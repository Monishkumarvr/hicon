/*
 * HiCon Melting Detection GStreamer Plugin
 */

#include "gsthiconmelting.h"

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_egl_interop.h>

#include "white_ratio_cuda_kernel.h"

GST_DEBUG_CATEGORY_STATIC(gst_hicon_melting_debug);
#define GST_CAT_DEFAULT gst_hicon_melting_debug

enum {
    PROP_0,
    PROP_CONFIG_INI,
    PROP_TAPPING_ZONE_COUNT,
    PROP_DESLAGGING_ZONE_COUNT,
    PROP_SPECTRO_ZONE_COUNT,
};

static GstStaticPadTemplate gst_hicon_melting_sink_template =
GST_STATIC_PAD_TEMPLATE("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

static GstStaticPadTemplate gst_hicon_melting_src_template =
GST_STATIC_PAD_TEMPLATE("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

#define gst_hicon_melting_parent_class parent_class
G_DEFINE_TYPE(GstHiConMelting, gst_hicon_melting, GST_TYPE_BASE_TRANSFORM);

static inline bool
is_supported_nv12_format(NvBufSurfaceColorFormat fmt)
{
    return fmt == NVBUF_COLOR_FORMAT_NV12 ||
           fmt == NVBUF_COLOR_FORMAT_NV12_ER ||
           fmt == NVBUF_COLOR_FORMAT_NV12_709 ||
           fmt == NVBUF_COLOR_FORMAT_NV12_709_ER ||
           fmt == NVBUF_COLOR_FORMAT_NV12_2020;
}

static inline bool
is_supported_rgba_format(NvBufSurfaceColorFormat fmt)
{
    return fmt == NVBUF_COLOR_FORMAT_RGBA;
}

static inline bool
is_supported_cuda_format(NvBufSurfaceColorFormat fmt)
{
    return is_supported_nv12_format(fmt) || is_supported_rgba_format(fmt);
}

struct CudaFrameView {
    NvBufSurfaceColorFormat color_format = NVBUF_COLOR_FORMAT_INVALID;
    const unsigned char *y_plane = nullptr;
    const unsigned char *uv_plane = nullptr;
    const unsigned char *rgba_plane = nullptr;
    int pitch_y = 0;
    int pitch_uv = 0;
    int pitch_rgba = 0;
    bool mapped_egl_here = false;
    cudaGraphicsResource_t cuda_resource = nullptr;
};

static void
release_cuda_frame_view(NvBufSurface *surface, guint batch_id, CudaFrameView &view)
{
    if (view.cuda_resource) {
        cudaGraphicsUnregisterResource(view.cuda_resource);
        view.cuda_resource = nullptr;
    }
#if defined(__aarch64__)
    if (view.mapped_egl_here) {
        NvBufSurfaceUnMapEglImage(surface, batch_id);
        view.mapped_egl_here = false;
    }
#endif
}

static bool
acquire_cuda_frame_view(
    GstHiConMelting *self,
    NvBufSurface *surface,
    guint batch_id,
    NvBufSurfaceParams &params,
    CudaFrameView &view
)
{
    view.color_format = params.colorFormat;
    view.pitch_y = (int)params.planeParams.pitch[0];
    view.pitch_uv = (int)(params.planeParams.pitch[1] ? params.planeParams.pitch[1]
                                                      : params.planeParams.pitch[0]);
    view.pitch_rgba = (int)(params.pitch ? params.pitch : params.planeParams.pitch[0]);
    if (view.pitch_y <= 0 || view.pitch_uv <= 0) {
        if (!is_supported_rgba_format(view.color_format)) {
            GST_WARNING_OBJECT(self, "Invalid NV12 plane pitch");
            return false;
        }
    }

    if (params.dataPtr) {
        if (is_supported_rgba_format(view.color_format)) {
            view.rgba_plane = reinterpret_cast<const unsigned char *>(params.dataPtr);
            return true;
        }
        int uv_offset = (int)params.planeParams.offset[1];
        if (uv_offset < 0) {
            GST_WARNING_OBJECT(self, "Invalid NV12 UV offset");
            return false;
        }
        view.y_plane = reinterpret_cast<const unsigned char *>(params.dataPtr);
        view.uv_plane = reinterpret_cast<const unsigned char *>(params.dataPtr) + uv_offset;
        return true;
    }

#if defined(__aarch64__)
    if (surface->memType == NVBUF_MEM_SURFACE_ARRAY || surface->memType == NVBUF_MEM_DEFAULT) {
        if (params.mappedAddr.eglImage == NULL) {
            if (NvBufSurfaceMapEglImage(surface, batch_id) != 0) {
                GST_WARNING_OBJECT(self, "NvBufSurfaceMapEglImage failed for batch %u", batch_id);
                return false;
            }
            view.mapped_egl_here = true;
        }

        cudaError_t cuda_err = cudaGraphicsEGLRegisterImage(
            &view.cuda_resource,
            (EGLImageKHR)params.mappedAddr.eglImage,
            cudaGraphicsRegisterFlagsNone
        );
        if (cuda_err != cudaSuccess) {
            GST_WARNING_OBJECT(
                self,
                "cudaGraphicsEGLRegisterImage failed for batch %u: %s",
                batch_id,
                cudaGetErrorString(cuda_err)
            );
            release_cuda_frame_view(surface, batch_id, view);
            return false;
        }

        cudaEglFrame egl_frame = {};
        cuda_err = cudaGraphicsResourceGetMappedEglFrame(&egl_frame, view.cuda_resource, 0, 0);
        if (cuda_err != cudaSuccess) {
            GST_WARNING_OBJECT(
                self,
                "cudaGraphicsResourceGetMappedEglFrame failed for batch %u: %s",
                batch_id,
                cudaGetErrorString(cuda_err)
            );
            release_cuda_frame_view(surface, batch_id, view);
            return false;
        }
        if (egl_frame.frameType != cudaEglFrameTypePitch || egl_frame.planeCount < 1) {
            GST_WARNING_OBJECT(
                self,
                "Unsupported EGL frame layout for batch %u: frameType=%u planeCount=%u",
                batch_id,
                (unsigned)egl_frame.frameType,
                egl_frame.planeCount
            );
            release_cuda_frame_view(surface, batch_id, view);
            return false;
        }

        if (is_supported_rgba_format(view.color_format)) {
            view.rgba_plane = reinterpret_cast<const unsigned char *>(egl_frame.frame.pPitch[0].ptr);
            if (egl_frame.planeDesc[0].pitch > 0) {
                view.pitch_rgba = (int)egl_frame.planeDesc[0].pitch;
            }
            return view.rgba_plane != nullptr;
        }

        view.y_plane = reinterpret_cast<const unsigned char *>(egl_frame.frame.pPitch[0].ptr);
        if (egl_frame.planeCount >= 2 && egl_frame.frame.pPitch[1].ptr != nullptr) {
            view.uv_plane = reinterpret_cast<const unsigned char *>(egl_frame.frame.pPitch[1].ptr);
        } else {
            int uv_offset = (int)params.planeParams.offset[1];
            if (view.y_plane == nullptr || uv_offset < 0) {
                GST_WARNING_OBJECT(self, "Invalid NV12 EGL frame layout for batch %u", batch_id);
                release_cuda_frame_view(surface, batch_id, view);
                return false;
            }
            view.uv_plane = view.y_plane + uv_offset;
        }
        if (egl_frame.planeDesc[0].pitch > 0) {
            view.pitch_y = (int)egl_frame.planeDesc[0].pitch;
        }
        if (egl_frame.planeCount >= 2 && egl_frame.planeDesc[1].pitch > 0) {
            view.pitch_uv = (int)egl_frame.planeDesc[1].pitch;
        }
        return view.y_plane != nullptr && view.uv_plane != nullptr;
    }
#endif

    GST_WARNING_OBJECT(self, "CUDA surface has no accessible device pointer (memType=%d)",
                       (int)surface->memType);
    return false;
}

static gpointer
hicon_melting_meta_copy_func(gpointer data, gpointer)
{
    NvDsUserMeta *src_user_meta = (NvDsUserMeta *)data;
    HiConMeltingMeta *src = (HiConMeltingMeta *)src_user_meta->user_meta_data;
    HiConMeltingMeta *dst = (HiConMeltingMeta *)g_malloc0(sizeof(HiConMeltingMeta));
    if (src && dst) {
        memcpy(dst, src, sizeof(HiConMeltingMeta));
    }
    return dst;
}

static void
hicon_melting_meta_release_func(gpointer data, gpointer)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    if (user_meta && user_meta->user_meta_data) {
        g_free(user_meta->user_meta_data);
        user_meta->user_meta_data = NULL;
    }
}

static void
attach_melting_meta(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta,
                    const HiConMeltingMeta &meta_data)
{
    NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
    if (!user_meta) {
        return;
    }

    HiConMeltingMeta *meta = (HiConMeltingMeta *)g_malloc0(sizeof(HiConMeltingMeta));
    memcpy(meta, &meta_data, sizeof(HiConMeltingMeta));

    user_meta->user_meta_data = meta;
    user_meta->base_meta.meta_type = HICON_MELTING_META_TYPE;
    user_meta->base_meta.copy_func = hicon_melting_meta_copy_func;
    user_meta->base_meta.release_func = hicon_melting_meta_release_func;
    nvds_add_user_meta_to_frame(frame_meta, user_meta);
}

static bool
parse_bbox(const gchar *value, MeltingZoneConfig &cfg)
{
    if (!value) {
        return false;
    }
    gint x1 = 0;
    gint y1 = 0;
    gint x2 = 0;
    gint y2 = 0;
    if (sscanf(value, "%d,%d,%d,%d", &x1, &y1, &x2, &y2) != 4) {
        return false;
    }
    cfg.x1 = x1;
    cfg.y1 = y1;
    cfg.x2 = x2;
    cfg.y2 = y2;
    return true;
}

static std::vector<MeltingZoneConfig>
load_zone_configs(GKeyFile *key_file, const gchar *section)
{
    std::vector<MeltingZoneConfig> zones;
    if (g_key_file_has_key(key_file, section, "zone_count", NULL)) {
        gint zone_count = g_key_file_get_integer(key_file, section, "zone_count", NULL);
        for (gint i = 0; i < zone_count && i < (gint)HICON_MELTING_MAX_ZONES; ++i) {
            MeltingZoneConfig cfg;

            std::string name_key = "zone_name." + std::to_string(i);
            if (g_key_file_has_key(key_file, section, name_key.c_str(), NULL)) {
                gchar *name_value = g_key_file_get_string(
                    key_file, section, name_key.c_str(), NULL
                );
                if (name_value) {
                    cfg.name = name_value;
                    g_free(name_value);
                }
            }
            if (cfg.name.empty()) {
                cfg.name = "zone-" + std::to_string(i + 1);
            }

            std::string bbox_key = "bbox." + std::to_string(i);
            gchar *bbox_value = g_key_file_get_string(
                key_file, section, bbox_key.c_str(), NULL
            );
            if (!parse_bbox(bbox_value, cfg)) {
                g_free(bbox_value);
                continue;
            }
            g_free(bbox_value);

            std::string on_frames_key = "on_frames." + std::to_string(i);
            if (g_key_file_has_key(key_file, section, on_frames_key.c_str(), NULL)) {
                cfg.on_frames = g_key_file_get_integer(
                    key_file, section, on_frames_key.c_str(), NULL
                );
            }

            std::string ar_key = "max_aspect_ratio." + std::to_string(i);
            if (g_key_file_has_key(key_file, section, ar_key.c_str(), NULL)) {
                cfg.max_aspect_ratio = (float)g_key_file_get_double(
                    key_file, section, ar_key.c_str(), NULL
                );
            }

            std::string cov_key = "max_coverage." + std::to_string(i);
            if (g_key_file_has_key(key_file, section, cov_key.c_str(), NULL)) {
                cfg.max_coverage = (float)g_key_file_get_double(
                    key_file, section, cov_key.c_str(), NULL
                );
            }

            zones.push_back(cfg);
        }
        return zones;
    }

    gsize names_len = 0;
    gchar **names = g_key_file_get_string_list(
        key_file, section, "zone_names", &names_len, NULL
    );
    if (!names) {
        return zones;
    }

    for (gsize i = 0; i < names_len && i < HICON_MELTING_MAX_ZONES; ++i) {
        MeltingZoneConfig cfg;
        cfg.name = names[i] ? names[i] : "";
        if (cfg.name.empty()) {
            continue;
        }

        std::string bbox_key = "bbox." + cfg.name;
        gchar *bbox_value = g_key_file_get_string(
            key_file, section, bbox_key.c_str(), NULL
        );
        if (!parse_bbox(bbox_value, cfg)) {
            g_free(bbox_value);
            continue;
        }
        g_free(bbox_value);

        std::string on_frames_key = "on_frames." + cfg.name;
        cfg.on_frames = g_key_file_get_integer(
            key_file, section, on_frames_key.c_str(), NULL
        );

        std::string ar_key = "max_aspect_ratio." + cfg.name;
        if (g_key_file_has_key(key_file, section, ar_key.c_str(), NULL)) {
            cfg.max_aspect_ratio = (float)g_key_file_get_double(
                key_file, section, ar_key.c_str(), NULL
            );
        }

        std::string cov_key = "max_coverage." + cfg.name;
        if (g_key_file_has_key(key_file, section, cov_key.c_str(), NULL)) {
            cfg.max_coverage = (float)g_key_file_get_double(
                key_file, section, cov_key.c_str(), NULL
            );
        }

        zones.push_back(cfg);
    }

    g_strfreev(names);
    return zones;
}

static void
rebuild_runtime_state(GstHiConMelting *self)
{
    self->tapping_states.clear();
    self->deslagging_states.clear();
    self->spectro_states.clear();

    for (const auto &cfg : self->config.tapping_zones) {
        TappingZoneState zs;
        zs.cfg = cfg;
        self->tapping_states.push_back(zs);
    }
    for (const auto &cfg : self->config.deslagging_zones) {
        BlobZoneState zs;
        zs.cfg = cfg;
        self->deslagging_states.push_back(zs);
    }
    for (const auto &cfg : self->config.spectro_zones) {
        BlobZoneState zs;
        zs.cfg = cfg;
        self->spectro_states.push_back(zs);
    }
}

static bool
load_melting_config(GstHiConMelting *self)
{
    self->config_loaded = FALSE;
    self->config_failed = FALSE;
    self->config = MeltingConfig();

    if (!self->config_ini || !*self->config_ini) {
        GST_ERROR_OBJECT(self, "Missing config-ini property");
        self->config_failed = TRUE;
        return false;
    }

    GKeyFile *key_file = g_key_file_new();
    GError *error = NULL;
    if (!g_key_file_load_from_data(
            key_file,
            self->config_ini,
            -1,
            G_KEY_FILE_NONE,
            &error)) {
        GST_ERROR_OBJECT(self, "Failed to parse melting config-ini: %s",
                         error ? error->message : "unknown");
        if (error) {
            g_error_free(error);
        }
        g_key_file_free(key_file);
        self->config_failed = TRUE;
        return false;
    }

    self->config.fps = (float)g_key_file_get_double(key_file, "global", "fps", NULL);
    if (self->config.fps <= 0.0f) {
        self->config.fps = 25.0f;
    }

    self->config.tapping_abs_threshold =
        g_key_file_get_integer(key_file, "tapping", "abs_brightness_threshold", NULL);
    self->config.tapping_on_ratio =
        (float)g_key_file_get_double(key_file, "tapping", "start_white_ratio", NULL);
    self->config.tapping_on_frames =
        g_key_file_get_integer(key_file, "tapping", "start_frame_count", NULL);
    self->config.tapping_off_ratio =
        (float)g_key_file_get_double(key_file, "tapping", "end_white_ratio", NULL);
    self->config.tapping_off_frames =
        g_key_file_get_integer(key_file, "tapping", "end_frame_count", NULL);
    self->config.tapping_zones = load_zone_configs(key_file, "tapping");

    self->config.deslagging_min_blob_area =
        g_key_file_get_integer(key_file, "deslagging", "min_blob_area", NULL);
    self->config.deslagging_brightness_thresh =
        g_key_file_get_integer(key_file, "deslagging", "brightness_thresh", NULL);
    self->config.deslagging_zones = load_zone_configs(key_file, "deslagging");

    self->config.spectro_min_blob_area =
        g_key_file_get_integer(key_file, "spectro", "min_blob_area", NULL);
    self->config.spectro_brightness_thresh =
        g_key_file_get_integer(key_file, "spectro", "brightness_thresh", NULL);
    self->config.spectro_zones = load_zone_configs(key_file, "spectro");

    g_key_file_free(key_file);
    rebuild_runtime_state(self);
    self->blackout_until_frame = 0;
    self->config_loaded = TRUE;

    GST_INFO_OBJECT(
        self,
        "Loaded melting config: tapping=%zu deslagging=%zu spectro=%zu fps=%.1f",
        self->config.tapping_zones.size(),
        self->config.deslagging_zones.size(),
        self->config.spectro_zones.size(),
        self->config.fps
    );
    g_printerr(
        "[hicon_melting] loaded config: tapping=%zu deslagging=%zu spectro=%zu fps=%.1f\n",
        self->config.tapping_zones.size(),
        self->config.deslagging_zones.size(),
        self->config.spectro_zones.size(),
        self->config.fps
    );
    return true;
}

static std::vector<DsMoltenBlob>
filter_blobs(const std::vector<DsMoltenBlob> &blobs, const BlobZoneState &state)
{
    std::vector<DsMoltenBlob> filtered;
    float zone_area = std::max(
        1.0f,
        (float)(state.cfg.x2 - state.cfg.x1 + 1) * (float)(state.cfg.y2 - state.cfg.y1 + 1)
    );

    for (const auto &blob : blobs) {
        float bw = (float)std::max(1, blob.x2 - blob.x1);
        float bh = (float)std::max(1, blob.y2 - blob.y1);
        if (state.cfg.max_aspect_ratio > 0.0f &&
            (std::max(bw, bh) / std::max(1.0f, std::min(bw, bh))) > state.cfg.max_aspect_ratio) {
            continue;
        }
        if (state.cfg.max_coverage > 0.0f &&
            ((float)blob.area / zone_area) > state.cfg.max_coverage) {
            continue;
        }
        filtered.push_back(blob);
    }

    return filtered;
}

static void
fill_zone_meta(HiConMeltingZoneMeta &dst, bool active, uint32_t raw_count,
               uint32_t filtered_count, float white_ratio,
               float max_blob_area, float max_blob_brightness)
{
    memset(&dst, 0, sizeof(dst));
    dst.valid = 1U;
    dst.active = active ? 1U : 0U;
    dst.raw_count = raw_count;
    dst.filtered_count = filtered_count;
    dst.white_ratio = white_ratio;
    dst.max_blob_area = max_blob_area;
    dst.max_blob_brightness = max_blob_brightness;
}

static inline int
scaled_temporal_threshold(int original_frames, int zone_stride_frames)
{
    if (original_frames <= 0) {
        return 0;
    }
    return std::max(1, (original_frames + zone_stride_frames - 1) / zone_stride_frames);
}

static GstFlowReturn
gst_hicon_melting_transform_ip(GstBaseTransform *btrans, GstBuffer *buf)
{
    GstHiConMelting *self = GST_HICON_MELTING(btrans);
    if (!self->config_loaded && !self->config_failed) {
        load_melting_config(self);
    }
    if (!self->config_loaded) {
        return GST_FLOW_OK;
    }

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) {
        return GST_FLOW_OK;
    }

    GstMapInfo map_info = GST_MAP_INFO_INIT;
    if (!gst_buffer_map(buf, &map_info, GST_MAP_READ)) {
        GST_WARNING_OBJECT(self, "Failed to map GstBuffer");
        return GST_FLOW_OK;
    }

    NvBufSurface *surface = reinterpret_cast<NvBufSurface *>(map_info.data);
    if (!surface) {
        gst_buffer_unmap(buf, &map_info);
        return GST_FLOW_OK;
    }

    NvDsMetaList *l_frame = batch_meta->frame_meta_list;
    while (l_frame) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        if (!frame_meta) {
            l_frame = l_frame->next;
            continue;
        }

        HiConMeltingMeta meta_out;
        memset(&meta_out, 0, sizeof(meta_out));
        meta_out.version = HICON_MELTING_META_VERSION;
        meta_out.frame_num = frame_meta->frame_num;
        meta_out.ntp_timestamp = frame_meta->ntp_timestamp;
        meta_out.tapping_zone_count = (uint32_t)self->tapping_states.size();
        meta_out.deslagging_zone_count = (uint32_t)self->deslagging_states.size();
        meta_out.spectro_zone_count = (uint32_t)self->spectro_states.size();
        meta_out.reserved0 = 0U;

        if (self->warmup_until_frame < 0) {
            self->warmup_until_frame =
                (gint64)frame_meta->frame_num +
                (gint64)std::llround(HICON_MELTING_STARTUP_WARMUP_S * self->config.fps);
            GST_INFO_OBJECT(
                self,
                "Starting CUDA warm-up: skipping melting compute for %.1fs (%lld frames)",
                (double)HICON_MELTING_STARTUP_WARMUP_S,
                (long long)std::max<gint64>(0, self->warmup_until_frame - (gint64)frame_meta->frame_num)
            );
        }
        if ((gint64)frame_meta->frame_num < self->warmup_until_frame) {
            meta_out.reserved0 = 4000U;
            attach_melting_meta(batch_meta, frame_meta, meta_out);
            l_frame = l_frame->next;
            continue;
        }

        const int tapping_zone_count = (int)std::min<size_t>(
            self->tapping_states.size(),
            HICON_MELTING_MAX_ZONES
        );
        const int zone_stride_frames = std::max(1, tapping_zone_count);

        int run_tapping_zone = -1;
        int run_deslagging_zone = -1;
        int run_spectro_zone = -1;
        if (tapping_zone_count > 0) {
            run_tapping_zone =
                (int)(((uint64_t)frame_meta->frame_num) % (uint64_t)tapping_zone_count);
        }

        guint batch_id = frame_meta->batch_id;
        if (batch_id >= surface->batchSize) {
            meta_out.reserved0 = 1000U + (uint32_t)batch_id;
            attach_melting_meta(batch_meta, frame_meta, meta_out);
            l_frame = l_frame->next;
            continue;
        }

        NvBufSurfaceParams &params = surface->surfaceList[batch_id];
        if (!is_supported_cuda_format(params.colorFormat)) {
            meta_out.reserved0 = 2000U + (uint32_t)params.colorFormat;
            attach_melting_meta(batch_meta, frame_meta, meta_out);
            l_frame = l_frame->next;
            continue;
        }

        CudaFrameView frame_view;
        if (!acquire_cuda_frame_view(self, surface, batch_id, params, frame_view)) {
            meta_out.reserved0 = 3000U + (uint32_t)surface->memType;
            attach_melting_meta(batch_meta, frame_meta, meta_out);
            l_frame = l_frame->next;
            continue;
        }
        meta_out.reserved0 = 9000U + (uint32_t)surface->memType;

        bool blackout_active = frame_meta->frame_num < self->blackout_until_frame;

        bool trigger_blackout = false;
        if (run_tapping_zone >= 0) {
            for (size_t i = 0; i < self->tapping_states.size() && i < HICON_MELTING_MAX_ZONES; ++i) {
                if ((int)i != run_tapping_zone) {
                    continue;
                }
                auto &state = self->tapping_states[i];
                float white_ratio = 0.0f;
                int rc = 0;
                if (is_supported_rgba_format(frame_view.color_format)) {
                    rc = ds_compute_white_ratio_from_device_rgba(
                        frame_view.rgba_plane,
                        (int)params.width,
                        (int)params.height,
                        frame_view.pitch_rgba,
                        state.cfg.x1,
                        state.cfg.y1,
                        state.cfg.x2,
                        state.cfg.y2,
                        self->config.tapping_abs_threshold,
                        &white_ratio
                    );
                } else {
                    rc = ds_compute_white_ratio_from_device_nv12(
                        frame_view.y_plane,
                        frame_view.uv_plane,
                        (int)params.width,
                        (int)params.height,
                        frame_view.pitch_y,
                        frame_view.pitch_uv,
                        state.cfg.x1,
                        state.cfg.y1,
                        state.cfg.x2,
                        state.cfg.y2,
                        self->config.tapping_abs_threshold,
                        &white_ratio
                    );
                }
                if (rc != 0) {
                    white_ratio = 0.0f;
                }
                state.white_ratio = white_ratio;
                if (white_ratio > HICON_NODULIZER_WHITE_RATIO_THRESHOLD) {
                    trigger_blackout = true;
                }
            }
        }

        if (trigger_blackout) {
            self->blackout_until_frame = frame_meta->frame_num +
                (uint64_t)std::round(HICON_NODULIZER_BLACKOUT_S * self->config.fps);
            blackout_active = true;
        }
        meta_out.blackout_active = blackout_active ? 1U : 0U;

        const int tapping_on_frames =
            scaled_temporal_threshold(self->config.tapping_on_frames, zone_stride_frames);
        const int tapping_off_frames =
            scaled_temporal_threshold(self->config.tapping_off_frames, zone_stride_frames);

        for (size_t i = 0; i < self->tapping_states.size() && i < HICON_MELTING_MAX_ZONES; ++i) {
            auto &state = self->tapping_states[i];
            if ((int)i == run_tapping_zone && !blackout_active) {
                if (!state.active) {
                    state.on_count = (state.white_ratio > self->config.tapping_on_ratio)
                        ? (state.on_count + 1)
                        : 0;
                    if (state.on_count >= tapping_on_frames) {
                        state.active = true;
                        state.on_count = 0;
                        state.off_count = 0;
                    }
                } else {
                    state.off_count = (state.white_ratio < self->config.tapping_off_ratio)
                        ? (state.off_count + 1)
                        : 0;
                    if (state.off_count >= tapping_off_frames) {
                        state.active = false;
                        state.on_count = 0;
                        state.off_count = 0;
                    }
                }
            }

            fill_zone_meta(
                meta_out.tapping[i],
                state.active,
                0U,
                0U,
                state.white_ratio,
                0.0f,
                0.0f
            );
        }

        for (size_t i = 0; i < self->deslagging_states.size() && i < HICON_MELTING_MAX_ZONES; ++i) {
            auto &state = self->deslagging_states[i];
            if ((int)i == run_deslagging_zone && !blackout_active) {
                DsMoltenBlob blobs[DS_MAX_MOLTEN_BLOBS];
                int num_blobs = 0;
                int rc = 0;
                if (is_supported_rgba_format(frame_view.color_format)) {
                    rc = ds_detect_molten_blobs_rgba_device(
                        frame_view.rgba_plane,
                        (int)params.width,
                        (int)params.height,
                        frame_view.pitch_rgba,
                        state.cfg.x1,
                        state.cfg.y1,
                        state.cfg.x2,
                        state.cfg.y2,
                        blobs,
                        DS_MAX_MOLTEN_BLOBS,
                        &num_blobs,
                        self->config.deslagging_min_blob_area,
                        self->config.deslagging_brightness_thresh
                    );
                } else {
                    rc = ds_detect_molten_blobs_nv12_device(
                        frame_view.y_plane,
                        frame_view.uv_plane,
                        (int)params.width,
                        (int)params.height,
                        frame_view.pitch_y,
                        frame_view.pitch_uv,
                        state.cfg.x1,
                        state.cfg.y1,
                        state.cfg.x2,
                        state.cfg.y2,
                        blobs,
                        DS_MAX_MOLTEN_BLOBS,
                        &num_blobs,
                        self->config.deslagging_min_blob_area,
                        self->config.deslagging_brightness_thresh
                    );
                }
                state.raw_count = (rc == 0) ? (uint32_t)std::max(0, num_blobs) : 0U;
                state.filtered_count = state.raw_count;
                state.active = state.filtered_count > 0U;
                state.max_blob_area = 0.0f;
                state.max_blob_brightness = 0.0f;
                for (uint32_t j = 0; j < state.filtered_count; ++j) {
                    state.max_blob_area = std::max(state.max_blob_area, (float)blobs[j].area);
                    state.max_blob_brightness = std::max(
                        state.max_blob_brightness, (float)blobs[j].avg_brightness
                    );
                }
            }

            fill_zone_meta(
                meta_out.deslagging[i],
                state.active,
                state.raw_count,
                state.filtered_count,
                0.0f,
                state.max_blob_area,
                state.max_blob_brightness
            );
        }

        for (size_t i = 0; i < self->spectro_states.size() && i < HICON_MELTING_MAX_ZONES; ++i) {
            auto &state = self->spectro_states[i];
            if ((int)i == run_spectro_zone && !blackout_active) {
                DsMoltenBlob blobs[DS_MAX_MOLTEN_BLOBS];
                int num_blobs = 0;
                int rc = 0;
                if (is_supported_rgba_format(frame_view.color_format)) {
                    rc = ds_detect_molten_blobs_rgba_device(
                        frame_view.rgba_plane,
                        (int)params.width,
                        (int)params.height,
                        frame_view.pitch_rgba,
                        state.cfg.x1,
                        state.cfg.y1,
                        state.cfg.x2,
                        state.cfg.y2,
                        blobs,
                        DS_MAX_MOLTEN_BLOBS,
                        &num_blobs,
                        self->config.spectro_min_blob_area,
                        self->config.spectro_brightness_thresh
                    );
                } else {
                    rc = ds_detect_molten_blobs_nv12_device(
                        frame_view.y_plane,
                        frame_view.uv_plane,
                        (int)params.width,
                        (int)params.height,
                        frame_view.pitch_y,
                        frame_view.pitch_uv,
                        state.cfg.x1,
                        state.cfg.y1,
                        state.cfg.x2,
                        state.cfg.y2,
                        blobs,
                        DS_MAX_MOLTEN_BLOBS,
                        &num_blobs,
                        self->config.spectro_min_blob_area,
                        self->config.spectro_brightness_thresh
                    );
                }
                state.raw_count = (rc == 0) ? (uint32_t)std::max(0, num_blobs) : 0U;

                std::vector<DsMoltenBlob> raw_vec;
                raw_vec.reserve(state.raw_count);
                for (uint32_t j = 0; j < state.raw_count; ++j) {
                    raw_vec.push_back(blobs[j]);
                }
                auto filtered = filter_blobs(raw_vec, state);
                state.filtered_count = (uint32_t)filtered.size();
                state.max_blob_area = 0.0f;
                state.max_blob_brightness = 0.0f;
                for (const auto &blob : filtered) {
                    state.max_blob_area = std::max(state.max_blob_area, (float)blob.area);
                    state.max_blob_brightness = std::max(
                        state.max_blob_brightness, (float)blob.avg_brightness
                    );
                }

                bool any_blob = state.filtered_count > 0U;
                const int spectro_on_frames =
                    scaled_temporal_threshold(state.cfg.on_frames, zone_stride_frames);
                if (any_blob) {
                    state.on_count = std::min(state.on_count + 1, spectro_on_frames + 1);
                } else {
                    state.on_count = 0;
                }
                bool now_active = (spectro_on_frames > 0)
                    ? (state.on_count >= spectro_on_frames)
                    : any_blob;
                state.active = now_active;
            }

            fill_zone_meta(
                meta_out.spectro[i],
                state.active,
                state.raw_count,
                state.filtered_count,
                0.0f,
                state.max_blob_area,
                state.max_blob_brightness
            );
        }

        attach_melting_meta(batch_meta, frame_meta, meta_out);
        release_cuda_frame_view(surface, batch_id, frame_view);
        l_frame = l_frame->next;
    }

    gst_buffer_unmap(buf, &map_info);
    return GST_FLOW_OK;
}

static gboolean
gst_hicon_melting_start(GstBaseTransform *btrans)
{
    GstHiConMelting *self = GST_HICON_MELTING(btrans);
    self->blackout_until_frame = 0;
    self->warmup_until_frame = -1;
    if (!self->config_loaded && !self->config_failed && self->config_ini && *self->config_ini) {
        load_melting_config(self);
    }
    return TRUE;
}

static gboolean
gst_hicon_melting_stop(GstBaseTransform *btrans)
{
    GstHiConMelting *self = GST_HICON_MELTING(btrans);
    self->config_loaded = FALSE;
    self->config_failed = FALSE;
    self->tapping_states.clear();
    self->deslagging_states.clear();
    self->spectro_states.clear();
    self->warmup_until_frame = -1;
    return TRUE;
}

static void
gst_hicon_melting_set_property(GObject *object, guint prop_id,
                               const GValue *value, GParamSpec *)
{
    GstHiConMelting *self = GST_HICON_MELTING(object);

    switch (prop_id) {
    case PROP_CONFIG_INI:
        g_free(self->config_ini);
        self->config_ini = g_value_dup_string(value);
        self->config_loaded = FALSE;
        self->config_failed = FALSE;
        if (self->config_ini && *self->config_ini) {
            load_melting_config(self);
        }
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, NULL);
        break;
    }
}

static void
gst_hicon_melting_get_property(GObject *object, guint prop_id,
                               GValue *value, GParamSpec *)
{
    GstHiConMelting *self = GST_HICON_MELTING(object);

    switch (prop_id) {
    case PROP_CONFIG_INI:
        g_value_set_string(value, self->config_ini);
        break;
    case PROP_TAPPING_ZONE_COUNT:
        g_value_set_uint(value, (guint)self->config.tapping_zones.size());
        break;
    case PROP_DESLAGGING_ZONE_COUNT:
        g_value_set_uint(value, (guint)self->config.deslagging_zones.size());
        break;
    case PROP_SPECTRO_ZONE_COUNT:
        g_value_set_uint(value, (guint)self->config.spectro_zones.size());
        break;
    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, NULL);
        break;
    }
}

static void
gst_hicon_melting_finalize(GObject *object)
{
    GstHiConMelting *self = GST_HICON_MELTING(object);
    g_free(self->config_ini);
    G_OBJECT_CLASS(parent_class)->finalize(object);
}

static void
gst_hicon_melting_class_init(GstHiConMeltingClass *klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass *gstelement_class = GST_ELEMENT_CLASS(klass);
    GstBaseTransformClass *gstbasetransform_class = GST_BASE_TRANSFORM_CLASS(klass);

    gobject_class->set_property = GST_DEBUG_FUNCPTR(gst_hicon_melting_set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR(gst_hicon_melting_get_property);
    gobject_class->finalize = GST_DEBUG_FUNCPTR(gst_hicon_melting_finalize);

    gstbasetransform_class->start = GST_DEBUG_FUNCPTR(gst_hicon_melting_start);
    gstbasetransform_class->stop = GST_DEBUG_FUNCPTR(gst_hicon_melting_stop);
    gstbasetransform_class->transform_ip = GST_DEBUG_FUNCPTR(gst_hicon_melting_transform_ip);

    g_object_class_install_property(
        gobject_class,
        PROP_CONFIG_INI,
        g_param_spec_string(
            "config-ini",
            "config-ini",
            "Serialized INI config for Stream 0 melting detection",
            NULL,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)
        )
    );
    g_object_class_install_property(
        gobject_class,
        PROP_TAPPING_ZONE_COUNT,
        g_param_spec_uint(
            "tapping-zone-count",
            "tapping-zone-count",
            "Number of parsed tapping zones",
            0,
            HICON_MELTING_MAX_ZONES,
            0,
            (GParamFlags)(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)
        )
    );
    g_object_class_install_property(
        gobject_class,
        PROP_DESLAGGING_ZONE_COUNT,
        g_param_spec_uint(
            "deslagging-zone-count",
            "deslagging-zone-count",
            "Number of parsed deslagging zones",
            0,
            HICON_MELTING_MAX_ZONES,
            0,
            (GParamFlags)(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)
        )
    );
    g_object_class_install_property(
        gobject_class,
        PROP_SPECTRO_ZONE_COUNT,
        g_param_spec_uint(
            "spectro-zone-count",
            "spectro-zone-count",
            "Number of parsed spectro zones",
            0,
            HICON_MELTING_MAX_ZONES,
            0,
            (GParamFlags)(G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)
        )
    );

    gst_element_class_add_pad_template(
        gstelement_class,
        gst_static_pad_template_get(&gst_hicon_melting_src_template)
    );
    gst_element_class_add_pad_template(
        gstelement_class,
        gst_static_pad_template_get(&gst_hicon_melting_sink_template)
    );

    gst_element_class_set_static_metadata(
        gstelement_class,
        "HiCon Melting Detector",
        "Filter/Effect/Video",
        "Native CUDA Stream 0 melting detector for tapping/deslagging/spectro",
        "OpenAI"
    );
}

static void
gst_hicon_melting_init(GstHiConMelting *self)
{
    self->config_ini = NULL;
    self->config_loaded = FALSE;
    self->config_failed = FALSE;
    self->blackout_until_frame = 0;
    self->warmup_until_frame = -1;

    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
    /* Keep the buffer in-place, but do not enable passthrough mode:
     * transform_ip must run on every frame so the plugin can attach
     * NvDsUserMeta to the outgoing buffer. */
    gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(self), FALSE);
}

static gboolean
hicon_melting_plugin_init(GstPlugin *plugin)
{
    GST_DEBUG_CATEGORY_INIT(gst_hicon_melting_debug, "hicon_melting_detect", 0,
                            "HiCon melting detector");

    return gst_element_register(plugin, "hicon_melting_detect", GST_RANK_PRIMARY,
                                GST_TYPE_HICON_MELTING);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    hiconmelting,
    DESCRIPTION,
    hicon_melting_plugin_init,
    VERSION,
    LICENSE,
    BINARY_PACKAGE,
    URL)
