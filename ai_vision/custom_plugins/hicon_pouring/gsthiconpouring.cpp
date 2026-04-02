/*
 * HiCon Pouring Detection GStreamer Plugin
 * =========================================
 * GstBaseTransform element — exports lightweight pouring session state from
 * tracked object metadata. Python owns brightness-based pour transitions,
 * mould business logic, screenshots, DB writes, heat-cycle updates, and sync.
 *
 * Pipeline placement:
 *   nvinfer → nvtracker → [hicon_pouring_detect] → nvosd → sink
 */

#include "gsthiconpouring.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <climits>
#include <cfloat>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <sstream>

GST_DEBUG_CATEGORY_STATIC(gst_hicon_pouring_debug);
#define GST_CAT_DEFAULT gst_hicon_pouring_debug

/* Default property values */
enum {
    PROP_0,
    PROP_GPU_DEVICE_ID,
    PROP_ENABLE_OSD,
};

#define DEFAULT_GPU_ID 0

/* Pad templates — match DeepStream NVMM video caps (gst-dsexample pattern) */
static GstStaticPadTemplate gst_hicon_pouring_sink_template =
    GST_STATIC_PAD_TEMPLATE("sink",
        GST_PAD_SINK,
        GST_PAD_ALWAYS,
        GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES("memory:NVMM",
            "{ NV12, RGBA, I420 }")));

static GstStaticPadTemplate gst_hicon_pouring_src_template =
    GST_STATIC_PAD_TEMPLATE("src",
        GST_PAD_SRC,
        GST_PAD_ALWAYS,
        GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES("memory:NVMM",
            "{ NV12, RGBA, I420 }")));

#define gst_hicon_pouring_parent_class parent_class
G_DEFINE_TYPE(GstHiConPouring, gst_hicon_pouring, GST_TYPE_BASE_TRANSFORM);

/* ================================================================
 * GEOMETRY UTILITIES
 * ================================================================ */

static inline float clamp_f(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

static inline void bbox_center(const float bbox[4], float &cx, float &cy) {
    cx = (bbox[0] + bbox[2]) / 2.0f;
    cy = (bbox[1] + bbox[3]) / 2.0f;
}

static inline bool point_in_bbox(float px, float py, const float bbox[4]) {
    return (px >= bbox[0] && px <= bbox[2] && py >= bbox[1] && py <= bbox[3]);
}

static inline void expand_bbox(const float in[4], float margin, float out[4]) {
    out[0] = in[0] - margin;
    out[1] = in[1] - margin;
    out[2] = in[2] + margin;
    out[3] = in[3] + margin;
}

static inline void expand_bbox_xy(const float in[4], float margin_x, float margin_y, float out[4]) {
    out[0] = in[0] - margin_x;
    out[1] = in[1] - margin_y;
    out[2] = in[2] + margin_x;
    out[3] = in[3] + margin_y;
}

static inline void norm_point_in_trolley(float px, float py, const float bbox[4],
                                          float &nx, float &ny) {
    float w = std::max(1.0f, bbox[2] - bbox[0]);
    float h = std::max(1.0f, bbox[3] - bbox[1]);
    nx = (px - bbox[0]) / w;
    ny = (py - bbox[1]) / h;
}

static inline void norm_point_in_expanded_trolley(float px, float py, const float bbox[4],
                                                  float margin_x, float margin_y,
                                                  float &nx, float &ny) {
    float expanded[4];
    expand_bbox_xy(bbox, margin_x, margin_y, expanded);
    norm_point_in_trolley(px, py, expanded, nx, ny);
}

static inline float l2_dist(float ax, float ay, float bx, float by) {
    float dx = ax - bx, dy = ay - by;
    return std::sqrt(dx*dx + dy*dy);
}

static inline void mouth_probe_point(const float bbox[4], float &px, float &py) {
    px = (bbox[0] + bbox[2]) / 2.0f;
    py = bbox[3] + POUR_DOT_BELOW_PX;
}

static inline void mouth_probe_point_scaled(const float bbox[4], int below_px, float &px, float &py) {
    px = (bbox[0] + bbox[2]) / 2.0f;
    py = bbox[3] + (float)below_px;
}

static inline void clear_active_probe(TrolleyState *st) {
    st->active_probe_track_id = UINT64_MAX;
    st->active_probe_last_seen_f = -999999;
    st->active_probe_bbox_valid = false;
    st->active_probe_pt_valid = false;
    st->active_probe_from_hold = false;
    memset(st->active_probe_bbox, 0, sizeof(st->active_probe_bbox));
    st->active_probe_pt_px[0] = 0.0f;
    st->active_probe_pt_px[1] = 0.0f;
}

static inline void clear_frozen_probe(TrolleyState *st) {
    st->frozen_probe_active = false;
    st->frozen_probe_x = 0.0f;
    st->frozen_probe_y = 0.0f;
    st->frozen_probe_bbox_valid = false;
    memset(st->frozen_probe_bbox, 0, sizeof(st->frozen_probe_bbox));
}

static inline bool probe_is_pouring(float head_score, float tail_score) {
    return head_score >= TH_ON && tail_score >= TH_OFF;
}

static gboolean env_flag_enabled(const char *name, gboolean default_value) {
    const gchar *raw = g_getenv(name);
    if (!raw || !*raw) {
        return default_value;
    }
    if (g_ascii_strcasecmp(raw, "1") == 0 ||
        g_ascii_strcasecmp(raw, "true") == 0 ||
        g_ascii_strcasecmp(raw, "yes") == 0 ||
        g_ascii_strcasecmp(raw, "on") == 0) {
        return TRUE;
    }
    if (g_ascii_strcasecmp(raw, "0") == 0 ||
        g_ascii_strcasecmp(raw, "false") == 0 ||
        g_ascii_strcasecmp(raw, "no") == 0 ||
        g_ascii_strcasecmp(raw, "off") == 0) {
        return FALSE;
    }
    return default_value;
}

static gint env_int_value(const char *name, gint default_value) {
    const gchar *raw = g_getenv(name);
    if (!raw || !*raw) {
        return default_value;
    }

    gchar *endptr = nullptr;
    long value = std::strtol(raw, &endptr, 10);
    if (endptr == raw || (endptr && *endptr != '\0')) {
        return default_value;
    }
    value = std::max<long>(G_MININT, std::min<long>(G_MAXINT, value));
    return (gint)value;
}

static inline gint scale_pixel_value_int(float value, float scale, gint minimum = 0) {
    return std::max(minimum, (gint)std::lround(value * scale));
}

static inline gint scale_offset_value(gint value, float scale) {
    return (gint)std::lround((float)value * scale);
}

static inline float scale_pixel_value_float(float value, float scale, float minimum = 0.0f) {
    return std::max(minimum, value * scale);
}

/* ================================================================
 * BRIGHTNESS PROBE — NvBufSurface RGBA pixel access
 *
 * HSV V-channel = max(R, G, B). No full HSV conversion needed.
 * ================================================================ */

static float brightness_probe(const uint8_t *data, uint32_t pitch,
                               int width, int height,
                               int cx, int cy, int r, int bpp)
{
    int x1 = (int)clamp_f((float)(cx - r), 0.0f, (float)(width  - 1));
    int x2 = (int)clamp_f((float)(cx + r), 0.0f, (float)(width  - 1));
    int y1 = (int)clamp_f((float)(cy - r), 0.0f, (float)(height - 1));
    int y2 = (int)clamp_f((float)(cy + r), 0.0f, (float)(height - 1));

    if (x2 < x1 || y2 < y1) return 0.0f;

    float sum = 0.0f;
    int count = 0;
    for (int y = y1; y <= y2; y++) {
        const uint8_t *row = data + (size_t)y * pitch;
        for (int x = x1; x <= x2; x++) {
            float V;
            if (bpp == 1) {
                V = (float)row[x];  /* NV12 Y plane: 1 byte = luminance */
            } else {
                int off = x * bpp;
                V = (float)std::max({row[off], row[off+1], row[off+2]});  /* RGBA */
            }
            sum += V;
            count++;
        }
    }
    return (count > 0) ? (sum / (float)count) : 0.0f;
}

static float flare_score(const uint8_t *data, uint32_t pitch,
                          int width, int height,
                          float pour_dot_x, float pour_dot_y, int bpp)
{
    float total = 0.0f;
    int valid = 0;
    for (int i = 0; i < NUM_PROBES; i++) {
        int cx = (int)clamp_f(pour_dot_x + PROBE_DX[i], 0.0f, (float)(width - 1));
        int cy = (int)clamp_f(pour_dot_y + PROBE_DY[i], 0.0f, (float)(height - 1));
        float v = brightness_probe(data, pitch, width, height, cx, cy, PROBE_R, bpp);
        if (v > 0.0f) { total += v; valid++; }
    }
    return (valid > 0) ? (total / (float)valid) : 0.0f;
}

/* ================================================================
 * SEGMENT SPLITTING — recursive
 * ================================================================ */

static void split_segment_by_motion(const PourSegment &seg, float d_split, int t_hold,
                                     std::vector<PourSegment> &out)
{
    std::vector<std::pair<int,std::pair<float,float>>> idx_pts;
    for (int i = 0; i < (int)seg.mouth_pts_norm.size(); i++) {
        if (seg.mouth_pts_norm[i].first >= 0.0f)
            idx_pts.push_back({i, seg.mouth_pts_norm[i]});
    }

    if ((int)idx_pts.size() < std::max(3, t_hold + 1)) {
        out.push_back(seg);
        return;
    }

    float anchor_x = idx_pts[0].second.first, anchor_y = idx_pts[0].second.second;
    int hold = 0;

    for (int k = 1; k < (int)idx_pts.size(); k++) {
        float d = l2_dist(idx_pts[k].second.first, idx_pts[k].second.second, anchor_x, anchor_y);
        if (d > d_split) {
            hold++;
        } else {
            hold = 0;
        }
        if (hold >= t_hold) {
            int split_pts_idx  = k - t_hold;
            int split_norm_idx = idx_pts[split_pts_idx].first;

            int total_frames = seg.end_f - seg.start_f + 1;
            int total_pts    = (int)seg.mouth_pts_norm.size();
            int split_frame;
            if (total_pts > 0)
                split_frame = seg.start_f + (split_norm_idx * total_frames) / total_pts;
            else
                split_frame = seg.start_f + split_norm_idx;

            split_frame = std::max(seg.start_f, std::min(seg.end_f - 1, split_frame));

            PourSegment left;
            left.start_f = seg.start_f;
            left.end_f   = split_frame;
            left.mouth_pts_norm.assign(seg.mouth_pts_norm.begin(),
                                        seg.mouth_pts_norm.begin() + split_norm_idx + 1);
            out.push_back(left);

            PourSegment right;
            right.start_f = split_frame + 1;
            right.end_f   = seg.end_f;
            right.mouth_pts_norm.assign(seg.mouth_pts_norm.begin() + split_norm_idx + 1,
                                         seg.mouth_pts_norm.end());
            split_segment_by_motion(right, d_split, t_hold, out);
            return;
        }
    }

    out.push_back(seg);
}

/* ================================================================
 * SPATIAL CLUSTERING
 * ================================================================ */

static int assign_to_cluster(float px, float py,
                              const std::vector<MouldCluster> &clusters,
                              float r_cluster)
{
    int latest_cid = 0;
    for (const auto &c : clusters) latest_cid = std::max(latest_cid, c.cid);
    int min_allowed_cid = std::max(1, latest_cid - CLUSTER_BACKTRACK_CID_GUARD);

    int best_i = -1;
    float best_d = 1e9f;
    for (int i = 0; i < (int)clusters.size(); i++) {
        if (clusters[i].cid < min_allowed_cid) continue;
        float d = l2_dist(px, py, clusters[i].centroid.first, clusters[i].centroid.second);
        if (d < best_d) { best_d = d; best_i = i; }
    }
    return (best_d <= r_cluster) ? best_i : -1;
}

static std::vector<MouldCluster> merge_clusters_fn(std::vector<MouldCluster> &clusters,
                                                     float r_merge)
{
    std::vector<MouldCluster> out;
    std::vector<bool> used(clusters.size(), false);
    int cid = 1;

    for (int i = 0; i < (int)clusters.size(); i++) {
        if (used[i]) continue;
        used[i] = true;

        std::vector<int> group = {i};
        for (int j = i + 1; j < (int)clusters.size(); j++) {
            if (used[j]) continue;
            float d = l2_dist(clusters[i].centroid.first, clusters[i].centroid.second,
                              clusters[j].centroid.first, clusters[j].centroid.second);
            if (d <= r_merge) { group.push_back(j); used[j] = true; }
        }

        float sx = 0, sy = 0;
        MouldCluster mc;
        mc.cid = cid++;
        for (int idx : group) {
            sx += clusters[idx].centroid.first;
            sy += clusters[idx].centroid.second;
            for (auto &s : clusters[idx].segments)
                mc.segments.push_back(s);
        }
        mc.centroid = {sx / (float)group.size(), sy / (float)group.size()};
        out.push_back(std::move(mc));
    }
    return out;
}

static std::vector<MouldCluster> build_clusters(const std::vector<PourSegment> &segments,
                                                  int min_pour_frames,
                                                  float r_cluster, float r_merge)
{
    std::vector<MouldCluster> clusters;
    int next_cid = 1;

    for (auto &seg : segments) {
        if (seg.duration_frames() < min_pour_frames) continue;
        auto p = seg.representative_point();
        if (p.first == 0.0f && p.second == 0.0f && seg.mouth_pts_norm.empty()) continue;

        int idx = assign_to_cluster(p.first, p.second, clusters, r_cluster);
        if (idx == -1) {
            MouldCluster mc;
            mc.cid = next_cid++;
            mc.centroid = p;
            mc.segments.push_back(seg);
            clusters.push_back(std::move(mc));
        } else {
            clusters[idx].segments.push_back(seg);
            float sx = 0, sy = 0;
            for (auto &s : clusters[idx].segments) {
                auto rp = s.representative_point();
                sx += rp.first; sy += rp.second;
            }
            int n = (int)clusters[idx].segments.size();
            clusters[idx].centroid = {sx / n, sy / n};
        }
    }
    return merge_clusters_fn(clusters, r_merge);
}

/* ================================================================
 * CLUSTER PREVIEW — for OSD overlay (real-time mould count)
 * ================================================================ */

static ClusterPreview build_cluster_preview_for_overlay(
    const GstHiConPouring *self, const TrolleyState *st, int frame_idx)
{
    ClusterPreview preview;
    std::vector<PourSegment> preview_segments = st->completed;
    bool has_active_point = false;
    std::pair<float,float> active_point = {0.0f, 0.0f};

    if (st->pour_active && st->current_seg != nullptr) {
        PourSegment active_seg = *(st->current_seg);
        active_seg.end_f = frame_idx;
        active_point = active_seg.representative_point();
        if (!(active_point.first == 0.0f && active_point.second == 0.0f &&
              active_seg.mouth_pts_norm.empty())) {
            has_active_point = true;
        }
        preview_segments.push_back(std::move(active_seg));
    }

    std::vector<PourSegment> split_segs;
    split_segs.reserve(preview_segments.size());
    for (auto &seg : preview_segments) {
        split_segment_by_motion(seg, D_SPLIT, self->T_HOLD_F, split_segs);
    }

    auto clusters = build_clusters(split_segs, self->MIN_POUR_DURATION_FRAMES, R_CLUSTER, R_MERGE_VAL);
    std::sort(clusters.begin(), clusters.end(),
              [](const MouldCluster &a, const MouldCluster &b) { return a.cid < b.cid; });

    int active_cluster_cid = -1;
    if (has_active_point) {
        float best_d = FLT_MAX;
        for (auto &c : clusters) {
            int frames = c.total_frames();
            if (frames < self->MIN_CLUSTER_POUR_F) continue;
            float d = l2_dist(active_point.first, active_point.second,
                              c.centroid.first, c.centroid.second);
            if (d < best_d) {
                best_d = d;
                active_cluster_cid = c.cid;
            }
        }
    }

    for (auto &c : clusters) {
        int frames = c.total_frames();
        if (frames >= self->MIN_CLUSTER_POUR_F) {
            if (c.cid == active_cluster_cid) {
                preview.active_cluster_pos = (int)preview.mould_total_frames.size();
            }
            preview.mould_total_frames.push_back(frames);
        }
    }

    preview.mould_count = (int)preview.mould_total_frames.size();
    return preview;
}

/* ================================================================
 * TROLLEY FINALIZATION — close open pours, cluster
 * ================================================================ */

static void finalize_trolley(const GstHiConPouring *self, TrolleyState *st, int current_frame) {
    // Close open pour
    if (st->pour_active && st->current_seg != nullptr) {
        st->current_seg->end_f = current_frame;
        if (st->current_seg->duration_frames() >= self->MIN_POUR_DURATION_FRAMES) {
            st->current_mould_id++;
            st->mould_completed_times[st->current_mould_id] = st->current_seg->duration_frames();
            (*self->trolley_id_to_count)[st->tid] = st->current_mould_id;
            st->completed.push_back(*st->current_seg);
        }
        delete st->current_seg;
        st->current_seg = nullptr;
        st->pour_active = false;
    }
    clear_frozen_probe(st);
    clear_active_probe(st);

    // Split + cluster
    std::vector<PourSegment> split_segs;
    for (auto &seg : st->completed) {
        split_segment_by_motion(seg, D_SPLIT, self->T_HOLD_F, split_segs);
    }
    auto clusters = build_clusters(split_segs, self->MIN_POUR_DURATION_FRAMES, R_CLUSTER, R_MERGE_VAL);

    int mould_count = 0;
    for (auto &c : clusters) {
        if (c.total_frames() >= self->MIN_CLUSTER_POUR_F) mould_count++;
    }

    st->final_clustered_count = mould_count;
    st->last_disappeared_f = current_frame;
}

/* ================================================================
 * NvDsUserMeta copy/release callbacks
 * ================================================================ */

/* NOTE: In DeepStream, copy_func receives (src_NvDsUserMeta, dst_NvDsUserMeta).
 * The `data` param is NvDsUserMeta*, NOT the user_meta_data pointer directly. */
static gpointer hicon_meta_copy_func(gpointer data, gpointer user_data) {
    NvDsUserMeta *src_user_meta = (NvDsUserMeta *)data;
    (void)user_data;
    HiConPouringMeta *src = (HiConPouringMeta *)src_user_meta->user_meta_data;
    if (!src) return NULL;
    HiConPouringMeta *dst = (HiConPouringMeta *)g_malloc0(sizeof(HiConPouringMeta));
    if (!dst) return NULL;
    memcpy(dst, src, sizeof(HiConPouringMeta));
    return (gpointer)dst;
}

/* NOTE: release_func receives (NvDsUserMeta*, NvDsUserMeta*).
 * Free user_meta_data, NOT the NvDsUserMeta (owned by pool). */
static void hicon_meta_release_func(gpointer data, gpointer user_data) {
    (void)user_data;
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    if (user_meta && user_meta->user_meta_data) {
        g_free(user_meta->user_meta_data);
        user_meta->user_meta_data = NULL;
    }
}

/* ================================================================
 * HELPER: configure a text param
 * ================================================================ */

static void setup_text(NvOSD_TextParams *tp, const char *text,
                       int x, int y, float font_size,
                       float r, float g, float b, float a)
{
    if (!tp) return;
    tp->display_text = g_strdup(text);
    tp->x_offset = x;
    tp->y_offset = y;
    tp->font_params.font_name  = (gchar *)"Sans";
    tp->font_params.font_size  = (guint)font_size;
    tp->font_params.font_color = {r, g, b, a};
    tp->set_bg_clr = 0;
}

/* ================================================================
 * HELPER: attach HiConPouringMeta to frame
 * ================================================================ */

static void attach_pouring_meta(NvDsBatchMeta *batch_meta, NvDsFrameMeta *frame_meta,
                                 const HiConPouringMeta &meta_data)
{
    NvDsUserMeta *user_meta = nvds_acquire_user_meta_from_pool(batch_meta);
    if (!user_meta) return;

    HiConPouringMeta *meta = (HiConPouringMeta *)g_malloc0(sizeof(HiConPouringMeta));
    memcpy(meta, &meta_data, sizeof(HiConPouringMeta));

    user_meta->user_meta_data = meta;
    user_meta->base_meta.meta_type = HICON_POURING_META_TYPE;
    user_meta->base_meta.copy_func = hicon_meta_copy_func;
    user_meta->base_meta.release_func = hicon_meta_release_func;

    nvds_add_user_meta_to_frame(frame_meta, user_meta);
}

/* ================================================================
 * RECOMPUTE THRESHOLDS from FPS
 * ================================================================ */

static void recompute_thresholds(GstHiConPouring *self) {
    float fps = self->video_fps;
    self->N_ENTER               = sec_to_frames(SESSION_ENTER_S, fps);
    self->N_EXIT                = sec_to_frames(SESSION_EXIT_S, fps);
    self->K_ON                  = sec_to_frames(POUR_ON_S, fps);
    self->K_OFF                 = sec_to_frames(POUR_OFF_S, fps);
    self->MOULD_SWITCH_HOLD_F   = sec_to_frames(MOULD_SWITCH_HOLD_S, fps);
    self->MIN_POUR_FRAMES       = sec_to_frames(MIN_POUR_S, fps);
    self->MIN_POUR_DURATION_FRAMES = sec_to_frames(MIN_POUR_DURATION_S, fps);
    self->MOUTH_MISSING_TOL     = sec_to_frames(MOUTH_MISSING_TOL_S, fps);
    self->MOUTH_HOLD_DUR        = sec_to_frames(MOUTH_HOLD_S, fps);
    self->T_HOLD_F              = sec_to_frames(T_HOLD_S, fps);
    self->MIN_CLUSTER_POUR_F    = sec_to_frames(MIN_CLUSTER_POUR_S, fps);
}

static void recompute_runtime_geometry(GstHiConPouring *self, int frame_w, int frame_h) {
    if (frame_w <= 0 || frame_h <= 0) {
        return;
    }

    if (self->runtime_frame_w == frame_w && self->runtime_frame_h == frame_h) {
        return;
    }

    const gint ref_w = std::max(1, self->pour_ref_width);
    const gint ref_h = std::max(1, self->pour_ref_height);
    const float scale_x = (float)frame_w / (float)ref_w;
    const float scale_y = (float)frame_h / (float)ref_h;
    const float scale_min = std::min(scale_x, scale_y);

    self->runtime_frame_w = frame_w;
    self->runtime_frame_h = frame_h;
    self->geometry_scale_x = scale_x;
    self->geometry_scale_y = scale_y;
    self->edge_expand_x_px = scale_pixel_value_float(EDGE_EXPAND, scale_x);
    self->edge_expand_y_px = scale_pixel_value_float(EDGE_EXPAND, scale_y);
    self->probe_below_px = scale_pixel_value_int((float)POUR_DOT_BELOW_PX, scale_y, 1);
    self->probe_tail_dy_px = scale_pixel_value_int((float)PROBE_TAIL_DY, scale_y, 1);
    self->probe_radius_px = scale_pixel_value_int((float)PROBE_R, scale_min, 1);
    self->split_min_dx_px = scale_pixel_value_float(MOULD_SPLIT_MIN_DX_PX, scale_x);
    self->split_min_dy_px = scale_pixel_value_float(MOULD_SPLIT_MIN_DY_PX, scale_y);
    self->split_rearm_dx_px = scale_pixel_value_float(MOULD_SPLIT_REARM_DX_PX, scale_x);
    self->split_rearm_dy_px = scale_pixel_value_float(MOULD_SPLIT_REARM_DY_PX, scale_y);

    self->probe_count = 0;
    std::set<std::pair<int, int>> seen;
    for (int i = 0; i < NUM_PROBES; i++) {
        const std::pair<int, int> item = {
            scale_offset_value(PROBE_DX[i], scale_x),
            scale_offset_value(PROBE_DY[i], scale_y),
        };
        if (seen.find(item) != seen.end()) {
            continue;
        }
        seen.insert(item);
        self->probe_dx_scaled[self->probe_count] = item.first;
        self->probe_dy_scaled[self->probe_count] = item.second;
        self->probe_count++;
    }
    if (self->probe_count == 0) {
        self->probe_dx_scaled[0] = 0;
        self->probe_dy_scaled[0] = 0;
        self->probe_count = 1;
    }

    std::ostringstream offsets;
    offsets << "[";
    for (int i = 0; i < self->probe_count; i++) {
        if (i > 0) offsets << ", ";
        offsets << "(" << self->probe_dx_scaled[i] << "," << self->probe_dy_scaled[i] << ")";
    }
    offsets << "]";

    GST_INFO_OBJECT(
        self,
        "Pour geometry scaled: ref=%dx%d actual=%dx%d scale=(%.3f, %.3f) "
        "edge_expand=(%.1f, %.1f) probe_below=%d probe_radius=%d probe_tail_dy=%d "
        "probe_offsets=%s split_min=(%.1f, %.1f) split_rearm=(%.1f, %.1f)",
        ref_w,
        ref_h,
        frame_w,
        frame_h,
        scale_x,
        scale_y,
        self->edge_expand_x_px,
        self->edge_expand_y_px,
        self->probe_below_px,
        self->probe_radius_px,
        self->probe_tail_dy_px,
        offsets.str().c_str(),
        self->split_min_dx_px,
        self->split_min_dy_px,
        self->split_rearm_dx_px,
        self->split_rearm_dy_px
    );
}

static GstPadProbeReturn
hicon_pouring_sink_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    GstHiConPouring *self = (GstHiConPouring *)user_data;
    GstBuffer *inbuf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!inbuf) return GST_PAD_PROBE_OK;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(inbuf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    /* Phase 3 — no gst_buffer_map / NvBufSurfaceMap.
     * Every NVMM buffer map operation (GST_MAP_READ, NvBufSurfaceMap in any
     * flag/sync combination) corrupts the Tegra SoC CUDA IOMMU while nvinfer
     * and nvtracker run concurrently → cudaErrorIllegalAddress crash.
     * Frame dimensions come from NvDsFrameMeta. Pixel brightness is disabled. */

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list;
         l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        int frame_idx = (int)(self->frame_num++);

        bool pixels_mapped = false;
        uint8_t *pixel_data = nullptr;
        uint32_t pixel_pitch = 0;
        int pixel_bpp = 1;
        int frame_w = (int)frame_meta->source_frame_width;
        int frame_h = (int)frame_meta->source_frame_height;
        recompute_runtime_geometry(self, frame_w, frame_h);

        /* Step 1: Parse tracked objects */
        struct Detection {
            uint64_t track_id;
            float    bbox[4];
            float    conf;
            int      class_id;
        };
        std::vector<Detection> trolley_dets, mouth_dets;
        int raw_obj_count = 0;
        int skipped_untracked = 0;
        std::map<int, int> class_hist;

        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list;
             l_obj != NULL; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
            raw_obj_count++;
            class_hist[obj_meta->class_id]++;
            if (obj_meta->object_id == UINT64_MAX) {
                skipped_untracked++;
                continue;
            }
            Detection det;
            det.track_id = obj_meta->object_id;
            det.bbox[0]  = clamp_f(obj_meta->rect_params.left, 0.0f, (float)(frame_w - 1));
            det.bbox[1]  = clamp_f(obj_meta->rect_params.top, 0.0f, (float)(frame_h - 1));
            det.bbox[2]  = clamp_f(obj_meta->rect_params.left + obj_meta->rect_params.width,
                                   0.0f, (float)(frame_w - 1));
            det.bbox[3]  = clamp_f(obj_meta->rect_params.top + obj_meta->rect_params.height,
                                   0.0f, (float)(frame_h - 1));
            det.conf     = obj_meta->confidence;
            det.class_id = obj_meta->class_id;
            if (det.bbox[2] <= det.bbox[0] || det.bbox[3] <= det.bbox[1]) continue;
            if (det.class_id == CLASS_TROLLEY)          trolley_dets.push_back(det);
            else if (det.class_id == CLASS_LADLE_MOUTH) mouth_dets.push_back(det);
        }

        /* Steps 2-4 restored from original probe code below */
        std::set<uint64_t> seen_tids;
        for (auto &tr : trolley_dets) {
            seen_tids.insert(tr.track_id);
            TrolleyState *st = nullptr;
            auto it = self->trolley_states->find(tr.track_id);
            if (it == self->trolley_states->end()) {
                st = new TrolleyState();
                st->tid = tr.track_id;
                (*self->trolley_states)[tr.track_id] = st;
                if (self->trolley_id_to_count->find(tr.track_id) == self->trolley_id_to_count->end())
                    (*self->trolley_id_to_count)[tr.track_id] = 0;
            } else {
                st = it->second;
                if (st->last_disappeared_f >= 0 &&
                    (frame_idx - st->last_disappeared_f) > sec_to_frames(TROLLEY_RESET_S, self->video_fps)) {
                    (*self->trolley_id_to_count)[tr.track_id] = 0;
                    st->final_clustered_count = -1;
                    st->last_disappeared_f = -1;
                }
            }
            memcpy(st->bbox, tr.bbox, sizeof(float) * 4);
            st->bbox_valid = true;
            st->last_seen_f = frame_idx;
        }

        /* Step 3: Finalize disappeared trolleys */
        std::vector<uint64_t> to_remove;
        for (auto &kv : *self->trolley_states) {
            TrolleyState *st = kv.second;
            if (st->last_seen_f >= 0 &&
                (frame_idx - st->last_seen_f) > sec_to_frames(TROLLEY_GONE_S, self->video_fps)) {
                if (!st->completed.empty() || st->session_active ||
                    !st->mould_completed_times.empty()) {
                    finalize_trolley(self, st, frame_idx);
                }
                to_remove.push_back(kv.first);
            }
        }
        for (uint64_t tid : to_remove) {
            delete (*self->trolley_states)[tid];
            self->trolley_states->erase(tid);
        }

        /* Step 4: Associate mouths to trolleys + session state export.
         *
         * The supported Stream 0/2 hybrid architecture is metadata-only here:
         * - C++ owns trolley/mouth association and session gating.
         * - Python owns brightness-based pour ON/OFF, mould counting/clustering,
         *   screenshots, DB writes, heat-cycle aggregation, and sync.
         */
        HiConPouringMeta meta_out;
        memset(&meta_out, 0, sizeof(meta_out));
        meta_out.version = HICON_POURING_META_VERSION;
        meta_out.event = HiConPouringMeta::NONE;

        struct ProbeHit {
            const Detection *det = nullptr;
            float probe_x = 0.0f;
            float probe_y = 0.0f;
        };

        std::unordered_map<uint64_t, std::vector<ProbeHit>> probes_by_trolley;
        probes_by_trolley.reserve(self->trolley_states->size());

        for (auto &kv : *self->trolley_states) {
            TrolleyState *st = kv.second;
            st->active_probe_from_hold = false;
        }

        for (auto &m : mouth_dets) {
            float probe_x = 0.0f;
            float probe_y = 0.0f;
            mouth_probe_point_scaled(m.bbox, self->probe_below_px, probe_x, probe_y);

            TrolleyState *best_st = nullptr;
            float best_center_y = -FLT_MAX;
            for (auto &kv : *self->trolley_states) {
                TrolleyState *st = kv.second;
                bool bbox_fresh = (st->bbox_valid && st->last_seen_f == frame_idx);
                if (!bbox_fresh) continue;
                float tb_exp[4];
                expand_bbox_xy(st->bbox, self->edge_expand_x_px, self->edge_expand_y_px, tb_exp);
                if (!point_in_bbox(probe_x, probe_y, tb_exp)) continue;

                float tcx = 0.0f, tcy = 0.0f;
                bbox_center(st->bbox, tcx, tcy);
                if (!best_st || tcy > best_center_y) {
                    best_st = st;
                    best_center_y = tcy;
                }
            }

            if (!best_st) continue;

            ProbeHit hit;
            hit.det = &m;
            hit.probe_x = probe_x;
            hit.probe_y = probe_y;
            probes_by_trolley[best_st->tid].push_back(hit);
        }

        auto meta_priority = [](const HiConPouringMeta &meta) -> int {
            if (meta.event == HiConPouringMeta::SESSION_START ||
                meta.event == HiConPouringMeta::SESSION_END) {
                return 4;
            }
            if (meta.session_active) {
                return meta.mouth_present_in_trolley ? 3 : 2;
            }
            if (meta.mouth_present_in_trolley || meta.probe_valid) {
                return 1;
            }
            return 0;
        };
        int best_meta_priority = -1;

        for (auto &kv : *self->trolley_states) {
            TrolleyState *st = kv.second;
            if (!st->bbox_valid) continue;

            bool bbox_fresh = (st->last_seen_f == frame_idx);
            auto probes_it = probes_by_trolley.find(st->tid);
            std::vector<ProbeHit> *probes =
                (probes_it != probes_by_trolley.end()) ? &probes_it->second : nullptr;
            bool has_current_probe = (probes && !probes->empty());
            const ProbeHit *selected = nullptr;

            if (has_current_probe) {
                for (auto &probe : *probes) {
                    if (probe.det->track_id == st->active_probe_track_id) {
                        selected = &probe;
                    }
                }
                if (!selected) {
                    selected = &(*probes)[0];
                    for (auto &probe : *probes) {
                        if (probe.det->conf > selected->det->conf) {
                            selected = &probe;
                        }
                    }
                }
            }

            if (selected) {
                st->active_probe_track_id = selected->det->track_id;
                memcpy(st->active_probe_bbox, selected->det->bbox, sizeof(float) * 4);
                st->active_probe_bbox_valid = true;
                st->active_probe_pt_px[0] = selected->probe_x;
                st->active_probe_pt_px[1] = selected->probe_y;
                st->active_probe_pt_valid = true;
                st->active_probe_last_seen_f = frame_idx;
                st->active_probe_from_hold = false;
                st->in_count++;
                st->out_count = 0;
            } else {
                st->in_count = 0;
                st->out_count++;
            }

            /* Session start */
            HiConPouringMeta local_meta;
            memset(&local_meta, 0, sizeof(local_meta));
            local_meta.version = HICON_POURING_META_VERSION;
            local_meta.event = HiConPouringMeta::NONE;
            local_meta.trolley_track_id = st->tid;
            memcpy(local_meta.trolley_bbox, st->bbox, sizeof(float) * 4);

            if (!st->session_active && has_current_probe && st->out_count == 0 && st->in_count >= self->N_ENTER) {
                st->session_active  = true;
                st->session_start_f = frame_idx - self->N_ENTER + 1;
                st->session_end_f   = -1;
                clear_frozen_probe(st);
                clear_active_probe(st);
                local_meta.event = HiConPouringMeta::SESSION_START;
            }

            /* Session end */
            if (st->session_active) {
                int probe_missing = frame_idx - st->active_probe_last_seen_f;
                if (st->out_count >= self->N_EXIT || probe_missing > self->MOUTH_MISSING_TOL) {
                    st->session_active = false;
                    st->session_end_f  = frame_idx;
                    clear_frozen_probe(st);
                    clear_active_probe(st);
                    local_meta.event = HiConPouringMeta::SESSION_END;
                    int priority = meta_priority(local_meta);
                    if (priority > best_meta_priority) {
                        meta_out = local_meta;
                        best_meta_priority = priority;
                    }
                    continue;
                }
            }

            bool use_probe = false;
            bool mouth_present_in_trolley = false;
            float mouth_bbox_use[4] = {0, 0, 0, 0};
            float probe_x_use = 0.0f, probe_y_use = 0.0f;
            float mouth_nx = -1.0f, mouth_ny = -1.0f;
            uint64_t mouth_track_id = UINT64_MAX;

            if (selected) {
                probe_x_use = selected->probe_x;
                probe_y_use = selected->probe_y;
                memcpy(mouth_bbox_use, selected->det->bbox, sizeof(float) * 4);
                mouth_track_id = selected->det->track_id;
                mouth_present_in_trolley = true;
                use_probe = true;
            } else if (st->active_probe_pt_valid &&
                       (frame_idx - st->active_probe_last_seen_f) <= self->MOUTH_HOLD_DUR) {
                probe_x_use = st->active_probe_pt_px[0];
                probe_y_use = st->active_probe_pt_px[1];
                if (st->active_probe_bbox_valid) {
                    memcpy(mouth_bbox_use, st->active_probe_bbox, sizeof(float) * 4);
                }
                mouth_track_id = st->active_probe_track_id;
                st->active_probe_from_hold = true;
                use_probe = true;
            }

            local_meta.session_active = st->session_active ? 1u : 0u;
            local_meta.mouth_present_in_trolley = mouth_present_in_trolley ? 1u : 0u;
            local_meta.probe_valid = use_probe ? 1u : 0u;
            local_meta.mouth_track_id = (mouth_track_id == UINT64_MAX) ? 0 : mouth_track_id;

            if (use_probe) {
                local_meta.probe_x_px = probe_x_use;
                local_meta.probe_y_px = probe_y_use;
                memcpy(local_meta.mouth_bbox, mouth_bbox_use, sizeof(float) * 4);
                norm_point_in_expanded_trolley(
                    probe_x_use,
                    probe_y_use,
                    st->bbox,
                    self->edge_expand_x_px,
                    self->edge_expand_y_px,
                    mouth_nx,
                    mouth_ny
                );
                if (bbox_fresh && mouth_nx >= 0.0f && mouth_ny >= 0.0f) {
                    local_meta.mouth_norm_x = mouth_nx;
                    local_meta.mouth_norm_y = mouth_ny;
                } else {
                    local_meta.mouth_norm_x = -1.0f;
                    local_meta.mouth_norm_y = -1.0f;
                }
            } else {
                local_meta.mouth_norm_x = -1.0f;
                local_meta.mouth_norm_y = -1.0f;
            }

            int priority = meta_priority(local_meta);
            if (priority > best_meta_priority) {
                meta_out = local_meta;
                best_meta_priority = priority;
            }
        }

        if (self->meta_attach_enabled) {
            attach_pouring_meta(batch_meta, frame_meta, meta_out);
        }

        /* Step 5: Cheap OSD via object metadata — only safe with downstream nvdsosd. */
        if (self->enable_osd) {
            for (NvDsMetaList *l_obj = frame_meta->obj_meta_list;
                 l_obj != NULL; l_obj = l_obj->next)
            {
                NvDsObjectMeta *obj = (NvDsObjectMeta *)(l_obj->data);
                if (obj->object_id == UINT64_MAX) continue;

                if (obj->class_id == CLASS_TROLLEY) {
                    TrolleyState *st = nullptr;
                    auto it = self->trolley_states->find(obj->object_id);
                    if (it != self->trolley_states->end()) st = it->second;
                    bool pouring = (st && st->pour_active);
                    obj->rect_params.border_width = 3;
                    obj->rect_params.border_color = pouring ?
                        (NvOSD_ColorParams){0.0f, 1.0f, 0.0f, 1.0f} :
                        (NvOSD_ColorParams){1.0f, 1.0f, 0.0f, 1.0f};
                    obj->rect_params.has_bg_color = 0;

                    char label[128];
                    int cnt = 0;
                    auto cit = self->trolley_id_to_count->find(obj->object_id);
                    if (cit != self->trolley_id_to_count->end()) cnt = cit->second;
                    if (cnt > 0) snprintf(label, sizeof(label), "T%lu (%dM)", (unsigned long)obj->object_id, cnt);
                    else snprintf(label, sizeof(label), "T%lu", (unsigned long)obj->object_id);

                    if (obj->text_params.display_text) g_free(obj->text_params.display_text);
                    obj->text_params.display_text = g_strdup(label);
                    obj->text_params.font_params.font_name = (gchar *)"Sans";
                    obj->text_params.font_params.font_size = 12;
                    obj->text_params.font_params.font_color = obj->rect_params.border_color;
                    obj->text_params.set_bg_clr = 0;

                } else if (obj->class_id == CLASS_LADLE_MOUTH) {
                    obj->rect_params.border_width = 3;
                    obj->rect_params.border_color = {1.0f, 0.0f, 1.0f, 1.0f};
                    obj->rect_params.has_bg_color = 0;
                    char label[64];
                    snprintf(label, sizeof(label), "M%lu", (unsigned long)obj->object_id);
                    if (obj->text_params.display_text) g_free(obj->text_params.display_text);
                    obj->text_params.display_text = g_strdup(label);
                    obj->text_params.font_params.font_name = (gchar *)"Sans";
                    obj->text_params.font_params.font_size = 10;
                    obj->text_params.font_params.font_color = {1.0f, 0.0f, 1.0f, 1.0f};
                    obj->text_params.set_bg_clr = 0;
                }
            }

            /* Deferred rich OSD: panel/probe display-meta needs guaranteed downstream
             * nvdsosd consumption and cached cluster previews before it is safe to re-enable. */
            constexpr bool kEnableDeferredRichOsd = false;
            if (kEnableDeferredRichOsd) {
                DisplayMetaHelper dm(batch_meta, frame_meta);
                if (!dm.exhausted) {
                    std::unordered_map<uint64_t, ClusterPreview> overlay_preview_by_tid;
                    overlay_preview_by_tid.reserve(self->trolley_states->size());
                    for (auto &kv : *self->trolley_states) {
                        TrolleyState *st = kv.second;
                        if (st->session_start_f >= 0) {
                            overlay_preview_by_tid.emplace(
                                kv.first,
                                build_cluster_preview_for_overlay(self, st, frame_idx));
                        }
                    }

                    {
                        struct TrolleyInfo { uint64_t tid; const ClusterPreview *preview; };
                        std::vector<TrolleyInfo> active_info;
                        for (auto &kv : *self->trolley_states) {
                            TrolleyState *s = kv.second;
                            if (s->session_start_f >= 0) {
                                auto pit = overlay_preview_by_tid.find(kv.first);
                                const ClusterPreview *preview = (pit != overlay_preview_by_tid.end()) ? &pit->second : nullptr;
                                int cnt = preview ? preview->mould_count : 0;
                                if (cnt > 0 || s->pour_active)
                                    active_info.push_back({kv.first, preview});
                            }
                        }
                        if (!active_info.empty() && !dm.exhausted) {
                            int panel_h = 30;
                            for (auto &ai : active_info) {
                                panel_h += 25 + 22;
                                int nm = ai.preview ? (int)ai.preview->mould_total_frames.size() : 0;
                                panel_h += nm * 20 + 10;
                            }
                            panel_h = std::max(60, panel_h);
                            NvOSD_RectParams *bg = dm.next_rect();
                            if (bg) {
                                bg->left = 10; bg->top = 10; bg->width = 350; bg->height = (unsigned int)panel_h;
                                bg->border_width = 2; bg->border_color = {1.0f, 1.0f, 0.0f, 1.0f};
                                bg->has_bg_color = 1; bg->bg_color = {0.0f, 0.0f, 0.0f, 0.6f};

                                int y_off = 40;
                                for (auto &ai : active_info) {
                                    char txt[256];
                                    snprintf(txt, sizeof(txt), "Trolley #%lu", (unsigned long)ai.tid);
                                    setup_text(dm.next_text(), txt, 25, y_off, 10, 1.0f, 1.0f, 1.0f, 1.0f);
                                    y_off += 25;
                                    int cnt = ai.preview ? ai.preview->mould_count : 0;
                                    snprintf(txt, sizeof(txt), "  Total Moulds: %d", cnt);
                                    setup_text(dm.next_text(), txt, 25, y_off, 9, 0.78f, 0.78f, 0.78f, 1.0f);
                                    y_off += 22;
                                    int num_moulds = ai.preview ? (int)ai.preview->mould_total_frames.size() : 0;
                                    for (int m = 0; m < num_moulds; m++) {
                                        float total_s = (float)ai.preview->mould_total_frames[m] / self->video_fps;
                                        bool is_pouring = (ai.preview && m == ai.preview->active_cluster_pos);
                                        if (is_pouring) {
                                            snprintf(txt, sizeof(txt), "  Mould #%d: %.1fs (pouring)", m + 1, total_s);
                                            setup_text(dm.next_text(), txt, 25, y_off, 8, 0.0f, 1.0f, 0.0f, 1.0f);
                                        } else {
                                            snprintf(txt, sizeof(txt), "  Mould #%d: %.1fs", m + 1, total_s);
                                            setup_text(dm.next_text(), txt, 25, y_off, 8, 0.78f, 0.78f, 0.78f, 1.0f);
                                        }
                                        y_off += 20;
                                    }
                                    y_off += 10;
                                }
                            }
                        }
                    }
                }
            }
        }

    } /* end per-frame loop */

    return GST_PAD_PROBE_OK;
}

/* ================================================================
 * transform_ip — stub (all processing runs in hicon_pouring_sink_probe)
 *
 * With passthrough=TRUE, GStreamer never calls this. Kept as a required
 * GstBaseTransform vfunc registration.
 * ================================================================ */

static GstFlowReturn
gst_hicon_pouring_transform_ip(GstBaseTransform *btrans, GstBuffer *inbuf)
{
    (void)btrans; (void)inbuf;
    return GST_FLOW_OK;
}

/* OLD transform_ip code removed — was ~560 lines of dead code duplicating
 * the sink pad probe logic. All processing now lives exclusively in
 * hicon_pouring_sink_probe() above.
 */

#if 0 /* === Dead transform_ip code — kept for reference only === */
static GstFlowReturn
gst_hicon_pouring_transform_ip_DEAD(GstBaseTransform *btrans, GstBuffer *inbuf)
{
    GstHiConPouring *self = GST_HICON_POURING(btrans);

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(inbuf);
    if (!batch_meta) {
        return GST_FLOW_OK;
    }

    /* Phase 3 — no gst_buffer_map / NvBufSurfaceMap (same reason as probe function). */

    /* --- Process each frame in batch --- */
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list;
         l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        int frame_idx = (int)(self->frame_num++);
        int batch_idx = frame_meta->batch_id;

        bool pixels_mapped = false;
        uint8_t *pixel_data = nullptr;
        uint32_t pixel_pitch = 0;
        int pixel_bpp = 1;
        int frame_w = (int)frame_meta->source_frame_width;
        int frame_h = (int)frame_meta->source_frame_height;

        /* ============================================================
         * STEP 1: Parse tracked objects — separate by class
         * ============================================================ */
        struct Detection {
            uint64_t track_id;
            float    bbox[4];  // x1, y1, x2, y2
            float    conf;
            int      class_id;
        };

        std::vector<Detection> trolley_dets, mouth_dets;

        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list;
             l_obj != NULL; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);

            if (obj_meta->object_id == UINT64_MAX) continue;

            Detection det;
            det.track_id = obj_meta->object_id;
            det.bbox[0]  = obj_meta->rect_params.left;
            det.bbox[1]  = obj_meta->rect_params.top;
            det.bbox[2]  = obj_meta->rect_params.left + obj_meta->rect_params.width;
            det.bbox[3]  = obj_meta->rect_params.top  + obj_meta->rect_params.height;
            det.conf     = obj_meta->confidence;
            det.class_id = obj_meta->class_id;

            if (det.class_id == CLASS_TROLLEY)          trolley_dets.push_back(det);
            else if (det.class_id == CLASS_LADLE_MOUTH) mouth_dets.push_back(det);
        }

        /* ============================================================
         * STEP 2: Update trolley states
         * ============================================================ */
        std::set<uint64_t> seen_tids;
        for (auto &tr : trolley_dets) {
            seen_tids.insert(tr.track_id);
            TrolleyState *st = nullptr;
            auto it = self->trolley_states->find(tr.track_id);
            if (it == self->trolley_states->end()) {
                st = new TrolleyState();
                st->tid = tr.track_id;
                (*self->trolley_states)[tr.track_id] = st;
                if (self->trolley_id_to_count->find(tr.track_id) == self->trolley_id_to_count->end())
                    (*self->trolley_id_to_count)[tr.track_id] = 0;
            } else {
                st = it->second;
                if (st->last_disappeared_f >= 0 &&
                    (frame_idx - st->last_disappeared_f) > sec_to_frames(TROLLEY_RESET_S, self->video_fps)) {
                    (*self->trolley_id_to_count)[tr.track_id] = 0;
                    st->final_clustered_count = -1;
                    st->last_disappeared_f = -1;
                }
            }
            memcpy(st->bbox, tr.bbox, sizeof(float) * 4);
            st->bbox_valid = true;
            st->last_seen_f = frame_idx;
        }

        /* ============================================================
         * STEP 3: Finalize disappeared trolleys
         * ============================================================ */
        std::vector<uint64_t> to_remove;
        for (auto &kv : *self->trolley_states) {
            TrolleyState *st = kv.second;
            if (st->last_seen_f >= 0 &&
                (frame_idx - st->last_seen_f) > sec_to_frames(TROLLEY_GONE_S, self->video_fps)) {
                if (!st->completed.empty() || st->session_active ||
                    !st->mould_completed_times.empty()) {
                    finalize_trolley(self, st, frame_idx);
                }
                to_remove.push_back(kv.first);
            }
        }
        for (uint64_t tid : to_remove) {
            delete (*self->trolley_states)[tid];
            self->trolley_states->erase(tid);
        }

        /* ============================================================
         * STEP 4: Associate mouths to trolleys + full state machine
         * ============================================================ */
        HiConPouringMeta meta_out;
        memset(&meta_out, 0, sizeof(meta_out));
        meta_out.event = HiConPouringMeta::NONE;

        for (auto &kv : *self->trolley_states) {
            TrolleyState *st = kv.second;
            if (!st->bbox_valid) continue;

            float tb_exp[4];
            expand_bbox(st->bbox, EDGE_EXPAND, tb_exp);

            const Detection *best_mouth = nullptr;
            for (auto &m : mouth_dets) {
                float mcx, mcy;
                bbox_center(m.bbox, mcx, mcy);
                if (point_in_bbox(mcx, mcy, tb_exp)) {
                    if (!best_mouth || m.conf > best_mouth->conf)
                        best_mouth = &m;
                }
            }

            if (best_mouth) {
                float mx, my;
                bbox_center(best_mouth->bbox, mx, my);
                st->last_mouth_seen_f = frame_idx;
                st->last_mouth_pt_px[0] = mx;
                st->last_mouth_pt_px[1] = my;
                st->last_mouth_valid = true;
                memcpy(st->last_mouth_bbox, best_mouth->bbox, sizeof(float) * 4);
                st->last_mouth_bbox_valid = true;
                st->in_count++;
                st->out_count = 0;
            } else {
                st->in_count = 0;
                st->out_count++;
            }

            /* ---- Session start ---- */
            if (!st->session_active && st->out_count == 0 && st->in_count >= self->N_ENTER) {
                st->session_active  = true;
                st->session_start_f = frame_idx - self->N_ENTER + 1;
                st->session_end_f   = -1;
                st->pour_active     = false;
                st->pour_on_count   = 0;
                st->pour_off_count  = 0;
                delete st->current_seg;
                st->current_seg = nullptr;

                meta_out.event = HiConPouringMeta::SESSION_START;
                meta_out.trolley_track_id = st->tid;
            }

            /* ---- Session end ---- */
            if (st->session_active) {
                int mouth_missing = frame_idx - st->last_mouth_seen_f;
                if (st->out_count >= self->N_EXIT || mouth_missing > self->MOUTH_MISSING_TOL) {
                    // Close open pour
                    if (st->pour_active && st->current_seg != nullptr) {
                        st->current_seg->end_f = frame_idx;
                        if (st->current_seg->duration_frames() >= self->MIN_POUR_DURATION_FRAMES) {
                            st->current_mould_id++;
                            st->mould_completed_times[st->current_mould_id] = st->current_seg->duration_frames();
                            (*self->trolley_id_to_count)[st->tid] = st->current_mould_id;
                            st->completed.push_back(*st->current_seg);
                        }
                    }
                    st->pour_active = false;
                    delete st->current_seg;
                    st->current_seg = nullptr;
                    st->session_active = false;
                    st->session_end_f  = frame_idx;

                    /* Cluster completed segments for final mould count */
                    {
                        std::vector<PourSegment> split_segs;
                        for (auto &seg : st->completed) {
                            split_segment_by_motion(seg, D_SPLIT, self->T_HOLD_F, split_segs);
                        }
                        auto clusters = build_clusters(split_segs, self->MIN_POUR_DURATION_FRAMES,
                                                        R_CLUSTER, R_MERGE_VAL);
                        int mi = 0;
                        for (auto &c : clusters) {
                            if (c.total_frames() >= self->MIN_CLUSTER_POUR_F && mi < 32) {
                                meta_out.per_mould_times[mi] =
                                    (float)c.total_frames() / self->video_fps;
                                mi++;
                            }
                        }
                        meta_out.per_mould_count = mi;
                        meta_out.mould_count = mi;
                    }

                    meta_out.event = HiConPouringMeta::SESSION_END;
                    meta_out.trolley_track_id = st->tid;
                    meta_out.session_duration_s = (float)(frame_idx - st->session_start_f) / self->video_fps;
                    continue;
                }

                /* ---- Pour logic ---- */
                bool use_mouth = false;
                float mouth_bbox_use[4] = {0,0,0,0};

                if (best_mouth) {
                    memcpy(mouth_bbox_use, st->last_mouth_bbox, sizeof(float) * 4);
                    use_mouth = true;
                } else if (st->last_mouth_bbox_valid &&
                           (frame_idx - st->last_mouth_seen_f) <= self->MOUTH_HOLD_DUR) {
                    memcpy(mouth_bbox_use, st->last_mouth_bbox, sizeof(float) * 4);
                    use_mouth = true;
                }

                if (!use_mouth) continue;

                float pour_dot_x = (mouth_bbox_use[0] + mouth_bbox_use[2]) / 2.0f;
                float pour_dot_y = mouth_bbox_use[3] + POUR_DOT_BELOW_PX;

                float fs = 0.0f;
                if (pixels_mapped) {
                    fs = flare_score(pixel_data, pixel_pitch, frame_w, frame_h,
                                     pour_dot_x, pour_dot_y, pixel_bpp);
                }

                /* Verbose brightness logging during active sessions (every 5th frame) */
                if (st->session_active && (frame_idx % 5) == 0) {
                    g_print("[CPP-BRIGHTNESS][%s] tid=%lu frame=%d dot=(%.0f,%.0f) fs=%.1f "
                            "TH_ON=%.0f TH_OFF=%.0f on_cnt=%d off_cnt=%d pour=%s mapped=%d bpp=%d\n",
                            GST_ELEMENT_NAME(self),
                            (unsigned long)st->tid, frame_idx,
                            pour_dot_x, pour_dot_y, fs,
                            TH_ON, TH_OFF,
                            st->pour_on_count, st->pour_off_count,
                            st->pour_active ? "ON" : "off",
                            pixels_mapped ? 1 : 0, pixel_bpp);
                }

                float mouth_nx = -1.0f, mouth_ny = -1.0f;
                norm_point_in_trolley(pour_dot_x, pour_dot_y, st->bbox, mouth_nx, mouth_ny);
                bool mouth_norm_valid = (mouth_nx >= 0.0f);

                if (!st->pour_active) {
                    /* ---- Pour OFF → check for pour start ---- */
                    st->pour_on_count = (fs > TH_ON) ? (st->pour_on_count + 1) : 0;

                    if (st->pour_on_count >= self->K_ON && mouth_norm_valid) {
                        st->pour_active = true;
                        int start_f = frame_idx - self->K_ON + 1;
                        st->current_seg = new PourSegment();
                        st->current_seg->start_f = start_f;
                        st->current_seg->end_f   = frame_idx;
                        st->current_seg->mouth_pts_norm.push_back({mouth_nx, mouth_ny});
                        st->pour_off_count = 0;
                        st->mould_switch_count = 0;
                        st->mould_anchor_pt[0] = mouth_nx;
                        st->mould_anchor_pt[1] = mouth_ny;
                        st->mould_anchor_valid = true;

                        meta_out.event = HiConPouringMeta::POUR_START;
                        meta_out.trolley_track_id = st->tid;
                        memcpy(meta_out.mouth_bbox, mouth_bbox_use, sizeof(float) * 4);
                    }
                } else {
                    /* ---- Pour ON → check displacement + pour-off ---- */

                    // Anchor-based mould switch detection
                    if (mouth_norm_valid && st->mould_anchor_valid) {
                        float dx = std::abs(mouth_nx - st->mould_anchor_pt[0]);
                        float dy = std::abs(mouth_ny - st->mould_anchor_pt[1]);

                        if (dx > D_SPLIT || dy > D_SPLIT) {
                            st->mould_switch_count++;
                            if (st->mould_switch_count >= self->MOULD_SWITCH_HOLD_F) {
                                st->current_seg->end_f = frame_idx - self->MOULD_SWITCH_HOLD_F + 1;
                                int left_dur = st->current_seg->duration_frames();
                                if (left_dur >= self->MIN_POUR_DURATION_FRAMES) {
                                    st->current_mould_id++;
                                    st->mould_completed_times[st->current_mould_id] = left_dur;
                                    st->completed.push_back(*st->current_seg);
                                    (*self->trolley_id_to_count)[st->tid] = st->current_mould_id;

                                    delete st->current_seg;
                                    st->current_seg = new PourSegment();
                                    st->current_seg->start_f = frame_idx - self->MOULD_SWITCH_HOLD_F + 1;
                                    st->current_seg->end_f   = frame_idx;
                                    st->current_seg->mouth_pts_norm.push_back({mouth_nx, mouth_ny});
                                    st->mould_anchor_pt[0] = mouth_nx;
                                    st->mould_anchor_pt[1] = mouth_ny;
                                    st->mould_switch_count = 0;

                                    meta_out.event = HiConPouringMeta::MOULD_SPLIT;
                                    meta_out.trolley_track_id = st->tid;
                                } else {
                                    st->mould_switch_count = 0;
                                }
                            }
                        } else {
                            st->mould_switch_count = 0;
                        }
                    }

                    // Update current segment
                    st->current_seg->end_f = frame_idx;
                    if (mouth_norm_valid) {
                        st->current_seg->mouth_pts_norm.push_back({mouth_nx, mouth_ny});
                    }

                    // Pour-off check
                    st->pour_off_count = (fs < TH_OFF) ? (st->pour_off_count + 1) : 0;

                    if (st->pour_off_count >= self->K_OFF) {
                        st->current_seg->end_f = frame_idx - self->K_OFF + 1;
                        float pour_dur_s = (float)st->current_seg->duration_frames() / self->video_fps;
                        if (st->current_seg->duration_frames() >= self->MIN_POUR_DURATION_FRAMES) {
                            st->current_mould_id++;
                            st->mould_completed_times[st->current_mould_id] =
                                st->current_seg->duration_frames();
                            (*self->trolley_id_to_count)[st->tid] = st->current_mould_id;
                            st->completed.push_back(*st->current_seg);
                        }
                        st->pour_active = false;
                        delete st->current_seg;
                        st->current_seg = nullptr;
                        st->pour_on_count  = 0;
                        st->pour_off_count = 0;
                        st->mould_switch_count = 0;

                        meta_out.event = HiConPouringMeta::POUR_END;
                        meta_out.trolley_track_id = st->tid;
                        meta_out.pour_duration_s = pour_dur_s;
                        memcpy(meta_out.mouth_bbox, mouth_bbox_use, sizeof(float) * 4);
                    }
                }

                /* Populate state snapshot in meta */
                meta_out.session_active = st->session_active;
                meta_out.pour_active = st->pour_active;
                meta_out.current_mould_id = st->current_mould_id;
                meta_out.brightness = fs;
                meta_out.trolley_track_id = st->tid;
            }
        } /* end per-trolley loop */

        /* BISECT: disable meta attachment to isolate crash */
        /* attach_pouring_meta(batch_meta, frame_meta, meta_out); */
        (void)meta_out;

        /* ============================================================
         * STEP 5: Update display metadata (bbox colors, labels, overlay)
         * BISECT: disabled to isolate crash
         * ============================================================ */
        if (false) {
        DisplayMetaHelper dm(batch_meta, frame_meta);
        std::unordered_map<uint64_t, ClusterPreview> overlay_preview_by_tid;
        overlay_preview_by_tid.reserve(self->trolley_states->size());
        for (auto &kv : *self->trolley_states) {
            TrolleyState *st = kv.second;
            if (st->session_start_f >= 0) {
                overlay_preview_by_tid.emplace(
                    kv.first,
                    build_cluster_preview_for_overlay(self, st, frame_idx));
            }
        }

        /* --- Modify existing object bbox colors and labels --- */
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list;
             l_obj != NULL; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj = (NvDsObjectMeta *)(l_obj->data);
            if (obj->object_id == UINT64_MAX) continue;

            if (obj->class_id == CLASS_TROLLEY) {
                TrolleyState *st = nullptr;
                auto it = self->trolley_states->find(obj->object_id);
                if (it != self->trolley_states->end()) st = it->second;
                bool pouring = (st && st->pour_active);

                obj->rect_params.border_width = 3;
                if (pouring) {
                    obj->rect_params.border_color = {0.0f, 1.0f, 0.0f, 1.0f};
                } else {
                    obj->rect_params.border_color = {1.0f, 1.0f, 0.0f, 1.0f};
                }
                obj->rect_params.has_bg_color = 0;

                char label[128];
                int cnt = 0;
                auto pit = overlay_preview_by_tid.find(obj->object_id);
                if (pit != overlay_preview_by_tid.end()) {
                    cnt = pit->second.mould_count;
                } else {
                    auto cit = self->trolley_id_to_count->find(obj->object_id);
                    if (cit != self->trolley_id_to_count->end()) cnt = cit->second;
                }
                if (cnt > 0)
                    snprintf(label, sizeof(label), "T%lu (%dM)", (unsigned long)obj->object_id, cnt);
                else
                    snprintf(label, sizeof(label), "T%lu", (unsigned long)obj->object_id);

                if (obj->text_params.display_text)
                    g_free(obj->text_params.display_text);
                obj->text_params.display_text = g_strdup(label);
                obj->text_params.font_params.font_name  = (gchar *)"Sans";
                obj->text_params.font_params.font_size  = 12;
                obj->text_params.font_params.font_color = obj->rect_params.border_color;
                obj->text_params.set_bg_clr = 0;

            } else if (obj->class_id == CLASS_LADLE_MOUTH) {
                obj->rect_params.border_width = 3;
                obj->rect_params.border_color = {1.0f, 0.0f, 1.0f, 1.0f};
                obj->rect_params.has_bg_color = 0;

                char label[64];
                snprintf(label, sizeof(label), "M%lu", (unsigned long)obj->object_id);
                if (obj->text_params.display_text)
                    g_free(obj->text_params.display_text);
                obj->text_params.display_text = g_strdup(label);
                obj->text_params.font_params.font_name  = (gchar *)"Sans";
                obj->text_params.font_params.font_size  = 10;
                obj->text_params.font_params.font_color = {1.0f, 0.0f, 1.0f, 1.0f};
                obj->text_params.set_bg_clr = 0;

                // Pour dot visualization
                if (pixels_mapped) {
                    float dot_x = obj->rect_params.left + obj->rect_params.width / 2.0f;
                    float dot_y = obj->rect_params.top + obj->rect_params.height + self->probe_below_px;
                    float fs_vis = flare_score(pixel_data, pixel_pitch, frame_w, frame_h,
                                               dot_x, dot_y, pixel_bpp);
                    if (fs_vis >= TH_ON) {
                        NvOSD_CircleParams *cp = dm.next_circle();
                        cp->xc = (unsigned int)dot_x;
                        cp->yc = (unsigned int)dot_y;
                        cp->radius = 10;
                        cp->circle_color = {0.0f, 1.0f, 0.0f, 1.0f};
                        cp->has_bg_color = 1;
                        cp->bg_color = {0.0f, 1.0f, 0.0f, 1.0f};
                    }
                }
            }
        }
        /* --- Info panel overlay (top-left) --- */
        {
            struct TrolleyInfo {
                uint64_t tid;
                const ClusterPreview *preview;
            };
            std::vector<TrolleyInfo> active_info;
            for (auto &kv : *self->trolley_states) {
                TrolleyState *s = kv.second;
                if (s->session_start_f >= 0) {
                    auto pit = overlay_preview_by_tid.find(kv.first);
                    const ClusterPreview *preview =
                        (pit != overlay_preview_by_tid.end()) ? &pit->second : nullptr;
                    int cnt = preview ? preview->mould_count : 0;
                    if (cnt > 0 || s->pour_active)
                        active_info.push_back({kv.first, preview});
                }
            }

            if (!active_info.empty()) {
                int panel_h = 30;
                for (auto &ai : active_info) {
                    panel_h += 25 + 22;
                    int nm = ai.preview ? (int)ai.preview->mould_total_frames.size() : 0;
                    panel_h += nm * 20 + 10;
                }
                panel_h = std::max(60, panel_h);

                NvOSD_RectParams *bg = dm.next_rect();
                bg->left   = 10;
                bg->top    = 10;
                bg->width  = 350;
                bg->height = (unsigned int)panel_h;
                bg->border_width  = 2;
                bg->border_color  = {1.0f, 1.0f, 0.0f, 1.0f};
                bg->has_bg_color  = 1;
                bg->bg_color      = {0.0f, 0.0f, 0.0f, 0.6f};

                int y_off = 40;
                for (auto &ai : active_info) {
                    char txt[256];

                    snprintf(txt, sizeof(txt), "Trolley #%lu", (unsigned long)ai.tid);
                    setup_text(dm.next_text(), txt, 25, y_off, 10, 1.0f, 1.0f, 1.0f, 1.0f);
                    y_off += 25;

                    int cnt = ai.preview ? ai.preview->mould_count : 0;
                    snprintf(txt, sizeof(txt), "  Total Moulds: %d", cnt);
                    setup_text(dm.next_text(), txt, 25, y_off, 9, 0.78f, 0.78f, 0.78f, 1.0f);
                    y_off += 22;

                    int num_moulds = ai.preview ? (int)ai.preview->mould_total_frames.size() : 0;
                    for (int m = 0; m < num_moulds; m++) {
                        float total_s = (float)ai.preview->mould_total_frames[m] / self->video_fps;
                        bool is_pouring = (ai.preview && m == ai.preview->active_cluster_pos);
                        if (is_pouring) {
                            snprintf(txt, sizeof(txt), "  Mould #%d: %.1fs (pouring)", m + 1, total_s);
                            setup_text(dm.next_text(), txt, 25, y_off, 8,
                                       0.0f, 1.0f, 0.0f, 1.0f);
                        } else {
                            snprintf(txt, sizeof(txt), "  Mould #%d: %.1fs", m + 1, total_s);
                            setup_text(dm.next_text(), txt, 25, y_off, 8,
                                       0.78f, 0.78f, 0.78f, 1.0f);
                        }
                        y_off += 20;
                    }
                    y_off += 10;
                }
            }
        }
        } /* end BISECT: if(false) OSD disabled */

    } /* end per-frame loop */

    return GST_FLOW_OK;
}
#endif /* === End dead transform_ip code === */

/* ================================================================
 * GstBaseTransform vfuncs: set_caps, start, stop
 * ================================================================ */

static gboolean
gst_hicon_pouring_set_caps(GstBaseTransform *btrans, GstCaps *incaps, GstCaps *outcaps)
{
    GstHiConPouring *self = GST_HICON_POURING(btrans);
    gst_video_info_from_caps(&self->video_info, incaps);

    /* Extract FPS from caps */
    gint fps_n = GST_VIDEO_INFO_FPS_N(&self->video_info);
    gint fps_d = GST_VIDEO_INFO_FPS_D(&self->video_info);
    if (fps_n > 0 && fps_d > 0) {
        self->video_fps = (float)fps_n / (float)fps_d;
    } else {
        self->video_fps = 25.0f;  /* default */
    }

    GST_INFO_OBJECT(self, "Caps set: %dx%d @ %.2f fps",
                     GST_VIDEO_INFO_WIDTH(&self->video_info),
                     GST_VIDEO_INFO_HEIGHT(&self->video_info),
                     self->video_fps);

    recompute_thresholds(self);
    recompute_runtime_geometry(
        self,
        GST_VIDEO_INFO_WIDTH(&self->video_info),
        GST_VIDEO_INFO_HEIGHT(&self->video_info)
    );
    return TRUE;
}

static gboolean
gst_hicon_pouring_start(GstBaseTransform *btrans)
{
    GstHiConPouring *self = GST_HICON_POURING(btrans);
    const gchar *element_name = GST_ELEMENT_NAME(self);

    /* Detect integrated GPU (Jetson) */
    int val = -1;
    cudaDeviceGetAttribute(&val, cudaDevAttrIntegrated, self->gpu_id);
    self->is_integrated = (val > 0) ? 1 : 0;

    self->meta_attach_enabled = TRUE;
    if (g_strcmp0(element_name, "hicon-pouring-0") == 0) {
        self->meta_attach_enabled = env_flag_enabled(
            "HICON_STREAM_0_CPP_META_ATTACH_ENABLED", TRUE
        );
    }

    GST_INFO_OBJECT(
        self,
        "HiCon Pouring plugin started (gpu=%d, integrated=%d, meta_attach=%d, ref=%dx%d)",
        self->gpu_id,
        self->is_integrated,
        self->meta_attach_enabled ? 1 : 0,
        self->pour_ref_width,
        self->pour_ref_height
    );

    /* Attach sink pad probe for per-buffer processing.
     * Using a probe instead of transform_ip avoids gst_buffer_make_writable()
     * overhead on NVMM buffers (which halved FPS from 25 to ~13). */
    GstPad *sink_pad = gst_element_get_static_pad(GST_ELEMENT(self), "sink");
    if (sink_pad) {
        gst_pad_add_probe(sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                          hicon_pouring_sink_probe, (gpointer)self, NULL);
        gst_object_unref(sink_pad);
        GST_INFO_OBJECT(self, "Sink pad probe attached for pouring detection");
    } else {
        GST_WARNING_OBJECT(self, "Failed to get sink pad — probe not attached");
    }

    return TRUE;
}

static gboolean
gst_hicon_pouring_stop(GstBaseTransform *btrans)
{
    GstHiConPouring *self = GST_HICON_POURING(btrans);

    /* Cleanup trolley states */
    if (self->trolley_states) {
        for (auto &kv : *self->trolley_states) {
            delete kv.second;
        }
        self->trolley_states->clear();
    }
    if (self->trolley_id_to_count) {
        self->trolley_id_to_count->clear();
    }

    GST_INFO_OBJECT(self, "HiCon Pouring plugin stopped");
    return TRUE;
}

/* ================================================================
 * GObject property get/set
 * ================================================================ */

static void
gst_hicon_pouring_set_property(GObject *object, guint prop_id,
                                const GValue *value, GParamSpec *pspec)
{
    GstHiConPouring *self = GST_HICON_POURING(object);
    switch (prop_id) {
        case PROP_GPU_DEVICE_ID:
            self->gpu_id = g_value_get_uint(value);
            break;
        case PROP_ENABLE_OSD:
            self->enable_osd = g_value_get_boolean(value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

static void
gst_hicon_pouring_get_property(GObject *object, guint prop_id,
                                GValue *value, GParamSpec *pspec)
{
    GstHiConPouring *self = GST_HICON_POURING(object);
    switch (prop_id) {
        case PROP_GPU_DEVICE_ID:
            g_value_set_uint(value, self->gpu_id);
            break;
        case PROP_ENABLE_OSD:
            g_value_set_boolean(value, self->enable_osd);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
            break;
    }
}

/* ================================================================
 * GObject finalize — cleanup heap-allocated maps
 * ================================================================ */

static void
gst_hicon_pouring_finalize(GObject *object)
{
    GstHiConPouring *self = GST_HICON_POURING(object);

    if (self->trolley_states) {
        for (auto &kv : *self->trolley_states) {
            delete kv.second;
        }
        delete self->trolley_states;
        self->trolley_states = nullptr;
    }
    if (self->trolley_id_to_count) {
        delete self->trolley_id_to_count;
        self->trolley_id_to_count = nullptr;
    }

    G_OBJECT_CLASS(parent_class)->finalize(object);
}

/* ================================================================
 * class_init — register vfuncs, properties, pad templates
 * ================================================================ */

static void
gst_hicon_pouring_class_init(GstHiConPouringClass *klass)
{
    GObjectClass *gobject_class = (GObjectClass *)klass;
    GstElementClass *gstelement_class = (GstElementClass *)klass;
    GstBaseTransformClass *gstbasetransform_class = (GstBaseTransformClass *)klass;

    /* Required for NvBufSurface access on Jetson */
    g_setenv("DS_NEW_BUFAPI", "1", TRUE);

    gobject_class->set_property = GST_DEBUG_FUNCPTR(gst_hicon_pouring_set_property);
    gobject_class->get_property = GST_DEBUG_FUNCPTR(gst_hicon_pouring_get_property);
    gobject_class->finalize     = GST_DEBUG_FUNCPTR(gst_hicon_pouring_finalize);

    gstbasetransform_class->set_caps     = GST_DEBUG_FUNCPTR(gst_hicon_pouring_set_caps);
    gstbasetransform_class->start        = GST_DEBUG_FUNCPTR(gst_hicon_pouring_start);
    gstbasetransform_class->stop         = GST_DEBUG_FUNCPTR(gst_hicon_pouring_stop);
    gstbasetransform_class->transform_ip = GST_DEBUG_FUNCPTR(gst_hicon_pouring_transform_ip);

    g_object_class_install_property(gobject_class, PROP_GPU_DEVICE_ID,
        g_param_spec_uint("gpu-id", "GPU Device ID",
            "GPU Device ID", 0, G_MAXUINT, DEFAULT_GPU_ID,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));
    g_object_class_install_property(gobject_class, PROP_ENABLE_OSD,
        g_param_spec_boolean("enable-osd", "Enable OSD",
            "Enable cheap object-meta OSD updates when a downstream nvdsosd is present", FALSE,
            (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY)));

    gst_element_class_add_pad_template(gstelement_class,
        gst_static_pad_template_get(&gst_hicon_pouring_src_template));
    gst_element_class_add_pad_template(gstelement_class,
        gst_static_pad_template_get(&gst_hicon_pouring_sink_template));

    gst_element_class_set_details_simple(gstelement_class,
        "HiCon Pouring Detection",
        "Filter/Analyzer/Video",
        "Pouring detection state machine with mould counting for DeepStream",
        "HiCon AI Vision");
}

/* ================================================================
 * init — instance initialization
 * ================================================================ */

static void
gst_hicon_pouring_init(GstHiConPouring *self)
{
    GstBaseTransform *btrans = GST_BASE_TRANSFORM(self);

    /* Passthrough mode: buffer passes through untouched (no copies).
     * Processing runs in a sink pad probe attached in start(). */
    gst_base_transform_set_in_place(btrans, TRUE);
    gst_base_transform_set_passthrough(btrans, TRUE);

    self->frame_num = 0;
    self->video_fps = 25.0f;
    self->gpu_id = DEFAULT_GPU_ID;
    self->is_integrated = 0;
    self->enable_osd = FALSE;
    self->meta_attach_enabled = TRUE;
    self->pour_ref_width = env_int_value("HICON_POUR_REF_WIDTH", 1920);
    self->pour_ref_height = env_int_value("HICON_POUR_REF_HEIGHT", 1080);
    self->runtime_frame_w = 0;
    self->runtime_frame_h = 0;
    self->geometry_scale_x = 1.0f;
    self->geometry_scale_y = 1.0f;
    self->edge_expand_x_px = EDGE_EXPAND;
    self->edge_expand_y_px = EDGE_EXPAND;
    self->probe_below_px = POUR_DOT_BELOW_PX;
    self->probe_tail_dy_px = PROBE_TAIL_DY;
    self->probe_radius_px = PROBE_R;
    self->probe_count = NUM_PROBES;
    for (int i = 0; i < NUM_PROBES; i++) {
        self->probe_dx_scaled[i] = PROBE_DX[i];
        self->probe_dy_scaled[i] = PROBE_DY[i];
    }
    self->split_min_dx_px = MOULD_SPLIT_MIN_DX_PX;
    self->split_min_dy_px = MOULD_SPLIT_MIN_DY_PX;
    self->split_rearm_dx_px = MOULD_SPLIT_REARM_DX_PX;
    self->split_rearm_dy_px = MOULD_SPLIT_REARM_DY_PX;

    /* Allocate state maps on heap (C++ objects in C-allocated struct) */
    self->trolley_states = new std::unordered_map<uint64_t, TrolleyState*>();
    self->trolley_id_to_count = new std::map<uint64_t, int>();

    recompute_thresholds(self);
}

/* ================================================================
 * PLUGIN REGISTRATION
 * ================================================================ */

static gboolean
hicon_pouring_plugin_init(GstPlugin *plugin)
{
    GST_DEBUG_CATEGORY_INIT(gst_hicon_pouring_debug, "hicon_pouring_detect", 0,
                             "HiCon pouring detection plugin");

    return gst_element_register(plugin, "hicon_pouring_detect", GST_RANK_PRIMARY,
                                 GST_TYPE_HICON_POURING);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    hiconpouring,
    DESCRIPTION,
    hicon_pouring_plugin_init,
    VERSION,
    LICENSE,
    BINARY_PACKAGE,
    URL
)
