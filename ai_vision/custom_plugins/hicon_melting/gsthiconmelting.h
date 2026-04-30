/*
 * HiCon Melting Detection GStreamer Plugin
 * ========================================
 * Native CUDA detector for Stream 0 tapping/deslagging/spectro analysis.
 * The plugin runs on an NV12 side branch, attaches lightweight NvDsUserMeta,
 * and leaves screenshots/DB/heat-cycle/sync orchestration to Python.
 */

#ifndef __GST_HICON_MELTING_H__
#define __GST_HICON_MELTING_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

#include <vector>
#include <string>
#include <cstdint>

#include "nvbufsurface.h"
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include "molten_detect.h"

G_BEGIN_DECLS

typedef struct _GstHiConMelting GstHiConMelting;
typedef struct _GstHiConMeltingClass GstHiConMeltingClass;

#define GST_TYPE_HICON_MELTING            (gst_hicon_melting_get_type())
#define GST_HICON_MELTING(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_HICON_MELTING, GstHiConMelting))
#define GST_HICON_MELTING_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_HICON_MELTING, GstHiConMeltingClass))
#define GST_IS_HICON_MELTING(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_HICON_MELTING))
#define GST_IS_HICON_MELTING_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_HICON_MELTING))

#define PACKAGE "hiconmelting"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "HiCon melting detection plugin for DeepStream"
#define BINARY_PACKAGE "HiCon AI Vision"
#define URL "https://hicon.ai/"

#define HICON_MELTING_META_TYPE (NvDsMetaType)(NVDS_START_USER_META + 2)
static constexpr uint32_t HICON_MELTING_META_VERSION = 1;
static constexpr uint32_t HICON_MELTING_MAX_ZONES = 8;
static constexpr float HICON_NODULIZER_WHITE_RATIO_THRESHOLD = 0.9f;
static constexpr float HICON_NODULIZER_BLACKOUT_S = 10.0f;
static constexpr float HICON_MELTING_STARTUP_WARMUP_S = 10.0f;

struct HiConMeltingZoneMeta {
    uint32_t valid;
    uint32_t active;
    uint32_t raw_count;
    uint32_t filtered_count;
    float white_ratio;
    float max_blob_area;
    float max_blob_brightness;
    uint32_t reserved0;
};

struct HiConMeltingMeta {
    uint32_t version;
    uint32_t blackout_active;
    uint32_t tapping_zone_count;
    uint32_t deslagging_zone_count;
    uint32_t spectro_zone_count;
    uint32_t reserved0;
    uint64_t frame_num;
    uint64_t ntp_timestamp;
    HiConMeltingZoneMeta tapping[HICON_MELTING_MAX_ZONES];
    HiConMeltingZoneMeta deslagging[HICON_MELTING_MAX_ZONES];
    HiConMeltingZoneMeta spectro[HICON_MELTING_MAX_ZONES];
};

struct MeltingZoneConfig {
    std::string name;
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
    int on_frames = 0;
    float max_aspect_ratio = -1.0f;
    float max_coverage = -1.0f;
};

struct TappingZoneState {
    MeltingZoneConfig cfg;
    bool active = false;
    int on_count = 0;
    int off_count = 0;
    float white_ratio = 0.0f;
};

struct BlobZoneState {
    MeltingZoneConfig cfg;
    bool active = false;
    int on_count = 0;
    uint32_t raw_count = 0;
    uint32_t filtered_count = 0;
    float max_blob_area = 0.0f;
    float max_blob_brightness = 0.0f;
};

struct MeltingConfig {
    float fps = 25.0f;

    int tapping_abs_threshold = 210;
    float tapping_on_ratio = 0.25f;
    int tapping_on_frames = 20;
    float tapping_off_ratio = 0.10f;
    int tapping_off_frames = 25;

    int deslagging_min_blob_area = 500;
    int deslagging_brightness_thresh = 200;

    int spectro_min_blob_area = 50;
    int spectro_brightness_thresh = 180;

    std::vector<MeltingZoneConfig> tapping_zones;
    std::vector<MeltingZoneConfig> deslagging_zones;
    std::vector<MeltingZoneConfig> spectro_zones;
};

struct _GstHiConMelting {
    GstBaseTransform base_trans;
    gchar* config_ini;
    gboolean config_loaded;
    gboolean config_failed;
    MeltingConfig config;
    std::vector<TappingZoneState> tapping_states;
    std::vector<BlobZoneState> deslagging_states;
    std::vector<BlobZoneState> spectro_states;
    uint64_t blackout_until_frame;
    gint64 warmup_until_frame;
};

struct _GstHiConMeltingClass {
    GstBaseTransformClass parent_class;
};

GType gst_hicon_melting_get_type(void);

G_END_DECLS

#endif
