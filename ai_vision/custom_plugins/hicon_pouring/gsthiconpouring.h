/*
 * HiCon Pouring Detection GStreamer Plugin
 * =========================================
 * Custom GstBaseTransform element that exports lightweight pouring session state from
 * tracked DeepStream object metadata. Python owns brightness-based pour
 * transitions, mould aggregation, screenshots, DB writes, and sync.
 *
 * The supported Stream 0 path is metadata-only: no pixel mapping, no RGBA side
 * branch, and no custom CUDA work beyond DeepStream's built-in decode/infer.
 * Detection results are attached as NvDsUserMeta for downstream Python probes.
 * Cheap object-meta OSD can still be enabled on paths with downstream nvdsosd.
 *
 * Build:  make -C ai_vision/custom_plugins/hicon_pouring/
 * Install: cp libgsthiconpouring.so /opt/nvidia/deepstream/deepstream-7.1/lib/gst-plugins/
 * Test:   gst-inspect-1.0 hicon_pouring_detect
 */

#ifndef __GST_HICON_POURING_H__
#define __GST_HICON_POURING_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "nvbufsurface.h"
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"

#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <string>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <algorithm>

/* Package metadata for GST_PLUGIN_DEFINE */
#define PACKAGE "hiconpouring"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "HiCon pouring detection plugin for DeepStream"
#define BINARY_PACKAGE "HiCon AI Vision"
#define URL "https://hicon.ai/"

G_BEGIN_DECLS

/* GObject boilerplate macros */
typedef struct _GstHiConPouring GstHiConPouring;
typedef struct _GstHiConPouringClass GstHiConPouringClass;

#define GST_TYPE_HICON_POURING            (gst_hicon_pouring_get_type())
#define GST_HICON_POURING(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_HICON_POURING, GstHiConPouring))
#define GST_HICON_POURING_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass), GST_TYPE_HICON_POURING, GstHiConPouringClass))
#define GST_IS_HICON_POURING(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), GST_TYPE_HICON_POURING))
#define GST_IS_HICON_POURING_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), GST_TYPE_HICON_POURING))

G_END_DECLS

/* ================================================================
 * CONSTANTS — mapped 1:1 from deepstream_pouring_app.cpp
 * ================================================================ */

// Class IDs (must match labels_pouring.txt order)
#define CLASS_LADLE_MOUTH  0
#define CLASS_TROLLEY      1

// Session gating
static constexpr float SESSION_ENTER_S     = 1.0f;
static constexpr float SESSION_EXIT_S      = 1.5f;
static constexpr float MOUTH_MISSING_TOL_S = 0.6f;
static constexpr float MOUTH_HOLD_S        = 0.4f;
static constexpr float TROLLEY_HOLD_S      = 1.0f;

// Pour ON/OFF hysteresis
static constexpr float POUR_ON_S           = 0.20f;
static constexpr float POUR_OFF_S          = 0.80f;
static constexpr float MIN_POUR_S          = 0.80f;
static constexpr float MIN_POUR_DURATION_S = 2.0f;

// Brightness probe
static constexpr int   PROBE_R             = 8;
static constexpr float TH_ON               = 205.0f;
static constexpr float TH_OFF              = 160.0f;
static const     int   PROBE_DX[]          = {0, 12, -12, 24, -24};
static const     int   PROBE_DY[]          = {0, 0, 0, 0, 0};
static constexpr int   NUM_PROBES          = 5;
static constexpr int   POUR_DOT_BELOW_PX   = 30;
static constexpr int   PROBE_TAIL_DY       = 20;

// Mould switching (anchor-based)
static constexpr float D_SPLIT             = 0.25f;
static constexpr float T_HOLD_S            = 0.30f;
static constexpr float MOULD_SWITCH_HOLD_S = 2.0f;
static constexpr float MOULD_SPLIT_MIN_DX_PX = 12.0f;
static constexpr float MOULD_SPLIT_MIN_DY_PX = 12.0f;
static constexpr float MOULD_SPLIT_REARM_DX_PX = 10.0f;
static constexpr float MOULD_SPLIT_REARM_DY_PX = 14.0f;

// Clustering
static constexpr float R_CLUSTER           = 0.08f;
static constexpr float R_MERGE_VAL         = 0.07f;
static constexpr float MIN_CLUSTER_POUR_S  = 1.5f;
static constexpr int   CLUSTER_BACKTRACK_CID_GUARD = 5;

// Edge expansion for trolley-mouth association
static constexpr float EDGE_EXPAND         = 180.0f;

// Trolley disappearance
static constexpr float TROLLEY_GONE_S      = 2.0f;
static constexpr float TROLLEY_RESET_S     = 5.0f;

/* ================================================================
 * FRAME-COUNT CONVERSION
 * ================================================================ */

static inline int sec_to_frames(float sec, float fps) {
    return std::max(1, (int)std::round(sec * fps));
}

/* ================================================================
 * DATA STRUCTURES — from deepstream_pouring_app.cpp
 * ================================================================ */

struct PourSegment {
    int start_f;
    int end_f;
    std::vector<std::pair<float,float>> mouth_pts_norm;  // (-1,-1) = None

    int duration_frames() const {
        return std::max(0, end_f - start_f + 1);
    }

    std::pair<float,float> representative_point() const {
        std::vector<float> xs, ys;
        for (auto &p : mouth_pts_norm) {
            if (p.first >= 0.0f) {
                xs.push_back(p.first);
                ys.push_back(p.second);
            }
        }
        if (xs.empty()) return {0.0f, 0.0f};
        std::sort(xs.begin(), xs.end());
        std::sort(ys.begin(), ys.end());
        size_t mid = xs.size() / 2;
        float mx = (xs.size() % 2 == 0) ? (xs[mid-1] + xs[mid]) / 2.0f : xs[mid];
        float my = (ys.size() % 2 == 0) ? (ys[mid-1] + ys[mid]) / 2.0f : ys[mid];
        return {mx, my};
    }
};

struct MouldCluster {
    int cid;
    std::pair<float,float> centroid;
    std::vector<PourSegment> segments;

    int total_frames() const {
        int sum = 0;
        for (auto &s : segments) sum += s.duration_frames();
        return sum;
    }
};

struct ClusterPreview {
    int mould_count = 0;
    std::vector<int> mould_total_frames;
    int active_cluster_pos = -1;
};

struct TrolleyState {
    uint64_t tid;

    float bbox[4]          = {0,0,0,0};   // x1, y1, x2, y2
    bool  bbox_valid        = false;
    int   last_seen_f       = -1;

    // Session gating
    int   in_count          = 0;
    int   out_count         = 0;
    bool  session_active    = false;
    int   session_start_f   = -1;
    int   session_end_f     = -1;

    // Pour state
    bool  pour_active       = false;
    int   pour_on_count     = 0;
    int   pour_off_count    = 0;

    // Probe tracking
    uint64_t active_probe_track_id = UINT64_MAX;
    int   active_probe_last_seen_f = -999999;
    float active_probe_bbox[4]     = {0,0,0,0};
    bool  active_probe_bbox_valid  = false;
    float active_probe_pt_px[2]    = {0,0};
    bool  active_probe_pt_valid    = false;
    bool  active_probe_from_hold   = false;
    bool  frozen_probe_active      = false;
    float frozen_probe_x           = 0.0f;
    float frozen_probe_y           = 0.0f;
    float frozen_probe_bbox[4]     = {0,0,0,0};
    bool  frozen_probe_bbox_valid  = false;

    // Current segment (heap-allocated, owned)
    PourSegment *current_seg = nullptr;
    std::vector<PourSegment> completed;

    // Mould counting (real-time, anchor-based)
    int   current_mould_id     = 0;
    int   mould_switch_count   = 0;
    float mould_anchor_pt[2]   = {0,0};
    bool  mould_anchor_valid   = false;
    std::map<int,int> mould_completed_times;

    // Clustering result
    int   final_clustered_count = -1;
    int   last_disappeared_f    = -1;

    ~TrolleyState() { delete current_seg; }
};

/* ================================================================
 * NvDsUserMeta: HiConPouringMeta
 *
 * Attached to each frame by the C++ plugin. Python reads this downstream and
 * owns the business state machine (brightness gating, pour ON/OFF, mould
 * splitting/clustering, DB writes, heat-cycle updates, screenshots, sync).
 * ================================================================ */

/* Unique meta type ID — must not collide with DeepStream built-in types.
 * NVDS_USER_CUSTOM_META starts at NVDS_START_USER_META + 4096. */
#define HICON_POURING_META_TYPE (NvDsMetaType)(NVDS_START_USER_META + 1)

static constexpr uint32_t HICON_POURING_META_VERSION = 2;

struct HiConPouringMeta {
    uint32_t version;

    // Current state snapshot (one authoritative trolley/mouth selection per frame)
    uint32_t session_active;
    uint32_t mouth_present_in_trolley;
    uint32_t probe_valid;
    uint32_t reserved0;

    // Events that occurred this frame
    enum EventType {
        NONE = 0,
        SESSION_START,
        SESSION_END,
    };
    uint32_t event;

    // Selected trolley/mouth identity
    uint64_t trolley_track_id;
    uint64_t mouth_track_id;

    // Geometry exported for Python-side brightness probing and screenshots.
    float trolley_bbox[4];       // x1, y1, x2, y2
    float mouth_bbox[4];         // x1, y1, x2, y2 (may be last-known if probe held)
    float probe_x_px;
    float probe_y_px;
    float mouth_norm_x;
    float mouth_norm_y;
};

/* ================================================================
 * DISPLAY METADATA HELPER
 *
 * DeepStream limits NvDsDisplayMeta to MAX_ELEMENTS_IN_DISPLAY_META=16
 * entries per struct for text/rect/circle. This helper acquires new
 * display_meta from the pool as needed.
 * ================================================================ */

struct DisplayMetaHelper {
    NvDsBatchMeta  *batch_meta;
    NvDsFrameMeta  *frame_meta;
    NvDsDisplayMeta *current;
    int text_idx, rect_idx, circle_idx;
    bool exhausted;

    DisplayMetaHelper(NvDsBatchMeta *bm, NvDsFrameMeta *fm)
        : batch_meta(bm), frame_meta(fm), current(nullptr),
          text_idx(0), rect_idx(0), circle_idx(0), exhausted(false) {
        acquire_new();
    }

    void acquire_new() {
        current = nvds_acquire_display_meta_from_pool(batch_meta);
        if (!current) {
            exhausted = true;
            return;
        }
        current->num_labels  = 0;
        current->num_rects   = 0;
        current->num_circles = 0;
        current->num_lines   = 0;
        current->num_arrows  = 0;
        text_idx = rect_idx = circle_idx = 0;
        nvds_add_display_meta_to_frame(frame_meta, current);
    }

    NvOSD_TextParams *next_text() {
        if (exhausted) return nullptr;
        if (text_idx >= MAX_ELEMENTS_IN_DISPLAY_META) acquire_new();
        if (exhausted) return nullptr;
        NvOSD_TextParams *tp = &current->text_params[text_idx++];
        current->num_labels = text_idx;
        return tp;
    }

    NvOSD_RectParams *next_rect() {
        if (exhausted) return nullptr;
        if (rect_idx >= MAX_ELEMENTS_IN_DISPLAY_META) acquire_new();
        if (exhausted) return nullptr;
        NvOSD_RectParams *rp = &current->rect_params[rect_idx++];
        current->num_rects = rect_idx;
        return rp;
    }

    NvOSD_CircleParams *next_circle() {
        if (exhausted) return nullptr;
        if (circle_idx >= MAX_ELEMENTS_IN_DISPLAY_META) acquire_new();
        if (exhausted) return nullptr;
        NvOSD_CircleParams *cp = &current->circle_params[circle_idx++];
        current->num_circles = circle_idx;
        return cp;
    }
};

/* ================================================================
 * GstHiConPouring — GstBaseTransform element instance struct
 * ================================================================ */

struct _GstHiConPouring {
    GstBaseTransform base_trans;

    /* Frame counter */
    guint64 frame_num;

    /* Video FPS (extracted from caps) */
    float video_fps;

    /* Frame-count thresholds (recomputed from FPS) */
    int N_ENTER, N_EXIT, K_ON, K_OFF;
    int MOULD_SWITCH_HOLD_F, MIN_POUR_FRAMES, MIN_POUR_DURATION_FRAMES;
    int MOUTH_MISSING_TOL, MOUTH_HOLD_DUR, TROLLEY_HOLD_DUR, SESSION_END_TOTAL_MISSING;
    int T_HOLD_F, MIN_CLUSTER_POUR_F;

    /* Per-trolley state */
    std::unordered_map<uint64_t, TrolleyState*> *trolley_states;
    std::map<uint64_t, int> *trolley_id_to_count;

    /* GPU info */
    guint gpu_id;
    guint is_integrated;
    gboolean enable_osd;
    gboolean meta_attach_enabled;
    gint pour_ref_width;
    gint pour_ref_height;
    gint runtime_frame_w;
    gint runtime_frame_h;
    float geometry_scale_x;
    float geometry_scale_y;
    float edge_expand_x_px;
    float edge_expand_y_px;
    int probe_below_px;
    int probe_tail_dy_px;
    int probe_radius_px;
    int probe_dx_scaled[NUM_PROBES];
    int probe_dy_scaled[NUM_PROBES];
    int probe_count;
    float split_min_dx_px;
    float split_min_dy_px;
    float split_rearm_dx_px;
    float split_rearm_dy_px;

    /* Video info (from caps) */
    GstVideoInfo video_info;
};

struct _GstHiConPouringClass {
    GstBaseTransformClass parent_class;
};

GType gst_hicon_pouring_get_type(void);

#endif /* __GST_HICON_POURING_H__ */
