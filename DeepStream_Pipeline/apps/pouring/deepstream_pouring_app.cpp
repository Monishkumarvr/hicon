/*
 * DeepStream Pouring Detection Pipeline
 * ======================================
 * Converts HI-CON inference_pouring.ipynb to DeepStream 7.1 C++ pipeline.
 *
 * Single PGIE detects both classes (ladle_mouth=0, trolley=1).
 * NvDCF tracker provides persistent track IDs.
 * OSD sink pad probe implements the full state machine:
 *   - Trolley-mouth association (center-in-expanded-bbox)
 *   - Session gating (enter/exit counters)
 *   - Pour ON/OFF via HSV brightness probes (NvBufSurface RGBA pixel access)
 *   - Anchor-based mould counting with displacement detection
 *   - Post-processing: recursive segment splitting + spatial clustering
 *   - Incremental CSV + final JSON output
 *
 * Pipeline:
 *   uridecodebin -> nvstreammux -> nvvideoconvert(bottom-pad) -> nvinfer(PGIE)
 *     -> nvtracker -> nvvideoconvert -> capsfilter(RGBA) -> nvosd
 *     -> nvvideoconvert(crop) -> nvv4l2h264enc
 *     -> h264parse -> mp4mux -> filesink
 *
 * Build:
 *   make -C /workspace/apps/pouring
 *
 * Run:
 *   ./deepstream-pouring-app <input_video.mp4> [output_dir]
 */

#include <gst/gst.h>
#include <glib.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <climits>
#include <cfloat>

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <functional>
#include <sys/stat.h>

#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
#include "nvbufsurface.h"
#include <cuda_runtime_api.h>

#include "nlohmann/json.hpp"
using json = nlohmann::json;

/* ================================================================
 * CONSTANTS — mapped 1:1 from inference_pouring.ipynb
 * ================================================================ */

// Pipeline config
#define PGIE_CONFIG_FILE   "/workspace/apps/pouring/config_pouring.txt"
#define TRACKER_CONFIG     "/opt/nvidia/deepstream/deepstream-7.1/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml"
#define TRACKER_LIB        "/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so"

#define MUXER_WIDTH        1920
#define MUXER_HEIGHT       1080
#define MUXER_BATCH_SIZE   1

static constexpr bool  ENABLE_INFER_BOTTOM_PADDING = true;
static constexpr int   INFER_BOTTOM_PAD_PX         = 160;
static constexpr int   VISIBLE_FRAME_HEIGHT        = MUXER_HEIGHT;
static constexpr int   INFER_FRAME_HEIGHT          =
    ENABLE_INFER_BOTTOM_PADDING ? (MUXER_HEIGHT + INFER_BOTTOM_PAD_PX) : MUXER_HEIGHT;

// Class IDs (must match labels_pouring.txt order)
#define CLASS_LADLE_MOUTH  0
#define CLASS_TROLLEY      1

// Session gating
static constexpr float SESSION_ENTER_S     = 1.0f;
static constexpr float SESSION_EXIT_S      = 1.5f;
static constexpr float MOUTH_MISSING_TOL_S = 0.6f;
static constexpr float MOUTH_HOLD_S        = 0.4f;

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

// Mould switching (anchor-based)
static constexpr float D_SPLIT             = 0.25f;
static constexpr float T_HOLD_S            = 0.30f;
static constexpr float MOULD_SWITCH_HOLD_S = 2.0f;

// Clustering
static constexpr float R_CLUSTER           = 0.08f;
static constexpr float R_MERGE_VAL         = 0.07f;
static constexpr float MIN_CLUSTER_POUR_S  = 1.5f;
// Recency guard: allow reassignment only to latest cluster or immediately previous.
static constexpr int   CLUSTER_BACKTRACK_CID_GUARD = 1;  // cid >= latest_cid - 1

// Edge expansion for trolley-mouth association
static constexpr float EDGE_EXPAND         = 180.0f;
static constexpr int   PROBE_TAIL_DY       = 20;

// Trolley disappearance
static constexpr float TROLLEY_GONE_S      = 2.0f;
static constexpr float TROLLEY_RESET_S     = 5.0f;

// Display / encoding
static constexpr int   ENCODER_BITRATE     = 8000000;

/* ================================================================
 * FRAME-COUNT CONVERSION
 * ================================================================ */

static inline int sec_to_frames(float sec, float fps) {
    return std::max(1, (int)std::round(sec * fps));
}

/* ================================================================
 * DATA STRUCTURES — translated from Python dataclasses
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
    int active_cluster_pos = -1;  // 0-based index into mould_total_frames
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
    std::map<int,int> mould_completed_times;  // mould_id -> accumulated frames

    // Clustering result
    int   final_clustered_count = -1;
    int   last_disappeared_f    = -1;

    ~TrolleyState() { delete current_seg; }
};

/* ================================================================
 * GLOBAL STATE
 * ================================================================ */

static std::unordered_map<uint64_t, TrolleyState*> g_trolley_states;
static std::map<uint64_t, int> g_trolley_id_to_count;
static std::vector<json> g_final_summaries;

static int   g_frame_count = 0;
static float g_video_fps   = 25.0f;

// Frame-count thresholds (recomputed from FPS in cb_newpad)
static int N_ENTER, N_EXIT, K_ON, K_OFF;
static int MOULD_SWITCH_HOLD_F, MIN_POUR_FRAMES, MIN_POUR_DURATION_FRAMES;
static int MOUTH_MISSING_TOL, MOUTH_HOLD_DUR, T_HOLD_F, MIN_CLUSTER_POUR_F;

// Output
static std::string   g_output_json_path;
static std::string   g_output_csv_path;
static std::string   g_output_video_path;
static std::string   g_input_uri;
static std::string   g_output_dir;

static std::chrono::steady_clock::time_point g_start_time;
static GMainLoop *g_main_loop = nullptr;

/* ================================================================
 * UTILITY: frame-count thresholds from FPS
 * ================================================================ */

static void recompute_thresholds(float fps) {
    N_ENTER               = sec_to_frames(SESSION_ENTER_S,     fps);
    N_EXIT                = sec_to_frames(SESSION_EXIT_S,      fps);
    K_ON                  = sec_to_frames(POUR_ON_S,           fps);
    K_OFF                 = sec_to_frames(POUR_OFF_S,          fps);
    MOULD_SWITCH_HOLD_F   = sec_to_frames(MOULD_SWITCH_HOLD_S, fps);
    MIN_POUR_FRAMES       = sec_to_frames(MIN_POUR_S,          fps);
    MIN_POUR_DURATION_FRAMES = sec_to_frames(MIN_POUR_DURATION_S, fps);
    MOUTH_MISSING_TOL     = sec_to_frames(MOUTH_MISSING_TOL_S, fps);
    MOUTH_HOLD_DUR        = sec_to_frames(MOUTH_HOLD_S,        fps);
    T_HOLD_F              = sec_to_frames(T_HOLD_S,            fps);
    MIN_CLUSTER_POUR_F    = sec_to_frames(MIN_CLUSTER_POUR_S,  fps);
}

/* ================================================================
 * UTILITY: filesystem
 * ================================================================ */

static void mkdir_p(const std::string &path) {
    std::string cmd = "mkdir -p \"" + path + "\"";
    system(cmd.c_str());
}

static std::string basename_no_ext(const std::string &path) {
    size_t slash = path.find_last_of("/\\");
    std::string fname = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = fname.find_last_of('.');
    return (dot == std::string::npos) ? fname : fname.substr(0, dot);
}

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

static inline void norm_point_in_trolley(float px, float py, const float bbox[4],
                                          float &nx, float &ny) {
    float w = std::max(1.0f, bbox[2] - bbox[0]);
    float h = std::max(1.0f, bbox[3] - bbox[1]);
    nx = (px - bbox[0]) / w;
    ny = (py - bbox[1]) / h;
}

static inline float l2_dist(float ax, float ay, float bx, float by) {
    float dx = ax - bx, dy = ay - by;
    return std::sqrt(dx*dx + dy*dy);
}

static inline void mouth_probe_point(const float bbox[4], float &px, float &py) {
    px = (bbox[0] + bbox[2]) / 2.0f;
    py = bbox[3] + POUR_DOT_BELOW_PX;
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

/* ================================================================
 * BRIGHTNESS PROBE — NvBufSurface RGBA pixel access
 *
 * HSV V-channel = max(R, G, B).  No full HSV conversion needed.
 * Buffer must be RGBA format (enforced by capsfilter after nvvidconv).
 * ================================================================ */

static float brightness_probe(const uint8_t *data, uint32_t pitch,
                               int width, int height,
                               int cx, int cy, int r)
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
            int off = x * 4;  // RGBA: 4 bytes per pixel
            uint8_t R = row[off + 0];
            uint8_t G = row[off + 1];
            uint8_t B = row[off + 2];
            uint8_t V = std::max({R, G, B});
            sum += (float)V;
            count++;
        }
    }
    return (count > 0) ? (sum / (float)count) : 0.0f;
}

static float flare_score(const uint8_t *data, uint32_t pitch,
                          int width, int height,
                          float pour_dot_x, float pour_dot_y)
{
    float total = 0.0f;
    int valid = 0;
    for (int i = 0; i < NUM_PROBES; i++) {
        int cx = (int)clamp_f(pour_dot_x + PROBE_DX[i], 0.0f, (float)(width - 1));
        int cy = (int)clamp_f(pour_dot_y + PROBE_DY[i], 0.0f, (float)(height - 1));
        float v = brightness_probe(data, pitch, width, height, cx, cy, PROBE_R);
        if (v > 0.0f) { total += v; valid++; }
    }
    return (valid > 0) ? (total / (float)valid) : 0.0f;
}

/* ================================================================
 * SEGMENT SPLITTING — recursive, from notebook split_segment_by_motion()
 * ================================================================ */

static void split_segment_by_motion(const PourSegment &seg, float d_split, int t_hold,
                                     std::vector<PourSegment> &out)
{
    /* mouth_pts_norm is sparse: it only has entries for frames where mouth was
     * valid, NOT one entry per frame. Tracking original indices into the full
     * mouth_pts_norm list is required so that split frame boundaries
     * (left.end_f, right.start_f) use the correct frame numbers, not the
     * compressed valid-point indices. */
    std::vector<std::pair<int,std::pair<float,float>>> idx_pts; // (orig_idx, point)
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
            int split_norm_idx = idx_pts[split_pts_idx].first; // index into mouth_pts_norm

            /* Use the frame number stored alongside each valid point.
             * mouth_pts_norm entry at split_norm_idx corresponds to frame:
             *   seg.start_f + (split_norm_idx * duration) / mouth_pts_norm.size()
             * But since mouth_pts_norm is sparse we approximate using the ratio
             * of valid-point index to total valid points vs total frames. */
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
 * SPATIAL CLUSTERING — from notebook assign_to_clusters / merge / build
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

static ClusterPreview build_cluster_preview_for_overlay(const TrolleyState *st, int frame_idx)
{
    ClusterPreview preview;
    std::vector<PourSegment> preview_segments = st->completed;
    bool has_active_point = false;
    std::pair<float,float> active_point = {0.0f, 0.0f};

    // Include in-progress segment so OSD follows the same split+cluster path as final outputs.
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
        split_segment_by_motion(seg, D_SPLIT, T_HOLD_F, split_segs);
    }

    auto clusters = build_clusters(split_segs, MIN_POUR_DURATION_FRAMES, R_CLUSTER, R_MERGE_VAL);
    std::sort(clusters.begin(), clusters.end(),
              [](const MouldCluster &a, const MouldCluster &b) { return a.cid < b.cid; });

    int active_cluster_cid = -1;
    if (has_active_point) {
        float best_d = FLT_MAX;
        for (auto &c : clusters) {
            int frames = c.total_frames();
            if (frames < MIN_CLUSTER_POUR_F) continue;
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
        if (frames >= MIN_CLUSTER_POUR_F) {
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
 * TROLLEY FINALIZATION — close open pours, cluster, build JSON
 * ================================================================ */

static void finalize_trolley(TrolleyState *st, int current_frame) {
    uint64_t tid = st->tid;

    // Close open pour
    if (st->pour_active && st->current_seg != nullptr) {
        st->current_seg->end_f = current_frame;
        if (st->current_seg->duration_frames() >= MIN_POUR_DURATION_FRAMES) {
            st->current_mould_id++;
            st->mould_completed_times[st->current_mould_id] = st->current_seg->duration_frames();
            g_trolley_id_to_count[tid] = st->current_mould_id;
            st->completed.push_back(*st->current_seg);
        }
        delete st->current_seg;
        st->current_seg = nullptr;
        st->pour_active = false;
    }
    clear_frozen_probe(st);

    // Split + cluster
    std::vector<PourSegment> split_segs;
    for (auto &seg : st->completed) {
        split_segment_by_motion(seg, D_SPLIT, T_HOLD_F, split_segs);
    }
    auto clusters = build_clusters(split_segs, MIN_POUR_DURATION_FRAMES, R_CLUSTER, R_MERGE_VAL);

    std::vector<MouldCluster> valid;
    for (auto &c : clusters) {
        if (c.total_frames() >= MIN_CLUSTER_POUR_F) valid.push_back(c);
    }
    int mould_count = (int)valid.size();

    // Build JSON summary
    json summary;
    summary["trolley_id"]          = (int)tid;
    summary["session_start_frame"] = st->session_start_f;
    summary["session_end_frame"]   = (st->session_end_f >= 0) ? st->session_end_f : current_frame;
    summary["mould_count"]         = mould_count;

    // mould_times derived from clustering (matches per_mould_summary exactly)
    json mould_times_j;
    for (auto &c : valid) {
        mould_times_j[std::to_string(c.cid)] = (float)c.total_frames() / g_video_fps;
    }
    summary["mould_times"] = mould_times_j;

    json per_mould_arr = json::array();
    for (auto &c : valid) {
        per_mould_arr.push_back({
            {"mould_cluster", c.cid},
            {"num_pours",     (int)c.segments.size()},
            {"total_pour_s",  (float)c.total_frames() / g_video_fps},
            {"centroid_rel",  {c.centroid.first, c.centroid.second}}
        });
    }
    summary["per_mould_summary"] = per_mould_arr;

    json pours_arr = json::array();
    for (auto &c : valid) {
        for (int j = 0; j < (int)c.segments.size(); j++) {
            auto &s = c.segments[j];
            auto rp = s.representative_point();
            pours_arr.push_back({
                {"mould_cluster", c.cid},
                {"pour_idx",      j + 1},
                {"start_frame",   s.start_f},
                {"end_frame",     s.end_f},
                {"duration_s",    (float)(s.end_f - s.start_f + 1) / g_video_fps},
                {"rep_rel",       {rp.first, rp.second}}
            });
        }
    }
    summary["pours"] = pours_arr;

    // UTC timestamp
    auto now   = std::chrono::system_clock::now();
    auto tt    = std::chrono::system_clock::to_time_t(now);
    std::stringstream ts_ss;
    ts_ss << std::put_time(std::gmtime(&tt), "%Y-%m-%dT%H:%M:%S") << "Z";
    summary["timestamp_utc"] = ts_ss.str();

    g_final_summaries.push_back(summary);
    st->final_clustered_count = mould_count;
    st->last_disappeared_f = current_frame;
}

/* ================================================================
 * JSON OUTPUT — write all summaries at EOS
 * ================================================================ */

static void write_json_output() {
    if (g_output_json_path.empty()) return;
    json arr = json::array();
    for (auto &s : g_final_summaries) arr.push_back(s);
    std::ofstream ofs(g_output_json_path);
    if (ofs.is_open()) {
        ofs << arr.dump(2);
        ofs.close();
        g_print("JSON saved: %s\n", g_output_json_path.c_str());
    }
}

static void write_csv_output() {
    if (g_output_csv_path.empty()) return;
    std::ofstream ofs(g_output_csv_path, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        g_printerr("Failed to write CSV: %s\n", g_output_csv_path.c_str());
        return;
    }

    ofs << "trolley_id,mould_id,pouring_time_s\n";
    ofs << std::fixed << std::setprecision(6);

    for (auto &s : g_final_summaries) {
        int tid = s.value("trolley_id", -1);
        if (!s.contains("per_mould_summary")) continue;

        for (auto &pm : s["per_mould_summary"]) {
            int mould_id = pm.value("mould_cluster", 0);
            double total_s = pm.value("total_pour_s", 0.0);
            ofs << tid << "," << mould_id << "," << total_s << "\n";
        }
    }

    ofs.close();
    g_print("CSV saved: %s\n", g_output_csv_path.c_str());
}

/* ================================================================
 * DISPLAY METADATA HELPER
 *
 * DeepStream limits NvDsDisplayMeta to MAX_ELEMENTS_IN_DISPLAY_META=16
 * entries per struct for text/rect/circle.  This helper acquires new
 * display_meta from the pool as needed.
 * ================================================================ */

struct DisplayMetaHelper {
    NvDsBatchMeta  *batch_meta;
    NvDsFrameMeta  *frame_meta;
    NvDsDisplayMeta *current;
    int text_idx, rect_idx, circle_idx;

    DisplayMetaHelper(NvDsBatchMeta *bm, NvDsFrameMeta *fm)
        : batch_meta(bm), frame_meta(fm), current(nullptr),
          text_idx(0), rect_idx(0), circle_idx(0) {
        acquire_new();
    }

    void acquire_new() {
        current = nvds_acquire_display_meta_from_pool(batch_meta);
        current->num_labels  = 0;
        current->num_rects   = 0;
        current->num_circles = 0;
        current->num_lines   = 0;
        current->num_arrows  = 0;
        text_idx = rect_idx = circle_idx = 0;
        nvds_add_display_meta_to_frame(frame_meta, current);
    }

    NvOSD_TextParams *next_text() {
        if (text_idx >= MAX_ELEMENTS_IN_DISPLAY_META) acquire_new();
        NvOSD_TextParams *tp = &current->text_params[text_idx++];
        current->num_labels = text_idx;
        return tp;
    }

    NvOSD_RectParams *next_rect() {
        if (rect_idx >= MAX_ELEMENTS_IN_DISPLAY_META) acquire_new();
        NvOSD_RectParams *rp = &current->rect_params[rect_idx++];
        current->num_rects = rect_idx;
        return rp;
    }

    NvOSD_CircleParams *next_circle() {
        if (circle_idx >= MAX_ELEMENTS_IN_DISPLAY_META) acquire_new();
        NvOSD_CircleParams *cp = &current->circle_params[circle_idx++];
        current->num_circles = circle_idx;
        return cp;
    }
};

struct Detection {
    uint64_t track_id;
    float    bbox[4];  // x1, y1, x2, y2
    float    conf;
    int      class_id;
};

/* ================================================================
 * HELPER: configure a text param (reused for info panel)
 * ================================================================ */

static void setup_text(NvOSD_TextParams *tp, const char *text,
                       int x, int y, float font_size,
                       float r, float g, float b, float a)
{
    tp->display_text = g_strdup(text);
    tp->x_offset = x;
    tp->y_offset = y;
    tp->font_params.font_name  = (gchar *)"Sans";
    tp->font_params.font_size  = (guint)font_size;
    tp->font_params.font_color = {r, g, b, a};
    tp->set_bg_clr = 0;
}

/* ================================================================
 * OSD SINK PAD PROBE — the main state machine callback
 * ================================================================ */

static GstPadProbeReturn osd_sink_pad_buffer_probe(
    GstPad *pad, GstPadProbeInfo *info, gpointer u_data)
{
    GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    /* --- Map NvBufSurface for pixel access --- */
    GstMapInfo map_info;
    memset(&map_info, 0, sizeof(map_info));
    if (!gst_buffer_map(buf, &map_info, GST_MAP_READ)) {
        g_printerr("Failed to map GstBuffer\n");
        return GST_PAD_PROBE_OK;
    }
    NvBufSurface *surface = (NvBufSurface *)map_info.data;

    /* --- Process each frame in batch (batch=1 for single stream) --- */
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list;
         l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
        int frame_idx = g_frame_count++;
        int batch_idx = frame_meta->batch_id;

        /* --- Copy surface to CPU for pixel reading (dGPU: CUDA device memory) --- */
        bool pixels_mapped = false;
        uint8_t *pixel_data = nullptr;
        uint32_t pixel_pitch = 0;
        int frame_w = (int)surface->surfaceList[batch_idx].width;
        int frame_h = (int)surface->surfaceList[batch_idx].height;
        int visible_frame_h = ENABLE_INFER_BOTTOM_PADDING
            ? std::max(1, frame_h - INFER_BOTTOM_PAD_PX)
            : frame_h;

        /* On dGPU, NvBufSurface uses CUDA device memory (type 2) which cannot
         * be mapped to CPU. Use cudaMemcpy2D to copy RGBA pixels to a persistent
         * CPU staging buffer. At 1080p RGBA ~8MB/frame, this is trivial for PCIe. */
        {
            static uint8_t *s_cpu_buf = nullptr;
            static size_t s_cpu_buf_size = 0;
            uint32_t src_pitch = surface->surfaceList[batch_idx].pitch;
            size_t needed = (size_t)src_pitch * frame_h;
            if (needed > s_cpu_buf_size) {
                free(s_cpu_buf);
                s_cpu_buf = (uint8_t *)malloc(needed);
                s_cpu_buf_size = needed;
            }
            if (s_cpu_buf) {
                void *src_ptr = surface->surfaceList[batch_idx].dataPtr;
                cudaError_t err = cudaMemcpy2D(
                    s_cpu_buf, src_pitch,       // dst, dst pitch
                    src_ptr,   src_pitch,       // src, src pitch
                    frame_w * 4, frame_h,       // width in bytes, height
                    cudaMemcpyDeviceToHost);
                if (err == cudaSuccess) {
                    pixel_data  = s_cpu_buf;
                    pixel_pitch = src_pitch;
                    pixels_mapped = true;
                }
            }
        }

        /* ============================================================
         * STEP 1: Parse tracked objects — separate by class
         * ============================================================ */
        std::vector<Detection> trolley_dets, mouth_dets;

        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list;
             l_obj != NULL; l_obj = l_obj->next)
        {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);

            // Skip untracked (NvDCF uses UINT64_MAX, track_id=0 is valid)
            if (obj_meta->object_id == UINT64_MAX) continue;

            Detection det;
            det.track_id = obj_meta->object_id;
            det.bbox[0]  = clamp_f(obj_meta->rect_params.left, 0.0f, (float)(frame_w - 1));
            det.bbox[1]  = clamp_f(obj_meta->rect_params.top, 0.0f, (float)(visible_frame_h - 1));
            det.bbox[2]  = clamp_f(obj_meta->rect_params.left + obj_meta->rect_params.width,
                                   0.0f, (float)(frame_w - 1));
            det.bbox[3]  = clamp_f(obj_meta->rect_params.top + obj_meta->rect_params.height,
                                   0.0f, (float)(visible_frame_h - 1));
            det.conf     = obj_meta->confidence;
            det.class_id = obj_meta->class_id;

            if (det.bbox[2] <= det.bbox[0] || det.bbox[3] <= det.bbox[1]) continue;

            if (det.class_id == CLASS_TROLLEY)      trolley_dets.push_back(det);
            else if (det.class_id == CLASS_LADLE_MOUTH) mouth_dets.push_back(det);
        }

        /* ============================================================
         * STEP 2: Update trolley states
         * ============================================================ */
        std::set<uint64_t> seen_tids;
        for (auto &tr : trolley_dets) {
            seen_tids.insert(tr.track_id);
            TrolleyState *st = nullptr;
            auto it = g_trolley_states.find(tr.track_id);
            if (it == g_trolley_states.end()) {
                st = new TrolleyState();
                st->tid = tr.track_id;
                g_trolley_states[tr.track_id] = st;
                if (g_trolley_id_to_count.find(tr.track_id) == g_trolley_id_to_count.end())
                    g_trolley_id_to_count[tr.track_id] = 0;
            } else {
                st = it->second;
                // Check new-trolley (reappeared after long absence)
                if (st->last_disappeared_f >= 0 &&
                    (frame_idx - st->last_disappeared_f) > sec_to_frames(TROLLEY_RESET_S, g_video_fps)) {
                    g_trolley_id_to_count[tr.track_id] = 0;
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
        for (auto &kv : g_trolley_states) {
            TrolleyState *st = kv.second;
            if (st->last_seen_f >= 0 &&
                (frame_idx - st->last_seen_f) > sec_to_frames(TROLLEY_GONE_S, g_video_fps)) {
                if (!st->completed.empty() || st->session_active ||
                    !st->mould_completed_times.empty()) {
                    finalize_trolley(st, frame_idx);
                }
                to_remove.push_back(kv.first);
            }
        }
        for (uint64_t tid : to_remove) {
            delete g_trolley_states[tid];
            g_trolley_states.erase(tid);
        }

        /* ============================================================
         * STEP 4: Associate mouths to trolleys + full state machine
         * ============================================================ */
        struct ProbeHit {
            const Detection *det = nullptr;
            float probe_x = 0.0f;
            float probe_y = 0.0f;
            float head_score = 0.0f;
            float tail_score = 0.0f;
            bool is_pouring = false;
        };

        std::unordered_map<uint64_t, std::vector<ProbeHit>> probes_by_trolley;
        probes_by_trolley.reserve(g_trolley_states.size());

        for (auto &kv : g_trolley_states) {
            TrolleyState *st = kv.second;
            st->active_probe_from_hold = false;
        }

        for (auto &m : mouth_dets) {
            float probe_x, probe_y;
            mouth_probe_point(m.bbox, probe_x, probe_y);

            TrolleyState *best_st = nullptr;
            float best_center_y = -FLT_MAX;
            for (auto &kv : g_trolley_states) {
                TrolleyState *st = kv.second;
                bool bbox_fresh = (st->bbox_valid && st->last_seen_f == frame_idx);
                if (!bbox_fresh) continue;
                if (!point_in_bbox(probe_x, probe_y, st->bbox)) continue;

                float tcx, tcy;
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
            if (pixels_mapped) {
                hit.head_score = flare_score(pixel_data, pixel_pitch, frame_w, frame_h,
                                             hit.probe_x, hit.probe_y);
                hit.tail_score = flare_score(pixel_data, pixel_pitch, frame_w, frame_h,
                                             hit.probe_x, hit.probe_y + PROBE_TAIL_DY);
            }
            hit.is_pouring = probe_is_pouring(hit.head_score, hit.tail_score);

            probes_by_trolley[best_st->tid].push_back(hit);
        }

        for (auto &kv : g_trolley_states) {
            TrolleyState *st = kv.second;
            if (!st->bbox_valid) continue;

            bool bbox_fresh = (st->last_seen_f == frame_idx);
            auto probes_it = probes_by_trolley.find(st->tid);
            std::vector<ProbeHit> *probes =
                (probes_it != probes_by_trolley.end()) ? &probes_it->second : nullptr;
            bool has_current_probe = (probes && !probes->empty());
            bool any_probe_pouring = false;
            const ProbeHit *selected = nullptr;

            if (has_current_probe) {
                for (auto &probe : *probes) {
                    any_probe_pouring = any_probe_pouring || probe.is_pouring;
                    if (probe.det->track_id == st->active_probe_track_id) {
                        selected = &probe;
                    }
                }
                if (!selected) {
                    for (auto &probe : *probes) {
                        if (!probe.is_pouring) continue;
                        if (!selected || probe.tail_score > selected->tail_score ||
                            (probe.tail_score == selected->tail_score &&
                             probe.head_score > selected->head_score) ||
                            (probe.tail_score == selected->tail_score &&
                             probe.head_score == selected->head_score &&
                             probe.det->conf > selected->det->conf)) {
                            selected = &probe;
                        }
                    }
                }
                if (!selected) {
                    selected = &(*probes)[0];
                    for (auto &probe : *probes) {
                        if (probe.head_score > selected->head_score ||
                            (probe.head_score == selected->head_score &&
                             probe.det->conf > selected->det->conf)) {
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

            /* ---- Session start ---- */
            if (!st->session_active && has_current_probe && st->out_count == 0 && st->in_count >= N_ENTER) {
                st->session_active  = true;
                st->session_start_f = frame_idx - N_ENTER + 1;
                st->session_end_f   = -1;
                st->pour_active     = false;
                st->pour_on_count   = 0;
                st->pour_off_count  = 0;
                st->mould_switch_count = 0;
                st->mould_anchor_valid = false;
                delete st->current_seg;
                st->current_seg = nullptr;
                clear_frozen_probe(st);
            }

            /* ---- Session end ---- */
            if (st->session_active) {
                int probe_missing = frame_idx - st->active_probe_last_seen_f;
                if (st->out_count >= N_EXIT || probe_missing > MOUTH_MISSING_TOL) {
                    // Close open pour
                    if (st->pour_active && st->current_seg != nullptr) {
                        st->current_seg->end_f = frame_idx;
                        if (st->current_seg->duration_frames() >= MIN_POUR_DURATION_FRAMES) {
                            st->current_mould_id++;
                            st->mould_completed_times[st->current_mould_id] = st->current_seg->duration_frames();
                            g_trolley_id_to_count[st->tid] = st->current_mould_id;
                            st->completed.push_back(*st->current_seg);
                        }
                    }
                    st->pour_active = false;
                    delete st->current_seg;
                    st->current_seg = nullptr;
                    st->session_active = false;
                    st->session_end_f  = frame_idx;
                    st->mould_switch_count = 0;
                    st->mould_anchor_valid = false;
                    clear_frozen_probe(st);
                    clear_active_probe(st);
                    continue;
                }

                /* ---- Pour logic ---- */
                bool use_mouth = false;
                bool using_hold = false;
                float mouth_nx = -1.0f, mouth_ny = -1.0f;
                float probe_x_use = 0.0f, probe_y_use = 0.0f;
                bool active_probe_pouring = false;

                if (st->pour_active && st->frozen_probe_active) {
                    probe_x_use = st->frozen_probe_x;
                    probe_y_use = st->frozen_probe_y;
                    float frozen_head = 0.0f, frozen_tail = 0.0f;
                    if (pixels_mapped) {
                        frozen_head = flare_score(pixel_data, pixel_pitch, frame_w, frame_h,
                                                  probe_x_use, probe_y_use);
                        frozen_tail = flare_score(pixel_data, pixel_pitch, frame_w, frame_h,
                                                  probe_x_use, probe_y_use + PROBE_TAIL_DY);
                    }
                    active_probe_pouring = probe_is_pouring(frozen_head, frozen_tail);
                    use_mouth = true;
                } else if (selected) {
                    probe_x_use = selected->probe_x;
                    probe_y_use = selected->probe_y;
                    active_probe_pouring = selected->is_pouring;
                    use_mouth = true;
                } else if (bbox_fresh && st->pour_active &&
                           st->active_probe_pt_valid &&
                           (frame_idx - st->active_probe_last_seen_f) <= MOUTH_HOLD_DUR) {
                    probe_x_use = st->active_probe_pt_px[0];
                    probe_y_use = st->active_probe_pt_px[1];
                    float held_head = 0.0f, held_tail = 0.0f;
                    if (pixels_mapped) {
                        held_head = flare_score(pixel_data, pixel_pitch, frame_w, frame_h,
                                                probe_x_use, probe_y_use);
                        held_tail = flare_score(pixel_data, pixel_pitch, frame_w, frame_h,
                                                probe_x_use, probe_y_use + PROBE_TAIL_DY);
                    }
                    active_probe_pouring = probe_is_pouring(held_head, held_tail);
                    use_mouth = true;
                    using_hold = true;
                    st->active_probe_from_hold = true;
                }

                if (!use_mouth) continue;

                // Normalized position of pour dot in trolley
                norm_point_in_trolley(probe_x_use, probe_y_use, st->bbox, mouth_nx, mouth_ny);
                bool mouth_norm_valid = bbox_fresh && (mouth_nx >= 0.0f) && (mouth_ny >= 0.0f);

                if (!st->pour_active) {
                    /* ---- Pour OFF → check for pour start ---- */
                    st->pour_on_count = any_probe_pouring ? (st->pour_on_count + 1) : 0;

                    if (st->pour_on_count >= K_ON && mouth_norm_valid) {
                        st->pour_active = true;
                        st->frozen_probe_active = true;
                        st->frozen_probe_x = probe_x_use;
                        st->frozen_probe_y = probe_y_use;
                        if (selected) {
                            memcpy(st->frozen_probe_bbox, selected->det->bbox, sizeof(float) * 4);
                            st->frozen_probe_bbox_valid = true;
                        } else if (st->active_probe_bbox_valid) {
                            memcpy(st->frozen_probe_bbox, st->active_probe_bbox, sizeof(float) * 4);
                            st->frozen_probe_bbox_valid = true;
                        } else {
                            st->frozen_probe_bbox_valid = false;
                        }
                        int start_f = frame_idx - K_ON + 1;
                        st->current_seg = new PourSegment();
                        st->current_seg->start_f = start_f;
                        st->current_seg->end_f   = frame_idx;
                        st->current_seg->mouth_pts_norm.push_back({mouth_nx, mouth_ny});
                        st->pour_off_count = 0;
                        st->mould_switch_count = 0;
                        if (mouth_norm_valid) {
                            st->mould_anchor_pt[0] = mouth_nx;
                            st->mould_anchor_pt[1] = mouth_ny;
                            st->mould_anchor_valid = true;
                        } else {
                            st->mould_anchor_valid = false;
                        }
                    }
                } else {
                    /* ---- Pour ON → check displacement + pour-off ---- */

                    // Anchor-based mould switch detection
                    if (mouth_norm_valid && st->mould_anchor_valid) {
                        float dx = std::abs(mouth_nx - st->mould_anchor_pt[0]);
                        float dy = std::abs(mouth_ny - st->mould_anchor_pt[1]);

                        if (dx > D_SPLIT || dy > D_SPLIT) {
                            st->mould_switch_count++;
                            if (st->mould_switch_count >= MOULD_SWITCH_HOLD_F) {
                                // Set end_f first, THEN check actual segment duration.
                                // current_pour_dur = frame_idx - start_f includes the hold period;
                                // the stored left segment ends at frame_idx - MOULD_SWITCH_HOLD_F + 1,
                                // so its actual duration can be shorter than MIN_POUR_DURATION_FRAMES.
                                st->current_seg->end_f = frame_idx - MOULD_SWITCH_HOLD_F + 1;
                                int left_dur = st->current_seg->duration_frames();
                                if (left_dur >= MIN_POUR_DURATION_FRAMES) {
                                    // Close current segment
                                    st->current_mould_id++;
                                    st->mould_completed_times[st->current_mould_id] = left_dur;
                                    st->completed.push_back(*st->current_seg);
                                    g_trolley_id_to_count[st->tid] = st->current_mould_id;

                                    // Start new segment
                                    delete st->current_seg;
                                    st->current_seg = new PourSegment();
                                    st->current_seg->start_f = frame_idx - MOULD_SWITCH_HOLD_F + 1;
                                    st->current_seg->end_f   = frame_idx;
                                    st->current_seg->mouth_pts_norm.push_back({mouth_nx, mouth_ny});
                                    st->mould_anchor_pt[0] = mouth_nx;
                                    st->mould_anchor_pt[1] = mouth_ny;
                                    st->mould_switch_count = 0;
                                } else {
                                    // Pour too short — ignore switch
                                    st->mould_switch_count = 0;
                                }
                            }
                        } else {
                            st->mould_switch_count = 0;
                        }
                    } else {
                        st->mould_switch_count = 0;
                    }

                    // Update current segment
                    st->current_seg->end_f = frame_idx;
                    if (mouth_norm_valid) {
                        st->current_seg->mouth_pts_norm.push_back({mouth_nx, mouth_ny});
                    }

                    // Pour-off check
                    bool trolley_still_pouring = any_probe_pouring || (using_hold && active_probe_pouring);
                    st->pour_off_count = trolley_still_pouring ? 0 : (st->pour_off_count + 1);

                    if (st->pour_off_count >= K_OFF) {
                        st->current_seg->end_f = frame_idx - K_OFF + 1;
                        if (st->current_seg->duration_frames() >= MIN_POUR_DURATION_FRAMES) {
                            st->current_mould_id++;
                            st->mould_completed_times[st->current_mould_id] =
                                st->current_seg->duration_frames();
                            g_trolley_id_to_count[st->tid] = st->current_mould_id;
                            st->completed.push_back(*st->current_seg);
                        }
                        st->pour_active = false;
                        delete st->current_seg;
                        st->current_seg = nullptr;
                        st->pour_on_count  = 0;
                        st->pour_off_count = 0;
                        st->mould_switch_count = 0;
                        st->mould_anchor_valid = false;
                        clear_frozen_probe(st);
                    }
                }
            }
        } /* end per-trolley loop */

        /* ============================================================
         * STEP 5: Update display metadata (bbox colors, labels, overlay)
         * ============================================================ */

        DisplayMetaHelper dm(batch_meta, frame_meta);
        std::unordered_map<uint64_t, ClusterPreview> overlay_preview_by_tid;
        overlay_preview_by_tid.reserve(g_trolley_states.size());
        for (auto &kv : g_trolley_states) {
            TrolleyState *st = kv.second;
            if (st->session_start_f >= 0) {
                overlay_preview_by_tid.emplace(
                    kv.first,
                    build_cluster_preview_for_overlay(st, frame_idx));
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
                auto it = g_trolley_states.find(obj->object_id);
                if (it != g_trolley_states.end()) st = it->second;
                bool pouring = (st && st->pour_active);

                obj->rect_params.border_width = 3;
                if (pouring) {
                    obj->rect_params.border_color = {0.0f, 1.0f, 0.0f, 1.0f}; // green
                } else {
                    obj->rect_params.border_color = {1.0f, 1.0f, 0.0f, 1.0f}; // cyan (RGBA)
                }
                obj->rect_params.has_bg_color = 0;

                // Label
                char label[128];
                int cnt = 0;
                auto pit = overlay_preview_by_tid.find(obj->object_id);
                if (pit != overlay_preview_by_tid.end()) {
                    cnt = pit->second.mould_count;
                } else {
                    auto cit = g_trolley_id_to_count.find(obj->object_id);
                    if (cit != g_trolley_id_to_count.end()) cnt = cit->second;
                }
                if (cnt > 0) {
                    snprintf(label, sizeof(label), "T%lu (%dM)", (unsigned long)obj->object_id, cnt);
                } else {
                    snprintf(label, sizeof(label), "T%lu", (unsigned long)obj->object_id);
                }

                if (obj->text_params.display_text)
                    g_free(obj->text_params.display_text);
                obj->text_params.display_text = g_strdup(label);
                obj->text_params.font_params.font_name  = (gchar *)"Sans";
                obj->text_params.font_params.font_size  = 12;
                obj->text_params.font_params.font_color = obj->rect_params.border_color;
                obj->text_params.set_bg_clr = 0;

            } else if (obj->class_id == CLASS_LADLE_MOUTH) {
                obj->rect_params.border_width = 3;
                obj->rect_params.border_color = {1.0f, 0.0f, 1.0f, 1.0f}; // magenta
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
                
                if (pixels_mapped) {
                    float dot_x = obj->rect_params.left + obj->rect_params.width / 2.0f;
                    float dot_y = obj->rect_params.top + obj->rect_params.height +
                                  POUR_DOT_BELOW_PX;
                    float head_score = flare_score(pixel_data, pixel_pitch, frame_w, frame_h,
                                                   dot_x, dot_y);
                    float tail_score = flare_score(pixel_data, pixel_pitch, frame_w, frame_h,
                                                   dot_x, dot_y + PROBE_TAIL_DY);
                    bool probe_green = probe_is_pouring(head_score, tail_score);

                    NvOSD_CircleParams *cp = dm.next_circle();
                    cp->xc = (unsigned int)dot_x;
                    cp->yc = (unsigned int)dot_y;
                    cp->radius = 10;
                    cp->circle_color = probe_green
                        ? NvOSD_ColorParams{0.0f, 1.0f, 0.0f, 1.0f}
                        : NvOSD_ColorParams{1.0f, 1.0f, 1.0f, 1.0f};
                    cp->has_bg_color = 1;
                    cp->bg_color = cp->circle_color;
                }
            }
        }

        /* --- Info panel overlay (top-left) --- */
        {
            // Collect active trolley info
            struct TrolleyInfo {
                uint64_t tid;
                const ClusterPreview *preview;
            };
            std::vector<TrolleyInfo> active_info;
            for (auto &kv : g_trolley_states) {
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
                // Calculate panel height
                int panel_h = 30;
                for (auto &ai : active_info) {
                    panel_h += 25 + 22;  // header + total line
                    int nm = ai.preview ? (int)ai.preview->mould_total_frames.size() : 0;
                    panel_h += nm * 20 + 10;
                }
                panel_h = std::max(60, panel_h);

                // Panel background
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

                    // Trolley header
                    snprintf(txt, sizeof(txt), "Trolley #%lu", (unsigned long)ai.tid);
                    setup_text(dm.next_text(), txt, 25, y_off, 10, 1.0f, 1.0f, 1.0f, 1.0f);
                    y_off += 25;

                    // Total moulds
                    int cnt = ai.preview ? ai.preview->mould_count : 0;
                    snprintf(txt, sizeof(txt), "  Total Moulds: %d", cnt);
                    setup_text(dm.next_text(), txt, 25, y_off, 9, 0.78f, 0.78f, 0.78f, 1.0f);
                    y_off += 22;

                    // Per-mould times
                    int num_moulds = ai.preview ? (int)ai.preview->mould_total_frames.size() : 0;
                    for (int m = 0; m < num_moulds; m++) {
                        float total_s = (float)ai.preview->mould_total_frames[m] / g_video_fps;
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

        /* pixels_mapped: CPU buffer is static, no unmap needed */
    } /* end per-frame loop */

    gst_buffer_unmap(buf, &map_info);
    return GST_PAD_PROBE_OK;
}

/* ================================================================
 * PIPELINE CALLBACKS
 * ================================================================ */

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
    GMainLoop *loop = (GMainLoop *)data;

    switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS: {
        g_print("\nEnd of stream\n");

        // Finalize remaining trolleys
        for (auto &kv : g_trolley_states) {
            if (!kv.second->mould_completed_times.empty()) {
                bool already = false;
                for (auto &s : g_final_summaries) {
                    if (s["trolley_id"].get<int>() == (int)kv.first) { already = true; break; }
                }
                if (!already) finalize_trolley(kv.second, g_frame_count);
            }
        }

        // Write JSON
        write_json_output();

        // Write CSV from final summaries (same source data as JSON)
        write_csv_output();

        // Print summary
        auto elapsed = std::chrono::steady_clock::now() - g_start_time;
        double secs = std::chrono::duration<double>(elapsed).count();
        double proc_fps = (secs > 0) ? (double)g_frame_count / secs : 0;
        int total_moulds = 0;
        for (auto &s : g_final_summaries) {
            total_moulds += s.value("mould_count", 0);
        }

        g_print("\nDeepStream Pouring Inference complete!\n");
        g_print("  Frames processed: %d\n", g_frame_count);
        g_print("  Processing time:  %.1fs\n", secs);
        g_print("  Processing speed: %.1f FPS\n", proc_fps);
        g_print("  Unique trolleys:  %lu\n", (unsigned long)g_trolley_id_to_count.size());
        g_print("  Total moulds:     %d\n", total_moulds);
        g_print("  Summaries:        %lu\n", (unsigned long)g_final_summaries.size());
        g_print("  Output video:     %s\n", g_output_video_path.c_str());
        g_print("  Output JSON:      %s\n", g_output_json_path.c_str());
        g_print("  Output CSV:       %s\n", g_output_csv_path.c_str());

        // Cleanup trolley states
        for (auto &kv : g_trolley_states) delete kv.second;
        g_trolley_states.clear();

        g_main_loop_quit(loop);
        break;
    }
    case GST_MESSAGE_ERROR: {
        gchar *debug;
        GError *error;
        gst_message_parse_error(msg, &error, &debug);
        g_printerr("ERROR from element %s: %s\n",
                    GST_OBJECT_NAME(msg->src), error->message);
        if (debug) g_printerr("Debug: %s\n", debug);
        g_free(debug);
        g_error_free(error);
        g_main_loop_quit(loop);
        break;
    }
    default:
        break;
    }
    return TRUE;
}

/* --- Dynamic pad callback for uridecodebin --- */
static GstPad *g_mux_sinkpad = nullptr;  // request pad, created once

static void cb_newpad(GstElement *decodebin, GstPad *decoder_src_pad, gpointer data)
{
    GstElement *streammux = (GstElement *)data;

    // Request pad once; reuse on subsequent calls
    if (!g_mux_sinkpad) {
        g_mux_sinkpad = gst_element_request_pad_simple(streammux, "sink_0");
    }
    GstPad *sinkpad = g_mux_sinkpad;

    if (!sinkpad) {
        g_printerr("Failed to request sink_0 pad from streammux\n");
        return;
    }
    if (gst_pad_is_linked(sinkpad)) {
        return;  // already linked
    }

    GstCaps *caps = gst_pad_get_current_caps(decoder_src_pad);
    if (!caps) caps = gst_pad_query_caps(decoder_src_pad, NULL);

    const GstStructure *str = gst_caps_get_structure(caps, 0);
    const gchar *name = gst_structure_get_name(str);

    if (g_strrstr(name, "video")) {
        // Extract FPS
        gint fps_n = 0, fps_d = 1;
        if (gst_structure_get_fraction(str, "framerate", &fps_n, &fps_d) && fps_d > 0) {
            g_video_fps = (float)fps_n / (float)fps_d;
            g_print("Detected video FPS: %.2f\n", g_video_fps);
            recompute_thresholds(g_video_fps);
        }

        if (gst_pad_link(decoder_src_pad, sinkpad) != GST_PAD_LINK_OK) {
            g_printerr("Failed to link decoder to streammux\n");
        }
    }

    gst_caps_unref(caps);
    // Note: sinkpad is g_mux_sinkpad (request pad), do NOT unref here — reused
}

/* ================================================================
 * MAIN
 * ================================================================ */

int main(int argc, char *argv[])
{
    /* --- Argument parsing --- */
    if (argc < 2) {
        g_printerr("Usage: %s <input_video.mp4> [output_dir]\n", argv[0]);
        return -1;
    }

    std::string input_path = argv[1];
    g_output_dir = (argc >= 3) ? argv[2] : "/workspace/output";

    std::string base = basename_no_ext(input_path);
    g_output_video_path = g_output_dir + "/" + base + "_annotated.mp4";
    g_output_json_path  = g_output_dir + "/" + base + "_annotated.json";
    g_output_csv_path   = g_output_dir + "/" + base + "_annotated.csv";
    g_input_uri = std::string("file://") + input_path;

    mkdir_p(g_output_dir);

    // Initialize thresholds with default FPS (recomputed in cb_newpad)
    g_video_fps = 25.0f;
    recompute_thresholds(g_video_fps);

    /* --- GStreamer init --- */
    gst_init(&argc, &argv);
    g_main_loop = g_main_loop_new(NULL, FALSE);

    /* --- Create elements --- */
    GstElement *pipeline   = gst_pipeline_new("pouring-pipeline");
    GstElement *source     = gst_element_factory_make("uridecodebin",     "source");
    GstElement *streammux  = gst_element_factory_make("nvstreammux",      "streammux");
    GstElement *prepadconv = gst_element_factory_make("nvvideoconvert",   "prepadconv");
    GstElement *prepadcaps = gst_element_factory_make("capsfilter",       "prepad_caps");
    GstElement *pgie       = gst_element_factory_make("nvinfer",          "pgie");
    GstElement *tracker    = gst_element_factory_make("nvtracker",        "tracker");
    GstElement *nvvidconv  = gst_element_factory_make("nvvideoconvert",   "nvvidconv");
    GstElement *capsfilter = gst_element_factory_make("capsfilter",       "rgba_caps");
    GstElement *nvosd      = gst_element_factory_make("nvdsosd",          "nvosd");
    GstElement *nvvidconv2 = gst_element_factory_make("nvvideoconvert",   "nvvidconv2");
    GstElement *cropcaps   = gst_element_factory_make("capsfilter",       "crop_caps");
    GstElement *encoder    = gst_element_factory_make("nvv4l2h264enc",    "encoder");
    GstElement *h264parse  = gst_element_factory_make("h264parse",        "h264parse");
    GstElement *mux        = gst_element_factory_make("mp4mux",           "mux");
    GstElement *sink       = gst_element_factory_make("filesink",         "filesink");

    if (!pipeline || !source || !streammux || !prepadconv || !prepadcaps || !pgie || !tracker ||
        !nvvidconv || !capsfilter || !nvosd || !nvvidconv2 || !cropcaps ||
        !encoder || !h264parse || !mux || !sink) {
        g_printerr("Failed to create one or more GStreamer elements.\n");
        return -1;
    }

    /* --- Configure elements --- */

    // Source
    g_object_set(G_OBJECT(source), "uri", g_input_uri.c_str(), NULL);

    // Streammux
    g_object_set(G_OBJECT(streammux),
        "width",               MUXER_WIDTH,
        "height",              MUXER_HEIGHT,
        "batch-size",          MUXER_BATCH_SIZE,
        "batched-push-timeout", 40000,
        "live-source",         FALSE,
        NULL);

    std::string prepad_dest_crop =
        "0:0:" + std::to_string(MUXER_WIDTH) + ":" + std::to_string(VISIBLE_FRAME_HEIGHT);
    g_object_set(G_OBJECT(prepadconv),
        "dest-crop", prepad_dest_crop.c_str(),
        "disable-passthrough", TRUE,
        NULL);
    std::string prepad_caps_str =
        "video/x-raw(memory:NVMM), format=NV12, width=" + std::to_string(MUXER_WIDTH) +
        ", height=" + std::to_string(INFER_FRAME_HEIGHT);
    GstCaps *prepad_caps = gst_caps_from_string(prepad_caps_str.c_str());
    g_object_set(G_OBJECT(prepadcaps), "caps", prepad_caps, NULL);
    gst_caps_unref(prepad_caps);

    // PGIE
    g_object_set(G_OBJECT(pgie),
        "config-file-path", PGIE_CONFIG_FILE,
        NULL);

    // Tracker
    g_object_set(G_OBJECT(tracker),
        "tracker-width",  960,
        "tracker-height", 544,
        "ll-lib-file",    TRACKER_LIB,
        "ll-config-file", TRACKER_CONFIG,
        "gpu-id",         0,
        NULL);

    // Capsfilter: force RGBA so brightness probe reads correct pixel format
    GstCaps *caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=RGBA");
    g_object_set(G_OBJECT(capsfilter), "caps", caps, NULL);
    gst_caps_unref(caps);

    // OSD
    g_object_set(G_OBJECT(nvosd),
        "process-mode",   0,     // CPU mode for reliable text/alpha
        "display-text",   TRUE,
        "display-bbox",   TRUE,
        "display-mask",   FALSE,
        NULL);

    // Encoder
    std::string output_src_crop =
        "0:0:" + std::to_string(MUXER_WIDTH) + ":" + std::to_string(VISIBLE_FRAME_HEIGHT);
    g_object_set(G_OBJECT(nvvidconv2),
        "src-crop", output_src_crop.c_str(),
        "disable-passthrough", TRUE,
        NULL);
    std::string crop_caps_str =
        "video/x-raw(memory:NVMM), format=NV12, width=" + std::to_string(MUXER_WIDTH) +
        ", height=" + std::to_string(VISIBLE_FRAME_HEIGHT);
    GstCaps *crop_caps = gst_caps_from_string(crop_caps_str.c_str());
    g_object_set(G_OBJECT(cropcaps), "caps", crop_caps, NULL);
    gst_caps_unref(crop_caps);

    g_object_set(G_OBJECT(encoder),
        "bitrate", ENCODER_BITRATE,
        NULL);

    // Filesink
    g_object_set(G_OBJECT(sink),
        "location", g_output_video_path.c_str(),
        "sync",     FALSE,
        NULL);

    /* --- Build pipeline --- */
    gst_bin_add_many(GST_BIN(pipeline),
        source, streammux, prepadconv, prepadcaps, pgie, tracker, nvvidconv,
        capsfilter, nvosd, nvvidconv2, cropcaps, encoder, h264parse, mux, sink, NULL);

    // Link: streammux -> prepadconv -> prepadcaps -> pgie -> tracker -> nvvidconv
    //   -> capsfilter(RGBA) -> nvosd -> nvvidconv2(crop) -> cropcaps -> encoder -> h264parse -> mux -> sink
    if (!gst_element_link_many(streammux, prepadconv, prepadcaps, pgie, tracker, nvvidconv,
                                capsfilter, nvosd, nvvidconv2, cropcaps,
                                encoder, h264parse, mux, sink, NULL)) {
        g_printerr("Failed to link pipeline elements.\n");
        return -1;
    }

    // Connect uridecodebin dynamic pad -> streammux sink_0
    g_signal_connect(G_OBJECT(source), "pad-added", G_CALLBACK(cb_newpad), streammux);

    /* --- Attach OSD probe --- */
    GstPad *osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
    if (!osd_sink_pad) {
        g_printerr("Unable to get nvosd sink pad\n");
        return -1;
    }
    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                       osd_sink_pad_buffer_probe, NULL, NULL);
    gst_object_unref(osd_sink_pad);

    /* --- Bus watch --- */
    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    guint bus_watch_id = gst_bus_add_watch(bus, bus_call, g_main_loop);
    gst_object_unref(bus);

    /* --- Run --- */
    g_start_time = std::chrono::steady_clock::now();

    g_print("========================================\n");
    g_print("DeepStream Pouring Detection Pipeline\n");
    g_print("  Input:  %s\n", input_path.c_str());
    g_print("  Output: %s\n", g_output_video_path.c_str());
    g_print("========================================\n");
    g_print("Setting pipeline to PLAYING...\n");

    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_main_loop_run(g_main_loop);

    /* --- Cleanup --- */
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(g_main_loop);

    return 0;
}
