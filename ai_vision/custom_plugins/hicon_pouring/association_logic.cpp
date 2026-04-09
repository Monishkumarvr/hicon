#include "association_logic.h"

#include <algorithm>
#include <cfloat>

namespace hicon_pouring_assoc {

namespace {

inline void bbox_center(const Box &bbox, float &cx, float &cy) {
    cx = (bbox.x1 + bbox.x2) / 2.0f;
    cy = (bbox.y1 + bbox.y2) / 2.0f;
}

inline bool point_in_box(float px, float py, const Box &bbox) {
    return (px >= bbox.x1 && px <= bbox.x2 && py >= bbox.y1 && py <= bbox.y2);
}

inline Box expand_box(const Box &bbox, float expand_x, float expand_y) {
    return Box{
        bbox.x1 - expand_x,
        bbox.y1 - expand_y,
        bbox.x2 + expand_x,
        bbox.y2 + expand_y,
    };
}

inline void normalize_point_in_box(float px, float py, const Box &bbox, float &nx, float &ny) {
    float w = std::max(1.0f, bbox.x2 - bbox.x1);
    float h = std::max(1.0f, bbox.y2 - bbox.y1);
    nx = (px - bbox.x1) / w;
    ny = (py - bbox.y1) / h;
}

}  // namespace

float bbox_iou(const Box &a, const Box &b) {
    const float inter_x1 = std::max(a.x1, b.x1);
    const float inter_y1 = std::max(a.y1, b.y1);
    const float inter_x2 = std::min(a.x2, b.x2);
    const float inter_y2 = std::min(a.y2, b.y2);
    const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    const float inter_area = inter_w * inter_h;
    if (inter_area <= 0.0f) {
        return 0.0f;
    }

    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float denom = area_a + area_b - inter_area;
    return denom > 0.0f ? (inter_area / denom) : 0.0f;
}

int find_best_trolley_for_mouth(const MouthCandidate &mouth,
                                const std::vector<TrolleyCandidate> &fresh_trolleys,
                                float expand_x,
                                float expand_y) {
    float mouth_cx = 0.0f;
    float mouth_cy = 0.0f;
    bbox_center(mouth.bbox, mouth_cx, mouth_cy);

    int best_idx = -1;
    float best_center_y = -FLT_MAX;
    for (size_t i = 0; i < fresh_trolleys.size(); ++i) {
        const auto &trolley = fresh_trolleys[i];
        Box expanded = expand_box(trolley.bbox, expand_x, expand_y);
        if (!point_in_box(mouth_cx, mouth_cy, expanded)) {
            continue;
        }

        float trolley_cx = 0.0f;
        float trolley_cy = 0.0f;
        bbox_center(trolley.bbox, trolley_cx, trolley_cy);
        if (best_idx < 0 || trolley_cy > best_center_y) {
            best_idx = static_cast<int>(i);
            best_center_y = trolley_cy;
        }
    }

    return best_idx;
}

int find_best_handoff_trolley(const Box &new_bbox,
                              const std::vector<TrolleyCandidate> &candidates,
                              float min_iou) {
    int best_idx = -1;
    float best_iou = min_iou;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const float iou = bbox_iou(new_bbox, candidates[i].bbox);
        if (iou >= best_iou) {
            best_idx = static_cast<int>(i);
            best_iou = iou;
        }
    }
    return best_idx;
}

int select_mouth_candidate_index(const std::vector<MouthCandidate> &candidates,
                                 uint64_t active_track_id) {
    if (candidates.empty()) {
        return -1;
    }

    if (active_track_id != 0 && active_track_id != UINT64_MAX) {
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (candidates[i].track_id == active_track_id) {
                return static_cast<int>(i);
            }
        }
    }

    int best_idx = 0;
    for (size_t i = 1; i < candidates.size(); ++i) {
        if (candidates[i].conf > candidates[best_idx].conf) {
            best_idx = static_cast<int>(i);
        }
    }
    return best_idx;
}

bool is_within_hold_window(int frame_idx, int last_seen_f, int hold_dur) {
    return hold_dur >= 0 && last_seen_f >= 0 && (frame_idx - last_seen_f) <= hold_dur;
}

void normalize_point_in_expanded_trolley(float px, float py,
                                         const Box &bbox,
                                         float expand_x,
                                         float expand_y,
                                         float &nx,
                                         float &ny) {
    Box expanded = expand_box(bbox, expand_x, expand_y);
    normalize_point_in_box(px, py, expanded, nx, ny);
}

}  // namespace hicon_pouring_assoc
