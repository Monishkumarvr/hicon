#pragma once

#include <cstdint>
#include <vector>

namespace hicon_pouring_assoc {

struct Box {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

struct TrolleyCandidate {
    uint64_t tid = 0;
    Box bbox;
};

struct MouthCandidate {
    uint64_t track_id = 0;
    float conf = 0.0f;
    Box bbox;
};

int find_best_trolley_for_mouth(const MouthCandidate &mouth,
                                const std::vector<TrolleyCandidate> &fresh_trolleys,
                                float expand_x,
                                float expand_y);

float bbox_iou(const Box &a, const Box &b);

int find_best_handoff_trolley(const Box &new_bbox,
                              const std::vector<TrolleyCandidate> &candidates,
                              float min_iou);

int select_mouth_candidate_index(const std::vector<MouthCandidate> &candidates,
                                 uint64_t active_track_id);

bool is_within_hold_window(int frame_idx, int last_seen_f, int hold_dur);

void normalize_point_in_expanded_trolley(float px, float py,
                                         const Box &bbox,
                                         float expand_x,
                                         float expand_y,
                                         float &nx,
                                         float &ny);

}  // namespace hicon_pouring_assoc
