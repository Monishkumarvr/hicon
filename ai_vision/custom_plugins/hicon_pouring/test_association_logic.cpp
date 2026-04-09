#include "association_logic.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

namespace assoc = hicon_pouring_assoc;

static assoc::Box make_box(float x1, float y1, float x2, float y2) {
    return assoc::Box{x1, y1, x2, y2};
}

int main() {
    {
        assoc::MouthCandidate mouth{101, 0.85f, make_box(145.0f, 145.0f, 155.0f, 155.0f)};
        std::vector<assoc::TrolleyCandidate> trolleys = {
            {1, make_box(100.0f, 100.0f, 200.0f, 200.0f)},
        };
        // Mouth center (150,150) is inside the expanded trolley even though a
        // derived below-mouth probe point would sit lower.
        assert(assoc::find_best_trolley_for_mouth(mouth, trolleys, 20.0f, 20.0f) == 0);
    }

    {
        assoc::MouthCandidate mouth{102, 0.60f, make_box(10.0f, 10.0f, 20.0f, 20.0f)};
        std::vector<assoc::TrolleyCandidate> trolleys = {
            {1, make_box(100.0f, 100.0f, 200.0f, 200.0f)},
        };
        assert(assoc::find_best_trolley_for_mouth(mouth, trolleys, 20.0f, 20.0f) == -1);
    }

    {
        std::vector<assoc::MouthCandidate> candidates = {
            {5001, 0.70f, make_box(0.0f, 0.0f, 1.0f, 1.0f)},
            {5002, 0.95f, make_box(0.0f, 0.0f, 1.0f, 1.0f)},
        };
        assert(assoc::select_mouth_candidate_index(candidates, 5001) == 0);
        assert(assoc::select_mouth_candidate_index(candidates, UINT64_MAX) == 1);
    }

    {
        assert(assoc::is_within_hold_window(100, 97, 3));
        assert(!assoc::is_within_hold_window(101, 97, 3));
    }

    {
        assoc::MouthCandidate mouth{103, 0.90f, make_box(150.0f, 175.0f, 170.0f, 195.0f)};
        std::vector<assoc::TrolleyCandidate> trolleys = {
            {10, make_box(100.0f, 100.0f, 240.0f, 240.0f)},
            {11, make_box(90.0f, 160.0f, 250.0f, 320.0f)},
        };
        // Both expanded bboxes contain the mouth center; tie-break should pick
        // the trolley with the larger center-y value.
        assert(assoc::find_best_trolley_for_mouth(mouth, trolleys, 10.0f, 10.0f) == 1);
    }

    {
        std::vector<assoc::TrolleyCandidate> previous_states = {
            {21, make_box(100.0f, 100.0f, 220.0f, 220.0f)},
            {22, make_box(300.0f, 100.0f, 420.0f, 220.0f)},
        };
        assoc::Box new_bbox = make_box(108.0f, 108.0f, 228.0f, 228.0f);
        assert(assoc::find_best_handoff_trolley(new_bbox, previous_states, 0.25f) == 0);
        assert(assoc::find_best_handoff_trolley(new_bbox, previous_states, 0.80f) == -1);
    }

    std::cout << "association_logic tests passed\n";
    return 0;
}
