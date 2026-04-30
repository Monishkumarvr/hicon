# Safer Mould Count Fix — Baseline Cluster Rescue + Lower Split Threshold

## Summary
- Keep the current baseline clustering as the default path.
- Replace the global adaptive `r_cluster` idea with a rescue pass that only runs on heavily compressed, high-count heats like `HEAT_0124`.
- Lower the default displacement threshold from `0.25` to `0.15` using the existing env var `HICON_MOULD_DISPLACEMENT`.

## Implementation Changes
- In [`pouring_processor.py`](/home/hicon/hicon/ai_vision/processors/pouring_processor.py), refactor `_build_clusters(...)` into two stages:
  - baseline clustering with the current `self.r_cluster` and `self.r_merge`
  - optional rescue refinement only when the baseline result looks pathologically over-collapsed
- Use these heat-level rescue gates before any refinement:
  - `valid_segment_count >= 18`
  - `baseline_cluster_count / valid_segment_count <= 0.65`
  - `valid_segment_count - baseline_cluster_count >= 8`
- If the heat passes those gates, refine only suspicious baseline clusters, not the whole heat:
  - inspect the cluster’s member representative `x` values
  - ignore internal `x` gaps `<= 0.008` as same-position revisits
  - only refine clusters with at least 2 member segments and at least 1 significant internal `x` gap
  - compute `local_r = min(self.r_cluster, max(0.005, typical_gap * 0.40))`
  - compute `local_r_merge = min(self.r_merge, max(0.003, local_r * 0.35))`
  - re-cluster only that cluster’s segments with the local radii
  - leave non-suspicious baseline clusters unchanged
- Keep the existing Euclidean distance on representative `(x, y)` points for assignment and merge decisions. Do not globally shrink `self.r_cluster`.
- After refinement, renumber cluster IDs sequentially and keep the existing `min_cluster_pour_s` filter.
- Add one `INFO` log when rescue refinement runs, including:
  - valid segment count
  - baseline cluster count
  - refined cluster count
  - trigger metrics and chosen local radii
- In [`config.py`](/home/hicon/hicon/ai_vision/config.py#L105), change the default to:
  - `MOULD_DISPLACEMENT_THRESHOLD = float(os.getenv('HICON_MOULD_DISPLACEMENT', '0.15'))`
- Keep the env var name unchanged. Do not introduce `HICON_MOULD_DISPLACEMENT_THRESHOLD`.

## Test Plan
- Add unit tests in [`test_pouring_transitions_and_screenshots.py`](/home/hicon/hicon/ai_vision/tests/test_pouring_transitions_and_screenshots.py) for:
  - rescue gate triggers on a `21 segments -> 10 clusters` pattern
  - rescue gate does not trigger on `15 -> 12`, `24 -> 20`, or similar recent revisit-heavy patterns
  - a suspicious baseline cluster with internal `x` gaps `> 0.008` splits into multiple subclusters
  - same-position revisit jitter stays merged
  - a synthetic `HEAT_0124`-like heat improves versus baseline
  - a synthetic `HEAT_0144`-like revisit heat stays unchanged because the heat-level gate blocks rescue
  - lowering displacement to `0.15` detects a slow-motion split that `0.25` misses
  - relock gating does not become noisy under the lower threshold
- Keep existing suites green:
  - `PYTHONPATH=ai_vision pytest -q ai_vision/tests/test_pouring_transitions_and_screenshots.py`
  - `PYTHONPATH=ai_vision pytest -q ai_vision/tests/test_heat_cycle_manager.py`

## Live Validation
- Remove the current verification rule that compares `CYCLE COMPLETE` to `[session] END`; that is not a valid acceptance check for revisit-heavy heats.
- Accept rescue behavior only against labeled or manually verified mould counts.
- Watch for the new rescue `INFO` log. It should appear on `HEAT_0124`-like compressed heats, and should stay absent on normal revisit-heavy heats.
- Roll back Change 2 by setting `HICON_MOULD_DISPLACEMENT=0.25` in `.env` if relock or split behavior regresses.

## Assumptions
- The target remains one final record per physical mould, not per cavity or per pour segment.
- Multi-cavity pours are rare enough that the safer bias is to preserve baseline revisit merging unless a heat looks strongly over-collapsed.
- Historical raw motion tracks are not retained, so offline validation must rely on labeled outcomes plus synthetic regression fixtures rather than exact post-hoc replay.

New data from user:
Ai data heat 0148  is 19 mould--Actual 44
Ai data heat 0146  is 20 mould--Actual 48
validate this plan


Source image - 720x1280
Deslagging 2:
[
    np.array([[855, 447], [991, 439], [1069, 617], [910, 635]])
]

Spectro zone 2:
[
    np.array([[1084, 551], [1111, 638], [1146, 627], [1198, 531]])
]

Spectro zone 1:
[
    np.array([[965, 304], [994, 383], [1015, 381], [1021, 341], [1069, 334], [1049, 289]])
]

Tapping 2:
[
    np.array([[1046, 439], [1085, 426], [1052, 371], [1026, 385]])
]