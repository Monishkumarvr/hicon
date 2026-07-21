# Plan: Mould Placement Detection + Pour Assignment

## Operational Reality (Final)

```
Session starts → Trolley detected
  → Worker places mould 1
  → Pour 1 starts (ladle mouth in trolley)
  → Pour 1 ends (mould 1 still on trolley, GLOWING for 10+ seconds)
  → Worker immediately places mould 2 (WHILE mould 1 still glowing)
  → Pour 2 starts
  → ...
```

**Key constraints confirmed:**
- Moulds never removed mid-session (accumulate on trolley)
- No cooling window between placements — glow persists 10+ seconds after pour end
- Moulds placed in rows; row size depends on mould size
- Fresh sand moulds are **dark**; just-poured metal is **bright**

---

## Core Algorithm: Pre-Pour Snapshot Comparison with Signed Diff

There is exactly one clean, noise-free moment per mould: **when the pour is about to start** (ladle mouth detected in trolley, pour not yet active). At this moment:
- The fresh (oven-baked) mould is on the trolley
- No active pour glow from the CURRENT pour (it hasn't started yet)
- Previous poured mould may still be glowing — but crucially, that glow was **already present at the previous pour-start snapshot `S_{N-1}`**

**The consecutive snapshot diff (`S_N − S_{N-1}`) inherently cancels the previous mould's glow.** The key is using a SIGNED diff (not absolute diff) to distinguish new objects from fading ones:
- **New mould (appeared): `S_N > S_{N-1}` → positive diff**
- **Fading glow (cooling): `S_N < S_{N-1}` → negative diff → filtered to zero**

This works regardless of mould temperature — we detect new objects appearing, not rely on them being dark.

### Snapshot Sequence

| Event | Action |
|-------|--------|
| Session starts (first trolley detected) | Capture `S_base` — initial trolley state |
| Pour N starts (mouth enters trolley, before pour fires) | Capture `S_N` |

### Detection at Pour N Start

```python
def detect_new_mould(self, S_N, S_prev):
    """
    S_N:     current pre-pour snapshot (128×64 grayscale, equalized)
    S_prev:  previous pre-pour snapshot (or session-start baseline)
    
    Returns: list of new object blobs (freshly placed moulds)
    """
    # 1. SIGNED diff: only positive changes (new objects appeared)
    #    cv2.subtract clips to 0 where S_N < S_prev (fading glow → ignored)
    pos_diff = cv2.subtract(S_N, S_prev)  # new or brighter objects
    
    # 2. Threshold the positive diff
    _, diff_mask = cv2.threshold(pos_diff, PLACEMENT_DIFF_THRESH, 255, cv2.THRESH_BINARY)
    
    # 3. Morphological cleanup
    cleaned = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, KERNEL_3x3)
    
    # 4. Connected components → blobs
    n, _, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    blobs = _filter_blobs(n, stats, centroids)
    
    # 5. Shape filter: reject irregular blobs (glow tends to be irregular,
    #    moulds tend to be compact rectangles)
    return [b for b in blobs if b['solidity'] >= PLACEMENT_MIN_SOLIDITY]  # e.g., 0.5
```

**Why signed diff works even with hot moulds:**
- Previous mould glow (from pour N-1): already in `S_{N-1}` → diff is ~0
- Fading glow as it cools: `S_N < S_{N-1}` → negative diff → clipped to 0
- New mould (from oven or ambient): newly appeared since `S_{N-1}` → positive diff ✓
- The mould does NOT need to be dark — it just needs to be NEW (not in `S_{N-1}`)

---

## Detection Timing

Detection happens at **pour start** (not during placement). This is intentionally delayed vs. physical placement — we detect the mould at the moment its pour begins, not the moment it's placed. This is acceptable because:
- We count moulds by pour cycle, and pour assignment to mould blob happens at the same moment
- No need to run continuous monitoring between pours
- Much simpler and more reliable than monitoring during the glowing period

---

## Session Snapshot State

```
Session start:  S_prev = S_base (first trolley ROI captured)
Pour 1 starts:  S_1 = snapshot; detect_new_mould(S_1, S_base) → mould 1 blob
                S_prev = S_1
Pour 2 starts:  S_2 = snapshot; detect_new_mould(S_2, S_1) → mould 2 blob
                S_prev = S_2
Pour N starts:  S_N = snapshot; detect_new_mould(S_N, S_{N-1}) → mould N blob
                S_prev = S_N
```

Only 2 snapshot frames held in memory at any time.

---

## Pour Assignment to Mould Blob (Phase 2)

At `_start_pour()`, detection fires and yields a blob centroid in trolley-relative [0,1]. The ladle mouth position is also in trolley-relative [0,1] via `_normalize_mouth_position()`. Same coordinate space — no transformation needed.

```python
# In _start_pour():
# 1. Capture pre-pour snapshot
S_N = self.placement_detector.capture_snapshot(frame_rgba, trolley_bbox)

# 2. Detect new mould blob
blob = self.placement_detector.detect_new_mould(S_N)

# 3. Assign pour to blob (or mouth position if detection fails)
norm_x, norm_y, _, _ = self._normalize_mouth_position(best_mouth, locked_trolley)
blob_id = self.placement_detector.lock_blob(blob, norm_x, norm_y, timestamp)
self._active_pour_blob_id = blob_id

# 4. Update S_prev = S_N
self.placement_detector.advance_snapshot(S_N)
```

```python
# In _end_pour():
if self._active_pour_blob_id is not None:
    self.placement_detector.close_pour(self._active_pour_blob_id, duration_s)

predictive_count = self.placement_detector.get_poured_blob_count()
effective_count = _select_count(self.mould_count, predictive_count, config.MOULD_TRACKING_MODE)
```

---

## Handling Row Arrangement (Multiple Moulds Per Row)

When moulds are placed side by side in a row:
- `S_N` vs `S_{N-1}`: only the NEWEST mould is different (previous moulds same position)
- If two moulds are placed very quickly between pours: diff shows two blobs (both new)
- `_filter_blobs` returns multiple blobs; take the one nearest to mouth position for assignment

---

## What Changes

### New file: `ai_vision/processors/mould_placement_detector.py`

```python
class MouldPlacementDetector:
    # Called at session start (first trolley frame)
    def on_session_start(self, gray_roi_128x64, timestamp) -> None
    
    # Called at each pour start — core detection
    def on_pour_start(self, frame_rgba, trolley_bbox, timestamp) -> Optional[Dict]
    # Returns: {"blob_id", "cx_norm", "cy_norm", "area_norm", "confidence"}
    
    # Called at pour end
    def on_pour_end(self, blob_id, duration_s) -> None
    
    # Pour count from placement perspective
    def get_poured_blob_count(self) -> int
    
    # Session cleanup
    def on_session_end(self) -> None
    def on_cycle_reset(self) -> None
    
    def get_result_dict(self) -> Dict
```

Internal state per session:
```python
self._S_prev: Optional[np.ndarray]      # 128×64 uint8 grayscale
self._locked_blobs: Dict[int, BlobState]
self._next_blob_id: int
```

`BlobState`:
```python
{blob_id, cx_norm, cy_norm, area_norm, pour_count, total_pour_s, detected_at}
```

### Modified: `ai_vision/processors/pouring_processor.py`

- Add `placement_detector=None` kwarg
- `_start_session()`: call `on_session_start(first_trolley_frame, ts)` on first frame with trolley
- `_start_pour()`: call `on_pour_start(frame, trolley_bbox, ts)` → lock blob → store `_active_pour_blob_id`
- `_end_pour()`: call `on_pour_end(blob_id, duration_s)` → compute `effective_mould_count`
- `_reset_all_state()`: call `on_cycle_reset()`
- Augmented `mould_wise` dict with `predictive_mould_count`, `reactive_mould_count`

### Modified: `ai_vision/config.py`

```python
MOULD_TRACKING_MODE = os.getenv('HICON_MOULD_TRACKING_MODE', 'reactive')
PLACEMENT_DIFF_THRESH = int(os.getenv('HICON_PLACEMENT_DIFF_THRESH', '30'))
PLACEMENT_MIN_BLOB_AREA_PX = int(os.getenv('HICON_PLACEMENT_MIN_BLOB_AREA_PX', '50'))
PLACEMENT_MAX_BLOB_AREA_PX = int(os.getenv('HICON_PLACEMENT_MAX_BLOB_AREA_PX', '2000'))
PLACEMENT_R_ASSIGN = float(os.getenv('HICON_PLACEMENT_R_ASSIGN', '0.15'))
PLACEMENT_HYBRID_CONF_THRESHOLD = float(os.getenv('HICON_PLACEMENT_HYBRID_CONF_THRESHOLD', '0.70'))
PLACEMENT_SAVE_DEBUG_IMAGES = os.getenv('HICON_PLACEMENT_SAVE_DEBUG_IMAGES', 'false').lower() == 'true'
PLACEMENT_CANONICAL_W = int(os.getenv('HICON_PLACEMENT_CANONICAL_W', '128'))
PLACEMENT_CANONICAL_H = int(os.getenv('HICON_PLACEMENT_CANONICAL_H', '64'))
PLACEMENT_MIN_SOLIDITY = float(os.getenv('HICON_PLACEMENT_MIN_SOLIDITY', '0.50'))
```

### Modified: `ai_vision/hicon_pipeline.py`

Construct `MouldPlacementDetector(config)` and inject into `PouringProcessor`.

The `frame` (RGBA) is passed to `on_pour_start()` — it's already available at `_start_pour()` if we thread it through from `process_frame(self, frame, ...)`. Currently `pouring_processor.process_frame()` already receives the frame argument.

---

## CPU Budget

| Operation | Cost | Frequency |
|-----------|------|-----------|
| Crop + resize trolley → 128×64 | ~0.2 ms | Once per pour start |
| equalizeHist + GaussianBlur | ~0.13 ms | Once per pour start |
| absdiff (diff + brightness mask) | ~0.06 ms | Once per pour start |
| threshold + morphology | ~0.15 ms | Once per pour start |
| connectedComponentsWithStats | ~0.15 ms | Once per pour start |
| Total per pour start | **~0.69 ms** | Very infrequent |
| Per-frame overhead | **0 ms** | Nothing runs per-frame |

Zero per-frame cost — detection only fires at pour-start events.

---

## Feature Flag Rollback

```ini
# .env — instant rollback
HICON_MOULD_TRACKING_MODE=reactive
```

---

## Verification

**Step 1 — Passive mode (`reactive` + `PLACEMENT_SAVE_DEBUG_IMAGES=true`):**
- Log `[placement] pour_N: detected blob at (cx, cy), conf=X` at each pour start
- Save 128×64 diff image + masked diff image per pour to `output/screenshots/placement_pour{N}_*.jpg`
- Review saved images vs. operator mould count
- Tune `PLACEMENT_GLOW_THRESH` (start at 160, adjust if glow bleeds through) and `PLACEMENT_DIFF_THRESH`

**Step 2 — Hybrid mode:**
- Watch `[placement] CONFIRMED/DIVERGE` per session
- Accept: CONFIRMED rate ≥ 80% over 10 cycles

**Step 3 — Predictive mode:**
- Monitor `mould_count` in DB vs. operator ground truth
- Roll back via `.env` if needed

**Tuning levers:**
- `HICON_PLACEMENT_DIFF_THRESH` — minimum positive pixel change to register a new object (lower = more sensitive, more noise from gradual glow changes)
- `HICON_PLACEMENT_MIN_BLOB_AREA_PX` — minimum new mould area in canonical pixels
- `HICON_PLACEMENT_MIN_SOLIDITY` — compactness filter (moulds are rectangular; glow is irregular)

---

## Open Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| First pour: no previous snapshot yet | Mould 1 not detected by diff | Use session-start `S_base` as reference; mould 1 detected vs. initial trolley state |
| Worker's hand/tool in frame at pour start | False blob | `PLACEMENT_R_ASSIGN=0.15` assigns pour to nearest blob within range; hand is dark but far from mouth position |
| Mould placed in same row position as previous one (very close) | Small diff, may miss blob | `S_{N-1}` reference includes the old mould, so new mould in same spot shows less diff — tune `DIFF_THRESH` down |
| Trolley moves between pours | Trolley bbox shifts, crop misaligns | Crop uses live YOLO trolley bbox (always current) — automatically handles trolley motion |
| New mould gradually heats up before pour (brightness creep) | May not appear as large positive diff immediately | `PLACEMENT_DIFF_THRESH` tuning; detect at pour-start when mould has been placed ~5–15s so any heat delta has already peaked |
| Pipeline restart mid-session | No `S_base` yet | First pour after restart: no detection (falls back to reactive for that cycle) |

---

## Critical Files

| File | Change |
|------|--------|
| `ai_vision/processors/mould_placement_detector.py` | **New** — snapshot comparison + brightness gating + blob tracking |
| `ai_vision/processors/pouring_processor.py` | Add kwarg + 4 lifecycle hooks + `on_pour_start/end` calls + tracking-mode output |
| `ai_vision/config.py` | 10 new constants |
| `ai_vision/hicon_pipeline.py` | Construct + inject `MouldPlacementDetector` |
