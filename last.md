# Plan: Attach short pours to adjacent valid pours instead of discarding

## Context

When a pour ends with duration < `pour_min_dur` (2s), it is discarded completely:
- `_active_segment` is cleared (never enters `completed_segments`)
- `_build_clusters()` never sees it
- Mould count is unaffected — real data is silently dropped

The correct behaviour: if a short pour starts within a short time gap after a previous valid pour ended (i.e. it's a continuation — brief brightness dip mid-fill), **extend the previous segment's end time** to cover it rather than throwing it away.

**Concrete example from HEAT_0103:**
```
13:14:43  pour START  → 13:14:46  pour END  2.3s  M10  (just above threshold)
13:14:50  pour START  → 13:14:56  pour END  5.4s  M11  (valid)
```
The 4s gap between M10 end and M11 start is the ladle repositioning. If M10 was 1.8s it would be discarded, even though it's a real mould.

---

## Fix — `ai_vision/processors/pouring_processor.py`

### Step 1 — Track last completed segment end time AND probe position

In `__init__` after `self.last_pour_duration` (~line 173), add:
```python
self.last_completed_segment_end_time: Optional[float] = None
self.last_completed_segment_rep_norm: Optional[tuple] = None  # normalized probe point of last valid pour
```

In `_reset_all_state()` add:
```python
self.last_completed_segment_end_time = None
self.last_completed_segment_rep_norm = None
```

### Step 2 — Update both after each valid pour

In `_end_pour()`, after line 1596 (`self.last_pour_duration = duration`), add:
```python
self.last_completed_segment_end_time = effective_end_ts
# Store representative probe point for proximity check on next short pour
if self.completed_segments:
    self.last_completed_segment_rep_norm = self._segment_representative_point(self.completed_segments[-1])
```

### Step 3 — Validate by probe position before merging

Replace the discard path (lines 1571–1594) with:

```python
if duration < self.pour_min_dur:
    merged = False
    merge_gap = config.POUR_MERGE_GAP_S
    if (self.last_completed_segment_end_time is not None and
            self.completed_segments and
            (self.pour_start_time - self.last_completed_segment_end_time) <= merge_gap):
        # Validate probe position — only merge if short pour is at same mould position
        short_rep = self._segment_representative_point(self._active_segment) if self._active_segment else None
        prev_rep = self.last_completed_segment_rep_norm
        if short_rep is not None and prev_rep is not None:
            dx = abs(short_rep[0] - prev_rep[0])
            dy = abs(short_rep[1] - prev_rep[1])
            position_close = (dx < config.R_CLUSTER and dy < config.R_CLUSTER)
        else:
            position_close = True  # no position data — give benefit of doubt

        if position_close:
            prev = self.completed_segments[-1]
            prev['end_time'] = effective_end_ts
            prev['end_datetime'] = effective_end_dt
            if self._active_segment and self._active_segment.get('samples'):
                prev.setdefault('samples', []).extend(self._active_segment['samples'])
            self.last_completed_segment_end_time = effective_end_ts
            merged = True
            logger.info(
                f"[pour] SHORT MERGED into prev segment - duration={duration:.1f}s, "
                f"gap={self.pour_start_time - self.last_completed_segment_end_time + duration:.1f}s, "
                f"probe_delta=({dx:.3f},{dy:.3f})"
            )

    if not merged:
        logger.info(f"[pour] DISCARDED - duration={duration:.1f}s < {self.pour_min_dur}s minimum")
        if self.pour_sync_id:
            try:
                self.db_manager.delete_pouring_event(self.pour_sync_id)
            except Exception as e:
                logger.warning(f"[pour] Failed to clean up discarded pour row: {e}")
            self.pour_sync_id = None
            self.pour_slno = None

    self.pour_start_time = None
    self.pour_start_datetime = None
    self._active_segment = None
    self.active_mould_id = None
    self.active_mould_start_time = None
    self.active_mould_start_datetime = None
    self.active_mould_start_norm = None
    self._materialize_mould_records(include_active=False)
    self.displacement_hold_frames = None
    self.split_hold_quadrant = None
    self.split_rearm_required = False
    self.split_rearm_below_since = None
    self.split_rearm_axis = None
    self._last_probe_is_pouring = None
    return
```

### Step 4 — Add `POUR_MERGE_GAP_S` to `config.py`

After `POUR_MIN_DURATION`:
```python
# Max gap (seconds) between a valid pour ending and a short pour starting
# for the short pour to be merged rather than discarded. Only merges if
# the probe point is within R_CLUSTER of the previous pour's position.
POUR_MERGE_GAP_S = float(os.getenv('HICON_POUR_MERGE_GAP_S', '8.0'))
```

---

## Critical Files

| File | Change |
|---|---|
| `ai_vision/processors/pouring_processor.py` `__init__` | Add `last_completed_segment_end_time` |
| `ai_vision/processors/pouring_processor.py` `_reset_all_state()` | Reset `last_completed_segment_end_time` |
| `ai_vision/processors/pouring_processor.py` `_end_pour()` line ~1596 | Update `last_completed_segment_end_time` on valid pour |
| `ai_vision/processors/pouring_processor.py` `_end_pour()` lines 1571–1594 | Merge short pour into prev segment if within gap |
| `ai_vision/config.py` | Add `POUR_MERGE_GAP_S = 8.0` |

---

## Verification

After restart, on next heat with short pours:
- Log: `[pour] SHORT MERGED into prev segment - duration=1.8s, gap=3.2s` instead of DISCARDED
- Mould count unchanged (no new splits) — extended segment stays same cluster
- Total pour duration increases slightly (correct — those seconds were real pouring)
