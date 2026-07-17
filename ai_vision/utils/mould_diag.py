"""Flag-gated per-frame mould diagnostics CSV writer (jitter analysis).

Bounded-queue background writer: the probe thread enqueues small row tuples and
never touches the filesystem. Rows are flushed in batches; the queue drops rows
(with a counter) when full rather than blocking the probe.

Enabled via HICON_MOULD_DIAG_CSV=true; one file per day:
    output/csv/mould_diag_YYYYMMDD.csv
"""

from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_HEADER = "frame,ts,n_raw,n_tracked,n_filtered,n_canonical,track_ids,confs,bboxes\n"
_FLUSH_EVERY = 100


class MouldDiagWriter:
    """Append-only daily CSV writer fed from a bounded queue."""

    _SENTINEL = object()

    def __init__(self, csv_dir: Path, maxsize: int = 512):
        self._dir = Path(csv_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._dropped = 0
        self._stopped = False
        self._thread = threading.Thread(
            target=self._worker, name="hicon-mould-diag", daemon=True
        )
        self._thread.start()
        logger.info("MouldDiagWriter initialized (dir=%s, maxsize=%d)", self._dir, maxsize)

    @property
    def dropped_rows(self) -> int:
        return self._dropped

    def write_row(self, frame: int, ts: float, n_raw: int, n_tracked: int,
                  n_filtered: int, n_canonical: int, track_ids, confs,
                  bboxes=()) -> None:
        """Enqueue one row; never blocks the caller."""
        if self._stopped:
            return
        ids_txt = ' '.join(str(i) for i in track_ids)
        confs_txt = ' '.join(f"{c:.2f}" for c in confs)
        bboxes_txt = ' '.join(
            f"{int(b[0])}:{int(b[1])}:{int(b[2])}:{int(b[3])}" for b in bboxes
        )
        row = (f"{frame},{ts:.3f},{n_raw},{n_tracked},{n_filtered},{n_canonical},"
               f"{ids_txt},{confs_txt},{bboxes_txt}\n")
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self._dropped += 1

    def stop(self, timeout: float = 5.0) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)
        if self._dropped:
            logger.warning("MouldDiagWriter dropped %d rows (queue full)", self._dropped)

    def _current_path(self) -> Path:
        return self._dir / f"mould_diag_{datetime.now().strftime('%Y%m%d')}.csv"

    def _worker(self) -> None:
        handle = None
        handle_path = None
        pending = 0
        try:
            while True:
                try:
                    item = self._queue.get(timeout=2.0)
                except queue.Empty:
                    if handle is not None and pending:
                        handle.flush()
                        pending = 0
                    if self._stopped:
                        break
                    continue
                if item is self._SENTINEL:
                    break
                path = self._current_path()
                if handle is None or path != handle_path:
                    if handle is not None:
                        handle.close()
                    new_file = not path.exists()
                    handle = open(path, "a", buffering=1024 * 64)
                    handle_path = path
                    if new_file:
                        handle.write(_HEADER)
                handle.write(item)
                pending += 1
                if pending >= _FLUSH_EVERY:
                    handle.flush()
                    pending = 0
        except Exception:
            logger.exception("MouldDiagWriter worker failed; diagnostics disabled")
        finally:
            if handle is not None:
                try:
                    handle.flush()
                    handle.close()
                except OSError:
                    pass
