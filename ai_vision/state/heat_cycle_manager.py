"""
Heat Cycle Manager - Track pouring-session-driven heat cycles.

Manages heat cycles where:
- A cycle is created by pouring-session presence or tapping
- Pouring-backed cycles stay alive only while mouth-in-trolley presence refreshes
- Tapping-only cycles are allowed and finalized after a confirmation window
- Aggregates pouring + melting-side timings for API POST
"""
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

_CHECKPOINT_INTERVAL_S = 30.0

logger = logging.getLogger(__name__)


@dataclass
class MouldPouringRecord:
    """Single mould pouring within a heat cycle."""
    mould_id: str
    mould_track_id: int
    start_time: float  # Wall clock timestamp
    start_datetime: datetime
    end_time: Optional[float] = None
    end_datetime: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    sync_id: str = ""
    slno: int = 0


@dataclass
class HeatCycle:
    """Represents one complete heat cycle."""
    heat_no: str  # Sequential: HEAT_0001, HEAT_0002, etc.
    ladle_number: str  # Legacy DB field; kept empty in the new logic
    ladle_track_ids: List[int]  # Track IDs seen during this cycle (legacy/debug)
    cycle_start_time: float
    cycle_start_datetime: datetime
    furnace_label: Optional[str] = None

    # Pouring aggregation
    mould_pourings: List[MouldPouringRecord] = field(default_factory=list)

    # Cycle state
    cycle_active: bool = True  # False when finalized
    has_pouring_session: bool = False
    last_pouring_presence_time: Optional[float] = None
    last_pouring_presence_datetime: Optional[datetime] = None

    # Tapping aggregation (first start = cycle tapping start, last end = cycle tapping end)
    tapping_start_time: Optional[float] = None
    tapping_start_datetime: Optional[datetime] = None
    tapping_end_time: Optional[float] = None
    tapping_end_datetime: Optional[datetime] = None
    last_tapping_end_time: Optional[float] = None
    last_tapping_end_datetime: Optional[datetime] = None
    tapping_events: List[Dict] = field(default_factory=list)

    # Deslagging aggregation
    deslagging_events: List[Dict] = field(default_factory=list)

    # Spectro aggregation
    spectro_events: List[Dict] = field(default_factory=list)

    # Pyrometer aggregation (rod events)
    pyrometer_events: List[Dict] = field(default_factory=list)

    # Trolley locking for pouring cycle
    locked_trolley_id: Optional[int] = None

    # Finalized cycle data (set when cycle completes)
    cycle_end_time: Optional[float] = None
    cycle_end_datetime: Optional[datetime] = None
    pouring_start_time: Optional[datetime] = None  # First mould start
    pouring_end_time: Optional[datetime] = None  # Last mould end
    total_pouring_time: Optional[int] = None  # Total seconds (int)
    mould_wise_pouring_time: Optional[List[Dict]] = None  # [{"mould_id": "M1", "start": "...", "end": "...", "duration": "..."}]


class HeatCycleManager:
    """
    Manage heat cycles and aggregate pouring data.
    
    Business logic:
    - New cycle starts from pouring-session presence or tapping
    - Pouring-backed cycles stay alive from mouth-in-trolley presence only
    - Tapping-only cycles are valid and finalized after a confirmation window
    - Finalized cycles are posted to API
    """
    
    def __init__(self, db_manager, ladle_absence_timeout: float = 300.0,
                 tapping_only_timeout: float = 900.0,
                 base_location: str = ""):
        """
        Initialize heat cycle manager.

        Args:
            db_manager: Database manager instance for querying existing counters
            ladle_absence_timeout: Seconds of pouring-session inactivity before finalizing cycle
            tapping_only_timeout: Seconds after tapping ends to wait for a pouring session before
                                  finalizing (longer than ladle_absence_timeout to allow ladle travel)
        """
        self.db_manager = db_manager
        self.ladle_absence_timeout = ladle_absence_timeout
        self.tapping_only_timeout = tapping_only_timeout
        self.base_location = base_location
        
        # Single active cycle (time-based, not per-ladle ID)
        self.active_cycle: Optional[HeatCycle] = None
        
        # Sequential counters - initialize from database
        self.heat_counter = self._get_last_heat_counter()

        # Finalized cycles ready for sync (keyed by heat_no)
        self.finalized_cycles: Dict[str, HeatCycle] = {}
        self.previous_cycle_end_time, self.previous_cycle_end_datetime = self._get_last_cycle_end()

        # Checkpoint throttle
        self._last_checkpoint_time: float = float('-inf')

        # Restore any active cycle that was in-progress before the last restart
        self._restore_from_checkpoint()

        logger.info(f"✓ HeatCycleManager initialized (ladle timeout: {ladle_absence_timeout}s, tapping-only timeout: {tapping_only_timeout}s)")
        logger.info(f"  Heat counter starting at: {self.heat_counter}")
    
    def _get_last_heat_counter(self) -> int:
        """Query database for the last used heat counter."""
        try:
            conn = self.db_manager._get_connection()
            c = conn.cursor()
            c.execute("SELECT heat_no FROM heat_cycles ORDER BY created_at DESC LIMIT 1")
            row = c.fetchone()
            conn.close()
            
            if row and row[0]:
                # Parse HEAT_XXXX format
                heat_no = row[0]
                if heat_no.startswith('HEAT_'):
                    try:
                        return int(heat_no.split('_')[1])
                    except (IndexError, ValueError):
                        logger.warning(f"Could not parse heat_no: {heat_no}")
            return 0
        except Exception as e:
            logger.warning(f"Error querying last heat counter: {e}")
            return 0
    
    def _get_last_cycle_end(self) -> tuple[Optional[float], Optional[datetime]]:
        """Query database for the last finalized heat cycle end time."""
        try:
            conn = self.db_manager._get_connection()
            c = conn.cursor()
            c.execute(
                "SELECT cycle_end_time FROM heat_cycles "
                "WHERE cycle_end_time IS NOT NULL AND cycle_end_time != '' "
                "ORDER BY created_at DESC LIMIT 1"
            )
            row = c.fetchone()
            conn.close()

            if not row or not row[0]:
                return None, None

            dt = datetime.fromisoformat(row[0])
            return dt.timestamp(), dt
        except Exception as e:
            logger.warning(f"Error querying last heat cycle end: {e}")
            return None, None
    
    def _get_next_heat_no(self) -> str:
        """Generate next sequential heat number."""
        self.heat_counter += 1
        return f"HEAT_{self.heat_counter:04d}"

    @staticmethod
    def infer_furnace_label(zone_name: str) -> str:
        """Map a zone name like tap-1 or furnace-2 to Furnace1/Furnace2."""
        if not zone_name:
            return ""
        digits = ""
        for ch in reversed(str(zone_name).strip()):
            if ch.isdigit():
                digits = ch + digits
            elif digits:
                break
        return f"Furnace{digits}" if digits else ""

    @staticmethod
    def location_with_furnace(base_location: str, furnace_label: str) -> str:
        """Append furnace label to the base location once."""
        location = (base_location or "").strip()
        if not furnace_label:
            return location
        if furnace_label.lower() in location.lower():
            return location
        return f"{location} {furnace_label}".strip()

    def _ensure_cycle_furnace(self, cycle: 'HeatCycle', zone_name: str, source: str) -> None:
        furnace_label = self.infer_furnace_label(zone_name)
        if not furnace_label:
            return
        if not cycle.furnace_label:
            cycle.furnace_label = furnace_label
            logger.info("  %s: Furnace associated as %s via %s", cycle.heat_no, furnace_label, source)
            self._sync_cycle_pouring_locations(cycle)
            return
        if cycle.furnace_label != furnace_label:
            logger.warning(
                "  %s: conflicting furnace association (%s via %s, already %s) - keeping existing",
                cycle.heat_no,
                furnace_label,
                source,
                cycle.furnace_label,
            )

    def _sync_cycle_pouring_locations(self, cycle: 'HeatCycle') -> None:
        """Propagate furnace-tagged location to pouring rows for the same heat."""
        if not self.base_location or not cycle.heat_no or not cycle.furnace_label:
            return
        update_fn = getattr(self.db_manager, "update_pouring_location_by_heat_no", None)
        if update_fn is None:
            return
        try:
            update_fn(
                heat_no=cycle.heat_no,
                location=self.location_with_furnace(self.base_location, cycle.furnace_label),
            )
        except Exception as exc:
            logger.warning("Failed to update pouring locations for %s: %s", cycle.heat_no, exc)

    def _get_cycle_start_seed(
        self,
        event_time: float,
        event_datetime: datetime,
    ) -> tuple[float, datetime]:
        """Backfill next cycle start from previous cycle end when available."""
        if self.previous_cycle_end_time is not None and self.previous_cycle_end_datetime is not None:
            return self.previous_cycle_end_time, self.previous_cycle_end_datetime
        return event_time, event_datetime

    MAX_BACKFILL_HOURS = 8  # Safety cap — don't pull events older than this

    def _backfill_pre_tapping_events(
        self, cycle: 'HeatCycle', tapping_start_dt: datetime
    ) -> None:
        """
        Retroactively attach deslagging/spectro/pyrometer events that fired
        before tapping opened the heat cycle.

        Lookback window:
          start = max(previous_cycle_end, tapping_start - MAX_BACKFILL_HOURS)
          end   = tapping_start (exclusive)
        """
        from datetime import timedelta
        window_end = tapping_start_dt
        cap = tapping_start_dt - timedelta(hours=self.MAX_BACKFILL_HOURS)

        if self.previous_cycle_end_datetime is not None:
            window_start = max(self.previous_cycle_end_datetime, cap)
        else:
            window_start = cap

        window_start_iso = window_start.isoformat()
        window_end_iso = window_end.isoformat()

        try:
            conn = self.db_manager._get_connection()
            c = conn.cursor()
            c.execute(
                """SELECT event_type, start_time, end_time, duration_sec, zone_name
                   FROM melting_events
                   WHERE event_type IN ('deslagging', 'spectro', 'pyrometer')
                     AND end_time > ? AND end_time <= ?
                   ORDER BY start_time ASC""",
                (window_start_iso, window_end_iso),
            )
            rows = c.fetchall()
            conn.close()
        except Exception as e:
            logger.warning(f"Backfill query failed: {e}")
            return

        counts = {"deslagging": 0, "spectro": 0, "pyrometer": 0}
        for row in rows:
            ev_type, start_iso, end_iso, dur, zone_name = row
            event = {"start": start_iso, "end": end_iso, "duration_sec": dur}
            if zone_name:
                event["zone_name"] = zone_name
            if ev_type == "deslagging":
                cycle.deslagging_events.append(event)
                counts["deslagging"] += 1
            elif ev_type == "spectro":
                cycle.spectro_events.append(event)
                counts["spectro"] += 1
            elif ev_type == "pyrometer":
                cycle.pyrometer_events.append(event)
                counts["pyrometer"] += 1
            if zone_name:
                self._ensure_cycle_furnace(cycle, zone_name, f"backfill:{ev_type}")

        if any(counts.values()):
            logger.info(
                f"  {cycle.heat_no}: Backfilled pre-tapping events — "
                f"deslagging={counts['deslagging']}, spectro={counts['spectro']}, "
                f"pyrometer={counts['pyrometer']} "
                f"(window {window_start_iso[:19]} \u2192 {window_end_iso[:19]})"
            )
            # Update cycle_start to the earliest backfilled event
            earliest_iso = rows[0][1]
            try:
                earliest_dt = datetime.fromisoformat(earliest_iso)
                if earliest_dt < cycle.cycle_start_datetime:
                    cycle.cycle_start_time = earliest_dt.timestamp()
                    cycle.cycle_start_datetime = earliest_dt
                    logger.info(f"  {cycle.heat_no}: cycle_start updated to {earliest_iso[:19]}")
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Checkpoint helpers                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dt_iso(dt: Optional[datetime]) -> Optional[str]:
        return dt.isoformat() if dt else None

    def _cycle_to_dict(self, cycle: HeatCycle) -> dict:
        """Serialize active HeatCycle to a JSON-safe dict."""
        return {
            'heat_no': cycle.heat_no,
            'ladle_number': cycle.ladle_number,
            'ladle_track_ids': cycle.ladle_track_ids,
            'cycle_start_time': cycle.cycle_start_time,
            'cycle_start_datetime': self._dt_iso(cycle.cycle_start_datetime),
            'furnace_label': cycle.furnace_label,
            'has_pouring_session': cycle.has_pouring_session,
            'last_pouring_presence_time': cycle.last_pouring_presence_time,
            'last_pouring_presence_datetime': self._dt_iso(cycle.last_pouring_presence_datetime),
            'tapping_start_time': cycle.tapping_start_time,
            'tapping_start_datetime': self._dt_iso(cycle.tapping_start_datetime),
            'tapping_end_time': cycle.tapping_end_time,
            'tapping_end_datetime': self._dt_iso(cycle.tapping_end_datetime),
            'last_tapping_end_time': cycle.last_tapping_end_time,
            'last_tapping_end_datetime': self._dt_iso(cycle.last_tapping_end_datetime),
            'tapping_events': cycle.tapping_events,
            'deslagging_events': cycle.deslagging_events,
            'spectro_events': cycle.spectro_events,
            'pyrometer_events': cycle.pyrometer_events,
            'locked_trolley_id': cycle.locked_trolley_id,
            'mould_pourings': [
                {
                    'mould_id': p.mould_id,
                    'mould_track_id': p.mould_track_id,
                    'start_time': p.start_time,
                    'start_datetime': self._dt_iso(p.start_datetime),
                    'end_time': p.end_time,
                    'end_datetime': self._dt_iso(p.end_datetime),
                    'duration_seconds': p.duration_seconds,
                    'sync_id': p.sync_id,
                    'slno': p.slno,
                }
                for p in cycle.mould_pourings
            ],
        }

    @staticmethod
    def _cycle_from_dict(d: dict) -> HeatCycle:
        """Reconstruct a HeatCycle from a serialized dict."""
        def _dt(s):
            return datetime.fromisoformat(s) if s else None

        mould_pourings = [
            MouldPouringRecord(
                mould_id=p['mould_id'],
                mould_track_id=p['mould_track_id'],
                start_time=p['start_time'],
                start_datetime=_dt(p.get('start_datetime')),
                end_time=p.get('end_time'),
                end_datetime=_dt(p.get('end_datetime')),
                duration_seconds=p.get('duration_seconds'),
                sync_id=p.get('sync_id', ''),
                slno=p.get('slno', 0),
            )
            for p in d.get('mould_pourings', [])
        ]

        return HeatCycle(
            heat_no=d['heat_no'],
            ladle_number=d.get('ladle_number', ''),
            ladle_track_ids=d.get('ladle_track_ids', []),
            cycle_start_time=d['cycle_start_time'],
            cycle_start_datetime=_dt(d.get('cycle_start_datetime')),
            furnace_label=d.get('furnace_label'),
            mould_pourings=mould_pourings,
            cycle_active=True,
            has_pouring_session=d.get('has_pouring_session', False),
            last_pouring_presence_time=d.get('last_pouring_presence_time'),
            last_pouring_presence_datetime=_dt(d.get('last_pouring_presence_datetime')),
            tapping_start_time=d.get('tapping_start_time'),
            tapping_start_datetime=_dt(d.get('tapping_start_datetime')),
            tapping_end_time=d.get('tapping_end_time'),
            tapping_end_datetime=_dt(d.get('tapping_end_datetime')),
            last_tapping_end_time=d.get('last_tapping_end_time'),
            last_tapping_end_datetime=_dt(d.get('last_tapping_end_datetime')),
            tapping_events=d.get('tapping_events', []),
            deslagging_events=d.get('deslagging_events', []),
            spectro_events=d.get('spectro_events', []),
            pyrometer_events=d.get('pyrometer_events', []),
            locked_trolley_id=d.get('locked_trolley_id'),
        )

    def _restore_from_checkpoint(self) -> None:
        """On startup, restore active cycle from DB checkpoint if present and fresh."""
        try:
            raw = self.db_manager.load_active_cycle_checkpoint()
            if not raw:
                return
            d = json.loads(raw)
            cycle = self._cycle_from_dict(d)

            # Discard if older than 4 hours (well beyond the 300s finalization window)
            age_s = time.time() - cycle.cycle_start_time
            if age_s > 4 * 3600:
                logger.warning(
                    f"Discarding stale active cycle checkpoint ({age_s / 3600:.1f}h old): {cycle.heat_no}"
                )
                self.db_manager.clear_active_cycle_checkpoint()
                return

            self.active_cycle = cycle

            # Ensure heat_counter is at least as high as the restored cycle
            try:
                counter = int(cycle.heat_no.split('_')[1])
                if counter > self.heat_counter:
                    self.heat_counter = counter
            except (IndexError, ValueError):
                pass

            logger.info(
                f"✓ Restored active cycle from checkpoint: {cycle.heat_no} "
                f"(age={age_s:.0f}s, tapping={len(cycle.tapping_events)}, "
                f"deslagging={len(cycle.deslagging_events)}, "
                f"moulds={len(cycle.mould_pourings)})"
            )
        except Exception as e:
            logger.error(f"Failed to restore active cycle checkpoint: {e}", exc_info=True)
            try:
                self.db_manager.clear_active_cycle_checkpoint()
            except Exception:
                pass

    def _maybe_checkpoint(self) -> None:
        """Write active cycle to DB if the checkpoint interval has elapsed."""
        if self.active_cycle is None:
            return
        now = time.monotonic()
        if now - self._last_checkpoint_time < _CHECKPOINT_INTERVAL_S:
            return
        try:
            d = self._cycle_to_dict(self.active_cycle)
            self.db_manager.save_active_cycle_checkpoint(json.dumps(d))
            self._last_checkpoint_time = now
            logger.debug(f"Checkpointed {self.active_cycle.heat_no}")
        except Exception as e:
            logger.warning(f"Active cycle checkpoint failed: {e}")

    def _clear_checkpoint(self) -> None:
        """Remove the checkpoint row after a cycle is finalized."""
        try:
            self.db_manager.clear_active_cycle_checkpoint()
        except Exception as e:
            logger.warning(f"Failed to clear active cycle checkpoint: {e}")

    # ------------------------------------------------------------------ #

    def _create_new_cycle(
        self,
        creator_track_id: Optional[int],
        event_time: float,
        event_datetime: datetime,
        reason: str,
    ) -> HeatCycle:
        """Create a new cycle seeded from the previous cycle end when available."""
        heat_no = self._get_next_heat_no()
        cycle_start_time, cycle_start_datetime = self._get_cycle_start_seed(event_time, event_datetime)

        track_ids: List[int] = []
        if creator_track_id is not None:
            track_ids.append(int(creator_track_id))

        cycle = HeatCycle(
            heat_no=heat_no,
            ladle_number="",
            ladle_track_ids=track_ids,
            cycle_start_time=cycle_start_time,
            cycle_start_datetime=cycle_start_datetime,
            cycle_active=True,
        )

        self.active_cycle = cycle
        logger.info(
            "🔥 NEW HEAT CYCLE: %s (reason=%s, start=%s)",
            heat_no,
            reason,
            cycle_start_datetime.isoformat(),
        )
        return cycle

    def update_pouring_session_presence(
        self,
        track_id: int,
        current_time: float,
        current_datetime: datetime,
    ) -> None:
        """
        Refresh cycle presence from a valid mouth-in-trolley observation.

        This is the only signal that should keep a pouring-backed cycle alive.
        """
        if self.active_cycle is None:
            self._create_new_cycle(track_id, current_time, current_datetime, reason="pouring_session")

        cycle = self.active_cycle
        cycle.has_pouring_session = True
        cycle.last_pouring_presence_time = current_time
        cycle.last_pouring_presence_datetime = current_datetime

        if track_id not in cycle.ladle_track_ids:
            cycle.ladle_track_ids.append(track_id)
        self._maybe_checkpoint()

    def update_ladle_presence(self, ladle_track_id: int, current_time: float, current_datetime: datetime) -> None:
        """Backward-compatible alias for older callers."""
        self.update_pouring_session_presence(ladle_track_id, current_time, current_datetime)

    def notify_session_ended(self) -> None:
        """Called when a pouring session ends. If no moulds were confirmed, reset the
        has_pouring_session flag so the cycle waits under the tapping_only_timeout path
        instead of the short ladle_absence_timeout."""
        if self.active_cycle is None:
            return
        cycle = self.active_cycle
        if cycle.has_pouring_session and not cycle.mould_pourings:
            cycle.has_pouring_session = False
            logger.info(
                "[%s] Session ended with no confirmed moulds — resetting session flag "
                "(will wait %.0fs for next pour)",
                cycle.heat_no, self.tapping_only_timeout,
            )

    def add_tapping_event(self, start_wall: float, start_dt: datetime,
                          end_wall: float, end_dt: datetime, duration: float,
                          zone_name: str = None) -> None:
        """
        Add a tapping event to the active heat cycle.
        Updates tapping_start_time (min of starts) and tapping_end_time (max of ends).
        """
        if self.active_cycle is None:
            self._create_new_cycle(None, start_wall, start_dt, reason="tapping")
            self._backfill_pre_tapping_events(self.active_cycle, start_dt)

        cycle = self.active_cycle
        if zone_name:
            self._ensure_cycle_furnace(cycle, zone_name, "tapping")
        event = {
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "duration_sec": duration,
        }
        if zone_name:
            event["zone_name"] = zone_name
        cycle.tapping_events.append(event)

        # Update aggregate: first start, last end
        if cycle.tapping_start_time is None or start_wall < cycle.tapping_start_time:
            cycle.tapping_start_time = start_wall
            cycle.tapping_start_datetime = start_dt
        if cycle.tapping_end_time is None or end_wall > cycle.tapping_end_time:
            cycle.tapping_end_time = end_wall
            cycle.tapping_end_datetime = end_dt
        cycle.last_tapping_end_time = end_wall
        cycle.last_tapping_end_datetime = end_dt

        zone_label = f" [{zone_name}]" if zone_name else ""
        logger.info(
            f"  {cycle.heat_no}: Added tapping event{zone_label} "
            f"({len(cycle.tapping_events)} total), duration={duration:.1f}s"
        )
        self._maybe_checkpoint()

    def add_deslagging_event(self, start_wall: float, start_dt: datetime,
                             end_wall: float, end_dt: datetime, duration: float,
                             zone_name: str = None) -> None:
        """Add a deslagging event to the active heat cycle."""
        if self.active_cycle is None:
            logger.warning("Cannot add deslagging event: no active cycle")
            return

        cycle = self.active_cycle
        if zone_name:
            self._ensure_cycle_furnace(cycle, zone_name, "deslagging")
        event = {
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "duration_sec": duration,
        }
        if zone_name:
            event["zone_name"] = zone_name
        cycle.deslagging_events.append(event)
        zone_label = f" [{zone_name}]" if zone_name else ""
        logger.info(
            f"  {cycle.heat_no}: Added deslagging event{zone_label} "
            f"({len(cycle.deslagging_events)} total), duration={duration:.1f}s"
        )
        self._maybe_checkpoint()

    def add_spectro_event(self, start_wall: float, start_dt: datetime,
                          end_wall: float, end_dt: datetime, duration: float,
                          zone_name: str = None) -> None:
        """Add a spectro event to the active heat cycle."""
        if self.active_cycle is None:
            logger.warning("Cannot add spectro event: no active cycle")
            return

        cycle = self.active_cycle
        if zone_name:
            self._ensure_cycle_furnace(cycle, zone_name, "spectro")
        event = {
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "duration_sec": duration,
        }
        if zone_name:
            event["zone_name"] = zone_name
        cycle.spectro_events.append(event)
        zone_label = f" [{zone_name}]" if zone_name else ""
        logger.info(
            f"  {cycle.heat_no}: Added spectro event{zone_label} "
            f"({len(cycle.spectro_events)} total), duration={duration:.1f}s"
        )
        self._maybe_checkpoint()

    def add_pyrometer_event(self, start_wall: float, start_dt: datetime,
                            end_wall: float, end_dt: datetime, duration: float,
                            zone_name: str = None) -> None:
        """Add a pyrometer event to the active heat cycle."""
        if self.active_cycle is None:
            logger.warning("Cannot add pyrometer event: no active cycle")
            return

        cycle = self.active_cycle
        if zone_name:
            self._ensure_cycle_furnace(cycle, zone_name, "pyrometer")
        event = {
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "duration_sec": duration,
        }
        if zone_name:
            event["zone_name"] = zone_name
        cycle.pyrometer_events.append(event)
        zone_label = f" [{zone_name}]" if zone_name else ""
        logger.info(
            f"  {cycle.heat_no}: Added pyrometer event{zone_label} "
            f"({len(cycle.pyrometer_events)} total), duration={duration:.1f}s"
        )
        self._maybe_checkpoint()

    def lock_trolley(self, trolley_track_id: int) -> None:
        """Lock the active cycle to a specific trolley (first pour trolley)."""
        if self.active_cycle is None:
            logger.warning("Cannot lock trolley: no active cycle")
            return

        self.active_cycle.locked_trolley_id = trolley_track_id
        logger.info(f"  {self.active_cycle.heat_no}: Locked to trolley T{trolley_track_id}")

    def add_pouring_to_cycle(
        self,
        ladle_track_id: int,
        mould_id: str,
        mould_track_id: int,
        start_time: float,
        start_datetime: datetime,
        sync_id: str,
        slno: int
    ) -> Optional[str]:
        """
        Add pouring start event to active cycle.
        
        Args:
            ladle_track_id: Ladle tracker ID
            mould_id: Mould identifier (e.g., "MOULD_T4")
            mould_track_id: Mould tracker ID
            start_time: Pouring start wall clock time
            start_datetime: Pouring start datetime
            sync_id: Sync ID for this pouring
            slno: Serial number
        
        Returns:
            heat_no if cycle exists, None otherwise
        """
        self.update_pouring_session_presence(ladle_track_id, start_time, start_datetime)
        cycle = self.active_cycle
        if ladle_track_id not in cycle.ladle_track_ids:
            cycle.ladle_track_ids.append(ladle_track_id)
        
        # Create pouring record
        pouring = MouldPouringRecord(
            mould_id=mould_id,
            mould_track_id=mould_track_id,
            start_time=start_time,
            start_datetime=start_datetime,
            sync_id=sync_id,
            slno=slno
        )
        
        cycle.mould_pourings.append(pouring)
        logger.info(f"  📊 Added {mould_id} to {cycle.heat_no} ({len(cycle.mould_pourings)} moulds)")
        self._maybe_checkpoint()

        return cycle.heat_no
    
    def update_pouring_end(
        self,
        ladle_track_id: int,
        mould_id: str,
        end_time: float,
        end_datetime: datetime,
        duration_seconds: float
    ) -> bool:
        """
        Update pouring end time for a mould in active cycle.
        
        Args:
            ladle_track_id: Ladle tracker ID
            mould_id: Mould identifier
            end_time: Pouring end wall clock time
            end_datetime: Pouring end datetime
            duration_seconds: Total pouring duration
        
        Returns:
            True if updated successfully, False otherwise
        """
        if self.active_cycle is None:
            logger.warning(f"Cannot update pouring end: no active cycle for ladle {ladle_track_id}")
            return False
        
        cycle = self.active_cycle
        
        # Find matching pouring record
        for pouring in cycle.mould_pourings:
            if pouring.mould_id == mould_id and pouring.end_time is None:
                pouring.end_time = end_time
                pouring.end_datetime = end_datetime
                pouring.duration_seconds = duration_seconds
                logger.debug(f"  ✓ Updated {mould_id} end time in {cycle.heat_no}")
                self._maybe_checkpoint()
                return True

        logger.warning(f"Could not find active pouring for {mould_id} in {cycle.heat_no}")
        return False

    def replace_mould_pourings(self, records: List[Dict], reason: str = "") -> bool:
        """Replace active-cycle mould pourings with finalized physical-count records."""
        if self.active_cycle is None:
            logger.warning("Cannot replace mould pourings: no active cycle")
            return False

        cycle = self.active_cycle
        pourings: List[MouldPouringRecord] = []
        for record in records:
            try:
                ladle_track_id = int(record.get("ladle_track_id") or record.get("mould_track_id") or 0)
                pouring = MouldPouringRecord(
                    mould_id=str(record["mould_id"]),
                    mould_track_id=int(record.get("mould_track_id") or ladle_track_id),
                    start_time=float(record["start_time"]),
                    start_datetime=record["start_datetime"],
                    end_time=float(record["end_time"]) if record.get("end_time") is not None else None,
                    end_datetime=record.get("end_datetime"),
                    duration_seconds=float(record.get("duration_seconds") or 0.0),
                    sync_id=str(record.get("sync_id") or ""),
                    slno=int(record.get("slno") or 0),
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping invalid physical mould record: %s (%s)", record, exc)
                continue

            pourings.append(pouring)
            if ladle_track_id > 0 and ladle_track_id not in cycle.ladle_track_ids:
                cycle.ladle_track_ids.append(ladle_track_id)

        cycle.mould_pourings = pourings
        logger.info(
            "  📊 Replaced %s mould pourings with %d physical moulds%s",
            cycle.heat_no,
            len(pourings),
            f" ({reason})" if reason else "",
        )
        self._maybe_checkpoint()
        return True
    
    def check_and_finalize_cycles(self, current_time: float, current_datetime: datetime) -> List[HeatCycle]:
        """
        Check for cycles that should be finalized after the configured inactivity window.
        
        Args:
            current_time: Current wall clock time
            current_datetime: Current datetime
        
        Returns:
            List of finalized HeatCycle objects ready for database insert
        """
        if self.active_cycle is None:
            return []

        cycle = self.active_cycle
        finalize_time = None
        finalize_datetime = None
        finalize_reason = None

        if cycle.has_pouring_session and cycle.last_pouring_presence_time is not None:
            time_since_last_presence = current_time - cycle.last_pouring_presence_time
            if time_since_last_presence >= self.ladle_absence_timeout:
                finalize_time = cycle.last_pouring_presence_time
                finalize_datetime = cycle.last_pouring_presence_datetime or current_datetime
                finalize_reason = f"pouring session absent for {time_since_last_presence:.1f}s"
        elif cycle.tapping_events and cycle.last_tapping_end_time is not None:
            time_since_tapping_end = current_time - cycle.last_tapping_end_time
            if time_since_tapping_end >= self.tapping_only_timeout:
                finalize_time = cycle.last_tapping_end_time
                finalize_datetime = cycle.last_tapping_end_datetime or current_datetime
                finalize_reason = f"no pouring after tapping for {time_since_tapping_end:.1f}s"

        if finalize_time is None or finalize_datetime is None:
            return []

        logger.info("⏱️  Finalizing %s: %s", cycle.heat_no, finalize_reason)
        self._finalize_cycle(cycle, finalize_time, finalize_datetime)
        self.previous_cycle_end_time = cycle.cycle_end_time
        self.previous_cycle_end_datetime = cycle.cycle_end_datetime
        self.finalized_cycles[cycle.heat_no] = cycle
        self.active_cycle = None
        self._clear_checkpoint()
        return [cycle]
    
    def _finalize_cycle(self, cycle: HeatCycle, end_time: float, end_datetime: datetime) -> None:
        """
        Finalize a heat cycle and prepare for sync.
        
        Calculates:
        - pouring_start_time (first mould start)
        - pouring_end_time (last mould end)
        - total_pouring_time (sum of all durations)
        - mould_wise_pouring_time (dict of mould: duration)
        """
        cycle.cycle_active = False
        cycle.cycle_end_time = end_time
        cycle.cycle_end_datetime = end_datetime
        
        if not cycle.mould_pourings and not cycle.tapping_events:
            logger.warning(f"Cycle {cycle.heat_no} has no pourings or tapping events")
            cycle.total_pouring_time = 0
            cycle.mould_wise_pouring_time = []
            return
        
        # Find first and last pouring times
        start_times = [p.start_datetime for p in cycle.mould_pourings]
        end_times = [p.end_datetime for p in cycle.mould_pourings if p.end_datetime]
        
        cycle.pouring_start_time = min(start_times) if start_times else None
        cycle.pouring_end_time = max(end_times) if end_times else None
        
        # Calculate total pouring time (sum of individual durations)
        total_seconds = sum(
            p.duration_seconds for p in cycle.mould_pourings
            if p.duration_seconds is not None
        )
        
        # Store as integer seconds for API
        cycle.total_pouring_time = int(total_seconds)
        
        # Build mould-wise pouring time array with API format
        mould_wise = []
        for pouring in cycle.mould_pourings:
            if pouring.duration_seconds and pouring.start_datetime and pouring.end_datetime:
                mould_wise.append({
                    "mould_id": pouring.mould_id,
                    "start": pouring.start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    "end": pouring.end_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration": str(int(pouring.duration_seconds))  # seconds as string
                })
        
        cycle.mould_wise_pouring_time = mould_wise

        if not cycle.mould_pourings:
            cycle.total_pouring_time = 0
            cycle.mould_wise_pouring_time = []
        
        # Move to finalized cycles
        self.finalized_cycles[cycle.heat_no] = cycle
        
        # Log tapping info if present
        tapping_info = ""
        if cycle.tapping_events:
            tapping_info = (
                f", Tapping: {len(cycle.tapping_events)} events"
                f" ({cycle.tapping_start_datetime.strftime('%H:%M:%S') if cycle.tapping_start_datetime else '?'}"
                f"-{cycle.tapping_end_datetime.strftime('%H:%M:%S') if cycle.tapping_end_datetime else '?'})"
            )

        deslag_count = len(cycle.deslagging_events)
        spectro_count = len(cycle.spectro_events)
        pyro_count = len(cycle.pyrometer_events)

        logger.info(
            f"CYCLE COMPLETE: {cycle.heat_no} - "
            f"{len(cycle.mould_pourings)} moulds, "
            f"Total: {cycle.total_pouring_time}s, "
            f"Ladle IDs: {cycle.ladle_track_ids}"
            f"{tapping_info}, Deslagging: {deslag_count}, Spectro: {spectro_count}, Pyrometer: {pyro_count}"
        )
    
    def get_finalized_cycles(self) -> List[HeatCycle]:
        """Get all finalized cycles ready for sync."""
        return list(self.finalized_cycles.values())
    
    def mark_cycle_synced(self, heat_no: str) -> bool:
        """
        Remove cycle from finalized list after successful sync.
        
        Args:
            heat_no: Heat cycle identifier
        
        Returns:
            True if removed, False if not found
        """
        if heat_no in self.finalized_cycles:
            del self.finalized_cycles[heat_no]
            logger.debug(f"Marked {heat_no} as synced and removed")
            return True
        return False
    
    def get_stats(self) -> Dict:
        """Get cycle manager statistics."""
        return {
            "active_cycles": 1 if self.active_cycle else 0,
            "finalized_cycles": len(self.finalized_cycles),
            "total_heat_counter": self.heat_counter,
            "active_ladles": self.active_cycle.ladle_track_ids if self.active_cycle else []
        }
