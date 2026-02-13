"""
Heat Cycle Manager - Track ladle-based pouring cycles

Manages heat cycles where:
- One continuous ladle presence window = One heat cycle
- Multiple mould pourings belong to same cycle
- Cycle ends after 5 minutes of NO ladle
- Aggregates pouring times for API POST
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

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
    """Represents one complete heat cycle (ladle-based)."""
    heat_no: str  # Sequential: HEAT_0001, HEAT_0002, etc.
    ladle_number: str  # Sequential: LAD_001, LAD_002, etc.
    ladle_track_ids: List[int]  # All tracker IDs seen during this cycle
    cycle_start_time: float  # Wall clock when ladle first appeared
    cycle_start_datetime: datetime

    # Pouring aggregation
    mould_pourings: List[MouldPouringRecord] = field(default_factory=list)

    # Cycle state
    ladle_last_seen: float = 0.0  # Last time ladle was in frame
    cycle_active: bool = True  # False when finalized

    # Tapping aggregation (first start = cycle tapping start, last end = cycle tapping end)
    tapping_start_time: Optional[float] = None
    tapping_start_datetime: Optional[datetime] = None
    tapping_end_time: Optional[float] = None
    tapping_end_datetime: Optional[datetime] = None
    tapping_events: List[Dict] = field(default_factory=list)

    # Deslagging aggregation
    deslagging_events: List[Dict] = field(default_factory=list)

    # Spectro aggregation
    spectro_events: List[Dict] = field(default_factory=list)

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
    - First ladle appearance creates new cycle
    - ANY ladle extends the current cycle (tracker ID changes do not split cycles)
    - Cycle ends after 5 minutes of NO ladle
    - Finalized cycles POSTed to API
    """
    
    def __init__(self, db_manager, ladle_absence_timeout: float = 300.0):
        """
        Initialize heat cycle manager.
        
        Args:
            db_manager: Database manager instance for querying existing counters
            ladle_absence_timeout: Seconds of ladle absence before finalizing cycle (default 300 = 5min)
        """
        self.db_manager = db_manager
        self.ladle_absence_timeout = ladle_absence_timeout
        
        # Single active cycle (time-based, not per-ladle ID)
        self.active_cycle: Optional[HeatCycle] = None
        
        # Sequential counters - initialize from database
        self.heat_counter = self._get_last_heat_counter()
        self.ladle_counter = self._get_last_ladle_counter()
        
        # Map ladle_track_id to assigned sequential ladle_number
        self.ladle_track_to_number: Dict[int, str] = {}
        
        # Finalized cycles ready for sync (keyed by heat_no)
        self.finalized_cycles: Dict[str, HeatCycle] = {}
        
        logger.info(f"✓ HeatCycleManager initialized (ladle timeout: {ladle_absence_timeout}s)")
        logger.info(f"  Heat counter starting at: {self.heat_counter}")
        logger.info(f"  Ladle counter starting at: {self.ladle_counter}")
    
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
    
    def _get_last_ladle_counter(self) -> int:
        """Query database for the last used ladle counter."""
        try:
            conn = self.db_manager._get_connection()
            c = conn.cursor()
            c.execute("SELECT DISTINCT ladle_number FROM heat_cycles ORDER BY ladle_number DESC")
            rows = c.fetchall()
            conn.close()
            
            max_counter = 0
            for row in rows:
                if row[0] and row[0].startswith('LAD_'):
                    try:
                        counter = int(row[0].split('_')[1])
                        max_counter = max(max_counter, counter)
                    except (IndexError, ValueError):
                        continue
            return max_counter
        except Exception as e:
            logger.warning(f"Error querying last ladle counter: {e}")
            return 0
    
    def _get_next_heat_no(self) -> str:
        """Generate next sequential heat number."""
        self.heat_counter += 1
        return f"HEAT_{self.heat_counter:04d}"
    
    def _get_ladle_number(self, ladle_track_id: int) -> str:
        """Get or assign sequential ladle number for a track_id."""
        if ladle_track_id not in self.ladle_track_to_number:
            self.ladle_counter += 1
            self.ladle_track_to_number[ladle_track_id] = f"LAD_{self.ladle_counter:03d}"
        return self.ladle_track_to_number[ladle_track_id]
    
    def update_ladle_presence(self, ladle_track_id: int, current_time: float, current_datetime: datetime) -> None:
        """
        Update ladle last-seen timestamp (call every frame where ladle is detected).
        
        Args:
            ladle_track_id: Ladle tracker ID
            current_time: Current wall clock time
            current_datetime: Current datetime object
        """
        if self.active_cycle is None:
            # Create new heat cycle
            self._create_new_cycle(ladle_track_id, current_time, current_datetime)
            return

        # Extend existing cycle with ANY ladle
        self.active_cycle.ladle_last_seen = current_time
        if ladle_track_id not in self.active_cycle.ladle_track_ids:
            self.active_cycle.ladle_track_ids.append(ladle_track_id)
            logger.info(
                f"🔄 {self.active_cycle.heat_no}: Detected ladle tracker_id={ladle_track_id} - extending cycle"
            )
    
    def add_tapping_event(self, start_wall: float, start_dt: datetime,
                          end_wall: float, end_dt: datetime, duration: float) -> None:
        """
        Add a tapping event to the active heat cycle.
        Updates tapping_start_time (min of starts) and tapping_end_time (max of ends).
        """
        if self.active_cycle is None:
            logger.warning("Cannot add tapping event: no active cycle")
            return

        cycle = self.active_cycle
        event = {
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "duration_sec": duration,
        }
        cycle.tapping_events.append(event)

        # Update aggregate: first start, last end
        if cycle.tapping_start_time is None or start_wall < cycle.tapping_start_time:
            cycle.tapping_start_time = start_wall
            cycle.tapping_start_datetime = start_dt
        if cycle.tapping_end_time is None or end_wall > cycle.tapping_end_time:
            cycle.tapping_end_time = end_wall
            cycle.tapping_end_datetime = end_dt

        logger.info(
            f"  {cycle.heat_no}: Added tapping event ({len(cycle.tapping_events)} total), "
            f"duration={duration:.1f}s"
        )

    def add_deslagging_event(self, start_wall: float, start_dt: datetime,
                             end_wall: float, end_dt: datetime, duration: float) -> None:
        """Add a deslagging event to the active heat cycle."""
        if self.active_cycle is None:
            logger.warning("Cannot add deslagging event: no active cycle")
            return

        cycle = self.active_cycle
        event = {
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "duration_sec": duration,
        }
        cycle.deslagging_events.append(event)
        logger.info(
            f"  {cycle.heat_no}: Added deslagging event ({len(cycle.deslagging_events)} total), "
            f"duration={duration:.1f}s"
        )

    def add_spectro_event(self, start_wall: float, start_dt: datetime,
                          end_wall: float, end_dt: datetime, duration: float) -> None:
        """Add a spectro event to the active heat cycle."""
        if self.active_cycle is None:
            logger.warning("Cannot add spectro event: no active cycle")
            return

        cycle = self.active_cycle
        event = {
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
            "duration_sec": duration,
        }
        cycle.spectro_events.append(event)
        logger.info(
            f"  {cycle.heat_no}: Added spectro event ({len(cycle.spectro_events)} total), "
            f"duration={duration:.1f}s"
        )

    def lock_trolley(self, trolley_track_id: int) -> None:
        """Lock the active cycle to a specific trolley (first pour trolley)."""
        if self.active_cycle is None:
            logger.warning("Cannot lock trolley: no active cycle")
            return

        self.active_cycle.locked_trolley_id = trolley_track_id
        logger.info(f"  {self.active_cycle.heat_no}: Locked to trolley T{trolley_track_id}")

    def _create_new_cycle(self, ladle_track_id: int, current_time: float, current_datetime: datetime) -> HeatCycle:
        """Create new heat cycle for ladle."""
        heat_no = self._get_next_heat_no()
        ladle_number = self._get_ladle_number(ladle_track_id)
        
        cycle = HeatCycle(
            heat_no=heat_no,
            ladle_number=ladle_number,
            ladle_track_ids=[ladle_track_id],
            cycle_start_time=current_time,
            cycle_start_datetime=current_datetime,
            ladle_last_seen=current_time,
            cycle_active=True
        )
        
        self.active_cycle = cycle
        logger.info(f"🔥 NEW HEAT CYCLE: {heat_no} (Ladle: {ladle_number}, tracker_id={ladle_track_id})")
        
        return cycle
    
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
        if self.active_cycle is None:
            logger.warning(f"Cannot add pouring: no active cycle for ladle {ladle_track_id}")
            return None
        
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
                return True
        
        logger.warning(f"Could not find active pouring for {mould_id} in {cycle.heat_no}")
        return False
    
    def check_and_finalize_cycles(self, current_time: float, current_datetime: datetime) -> List[HeatCycle]:
        """
        Check for cycles that should be finalized (ladle absent for timeout period).
        
        Args:
            current_time: Current wall clock time
            current_datetime: Current datetime
        
        Returns:
            List of finalized HeatCycle objects ready for database insert
        """
        if self.active_cycle is None:
            return []

        cycle = self.active_cycle
        time_since_last_seen = current_time - cycle.ladle_last_seen

        if time_since_last_seen < self.ladle_absence_timeout:
            return []

        logger.info(
            f"⏱️  Finalizing {cycle.heat_no}: ladle absent for {time_since_last_seen:.1f}s"
        )
        self._finalize_cycle(cycle, current_time, current_datetime)
        self.finalized_cycles[cycle.heat_no] = cycle
        self.active_cycle = None
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
        
        if not cycle.mould_pourings:
            logger.warning(f"Cycle {cycle.heat_no} has no pourings!")
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

        logger.info(
            f"CYCLE COMPLETE: {cycle.heat_no} - "
            f"{len(cycle.mould_pourings)} moulds, "
            f"Total: {cycle.total_pouring_time}s, "
            f"Ladle IDs: {cycle.ladle_track_ids}"
            f"{tapping_info}, Deslagging: {deslag_count}, Spectro: {spectro_count}"
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
