"""
Pouring Tracker - State machine for tracking pouring lifecycle

Simplified state machine (most logic is in PouringProcessor):
- IDLE → PREPARING → POURING → COOLING → IDLE
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PouringTracker:
    """
    State machine for tracking pouring lifecycle.

    States:
    - IDLE: No active pouring
    - PREPARING: Ladle detected above mould, brightness rising
    - POURING: Active pouring detected (brightness > threshold)
    - COOLING: Pouring ended, in cooldown period

    Note: Most pouring logic is implemented in PouringProcessor.
    This class provides a simple state tracking interface if needed.
    """

    def __init__(self, cooldown_seconds: int = 600):
        """
        Initialize pouring tracker.

        Args:
            cooldown_seconds: Cooldown period between pours (default 600s = 10 min)
        """
        self.cooldown_seconds = cooldown_seconds
        self.active_sessions = {}  # mould_id -> session dict
        self.mould_states = {}     # mould_id -> 'IDLE', 'PREPARING', 'POURING', 'COOLING'

        logger.info(f"✓ PouringTracker initialized (cooldown: {cooldown_seconds}s)")

    def detect_pouring_start(self, mould_id: str, ladle_id: int, timestamp: datetime) -> bool:
        """
        Detect pouring start for a mould.

        Args:
            mould_id: Mould identifier
            ladle_id: Ladle track ID
            timestamp: Current timestamp

        Returns:
            True if pouring started, False if in cooldown
        """
        current_state = self.mould_states.get(mould_id, 'IDLE')

        if current_state == 'COOLING':
            # Check if cooldown period has passed
            session = self.active_sessions.get(mould_id)
            if session and 'end_time' in session:
                cooldown_elapsed = (timestamp - session['end_time']).total_seconds()
                if cooldown_elapsed < self.cooldown_seconds:
                    logger.debug(f"Mould {mould_id} still in cooldown ({cooldown_elapsed:.0f}s / {self.cooldown_seconds}s)")
                    return False

        # Start new pouring session
        self.active_sessions[mould_id] = {
            'mould_id': mould_id,
            'ladle_id': ladle_id,
            'start_time': timestamp,
            'end_time': None
        }
        self.mould_states[mould_id] = 'POURING'

        logger.info(f"🌊 Pouring started: {mould_id} (ladle {ladle_id})")
        return True

    def detect_pouring_end(self, mould_id: str, timestamp: datetime) -> Optional[Dict]:
        """
        Detect pouring end for a mould.

        Args:
            mould_id: Mould identifier
            timestamp: Current timestamp

        Returns:
            Session dict with start_time, end_time, duration, or None
        """
        if mould_id not in self.active_sessions:
            return None

        session = self.active_sessions[mould_id]

        if session.get('end_time') is not None:
            return None  # Already ended

        # End pouring session
        session['end_time'] = timestamp
        session['duration'] = (timestamp - session['start_time']).total_seconds()

        self.mould_states[mould_id] = 'COOLING'

        logger.info(f"✅ Pouring ended: {mould_id} - Duration: {session['duration']:.1f}s")

        return session

    def update_mould_pour(self, ladle_id: int, mould_id: str,
                         timestamp: datetime, is_pouring: bool):
        """
        Update mould-specific pouring state.

        Args:
            ladle_id: Ladle track ID
            mould_id: Mould identifier
            timestamp: Current timestamp
            is_pouring: Whether pouring is active
        """
        if is_pouring:
            self.detect_pouring_start(mould_id, ladle_id, timestamp)
        else:
            self.detect_pouring_end(mould_id, timestamp)

    def get_active_session(self, mould_id: str) -> Optional[Dict]:
        """
        Get active pouring session for a mould.

        Returns:
            Session dict or None
        """
        session = self.active_sessions.get(mould_id)
        if session and session.get('end_time') is None:
            return session
        return None

    def get_mould_state(self, mould_id: str) -> str:
        """
        Get current state of a mould.

        Returns:
            State string: 'IDLE', 'PREPARING', 'POURING', 'COOLING'
        """
        return self.mould_states.get(mould_id, 'IDLE')
