"""
Brightness Tracker - IDLE/ACTIVE state machine for tapping and deslagging detection.
Uses consecutive frame counting (not timers) for state transitions.
"""
import logging
import time
from datetime import datetime
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class BrightnessTracker:
    """
    Frame-counter based state machine for brightness-driven event detection.

    States: IDLE -> ACTIVE -> IDLE
    Transition to ACTIVE: N consecutive frames with white_ratio >= start_threshold
    Transition to IDLE:   N consecutive frames with white_ratio < end_threshold
    """

    def __init__(self, name: str, brightness_threshold: int,
                 start_white_ratio: float, start_frame_count: int,
                 end_white_ratio: float, end_frame_count: int,
                 max_white_ratio: float = None):
        """
        Args:
            name: Event type name (e.g., "tapping", "deslagging", "spectro")
            brightness_threshold: Y-channel threshold for white pixel detection
            start_white_ratio: White pixel ratio to trigger start
            start_frame_count: Consecutive frames needed to start
            end_white_ratio: White pixel ratio threshold for end
            end_frame_count: Consecutive frames needed to end
            max_white_ratio: If set, events where ratio exceeds this are discarded
                             as false positives (e.g., 0.2 for spectro)
        """
        self.name = name
        self.brightness_threshold = brightness_threshold
        self.start_white_ratio = start_white_ratio
        self.start_frame_count = start_frame_count
        self.end_white_ratio = end_white_ratio
        self.end_frame_count = end_frame_count
        self.max_white_ratio = max_white_ratio

        # State
        self.state = "IDLE"  # IDLE or ACTIVE
        self.start_counter = 0
        self.end_counter = 0

        # Current event
        self.event_start_time = None
        self.event_start_datetime = None
        self._exceeded_max = False  # Tracks if max_white_ratio was exceeded during event

        max_info = f", max_ratio<{max_white_ratio}" if max_white_ratio else ""
        logger.info(
            f"BrightnessTracker[{name}] initialized: "
            f"Y>{brightness_threshold}, start_ratio>={start_white_ratio} x{start_frame_count}f, "
            f"end_ratio<{end_white_ratio} x{end_frame_count}f{max_info}"
        )

    def update(self, white_ratio: float) -> Optional[Dict]:
        """
        Update state machine with new frame's white ratio.

        Args:
            white_ratio: Ratio of white pixels in ROI (0.0 - 1.0)

        Returns:
            Event dict on state transition, None otherwise.
            On ACTIVE->IDLE: {"type": name, "start": datetime, "end": datetime, "duration_sec": float}
        """
        if self.state == "IDLE":
            if white_ratio >= self.start_white_ratio:
                self.start_counter += 1
                if self.start_counter >= self.start_frame_count:
                    # Transition to ACTIVE
                    self.state = "ACTIVE"
                    self.event_start_time = time.time()
                    self.event_start_datetime = datetime.now()
                    self.start_counter = 0
                    self.end_counter = 0
                    self._exceeded_max = False
                    # Check if already exceeding max on start
                    if self.max_white_ratio and white_ratio >= self.max_white_ratio:
                        self._exceeded_max = True
                    logger.info(
                        f"[{self.name}] ACTIVE - white_ratio={white_ratio:.3f} "
                        f"sustained for {self.start_frame_count} frames"
                    )
                    return {
                        "type": self.name,
                        "phase": "start",
                        "start": self.event_start_datetime.isoformat(),
                        "start_wall": self.event_start_time,
                        "start_datetime": self.event_start_datetime,
                    }
            else:
                self.start_counter = 0

        elif self.state == "ACTIVE":
            # Track if max threshold is exceeded during event
            if self.max_white_ratio and white_ratio >= self.max_white_ratio:
                self._exceeded_max = True

            if white_ratio < self.end_white_ratio:
                self.end_counter += 1
                if self.end_counter >= self.end_frame_count:
                    # Transition to IDLE
                    end_time = time.time()
                    end_datetime = datetime.now()
                    duration = end_time - self.event_start_time

                    # Discard if max threshold was exceeded (false positive)
                    if self._exceeded_max:
                        logger.info(
                            f"[{self.name}] DISCARDED - exceeded max_ratio "
                            f"{self.max_white_ratio}, duration={duration:.1f}s"
                        )
                        self.state = "IDLE"
                        self.end_counter = 0
                        self.start_counter = 0
                        self.event_start_time = None
                        self.event_start_datetime = None
                        self._exceeded_max = False
                        return None

                    event = {
                        "type": self.name,
                        "phase": "end",
                        "start": self.event_start_datetime.isoformat(),
                        "end": end_datetime.isoformat(),
                        "duration_sec": round(duration, 1),
                        "start_wall": self.event_start_time,
                        "end_wall": end_time,
                        "start_datetime": self.event_start_datetime,
                        "end_datetime": end_datetime,
                    }

                    logger.info(
                        f"[{self.name}] IDLE - event ended, duration={duration:.1f}s"
                    )

                    self.state = "IDLE"
                    self.end_counter = 0
                    self.start_counter = 0
                    self.event_start_time = None
                    self.event_start_datetime = None

                    return event
            else:
                self.end_counter = 0

        return None

    @property
    def is_active(self) -> bool:
        return self.state == "ACTIVE"
