"""
Base Stream Processor
Abstract base class for stream-specific processing logic
"""
from abc import ABC, abstractmethod
from typing import Any, Optional


class StreamProcessor(ABC):
    """
    Base class for stream-specific processing logic.

    Each processor handles business logic for a specific detection task:
    - PPE detection and violation tracking
    - Heat cycle tracking (pan zone entry)
    - Tapping detection (ladle activity)
    - Deslagging monitoring (brightness analysis)
    """

    def __init__(self, db: Any, config: dict):
        """
        Initialize stream processor.

        Args:
            db: Database manager instance (HiConDatabase)
            config: Configuration dictionary with processor-specific settings
        """
        self.db = db
        self.config = config

    @abstractmethod
    def process_frame(self, frame_meta: Any, frame: Any, timestamp: float,
                      datetime_obj: Any) -> Optional[str]:
        """
        Process a single frame with detections.

        Args:
            frame_meta: DeepStream frame metadata (NvDsFrameMeta)
            frame: Numpy array of the frame image
            timestamp: Current timestamp (seconds)
            datetime_obj: Datetime object for current frame time

        Returns:
            Optional signal string for pipeline control (e.g., 'disable_pan_model', 'enable_pan_model')
            or None if no signal needed
        """
        raise NotImplementedError("Subclasses must implement process_frame()")
