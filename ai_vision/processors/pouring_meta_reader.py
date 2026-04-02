"""
Low-level metadata decoder for the C++ pouring plugin.

The hybrid pouring architecture keeps the native plugin focused on object-meta
association + session gating and moves business logic back into Python. This
module only decodes the C++ NvDsUserMeta payload and emits lightweight
heartbeats for runtime visibility.
"""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass

import pyds

logger = logging.getLogger(__name__)


try:
    HICON_POURING_META_TYPE = int(pyds.NVDS_START_USER_META) + 1
except (AttributeError, TypeError):
    HICON_POURING_META_TYPE = 256 + 1


EVENT_NONE = 0
EVENT_SESSION_START = 1
EVENT_SESSION_END = 2

_EVENT_NAMES = {
    EVENT_NONE: "NONE",
    EVENT_SESSION_START: "SESSION_START",
    EVENT_SESSION_END: "SESSION_END",
}


class HiConPouringMeta(ctypes.Structure):
    """ctypes mirror of gsthiconpouring.h::HiConPouringMeta (version 2)."""

    _fields_ = [
        ("version", ctypes.c_uint32),
        ("session_active", ctypes.c_uint32),
        ("mouth_present_in_trolley", ctypes.c_uint32),
        ("probe_valid", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("event", ctypes.c_uint32),
        ("trolley_track_id", ctypes.c_uint64),
        ("mouth_track_id", ctypes.c_uint64),
        ("trolley_bbox", ctypes.c_float * 4),
        ("mouth_bbox", ctypes.c_float * 4),
        ("probe_x_px", ctypes.c_float),
        ("probe_y_px", ctypes.c_float),
        ("mouth_norm_x", ctypes.c_float),
        ("mouth_norm_y", ctypes.c_float),
    ]


@dataclass
class DecodedPouringState:
    version: int
    session_active: bool
    mouth_present_in_trolley: bool
    probe_valid: bool
    event: int
    trolley_track_id: int
    mouth_track_id: int
    trolley_bbox: tuple[float, float, float, float]
    mouth_bbox: tuple[float, float, float, float]
    probe_x_px: float
    probe_y_px: float
    mouth_norm_x: float
    mouth_norm_y: float

    @property
    def has_valid_mouth_norm(self) -> bool:
        return self.mouth_norm_x >= 0.0 and self.mouth_norm_y >= 0.0


class PouringMetaReader:
    """Decode one authoritative pouring-state snapshot from each frame."""

    def __init__(self, stream_label="stream", heartbeat_every=250, **_unused):
        self.stream_label = stream_label
        self.heartbeat_every = heartbeat_every
        self._frames_seen = 0
        self._meta_frames = 0
        self._last_state: DecodedPouringState | None = None

    def process_frame_meta(self, frame_meta):
        """Backward-compatible alias for decode_frame_meta()."""
        return self.decode_frame_meta(frame_meta)

    def decode_frame_meta(self, frame_meta):
        """Decode the plugin payload from one frame, if present."""
        self._frames_seen += 1
        decoded = None

        l_user = frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            if user_meta.base_meta.meta_type == HICON_POURING_META_TYPE:
                try:
                    meta = self._parse_meta(user_meta)
                    if meta is not None:
                        decoded = self._decode_struct(meta)
                        self._meta_frames += 1
                        self._last_state = decoded
                        break
                except Exception as exc:
                    logger.error("Error parsing pouring meta: %s", exc, exc_info=True)

            try:
                l_user = l_user.next
            except StopIteration:
                break

        self._log_heartbeat()
        return decoded

    def _log_heartbeat(self):
        if not self.heartbeat_every or (self._frames_seen % self.heartbeat_every) != 0:
            return

        if self._last_state is None:
            logger.info(
                "[CPP-POURING][%s] heartbeat: meta_frames=%d/%d, no meta decoded yet",
                self.stream_label,
                self._meta_frames,
                self._frames_seen,
            )
            return

        logger.info(
            "[CPP-POURING][%s] heartbeat: meta_frames=%d/%d, session=%s, mouth=%s, "
            "probe=%s, event=%s, trolley=%d, mouth_track=%d, norm=(%.3f, %.3f)",
            self.stream_label,
            self._meta_frames,
            self._frames_seen,
            self._last_state.session_active,
            self._last_state.mouth_present_in_trolley,
            self._last_state.probe_valid,
            _EVENT_NAMES.get(self._last_state.event, str(self._last_state.event)),
            self._last_state.trolley_track_id,
            self._last_state.mouth_track_id,
            self._last_state.mouth_norm_x,
            self._last_state.mouth_norm_y,
        )

    @staticmethod
    def _parse_meta(user_meta):
        data = user_meta.user_meta_data
        if data is None or data == 0:
            return None

        addr = None
        try:
            addr = int(data)
        except (TypeError, ValueError):
            try:
                addr = pyds.get_ptr(data)
            except Exception:
                logger.warning("Cannot cast user_meta_data: type=%s", type(data))
                return None

        if not addr:
            return None

        meta_ptr = ctypes.cast(addr, ctypes.POINTER(HiConPouringMeta))
        return meta_ptr.contents

    @staticmethod
    def _decode_bbox(values):
        return tuple(float(values[i]) for i in range(4))

    @classmethod
    def _decode_struct(cls, meta):
        return DecodedPouringState(
            version=int(meta.version),
            session_active=bool(meta.session_active),
            mouth_present_in_trolley=bool(meta.mouth_present_in_trolley),
            probe_valid=bool(meta.probe_valid),
            event=int(meta.event),
            trolley_track_id=int(meta.trolley_track_id),
            mouth_track_id=int(meta.mouth_track_id),
            trolley_bbox=cls._decode_bbox(meta.trolley_bbox),
            mouth_bbox=cls._decode_bbox(meta.mouth_bbox),
            probe_x_px=float(meta.probe_x_px),
            probe_y_px=float(meta.probe_y_px),
            mouth_norm_x=float(meta.mouth_norm_x),
            mouth_norm_y=float(meta.mouth_norm_y),
        )
