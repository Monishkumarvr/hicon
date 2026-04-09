"""
Low-level metadata decoder for the C++ melting plugin.
"""

from __future__ import annotations

import ctypes
import logging
from dataclasses import dataclass

import pyds

logger = logging.getLogger(__name__)


try:
    HICON_MELTING_META_TYPE = int(pyds.NVDS_START_USER_META) + 2
except (AttributeError, TypeError):
    HICON_MELTING_META_TYPE = 256 + 2

HICON_MELTING_MAX_ZONES = 8


class HiConMeltingZoneMeta(ctypes.Structure):
    _fields_ = [
        ("valid", ctypes.c_uint32),
        ("active", ctypes.c_uint32),
        ("raw_count", ctypes.c_uint32),
        ("filtered_count", ctypes.c_uint32),
        ("white_ratio", ctypes.c_float),
        ("max_blob_area", ctypes.c_float),
        ("max_blob_brightness", ctypes.c_float),
        ("reserved0", ctypes.c_uint32),
    ]


class HiConMeltingMeta(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint32),
        ("blackout_active", ctypes.c_uint32),
        ("tapping_zone_count", ctypes.c_uint32),
        ("deslagging_zone_count", ctypes.c_uint32),
        ("spectro_zone_count", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("frame_num", ctypes.c_uint64),
        ("ntp_timestamp", ctypes.c_uint64),
        ("tapping", HiConMeltingZoneMeta * HICON_MELTING_MAX_ZONES),
        ("deslagging", HiConMeltingZoneMeta * HICON_MELTING_MAX_ZONES),
        ("spectro", HiConMeltingZoneMeta * HICON_MELTING_MAX_ZONES),
    ]


@dataclass
class DecodedMeltingZoneState:
    valid: bool
    active: bool
    raw_count: int
    filtered_count: int
    white_ratio: float
    max_blob_area: float
    max_blob_brightness: float


@dataclass
class DecodedMeltingState:
    version: int
    debug_code: int
    blackout_active: bool
    frame_num: int
    ntp_timestamp: int
    tapping: list[DecodedMeltingZoneState]
    deslagging: list[DecodedMeltingZoneState]
    spectro: list[DecodedMeltingZoneState]


class MeltingMetaReader:
    """Decode one authoritative melting-state snapshot from each frame."""

    def __init__(self, stream_label="stream", heartbeat_every=250):
        self.stream_label = stream_label
        self.heartbeat_every = heartbeat_every
        self._frames_seen = 0
        self._meta_frames = 0
        self._last_state: DecodedMeltingState | None = None
        self._last_seen_meta_types: list[int] = []

    def decode_frame_meta(self, frame_meta):
        self._frames_seen += 1
        decoded = None
        seen_meta_types: list[int] = []

        l_user = frame_meta.frame_user_meta_list
        while l_user is not None:
            try:
                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
            except StopIteration:
                break

            try:
                seen_meta_types.append(int(user_meta.base_meta.meta_type))
            except Exception:
                pass

            if user_meta.base_meta.meta_type == HICON_MELTING_META_TYPE:
                try:
                    meta = self._parse_meta(user_meta)
                    if meta is not None:
                        decoded = self._decode_struct(meta)
                        self._meta_frames += 1
                        self._last_state = decoded
                        break
                except Exception as exc:
                    logger.error("Error parsing melting meta: %s", exc, exc_info=True)

            try:
                l_user = l_user.next
            except StopIteration:
                break

        self._last_seen_meta_types = seen_meta_types
        self._log_heartbeat()
        return decoded

    def _log_heartbeat(self):
        if not self.heartbeat_every or (self._frames_seen % self.heartbeat_every) != 0:
            return

        if self._last_state is None:
            logger.info(
                "[CPP-MELTING][%s] heartbeat: meta_frames=%d/%d, no meta decoded yet, "
                "seen_meta_types=%s expected=%d",
                self.stream_label,
                self._meta_frames,
                self._frames_seen,
                self._last_seen_meta_types,
                HICON_MELTING_META_TYPE,
            )
            return

        def _summary(values):
            active = sum(1 for item in values if item.active)
            valid = sum(1 for item in values if item.valid)
            return f"{active}/{valid}"

        logger.info(
            "[CPP-MELTING][%s] heartbeat: meta_frames=%d/%d blackout=%s debug=%d "
            "counts=%d/%d/%d tapping=%s deslag=%s spectro=%s",
            self.stream_label,
            self._meta_frames,
            self._frames_seen,
            self._last_state.blackout_active,
            self._last_state.debug_code,
            len(self._last_state.tapping),
            len(self._last_state.deslagging),
            len(self._last_state.spectro),
            _summary(self._last_state.tapping),
            _summary(self._last_state.deslagging),
            _summary(self._last_state.spectro),
        )

    @staticmethod
    def _parse_meta(user_meta):
        data = user_meta.user_meta_data
        if data is None or data == 0:
            return None

        addr = None
        try:
            addr = pyds.get_ptr(data)
        except Exception:
            try:
                addr = int(data)
            except (TypeError, ValueError):
                logger.warning("Cannot cast melting user_meta_data: type=%s", type(data))
                return None

        if not addr:
            return None

        meta_ptr = ctypes.cast(addr, ctypes.POINTER(HiConMeltingMeta))
        return meta_ptr.contents

    @staticmethod
    def _decode_zone(zone):
        return DecodedMeltingZoneState(
            valid=bool(zone.valid),
            active=bool(zone.active),
            raw_count=int(zone.raw_count),
            filtered_count=int(zone.filtered_count),
            white_ratio=float(zone.white_ratio),
            max_blob_area=float(zone.max_blob_area),
            max_blob_brightness=float(zone.max_blob_brightness),
        )

    @classmethod
    def _decode_struct(cls, meta):
        tapping_count = min(int(meta.tapping_zone_count), HICON_MELTING_MAX_ZONES)
        deslagging_count = min(int(meta.deslagging_zone_count), HICON_MELTING_MAX_ZONES)
        spectro_count = min(int(meta.spectro_zone_count), HICON_MELTING_MAX_ZONES)
        return DecodedMeltingState(
            version=int(meta.version),
            debug_code=int(meta.reserved0),
            blackout_active=bool(meta.blackout_active),
            frame_num=int(meta.frame_num),
            ntp_timestamp=int(meta.ntp_timestamp),
            tapping=[cls._decode_zone(meta.tapping[i]) for i in range(tapping_count)],
            deslagging=[cls._decode_zone(meta.deslagging[i]) for i in range(deslagging_count)],
            spectro=[cls._decode_zone(meta.spectro[i]) for i in range(spectro_count)],
        )
