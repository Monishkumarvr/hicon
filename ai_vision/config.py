"""
HiCon Edge AI Configuration
Centralized configuration with environment variable support
"""
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env file from ai_vision directory
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"Loaded environment from: {env_path}")
else:
    print(f"No .env file found at: {env_path} (using system environment variables)")


_RTSP_PROTOCOLS = frozenset({"auto", "tcp", "udp"})


def _get_rtsp_protocol(env_name: str, default: str) -> str:
    """Load and validate per-stream RTSP transport preference."""
    value = os.getenv(env_name, default).strip().lower()
    if value not in _RTSP_PROTOCOLS:
        raise ValueError(
            f"{env_name} must be one of {sorted(_RTSP_PROTOCOLS)} (got {value!r})"
        )
    return value


# =============================================================================
# API CONFIGURATION
# =============================================================================

# API Base URL (environment variable with fallback)
API_URL = os.getenv(
    'HICON_API_URL',
    'http://ai-bakend-v2.ap-south-1.elasticbeanstalk.com/api/v1'
)

# HMAC Secret for API authentication (REQUIRED from environment variable)
HMAC_SECRET = os.getenv('HICON_HMAC_SECRET', 'jIQI1O3rR4QyUceWgTiJ09Xn4Yo-CyiaabLE6vR8R_0lxgkN7or_298cEhOIfg9I0IU-eWA2NKHXkOoHx9shLA')

# Customer and Device Identification
CUSTOMER_ID = os.getenv('HICON_CUSTOMER_ID', '1157')
DEVICE_ID = os.getenv('HICON_DEVICE_ID', 'DEVICE001')

# Camera and Location Identification
# Stream 0 = Process camera (tapping, pouring, deslagging, spectro)
# Stream 1 = Pyrometer camera (rod insertion detection)
CAMERA_ID_STREAM_0 = os.getenv('HICON_CAMERA_ID_STREAM_0', 'Cam-Process')
CAMERA_ID_STREAM_1 = os.getenv('HICON_CAMERA_ID_STREAM_1', 'Cam-Pyrometer')
CAMERA_ID_STREAM_2 = os.getenv('HICON_CAMERA_ID_STREAM_2', 'Cam-Pouring2')
LOCATION = os.getenv('HICON_LOCATION', 'Casting Section')
FURNACE_ID = os.getenv('HICON_FURNACE_ID', LOCATION)

# =============================================================================
# SYNC CONFIGURATION
# =============================================================================

# Sync interval (seconds between cloud sync attempts)
SYNC_INTERVAL = int(os.getenv('HICON_SYNC_INTERVAL', '30'))

# Batch size (number of records to sync per request)
BATCH_SIZE = int(os.getenv('HICON_BATCH_SIZE', '50'))

# Screenshot compression settings (for cloud sync)
SCREENSHOT_MAX_WIDTH = int(os.getenv('HICON_SCREENSHOT_MAX_WIDTH', '1280'))
SCREENSHOT_JPEG_QUALITY = int(os.getenv('HICON_SCREENSHOT_JPEG_QUALITY', '75'))
SCREENSHOT_RETENTION_DAYS = int(os.getenv('HICON_SCREENSHOT_RETENTION_DAYS', '7'))

# Maximum retry attempts for failed API requests
MAX_RETRY_ATTEMPTS = int(os.getenv('HICON_MAX_RETRIES', '3'))

# Request timeout (seconds)
REQUEST_TIMEOUT = int(os.getenv('HICON_REQUEST_TIMEOUT', '30'))

# =============================================================================
# POURING DETECTION CONFIGURATION (per HiCon_Systems_Design.md Section 7.1)
# =============================================================================

# Detection confidence thresholds
MOUTH_CONFIDENCE = float(os.getenv('HICON_MOUTH_CONF', '0.40'))
TROLLEY_CONFIDENCE = float(os.getenv('HICON_TROLLEY_CONF', '0.25'))

# Session timing (ladle_mouth inside trolley bbox)
SESSION_START_DURATION = float(os.getenv('HICON_SESSION_START_DURATION', '1.0'))
SESSION_END_DURATION = float(os.getenv('HICON_SESSION_END_DURATION', '1.5'))

# Legacy single-probe offset (kept for backward compatibility; multi-probe is used)
POUR_PROBE_OFFSET_PX = int(os.getenv('HICON_POUR_PROBE_OFFSET', '50'))
POUR_PROBE_RADIUS_PX = int(os.getenv('HICON_POUR_PROBE_RADIUS', '6'))
POUR_BRIGHTNESS_START = int(os.getenv('HICON_POUR_BRIGHTNESS_START', '230'))
POUR_BRIGHTNESS_END = int(os.getenv('HICON_POUR_BRIGHTNESS_END', '180'))
POUR_START_DURATION = float(os.getenv('HICON_POUR_START_DURATION', '0.25'))
POUR_END_DURATION = float(os.getenv('HICON_POUR_END_DURATION', '1.0'))
POUR_MIN_DURATION = float(os.getenv('HICON_POUR_MIN_DURATION', '2.0'))

# Mould counting: anchor-based trolley motion + spatial clustering
MOULD_DISPLACEMENT_THRESHOLD = float(os.getenv('HICON_MOULD_DISPLACEMENT', '0.15'))
MOULD_SUSTAINED_DURATION = float(os.getenv('HICON_MOULD_SUSTAINED', '1.5'))
CLUSTER_R_CLUSTER = float(os.getenv('HICON_CLUSTER_R_CLUSTER', '0.08'))
CLUSTER_R_MERGE = float(os.getenv('HICON_CLUSTER_R_MERGE', '0.05'))  # Euclidean distance threshold
MOULD_SPLIT_MIN_DX_PX = float(os.getenv('HICON_MOULD_SPLIT_MIN_DX_PX', '12.0'))
MOULD_SPLIT_MIN_DY_PX = float(os.getenv('HICON_MOULD_SPLIT_MIN_DY_PX', '12.0'))
MOULD_SPLIT_COOLDOWN_S = float(os.getenv('HICON_MOULD_SPLIT_COOLDOWN_S', '1.5'))
MOULD_SPLIT_REARM_BASELINE_S = float(os.getenv('HICON_MOULD_SPLIT_REARM_BASELINE_S', '0.5'))
MOULD_SPLIT_DOM_RATIO = float(os.getenv('HICON_MOULD_SPLIT_DOM_RATIO', '1.35'))
MOULD_AXIS_ONLY_MIN_MAG = float(os.getenv('HICON_MOULD_AXIS_ONLY_MIN_MAG', '0.05'))
MOULD_SPLIT_REARM_DX_PX = float(os.getenv('HICON_MOULD_SPLIT_REARM_DX_PX', '10.0'))
MOULD_SPLIT_REARM_DY_PX = float(os.getenv('HICON_MOULD_SPLIT_REARM_DY_PX', '14.0'))
LOG_MOULD_DISPLACEMENT = os.getenv('HICON_LOG_MOULD_DISPLACEMENT', 'false').lower() == 'true'
MOULD_DISP_LOG_INTERVAL_S = float(os.getenv('HICON_MOULD_DISP_LOG_INTERVAL_S', '0.25'))

# Trolley bbox expansion for mouth-inside check (top edge only, ladle above trolley)
EDGE_EXPAND_PX = int(os.getenv('HICON_EDGE_EXPAND_PX', '200'))

# Mouth absence tolerance during active session (before session-end timer starts)
MOUTH_MISSING_TOL_S = float(os.getenv('HICON_MOUTH_MISSING_TOL_S', '0.8'))

# Multiple brightness probe offsets: (dx, dy) from mouth bottom-center
POUR_PROBE_OFFSETS = [(20, 0), (30, 0), (40, 0)]
POUR_PROBE_BELOW_PX = int(os.getenv('HICON_POUR_PROBE_BELOW_PX', '50'))

# Pouring cycle timeout: mouth absent from locked trolley region (5 min)
POURING_CYCLE_TIMEOUT_S = float(os.getenv('HICON_POURING_CYCLE_TIMEOUT', '300.0'))

# Mould switch requires last pour >= this duration
MOULD_SWITCH_MIN_POUR_S = float(os.getenv('HICON_MOULD_SWITCH_MIN_POUR', '2.0'))

# Minimum cumulative pour time per mould cluster (offline parity filter)
MIN_CLUSTER_POUR_S = float(os.getenv('HICON_MIN_CLUSTER_POUR_S', '1.5'))

# Enable per-frame CPU extraction for processors
ENABLE_FRAME_PROCESSING = os.getenv('HICON_ENABLE_FRAME_PROCESSING', 'true').lower() == 'true'

# Annotated inference video (DS-native: tee + nvosd + RecordingManager)
ENABLE_INFERENCE_VIDEO = os.getenv('HICON_ENABLE_INFERENCE_VIDEO', 'false').lower() == 'true'
INFERENCE_VIDEO_FPS = float(os.getenv('HICON_INFERENCE_VIDEO_FPS', '10'))
INFERENCE_VIDEO_WIDTH = int(os.getenv('HICON_INFERENCE_VIDEO_WIDTH', '640'))
INFERENCE_VIDEO_HEIGHT = int(os.getenv('HICON_INFERENCE_VIDEO_HEIGHT', '360'))

# Recording schedule: 'always' for 24/7, or time windows like '06:00-08:00,18:00-20:00'
# Format: comma-separated HH:MM-HH:MM pairs (24h clock, local time)
# Examples: 'always', '06:00-08:00', '06:00-08:00,18:00-20:00'
INFERENCE_VIDEO_SCHEDULE = os.getenv('HICON_INFERENCE_VIDEO_SCHEDULE', 'always')
# Max duration per recording file in seconds (0 = no rotation)
INFERENCE_VIDEO_MAX_DURATION_S = int(os.getenv('HICON_INFERENCE_VIDEO_MAX_DURATION_S', '3600'))
# Retention: auto-delete recordings older than N days (0 = keep forever)
INFERENCE_VIDEO_RETENTION_DAYS = int(os.getenv('HICON_INFERENCE_VIDEO_RETENTION_DAYS', '3'))

# =============================================================================
# RTSP STREAMS (3-stream HiCon pipeline)
# =============================================================================

# Stream 0: Process camera (tapping, pouring, deslagging, spectro)
# Stream 1: Pyrometer camera (rod insertion detection)
# Stream 2: Second pouring camera (pouring detection only)
ENABLE_RTSP_STREAM_0 = os.getenv('HICON_ENABLE_RTSP_STREAM_0', 'true').lower() == 'true'
ENABLE_RTSP_STREAM_1 = os.getenv('HICON_ENABLE_RTSP_STREAM_1', 'true').lower() == 'true'
ENABLE_RTSP_STREAM_2 = os.getenv('HICON_ENABLE_RTSP_STREAM_2', 'true').lower() == 'true'

RTSP_STREAM_0 = os.getenv('HICON_RTSP_STREAM_0', '') if ENABLE_RTSP_STREAM_0 else ''
RTSP_STREAM_1 = os.getenv('HICON_RTSP_STREAM_1', '') if ENABLE_RTSP_STREAM_1 else ''
RTSP_STREAM_2 = os.getenv('HICON_RTSP_STREAM_2', '') if ENABLE_RTSP_STREAM_2 else ''

# Per-stream codec selection: 'h265' (default for all cameras) or 'h264'
RTSP_CODEC_0 = os.getenv('HICON_RTSP_CODEC_0', 'h265')
RTSP_CODEC_1 = os.getenv('HICON_RTSP_CODEC_1', 'h265')
RTSP_CODEC_2 = os.getenv('HICON_RTSP_CODEC_2', 'h265')

# Per-stream transport preference
RTSP_PROTOCOL_0 = _get_rtsp_protocol('HICON_RTSP_PROTOCOL_0', 'tcp')
RTSP_PROTOCOL_1 = _get_rtsp_protocol('HICON_RTSP_PROTOCOL_1', 'auto')
RTSP_PROTOCOL_2 = _get_rtsp_protocol('HICON_RTSP_PROTOCOL_2', 'auto')

# RTSP transport timeouts (microseconds, 0 disables)
RTSP_TCP_TIMEOUT_US = int(os.getenv('HICON_RTSP_TCP_TIMEOUT_US', '60000000'))
RTSP_UDP_TIMEOUT_US = int(os.getenv('HICON_RTSP_UDP_TIMEOUT_US', '0'))

_legacy_rtsp_timeout_sec = os.getenv('HICON_RTSP_TIMEOUT_SEC')
if _legacy_rtsp_timeout_sec is not None:
    logger.warning(
        "HICON_RTSP_TIMEOUT_SEC is obsolete and ignored; "
        "use HICON_RTSP_UDP_TIMEOUT_US instead because rtspsrc.timeout is in microseconds."
    )

# RTSP reconnection behavior (rtspsrc properties)
RTSP_PORT_RETRY = int(os.getenv('HICON_RTSP_PORT_RETRY', '20'))
RTSP_DO_RETRANSMISSION = os.getenv('HICON_RTSP_DO_RETRANSMISSION', 'true').lower() == 'true'
RTSP_LATENCY_MS = int(os.getenv('HICON_RTSP_LATENCY_MS', '2000'))

# Stream-level auto-restart thresholds (seconds)
RTSP_RESTART_STALE_SEC = int(os.getenv('HICON_RTSP_RESTART_STALE_SEC', '90'))
RTSP_RESTART_COOLDOWN_SEC = int(os.getenv('HICON_RTSP_RESTART_COOLDOWN_SEC', '60'))
RTSP_RESTART_BACKOFF_SEC = int(os.getenv('HICON_RTSP_RESTART_BACKOFF_SEC', '5'))

# Per-stream 0fps watchdog policy: 'restart' (kill pipeline) or 'warn' (log, wait for recovery)
STREAM_0_ZERO_FPS_POLICY = os.getenv('HICON_STREAM_0_ZERO_FPS_POLICY', 'warn')
STREAM_1_ZERO_FPS_POLICY = os.getenv('HICON_STREAM_1_ZERO_FPS_POLICY', 'restart')

# Use nvurisrcbin (with built-in RTSP reconnection) instead of rtspsrc for Stream 0
USE_NVURISRCBIN_0 = os.getenv('HICON_USE_NVURISRCBIN_0', 'true').lower() == 'true'

# Use ffmpeg subprocess as RTSP-to-pipe bridge (zero-drop, handles keepalives correctly).
# ffmpeg -c:v copy remuxes only (~1% CPU); GStreamer fdsrc reads from pipe, NVDEC decodes.
# Takes priority over nvurisrcbin/rtspsrc when enabled.
USE_FFMPEG_SRC_0 = os.getenv('HICON_USE_FFMPEG_SRC_0', 'false').lower() == 'true'
USE_FFMPEG_SRC_2 = os.getenv('HICON_USE_FFMPEG_SRC_2', 'false').lower() == 'true'

# Use delayed tmpfs segment buffer for Stream 0.
# A helper process records the direct RTSP stream into MPEG-TS segments under /dev/shm
# and feeds DeepStream from a FIFO with a fixed playback delay.
USE_SEGMENT_BUFFER_0 = os.getenv('HICON_USE_SEGMENT_BUFFER_0', 'false').lower() == 'true'
SEGMENT_BUFFER_DIR_0 = os.getenv(
    'HICON_SEGMENT_BUFFER_DIR_0',
    '/dev/shm/hicon/stream0-buffer',
)
SEGMENT_BUFFER_SEGMENT_SEC_0 = int(os.getenv('HICON_SEGMENT_BUFFER_SEGMENT_SEC_0', '2'))
SEGMENT_BUFFER_DELAY_SEC_0 = int(os.getenv('HICON_SEGMENT_BUFFER_DELAY_SEC_0', '60'))
SEGMENT_BUFFER_RETENTION_SEC_0 = int(os.getenv('HICON_SEGMENT_BUFFER_RETENTION_SEC_0', '120'))

USE_SEGMENT_BUFFER_2 = os.getenv('HICON_USE_SEGMENT_BUFFER_2', 'false').lower() == 'true'
SEGMENT_BUFFER_DIR_2 = os.getenv(
    'HICON_SEGMENT_BUFFER_DIR_2',
    '/dev/shm/hicon/stream2-buffer',
)
SEGMENT_BUFFER_SEGMENT_SEC_2 = int(os.getenv('HICON_SEGMENT_BUFFER_SEGMENT_SEC_2', '2'))
SEGMENT_BUFFER_DELAY_SEC_2 = int(os.getenv('HICON_SEGMENT_BUFFER_DELAY_SEC_2', '120'))
SEGMENT_BUFFER_RETENTION_SEC_2 = int(os.getenv('HICON_SEGMENT_BUFFER_RETENTION_SEC_2', '180'))

# Use UDP loopback bridge instead of pipe bridge for CP Plus cameras.
# ffmpeg reads camera via TCP, sends MPEGTS to udp://127.0.0.1:PORT.
# GStreamer udpsrc reads from localhost UDP — non-blocking send decouples ffmpeg's
# TCP read loop from pipeline backpressure, eliminating the ~5-min session timeout.
# Takes priority over fdsrc pipe when use_ffmpeg_src is also enabled.
USE_UDP_LOOPBACK_0 = os.getenv('HICON_USE_UDP_LOOPBACK_0', 'false').lower() == 'true'
USE_UDP_LOOPBACK_2 = os.getenv('HICON_USE_UDP_LOOPBACK_2', 'false').lower() == 'true'
UDP_LOOPBACK_PORT_0 = int(os.getenv('HICON_UDP_LOOPBACK_PORT_0', '5000'))
UDP_LOOPBACK_PORT_2 = int(os.getenv('HICON_UDP_LOOPBACK_PORT_2', '5002'))

# Healthchecks.io heartbeat URL (empty = disabled)
# Ping every 60s from watchdog; /fail on fatal error
HEALTHCHECK_URL = os.getenv('HICON_HEALTHCHECK_URL', '')

# Debug/diagnostic flags
BYPASS_STREAM_0_INFER = os.getenv('HICON_BYPASS_STREAM_0_INFER', 'false').lower() == 'true'
BYPASS_STREAM_1_INFER = os.getenv('HICON_BYPASS_STREAM_1_INFER', 'false').lower() == 'true'
BYPASS_STREAM_2_INFER = os.getenv('HICON_BYPASS_STREAM_2_INFER', 'false').lower() == 'true'
STREAM_0_BYPASS_TRACKER = os.getenv('HICON_BYPASS_STREAM_0_TRACKER', 'false').lower() == 'true'
STREAM_0_BYPASS_PGIE = os.getenv('HICON_BYPASS_STREAM_0_PGIE', 'false').lower() == 'true'
STREAM_0_DECODE_ONLY_MODE = os.getenv('HICON_STREAM_0_DECODE_ONLY_MODE', 'false').lower() == 'true'
STREAM_0_POSTMUX_ONLY_MODE = os.getenv('HICON_STREAM_0_POSTMUX_ONLY_MODE', 'false').lower() == 'true'
STREAM_0_POSTCONV_ONLY_MODE = os.getenv('HICON_STREAM_0_POSTCONV_ONLY_MODE', 'false').lower() == 'true'
STREAM_0_PREOSD_ONLY_MODE = os.getenv('HICON_STREAM_0_PREOSD_ONLY_MODE', 'false').lower() == 'true'
STREAM_0_DECOUPLED_ANALYSIS_MODE = os.getenv(
    'HICON_STREAM_0_DECOUPLED_ANALYSIS_MODE', 'false'
).lower() == 'true'
ENABLE_STREAM_0_POURING_PROCESSOR = os.getenv('HICON_ENABLE_STREAM_0_POURING_PROCESSOR', 'true').lower() == 'true'
ENABLE_STREAM_0_BRIGHTNESS_PROCESSOR = os.getenv('HICON_ENABLE_STREAM_0_BRIGHTNESS_PROCESSOR', 'true').lower() == 'true'
ENABLE_DEBUG_PROBES = os.getenv('HICON_ENABLE_DEBUG_PROBES', 'false').lower() == 'true'
LOG_SOURCE_IDS = os.getenv('HICON_LOG_SOURCE_IDS', 'true').lower() == 'true'
ENABLE_STREAM_0_PROBE = os.getenv('HICON_ENABLE_STREAM_0_PROBE', 'true').lower() == 'true'
ENABLE_STREAM_1_PROBE = os.getenv('HICON_ENABLE_STREAM_1_PROBE', 'true').lower() == 'true'
ENABLE_STREAM_2_PROBE = os.getenv('HICON_ENABLE_STREAM_2_PROBE', 'true').lower() == 'true'
ENABLE_BRIGHTNESS_STREAM_2 = False  # Stream 2 is pouring-only; no brightness checks

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

BASE_DIR = Path(os.getenv('HICON_BASE_DIR', '/home/hicon/hicon/ai_vision'))
CONFIG_DIR = Path(os.getenv('HICON_CONFIG_DIR', str(BASE_DIR / 'configs')))
DATA_DIR = Path(os.getenv('HICON_DATA_DIR', str(BASE_DIR / 'data')))
LOG_DIR = Path(os.getenv('HICON_LOG_DIR', str(BASE_DIR / 'logs')))
SCREENSHOT_DIR = Path(os.getenv('HICON_SCREENSHOT_DIR', str(BASE_DIR / 'output/screenshots')))
VIDEO_DIR = Path(os.getenv('HICON_VIDEO_DIR', str(BASE_DIR / 'output/videos')))

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# DeepStream nvinfer config files (HiCon 3-stream pipeline)
CONFIG_POURING = str(CONFIG_DIR / 'config_pouring_pgie.txt')
CONFIG_PYROMETER = str(CONFIG_DIR / 'config_pyrometer_pgie.txt')
CONFIG_POURING_2 = str(CONFIG_DIR / 'config_pouring2_pgie.txt')  # GIE-3: Stream 2 pouring

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

DB_PATH = DATA_DIR / 'hicon.db'

# =============================================================================
# TRACKER CONFIGURATION
# =============================================================================

TRACKER_LIB = os.getenv(
    'HICON_TRACKER_LIB',
    '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so'
)

TRACKER_CONFIG = os.getenv(
    'HICON_TRACKER_CONFIG',
    str(CONFIG_DIR / 'config_tracker.yml')
)

# =============================================================================
# FEATURE FLAGS
# =============================================================================

ENABLE_SYNC = os.getenv('HICON_ENABLE_SYNC', 'true').lower() == 'true'

# =============================================================================
# LIVE STREAMING CONFIGURATION
# =============================================================================

ENABLE_LIVE_STREAM = os.getenv('HICON_ENABLE_LIVE_STREAM', 'false').lower() == 'true'
LIVE_STREAM_HOST = os.getenv('HICON_LIVE_STREAM_HOST', '0.0.0.0')
LIVE_STREAM_PORT = int(os.getenv('HICON_LIVE_STREAM_PORT', '8080'))
LIVE_STREAM_QUALITY = int(os.getenv('HICON_LIVE_STREAM_QUALITY', '85'))  # JPEG quality 0-100
LIVE_STREAM_FPS = int(os.getenv('HICON_LIVE_STREAM_FPS', '15'))  # Max FPS for stream

# Per-stream live streaming control (default: follow ENABLE_LIVE_STREAM)
ENABLE_LIVE_STREAM_0 = os.getenv('HICON_ENABLE_LIVE_STREAM_0', str(ENABLE_LIVE_STREAM)).lower() == 'false'
ENABLE_LIVE_STREAM_1 = os.getenv('HICON_ENABLE_LIVE_STREAM_1', str(ENABLE_LIVE_STREAM)).lower() == 'false'


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config():
    """Validate configuration values on import"""
    if not API_URL:
        raise ValueError("API_URL cannot be empty")

    if ENABLE_SYNC and not HMAC_SECRET:
        raise ValueError(
            "HMAC_SECRET is required when ENABLE_SYNC=true.\n"
            "Set environment variable: export HICON_HMAC_SECRET='your-secret-key'\n"
            "Or disable sync: export HICON_ENABLE_SYNC=false"
        )

    if not CUSTOMER_ID:
        raise ValueError("CUSTOMER_ID cannot be empty")

    if SYNC_INTERVAL < 1:
        raise ValueError("SYNC_INTERVAL must be at least 1 second")

    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be at least 1")

    if RTSP_TCP_TIMEOUT_US < 0:
        raise ValueError("RTSP_TCP_TIMEOUT_US must be >= 0")

    if RTSP_UDP_TIMEOUT_US < 0:
        raise ValueError("RTSP_UDP_TIMEOUT_US must be >= 0")

    if RTSP_PORT_RETRY < 0:
        raise ValueError("RTSP_PORT_RETRY must be >= 0")

    if SEGMENT_BUFFER_SEGMENT_SEC_0 < 1:
        raise ValueError("SEGMENT_BUFFER_SEGMENT_SEC_0 must be >= 1")

    if SEGMENT_BUFFER_DELAY_SEC_0 < SEGMENT_BUFFER_SEGMENT_SEC_0:
        raise ValueError("SEGMENT_BUFFER_DELAY_SEC_0 must be >= SEGMENT_BUFFER_SEGMENT_SEC_0")

    if SEGMENT_BUFFER_RETENTION_SEC_0 < SEGMENT_BUFFER_DELAY_SEC_0:
        raise ValueError("SEGMENT_BUFFER_RETENTION_SEC_0 must be >= SEGMENT_BUFFER_DELAY_SEC_0")

    if SEGMENT_BUFFER_SEGMENT_SEC_2 < 1:
        raise ValueError("SEGMENT_BUFFER_SEGMENT_SEC_2 must be >= 1")

    if SEGMENT_BUFFER_DELAY_SEC_2 < SEGMENT_BUFFER_SEGMENT_SEC_2:
        raise ValueError("SEGMENT_BUFFER_DELAY_SEC_2 must be >= SEGMENT_BUFFER_SEGMENT_SEC_2")

    if SEGMENT_BUFFER_RETENTION_SEC_2 < SEGMENT_BUFFER_DELAY_SEC_2:
        raise ValueError("SEGMENT_BUFFER_RETENTION_SEC_2 must be >= SEGMENT_BUFFER_DELAY_SEC_2")


# Validate configuration on import
validate_config()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_config_summary():
    """Get a summary of current configuration for logging"""
    return {
        'api_url': API_URL,
        'customer_id': CUSTOMER_ID,
        'device_id': DEVICE_ID,
        'location': LOCATION,
        'furnace_id': FURNACE_ID,
        'sync_interval': SYNC_INTERVAL,
        'batch_size': BATCH_SIZE,
        'enable_sync': ENABLE_SYNC,
        'hmac_configured': bool(HMAC_SECRET),
        'rtsp_stream_0': RTSP_STREAM_0,
        'rtsp_stream_1': RTSP_STREAM_1,
        'rtsp_protocol_0': RTSP_PROTOCOL_0,
        'rtsp_protocol_1': RTSP_PROTOCOL_1,
        'rtsp_protocol_2': RTSP_PROTOCOL_2,
        'rtsp_udp_timeout_us': RTSP_UDP_TIMEOUT_US,
        'rtsp_tcp_timeout_us': RTSP_TCP_TIMEOUT_US,
        'rtsp_port_retry': RTSP_PORT_RETRY,
        'rtsp_do_retransmission': RTSP_DO_RETRANSMISSION,
        'stream_0_bypass_tracker': STREAM_0_BYPASS_TRACKER,
        'stream_0_bypass_pgie': STREAM_0_BYPASS_PGIE,
        'stream_0_decode_only_mode': STREAM_0_DECODE_ONLY_MODE,
        'stream_0_postmux_only_mode': STREAM_0_POSTMUX_ONLY_MODE,
        'stream_0_postconv_only_mode': STREAM_0_POSTCONV_ONLY_MODE,
        'stream_0_preosd_only_mode': STREAM_0_PREOSD_ONLY_MODE,
        'stream_0_decoupled_analysis_mode': STREAM_0_DECOUPLED_ANALYSIS_MODE,
        'enable_stream_0_pouring_processor': ENABLE_STREAM_0_POURING_PROCESSOR,
        'enable_stream_0_brightness_processor': ENABLE_STREAM_0_BRIGHTNESS_PROCESSOR,
        'base_dir': str(BASE_DIR),
        'config_dir': str(CONFIG_DIR),
        'data_dir': str(DATA_DIR),
        'db_path': str(DB_PATH),
        'mouth_confidence': MOUTH_CONFIDENCE,
        'trolley_confidence': TROLLEY_CONFIDENCE,
        'session_start_duration': SESSION_START_DURATION,
        'session_end_duration': SESSION_END_DURATION,
        'pour_brightness_start': POUR_BRIGHTNESS_START,
        'pour_brightness_end': POUR_BRIGHTNESS_END,
        'pour_min_duration': POUR_MIN_DURATION,
        'edge_expand_px': EDGE_EXPAND_PX,
        'mouth_missing_tol_s': MOUTH_MISSING_TOL_S,
        'pouring_cycle_timeout_s': POURING_CYCLE_TIMEOUT_S,
        'mould_switch_min_pour_s': MOULD_SWITCH_MIN_POUR_S,
        'min_cluster_pour_s': MIN_CLUSTER_POUR_S,
        'mould_split_min_dx_px': MOULD_SPLIT_MIN_DX_PX,
        'mould_split_min_dy_px': MOULD_SPLIT_MIN_DY_PX,
        'mould_split_cooldown_s': MOULD_SPLIT_COOLDOWN_S,
        'mould_split_rearm_baseline_s': MOULD_SPLIT_REARM_BASELINE_S,
        'mould_split_dom_ratio': MOULD_SPLIT_DOM_RATIO,
        'mould_axis_only_min_mag': MOULD_AXIS_ONLY_MIN_MAG,
        'mould_split_rearm_dx_px': MOULD_SPLIT_REARM_DX_PX,
        'mould_split_rearm_dy_px': MOULD_SPLIT_REARM_DY_PX,
        'log_mould_displacement': LOG_MOULD_DISPLACEMENT,
        'mould_disp_log_interval_s': MOULD_DISP_LOG_INTERVAL_S,
        'enable_inference_video': ENABLE_INFERENCE_VIDEO,
        'inference_video_fps': INFERENCE_VIDEO_FPS,
        'inference_video_width': INFERENCE_VIDEO_WIDTH,
        'inference_video_height': INFERENCE_VIDEO_HEIGHT,
        'inference_video_schedule': INFERENCE_VIDEO_SCHEDULE,
        'inference_video_max_duration_s': INFERENCE_VIDEO_MAX_DURATION_S,
        'inference_video_retention_days': INFERENCE_VIDEO_RETENTION_DAYS,
        'use_segment_buffer_0': USE_SEGMENT_BUFFER_0,
        'segment_buffer_dir_0': SEGMENT_BUFFER_DIR_0,
        'segment_buffer_segment_sec_0': SEGMENT_BUFFER_SEGMENT_SEC_0,
        'segment_buffer_delay_sec_0': SEGMENT_BUFFER_DELAY_SEC_0,
        'segment_buffer_retention_sec_0': SEGMENT_BUFFER_RETENTION_SEC_0,
        'use_segment_buffer_2': USE_SEGMENT_BUFFER_2,
        'segment_buffer_dir_2': SEGMENT_BUFFER_DIR_2,
        'segment_buffer_segment_sec_2': SEGMENT_BUFFER_SEGMENT_SEC_2,
        'segment_buffer_delay_sec_2': SEGMENT_BUFFER_DELAY_SEC_2,
        'segment_buffer_retention_sec_2': SEGMENT_BUFFER_RETENTION_SEC_2,
        'use_ffmpeg_src_0': USE_FFMPEG_SRC_0,
        'use_ffmpeg_src_2': USE_FFMPEG_SRC_2,
        'use_udp_loopback_0': USE_UDP_LOOPBACK_0,
        'use_udp_loopback_2': USE_UDP_LOOPBACK_2,
        'udp_loopback_port_0': UDP_LOOPBACK_PORT_0,
        'udp_loopback_port_2': UDP_LOOPBACK_PORT_2,
    }
