"""
HiCon Edge AI Configuration
Centralized configuration with environment variable support
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from ai_vision directory
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"Loaded environment from: {env_path}")
else:
    print(f"No .env file found at: {env_path} (using system environment variables)")


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
CLUSTER_R_MERGE = float(os.getenv('HICON_CLUSTER_R_MERGE', '0.05'))
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

# =============================================================================
# RTSP STREAMS (2-stream HiCon pipeline)
# =============================================================================

# Stream 0: Process camera (tapping, pouring, deslagging, spectro)
# Stream 1: Pyrometer camera (rod insertion detection)
ENABLE_RTSP_STREAM_0 = os.getenv('HICON_ENABLE_RTSP_STREAM_0', 'true').lower() == 'true'
ENABLE_RTSP_STREAM_1 = os.getenv('HICON_ENABLE_RTSP_STREAM_1', 'true').lower() == 'true'

RTSP_STREAM_0 = os.getenv('HICON_RTSP_STREAM_0', 'rtsp://100.78.173.43:8554/mystream') if ENABLE_RTSP_STREAM_0 else ''
RTSP_STREAM_1 = os.getenv('HICON_RTSP_STREAM_1', 'rtsp://100.78.173.43:8554/mystream1') if ENABLE_RTSP_STREAM_1 else ''

# RTSP connection timeout (microseconds, 0 disables)
RTSP_TCP_TIMEOUT_US = int(os.getenv('HICON_RTSP_TCP_TIMEOUT_US', '60000000'))

# RTSP reconnection behavior (rtspsrc properties)
RTSP_RETRY = int(os.getenv('HICON_RTSP_RETRY', '65535'))
RTSP_TIMEOUT_SEC = int(os.getenv('HICON_RTSP_TIMEOUT_SEC', '20'))
RTSP_DO_RETRANSMISSION = os.getenv('HICON_RTSP_DO_RETRANSMISSION', 'true').lower() == 'true'

# Stream-level auto-restart thresholds (seconds)
RTSP_RESTART_STALE_SEC = int(os.getenv('HICON_RTSP_RESTART_STALE_SEC', '90'))
RTSP_RESTART_COOLDOWN_SEC = int(os.getenv('HICON_RTSP_RESTART_COOLDOWN_SEC', '60'))
RTSP_RESTART_BACKOFF_SEC = int(os.getenv('HICON_RTSP_RESTART_BACKOFF_SEC', '5'))

# Debug/diagnostic flags
BYPASS_STREAM_0_INFER = os.getenv('HICON_BYPASS_STREAM_0_INFER', 'false').lower() == 'true'
BYPASS_STREAM_1_INFER = os.getenv('HICON_BYPASS_STREAM_1_INFER', 'false').lower() == 'true'
ENABLE_DEBUG_PROBES = os.getenv('HICON_ENABLE_DEBUG_PROBES', 'true').lower() == 'true'
LOG_SOURCE_IDS = os.getenv('HICON_LOG_SOURCE_IDS', 'true').lower() == 'true'
ENABLE_STREAM_0_PROBE = os.getenv('HICON_ENABLE_STREAM_0_PROBE', 'true').lower() == 'true'
ENABLE_STREAM_1_PROBE = os.getenv('HICON_ENABLE_STREAM_1_PROBE', 'true').lower() == 'true'

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

# DeepStream nvinfer config files (HiCon 2-model pipeline)
CONFIG_POURING = str(CONFIG_DIR / 'config_pouring_pgie.txt')
CONFIG_PYROMETER = str(CONFIG_DIR / 'config_pyrometer_pgie.txt')

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
        'log_mould_displacement': LOG_MOULD_DISPLACEMENT,
        'mould_disp_log_interval_s': MOULD_DISP_LOG_INTERVAL_S,
        'enable_inference_video': ENABLE_INFERENCE_VIDEO,
        'inference_video_fps': INFERENCE_VIDEO_FPS,
        'inference_video_width': INFERENCE_VIDEO_WIDTH,
        'inference_video_height': INFERENCE_VIDEO_HEIGHT,
    }
