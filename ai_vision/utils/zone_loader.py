"""
Zone Configuration Loader

This module provides utilities for loading zone configurations from the unified
zones.json file and converting them to formats expected by the HiCon pipeline.

Key Features:
- Loads zones from JSON config file
- Converts point arrays to numpy int32 arrays
- Converts color lists to tuples for OpenCV
- Handles field name mapping for state machine compatibility
- Provides error handling for missing/invalid configurations

Usage:
    from zone_loader import get_ppe_zones, get_mould_zones, get_pan_zones, get_ladle_zones

    PPE_ZONES = get_ppe_zones('configs/zones.json')
    MOULD_ZONES = get_mould_zones('configs/zones.json')
    PAN_ZONES = get_pan_zones('configs/zones.json')
    LADLE_BASE_ZONES = get_ladle_zones('configs/zones.json')
"""

import json
import numpy as np
from typing import Dict, Tuple
from pathlib import Path


def load_zones_config(config_path: str) -> dict:
    """
    Load zone configuration from JSON file.

    Args:
        config_path: Path to zones.json configuration file

    Returns:
        Dictionary containing all zone configurations

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file contains invalid JSON
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Zone configuration file not found: {config_path}\n"
            f"Expected location: {config_file.absolute()}"
        )

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in zone configuration file {config_path}: {e}")

    return config


def get_ppe_zone(config_path: str) -> np.ndarray:
    """
    Get PPE zone as numpy array (legacy single-zone support).

    Args:
        config_path: Path to zones.json configuration file

    Returns:
        Numpy array of PPE zone polygon points (shape: Nx2, dtype: int32)

    Raises:
        KeyError: If ppe_zone not found in config
    """
    config = load_zones_config(config_path)

    if 'ppe_zone' not in config:
        raise KeyError(f"'ppe_zone' not found in configuration file {config_path}")

    ppe_zone_config = config['ppe_zone']

    if 'points' not in ppe_zone_config:
        raise KeyError(f"'points' field not found in ppe_zone configuration")

    # Convert points list to numpy array with int32 dtype
    ppe_zone = np.array(ppe_zone_config['points'], dtype=np.int32)

    return ppe_zone


def get_ppe_zones(config_path: str) -> Dict:
    """
    Get per-stream PPE zones dictionary for multi-stream PPE monitoring.

    Args:
        config_path: Path to zones.json configuration file

    Returns:
        Dictionary of PPE zones:
        {
            'stream_0': {
                'name': str,
                'camera_id': str,
                'points': np.array(..., dtype=int32),
                'color': tuple,
                'enabled': bool
            },
            'stream_1': {...},
            'stream_2': {...}
        }

    Raises:
        KeyError: If ppe_zones not found in config
    """
    config = load_zones_config(config_path)

    if 'ppe_zones' not in config:
        raise KeyError(f"'ppe_zones' not found in configuration file {config_path}")

    ppe_zones_config = config['ppe_zones']
    ppe_zones = {}

    for stream_key, zone_config in ppe_zones_config.items():
        if stream_key == 'description':
            continue  # Skip description field

        if 'points' not in zone_config:
            raise KeyError(f"'points' field not found in ppe_zones.{stream_key}")

        ppe_zones[stream_key] = {
            'name': zone_config.get('name', f'PPE Zone {stream_key}'),
            'camera_id': zone_config.get('camera_id', f'PPE_Camera_{stream_key}'),
            'points': np.array(zone_config['points'], dtype=np.int32),
            'enabled': zone_config.get('enabled', True)
        }

        # Convert color to tuple if present
        if 'color' in zone_config:
            ppe_zones[stream_key]['color'] = tuple(zone_config['color'])

    return ppe_zones


def get_pan_zones(config_path: str) -> Dict:
    """
    Get pan zones dictionary with proper formatting for HeatCycleTracker.

    Performs field name mapping:
    - 'confidence_threshold' → 'threshold' (required by HeatCycleTracker)
    - 'points' → numpy int32 array
    - 'color' → tuple (for OpenCV)

    Args:
        config_path: Path to zones.json configuration file

    Returns:
        Dictionary of pan zones:
        {
            'FURNACE_1': {
                'points': np.array(..., dtype=int32),
                'threshold': float,
                'color': tuple
            },
            ...
        }

    Raises:
        KeyError: If pan_zones not found in config
    """
    config = load_zones_config(config_path)

    if 'pan_zones' not in config:
        raise KeyError(f"'pan_zones' not found in configuration file {config_path}")

    pan_zones_config = config['pan_zones']
    pan_zones = {}

    for furnace_name, zone_config in pan_zones_config.items():
        # Convert points to numpy array
        if 'points' not in zone_config:
            raise KeyError(f"'points' field not found in pan_zones.{furnace_name}")

        # Rename confidence_threshold → threshold for state machine compatibility
        if 'confidence_threshold' not in zone_config:
            raise KeyError(f"'confidence_threshold' field not found in pan_zones.{furnace_name}")

        pan_zones[furnace_name] = {
            'points': np.array(zone_config['points'], dtype=np.int32),
            'threshold': zone_config['confidence_threshold'],  # Rename field
        }

        # Convert color to tuple if present
        if 'color' in zone_config:
            pan_zones[furnace_name]['color'] = tuple(zone_config['color'])

    return pan_zones


def get_ladle_zones(config_path: str) -> Dict:
    """
    Get ladle zones dictionary with proper formatting for TappingTracker.

    Performs field name mapping:
    - 'points' → numpy int32 array (base detection zone)
    - 'brightness_zone_points' → 'brightness_zone' (numpy int32 array)
    - 'color' → tuple (for OpenCV)

    Args:
        config_path: Path to zones.json configuration file

    Returns:
        Dictionary of ladle zones:
        {
            'FURNACE_1': {
                'points': np.array(..., dtype=int32),
                'brightness_zone': np.array(..., dtype=int32),
                'brightness_threshold': int,
                'max_ladle_height': int,
                'color': tuple
            },
            ...
        }

    Raises:
        KeyError: If ladle_zones not found in config
    """
    config = load_zones_config(config_path)

    if 'ladle_zones' not in config:
        raise KeyError(f"'ladle_zones' not found in configuration file {config_path}")

    ladle_zones_config = config['ladle_zones']
    ladle_zones = {}

    for furnace_name, zone_config in ladle_zones_config.items():
        # Validate required fields
        if 'points' not in zone_config:
            raise KeyError(f"'points' field not found in ladle_zones.{furnace_name}")
        if 'brightness_zone_points' not in zone_config:
            raise KeyError(f"'brightness_zone_points' field not found in ladle_zones.{furnace_name}")
        if 'brightness_threshold' not in zone_config:
            raise KeyError(f"'brightness_threshold' field not found in ladle_zones.{furnace_name}")
        if 'max_ladle_height' not in zone_config:
            raise KeyError(f"'max_ladle_height' field not found in ladle_zones.{furnace_name}")

        # Convert to format expected by TappingTracker
        ladle_zones[furnace_name] = {
            'points': np.array(zone_config['points'], dtype=np.int32),
            'brightness_zone': np.array(zone_config['brightness_zone_points'], dtype=np.int32),  # Rename field
            'brightness_threshold': zone_config['brightness_threshold'],
            'max_ladle_height': zone_config['max_ladle_height'],
        }

        # Convert color to tuple if present
        if 'color' in zone_config:
            ladle_zones[furnace_name]['color'] = tuple(zone_config['color'])

    return ladle_zones


def get_mould_zones(config_path: str) -> Dict:
    """
    Get mould zones dictionary for pouring tracking.

    Args:
        config_path: Path to zones.json configuration file

    Returns:
        Dictionary of mould zones:
        {
            'MOULD_1': {
                'name': str,
                'mould_id': str,
                'points': np.array(..., dtype=int32),
                'color': tuple,
                'enabled': bool
            },
            ...
        }

    Raises:
        KeyError: If mould_zones not found in config
    """
    config = load_zones_config(config_path)

    if 'mould_zones' not in config:
        raise KeyError(f"'mould_zones' not found in configuration file {config_path}")

    mould_zones_config = config['mould_zones']
    mould_zones = {}

    for mould_key, zone_config in mould_zones_config.items():
        if mould_key == 'description':
            continue  # Skip description field

        if 'points' not in zone_config:
            raise KeyError(f"'points' field not found in mould_zones.{mould_key}")

        mould_zones[mould_key] = {
            'name': zone_config.get('name', f'Mould {mould_key}'),
            'mould_id': zone_config.get('mould_id', mould_key),
            'points': np.array(zone_config['points'], dtype=np.int32),
            'enabled': zone_config.get('enabled', True)
        }

        # Convert color to tuple if present
        if 'color' in zone_config:
            mould_zones[mould_key]['color'] = tuple(zone_config['color'])

    return mould_zones


def get_deslagging_config(config_path: str) -> Tuple[Dict, float, float]:
    """
    Get deslagging zones configuration with detection parameters.

    Performs transformations:
    - Converts zone string keys to integers
    - Converts points to numpy int32 arrays
    - Extracts detection parameters

    Args:
        config_path: Path to zones.json configuration file

    Returns:
        Tuple of (zones_dict, duration_threshold, cooldown_duration):
        - zones_dict: {
              1: {
                  'name': str,
                  'furnace_id': str,
                  'points': np.array(..., dtype=int32),
                  'brightness_threshold': int
              },
              ...
          }
        - duration_threshold: float (seconds for continuous brightness)
        - cooldown_duration: float (seconds between events)

    Raises:
        KeyError: If deslagging_zones not found in config
    """
    config = load_zones_config(config_path)

    if 'deslagging_zones' not in config:
        raise KeyError(f"'deslagging_zones' not found in configuration file {config_path}")

    deslagging_config = config['deslagging_zones']

    # Extract detection parameters
    if 'detection_params' not in deslagging_config:
        raise KeyError(f"'detection_params' not found in deslagging_zones configuration")

    params = deslagging_config['detection_params']
    duration_threshold = params.get('duration_threshold', 1.0)
    cooldown_duration = params.get('cooldown_duration', 10.0)

    # Extract and convert zones
    if 'zones' not in deslagging_config:
        raise KeyError(f"'zones' not found in deslagging_zones configuration")

    zones_raw = deslagging_config['zones']
    deslagging_zones = {}

    for zone_id_str, zone_config in zones_raw.items():
        # Convert string key to integer
        zone_id = int(zone_id_str)

        if 'points' not in zone_config:
            raise KeyError(f"'points' field not found in deslagging_zones.zones.{zone_id_str}")

        # Convert points to numpy array and preserve other fields
        deslagging_zones[zone_id] = {
            **zone_config,  # Copy all fields
            'points': np.array(zone_config['points'], dtype=np.int32)  # Override points with numpy array
        }

    return deslagging_zones, duration_threshold, cooldown_duration
