"""
Sync Manager for HiCon - Periodic cloud synchronization
Handles batched syncing of local DB to cloud API with retry logic
"""
import time
import json
import logging
import base64
from pathlib import Path
from typing import Dict, List
from datetime import datetime, timedelta

from config import SCREENSHOT_DIR, SCREENSHOT_RETENTION_DAYS

logger = logging.getLogger(__name__)


def format_timestamp_for_api(iso_timestamp: str) -> str:
    """
    Convert ISO8601 timestamp to API format (YYYY-MM-DD HH:MM:SS).

    Args:
        iso_timestamp: ISO8601 format timestamp (e.g., "2025-12-26T14:06:01.885957")

    Returns:
        API format timestamp (e.g., "2025-12-26 14:06:01") or None if input is None
    """
    # Handle None/null timestamps (e.g., ongoing pouring events without end_time)
    if iso_timestamp is None:
        return None

    try:
        # Parse ISO8601 timestamp
        dt = datetime.fromisoformat(iso_timestamp)
        # Format as YYYY-MM-DD HH:MM:SS
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError, TypeError) as e:
        logger.warning(f"Failed to parse timestamp '{iso_timestamp}': {e}")
        return iso_timestamp  # Return as-is if parsing fails


class SyncManager:
    """
    Manage periodic synchronization of local data to cloud API.
    
    - Batches records for efficiency
    - Handles retries with exponential backoff
    - Uploads screenshots to S3 (base64 encoded)
    - Updates sync status in local DB
    """
    
    def __init__(self, database, api_client, customer_id: str,
                 camera_id: str, location: str, furnace_id: str = "",
                 sync_interval: int = 30, batch_size: int = 50):
        """
        Initialize sync manager.

        Args:
            database: HiConDatabase instance
            api_client: APIClient instance
            customer_id: Customer ID
            camera_id: Camera identifier (e.g., "IPCamera2")
            location: Location identifier (e.g., "Melting Section")
            sync_interval: Seconds between sync attempts
            batch_size: Max records per sync batch
        """
        self.db = database
        self.api = api_client
        self.customer_id = customer_id
        self.camera_id = camera_id
        self.location = location
        self.furnace_id = furnace_id
        self.sync_interval = sync_interval
        self.batch_size = batch_size
        
        self.last_sync_time = 0
        self.last_cleanup_time = 0
        self.cleanup_interval = 86400  # 24 hours
        
        # Statistics
        self.stats = {
            'total_melting_synced': 0,
            'total_heat_cycles_synced': 0,
            'total_pouring_synced': 0,  # Backward-compatible alias for heat-cycle sync count
            'total_sync_attempts': 0,
            'total_sync_failures': 0,
            'last_sync_success': None,
            'last_sync_failure': None
        }
        
        logger.info(f"✓ SyncManager initialized - Interval: {sync_interval}s, Batch: {batch_size}")
    
    def should_sync(self, current_time: float) -> bool:
        """Check if sync should run"""
        return current_time - self.last_sync_time >= self.sync_interval
    
    def should_cleanup(self, current_time: float) -> bool:
        """Check if cleanup should run"""
        return current_time - self.last_cleanup_time >= self.cleanup_interval
    
    def sync_all(self):
        """Sync all pending data to cloud"""
        current_time = time.time()
        
        if not self.should_sync(current_time):
            return
        
        self.last_sync_time = current_time
        self.stats['total_sync_attempts'] += 1
        
        try:
            # Sync heat cycles (aggregated pouring + melting cycle payloads)
            melting_synced, pouring_synced, finalized_synced = self._sync_heat_cycles()

            # Update stats
            self.stats['total_melting_synced'] += melting_synced
            self.stats['total_heat_cycles_synced'] += finalized_synced
            self.stats['total_pouring_synced'] += pouring_synced
            self.stats['last_sync_success'] = datetime.now().isoformat()

            if melting_synced > 0 or pouring_synced > 0:
                logger.info(
                    "Sync complete - Melting: %s, Pouring: %s, Finalized cycles: %s",
                    melting_synced, pouring_synced, finalized_synced,
                )
            
        except Exception as e:
            self.stats['total_sync_failures'] += 1
            self.stats['last_sync_failure'] = datetime.now().isoformat()
            logger.error(f"✗ Sync failed: {e}", exc_info=True)
        
        # Periodic cleanup
        if self.should_cleanup(current_time):
            self._run_cleanup()
    
    @staticmethod
    def _format_duration_hhmmss(start_iso: str, end_iso: str) -> str:
        """Format duration between ISO timestamps as HH:MM:SS."""
        if not start_iso or not end_iso:
            return ""
        try:
            start_dt = datetime.fromisoformat(start_iso)
            end_dt = datetime.fromisoformat(end_iso)
            total_seconds = int(max(0, (end_dt - start_dt).total_seconds()))
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        except Exception:
            return ""
    
    def _sync_heat_cycles(self) -> tuple:
        """
        Sync pending heat cycles to both endpoints:
        - /pouring (mould-wise pouring payload)
        - /melting (melting cycle summary payload)
        """
        # Get unsynced heat cycles
        cycles = self.db.get_unsynced_heat_cycles(limit=self.batch_size)

        if not cycles:
            return 0, 0, 0

        # Prepare API payload
        pouring_items = []
        melting_items = []
        cycle_by_sync = {}
        for cycle in cycles:
            cycle_by_sync[cycle['sync_id']] = cycle
            # Parse JSON mould_wise_pouring_time array
            # Handle both single and double-encoded JSON (for backward compatibility)
            mould_wise_timing = []
            if cycle.get('mould_wise_pouring_time'):
                try:
                    # First parse
                    parsed = json.loads(cycle['mould_wise_pouring_time'])
                    # Check if it's still a string (double-encoded)
                    if isinstance(parsed, str):
                        mould_wise_timing = json.loads(parsed)  # Parse again
                    else:
                        mould_wise_timing = parsed  # Already a list
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse mould_wise_pouring_time for {cycle['heat_no']}: {e}")
                    mould_wise_timing = []

            def _parse_json_list(raw_value):
                if not raw_value:
                    return []
                try:
                    parsed = json.loads(raw_value)
                    return parsed if isinstance(parsed, list) else []
                except (json.JSONDecodeError, TypeError):
                    return []

            mould_count = len(mould_wise_timing) if mould_wise_timing else 0

            # Pouring payload (new API format)
            pouring_items.append({
                'sync_id': cycle['sync_id'],
                'customer_id': cycle['customer_id'],
                'date': cycle['date'],
                'heat_no': cycle.get('heat_no') or "",
                'location': cycle['location'],
                'camera_id': cycle['camera_id'],
                'mould_count': mould_count,
                'pouring_start_time': format_timestamp_for_api(cycle['pouring_start_time']),
                'pouring_end_time': format_timestamp_for_api(cycle.get('pouring_end_time')),
                'total_pouring_time': str(cycle.get('total_pouring_time', '0')),  # Seconds as string
                'mould_wise_pouring_time': mould_wise_timing,  # Array of {mould_id, start, end, duration}
            })

            # Melting payload (new API format)
            tapping_start = format_timestamp_for_api(cycle.get('tapping_start_time'))
            tapping_end = format_timestamp_for_api(cycle.get('tapping_end_time'))
            deslag_events = _parse_json_list(cycle.get('deslagging_events'))
            spectro_events = _parse_json_list(cycle.get('spectro_events'))
            pyro_events = _parse_json_list(cycle.get('pyrometer_events'))

            cycle_start_iso = cycle.get('cycle_start_time')
            cycle_end_iso = cycle.get('cycle_end_time')
            melting_items.append({
                'sync_id': cycle['sync_id'],
                'customer_id': cycle['customer_id'],
                'date': cycle['date'],
                'camera_id': cycle['camera_id'],
                'location': cycle['location'],
                'pyrometer': bool(len(pyro_events) > 0),
                'spectro': bool(len(spectro_events) > 0),
                'furnace': self.furnace_id or "",
                'heat_no': cycle.get('heat_no') or "",
                'heat_start_time': format_timestamp_for_api(cycle_start_iso),
                'heat_end_time': format_timestamp_for_api(cycle_end_iso),
                'heat_duration': self._format_duration_hhmmss(cycle_start_iso, cycle_end_iso),
                'tapping_start_time': tapping_start,
                'tapping_end_time': tapping_end,
                'deslagging': bool(len(deslag_events) > 0),
            })

        # Send to API
        pouring_result = self.api.send_pouring_data(pouring_items)
        melting_result = self.api.send_melting_data(melting_items)

        # Extract successful sync_ids from results array
        pouring_results = pouring_result.get('results', [])
        pouring_success = [
            r['sync_id'] for r in pouring_results
            if r.get('success', False)
        ]
        melting_results = melting_result.get('results', [])
        melting_success = [
            r['sync_id'] for r in melting_results
            if r.get('success', False)
        ]

        # Log any failures for debugging
        failed_ids = [
            (r['sync_id'], r.get('error', 'Unknown error'))
            for r in pouring_results
            if not r.get('success', False)
        ]

        if failed_ids:
            for sync_id, error in failed_ids:
                logger.warning(f"    Failed to sync heat cycle {sync_id}: {error}")

            # Update sync_attempts for failed cycles
            for sync_id, error_msg in failed_ids:
                self.db.update_heat_cycle_sync_status(sync_id, error_msg)

        failed_melting_ids = [
            (r['sync_id'], r.get('error', 'Unknown error'))
            for r in melting_results
            if not r.get('success', False)
        ]
        if failed_melting_ids:
            for sync_id, error in failed_melting_ids:
                logger.warning(f"    Failed to sync melting cycle {sync_id}: {error}")
            for sync_id, error_msg in failed_melting_ids:
                self.db.update_heat_cycle_sync_status(sync_id, error_msg)

        # Mark successful records as synced
        successful_ids = set(pouring_success) & set(melting_success)
        if successful_ids:
            self.db.mark_heat_cycles_synced(list(successful_ids))
            logger.info(f"  ✓ Marked {len(successful_ids)} heat cycles as synced")
            # Mark melting events within cycle window as synced (tapping/deslagging/spectro/pyrometer)
            for sync_id in successful_ids:
                cycle = cycle_by_sync.get(sync_id)
                if cycle:
                    self.db.mark_melting_events_synced_by_window(
                        cycle.get('cycle_start_time'),
                        cycle.get('cycle_end_time'),
                    )

        return len(melting_success), len(pouring_success), len(successful_ids)
    
    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 with compression.

        Applies JPEG quality reduction and resizing before Base64 encoding
        to reduce API payload size.

        Args:
            image_path: Path to screenshot file

        Returns:
            Base64-encoded compressed JPEG string, or None on failure
        """
        try:
            import cv2
            from config import SCREENSHOT_MAX_WIDTH, SCREENSHOT_JPEG_QUALITY

            path = Path(image_path)
            if not path.exists():
                logger.warning(f"Screenshot not found: {image_path}")
                return None

            # Read image
            img = cv2.imread(str(path))
            if img is None:
                logger.error(f"Failed to read image: {image_path}")
                return None

            # Resize to 50% of original dimensions for compression
            height, width = img.shape[:2]
            new_width = int(width * 0.5)
            new_height = int(height * 0.5)
            img = cv2.resize(
                img,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA  # Best for downscaling
            )
            logger.debug(f"Compressed screenshot from {width}×{height} to {new_width}×{new_height} (50%)")

            # Encode to JPEG with 50% quality (aggressive compression for API)
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 50]
            success, buffer = cv2.imencode('.jpg', img, encode_params)

            if not success:
                logger.error(f"Failed to encode image: {image_path}")
                return None

            # Convert to base64
            base64_string = base64.b64encode(buffer).decode('utf-8')

            # Log compression results
            original_size = path.stat().st_size / 1024  # KB
            compressed_size = len(buffer) / 1024  # KB
            base64_size = len(base64_string) / 1024  # KB
            reduction = ((original_size - compressed_size) / original_size) * 100

            logger.debug(
                f"Screenshot compression: {original_size:.1f}KB → {compressed_size:.1f}KB "
                f"({reduction:.0f}% reduction) → Base64: {base64_size:.1f}KB"
            )

            return base64_string

        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return None
    
    def _run_cleanup(self):
        """Run periodic cleanup tasks"""
        try:
            logger.info("Running screenshot cleanup...")
            self._cleanup_screenshots(days=SCREENSHOT_RETENTION_DAYS)
            self.last_cleanup_time = time.time()
            logger.info("✓ Cleanup complete")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}", exc_info=True)

    def _cleanup_screenshots(self, days: int = 7):
        """Delete screenshot files older than specified days (no DB row deletion)."""
        if days <= 0:
            logger.info("Screenshot cleanup disabled (retention <= 0)")
            return

        if not SCREENSHOT_DIR.exists():
            logger.info(f"Screenshot directory does not exist: {SCREENSHOT_DIR}")
            return

        cutoff = time.time() - (days * 86400)
        deleted = 0

        for path in SCREENSHOT_DIR.iterdir():
            if not path.is_file():
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    deleted += 1
            except OSError as exc:
                logger.warning(f"Failed to delete screenshot {path}: {exc}")

        if deleted:
            logger.info(f"✓ Deleted {deleted} screenshots older than {days} days")
    
    def get_stats(self) -> Dict:
        """Get sync statistics"""
        stats = self.stats.copy()
        db_stats = self.db.get_stats()
        stats['db_stats'] = db_stats
        return stats
    
    def force_sync(self):
        """Force immediate sync (ignores interval)"""
        logger.info("Forcing immediate sync...")
        self.last_sync_time = 0
        self.sync_all()
    
    def force_cleanup(self):
        """Force immediate cleanup"""
        logger.info("Forcing immediate cleanup...")
        self._run_cleanup()
