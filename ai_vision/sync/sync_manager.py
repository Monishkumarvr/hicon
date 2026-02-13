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
                 camera_id: str, location: str,
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
            # Sync melting events (tapping, deslagging, pyrometer, spectro)
            melting_synced = self._sync_melting_events()

            # Sync heat cycles (aggregated pouring data)
            heat_cycles_synced = self._sync_heat_cycles()

            # Update stats
            self.stats['total_melting_synced'] += melting_synced
            self.stats['total_heat_cycles_synced'] += heat_cycles_synced
            self.stats['total_pouring_synced'] += heat_cycles_synced
            self.stats['last_sync_success'] = datetime.now().isoformat()

            if melting_synced > 0 or heat_cycles_synced > 0:
                logger.info(f"Sync complete - Melting: {melting_synced}, Heat Cycles: {heat_cycles_synced}")
            
        except Exception as e:
            self.stats['total_sync_failures'] += 1
            self.stats['last_sync_failure'] = datetime.now().isoformat()
            logger.error(f"✗ Sync failed: {e}", exc_info=True)
        
        # Periodic cleanup
        if self.should_cleanup(current_time):
            self._run_cleanup()
    
    def _sync_melting_events(self) -> int:
        """Sync pending melting events (tapping, deslagging, pyrometer, spectro)."""
        records = self.db.get_unsynced_melting_events(limit=self.batch_size)

        if not records:
            return 0

        items = []
        for record in records:
            item = {
                'sync_id': record['sync_id'],
                'customer_id': record['customer_id'],
                'slno': record['slno'],
                'date': record['date'],
                'event_type': record['event_type'],
                'start_time': format_timestamp_for_api(record['start_time']),
                'end_time': format_timestamp_for_api(record.get('end_time')),
                'duration_sec': record.get('duration_sec', 0),
                'camera_id': record['camera_id'],
                'location': record['location'],
            }

            # Encode screenshot if present
            screenshot_path = record.get('screenshot_path', '')
            if screenshot_path:
                screenshot_b64 = self._encode_image(screenshot_path)
                if screenshot_b64:
                    item['screenshot'] = screenshot_b64

            items.append(item)

        if not items:
            return 0

        result = self.api.send_melting_data(items)

        results = result.get('results', [])
        successful_ids = [r['sync_id'] for r in results if r.get('success', False)]

        failed_ids = [
            (r['sync_id'], r.get('error', 'Unknown error'))
            for r in results if not r.get('success', False)
        ]

        if failed_ids:
            for sync_id, error in failed_ids:
                logger.warning(f"  Failed to sync melting {sync_id}: {error}")
            failed_sync_ids = [sync_id for sync_id, _ in failed_ids]
            error_msg = failed_ids[0][1] if failed_ids else 'Sync failed'
            self.db.increment_sync_attempts('melting_events', failed_sync_ids, error_msg)

        if successful_ids:
            self.db.mark_melting_events_synced(successful_ids)
            logger.info(f"  Marked {len(successful_ids)} melting events as synced")

        return len(successful_ids)
    
    def _sync_heat_cycles(self) -> int:
        """Sync pending heat cycles (aggregated pouring data)"""
        # Get unsynced heat cycles
        cycles = self.db.get_unsynced_heat_cycles(limit=self.batch_size)

        if not cycles:
            return 0

        # Prepare API payload
        items = []
        for cycle in cycles:
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

            item = {
                'sync_id': cycle['sync_id'],
                'customer_id': cycle['customer_id'],
                'slno': str(cycle.get('slno', '01')),
                'date': cycle['date'],
                'shift': cycle.get('shift') or "",
                'heat_no': cycle.get('heat_no') or "",
                'ladle_number': cycle.get('ladle_number') or "",
                'location': cycle['location'],
                'camera_id': cycle['camera_id'],
                'pouring_start_time': format_timestamp_for_api(cycle['pouring_start_time']),
                'pouring_end_time': format_timestamp_for_api(cycle.get('pouring_end_time')),
                'total_pouring_time': str(cycle.get('total_pouring_time', '0')),  # Seconds as string
                'mould_wise_pouring_time': mould_wise_timing,  # Array of {mould_id, start, end, duration}
                'tapping_start_time': format_timestamp_for_api(cycle.get('tapping_start_time')),
                'tapping_end_time': format_timestamp_for_api(cycle.get('tapping_end_time')),
                'tapping_events': _parse_json_list(cycle.get('tapping_events')),
                'deslagging_events': _parse_json_list(cycle.get('deslagging_events')),
                'spectro_events': _parse_json_list(cycle.get('spectro_events')),
            }
            items.append(item)

        # Send to API
        result = self.api.send_pouring_data(items)

        # Extract successful sync_ids from results array
        results = result.get('results', [])
        successful_ids = [
            r['sync_id'] for r in results
            if r.get('success', False)
        ]

        # Log any failures for debugging
        failed_ids = [
            (r['sync_id'], r.get('error', 'Unknown error'))
            for r in results
            if not r.get('success', False)
        ]

        if failed_ids:
            for sync_id, error in failed_ids:
                logger.warning(f"    Failed to sync heat cycle {sync_id}: {error}")

            # Update sync_attempts for failed cycles
            for sync_id, error_msg in failed_ids:
                self.db.update_heat_cycle_sync_status(sync_id, error_msg)

        # Mark successful records as synced
        if successful_ids:
            self.db.mark_heat_cycles_synced(successful_ids)
            logger.info(f"  ✓ Marked {len(successful_ids)} heat cycles as synced")

        return len(successful_ids)
    
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
