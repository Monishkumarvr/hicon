"""
Database module for HiCon - Local SQLite with 7-day rotation
"""
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import json

logger = logging.getLogger(__name__)


class HiConDatabase:
    """Local SQLite database with 7-day rotation"""

    def __init__(self, db_path: str):
        """
        Initialize database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self.migrate_add_sync_tracking()
        self.migrate_add_heat_cycle_melting_columns()
        logger.info(f"Database initialized: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
        except sqlite3.OperationalError as exc:
            logger.warning(f"Failed to set SQLite pragmas: {exc}")
        return conn

    def _init_schema(self):
        """Initialize database schema"""
        conn = self._get_connection()
        c = conn.cursor()

        # Melting events table (tapping, deslagging, pyrometer, spectro)
        c.execute('''CREATE TABLE IF NOT EXISTS melting_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sync_id TEXT UNIQUE NOT NULL,
            customer_id TEXT NOT NULL,
            slno TEXT NOT NULL,
            date TEXT NOT NULL,
            event_type TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            duration_sec REAL,
            camera_id TEXT NOT NULL,
            location TEXT NOT NULL,
            screenshot_path TEXT,
            synced INTEGER DEFAULT 0,
            sync_attempts INTEGER DEFAULT 0,
            last_sync_error TEXT,
            created_at TEXT NOT NULL
        )''')

        # Pouring events table (replaces agni_heats, tapping_events)
        c.execute('''CREATE TABLE IF NOT EXISTS pouring_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sync_id TEXT UNIQUE NOT NULL,
            customer_id TEXT NOT NULL,
            slno TEXT NOT NULL,
            date TEXT NOT NULL,
            shift TEXT,
            heat_no TEXT,
            ladle_number TEXT,
            location TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            pouring_start_time TEXT NOT NULL,
            pouring_end_time TEXT,
            total_pouring_time TEXT,
            mould_wise_pouring_time TEXT,
            synced INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )''')

        # Heat cycles table (aggregated cycle data for API POST)
        c.execute('''CREATE TABLE IF NOT EXISTS heat_cycles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sync_id TEXT UNIQUE NOT NULL,
            heat_no TEXT UNIQUE NOT NULL,
            customer_id TEXT NOT NULL,
            slno TEXT NOT NULL,
            date TEXT NOT NULL,
            shift TEXT,
            ladle_number TEXT NOT NULL,
            location TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            cycle_start_time TEXT NOT NULL,
            cycle_end_time TEXT NOT NULL,
            pouring_start_time TEXT NOT NULL,
            pouring_end_time TEXT NOT NULL,
            total_pouring_time TEXT NOT NULL,
            mould_wise_pouring_time TEXT NOT NULL,
            tapping_start_time TEXT NOT NULL,
            tapping_end_time TEXT NOT NULL,
            tapping_events TEXT,
            deslagging_events TEXT,
            spectro_events TEXT,
            pyrometer_events TEXT,
            synced INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )''')

        # Metadata table for counters and configuration
        c.execute('''CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )''')

        # Initialize slno counter if not exists
        c.execute('''INSERT OR IGNORE INTO metadata (key, value, updated_at)
                     VALUES (?, ?, ?)''',
                  ('slno_counter', '0', datetime.now().isoformat()))

        # Create indices for melting events
        c.execute('CREATE INDEX IF NOT EXISTS idx_melting_synced ON melting_events(synced)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_melting_created ON melting_events(created_at)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_melting_date ON melting_events(date)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_melting_type ON melting_events(event_type)')

        # Create indices for pouring events
        c.execute('CREATE INDEX IF NOT EXISTS idx_pouring_synced ON pouring_events(synced)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_pouring_created ON pouring_events(created_at)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_pouring_date ON pouring_events(date)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_pouring_start ON pouring_events(pouring_start_time)')

        # Create indices for heat_cycles
        c.execute('CREATE INDEX IF NOT EXISTS idx_heat_synced ON heat_cycles(synced)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_heat_no ON heat_cycles(heat_no)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_heat_date ON heat_cycles(date)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_heat_ladle ON heat_cycles(ladle_number)')

        conn.commit()
        conn.close()

    def migrate_add_sync_tracking(self):
        """Add sync_attempts and last_sync_error columns if they don't exist (SQLite migration)"""
        conn = self._get_connection()
        c = conn.cursor()

        # Pouring events
        try:
            c.execute("ALTER TABLE pouring_events ADD COLUMN sync_attempts INTEGER DEFAULT 0")
            c.execute("ALTER TABLE pouring_events ADD COLUMN last_sync_error TEXT")
            logger.info("Added sync tracking columns to pouring_events")
        except sqlite3.OperationalError:
            logger.debug("Sync tracking columns already exist in pouring_events")

        # Heat cycles
        try:
            c.execute("ALTER TABLE heat_cycles ADD COLUMN sync_attempts INTEGER DEFAULT 0")
            c.execute("ALTER TABLE heat_cycles ADD COLUMN last_sync_error TEXT")
            logger.info("Added sync tracking columns to heat_cycles")
        except sqlite3.OperationalError:
            logger.debug("Sync tracking columns already exist in heat_cycles")

        conn.commit()
        conn.close()

    def migrate_add_heat_cycle_melting_columns(self):
        """Add tapping/deslagging/spectro/pyrometer columns to heat_cycles if they don't exist."""
        conn = self._get_connection()
        c = conn.cursor()

        columns = [
            ("tapping_start_time", "TEXT"),
            ("tapping_end_time", "TEXT"),
            ("tapping_events", "TEXT"),
            ("deslagging_events", "TEXT"),
            ("spectro_events", "TEXT"),
            ("pyrometer_events", "TEXT"),
        ]
        for col_name, col_type in columns:
            try:
                c.execute(f"ALTER TABLE heat_cycles ADD COLUMN {col_name} {col_type}")
                logger.info(f"Added column {col_name} to heat_cycles")
            except sqlite3.OperationalError:
                pass  # Column already exists

        conn.commit()
        conn.close()

    def _get_next_slno(self, conn: sqlite3.Connection) -> str:
        """
        Get next auto-incremented serial number (thread-safe).

        Uses database-level transaction locking to ensure atomicity.

        Args:
            conn: Active database connection (must be in transaction)

        Returns:
            Zero-padded serial number (e.g., "0001", "0002", ..., "9999")
        """
        c = conn.cursor()

        # Read current counter
        c.execute('SELECT value FROM metadata WHERE key = ?', ('slno_counter',))
        row = c.fetchone()

        if row is None:
            # Initialize if missing (defensive)
            current = 0
            c.execute('''INSERT INTO metadata (key, value, updated_at)
                         VALUES (?, ?, ?)''',
                      ('slno_counter', '0', datetime.now().isoformat()))
        else:
            current = int(row[0])

        # Increment counter
        next_value = current + 1

        # Update counter in database
        c.execute('''UPDATE metadata
                     SET value = ?, updated_at = ?
                     WHERE key = ?''',
                  (str(next_value), datetime.now().isoformat(), 'slno_counter'))

        # Format as zero-padded 4-digit string
        return f"{next_value:04d}"

    # === MELTING EVENTS (tapping, deslagging, pyrometer, spectro) ===

    def insert_melting_event(self, sync_id: str, customer_id: str, event_type: str,
                             start_time: str, end_time: str, duration_sec: float,
                             camera_id: str, location: str,
                             screenshot_path: str = "") -> str:
        """
        Insert melting event record.

        Args:
            sync_id: Unique sync identifier
            customer_id: Customer identifier
            event_type: Event type (tapping, deslagging, pyrometer, spectro)
            start_time: ISO8601 start timestamp
            end_time: ISO8601 end timestamp
            duration_sec: Event duration in seconds
            camera_id: Camera identifier
            location: Location description
            screenshot_path: Path to screenshot file

        Returns:
            Auto-generated serial number
        """
        conn = self._get_connection()
        c = conn.cursor()

        try:
            slno = self._get_next_slno(conn)
            date = datetime.fromisoformat(start_time).strftime("%Y-%m-%d")

            c.execute('''INSERT INTO melting_events
                (sync_id, customer_id, slno, date, event_type, start_time, end_time,
                 duration_sec, camera_id, location, screenshot_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (sync_id, customer_id, slno, date, event_type, start_time, end_time,
                 duration_sec, camera_id, location, screenshot_path,
                 datetime.now().isoformat()))
            conn.commit()
            logger.debug(f"Inserted melting event: {event_type} {sync_id} (slno: {slno})")
            return slno
        except sqlite3.IntegrityError:
            logger.warning(f"Duplicate melting event: {sync_id}")
            conn.rollback()
            c.execute('SELECT slno FROM melting_events WHERE sync_id = ?', (sync_id,))
            row = c.fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def get_unsynced_melting_events(self, limit: int = 100) -> List[Dict]:
        """Get unsynced melting events."""
        conn = self._get_connection()
        c = conn.cursor()

        c.execute('''SELECT sync_id, customer_id, slno, date, event_type,
                     start_time, end_time, duration_sec, camera_id, location,
                     screenshot_path
                     FROM melting_events
                     WHERE synced = 0
                     ORDER BY created_at ASC
                     LIMIT ?''', (limit,))

        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def mark_melting_events_synced(self, sync_ids: List[str]):
        """Mark melting events as synced."""
        if not sync_ids:
            return
        conn = self._get_connection()
        c = conn.cursor()
        c.executemany('UPDATE melting_events SET synced = 1, sync_attempts = 0 WHERE sync_id = ?',
                     [(sid,) for sid in sync_ids])
        conn.commit()
        conn.close()
        logger.debug(f"Marked {len(sync_ids)} melting events as synced")

    def mark_melting_events_synced_by_window(self, start_time: str, end_time: str):
        """Mark melting events as synced within a time window (inclusive)."""
        if not start_time or not end_time:
            return
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('''UPDATE melting_events
                     SET synced = 1, sync_attempts = 0, last_sync_error = NULL
                     WHERE start_time >= ? AND start_time <= ?''',
                  (start_time, end_time))
        conn.commit()
        conn.close()

    def has_melting_event_type_in_window(self, event_type: str, start_time: str, end_time: str) -> bool:
        """Check if a melting event of a given type exists in a time window."""
        if not event_type or not start_time or not end_time:
            return False
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('''SELECT 1 FROM melting_events
                     WHERE event_type = ?
                       AND start_time >= ? AND start_time <= ?
                     LIMIT 1''',
                  (event_type, start_time, end_time))
        row = c.fetchone()
        conn.close()
        return row is not None

    # === POURING EVENTS ===

    def insert_pouring_event(self, sync_id: str, customer_id: str, date: str,
                            location: str, camera_id: str, pouring_start_time: str,
                            shift: Optional[str] = None, heat_no: Optional[str] = None,
                            ladle_number: Optional[str] = None,
                            pouring_end_time: Optional[str] = None,
                            total_pouring_time: Optional[str] = None,
                            mould_wise_pouring_time: Optional[Dict] = None) -> str:
        """
        Insert pouring event record with auto-generated serial number.

        Args:
            sync_id: Unique sync identifier
            customer_id: Customer identifier
            date: Event date (YYYY-MM-DD)
            location: Location identifier
            camera_id: Camera identifier
            pouring_start_time: ISO8601 timestamp
            shift: Optional shift identifier
            heat_no: Optional heat number
            ladle_number: Optional ladle identifier
            pouring_end_time: Optional end timestamp
            total_pouring_time: Optional total duration string
            mould_wise_pouring_time: Optional dict of mould-wise times

        Returns:
            Auto-generated serial number (e.g., "0001", "0002")
        """
        conn = self._get_connection()
        c = conn.cursor()

        try:
            # Generate next serial number (thread-safe, in transaction)
            slno = self._get_next_slno(conn)

            # Convert mould_wise_pouring_time dict to JSON string
            mould_wise_json = json.dumps(mould_wise_pouring_time) if mould_wise_pouring_time else None

            c.execute('''INSERT INTO pouring_events
                (sync_id, customer_id, slno, date, shift, heat_no, ladle_number,
                 location, camera_id, pouring_start_time, pouring_end_time,
                 total_pouring_time, mould_wise_pouring_time, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (sync_id, customer_id, slno, date, shift, heat_no, ladle_number,
                 location, camera_id, pouring_start_time, pouring_end_time,
                 total_pouring_time, mould_wise_json, datetime.now().isoformat()))
            conn.commit()
            logger.debug(f"✓ Inserted pouring event: {sync_id} (slno: {slno})")
            return slno
        except sqlite3.IntegrityError:
            logger.warning(f"Duplicate pouring event: {sync_id}")
            conn.rollback()
            # Return existing slno for duplicate
            c.execute('SELECT slno FROM pouring_events WHERE sync_id = ?', (sync_id,))
            row = c.fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def update_pouring_end(self, sync_id: str, pouring_end_time: str,
                          total_pouring_time: str, mould_wise_pouring_time: Dict):
        """
        Update pouring event when pouring completes.

        Note:
            Pouring events are stored for local traceability, while cloud sync uses
            finalized heat cycles. Completed records are marked synced here to avoid
            unsynced drift in local stats.

        Args:
            sync_id: Unique sync identifier
            pouring_end_time: ISO8601 end timestamp
            total_pouring_time: Total duration string
            mould_wise_pouring_time: Dict of mould-wise pouring times
        """
        conn = self._get_connection()
        c = conn.cursor()

        # Convert mould_wise_pouring_time dict to JSON string
        mould_wise_json = json.dumps(mould_wise_pouring_time)

        c.execute('''UPDATE pouring_events
                     SET pouring_end_time = ?,
                         total_pouring_time = ?,
                         mould_wise_pouring_time = ?,
                         synced = 1,
                         sync_attempts = 0,
                         last_sync_error = NULL
                     WHERE sync_id = ?''',
                  (pouring_end_time, total_pouring_time, mould_wise_json, sync_id))
        conn.commit()
        conn.close()
        logger.debug(f"✓ Updated pouring event end: {sync_id}")

    # === HEAT CYCLES ===

    def insert_heat_cycle(self, sync_id: str, heat_no: str, customer_id: str, date: str,
                         ladle_number: str, location: str, camera_id: str,
                         cycle_start_time: str, cycle_end_time: str,
                         pouring_start_time: str, pouring_end_time: str,
                         total_pouring_time: str, mould_wise_pouring_time: Dict,
                         shift: Optional[str] = None,
                         tapping_start_time: Optional[str] = None,
                         tapping_end_time: Optional[str] = None,
                         tapping_events: Optional[List] = None,
                         deslagging_events: Optional[List] = None,
                         spectro_events: Optional[List] = None,
                         pyrometer_events: Optional[List] = None) -> str:
        """
        Insert completed heat cycle record.

        Args:
            sync_id: Unique sync identifier
            heat_no: Heat cycle identifier
            customer_id: Customer identifier
            date: Event date (YYYY-MM-DD)
            ladle_number: Ladle identifier
            location: Location identifier
            camera_id: Camera identifier
            cycle_start_time: Cycle start timestamp (ISO8601)
            cycle_end_time: Cycle end timestamp (ISO8601)
            pouring_start_time: First mould pour start (ISO8601)
            pouring_end_time: Last mould pour end (ISO8601)
            total_pouring_time: Total duration (HH:MM:SS)
            mould_wise_pouring_time: Dict of mould-wise times
            shift: Optional shift identifier
            tapping_start_time: First tapping start (ISO8601)
            tapping_end_time: Last tapping end (ISO8601)
            tapping_events: List of tapping event dicts
            deslagging_events: List of deslagging event dicts
            spectro_events: List of spectro event dicts

        Returns:
            Auto-generated serial number
        """
        conn = self._get_connection()
        c = conn.cursor()

        try:
            # Generate next serial number
            slno = self._get_next_slno(conn)

            # Convert to JSON
            mould_wise_json = json.dumps(mould_wise_pouring_time)
            tapping_events_json = json.dumps(tapping_events) if tapping_events else None
            deslagging_events_json = json.dumps(deslagging_events) if deslagging_events else None
            spectro_events_json = json.dumps(spectro_events) if spectro_events else None
            pyrometer_events_json = json.dumps(pyrometer_events) if pyrometer_events else None

            c.execute('''INSERT INTO heat_cycles
                (sync_id, heat_no, customer_id, slno, date, shift, ladle_number,
                 location, camera_id, cycle_start_time, cycle_end_time,
                 pouring_start_time, pouring_end_time, total_pouring_time,
                 mould_wise_pouring_time, tapping_start_time, tapping_end_time,
                 tapping_events, deslagging_events, spectro_events, pyrometer_events, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (sync_id, heat_no, customer_id, slno, date, shift, ladle_number,
                 location, camera_id, cycle_start_time, cycle_end_time,
                 pouring_start_time, pouring_end_time, total_pouring_time,
                 mould_wise_json, tapping_start_time, tapping_end_time,
                 tapping_events_json, deslagging_events_json, spectro_events_json, pyrometer_events_json,
                 datetime.now().isoformat()))
            conn.commit()
            logger.info(f"Inserted heat cycle: {heat_no} (slno: {slno})")
            return slno

        finally:
            conn.close()

    def get_unsynced_heat_cycles(self, limit: int = 50) -> List[Dict]:
        """Get unsynced heat cycles for API POST."""
        conn = self._get_connection()
        c = conn.cursor()

        c.execute('''SELECT sync_id, heat_no, customer_id, slno, date, shift, ladle_number,
                     location, camera_id, cycle_start_time, cycle_end_time,
                     pouring_start_time, pouring_end_time, total_pouring_time,
                     mould_wise_pouring_time, tapping_start_time, tapping_end_time,
                     tapping_events, deslagging_events, spectro_events, pyrometer_events
                     FROM heat_cycles
                     WHERE synced = 0
                     ORDER BY created_at ASC
                     LIMIT ?''', (limit,))

        rows = c.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def mark_heat_cycles_synced(self, sync_ids: List[str]):
        """Mark heat cycles as synced."""
        if not sync_ids:
            return

        conn = self._get_connection()
        c = conn.cursor()

        placeholders = ','.join('?' * len(sync_ids))
        c.execute(f'''UPDATE heat_cycles
                     SET synced = 1
                     WHERE sync_id IN ({placeholders})''', sync_ids)
        conn.commit()
        conn.close()

        logger.debug(f"✓ Marked {len(sync_ids)} heat cycles as synced")

    def update_heat_cycle_sync_status(self, sync_id: str, error: str):
        """Update sync attempt count and last error for a heat cycle."""
        conn = self._get_connection()
        c = conn.cursor()

        c.execute('''UPDATE heat_cycles
                     SET sync_attempts = sync_attempts + 1,
                         last_sync_error = ?
                     WHERE sync_id = ?''', (error, sync_id))
        conn.commit()
        conn.close()

    def get_unsynced_pouring_events(self, limit: int = 100) -> List[Dict]:
        """Get unsynced COMPLETED pouring events (with end time)"""
        conn = self._get_connection()
        c = conn.cursor()

        # Only get completed events (pouring_end_time IS NOT NULL)
        c.execute('''SELECT sync_id, customer_id, slno, date, shift, heat_no, ladle_number,
                     location, camera_id, pouring_start_time, pouring_end_time,
                     total_pouring_time, mould_wise_pouring_time
                     FROM pouring_events
                     WHERE synced = 0 AND pouring_end_time IS NOT NULL
                     ORDER BY created_at ASC
                     LIMIT ?''', (limit,))

        rows = c.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def mark_pouring_synced(self, sync_ids: List[str]):
        """Mark pouring events as synced and reset sync_attempts"""
        conn = self._get_connection()
        c = conn.cursor()

        c.executemany('UPDATE pouring_events SET synced = 1, sync_attempts = 0 WHERE sync_id = ?',
                     [(sid,) for sid in sync_ids])
        conn.commit()
        conn.close()

        logger.debug(f"✓ Marked {len(sync_ids)} pouring events as synced")

    def get_active_pouring_session(self) -> Optional[Dict]:
        """
        Get currently active pouring session (pouring_end_time IS NULL).

        Returns:
            Dict with pouring event data, or None if no active session
        """
        conn = self._get_connection()
        c = conn.cursor()

        c.execute('''SELECT sync_id, customer_id, slno, date, shift, heat_no, ladle_number,
                     location, camera_id, pouring_start_time, pouring_end_time,
                     total_pouring_time, mould_wise_pouring_time
                     FROM pouring_events
                     WHERE pouring_end_time IS NULL
                     ORDER BY id DESC LIMIT 1''')

        row = c.fetchone()
        conn.close()

        return dict(row) if row else None

    # === SYNC TRACKING ===

    def increment_sync_attempts(self, table_name: str, sync_ids: List[str], error_message: str):
        """Increment sync_attempts and store last error for failed syncs"""
        if not sync_ids:
            return

        conn = self._get_connection()
        c = conn.cursor()

        placeholders = ','.join(['?' for _ in sync_ids])
        c.execute(f'''UPDATE {table_name}
                      SET sync_attempts = sync_attempts + 1,
                          last_sync_error = ?
                      WHERE sync_id IN ({placeholders})''',
                  [error_message] + sync_ids)

        conn.commit()
        conn.close()
        logger.debug(f"Incremented sync_attempts for {len(sync_ids)} {table_name} records")

    def get_stuck_records(self, max_attempts: int = 5) -> Dict[str, List[Dict]]:
        """Get records that have failed sync multiple times"""
        conn = self._get_connection()
        c = conn.cursor()

        # Melting events with many failures
        c.execute('''SELECT 'melting' as table_name, sync_id, sync_attempts,
                            last_sync_error, start_time
                     FROM melting_events
                     WHERE synced = 0 AND sync_attempts >= ?
                     ORDER BY sync_attempts DESC''', (max_attempts,))
        melting_stuck = [dict(row) for row in c.fetchall()]

        # Pouring events with many failures
        c.execute('''SELECT 'pouring' as table_name, sync_id, sync_attempts,
                            last_sync_error, pouring_start_time
                     FROM pouring_events
                     WHERE synced = 0 AND sync_attempts >= ?
                     ORDER BY sync_attempts DESC''', (max_attempts,))
        pouring_stuck = [dict(row) for row in c.fetchall()]

        conn.close()
        return {'melting': melting_stuck, 'pouring': pouring_stuck}

    # === DATA ROTATION (7-day cleanup) ===

    def cleanup_old_data(self, days: int = 7):
        """
        Delete data older than specified days.

        Args:
            days: Number of days to retain
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        conn = self._get_connection()
        c = conn.cursor()

        tables = [
            'melting_events',
            'pouring_events',
            'heat_cycles',
        ]

        total_deleted = 0
        for table in tables:
            c.execute(f'DELETE FROM {table} WHERE created_at < ?', (cutoff,))
            deleted = c.rowcount
            total_deleted += deleted
            if deleted > 0:
                logger.info(f"✓ Deleted {deleted} old records from {table}")

        conn.commit()
        conn.close()

        if total_deleted > 0:
            # Vacuum to reclaim space
            conn = self._get_connection()
            conn.execute('VACUUM')
            conn.close()
            logger.info(f"✓ Total cleanup: {total_deleted} records, database vacuumed")

    # === STATISTICS ===

    def get_stats(self) -> Dict:
        """Get database statistics"""
        conn = self._get_connection()
        c = conn.cursor()

        stats = {}

        c.execute('SELECT COUNT(*) as total, COALESCE(SUM(synced), 0) as synced FROM melting_events')
        row = c.fetchone()
        total = row['total'] or 0
        synced = row['synced'] or 0
        stats['melting'] = {'total': total, 'synced': synced, 'pending': total - synced}

        c.execute('SELECT COUNT(*) as total, COALESCE(SUM(synced), 0) as synced FROM pouring_events')
        row = c.fetchone()
        total = row['total'] or 0
        synced = row['synced'] or 0
        stats['pouring'] = {'total': total, 'synced': synced, 'pending': total - synced}

        conn.close()

        return stats
