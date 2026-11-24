import os
import sqlite3
from datetime import datetime

DB_PATH = os.environ.get("DB_PATH", os.path.join("cryptoshield.db"))


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_columns(conn):
    user_cols = {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "phone" not in user_cols:
        conn.execute("ALTER TABLE users ADD COLUMN phone TEXT NOT NULL DEFAULT ''")

    scan_cols = {row["name"] for row in conn.execute("PRAGMA table_info(scans)").fetchall()}
    if "user_id" not in scan_cols:
        conn.execute("ALTER TABLE scans ADD COLUMN user_id INTEGER")
    if "fake_probability" not in scan_cols:
        conn.execute("ALTER TABLE scans ADD COLUMN fake_probability REAL NOT NULL DEFAULT 0.0")
    if "is_fake" not in scan_cols:
        conn.execute("ALTER TABLE scans ADD COLUMN is_fake INTEGER NOT NULL DEFAULT 0")
    if "summary" not in scan_cols:
        conn.execute("ALTER TABLE scans ADD COLUMN summary TEXT NOT NULL DEFAULT ''")
    if "processing_time" not in scan_cols:
        conn.execute("ALTER TABLE scans ADD COLUMN processing_time REAL NOT NULL DEFAULT 0.0")
    if "media_type" not in scan_cols:
        conn.execute("ALTER TABLE scans ADD COLUMN media_type TEXT NOT NULL DEFAULT 'video'")
    if "created_at" not in scan_cols:
        conn.execute("ALTER TABLE scans ADD COLUMN created_at TEXT NOT NULL DEFAULT ''")


def init_db():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                phone TEXT NOT NULL DEFAULT '',
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                filename TEXT NOT NULL,
                fake_probability REAL NOT NULL,
                is_fake INTEGER NOT NULL,
                summary TEXT NOT NULL,
                processing_time REAL NOT NULL DEFAULT 0.0,
                media_type TEXT NOT NULL DEFAULT 'video',
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                rating INTEGER NOT NULL,
                comment TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        _ensure_columns(conn)
        conn.commit()


def create_user(full_name, email, phone, password_hash):
    clean_name = (full_name or "").strip()[:100]
    clean_email = (email or "").strip().lower()[:150]
    clean_phone = (phone or "").strip()[:25]
    if not clean_name or not clean_email or "@" not in clean_email or not clean_phone:
        return False

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO users (full_name, email, phone, password_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (clean_name, clean_email, clean_phone, password_hash, now),
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def get_user_by_email(email):
    clean_email = (email or "").strip().lower()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (clean_email,),
        ).fetchone()
    return dict(row) if row else None


def get_all_users_for_owner():
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, full_name, email, phone, created_at
            FROM users
            ORDER BY id DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def add_scan(
    filename,
    fake_probability,
    is_fake,
    summary,
    user_id=None,
    processing_time=0.0,
    media_type="video",
):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    clean_name = (filename or "unknown_media.mp4")[:255]
    clean_summary = (summary or "")[:500]
    clean_media_type = "image" if str(media_type).lower() == "image" else "video"
    prob_val = float(fake_probability or 0.0)
    fake_int = 1 if is_fake else 0
    proc_val = round(float(processing_time or 0.0), 2)

    with get_connection() as conn:
        scan_cols = {row["name"] for row in conn.execute("PRAGMA table_info(scans)").fetchall()}
        if "probability" in scan_cols:
            conn.execute(
                """
                INSERT INTO scans (
                    user_id, filename, probability, fake_probability, is_fake,
                    summary, processing_time, media_type, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    clean_name,
                    prob_val,
                    prob_val,
                    fake_int,
                    clean_summary,
                    proc_val,
                    clean_media_type,
                    now,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO scans (
                    user_id, filename, fake_probability, is_fake,
                    summary, processing_time, media_type, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    clean_name,
                    prob_val,
                    fake_int,
                    clean_summary,
                    proc_val,
                    clean_media_type,
                    now,
                ),
            )
        conn.commit()


def get_user_scans(user_id, limit=25):
    if not user_id:
        return []
    safe_limit = max(1, min(100, int(limit)))
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, filename, fake_probability, is_fake, summary,
                   processing_time, media_type, created_at
            FROM scans
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(user_id), safe_limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_scans(limit=10):
    safe_limit = max(1, min(50, int(limit)))
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, user_id, filename, fake_probability, is_fake, summary,
                   processing_time, media_type, created_at
            FROM scans
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def add_feedback(name, rating, comment):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    try:
        rating_int = max(1, min(5, int(rating)))
    except (TypeError, ValueError):
        rating_int = 5

    clean_name = (name or "").strip()[:100]
    clean_comment = (comment or "").strip()[:1000]
    if not clean_name or not clean_comment:
        return False

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO feedback (name, rating, comment, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (clean_name, rating_int, clean_comment, now),
        )
        conn.commit()
    return True


def get_recent_feedback(limit=10):
    safe_limit = max(1, min(50, int(limit)))
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, name, rating, comment, created_at
            FROM feedback
            ORDER BY id DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
    return [dict(r) for r in rows]