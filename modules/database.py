"""
database.py — SQLite database handling for user authentication + analyses
"""

import sqlite3
import hashlib
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "users.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT    NOT NULL,
            email      TEXT    NOT NULL UNIQUE,
            password   TEXT    NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            locality    TEXT,
            budget      TEXT,
            team_size   INTEGER,
            domain      TEXT,
            idea        TEXT,
            result      TEXT,
            title       TEXT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    # Add title column if upgrading existing DB
    try:
        cursor.execute("ALTER TABLE analyses ADD COLUMN title TEXT")
    except Exception:
        pass
    conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(name: str, email: str, password: str) -> dict:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            (name.strip(), email.strip().lower(), hash_password(password))
        )
        conn.commit()
        return {"success": True, "message": "Account created successfully!"}
    except sqlite3.IntegrityError:
        return {"success": False, "message": "An account with this email already exists."}
    finally:
        conn.close()


def login_user(email: str, password: str) -> dict:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE email = ? AND password = ?",
        (email.strip().lower(), hash_password(password))
    ).fetchone()
    conn.close()
    if row:
        return {"success": True, "user": dict(row), "message": "Login successful!"}
    return {"success": False, "user": None, "message": "Invalid email or password."}


def save_analysis(user_id: int, locality: str, budget: str, team_size: int,
                  domain: str, idea: str, result: str, title: str = "") -> int:
    conn = get_connection()
    cur = conn.execute(
        "INSERT INTO analyses (user_id, locality, budget, team_size, domain, idea, result, title) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, locality, budget, team_size, domain, idea, result, title)
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_user_analyses(user_id: int, limit: int = 20) -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_analysis_by_id(analysis_id: int) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_analysis(analysis_id: int, user_id: int):
    conn = get_connection()
    conn.execute(
        "DELETE FROM analyses WHERE id = ? AND user_id = ?", (analysis_id, user_id)
    )
    conn.commit()
    conn.close()


