"""Minimal account system: bcrypt-hashed passwords in SQLite."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import bcrypt

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8


class AuthError(ValueError):
    pass


class UserStore:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash BLOB NOT NULL,
                    display_name TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

    def register(self, email: str, password: str, display_name: str = "") -> Dict:
        email = email.strip().lower()
        if not EMAIL_PATTERN.match(email):
            raise AuthError("that doesn't look like a valid email address")
        if len(password) < MIN_PASSWORD_LENGTH:
            raise AuthError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")

        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    "INSERT INTO users (email, password_hash, display_name, created_at) VALUES (?, ?, ?, ?)",
                    (email, password_hash, display_name.strip(),
                     datetime.now(timezone.utc).isoformat()),
                )
                user_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            raise AuthError("that email is already registered - sign in instead")
        return {"id": user_id, "email": email, "display_name": display_name.strip()}

    def authenticate(self, email: str, password: str) -> Optional[Dict]:
        email = email.strip().lower()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, email, password_hash, display_name FROM users WHERE email = ?",
                (email,),
            ).fetchone()
        if row is None:
            return None
        if not bcrypt.checkpw(password.encode(), row["password_hash"]):
            return None
        return {"id": row["id"], "email": row["email"], "display_name": row["display_name"]}
