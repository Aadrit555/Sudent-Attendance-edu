import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional


class EmbeddingDBService:
    def __init__(self, db_path: str = "data/attendance.db") -> None:
        self.db_path = db_path
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._initialize_table()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize_table(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS student_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_name TEXT NOT NULL,
                    reg_no TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def insert_embeddings(
        self,
        student_name: str,
        reg_no: str,
        captured_at: Optional[str],
        embeddings: List,
    ) -> int:
        if not embeddings:
            return 0

        normalized_timestamp = captured_at or datetime.now(timezone.utc).isoformat()
        rows = []
        for embedding in embeddings:
            if embedding is None:
                continue

            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()

            rows.append(
                (
                    student_name,
                    reg_no,
                    normalized_timestamp,
                    json.dumps(embedding),
                )
            )

        if not rows:
            return 0

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO student_embeddings (
                    student_name, reg_no, captured_at, embedding_json
                ) VALUES (?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

        return len(rows)
