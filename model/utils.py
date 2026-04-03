import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class EmbeddingDataset:
    X: np.ndarray
    y_reg_no: np.ndarray
    reg_to_name: Dict[str, str]
    total_rows: int
    valid_rows: int
    skipped_rows: int
    embedding_dim: int


def _normalize_embedding_shape(value: List) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError("Embedding cannot be scalar.")
    if arr.ndim > 1:
        # Handles cases like [[...]] by flattening to a single vector.
        arr = arr.reshape(-1)
    if arr.size == 0:
        raise ValueError("Embedding cannot be empty.")
    return arr


def load_embeddings_from_db(db_path: str) -> EmbeddingDataset:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT student_name, reg_no, embedding_json
            FROM student_embeddings
            ORDER BY id ASC
            """
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError(
            f"No embeddings found in '{db_path}'. Upload student data before training."
        )

    vectors: List[np.ndarray] = []
    labels: List[str] = []
    name_votes = defaultdict(Counter)
    skipped_rows = 0
    expected_dim = None

    for student_name, reg_no, embedding_json in rows:
        try:
            parsed = json.loads(embedding_json)
            vector = _normalize_embedding_shape(parsed)
        except Exception:
            skipped_rows += 1
            continue

        if expected_dim is None:
            expected_dim = int(vector.shape[0])
        if vector.shape[0] != expected_dim:
            skipped_rows += 1
            continue

        vectors.append(vector)
        labels.append(reg_no)
        name_votes[reg_no][student_name] += 1

    if not vectors:
        raise ValueError(
            "All embeddings were invalid/inconsistent. Check DB rows and try again."
        )

    reg_to_name = {
        reg_no: counter.most_common(1)[0][0] for reg_no, counter in name_votes.items()
    }

    X = np.vstack(vectors).astype(np.float32)
    y = np.asarray(labels)
    return EmbeddingDataset(
        X=X,
        y_reg_no=y,
        reg_to_name=reg_to_name,
        total_rows=len(rows),
        valid_rows=len(vectors),
        skipped_rows=skipped_rows,
        embedding_dim=expected_dim or 0,
    )


def min_samples_per_class(labels: np.ndarray) -> int:
    if labels.size == 0:
        return 0
    counter = Counter(labels.tolist())
    return min(counter.values())
