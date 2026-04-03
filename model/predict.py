import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np


def _normalize_embedding(raw: Any) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError("Embedding must be a vector/list.")
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        raise ValueError("Embedding cannot be empty.")

    return arr


def _l2_normalize_vector(x: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(x))
    if norm == 0.0:
        return x
    return x / norm


def _load_embedding(
    embedding_json: Optional[str],
    embedding_file: Optional[str],
) -> np.ndarray:
    if not embedding_json and not embedding_file:
        raise ValueError("Provide --embedding-json or --embedding-file.")

    if embedding_json:
        raw: Any = json.loads(embedding_json)
    else:
        raw = json.loads(Path(embedding_file).read_text())

    return _normalize_embedding(raw)


def _top_k_predictions(
    proba: np.ndarray,
    label_encoder,
    top_k: int,
) -> List[Tuple[str, float]]:
    top_indices = np.argsort(proba)[::-1][:top_k]
    labels = label_encoder.inverse_transform(top_indices)
    return [(label, float(proba[idx])) for label, idx in zip(labels, top_indices)]


def predict_from_embedding(
    embedding: Any,
    artifacts_dir: str = "model/artifacts",
    top_k: int = 3,
    threshold: float = 0.0,
    similarity_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    artifacts_path = Path(artifacts_dir)

    classifier_path = artifacts_path / "face_classifier.joblib"
    label_encoder_path = artifacts_path / "label_encoder.joblib"
    metadata_path = artifacts_path / "metadata.json"

    if not classifier_path.exists() or not label_encoder_path.exists():
        raise FileNotFoundError(
            "Missing model artifacts. Run training first: python3 model/train.py"
        )

    model = joblib.load(classifier_path)
    label_encoder = joblib.load(label_encoder_path)
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    embedding_vector = _normalize_embedding(embedding)
    embedding_vector_norm = _l2_normalize_vector(embedding_vector.astype(np.float32))

    expected_dim = getattr(model, "n_features_in_", None)
    if expected_dim is not None and int(embedding_vector.shape[0]) != int(expected_dim):
        raise ValueError(
            f"Embedding dimension mismatch. Expected {expected_dim}, "
            f"got {embedding_vector.shape[0]}"
        )

    sample = embedding_vector.reshape(1, -1)
    confidence = None
    top_k_predictions: List[Tuple[str, float]] = []
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(sample)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        top_k_predictions = _top_k_predictions(probs, label_encoder, max(top_k, 1))
    else:
        pred_idx = int(model.predict(sample)[0])

    pred_reg_no = label_encoder.inverse_transform([pred_idx])[0]
    pred_name = metadata.get("reg_to_name", {}).get(pred_reg_no, "UNKNOWN_NAME")

    # Open-set rejection: if embedding is not close enough to any known class centroid,
    # mark as UNKNOWN even if classifier predicts a known label.
    class_centroids = metadata.get("class_centroids_reg_no", {})
    best_similarity = None
    best_similarity_reg_no = None
    if class_centroids:
        for reg_no, centroid in class_centroids.items():
            centroid_vec = _l2_normalize_vector(np.asarray(centroid, dtype=np.float32))
            sim = float(np.dot(embedding_vector_norm, centroid_vec))
            if best_similarity is None or sim > best_similarity:
                best_similarity = sim
                best_similarity_reg_no = reg_no

    effective_similarity_threshold = similarity_threshold
    if effective_similarity_threshold is None:
        effective_similarity_threshold = metadata.get("recommended_similarity_threshold")
    if effective_similarity_threshold is None:
        effective_similarity_threshold = 0.8

    if best_similarity is not None and best_similarity < effective_similarity_threshold:
        return {
            "prediction": "UNKNOWN",
            "reason": "similarity_below_threshold",
            "best_similarity": best_similarity,
            "similarity_threshold": effective_similarity_threshold,
            "best_similarity_reg_no": best_similarity_reg_no,
            "confidence": confidence,
            "top_k_reg_no": top_k_predictions,
        }

    if confidence is not None and confidence < threshold:
        return {
            "prediction": "UNKNOWN",
            "reason": "confidence_below_threshold",
            "confidence": confidence,
            "threshold": threshold,
            "top_k_reg_no": top_k_predictions,
            "best_similarity": best_similarity,
            "similarity_threshold": effective_similarity_threshold,
        }

    return {
        "predicted_reg_no": pred_reg_no,
        "predicted_name": pred_name,
        "confidence": confidence,
        "top_k_reg_no": top_k_predictions,
        "best_similarity": best_similarity,
        "similarity_threshold": effective_similarity_threshold,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict student identity from one face embedding."
    )
    parser.add_argument("--artifacts-dir", default="model/artifacts")
    parser.add_argument("--embedding-json", default=None)
    parser.add_argument("--embedding-file", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="If confidence is below threshold, return UNKNOWN.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="If cosine similarity to nearest class centroid is below this, return UNKNOWN.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedding = _load_embedding(args.embedding_json, args.embedding_file)
    result = predict_from_embedding(
        embedding=embedding,
        artifacts_dir=args.artifacts_dir,
        top_k=args.top_k,
        threshold=args.threshold,
        similarity_threshold=args.similarity_threshold,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
