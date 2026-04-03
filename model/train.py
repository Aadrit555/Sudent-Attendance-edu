import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC

try:
    from model.utils import load_embeddings_from_db, min_samples_per_class
except ModuleNotFoundError:
    from utils import load_embeddings_from_db, min_samples_per_class


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


def _build_centroid_metadata(X: np.ndarray, y_reg_no: np.ndarray) -> Dict[str, Any]:
    Xn = _l2_normalize_rows(X.astype(np.float32))
    class_centroids: Dict[str, np.ndarray] = {}
    per_sample_similarities = []

    unique_labels = np.unique(y_reg_no)
    for reg_no in unique_labels:
        class_vectors = Xn[y_reg_no == reg_no]
        centroid = class_vectors.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            continue
        centroid = centroid / centroid_norm
        class_centroids[str(reg_no)] = centroid

        sims = class_vectors @ centroid
        per_sample_similarities.extend(sims.tolist())

    if per_sample_similarities:
        p05 = float(np.percentile(per_sample_similarities, 5))
        recommended_similarity_threshold = max(0.35, min(0.95, p05 - 0.05))
    else:
        recommended_similarity_threshold = 0.6

    return {
        "class_centroids_reg_no": {
            reg_no: centroid.tolist() for reg_no, centroid in class_centroids.items()
        },
        "recommended_similarity_threshold": recommended_similarity_threshold,
        "centroid_similarity_p05": float(np.percentile(per_sample_similarities, 5))
        if per_sample_similarities
        else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a student face classifier from embeddings stored in SQLite."
    )
    parser.add_argument("--db-path", default="data/attendance.db")
    parser.add_argument("--output-dir", default="model/artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log train params/metrics to MLflow if installed.",
    )
    parser.add_argument("--mlflow-experiment", default="attendance-classifier")
    return parser.parse_args()


def maybe_log_mlflow(
    enabled: bool,
    experiment_name: str,
    params: dict,
    metrics: dict,
) -> None:
    if not enabled:
        return
    try:
        import mlflow
    except Exception:
        print("MLflow not installed. Skipping MLflow logging.")
        return

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(params)
        for key, value in metrics.items():
            if value is not None:
                mlflow.log_metric(key, float(value))


def train_classifier(
    db_path: str = "data/attendance.db",
    output_dir: str = "model/artifacts",
    test_size: float = 0.2,
    random_state: int = 42,
    use_mlflow: bool = False,
    mlflow_experiment: str = "attendance-classifier",
) -> Dict[str, Any]:
    dataset = load_embeddings_from_db(db_path)

    class_counts = Counter(dataset.y_reg_no.tolist())
    if len(class_counts) < 2:
        raise ValueError(
            "Need at least 2 students/classes in DB to train a classifier."
        )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(dataset.y_reg_no)

    model = Pipeline(
        steps=[
            ("normalize", Normalizer(norm="l2")),
            (
                "svc",
                SVC(
                    kernel="linear",
                    probability=True,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    val_accuracy = None
    report_text = None

    if min_samples_per_class(dataset.y_reg_no) >= 2 and dataset.valid_rows >= 4:
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                dataset.X,
                y_encoded,
                test_size=test_size,
                random_state=random_state,
                stratify=y_encoded,
            )
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            report_text = classification_report(
                y_val,
                y_val_pred,
                target_names=label_encoder.inverse_transform(np.unique(y_val)),
                zero_division=0,
            )
        except ValueError:
            print(
                "Skipping holdout validation due to small split size for "
                "current class distribution."
            )
    else:
        print(
            "Skipping holdout validation: need at least 2 samples per student "
            "and enough total rows."
        )

    model.fit(dataset.X, y_encoded)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    classifier_path = output_dir_path / "face_classifier.joblib"
    label_encoder_path = output_dir_path / "label_encoder.joblib"
    metadata_path = output_dir_path / "metadata.json"

    joblib.dump(model, classifier_path)
    joblib.dump(label_encoder, label_encoder_path)

    centroid_meta = _build_centroid_metadata(dataset.X, dataset.y_reg_no)
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "db_path": db_path,
        "samples_total": dataset.total_rows,
        "samples_used": dataset.valid_rows,
        "samples_skipped": dataset.skipped_rows,
        "embedding_dim": dataset.embedding_dim,
        "num_classes": len(class_counts),
        "classes_reg_no": sorted(class_counts.keys()),
        "reg_to_name": dataset.reg_to_name,
        "holdout_accuracy": val_accuracy,
        "class_centroids_reg_no": centroid_meta["class_centroids_reg_no"],
        "recommended_similarity_threshold": centroid_meta[
            "recommended_similarity_threshold"
        ],
        "centroid_similarity_p05": centroid_meta["centroid_similarity_p05"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    maybe_log_mlflow(
        enabled=use_mlflow,
        experiment_name=mlflow_experiment,
        params={
            "samples_used": dataset.valid_rows,
            "num_classes": len(class_counts),
            "embedding_dim": dataset.embedding_dim,
            "classifier": "SVC(linear)",
        },
        metrics={"holdout_accuracy": val_accuracy},
    )

    return {
        "classifier_path": str(classifier_path),
        "label_encoder_path": str(label_encoder_path),
        "metadata_path": str(metadata_path),
        "samples_total": dataset.total_rows,
        "samples_used": dataset.valid_rows,
        "samples_skipped": dataset.skipped_rows,
        "embedding_dim": dataset.embedding_dim,
        "num_classes": len(class_counts),
        "classes_reg_no": sorted(class_counts.keys()),
        "holdout_accuracy": val_accuracy,
        "validation_report": report_text,
        "recommended_similarity_threshold": metadata[
            "recommended_similarity_threshold"
        ],
    }


def main() -> None:
    args = parse_args()
    result = train_classifier(
        db_path=args.db_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        use_mlflow=args.mlflow,
        mlflow_experiment=args.mlflow_experiment,
    )

    print("Training complete.")
    print(f"Classifier saved to: {result['classifier_path']}")
    print(f"Label encoder saved to: {result['label_encoder_path']}")
    print(f"Metadata saved to: {result['metadata_path']}")
    print(
        f"Rows in DB: total={result['samples_total']}, "
        f"used={result['samples_used']}, skipped={result['samples_skipped']}"
    )
    if result["holdout_accuracy"] is not None:
        print(f"Holdout accuracy: {result['holdout_accuracy']:.4f}")
        print("Validation report:")
        print(result["validation_report"])


if __name__ == "__main__":
    main()
