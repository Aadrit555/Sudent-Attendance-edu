import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    from model.utils import load_embeddings_from_db
except ModuleNotFoundError:
    from utils import load_embeddings_from_db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved face classifier on embeddings present in SQLite."
    )
    parser.add_argument("--db-path", default="data/attendance.db")
    parser.add_argument("--artifacts-dir", default="model/artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    classifier_path = artifacts_dir / "face_classifier.joblib"
    label_encoder_path = artifacts_dir / "label_encoder.joblib"
    metadata_path = artifacts_dir / "metadata.json"

    if not classifier_path.exists() or not label_encoder_path.exists():
        raise FileNotFoundError(
            "Missing model artifacts. Run training first: python3 model/train.py"
        )

    dataset = load_embeddings_from_db(args.db_path)
    model = joblib.load(classifier_path)
    label_encoder = joblib.load(label_encoder_path)

    y_true = label_encoder.transform(dataset.y_reg_no)
    y_pred = model.predict(dataset.X)

    accuracy = accuracy_score(y_true, y_pred)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    class_names = label_encoder.inverse_transform(labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"Evaluation on {dataset.valid_rows} rows from DB '{args.db_path}'")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(report)
    print("Confusion matrix:")
    print(cm)

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        print("Training metadata:")
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
