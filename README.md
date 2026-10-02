# Student Attendance Face Recognition System (`Sudent-Attendance-edu`)

An automated, deep-learning-powered student attendance verification backend. The system ingests student registration videos or images, extracts facial features using MTCNN and Inception-ResNet-v1 (FaceNet), stores high-dimensional embeddings, and classifies identities using a Support Vector Machine (SVM) pipeline with open-set rejection.

---

## Key Features

- **Automated Face Detection & Cropping**: Uses MTCNN to extract high-confidence face crops from video frames and images.
- **Deep Face Embeddings**: Generates 512-dimensional normalized face embeddings via `facenet-pytorch` (Inception-ResNet-v1 pretrained on VGGFace2).
- **Dual Video Storage (Cloud & Local)**:
  - **AWS S3**: Automatically archives incoming student registration videos to an Amazon S3 bucket if configured.
  - **Local Fallback**: Automatically saves to local storage if AWS S3 environment variables are not set.
- **Embedding Database**: Stores vector embeddings and metadata in SQLite for training and retrieval.
- **Machine Learning Classification**:
  - Linear SVC with L2-normalized feature representation and balanced class weighting.
  - Class centroid calculation for cosine-similarity gating (open-set rejection of unregistered individuals).
- **FastAPI REST Service**: Production-ready endpoints for uploading video, continuous training, health checks, and single-image identity prediction.
- **CLI Utilities**: Standalone scripts for model training, evaluation metrics, and offline inference.

---

## System Architecture

```text
[ Student Video / Image ]
           │
           ▼
[ Video / Image Preprocessing (OpenCV) ]
           │
           ▼
[ Face Detection & Alignment (MTCNN) ]
           │
           ▼
[ Embedding Generation (InceptionResnetV1) ] ──► [ SQLite DB: student_embeddings ]
           │                                                │
           ▼                                                ▼
[ Prediction / Classification (Linear SVC) ] ◄─── [ Train Model: SVC + Centroids ]
           │
           ▼
[ Open-Set Verification & Identity Match ]
```

---

## Project Structure

```text
├── api/
│   ├── __init__.py
│   ├── api.py               # FastAPI application and endpoints
│   └── schemas.py           # Pydantic schemas for request validation
├── data/                    # Local SQLite database directory
│   └── .gitkeep
├── data_process/
│   ├── __init__.py
│   ├── db_service.py        # SQLite database service for student embeddings
│   ├── image_service.py     # MTCNN face detection and FaceNet embedding service
│   └── video_service.py     # Video frame extraction and S3 utilities
├── model/
│   ├── __init__.py
│   ├── artifacts/           # Trained models, encoders, and centroid metadata
│   │   └── .gitkeep
│   ├── evaluate.py          # Script for classification metrics and confusion matrix
│   ├── predict.py           # Single-embedding prediction with open-set rejection
│   ├── train.py             # Classifier training with class centroid calculation
│   └── utils.py             # Database embedding loaders and validation utilities
├── .env.example             # Template for environment variables
├── .gitignore               # Git ignore rules for media, caches, and secrets
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation
```

---

## Quickstart Guide

### 1. Prerequisites

- Python 3.9+
- Recommended: NVIDIA GPU with CUDA support for faster face embedding generation (CPU fallback is automatic).

### 2. Clone and Setup Environment

```bash
git clone https://github.com/Aadrit555/Sudent-Attendance-edu.git
cd Sudent-Attendance-edu

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables (Optional for S3)

To enable cloud storage for videos, copy `.env.example` to `.env` and fill in your AWS details:

```bash
cp .env.example .env
```

Edit `.env`:

```ini
AWS_S3_BUCKET_NAME=attendance-video-data
AWS_REGION=us-east-1
AWS_S3_VIDEO_PREFIX=uploaded_videos
# AWS_ACCESS_KEY_ID=your_access_key
# AWS_SECRET_ACCESS_KEY=your_secret_key
```

*Note: If no S3 bucket is configured, video files will be stored locally in `data_process/data/images`.*

---

## Running the API Server

Start the FastAPI application using `uvicorn`:

```bash
uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## API Endpoints

### 1. `GET /health`
Verifies server health.
- **Response**: `{"status": "healthy"}`

### 2. `POST /upload_video`
Uploads a student video, extracts face crops at sampled intervals, computes face embeddings, and saves them to the database.
- **Form Data**:
  - `name`: Student full name (e.g. `"John Doe"`)
  - `reg_no`: Unique registration number (e.g. `"REG101"`)
  - `timestamp`: (Optional) ISO timestamp string
  - `video`: Video file upload (`.mp4`, `.mov`)
- **Response**:
  ```json
  {
    "message": "Video processed successfully",
    "storage_mode": "s3",
    "s3_bucket": "attendance-video-data",
    "s3_object_key": "uploaded_videos/John_Doe_REG101.mp4",
    "faces_detected": 15,
    "embeddings_saved": 15
  }
  ```

### 3. `POST /upload_video_and_train`
Ingests a student video, extracts embeddings, stores them in the database, and immediately triggers classifier re-training.
- **Form Data**: Same as `/upload_video`
- **Response**: Returns processing results and training status (`"trained"` or `"skipped"` if minimum student count is not yet reached).

### 4. `POST /predict_person`
Uploads a single query photo (e.g. from an attendance kiosk or webcam snapshot), detects the face, and identifies the student.
- **Form Data**:
  - `image`: Image file (`.jpg`, `.png`)
  - `threshold`: Minimum classifier probability threshold (default: `0.0`)
  - `similarity_threshold`: Cosine similarity threshold against known class centroids (optional)
  - `top_k`: Number of top predictions to return (default: `3`)
- **Response**:
  ```json
  {
    "message": "Prediction completed",
    "faces_detected": 1,
    "prediction": {
      "predicted_reg_no": "REG101",
      "predicted_name": "John Doe",
      "confidence": 0.982,
      "best_similarity": 0.914,
      "top_k_reg_no": [
        ["REG101", 0.982],
        ["REG102", 0.018]
      ]
    }
  }
  ```

---

## Command Line Utilities

### Train Face Classifier
Train an SVM classifier on existing embeddings stored in SQLite:
```bash
python model/train.py --db-path data/attendance.db --output-dir model/artifacts
```

### Model Evaluation
Evaluate the saved model artifacts against the embeddings database:
```bash
python model/evaluate.py --db-path data/attendance.db --artifacts-dir model/artifacts
```

### Run Embedding Prediction
Test identity prediction from a JSON embedding vector:
```bash
python model/predict.py --artifacts-dir model/artifacts --embedding-file path/to/sample_embedding.json
```

---

## License

This project is licensed under the MIT License.
