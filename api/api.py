import asyncio
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
from tempfile import NamedTemporaryFile
import boto3
from botocore.exceptions import BotoCoreError, ClientError

from fastapi import FastAPI, File, HTTPException, UploadFile

from data_process.db_service import EmbeddingDBService
from data_process.image_service import ImageService
from model.predict import predict_from_embedding
from model.train import train_classifier

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="Student Attendance Face Recognition API")
embedding_db = EmbeddingDBService()


def _sanitize_s3_component(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", (value or "").strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or fallback


async def _save_uploaded_video(video: UploadFile) -> str:
    os.makedirs("data_process/data/images", exist_ok=True)
    safe_filename = Path(video.filename or "uploaded_video.mp4").name
    file_path = f"data_process/data/images/{safe_filename}"
    content = await video.read()
    if not content:
        raise ValueError("Uploaded video is empty.")
    with open(file_path, "wb") as f:
        f.write(content)
    return file_path


async def _save_uploaded_image(image: UploadFile) -> str:
    os.makedirs("data_process/data/predict_images", exist_ok=True)
    safe_filename = Path(image.filename or "uploaded_image.jpg").name
    file_path = f"data_process/data/predict_images/{safe_filename}"
    content = await image.read()
    if not content:
        raise ValueError("Uploaded image is empty.")
    with open(file_path, "wb") as f:
        f.write(content)
    return file_path


def _upload_video_to_s3(
    video: UploadFile,
    name: str,
    reg_no: str,
) -> Dict[str, str]:
    bucket_name = os.getenv("AWS_S3_BUCKET_NAME") or os.getenv("S3_BUCKET_NAME")
    if not bucket_name:
        raise RuntimeError(
            "S3 bucket is not configured. Set AWS_S3_BUCKET_NAME (or S3_BUCKET_NAME)."
        )

    region_name = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    s3_client = (
        boto3.client("s3", region_name=region_name)
        if region_name
        else boto3.client("s3")
    )

    original_filename = video.filename or "uploaded_video.mp4"
    extension = Path(original_filename).suffix or ".mp4"
    safe_name = _sanitize_s3_component(name, "unknown")
    safe_reg_no = _sanitize_s3_component(reg_no, "unknown")
    s3_filename = f"{safe_name}_{safe_reg_no}{extension}"
    s3_prefix = (os.getenv("AWS_S3_VIDEO_PREFIX") or "uploaded_videos").strip("/")
    object_key = f"{s3_prefix}/{s3_filename}" if s3_prefix else s3_filename

    extra_args = {}
    if video.content_type:
        extra_args["ContentType"] = video.content_type

    try:
        video.file.seek(0)
        if extra_args:
            s3_client.upload_fileobj(
                video.file, bucket_name, object_key, ExtraArgs=extra_args
            )
        else:
            s3_client.upload_fileobj(video.file, bucket_name, object_key)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to upload video to S3: {exc}") from exc
    finally:
        video.file.seek(0)

    return {
        "s3_bucket": bucket_name,
        "s3_object_key": object_key,
        "s3_uri": f"s3://{bucket_name}/{object_key}",
    }


def _extract_embeddings(file_path: str) -> Tuple[int, list]:
    image_service = ImageService(file_path)
    faces = image_service.reg_face()
    embeddings = image_service.face_embeddings(faces)
    final_embeddings = [embedding for embedding in embeddings if embedding is not None]
    if not final_embeddings:
        raise ValueError("No valid face embeddings could be generated from the video.")
    return len(faces), final_embeddings


def _extract_embeddings_from_s3(bucket_name: str, object_key: str) -> Tuple[int, list]:
    region_name = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    s3_client = (
        boto3.client("s3", region_name=region_name)
        if region_name
        else boto3.client("s3")
    )

    suffix = Path(object_key).suffix or ".mp4"
    temp_path = None
    try:
        with NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = temp_file.name
        s3_client.download_file(bucket_name, object_key, temp_path)
        return _extract_embeddings(temp_path)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to read video from S3: {exc}") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def _extract_first_embedding_from_image(image_path: str) -> Tuple[int, object]:
    image_service = ImageService(image_path)
    faces = image_service.reg_face_from_image(image_path)
    if not faces:
        raise ValueError("No faces detected in image.")
    embeddings = image_service.face_embeddings(faces)
    final_embeddings = [embedding for embedding in embeddings if embedding is not None]
    if not final_embeddings:
        raise ValueError("No valid embeddings generated for detected faces.")
    return len(faces), final_embeddings[0]


async def _process_video_and_store_embeddings(
    name: str,
    reg_no: str,
    timestamp: Optional[str],
    video: UploadFile,
) -> Dict:
    bucket_name = os.getenv("AWS_S3_BUCKET_NAME") or os.getenv("S3_BUCKET_NAME")
    if bucket_name:
        s3_upload_result = await asyncio.to_thread(
            _upload_video_to_s3,
            video,
            name,
            reg_no,
        )
        faces_detected, embeddings = await asyncio.to_thread(
            _extract_embeddings_from_s3,
            s3_upload_result["s3_bucket"],
            s3_upload_result["s3_object_key"],
        )
        storage_info = {
            "storage_mode": "s3",
            "s3_bucket": s3_upload_result["s3_bucket"],
            "s3_object_key": s3_upload_result["s3_object_key"],
            "s3_uri": s3_upload_result["s3_uri"],
            "file_path": None,
        }
    else:
        file_path = await _save_uploaded_video(video)
        faces_detected, embeddings = await asyncio.to_thread(
            _extract_embeddings, file_path
        )
        storage_info = {
            "storage_mode": "local",
            "s3_bucket": None,
            "s3_object_key": None,
            "s3_uri": None,
            "file_path": file_path,
        }

    saved_embeddings = await asyncio.to_thread(
        embedding_db.insert_embeddings,
        student_name=name,
        reg_no=reg_no,
        captured_at=timestamp,
        embeddings=embeddings,
    )
    return {
        **storage_info,
        "faces_detected": faces_detected,
        "embeddings_saved": saved_embeddings,
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/upload_video")
async def upload_video(
    name: str,
    reg_no: str,
    timestamp: Optional[str] = None,
    video: UploadFile = File(...),
):
    """
    Upload a student video, crop faces, generate embeddings, and store them in DB.
    """
    try:
        result = await _process_video_and_store_embeddings(
            name=name,
            reg_no=reg_no,
            timestamp=timestamp,
            video=video,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "message": "Video processed successfully",
        "storage_mode": result.get("storage_mode"),
        "s3_bucket": result.get("s3_bucket"),
        "s3_object_key": result.get("s3_object_key"),
        "s3_uri": result.get("s3_uri"),
        "file_path": result.get("file_path"),
        "faces_detected": result["faces_detected"],
        "embeddings_saved": result["embeddings_saved"],
    }


@app.post("/upload_video_and_train")
async def upload_video_and_train(
    name: str,
    reg_no: str,
    timestamp: Optional[str] = None,
    video: UploadFile = File(...),
):
    """
    Upload a student video and run the full pipeline:
    crop faces -> embeddings -> save to DB -> train classifier.
    """
    try:
        result = await _process_video_and_store_embeddings(
            name=name,
            reg_no=reg_no,
            timestamp=timestamp,
            video=video,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        train_result = await asyncio.to_thread(
            train_classifier,
            db_path=embedding_db.db_path,
            output_dir="model/artifacts",
        )
        training_status = "trained"
    except ValueError as exc:
        # This commonly happens until enough students/classes are in DB.
        train_result = {"reason": str(exc)}
        training_status = "skipped"

    return {
        "message": "Video processed, embeddings saved, and training step finished",
        "storage_mode": result.get("storage_mode"),
        "s3_bucket": result.get("s3_bucket"),
        "s3_object_key": result.get("s3_object_key"),
        "s3_uri": result.get("s3_uri"),
        "file_path": result.get("file_path"),
        "faces_detected": result["faces_detected"],
        "embeddings_saved": result["embeddings_saved"],
        "training_status": training_status,
        "training_result": train_result,
    }


@app.post("/predict_person")
async def predict_person(
    image: UploadFile = File(...),
    threshold: float = 0.0,
    similarity_threshold: Optional[float] = None,
    top_k: int = 3,
):
    """
    Upload one image and predict the student using trained classifier artifacts.
    """
    try:
        image_path = await _save_uploaded_image(image)
        faces_detected, embedding = await asyncio.to_thread(
            _extract_first_embedding_from_image, image_path
        )
        prediction = await asyncio.to_thread(
            predict_from_embedding,
            embedding,
            "model/artifacts",
            top_k,
            threshold,
            similarity_threshold,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Prediction completed",
        "faces_detected": faces_detected,
        "prediction": prediction,
    }
