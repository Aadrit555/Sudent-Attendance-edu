import asyncio
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile

from data_process.db_service import EmbeddingDBService
from data_process.image_service import ImageService
from model.predict import predict_from_embedding
from model.train import train_classifier

app = FastAPI()
embedding_db = EmbeddingDBService()


async def _save_uploaded_video(video: UploadFile) -> str:
    os.makedirs("data_process/data/images", exist_ok=True)
    safe_filename = Path(video.filename or "uploaded_video.mp4").name
    file_path = f"data_process/data/images/{safe_filename}"
    with open(file_path, "wb") as f:
        f.write(await video.read())
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


def _extract_embeddings(file_path: str) -> Tuple[int, list]:
    image_service = ImageService(file_path)
    faces = image_service.reg_face()
    embeddings = image_service.face_embeddings(faces)
    final_embeddings = [embedding for embedding in embeddings if embedding is not None]
    if not final_embeddings:
        raise ValueError("No valid face embeddings could be generated from the video.")
    return len(faces), final_embeddings


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
    file_path = await _save_uploaded_video(video)
    faces_detected, embeddings = await asyncio.to_thread(_extract_embeddings, file_path)
    saved_embeddings = await asyncio.to_thread(
        embedding_db.insert_embeddings,
        student_name=name,
        reg_no=reg_no,
        captured_at=timestamp,
        embeddings=embeddings,
    )
    return {
        "file_path": file_path,
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

    return {
        "message": "Video processed successfully",
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
