from fastapi import FastAPI , UploadFile , File
from typing import List , Optional
from pydantic import BaseModel 
import asyncio 
import os 
import boto3 
from data_process.image_service import ImageService 

app = FastAPI() 

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/upload_video")
async def upload_video(
    name : str ,
    reg_no : str , 
    timestamp : Optional[str] = None ,
    video : UploadFile = File(...) ,
):
    """
        Endpoint to upload the video of the user and save it to the S3 bucket.
    """ 
    os.makedirs("data_process/data/images", exist_ok=True)

    file_path = f"data_process/data/images/{video.filename}"

    # Save file
    with open(file_path, "wb") as f:
        f.write(await video.read())

    # Pass FILE PATH (not UploadFile)
    response = ImageService(file_path).reg_face()

    return {
        "message": "Video processed successfully",
        "faces_detected": len(response)
    }

