from pydantic import BaseModel, Field
from typing import Optional 

class AttendanceCpatured(BaseModel):
    image_url: str = Field(..., description="URL of the captured image")
    timestamp: Optional[str] = Field(None, description="Timestamp of when the image was captured")