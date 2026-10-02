from typing import List
import os 
import cv2
import numpy as np 

class VideoService:
    def __init__(self, video_dir: str):
        self.video_dir = video_dir

    def process_vid_to_img(self) -> List: 
        """
            This is the function to process the video to images 
            This function helps for training the model with images 
        """
        cap = cv2.VideoCapture(self.video_dir) 
        if not cap.isOpened():
            raise ValueError("Video not opened. Check path or file.")
        frame_count = 0
        img = []
        os.makedirs("./data/images", exist_ok=True)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) 
        duration = frames / fps 
        print(f"Frames : {frames}")
        print(f"fps : {fps}")
        print(f"duration : {duration}")
        while True:
            # img = []
            sucess , frame = cap.read() 
            if not sucess: 
                break 
            else : 
                # Taking every 15 frames 
                if frame_count % 30 == 0:
                    img.append(frame) 
                    frame_count += 1 
                # os.makedirs("./data/images", exist_ok=True)
                    #cv2.imwrite(f'./data/images/frame_{frame_count}.jpg', frame)
                else : 
                    frame_count += 1 
        cap.release()
        print(f"Extracted {len(img)} frames")
        return img 

    def upload_video_to_s3(self, video_path: str, bucket_name: str, object_name: str):
        """
        Upload the video to an S3 bucket.
        """
        import boto3
        s3_client = boto3.client("s3")
        s3_client.upload_file(video_path, bucket_name, object_name)


if __name__ == "__main__":
    sample_path = os.getenv("SAMPLE_VIDEO_PATH", "data_process/5b918baf-dfce-4897-8892-f8771cdcddc6.MP4")
    if os.path.exists(sample_path):
        VideoService(video_dir=sample_path).process_vid_to_img()