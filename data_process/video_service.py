from typing import List
import os 
import cv2

class VideoService:
    def __init__(self, video_dir: str):
        self.video_dir = video_dir

    def process_vid_to_img(self): 
        """
            This is the function to process the video to images 
            This function helps for training the model with images 
        """
        cap = cv2.VideoCapture(self.video_dir) 
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
                if frame_count % 15 == 0:
                    img.append(frame) 
                    frame_count += 1 
                # os.makedirs("./data/images", exist_ok=True)
                    cv2.imwrite(f'./data/images/frame_{frame_count}.jpg', frame)
                else : 
                    frame_count += 1 
        cap.release()

if __name__ == "__main__": 
    VideoService(video_dir="/Users/muthuamuthan/Documents/img_reg/data_process/5b918baf-dfce-4897-8892-f8771cdcddc6.MP4").process_vid_to_img()