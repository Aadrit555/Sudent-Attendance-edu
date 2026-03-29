from typing import List 
import os 
import cv2 
from video_service import VideoService 
from PIL import Image 
from facenet_pytorch import MTCNN 
import torch 

class ImageService: 
    def __init__(self , vid_path : str) -> None : 
        self.vid_path = vid_path
        self.video = VideoService(self.vid_path) 
        self.mtcnn = MTCNN(device='cuda' if torch.cuda.is_available() else 'cpu') 
    def reg_face(self) -> List : 
        frames = self.video.process_vid_to_img() 
        #Checking the frames 
        if len(frames) == 0:
            raise ValueError("No frames from the video.")
        frame = frames[0] 
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, probs = self.mtcnn.detect(img) 
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                print(f"Detected face with probability {prob:.4f} at box {box}")
        else : 
            print("No faces detected.") 

if __name__ == "__main__": 
    ImageService("IMG_3352.MOV").reg_face() 