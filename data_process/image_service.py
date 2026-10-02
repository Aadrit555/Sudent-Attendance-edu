from typing import List 
import os 
import cv2 
from data_process.video_service import VideoService 
from PIL import Image 
from facenet_pytorch import MTCNN , InceptionResnetV1
import torch 
import numpy as np 

class ImageService: 
    def __init__(self , vid_path : str) -> None : 
        self.vid_path = vid_path
        self.video = VideoService(self.vid_path) 
        self.mtcnn = MTCNN(device='cuda' if torch.cuda.is_available() else 'cpu') 
    
    def _crop_faces_from_rgb_frame(self, rgb_frame: np.ndarray) -> List:
        face_images = []
        img = Image.fromarray(rgb_frame)
        boxes, probs = self.mtcnn.detect(img)
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob < 0.9:
                    continue
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(rgb_frame.shape[1], x2)
                y2 = min(rgb_frame.shape[0], y2)
                face = rgb_frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                face_images.append(Image.fromarray(face))
        return face_images

    def reg_face(self) -> List:
        frames = self.video.process_vid_to_img() 
        if len(frames) == 0:
            raise ValueError("No frames from the video.")
        face_images = []  # store cropped faces
        for frame in frames:
            # Convert BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_images.extend(self._crop_faces_from_rgb_frame(rgb_frame))

        if len(face_images) == 0:
            print("No faces detected in any frame.")

        return face_images

    def reg_face_from_image(self, image_path: str) -> List:
        rgb_frame = None

        # First try OpenCV decode.
        frame = cv2.imread(image_path)
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Fallback to PIL decode for files OpenCV fails to read.
        if rgb_frame is None:
            try:
                with Image.open(image_path) as pil_img:
                    rgb_frame = np.array(pil_img.convert("RGB"))
            except Exception as exc:
                raise ValueError(
                    "Image not opened. Ensure the uploaded file is a valid image."
                ) from exc

        face_images = self._crop_faces_from_rgb_frame(rgb_frame)
        if len(face_images) == 0:
            print("No faces detected in image.")
        return face_images
    
    def face_embeddings(self , face_images : List) -> List: 
        """
            This is the function to get the face embeddings from the cropped face images 
        """
        if len(face_images) == 0:
            raise ValueError("No face images provided for embedding.") 
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        model = InceptionResnetV1(pretrained='vggface2').eval().to(device) 

        embeddings = [] 

        for img in face_images: 
            img = img.resize((160, 160))
            img_tensor = torch.tensor(np.array(img)).float()
            img_tensor = (img_tensor - 127.5) / 128.0
            img_tensor = img_tensor.permute(2, 0, 1)
            img_tensor = img_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model(img_tensor)

            embeddings.append(emb.squeeze().cpu().numpy())

        return embeddings
        

if __name__ == "__main__": 
    import sys
    test_video = sys.argv[1] if len(sys.argv) > 1 else "IMG_3352.MOV"
    if os.path.exists(test_video):
        ImageService(test_video).reg_face()
    else:
        print(f"Video file '{test_video}' not found.") 
