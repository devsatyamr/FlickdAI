import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path='models/best.pt'):
        self.model = YOLO(model_path)
        # Define fashion category names
        self.class_names = {
            0: "top",
            1: "dress",
            2: "pants",
            3: "skirt",
            4: "outerwear",
            5: "shoes",
            6: "bag",
            7: "accessories"
            # Add more classes based on your YOLO model
        }
        self.confidence_threshold = 0.3

    def detect(self, image):
        """Detect fashion objects in image with proper class mapping"""
        results = self.model(image, conf=self.confidence_threshold)
        # Map class indices to fashion category names
        for r in results:
            boxes = []
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.class_names.get(cls_id, f"class{cls_id}")
                conf = float(box.conf[0].item())
                # Only include if confidence above threshold
                if conf >= self.confidence_threshold:
                    box.cls_name = cls_name
                    boxes.append(box)
            r.boxes = boxes
        return results

def extract_frames(video_path, interval=30, max_frames=20):
    """Extract frames from video with smart sampling
    
    Args:
        video_path: Path to video file
        interval: Frame interval for extraction
        max_frames: Maximum number of frames to extract
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Calculate optimal interval to get max_frames
    if total_frames > max_frames * interval:
        interval = total_frames // max_frames

    frames = []
    frame_num = 0
    prev_frame = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_num % interval == 0:
            # Skip if too similar to previous frame
            if prev_frame is not None:
                diff = cv2.absdiff(frame, prev_frame)
                if np.mean(diff) < 10:  # Threshold for difference
                    frame_num += 1
                    continue
                    
            frames.append((frame_num, frame))
            prev_frame = frame.copy()
            
            # Stop if we have enough frames
            if len(frames) >= max_frames:
                break
                
        frame_num += 1
        
    cap.release()
    return frames