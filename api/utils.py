# Utility functions for the backend (frame saving, image cropping, etc.) 

import shutil
import numpy as np

def save_upload_file(upload_file, destination_path):
    with open(destination_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)


def crop_image(image: np.ndarray, bbox):
    """
    Crop an image (numpy array) using bbox = (x, y, w, h)
    """
    x, y, w, h = bbox
    return image[y:y+h, x:x+w] 

def detect_dominant_color(image):
    # Implementation of detect_dominant_color function
    pass

def rgb_to_color_name(rgb):
    # Implementation of rgb_to_color_name function
    pass 