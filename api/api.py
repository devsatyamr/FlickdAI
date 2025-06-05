from fastapi import FastAPI, UploadFile, File, HTTPException
from api.detection import ObjectDetector, extract_frames
from api.matching import ProductMatcher
from api.vibe import VibeClassifier
from api.utils import save_upload_file, crop_image, detect_dominant_color, rgb_to_color_name
import os
import numpy as np
import uuid
from typing import List
import cv2

app = FastAPI()

# Initialize paths
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
faiss_index_path = os.path.join(data_dir, 'faiss_index.bin')
product_ids_path = os.path.join(data_dir, 'product_ids.npy')

try:
    # Load product IDs and initialize components
    product_ids = np.load(product_ids_path, allow_pickle=True)
    matcher = ProductMatcher(faiss_index_path, None, product_ids)
    detector = ObjectDetector()
    vibe_classifier = VibeClassifier()
except Exception as e:
    print(f"Error initializing components: {str(e)}")
    raise

@app.post("/process_video/")
async def process_video(video: UploadFile = File(...), caption: str = ""):
    try:
        # 1. Save uploaded video
        video_id = str(uuid.uuid4())
        video_path = os.path.join(data_dir, f"{video_id}.mp4")
        save_upload_file(video, video_path)

        # 2. Extract frames with smart sampling
        frames = extract_frames(video_path, interval=30, max_frames=20)
        if not frames:
            raise HTTPException(status_code=400, detail="Could not extract frames from video")

        # Save extracted frames to frames folder
        frames_dir = os.path.join(os.path.dirname(__file__), '..', 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        for frame_num, frame in frames:
            frame_path = os.path.join(frames_dir, f"{video_id}_frame{frame_num}.jpg")
            cv2.imwrite(frame_path, frame)

        # 3. Process frames
        products = []
        seen_products = set()  # Track unique products to avoid duplicates
        
        for frame_num, frame in frames:
            # Detect objects
            results = detector.detect(frame)
            if not results:
                continue
                
            for r in results[0].boxes:
                try:
                    # Get coordinates and class info
                    x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
                    w, h = x2 - x1, y2 - y1
                    bbox = (x1, y1, w, h)
                    
                    # Skip if detection confidence is too low
                    conf = float(r.conf[0].item())
                    if conf < detector.confidence_threshold:
                        continue
                        
                    class_name = r.cls_name
                    
                    # Get crop and analyze
                    crop = crop_image(frame, bbox)
                    if crop is None or crop.size == 0:
                        continue
                        
                    color = detect_dominant_color(crop)
                    
                    # Match product
                    try:
                        print(f"\nProcessing {class_name} in {color}")
                        
                        # Get embedding and normalize
                        emb = matcher.get_embedding(crop)
                        if emb is None:
                            print("Failed to generate embedding")
                            raise ValueError("Failed to generate embedding")
                            
                        # Match and get similarity scores
                        print("Running FAISS search...")
                        scores, matched_ids = matcher.match(emb, top_k=1)
                        
                        # Convert similarity score and adjust thresholds
                        sim = float(scores[0])
                        print(f"Product match result - Score: {sim:.3f}, ID: {matched_ids[0] if matched_ids else 'none'}")
                        
                        if sim > 0.25:  # Further lowered from 0.35
                            match_type = "exact"
                        elif sim > 0.15:  # Further lowered from 0.25
                            match_type = "similar"
                        else:
                            match_type = "none"
                            
                        matched_product_id = str(matched_ids[0]) if match_type != "none" else None
                        
                        # Create unique key for product
                        product_key = f"{class_name}_{color}_{matched_product_id}"
                        if product_key in seen_products:
                            continue
                            
                        seen_products.add(product_key)
                        
                    except Exception as e:
                        print(f"Product matching error: {str(e)}")
                        match_type = "none"
                        matched_product_id = None
                        sim = 0.0
                    
                    products.append({
                        "type": class_name,
                        "color": color,
                        "match_type": match_type,
                        "matched_product_id": matched_product_id,
                        "confidence": float(round(sim, 2))
                    })
                    
                except Exception as e:
                    print(f"Error processing detection: {str(e)}")
                    continue        # 4. Vibe classification
        vibes = vibe_classifier.classify(caption, products=products)

        # 5. Prepare output
        output = {
            "video_id": video_id,
            "vibes": vibes,
            "products": products
        }

        # Clean up
        os.remove(video_path)
        return output

    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        raise HTTPException(status_code=500, detail=str(e))

def detect_dominant_color(image):
    # Resize to speed up
    img = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
    img = img.reshape((-1, 3))
    # Convert to float and k-means
    img = np.float32(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, palette = cv2.kmeans(img, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    # Convert BGR to RGB
    dominant_rgb = tuple(int(c) for c in dominant[::-1])
    # Map to color name
    return rgb_to_color_name(dominant_rgb)

def rgb_to_color_name(rgb):
    # Simple mapping, you can expand this
    colors = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "purple": (128, 0, 128),
        "orange": (255, 165, 0),
        "gray": (128, 128, 128),
        "brown": (165, 42, 42),
        # Add more as needed
    }
    # Find closest color
    min_dist = float('inf')
    color_name = "unknown"
    for name, value in colors.items():
        dist = np.linalg.norm(np.array(rgb) - np.array(value))
        if dist < min_dist:
            min_dist = dist
            color_name = name
    return color_name