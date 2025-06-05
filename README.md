<!-- Flickd: AI-Powered Fashion Object Detection and Video Frame Analysis -->

<p align="center">
  <img src="https://user-images.githubusercontent.com/placeholder/flickd-logo.png" alt="Flickd Logo" width="200"/>
</p>

---

# Flickd: AI-Powered Fashion Object Detection & Video Frame Analysis

Flickd is a cutting-edge AI application for the fashion industry, designed to automatically detect, classify, and analyze fashion items in both images and videos. Built on top of the powerful YOLO object detection framework, Flickd enables smart frame extraction, robust object recognition, and seamless integration for fashion tech, e-commerce, and content analysis workflows.

---

## 🚀 Features

- **🎯 Fashion Object Detection:**
  - Detects and classifies fashion items (tops, dresses, pants, shoes, accessories, and more) in images and video frames using a YOLO-based deep learning model.
- **🧠 Smart Frame Extraction:**
  - Efficiently samples frames from videos, skipping near-duplicate frames to optimize for unique and relevant content.
- **🔗 Extensible Class Mapping:**
  - Easily update or expand the set of fashion categories by modifying a single dictionary in the code.
- **🛠️ Modular & Clean API:**
  - Well-structured Python modules for easy integration, extension, and maintenance.
- **📦 Ready for Integration:**
  - Designed for use in larger pipelines, web services, or as a standalone tool.

---

## 🗂️ Project Structure

```text
Flickd/
├── api/
│   ├── detection.py      # Object detection & frame extraction logic
│   ├── matching.py      # Matching logic for detected items (extendable)
│   ├── utils.py         # Utility functions
│   ├── vibe.py          # Vibe/category logic (extendable)
│   └── __pycache__/     # Compiled Python files
├── data/                # Data files (e.g., FAISS index, product IDs)
├── frames/              # Extracted frames & related metadata
├── models/              # YOLO model weights & training artifacts
├── requirements.txt     # Python dependencies
├── vibeslist.json       # List of vibes or categories
└── .gitignore           # Git ignore rules
```

---

## 🧩 How It Works

### 1. Object Detection
- The `ObjectDetector` class loads a YOLO model and defines a mapping from class indices to fashion categories.
- The `detect` method processes an image, returning detected objects with mapped class names and confidence scores.
- Easily extend the `class_names` dictionary to support new categories.

### 2. Frame Extraction
- The `extract_frames` function samples frames from a video at a specified interval, skipping frames that are too similar to the previous one (to avoid redundancy).
- Returns a list of `(frame number, frame)` tuples, up to a maximum number of frames.
- Smart difference thresholding ensures only unique frames are kept.

### 3. Customization
- **Add/Modify Fashion Categories:**
  - Update the `class_names` dictionary in `api/detection.py`.
- **Change Model Weights:**
  - Replace the YOLO weights in `models/best.pt` with your own trained model.
- **Integrate with Other Systems:**
  - Use the modular API to plug Flickd into web apps, batch pipelines, or research tools.

---

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd Flickd
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Download/Place YOLO model weights:**
   - Place your trained YOLO weights in `models/best.pt`.

---

## 💡 Usage Examples

### Detect Fashion Items in an Image
```python
from api.detection import ObjectDetector

detector = ObjectDetector('models/best.pt')
results = detector.detect('path/to/image.jpg')
for r in results:
    for box in r.boxes:
        print(f"Detected: {box.cls_name} (confidence: {box.conf[0].item():.2f})")
```

### Extract Unique Frames from a Video
```python
from api.detection import extract_frames

frames = extract_frames('path/to/video.mp4', interval=30, max_frames=20)
for frame_num, frame in frames:
    # Process or save frame
    pass
```

### Full Pipeline Example
```python
from api.detection import ObjectDetector, extract_frames

detector = ObjectDetector('models/best.pt')
frames = extract_frames('path/to/video.mp4')
for frame_num, frame in frames:
    results = detector.detect(frame)
    # Process results for each frame
```

---

## 📚 Documentation

### `api/detection.py`
- **ObjectDetector**
  - `__init__(model_path)`: Loads YOLO model from the given path.
  - `detect(image)`: Detects fashion objects in the image, returns results with class names and confidence.
- **extract_frames**
  - `extract_frames(video_path, interval=30, max_frames=20)`: Extracts unique frames from a video, skipping near-duplicates.

### Extending Flickd
- Add new detection logic, matching algorithms, or vibe analysis by creating new modules in the `api/` directory.
- Update `requirements.txt` for new dependencies.

---

## 🤝 Contributing

We welcome contributions! To get started:
- Fork the repository
- Create a new branch for your feature or bugfix
- Submit a pull request with a clear description

For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License
This project is for educational and hackathon use. For commercial or production use, please contact the author.

---

## 🙏 Acknowledgements
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV, NumPy
- Hackathon organizers and the open-source community

---

<p align="center">
  <em>For more details, see the code in the <code>api/</code> directory and the competition document.</em>
</p>
