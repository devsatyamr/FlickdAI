# Backend MVP for FlickdAI Hackathon


---

## 🚀 Overview
FlickdAI is an AI-powered backend MVP designed for rapid, accurate fashion product detection and vibe analysis from video content. Built for the FlickdAI Hackathon, this backend leverages state-of-the-art computer vision (YOLO), similarity search (FAISS), and FastAPI for a seamless, scalable solution.

---

## 🗂️ Filesystem & Directory Map

```
Flickd/
├── api/           # FastAPI app, detection, and core logic
│   ├── api.py         # Main FastAPI application
│   ├── detection.py   # YOLO-based detection logic
│   ├── matching.py    # Product matching using FAISS
│   ├── vibe.py        # Vibe classification
│   ├── utils.py       # Helper utilities
│   └── __pycache__/   # Python cache files
├── data/          # Data for similarity search
│   ├── faiss_index.bin    # FAISS index for fast product search
│   └── product_ids.npy    # Numpy array of product IDs (maps index to product)
├── frames/        # Extracted frames from uploaded videos
├── models/        # YOLO model weights and training artifacts
│   └── best.pt        # Trained YOLO model
├── outputs/       # JSON outputs for processed videos
├── videos/        # Organizer-provided video clips
├── requirements.txt   # Python dependencies
├── venv/          # Virtual environment (recommended)
├── README.md      # This documentation
├── demo.mp4       # Demo video of the full MVP process
└── vibeslist.json # List of vibes/categories
```

---

## ⚡ Quick Setup

1. **Clone the repository**
   ```sh
   git clone https://github.com/devsatyamr/FlickdAI
   cd FlickdAI
   ```
2. **Create and activate a virtual environment**
   ```sh
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   # Or
   source venv/bin/activate  # On Mac/Linux
   ```
3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the FastAPI server**
   ```sh
   uvicorn api.api:app --reload
   ```
5. **Upload a video for processing**
   - Use the `/process_video/` endpoint (e.g., via Swagger UI at `http://127.0.0.1:8000/docs`)
   - Upload your `.mp4` file and (optionally) a caption
   - Download the JSON output from the response or from the `outputs/` folder

---

## 🏗️ Working Process (End-to-End)

1. **Image Dataset Downloading**
   - Gather a large, labeled fashion image dataset for training the YOLO model.
2. **YOLO Model Training**
   - Train YOLO on the dataset to detect fashion categories (e.g., tops, dresses, shoes).
   - Save the trained weights as `models/best.pt`.
3. **Product Embedding & FAISS Indexing**
   - Extract feature embeddings for each product image using a suitable model.
   - Build a FAISS index (`data/faiss_index.bin`) for fast similarity search.
   - Store product IDs in `data/product_ids.npy` to map FAISS results to real products.
4. **FastAPI Backend**
   - Handles video uploads, frame extraction, detection, product matching, and vibe classification.
5. **Video Processing Pipeline**
   - Video is uploaded → frames are extracted (smart sampling) → YOLO detects products in each frame.
   - Each detected product is cropped, its dominant color is analyzed, and an embedding is generated.
   - FAISS index is queried for similar products; results are scored for confidence.
   - Vibe classifier analyzes the caption and detected products to assign a vibe.
6. **Output**
   - JSON output includes detected products, their types, colors, match confidence, and vibe.
   - Output is downloadable or viewable via API.

---

## 📦 Folder/Component Roles

- **api/**: All backend logic (FastAPI app, detection, matching, vibe analysis)
- **data/**: FAISS index and product ID mapping for similarity search
- **frames/**: Temporary storage for frames extracted from videos
- **models/**: YOLO model weights for detection
- **outputs/**: JSON results for each processed video
- **videos/**: Raw video clips for processing
- **venv/**: Python virtual environment (prevents dependency conflicts)

---

## 🎬 Demo Process

1. Upload a video via the `/process_video/` endpoint.
2. Frames are extracted and analyzed for fashion products.
3. Each product is matched to the closest item in the FAISS index.
4. Vibe is classified based on caption and detected products.
5. Download the structured JSON output with all results.

Demo Link : https://www.loom.com/share/c2993e5751c541b599d95354e1699286
---

## 📝 Example API Usage

- **Start the server:**
  ```sh
  uvicorn api.api:app --reload
  ```
- **Upload a video:**
  - Go to `http://127.0.0.1:8000/docs`
  - Use `/process_video/` to upload your `.mp4` file
- **Get results:**
  - Download the JSON output from the API response or from the `outputs/` folder

---

## 🧠 Key Concepts
- **YOLO Model:** Detects and classifies fashion items in video frames.
- **FAISS Index:** Enables fast, scalable similarity search for product matching.
- **Product IDs:** Map FAISS search results to real product catalog entries.
- **Vibe Classifier:** Assigns a vibe to the video based on caption and detected products.
- **Confidence Score:** Indicates how closely a detected product matches the catalog.

---

## 👨‍💻 Author & Copyright

**Devsatyam Ray**  
[LinkedIn](https://linkedin.com/in/devsatyamr)  
devsatyamr@gmail.com

---

<p align="center">
  <b>Made with ❤️ for the FlickdAI Hackathon</b>
</p>
