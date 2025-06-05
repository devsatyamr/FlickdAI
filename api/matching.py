import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
import cv2
from PIL import Image
from functools import lru_cache

class ProductMatcher:
    def __init__(self, faiss_index_path, product_embeddings=None, product_ids=None):
        try:
            # Load CLIP model and processor
            print("Initializing CLIP model and processor...")
            self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16')
            self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')
            
            # Load FAISS index
            print(f"Loading FAISS index from {faiss_index_path}")
            self.index = faiss.read_index(faiss_index_path)
            print(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
            if product_ids is None:
                raise ValueError("product_ids cannot be None")
            self.product_ids = product_ids
            print(f"Loaded {len(self.product_ids)} product IDs")
            
            # Enable GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            print(f"Using device: {self.device}")
            
        except Exception as e:
            print(f"Error initializing ProductMatcher: {str(e)}")
            raise

    def get_embedding(self, image):
        """Get CLIP embedding for image with caching"""
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Preprocess image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embedding
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                embedding = features.cpu().numpy().flatten()
            
            # Normalize embedding
            embedding = embedding.reshape(1, -1)
            faiss.normalize_L2(embedding)
            embedding = embedding.flatten()
            
            print(f"Generated embedding shape: {embedding.shape}, norm: {np.linalg.norm(embedding):.3f}")
            return embedding
            
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            return None

    def match(self, embedding, top_k=1):
        """Find similar products using FAISS"""
        try:
            if embedding is None:
                print("Error: embedding is None")
                return [0.0], []
                
            # Reshape and ensure normalized
            embedding = embedding.reshape(1, -1)
            norm = np.linalg.norm(embedding)
            print(f"Input embedding norm before normalization: {norm:.3f}")
            
            if norm > 0:
                embedding = embedding / norm
            
            # Search index
            D, I = self.index.search(embedding, top_k)
            print(f"Raw FAISS distances: {D[0]}")
            
            # Convert distances to cosine similarities
            similarities = 1 - D[0]/2  # Convert L2 distance to cosine similarity
            matched_product_ids = [self.product_ids[i] for i in I[0]]
            
            print(f"Similarities: {similarities}")
            print(f"Matched product IDs: {matched_product_ids}")
            
            return similarities, matched_product_ids
            
        except Exception as e:
            print(f"Error in matching: {str(e)}")
            return [0.0], []