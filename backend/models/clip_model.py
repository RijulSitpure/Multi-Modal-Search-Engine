# backend/models/clip_model.py
# This file defines the custom LangChain Embeddings class for the CLIP model.

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from langchain.embeddings.base import Embeddings
from typing import List

class ClipEmbeddings(Embeddings):
    """Custom LangChain embeddings class for the CLIP model."""
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"CLIP model loaded on device: {self.device}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts (captions)."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_embeddings = self.model.get_text_features(**inputs)
        return text_embeddings.cpu().numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single text query."""
        return self.embed_documents([text])[0]

    def embed_image(self, images: List[Image.Image]) -> List[List[float]]:
        """A custom method to embed a list of PIL Images."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            image_embeddings = self.model.get_image_features(**inputs)
        return image_embeddings.cpu().numpy().tolist()
