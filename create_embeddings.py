# create_embeddings.py
# Place this in the root directory of your project.
# Run this script once to create the FAISS index.

import os
from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
# Assuming clip_model.py is in backend/models/
from backend.models.clip_model import ClipEmbeddings

print("--- Starting Embedding Creation ---")

# --- 1. Define Paths ---
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
EMBEDDING_DIR = "embeddings"
FAISS_INDEX_PATH = os.path.join(EMBEDDING_DIR, "faiss_index")

# --- 2. Create Directories if they don't exist ---
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# --- 3. Prepare Your Data ---
# Create dummy image files for demonstration
print("Creating dummy image data...")
Image.new('RGB', (100, 100), color = 'red').save(os.path.join(IMAGE_DIR, 'red_square.jpg'))
Image.new('RGB', (100, 100), color = 'blue').save(os.path.join(IMAGE_DIR, 'blue_square.jpg'))
Image.new('RGB', (100, 100), color = 'green').save(os.path.join(IMAGE_DIR, 'green_square.jpg'))
print("Dummy images created in", IMAGE_DIR)

# Define image paths and captions
image_caption_pairs = [
    {"image_path": os.path.join(IMAGE_DIR, "red_square.jpg"), "caption": "a photo of a vibrant red square"},
    {"image_path": os.path.join(IMAGE_DIR, "blue_square.jpg"), "caption": "a clear image of a blue block"},
    {"image_path": os.path.join(IMAGE_DIR, "green_square.jpg"), "caption": "a simple green rectangle on a white background"},
]

# --- 4. Create LangChain Documents ---
print("Creating LangChain documents...")
documents = [
    Document(
        page_content=pair["caption"],
        metadata={"image_path": pair["image_path"]}
    )
    for pair in image_caption_pairs
]

# --- 5. Instantiate Embedder and Index Data ---
print("Initializing CLIP embeddings...")
clip_embedder = ClipEmbeddings()

print(f"Indexing documents into FAISS vector store at '{FAISS_INDEX_PATH}'...")
db = FAISS.from_documents(documents, clip_embedder)

# --- 6. Save the Index ---
db.save_local(FAISS_INDEX_PATH)
print("\nFAISS index created and saved successfully.")
print("--- Embedding Creation Finished ---")
