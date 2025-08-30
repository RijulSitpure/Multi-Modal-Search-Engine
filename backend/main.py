# backend/main.py
# Reverted to a stable version with static file serving and two search endpoints.

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import os
import io
import numpy as np

from models.clip_model import ClipEmbeddings
from PIL import Image
from langchain_community.vectorstores import FAISS

# --- 1. FastAPI Setup & Global Objects ---

app = FastAPI(title="Multi-Modal Search Engine API")

# Mount the static directory to serve images
app.mount(
    "/static/images",
    StaticFiles(directory=os.path.join("..", "data", "images")),
    name="static_images"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBEDDINGS_PATH = os.path.join("..", "embeddings", "faiss_index")
CLIP_EMBEDDER = None
DB = None

# --- 2. API Request and Response Models ---

class SearchResult(BaseModel):
    caption: str
    image_url: str
    score: float

class TextSearchRequest(BaseModel):
    query: str
    k: int = 3

# --- 3. API Endpoints ---

@app.on_event("startup")
async def startup_event():
    """Load the model and FAISS index at startup."""
    global CLIP_EMBEDDER, DB
    print("Loading CLIP embedder and FAISS index...")
    try:
        CLIP_EMBEDDER = ClipEmbeddings()
        if os.path.exists(EMBEDDINGS_PATH):
            DB = FAISS.load_local(
                EMBEDDINGS_PATH,
                CLIP_EMBEDDER,
                allow_dangerous_deserialization=True
            )
            print(f"FAISS index loaded successfully from {EMBEDDINGS_PATH}")
        else:
            print(f"Error: FAISS index not found at {EMBEDDINGS_PATH}")
            DB = None
    except Exception as e:
        print(f"An error occurred during startup: {e}")
        DB = None

    if DB is None or CLIP_EMBEDDER is None:
        raise RuntimeError("Failed to load model or FAISS index on startup.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Multi-Modal Search Engine API!"}

@app.post("/search/text", response_model=List[SearchResult])
async def search_text(req: TextSearchRequest, http_request: Request):
    """Search for images using a text query."""
    print(f"Received text search request: query='{req.query}', k={req.k}")
    if not DB:
        raise HTTPException(status_code=503, detail="Vector database is not available.")

    results_with_scores = DB.similarity_search_with_score(req.query, k=req.k)
    
    output = []
    base_url = str(http_request.base_url)
    for doc, score in results_with_scores:
        image_path = doc.metadata.get("image_path", "not_found")
        filename = os.path.basename(image_path)
        output.append(SearchResult(
            caption=doc.page_content,
            image_url=f"{base_url}static/images/{filename}",
            score=score
        ))
    return output

@app.post("/search/image", response_model=List[SearchResult])
async def search_image(http_request: Request, k: int = Form(3), file: UploadFile = File(...)):
    """Search for similar images using an uploaded image."""
    print(f"Received image search request: k={k}")
    if not DB or not CLIP_EMBEDDER:
        raise HTTPException(status_code=503, detail="Vector database or embedder is not available.")

    try:
        contents = await file.read()
        query_image = Image.open(io.BytesIO(contents))
        query_image = query_image.convert("RGB")
        query_embedding = CLIP_EMBEDDER.embed_image([query_image])[0]
        
        query_vec_np = np.array([query_embedding], dtype=np.float32)
        distances, indices = DB.index.search(query_vec_np, k)
        
        output = []
        base_url = str(http_request.base_url)
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            docstore_id = DB.index_to_docstore_id.get(int(idx))
            if docstore_id:
                doc = DB.docstore.search(docstore_id)
                score = float(distances[0][i])
                if doc:
                    image_path = doc.metadata.get("image_path", "not_found")
                    filename = os.path.basename(image_path)
                    output.append(SearchResult(
                        caption=doc.page_content,
                        image_url=f"{base_url}static/images/{filename}",
                        score=score
                    ))
        return output
    except Exception as e:
        print(f"ERROR during image search: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during image processing: {e}")

