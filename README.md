🧠 Multi-Modal AI Search Engine 🚀
A powerful, interactive search engine that goes beyond text. Search for images using natural language, find similar images by uploading your own, and even select specific objects within an image to search for. Built with a modern Python stack including CLIP, LangChain, FAISS, FastAPI, and Streamlit.

➡️ Live Demo Link (Coming Soon!)

✨ Features
📝 Text-to-Image Search: Type a description (e.g., "a red car on a sunny day") and get visually similar images.

🖼️ Image-to-Image Search: Upload an image to find the most visually similar images in the dataset.

- (Planned) 🎯 Region-Based Search ("Circle/Box to Search"): Upload an image, draw a box or circle around a specific object (like a pair of shoes in a full-body photo), and find images containing that specific object.

- (Planned) 🔊 Audio Search: Search for audio clips by uploading your own sample.

- (Planned) 🌐 Cross-Modal Search: Bridge the gap between modalities, enabling searches like text-to-audio.

🏛️ Architecture Overview
This project follows a decoupled frontend-backend architecture, ensuring scalability and separation of concerns.

Indexing (Offline Job): The create_embeddings.py script first processes a large dataset (e.g., COCO Captions). It uses the CLIP model to generate vector embeddings for every image and its caption. These embeddings are stored in a highly efficient FAISS vector index.

Frontend (Streamlit): The user interacts with a web interface built with Streamlit. It handles user input, such as text queries or file uploads.

Backend (FastAPI): The Streamlit frontend sends API requests to the FastAPI backend. The backend is responsible for:

Using the same CLIP model to embed the user's query (text or image) in real-time.

Searching the pre-computed FAISS index for the most similar vectors.

Returning a JSON response containing the search results, including URLs to the matching images.

Image Serving: The FastAPI server also serves the dataset images from a static directory, allowing the frontend to display them directly from a URL.

graph TD
    A[User] --> B{Frontend (Streamlit)};
    B -->|Text/Image Query| C{Backend (FastAPI)};
    C -->|Embed Query| D[CLIP Model];
    C -->|Search by Vector| E[FAISS Vector Index];
    E --> C;
    C -->|Image URLs| B;
    B -->|Displays Results| A;

    subgraph "Offline Indexing"
        F[COCO Dataset] --> G[create_embeddings.py];
        G --> D;
        G --> E;
    end

🛠️ Tech Stack
AI / Machine Learning:

PyTorch: Core deep learning framework.

transformers: For easy access to the pre-trained CLIP model from Hugging Face.

LangChain: For orchestrating interactions with the vector database and embedding models.

Vector Database:

faiss-cpu: For efficient similarity search on vector embeddings.

Backend:

FastAPI: High-performance Python web framework for building the API.

uvicorn: ASGI server to run the FastAPI application.

Frontend:

Streamlit: To rapidly create and deploy the interactive web UI.

streamlit-drawable-canvas (for region search): For the interactive image selection feature.

Data Handling:

Pillow: For image manipulation.

numpy: For numerical operations on embeddings.

🚀 Getting Started
Follow these instructions to set up and run the project locally.

1. Prerequisites
Python 3.9+

An environment manager like venv or conda.

2. Clone the Repository
git clone [https://github.com/RijulSitpure/Multi-Modal-Search-Engine](https://github.com/RijulSitpure/Multi-Modal-Search-Engine)
cd Multi-Modal-Search-Engine

3. Set Up the Python Environment
# Create a virtual environment
python -m venv multimodal_env

# Activate it
# On Windows:
.\multimodal_env\Scripts\activate
# On macOS/Linux:
source multimodal_env/bin/activate

4. Install Dependencies
Install all required libraries from the requirements.txt file.

pip install -r requirements.txt

5. Download the Dataset (Optional, for large-scale search)
To get the best results, download the COCO 2017 dataset.

Images (18 GB): 2017 Train images

Annotations (241 MB): 2017 Train/Val annotations

Unzip them and place them in the data/ folder according to the structure in create_embeddings.py.

6. Create the Vector Embeddings
Run the indexing script from the project's root directory. This will process your data and create the FAISS index in the embeddings/ folder.

python create_embeddings.py

Note: This will take a long time for the full COCO dataset. It is much faster with a GPU.

7. Run the Application
You need to run the backend and frontend servers in two separate terminals.

Terminal 1: Start the Backend

# Navigate to the backend directory
cd backend

# Start the FastAPI server
uvicorn main:app --reload

Terminal 2: Start the Frontend

# Make sure you are in the project's root directory
streamlit run frontend/app.py

Your browser should automatically open to the Streamlit application, ready for you to use!

📂 Project Structure
multimodal-search-engine/
│
├── backend/
│   ├── models/             # Contains the custom ClipEmbeddings class
│   │   └── clip_model.py
│   ├── __init__.py
│   └── main.py             # FastAPI application logic and API endpoints
│
├── frontend/
│   └── app.py              # Streamlit application UI and logic
│
├── data/                   # For storing raw images and annotation files
│
├── embeddings/             # Stores the generated FAISS vector index
│
├── create_embeddings.py    # Standalone script to generate the vector index
├── requirements.txt
└── README.md

LICENSE
This project is licensed under the MIT License. See the LICENSE file for details.