# frontend/app.py
# Reverted to a stable version with Text and Full Image Search.

import streamlit as st
import requests

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(layout="wide")
st.title("🤖 Multi-Modal Search Engine 🧠")
st.markdown("Built with `CLIP`, `LangChain`, `FAISS`, and `FastAPI`.")

# --- Helper Function ---
def display_results(results):
    """Helper function to display search results in columns."""
    if not results:
        st.warning("No results found.")
        return

    num_results = len(results)
    cols = st.columns(num_results)

    for i, r in enumerate(results):
        with cols[i]:
            st.markdown(f"**Score:** `{r['score']:.2f}`")
            # Streamlit can directly display images from URLs
            st.image(r['image_url'], caption=r['caption'], use_column_width=True)


# --- UI Tabs ---
tabs = st.tabs(["**📝 Text Search**", "**🖼️ Full Image Search**"])

# --- Text Search Tab ---
with tabs[0]:
    st.header("Find Images with Text")
    text_query = st.text_input("Enter your search query:", value="a blue block", key="text_query_input")
    k_text = st.slider("Number of results", 1, 5, 3, key="text_k")

    if st.button("Search with Text 🚀", key="text_search_button"):
        if not text_query.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching..."):
                try:
                    payload = {"query": text_query, "k": k_text}
                    resp = requests.post(f"{API_BASE_URL}/search/text", json=payload)
                    resp.raise_for_status()
                    results = resp.json()
                    st.subheader("Search Results")
                    display_results(results)
                except requests.exceptions.RequestException as e:
                    st.error(f"Search failed. Is the backend running? Error: {e}")

# --- Image Search Tab ---
with tabs[1]:
    st.header("Find Similar Images")
    uploaded_image_full = st.file_uploader("Upload an image to find similar ones", type=["png", "jpg", "jpeg"], key="full_image_uploader")
    k_image = st.slider("Number of results", 1, 5, 3, key="img_k")

    if st.button("Search with Image 🚀", key="full_image_search_button"):
        if uploaded_image_full:
            files = {"file": (uploaded_image_full.name, uploaded_image_full.getvalue(), uploaded_image_full.type)}
            form_data = {"k": k_image}
            
            with st.spinner("Embedding image and searching..."):
                try:
                    resp = requests.post(f"{API_BASE_URL}/search/image", files=files, data=form_data)
                    resp.raise_for_status()
                    results = resp.json()

                    st.subheader("Your Query Image:")
                    st.image(uploaded_image_full, width=200)
                    st.subheader("Search Results:")
                    display_results(results)
                except requests.exceptions.RequestException as e:
                    st.error(f"Search failed. Is the backend running? Error: {e}")
        else:
            st.warning("Please upload an image first.")

