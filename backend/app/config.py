"""
config.py

This file stores all project-level configuration values.

It loads environment variables from the .env file and provides:
- Gemini API key
- ChromaDB storage path
- Media storage path
- Gemini embedding model name
- Gemini LLM model name
- Final threshold values for each modality

These values are used across the multimodal RAG pipeline.
"""

import os
from dotenv import load_dotenv

# Load variables from .env file into the application environment
load_dotenv()

# Gemini API key used for embeddings, media processing, and LLM calls
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Path where ChromaDB vector database is stored
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")

# Path where downloaded media files are stored
MEDIA_DIR = os.getenv("MEDIA_DIR", "./data/media")

# Gemini embedding model used for text, image, audio, and video embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")

# Gemini LLM model used for answer generation and media description
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# Final relevance threshold for text results
TEXT_FINAL_THRESHOLD = float(os.getenv("TEXT_FINAL_THRESHOLD", "0.07"))

# Final relevance threshold for image results
IMAGE_FINAL_THRESHOLD = float(os.getenv("IMAGE_FINAL_THRESHOLD", "0.08"))

# Final relevance threshold for audio results
AUDIO_FINAL_THRESHOLD = float(os.getenv("AUDIO_FINAL_THRESHOLD", "0.08"))

# Final relevance threshold for video results
VIDEO_FINAL_THRESHOLD = float(os.getenv("VIDEO_FINAL_THRESHOLD", "0.08"))

# Final relevance threshold for mixed multimodal results
MIXED_FINAL_THRESHOLD = float(os.getenv("MIXED_FINAL_THRESHOLD", "0.08"))