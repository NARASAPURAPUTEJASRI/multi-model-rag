"""
config.py

Stores project configuration.

Thresholds are normalized confidence scores:
0.70 = 70% confidence
0.80 = 80% confidence
"""

import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")
MEDIA_DIR = os.getenv("MEDIA_DIR", "./data/media")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

TEXT_CONFIDENCE_THRESHOLD = float(os.getenv("TEXT_CONFIDENCE_THRESHOLD", "0.70"))
IMAGE_CONFIDENCE_THRESHOLD = float(os.getenv("IMAGE_CONFIDENCE_THRESHOLD", "0.80"))
AUDIO_CONFIDENCE_THRESHOLD = float(os.getenv("AUDIO_CONFIDENCE_THRESHOLD", "0.80"))
VIDEO_CONFIDENCE_THRESHOLD = float(os.getenv("VIDEO_CONFIDENCE_THRESHOLD", "0.80"))
MIXED_CONFIDENCE_THRESHOLD = float(os.getenv("MIXED_CONFIDENCE_THRESHOLD", "0.80"))