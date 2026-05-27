import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")
MEDIA_DIR = os.getenv("MEDIA_DIR", "./data/media")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

TEXT_FINAL_THRESHOLD = float(os.getenv("TEXT_FINAL_THRESHOLD", "0.07"))
IMAGE_FINAL_THRESHOLD = float(os.getenv("IMAGE_FINAL_THRESHOLD", "0.08"))
AUDIO_FINAL_THRESHOLD = float(os.getenv("AUDIO_FINAL_THRESHOLD", "0.08"))
VIDEO_FINAL_THRESHOLD = float(os.getenv("VIDEO_FINAL_THRESHOLD", "0.08"))
MIXED_FINAL_THRESHOLD = float(os.getenv("MIXED_FINAL_THRESHOLD", "0.08"))