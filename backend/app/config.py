import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")
MEDIA_DIR = os.getenv("MEDIA_DIR", "./data/media")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))

EMBEDDING_MODEL = "models/gemini-embedding-2"
LLM_MODEL = "gemini-2.5-flash"