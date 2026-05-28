"""
embeddings.py

This file handles Gemini-based embedding and media description generation.

Main responsibilities:
- Create text embeddings
- Upload image/audio/video files to Gemini
- Create raw media embeddings
- Wait until uploaded media files become active
- Generate media descriptions for BM25 keyword search

Important design rule:
- Raw media files are used for semantic embeddings.
- Captions/descriptions are used only for BM25 keyword search and metadata.
"""

import time
import google.generativeai as genai
from app.config import GEMINI_API_KEY, EMBEDDING_MODEL, LLM_MODEL

# Configure Gemini API using API key from config.py
genai.configure(api_key=GEMINI_API_KEY)


def embed_text(text: str):
    # Creates semantic embedding for text query or text chunk

    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text
    )

    return result["embedding"]


def embed_media(file_path: str):
    # Creates semantic embedding directly from raw media file

    uploaded_file = genai.upload_file(file_path)

    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=uploaded_file
    )

    return result["embedding"]


def wait_for_file_active(uploaded_file, timeout=120):
    # Waits until Gemini finishes processing uploaded media file

    start_time = time.time()

    while time.time() - start_time < timeout:
        file = genai.get_file(uploaded_file.name)

        state = getattr(file, "state", None)

        # Continue only when file becomes active
        if str(state).lower().endswith("active"):
            return file

        time.sleep(5)

    # Return uploaded file even if timeout is reached
    return uploaded_file


def describe_media(file_path: str, modality: str):
    # Generates a short description for image/audio/video
    # This description is used for BM25 keyword search, not for media embedding

    uploaded_file = genai.upload_file(file_path)

    uploaded_file = wait_for_file_active(uploaded_file)

    model = genai.GenerativeModel(LLM_MODEL)

    # Prompt for image description
    if modality == "image":
        prompt = """
Describe this image in one clear sentence.
Mention important visible objects, place, person, text, or concept.
This description will be used for keyword search.
"""

    # Prompt for audio summary
    elif modality == "audio":
        prompt = """
Listen to this audio and summarize it in 2 to 3 short sentences.
Mention speech topic, sounds, speaker hints, or events if present.
This summary will be used for keyword search.
"""

    # Prompt for video summary
    elif modality == "video":
        prompt = """
Watch this video and summarize it in 2 to 3 short sentences.
Mention main scene, objects, actions, places, people, and events.
This summary will be used for keyword search.
"""

    # Fallback prompt for unknown media type
    else:
        prompt = "Describe this media briefly."

    response = model.generate_content([prompt, uploaded_file])

    return response.text.strip()