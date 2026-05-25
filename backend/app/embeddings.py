import time
import google.generativeai as genai
from app.config import GEMINI_API_KEY, EMBEDDING_MODEL, LLM_MODEL

genai.configure(api_key=GEMINI_API_KEY)


def embed_text(text: str):
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text
    )
    return result["embedding"]


def embed_media(file_path: str):
    uploaded_file = genai.upload_file(file_path)

    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=uploaded_file
    )

    return result["embedding"]


def wait_for_file_active(uploaded_file, timeout=120):
    start_time = time.time()

    while time.time() - start_time < timeout:
        file = genai.get_file(uploaded_file.name)

        state = getattr(file, "state", None)

        if str(state).lower().endswith("active"):
            return file

        time.sleep(5)

    return uploaded_file


def describe_media(file_path: str, modality: str):
    uploaded_file = genai.upload_file(file_path)

    uploaded_file = wait_for_file_active(uploaded_file)

    model = genai.GenerativeModel(LLM_MODEL)

    if modality == "image":
        prompt = """
Describe this image in one clear sentence.
Mention important visible objects, place, person, text, or concept.
This description will be used for keyword search.
"""

    elif modality == "audio":
        prompt = """
Listen to this audio and summarize it in 2 to 3 short sentences.
Mention speech topic, sounds, speaker hints, or events if present.
This summary will be used for keyword search.
"""

    elif modality == "video":
        prompt = """
Watch this video and summarize it in 2 to 3 short sentences.
Mention main scene, objects, actions, places, people, and events.
This summary will be used for keyword search.
"""

    else:
        prompt = "Describe this media briefly."

    response = model.generate_content([prompt, uploaded_file])

    return response.text.strip()