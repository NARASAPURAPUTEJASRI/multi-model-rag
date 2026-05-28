"""
response_router.py

This file decides what response should be returned to the frontend.

Rules:
- text query returns text answer + related media if available
- image query returns only image results
- audio query returns only audio results
- video query returns only video results

LLM is used only for text answer generation.
Media-only responses do not call LLM for answer generation.
"""

import google.generativeai as genai
from app.config import GEMINI_API_KEY, LLM_MODEL

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Gemini model used for text answer generation
model = genai.GenerativeModel(LLM_MODEL)


def generate_text_answer(query, results):
    # Generates text answer using only retrieved text contexts

    text_contexts = []

    for result in results:
        metadata = result.get("metadata", {})

        # Use only text modality for answer generation
        if metadata.get("modality") == "text":
            content = metadata.get("content") or result.get("document", "")
            if content:
                text_contexts.append(content)

    # Use top 4 text contexts
    context = "\n\n".join(text_contexts[:4])

    # If no text context exists, return fallback answer
    if not context.strip():
        return {
            "type": "text",
            "answer": "No relevant text data found."
        }

    # Prompt for grounded answer generation
    prompt = f"""
Answer the user question using only the context below.
Give a short, clear answer.

Context:
{context}

Question:
{query}
"""

    response = model.generate_content(prompt)

    return {
        "type": "text",
        "answer": response.text
    }


def get_media_limit(intent):
    # Returns maximum number of media results for each intent

    if intent == "image":
        return 2
    if intent == "audio":
        return 1
    if intent == "video":
        return 1
    return 3


def empty_media_message(intent):
    # Returns message when no relevant media is found

    if intent == "image":
        return "No relevant image found for this query."
    if intent == "audio":
        return "No relevant audio found for this query."
    if intent == "video":
        return "No relevant video found for this query."
    return "No relevant media found for this query."


def media_response(intent, results):
    # Builds media-only response for image/audio/video queries

    media = []
    limit = get_media_limit(intent)

    for result in results:
        metadata = result.get("metadata", {})

        # Only include media matching requested intent
        if metadata.get("modality") == intent:
            media.append({
                "url": metadata.get("url", ""),
                "caption": metadata.get("caption", ""),
                "description": metadata.get("description", ""),
                "modality": metadata.get("modality", ""),
                "page_title": metadata.get("page_title", ""),
                "source_topic": metadata.get("source_topic", ""),
                "score": result.get("score", 0)
            })

    return {
        "type": intent,
        "results": media[:limit],
        "message": "" if media else empty_media_message(intent)
    }


def route_response(intent, query, results):
    # Main response routing function

    if intent == "text":
        # Generate text answer
        text_answer = generate_text_answer(query, results)

        images = []
        audios = []
        videos = []

        # Collect related media for text query
        for result in results:
            metadata = result.get("metadata", {})
            modality = metadata.get("modality")

            media_item = {
                "url": metadata.get("url", ""),
                "caption": metadata.get("caption", ""),
                "description": metadata.get("description", ""),
                "modality": modality,
                "page_title": metadata.get("page_title", ""),
                "source_topic": metadata.get("source_topic", ""),
                "score": result.get("score", 0)
            }

            # Add maximum 2 images
            if modality == "image" and len(images) < 2:
                images.append(media_item)

            # Add maximum 1 audio
            elif modality == "audio" and len(audios) < 1:
                audios.append(media_item)

            # Add maximum 1 video
            elif modality == "video" and len(videos) < 1:
                videos.append(media_item)

        return {
            "type": "text",
            "answer": text_answer.get("answer", ""),
            "media": images + audios + videos
        }

    # Media-only response routing
    if intent in ["image", "audio", "video"]:
        return media_response(intent, results)

    # Fallback response
    return generate_text_answer(query, results)