"""
response_router.py

Routes final response to frontend.

Rules:
- text query -> text answer + related media only if topic matches
- image query -> only images
- audio query -> only audio
- video query -> only video
"""

import google.generativeai as genai

from app.config import GEMINI_API_KEY, LLM_MODEL
from app.text_utils import has_query_entity_match


genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel(LLM_MODEL)


def generate_text_answer(query, results):
    # Generate answer using only text chunks
    text_contexts = []

    for result in results:
        metadata = result.get("metadata", {})

        if metadata.get("modality") == "text":
            content = metadata.get("content") or result.get("document", "")

            if content:
                text_contexts.append(content)

    context = "\n\n".join(text_contexts[:4])

    if not context.strip():
        return {
            "type": "text",
            "answer": "No relevant text data found."
        }

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
    # Media result limits
    if intent == "image":
        return 2

    if intent == "audio":
        return 1

    if intent == "video":
        return 1

    return 3


def empty_media_message(intent):
    # Empty result messages
    if intent == "image":
        return "No relevant image found for this query."

    if intent == "audio":
        return "No relevant audio found for this query."

    if intent == "video":
        return "No relevant video found for this query."

    return "No relevant media found for this query."


def media_response(intent, results):
    # Build media-only response
    media = []
    limit = get_media_limit(intent)

    for result in results:
        metadata = result.get("metadata", {})

        if metadata.get("modality") == intent:
            media.append({
                "url": metadata.get("url", ""),
                "caption": metadata.get("caption", ""),
                "description": metadata.get("description", ""),
                "modality": metadata.get("modality", ""),
                "page_title": metadata.get("page_title", ""),
                "source_topic": metadata.get("source_topic", ""),
                "score": result.get("score", 0),
                "confidence_score": result.get("confidence_score", 0),
            })

    return {
        "type": intent,
        "results": media[:limit],
        "message": "" if media else empty_media_message(intent)
    }


def route_response(intent, query, results):
    # Text response with related media
    if intent == "text":
        text_answer = generate_text_answer(query, results)

        images = []
        audios = []
        videos = []

        for result in results:
            metadata = result.get("metadata", {})
            modality = metadata.get("modality")

            if modality not in ["image", "audio", "video"]:
                continue

            if not has_query_entity_match(query, result):
                continue

            media_item = {
                "url": metadata.get("url", ""),
                "caption": metadata.get("caption", ""),
                "description": metadata.get("description", ""),
                "modality": modality,
                "page_title": metadata.get("page_title", ""),
                "source_topic": metadata.get("source_topic", ""),
                "score": result.get("score", 0),
                "confidence_score": result.get("confidence_score", 0),
            }

            if modality == "image" and len(images) < 2:
                images.append(media_item)

            elif modality == "audio" and len(audios) < 1:
                audios.append(media_item)

            elif modality == "video" and len(videos) < 1:
                videos.append(media_item)

        return {
            "type": "text",
            "answer": text_answer.get("answer", ""),
            "media": images + audios + videos
        }

    # Media-only response
    if intent in ["image", "audio", "video"]:
        return media_response(intent, results)

    return generate_text_answer(query, results)