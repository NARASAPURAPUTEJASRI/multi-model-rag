import google.generativeai as genai
from app.config import LLM_MODEL

model = genai.GenerativeModel(LLM_MODEL)


def generate_text_answer(query, results):
    text_contexts = []

    for result in results:
        metadata = result.get("metadata", {})

        if metadata.get("modality") == "text":
            content = metadata.get("content") or result.get("document", "")
            if content:
                text_contexts.append(content)

    context = "\n\n".join(text_contexts[:4])

    prompt = f"""
Answer the user question using only the context below.
Give a short, clear answer in simple paragraphs.

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
    if intent == "image":
        return 2

    if intent == "audio":
        return 1

    if intent == "video":
        return 1

    return 3


def media_response(intent, results):
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
                "score": result.get("score", 0)
            })

    return {
        "type": intent,
        "results": media[:limit]
    }


def route_response(intent, query, results):
    if intent == "text":
        return generate_text_answer(query, results)

    if intent in ["image", "audio", "video"]:
        return media_response(intent, results)

    if intent == "mixed":
        text_answer = generate_text_answer(query, results)

        media_items = []

        for result in results:
            metadata = result.get("metadata", {})

            if metadata.get("modality") in ["image", "audio", "video"]:
                media_items.append({
                    "url": metadata.get("url", ""),
                    "caption": metadata.get("caption", ""),
                    "description": metadata.get("description", ""),
                    "modality": metadata.get("modality", ""),
                    "score": result.get("score", 0)
                })

        return {
            "type": "mixed",
            "answer": text_answer["answer"],
            "media": media_items[:4]
        }

    return generate_text_answer(query, results)