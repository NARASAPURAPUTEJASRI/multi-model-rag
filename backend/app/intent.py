"""
intent.py

This file detects the user query intent.

Supported intents:
- text
- image
- audio
- video

Intent is used to decide:
- what modality should be retrieved
- what response type should be sent to frontend
- whether LLM answer generation is needed
"""


def classify_intent(query: str):
    # Convert query to lowercase for easier matching
    q = query.lower().strip()

    # Phrases that indicate text/topic query
    text_phrases = [
        "tell me",
        "explain",
        "what is",
        "who is",
        "describe",
        "information about",
        "details about",
        "give me information",
        "give information",
    ]

    # Phrases that indicate image query
    image_phrases = [
        "give me image",
        "give me images",
        "show image",
        "show images",
        "display image",
        "display images",
        "photo",
        "photos",
        "picture",
        "pictures",
    ]

    # Phrases that indicate audio query
    audio_phrases = [
        "give me audio",
        "give me sound",
        "play audio",
        "play sound",
        "show audio",
        "audio file",
        "sound file",
    ]

    # Phrases that indicate video query
    video_phrases = [
        "give me video",
        "show video",
        "play video",
        "display video",
        "video clip",
        "videos",
    ]

    # Text intent has first priority for explanation-style queries
    if any(phrase in q for phrase in text_phrases):
        return "text"

    # Image intent detection
    if any(phrase in q for phrase in image_phrases):
        return "image"

    # Audio intent detection
    if any(phrase in q for phrase in audio_phrases):
        return "audio"

    # Video intent detection
    if any(phrase in q for phrase in video_phrases):
        return "video"

    # Fallback keyword checks for image
    if "image" in q or "photo" in q or "picture" in q:
        return "image"

    # Fallback keyword checks for audio
    if "audio" in q or "sound" in q:
        return "audio"

    # Fallback keyword check for video
    if "video" in q:
        return "video"

    # Default intent is text
    return "text"