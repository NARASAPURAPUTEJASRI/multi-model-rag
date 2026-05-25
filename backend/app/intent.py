def classify_intent(query: str):
    q = query.lower().strip()

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

    audio_phrases = [
        "give me audio",
        "give me sound",
        "play audio",
        "play sound",
        "show audio",
        "audio file",
        "sound file",
    ]

    video_phrases = [
        "give me video",
        "show video",
        "play video",
        "display video",
        "video clip",
        "videos",
    ]

    if any(phrase in q for phrase in text_phrases):
        return "text"

    if any(phrase in q for phrase in image_phrases):
        return "image"

    if any(phrase in q for phrase in audio_phrases):
        return "audio"

    if any(phrase in q for phrase in video_phrases):
        return "video"

    if "image" in q or "photo" in q or "picture" in q:
        return "image"

    if "audio" in q or "sound" in q:
        return "audio"

    if "video" in q:
        return "video"

    return "text"