"""
text_utils.py

Shared NLP utility functions.

Used for:
- predefined stopwords
- project-specific stopwords
- tokenization
- entity extraction
- retrieved result text building
- entity matching
"""

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Project-specific words that are not topic entities
INTENT_WORDS = {
    "image", "images", "photo", "photos", "picture", "pictures",
    "audio", "audios", "sound", "sounds", "voice", "voices",
    "video", "videos", "clip", "clips",
    "give", "gave", "show", "tell", "about", "me",
    "play", "display", "explain", "describe",
    "information", "details", "want", "need", "please"
}


# Final stopwords = NLP library stopwords + project-specific words
STOPWORDS = set(ENGLISH_STOP_WORDS).union(INTENT_WORDS)


def tokenize(text: str):
    # Convert text into lowercase alphanumeric tokens
    return re.findall(r"[a-z0-9]+", str(text).lower())


def extract_query_entities(query: str):
    # Extract meaningful topic words from query
    tokens = tokenize(query)

    return [
        token for token in tokens
        if token not in STOPWORDS and len(token) > 2
    ]


def result_text(result):
    # Combine retrieved result document and metadata into one searchable text
    metadata = result.get("metadata", {})

    parts = [
        result.get("document", ""),
        metadata.get("content", ""),
        metadata.get("caption", ""),
        metadata.get("description", ""),
        metadata.get("page_title", ""),
        metadata.get("source_topic", ""),
        metadata.get("modality", ""),
    ]

    return " ".join([str(p) for p in parts if p]).lower()


def entity_coverage(query, results):
    # Measures how many query entities appear in retrieved results
    entities = extract_query_entities(query)

    if not entities or not results:
        return 0.0

    combined_text = " ".join([result_text(r) for r in results])

    matched = 0

    for entity in entities:
        if entity in combined_text:
            matched += 1

    return round(matched / len(entities), 4)


def has_query_entity_match(query, result):
    # Checks whether one result matches at least one important query entity
    entities = extract_query_entities(query)

    if not entities:
        return True

    tokens = set(tokenize(result_text(result)))

    return any(entity in tokens for entity in entities)