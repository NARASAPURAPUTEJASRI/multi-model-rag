"""
bm25_store.py

This file manages the BM25 keyword search index.

BM25 is used for keyword-based retrieval in the multimodal RAG pipeline.
It works together with vector search.

Main responsibilities:
- Tokenize text
- Remove stopwords
- Apply simple stemming
- Build BM25 index
- Save and load BM25 index from disk
- Add new items into BM25
- Search BM25 results for a user query

For media files, captions and descriptions are used for BM25 search.
Raw media embeddings are still handled separately in vector search.
"""

import os
import re
import pickle
from rank_bm25 import BM25Okapi

# File path where the BM25 index is saved
BM25_PATH = "./data/bm25_index.pkl"

# Global BM25 model object
bm25_model = None

# Stores all items added to BM25
bm25_items = []

# Common words and intent words removed before BM25 tokenization
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am",
    "to", "of", "in", "on", "at", "for", "from", "by",
    "me", "give", "gave", "show", "tell", "about",
    "image", "images", "photo", "photos", "picture", "pictures",
    "audio", "audios", "sound", "sounds",
    "video", "videos", "clip", "clips",
    "and", "or", "with", "as", "this", "that"
}


def simple_stem(word: str):
    # Removes simple word endings to improve keyword matching
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]

    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]

    if len(word) > 4 and word.endswith("es"):
        return word[:-2]

    if len(word) > 3 and word.endswith("s"):
        return word[:-1]

    return word


def tokenize(text: str):
    # Converts text into clean searchable tokens for BM25
    if not text:
        return []

    text = text.lower()

    # Extract only letters and numbers
    words = re.findall(r"[a-z0-9]+", text)

    tokens = []

    for word in words:

        # Skip common stopwords
        if word in STOPWORDS:
            continue

        # Apply simple stemming
        stemmed = simple_stem(word)

        if stemmed:
            tokens.append(stemmed)

    return tokens


def get_item_text(item):
    # Returns the text that should be indexed by BM25

    # Prefer direct bm25_text if available
    if item.get("bm25_text"):
        return item["bm25_text"]

    # Otherwise build searchable text from metadata
    metadata = item.get("metadata", {})

    parts = [
        metadata.get("content", ""),
        metadata.get("description", ""),
        metadata.get("caption", ""),
        metadata.get("page_title", ""),
        metadata.get("source_topic", ""),
        metadata.get("modality", ""),
    ]

    return " ".join([p for p in parts if p])


def save_bm25_index():
    # Saves BM25 model and indexed items to disk

    os.makedirs("./data", exist_ok=True)

    with open(BM25_PATH, "wb") as f:

        pickle.dump(
            {
                "items": bm25_items,
                "model": bm25_model
            },
            f
        )


def rebuild_bm25_model():
    # Rebuilds the BM25 model whenever new valid items are added

    global bm25_model
    global bm25_items

    valid_items = []

    for item in bm25_items:

        text = get_item_text(item).strip()

        tokens = tokenize(text)

        # Only keep items that contain valid searchable text
        if text and tokens:
            valid_items.append(item)

    # If no valid items exist, reset the BM25 index
    if not valid_items:

        bm25_model = None
        bm25_items = []

        save_bm25_index()

        return

    # Build BM25 corpus from tokenized item text
    corpus = [
        tokenize(get_item_text(item))
        for item in valid_items
    ]

    # Create BM25 model
    bm25_model = BM25Okapi(corpus)

    # Keep only valid indexed items
    bm25_items = valid_items

    # Save updated BM25 index
    save_bm25_index()


def load_bm25_index():
    # Loads BM25 model and indexed items from disk

    global bm25_model
    global bm25_items

    if not os.path.exists(BM25_PATH):

        bm25_model = None
        bm25_items = []

        return

    with open(BM25_PATH, "rb") as f:

        data = pickle.load(f)

    bm25_items = data.get("items", [])

    bm25_model = data.get("model")


def add_items_to_bm25(new_items):
    # Adds newly ingested text/media items to BM25 index

    global bm25_items

    if not new_items:
        return

    # Track existing IDs to avoid duplicate indexing
    existing_ids = {
        item.get("id")
        for item in bm25_items
    }

    added_count = 0

    for item in new_items:

        item_id = item.get("id")

        text = get_item_text(item).strip()

        if not item_id:
            continue

        if not text:
            continue

        if item_id in existing_ids:
            continue

        bm25_items.append(item)

        existing_ids.add(item_id)

        added_count += 1

    # Rebuild BM25 only if new items were added
    if added_count > 0:
        rebuild_bm25_model()


def build_bm25_index(items):
    # Builds BM25 index from a complete list of items

    global bm25_items

    bm25_items = []

    for item in items:

        text = get_item_text(item).strip()

        if text:
            bm25_items.append(item)

    rebuild_bm25_model()


def bm25_search(query: str, top_k=10):
    # Searches BM25 index using the user query

    global bm25_model
    global bm25_items

    if bm25_model is None:
        return []

    if not bm25_items:
        return []

    # Tokenize the user query
    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    # Get BM25 scores for all indexed items
    scores = bm25_model.get_scores(query_tokens)

    # Sort results by BM25 score in descending order
    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    results = []

    for rank, (idx, score) in enumerate(ranked, start=1):

        if idx >= len(bm25_items):
            continue

        # Ignore zero or negative BM25 scores
        if score <= 0:
            continue

        item = bm25_items[idx]

        # Return result in common retrieval format
        results.append({
            "id": item["id"],
            "score": float(score),
            "bm25_score": float(score),
            "metadata": item["metadata"],
            "document": get_item_text(item),
            "rank": rank
        })

    return results