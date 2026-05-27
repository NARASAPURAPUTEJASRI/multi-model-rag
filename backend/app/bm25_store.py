import os
import re
import pickle
from rank_bm25 import BM25Okapi

BM25_PATH = "./data/bm25_index.pkl"

bm25_model = None
bm25_items = []

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
    if not text:
        return []

    text = text.lower()

    words = re.findall(r"[a-z0-9]+", text)

    tokens = []

    for word in words:

        if word in STOPWORDS:
            continue

        stemmed = simple_stem(word)

        if stemmed:
            tokens.append(stemmed)

    return tokens


def get_item_text(item):

    if item.get("bm25_text"):
        return item["bm25_text"]

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

    global bm25_model
    global bm25_items

    valid_items = []

    for item in bm25_items:

        text = get_item_text(item).strip()

        tokens = tokenize(text)

        if text and tokens:
            valid_items.append(item)

    if not valid_items:

        bm25_model = None
        bm25_items = []

        save_bm25_index()

        return

    corpus = [
        tokenize(get_item_text(item))
        for item in valid_items
    ]

    bm25_model = BM25Okapi(corpus)

    bm25_items = valid_items

    save_bm25_index()


def load_bm25_index():

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

    global bm25_items

    if not new_items:
        return

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

    if added_count > 0:
        rebuild_bm25_model()


def build_bm25_index(items):

    global bm25_items

    bm25_items = []

    for item in items:

        text = get_item_text(item).strip()

        if text:
            bm25_items.append(item)

    rebuild_bm25_model()


def bm25_search(query: str, top_k=10):

    global bm25_model
    global bm25_items

    if bm25_model is None:
        return []

    if not bm25_items:
        return []

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    scores = bm25_model.get_scores(query_tokens)

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    results = []

    for rank, (idx, score) in enumerate(ranked, start=1):

        if idx >= len(bm25_items):
            continue

        if score <= 0:
            continue

        item = bm25_items[idx]

        results.append({
            "id": item["id"],
            "score": float(score),
            "bm25_score": float(score),
            "metadata": item["metadata"],
            "document": get_item_text(item),
            "rank": rank
        })

    return results