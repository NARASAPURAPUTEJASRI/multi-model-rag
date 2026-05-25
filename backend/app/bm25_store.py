import os
import pickle
from rank_bm25 import BM25Okapi

BM25_PATH = "./data/bm25_index.pkl"

bm25_model = None
bm25_items = []


def tokenize(text: str):
    return text.lower().split()


def get_item_text(item):
    if item.get("bm25_text"):
        return item["bm25_text"]

    metadata = item.get("metadata", {})

    return (
        metadata.get("content")
        or metadata.get("description")
        or metadata.get("caption")
        or ""
    )


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

    valid_items = []

    for item in bm25_items:
        text = get_item_text(item).strip()
        if text:
            valid_items.append(item)

    if not valid_items:
        bm25_model = None
        return

    corpus = [tokenize(get_item_text(item)) for item in valid_items]

    bm25_model = BM25Okapi(corpus)

    bm25_items.clear()
    bm25_items.extend(valid_items)

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

    existing_ids = {item["id"] for item in bm25_items}
    added_count = 0

    for item in new_items:
        item_id = item.get("id")
        text = get_item_text(item).strip()

        if not item_id or not text:
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

    if bm25_model is None or not bm25_items:
        return []

    scores = bm25_model.get_scores(tokenize(query))

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    results = []

    for rank, (idx, score) in enumerate(ranked, start=1):
        if idx >= len(bm25_items):
            continue

        item = bm25_items[idx]

        results.append({
            "id": item["id"],
            "score": float(score),
            "metadata": item["metadata"],
            "document": get_item_text(item),
            "rank": rank
        })

    return results