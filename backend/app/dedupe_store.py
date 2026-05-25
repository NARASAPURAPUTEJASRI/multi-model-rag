import json
import os
import hashlib

STATE_FILE = "./data/dedupe_state.json"


def load_state():
    if not os.path.exists(STATE_FILE):
        return {
            "media_urls": [],
            "text_hashes": []
        }

    with open(STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state):
    os.makedirs("./data", exist_ok=True)

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def make_hash(value: str):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def is_media_url_seen(url: str):
    state = load_state()
    url_hash = make_hash(url)
    return url_hash in state.get("media_urls", [])


def mark_media_url_seen(url: str):
    state = load_state()
    url_hash = make_hash(url)

    if url_hash not in state.get("media_urls", []):
        state["media_urls"].append(url_hash)

    save_state(state)


def is_text_seen(text: str):
    state = load_state()
    text_hash = make_hash(text[:1000])
    return text_hash in state.get("text_hashes", [])


def mark_text_seen(text: str):
    state = load_state()
    text_hash = make_hash(text[:1000])

    if text_hash not in state.get("text_hashes", []):
        state["text_hashes"].append(text_hash)

    save_state(state)