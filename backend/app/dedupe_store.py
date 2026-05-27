import os
import json
import hashlib

DEDUPE_PATH = "./data/dedupe_state.json"


def load_state():
    if not os.path.exists(DEDUPE_PATH):
        return {
            "media_urls": [],
            "file_hashes": [],
            "text_hashes": []
        }

    with open(DEDUPE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state):
    os.makedirs("./data", exist_ok=True)

    with open(DEDUPE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def hash_text(text: str):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_file(file_path: str):
    sha = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)

    return sha.hexdigest()


def is_text_seen(text: str):
    state = load_state()
    text_hash = hash_text(text)

    return text_hash in state.get("text_hashes", [])


def mark_text_seen(text: str):
    state = load_state()
    text_hash = hash_text(text)

    if text_hash not in state["text_hashes"]:
        state["text_hashes"].append(text_hash)

    save_state(state)


def is_media_url_seen(url: str):
    state = load_state()

    return url in state.get("media_urls", [])


def mark_media_url_seen(url: str):
    state = load_state()

    if url not in state["media_urls"]:
        state["media_urls"].append(url)

    save_state(state)


def is_file_hash_seen(file_path: str):
    state = load_state()
    file_hash = hash_file(file_path)

    return file_hash in state.get("file_hashes", [])


def mark_file_hash_seen(file_path: str):
    state = load_state()
    file_hash = hash_file(file_path)

    if file_hash not in state["file_hashes"]:
        state["file_hashes"].append(file_hash)

    save_state(state)