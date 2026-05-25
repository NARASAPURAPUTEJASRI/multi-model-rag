import os
import uuid
import time
import requests
import wikipediaapi

from app.ingestion import ingest_text, ingest_media
from app.bm25_store import add_items_to_bm25
from app.logger import get_logger
from app.config import MEDIA_DIR
from app.dedupe_store import (
    is_media_url_seen,
    mark_media_url_seen,
    is_text_seen,
    mark_text_seen,
)

log = get_logger("cold_start")

IMAGE_LIMIT = 2
AUDIO_LIMIT = 1
VIDEO_LIMIT = 1
MAX_FILE_CHECKS = 20

HEADERS = {"User-Agent": "multimodal-rag-project/1.0"}

wiki = wikipediaapi.Wikipedia(
    user_agent="multimodal-rag-project/1.0",
    language="en"
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
AUDIO_EXTS = (".mp3", ".wav", ".ogg", ".oga", ".flac", ".m4a")
VIDEO_EXTS = (".mp4", ".mov", ".webm", ".ogv")


def normalize_wiki_query(query: str):
    q = query.lower()

    remove_words = [
        "tell me about", "explain", "what is", "who is",
        "show me", "give me", "gave me",
        "images", "image", "photos", "pictures",
        "audio", "audios", "sound", "sounds", "voice",
        "video", "videos", "clip", "clips",
        "information about", "details about",
    ]

    for word in remove_words:
        q = q.replace(word, " ")

    return " ".join(q.split()).strip()


def safe_get(url, params=None, timeout=10, retries=3):
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)

            if response.status_code == 429:
                wait_time = 5 * attempt
                log.warning("Rate limit hit | attempt=%s | waiting=%s sec", attempt, wait_time)
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            return response

        except Exception as e:
            last_error = e
            log.warning("HTTP request failed | attempt=%s | error=%s", attempt, str(e))
            time.sleep(3 * attempt)

    if last_error:
        raise last_error

    raise RuntimeError("HTTP request failed after retries")


def search_wikipedia_pages(query: str):
    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 5,
    }

    response = safe_get(url, params=params)
    data = response.json()

    return data.get("query", {}).get("search", [])


def select_best_page(query: str):
    clean_query = normalize_wiki_query(query)

    log.info("Wikipedia search query normalized | original=%s | clean=%s", query, clean_query)

    search_results = search_wikipedia_pages(clean_query)

    if not search_results:
        return None

    best_title = search_results[0]["title"]
    log.info("Wikipedia best page selected | title=%s", best_title)

    return wiki.page(best_title)


def select_relevant_sections(page, query):
    selected = []
    query_words = normalize_wiki_query(query).split()

    for section in page.sections:
        section_text = section.text or ""
        section_title = section.title or ""
        combined = (section_title + " " + section_text).lower()

        if any(word in combined for word in query_words):
            selected.append({
                "title": section_title,
                "text": section_text,
            })

    return selected[:3]


def get_wikipedia_files(page_title: str):
    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "titles": page_title,
        "prop": "images",
        "format": "json",
        "imlimit": 50,
    }

    response = safe_get(url, params=params)
    data = response.json()

    pages = data.get("query", {}).get("pages", {})
    file_titles = []

    for page in pages.values():
        for item in page.get("images", []):
            file_titles.append(item.get("title", ""))

    return file_titles[:MAX_FILE_CHECKS]


def get_file_url(file_title: str):
    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "titles": file_title,
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "format": "json",
    }

    response = safe_get(url, params=params)
    data = response.json()

    pages = data.get("query", {}).get("pages", {})

    for page in pages.values():
        info = page.get("imageinfo", [])
        if info:
            return {
                "url": info[0].get("url"),
                "mime": info[0].get("mime", ""),
                "caption": file_title.replace("File:", "").replace("_", " "),
            }

    return None


def split_files_by_modality(file_titles):
    images, audios, videos = [], [], []

    for title in file_titles:
        lower = title.lower()

        try:
            file_info = get_file_url(title)

            if not file_info or not file_info.get("url"):
                continue

            if lower.endswith(IMAGE_EXTS):
                images.append(file_info)

            elif lower.endswith(AUDIO_EXTS):
                audios.append(file_info)

            elif lower.endswith(VIDEO_EXTS):
                videos.append(file_info)

        except Exception as e:
            log.warning("File info failed | title=%s | error=%s", title, str(e))

    return {
        "image": images,
        "audio": audios,
        "video": videos,
    }


def attach_source_metadata(media_groups, query, page_title):
    source_topic = normalize_wiki_query(query)

    for modality in ["image", "audio", "video"]:
        for item in media_groups[modality]:
            item["source_topic"] = source_topic
            item["page_title"] = page_title


def download_media(media_url, modality):
    os.makedirs(MEDIA_DIR, exist_ok=True)

    ext = media_url.split(".")[-1].split("?")[0].lower()

    if modality == "image" and ext not in ["jpg", "jpeg", "png", "webp"]:
        ext = "jpg"

    if modality == "audio" and ext not in ["mp3", "wav", "ogg", "oga", "flac", "m4a"]:
        ext = "mp3"

    if modality == "video" and ext not in ["mp4", "mov", "webm", "ogv"]:
        ext = "mp4"

    filename = f"{uuid.uuid4().hex}.{ext}"
    file_path = os.path.join(MEDIA_DIR, filename)

    response = safe_get(media_url, params=None, timeout=30, retries=3)

    with open(file_path, "wb") as f:
        f.write(response.content)

    frontend_url = f"http://127.0.0.1:8000/media/{filename}"

    return file_path, frontend_url


def ingest_media_group(media_items, modality, new_items, limit):
    added = 0

    for item in media_items:
        if added >= limit:
            break

        try:
            media_url = item["url"]

            if is_media_url_seen(media_url):
                log.info("Skipping duplicate media | modality=%s | url=%s", modality, media_url)
                continue

            file_path, frontend_url = download_media(media_url, modality)

            media_item = ingest_media(
                file_path=file_path,
                modality=modality,
                caption=item["caption"],
                url=frontend_url,
                source_topic=item.get("source_topic", ""),
                page_title=item.get("page_title", ""),
            )

            mark_media_url_seen(media_url)
            new_items.append(media_item)
            added += 1

        except Exception as e:
            log.warning("%s ingestion failed | url=%s | error=%s", modality, item.get("url"), str(e))


def wikipedia_cold_start(query, intent="text"):
    log.info("Cold start triggered | query=%s | intent=%s", query, intent)

    page = select_best_page(query)

    if page is None or not page.exists():
        log.warning("Wikipedia page not found after search | query=%s", query)
        return []

    new_items = []
    source_topic = normalize_wiki_query(query)

    sections = select_relevant_sections(page, query)

    if not sections:
        sections = [{"title": "Summary", "text": page.summary}]

    full_relevant_text = "\n".join(
        [section["title"] + "\n" + section["text"] for section in sections]
    )

    if is_text_seen(full_relevant_text):
        log.info("Skipping duplicate text content | page=%s", page.title)
        text_items = []
    else:
        text_items = ingest_text(
            text=full_relevant_text,
            source_url=page.fullurl,
            source_topic=source_topic,
            page_title=page.title,
        )
        mark_text_seen(full_relevant_text)
        new_items.extend(text_items)

    file_titles = get_wikipedia_files(page.title)
    media_groups = split_files_by_modality(file_titles)
    attach_source_metadata(media_groups, query, page.title)

    if intent == "image":
        ingest_media_group(media_groups["image"], "image", new_items, IMAGE_LIMIT)

    elif intent == "audio":
        ingest_media_group(media_groups["audio"], "audio", new_items, AUDIO_LIMIT)

    elif intent == "video":
        ingest_media_group(media_groups["video"], "video", new_items, VIDEO_LIMIT)

    elif intent == "mixed":
        ingest_media_group(media_groups["image"], "image", new_items, IMAGE_LIMIT)
        ingest_media_group(media_groups["audio"], "audio", new_items, AUDIO_LIMIT)
        ingest_media_group(media_groups["video"], "video", new_items, VIDEO_LIMIT)

    else:
        log.info("Text intent detected, skipping media ingestion to reduce cold-start time")

    add_items_to_bm25(new_items)

    log.info(
        "Cold start ingestion completed | page=%s | intent=%s | text_items=%s | images_found=%s | audios_found=%s | videos_found=%s | total_new_unique=%s",
        page.title,
        intent,
        len(text_items),
        len(media_groups["image"]),
        len(media_groups["audio"]),
        len(media_groups["video"]),
        len(new_items),
    )

    return new_items