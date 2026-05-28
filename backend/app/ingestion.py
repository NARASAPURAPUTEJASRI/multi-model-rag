"""
ingestion.py

This file handles ingestion of text and media into the multimodal RAG system.

Main responsibilities:
- Split text into chunks
- Generate text embeddings
- Generate raw media embeddings
- Generate media descriptions for BM25 keyword search
- Store all embeddings in ChromaDB
- Return BM25-ready items

Important:
- Text is embedded directly as text.
- Image/audio/video are embedded as raw media files.
- Media descriptions are used only for BM25 and metadata.
"""

import uuid
import time

from app.embeddings import embed_text, embed_media, describe_media
from app.vector_store import add_to_vector_db
from app.logger import get_logger

# Logger for ingestion process
log = get_logger("ingestion")


def chunk_text(text, chunk_size=350, overlap=70):
    # Splits long text into overlapping chunks

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size

        # Create one chunk
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        # Move start forward with overlap
        start += chunk_size - overlap

    return chunks


def safe_embed_media(file_path, retries=3):
    # Creates media embedding with retry handling

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            log.info("Media embedding attempt | file=%s | attempt=%s", file_path, attempt)
            return embed_media(file_path)

        except Exception as e:
            last_error = e
            log.warning("Media embedding failed | file=%s | attempt=%s | error=%s", file_path, attempt, str(e))

            # Wait before retrying
            time.sleep(3 * attempt)

    # Raise last error if all retries fail
    raise last_error


def safe_describe_media(file_path, modality, fallback_caption="", retries=2):
    # Generates media description with retry handling

    for attempt in range(1, retries + 1):
        try:
            log.info("Media description attempt | modality=%s | file=%s | attempt=%s", modality, file_path, attempt)

            description = describe_media(file_path, modality)

            if description:
                return description

        except Exception as e:
            log.warning("Media description failed | modality=%s | file=%s | attempt=%s | error=%s", modality, file_path, attempt, str(e))

            # Wait before retrying
            time.sleep(2 * attempt)

    # Use fallback if description generation fails
    return fallback_caption or f"{modality} media related to the topic"


def ingest_text(text, source_url=None, source_topic="", page_title=""):
    # Ingests text into vector DB and prepares BM25 items

    chunks = chunk_text(text)
    items = []

    for chunk in chunks:
        # Create unique text item ID
        item_id = f"text_{uuid.uuid4().hex}"

        # Generate text embedding
        embedding = embed_text(chunk)

        # Metadata stored with vector item
        metadata = {
            "modality": "text",
            "url": source_url or "",
            "content": chunk,
            "source_topic": source_topic,
            "page_title": page_title,
        }

        # Store text embedding in vector database
        add_to_vector_db(
            item_id=item_id,
            embedding=embedding,
            document=chunk,
            metadata=metadata,
        )

        # Return item for BM25 indexing
        items.append({
            "id": item_id,
            "bm25_text": chunk,
            "metadata": metadata,
        })

    log.info("Text ingestion completed | chunks=%s", len(chunks))

    return items


def ingest_media(file_path, modality, caption, url, source_topic="", page_title=""):
    # Ingests image/audio/video into vector DB and prepares BM25 item

    # Create unique media item ID
    item_id = f"{modality}_{uuid.uuid4().hex}"

    # Raw media semantic embedding
    embedding = safe_embed_media(file_path)

    # Description only for BM25 keyword search and frontend captions
    description = safe_describe_media(
        file_path=file_path,
        modality=modality,
        fallback_caption=caption,
    )

    # BM25 searchable text for media
    bm25_text = f"{caption}. {description}. {source_topic}. {page_title}"

    # Metadata stored with media result
    metadata = {
        "modality": modality,
        "url": url,
        "caption": caption,
        "description": description,
        "file_path": file_path,
        "source_topic": source_topic,
        "page_title": page_title,
    }

    # Store media embedding in vector database
    add_to_vector_db(
        item_id=item_id,
        embedding=embedding,
        document=bm25_text,
        metadata=metadata,
    )

    log.info("Media ingestion completed | modality=%s | id=%s", modality, item_id)

    # Return item for BM25 indexing
    return {
        "id": item_id,
        "bm25_text": bm25_text,
        "metadata": metadata,
    }