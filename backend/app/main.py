from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.logger import get_logger
from app.intent import classify_intent
from app.embeddings import embed_text
from app.vector_store import vector_search
from app.bm25_store import bm25_search, load_bm25_index
from app.hybrid import reciprocal_rank_fusion
from app.cold_start import wikipedia_cold_start
from app.response_router import route_response
from app.config import SIMILARITY_THRESHOLD

log = get_logger("main")

app = FastAPI(title="Multimodal RAG Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory="data/media"), name="media")


class ChatRequest(BaseModel):
    query: str


@app.on_event("startup")
def startup():
    load_bm25_index()
    log.info("Backend started successfully")


def get_search_k(intent):
    return 50 if intent in ["image", "audio", "video", "mixed"] else 15


def get_intent_threshold(intent):
    if intent == "text":
        return SIMILARITY_THRESHOLD
    if intent == "image":
        return max(0.10, SIMILARITY_THRESHOLD - 0.10)
    if intent == "audio":
        return max(0.08, SIMILARITY_THRESHOLD - 0.12)
    if intent == "video":
        return max(0.08, SIMILARITY_THRESHOLD - 0.12)
    if intent == "mixed":
        return max(0.10, SIMILARITY_THRESHOLD - 0.10)
    return SIMILARITY_THRESHOLD


def filter_results_by_intent(intent, results):
    if intent == "text":
        return [r for r in results if r.get("metadata", {}).get("modality") == "text"]
    if intent == "image":
        return [r for r in results if r.get("metadata", {}).get("modality") == "image"]
    if intent == "audio":
        return [r for r in results if r.get("metadata", {}).get("modality") == "audio"]
    if intent == "video":
        return [r for r in results if r.get("metadata", {}).get("modality") == "video"]
    return results


def best_score(results):
    scores = []
    for r in results:
        try:
            scores.append(float(r.get("score", 0)))
        except Exception:
            pass
    return max(scores) if scores else 0


def should_cold_start(vector_results, filtered_results, threshold):
    if not vector_results:
        return True
    if not filtered_results:
        return True
    return best_score(filtered_results) < threshold


@app.post("/chat")
def chat(req: ChatRequest):
    query = req.query.strip()
    log.info("Query received | query=%s", query)

    intent = classify_intent(query)
    log.info("Intent detected | intent=%s", intent)

    search_k = get_search_k(intent)
    threshold = get_intent_threshold(intent)

    log.info("Search config | intent=%s | top_k=%s | threshold=%s", intent, search_k, threshold)

    query_embedding = embed_text(query)
    log.info("Query embedding created")

    vector_results = vector_search(query_embedding, top_k=search_k)
    log.info("Vector search completed | count=%s", len(vector_results))

    bm25_results = bm25_search(query, top_k=search_k)
    log.info("BM25 search completed | count=%s", len(bm25_results))

    merged_results = reciprocal_rank_fusion(vector_results, bm25_results)
    filtered_results = filter_results_by_intent(intent, merged_results)

    log.info(
        "Hybrid merge completed | total_count=%s | filtered_count=%s | best_score=%s",
        len(merged_results),
        len(filtered_results),
        best_score(filtered_results),
    )

    if should_cold_start(vector_results, filtered_results, threshold):
        log.info("Cold start needed | intent=%s", intent)

        new_items = wikipedia_cold_start(query, intent)
        log.info("Cold start completed | new_items=%s", len(new_items))

        vector_results = vector_search(query_embedding, top_k=search_k)
        log.info("Vector search rerun completed | count=%s", len(vector_results))

        bm25_results = bm25_search(query, top_k=search_k)
        log.info("BM25 search rerun completed | count=%s", len(bm25_results))

        merged_results = reciprocal_rank_fusion(vector_results, bm25_results)
        filtered_results = filter_results_by_intent(intent, merged_results)

        log.info(
            "Hybrid merge rerun completed | total_count=%s | filtered_count=%s | best_score=%s",
            len(merged_results),
            len(filtered_results),
            best_score(filtered_results),
        )
    else:
        log.info("Cold start skipped | intent=%s", intent)

    response = route_response(intent, query, filtered_results)
    log.info("Response sent | type=%s", response.get("type"))

    return response