from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.evaluation import evaluate_pipeline
from app.logger import get_logger
from app.intent import classify_intent
from app.embeddings import embed_text
from app.vector_store import vector_search
from app.bm25_store import bm25_search, load_bm25_index
from app.hybrid import reciprocal_rank_fusion
from app.reranker import rerank_results_after_rrf, has_required_entity
from app.cold_start import wikipedia_cold_start
from app.response_router import route_response
from app.config import (
    TEXT_FINAL_THRESHOLD,
    IMAGE_FINAL_THRESHOLD,
    AUDIO_FINAL_THRESHOLD,
    VIDEO_FINAL_THRESHOLD,
    MIXED_FINAL_THRESHOLD,
)

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
    if intent in ["image", "audio", "video", "mixed"]:
        return 50
    return 15


def get_final_threshold(intent):
    if intent == "text":
        return TEXT_FINAL_THRESHOLD
    if intent == "image":
        return IMAGE_FINAL_THRESHOLD
    if intent == "audio":
        return AUDIO_FINAL_THRESHOLD
    if intent == "video":
        return VIDEO_FINAL_THRESHOLD
    if intent == "mixed":
        return MIXED_FINAL_THRESHOLD
    return MIXED_FINAL_THRESHOLD


def filter_results_by_intent(intent, results):
    if intent == "text":
        return [
            r for r in results
            if r.get("metadata", {}).get("modality") in ["text", "image", "audio", "video"]
        ]

    if intent in ["image", "audio", "video"]:
        return [
            r for r in results
            if r.get("metadata", {}).get("modality") == intent
        ]

    return results


def best_score(results):
    if not results:
        return 0.0

    return max(float(r.get("score", 0)) for r in results)


def apply_final_relevance_gate(query, intent, results, threshold):
    accepted = []

    for result in results:
        score = float(result.get("score", 0))

        if score < threshold:
            continue

        if intent in ["text","image", "audio", "video"]:
            if not has_required_entity(query, result):
                continue

        accepted.append(result)

    return accepted


def should_cold_start(filtered_results, threshold):
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
    final_threshold = get_final_threshold(intent)

    log.info(
        "Search config | intent=%s | top_k=%s | final_threshold=%s",
        intent,
        search_k,
        final_threshold,
    )

    query_embedding = embed_text(query)
    log.info("Query embedding created")

    vector_results = vector_search(query_embedding, top_k=search_k)
    log.info("Vector search completed | count=%s", len(vector_results))

    bm25_results = bm25_search(query, top_k=search_k)
    log.info("BM25 search completed | count=%s", len(bm25_results))

    merged_results = reciprocal_rank_fusion(vector_results, bm25_results)
    merged_results = rerank_results_after_rrf(query, merged_results)

    filtered_results = filter_results_by_intent(intent, merged_results)

    log.info(
        "Hybrid merge completed | total_count=%s | filtered_count=%s | best_score=%s",
        len(merged_results),
        len(filtered_results),
        best_score(filtered_results),
    )

    gated_results = apply_final_relevance_gate(
        query=query,
        intent=intent,
        results=filtered_results,
        threshold=final_threshold,
    )

    if should_cold_start(gated_results, final_threshold):
        log.info(
            "Cold start needed | intent=%s | gated_count=%s | best_score=%s | final_threshold=%s",
            intent,
            len(gated_results),
            best_score(gated_results),
            final_threshold,
        )

        new_items = wikipedia_cold_start(query, intent)

        if new_items:
            log.info("New unique data ingested from cold start | count=%s", len(new_items))
        else:
            log.info("Cold start completed but no new unique data found")

        vector_results = vector_search(query_embedding, top_k=search_k)
        log.info("Vector search rerun completed | count=%s", len(vector_results))

        bm25_results = bm25_search(query, top_k=search_k)
        log.info("BM25 search rerun completed | count=%s", len(bm25_results))

        merged_results = reciprocal_rank_fusion(vector_results, bm25_results)
        merged_results = rerank_results_after_rrf(query, merged_results)

        filtered_results = filter_results_by_intent(intent, merged_results)

        log.info(
            "Hybrid merge rerun completed | total_count=%s | filtered_count=%s | best_score=%s",
            len(merged_results),
            len(filtered_results),
            best_score(filtered_results),
        )

        gated_results = apply_final_relevance_gate(
            query=query,
            intent=intent,
            results=filtered_results,
            threshold=final_threshold,
        )

    else:
        log.info(
            "Cold start skipped | intent=%s | gated_count=%s | best_score=%s",
            intent,
            len(gated_results),
            best_score(gated_results),
        )

    if not gated_results:
        log.info("No relevant final results found | intent=%s | query=%s", intent, query)

    response = route_response(intent, query, gated_results)
    
    evaluate_pipeline(
        query=query,
        intent=intent,
        results=gated_results,
        response=response
    )

    log.info("Response sent | type=%s", response.get("type"))

    return response