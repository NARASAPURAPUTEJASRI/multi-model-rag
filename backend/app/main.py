"""
main.py

Main FastAPI backend entry point.

Updated:
- Uses normalized confidence thresholds.
- Uses NLP intent detection.
- Filters unrelated media using query entity matching.
- Evaluation runs only after response generation.
"""

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
from app.reranker import rerank_results_after_rrf
from app.cold_start import wikipedia_cold_start
from app.response_router import route_response
from app.confidence import calculate_result_confidence, best_confidence
from app.text_utils import has_query_entity_match

from app.config import (
    TEXT_CONFIDENCE_THRESHOLD,
    IMAGE_CONFIDENCE_THRESHOLD,
    AUDIO_CONFIDENCE_THRESHOLD,
    VIDEO_CONFIDENCE_THRESHOLD,
    MIXED_CONFIDENCE_THRESHOLD,
)


log = get_logger("main")

app = FastAPI(title="Multimodal RAG Pipeline")


# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve media files from data/media
app.mount("/media", StaticFiles(directory="data/media"), name="media")


class ChatRequest(BaseModel):
    # Request body from frontend
    query: str


@app.on_event("startup")
def startup():
    # Load BM25 index when backend starts
    load_bm25_index()
    log.info("Backend started successfully")


def get_search_k(intent):
    # Media search needs more candidates
    if intent in ["image", "audio", "video", "mixed"]:
        return 50

    return 15


def get_final_threshold(intent):
    # Return confidence threshold according to intent
    if intent == "text":
        return TEXT_CONFIDENCE_THRESHOLD

    if intent == "image":
        return IMAGE_CONFIDENCE_THRESHOLD

    if intent == "audio":
        return AUDIO_CONFIDENCE_THRESHOLD

    if intent == "video":
        return VIDEO_CONFIDENCE_THRESHOLD

    if intent == "mixed":
        return MIXED_CONFIDENCE_THRESHOLD

    return MIXED_CONFIDENCE_THRESHOLD


def filter_results_by_intent(intent, results):
    # Text query can keep all modalities
    if intent == "text":
        return [
            r for r in results
            if r.get("metadata", {}).get("modality") in ["text", "image", "audio", "video"]
        ]

    # Media query keeps only requested modality
    if intent in ["image", "audio", "video"]:
        return [
            r for r in results
            if r.get("metadata", {}).get("modality") == intent
        ]

    return results


def apply_final_relevance_gate(query, intent, results, threshold):
    # Accept results only if confidence passes threshold and topic entity matches
    accepted = []

    for result in results:
        confidence = calculate_result_confidence(query, intent, result)

        result["confidence_score"] = confidence

        if confidence < threshold:
            continue

        if not has_query_entity_match(query, result):
            continue

        accepted.append(result)

    return accepted


def should_cold_start(query, intent, results, threshold):
    # Trigger cold start when no valid results or confidence is weak
    if not results:
        return True

    return best_confidence(query, intent, results) < threshold


@app.post("/chat")
def chat(req: ChatRequest):
    # Read query
    query = req.query.strip()

    log.info("Query received | query=%s", query)

    # Detect intent
    intent = classify_intent(query)
    log.info("Intent detected | intent=%s", intent)

    # Search settings
    search_k = get_search_k(intent)
    final_threshold = get_final_threshold(intent)

    log.info(
        "Search config | intent=%s | top_k=%s | confidence_threshold=%s",
        intent,
        search_k,
        final_threshold,
    )

    # Query embedding
    query_embedding = embed_text(query)
    log.info("Query embedding created")

    # Vector search
    vector_results = vector_search(query_embedding, top_k=search_k)
    log.info("Vector search completed | count=%s", len(vector_results))

    # BM25 search
    bm25_results = bm25_search(query, top_k=search_k)
    log.info("BM25 search completed | count=%s", len(bm25_results))

    # Hybrid RRF merge
    merged_results = reciprocal_rank_fusion(vector_results, bm25_results)

    # Entity reranking
    merged_results = rerank_results_after_rrf(query, merged_results)

    # Intent filtering
    filtered_results = filter_results_by_intent(intent, merged_results)

    log.info(
        "Hybrid merge completed | total_count=%s | filtered_count=%s | best_confidence=%s",
        len(merged_results),
        len(filtered_results),
        best_confidence(query, intent, filtered_results),
    )

    # Final confidence gate
    gated_results = apply_final_relevance_gate(
        query=query,
        intent=intent,
        results=filtered_results,
        threshold=final_threshold,
    )

    # Cold start check
    if should_cold_start(query, intent, gated_results, final_threshold):
        log.info(
            "Cold start needed | intent=%s | gated_count=%s | best_confidence=%s | confidence_threshold=%s",
            intent,
            len(gated_results),
            best_confidence(query, intent, gated_results),
            final_threshold,
        )

        # Wikipedia cold start
        new_items = wikipedia_cold_start(query, intent)

        if new_items:
            log.info("New unique data ingested from cold start | count=%s", len(new_items))
        else:
            log.info("Cold start completed but no new unique data found")

        # Rerun vector search
        vector_results = vector_search(query_embedding, top_k=search_k)
        log.info("Vector search rerun completed | count=%s", len(vector_results))

        # Rerun BM25 search
        bm25_results = bm25_search(query, top_k=search_k)
        log.info("BM25 search rerun completed | count=%s", len(bm25_results))

        # Rerun merge and reranking
        merged_results = reciprocal_rank_fusion(vector_results, bm25_results)
        merged_results = rerank_results_after_rrf(query, merged_results)

        # Rerun intent filtering
        filtered_results = filter_results_by_intent(intent, merged_results)

        log.info(
            "Hybrid merge rerun completed | total_count=%s | filtered_count=%s | best_confidence=%s",
            len(merged_results),
            len(filtered_results),
            best_confidence(query, intent, filtered_results),
        )

        # Rerun gate
        gated_results = apply_final_relevance_gate(
            query=query,
            intent=intent,
            results=filtered_results,
            threshold=final_threshold,
        )

    else:
        log.info(
            "Cold start skipped | intent=%s | gated_count=%s | best_confidence=%s",
            intent,
            len(gated_results),
            best_confidence(query, intent, gated_results),
        )

    # Log no result
    if not gated_results:
        log.info("No relevant final results found | intent=%s | query=%s", intent, query)

    # Generate frontend response
    response = route_response(intent, query, gated_results)

    # Backend-only evaluation
    evaluate_pipeline(
        query=query,
        intent=intent,
        results=gated_results,
        response=response,
    )

    log.info("Response sent | type=%s", response.get("type"))

    return response