"""
main.py

This is the main backend entry point for the Multimodal RAG Pipeline.

Responsibilities:
- Start FastAPI application
- Enable frontend CORS access
- Serve downloaded media files
- Receive user query from frontend
- Detect query intent
- Perform vector search
- Perform BM25 keyword search
- Merge results using RRF
- Apply entity-based reranking
- Filter results by intent
- Apply final relevance gate
- Trigger Wikipedia cold start if results are weak or missing
- Generate final response
- Run backend-only evaluation metrics
- Return response to frontend

Important:
- Cold start is controlled only by retrieval score and threshold.
- Evaluation runs only after response generation.
- Evaluation is logged in backend only.
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

# Logger for main pipeline
log = get_logger("main")

# Create FastAPI app
app = FastAPI(title="Multimodal RAG Pipeline")

# Allow frontend to call backend API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve downloaded media files from data/media folder
app.mount("/media", StaticFiles(directory="data/media"), name="media")


class ChatRequest(BaseModel):
    # Request body structure for /chat endpoint
    query: str


@app.on_event("startup")
def startup():
    # Load BM25 index when backend starts
    load_bm25_index()
    log.info("Backend started successfully")


def get_search_k(intent):
    # Media queries need more candidates because media may be sparse
    if intent in ["image", "audio", "video", "mixed"]:
        return 50

    # Text queries use smaller top_k
    return 15


def get_final_threshold(intent):
    # Return threshold based on detected modality
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
    # For text query, keep all modalities because text response may include related media
    if intent == "text":
        return [
            r for r in results
            if r.get("metadata", {}).get("modality") in ["text", "image", "audio", "video"]
        ]

    # For media-only query, keep only requested modality
    if intent in ["image", "audio", "video"]:
        return [
            r for r in results
            if r.get("metadata", {}).get("modality") == intent
        ]

    return results


def best_score(results):
    # Returns highest score from result list
    if not results:
        return 0.0

    return max(float(r.get("score", 0)) for r in results)


def apply_final_relevance_gate(query, intent, results, threshold):
    # Filters results using final threshold and entity matching

    accepted = []

    for result in results:
        score = float(result.get("score", 0))

        # Remove results below threshold
        if score < threshold:
            continue

        # Ensure result contains required query entity
        if intent in ["text", "image", "audio", "video"]:
            if not has_required_entity(query, result):
                continue

        accepted.append(result)

    return accepted


def should_cold_start(filtered_results, threshold):
    # Cold start is needed when no valid results exist or best score is weak
    if not filtered_results:
        return True

    return best_score(filtered_results) < threshold


@app.post("/chat")
def chat(req: ChatRequest):
    # Read user query from frontend request
    query = req.query.strip()

    log.info("Query received | query=%s", query)

    # Detect user intent: text/image/audio/video
    intent = classify_intent(query)
    log.info("Intent detected | intent=%s", intent)

    # Get retrieval configuration
    search_k = get_search_k(intent)
    final_threshold = get_final_threshold(intent)

    log.info(
        "Search config | intent=%s | top_k=%s | final_threshold=%s",
        intent,
        search_k,
        final_threshold,
    )

    # Convert user query into embedding
    query_embedding = embed_text(query)
    log.info("Query embedding created")

    # Run vector search from ChromaDB
    vector_results = vector_search(query_embedding, top_k=search_k)
    log.info("Vector search completed | count=%s", len(vector_results))

    # Run BM25 keyword search
    bm25_results = bm25_search(query, top_k=search_k)
    log.info("BM25 search completed | count=%s", len(bm25_results))

    # Merge vector and BM25 results using RRF
    merged_results = reciprocal_rank_fusion(vector_results, bm25_results)

    # Apply entity-based reranking
    merged_results = rerank_results_after_rrf(query, merged_results)

    # Filter results based on intent
    filtered_results = filter_results_by_intent(intent, merged_results)

    log.info(
        "Hybrid merge completed | total_count=%s | filtered_count=%s | best_score=%s",
        len(merged_results),
        len(filtered_results),
        best_score(filtered_results),
    )

    # Apply final relevance threshold and entity gate
    gated_results = apply_final_relevance_gate(
        query=query,
        intent=intent,
        results=filtered_results,
        threshold=final_threshold,
    )

    # Trigger cold start if results are missing or weak
    if should_cold_start(gated_results, final_threshold):
        log.info(
            "Cold start needed | intent=%s | gated_count=%s | best_score=%s | final_threshold=%s",
            intent,
            len(gated_results),
            best_score(gated_results),
            final_threshold,
        )

        # Ingest new data from Wikipedia
        new_items = wikipedia_cold_start(query, intent)

        if new_items:
            log.info("New unique data ingested from cold start | count=%s", len(new_items))
        else:
            log.info("Cold start completed but no new unique data found")

        # Rerun vector search after cold start ingestion
        vector_results = vector_search(query_embedding, top_k=search_k)
        log.info("Vector search rerun completed | count=%s", len(vector_results))

        # Rerun BM25 search after cold start ingestion
        bm25_results = bm25_search(query, top_k=search_k)
        log.info("BM25 search rerun completed | count=%s", len(bm25_results))

        # Merge rerun results
        merged_results = reciprocal_rank_fusion(vector_results, bm25_results)

        # Rerank rerun results
        merged_results = rerank_results_after_rrf(query, merged_results)

        # Filter rerun results by intent
        filtered_results = filter_results_by_intent(intent, merged_results)

        log.info(
            "Hybrid merge rerun completed | total_count=%s | filtered_count=%s | best_score=%s",
            len(merged_results),
            len(filtered_results),
            best_score(filtered_results),
        )

        # Apply final gate again after cold start
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

    # Log when no final relevant results exist
    if not gated_results:
        log.info("No relevant final results found | intent=%s | query=%s", intent, query)

    # Create final frontend response
    response = route_response(intent, query, gated_results)

    # Run evaluation after response generation
    evaluate_pipeline(
        query=query,
        intent=intent,
        results=gated_results,
        response=response
    )

    log.info("Response sent | type=%s", response.get("type"))

    # Return response to frontend
    return response