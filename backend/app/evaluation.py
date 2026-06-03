"""
evaluation.py

Ragas-based backend evaluation.

Text:
- answer_relevancy
- faithfulness

Media:
- MultiModalRelevance

Evaluation does not affect retrieval, cold start, or frontend display.
"""

import asyncio

from datasets import Dataset
from ragas import evaluate, SingleTurnSample
from ragas.metrics import answer_relevancy, faithfulness
from ragas.metrics import MultiModalRelevance

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import GEMINI_API_KEY, LLM_MODEL, EMBEDDING_MODEL
from app.logger import get_logger


log = get_logger("evaluation")


ragas_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0,
    )
)


ragas_embeddings = LangchainEmbeddingsWrapper(
    GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY,
    )
)


def safe_float(value, default=0.0):
    # Convert value safely to float
    try:
        return float(value)
    except Exception:
        return default


def run_async(coro):
    # Run async Ragas metric from sync FastAPI function
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


def build_text_contexts(results):
    # Build text contexts for Ragas text metrics
    contexts = []

    for result in results:
        metadata = result.get("metadata", {})

        if metadata.get("modality") == "text":
            content = metadata.get("content") or result.get("document", "")

            if content:
                contexts.append(content)

    return contexts


def build_media_contexts(intent, results):
    # Build media contexts for Ragas MultiModalRelevance
    contexts = []

    for result in results:
        metadata = result.get("metadata", {})

        if metadata.get("modality") != intent:
            continue

        if intent == "image":
            image_context = metadata.get("file_path") or metadata.get("url")

            if image_context:
                contexts.append(image_context)

        contexts.append(
            f"""
Modality: {metadata.get("modality", "")}
URL: {metadata.get("url", "")}
Caption: {metadata.get("caption", "")}
Description: {metadata.get("description", "")}
Page Title: {metadata.get("page_title", "")}
Source Topic: {metadata.get("source_topic", "")}
"""
        )

    return contexts


def build_media_response_text(intent, results):
    # Build backend-only response text for media evaluation
    parts = []

    for result in results:
        metadata = result.get("metadata", {})

        if metadata.get("modality") != intent:
            continue

        parts.append(
            f"""
Retrieved {intent} result.
Caption: {metadata.get("caption", "")}
Description: {metadata.get("description", "")}
Page Title: {metadata.get("page_title", "")}
Source Topic: {metadata.get("source_topic", "")}
"""
        )

    if not parts:
        return f"No relevant {intent} result was retrieved."

    return "\n".join(parts)


def evaluate_text_with_ragas(query, answer, contexts):
    # Evaluate text answer using Ragas
    if not answer or not contexts:
        return {
            "ragas_answer_relevancy": 0.0,
            "ragas_faithfulness": 0.0,
        }

    try:
        dataset = Dataset.from_dict({
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        })

        result = evaluate(
            dataset,
            metrics=[
                answer_relevancy,
                faithfulness,
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )

        row = result.to_pandas().iloc[0].to_dict()

        return {
            "ragas_answer_relevancy": round(safe_float(row.get("answer_relevancy")), 4),
            "ragas_faithfulness": round(safe_float(row.get("faithfulness")), 4),
        }

    except Exception as e:
        log.warning("Ragas text evaluation failed | error=%s", str(e))

        return {
            "ragas_answer_relevancy": 0.0,
            "ragas_faithfulness": 0.0,
        }


def evaluate_media_with_ragas(query, intent, results):
    # Evaluate image/audio/video using Ragas MultiModalRelevance
    contexts = build_media_contexts(intent, results)

    if not contexts:
        return {
            "ragas_multimodal_relevance": 0.0,
        }

    response_text = build_media_response_text(intent, results)

    try:
        sample = SingleTurnSample(
            user_input=query,
            response=response_text,
            retrieved_contexts=contexts,
        )

        metric = MultiModalRelevance(llm=ragas_llm)

        score = run_async(metric.single_turn_ascore(sample))

        return {
            "ragas_multimodal_relevance": round(safe_float(score), 4),
        }

    except Exception as e:
        log.warning("Ragas multimodal evaluation failed | error=%s", str(e))

        return {
            "ragas_multimodal_relevance": 0.0,
        }


def modality_score_from_response(intent, response):
    # Validate response modality format
    if response.get("type") != intent:
        return 0.0

    if intent == "text":
        return 1.0 if response.get("answer", "") else 0.0

    results = response.get("results", [])

    if not results:
        return 0.0

    for item in results:
        if item.get("modality") != intent:
            return 0.0

    return 1.0


def calculate_final_score(intent, ragas_scores, modality_score):
    # Combine Ragas metrics into one final project score
    if intent == "text":
        answer_rel = ragas_scores.get("ragas_answer_relevancy", 0.0)
        faith = ragas_scores.get("ragas_faithfulness", 0.0)

        return round(
            (0.45 * answer_rel) +
            (0.40 * faith) +
            (0.15 * modality_score),
            4
        )

    multimodal_rel = ragas_scores.get("ragas_multimodal_relevance", 0.0)

    return round(
        (0.80 * multimodal_rel) +
        (0.20 * modality_score),
        4
    )


def evaluate_pipeline(query, intent, results, response):
    # Main evaluation function
    modality_score = modality_score_from_response(intent, response)

    if intent == "text":
        answer = response.get("answer", "")

        contexts = build_text_contexts(results)

        ragas_scores = evaluate_text_with_ragas(
            query=query,
            answer=answer,
            contexts=contexts,
        )

    else:
        ragas_scores = evaluate_media_with_ragas(
            query=query,
            intent=intent,
            results=results,
        )

    final_score = calculate_final_score(
        intent=intent,
        ragas_scores=ragas_scores,
        modality_score=modality_score,
    )

    metrics = {
        **ragas_scores,
        "modality_score": modality_score,
        "final_correctness_score": final_score,
        "retrieved_count": len(results),
        "response_type": response.get("type"),
    }

    log.info("Evaluation metrics | metrics=%s", metrics)

    return metrics