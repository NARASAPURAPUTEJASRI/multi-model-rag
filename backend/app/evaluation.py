"""
evaluation.py

This file evaluates the final output of the multimodal RAG pipeline.

Important:
- Evaluation runs only after final response generation.
- Evaluation does not control retrieval.
- Evaluation does not control cold start.
- Evaluation does not affect frontend display.
- Evaluation metrics are logged only in backend logs.

Metrics calculated:
- context_precision
- retrieval_score_quality
- entity_coverage
- modality_score
- relevancy_score
- final_correctness_score
"""

import re
import google.generativeai as genai

from app.config import GEMINI_API_KEY, LLM_MODEL
from app.logger import get_logger

# Create logger for evaluation module
log = get_logger("evaluation")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Gemini model used as judge for relevancy scoring
model = genai.GenerativeModel(LLM_MODEL)


# Words that should not be treated as important query entities
INTENT_WORDS = {
    "image", "images", "photo", "photos", "picture", "pictures",
    "audio", "audios", "sound", "sounds",
    "video", "videos", "clip", "clips",
    "give", "show", "tell", "about", "me", "play", "display",
    "what", "is", "who", "explain", "describe", "information", "details"
}


def tokenize(text):
    # Converts text into lowercase alphanumeric tokens
    return re.findall(r"[a-z0-9]+", str(text).lower())


def extract_query_entities(query):
    # Extracts important words from user query by removing intent/common words
    words = tokenize(query)

    return [
        word for word in words
        if word not in INTENT_WORDS and len(word) > 2
    ]


def result_text(result):
    # Builds searchable evaluation text from retrieved result and metadata
    metadata = result.get("metadata", {})

    parts = [
        result.get("document", ""),
        metadata.get("content", ""),
        metadata.get("caption", ""),
        metadata.get("description", ""),
        metadata.get("page_title", ""),
        metadata.get("source_topic", ""),
    ]

    return " ".join([str(p) for p in parts if p]).lower()


def retrieval_score_quality(results):
    """
    Uses actual final retrieval scores from RRF/entity-boosted results.
    This gives continuous values like 0.22, 0.46, 0.78.
    """

    # If there are no final retrieved results, retrieval quality is zero
    if not results:
        return 0.0

    # Extract final scores from retrieved results
    scores = [float(r.get("score", 0.0)) for r in results]

    if not scores:
        return 0.0

    # Average final retrieval score
    avg_score = sum(scores) / len(scores)

    # Normalize because RRF scores are usually small
    # Here 0.30 is treated as strong retrieval confidence
    normalized = avg_score / 0.30

    # Keep score between 0 and 1
    return round(max(0.0, min(1.0, normalized)), 4)


def entity_coverage(query, results):
    """
    Checks how many query entities are covered by retrieved result metadata/content.
    This gives a continuous score instead of only 0 or 1.
    """

    # Extract important query words
    entities = extract_query_entities(query)

    if not entities or not results:
        return 0.0

    # Combine all retrieved result text
    combined_text = " ".join([result_text(r) for r in results])

    matched = 0

    # Count how many query entities are present in retrieved context
    for entity in entities:
        if entity in combined_text:
            matched += 1

    return round(matched / len(entities), 4)


def context_precision(query, results):
    """
    Combines entity relevance and retrieval score quality.
    This is more realistic than only keyword matching.
    """

    if not results:
        return 0.0

    # Entity matching score
    entity_score = entity_coverage(query, results)

    # Retrieval confidence score
    retrieval_quality = retrieval_score_quality(results)

    # Combine both scores equally
    return round(
        (0.5 * entity_score) + (0.5 * retrieval_quality),
        4
    )


def modality_score(intent, response):
    """
    Continuous modality score.

    This checks whether the response follows the expected modality rule:
    - text query should return answer + related media if available
    - image query should return only images
    - audio query should return only audio
    - video query should return only video
    """

    response_type = response.get("type")

    # If response type does not match detected intent, score is zero
    if response_type != intent:
        return 0.0

    if intent == "text":
        score = 0.0

        answer = response.get("answer", "")
        media = response.get("media", [])

        # Text answer contributes 40%
        if answer:
            score += 0.4

        # Count media returned with text answer
        image_count = len([m for m in media if m.get("modality") == "image"])
        audio_count = len([m for m in media if m.get("modality") == "audio"])
        video_count = len([m for m in media if m.get("modality") == "video"])

        # Text query expected output:
        # answer + max 2 images + max 1 audio + max 1 video
        score += min(image_count, 2) / 2 * 0.25
        score += min(audio_count, 1) / 1 * 0.175
        score += min(video_count, 1) / 1 * 0.175

        return round(max(0.0, min(1.0, score)), 4)

    if intent == "image":
        results = response.get("results", [])

        # Count only image results
        image_count = len([r for r in results if r.get("modality") == "image"])

        # Expected output is 2 images
        return round(min(image_count, 2) / 2, 4)

    if intent == "audio":
        results = response.get("results", [])

        # Count only audio results
        audio_count = len([r for r in results if r.get("modality") == "audio"])

        # Expected output is 1 audio
        return round(min(audio_count, 1) / 1, 4)

    if intent == "video":
        results = response.get("results", [])

        # Count only video results
        video_count = len([r for r in results if r.get("modality") == "video"])

        # Expected output is 1 video
        return round(min(video_count, 1) / 1, 4)

    return 0.0


def answer_relevancy_llm(query, answer):
    # Uses Gemini as judge to score text answer relevancy

    if not answer:
        return 0.0

    prompt = f"""
You are evaluating a RAG answer.

Score from 0 to 1.

0 = irrelevant
0.25 = weakly related
0.5 = partially relevant
0.75 = mostly relevant
1 = highly relevant and directly answers the question

Question:
{query}

Answer:
{answer}

Return only one decimal number.
"""

    try:
        response = model.generate_content(prompt)
        score = float(response.text.strip())

        # Keep score between 0 and 1
        return round(max(0.0, min(1.0, score)), 4)

    except Exception as e:
        log.warning("Answer relevancy evaluation failed | error=%s", str(e))
        return 0.0


def media_relevancy_llm(query, results):
    # Uses Gemini as judge to score media relevance using metadata only

    if not results:
        return 0.0

    media_context = []

    # Build evaluation-only metadata context
    for result in results:
        metadata = result.get("metadata", {})

        media_context.append(
            f"""
Modality: {metadata.get("modality", "")}
Caption: {metadata.get("caption", "")}
Description: {metadata.get("description", "")}
Page Title: {metadata.get("page_title", "")}
Source Topic: {metadata.get("source_topic", "")}
"""
        )

    prompt = f"""
You are evaluating whether retrieved media is relevant to the query.

Score from 0 to 1.

0 = irrelevant media
0.25 = weakly related media
0.5 = partially relevant media
0.75 = mostly relevant media
1 = highly relevant media

Query:
{query}

Retrieved Media Metadata:
{" ".join(media_context)}

Return only one decimal number.
"""

    try:
        response = model.generate_content(prompt)
        score = float(response.text.strip())

        # Keep score between 0 and 1
        return round(max(0.0, min(1.0, score)), 4)

    except Exception as e:
        log.warning("Media relevancy evaluation failed | error=%s", str(e))
        return 0.0


def final_correctness_score(context_score, modality_score_value, relevancy_score):
    # Combines retrieval quality, modality correctness, and relevancy into final score
    return round(
        (0.4 * context_score) +
        (0.3 * modality_score_value) +
        (0.3 * relevancy_score),
        4
    )


def evaluate_pipeline(query, intent, results, response):
    # Main evaluation function called after final response is generated

    # Calculate context quality score
    context_score = context_precision(query, results)

    # Calculate modality output score
    modality_score_value = modality_score(intent, response)

    # For text intent, evaluate generated answer
    if intent == "text":
        relevancy_score = answer_relevancy_llm(
            query=query,
            answer=response.get("answer", "")
        )

    # For image/audio/video intent, evaluate media metadata relevance
    else:
        relevancy_score = media_relevancy_llm(
            query=query,
            results=results
        )

    # Calculate final score
    final_score = final_correctness_score(
        context_score=context_score,
        modality_score_value=modality_score_value,
        relevancy_score=relevancy_score
    )

    # Store all evaluation metrics
    metrics = {
        "context_precision": context_score,
        "retrieval_score_quality": retrieval_score_quality(results),
        "entity_coverage": entity_coverage(query, results),
        "modality_score": modality_score_value,
        "relevancy_score": relevancy_score,
        "final_correctness_score": final_score,
        "retrieved_count": len(results),
        "response_type": response.get("type")
    }

    # Log metrics only in backend logs
    log.info("Evaluation metrics | metrics=%s", metrics)

    return metrics