import re
import google.generativeai as genai

from app.config import GEMINI_API_KEY, LLM_MODEL
from app.logger import get_logger

log = get_logger("evaluation")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(LLM_MODEL)


INTENT_WORDS = {
    "image", "images", "photo", "photos", "picture", "pictures",
    "audio", "audios", "sound", "sounds",
    "video", "videos", "clip", "clips",
    "give", "show", "tell", "about", "me", "play", "display",
    "what", "is", "who", "explain", "describe"
}


def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def extract_query_entities(query):
    words = tokenize(query)
    return [
        w for w in words
        if w not in INTENT_WORDS and len(w) > 2
    ]


def result_text(result):
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


def context_precision(query, results):
    if not results:
        return 0.0

    entities = extract_query_entities(query)

    if not entities:
        return 0.0

    relevant = 0

    for result in results:
        text = result_text(result)

        if any(entity in text for entity in entities):
            relevant += 1

    return round(relevant / len(results), 4)


def modality_correctness(intent, response):
    if intent == "text":
        return 1.0 if response.get("type") == "text" else 0.0

    if intent in ["image", "audio", "video"]:
        if response.get("type") != intent:
            return 0.0

        results = response.get("results", [])

        if not results:
            return 0.0

        for item in results:
            if item.get("modality") != intent:
                return 0.0

        return 1.0

    return 0.0


def answer_relevancy_llm(query, answer):
    if not answer:
        return 0.0

    prompt = f"""
You are evaluating a RAG system answer.

Give a score from 0 to 1.

0 = answer is irrelevant
0.5 = partially relevant
1 = highly relevant and directly answers the question

Question:
{query}

Answer:
{answer}

Return only one number.
"""

    try:
        response = model.generate_content(prompt)
        score_text = response.text.strip()
        score = float(score_text)
        return round(max(0.0, min(1.0, score)), 4)

    except Exception as e:
        log.warning("LLM answer relevancy failed | error=%s", str(e))
        return 0.0


def final_correctness_score(intent, context_score, modality_score, answer_score):
    if intent == "text":
        return round(
            (0.4 * context_score) +
            (0.3 * modality_score) +
            (0.3 * answer_score),
            4
        )

    return round(
        (0.6 * context_score) +
        (0.4 * modality_score),
        4
    )


def evaluate_pipeline(query, intent, results, response):
    context_score = context_precision(query, results)
    modality_score = modality_correctness(intent, response)

    answer_score = None

    if intent == "text":
        answer_score = answer_relevancy_llm(
            query=query,
            answer=response.get("answer", "")
        )

    final_score = final_correctness_score(
        intent=intent,
        context_score=context_score,
        modality_score=modality_score,
        answer_score=answer_score or 0.0
    )

    metrics = {
        "context_precision": context_score,
        "modality_correctness": modality_score,
        "answer_relevancy": answer_score,
        "final_correctness_score": final_score,
        "retrieved_count": len(results),
        "response_type": response.get("type")
    }

    log.info("Evaluation metrics | metrics=%s", metrics)

    return metrics