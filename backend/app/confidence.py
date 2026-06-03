"""
confidence.py

Converts raw retrieval scores into normalized confidence scores.

Old score:
- RRF + entity boost values were small, like 0.07 or 0.08.

New score:
- confidence_score is normalized between 0.0 and 1.0.
- This allows thresholds like 0.70 and 0.80.
"""

from app.text_utils import entity_coverage


def clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def normalized_vector_score(result):
    score = float(result.get("vector_score", 0.0) or 0.0)
    return clamp(score)


def normalized_rrf_score(result):
    rrf_score = float(result.get("rrf_score", result.get("score", 0.0)) or 0.0)

    # Practical max RRF when same item appears high in vector + BM25
    practical_max_rrf = 0.033

    return clamp(rrf_score / practical_max_rrf)


def modality_match_score(intent, result):
    modality = result.get("metadata", {}).get("modality")

    if intent == "text":
        return 1.0 if modality in ["text", "image", "audio", "video"] else 0.0

    return 1.0 if modality == intent else 0.0


def calculate_result_confidence(query, intent, result):
    vector_part = normalized_vector_score(result)
    rrf_part = normalized_rrf_score(result)
    entity_part = entity_coverage(query, [result])
    modality_part = modality_match_score(intent, result)

    confidence = (
        0.45 * vector_part +
        0.25 * rrf_part +
        0.20 * entity_part +
        0.10 * modality_part
    )

    return round(clamp(confidence), 4)


def best_confidence(query, intent, results):
    if not results:
        return 0.0

    scores = [
        calculate_result_confidence(query, intent, result)
        for result in results
    ]

    return round(max(scores), 4)