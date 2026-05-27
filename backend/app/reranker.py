import re

INTENT_WORDS = {
    "image", "images", "photo", "photos", "picture", "pictures",
    "audio", "audios", "sound", "sounds",
    "video", "videos", "clip", "clips",
    "give", "gave", "show", "tell", "about", "me", "information",
    "play", "display"
}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "to", "of", "in", "on", "at", "for", "from", "by",
    "and", "or", "with", "this", "that"
}


def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def extract_query_entities(query: str):
    words = tokenize(query)

    entities = []

    for word in words:
        if word in INTENT_WORDS:
            continue
        if word in STOPWORDS:
            continue
        if len(word) <= 2:
            continue

        entities.append(word)

    return entities


def get_result_text(result):
    metadata = result.get("metadata", {})

    parts = [
        result.get("document", ""),
        metadata.get("content", ""),
        metadata.get("description", ""),
        metadata.get("caption", ""),
        metadata.get("page_title", ""),
        metadata.get("source_topic", ""),
        metadata.get("modality", ""),
    ]

    return " ".join([str(p) for p in parts if p])


def exact_entity_match_count(query_entities, result_text):
    result_tokens = set(tokenize(result_text))

    count = 0

    for entity in query_entities:
        if entity in result_tokens:
            count += 1

    return count


def has_required_entity(query, result):
    query_entities = extract_query_entities(query)

    if not query_entities:
        return True

    result_text = get_result_text(result).lower()
    result_tokens = set(tokenize(result_text))

    # For multi-word query entities, require all entity words
    if len(query_entities) >= 2:
        return all(entity in result_tokens for entity in query_entities)

    return query_entities[0] in result_tokens

def rerank_results_after_rrf(query, results):
    query_entities = extract_query_entities(query)

    if not query_entities:
        return results

    reranked = []

    for result in results:
        result_text = get_result_text(result)

        base_score = float(result.get("score", 0))
        match_count = exact_entity_match_count(query_entities, result_text)

        entity_boost = match_count * 0.08

        result["base_rrf_score"] = base_score
        result["entity_match_count"] = match_count
        result["entity_boost"] = entity_boost
        result["score"] = base_score + entity_boost

        reranked.append(result)

    reranked.sort(key=lambda x: x.get("score", 0), reverse=True)

    return reranked