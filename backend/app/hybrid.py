def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    fused = {}

    def ensure_item(result):
        item_id = result["id"]

        if item_id not in fused:
            fused[item_id] = {
                "id": item_id,
                "score": 0.0,
                "rrf_score": 0.0,
                "vector_score": result.get("vector_score"),
                "bm25_score": result.get("bm25_score"),
                "distance": result.get("distance"),
                "metadata": result.get("metadata", {}),
                "document": result.get("document", ""),
                "sources": [],
            }

        return fused[item_id]

    for rank, result in enumerate(vector_results, start=1):
        item = ensure_item(result)
        rrf = 1 / (k + rank)

        item["rrf_score"] += rrf
        item["score"] += rrf
        item["sources"].append("vector")

        item["vector_score"] = result.get("vector_score", result.get("score"))
        item["distance"] = result.get("distance")

    for rank, result in enumerate(bm25_results, start=1):
        item = ensure_item(result)
        rrf = 1 / (k + rank)

        item["rrf_score"] += rrf
        item["score"] += rrf
        item["sources"].append("bm25")

        item["bm25_score"] = result.get("score")

    return sorted(
        fused.values(),
        key=lambda x: x["score"],
        reverse=True,
    )