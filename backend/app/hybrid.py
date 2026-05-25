def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    fused = {}

    def add_result(result, rank, source):
        item_id = result["id"]

        if item_id not in fused:
            fused[item_id] = {
                "id": item_id,
                "score": 0.0,
                "metadata": result.get("metadata", {}),
                "document": result.get("document", ""),
                "sources": [],
            }

        fused[item_id]["score"] += 1 / (k + rank)
        fused[item_id]["sources"].append(source)

    for rank, result in enumerate(vector_results, start=1):
        add_result(result, rank, "vector")

    for rank, result in enumerate(bm25_results, start=1):
        add_result(result, rank, "bm25")

    return sorted(
        fused.values(),
        key=lambda x: x["score"],
        reverse=True,
    )