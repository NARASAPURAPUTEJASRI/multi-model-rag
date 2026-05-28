"""
hybrid.py

This file combines vector search results and BM25 keyword search results.

It uses Reciprocal Rank Fusion (RRF).

Purpose:
- Vector search captures semantic similarity.
- BM25 captures keyword matching.
- RRF merges both rankings into one final ranked list.

This helps the multimodal RAG pipeline retrieve better results.
"""


def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    # Dictionary used to merge duplicate items from vector and BM25 results
    fused = {}

    def ensure_item(result):
        # Get unique item ID
        item_id = result["id"]

        # If item is not already added, create a new fused entry
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

    # Add vector search ranking contribution
    for rank, result in enumerate(vector_results, start=1):
        item = ensure_item(result)

        # RRF formula: 1 / (k + rank)
        rrf = 1 / (k + rank)

        item["rrf_score"] += rrf
        item["score"] += rrf
        item["sources"].append("vector")

        item["vector_score"] = result.get("vector_score", result.get("score"))
        item["distance"] = result.get("distance")

    # Add BM25 ranking contribution
    for rank, result in enumerate(bm25_results, start=1):
        item = ensure_item(result)

        # RRF formula: 1 / (k + rank)
        rrf = 1 / (k + rank)

        item["rrf_score"] += rrf
        item["score"] += rrf
        item["sources"].append("bm25")

        item["bm25_score"] = result.get("score")

    # Return merged results sorted by final RRF score
    return sorted(
        fused.values(),
        key=lambda x: x["score"],
        reverse=True,
    )