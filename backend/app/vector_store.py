import chromadb
from app.config import CHROMA_PATH

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name="multimodal_rag",
    metadata={"hnsw:space": "cosine"}
)


def add_to_vector_db(item_id, embedding, document, metadata):
    collection.add(
        ids=[item_id],
        embeddings=[embedding],
        documents=[document],
        metadatas=[metadata]
    )


def vector_search(query_embedding, top_k=10):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    output = []

    if not results["ids"] or not results["ids"][0]:
        return output

    for i, item_id in enumerate(results["ids"][0]):
        distance = float(results["distances"][0][i])

        cosine_similarity = 1 - distance

        output.append({
            "id": item_id,
            "score": cosine_similarity,
            "vector_score": cosine_similarity,
            "distance": distance,
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "rank": i + 1
        })

    return output