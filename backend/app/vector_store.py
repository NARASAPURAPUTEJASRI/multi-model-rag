import chromadb
from app.config import CHROMA_PATH

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name="multimodal_rag"
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
        n_results=top_k
    )

    output = []

    for i, item_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][i]
        score = 1 - distance

        output.append({
            "id": item_id,
            "score": score,
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "rank": i + 1
        })

    return output