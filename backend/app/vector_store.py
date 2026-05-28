"""
vector_store.py

This file manages ChromaDB vector storage and vector search.

Main responsibilities:
- Create/load persistent ChromaDB collection
- Add embeddings into vector database
- Search embeddings using query embedding
- Return cosine similarity scores

This stores all modalities together:
- text embeddings
- image embeddings
- audio embeddings
- video embeddings
"""

import chromadb
from app.config import CHROMA_PATH

# Create persistent ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Create or load multimodal RAG collection using cosine distance
collection = client.get_or_create_collection(
    name="multimodal_rag",
    metadata={"hnsw:space": "cosine"}
)


def add_to_vector_db(item_id, embedding, document, metadata):
    # Adds one item embedding into ChromaDB

    collection.add(
        ids=[item_id],
        embeddings=[embedding],
        documents=[document],
        metadatas=[metadata]
    )


def vector_search(query_embedding, top_k=10):
    # Searches ChromaDB using query embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    output = []

    # Return empty list if no results exist
    if not results["ids"] or not results["ids"][0]:
        return output

    for i, item_id in enumerate(results["ids"][0]):
        # Chroma returns cosine distance
        distance = float(results["distances"][0][i])

        # Convert cosine distance to cosine similarity
        cosine_similarity = 1 - distance

        # Return result in common retrieval format
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