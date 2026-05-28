"""
debug_vectors.py

This file is used only for debugging the ChromaDB vector store.

It prints:
- Total number of stored items
- Item IDs
- Metadata
- Stored documents
- Embedding dimension
- First 20 embedding values

This helps verify whether text, image, audio, and video embeddings are stored correctly.
"""

from app.vector_store import collection

# Get all stored vector database items with embeddings, metadata, and documents
results = collection.get(
    include=["embeddings", "metadatas", "documents"]
)

# Print total number of stored vector items
print("\nTOTAL ITEMS:", len(results["ids"]))

# Loop through each stored item and print debug information
for i in range(len(results["ids"])):

    print("\n==============================")
    print("ID:", results["ids"][i])

    print("\nMETADATA:")
    print(results["metadatas"][i])

    print("\nDOCUMENT:")
    print(results["documents"][i])

    embedding = results["embeddings"][i]

    print("\nEMBEDDING DIMENSION:", len(embedding))

    print("\nFIRST 20 VECTOR VALUES:")
    print(embedding[:20])