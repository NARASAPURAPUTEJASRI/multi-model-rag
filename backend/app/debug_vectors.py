from app.vector_store import collection

results = collection.get(
    include=["embeddings", "metadatas", "documents"]
)

print("\nTOTAL ITEMS:", len(results["ids"]))

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