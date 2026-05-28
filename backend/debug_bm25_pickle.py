"""
debug_bm25_pickle.py

This file is used to debug the saved BM25 index pickle file.

It prints:
- keys available inside the pickle file
- total indexed BM25 items
- type of BM25 model object

This helps verify whether BM25 index was saved correctly.
"""

import pickle

# Open saved BM25 index file
with open("./data/bm25_index.pkl", "rb") as f:
    data = pickle.load(f)

# Print available dictionary keys
print(data.keys())

# Print total BM25 indexed items
print("\nTOTAL ITEMS:")
print(len(data["items"]))

# Print BM25 model type
print("\nMODEL TYPE:")
print(type(data["model"]))