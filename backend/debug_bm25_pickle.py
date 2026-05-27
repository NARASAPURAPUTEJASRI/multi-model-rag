import pickle

with open("./data/bm25_index.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())

print("\nTOTAL ITEMS:")
print(len(data["items"]))

print("\nMODEL TYPE:")
print(type(data["model"]))