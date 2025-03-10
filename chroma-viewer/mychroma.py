import chromadb

# Connect to the local ChromaDB (Adjust path if needed)
client = chromadb.PersistentClient(path="../chroma_db")  

# List available collections
collections = client.list_collections()
print("Available Collections:", collections)

collection_name = "B2B_KM"
collection = client.get_collection(collection_name)
documents = collection.get()
print("retrievd documents:", documents)