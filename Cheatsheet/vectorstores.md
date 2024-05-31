#  Vector stores

Pieces of code for vector databases

## Chroma

```
import chromadb

class ChromaClient:
    def __init__(self, mode="disk"):
        if mode == "disk":
		    # ex "./chroma_db"
            self.client = chromadb.PersistentClient(path="./chroma_db")
        elif mode == "client-server":
            self.client = self.client = chromadb.HttpClient(host="localhost", port=8000)

    def get_collections_metadata(self):
        return [col for col in self.client.list_collections()]

    def get_collections_names(self):
        return [col.name for col in self.client.list_collections()]

    def add_collection(self, collection_name):
        return self.client.create_collection(f"{collection_name}")
    
    def get_collection(self, collection_name):
        return self.client.get_collection(name=f"{collection_name}")

    def get_collection_data(self, collection_name):
        # return docs,embeddings
        collection = self.client.get_collection(collection_name)
        return (
            collection.get(include=["documents"])["documents"],
            collection.get(include=["embeddings"])["embeddings"],
        )

    def delete_collection(self, collection_name):
        return self.client.delete_collection(collection_name)
    
    def get_vectorstore(self,collection_name):
        collection = self.client.get_collection(collection_name)
        return ChromaVectorStore(chroma_collection=collection)

    def perform_similarity_search(self, query, collection_name):
        collection = self.client.get_collection(collection_name)
        docs = collection.similarity_search(query)
        return docs
```

## Qdrant

```
class QdrantClient:
    def __init__(self, mode="disk"):
        if mode == "disk":
            # ex "./qdrant_db"
            self.client = qdrant_client.QdrantClient(path="./qdrant_db")
        elif mode == "client-server":
            self.client = qdrant_client.QdrantClient(host="localhost", port=6333)

    def get_collections_metadata(self):
        return [col for col in self.client.get_collections()]

    def get_collections_names(self):
        collections = self.client.get_collections()
        return [col.name for col in collections.collections]

    def add_collection(self, collection_name):
        return self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=qdrant_client.http.models.VectorParams(size=100, distance=qdrant_client.http.models.Distance.COSINE)
            )

    def get_collection_data(self, collection_name):
        return self.client.retrieve(
            collection_name=collection_name,
            ids=[0, 3, 100],
            with_vectors=True
      )

    def delete_collection(self, collection_name):
        return self.client.delete_collection(collection_name)
```		
