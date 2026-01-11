from embeddings.embedder import Embedder

class Retriever:
    def __init__(self, store, embedder: Embedder, top_k: int = 5):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str):
        query_embedding = self.embedder.embed([query])[0]
        return self.store.search(query_embedding, k=self.top_k)
