from rag.sparse_retriever import SparseRetriever
from rag.rrf import reciprocal_rank_fusion

class HybridRetriever:
    def __init__(self, dense_retriever, chunks, top_k=5):
        self.dense = dense_retriever
        self.sparse = SparseRetriever(chunks)
        self.top_k = top_k

    def retrieve(self, query):
        dense_results = self.dense.retrieve(query)
        sparse_results = self.sparse.retrieve(query)

        fused = reciprocal_rank_fusion(dense_results, sparse_results)
        return fused[:self.top_k]
