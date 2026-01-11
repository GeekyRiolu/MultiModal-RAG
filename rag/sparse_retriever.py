from collections import defaultdict
import re

class SparseRetriever:
    def __init__(self, chunks):
        self.index = defaultdict(list)
        self.chunks = chunks
        self._build_index()

    def _build_index(self):
        for idx, chunk in enumerate(self.chunks):
            tokens = set(re.findall(r"\w+", chunk.content.lower()))
            for token in tokens:
                self.index[token].append(idx)

    def retrieve(self, query, top_k=5):
        scores = defaultdict(int)
        tokens = set(re.findall(r"\w+", query.lower()))

        for token in tokens:
            for idx in self.index.get(token, []):
                scores[idx] += 1

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self.chunks[idx] for idx, _ in ranked[:top_k]]
