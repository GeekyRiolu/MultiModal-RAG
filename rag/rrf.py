def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    scores = {}

    for rank, chunk in enumerate(dense_results):
        scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (k + rank + 1)

    for rank, chunk in enumerate(sparse_results):
        scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (k + rank + 1)

    ranked_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    chunk_map = {c.id: c for c in dense_results + sparse_results}

    return [chunk_map[cid] for cid, _ in ranked_ids]
