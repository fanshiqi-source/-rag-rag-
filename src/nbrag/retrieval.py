from pymilvus import MilvusClient

def naive_rag_search(query: str, collection_name: str, milvus_client: MilvusClient, embedding_func) -> list:
    if not collection_name:
        return []
    vec = embedding_func(query)
    res = milvus_client.search(
        collection_name=collection_name,
        data=[vec],
        limit=3,
        output_fields=["text"]
    )
    return [hit["entity"]["text"] for hit in res[0]]

def rrf_fusion(dense_results: list, sparse_results: list, k=60) -> list:
    rrf_scores = {}
    for rank, doc in enumerate(dense_results):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (k + rank + 1)
    for rank, doc in enumerate(sparse_results):
        rrf_scores[doc] = rrf_scores.get(doc, 0) + 1 / (k + rank + 1)
    return sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
