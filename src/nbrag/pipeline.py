from .retrieval import rrf_fusion
from .chunking import tokenize

def advanced_rag_pipeline(
    query: str,
    collection_name: str,
    milvus_client,
    llm_client,
    bm25_model,
    doc_chunks: list
) -> tuple:
    debug_log = {}

    opt_prompt = (
        "请分析以下用户问题。如果是包含多个条件的复杂问题，请拆分为简单子问题。"
        "只输出拆分后的子问题文本，每行一个，不要输出编号和多余废话。如果问题很简单，直接原样输出。\n"
        f"用户问题：{query}"
    )
    sub_text = llm_client.chat("", opt_prompt, max_tokens=150)
    sub_queries = [sq.strip() for sq in sub_text.split('\n') if sq.strip()]
    if not sub_queries:
        sub_queries = [query]
    debug_log["sub_queries"] = sub_queries

    final_all_docs = []
    for sq in sub_queries:
        vec = llm_client.embed(sq)
        res = milvus_client.search(
            collection_name=collection_name,
            data=[vec],
            limit=5,
            output_fields=["text"]
        )
        dense_docs = [hit["entity"]["text"] for hit in res[0]]

        tok_q = tokenize(sq)
        scores = bm25_model.get_scores(tok_q)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        sparse_docs = [doc_chunks[i] for i in top_idx]

        unique_dense = list(dict.fromkeys(dense_docs))
        unique_sparse = list(dict.fromkeys(sparse_docs))
        fused = rrf_fusion(unique_dense, unique_sparse)[:10]

        rerank_scores = llm_client.rerank(sq, fused)
        scored = sorted(zip(fused, rerank_scores), key=lambda x: x[1], reverse=True)

        retained = [doc for doc, _ in scored[:2]]
        final_all_docs.extend(retained)

    unique_final = list(dict.fromkeys(final_all_docs))
    debug_log["final_compressed_count"] = len(unique_final)
    return unique_final, debug_log
