import time
from pymilvus import MilvusClient
from rank_bm25 import BM25Okapi
from .chunking import tokenize

def build_index(chunks: list, embedding_func, milvus_client: MilvusClient) -> tuple:
    new_collection = f"rag_docs_{int(time.time())}"
    milvus_client.create_collection(
        collection_name=new_collection,
        dimension=1024,
        metric_type="COSINE"
    )
    data = []
    for i, chunk in enumerate(chunks):
        vec = embedding_func(chunk)
        data.append({"id": i, "vector": vec, "text": chunk})
    milvus_client.insert(collection_name=new_collection, data=data)
    milvus_client.load_collection(new_collection)

    tokenized_corpus = [tokenize(doc) for doc in chunks]
    bm25_model = BM25Okapi(tokenized_corpus)

    return new_collection, bm25_model, chunks
