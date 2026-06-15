import os

SILICON_API_KEY = os.getenv("SILICON_API_KEY")
if not SILICON_API_KEY:
    raise ValueError("SILICON_API_KEY environment variable must be set")

SILICON_BASE_URL = "https://api.siliconflow.cn/v1"

LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct"
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 200
