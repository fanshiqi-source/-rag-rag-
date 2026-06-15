import logging
import requests
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .config import SILICON_API_KEY, SILICON_BASE_URL, EMBED_MODEL, RERANK_MODEL, LLM_MODEL

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key or SILICON_API_KEY
        self.base_url = base_url or SILICON_BASE_URL
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=10),
           retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)))
    def embed(self, text: str) -> list:
        logger.info("Computing embedding...")
        resp = self.client.embeddings.create(input=text, model=EMBED_MODEL)
        return resp.data[0].embedding

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=10),
           retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)))
    def rerank(self, query: str, docs: list) -> list:
        if not docs:
            return []
        payload = {
            "model": RERANK_MODEL,
            "query": query,
            "texts": docs,
            "return_documents": False
        }
        logger.info("Calling rerank API...")
        resp = self.session.post(
            f"{self.base_url}/rerank",
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        results.sort(key=lambda x: x["index"])
        return [r["relevance_score"] for r in results]

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=10),
           retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)))
    def chat(self, system_prompt: str, user_prompt: str, max_tokens=500) -> str:
        logger.info("Chat completion...")
        resp = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()

    def chat_stream(self, system_prompt: str, user_prompt: str, max_tokens=500):
        logger.info("Streaming chat...")
        stream = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                yield delta
