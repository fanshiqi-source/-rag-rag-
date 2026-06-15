import re
import jieba

def tokenize(text: str) -> list:
    return list(jieba.cut_for_search(text))

def smart_chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 200) -> list:
    paragraphs = re.split(r'\n+', text)
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(current_chunk) + len(p) < chunk_size:
            current_chunk += p + "\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(p) >= chunk_size:
                start = 0
                while start < len(p):
                    chunks.append(p[start:start + chunk_size])
                    start += (chunk_size - chunk_overlap)
                current_chunk = ""
            else:
                current_chunk = p + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks
