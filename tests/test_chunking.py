from nbrag.chunking import smart_chunk_text, tokenize

def test_short_text():
    res = smart_chunk_text("Hello world", 100, 20)
    assert len(res) == 1

def test_paragraph_boundary():
    text = "段落一\n\n段落二"
    res = smart_chunk_text(text, 3, 1)
    assert len(res) >= 2

def test_tokenize():
    tokens = tokenize("主管岗转正")
    assert "主管" in tokens
