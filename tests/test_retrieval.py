from nbrag.retrieval import rrf_fusion

def test_rrf_empty():
    assert rrf_fusion([], []) == []

def test_rrf_merge():
    dense = ["A", "B"]
    sparse = ["B", "C"]
    merged = rrf_fusion(dense, sparse)
    assert merged[0] == "B"
