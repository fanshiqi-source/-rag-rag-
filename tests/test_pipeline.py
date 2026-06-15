from nbrag.pipeline import advanced_rag_pipeline

def test_pipeline_no_subqueries(mocker):
    mock_llm = mocker.MagicMock()
    mock_llm.chat.return_value = ""
    mock_llm.embed.return_value = [0.1] * 1024
    mock_llm.rerank.return_value = [0.9, 0.8]
    mock_milvus = mocker.MagicMock()
    mock_milvus.search.return_value = [[
        mocker.MagicMock(entity={"text": "doc1"}),
        mocker.MagicMock(entity={"text": "doc2"})
    ]]
    mock_bm25 = mocker.MagicMock()
    mock_bm25.get_scores.return_value = [0.5, 0.3]

    docs, debug = advanced_rag_pipeline(
        "test query", "col", mock_milvus, mock_llm, mock_bm25, ["doc1", "doc2"]
    )
    assert len(docs) > 0
    assert "final_compressed_count" in debug
