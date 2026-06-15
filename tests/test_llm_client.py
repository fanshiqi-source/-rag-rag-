import os

def test_embed_success(mocker):
    os.environ["SILICON_API_KEY"] = "test-key"
    mocker.patch("nbrag.llm_client.OpenAI")
    from unittest.mock import MagicMock
    from nbrag.llm_client import LLMClient
    mock_data = MagicMock()
    mock_data.data = [MagicMock(embedding=[0.1] * 1024)]
    import nbrag.llm_client as llm_module
    llm_module.OpenAI.return_value.embeddings.create.return_value = mock_data
    client = LLMClient(api_key="fake", base_url="http://fake")
    emb = client.embed("test")
    assert len(emb) == 1024
