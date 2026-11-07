from unittest.mock import Mock

import pytest
from haystack import Document

from retreevo.chunks import ChunkSet
from retreevo.retrieval import RerankOperator


def test_rerank_matches_haystack_and_replaces_scores_without_mutation() -> None:
    integration = pytest.importorskip(
        "haystack_integrations.components.rankers.sentence_transformers"
    )
    ranker = integration.SentenceTransformersSimilarityRanker(top_k=2)
    ranker._cross_encoder = Mock()
    ranker._cross_encoder.rank.return_value = [
        {"corpus_id": 1, "score": 0.9},
        {"corpus_id": 2, "score": 0.4},
        {"corpus_id": 0, "score": 0.1},
    ]
    chunks = ChunkSet(
        "query",
        (
            Document(id="d1", content="unrelated", score=100.0),
            Document(id="d2", content="relevant"),
            Document(id="d3", content="partly relevant", score=50.0),
        ),
    )
    direct = ranker.run(query="query", documents=list(chunks.documents))["documents"]
    rerank = RerankOperator(ranker)
    result = rerank(chunks)

    assert list(result.documents) == direct
    assert result.query == chunks.query
    assert [doc.id for doc in result.documents] == ["d2", "d3"]
    assert [doc.score for doc in result.documents] == [0.9, 0.4]
    assert [doc.score for doc in chunks.documents] == [100.0, None, 50.0]
    assert ranker._cross_encoder.rank.call_args.kwargs["documents"] == [
        "unrelated",
        "relevant",
        "partly relevant",
    ]
    ranker._cross_encoder.rank.return_value = [
        {"corpus_id": 0, "score": 0.9},
        {"corpus_id": 1, "score": 0.4},
    ]
    assert rerank(result) == result


@pytest.mark.parametrize("query, empty", [("query", True), ("", False), ("  ", False)])
def test_empty_inputs_do_not_invoke_ranker(query: str, empty: bool) -> None:
    ranker = Mock()
    documents = () if empty else (Document(id="d1", content="text"),)

    result = RerankOperator(ranker)(ChunkSet(query=query, documents=documents))

    assert result == ChunkSet(query=query, documents=())
    ranker.run.assert_not_called()
