from unittest.mock import Mock, patch

import pytest
from haystack import Document

from retreevo.chunks import ChunkSet
from retreevo.retrieval import DiversityOperator
from retreevo.retrieval.model_pool import DiversityModel
from retreevo.retrieval.model_worker import ModelWorker, _serve


@pytest.mark.parametrize(
    "strategy", ["maximum_margin_relevance", "greedy_diversity_order"]
)
def test_diversity_matches_native_and_preserves_scores(strategy: str) -> None:
    integration = pytest.importorskip(
        "haystack_integrations.components.rankers.sentence_transformers"
    )
    torch = pytest.importorskip("torch")
    ranker = integration.SentenceTransformersDiversityRanker(
        top_k=2, strategy=strategy, lambda_threshold=0.3
    )
    ranker.model = Mock()
    ranker.model.encode.side_effect = lambda texts, **kwargs: (
        torch.tensor([[1.0, 0.0]])
        if len(texts) == 1
        else torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    )
    corpus = tuple(
        Document(id=name, content=name, score=score)
        for name, score in (("a", 0.1), ("b", 0.9), ("c", 0.5))
    )
    chunks = ChunkSet("query", corpus)
    direct = ranker.run(query=chunks.query, documents=list(corpus))["documents"]
    result = DiversityOperator(ranker)(chunks)
    assert list(result.documents) == direct
    assert [doc.id for doc in result.documents] == ["a", "c"]
    assert [doc.score for doc in result.documents] == [0.1, 0.5]
    assert [doc.score for doc in corpus] == [0.1, 0.9, 0.5]


def test_diversity_worker_preserves_incoming_scores_across_transport() -> None:
    corpus = (Document(id="a", content="cat"), Document(id="b", content="dog"))
    model = DiversityModel("model", "revision")
    worker = ModelWorker(corpus)
    chunks = ChunkSet("query", (Document(id="b", score=0.9), Document(id="a")))
    with patch.object(
        worker, "_request", return_value=[("b", 0.9), ("a", None)]
    ) as request:
        result = worker.run(model, chunks, 2)
        request.assert_called_once_with(
            model, "query", ("b", "a"), 2, candidate_scores=(0.9, None)
        )
        assert [(doc.id, doc.score) for doc in result.documents] == [
            ("b", 0.9),
            ("a", None),
        ]
    integration = pytest.importorskip(
        "haystack_integrations.components.rankers.sentence_transformers"
    )
    connection = Mock()
    connection.recv.side_effect = [("query", ("b", "a"), 2, (0.9, None)), None]
    ranker = Mock()
    ranker.run.side_effect = lambda **kwargs: {"documents": kwargs["documents"]}
    with patch.object(
        integration, "SentenceTransformersDiversityRanker", return_value=ranker
    ):
        _serve(connection, corpus, model)
    assert [
        (doc.id, doc.score) for doc in ranker.run.call_args.kwargs["documents"]
    ] == [("b", 0.9), ("a", None)]
    assert connection.send.call_args.args[0][:2] == (True, [("b", 0.9), ("a", None)])
    assert all(doc.score is None for doc in corpus)


def test_empty_diversity_input_does_not_load_model() -> None:
    ranker = Mock()
    assert DiversityOperator(ranker)(ChunkSet("query", ())) == ChunkSet("query", ())
    ranker.run.assert_not_called()
