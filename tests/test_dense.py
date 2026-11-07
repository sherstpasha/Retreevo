from unittest.mock import Mock, patch

import pytest
from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from retreevo.chunks import ChunkSet
from retreevo.retrieval import DenseOperator


@pytest.fixture
def corpus() -> list[Document]:
    return [
        Document(id="d1", content="first", embedding=[1.0, 0.0]),
        Document(id="d2", content="second", embedding=[0.8, 0.2]),
        Document(id="d3", content="third", embedding=[0.0, 1.0]),
    ]


@pytest.fixture
def retriever(corpus: list[Document]) -> InMemoryEmbeddingRetriever:
    store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    store.write_documents(corpus)
    return InMemoryEmbeddingRetriever(store, top_k=2)


def test_dense_matches_native_and_composes(
    corpus: list[Document], retriever: InMemoryEmbeddingRetriever
) -> None:
    embedder = Mock()
    embedder.run.return_value = {"embedding": [1.0, 0.0]}
    dense = DenseOperator(retriever, embedder)
    chunks = ChunkSet(query="query", documents=tuple(corpus))
    direct = retriever.run(query_embedding=[1.0, 0.0])["documents"]

    with patch.object(retriever, "run", wraps=retriever.run) as run:
        result = dense(chunks)

    assert run.call_args.kwargs["filters"] is None
    assert list(result.documents) == direct
    assert [doc.id for doc in result.documents] == ["d1", "d2"]
    assert dense(result) == result
    assert all(doc.score is None for doc in corpus)
    embedder.run.assert_called_with(text="query")


def test_dense_subset_excludes_better_global_match(
    corpus: list[Document], retriever: InMemoryEmbeddingRetriever
) -> None:
    embedder = Mock()
    embedder.run.return_value = {"embedding": [1.0, 0.0]}
    chunks = ChunkSet(query="query", documents=tuple(corpus[1:]))
    result = DenseOperator(retriever, embedder)(chunks)
    direct = retriever.run(
        query_embedding=[1.0, 0.0],
        filters={"field": "id", "operator": "in", "value": ["d2", "d3"]},
    )["documents"]

    assert list(result.documents) == direct
    assert [doc.id for doc in result.documents] == ["d2", "d3"]


@pytest.mark.parametrize("query, empty", [("query", True), ("", False), ("  ", False)])
def test_empty_inputs_skip_embedding(
    corpus: list[Document],
    retriever: InMemoryEmbeddingRetriever,
    query: str,
    empty: bool,
) -> None:
    embedder = Mock()
    chunks = ChunkSet(query=query, documents=() if empty else tuple(corpus))
    result = DenseOperator(retriever, embedder)(chunks)

    assert result == ChunkSet(query=query, documents=())
    embedder.run.assert_not_called()
