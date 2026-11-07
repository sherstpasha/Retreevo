from unittest.mock import patch

import pytest
from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from retreevo.chunks import ChunkSet
from retreevo.retrieval import BM25Operator


@pytest.fixture
def corpus() -> list[Document]:
    return [
        Document(id="d1", content="cat cat"),
        Document(id="d2", content="cat dog"),
        Document(id="d3", content="bird fish"),
        Document(id="d4", content="horse sheep"),
        Document(id="d5", content="mouse rabbit"),
    ]


@pytest.fixture
def retriever(corpus: list[Document]) -> InMemoryBM25Retriever:
    store = InMemoryDocumentStore(bm25_algorithm="BM25Okapi")
    store.write_documents(corpus)
    return InMemoryBM25Retriever(document_store=store, top_k=2)


def test_operator_matches_haystack_and_composes(
    corpus: list[Document], retriever: InMemoryBM25Retriever
) -> None:
    bm25 = BM25Operator(retriever)
    chunks = ChunkSet(query="cat", documents=tuple(corpus))
    result = bm25(chunks)
    direct = retriever.run(query="cat")["documents"]

    assert list(result.documents) == direct
    assert [document.id for document in result.documents] == ["d1", "d2"]
    assert bm25(result) == result
    assert all(document.score is None for document in chunks.documents)
    assert all(
        document.score is None
        for document in retriever.document_store.filter_documents()
    )


def test_subset_retains_global_scores(
    corpus: list[Document], retriever: InMemoryBM25Retriever
) -> None:
    chunks = ChunkSet(query="cat", documents=(corpus[1],))
    result = BM25Operator(retriever)(chunks)
    direct = retriever.run(
        query="cat", filters={"field": "id", "operator": "in", "value": ["d2"]}
    )["documents"]
    full = retriever.run(query="cat")["documents"]

    assert list(result.documents) == direct
    assert result.documents[0].id == "d2"
    assert result.documents[0].score == full[1].score


@pytest.mark.parametrize("query, empty", [("cat", True), ("", False), ("  ", False)])
def test_operator_accepts_empty_inputs(
    corpus: list[Document],
    retriever: InMemoryBM25Retriever,
    query: str,
    empty: bool,
) -> None:
    documents = () if empty else tuple(corpus)
    chunks = ChunkSet(query=query, documents=documents)
    result = BM25Operator(retriever)(chunks)

    assert result.query == query
    assert result.documents == ()


@pytest.mark.parametrize("documents", [(), (Document(id="d1", content=""),)])
def test_empty_corpus_content(documents: tuple[Document, ...]) -> None:
    store = InMemoryDocumentStore(bm25_algorithm="BM25Okapi")
    store.write_documents(list(documents))
    retriever = InMemoryBM25Retriever(document_store=store)
    result = BM25Operator(retriever)(ChunkSet(query="cat", documents=documents))

    assert list(result.documents) == retriever.run(query="cat")["documents"]
    assert all(document.score == 0.0 for document in result.documents)


def test_chunk_set_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        ChunkSet(
            query="cat",
            documents=(
                Document(id="d1", content="cat"),
                Document(id="d1", content="dog"),
            ),
        )


def test_full_corpus_bypasses_filter_regardless_of_input_order(
    corpus: list[Document], retriever: InMemoryBM25Retriever
) -> None:
    expected = retriever.run(query="cat")["documents"]
    chunks = ChunkSet(query="cat", documents=tuple(reversed(corpus)))

    with patch.object(retriever, "run", wraps=retriever.run) as run:
        result = BM25Operator(retriever)(chunks)

    assert run.call_args.kwargs["filters"] is None
    assert list(result.documents) == expected


def test_equal_candidate_count_with_foreign_id_keeps_filter(
    corpus: list[Document], retriever: InMemoryBM25Retriever
) -> None:
    candidates = tuple(corpus[:-1]) + (Document(id="foreign", content="rabbit"),)
    retriever.top_k = len(corpus)
    chunks = ChunkSet(query="rabbit", documents=candidates)

    with patch.object(retriever, "run", wraps=retriever.run) as run:
        result = BM25Operator(retriever)(chunks)

    assert run.call_args.kwargs["filters"] is not None
    assert {document.id for document in result.documents} == {"d1", "d2", "d3", "d4"}


def test_full_corpus_preserves_existing_default_filter_behavior(
    corpus: list[Document], retriever: InMemoryBM25Retriever
) -> None:
    retriever.filters = {"field": "id", "operator": "==", "value": "d3"}
    expected = retriever.run(
        query="cat",
        filters={"field": "id", "operator": "in", "value": [doc.id for doc in corpus]},
    )["documents"]
    result = BM25Operator(retriever)(ChunkSet(query="cat", documents=tuple(corpus)))

    assert list(result.documents) == expected
