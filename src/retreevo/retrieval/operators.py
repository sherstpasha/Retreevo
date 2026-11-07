from __future__ import annotations

from typing import TYPE_CHECKING

from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)

if TYPE_CHECKING:
    from haystack_integrations.components.embedders.sentence_transformers import (
        SentenceTransformersTextEmbedder,
    )
    from haystack_integrations.components.rankers.sentence_transformers import (
        SentenceTransformersDiversityRanker,
        SentenceTransformersSimilarityRanker,
    )

from ..chunks import ChunkSet


class BM25Operator:
    """Adapt Haystack BM25 to ChunkSet -> ChunkSet without rebuilding the index.

    Input document IDs select candidates from the prepared retriever's store.
    Full-corpus inputs bypass the ID filter when no default filters are set.
    Scores and ordering follow Haystack. Empty candidates or a blank query
    produce an empty ChunkSet without calling the retriever.
    """

    def __init__(self, retriever: InMemoryBM25Retriever) -> None:
        self.retriever = retriever

    def __call__(self, chunks: ChunkSet) -> ChunkSet:
        if not chunks.documents or not chunks.query.strip():
            return ChunkSet(query=chunks.query, documents=())

        candidate_filter = _candidate_filter(chunks, self.retriever)
        result = self.retriever.run(query=chunks.query, filters=candidate_filter)
        return ChunkSet(query=chunks.query, documents=tuple(result["documents"]))


class DenseOperator:
    """Retrieve candidates using prepared Haystack embedding components.

    The text embedder must use the same embedding space as the store's document
    embeddings. Models and document embeddings are prepared outside this class;
    each call embeds the query and searches only the input document IDs.
    """

    def __init__(
        self,
        retriever: InMemoryEmbeddingRetriever,
        embedder: SentenceTransformersTextEmbedder,
    ) -> None:
        self.retriever = retriever
        self.embedder = embedder

    def __call__(self, chunks: ChunkSet) -> ChunkSet:
        if not chunks.documents or not chunks.query.strip():
            return ChunkSet(query=chunks.query, documents=())

        query_embedding = self.embedder.run(text=chunks.query)["embedding"]
        candidate_filter = _candidate_filter(chunks, self.retriever)
        result = self.retriever.run(
            query_embedding=query_embedding, filters=candidate_filter
        )
        return ChunkSet(query=chunks.query, documents=tuple(result["documents"]))


class RerankOperator:
    """Rescore input candidates with a prepared Haystack cross-encoder ranker.

    The ranker evaluates query-document pairs for all candidates before selecting
    top_k. Model loading is performed outside the operator. Existing retrieval
    scores are replaced by the ranker's scores without modifying input documents.
    """

    def __init__(self, ranker: SentenceTransformersSimilarityRanker) -> None:
        self.ranker = ranker

    def __call__(self, chunks: ChunkSet) -> ChunkSet:
        if not chunks.documents or not chunks.query.strip():
            return ChunkSet(query=chunks.query, documents=())

        result = self.ranker.run(query=chunks.query, documents=list(chunks.documents))
        return ChunkSet(query=chunks.query, documents=tuple(result["documents"]))


class DiversityOperator:
    """Return Haystack's diversity ordering while preserving document scores."""

    def __init__(self, ranker: SentenceTransformersDiversityRanker) -> None:
        self.ranker = ranker

    def __call__(self, chunks: ChunkSet) -> ChunkSet:
        if not chunks.documents or not chunks.query.strip():
            return ChunkSet(query=chunks.query, documents=())
        result = self.ranker.run(query=chunks.query, documents=list(chunks.documents))
        return ChunkSet(query=chunks.query, documents=tuple(result["documents"]))


class FusionOperator:
    """Adapt a Haystack DocumentJoiner to two ChunkSets for the same query.

    Documents are passed directly to the joiner, which controls scoring,
    ordering, and output limits.
    """

    def __init__(self, joiner: DocumentJoiner) -> None:
        if joiner.weights is not None and len(joiner.weights) != 2:
            raise ValueError("FusionOperator requires exactly two input weights.")
        if joiner.top_k is not None and joiner.top_k <= 0:
            raise ValueError("top_k must be positive or None.")
        self.joiner = joiner

    def __call__(self, left: ChunkSet, right: ChunkSet) -> ChunkSet:
        if left.query != right.query:
            raise ValueError("Fusion inputs must share the same query.")

        result = self.joiner.run(
            documents=[list(left.documents), list(right.documents)]
        )
        return ChunkSet(query=left.query, documents=tuple(result["documents"]))


def _candidate_filter(
    chunks: ChunkSet,
    retriever: InMemoryBM25Retriever | InMemoryEmbeddingRetriever,
) -> dict[str, str | list[str]] | None:
    candidate_ids = [document.id for document in chunks.documents]
    store = retriever.document_store
    is_full_corpus = (
        not retriever.filters
        and len(candidate_ids) == store.count_documents()
        and set(candidate_ids) == {document.id for document in store.filter_documents()}
    )
    if is_full_corpus:
        return None
    return {"field": "id", "operator": "in", "value": candidate_ids}
