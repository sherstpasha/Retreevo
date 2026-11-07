from collections.abc import Mapping
from dataclasses import dataclass

from haystack import Document
from haystack.components.evaluators import (
    DocumentMRREvaluator,
    DocumentNDCGEvaluator,
    DocumentRecallEvaluator,
)

from .chunks import ChunkSet


@dataclass(frozen=True)
class MetricResult:
    score: float
    per_query: dict[str, float]


class _DocumentMetric:
    _evaluator: DocumentRecallEvaluator | DocumentMRREvaluator | DocumentNDCGEvaluator

    def __init__(self, k: int | None) -> None:
        if k is not None and k <= 0:
            raise ValueError("k must be positive.")
        self.k = k

    def evaluate(
        self,
        qrels: Mapping[str, Mapping[str, int]],
        results: Mapping[str, ChunkSet],
    ) -> MetricResult:
        if not qrels:
            raise ValueError("qrels must contain at least one query.")
        if results.keys() - qrels.keys():
            raise ValueError("Result query IDs must be present in qrels.")

        query_ids = list(qrels)
        ground_truth_documents = [
            [
                Document(id=document_id, score=float(grade))
                for document_id, grade in qrels[query_id].items()
                if grade > 0
            ]
            for query_id in query_ids
        ]
        retrieved_documents = [
            list(results[query_id].documents[: self.k]) if query_id in results else []
            for query_id in query_ids
        ]
        evaluation = self._evaluator.run(
            ground_truth_documents=ground_truth_documents,
            retrieved_documents=retrieved_documents,
        )
        return MetricResult(
            score=evaluation["score"],
            per_query=dict(
                zip(query_ids, evaluation["individual_scores"], strict=True)
            ),
        )


class RecallAtK(_DocumentMetric):
    """Evaluate Haystack multi-hit recall by document ID within the first k hits.

    Positive grades count as relevant. Missing results and queries without
    positive judgments contribute zero to the mean.
    """

    def __init__(self, k: int = 100) -> None:
        super().__init__(k)
        self._evaluator = DocumentRecallEvaluator(
            mode="multi_hit", document_comparison_field="id"
        )


class MRR(_DocumentMetric):
    """Average reciprocal rank of the first relevant document using Haystack.

    Positive grades count as relevant. Missing results and queries without a
    relevant hit contribute zero. With k=None, the entire returned ranking is
    considered; a positive k restricts evaluation to the first k documents.
    """

    def __init__(self, k: int | None = None) -> None:
        super().__init__(k)
        self._evaluator = DocumentMRREvaluator(document_comparison_field="id")


class NDCG(_DocumentMetric):
    """Evaluate native Haystack NDCG by ID using positive qrels grades as gains.

    All returned positions contribute, with logarithmic rank discounts. Haystack
    normalizes against all positive judgments, even when retrieval is limited.
    Missing results and queries without positive judgments contribute zero.
    """

    def __init__(self) -> None:
        super().__init__(None)
        self._evaluator = DocumentNDCGEvaluator(document_comparison_field="id")
