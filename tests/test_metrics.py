from math import log2

import pytest
from haystack import Document
from haystack.components.evaluators import DocumentNDCGEvaluator

from retreevo.chunks import ChunkSet
from retreevo.metrics import MRR, NDCG, RecallAtK


def test_recall_uses_ids_positive_grades_and_cutoff() -> None:
    qrels = {"q1": {"d1": 2, "d2": 1, "d3": 0}, "q2": {"d4": 1}}
    results = {
        "q2": ChunkSet("second", (Document(id="d4", content="same"),)),
        "q1": ChunkSet(
            "first",
            (Document(id="d1", content="same"), Document(id="d2", content="same")),
        ),
    }

    first = RecallAtK(k=1).evaluate(qrels, results)
    second = RecallAtK(k=2).evaluate(qrels, results)

    assert first.per_query == {"q1": 0.5, "q2": 1.0}
    assert first.score == 0.75
    assert second.score == 1.0


def test_missing_results_contribute_zero() -> None:
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    results = {"q1": ChunkSet("first", (Document(id="d1"),))}

    result = RecallAtK().evaluate(qrels, results)

    assert result.per_query == {"q1": 1.0, "q2": 0.0}
    assert result.score == 0.5


def test_no_positive_judgments_score_zero() -> None:
    result = RecallAtK().evaluate(
        {"q1": {"d1": 0}}, {"q1": ChunkSet("first", (Document(id="d1"),))}
    )

    assert result.score == 0.0


def test_rejects_invalid_evaluation_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        RecallAtK(k=0)
    with pytest.raises(ValueError, match="at least one"):
        RecallAtK().evaluate({}, {})
    with pytest.raises(ValueError, match="query IDs"):
        RecallAtK().evaluate({"q1": {"d1": 1}}, {"unknown": ChunkSet("query", ())})


def test_mrr_uses_first_positive_hit_and_preserves_all_queries() -> None:
    qrels = {
        "q1": {"a": 1, "b": 2, "zero": 0},
        "q2": {"a": 1},
        "missing": {"a": 1},
        "no_positive": {"a": 0},
    }
    results = {
        "q1": ChunkSet("first", tuple(Document(id=doc) for doc in ("zero", "a", "b"))),
        "q2": ChunkSet("second", (Document(id="a"),)),
        "no_positive": ChunkSet("third", (Document(id="a"),)),
    }
    result = MRR().evaluate(qrels, results)
    assert result.per_query == {
        "q1": 0.5,
        "q2": 1.0,
        "missing": 0.0,
        "no_positive": 0.0,
    }
    assert result.score == 0.375
    assert MRR(k=1).evaluate(qrels, results).score == 0.25


def test_mrr_distinguishes_rankings_with_identical_recall() -> None:
    qrels = {"q": {"relevant": 1}}
    first = {"q": ChunkSet("query", (Document(id="relevant"), Document(id="other")))}
    second = {"q": ChunkSet("query", (Document(id="other"), Document(id="relevant")))}
    assert (
        RecallAtK(2).evaluate(qrels, first).score
        == RecallAtK(2).evaluate(qrels, second).score
    )
    assert MRR().evaluate(qrels, first).score == 1.0
    assert MRR().evaluate(qrels, second).score == 0.5


def test_ndcg_uses_grades_and_matches_native_haystack() -> None:
    qrels = {"q": {"a": 3, "b": 1, "zero": 0}}
    documents = (Document(id="b", score=99), Document(id="zero"), Document(id="a"))
    result = NDCG().evaluate(qrels, {"q": ChunkSet("query", documents)})
    native = DocumentNDCGEvaluator(document_comparison_field="id").run(
        ground_truth_documents=[[Document(id="a", score=3), Document(id="b", score=1)]],
        retrieved_documents=[list(documents)],
    )
    assert result.score == native["score"]
    assert result.score == pytest.approx((1 + 3 / log2(4)) / (3 + 1 / log2(3)))


def test_ndcg_sees_later_relevant_documents_that_mrr_ignores() -> None:
    qrels = {"q": {"a": 1, "b": 1}}
    better = {"q": ChunkSet("query", (Document(id="a"), Document(id="b")))}
    worse = {
        "q": ChunkSet(
            "query", (Document(id="a"), Document(id="other"), Document(id="b"))
        )
    }
    assert MRR().evaluate(qrels, better).score == MRR().evaluate(qrels, worse).score
    assert NDCG().evaluate(qrels, better).score == 1.0
    assert NDCG().evaluate(qrels, worse).score < 1.0


def test_ndcg_missing_hits_and_nonpositive_judgments_score_zero() -> None:
    qrels = {"hit": {"a": 1}, "missing": {"b": 1}, "none": {"a": 0, "b": -1}}
    results = {
        "hit": ChunkSet("query", (Document(id="a"),)),
        "none": ChunkSet("query", (Document(id="a"), Document(id="b"))),
    }
    result = NDCG().evaluate(qrels, results)
    assert result.per_query == {"hit": 1.0, "missing": 0.0, "none": 0.0}
    assert result.score == pytest.approx(1 / 3)
