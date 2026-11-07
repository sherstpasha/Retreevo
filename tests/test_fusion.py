import pytest
from haystack import Document
from haystack.components.joiners import DocumentJoiner

from retreevo.chunks import ChunkSet
from retreevo.retrieval import FusionOperator


MODES = [
    "concatenate",
    "merge",
    "reciprocal_rank_fusion",
    "distribution_based_rank_fusion",
]


@pytest.mark.parametrize("mode", MODES)
def test_fusion_matches_native_and_preserves_inputs(mode: str) -> None:
    left = ChunkSet("query", (Document(id="a", score=5.0), Document(id="b", score=3.0)))
    right = ChunkSet(
        "query", (Document(id="b", score=0.9), Document(id="c", score=0.5))
    )
    joiner = DocumentJoiner(join_mode=mode, weights=[0.3, 0.7], top_k=2)
    direct = joiner.run(documents=[list(left.documents), list(right.documents)])

    result = FusionOperator(joiner)(left, right)

    assert list(result.documents) == direct["documents"]
    assert result.query == "query"
    assert [doc.score for doc in left.documents] == [5.0, 3.0]
    assert [doc.score for doc in right.documents] == [0.9, 0.5]


@pytest.mark.parametrize("mode", MODES)
def test_empty_inputs_match_native(mode: str) -> None:
    joiner = DocumentJoiner(join_mode=mode, top_k=3)
    empty = ChunkSet("query", ())
    ranked = ChunkSet("query", (Document(id="a", score=1.0),))
    fusion = FusionOperator(joiner)
    assert fusion(empty, empty) == empty
    assert (
        list(fusion(empty, ranked).documents)
        == joiner.run(documents=[[], list(ranked.documents)])["documents"]
    )


def test_unranked_documents_are_forwarded_to_native_joiner() -> None:
    joiner = DocumentJoiner(join_mode="concatenate", top_k=3)
    left = ChunkSet("query", (Document(id="b"),))
    right = ChunkSet("query", (Document(id="a", score=1.0),))
    result = FusionOperator(joiner)(left, right)
    assert (
        list(result.documents)
        == joiner.run(documents=[list(left.documents), list(right.documents)])[
            "documents"
        ]
    )
    assert {doc.id for doc in result.documents} == {"a", "b"}


def test_rejects_mismatched_queries_and_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="same query"):
        FusionOperator(DocumentJoiner())(ChunkSet("first", ()), ChunkSet("second", ()))
    with pytest.raises(ValueError, match="two input weights"):
        FusionOperator(DocumentJoiner(weights=[1.0]))
    with pytest.raises(ValueError, match="positive"):
        FusionOperator(DocumentJoiner(top_k=0))
