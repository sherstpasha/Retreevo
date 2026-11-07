import pytest
from haystack import Document
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from thefittest.base import FunctionalNode, TerminalNode, Tree
from thefittest.utils import create_operator
from thefittest.utils.mutations import point_mutation
from thefittest.utils.random import check_random_state, randint

from retreevo.chunks import ChunkSet
from retreevo.retrieval import BM25Operator, FusionOperator
from retreevo.gp.uniset import (
    EphemeralFunctionalNode,
    RetrievalUniversalSet,
    init_retrieval_uniset,
)


def test_tree_matches_operator_and_rebinds_terminal() -> None:
    corpus = (
        Document(id="cat", content="cat kitten"),
        Document(id="dog", content="dog puppy"),
        Document(id="bird", content="bird parrot"),
    )
    store = InMemoryDocumentStore(bm25_algorithm="BM25Okapi")
    store.write_documents(list(corpus))
    try:
        bm25 = BM25Operator(InMemoryBM25Retriever(store, top_k=1))
        initial = ChunkSet(query="cat", documents=corpus)
        uniset = init_retrieval_uniset(initial, {"bm25_1": bm25}, {})
        tree = Tree.random_tree(uniset, max_level=1)

        assert str(tree) == "bm25_1(c)"
        assert tree() == bm25(initial)
        replacement = ChunkSet(query="dog", documents=corpus)
        assert tree.set_terminals(c=replacement)() == bm25(replacement)
        assert tree().documents[0].id == "cat"
        assert all(doc.score is None for doc in corpus)
    finally:
        store.shutdown()


def test_random_trees_are_closed_and_rebind_all_leaves() -> None:
    original = ChunkSet(query="original", documents=(Document(id="d1"),))
    replacement = ChunkSet(query="replacement", documents=(Document(id="d2"),))
    observed_queries = []

    def identity(chunks: ChunkSet) -> ChunkSet:
        observed_queries.append(chunks.query)
        return chunks

    fusion = FusionOperator(DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=2))
    uniset = init_retrieval_uniset(original, {"identity": identity}, {"rrf_2": fusion})
    check_random_state(42)
    trees = [Tree.random_tree(uniset, max_level=3) for _ in range(20)]
    assert any("rrf_2" in str(tree) for tree in trees)
    for tree in trees:
        result = tree.set_terminals(c=replacement)()
        assert result.query == replacement.query
        assert all(doc.id == "d2" for doc in result.documents)
        original_result = tree()
        assert original_result.query == original.query
        assert all(doc.id == "d1" for doc in original_result.documents)
    assert "replacement" in observed_queries

    for chunks in (
        ChunkSet(query="query", documents=()),
        ChunkSet(query="", documents=()),
    ):
        for tree in trees:
            assert tree.set_terminals(c=chunks)() == chunks


def test_invalid_operator_sets_are_rejected() -> None:
    chunks = ChunkSet(query="", documents=())
    with pytest.raises(ValueError, match="At least one"):
        init_retrieval_uniset(chunks, {}, {})
    with pytest.raises(ValueError, match="distinct"):
        init_retrieval_uniset(chunks, {"same": lambda c: c}, {"same": lambda a, b: a})


def test_ephemeral_functions_are_fixed_reproducible_and_mutable() -> None:
    sampled = []

    def generate_shift() -> FunctionalNode:
        shift = int(randint(1, 1_000_000, 1)[0])
        sampled.append(shift)

        def operation(value: int) -> int:
            return value + shift

        name = f"shift_{shift}"
        return FunctionalNode(create_operator(name + "({})", name, name, operation))

    uniset = RetrievalUniversalSet(
        (EphemeralFunctionalNode(generate_shift, n_args=1),), (TerminalNode(2, "c"),)
    )
    assert sampled == []
    check_random_state(42)
    trees = [Tree.random_tree(uniset, max_level=1) for _ in range(5)]
    assert len(set(map(str, trees))) == 5
    for tree, shift in zip(trees, sampled.copy(), strict=True):
        assert type(tree._nodes[0]) is FunctionalNode
        assert str(tree) == f"shift_{shift}(c)"
        assert tree() == tree() == tree.copy()() == 2 + shift
        assert tree.set_terminals(c=10)() == 10 + shift
    assert len(sampled) == 5
    check_random_state(42)
    assert [str(Tree.random_tree(uniset, 1)) for _ in range(5)] == list(map(str, trees))

    original = trees[0]
    original_name, original_result = str(original), original()
    mutants = [point_mutation(original, uniset, 1.0, 1) for _ in range(20)]
    assert any(str(mutant) != original_name for mutant in mutants)
    for mutant in mutants:
        assert mutant._nodes[0]._n_args == 1
        assert mutant() == mutant.copy()()
    assert str(original) == original_name
    assert original() == original_result


def test_ephemeral_function_selection_respects_arity_and_fixed_nodes() -> None:
    def identity(value: int) -> int:
        return value

    def add(left: int, right: int) -> int:
        return left + right

    unary = FunctionalNode(
        create_operator("identity({})", "identity", "identity", identity)
    )
    binary = FunctionalNode(create_operator("add({}, {})", "add", "add", add))
    uniset = RetrievalUniversalSet(
        (EphemeralFunctionalNode(lambda: unary, 1), binary), (TerminalNode(2, "c"),)
    )
    assert uniset._random_functional(1) is unary
    assert uniset._random_functional(2) is binary
    binary_uniset = RetrievalUniversalSet(
        (EphemeralFunctionalNode(lambda: binary, 2),), (TerminalNode(2, "c"),)
    )
    tree = Tree.random_tree(binary_uniset, max_level=1)
    assert str(tree) == "add(c, c)"
    assert tree() == 4


def test_ephemeral_function_rejects_invalid_generator_results() -> None:
    with pytest.raises(ValueError, match="positive arity"):
        EphemeralFunctionalNode(lambda: None, n_args=0)
    with pytest.raises(TypeError, match="FunctionalNode"):
        EphemeralFunctionalNode(lambda: TerminalNode(1, "c"), n_args=1)()

    def identity(value: int) -> int:
        return value

    unary = FunctionalNode(
        create_operator("identity({})", "identity", "identity", identity)
    )
    with pytest.raises(ValueError, match="declared arity"):
        EphemeralFunctionalNode(lambda: unary, n_args=2)()
