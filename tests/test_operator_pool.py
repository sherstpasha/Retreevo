from collections.abc import Callable
from functools import partial
from dataclasses import asdict
import json
from unittest.mock import patch

import numpy as np
import pytest
from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from thefittest.base import FunctionalNode, TerminalNode, Tree
from thefittest.optimizers import GeneticProgramming
from thefittest.utils import create_operator
from thefittest.utils.random import check_random_state

from retreevo.chunks import ChunkSet
from retreevo.retrieval.model_worker import ModelWorker
from retreevo.gp.operator_pool import (
    GP_DENSE_MODELS,
    GP_RERANK_MODELS,
    GP_DIVERSITY_MODELS,
    OperatorPool,
    BM25Config,
    NeuralConfig,
    generate_bm25_node,
    generate_dense_node,
    generate_rerank_node,
    generate_fusion_node,
    generate_diversity_node,
)
from retreevo.gp.uniset import EphemeralFunctionalNode, RetrievalUniversalSet
from retreevo.retrieval.model_pool import DenseModel, RerankModel, DiversityModel


def test_pool_generates_and_binds_without_loading_resources() -> None:
    corpus = (
        Document(id="a", content="cat kitten"),
        Document(id="b", content="dog puppy"),
    )
    with OperatorPool(corpus) as pool:
        assert len(pool.uniset._functional_set[-1]) == 8
        check_random_state(42)
        population = np.array(
            [Tree.random_tree(pool.uniset, 2) for _ in range(10)], dtype=object
        )
        originals = list(map(str, population))
        phenotypes = pool.prepare_population(population)
        assert list(map(str, population)) == originals
        assert len(phenotypes) == 10
        assert pool._store is None
        assert pool.worker.loads == []
        assert all(
            tree.set_terminals(c=ChunkSet(query="", documents=()))().documents == ()
            for tree in phenotypes
        )
        assert pool._store is None
        assert pool.worker.loads == []


def test_bm25_execution_is_native_and_releases_previous_index() -> None:
    corpus = tuple(
        Document(id=str(i), content="cat " * (i + 1) + "dog " * (120 - i))
        for i in range(120)
    )
    chunks = ChunkSet(query="cat", documents=corpus)
    first = generate_bm25_node(
        top_k_range=(10, 10), algorithms=("BM25Okapi",), k1_range=(1.5, 1.5)
    )
    second = generate_bm25_node(top_k_range=(50, 50), algorithms=("BM25L",))
    with OperatorPool(corpus) as pool:
        raw = Tree([first, TerminalNode(chunks, "c")])
        with pytest.raises(RuntimeError, match="prepare_population"):
            raw()
        phenotype = pool.prepare_population(np.array([raw], dtype=object))[0]
        result = phenotype()
        old_store = pool._store
        direct = InMemoryBM25Retriever(old_store, top_k=10).run(query="cat")[
            "documents"
        ]
        assert list(result.documents) == direct
        assert str(raw) == str(phenotype)
        second_tree = Tree([second, TerminalNode(chunks, "c")])
        pool.prepare_population(np.array([second_tree], dtype=object))[0]()
        assert old_store.count_documents() == 0
        assert pool._store is not old_store
        last_store = pool._store
    assert last_store.count_documents() == 0
    assert all(doc.score is None and doc.embedding is None for doc in corpus)


def test_neural_worker_rejects_unknown_candidates_without_starting() -> None:
    worker = ModelWorker((Document(id="known"),))
    with pytest.raises(ValueError, match="Unknown candidate"):
        worker.run(
            GP_DENSE_MODELS[0],
            ChunkSet(query="query", documents=(Document(id="unknown"),)),
            10,
        )
    assert worker.active_pid is None
    worker.close()
    worker.close()


@pytest.mark.parametrize("model", [GP_DENSE_MODELS[0], GP_RERANK_MODELS[0]])
def test_neural_worker_requests_each_call_without_caching(
    model: DenseModel | RerankModel,
) -> None:
    corpus = (Document(id="a", content="cat"),)
    worker = ModelWorker(corpus)
    chunks = ChunkSet("cat", corpus)
    with patch.object(worker, "_request", return_value=[("a", 0.5)]) as request:
        first = worker.run(model, chunks, 1)
        second = worker.run(model, chunks, 1)
        assert first == second
        assert request.call_count == 2
        request.assert_called_with(model, "cat", ("a",), 1)
    assert corpus[0].score is None
    worker.close()


def test_gp_repairs_ephemeral_rerank_without_mutating_genotypes() -> None:
    corpus = tuple(Document(id=str(i), content=f"cat document {i}") for i in range(120))
    chunks = ChunkSet(query="cat", documents=corpus)
    model = RerankModel("test-rerank", "v1")
    factory = partial(generate_rerank_node, models=(model,), top_k_range=(10, 50))
    unsafe = Tree([factory(), TerminalNode(chunks, "c")])
    original = str(unsafe)
    with (
        OperatorPool(corpus) as pool,
        patch.object(pool.worker, "run") as run,
    ):
        uniset = RetrievalUniversalSet(
            (EphemeralFunctionalNode(factory, 1),), (TerminalNode(chunks, "c"),)
        )
        gp = GeneticProgramming(
            fitness_function=lambda population: np.array(
                [len(tree().documents) for tree in population], dtype=float
            ),
            genotype_to_phenotype=pool.prepare_population,
            uniset=uniset,
            iters=3,
            pop_size=4,
            init_population=np.array([unsafe] * 4, dtype=object),
            mutation="gp_custom_rate_grow",
            mutation_rate=1.0,
            max_level=3,
            random_state=42,
            keep_history=True,
        )
        gp.fit()
        assert str(unsafe) == original
        for population in gp.get_stats()["population_ph"]:
            for tree in population:
                assert str(tree) == "c"
                assert str(pool.repair_tree(tree)) == "c"
        run.assert_not_called()
        assert pool._store is None
        assert pool.worker.loads == []


def test_repair_compares_configurations_and_preserves_other_operations() -> None:
    chunks = ChunkSet(query="cat", documents=())
    model = RerankModel("test", "v1")
    inner = generate_rerank_node(models=(model,), top_k_range=(10, 10))
    outer = generate_rerank_node(models=(model,), top_k_range=(50, 50))
    other = generate_rerank_node(
        models=(RerankModel("other", "v1"),), top_k_range=(50, 50)
    )
    dense_inner = generate_dense_node(
        models=(GP_DENSE_MODELS[0],), top_k_range=(10, 10)
    )
    dense_outer = generate_dense_node(
        models=(GP_DENSE_MODELS[0],), top_k_range=(100, 100)
    )
    bm25_inner = generate_bm25_node(
        top_k_range=(10, 10),
        algorithms=("BM25Okapi",),
        k1_range=(1.5, 1.5),
        b_range=(0.75, 0.75),
    )
    bm25_outer = generate_bm25_node(
        top_k_range=(100, 100),
        algorithms=("BM25Okapi",),
        k1_range=(1.5, 1.5),
        b_range=(0.75, 0.75),
    )
    with OperatorPool(()) as pool:
        for larger, smaller in (
            (outer, inner),
            (dense_outer, dense_inner),
            (bm25_outer, bm25_inner),
        ):
            for nodes in (
                (larger, smaller),
                (smaller, larger),
                (larger, smaller, larger),
            ):
                tree = Tree([*nodes, TerminalNode(chunks, "c")])
                repaired = pool.repair_tree(tree)
                expected = smaller._name + "(c)"
                if smaller is inner:
                    expected = "c"
                assert str(repaired) == expected
                assert str(pool.repair_tree(repaired)) == expected
                assert len(tree) == len(nodes) + 1
        different = Tree([other, inner, TerminalNode(chunks, "c")])
        assert str(pool.repair_tree(different)) == "c"
        for larger, smaller in ((outer, inner), (inner, outer)):
            tree = Tree([larger, smaller, dense_inner, TerminalNode(chunks, "c")])
            assert (
                str(pool.repair_tree(tree))
                == inner._name + "(" + dense_inner._name + "(c))"
            )
        different = Tree([other, inner, dense_inner, TerminalNode(chunks, "c")])
        assert str(pool.repair_tree(different)) == str(different)
        identity = FunctionalNode(
            create_operator("custom({})", "custom", "custom", lambda value: value)
        )
        custom = Tree([identity, identity, TerminalNode(chunks, "c")])
        assert str(pool.repair_tree(custom)) == str(custom)


@pytest.mark.parametrize(
    "mode",
    [
        "merge",
        "concatenate",
        "reciprocal_rank_fusion",
        "distribution_based_rank_fusion",
    ],
)
def test_fusion_repair_collapses_raw_inputs_and_propagates(mode: str) -> None:
    chunks = ChunkSet("query", (Document(id="a", content="cat"),))
    terminal = TerminalNode(chunks, "c")
    fusion = generate_fusion_node(join_modes=(mode,))
    search = generate_dense_node(models=(GP_DENSE_MODELS[0],))
    rerank = generate_rerank_node(models=(GP_RERANK_MODELS[0],))
    with OperatorPool(chunks.documents) as pool:
        for nodes in (
            [fusion, terminal, terminal],
            [fusion, search, terminal, terminal],
            [fusion, terminal, search, terminal],
            [rerank, fusion, search, terminal, terminal],
            [fusion, search, terminal, rerank, terminal],
        ):
            genotype = Tree(nodes)
            original = str(genotype)
            phenotype = pool.prepare_population(np.array([genotype], dtype=object))[0]
            assert str(phenotype) == "c"
            assert phenotype() == chunks
            assert str(genotype) == original
            assert str(pool.repair_tree(phenotype)) == "c"
        nested = Tree([search, fusion, search, terminal, terminal])
        assert str(pool.repair_tree(nested)) == search._name + "(c)"
        valid = Tree([fusion, search, terminal, search, terminal])
        assert str(pool.repair_tree(valid)) == str(valid)
        assert pool.worker.loads == []


def test_generated_bm25_nodes_sample_fixed_reproducible_configurations() -> None:
    configurations: list[BM25Config] = []

    def build(config: BM25Config) -> Callable[[ChunkSet], ChunkSet]:
        configurations.append(config)
        return lambda chunks: chunks

    chunks = ChunkSet(query="query", documents=())
    uniset = RetrievalUniversalSet(
        (EphemeralFunctionalNode(partial(generate_bm25_node, build), 1),),
        (TerminalNode(chunks, "c"),),
    )
    check_random_state(42)
    trees = [Tree.random_tree(uniset, 1) for _ in range(10)]
    assert len(set(map(str, trees))) == 10
    for tree, config in zip(trees, configurations.copy(), strict=True):
        assert str(tree) == config.name + "(c)"
        parameters = asdict(config)
        parameters.pop("name")
        assert json.loads(config.name[len("bm25[") : -1]) == parameters
        assert 0.5 <= config.k1 <= 2.5
        assert 0 <= config.b <= 1
        assert 1 <= config.top_k <= 50
        assert (config.delta is None) == (config.algorithm == "BM25Okapi")
        assert tree() == tree.copy()() == chunks
    assert len(configurations) == 10
    check_random_state(42)
    assert [str(Tree.random_tree(uniset, 1)) for _ in range(10)] == list(
        map(str, trees)
    )


@pytest.mark.parametrize(
    "generate,model",
    [
        (
            generate_dense_node,
            DenseModel("dense", query_prefix="query: {text}", revision="v1"),
        ),
        (generate_rerank_node, RerankModel("rerank", "v2")),
    ],
)
def test_generated_neural_names_include_model_settings(
    generate: Callable[..., FunctionalNode], model: DenseModel | RerankModel
) -> None:
    configurations: list[NeuralConfig] = []

    def build(config: NeuralConfig) -> Callable[[ChunkSet], ChunkSet]:
        configurations.append(config)
        return lambda chunks: chunks

    node = generate(build, models=(model,), top_k_range=(10, 10))
    tree = Tree([node, TerminalNode(ChunkSet(query="", documents=()), "c")])
    config = configurations[0]
    assert config.model == model
    assert config.top_k == 10
    assert str(tree) == config.name + "(c)"
    parameters = json.loads(config.name.split("[", 1)[1][:-1])
    assert parameters == {"model": asdict(model), "top_k": 10}
    assert tree() == tree.copy()()
    assert len(configurations) == 1


@pytest.mark.parametrize("kind", ["bm25", "dense", "rerank", "fusion"])
def test_limits_cover_every_integer_in_inclusive_range(kind: str) -> None:
    check_random_state(42)
    seen = set()
    for _ in range(100):
        if kind == "bm25":
            node = generate_bm25_node(top_k_range=(6, 9))
        elif kind == "dense":
            node = generate_dense_node(models=(GP_DENSE_MODELS[0],), top_k_range=(6, 9))
        elif kind == "rerank":
            node = generate_rerank_node(
                models=(RerankModel("test", "v1"),), top_k_range=(6, 9)
            )
        else:
            node = generate_fusion_node(top_k_range=(6, 9))
        parameters = json.loads(node._name.split("[", 1)[1][:-1])
        seen.add(parameters["top_k"])
    assert seen == {6, 7, 8, 9}


@pytest.mark.parametrize("bounds", [(0, 5), (5, 4), (1.5, 5)])
def test_invalid_limit_ranges_are_rejected(bounds: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="top_k_range"):
        generate_bm25_node(top_k_range=bounds)


def test_diversity_nodes_are_reproducible_lazy_and_not_reduced() -> None:
    def sample() -> list[FunctionalNode]:
        return [generate_diversity_node(models=GP_DIVERSITY_MODELS) for _ in range(10)]

    check_random_state(42)
    nodes = sample()
    check_random_state(42)
    assert [node._name for node in nodes] == [node._name for node in sample()]
    chunks = ChunkSet("query", (Document(id="a", score=0.9),))
    search = generate_dense_node(models=GP_DENSE_MODELS)
    tree = Tree([nodes[0], nodes[0], search, TerminalNode(chunks, "c")])
    with OperatorPool(chunks.documents) as pool:
        assert str(pool.repair_tree(tree)) == str(tree)
        phenotype = pool.prepare_population(np.array([tree], dtype=object))[0]
        with patch.object(pool.worker, "run", return_value=chunks) as run:
            assert phenotype() == chunks
            assert run.call_count == 3
        assert pool.worker.loads == []
    parameters = [json.loads(node._name.split("[", 1)[1][:-1]) for node in nodes]
    assert {item["model"]["strategy"] for item in parameters} == {
        "maximum_margin_relevance",
        "greedy_diversity_order",
    }


@pytest.mark.parametrize(
    "strategy", ["maximum_margin_relevance", "greedy_diversity_order"]
)
def test_diversity_on_c_collapses_and_propagates(strategy: str) -> None:
    chunks = ChunkSet("query", (Document(id="a", content="text"),))
    terminal = TerminalNode(chunks, "c")
    diversity = generate_diversity_node(
        models=GP_DIVERSITY_MODELS, strategies=(strategy,)
    )
    search = generate_dense_node(models=GP_DENSE_MODELS)
    rerank = generate_rerank_node(models=GP_RERANK_MODELS)
    fusion = generate_fusion_node(join_modes=("merge",))
    with OperatorPool(chunks.documents) as pool:
        for nodes in (
            [diversity, terminal],
            [diversity, diversity, terminal],
            [rerank, diversity, terminal],
            [fusion, search, terminal, diversity, terminal],
            [fusion, diversity, terminal, search, terminal],
        ):
            genotype = Tree(nodes)
            original = str(genotype)
            phenotype = pool.prepare_population(np.array([genotype], dtype=object))[0]
            assert str(phenotype) == "c"
            assert phenotype() == chunks
            assert str(pool.repair_tree(phenotype)) == "c"
            assert str(genotype) == original
        assert pool.worker.loads == []


@pytest.mark.parametrize(
    "mode",
    [
        "merge",
        "concatenate",
        "reciprocal_rank_fusion",
        "distribution_based_rank_fusion",
    ],
)
def test_generated_fusion_matches_native_with_sampled_weights(mode: str) -> None:
    from haystack.components.joiners import DocumentJoiner

    check_random_state(42)
    node = generate_fusion_node(
        join_modes=(mode,), left_weight_range=(0.2, 0.2), top_k_range=(2, 2)
    )
    parameters = json.loads(node._name.split("[", 1)[1][:-1])
    weights = [0.2, 0.8] if mode in ("merge", "reciprocal_rank_fusion") else None
    assert parameters["weights"] == weights
    left = ChunkSet("query", (Document(id="a", score=0.9), Document(id="b", score=0.2)))
    right = ChunkSet(
        "query", (Document(id="b", score=0.8), Document(id="c", score=0.1))
    )
    native = DocumentJoiner(join_mode=mode, top_k=2, weights=weights).run(
        documents=[list(left.documents), list(right.documents)]
    )["documents"]
    tree = Tree([node, TerminalNode(left, "left"), TerminalNode(right, "right")])
    assert list(tree().documents) == native
