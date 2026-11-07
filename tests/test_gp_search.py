from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from haystack import Document
from thefittest.base import TerminalNode, Tree
from thefittest.optimizers import GeneticProgramming, SelfCGP

from retreevo.chunks import ChunkSet
from retreevo.data import RetrievalDataset
from retreevo.gp import GeneticProgrammingRetriever
from retreevo.gp.operator_pool import (
    generate_bm25_node,
    generate_rerank_node,
    GP_RERANK_MODELS,
)
from retreevo.metrics import MRR, NDCG, RecallAtK


def initial_population() -> np.ndarray:
    node = generate_bm25_node(
        algorithms=("BM25Okapi",),
        top_k_range=(1, 1),
        k1_range=(1.5, 1.5),
        b_range=(0.75, 0.75),
    )
    tree = Tree([node, TerminalNode(ChunkSet(query="", documents=()), "c")])
    return np.array([tree] * 4, dtype=object)


@pytest.mark.parametrize(
    "optimizer,cached,calls", [(SelfCGP, True, 1), (GeneticProgramming, False, 4)]
)
def test_fit_search_and_native_signature_cache(
    optimizer: type[SelfCGP] | type[GeneticProgramming], cached: bool, calls: int
) -> None:
    dataset = RetrievalDataset(
        [
            Document(id="a", content="cat kitten"),
            Document(id="b", content="dog puppy"),
            Document(id="c", content="bird parrot"),
        ],
        {"q": "cat"},
        {"q": {"a": 1}},
    )
    metric = RecallAtK(1)
    population = initial_population()
    optimizer_args = {"init_population": population}
    with (
        GeneticProgrammingRetriever(
            dataset,
            n_iter=1,
            pop_size=4,
            metric=metric,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            use_fitness_cache=cached,
            random_state=42,
        ) as retriever,
        patch.object(metric, "evaluate", wraps=metric.evaluate) as evaluate,
    ):
        assert retriever.fit() is retriever
        assert retriever._pool.worker.device == retriever.device == "cpu"
        assert evaluate.call_count == calls
        assert retriever.fitness_ == 1.0
        assert len(retriever.get_stats()["fitness"]) == 1
        assert retriever.search("cat").documents[0].id == "a"
        assert retriever.search("dog").documents[0].id == "b"
        assert str(retriever.genotype_) == str(population[0])
        with pytest.raises(RuntimeError, match="prepare_population"):
            retriever.genotype_()
        assert list(optimizer_args) == ["init_population"]
        store = retriever._pool._store
        if cached:
            assert list(retriever._fitness_cache) == [retriever.get_tree().signature()]
        retriever.fit()
        assert evaluate.call_count == 2 * calls
    assert store.count_documents() == 0
    with pytest.raises(RuntimeError, match="closed retriever"):
        retriever.search("cat")


def test_search_loads_local_dataset(tmp_path: Path) -> None:
    (tmp_path / "corpus.jsonl").write_text('{"_id":"a","text":"cat"}\n')
    (tmp_path / "queries.jsonl").write_text('{"_id":"q","text":"cat"}\n')
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\nq\ta\t1\n")
    with GeneticProgrammingRetriever(
        tmp_path,
        n_iter=1,
        pop_size=4,
        optimizer_args={"init_population": initial_population()},
    ) as retriever:
        retriever.fit()
        assert retriever.fitness_ == 1.0
        assert retriever.search("cat").documents[0].id == "a"


@pytest.mark.parametrize("args", [{"iters": 2}, {"minimization": True}, {"n_jobs": 2}])
def test_search_rejects_conflicting_optimizer_arguments(
    args: dict[str, object],
) -> None:
    retriever = GeneticProgrammingRetriever("unused", optimizer_args=args)
    with pytest.raises(ValueError):
        retriever.fit()
    assert retriever._pool is None


def test_search_cleans_up_on_optimizer_error() -> None:
    dataset = RetrievalDataset(
        [Document(id="a", content="cat")], {"q": "cat"}, {"q": {"a": 1}}
    )
    retriever = GeneticProgrammingRetriever(dataset)
    with patch.object(SelfCGP, "fit", side_effect=RuntimeError("optimizer failed")):
        with pytest.raises(RuntimeError, match="optimizer failed"):
            retriever.fit()
    assert retriever._pool is None
    with pytest.raises(RuntimeError, match="fit"):
        retriever.get_tree()


@pytest.mark.parametrize(
    "top_k,expected_score", [(None, 0.5), (1, 0.0), (2, 0.5), (5, 0.5)]
)
def test_final_limit_applies_to_fitness_and_search(
    top_k: int | None, expected_score: float
) -> None:
    corpus = [Document(id=name, content="cat") for name in ("a", "b", "c")]
    dataset = RetrievalDataset(corpus, {"q": "cat"}, {"q": {"b": 1}})
    node = generate_bm25_node(
        algorithms=("BM25L",),
        top_k_range=(3, 3),
        k1_range=(1.5, 1.5),
        b_range=(0.75, 0.75),
        delta_range=(0.5, 0.5),
    )
    tree = Tree([node, TerminalNode(ChunkSet("", ()), "c")])
    with GeneticProgrammingRetriever(
        dataset,
        n_iter=1,
        pop_size=4,
        top_k=top_k,
        metric=MRR(),
        optimizer_args={"init_population": np.array([tree] * 4, dtype=object)},
    ) as retriever:
        retriever.fit()
        assert retriever.fitness_ == expected_score
        assert [doc.id for doc in retriever.search("cat").documents] == ["a", "b", "c"][
            :top_k
        ]
        raw = retriever.get_tree().set_terminals(c=ChunkSet("cat", tuple(corpus)))()
        assert len(raw.documents) == 3


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_rejects_invalid_final_limit(top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k"):
        GeneticProgrammingRetriever("unused", top_k=top_k)


def test_default_metric_is_ndcg() -> None:
    assert isinstance(GeneticProgrammingRetriever("unused").metric, NDCG)


def test_collapsed_c_is_evaluated_normally_and_cached() -> None:
    from math import log2

    corpus = [Document(id="a", content="first"), Document(id="b", content="second")]
    dataset = RetrievalDataset(corpus, {"q": "query"}, {"q": {"b": 1}})
    node = generate_rerank_node(models=GP_RERANK_MODELS)
    tree = Tree([node, TerminalNode(ChunkSet("", ()), "c")])
    with GeneticProgrammingRetriever(
        dataset,
        n_iter=1,
        pop_size=4,
        optimizer_args={"init_population": np.array([tree] * 4, dtype=object)},
    ) as retriever:
        retriever.fit()
        assert str(retriever.genotype_) == str(tree)
        assert str(retriever.get_tree()) == "c"
        assert retriever.fitness_ == pytest.approx(1 / log2(3))
        assert len(retriever._fitness_cache) == 1
        assert retriever.search("query").documents == tuple(corpus)
        assert retriever._pool.worker.loads == []
