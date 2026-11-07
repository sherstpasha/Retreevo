from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
from thefittest.base import FunctionalNode, Tree
from thefittest.optimizers import GeneticProgramming, SelfCGP

from ..chunks import ChunkSet
from ..data import RetrievalDataset, load_local
from ..metrics import MRR, NDCG, RecallAtK
from .operator_pool import OperatorPool


def evaluate_tree(tree: Tree, inputs: Mapping[str, ChunkSet]) -> dict[str, ChunkSet]:
    """Evaluate all queries per node to avoid switching models per query."""
    stack: list[dict[str, ChunkSet]] = []
    for node in reversed(tree._nodes):
        if isinstance(node, FunctionalNode):
            arguments = [stack.pop() for _ in range(node._n_args)]
            results = {
                qid: node._value(*(argument[qid] for argument in arguments))
                for qid in inputs
            }
        elif node._name == "c":
            results = dict(inputs)
        else:
            raise ValueError(f"Unsupported terminal: {node._name}")
        stack.append(results)
    return stack[0]


class GeneticProgrammingRetriever:
    """Fit a retrieval tree on judged queries and retain it for search.

    A path refers to a local BEIR-format directory. Fitting maximizes the supplied
    retrieval metric. Fitness caching is scoped to one fit and uses Tree.signature().
    An optional final top_k limit applies before metric evaluation and to search
    results; internal operator limits are independent of it. None adds no limit.
    Execution is serial because the operator pool shares model processes.
    Use as a context manager or call close() to release indexes and models.
    """

    def __init__(
        self,
        dataset: RetrievalDataset | str | Path,
        *,
        split: str = "test",
        n_iter: int = 10,
        pop_size: int = 20,
        top_k: int | None = None,
        metric: RecallAtK | MRR | NDCG | None = None,
        optimizer: type[SelfCGP] | type[GeneticProgramming] = SelfCGP,
        optimizer_args: dict[str, Any] | None = None,
        random_state: int | None = None,
        use_fitness_cache: bool = True,
        device: str = "cpu",
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.n_iter = n_iter
        self.pop_size = pop_size
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer or None.")
        self.top_k = top_k
        self.metric = metric if metric is not None else NDCG()
        self.optimizer = optimizer
        self.optimizer_args = (
            optimizer_args.copy() if optimizer_args is not None else {}
        )
        self.random_state = random_state
        self.use_fitness_cache = use_fitness_cache
        self.device = device
        self._pool: OperatorPool | None = None
        self._inputs: dict[str, ChunkSet] = {}
        self._fitness_cache: dict[str, float] = {}

    def fit(self) -> Self:
        args = self.optimizer_args.copy()
        reserved = {
            "fitness_function",
            "fitness_function_args",
            "uniset",
            "iters",
            "pop_size",
            "genotype_to_phenotype",
            "genotype_to_phenotype_args",
            "minimization",
            "random_state",
        }
        conflicts = reserved & args.keys()
        if conflicts:
            raise ValueError(
                f"Optimizer arguments are managed by the retriever: {sorted(conflicts)}"
            )
        if args.get("n_jobs", 1) != 1:
            raise ValueError("The shared operator pool requires n_jobs=1.")
        if self.n_iter < 1 or self.pop_size < 2:
            raise ValueError("n_iter must be positive and pop_size must be at least 2.")
        dataset = (
            self.dataset
            if isinstance(self.dataset, RetrievalDataset)
            else load_local(self.dataset, self.split)
        )
        if not dataset.corpus or not dataset.queries:
            raise ValueError("The dataset must contain documents and judged queries.")
        if dataset.queries.keys() != dataset.qrels.keys():
            raise ValueError("Queries and relevance judgments must have matching IDs.")
        self.close()
        for name in ("tree_", "genotype_", "fitness_", "optimizer_stats_"):
            self.__dict__.pop(name, None)
        self.dataset_ = dataset
        self._corpus = tuple(dataset.corpus)
        self._inputs = {
            qid: ChunkSet(query=query, documents=self._corpus)
            for qid, query in dataset.queries.items()
        }
        self._fitness_cache = {}
        self._pool = OperatorPool(self._corpus, device=self.device)
        args.setdefault("init_level", 3)
        args.setdefault("max_level", 4)
        args.setdefault("keep_history", True)
        args.setdefault("n_jobs", 1)
        try:
            optimizer = self.optimizer(
                fitness_function=self._evaluate_population,
                uniset=self._pool.uniset,
                iters=self.n_iter,
                pop_size=self.pop_size,
                genotype_to_phenotype=self._pool.prepare_population,
                minimization=False,
                random_state=self.random_state,
                **args,
            )
            optimizer.fit()
            winner = optimizer.get_fittest()
            self.tree_: Tree = winner["phenotype"]
            self.genotype_: Tree = winner["genotype"]
            self.fitness_ = float(winner["fitness"])
            self.optimizer_stats_ = optimizer.get_stats()
        except BaseException:
            self.close()
            raise
        return self

    def _evaluate_population(self, population: NDArray) -> NDArray[np.float64]:
        scores = []
        for tree in population:
            signature = tree.signature()
            if self.use_fitness_cache and signature in self._fitness_cache:
                score = self._fitness_cache[signature]
            else:
                results = evaluate_tree(tree, self._inputs)
                results = {
                    qid: self._limit_results(chunks) for qid, chunks in results.items()
                }
                score = self.metric.evaluate(self.dataset_.qrels, results).score
                if self.use_fitness_cache:
                    self._fitness_cache[signature] = score
            scores.append(score)
        return np.asarray(scores, dtype=np.float64)

    def _limit_results(self, chunks: ChunkSet) -> ChunkSet:
        return ChunkSet(query=chunks.query, documents=chunks.documents[: self.top_k])

    def get_tree(self) -> Tree:
        if not hasattr(self, "tree_"):
            raise RuntimeError("Call fit() before requesting the fitted tree.")
        return self.tree_.copy()

    def get_stats(self) -> dict[str, list[object]]:
        if not hasattr(self, "optimizer_stats_"):
            raise RuntimeError("Call fit() before requesting optimizer statistics.")
        return self.optimizer_stats_

    def search(self, query: str) -> ChunkSet:
        if self._pool is None:
            raise RuntimeError(
                "Call fit() before search; a closed retriever must be fitted again."
            )
        chunks = ChunkSet(query=query, documents=self._corpus)
        results = self.get_tree().set_terminals(c=chunks)()
        return self._limit_results(results)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
