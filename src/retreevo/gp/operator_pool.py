from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
import json
from functools import partial
from types import TracebackType
from typing import Self, TypeVar

import numpy as np
from haystack import Document
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from numpy.typing import NDArray
from thefittest.base import FunctionalNode, TerminalNode, Tree
from thefittest.utils import create_operator
from thefittest.utils.random import randint, uniform

from ..chunks import ChunkSet
from ..retrieval.model_pool import (
    DENSE_MODELS,
    RERANK_MODELS,
    DenseModel,
    RerankModel,
    DiversityModel,
)
from ..retrieval.model_worker import ModelWorker
from ..retrieval import BM25Operator, FusionOperator
from .uniset import (
    EphemeralFunctionalNode,
    RetrievalUniversalSet,
    repair_retrieval_tree,
)


@dataclass(frozen=True)
class BM25Config:
    name: str
    algorithm: str
    k1: float
    b: float
    delta: float | None
    top_k: int


@dataclass(frozen=True)
class NeuralConfig:
    name: str
    model: DenseModel | RerankModel | DiversityModel
    top_k: int


UnaryOperator = Callable[[ChunkSet], ChunkSet]
BinaryOperator = Callable[[ChunkSet, ChunkSet], ChunkSet]


@dataclass(frozen=True)
class _ConfiguredOperator:
    config: BM25Config | NeuralConfig
    operation: UnaryOperator | None = None

    def __call__(self, chunks: ChunkSet) -> ChunkSet:
        if self.operation is None:
            raise RuntimeError(
                "Convert the genotype through prepare_population before execution."
            )
        return self.operation(chunks)


_Choice = TypeVar("_Choice")


def _choose(values: tuple[_Choice, ...]) -> _Choice:
    if not values:
        raise ValueError("At least one sampling choice is required.")
    return values[int(randint(0, len(values), 1)[0])]


def _sample_top_k(bounds: tuple[int, int]) -> int:
    low, high = bounds
    if not isinstance(low, int) or not isinstance(high, int) or not 1 <= low <= high:
        raise ValueError("top_k_range must contain ordered positive integer bounds.")
    return int(randint(low, high + 1, 1)[0])


def _configuration_name(kind: str, parameters: dict[str, object]) -> str:
    return (
        kind + "[" + json.dumps(parameters, sort_keys=True, separators=(",", ":")) + "]"
    )


def _configured_node(
    name: str, operation: Callable[[ChunkSet], ChunkSet]
) -> FunctionalNode:
    formula = name.replace("{", "{{").replace("}", "}}") + "({})"
    return FunctionalNode(create_operator(formula, name, name, operation))


def generate_bm25_node(
    build_operator: Callable[[BM25Config], UnaryOperator] = _ConfiguredOperator,
    top_k_range: tuple[int, int] = (1, 50),
    algorithms: tuple[str, ...] = ("BM25Okapi", "BM25L", "BM25Plus"),
    k1_range: tuple[float, float] = (0.5, 2.5),
    b_range: tuple[float, float] = (0.0, 1.0),
    delta_range: tuple[float, float] = (0.0, 1.5),
) -> FunctionalNode:
    """Sample fixed BM25 settings and delegate lightweight wrapper creation.

    The builder must retain the configuration without preparing an index.
    Sampling uses thefittest's seeded random functions. Numeric parameters are
    sampled uniformly, including every integer in the inclusive top_k_range.
    """
    if not algorithms or any(
        algorithm not in ("BM25Okapi", "BM25L", "BM25Plus") for algorithm in algorithms
    ):
        raise ValueError("Choose supported BM25 algorithms.")
    if not 0 < k1_range[0] <= k1_range[1]:
        raise ValueError("k1_range must be positive and ordered.")
    if not 0 <= b_range[0] <= b_range[1] <= 1:
        raise ValueError("b_range must lie within [0, 1] and be ordered.")
    if not 0 <= delta_range[0] <= delta_range[1]:
        raise ValueError("delta_range must be nonnegative and ordered.")
    algorithm = _choose(algorithms)
    parameters = {
        "algorithm": algorithm,
        "k1": float(uniform(*k1_range, 1)[0]),
        "b": float(uniform(*b_range, 1)[0]),
        "delta": (
            None if algorithm == "BM25Okapi" else float(uniform(*delta_range, 1)[0])
        ),
        "top_k": _sample_top_k(top_k_range),
    }
    name = _configuration_name("bm25", parameters)
    config = BM25Config(name=name, **parameters)
    return _configured_node(name, build_operator(config))


def generate_dense_node(
    build_operator: Callable[[NeuralConfig], UnaryOperator] = _ConfiguredOperator,
    *,
    models: tuple[DenseModel, ...],
    top_k_range: tuple[int, int] = (1, 50),
) -> FunctionalNode:
    """Sample a dense model and limit without loading model weights."""
    model = _choose(models)
    limit = _sample_top_k(top_k_range)
    name = _configuration_name("dense", {"model": asdict(model), "top_k": limit})
    config = NeuralConfig(name, model, limit)
    return _configured_node(name, build_operator(config))


def generate_rerank_node(
    build_operator: Callable[[NeuralConfig], UnaryOperator] = _ConfiguredOperator,
    *,
    models: tuple[RerankModel, ...],
    top_k_range: tuple[int, int] = (1, 50),
) -> FunctionalNode:
    """Sample a reranker and limit without loading model weights."""
    model = _choose(models)
    limit = _sample_top_k(top_k_range)
    name = _configuration_name("rerank", {"model": asdict(model), "top_k": limit})
    config = NeuralConfig(name, model, limit)
    return _configured_node(name, build_operator(config))


def generate_diversity_node(
    build_operator: Callable[[NeuralConfig], UnaryOperator] = _ConfiguredOperator,
    *,
    models: tuple[DiversityModel, ...],
    top_k_range: tuple[int, int] = (1, 50),
    strategies: tuple[str, ...] = (
        "maximum_margin_relevance",
        "greedy_diversity_order",
    ),
    lambda_range: tuple[float, float] = (0.0, 1.0),
) -> FunctionalNode:
    if not strategies or any(
        strategy not in ("maximum_margin_relevance", "greedy_diversity_order")
        for strategy in strategies
    ):
        raise ValueError("Choose supported diversity strategies.")
    if not 0 <= lambda_range[0] <= lambda_range[1] <= 1:
        raise ValueError("lambda_range must lie within [0, 1] and be ordered.")
    model = _choose(models)
    strategy = _choose(strategies)
    model = replace(
        model,
        strategy=strategy,
        lambda_threshold=(
            float(uniform(*lambda_range, 1)[0])
            if strategy == "maximum_margin_relevance"
            else 0.5
        ),
    )
    limit = _sample_top_k(top_k_range)
    name = _configuration_name("diversity", {"model": asdict(model), "top_k": limit})
    return _configured_node(name, build_operator(NeuralConfig(name, model, limit)))


def generate_fusion_node(
    join_modes: tuple[str, ...] = (
        "reciprocal_rank_fusion",
        "distribution_based_rank_fusion",
        "merge",
        "concatenate",
    ),
    top_k_range: tuple[int, int] = (1, 50),
    left_weight_range: tuple[float, float] = (0.0, 1.0),
) -> FunctionalNode:
    if not 0 <= left_weight_range[0] <= left_weight_range[1] <= 1:
        raise ValueError("left_weight_range must lie within [0, 1] and be ordered.")
    mode = _choose(join_modes)
    limit = _sample_top_k(top_k_range)
    weights = None
    if mode in ("merge", "reciprocal_rank_fusion"):
        left_weight = float(uniform(*left_weight_range, 1)[0])
        weights = [left_weight, 1.0 - left_weight]
    name = _configuration_name(
        "fusion", {"join_mode": mode, "top_k": limit, "weights": weights}
    )
    operation = FusionOperator(
        DocumentJoiner(join_mode=mode, top_k=limit, weights=weights)
    )
    formula = name.replace("{", "{{").replace("}", "}}") + "({}, {})"
    return FunctionalNode(create_operator(formula, name, name, operation))


_dense_by_id = {model.model: model for model in DENSE_MODELS}
_dense_choices = (
    (
        "minilm",
        replace(
            _dense_by_id["sentence-transformers/all-MiniLM-L6-v2"],
            revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        ),
    ),
    (
        "bge_small",
        replace(
            _dense_by_id["BAAI/bge-small-en-v1.5"],
            revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        ),
    ),
    (
        "gte",
        replace(
            _dense_by_id["Alibaba-NLP/gte-modernbert-base"],
            revision="e7f32e3c00f91d699e8c43b53106206bcc72bb22",
        ),
    ),
)
GP_DENSE_MODELS = tuple(model for _, model in _dense_choices)
GP_DIVERSITY_MODELS = (
    DiversityModel(GP_DENSE_MODELS[0].model, GP_DENSE_MODELS[0].revision),
)

_rerank_choices = (
    (
        "minilm",
        RerankModel(
            "cross-encoder/ms-marco-MiniLM-L6-v2",
            "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        ),
    ),
    (
        "tinybert",
        RerankModel(
            "cross-encoder/ms-marco-TinyBERT-L2-v2",
            "81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc",
        ),
    ),
    (
        "gte",
        RerankModel(
            "Alibaba-NLP/gte-reranker-modernbert-base",
            "f7481e6055501a30fb19d090657df9ec1f79ab2c",
        ),
    ),
)
if any(model.model not in RERANK_MODELS for _, model in _rerank_choices):
    raise ValueError("Rerank configurations must refer to the model pool.")
GP_RERANK_MODELS = tuple(model for _, model in _rerank_choices)


def _node_config(node: FunctionalNode) -> BM25Config | NeuralConfig | None:
    operation = node._value._operation
    return operation.config if isinstance(operation, _ConfiguredOperator) else None


def _requires_candidates(node: FunctionalNode) -> bool:
    config = _node_config(node)
    return isinstance(config, NeuralConfig) and isinstance(
        config.model, (RerankModel, DiversityModel)
    )


def _is_fusion(node: FunctionalNode) -> bool:
    return isinstance(node._value._operation, FusionOperator)


def _scorer_key(config: BM25Config | NeuralConfig) -> tuple:
    if isinstance(config, BM25Config):
        return config.algorithm, config.k1, config.b, config.delta
    return (config.model,)


def _reduce_pair(outer: FunctionalNode, inner: FunctionalNode) -> FunctionalNode | None:
    outer_config, inner_config = _node_config(outer), _node_config(inner)
    if outer_config is None or inner_config is None:
        return None
    if any(
        isinstance(config, NeuralConfig) and isinstance(config.model, DiversityModel)
        for config in (outer_config, inner_config)
    ):
        return None
    if _scorer_key(outer_config) != _scorer_key(inner_config):
        return None
    return outer if outer_config.top_k <= inner_config.top_k else inner


class OperatorPool:
    """Generate configuration-only trees and bind repaired phenotypes lazily.

    At most one BM25 index and one neural worker are retained. Generating and
    binding nodes do not load models or build indexes. Execution is serial.
    """

    def __init__(
        self,
        corpus: tuple[Document, ...],
        device: str = "cpu",
    ) -> None:
        ChunkSet(query="", documents=corpus)
        self.corpus = corpus
        self.worker = ModelWorker(corpus, device=device)
        self._store: InMemoryDocumentStore | None = None
        self._store_key: tuple | None = None
        functions = [
            EphemeralFunctionalNode(generate_bm25_node, 1),
            EphemeralFunctionalNode(
                partial(generate_dense_node, models=GP_DENSE_MODELS), 1
            ),
            EphemeralFunctionalNode(
                partial(generate_rerank_node, models=GP_RERANK_MODELS), 1
            ),
        ]
        functions.append(
            EphemeralFunctionalNode(
                partial(generate_diversity_node, models=GP_DIVERSITY_MODELS), 1
            )
        )
        for mode in (
            "reciprocal_rank_fusion",
            "distribution_based_rank_fusion",
            "merge",
            "concatenate",
        ):
            functions.append(
                EphemeralFunctionalNode(
                    partial(
                        generate_fusion_node,
                        join_modes=(mode,),
                    ),
                    2,
                )
            )
        self.uniset = RetrievalUniversalSet(
            tuple(functions), (TerminalNode(ChunkSet(query="", documents=corpus), "c"),)
        )

    def run(self, config: BM25Config | NeuralConfig, chunks: ChunkSet) -> ChunkSet:
        if not chunks.documents or not chunks.query.strip():
            return ChunkSet(query=chunks.query, documents=())
        if isinstance(config, NeuralConfig):
            return self.worker.run(config.model, chunks, config.top_k)
        key = _scorer_key(config)
        if self._store_key != key:
            self._close_store()
            parameters = {"k1": config.k1, "b": config.b}
            parameters.update(
                {"epsilon": 0.25} if config.delta is None else {"delta": config.delta}
            )
            store = InMemoryDocumentStore(
                bm25_algorithm=config.algorithm, bm25_parameters=parameters
            )
            try:
                store.write_documents(list(self.corpus))
            except Exception:
                store.delete_all_documents()
                store.shutdown()
                raise
            self._store, self._store_key = store, key
        return BM25Operator(InMemoryBM25Retriever(self._store, top_k=config.top_k))(
            chunks
        )

    def repair_tree(self, tree: Tree) -> Tree:
        return repair_retrieval_tree(
            tree, _requires_candidates, _is_fusion, _reduce_pair
        )

    def prepare_population(self, population: NDArray) -> NDArray:
        """Repair and bind phenotypes without changing the source genotypes."""
        phenotypes = []
        for tree in population:
            repaired = self.repair_tree(tree)
            nodes = []
            for node in repaired._nodes:
                config = (
                    _node_config(node) if isinstance(node, FunctionalNode) else None
                )
                if config is not None:
                    operation = partial(self.run, config)
                    node = _configured_node(
                        config.name, _ConfiguredOperator(config, operation)
                    )
                nodes.append(node)
            phenotypes.append(Tree(nodes))
        return np.array(phenotypes, dtype=object)

    def _close_store(self) -> None:
        if self._store is not None:
            self._store.delete_all_documents()
            self._store.shutdown()
        self._store = self._store_key = None

    def close(self) -> None:
        self.worker.close()
        self._close_store()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
