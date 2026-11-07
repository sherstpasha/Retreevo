from dataclasses import dataclass


@dataclass(frozen=True)
class DenseModel:
    """Keep query and document formatting with the embedding model identity.

    Presets describe candidate configurations, not verified model quality or
    runtime compatibility. Loading and warming up models are separate steps.
    """

    model: str
    query_prefix: str = ""
    document_prefix: str = ""
    query_prompt_name: str | None = None
    document_prompt_name: str | None = None
    revision: str | None = None


@dataclass(frozen=True)
class RerankModel:
    model: str
    revision: str


@dataclass(frozen=True)
class DiversityModel:
    model: str
    revision: str
    strategy: str = "maximum_margin_relevance"
    lambda_threshold: float = 0.5


DENSE_MODELS: tuple[DenseModel, ...] = (
    DenseModel("sentence-transformers/all-MiniLM-L6-v2"),
    DenseModel(
        "BAAI/bge-small-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    DenseModel("Alibaba-NLP/gte-modernbert-base"),
    DenseModel("BAAI/bge-m3"),
    DenseModel(
        "intfloat/multilingual-e5-base",
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
    DenseModel("Qwen/Qwen3-Embedding-0.6B", query_prompt_name="query"),
)


RERANK_MODELS: tuple[str, ...] = (
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "cross-encoder/ms-marco-TinyBERT-L2-v2",
    "Alibaba-NLP/gte-reranker-modernbert-base",
    "BAAI/bge-reranker-v2-m3",
    "mixedbread-ai/mxbai-rerank-base-v2",
    "Qwen/Qwen3-Reranker-0.6B",
)
