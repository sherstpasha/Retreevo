from dataclasses import dataclass
from haystack import Document


@dataclass(frozen=True)
class ChunkSet:
    """Candidates or ranked Haystack documents for one query.

    Documents are shared by reference and must not be modified during execution.
    An unranked input has score=None; its order does not represent relevance.
    """

    query: str
    documents: tuple[Document, ...]

    def __post_init__(self) -> None:
        ids = [document.id for document in self.documents]
        if len(set(ids)) != len(ids):
            raise ValueError("Chunk IDs must be unique.")
