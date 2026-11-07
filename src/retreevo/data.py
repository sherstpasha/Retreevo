import re
from dataclasses import dataclass
from pathlib import Path
from random import Random

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from haystack import Document


@dataclass
class RetrievalDataset:
    """A corpus and the judged queries for one evaluation split, held in memory."""

    corpus: list[Document]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def sample_dataset(
    dataset: RetrievalDataset,
    query_count: int = 10,
    document_count: int = 500,
    seed: int = 42,
) -> RetrievalDataset:
    """Select a reproducible debugging subset, preserving positive judgments.

    IDs are sorted before sampling, so source ordering does not affect selection.
    All positively judged documents for selected queries are retained; remaining
    slots are sampled from the corpus. Scores on this reduced corpus are not
    comparable to the original benchmark. Documents are shared with the source.
    """
    if not 1 <= query_count <= len(dataset.queries):
        raise ValueError("query_count must be positive and not exceed dataset size.")
    if not 1 <= document_count <= len(dataset.corpus):
        raise ValueError("document_count must be positive and not exceed corpus size.")
    random = Random(seed)
    query_ids = sorted(random.sample(sorted(dataset.queries), query_count))
    corpus = {document.id: document for document in dataset.corpus}
    required = {
        document_id
        for query_id in query_ids
        for document_id, grade in dataset.qrels[query_id].items()
        if grade > 0
    }
    if not required <= corpus.keys():
        raise ValueError("Positive judgments reference unknown documents.")
    if len(required) > document_count:
        raise ValueError("document_count is smaller than the relevant document count.")
    extra = random.sample(
        sorted(corpus.keys() - required), document_count - len(required)
    )
    document_ids = required | set(extra)
    return RetrievalDataset(
        corpus=[corpus[document_id] for document_id in sorted(document_ids)],
        queries={query_id: dataset.queries[query_id] for query_id in query_ids},
        qrels={
            query_id: {
                document_id: grade
                for document_id, grade in sorted(dataset.qrels[query_id].items())
                if document_id in document_ids
            }
            for query_id in query_ids
        },
    )


def load_local(path: str | Path, split: str = "test") -> RetrievalDataset:
    """Load corpus.jsonl, queries.jsonl and qrels/<split>.tsv in BEIR format.

    Only queries with judgments in the selected split are returned. Relevance
    grades, including zero, are preserved without conversion to binary labels.
    Corpus documents use Haystack's Document type, preserving BEIR IDs. Titles
    are prepended to content for retrieval and also stored in metadata.
    """
    _validate_name(split)
    loader = GenericDataLoader(data_folder=str(path))
    corpus, queries, qrels = loader.load(split=split)
    for query_id, judgments in qrels.items():
        for document_id in judgments:
            if document_id not in corpus:
                raise ValueError(
                    f"Judgment for query {query_id!r} references "
                    f"unknown document {document_id!r}."
                )
    documents = [
        Document(
            id=document_id,
            content="\n".join(
                part for part in (document.get("title"), document["text"]) if part
            ),
            meta={"title": document.get("title") or ""},
        )
        for document_id, document in corpus.items()
    ]
    return RetrievalDataset(corpus=documents, queries=queries, qrels=qrels)


def load_beir(
    name: str,
    cache_dir: str | Path = ".cache/retreevo/beir",
    split: str = "test",
) -> RetrievalDataset:
    """Download a public BEIR archive, reuse its local copy and load one split.

    Dataset names refer to archives on the official BEIR server, for example
    ``scifact`` or ``nfcorpus``. Available splits depend on the dataset.
    """
    _validate_name(name)
    _validate_name(split)
    url = (
        f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    )
    dataset_path = util.download_and_unzip(url, str(cache_dir))
    return load_local(dataset_path, split=split)


def _validate_name(value: str) -> None:
    if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", value) is None:
        raise ValueError(f"Expected a dataset or split name, got {value!r}.")
