import json
from pathlib import Path

import pytest
from haystack import Document

from retreevo.data import RetrievalDataset, load_beir, load_local, sample_dataset


@pytest.fixture
def dataset_path(tmp_path: Path) -> Path:
    corpus = [
        {"_id": "d1", "title": "Title", "text": "First document"},
        {"_id": "d2", "title": "", "text": "Second document"},
    ]
    queries = [
        {"_id": "q1", "text": "Test query"},
        {"_id": "q2", "text": "Train query"},
    ]
    for filename, rows in [("corpus.jsonl", corpus), ("queries.jsonl", queries)]:
        (tmp_path / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels/test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t0\n",
        encoding="utf-8",
    )
    (tmp_path / "qrels/train.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq2\td2\t1\n", encoding="utf-8"
    )
    return tmp_path


def test_load_local_preserves_corpus_and_grades(dataset_path: Path) -> None:
    dataset = load_local(dataset_path)

    assert len(dataset.corpus) == 2
    assert dataset.corpus[0] == Document(
        id="d1", content="Title\nFirst document", meta={"title": "Title"}
    )
    assert dataset.corpus[1].content == "Second document"
    assert dataset.corpus[1].id == "d2"
    assert dataset.queries == {"q1": "Test query"}
    assert dataset.qrels == {"q1": {"d1": 2, "d2": 0}}


def test_load_local_selects_split(dataset_path: Path) -> None:
    dataset = load_local(dataset_path, split="train")

    assert dataset.queries == {"q2": "Train query"}
    assert dataset.qrels == {"q2": {"d2": 1}}


def test_load_local_rejects_unknown_document(dataset_path: Path) -> None:
    (dataset_path / "qrels/test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\tmissing\t1\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unknown document 'missing'"):
        load_local(dataset_path)


def test_load_beir_uses_shared_loader(
    dataset_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def download(url: str, out_dir: str) -> str:
        assert url.endswith("/scifact.zip")
        assert out_dir == str(tmp_path)
        return str(dataset_path)

    monkeypatch.setattr("retreevo.data.util.download_and_unzip", download)
    dataset = load_beir("scifact", cache_dir=tmp_path, split="train")

    assert dataset.queries == {"q2": "Train query"}


def test_load_beir_rejects_path_as_name() -> None:
    with pytest.raises(ValueError, match="Expected a dataset or split name"):
        load_beir("../scifact")


def test_sample_is_reproducible_and_preserves_relevance() -> None:
    corpus = [Document(id=str(i), content=str(i)) for i in range(20)]
    dataset = RetrievalDataset(
        corpus,
        {"q1": "first", "q2": "second", "q3": "third"},
        {"q1": {"1": 2, "2": 1, "3": 0}, "q2": {"4": 1}, "q3": {"5": 1}},
    )
    subset = sample_dataset(dataset, query_count=2, document_count=6)
    reordered = RetrievalDataset(
        list(reversed(corpus)),
        dict(reversed(list(dataset.queries.items()))),
        dataset.qrels,
    )
    assert sample_dataset(reordered, 2, 6) == subset
    assert len(subset.corpus) == 6
    assert len(subset.queries) == 2
    ids = {document.id for document in subset.corpus}
    for query_id, judgments in subset.qrels.items():
        assert set(judgments) <= ids
        assert {doc: grade for doc, grade in judgments.items() if grade > 0} == {
            doc: grade for doc, grade in dataset.qrels[query_id].items() if grade > 0
        }
    assert len(dataset.corpus) == 20
    assert len(dataset.queries) == 3


def test_sample_rejects_loss_of_relevant_documents(dataset_path: Path) -> None:
    dataset = load_local(dataset_path)
    dataset.qrels["q1"]["d2"] = 1
    with pytest.raises(ValueError, match="relevant document count"):
        sample_dataset(dataset, query_count=1, document_count=1)
