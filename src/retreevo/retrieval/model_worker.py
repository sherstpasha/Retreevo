from __future__ import annotations

import multiprocessing as mp
import resource
import sys
import traceback
from dataclasses import replace
from multiprocessing.connection import Connection

from haystack import Document

from ..chunks import ChunkSet
from .model_pool import DenseModel, RerankModel, DiversityModel


def _serve(
    connection: Connection,
    corpus: tuple[Document, ...],
    model: DenseModel | RerankModel | DiversityModel,
    device_name: str = "cpu",
) -> None:
    import torch
    from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.utils import ComponentDevice

    store = None
    try:
        device = ComponentDevice.from_str(device_name)
        by_id = {doc.id: doc for doc in corpus}
        if isinstance(model, DenseModel):
            from haystack_integrations.components.embedders.sentence_transformers import (
                SentenceTransformersDocumentEmbedder,
                SentenceTransformersTextEmbedder,
            )

            embedder = SentenceTransformersTextEmbedder(
                model=model.model,
                revision=model.revision,
                device=device,
                prefix=model.query_prefix,
                normalize_embeddings=True,
                progress_bar=False,
                model_kwargs={"dtype": torch.float32},
                encode_kwargs=(
                    {"prompt_name": model.query_prompt_name}
                    if model.query_prompt_name
                    else {}
                ),
            )
            embedder.warm_up()
            document_embedder = SentenceTransformersDocumentEmbedder(
                model=model.model,
                revision=model.revision,
                device=device,
                prefix=model.document_prefix,
                normalize_embeddings=True,
                progress_bar=False,
                model_kwargs={"dtype": torch.float32},
                encode_kwargs=(
                    {"prompt_name": model.document_prompt_name}
                    if model.document_prompt_name
                    else {}
                ),
            )
            document_embedder.warm_up()
            assert document_embedder.embedding_backend is embedder.embedding_backend
            embedded = document_embedder.run(documents=list(corpus))["documents"]
            store = InMemoryDocumentStore(embedding_similarity_function="cosine")
            store.write_documents(embedded)
            retriever = InMemoryEmbeddingRetriever(store)
        elif isinstance(model, DiversityModel):
            from haystack_integrations.components.rankers.sentence_transformers import (
                SentenceTransformersDiversityRanker,
            )

            ranker = SentenceTransformersDiversityRanker(
                model=model.model,
                device=device,
                strategy=model.strategy,
                lambda_threshold=model.lambda_threshold,
                model_kwargs={"revision": model.revision, "dtype": torch.float32},
                tokenizer_kwargs={"revision": model.revision},
                config_kwargs={"revision": model.revision},
            )
            ranker.warm_up()
        else:
            from haystack_integrations.components.rankers.sentence_transformers import (
                SentenceTransformersSimilarityRanker,
            )

            ranker = SentenceTransformersSimilarityRanker(
                model=model.model,
                device=device,
                batch_size=16,
                scale_score=True,
                model_kwargs={"revision": model.revision, "dtype": torch.float32},
                tokenizer_kwargs={"revision": model.revision},
                config_kwargs={"revision": model.revision},
            )
            ranker.warm_up()
        while (request := connection.recv()) is not None:
            query, candidate_ids, top_k, candidate_scores = request
            if isinstance(model, DenseModel):
                filters = (
                    None
                    if candidate_ids is None
                    else {"field": "id", "operator": "in", "value": list(candidate_ids)}
                )
                documents = retriever.run(
                    query_embedding=embedder.run(text=query)["embedding"],
                    filters=filters,
                    top_k=top_k,
                )["documents"]
            else:
                candidates = (
                    list(corpus)
                    if candidate_ids is None
                    else [by_id[doc_id] for doc_id in candidate_ids]
                )
                if isinstance(model, DiversityModel):
                    candidates = [
                        replace(doc, score=score)
                        for doc, score in zip(candidates, candidate_scores, strict=True)
                    ]
                documents = ranker.run(query=query, documents=candidates, top_k=top_k)[
                    "documents"
                ]
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_mb = peak_rss / (1024**2 if sys.platform == "darwin" else 1024)
            connection.send((True, [(doc.id, doc.score) for doc in documents], peak_mb))
    except EOFError:
        pass
    except Exception:
        connection.send((False, traceback.format_exc(), None))
    finally:
        if store is not None:
            store.shutdown()
        connection.close()


class ModelWorker:
    """Keep at most one neural-model process alive; switching joins the old one.

    Calls are synchronous and intended for serial tree evaluation. Ending the
    process releases Haystack's global model cache as well as tensor allocations.
    The fixed corpus is treated as immutable for the worker's lifetime.
    """

    def __init__(
        self,
        corpus: tuple[Document, ...],
        device: str = "cpu",
    ) -> None:
        self.corpus = corpus
        self.device = device
        self._ids = tuple(doc.id for doc in corpus)
        self._by_id = {doc.id: doc for doc in corpus}
        self._model: DenseModel | RerankModel | DiversityModel | None = None
        self._process: mp.Process | None = None
        self._connection: Connection | None = None
        self.loads: list[dict] = []
        self.closed_pids: list[int] = []

    @property
    def active_pid(self) -> int | None:
        return self._process.pid if self._process and self._process.is_alive() else None

    def run(
        self,
        model: DenseModel | RerankModel | DiversityModel,
        chunks: ChunkSet,
        top_k: int,
    ) -> ChunkSet:
        if not chunks.documents or not chunks.query.strip():
            return ChunkSet(query=chunks.query, documents=())
        if top_k <= 0 or not model.revision:
            raise ValueError(
                "A positive top_k and a pinned model revision are required."
            )
        ids = tuple(doc.id for doc in chunks.documents)
        unknown = set(ids) - self._by_id.keys()
        if unknown:
            raise ValueError(f"Unknown candidate IDs: {sorted(unknown)}")
        if isinstance(model, DiversityModel):
            result = self._request(
                model,
                chunks.query,
                ids,
                top_k,
                candidate_scores=tuple(doc.score for doc in chunks.documents),
            )
        else:
            result = self._request(model, chunks.query, ids, top_k)
        documents = tuple(
            replace(self._by_id[doc_id], score=score, embedding=None)
            for doc_id, score in result
        )
        return ChunkSet(query=chunks.query, documents=documents)

    def _request(
        self,
        model: DenseModel | RerankModel | DiversityModel,
        query: str,
        ids: tuple[str, ...],
        top_k: int,
        candidate_scores: tuple[float | None, ...] | None = None,
    ) -> list[tuple[str, float | None]]:
        if model != self._model or self.active_pid is None:
            self._close_process()
            context = mp.get_context("spawn")
            parent, child = context.Pipe()
            process = context.Process(
                target=_serve,
                args=(child, self.corpus, model, self.device),
            )
            process.start()
            child.close()
            self._model, self._process, self._connection = model, process, parent
            self.loads.append(
                {"model": model.model, "revision": model.revision, "pid": process.pid}
            )
        try:
            self._connection.send(
                (query, None if ids == self._ids else ids, top_k, candidate_scores)
            )
            success, result, peak_mb = self._connection.recv()
            if not success:
                raise RuntimeError(result)
        except (EOFError, BrokenPipeError, RuntimeError):
            self._close_process()
            raise
        self.loads[-1]["peak_rss_mb"] = peak_mb
        return result

    def close(self) -> None:
        self._close_process()

    def _close_process(self) -> None:
        if self._process is not None:
            pid = self._process.pid
            try:
                self._connection.send(None)
            except (BrokenPipeError, OSError):
                pass
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join()
            self.closed_pids.append(pid)
            self._process.close()
            self._connection.close()
        self._model = self._process = self._connection = None
