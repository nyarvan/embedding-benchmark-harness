"""
Encoding, persistence and cosine search for document, query and chunk
embeddings.
"""
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np
from sentence_transformers import util

from domain.chunk import Chunk

# A loaded .npz archive: maps array names to arrays
EmbeddingFile = Mapping[str, np.ndarray]


class Encoder(Protocol):
    """Anything that turns texts into embedding vectors.

    Satisfied by sentence-transformers models and by `OpenAIEncoder`.
    """

    def encode(self, inputs: list[str], /, **kwargs) -> np.ndarray:
        """Return an array of shape (len(inputs), dim), in input order."""
        ...


class InputMode(Enum):
    """Which parts of a document are fed to the embedding model."""

    TITLE_TEXT = 'title+text'
    TITLE = 'title'
    TEXT = 'text'


class Embedding:
    """Encode texts with an `Encoder` model, save and search the embeddings.

    Saved .npz layouts:
        documents and queries: `document_ids`, `embeddings`. Query files store
            query ids under the same `document_ids` key (kept for
            compatibility with already saved files).
        chunks: `chunk_ids`, `document_ids`, `embeddings`.

    `model` may be None when the instance is only used for search.
    """

    def __init__(
            self, model: Encoder | None,
            mode: InputMode = InputMode.TITLE_TEXT,
    ):
        self.model = model
        self.mode = mode
        self.embeddings: np.ndarray | None = None

    def encode(self, inputs: list[str]) -> np.ndarray:
        """Encode texts and keep the result in `self.embeddings`."""
        self.embeddings = self.model.encode(
            inputs, batch_size=32, show_progress_bar=True,
            convert_to_numpy=True
        )

        return self.embeddings

    def save_document_embeddings(
            self, path: str | Path, document_ids: np.ndarray
    ) -> None:
        """Save embeddings with their ids (document or query ids)."""
        embeddings = self._require_embeddings()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, document_ids=document_ids, embeddings=embeddings)

    def save_chunk_embeddings(
            self, path: str | Path, *, chunk_ids: np.ndarray,
            document_ids: np.ndarray
    ) -> None:
        """Save chunk embeddings with chunk ids and their parent document ids.

        The ids are keyword-only: both are string arrays, so swapping them
        would go unnoticed.
        """
        embeddings = self._require_embeddings()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path, chunk_ids=chunk_ids, document_ids=document_ids,
            embeddings=embeddings
        )

    def search(
            self, queries: EmbeddingFile, documents: EmbeddingFile,
            top_k: int = 10
    ) -> dict[str, dict[str, float]]:
        """Cosine search over document embeddings.

        Returns {query_id: {document_id: score}}, best match first.
        """
        # top_k must be a keyword: the third positional parameter of
        # semantic_search is query_chunk_size, not top_k
        outputs = util.semantic_search(
            queries['embeddings'], documents['embeddings'], top_k=top_k
        )
        result = {}
        for i, output in enumerate(outputs):
            q_id = queries['document_ids'][i]
            result[q_id] = {}
            for value in output:
                corpus_id = value['corpus_id']
                score = value['score']
                doc_id = documents['document_ids'][corpus_id]
                result[q_id][doc_id] = score

        return result

    def search_chunks(
            self, queries: EmbeddingFile, chunks: EmbeddingFile,
            top_k: int = 50
    ) -> dict[str, dict[str, float]]:
        """Cosine search over chunk embeddings.

        Returns {query_id: {chunk_id: score}}, best match first. Use
        `aggregate_to_documents` to get a document ranking.
        """
        outputs = util.semantic_search(
            queries['embeddings'], chunks['embeddings'], top_k=top_k
        )
        result = {}
        for i, output in enumerate(outputs):
            q_id = queries['document_ids'][i]
            result[q_id] = {}
            for value in output:
                corpus_id = value['corpus_id']
                chunk_id = chunks['chunk_ids'][corpus_id]
                score = value['score']
                result[q_id][chunk_id] = score

        return result

    def _require_embeddings(self) -> np.ndarray:
        if self.embeddings is None:
            raise RuntimeError("Call encode() before saving.")

        return self.embeddings

    @staticmethod
    def build_document_inputs(
        documents: list[tuple[str, dict[str, str]]],
        mode: InputMode = InputMode.TITLE_TEXT,
    ) -> list[str]:
        """Build one model input per (doc_id, doc) pair, in the same order."""
        if mode is InputMode.TITLE_TEXT:
            return [
                f"Title: {doc['title']}\nText: {doc['text']}"
                for _, doc in documents
            ]

        if mode is InputMode.TITLE:
            return [doc['title'] for _, doc in documents]

        if mode is InputMode.TEXT:
            return [doc['text'] for _, doc in documents]

        raise ValueError(f"Unsupported embedding mode: {mode}")

    @staticmethod
    def build_chunk_inputs(
        chunks: list[Chunk], documents: list[tuple[str, dict[str, str]]]
    ) -> list[str]:
        """Build one model input per chunk, in the same order.

        Every chunk gets its document title in the same 'Title/Text' format as
        whole documents, so chunking is the only difference from the baseline.
        """
        titles = {doc_id: doc['title'] for doc_id, doc in documents}
        result = []
        for chunk in chunks:
            title = titles[chunk.document_id]
            result.append(f'Title: {title}\nText: {chunk.text}')

        return result
