"""Benchmark pipeline: cached embeddings, retrieval, evaluation, metrics file.

For every model the pipeline embeds whole documents, queries and (optionally)
chunks of the documents, caches the vectors on disk, retrieves with cosine
search and evaluates against the qrels. Chunk rankings are aggregated back
to documents first, so all variants go through the same evaluation.
"""
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from domain.chunk import Chunking
from harness.aggregate import aggregate_to_documents
from harness.cache import chunks_path, documents_path, queries_path
from harness.config import (
    BASELINE, CHUNK_STRATEGIES, EVAL_DEPTH, BenchmarkConfig, variant_name,
)
from harness.embed import Embedding, EmbeddingFile, Encoder
from harness.evaluate import evaluate
from harness.loaders import load_beir_dataset, save_metrics
from harness.models import get_spec

METRIC_NAMES = ('P@5', 'Recall@10', 'NDCG@10')

Documents = list[tuple[str, dict[str, str]]]
Metrics = dict[str, float]


class LazyEncoder:
    """Builds the wrapped model on the first `encode` call.

    A run whose embeddings are all cached never loads a model or calls an API.
    """

    def __init__(self, create: Callable[[], Encoder]):
        self._create = create
        self._model: Encoder | None = None

    def encode(self, inputs: list[str], **kwargs) -> np.ndarray:
        if self._model is None:
            self._model = self._create()
        return self._model.encode(inputs, **kwargs)


def build_document_embeddings(
        model: Encoder, documents: Documents, path: str | Path
) -> None:
    """Encode whole documents and save them, unless `path` already exists.

    Delete the file to force a recompute: an existing file is never checked
    against the current documents or model.
    """
    if Path(path).exists():
        return
    embedding = Embedding(model)
    embedding.encode(embedding.build_document_inputs(documents))
    doc_ids = np.array([doc_id for doc_id, _ in documents], dtype=str)
    embedding.save_document_embeddings(path, doc_ids)
    print_embeddings_report(path)


def build_query_embeddings(
        model: Encoder, queries: dict[str, str], path: str | Path
) -> None:
    """Encode queries and save them, unless `path` already exists."""
    if Path(path).exists():
        return
    embedding = Embedding(model)
    queries = list(queries.items())
    q_inputs = [query for _, query in queries]
    q_ids = np.array([id_ for id_, _ in queries], dtype=str)

    embedding.encode(q_inputs)
    embedding.save_document_embeddings(path, q_ids)
    print_embeddings_report(path)


def build_chunk_embeddings(
        model: Encoder, documents: Documents, path: str | Path,
        chunker: Chunking, splitter: Callable[[str], list[str]]
) -> None:
    """Chunk documents with `splitter`, encode the chunks and save them.

    Skipped when `path` already exists.
    """
    if Path(path).exists():
        return

    embedding = Embedding(model)
    chunks = chunker.chunk_documents(documents, splitter)
    inputs = embedding.build_chunk_inputs(chunks, documents)
    chunk_ids = np.array([chunk.chunk_id for chunk in chunks])
    document_ids = np.array([chunk.document_id for chunk in chunks])
    embedding.encode(inputs)
    embedding.save_chunk_embeddings(
        path, chunk_ids=chunk_ids, document_ids=document_ids
    )
    print_embeddings_report(path)


def search_saved_embeddings(
        doc_path: str | Path, q_path: str | Path,
        search: Callable[[EmbeddingFile, EmbeddingFile, int], dict],
        top_k: int
) -> dict[str, dict[str, float]]:
    """Load saved document/chunk and query embeddings and run `search`."""
    with np.load(doc_path) as docs, np.load(q_path) as queries:
        return search(queries, docs, top_k)


def print_embeddings_report(path: str | Path) -> None:
    """Print shapes, ids and sanity checks (NaN/inf) of a saved .npz file."""
    data = np.load(path)

    print(data.files)

    document_ids = data["document_ids"]
    embeddings = data["embeddings"]

    print("document_ids:", document_ids.shape)
    print("embeddings:", embeddings.shape)

    print("First document ID:", document_ids[0])
    print("First embedding:", embeddings[0][:10])

    if 'chunk_ids' in data.files:
        chunk_ids = data['chunk_ids']
        print("chunk_ids == embeddings:", len(chunk_ids) == len(embeddings))
        print("First chunk ID:", chunk_ids[0])

    print("IDs == embeddings:", len(document_ids) == len(embeddings))

    print("dtype:", embeddings.dtype)
    print("has NaN:", np.isnan(embeddings).any())
    print("has inf:", np.isinf(embeddings).any())


def run_benchmark(
        config: BenchmarkConfig, model_names: Iterable[str],
        output_path: str | Path,
) -> list[tuple[str, Metrics]]:
    """Embed, retrieve and evaluate every variant of every model.

    Variants per model: whole documents ('classic') and one per chunking
    strategy ('<strategy>_chunk'). Run names are '<model>_<variant>'. The
    metrics are saved to `output_path` and returned.

    Unknown models or strategies raise ValueError before any data is loaded.
    """
    unknown = [s for s in config.strategies if s not in CHUNK_STRATEGIES]
    if unknown:
        raise ValueError(
            f'Unknown chunking strategy {unknown}. '
            f'Available: {", ".join(CHUNK_STRATEGIES)}'
        )
    specs = {name: get_spec(name) for name in dict.fromkeys(model_names)}

    documents, queries, qrels = load_beir_dataset(
        config.dataset, config.split, config.subset_size
    )

    results = []
    for name, spec in specs.items():
        encoder = LazyEncoder(spec.create)
        results += _run_model(
            name, encoder, config, documents, queries, qrels
        )

    save_metrics(results, output_path)
    return results


def format_table(results: list[tuple[str, Metrics]]) -> str:
    """Aligned plain-text table: run name, then the three metrics."""
    header = ('run', *METRIC_NAMES)
    rows = [
        (name, *(f'{metrics[m]:.4f}' for m in METRIC_NAMES))
        for name, metrics in results
    ]
    widths = [
        max(len(row[i]) for row in [header, *rows])
        for i in range(len(header))
    ]

    def line(row):
        first, *numbers = row
        cells = [first.ljust(widths[0])]
        cells += [n.rjust(w) for n, w in zip(numbers, widths[1:])]
        return '  '.join(cells)

    return '\n'.join([line(header), *(line(row) for row in rows)])


def _run_model(
        name: str, encoder: LazyEncoder, config: BenchmarkConfig,
        documents: Documents, queries: dict[str, str],
        qrels: dict[str, dict[str, int]],
) -> list[tuple[str, Metrics]]:
    """Build the missing embeddings of one model, then evaluate variants."""
    dataset, size = config.dataset, config.subset_size
    data_dir = config.data_dir
    doc_path = documents_path(dataset, size, name, data_dir)
    q_path = queries_path(dataset, size, name, config.split, data_dir)
    chunker = Chunking(config.chunk_size)
    splitters = {
        'fixed': chunker.fixed_chunker,
        'recursive': chunker.recursive_chunker,
    }
    chunk_paths = {
        strategy: chunks_path(
            dataset, size, name, strategy, config.chunk_size, data_dir
        )
        for strategy in config.strategies
    }

    build_document_embeddings(encoder, documents, doc_path)
    build_query_embeddings(encoder, queries, q_path)
    for strategy, path in chunk_paths.items():
        build_chunk_embeddings(
            encoder, documents, path, chunker, splitters[strategy]
        )

    searcher = Embedding(None)  # search needs no model
    rankings = {
        variant_name(BASELINE): search_saved_embeddings(
            doc_path, q_path, searcher.search, EVAL_DEPTH
        )
    }
    for strategy, path in chunk_paths.items():
        chunk_hits = search_saved_embeddings(
            path, q_path, searcher.search_chunks, config.top_k
        )
        rankings[variant_name(strategy)] = aggregate_to_documents(
            chunk_hits, EVAL_DEPTH
        )

    return [
        (f'{name}_{variant}', evaluate(qrels, ranking))
        for variant, ranking in rankings.items()
    ]
