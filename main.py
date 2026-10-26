"""
Embedding pipeline: build and cache embeddings, retrieve, evaluate, save
metrics.
"""
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from domain.chunk import Chunking
from harness.aggregate import aggregate_to_documents
from harness.embed import Embedding, Encoder
from harness.evaluate import evaluate
from harness.loaders import load_beir_dataset, save_metrics
from harness.openai_encoder import OpenAIEncoder

CHUNK_SIZE = 500
MODEL_IDS = {
    # 'bge_m3': 'BAAI/bge-m3',
    # 'e5_large_v2': 'intfloat/e5-large-v2',
    'openai_3_small': 'text-embedding-3-small'
}
RESULTS_PATH = 'results/scifact_500docs_openai.json'


def load_models() -> dict[str, Encoder]:
    """
    Load the embedding models by short name (heavy, call only when needed).
    """
    models = {}
    for name, model in MODEL_IDS.items():
        if name.startswith('openai_'):
            models[name] = OpenAIEncoder(model)
            continue
        models[name] = SentenceTransformer(model)
    return models


def build_document_embeddings(
        model: Encoder,
        documents: list[tuple[str, dict[str, str]]], path: str | Path
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
        model: Encoder,
        documents: list[tuple[str, dict[str, str]]], path: str | Path,
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
        doc_path: str | Path, q_path: str | Path, search: Callable,
        top_k: int
) -> dict[str, dict[str, float]]:
    """Load saved document/chunk and query embeddings and run `search`."""
    docs = np.load(doc_path)
    queries = np.load(q_path)

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


def find_relevant(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]]
) -> list[tuple[str, str]]:
    """
    List (query_id, doc_id) pairs of retrieved documents that are relevant.
    """
    relevants = []

    for q_id, docs in results.items():
        qrel_docs = qrels[q_id]

        for doc_id in docs:
            if doc_id in qrel_docs:
                relevants.append((q_id, doc_id))

    return relevants


def hit_at_k(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]]
) -> float:
    """Share of queries with at least one relevant document in the results."""
    hits = 0

    for q_id, docs in results.items():
        qrel_docs = qrels[q_id]

        if any(doc_id in qrel_docs for doc_id in docs):
            hits += 1

    return hits / len(results)


def run_benchmark(
        documents: list[tuple[str, dict[str, str]]], queries: dict[str, str],
        qrels: dict[str, dict[str, int]], top_k: int,
        output_path: str | Path = RESULTS_PATH
) -> None:
    """Embed, retrieve and evaluate three variants per model, save metrics.

    Variants: whole documents ('classic'), fixed chunks, recursive chunks.
    Chunk rankings are aggregated back to documents before evaluation.
    """
    chunker = Chunking(CHUNK_SIZE)

    retrieval_results = []

    for name, model in load_models().items():
        embedding = Embedding(model)
        runs = {
            'classic': {'path': f'data/embeddings_{name}.npz'},
            'fixed_chunk': {
                'path': f'data/chunks_fixed_{name}.npz',
                'splitter': chunker.fixed_chunker
            },
            'recursive_chunk': {
                'path': f'data/chunks_recursive_{name}.npz',
                'splitter': chunker.recursive_chunker
            },
            'queries': {'path': f'data/embeddings_queries_{name}.npz'}
        }

        for mode, values in runs.items():
            if mode == 'classic':
                build_document_embeddings(model, documents, values['path'])
            elif mode == 'queries':
                build_query_embeddings(model, queries, values['path'])
            else:
                build_chunk_embeddings(
                    model, documents, values['path'], chunker,
                    values['splitter']
                )

        for mode, values in runs.items():
            if mode == 'queries':
                continue
            elif mode in ['fixed_chunk', 'recursive_chunk']:
                search_results = search_saved_embeddings(
                    values['path'], runs['queries']['path'],
                    embedding.search_chunks, top_k
                )
                search_results = aggregate_to_documents(search_results, top_k)
            else:
                search_results = search_saved_embeddings(
                    values['path'], runs['queries']['path'], embedding.search,
                    top_k
                )

            retrieval_results.append(
                (f'{name}_{mode}', evaluate(qrels, search_results))
            )

    save_metrics(retrieval_results, output_path)


def summarize_aggregated_lengths(top_k: int, name: str = 'bge_m3') -> None:
    """Print how many documents each chunk strategy returns per query.

    Diagnostic for a too small chunk-level `top_k`: when many chunks come from
    the same document, fewer than 10 distinct documents are left after
    aggregation and Recall@10 is capped.
    """
    embedding = Embedding(None)

    modes = {
        'fixed': f'data/chunks_fixed_{name}.npz',
        'recursive': f'data/chunks_recursive_{name}.npz'
    }
    queries = np.load(f'data/embeddings_queries_{name}.npz')

    results = {}

    for mode, path in modes.items():
        chunks = np.load(path)

        search_results = embedding.search_chunks(queries, chunks, top_k)
        aggregated = aggregate_to_documents(search_results, top_k)

        lengths = np.array([len(docs) for docs in aggregated.values()])

        results[mode] = {
            'len < 10': float(np.mean(lengths < 10)),
            'min': int(np.min(lengths)),
            'mean': float(np.mean(lengths, dtype=float))
        }

    print(results)


if __name__ == '__main__':
    documents, queries, qrels = load_beir_dataset(split='train')

    run_benchmark(documents, queries, qrels, top_k=200)
