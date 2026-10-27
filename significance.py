"""Per-query metrics and paired bootstrap comparison of retrieval runs."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from harness.aggregate import aggregate_to_documents
from harness.cache import chunks_path, documents_path, queries_path
from harness.config import CHUNK_STRATEGIES, BenchmarkConfig
from harness.embed import Embedding, EmbeddingFile
from harness.evaluate import evaluate
from harness.loaders import load_beir_dataset

# The analysis covers the default benchmark setup of the pipeline
_DEFAULTS = BenchmarkConfig()
DATASET = _DEFAULTS.dataset
SUBSET_SIZE = _DEFAULTS.subset_size
SPLIT = _DEFAULTS.split
CHUNK_SIZE = _DEFAULTS.chunk_size
METRICS = ('NDCG@10', 'Recall@10')
EVAL_DEPTH = 10  # documents kept per query, matches the @10 metrics
# Model name -> file with the metrics saved by the pipeline for that model
SAVED_METRICS_PATHS = {
    'bge_m3': 'results/scifact_500docs_chunking_topk200.json',
    'openai_3_small': 'results/scifact_500docs_openai.json',
}
# Strategy name -> suffix of its run name in the saved metrics file
RESULT_KEY_SUFFIXES = {
    'baseline': 'classic',
    'fixed': 'fixed_chunk',
    'recursive': 'recursive_chunk',
}


def get_active_queries(
    qrels: dict[str, dict[str, int]],
    documents: list[tuple[str, dict[str, str]]],
) -> list[str]:
    """Query ids with at least one relevant document in the corpus subset.

    Only these queries can score above zero; the rest add zeros to every
    strategy and only dilute the averages.
    """
    doc_ids = {doc_id for doc_id, _ in documents}
    active = [q for q, rel in qrels.items() if any(d in doc_ids for d in rel)]
    return active


def strategy_paths(
    model: str, dataset: str = DATASET, subset_size: int = SUBSET_SIZE,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, Path]:
    """Cache files of the three compared strategies for one model."""
    return {
        'baseline': documents_path(dataset, subset_size, model),
        'fixed': chunks_path(dataset, subset_size, model, 'fixed', chunk_size),
        'recursive': chunks_path(
            dataset, subset_size, model, 'recursive', chunk_size
        ),
    }


def retrieve_documents(
    strategy: str, queries: EmbeddingFile, docs_path: str | Path,
    top_k: int = 50
) -> dict[str, dict[str, float]]:
    """Rank documents for every query: {query_id: {doc_id: score}}.

    Chunk strategies search `top_k` chunks and aggregate them to documents;
    the baseline searches whole-document embeddings.
    """
    embedding = Embedding(None)
    docs = np.load(docs_path)

    if strategy in CHUNK_STRATEGIES:
        results = embedding.search_chunks(queries, docs, top_k)
        results = aggregate_to_documents(results, EVAL_DEPTH)
    else:
        results = embedding.search(queries, docs, top_k)

    return results


def per_query_scores(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    active: list[str],
) -> dict[str, np.ndarray]:
    """Score every active query separately.

    Returns {metric: array}, with the arrays ordered as `active`.
    """
    scores = {metric: [] for metric in METRICS}
    for q in active:
        eval_result = evaluate({q: qrels[q]}, {q: results[q]})
        for metric in METRICS:
            scores[metric].append(eval_result[metric])

    return {metric: np.array(values) for metric, values in scores.items()}


def load_saved_metrics(path: str, name: str) -> dict[str, dict[str, float]]:
    """Read saved metrics as {strategy: {metric: value}} for one model."""
    records = {
        key: value
        for record in json.loads(Path(path).read_text(encoding='utf-8'))
        for key, value in record.items()
    }
    return {
        strategy: records[f'{name}_{suffix}']
        for strategy, suffix in RESULT_KEY_SUFFIXES.items()
    }


def check_against_saved_metrics(
    scores: dict[str, dict[str, np.ndarray]], total_queries: int,
    expected: dict[str, dict[str, float]], atol: float = 1e-4,
) -> None:
    """Raise if sum(per-query scores) / total_queries differs from the table.

    `total_queries` is the number of all evaluated queries, because the saved
    metrics are averages over all of them, not only over the active ones.
    """
    mismatches = []
    for strategy, metrics in scores.items():
        for metric, values in metrics.items():
            actual = values.sum() / total_queries
            target = expected[strategy][metric]
            if not np.isclose(actual, target, atol=atol):
                mismatches.append(
                    f'{strategy} {metric}: got {actual:.5f}, '
                    f'expected {target:.5f}'
                )

    if mismatches:
        raise ValueError(
            'Per-query scores do not match the table:\n'
            + '\n'.join(mismatches)
        )


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000, seed: int = 0,
) -> tuple[float, float, float]:
    """Mean per-query difference (a - b) with its 95% bootstrap interval.

    The two arrays must hold scores of the same queries in the same order.
    Returns (mean, low, high).
    """
    rng = np.random.default_rng(seed)
    diff = a - b
    idx = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)


def changed_queries(
    a: np.ndarray, b: np.ndarray, active: list[str],
) -> list[str]:
    """Query ids where the two strategies scored differently."""
    return [q for q, x, y in zip(active, a, b) if not np.isclose(x, y)]


def run_significance_analysis(name: str = 'bge_m3', top_k: int = 50) -> None:
    """Compare baseline, fixed and recursive chunking on the active queries.

    Verifies the per-query scores against the saved metrics table first, then
    prints the paired bootstrap result and the changed queries for every pair
    of strategies and metric.
    """
    documents, _, qrels = load_beir_dataset(DATASET, SPLIT, SUBSET_SIZE)
    active = get_active_queries(qrels, documents)
    queries = np.load(queries_path(DATASET, SUBSET_SIZE, name, SPLIT))

    strategies = strategy_paths(name)

    scores = {s: per_query_scores(
        qrels, retrieve_documents(s, queries, path, top_k), active
    ) for s, path in strategies.items()}

    expected = load_saved_metrics(SAVED_METRICS_PATHS[name], name)

    check_against_saved_metrics(scores, len(qrels), expected)

    for a, b in itertools.combinations(strategies, 2):
        for metric in METRICS:
            mean, low, high = paired_bootstrap(
                scores[a][metric], scores[b][metric]
            )
            changed = changed_queries(
                scores[a][metric], scores[b][metric], active
            )
            print(f'{a}-{b}: {metric} - mean={mean}, low={low}, high={high}')
            print(f'{changed}\n\n')


if __name__ == '__main__':
    # Usage: python significance.py [model_name], default bge_m3
    run_significance_analysis(sys.argv[1] if len(sys.argv) > 1 else 'bge_m3')
