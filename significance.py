"""Per-query metrics and paired bootstrap comparison of retrieval runs.

Compares whole-document retrieval (the baseline) with each chunking strategy
of one model on the queries that can score at all, and shows whether the
differences can be told from noise.
"""
import json
from dataclasses import dataclass
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

from harness.aggregate import aggregate_to_documents
from harness.cache import chunks_path, documents_path, queries_path
from harness.config import BASELINE, EVAL_DEPTH, BenchmarkConfig, variant_name
from harness.embed import Embedding, EmbeddingFile
from harness.evaluate import evaluate
from harness.loaders import load_beir_dataset

METRICS = ('NDCG@10', 'Recall@10')
EPS = 1e-9  # scores closer than this count as equal

Scores = dict[str, dict[str, np.ndarray]]


@dataclass(frozen=True)
class Comparison:
    """Paired comparison of strategy `a` against `b` on one metric (a - b)."""

    a: str
    b: str
    metric: str
    mean_diff: float
    low: float  # 95% bootstrap interval of the mean difference
    high: float
    wins: int  # queries where `a` scores higher
    losses: int  # queries where `a` scores lower
    p_sign: float  # exact two-sided sign test on wins and losses
    changed: tuple[str, ...]  # ids of the queries that differ


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


def strategy_paths(model: str, config: BenchmarkConfig) -> dict[str, Path]:
    """Cache file of the baseline and of every chunking strategy of a run."""
    paths = {
        BASELINE: documents_path(
            config.dataset, config.subset_size, model, config.data_dir
        )
    }
    for strategy in config.strategies:
        paths[strategy] = chunks_path(
            config.dataset, config.subset_size, model, strategy,
            config.chunk_size, config.data_dir,
        )
    return paths


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

    if strategy == BASELINE:
        return embedding.search(queries, docs, top_k)

    results = embedding.search_chunks(queries, docs, top_k)
    return aggregate_to_documents(results, EVAL_DEPTH)


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


def load_saved_metrics(
    path: str | Path, model: str, strategies: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    """Read saved metrics as {strategy: {metric: value}} for one model.

    The baseline and every strategy in `strategies` must be in the file.
    """
    records = {
        key: value
        for record in json.loads(Path(path).read_text(encoding='utf-8'))
        for key, value in record.items()
    }
    names = {
        strategy: f'{model}_{variant_name(strategy)}'
        for strategy in (BASELINE, *strategies)
    }
    missing = [name for name in names.values() if name not in records]
    if missing:
        raise ValueError(
            f'{path} has no run named {", ".join(missing)}; was it produced '
            'with the same model and chunking strategies?'
        )
    return {strategy: records[name] for strategy, name in names.items()}


def check_against_saved_metrics(
    scores: Scores, total_queries: int,
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


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test: how likely is a split this uneven by chance.

    Queries without a difference are ignored. Without any, returns 1.0.
    """
    total = wins + losses
    if total == 0:
        return 1.0
    tail = sum(comb(total, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / 2 ** total)


def changed_queries(
    a: np.ndarray, b: np.ndarray, active: list[str],
) -> list[str]:
    """Query ids where the two strategies scored differently."""
    return [q for q, x, y in zip(active, a, b) if abs(x - y) > EPS]


def comparison_pairs(strategies: list[str]) -> list[tuple[str, str]]:
    """(a, b) pairs: every strategy against the baseline, then in pairs."""
    pairs = [(strategy, BASELINE) for strategy in strategies]
    pairs += [(b, a) for a, b in combinations(strategies, 2)]
    return pairs


def compare_strategies(
    scores: Scores, active: list[str], n_resamples: int = 10_000,
    seed: int = 0,
) -> list[Comparison]:
    """Compare every chunking strategy with the baseline and with the others.

    `scores` maps the baseline and each strategy to its per-query metrics.
    """
    strategies = [name for name in scores if name != BASELINE]
    comparisons = []
    for a, b in comparison_pairs(strategies):
        for metric in METRICS:
            x, y = scores[a][metric], scores[b][metric]
            mean, low, high = paired_bootstrap(x, y, n_resamples, seed)
            diff = x - y
            wins = int((diff > EPS).sum())
            losses = int((diff < -EPS).sum())
            comparisons.append(Comparison(
                a, b, metric, mean, low, high, wins, losses,
                sign_test_p(wins, losses),
                tuple(changed_queries(x, y, active)),
            ))
    return comparisons


def format_comparisons(comparisons: list[Comparison]) -> str:
    """Aligned table of the comparisons with a short legend."""
    header = ('comparison', 'metric', 'mean diff', '95% CI', 'better/worse',
              'p (sign)')
    rows = [
        (
            f'{c.a} - {c.b}', c.metric, f'{c.mean_diff:+.4f}',
            f'[{c.low:+.4f}, {c.high:+.4f}]', f'{c.wins}/{c.losses}',
            f'{c.p_sign:.3f}',
        )
        for c in comparisons
    ]
    widths = [
        max(len(row[i]) for row in [header, *rows])
        for i in range(len(header))
    ]
    lines = [
        '  '.join(cell.ljust(width) for cell, width in zip(row, widths))
        .rstrip()
        for row in [header, *rows]
    ]
    lines.append(
        '\nmean diff = first minus second, positive means the first is '
        'better; better/worse = queries where the first scores higher/lower; '
        'p = exact two-sided sign test on those queries.'
    )
    return '\n'.join(lines)


def run_significance_analysis(
    model: str, config: BenchmarkConfig, metrics_path: str | Path,
    n_resamples: int = 10_000,
) -> str:
    """Compare the baseline and the chunking strategies of one model.

    Needs the cache of a finished benchmark run and its saved metrics file.
    The per-query scores are verified against the saved metrics first, so an
    analysis never rests on a mismatching cache. Returns the report text.
    """
    paths = strategy_paths(model, config)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise ValueError(
            'Missing cache files, run the benchmark first: '
            + ', '.join(missing)
        )

    documents, _, qrels = load_beir_dataset(
        config.dataset, config.split, config.subset_size
    )
    active = get_active_queries(qrels, documents)
    if not active:
        raise ValueError('No query has a relevant document in the subset')

    queries = np.load(
        queries_path(
            config.dataset, config.subset_size, model, config.split,
            config.data_dir,
        )
    )
    scores = {
        strategy: per_query_scores(
            qrels, retrieve_documents(strategy, queries, path, config.top_k),
            active,
        )
        for strategy, path in paths.items()
    }
    expected = load_saved_metrics(metrics_path, model, config.strategies)
    check_against_saved_metrics(scores, len(qrels), expected)

    comparisons = compare_strategies(scores, active, n_resamples)
    title = (
        f'{model} on {config.dataset} ({config.subset_size} documents, '
        f'{config.split} queries): {len(active)} of {len(qrels)} queries '
        'have a relevant document'
    )
    return f'{title}\n\n{format_comparisons(comparisons)}'
