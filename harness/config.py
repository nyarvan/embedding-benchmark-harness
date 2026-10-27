"""Benchmark settings shared by the pipeline and the command line.

Kept free of heavy imports (torch, beir) so that the command line can read the
defaults and validate its arguments before any of them is loaded.
"""
from dataclasses import dataclass
from pathlib import Path

from harness.cache import DEFAULT_DATA_DIR

CHUNK_STRATEGIES = ('fixed', 'recursive')
BASELINE = 'baseline'  # whole-document retrieval, the reference for chunking
EVAL_DEPTH = 10  # documents kept per query, matches the @10 metrics


def variant_name(strategy: str) -> str:
    """Name of a strategy inside a run name: 'classic' or '<strategy>_chunk'.

    A run is saved as '<model>_<variant>', e.g. 'bge_m3_fixed_chunk'.
    """
    return 'classic' if strategy == BASELINE else f'{strategy}_chunk'


@dataclass(frozen=True)
class BenchmarkConfig:
    """Everything except the models that defines one benchmark run.

    `top_k` is the retrieval depth over chunks; it must be well above the
    number of documents evaluated because several chunks of one document
    collapse into one.
    """

    dataset: str = 'scifact'
    subset_size: int = 500
    split: str = 'train'
    chunk_size: int = 500
    strategies: tuple[str, ...] = CHUNK_STRATEGIES
    top_k: int = 50
    data_dir: str | Path = DEFAULT_DATA_DIR
