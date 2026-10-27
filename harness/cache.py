"""Layout of the embedding cache on disk.

    <data_dir>/<dataset>-<subset_size>/<model>/
        documents.npz
        queries_<split>.npz
        chunks_<strategy>_<chunk_size>.npz

Every parameter that changes the embeddings is part of the path, so an
existing file can never be mistaken for the result of another configuration.
"""
import re
from pathlib import Path

DEFAULT_DATA_DIR = 'data'
# Letters, digits, '_', '.' and '-'; must start with a letter or a digit
_SAFE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*')


def cache_dir(
        dataset: str, subset_size: int, model: str,
        data_dir: str | Path = DEFAULT_DATA_DIR,
) -> Path:
    """Directory with all cached embeddings of one model on one subset."""
    dataset = _check_name(dataset, 'dataset')
    model = _check_name(model, 'model')
    _check_positive(subset_size, 'subset_size')
    return Path(data_dir) / f'{dataset}-{subset_size}' / model


def documents_path(
        dataset: str, subset_size: int, model: str,
        data_dir: str | Path = DEFAULT_DATA_DIR,
) -> Path:
    """File with whole-document embeddings."""
    return cache_dir(dataset, subset_size, model, data_dir) / 'documents.npz'


def queries_path(
        dataset: str, subset_size: int, model: str, split: str,
        data_dir: str | Path = DEFAULT_DATA_DIR,
) -> Path:
    """File with query embeddings of one dataset split."""
    split = _check_name(split, 'split')
    directory = cache_dir(dataset, subset_size, model, data_dir)
    return directory / f'queries_{split}.npz'


def chunks_path(
        dataset: str, subset_size: int, model: str, strategy: str,
        chunk_size: int, data_dir: str | Path = DEFAULT_DATA_DIR,
) -> Path:
    """File with chunk embeddings for one strategy and chunk size."""
    strategy = _check_name(strategy, 'strategy')
    _check_positive(chunk_size, 'chunk_size')
    directory = cache_dir(dataset, subset_size, model, data_dir)
    return directory / f'chunks_{strategy}_{chunk_size}.npz'


def _check_name(value: str, label: str) -> str:
    """Strip the value and reject anything that could leave the cache dir."""
    value = value.strip()
    if not _SAFE_NAME.fullmatch(value) or '..' in value:
        raise ValueError(f'Invalid {label}: {value!r}')
    return value


def _check_positive(value: int, label: str) -> None:
    if value <= 0:
        raise ValueError(f'{label} must be positive, got {value}')
