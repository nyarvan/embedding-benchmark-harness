from pathlib import Path

import pytest

from harness.cache import cache_dir, documents_path, queries_path, chunks_path


DATASET = 'scifact'
SUBSET_SIZE = 500
MODEL = 'bge_m3'
SPLIT = 'train'
STRATEGY = 'fixed'
CHUNK_SIZE = 500


def test_cache_dir():
    path = cache_dir(DATASET, SUBSET_SIZE, MODEL)

    assert path == Path(f'data/{DATASET}-{SUBSET_SIZE}/{MODEL}')


def test_documents_path():
    path = documents_path(DATASET, SUBSET_SIZE, MODEL)

    assert path == Path(f'data/{DATASET}-{SUBSET_SIZE}/{MODEL}/documents.npz')


def test_queries_path():
    path = queries_path(DATASET, SUBSET_SIZE, MODEL, SPLIT)

    assert path == Path(
        f'data/{DATASET}-{SUBSET_SIZE}/{MODEL}/queries_{SPLIT}.npz'
    )


def test_chunks_path():
    path = chunks_path(DATASET, SUBSET_SIZE, MODEL, STRATEGY, CHUNK_SIZE)

    assert path == Path(
        f'data/{DATASET}-{SUBSET_SIZE}/{MODEL}'
        f'/chunks_{STRATEGY}_{CHUNK_SIZE}.npz'
    )


def all_paths(**changes):
    """Every cache path for the default configuration, with `changes`."""
    params = dict(
        dataset=DATASET, subset_size=SUBSET_SIZE, model=MODEL, split=SPLIT,
        strategy=STRATEGY, chunk_size=CHUNK_SIZE,
    )
    params.update(changes)
    return {
        'documents': documents_path(
            params['dataset'], params['subset_size'], params['model']
        ),
        'queries': queries_path(
            params['dataset'], params['subset_size'], params['model'],
            params['split']
        ),
        'chunks': chunks_path(
            params['dataset'], params['subset_size'], params['model'],
            params['strategy'], params['chunk_size']
        ),
    }


@pytest.mark.parametrize('change', [
    {'dataset': 'nfcorpus'},
    {'subset_size': 1000},
    {'model': 'e5_large_v2'},
])
def test_paths_differ_when_dataset_subset_or_model_changes(change):
    base = all_paths()
    changed = all_paths(**change)

    assert all(base[kind] != changed[kind] for kind in base)


def test_queries_path_differs_between_splits():
    assert all_paths()['queries'] != all_paths(split='test')['queries']


@pytest.mark.parametrize('change', [
    {'strategy': 'recursive'},
    {'chunk_size': 1000},
])
def test_chunks_path_differs_when_strategy_or_chunk_size_changes(change):
    assert all_paths()['chunks'] != all_paths(**change)['chunks']


def test_paths_are_stable_for_equal_parameters():
    assert all_paths() == all_paths()


def test_surrounding_whitespace_in_names_is_ignored():
    assert all_paths(model=f' {MODEL} ') == all_paths()


@pytest.mark.parametrize('bad_name', [
    '', ' ', '..', '../x', 'a/b', 'a\\b', 'a b', '.hidden', 'x..y',
])
def test_unsafe_names_are_rejected(bad_name):
    with pytest.raises(ValueError):
        cache_dir(DATASET, SUBSET_SIZE, bad_name)
    with pytest.raises(ValueError):
        cache_dir(bad_name, SUBSET_SIZE, MODEL)
    with pytest.raises(ValueError):
        queries_path(DATASET, SUBSET_SIZE, MODEL, bad_name)
    with pytest.raises(ValueError):
        chunks_path(DATASET, SUBSET_SIZE, MODEL, bad_name, CHUNK_SIZE)


@pytest.mark.parametrize('size', [0, -1])
def test_non_positive_sizes_are_rejected(size):
    with pytest.raises(ValueError):
        cache_dir(DATASET, size, MODEL)
    with pytest.raises(ValueError):
        chunks_path(DATASET, SUBSET_SIZE, MODEL, STRATEGY, size)


def test_data_dir_can_be_overridden(tmp_path):
    path = documents_path(DATASET, SUBSET_SIZE, MODEL, data_dir=tmp_path)

    assert path == tmp_path / f'{DATASET}-{SUBSET_SIZE}' / MODEL / 'documents.npz'
