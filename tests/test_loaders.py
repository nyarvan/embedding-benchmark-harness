"""Tests for dataset loading and metrics persistence (no network)."""
import json
from types import SimpleNamespace

import pytest

from harness import loaders
from harness.loaders import load_beir_dataset, save_metrics

METRICS = {'P@5': 0.2, 'Recall@10': 1.0, 'NDCG@10': 0.63093}


class FakeLoader:
    """Stands in for beir's GenericDataLoader with a tiny fixed dataset."""

    def __init__(self, data_path):
        self.data_path = data_path

    def load(self, split):
        self.split = split
        corpus = {
            f'd{i}': {'title': f't{i}', 'text': f'text {i}'}
            for i in range(5)
        }
        queries = {'q1': 'first query', 'q2': 'second query'}
        qrels = {'q1': {'d0': 1}, 'q2': {'d3': 1, 'd99': 1}}
        return corpus, queries, qrels


@pytest.fixture
def fake_beir(monkeypatch):
    """Patch download and loader; the result records how they were used."""
    calls = SimpleNamespace(download=None, loaders=[])

    def fake_download(url, out_dir):
        calls.download = (url, out_dir)
        return 'fake/data/path'

    def fake_loader(data_path):
        loader = FakeLoader(data_path)
        calls.loaders.append(loader)
        return loader

    monkeypatch.setattr(loaders.util, 'download_and_unzip', fake_download)
    monkeypatch.setattr(loaders, 'GenericDataLoader', fake_loader)
    return calls


def test_save_metrics_writes_one_record_per_run(tmp_path):
    path = tmp_path / 'out.json'

    save_metrics([('model_a', METRICS), ('model_b', METRICS)], path)

    assert json.loads(path.read_text(encoding='utf-8')) == [
        {'model_a': METRICS},
        {'model_b': METRICS},
    ]


def test_save_metrics_keeps_only_the_three_reported_metrics(tmp_path):
    path = tmp_path / 'out.json'
    results = {**METRICS, 'P@1': 0.5, 'NDCG@100': 0.7}

    save_metrics([('model_a', results)], path)

    saved = json.loads(path.read_text(encoding='utf-8'))
    assert saved == [{'model_a': METRICS}]


def test_save_metrics_creates_missing_parent_directories(tmp_path):
    path = tmp_path / 'results' / 'nested' / 'out.json'

    save_metrics([('model_a', METRICS)], path)

    assert path.exists()


def test_save_metrics_accepts_string_path(tmp_path):
    path = tmp_path / 'out.json'

    save_metrics([('model_a', METRICS)], str(path))

    assert path.exists()


def test_save_metrics_keeps_run_order(tmp_path):
    path = tmp_path / 'out.json'
    names = ['z_last_alphabetically', 'a_first_alphabetically', 'm_middle']

    save_metrics([(name, METRICS) for name in names], path)

    saved = json.loads(path.read_text(encoding='utf-8'))
    assert [next(iter(record)) for record in saved] == names


def test_save_metrics_writes_non_ascii_run_names_unescaped(tmp_path):
    path = tmp_path / 'out.json'

    save_metrics([('модель', METRICS)], path)

    assert 'модель' in path.read_text(encoding='utf-8')


def test_save_metrics_without_runs_writes_empty_list(tmp_path):
    path = tmp_path / 'out.json'

    save_metrics([], path)

    assert json.loads(path.read_text(encoding='utf-8')) == []


def test_save_metrics_raises_when_a_metric_is_missing(tmp_path):
    incomplete = {'P@5': 0.2, 'Recall@10': 1.0}

    with pytest.raises(KeyError):
        save_metrics([('model_a', incomplete)], tmp_path / 'out.json')


def test_load_beir_dataset_keeps_first_size_documents_in_order(fake_beir):
    documents, _, _ = load_beir_dataset(size=3)

    assert [doc_id for doc_id, _ in documents] == ['d0', 'd1', 'd2']


def test_load_beir_dataset_returns_id_document_pairs(fake_beir):
    documents, _, _ = load_beir_dataset(size=1)

    assert documents == [('d0', {'title': 't0', 'text': 'text 0'})]


def test_load_beir_dataset_returns_all_documents_if_size_is_larger(fake_beir):
    documents, _, _ = load_beir_dataset(size=500)

    assert len(documents) == 5


def test_load_beir_dataset_does_not_cut_queries_or_qrels(fake_beir):
    _, queries, qrels = load_beir_dataset(size=1)

    assert set(queries) == {'q1', 'q2'}
    # qrels keep relevant documents outside the kept subset (d3, d99)
    assert qrels['q2'] == {'d3': 1, 'd99': 1}


def test_load_beir_dataset_passes_split_to_loader(fake_beir):
    load_beir_dataset(split='train')

    assert fake_beir.loaders[0].split == 'train'


def test_load_beir_dataset_downloads_named_dataset_to_datasets_dir(fake_beir):
    load_beir_dataset(name='nfcorpus')

    url, out_dir = fake_beir.download
    assert url.endswith('/nfcorpus.zip')
    assert out_dir == 'datasets'
    assert fake_beir.loaders[0].data_path == 'fake/data/path'
