"""End-to-end tests of the pipeline with a fake encoder and a tiny dataset."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from harness import pipeline
from harness.cache import chunks_path, documents_path, queries_path
from harness.models import ModelSpec
from harness.pipeline import BenchmarkConfig, format_table, run_benchmark

VOCAB = ['apple', 'banana', 'cherry', 'date', 'elder', 'fig', 'grape', 'honey']
DOCUMENTS = [
    ('d1', {'title': 'fruit one', 'text': 'apple banana apple banana apple'}),
    ('d2', {'title': 'fruit two', 'text': 'cherry date cherry date cherry'}),
    ('d3', {'title': 'berries', 'text': 'elder fig elder fig elder fig'}),
    ('d4', {'title': 'misc', 'text': 'grape honey grape honey grape honey'}),
]
QUERIES = {'q1': 'apple', 'q2': 'cherry', 'q3': 'fig'}
QRELS = {'q1': {'d1': 1}, 'q2': {'d2': 1}, 'q3': {'d3': 1}}
PERFECT = {'P@5': 0.2, 'Recall@10': 1.0, 'NDCG@10': 1.0}


class CountingEncoder:
    """Bag-of-words vectors over VOCAB; counts how often it is used."""

    def __init__(self):
        self.encode_calls = 0

    def encode(self, inputs, **kwargs):
        self.encode_calls += 1
        vectors = np.zeros((len(inputs), len(VOCAB)), dtype=np.float32)
        for row, text in enumerate(inputs):
            for word in text.lower().replace('\n', ' ').split():
                if word in VOCAB:
                    vectors[row, VOCAB.index(word)] += 1
        return vectors


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Fake dataset and model; `env.created` counts model constructions."""
    encoder = CountingEncoder()
    created = SimpleNamespace(count=0)

    def create():
        created.count += 1
        return encoder

    monkeypatch.setattr(
        pipeline, 'load_beir_dataset',
        lambda *args, **kwargs: (DOCUMENTS, QUERIES, QRELS),
    )
    monkeypatch.setattr(
        pipeline, 'get_spec', lambda name: ModelSpec(name, create)
    )
    monkeypatch.setattr(pipeline, 'print_embeddings_report', lambda path: None)

    config = BenchmarkConfig(
        dataset='tiny', subset_size=4, split='train', chunk_size=25,
        top_k=10, data_dir=tmp_path / 'data',
    )
    return SimpleNamespace(
        config=config, encoder=encoder, created=created,
        output=tmp_path / 'results' / 'out.json',
    )


def test_run_benchmark_names_runs_after_model_and_variant(env):
    results = run_benchmark(env.config, ['fake'], env.output)

    assert [name for name, _ in results] == [
        'fake_classic', 'fake_fixed_chunk', 'fake_recursive_chunk'
    ]


def test_run_benchmark_scores_perfect_retrieval_for_every_variant(env):
    results = run_benchmark(env.config, ['fake'], env.output)

    for _, metrics in results:
        assert metrics == pytest.approx(PERFECT, abs=1e-5)


def test_run_benchmark_writes_the_returned_metrics_to_the_output_file(env):
    results = run_benchmark(env.config, ['fake'], env.output)

    saved = json.loads(env.output.read_text(encoding='utf-8'))
    assert saved == [{name: metrics} for name, metrics in results]


def test_run_benchmark_caches_embeddings_under_the_cache_layout(env):
    run_benchmark(env.config, ['fake'], env.output)

    c = env.config
    expected = [
        documents_path('tiny', 4, 'fake', c.data_dir),
        queries_path('tiny', 4, 'fake', 'train', c.data_dir),
        chunks_path('tiny', 4, 'fake', 'fixed', 25, c.data_dir),
        chunks_path('tiny', 4, 'fake', 'recursive', 25, c.data_dir),
    ]
    assert all(path.exists() for path in expected)


def test_chunk_cache_keeps_the_link_to_the_source_document(env):
    run_benchmark(env.config, ['fake'], env.output)

    path = chunks_path('tiny', 4, 'fake', 'fixed', 25, env.config.data_dir)
    saved = np.load(path)
    chunk_to_doc = dict(zip(saved['chunk_ids'], saved['document_ids']))
    assert len(chunk_to_doc) > len(DOCUMENTS)  # documents were really split
    assert all(
        chunk_id.rsplit('_', 1)[0] == doc_id
        for chunk_id, doc_id in chunk_to_doc.items()
    )
    assert set(chunk_to_doc.values()) == {'d1', 'd2', 'd3', 'd4'}


def test_second_run_reuses_the_cache_and_never_builds_the_model(env):
    first = run_benchmark(env.config, ['fake'], env.output)
    built_before = env.created.count
    calls_before = env.encoder.encode_calls

    second = run_benchmark(env.config, ['fake'], env.output)

    assert env.created.count == built_before
    assert env.encoder.encode_calls == calls_before
    assert second == first


def test_run_recomputes_only_the_cache_file_that_is_missing(env):
    run_benchmark(env.config, ['fake'], env.output)
    calls_before = env.encoder.encode_calls
    chunks_path(
        'tiny', 4, 'fake', 'fixed', 25, env.config.data_dir
    ).unlink()

    run_benchmark(env.config, ['fake'], env.output)

    assert env.encoder.encode_calls == calls_before + 1


def test_empty_strategies_run_only_the_classic_variant(env, tmp_path):
    config = BenchmarkConfig(
        dataset='tiny', subset_size=4, chunk_size=25, strategies=(),
        data_dir=tmp_path / 'data',
    )

    results = run_benchmark(config, ['fake'], env.output)

    assert [name for name, _ in results] == ['fake_classic']
    assert not chunks_path(
        'tiny', 4, 'fake', 'fixed', 25, config.data_dir
    ).exists()


def test_several_models_get_separate_caches_and_runs(env):
    results = run_benchmark(env.config, ['fake_a', 'fake_b'], env.output)

    names = [name for name, _ in results]
    assert names[::3] == ['fake_a_classic', 'fake_b_classic']
    assert documents_path('tiny', 4, 'fake_a', env.config.data_dir).exists()
    assert documents_path('tiny', 4, 'fake_b', env.config.data_dir).exists()


def test_duplicate_model_names_are_run_once(env):
    results = run_benchmark(env.config, ['fake', 'fake'], env.output)

    assert len(results) == 3


def test_unknown_strategy_is_rejected_before_any_work(env):
    config = BenchmarkConfig(
        dataset='tiny', subset_size=4, strategies=('semantic',),
        data_dir=env.config.data_dir,
    )

    with pytest.raises(ValueError, match='semantic'):
        run_benchmark(config, ['fake'], env.output)

    assert env.created.count == 0
    assert not env.config.data_dir.exists()
    assert not env.output.exists()


def test_unknown_model_is_rejected_before_any_data_is_loaded(
        monkeypatch, env
):
    def unknown(name):
        raise ValueError(f'Unknown model {name!r}')

    loaded = SimpleNamespace(count=0)

    def load(*args, **kwargs):
        loaded.count += 1
        return DOCUMENTS, QUERIES, QRELS

    monkeypatch.setattr(pipeline, 'get_spec', unknown)
    monkeypatch.setattr(pipeline, 'load_beir_dataset', load)

    with pytest.raises(ValueError, match='nope'):
        run_benchmark(env.config, ['nope'], env.output)

    assert loaded.count == 0


def test_format_table_aligns_columns_and_rounds_to_four_decimals():
    openai = {'P@5': 0.02299, 'Recall@10': 0.10764, 'NDCG@10': 0.09949}
    results = [('openai_classic', openai), ('x', PERFECT)]

    lines = format_table(results).splitlines()

    assert lines[0].split() == ['run', 'P@5', 'Recall@10', 'NDCG@10']
    assert lines[1].split() == ['openai_classic', '0.0230', '0.1076', '0.0995']
    assert lines[2].split() == ['x', '0.2000', '1.0000', '1.0000']
    assert len({len(line) for line in lines}) == 1  # equal width


def test_format_table_without_results_returns_only_the_header():
    assert format_table([]).split() == ['run', 'P@5', 'Recall@10', 'NDCG@10']
