"""Tests for the significance analysis: statistics, comparison, report."""
import json
import shutil
from math import log2
from types import SimpleNamespace

import numpy as np
import pytest

import significance
from harness import pipeline
from harness.cache import chunks_path, documents_path
from harness.config import BASELINE, BenchmarkConfig
from harness.models import ModelSpec
from significance import (
    Comparison, changed_queries, check_against_saved_metrics,
    compare_strategies, comparison_pairs, format_comparisons,
    get_active_queries, load_saved_metrics, paired_bootstrap,
    per_query_scores, run_significance_analysis, sign_test_p, strategy_paths,
)
from tests.fakes import DOCUMENTS, QRELS, QUERIES, CountingEncoder


def arrays(ndcg, recall=None):
    recall = ndcg if recall is None else recall
    return {'NDCG@10': np.array(ndcg), 'Recall@10': np.array(recall)}


# --- active queries ------------------------------------------------------

def test_get_active_queries_keeps_queries_with_a_relevant_document_in_subset():
    documents = [('d1', {}), ('d2', {})]
    qrels = {'q1': {'d1': 1}, 'q2': {'d99': 1}, 'q3': {'d99': 1, 'd2': 1}}

    assert get_active_queries(qrels, documents) == ['q1', 'q3']


def test_get_active_queries_is_empty_when_no_relevant_document_is_in_subset():
    assert get_active_queries({'q1': {'d99': 1}}, [('d1', {})]) == []


# --- paths ---------------------------------------------------------------

def test_strategy_paths_start_with_the_baseline_then_follow_the_config(
        tmp_path
):
    config = BenchmarkConfig(
        dataset='tiny', subset_size=4, chunk_size=25, data_dir=tmp_path,
        strategies=('recursive', 'fixed'),
    )

    paths = strategy_paths('m', config)

    assert list(paths) == [BASELINE, 'recursive', 'fixed']
    assert paths[BASELINE] == documents_path('tiny', 4, 'm', tmp_path)
    assert paths['fixed'] == chunks_path('tiny', 4, 'm', 'fixed', 25, tmp_path)


def test_strategy_paths_without_strategies_have_only_the_baseline(tmp_path):
    config = BenchmarkConfig(strategies=(), data_dir=tmp_path)

    assert list(strategy_paths('m', config)) == [BASELINE]


# --- per-query scores ----------------------------------------------------

def test_per_query_scores_are_ordered_as_the_active_queries():
    qrels = {'q1': {'d1': 1}, 'q2': {'d1': 1}}
    results = {
        'q1': {'d1': 0.9, 'd2': 0.1},   # relevant document first
        'q2': {'d2': 0.9, 'd1': 0.1},   # relevant document second
    }

    scores = per_query_scores(qrels, results, ['q2', 'q1'])

    assert scores['NDCG@10'] == pytest.approx([1 / log2(3), 1.0], abs=1e-5)
    assert scores['Recall@10'].tolist() == [1.0, 1.0]


def test_per_query_scores_return_numpy_arrays():
    scores = per_query_scores(
        {'q1': {'d1': 1}}, {'q1': {'d1': 0.9}}, ['q1']
    )

    assert all(isinstance(values, np.ndarray) for values in scores.values())


# --- saved metrics -------------------------------------------------------

def write_metrics(path, runs):
    path.write_text(json.dumps([{name: m} for name, m in runs.items()]))


def test_load_saved_metrics_maps_strategies_to_their_run_names(tmp_path):
    path = tmp_path / 'metrics.json'
    classic = {'NDCG@10': 0.1, 'Recall@10': 0.2}
    fixed = {'NDCG@10': 0.3, 'Recall@10': 0.4}
    write_metrics(path, {'m_classic': classic, 'm_fixed_chunk': fixed})

    loaded = load_saved_metrics(path, 'm', ('fixed',))

    assert loaded == {BASELINE: classic, 'fixed': fixed}


def test_load_saved_metrics_names_the_missing_run(tmp_path):
    path = tmp_path / 'metrics.json'
    write_metrics(path, {'m_classic': {}})

    with pytest.raises(ValueError, match='m_recursive_chunk'):
        load_saved_metrics(path, 'm', ('recursive',))


def test_check_against_saved_metrics_accepts_matching_averages():
    scores = {BASELINE: arrays([0.5, 0.5])}
    expected = {BASELINE: {'NDCG@10': 0.25, 'Recall@10': 0.25}}

    check_against_saved_metrics(scores, 4, expected)  # no exception


def test_check_against_saved_metrics_names_the_mismatching_metric():
    scores = {BASELINE: arrays([0.5, 0.5])}
    expected = {BASELINE: {'NDCG@10': 0.30, 'Recall@10': 0.25}}

    with pytest.raises(ValueError, match='baseline NDCG@10'):
        check_against_saved_metrics(scores, 4, expected)


# --- statistics ----------------------------------------------------------

def test_paired_bootstrap_of_identical_scores_is_zero():
    scores = np.array([0.2, 0.5, 1.0])

    assert paired_bootstrap(scores, scores) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_of_a_constant_difference_has_a_point_interval():
    a, b = np.ones(5), np.full(5, 0.5)

    mean, low, high = paired_bootstrap(a, b)

    assert mean == pytest.approx(0.5)
    assert low == pytest.approx(0.5)
    assert high == pytest.approx(0.5)


def test_paired_bootstrap_is_positive_when_the_first_is_better():
    mean, _, _ = paired_bootstrap(np.array([1.0, 1.0]), np.array([0.0, 0.5]))

    assert mean > 0


def test_paired_bootstrap_interval_contains_the_mean_difference():
    rng = np.random.default_rng(1)
    a, b = rng.random(60), rng.random(60)

    mean, low, high = paired_bootstrap(a, b)

    assert low <= mean <= high


def test_paired_bootstrap_is_reproducible_for_a_seed():
    rng = np.random.default_rng(2)
    a, b = rng.random(40), rng.random(40)

    assert paired_bootstrap(a, b, seed=7) == paired_bootstrap(a, b, seed=7)


def test_paired_bootstrap_interval_narrows_with_more_queries():
    rng = np.random.default_rng(3)
    small_a, small_b = rng.random(20), rng.random(20)
    big_a, big_b = rng.random(800), rng.random(800)

    _, low_s, high_s = paired_bootstrap(small_a, small_b)
    _, low_b, high_b = paired_bootstrap(big_a, big_b)

    assert (high_b - low_b) < (high_s - low_s)


@pytest.mark.parametrize('wins, losses, expected', [
    (13, 4, 0.0490),   # the recursive vs baseline NDCG split of the README
    (3, 0, 0.25),
    (13, 7, 0.2632),
    (5, 5, 1.0),
    (0, 0, 1.0),
])
def test_sign_test_p_matches_hand_computed_values(wins, losses, expected):
    assert sign_test_p(wins, losses) == pytest.approx(expected, abs=1e-4)


def test_sign_test_p_is_symmetric():
    assert sign_test_p(13, 4) == sign_test_p(4, 13)


def test_sign_test_p_never_exceeds_one():
    assert all(sign_test_p(k, 10 - k) <= 1.0 for k in range(11))


# --- changed queries and comparisons -------------------------------------

def test_changed_queries_lists_only_queries_with_different_scores():
    a = np.array([1.0, 0.5, 0.0])
    b = np.array([1.0, 0.4, 0.0])

    assert changed_queries(a, b, ['q1', 'q2', 'q3']) == ['q2']


def test_changed_queries_ignores_floating_point_noise():
    assert changed_queries(
        np.array([0.5]), np.array([0.5 + 1e-12]), ['q1']
    ) == []


def test_comparison_pairs_put_every_strategy_against_the_baseline_first():
    assert comparison_pairs(['fixed', 'recursive']) == [
        ('fixed', BASELINE), ('recursive', BASELINE), ('recursive', 'fixed'),
    ]


def test_comparison_pairs_of_a_single_strategy_and_of_none():
    assert comparison_pairs(['fixed']) == [('fixed', BASELINE)]
    assert comparison_pairs([]) == []


def make_scores():
    return {
        BASELINE: arrays([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]),
        'fixed': arrays([1.0, 1.0, 0.5, 0.0], [1, 1, 0, 0]),
    }


def test_compare_strategies_reports_direction_wins_and_losses():
    comparisons = compare_strategies(
        make_scores(), ['q1', 'q2', 'q3', 'q4'], n_resamples=500
    )

    ndcg = next(c for c in comparisons if c.metric == 'NDCG@10')
    assert (ndcg.a, ndcg.b) == ('fixed', BASELINE)
    assert ndcg.mean_diff == pytest.approx(0.125)  # chunking is better
    assert (ndcg.wins, ndcg.losses) == (2, 1)
    assert ndcg.changed == ('q1', 'q2', 'q4')
    assert ndcg.p_sign == sign_test_p(2, 1)


def test_compare_strategies_reports_no_change_when_scores_are_equal():
    comparisons = compare_strategies(
        make_scores(), ['q1', 'q2', 'q3', 'q4'], n_resamples=500
    )

    recall = next(c for c in comparisons if c.metric == 'Recall@10')
    assert (recall.wins, recall.losses, recall.changed) == (0, 0, ())
    assert recall.p_sign == 1.0
    assert (recall.mean_diff, recall.low, recall.high) == (0.0, 0.0, 0.0)


def test_compare_strategies_yields_both_metrics_for_every_pair():
    scores = make_scores()
    scores['recursive'] = arrays([0.5, 0.5, 0.5, 0.5])

    comparisons = compare_strategies(scores, ['a', 'b', 'c', 'd'], 100)

    assert [(c.a, c.b, c.metric) for c in comparisons] == [
        ('fixed', BASELINE, 'NDCG@10'), ('fixed', BASELINE, 'Recall@10'),
        ('recursive', BASELINE, 'NDCG@10'),
        ('recursive', BASELINE, 'Recall@10'),
        ('recursive', 'fixed', 'NDCG@10'), ('recursive', 'fixed', 'Recall@10'),
    ]


def test_format_comparisons_shows_pair_metric_interval_and_p_value():
    comparison = Comparison(
        'fixed', BASELINE, 'NDCG@10', 0.125, -0.0248, 0.038, 13, 7, 0.2632,
        ('q1',),
    )

    lines = format_comparisons([comparison]).splitlines()

    assert lines[0].split()[:2] == ['comparison', 'metric']
    row = lines[1].split()
    assert row[:3] == ['fixed', '-', 'baseline']
    assert row[3:] == [
        'NDCG@10', '+0.1250', '[-0.0248,', '+0.0380]', '13/7', '0.263'
    ]
    assert 'sign test' in lines[-1]


# --- end to end ----------------------------------------------------------

@pytest.fixture
def benchmark(monkeypatch, tmp_path):
    """A finished fake benchmark run: cache on disk and its metrics file."""
    encoder = CountingEncoder()
    data = (DOCUMENTS, QUERIES, QRELS)
    for module in (pipeline, significance):
        monkeypatch.setattr(
            module, 'load_beir_dataset', lambda *args, **kwargs: data
        )
    monkeypatch.setattr(
        pipeline, 'get_spec', lambda name: ModelSpec(name, lambda: encoder)
    )
    monkeypatch.setattr(pipeline, 'print_embeddings_report', lambda path: None)

    config = BenchmarkConfig(
        dataset='tiny', subset_size=4, chunk_size=25, top_k=10,
        data_dir=tmp_path / 'data',
    )
    metrics = tmp_path / 'metrics.json'
    pipeline.run_benchmark(config, ['fake'], metrics)
    return SimpleNamespace(config=config, metrics=metrics, tmp=tmp_path)


def test_analysis_report_names_the_setup_and_every_comparison(benchmark):
    report = run_significance_analysis(
        'fake', benchmark.config, benchmark.metrics, n_resamples=200
    )

    assert 'fake on tiny (4 documents, train queries)' in report
    assert '3 of 3 queries have a relevant document' in report
    for pair in ('fixed - baseline', 'recursive - baseline',
                 'recursive - fixed'):
        assert pair in report


def test_analysis_follows_the_strategies_of_the_config(benchmark):
    config = BenchmarkConfig(
        dataset='tiny', subset_size=4, chunk_size=25, top_k=10,
        strategies=('fixed',), data_dir=benchmark.config.data_dir,
    )

    report = run_significance_analysis(
        'fake', config, benchmark.metrics, n_resamples=200
    )

    assert 'fixed - baseline' in report
    assert 'recursive' not in report


def test_analysis_refuses_metrics_that_do_not_match_the_cache(benchmark):
    runs = json.loads(benchmark.metrics.read_text())
    runs[0]['fake_classic']['NDCG@10'] = 0.5  # was 1.0
    benchmark.metrics.write_text(json.dumps(runs))

    with pytest.raises(ValueError, match='do not match'):
        run_significance_analysis(
            'fake', benchmark.config, benchmark.metrics, n_resamples=200
        )


def test_analysis_reports_missing_cache_files(benchmark):
    chunks_path(
        'tiny', 4, 'fake', 'fixed', 25, benchmark.config.data_dir
    ).unlink()

    with pytest.raises(ValueError, match='Missing cache files'):
        run_significance_analysis(
            'fake', benchmark.config, benchmark.metrics, n_resamples=200
        )


def test_analysis_reports_a_metrics_file_of_another_model(benchmark):
    cache = benchmark.config.data_dir / 'tiny-4'
    shutil.copytree(cache / 'fake', cache / 'other')  # cache, other metrics

    with pytest.raises(ValueError, match='other_classic'):
        run_significance_analysis(
            'other', benchmark.config, benchmark.metrics, n_resamples=200
        )


def test_analysis_without_active_queries_is_an_error(
        monkeypatch, benchmark
):
    monkeypatch.setattr(
        significance, 'load_beir_dataset',
        lambda *args, **kwargs: (DOCUMENTS, QUERIES, {'q1': {'gone': 1}}),
    )

    with pytest.raises(ValueError, match='No query'):
        run_significance_analysis(
            'fake', benchmark.config, benchmark.metrics, n_resamples=200
        )
