"""Tests for evaluate(): metrics on tiny cases that can be computed by hand."""
from math import log2

import pytest

from harness.evaluate import evaluate

QRELS = {'q1': {'d1': 1}}
# BEIR rounds metrics to 5 decimals
TOLERANCE = 1e-5
ZERO_METRICS = {'P@5': 0.0, 'Recall@10': 0.0, 'NDCG@10': 0.0}


def ranked(*doc_ids: str) -> dict[str, float]:
    """Scores for `doc_ids`, descending in the given order."""
    return {doc_id: 1.0 - i * 0.01 for i, doc_id in enumerate(doc_ids)}


def test_evaluate_returns_the_three_reported_metrics():
    results = {'q1': ranked('d1', 'd2')}

    eval_result = evaluate(QRELS, results)

    assert set(eval_result) == {'P@5', 'Recall@10', 'NDCG@10'}


def test_evaluate_relevant_document_on_first_place_gives_ideal_ndcg():
    results = {'q1': ranked('d1', 'd2')}

    eval_result = evaluate(QRELS, results)

    # One relevant document in the top 5: precision 1/5
    assert eval_result == pytest.approx(
        {'P@5': 0.2, 'Recall@10': 1.0, 'NDCG@10': 1.0}, abs=TOLERANCE
    )


def test_evaluate_relevant_document_on_second_place_discounts_ndcg():
    results = {'q1': ranked('d2', 'd1')}

    eval_result = evaluate(QRELS, results)

    assert eval_result['NDCG@10'] == pytest.approx(1 / log2(3), abs=TOLERANCE)
    assert eval_result['Recall@10'] == 1.0


def test_evaluate_ranks_by_score_not_by_dict_order():
    results = {'q1': {'d2': 0.1, 'd1': 0.9}}

    eval_result = evaluate(QRELS, results)

    assert eval_result['NDCG@10'] == pytest.approx(1.0, abs=TOLERANCE)


def test_evaluate_relevant_document_below_top_5_has_zero_precision_at_5():
    results = {'q1': ranked('d2', 'd3', 'd4', 'd5', 'd6', 'd1')}

    eval_result = evaluate(QRELS, results)

    assert eval_result['P@5'] == 0.0
    assert eval_result['Recall@10'] == 1.0
    assert eval_result['NDCG@10'] == pytest.approx(1 / log2(7), abs=TOLERANCE)


def test_evaluate_relevant_document_below_top_10_is_not_counted():
    others = [f'x{i}' for i in range(10)]
    results = {'q1': ranked(*others, 'd1')}

    eval_result = evaluate(QRELS, results)

    assert eval_result == ZERO_METRICS


def test_evaluate_relevant_document_not_retrieved_gives_zero_metrics():
    results = {'q1': ranked('d2')}

    eval_result = evaluate(QRELS, results)

    assert eval_result == ZERO_METRICS


def test_evaluate_averages_metrics_over_queries():
    qrels = {'q1': {'d1': 1}, 'q2': {'d1': 1}}
    results = {
        'q1': ranked('d1', 'd2'),  # found on the first place
        'q2': ranked('d2'),        # relevant document not retrieved
    }

    eval_result = evaluate(qrels, results)

    assert eval_result == pytest.approx(
        {'P@5': 0.1, 'Recall@10': 0.5, 'NDCG@10': 0.5}, abs=TOLERANCE
    )


def test_evaluate_query_with_empty_results_counts_as_zero():
    qrels = {'q1': {'d1': 1}, 'q2': {'d1': 1}}
    results = {'q1': ranked('d1', 'd2'), 'q2': {}}

    eval_result = evaluate(qrels, results)

    assert eval_result == pytest.approx(
        {'P@5': 0.1, 'Recall@10': 0.5, 'NDCG@10': 0.5}, abs=TOLERANCE
    )
