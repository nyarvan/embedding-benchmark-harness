from harness.aggregate import aggregate_to_documents


def test_aggregate_to_documents():
    chunk_results = {
        'q1': {
            '2613775_3': 0.82, '1263446_2': 0.79,
            '2613775_7': 0.75, '9999_0': 0.66
        },
        'q2': {'9999_0': 0.90, '2613775_3': 0.40, '2613775_1': 0.50},
    }

    result = aggregate_to_documents(chunk_results, 2)

    assert result == {
        'q1': {'2613775': 0.82, '1263446': 0.79},
        'q2': {'9999': 0.90, '2613775': 0.50},
    }
    assert list(result['q1']) == ['2613775', '1263446']
    assert list(result['q2']) == ['9999', '2613775']
