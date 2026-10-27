"""A tiny dataset and a fake encoder shared by the pipeline-level tests."""
import numpy as np

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
