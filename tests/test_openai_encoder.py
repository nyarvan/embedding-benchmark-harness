"""Tests for OpenAIEncoder with a fake OpenAI client (no network)."""
from types import SimpleNamespace

import numpy as np
import pytest

from harness import openai_encoder
from harness.openai_encoder import OpenAIEncoder


class FakeEmbeddings:
    """Records requests; returns the vector [len(text), 1.0] per text.

    Items come back in reversed order, as a client must not rely on the
    response order. `drop_last` simulates an API that loses an item.
    """

    def __init__(self, drop_last: bool = False):
        self.requests = []
        self.client_kwargs = {}
        self.drop_last = drop_last

    def create(self, input, model):
        self.requests.append({'input': list(input), 'model': model})
        items = [
            SimpleNamespace(index=i, embedding=[float(len(text)), 1.0])
            for i, text in enumerate(input)
        ]
        if self.drop_last:
            items = items[:-1]
        return SimpleNamespace(data=list(reversed(items)))


def install_fake_client(monkeypatch, fake: FakeEmbeddings) -> None:
    def fake_openai(**kwargs):
        fake.client_kwargs = kwargs
        return SimpleNamespace(embeddings=fake)

    monkeypatch.setattr(openai_encoder.openai, 'OpenAI', fake_openai)
    monkeypatch.setattr(openai_encoder, 'load_dotenv', lambda: None)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')


@pytest.fixture
def fake(monkeypatch):
    embeddings = FakeEmbeddings()
    install_fake_client(monkeypatch, embeddings)
    return embeddings


TEXTS = ['a' * n for n in range(1, 9)]


def test_encode_splits_inputs_into_batches_without_resending_tail(fake):
    OpenAIEncoder(batch_size=3).encode(TEXTS)

    sizes = [len(request['input']) for request in fake.requests]
    sent = [text for request in fake.requests for text in request['input']]

    assert sizes == [3, 3, 2]
    assert sent == TEXTS


def test_encode_keeps_input_order_even_if_api_reverses_items(fake):
    vectors = OpenAIEncoder(batch_size=3).encode(TEXTS)

    assert vectors[:, 0].tolist() == [float(len(text)) for text in TEXTS]


def test_encode_returns_float32_matrix(fake):
    vectors = OpenAIEncoder(batch_size=3).encode(TEXTS)

    assert vectors.shape == (len(TEXTS), 2)
    assert vectors.dtype == np.float32


def test_encode_empty_input_sends_no_requests(fake):
    vectors = OpenAIEncoder().encode([])

    assert len(vectors) == 0
    assert fake.requests == []


def test_encode_ignores_sentence_transformers_kwargs(fake):
    vectors = OpenAIEncoder().encode(
        TEXTS, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    )

    assert len(vectors) == len(TEXTS)
    assert len(fake.requests) == 1  # the encoder's own batch_size (100) wins


def test_encode_passes_model_name_to_api(fake):
    OpenAIEncoder(model='text-embedding-3-large').encode(['x'])

    assert fake.requests[0]['model'] == 'text-embedding-3-large'


def test_client_is_created_with_retries(fake):
    OpenAIEncoder()

    assert fake.client_kwargs['max_retries'] > 0


def test_encode_raises_when_api_returns_fewer_embeddings(monkeypatch):
    install_fake_client(monkeypatch, FakeEmbeddings(drop_last=True))

    with pytest.raises(ValueError):
        OpenAIEncoder(batch_size=3).encode(TEXTS)


def test_missing_api_key_raises_a_clear_error_before_a_client_is_built(
        monkeypatch
):
    fake = FakeEmbeddings()
    install_fake_client(monkeypatch, fake)
    monkeypatch.delenv('OPENAI_API_KEY')

    with pytest.raises(ValueError, match='OPENAI_API_KEY'):
        OpenAIEncoder()

    assert fake.client_kwargs == {}  # the client was never constructed
