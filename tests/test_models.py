"""Tests for the model registry (no model is loaded, no API is called)."""
import subprocess
import sys
from types import SimpleNamespace

import pytest

from harness.models import MODELS, available_models, get_spec

EXPECTED_MODELS = ['bge_m3', 'e5_large_v2', 'openai_3_small']


class FakeEncoder:
    """Records the model id it was built with."""

    def __init__(self, model_id):
        self.model_id = model_id


def test_available_models_lists_the_three_registered_models():
    assert sorted(available_models()) == EXPECTED_MODELS


def test_get_spec_returns_the_spec_registered_under_the_name():
    spec = get_spec('bge_m3')

    assert spec is MODELS['bge_m3']
    assert spec.name == 'bge_m3'


def test_get_spec_rejects_unknown_name_and_lists_available_models():
    with pytest.raises(ValueError) as error:
        get_spec('no_such_model')

    message = str(error.value)
    assert 'no_such_model' in message
    assert all(name in message for name in EXPECTED_MODELS)


def test_get_spec_is_case_sensitive():
    with pytest.raises(ValueError):
        get_spec('BGE_M3')


def test_spec_names_match_their_registry_keys():
    assert all(spec.name == key for key, spec in MODELS.items())


def test_only_the_openai_model_is_marked_as_paid():
    paid = [name for name, spec in MODELS.items() if spec.paid]

    assert paid == ['openai_3_small']


def test_importing_the_registry_does_not_load_model_libraries():
    # A fresh interpreter: in the pytest process these modules are loaded
    code = (
        'import sys, harness.models; '
        'assert "sentence_transformers" not in sys.modules; '
        'assert "openai" not in sys.modules'
    )

    result = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('name, model_id', [
    ('bge_m3', 'BAAI/bge-m3'),
    ('e5_large_v2', 'intfloat/e5-large-v2'),
])
def test_sentence_transformer_models_are_built_from_their_hub_id(
        monkeypatch, name, model_id
):
    monkeypatch.setattr('sentence_transformers.SentenceTransformer', FakeEncoder)

    model = get_spec(name).create()

    assert model.model_id == model_id


def test_openai_model_is_built_with_the_embedding_model_name(monkeypatch):
    monkeypatch.setattr('harness.openai_encoder.OpenAIEncoder', FakeEncoder)

    model = get_spec('openai_3_small').create()

    assert model.model_id == 'text-embedding-3-small'


def test_creating_one_model_does_not_build_the_others(monkeypatch):
    built = SimpleNamespace(count=0)

    class CountingEncoder(FakeEncoder):
        def __init__(self, model_id):
            super().__init__(model_id)
            built.count += 1

    monkeypatch.setattr(
        'sentence_transformers.SentenceTransformer', CountingEncoder
    )
    monkeypatch.setattr('harness.openai_encoder.OpenAIEncoder', CountingEncoder)

    get_spec('bge_m3').create()

    assert built.count == 1
