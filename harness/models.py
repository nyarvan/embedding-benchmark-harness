"""Registry of the embedding models the harness can run.

Models are described, not built: `ModelSpec.create` builds the model when
called, so importing this module neither loads weights nor needs an API key,
and a run with one model never imports the libraries of the others.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type-only: harness.embed imports sentence_transformers at import time
    from harness.embed import Encoder


@dataclass(frozen=True)
class ModelSpec:
    """How to build one model and whether using it costs money."""

    name: str
    create: Callable[[], Encoder]
    paid: bool = False  # calls a paid API


def _sentence_transformer(model_id: str) -> Callable[[], Encoder]:
    def create() -> Encoder:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_id)

    return create


def _openai(model_id: str) -> Callable[[], Encoder]:
    def create() -> Encoder:
        from harness.openai_encoder import OpenAIEncoder
        return OpenAIEncoder(model_id)

    return create


MODELS: dict[str, ModelSpec] = {
    'bge_m3': ModelSpec('bge_m3', _sentence_transformer('BAAI/bge-m3')),
    'e5_large_v2': ModelSpec(
        'e5_large_v2', _sentence_transformer('intfloat/e5-large-v2')
    ),
    'openai_3_small': ModelSpec(
        'openai_3_small', _openai('text-embedding-3-small'), paid=True
    ),
}


def get_spec(name: str) -> ModelSpec:
    """Return the spec registered under `name`, or raise ValueError."""
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(
            f'Unknown model {name!r}. Available: {", ".join(MODELS)}'
        ) from None


def available_models() -> list[str]:
    """Names of all registered models."""
    return list(MODELS)
