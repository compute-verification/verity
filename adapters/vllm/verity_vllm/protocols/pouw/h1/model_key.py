"""The auditor's registration of the served model, as the gate's ``model-key`` answers it (`ModelKey`, `model_key`): the key a
window's ``b/<K>`` rows are committed under, and the weights' root ``period-open --model M`` signs."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelKey:
    """The auditor's registration of the served model under the period's circuit, as the gate's ``model-key --model M
    --circuit C`` answers a lane's gate key: its role ``model``, ``circuit``, the weights' root ``weights_root`` (hex) that
    ``period-open --model M`` signs, and ``key``, the 32-byte salt key its ``b/<K>`` rows were committed under."""

    model: str
    circuit: str
    weights_root: str
    key: bytes = field(repr=False)


def _hex512(v: Any) -> bool:
    return type(v) is str and len(v) == 128 and all(c in "0123456789abcdef" for c in v)


def model_key(answer: Mapping[str, Any]) -> ModelKey:
    """The gate's ``model-key`` answer as a `ModelKey`; ValueError on a refusal, or an answer whose key isn't 32 bytes
    whose SHA-512 is its ``key_sha512``, or whose circuit or weights' root isn't a SHA-512 digest."""
    if "refused" in answer:
        raise ValueError(f"model-key: refused: {answer['refused']}")
    try:
        key = bytes.fromhex(answer["key"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("model-key: the answer has no key in hex") from None
    if len(key) != 32 or hashlib.sha512(key).hexdigest() != answer.get("key_sha512"):
        raise ValueError("model-key: the key is not 32 bytes whose SHA-512 is the answer's key_sha512")
    if not (type(answer.get("model")) is str and _hex512(answer.get("circuit")) and _hex512(answer.get("weights_root"))):
        raise ValueError("model-key: the answer names no model, circuit and weights' root (SHA-512 digests in hex)")
    return ModelKey(answer["model"], answer["circuit"], answer["weights_root"], key)
