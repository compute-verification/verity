"""circuit-check's bindings of `verity.definitions.registry.torch_words`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _core_roots() -> list[Any]:
    from verity.definitions.registry import torch_words as W

    return list(W.TORCH_WORDS)
