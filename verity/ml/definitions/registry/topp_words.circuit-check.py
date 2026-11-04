"""circuit-check's bindings of `verity.ml.definitions.registry.topp_words`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """The seeded top-p token select over V = 16 at the split count S = 4 (the frontend's `token_select_stochastic_split`
    instance), which reaches its mask and keep word."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import topp_words as TW

    return [
        bind(TW.GumbelTopPTokenSelectV2, V=16, S=4),
    ]
