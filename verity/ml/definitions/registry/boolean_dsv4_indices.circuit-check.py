"""circuit-check's bindings of `verity.ml.definitions.registry.boolean_dsv4_indices`' Definition, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definition at ratio 128's narrowest row (`active_topk_width` 128, window 128) and at a small one (top-k 8,
    ratio 4, window 4, row 16); the served ratio-4 row is over `targets.MAX_GATES`, so it is the word Definition's
    (`dsv4_indices.circuit-check.py`)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_dsv4_indices as BX

    return [
        ("Dsv4CombineTopkSwa_v2:1", lambda: bind(BX.Dsv4CombineTopkSwaV2, TOPK=128, RATIO=128, WINDOW=128, WIDTH=256)),
        ("Dsv4CombineTopkSwa_v2:2", lambda: bind(BX.Dsv4CombineTopkSwaV2, TOPK=8, RATIO=4, WINDOW=4, WIDTH=16)),
    ]
