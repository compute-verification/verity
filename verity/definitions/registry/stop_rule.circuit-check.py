"""circuit-check's bindings of `verity.definitions.registry.stop_rule`'s `StopRule_v1`, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`StopRule_v1` at small classes (4 outputs and one stop id, 5 outputs and two), at GLM-5.3's served class (32
    outputs, its three EOS ids), and at 1,024 outputs with those stop ids: the class of a response of 513 to 1,024 ids
    where a request's outputs are classed by the least power of two that holds them."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import stop_rule as X

    return [("StopRule_v1:4:1", lambda: bind(X.StopRule, T=4, NSTOP=1)),
            ("StopRule_v1:5:2", lambda: bind(X.StopRule, T=5, NSTOP=2)),
            ("StopRule_v1:32:3", lambda: bind(X.StopRule, T=32, NSTOP=3)),
            ("StopRule_v1:1024:3", lambda: bind(X.StopRule, T=1024, NSTOP=3))]
