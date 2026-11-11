"""circuit-check's bindings of `verity.definitions.registry.stop_rule`'s `StopRule_v1` and `StopRule16_v1`, in
`circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`StopRule_v1` at small classes (4 outputs and one stop id, 5 outputs and two) and at GLM-5.3's served class (32
    outputs, its three EOS ids), and `StopRule16_v1` at the two small classes."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import stop_rule as X

    return [("StopRule_v1:4:1", lambda: bind(X.StopRule, T=4, NSTOP=1)),
            ("StopRule_v1:5:2", lambda: bind(X.StopRule, T=5, NSTOP=2)),
            ("StopRule_v1:32:3", lambda: bind(X.StopRule, T=32, NSTOP=3)),
            ("StopRule16_v1:4:1", lambda: bind(X.StopRule16, T=4, NSTOP=1)),
            ("StopRule16_v1:5:2", lambda: bind(X.StopRule16, T=5, NSTOP=2))]
