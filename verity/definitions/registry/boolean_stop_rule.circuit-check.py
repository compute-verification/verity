"""circuit-check's bindings of `verity.definitions.registry.boolean_stop_rule`'s `StopRule_v2` and `StopRule16_v2`, in
`circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`StopRule_v2` at small classes (4 outputs and one stop id, 5 outputs and two) and at GLM-5.3's served class (32
    outputs, its three EOS ids), and `StopRule16_v2` at the two small classes and at 1,024 outputs with three stop ids, a
    binding over `targets.MAX_GATES` that its ``digests`` pin keeps in ``--all``."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_stop_rule as BS

    return [("StopRule_v2:4:1", lambda: bind(BS.StopRuleV2, T=4, NSTOP=1)),
            ("StopRule_v2:5:2", lambda: bind(BS.StopRuleV2, T=5, NSTOP=2)),
            ("StopRule_v2:32:3", lambda: bind(BS.StopRuleV2, T=32, NSTOP=3)),
            ("StopRule16_v2:4:1", lambda: bind(BS.StopRule16V2, T=4, NSTOP=1)),
            ("StopRule16_v2:5:2", lambda: bind(BS.StopRule16V2, T=5, NSTOP=2)),
            ("StopRule16_v2:1024:3", lambda: bind(BS.StopRule16V2, T=1024, NSTOP=3))]
