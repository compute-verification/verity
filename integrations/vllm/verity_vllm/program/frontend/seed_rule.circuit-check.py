"""circuit-check's bindings of `verity_vllm.program.frontend.seed_rule`'s `ServedSeed_v1` and `SeedCheck_v1`, in
`circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """Each at a message of one SHA-512 block (a 16-byte coin, a 4-byte id) and of two (a 64-byte coin, a 16-byte id)."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.frontend import seed_rule as SR

    return [("ServedSeed_v1:16:4", lambda: bind(SR.ServedSeed, COIN=16, RID=4)),
            ("ServedSeed_v1:64:16", lambda: bind(SR.ServedSeed, COIN=64, RID=16)),
            ("SeedCheck_v1:16:4", lambda: bind(SR.SeedCheck, COIN=16, RID=4)),
            ("SeedCheck_v1:64:16", lambda: bind(SR.SeedCheck, COIN=64, RID=16))]
