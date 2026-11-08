"""circuit-check's bindings of `verity_vllm.program.frontend.seed_rule`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _by_calls_roots() -> list[Any]:
    """`ServedSeed_v1` and `SeedCheck_v1` at a message of one SHA-512 block (a 16-byte coin, a 4-byte id) and of two (a 64-byte
    coin, a 16-byte id), checked by their calls: each is `Sha512Compress_v1` Calls, a few hundred thousand gates a block, then
    `SeedSelect_v1` (and `SeedEqual_v1`), which are static-free and built whole."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.frontend import seed_rule as SR

    return [bind(d, COIN=coin, RID=rid) for d in (SR.ServedSeed, SR.SeedCheck) for coin, rid in ((16, 4), (64, 16))]
