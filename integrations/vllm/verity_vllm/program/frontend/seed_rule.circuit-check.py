"""circuit-check's bindings of `verity_vllm.program.frontend.seed_rule`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable

SIZES = ((16, 4), (64, 16))


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`SeedBlocks_v1` at a message of one SHA-512 block (a 16-byte coin, a 4-byte id) and of two (a 64-byte coin, a 16-byte
    id); `SeedSelect_v1` and `SeedEqual_v1` are static-free."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.frontend import seed_rule as SR

    return [(f"SeedBlocks_v1:{c}:{r}", lambda c=c, r=r: bind(SR.SeedBlocks, COIN=c, RID=r)) for c, r in SIZES] + [
        ("SeedSelect_v1", lambda: SR.SeedSelect), ("SeedEqual_v1", lambda: SR.SeedEqual)]


def _by_calls_roots() -> list[Any]:
    """`ServedSeed_v1` and `SeedCheck_v1` at the same two sizes, checked by their calls: `SeedBlocks_v1`, a
    `Sha512Compress_v1` Call (a few hundred thousand gates) a block, then `SeedSelect_v1` (and `SeedEqual_v1`)."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.frontend import seed_rule as SR

    return [bind(d, COIN=c, RID=r) for d in (SR.ServedSeed, SR.SeedCheck) for c, r in SIZES]
