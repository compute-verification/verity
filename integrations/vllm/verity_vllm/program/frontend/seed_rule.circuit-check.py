"""circuit-check's bindings of `verity_vllm.program.frontend.seed_rule`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """`SeedBlocks_v1` at a message of one SHA-512 block (a 16-byte coin, a 4-byte id) and of two (a 64-byte coin, a 16-byte
    id); `SeedSelect_v1`, `SeedEqual_v1` and `SeedWord_v1` are static-free."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.frontend import seed_rule as SR

    return [
        ("SeedBlocks_v1:16:4", lambda: bind(SR.SeedBlocks, COIN=16, RID=4)),
        ("SeedBlocks_v1:64:16", lambda: bind(SR.SeedBlocks, COIN=64, RID=16)),
        ("SeedSelect_v1", lambda: SR.SeedSelect),
        ("SeedEqual_v1", lambda: SR.SeedEqual),
        ("SeedWord_v1", lambda: SR.SeedWord),
    ]


def _by_calls_roots() -> list[Any]:
    """`ServedSeed_v1`, `SeedCheck_v1` and `SeedCheck_v2` at the same two sizes, checked by their calls: `SeedBlocks_v1`, a
    `Sha512Compress_v1` Call (a few hundred thousand gates) for each block but the last, then `SeedSelect_v1`, which compresses the
    last itself (and `SeedEqual_v1`); `SeedCheck_v2`'s are `SeedWord_v1` and then `SeedCheck_v1`'s."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.frontend import seed_rule as SR

    return [bind(d, COIN=c, RID=r) for d in (SR.ServedSeed, SR.SeedCheck, SR.SeedCheckWord) for c, r in ((16, 4), (64, 16))]
