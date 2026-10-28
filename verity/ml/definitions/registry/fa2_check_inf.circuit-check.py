"""circuit-check's bindings of `verity.ml.definitions.registry.fa2_check_inf`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics: a key block's softmax step at its first block of 24 keys (groups of 16 and
    8) and below it guarded and not over 32 keys, and at LAST (no running max out) a head's one block of 24 keys and its block 0
    unguarded, and its P·V group from +0 at 8 keys (padded to a k-step) and over 32 keys, a key's value 4 slices of 4 columns,
    also at Ampere's k16 step."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import b1, fa2_check_inf, hopper

    dot = hopper.HopperBF16WgmmaDot16
    softmax = {"D": 16, "DOT": dot, "PK": 16}
    step = lambda **kw: lambda: bind(fa2_check_inf.AttnBlockSoftmaxV1, **softmax, **kw)  # noqa: E731
    return [
        ("Attention_v11", lambda: bind(fa2_check_inf.AttentionV11, T=17, NH=1, KVH=1, D=16, BN=16, MASKED_FROM=1,
                                       DOT=hopper.HopperBF16WgmmaDot16, INV=b1.P.Fa2InvSum)),
        ("AttnBlockSoftmax_v1-first", step(NVIS=24, FIRST=True, CHECK=True, LAST=False)),
        ("AttnBlockSoftmax_v1", step(NVIS=32, FIRST=False, CHECK=True, LAST=False)),
        ("AttnBlockSoftmax_v1-unguarded", step(NVIS=32, FIRST=False, CHECK=False, LAST=False)),
        ("AttnBlockSoftmax_v1-first-last", step(NVIS=24, FIRST=True, CHECK=True, LAST=True)),
        ("AttnBlockSoftmax_v1-last", step(NVIS=32, FIRST=False, CHECK=False, LAST=True)),
        ("AttnBlockPv_v1-first", lambda: bind(fa2_check_inf.AttnBlockPvV1, D=16, N=8, VS=4, DOT=dot, FIRST=True)),
        ("AttnBlockPv_v1", lambda: bind(fa2_check_inf.AttnBlockPvV1, D=16, N=32, VS=4, DOT=dot, FIRST=False)),
        ("AttnBlockPv_v1-ampere", lambda: bind(fa2_check_inf.AttnBlockPvV1, D=16, N=8, VS=4, DOT=b1.P.AmpereBF16TcDot16, FIRST=True)),
    ]
