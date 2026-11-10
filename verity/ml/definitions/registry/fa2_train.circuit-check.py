"""circuit-check's bindings of `verity.ml.definitions.registry.fa2_train`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """At D = 16 and BN = 16: a diagonal block of 5 visible keys (its fourth lane empty), a full block after it, a row over those
    two blocks, a split of 5 visible keys, the combine of four splits and a row over two of four splits (two empty).  A row's
    body computes its end itself, so it is a target of its own, which the heads' calls rely on."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import fa2_train as W

    return [
        ("Fa2TrainFirst_v1", lambda: bind(W.Fa2TrainFirst, D=16, BN=16, NVIS=5)),
        ("Fa2TrainBlock_v1", lambda: bind(W.Fa2TrainBlock, D=16, BN=16)),
        ("Fa2TrainRow_v1", lambda: bind(W.Fa2TrainRow, D=16, BN=16, NB=2, NVIS=5)),
        ("Fa2TrainPart_v1", lambda: bind(W.Fa2TrainPart, D=16, BN=16, NVIS=5)),
        ("Fa2TrainCombine_v1", lambda: bind(W.Fa2TrainCombine, D=16)),
        ("Fa2TrainSplitRow_v1", lambda: bind(W.Fa2TrainSplitRow, D=16, BN=16, NS=2, NVIS=5)),
    ]


def _by_calls_roots() -> list[Any]:
    """At D = 16 and BN = 16, checked by their calls: a head of 16 rows (one block) and a split head of 64 rows."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import fa2_train as W

    return [bind(W.Fa2TrainHead, T=16, D=16, BN=16), bind(W.Fa2TrainSplitHead, T=64, D=16, BN=16)]
