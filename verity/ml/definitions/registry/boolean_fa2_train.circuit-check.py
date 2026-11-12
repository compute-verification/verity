"""circuit-check's bindings of `verity.ml.definitions.registry.boolean_fa2_train`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """At `fa2_train`'s sidecar's statics: a diagonal block of 5 visible keys, a full block, a row over those two blocks, a split
    of 5 visible keys, the combine and a row over two of four splits, at D = 16 and BN = 16."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_fa2_train as BX

    return [bind(BX.Fa2TrainFirst, D=16, BN=16, NVIS=5), bind(BX.Fa2TrainBlock, D=16, BN=16),
            bind(BX.Fa2TrainRow, D=16, BN=16, NB=2, NVIS=5), bind(BX.Fa2TrainPart, D=16, BN=16, NVIS=5),
            bind(BX.Fa2TrainCombine, D=16), bind(BX.Fa2TrainSplitRow, D=16, BN=16, NS=2, NVIS=5)]


def _by_calls_roots() -> list[Any]:
    """The heads at `fa2_train`'s sidecar's statics, checked by their calls."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_fa2_train as BX

    return [bind(BX.Fa2TrainHead, T=16, D=16, BN=16), bind(BX.Fa2TrainSplitHead, T=64, D=16, BN=16)]
