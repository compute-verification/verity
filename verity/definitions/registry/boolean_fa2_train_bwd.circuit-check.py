"""circuit-check's bindings of `verity.definitions.registry.boolean_fa2_train_bwd`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """At `fa2_train_bwd`'s sidecar's statics: a row's sums with four threads, a row against a key block of which it sees 5
    keys, a key through its first query block and through a later one, a key after the loop and a row's dQ from five parts in
    two buffers."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_fa2_train_bwd as BX

    return [bind(BX.Fa2TrainBwdPre, D=32), bind(BX.Fa2TrainBwdRow, D=32, BN=32, NVIS=5), bind(BX.Fa2TrainBwdKeyFirst, D=32, BM=16),
            bind(BX.Fa2TrainBwdKey, D=32, BM=16), bind(BX.Fa2TrainBwdDkDv, D=32), bind(BX.Fa2TrainBwdDq, D=32, NP=5, NS=2)]


def _by_calls_roots() -> list[Any]:
    """The head at `fa2_train_bwd`'s sidecar's statics, checked by its calls."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_fa2_train_bwd as BX

    return [bind(BX.Fa2TrainBwdHead, T=64, D=32, BM=16, BN=32, NS=2)]
