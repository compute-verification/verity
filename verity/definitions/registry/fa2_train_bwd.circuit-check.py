"""circuit-check's bindings of `verity.definitions.registry.fa2_train_bwd`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """At D = 32, BM = 16 and BN = 32: a row's sums with four threads a row, and at D = 64 with eight; a row against a key block
    of which it sees 5 keys, and all; a key block's cleared accumulators; a key through a query block; a tile on the diagonal,
    and one a query block below it; a key after the loop; and a row's dQ from five parts in two buffers."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import fa2_train_bwd as W

    blocks = {"D": 32, "BM": 16, "BN": 32}
    return [
        ("Fa2TrainBwdPre_v1:1", lambda: bind(W.Fa2TrainBwdPre, D=32)),
        ("Fa2TrainBwdPre_v1:2", lambda: bind(W.Fa2TrainBwdPre, D=64)),
        ("Fa2TrainBwdRow_v1:1", lambda: bind(W.Fa2TrainBwdRow, D=32, BN=32, NVIS=5)),
        ("Fa2TrainBwdRow_v1:2", lambda: bind(W.Fa2TrainBwdRow, D=32, BN=32, NVIS=32)),
        ("Fa2TrainBwdClear_v1", lambda: bind(W.Fa2TrainBwdClear, D=32, BN=32)),
        ("Fa2TrainBwdKey_v1", lambda: bind(W.Fa2TrainBwdKey, D=32, BM=16)),
        ("Fa2TrainBwdTile_v1:1", lambda: bind(W.Fa2TrainBwdTile, OFF=0, **blocks)),
        ("Fa2TrainBwdTile_v1:2", lambda: bind(W.Fa2TrainBwdTile, OFF=16, **blocks)),
        ("Fa2TrainBwdDkDv_v1", lambda: bind(W.Fa2TrainBwdDkDv, D=32)),
        ("Fa2TrainBwdDq_v1", lambda: bind(W.Fa2TrainBwdDq, D=32, NP=5, NS=2)),
    ]


def _by_calls_roots() -> list[Any]:
    """At D = 32, BM = 16 and BN = 32, checked by its calls: a head of 64 rows, two key blocks, its dQ in two buffers."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import fa2_train_bwd as W

    return [bind(W.Fa2TrainBwdHead, T=64, D=32, BM=16, BN=32, NS=2)]
