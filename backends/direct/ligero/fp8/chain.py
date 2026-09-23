"""GemmCoordinate<K=1536> over ``sm89.mma.m16n8k32.e4m3`` on the B-Ligero chain prover (``--relation fp8-ada``).

Since lane b-hopper this is ``relchain.py`` at ``RELATIONS["fp8-ada"]``: the runner, the gate, the bench and the
synthetic instances are the generic ones (the FP8 unit itself is ``relation.py``, parametrised by ``FP8Params``).
The names below are kept for the fp8-proof lane's tests and tool declaration.

    python -m backends.direct.ligero.run --relation fp8-ada gate-vu  --device cuda --vus 64
    python -m backends.direct.ligero.run --relation fp8-ada bench-vu --device cuda --batch 4096 --total-vus 4096 --reps 3 --out result.json
"""
from __future__ import annotations

from .. import protocol as protocol_mod
from ..relations import RELATIONS
from ..relchain import RelationChainRunner, bench_vu_rel, gate_vu_rel
from ..relchain import instances as _instances
from ..relchain import instances_digest as _instances_digest
from ..relchain import vu_units as _vu_units

__all__ = ["FP8_HOOKS", "FP8ChainRunner", "RELATION_NAME", "MODEL_NAME", "bench_vu_fp8", "gate_vu_fp8", "instances_fp8"]

ADA_REL = RELATIONS["fp8-ada"]
RELATION_NAME = ADA_REL.relation_name
MODEL_NAME = ADA_REL.model_name
FP8_HOOKS = ADA_REL.hooks
INSTANCE_SEED = ADA_REL.instance_seed


def instances_fp8(n: int, seed: int = INSTANCE_SEED) -> list[tuple[list, list, list, int]]:
    """``n`` VUs: (a (1536,), b (1536,), accs (48,), y_final) with the accumulators recorded from the model."""
    return _instances(ADA_REL, n, seed=seed)


def instances_digest(n: int, seed: int = INSTANCE_SEED) -> str:
    return _instances_digest(ADA_REL, n, seed=seed)


def vu_units(a: list[int], b: list[int], accs: list[int]) -> list[tuple]:
    return _vu_units(ADA_REL, a, b, accs)


class FP8ChainRunner(RelationChainRunner):
    """``RelationChainRunner`` for the Ada FP8 relation."""

    def __init__(self, device: str, target: float, rate_log2: int = 2, n_proofs: int = 1, zk: bool = False,
                 mode: str = protocol_mod.MODE_INTERACTIVE):
        super().__init__(ADA_REL, device, target, rate_log2, n_proofs=n_proofs, zk=zk, mode=mode)


def gate_vu_fp8(args) -> int:
    return gate_vu_rel(args, ADA_REL)


def bench_vu_fp8(args) -> int:
    return bench_vu_rel(args, ADA_REL)
