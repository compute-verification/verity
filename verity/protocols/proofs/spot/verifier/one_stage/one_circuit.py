"""The one circuit at the served verifier's door: C_S, which the door derives from C (the registered Program), the
partition and the block's units (`restrict`), and the holes it does not yet derive, each refusing by name
(`NotDerived`)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from verity.circuits import partition_object as PO
from verity.circuits import restrict as R

PREFIX = "one-circuit: not derived yet: "

RESTRICT = "restrict : C -> Partition -> Block -> Circuit (C_S, the lowering of the block's Definition in C)"

LOWER = "lower : C_S -> the base argument's circuit (Flock's verifier takes a Boolean block circuit and lowers it to rows)"

Refused = R.Refused


class NotDerived(ValueError):
    """A derivation from C and the partition that is not written yet (`Flock.OneCircuit.notDerived`)."""

    def __init__(self, what: str) -> None:
        super().__init__(PREFIX + what)


def block(c: bytes, partition: Mapping[str, Any], k: int) -> list[tuple[int, int]]:
    """Proof block ``k``'s units, the ``k``-th pair of the partition object's ``blocks``, which cover every unit of C's
    population exactly once (`partition_object.blocks`); `Refused` where the object is not valid, names no blocks, its
    blocks end short of or past the population, or has no block ``k``."""
    v = PO.validate(partition)
    if not v.ok:
        raise Refused("E_PARTITION", f"the partition object is not valid: {list(v.codes)}")
    bs = PO.blocks(partition)
    if bs is None:
        raise Refused("E_BLOCKS", "the partition object names no blocks")
    try:
        population = PO.TemplatePartition(R.program(c), partition["query"]).population
    except PO.QueryInapplicable as e:
        raise Refused("E_QUERY", str(e)) from e
    if bs[-1][1] != population:
        raise Refused("E_BLOCKS", f"the partition's blocks end at {bs[-1][1]}, not at C's population {population}")
    if not (type(k) is int and 0 <= k < len(bs)):
        raise Refused("E_BLOCKS", f"no block {k!r} among the partition's {len(bs)}")
    return [bs[k]]


def restrict(c: bytes, partition: Mapping[str, Any], units: Sequence[tuple[int, int]]) -> bytes:
    """C_S (`OneStageZk.restrict`): the descriptor bytes of C compacted to the block of ``units`` (ascending ``[lo, hi)``
    ranges of the partition's units), Core's `Circuit.restrict` (`restrict.restrict`); `Refused` where C, the partition
    or the units give no block."""
    return R.restrict(c, partition, [tuple(u) for u in units]).descriptor


def lower(c_s: bytes) -> bytes:
    """The base argument's circuit of C_S, the Boolean block circuit (`restrict`): the lowering to rows is the base
    argument's, and Flock's verifier takes no Boolean block circuit yet."""
    raise NotDerived(LOWER)
