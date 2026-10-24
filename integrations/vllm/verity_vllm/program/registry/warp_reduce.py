"""Lane 0's value of a warp's xor-butterfly reduction (`v = v + shfl_xor(v, m)` for each mask in turn).

On the GPU every lane adds in every round, but a circuit reads only lane 0's sum, so it computes only the adds lane 0's value
depends on: at masks 16, 8, 4, 2, 1 over 32 lanes, 31 of the 160.
"""
from __future__ import annotations

from verity.ml.circuits.defs import Coll


def lane0_butterfly(B, vals: list[Coll], masks: tuple[int, ...], add) -> Coll:
    """Lane 0's value after the rounds `v[i] = add(v[i], v[i ^ m])` (a lane whose peer `i ^ m` is past the last keeps its value), each
    round computed on the lanes the later rounds read for lane 0: those an xor of the later masks reaches."""
    cur = dict(enumerate(vals))
    for k, m in enumerate(masks):
        reads = {0}
        for r in masks[k + 1:]:
            reads |= {i ^ r for i in reads}
        cur = {i: B.call(add, cur[i], cur[i ^ m]) if (i ^ m) in cur else cur[i] for i in sorted(reads) if i in cur}
    return cur[0]


__all__ = ["lane0_butterfly"]
