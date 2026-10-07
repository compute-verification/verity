"""circuit-check's bindings of `verity.ml.circuits.ops.boolean.gf2k`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _gf2k_roots() -> list[Any]:
    """GF(2^128) arithmetic (`verity.ml.circuits.ops.boolean.gf2k`): Gf128Mul and Gf256Mul are static-free; GfScale at L = 2, and GfResiduals
    on two residuals over 129 + 2 elements whose operands use every term kind (an element, a transposed block, a sum)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gf2k as G

    S = {"nw": 129, "nv": 2, "ops": [[["w", 0]], [["wt", 1, 3], ["v", 0]]],
         "res": [{"mul": [[0, 1]], "add": [["v", 1]]}, {"mul": [], "add": [["w", 0]]}]}
    return [
        bind(G.GfScale, L=2),
        bind(G.GfResiduals, S=S),
    ]
