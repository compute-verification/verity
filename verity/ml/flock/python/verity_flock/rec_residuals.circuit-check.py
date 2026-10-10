"""circuit-check's bindings of `verity_flock.rec_residuals`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _rec_roots() -> list[Any]:
    """V*'s algebra (`verity_flock.rec_residuals`) on two row ports (an element alone, a transposed block) and v, under
    `_gf2k_roots`' GfResiduals: a rep's is the same Calls over more ports and products; and a holo shape's lincheck L4′ in
    small, ``Σ_l (comb_l + h_l) z_partial_l + claim`` over two lanes, z_partial and h each a row port, the comb and the claim in
    v."""
    from verity_flock import rec_residuals as RR

    S = {"nw": 129, "nv": 2, "ops": [[["w", 0]], [["wt", 1, 3], ["v", 0]]],
         "res": [{"mul": [[0, 1]], "add": [["v", 1]]}, {"mul": [], "add": [["w", 0]]}]}
    L4 = {"nw": 4, "nv": 3, "ops": [[["v", 0], ["w", 2]], [["w", 0]], [["v", 1], ["w", 3]], [["w", 1]]],
          "res": [{"mul": [[0, 1], [2, 3]], "add": [["v", 2]]}]}
    return [
        RR.definition({"gf": S, "ports": [["m0", 1], ["m1", 128]]}),
        RR.definition({"gf": L4, "ports": [["m0", 2], ["m1", 2]]}),
    ]
