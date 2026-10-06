"""circuit-check's bindings of `verity_flock.rec_residuals`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _rec_roots() -> list[Any]:
    """V*'s algebra (`verity_flock.rec_residuals`) on two row ports (an element alone, a transposed block) and v, under
    `_gf2k_roots`' GfResiduals: a rep's is the same Calls over more ports and products."""
    from verity_flock import rec_residuals as RR

    S = {"nw": 129, "nv": 2, "ops": [[["w", 0]], [["wt", 1, 3], ["v", 0]]],
         "res": [{"mul": [[0, 1]], "add": [["v", 1]]}, {"mul": [], "add": [["w", 0]]}]}
    return [
        RR.definition({"gf": S, "ports": [["m0", 1], ["m1", 128]]}),
    ]
