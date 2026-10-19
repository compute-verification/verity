"""CA's H1V3 itself on a few J10-banded units: h1v3_ref.py's `h1tile.reference(..., stream=True)` from the tree on
PYTHONPATH, with one change: the tree's own `noise` taken twice, each unit's rows from its band leader's seed_j and its
columns from its own (the band-leader map is kernels' reading of SharedNoise.lean, j10b_ref.py; not CA's reference). So
the banded kernels' z, fold and receipts are held to CA's code everywhere but that map.

    PYTHONPATH=TREE python j10v3_ref.py IN.json OUT.json

IN: h1v3_ref.py's units, each with leader_position; OUT: h1v3_ref.py's.
"""
from __future__ import annotations

import json
import sys

from verity.core.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.pouw.circuit import seeds as SD


def main() -> int:
    units = json.loads(open(sys.argv[1]).read())
    noise0 = HT.noise
    out = []
    for u in units:
        salt = bytes.fromhex(u["salt"])
        roots = (bytes.fromhex(u["roots"][0]), bytes.fromhex(u["roots"][1]))
        seed_l = SD.seed(salt, u["leader_position"], *roots)
        HT.noise = lambda seed_j, lay, seed_l=seed_l: (noise0(seed_l, lay)[0], noise0(seed_j, lay)[1])
        try:
            z, rc = HT.reference(u["a"], u["b"], salt, u["position"], roots, stream=True)
        finally:
            HT.noise = noise0
        c = len(u["b"])
        out.append({"z": [[(v - (1 << 32)) if v >> 31 else v for v in z[i * c:(i + 1) * c]] for i in range(len(u["a"]))],
                    "receipt": rc, "leader_position": u["leader_position"]})
    open(sys.argv[2], "w").write(json.dumps({"h1tile": HT.__file__, "peeled": hasattr(HT, "peel_seg"), "units": out,
                                             "noise": "j10-banded/kernels-reading"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
