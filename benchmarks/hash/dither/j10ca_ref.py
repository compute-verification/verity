"""The credit's banded reference itself on a few J10-banded units: `circuit.banded.banded_reference` from the tree on
PYTHONPATH, which the banded kernels' z and receipts are held to.

    PYTHONPATH=TREE python j10ca_ref.py IN.json OUT.json [WORKERS]

IN: h1v3_ref.py's units (a, b, salt, position, roots), each with its tile's I, J, the band g and `row_positions`, row I of
the call's grid of unit positions (banded_noise reads only that row); OUT: h1v3_ref.py's, z signed (A B^T: peeled).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor

from verity.core.protocols.pouw.circuit import banded as BD


def one(u: dict) -> dict:
    salt = bytes.fromhex(u["salt"])
    roots = (bytes.fromhex(u["roots"][0]), bytes.fromhex(u["roots"][1]))
    I, J, g = u["I"], u["J"], u["g"]
    positions = [[] for _ in range(I)] + [u["row_positions"]]
    if positions[I][J] != u["position"]:
        raise ValueError(f"tile ({I}, {J}) sits at {positions[I][J]}, not {u['position']}")
    z, rc = BD.banded_reference(u["a"], u["b"], salt, positions, roots, I, J, g)
    c = len(u["b"])
    return {"z": [[(v - (1 << 32)) if v >> 31 else v for v in z[i * c:(i + 1) * c]] for i in range(len(u["a"]))],
            "receipt": rc, "band_start": BD.band_start(J, g), "leader_position": positions[I][BD.band_start(J, g)]}


def main() -> int:
    units = json.loads(open(sys.argv[1]).read())
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else min(8, len(units)) or 1
    with ProcessPoolExecutor(workers) as pool:
        out = list(pool.map(one, units))
    open(sys.argv[2], "w").write(json.dumps({"banded": BD.__file__, "peeled": True, "units": out,
                                             "noise": "j10-banded (credit's banded reference @ 1a4d1de024)"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
