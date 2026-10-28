"""H1V3 itself on a few units: `h1tile.reference(..., stream=True)` from the tree on PYTHONPATH (this source's own,
PouwH1Tile_v4 ungated, unless a launcher's H1V3_SHA extracted another commit's `verity/`), so the gate holds the served
kernel's z and receipts to the Definition's own code, not only to dither_one's CPU twin.

    PYTHONPATH=TREE python h1v3_ref.py IN.json OUT.json

IN: a list of units {a, b (32 rows of K int8 encodings each), salt (hex), position, roots ([hex root_a, hex root_b])},
each with "j10r" ({call, m, ri, cj}: J10-R's call position, layout rows and unit) by `h1tile.j10r_reference` at
`j10r_blocks(m, K, tile, ri, cj)` instead (J10R's tree; j10r_so.py);
OUT: {"h1tile": the module's file, "peeled": whether its z is A B^T (its `peel_seg`, 4733b9ab0c on) rather than
X (B + c)^T, "units": [{z (32 x 32 row-major, s32), receipt}]}.  IN {"wrap_table": true}: OUT {"h1tile", "peeled",
"wrap": `wrap_bits(b, c, u)` at index (256 c + u) * 256 + b for every byte b, u and bit c}.  Exit 3 when the h1tile
imported is not under TREE (a TREE without `verity/` would fall back to the environment's installed verity).
"""
from __future__ import annotations

import json
import os
import sys

from verity.core.protocols.pouw.circuit import h1tile as HT


def main() -> int:
    tree = os.path.realpath(os.environ.get("PYTHONPATH", "").split(os.pathsep)[0] or ".")
    if not os.path.realpath(HT.__file__).startswith(tree + os.sep):
        sys.stderr.write(f"h1v3_ref: h1tile is {HT.__file__}, not under the tree {tree} (PYTHONPATH)\n")
        return 3
    units = json.loads(open(sys.argv[1]).read())
    peeled = hasattr(HT, "peel_seg")
    if isinstance(units, dict) and units.get("wrap_table"):
        tab = [HT.wrap_bits(b, c, u) for c in range(2) for u in range(256) for b in range(256)]
        open(sys.argv[2], "w").write(json.dumps({"h1tile": HT.__file__, "peeled": peeled, "wrap": tab}))
        return 0
    out = []
    for u in units:
        roots = (bytes.fromhex(u["roots"][0]), bytes.fromhex(u["roots"][1]))
        if "j10r" in u:
            j = u["j10r"]
            blk = HT.j10r_blocks(j["m"], len(u["a"][0]), (len(u["a"]), len(u["b"])), j["ri"], j["cj"])
            z, rc = HT.j10r_reference(u["a"], u["b"], bytes.fromhex(u["salt"]), u["position"], roots, j["call"], blk,
                                      stream=True)
        else:
            z, rc = HT.reference(u["a"], u["b"], bytes.fromhex(u["salt"]), u["position"], roots, stream=True)
        c = len(u["b"])
        out.append({"z": [[(v - (1 << 32)) if v >> 31 else v for v in z[i * c:(i + 1) * c]] for i in range(len(u["a"]))],
                    "receipt": rc})
    open(sys.argv[2], "w").write(json.dumps({"h1tile": HT.__file__, "peeled": peeled, "units": out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
