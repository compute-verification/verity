"""CA's H1V3 itself on a few units: `h1tile.reference(..., stream=True)` from the tree on PYTHONPATH (sbench.sh extracts
cursor/h1-v3-1172's `verity/` at DITHER_H1V3_SHA), so the gate holds the served kernel's z and receipts to the Definition's
own code, not only to this lane's CPU twin.

    PYTHONPATH=TREE python h1v3_ref.py IN.json OUT.json

IN: a list of units {a, b (32 rows of K int8 encodings each), salt (hex), position, roots ([hex root_a, hex root_b])};
OUT: {"h1tile": the module's file, "units": [{z (32 x 32 row-major, s32), receipt}]}.  IN {"wrap_table": true}: OUT
{"h1tile", "wrap": `wrap_bits(b, c, u)` at index (256 c + u) * 256 + b for every byte b, u and bit c}.
"""
from __future__ import annotations

import json
import sys

from verity.core.protocols.pouw.circuit import h1tile as HT


def main() -> int:
    units = json.loads(open(sys.argv[1]).read())
    if isinstance(units, dict) and units.get("wrap_table"):
        tab = [HT.wrap_bits(b, c, u) for c in range(2) for u in range(256) for b in range(256)]
        open(sys.argv[2], "w").write(json.dumps({"h1tile": HT.__file__, "wrap": tab}))
        return 0
    out = []
    for u in units:
        z, rc = HT.reference(u["a"], u["b"], bytes.fromhex(u["salt"]), u["position"],
                             (bytes.fromhex(u["roots"][0]), bytes.fromhex(u["roots"][1])), stream=True)
        c = len(u["b"])
        out.append({"z": [[(v - (1 << 32)) if v >> 31 else v for v in z[i * c:(i + 1) * c]] for i in range(len(u["a"]))],
                    "receipt": rc})
    open(sys.argv[2], "w").write(json.dumps({"h1tile": HT.__file__, "units": out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
