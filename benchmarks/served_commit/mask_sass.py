"""M_key·y's priced work a salt on the H200 (sm_90a), from the SASS of mask_probe.cu: stdlib only.

    python3 mask_sass.py ROLLED_SASS UNROLLED_SASS OUT_JSON

Each probe's loop trip is one salt: its loads (LDG) left out, its stores writing nothing. In the rolled build the product's
loop over r is nested inside, 16 trips a salt, and ptxas splits its body on r < 8 (which of Y's two words a byte reads) into
two copies, one a trip: each copy is priced for its 8 trips, the code they share for 16. Prices are sha512_sass.py's (32 W1
units a register at the floor, each instruction alone); a 128-bit shared load writes 4 registers.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sha512_sass as S  # noqa: E402

TRIPS = 16


def per_salt(ins: list[dict]) -> dict:
    """A salt's price: the probe's loop trip, a nested loop over r weighted path by path."""
    ls = S.loops(ins)
    a, b = ls[0]
    inner = [(c, d) for c, d in ls[1:] if a <= c and d <= b]
    if not inner:
        return S.per_compression(ins, TRIPS)
    (c, d), = inner
    ps = S.paths(ins, c, d)
    if len(ps) != 2 or len(S.paths(ins, a, b)) != 2:
        raise SystemExit(f"expected the loop over r split in two (r < 8 and r >= 8) and nothing else: {len(ps)} paths")
    mult: Counter = Counter({i["addr"]: 1 for i in ins[a:b + 1]})
    for n in range(c, d + 1):
        mult[ins[n]["addr"]] = 0
    for p in ps:
        for n in p:
            mult[ins[n]["addr"]] += TRIPS // 2
    r = S.price(ins[a:b + 1], mult)
    r["loads_left_out"] = sum(1 for i in ins[a:b + 1] if i["op"].startswith("LDG"))
    r["inner_loops"] = 1
    r["paths"] = [S.price([ins[n] for n in p]) for p in ps]
    return r


def main() -> None:
    out = {}
    for form, path in (("rolled", sys.argv[1]), ("unrolled", sys.argv[2])):
        r = per_salt(S.parse(Path(path))["probe"])
        out[form] = r
        each = "; a trip by path " + ", ".join(f"{p['registers_written']:.0f}" for p in r["paths"]) if "paths" in r else ""
        print(f"{form}: {r['registers_written']:.0f} registers written a salt ({r['instructions']:.0f} instructions{each}), "
              f"W1 floor {r['w1_floor']:.0f}, alone {r['w1_alone']:.0f}; top {list(r['by_opcode'].items())[:6]}")
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
