"""M_key·y's priced work a salt on the H200 (sm_90a), from the SASS of mask_probe.cu: stdlib only.

    python3 mask_sass.py ROLLED_SASS UNROLLED_SASS OUT_JSON

Each probe's loop trip is one salt: its loads (LDG) left out, its stores writing nothing, and in the rolled build the
product's loop over r nested inside, 16 trips a salt. Prices are sha512_sass.py's (32 W1 units a register at the floor,
each instruction alone); a 128-bit shared load writes 4 registers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sha512_sass as S  # noqa: E402

TRIPS = 16


def main() -> None:
    out = {}
    for form, path in (("rolled", sys.argv[1]), ("unrolled", sys.argv[2])):
        (name, ins), = [(k, v) for k, v in S.parse(Path(path)).items() if "probe" in k]
        r = S.per_compression(ins, TRIPS)
        out[form] = r
        print(f"{form}: {r['registers_written']:.0f} registers written a salt ({r['instructions']:.0f} instructions, inner loops "
              f"{r['inner_loops']}), W1 floor {r['w1_floor']:.0f}, alone {r['w1_alone']:.0f}; top {list(r['by_opcode'].items())[:6]}")
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
