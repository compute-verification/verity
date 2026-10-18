"""sharaw's bit-exactness check: commit_gpu's reference.py (verity.ml.commitments' hm96.SHA512, rowleaf, merkle) on 4 KiB
`sha512/row/v2` rows with derived salts, the only form sharaw commits.

    python3 check.py prepare WORK             the cases' inputs and the reference's levels and b || c (CPU only)
    python3 check.py check BIN WORK TAG       BIN's `check` on every case against the reference (exit 1 on any difference)

Ragged leaf counts (pads; one CTA or many; a climb of 1 to 5 levels above a 256-leaf CTA) and a fresh key. Run from the
source tree's root.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pouw", "commit_gpu"))
import reference as R  # noqa: E402

BITS, STRIDE = 32768, 4096
NS = (1, 2, 3, 5, 37, 127, 128, 129, 255, 256, 257, 300, 511, 513, 1000, 4097, 8192)


def cases() -> list[tuple]:
    out = [("rows", n, BITS, STRIDE, "prf", R.KEY) for n in NS]
    out.append(("rows", 777, BITS, STRIDE, "prf", R.INST.fresh_key()))
    return out


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "prepare":
        for i, case in enumerate(cases()):
            print(json.dumps(R.prepare_case(os.path.join(argv[2], f"c{i:03d}"), *case, seed=7000 + i)), flush=True)
        return 0
    if len(argv) == 5 and argv[1] == "check":
        binary, work, tag = argv[2:5]
        failed = total = 0
        for d in sorted(x for x in os.listdir(work) if x.startswith("c")):
            r = R.check_case(binary, os.path.join(work, d), tag)
            failed += not r["ok"]
            total += 1
            print(json.dumps(r), flush=True)
        print(json.dumps({"summary": {"tag": tag, "cases": total, "failed": failed}}), flush=True)
        return 1 if failed or not total else 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
