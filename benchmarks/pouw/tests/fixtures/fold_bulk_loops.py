"""Writes fold_bulk_loops.json, test_lean_headline.py's bulk folds, from the sm_120 SASS of fold_bulk.cu.

From a run of fold_bulk.sh on sm_120 (every kernel bit-exact, 0 bad words):
`uv run benchmarks/pouw/tests/fixtures/fold_bulk_loops.py RUN_DIR/out/sass/fold_bulk.sass`.

Each kernel's timed loop is its longest backward branch. Its opcodes split as fold_bulk.sh's: what every warp issues,
and the bodies of forward predicated branches (thread 0's), once a block of `BT/32` warps. A bulk instruction's key is
its opcode and its bytes, which the SASS's size register holds over 16 (the last `UMOV` into it before the
instruction). A unit's counts are `(every warp + body/W)/(8K)`, 8 units of 128 words a logical warp a set.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "fold_bulk_loops.json"
INS = re.compile(r"/\*([0-9a-f]{4,})\*/\s+((?:@!?U?P[T0-9]+\s+)?)([A-Z][A-Z0-9_.]*)([^;]*);")
NAME = re.compile(r"(fb[kuw])ILi(\d+)ELi(\d+)E(?:Li(\d+)E)?")
BULK = re.compile(r"UBLK(CP|RED)|UTMA(LDG|STG|REDG)")


def _targets(rest: str) -> list[int]:
    return [int(t, 16) for t in re.findall(r"0x([0-9a-f]+)", rest)]


def loops(sass: str) -> dict[str, dict]:
    parts = re.split(r"\n\s*Function : (\S+)\n", "\n" + sass)
    out = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        m = NAME.search(name)
        if not m:
            continue
        kern, K, BT = m[1], int(m[2]), int(m[3])
        tag = {"fbk": f"fbk{'256' if m[4] == '1' else '128'}", "fbu": "fbv" if m[4] == "1" else "fbu",
               "fbw": f"fbw{m[4]}"}[kern]
        ins = [(int(a, 16), p.strip(), op, rest) for a, p, op, rest in INS.findall(body)]
        back = [(t, a) for a, p, op, rest in ins if op.startswith("BRA") for t in _targets(rest) if t < a]
        head, tail = max(back, key=lambda lp: lp[1] - lp[0])
        loop = [x for x in ins if head <= x[0] <= tail]
        bodies = [(a, _targets(rest)[0]) for a, p, op, rest in loop
                  if op in ("BRA", "BRA.U") and p and _targets(rest) and _targets(rest)[0] > a]
        size: dict[str, int] = {}
        every, inside = collections.Counter(), collections.Counter()
        for a, p, op, rest in loop:
            args = [s.strip() for s in rest.split(",")]
            if op == "UMOV" and len(args) == 2:
                size[args[0]] = int(args[1], 16) if args[1].startswith("0x") else 0
            key = op
            if BULK.match(op.split(".")[0]):
                key = f"{op} {16 * size[args[-1]]}"
            (inside if any(b < a < t for b, t in bodies) else every)[key] += 1
        out[f"{tag}_K{K}_BT{BT}"] = {"K": K, "BT": BT, "every_warp": dict(sorted(every.items())),
                                     "body": dict(sorted(inside.items()))}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sass", type=Path, help="fold_bulk.sh's out/sass/fold_bulk.sass on sm_120")
    a = ap.parse_args(argv)
    OUT.write_text(json.dumps({
        "about": "fold_bulk.cu's timed loops on sm_120 (RT4-SHEET's whole-block folds, fx1's fold with its XOR on the "
                 "bulk engine), from r20261019-234605-a812's out/sass/fold_bulk.sass: each kernel's opcodes in its "
                 "timed loop, what every warp issues and the bodies thread 0 issues once a block of BT/32 warps; a "
                 "bulk instruction keyed by its opcode and its bytes (fold_bulk_loops.py).",
        "run": "r20261019-234605-a812", "loops": loops(a.sass.read_text())}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
