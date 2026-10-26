"""FOLD-OPT at π* for a cpw_routes run (`cpw_routes.sh`'s out/), from its own SASS: each mode's timed loop, IMMA16832
dropped, a unit 1/8 of a warp-set, against the menu's floor N_w φ* (8 IMAD + 8 ALU a unit, `Sm120.flatFold`).

The loop splits into what every warp issues and the bodies strictly between a forward predicated `@P BRA X` (not
.U.ANY) and X. In modes 9 and 10 those bodies are thread 0's of the block, issued once a block of W warps, so the run's
fold a unit is (outside + bodies / W) / 8; in every other mode each warp issues its own (W = 1). The one-time prologue
is printed apart (straight-line once, and each loop body in it a trip), to amortize over a run's sets.

Rows: VF3s-16's JI3t_0 through `loads.class_of`, an opcode with no class or a class JI3t_0 omits at 0, as the measured
headline's (I1) has no catch-all; the same with that catch-all at NOROW's coefficient; DS-120's issue row, every warp
instruction one slot. NOPs are dropped.

    uv run python theory/cpw_price.py OUT          # from benchmarks/pouw
"""
from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "facet_break"))
import loads as L  # noqa: E402
from verity_machine_model import eps as E  # noqa: E402

BLOCK_WARPS = {9: 4, 10: 16}
INS = re.compile(r"/\*([0-9a-f]{4,})\*/\s+((?:@!?U?P[T0-9]+\s+)?)([A-Z][A-Z0-9_.]*)([^;]*);")


def targets(rest: str) -> list[int]:
    return [int(t, 16) for t in re.findall(r"0x([0-9a-f]+)", rest)]


def price(ops, c: dict, row: str) -> float:
    tot = 0.0
    for op in ops:
        k = L.class_of(op)
        if k == "IMMA16832" or op == "NOP":
            continue
        if row == "DS_issue":
            tot += 1.0
            continue
        v = c.get(k, {}).get("JI3t_0", 0.0) if k else 0.0
        tot += c["NOROW"]["JI3t_0"] if v == 0 and row == "JI3t_0+catch-all" else v
    return tot


def split(body: str):
    ins = [(int(a, 16), p.strip(), op, rest) for a, p, op, rest in INS.findall(body)]
    loops = [(t, a) for a, p, op, rest in ins if op.startswith("BRA") for t in targets(rest) if t < a]
    head, tail = max(loops, key=lambda lp: lp[1] - lp[0])
    loop = [x for x in ins if head <= x[0] <= tail]
    bodies = [(a, targets(rest)[0]) for a, p, op, rest in loop
              if op == "BRA" and p and targets(rest) and targets(rest)[0] > a]
    inbody = [any(b < a < t for b, t in bodies) for a, *_ in loop]
    outside = [op for (a, p, op, r), i in zip(loop, inbody) if not i]
    inside = [op for (a, p, op, r), i in zip(loop, inbody) if i]
    pro = [x for x in ins if x[0] < head]
    inner = [(t, a) for a, p, op, rest in pro if op.startswith("BRA") for t in targets(rest) if t < a]
    once = [op for a, p, op, r in pro if not any(t <= a <= b for t, b in inner)]
    trips = [[op for a, p, op, r in pro if t <= a <= b] for t, b in inner]
    return head, tail, outside, inside, len(bodies), once, trips


def main(argv: list[str]) -> int:
    sass = (Path(argv[0]) / "sass" / "cpw_routes.sass").read_text()
    c = E.machine("VF3s-16").c
    floor = {"JI3t_0": 8 * (c["IMAD"]["JI3t_0"] + c["ALU"]["JI3t_0"]), "DS_issue": 16.0}
    floor["JI3t_0+catch-all"] = floor["JI3t_0"]
    parts = re.split(r"\n\s*Function : (\S+)\n", "\n" + sass)
    for name, body in sorted(zip(parts[1::2], parts[2::2]), key=lambda nb: int(re.search(r"ILi(\d+)E", nb[0])[1])
                             if re.search(r"cpw[mbx]?ILi\d+E", nb[0]) else -1):
        m = re.search(r"cpw[mbx]?ILi(\d+)E", name)
        if not m:
            continue
        mode = int(m[1])
        w = BLOCK_WARPS.get(mode, 1)
        head, tail, outside, inside, nb, once, trips = split(body)
        print(f"mode {mode} ({name}): loop {head:#x}-{tail:#x}, {len(outside)} instructions every warp issues a set, "
              f"{len(inside)} in {nb} branch bodies, issued by {'one warp of ' + str(w) if w > 1 else 'every warp'}")
        if inside:
            print(f"  bodies: {dict(collections.Counter(inside).most_common())}")
        for row in ("JI3t_0", "JI3t_0+catch-all", "DS_issue"):
            o, b = price(outside, c, row), price(inside, c, row)
            run = (o + b / w) / 8
            print(f"  {row}: the run's fold {run:.4f} a unit (floor {floor[row]:.6f}, ratio {run / floor[row]:.4f}); "
                  f"static {(o + b) / 8:.4f}; prologue {price(once, c, row):.3f} once and "
                  f"{[round(price(t, c, row), 3) for t in trips]} a trip of each loop in it")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
