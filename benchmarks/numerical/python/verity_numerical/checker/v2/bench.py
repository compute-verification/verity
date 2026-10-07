"""checker v2 against ``bench-instances/v1``: every ``tu-k16`` unit (honest witness accepted with the recorded
``out``; the v1 checker's output agrees), the first N ``vu-k1536`` VUs, and every ``vu-k1536-neg`` negative
rejected (the claimed word is not accepted under the honest witness; a ``reject`` verdict is refused outright).

    python -m verity_numerical.checker.v2.bench --root fixtures/bench-instances/v1 --vus 256 --json out.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..params import REAL, Reject
from ..step import step_output as step_output_v1
from ..step import vu_output as vu_output_v1
from . import step as v2

__all__ = ["main", "run"]


def run(root: Path, *, vus: int = 256, tus: int | None = None, v1_cross: bool = True) -> dict:
    from verity_numerical.bench import instances as I

    p = REAL
    out: dict = {}
    t0 = time.perf_counter()
    tu = I.load("tu-k16", root)
    n = tu.n if tus is None else min(tus, tu.n)
    bad = []
    for i in range(n):
        c, a, b, y = tu.instance(i)
        try:
            h = v2.gen_step(p, c, a, b)
            v2.check_step(p, c, a, b, y, h)
            if v2.step_output(p, c, a, b) != y:
                bad.append((i, "output"))
            if v1_cross and step_output_v1(p, c, a, b) != y:
                bad.append((i, "v1-output"))
        except Reject as e:
            bad.append((i, f"reject: {e}"))
    out["tu-k16"] = {"n": n, "failures": bad, "seconds": round(time.perf_counter() - t0, 1)}

    t0 = time.perf_counter()
    vu = I.load("vu-k1536", root)
    n = min(vus, vu.n)
    bad = []
    for i in range(n):
        x, w, y = vu.instance(i)
        try:
            h = v2.gen_vu(p, x, w)
            v2.check_vu(p, x, w, y, h)
            if v1_cross and vu_output_v1(p, x, w) != y:
                bad.append((i, "v1-output"))
        except Reject as e:
            bad.append((i, f"reject: {e}"))
    out["vu-k1536"] = {"n": n, "failures": bad, "seconds": round(time.perf_counter() - t0, 1)}

    t0 = time.perf_counter()
    neg = I.load("vu-k1536-neg", root)
    accepted, by_verdict = [], {}
    for i in range(neg.n):
        x, w, claimed = neg.instance(i)
        _, mistake, _base, _cy, correct, verdict, _ = neg.index[i]
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
        try:
            h = v2.gen_vu(p, x, w)
        except Reject:
            if verdict != "reject":
                accepted.append((i, mistake, "honest gen refused an in-domain instance"))
            continue
        if verdict == "reject":
            accepted.append((i, mistake, "gen produced a witness for an out-of-domain instance"))
            continue
        if v2.accepts_vu(p, x, w, claimed, h):
            accepted.append((i, mistake, "claimed word accepted"))
        if not v2.accepts_vu(p, x, w, correct, h):
            accepted.append((i, mistake, "correct word not accepted"))
    out["vu-k1536-neg"] = {"n": neg.n, "verdicts": by_verdict, "accepted": accepted,
                           "seconds": round(time.perf_counter() - t0, 1)}
    out["ok"] = not (out["tu-k16"]["failures"] or out["vu-k1536"]["failures"] or accepted)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path("fixtures/bench-instances/v1"))
    ap.add_argument("--vus", type=int, default=256)
    ap.add_argument("--tus", type=int, default=None)
    ap.add_argument("--no-v1-cross", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    r = run(a.root, vus=a.vus, tus=a.tus, v1_cross=not a.no_v1_cross)
    print(json.dumps(r, indent=1))
    if a.json:
        a.json.write_text(json.dumps(r, indent=1))
    return 0 if r["ok"] else 1


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
