"""The graded failure rule's floor over a window of rounds (the frozen definition v2.2, decision 21):

    W_k = sum_{w passed} F_w - top_{D(k)} {F_w : w passed},   D(k) = mu_k / (1 - p),
    mu_k - k - k ln(mu_k / k) = ln(2^40 (k + 1)(k + 2)),      mu_0 = 41 ln 2,

k the failed rounds (after their noise: abandoned before (c), a proof session started and not finished, a failed check or
proof), p the per-round pass probability of a round that hides its work (3/4 at F in {±1}). top_D takes the D largest,
the last one fractionally (d_w in [0, 1] with sum d_w < D).

    window.py ROUNDS.jsonl [--p 0.75]    the challenger's record: one line a round, {"round", "status", "F"}
"""
from __future__ import annotations

import argparse
import json
import math
import sys


def mu(k: int) -> float:
    rhs = math.log(2.0 ** 40 * (k + 1) * (k + 2))
    if k == 0:
        return rhs
    f = lambda x: x - k - k * math.log(x / k)  # noqa: E731
    lo, hi = float(k), 2.0 * k + rhs
    while f(hi) < rhs:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) < rhs:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def depth(k: int, p: float) -> float:
    return mu(k) / (1 - p)


def top(fs: list[int], d: float) -> float:
    s, left = 0.0, d
    for f in sorted(fs, reverse=True):
        if left <= 0:
            break
        s += f * min(1.0, left)
        left -= 1
    return s


def floor(rounds: list[dict], p: float = 0.75) -> dict:
    passed = [r["F"] for r in rounds if r["status"] == "passed"]
    k = sum(r["status"] == "failed" for r in rounds)
    skipped = sum(r["status"] == "skipped" for r in rounds)
    d = depth(k, p)
    w = max(0.0, sum(passed) - top(passed, d))
    return {"rounds": len(rounds), "passed": len(passed), "failed": k, "skipped": skipped, "p": p, "mu_k": mu(k),
            "D_k": d, "work_passed": sum(passed), "W_k": w, "share": w / sum(passed) if passed else 0.0,
            "rounds_for_nonzero": math.floor(d) + 1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rounds")
    ap.add_argument("--p", type=float, default=0.75)
    a = ap.parse_args()
    print(json.dumps(floor([json.loads(x) for x in open(a.rounds)], a.p)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
