"""Run fold_super's randomised superopt hunt at scale across every core (RT-FOLD round 2), for a pod via `research run`.

Each worker runs fold_super.superopt_hunt with a distinct shard seed (varying both the program RNG and the filter points);
the programs-tried counts sum and any hit is collected. The hunt covers 2- and 3-word duo/fx1/li/xin13 receipts over the
real sm_120 menu (IMAD.WIDE with two output words, PRMT byte selects, the SHF funnels and the uniform-datapath ops), at
lengths strictly below honest, and verifies any match on 10^4 points. Writes OUT/hunt.json.

Usage: python fold_hunt_pod.py [--total 150000000] [--procs N] [--shards K] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RECEIPTS = _HERE.parents[0] / "receipts"
_EXACT = _HERE.parents[0] / "exact_kit"
for _p in (str(_HERE), str(_RECEIPTS), str(_EXACT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fold_super as F  # noqa: E402


def _worker(args):
    budget, seed = args
    hits, tried, ntargets = F.superopt_hunt(budget, F.default_keys(), seed=seed)
    return tried, ntargets, [(seed,) + tuple(h) for h in hits]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=150_000_000)
    ap.add_argument("--procs", type=int, default=max(1, os.cpu_count() or 2))
    ap.add_argument("--shards", type=int, default=None)
    ap.add_argument("--out", default=os.environ.get("OUT", "."))
    args = ap.parse_args(argv)
    shards = args.shards or args.procs * 4
    per = max(1, args.total // shards)
    tasks = [(per, s + 1) for s in range(shards)]
    t0 = time.time()
    tried = 0
    ntargets = 0
    hits = []
    with mp.get_context("fork").Pool(args.procs) as pool:
        for tr, nt, hs in pool.imap_unordered(_worker, tasks):
            tried += tr
            ntargets = nt
            hits += hs
    doc = {"total_tried": tried, "targets": ntargets, "shards": shards, "procs": args.procs,
           "menu_size": len(F._MENU), "secs": round(time.time() - t0, 1), "hits": hits,
           "verdict": "BREAK" if hits else "no break"}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hunt.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(json.dumps(doc, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
