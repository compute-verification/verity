"""Sweeps of (D-proj), the served receipt's binding, at small widths (`duo_proj.c`), as JSON lines.

(D-proj): two different runs of L duo words give served receipts R = y ^ rotl(z, n/2) whose difference is not one
constant over every start state (y, z). The served receipt XORs a thread's 8 chains, so the constant is an XOR
constant (``xor``); ``add`` checks an additive constant too, and ``xor`` rows with kappa 0 are the exact collisions.

Each row is one invocation: width n, run length L, mode, keys and rotations, how many key assignments it swept and how
many had a kernel element, the kappas seen and up to one kernel element an assignment. Sets:

- ``served``: the served schedule scaled to width n. A chain of group g (chains 4g..4g+3) absorbs positions g, g+2,
  g+4, g+6 in turn, with multiplier SERVED_KA[t] mod 2^n and rotation SERVED_ROT[t] scaled to round(r n / 32) in
  1..n-1. A difference first at word i of a window is a run of the window's last L words (a shared prefix permutes the
  start states), so every L and every phase of its end is a row; the window's own end is phase 3.
- ``served-keys``: the scaled served rotations with every odd multiplier at each position.
- ``generic``: every odd multiplier and every rotation 1..n-1 at each position.
- ``sample``: random odd multipliers and rotations at lengths the exhaustive sweep can't reach.
- ``curve``: the served schedule with start states on the curve (y, pi(y)) instead of every (y, z).
- ``full``: the rows above over the full state (``-F``) under xor: runs whose states differ by one constant on every
  start state, which is what wrong words in two chains need to cancel in a checkpoint that XORs every chain's y and
  every chain's z, in three parts that can run side by side (``full-served``, ``full-generic``, ``full-sample``).
  ``full-big`` is served n = 10, L = 4 (a 32 GiB table).

    python duo_proj_sweep.py --set quick            # seconds: the tests' rows
    python duo_proj_sweep.py --set pod --out DIR    # the pod rows (see the note)
    python duo_proj_sweep.py --set pod-big --out DIR  # served n = 6, L = 6: a 32 GiB table
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "receipts"))
import receipt_folds as F  # noqa: E402


def build(out: Path) -> Path:
    exe = out / "duo_proj"
    subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", "-o", str(exe), str(HERE / "duo_proj.c")], check=True)
    return exe


def scaled_rot(r: int, n: int) -> int:
    return min(n - 1, max(1, (r * n + 16) // 32))


def served_keys(n: int, g: int, L: int, end: int = 3) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Multipliers and rotations of the last L words of a group-g chain whose last word is at cycle index `end`."""
    cyc = [g + 2 * i for i in range(4)]
    pos = [cyc[(end - L + 1 + j) % 4] for j in range(L)]
    return (tuple(F.SERVED_KA[t] & ((1 << n) - 1) for t in pos), tuple(scaled_rot(F.SERVED_ROT[t], n) for t in pos))


def invoke(exe: Path, job: dict) -> dict:
    args = [str(exe), str(job["n"]), str(job["L"]), job["mode"], "-r", ",".join(map(str, job["rot"])), "-v", "4"]
    args += ["-a", ",".join(map(str, job["keys"]))] if job.get("keys") else ["-s"]
    args += ["-C"] if job.get("curve") else []
    args += ["-F"] if job.get("full") else []
    args += ["-h", str(job["h"])] if job.get("h") is not None else []
    env = {**os.environ, "OMP_NUM_THREADS": str(job.get("threads", 1))}
    p = subprocess.run(args, capture_output=True, text=True, check=True, env=env)
    lines = p.stdout.strip().splitlines()
    f = lines[-1].split()
    nk = int(f[f.index("kappas") + 1])
    return {**{k: v for k, v in job.items() if k != "threads"}, "keys": job.get("keys") or "every odd",
            "configs": int(f[f.index("configs") + 1]), "hit_configs": int(f[f.index("hit_configs") + 1]),
            "hits": int(f[f.index("hits") + 1]), "toponly": int(f[f.index("toponly") + 1]),
            "kappas": [int(x) for x in f[f.index("kappas") + 2:f.index("kappas") + 2 + nk]], "hit_lines": lines[:-1][:400]}


def served(n: int, Lmax: int, modes=("xor", "add"), curve: bool = False, Lmin: int = 1,
           full: bool = False) -> list[dict]:
    jobs = []
    for g in (0, 1):
        for L in range(Lmin, Lmax + 1):
            for end in range(4):
                keys, rot = served_keys(n, g, L, end)
                for mode in modes:
                    jobs.append({"set": "curve" if curve else "served", "n": n, "L": L, "mode": mode, "group": g,
                                 "end": end, "keys": list(keys), "rot": list(rot), "curve": curve, "full": full,
                                 "threads": 0})
    return jobs


def served_any_keys(n: int, Lmax: int, modes=("xor", "add"), full: bool = False) -> list[dict]:
    jobs, seen = [], set()
    for g in (0, 1):
        for L in range(1, Lmax + 1):
            for end in range(4):
                rot = served_keys(n, g, L, end)[1]
                if rot in seen:
                    continue
                seen.add(rot)
                jobs += [{"set": "served-keys", "n": n, "L": L, "mode": m, "rot": list(rot), "full": full}
                         for m in modes]
    return jobs


def generic(n: int, L: int, modes=("xor", "add"), full: bool = False) -> list[dict]:
    return [{"set": "generic", "n": n, "L": L, "mode": m, "rot": list(rot), "full": full}
            for rot in itertools.product(range(1, n), repeat=L) for m in modes]


def sample(n: int, L: int, count: int, seed: int, modes=("xor", "add"), full: bool = False) -> list[dict]:
    rng = random.Random(seed)
    jobs = []
    for _ in range(count):
        rot = [rng.randrange(1, n) for _ in range(L)]
        keys = [rng.randrange(1 << n) | 1 for _ in range(L)]
        jobs += [{"set": "sample", "n": n, "L": L, "mode": m, "rot": rot, "keys": keys, "full": full,
                  "threads": 0 if n * (L - 1) >= 20 else 1} for m in modes]
    return jobs


def plan(name: str) -> list[dict]:
    if name == "quick":
        return (generic(4, 3, ("xor",))[:6] + generic(6, 1) + served(6, 2) + served(8, 2, ("xor",))
                + served_any_keys(6, 2, ("xor",)))
    if name == "pod":
        jobs = served(6, 5) + served(8, 4) + served(10, 3) + served(12, 3)
        jobs += served(6, 4, ("xor",), curve=True) + served(8, 3, ("xor",), curve=True)
        jobs += served_any_keys(6, 3) + served_any_keys(8, 2)
        jobs += generic(6, 1) + generic(6, 2) + generic(6, 3) + generic(8, 1) + generic(8, 2, ("xor",))
        jobs += generic(4, 4, ("xor",)) + generic(4, 5, ("xor",))
        jobs += sample(8, 3, 300, 11) + sample(10, 2, 200, 12) + sample(10, 3, 40, 13) + sample(12, 2, 100, 14)
        jobs += sample(6, 4, 300, 15) + sample(6, 5, 30, 16)
        return jobs
    if name == "pod-big":
        return served(6, 6, ("xor",), Lmin=6)
    if name == "full-quick":
        return generic(4, 2, ("xor",), full=True) + served(6, 2, ("xor",), full=True)
    x = ("xor",)
    if name == "full-served":
        return (served(6, 4, x, full=True) + served(8, 4, x, full=True) + served(10, 3, x, full=True)
                + plan("full-served-wide"))
    if name == "full-served-wide":
        return served(16, 2, x, full=True) + served(12, 3, x, full=True)
    if name == "full-generic":
        jobs = served_any_keys(6, 3, x, full=True) + served_any_keys(8, 2, x, full=True)
        jobs += generic(4, 1, x, full=True) + generic(4, 2, x, full=True) + generic(4, 3, x, full=True)
        jobs += generic(4, 4, x, full=True) + generic(5, 2, x, full=True) + generic(5, 3, x, full=True)
        jobs += generic(6, 1, x, full=True) + generic(6, 2, x, full=True)
        return jobs + sample(6, 4, 300, 21, x, full=True) + sample(16, 2, 20, 25, x, full=True)
    if name == "full-sample":
        return (sample(8, 4, 24, 22, x, full=True) + sample(10, 3, 60, 23, x, full=True)
                + sample(12, 3, 8, 24, x, full=True))
    if name == "full":
        return plan("full-served") + plan("full-generic") + plan("full-sample")
    if name == "full-big":
        return served(10, 4, ("xor",), Lmin=4, full=True)
    raise SystemExit(f"unknown set {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="quick")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    out = a.out or Path(tempfile.mkdtemp())
    out.mkdir(parents=True, exist_ok=True)
    exe = build(out)
    jobs = plan(a.set)
    for j in jobs:
        if j.get("threads") == 0:
            j["threads"] = a.jobs
    rows = out / f"duo_proj_{a.set}.jsonl"
    total = hit = 0
    with rows.open("w") as fh:
        def emit(r: dict) -> None:
            nonlocal total, hit
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            total += r["configs"]
            hit += r["hit_configs"]
            if r["hit_configs"]:
                print("HIT", json.dumps({k: v for k, v in r.items() if k != "hit_lines"}), flush=True)
        for j in (j for j in jobs if j.get("threads", 1) > 1):
            emit(invoke(exe, j))
        with ThreadPoolExecutor(a.jobs) as pool:
            futures = [pool.submit(invoke, exe, j) for j in jobs if j.get("threads", 1) == 1]
            for f in as_completed(futures):
                emit(f.result())
    print(json.dumps({"set": a.set, "invocations": len(jobs), "configs": total, "hit_configs": hit, "rows": str(rows),
                      "cpus": os.cpu_count()}))


if __name__ == "__main__":
    main()
