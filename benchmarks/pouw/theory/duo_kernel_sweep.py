"""Sweeps of duo's all-state kernel at small widths (`duo_kernel.c`), as JSON lines.

Each row is one invocation: width n, run length L, the committed value (``full``: the final state (y, z); ``proj``:
the served receipt y ^ rotl(z, n/2), ``proj-K`` allowing one constant difference across states), the key pattern and
rotations, and how many key assignments it swept and how many had a kernel element. Sets:

- ``generic``: every odd multiplier and every nonzero rotation at each position that matters (1..L-1 for ``full``,
  1..L for ``proj``), independently: this covers pfill's key_ok keys and the served SERVED_KA keys at once, since both
  are odd keys. Exhaustive where the row says so.
- ``pfill``: pfill's shape, keys (a0, a1) alternating and rotations (5, 11, 17, 29) mod n, a run starting at any of
  the four positions of an MMA, every odd pair.
- ``sample``: independent random odd keys and nonzero rotations at each position, for lengths the exhaustive sweep
  can't reach.

    python duo_kernel_sweep.py --set quick            # seconds: the tests' rows
    python duo_kernel_sweep.py --set pod --out DIR    # the pod rows (see the note), one process per core
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PFILL_ROT = (5, 11, 17, 29)


def build(out: Path) -> Path:
    exe = out / "duo_kernel"
    subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", "-o", str(exe), str(HERE / "duo_kernel.c")], check=True)
    return exe


def invoke(exe: Path, n: int, L: int, mode: str, idx: tuple, rot: tuple, phase: int = 0, keys: tuple | None = None,
           threads: int = 1) -> dict:
    args = [str(exe), str(n), str(L), "proj" if mode.startswith("proj") else "full", "-i", ",".join(map(str, idx)),
            "-r", ",".join(map(str, rot)), "-p", str(phase)]
    args += ["-K"] if mode == "proj-K" else []
    args += ["-a", ",".join(map(str, keys))] if keys else ["-s"]
    p = subprocess.run(args, capture_output=True, text=True, check=True, env={**os.environ, "OMP_NUM_THREADS": str(threads)})
    lines = p.stdout.strip().splitlines()
    f = lines[-1].split()
    return {"n": n, "L": L, "mode": mode, "idx": list(idx), "rot": list(rot), "phase": phase,
            "keys": list(keys) if keys else "every odd", "configs": int(f[f.index("configs") + 1]),
            "hit_configs": int(f[f.index("hit_configs") + 1]), "hits": int(f[f.index("hits") + 1]), "nontop": int(f[f.index("nontop") + 1]),
            "hit_lines": lines[:-1]}


def generic(n: int, L: int, mode: str) -> list[tuple]:
    """Every nonzero rotation at each position that matters, keys swept by the binary (positions independent)."""
    m = L - 1 if mode == "full" else L
    if m == 0:
        return []
    return [(n, L, mode, tuple(range(L)), rots + (1,) * (L - m), 0, None)
            for rots in itertools.product(range(1, n), repeat=m)]


def pfill(n: int, L: int, mode: str = "full") -> list[tuple]:
    rot = tuple(r % n for r in PFILL_ROT)
    if 0 in rot:
        return []
    return [(n, L, mode, (0, 1), rot, ph, None) for ph in range(4)]


def sample(n: int, L: int, mode: str, count: int, seed: int) -> list[tuple]:
    rng = random.Random(seed)
    return [(n, L, mode, tuple(range(L)), tuple(rng.randrange(1, n) for _ in range(L)), 0,
             tuple(rng.randrange(1 << n) | 1 for _ in range(L))) for _ in range(count)]


def plan(name: str) -> list[tuple]:
    if name == "quick":
        return (generic(4, 3, "full") + generic(6, 3, "full") + generic(4, 4, "full") + pfill(6, 4)
                + generic(6, 1, "proj") + generic(6, 1, "proj-K") + generic(6, 2, "proj") + generic(6, 2, "proj-K"))
    if name == "pod":
        jobs = []
        for n in (4, 6, 8):
            jobs += generic(n, 3, "full")
        jobs += generic(4, 4, "full") + generic(4, 5, "full") + generic(6, 4, "full")
        jobs += pfill(6, 4) + pfill(6, 5) + pfill(8, 4)
        jobs += sample(6, 5, "full", 4000, 1) + sample(8, 5, "full", 64, 2)
        for n in (4, 6, 8):
            for mode in ("proj", "proj-K"):
                jobs += generic(n, 1, mode) + generic(n, 2, mode)
        jobs += generic(4, 3, "proj") + generic(4, 3, "proj-K") + generic(4, 4, "proj") + generic(4, 4, "proj-K")
        jobs += generic(6, 3, "proj") + generic(6, 3, "proj-K") + sample(8, 3, "proj", 1000, 5)
        return jobs
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
    rows = out / f"duo_kernel_{a.set}.jsonl"
    total = hits = 0
    with rows.open("w") as fh, ThreadPoolExecutor(a.jobs) as pool:
        for r in pool.map(lambda j: invoke(exe, *j), jobs):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            total += r["configs"]
            hits += r["hit_configs"]
            if r["hit_configs"]:
                print("HIT", json.dumps(r), flush=True)
    print(json.dumps({"set": a.set, "invocations": len(jobs), "configs": total, "hit_configs": hits, "rows": str(rows),
                      "cpus": os.cpu_count()}))
    sys.exit(0)


if __name__ == "__main__":
    main()
