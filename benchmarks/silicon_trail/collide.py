"""Measure how often a cancelling trail's input differences make the chain collide, and compare with the model.

    python3 collide.py --trail FILE.json [--forced] [--x zero|random|fixed:SEED] --log2-trials T --jobs J --out F.json

FILE.json is a trail as search.py writes it (the file's "best", or the trail itself).  Direct mode runs the chain from
uniform start states on (x, x xor d) and counts equal final states; the model's rate for one trail is 2^-weight.
--forced samples the start state uniformly among those on which round 1 follows the trail (an exact factor
2^-weight_1, since round 1's conditions are on separate rows of a uniform state) and counts over the remaining rounds;
the model's conditional rate is 2^-(weight - weight_1).  It reaches trails too rare to count directly, and it tests
the one thing the model assumes: that the rounds' conditions hold independently.

The common input x matters from step 2 on (step 1's is absorbed by the uniform start state): zero, fresh random bits
every trial, or one fixed random x per SEED.

--exact (w = 1 only) enumerates all 2^25 start states in Python instead: the exact collision probability.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import tempfile
from pathlib import Path

import diff
import keccak as K

HERE = Path(__file__).resolve().parent


def _chi_row(v: int) -> int:
    return sum(((v >> x & 1) ^ ((~(v >> (x + 1) % 5) & 1) & (v >> (x + 2) % 5 & 1))) << x for x in range(5))


def forced_rows(t: diff.Trail) -> list[str]:
    """For each active row of b_1: its bit indices, a value giving c_1's row, a basis of the directions keeping it."""
    out = []
    w = t.w
    crow = dict(diff.rows(t.c[0], w))
    for (y, z), b in diff.rows(t.b[0], w):
        if not b:
            continue
        sol = [u for u in range(32) if _chi_row(u ^ b) ^ _chi_row(u) == crow[(y, z)]]
        assert len(sol) == 1 << (5 - diff.ROW_WEIGHT[b])
        base, basis, span = sol[0], [], {0}
        for u in sol:
            if u ^ base not in span:
                basis.append(u ^ base)
                span |= {s ^ u ^ base for s in span}
        idx = [w * (5 * y + x) + z for x in range(5)]
        out.append("row " + " ".join(map(str, idx + [base, len(basis)] + basis)))
    return out


def spec(t: diff.Trail, xmode: str, forced: bool) -> str:
    w = t.w

    def hexs(s: list[int]) -> str:
        return f"{K.to_int(s, w):x}"

    lines = [f"w {w}", f"k {t.k}"]
    full = (1 << (4 * w)) - 1
    lines.append(f"outer {hexs(K.place(full, w, t.placement))}")
    if xmode == "zero":
        lines.append("xmode 0")
    elif xmode == "random":
        lines.append("xmode 1")
    else:
        rng = random.Random(int(xmode.split(":")[1]))
        lines.append("xmode 2")
        lines += [f"x {hexs(K.place(rng.getrandbits(4 * w), w, t.placement))}" for _ in range(t.k - 1)]
    lines += [f"d {hexs(K.place(d, w, t.placement))}" for d in t.d]
    lines += [f"c {hexs(c)}" for c in t.c]
    rows = forced_rows(t) if forced else []
    lines.append(f"force {len(rows)}")
    if forced:
        lines.append(f"b1 {hexs(t.b[0])}")
        lines += rows
    return "\n".join(lines) + "\n"


def exact(t: diff.Trail, xs: list[int]) -> dict:
    """w = 1: every start state.  Bit j of the integer i is the value of state bit j in start state i."""
    import numpy as np
    assert t.w == 1
    n = 1 << 25
    idx = np.arange(n, dtype=np.uint32)
    planes = [((idx >> j) & 1).astype(np.uint8) for j in range(25)]

    def rnd(s: list) -> list:
        par = [s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20] for x in range(5)]
        s = [s[x + 5 * y] ^ par[(x - 1) % 5] ^ par[(x + 1) % 5] for y in range(5) for x in range(5)]
        b = [None] * 25
        for x in range(5):
            for y in range(5):
                b[y + 5 * ((2 * x + 3 * y) % 5)] = s[x + 5 * y]
        return [b[x + 5 * y] ^ ((b[(x + 1) % 5 + 5 * y] ^ 1) & b[(x + 2) % 5 + 5 * y]) for y in range(5) for x in range(5)]

    def absorb(s: list, v: int) -> list:
        m = K.place(v, 1, t.placement)
        return [s[j] ^ 1 if m[j] else s[j] for j in range(25)]

    a, b = planes, absorb(planes, t.d[0])
    for i in range(1, t.k):
        a, b = rnd(a), rnd(b)
        a, b = absorb(a, xs[i - 1]), absorb(absorb(b, xs[i - 1]), t.d[i])
    differ = np.zeros(n, dtype=np.uint8)
    for j in range(25):
        differ |= a[j] ^ b[j]
    hits = int(n - np.count_nonzero(differ))
    return {"trials": n, "collisions": hits}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trail", type=Path, required=True)
    ap.add_argument("--forced", action="store_true")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--x", default="random")
    ap.add_argument("--log2-trials", type=float, default=30.0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    j = json.loads(a.trail.read_text())
    t = diff.Trail.from_json(j.get("best", j))
    t.check()
    res = {"w": t.w, "placement": t.placement, "k": t.k, "weight": t.weight, "weights": t.weights,
           "d": [f"{d:x}" for d in t.d], "x": a.x, "forced": a.forced, "exact": a.exact}
    if a.exact:
        if a.x == "zero":
            xs = [0] * (t.k - 1)
        else:
            rng = random.Random(int(a.x.split(":")[1]))
            xs = [rng.getrandbits(4 * t.w) for _ in range(t.k - 1)]
        res.update(exact(t, xs))
        res["model_log2"] = -t.weight
    else:
        work = Path(tempfile.mkdtemp(prefix="silicon-collide-"))
        exe = work / "collide"
        subprocess.run(["cc", "-O3", "-march=native", "-o", str(exe), str(HERE / "collide.c")], check=True)
        (work / "spec").write_text(spec(t, a.x, a.forced))
        batches = max(1, round(2 ** a.log2_trials / 512 / a.jobs))
        procs = [subprocess.Popen([str(exe), str(work / "spec"), str(batches), str(a.seed * 1000 + i)],
                                  stdout=subprocess.PIPE, text=True) for i in range(a.jobs)]
        outs = [json.loads(p.communicate()[0]) for p in procs]
        assert all(p.returncode == 0 for p in procs)
        res["trials"] = sum(o["trials"] for o in outs)
        res["collisions"] = sum(o["collisions"] for o in outs)
        res["collisions_on_trail"] = sum(o["collisions_on_trail"] for o in outs)
        res["follow"] = [sum(o["follow"][i] for o in outs) for i in range(t.k - 1)]
        res["model_log2"] = -(t.weight - t.weights[0]) if a.forced else -t.weight
        res["model_follow_log2"] = [-(sum(t.weights[:i + 1]) - (t.weights[0] if a.forced else 0)) for i in range(t.k - 1)]
        res["cpu_seconds"] = round(os.times().children_user + os.times().children_system, 1)
    c, n = res["collisions"], res["trials"]
    res["measured_log2"] = round(math.log2(c / n), 3) if c else None
    res["expected_by_model"] = n * 2.0 ** res["model_log2"]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    shown = f"2^{res['measured_log2']}" if c else f"none in 2^{math.log2(n):.1f}"
    print(f"w={t.w} {t.placement} k={t.k} weights {t.weights} x={a.x}{' forced' if a.forced else ''}: "
          f"{c} collisions in 2^{math.log2(n):.2f} trials, measured {shown}, model 2^{res['model_log2']}"
          f" (expects {res['expected_by_model']:.3g})" + (f"; follow {res['follow']}" if "follow" in res else ""))


if __name__ == "__main__":
    main()
