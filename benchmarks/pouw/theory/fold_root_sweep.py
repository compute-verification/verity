#!/usr/bin/env python3
"""K_R = K_1 = P/X/N for the same-word R-round XOR-in fold, swept with `fold_root_kernel.c` (stdlib only, for a pod).

Each job is a key pair and a word at width n; `fold_root_kernel` finds K_1 .. K_rmax on every state. The sets:

  all        every odd key pair at width n, one seeded random word each
  filtered   every pair whose keys both pass the scaled KJ (`kj_scaled`, the 32-bit KJ19 read at width n), with the
             structured words of `WORDS` and `--random-words` seeded random words each
  lastsample `--keys` seeded random pairs whose keys pass only KJ19's last clause (`kj_last`), with the words of
             `filtered`
  sample     `--keys` seeded random pairs passing the scaled KJ and as many unfiltered, the zero word and one random word
  addsweep   `--keys` pairs passing the scaled KJ (all of them if fewer), every add-limb pair (d1, d3) with one seeded
             random (d0, d2) a pair: every word where an add-limb condition could open the kernel
  xorsweep   the same over every XOR-limb pair (d0, d2), with one seeded random (d1, d3) a pair
  symmetric  `--keys` seeded random pairs passing the scaled KJ, the four NOT-symmetric words (`not_symmetric`) a pair
  recut      the jobs an earlier output (`--from`, a .txt) flagged degenerate, rerun with `--probes` and `--log2cap`

Writes `<set>-n<n>.txt` (the enumerator's S and E lines) and `<set>-n<n>.json` (counts) under --out.

  python3 fold_root_sweep.py --set all --n 10 --out DIR [--threads T]
  python3 fold_root_sweep.py --set recut --n 10 --from DIR/filtered-n10.txt --probes 6 --log2cap 28 --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: rotations at width n standing in for (13, 19) at 32: the pair sums to n, the smaller about 0.4 n
ROT = {6: (2, 4), 8: (3, 5), 10: (4, 6), 12: (5, 7), 14: (6, 8), 16: (6, 10)}


def signed(a: int, m: int) -> int:
    r = a % (1 << m)
    return r - (1 << m) if r >= 1 << (m - 1) else r


def kj_scaled(a: int, n: int, r2: int) -> bool:
    """receipt_folds.kj19 at width n with r2 the larger rotation: for each j < r2, the signed residue of a mod 2^(n−1−j)
    is at least 2^(r2−j) in size. At n = 32, r2 = 19 it is KJ19; for odd a it says no r1 − 1 consecutive bits of
    a's bits 1 .. n − 2 are equal."""
    return all(abs(signed(a, n - 1 - j)) >= 1 << (r2 - j) for j in range(r2))


def kj_last(a: int, r1: int) -> bool:
    """Only KJ19's last clause (j = r2 − 1), a ≢ ±1 mod 2^r1: `egemm_depthr_check.key_ok_r11_32`'s KJ19, and
    the rescaled rule's."""
    return a % (1 << r1) not in (1, (1 << r1) - 1)


def near_inverse(a0: int, a1: int, n: int) -> bool:
    """egemm_depthr_check's reduced pair clause (near_inverse): v2(a0 a1 ∓ 1) >= n − 5."""
    m = (1 << n) - 1
    p = (a0 * a1) & m

    def v2(x):
        return n if x == 0 else (x & -x).bit_length() - 1
    return v2((p - 1) & m) >= n - 5 or v2((p + 1) & m) >= n - 5


def not_symmetric(a0: int, a1: int, w, n: int) -> bool:
    """2·d1 ≡ a0 − 1 and 2·d3 ≡ a1 − 1 mod 2^n: both half-rounds then commute with bitwise NOT, so F_w does, and
    complementing d0 alone gives F_{w'} = NOT ∘ F_w, which agrees with F_w at every even R."""
    m = (1 << n) - 1
    return (2 * w[1] - a0 + 1) & m == 0 and (2 * w[3] - a1 + 1) & m == 0


def words(n: int) -> list[tuple[int, int, int, int]]:
    """Structured words: zero, all ones, top bits, ones, and the half-zero mixes."""
    m, t = (1 << n) - 1, 1 << (n - 1)
    return [(0, 0, 0, 0), (m, m, m, m), (t, t, t, t), (1, 1, 1, 1), (0, t, 0, t), (t, 0, t, 0), (m, 0, m, 0),
            (0, m, 0, m), (1, 0, 0, 0), (0, 0, 1, 0)]


def build(tmp: Path) -> Path:
    exe = tmp / "fold_root_kernel"
    subprocess.run(["gcc", "-O2", "-fopenmp", "-o", str(exe), str(HERE / "fold_root_kernel.c")], check=True)
    return exe


def jobs_for(kind: str, n: int, rng: random.Random, nkeys: int, nrand: int, word=None, src=None) -> list[tuple[int, ...]]:
    r1, r2 = ROT[n]
    if kind == "recut":
        return [tuple(map(int, p[1:7])) for p in (line.split() for line in Path(src).read_text().splitlines())
                if p and p[0] == "S" and p[-1] == "1"]
    rw = lambda: tuple(rng.randrange(1 << n) for _ in range(4))
    odd = range(1, 1 << n, 2)
    if kind == "all":
        return [(a0, a1) + (word or rw()) for a0 in odd for a1 in odd]
    ok = [a for a in odd if kj_scaled(a, n, r2)]
    if kind == "filtered":
        ws = words(n)
        return [(a0, a1) + w for a0 in ok for a1 in ok for w in ws + [rw() for _ in range(nrand)]]
    if kind == "lastsample":
        last = [a for a in odd if kj_last(a, r1)]
        ws = words(n)
        return [k + w for k in ((rng.choice(last), rng.choice(last)) for _ in range(nkeys))
                for w in ws + [rw() for _ in range(nrand)]]
    if kind == "sample":
        out = []
        for _ in range(nkeys):
            k = (rng.choice(ok), rng.choice(ok))
            out += [k + (0, 0, 0, 0), k + rw()]
        for _ in range(nkeys):
            out.append((rng.randrange(1 << n) | 1, rng.randrange(1 << n) | 1) + rw())
        return out
    pairs = [(a0, a1) for a0 in ok for a1 in ok]
    if kind in ("addsweep", "xorsweep"):
        pairs = pairs if len(pairs) <= nkeys else rng.sample(pairs, nkeys)
        out = []
        for a0, a1 in pairs:
            p, q = rng.randrange(1 << n), rng.randrange(1 << n)
            if kind == "addsweep":
                out += [(a0, a1, p, d1, q, d3) for d1 in range(1 << n) for d3 in range(1 << n)]
            else:
                out += [(a0, a1, d0, p, d2, q) for d0 in range(1 << n) for d2 in range(1 << n)]
        return out
    if kind == "symmetric":
        h, out = 1 << (n - 1), []
        for a0, a1 in (rng.choice(pairs) for _ in range(nkeys)):
            out += [(a0, a1, rng.randrange(1 << n), ((a0 - 1) // 2 + s) % (1 << n), rng.randrange(1 << n),
                     ((a1 - 1) // 2 + u) % (1 << n)) for s in (0, h) for u in (0, h)]
        return out
    raise SystemExit(f"unknown set {kind}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True,
                    choices=("all", "filtered", "lastsample", "sample", "addsweep", "xorsweep", "symmetric", "recut"))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--rmax", type=int, default=6)
    ap.add_argument("--keys", type=int, default=1000)
    ap.add_argument("--random-words", type=int, default=2)
    ap.add_argument("--word", default=None, help="with --set all: this word (w0,w1,w2,w3) at every pair instead of a random one")
    ap.add_argument("--threads", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--from", dest="src", default=None, help="with --set recut: an earlier <set>-n<n>.txt")
    ap.add_argument("--probes", type=int, default=3)
    ap.add_argument("--log2cap", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if (a.set == "recut") != (a.src is not None):
        ap.error("--from goes with --set recut")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r1, r2 = ROT[a.n]
    rng = random.Random(a.seed + a.n)
    word = tuple(int(x, 0) for x in a.word.split(",")) if a.word else None
    jobs = jobs_for(a.set, a.n, rng, a.keys, a.random_words, word, a.src)
    tag = f"{a.set}-n{a.n}" + (f"-w{'_'.join(map(str, word))}" if word else "")
    print(f"{tag}: {len(jobs)} jobs, rotations ({r1}, {r2}), R = 1..{a.rmax}, {a.threads} threads", file=sys.stderr, flush=True)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        exe = build(Path(tmp))
        inp = "".join(" ".join(map(str, j)) + "\n" for j in jobs)
        env = dict(os.environ, OMP_NUM_THREADS=str(a.threads))
        with open(out / f"{tag}.txt", "w") as f:
            subprocess.run([str(exe), "-p", str(a.probes), "-c", str(a.log2cap), str(a.n), str(r1), str(r2), str(a.rmax)],
                           input=inp, stdout=f, text=True, check=True, env=env)
    res = (out / f"{tag}.txt").read_text()
    summ = {"set": a.set, "n": a.n, "rotations": [r1, r2], "rmax": a.rmax, "jobs": len(jobs), "seconds": round(time.time() - t0, 1),
            "threads": a.threads, "probes": a.probes, "log2cap": a.log2cap}
    S = [line.split() for line in res.splitlines() if line.startswith("S ")]
    summ["reported"] = len(S)
    bad = []
    for p in S:
        a0, a1, *w = map(int, p[1:7])
        ks = list(map(int, p[7:7 + a.rmax]))
        pxn, g3bad, degen = int(p[7 + a.rmax]), int(p[8 + a.rmax]), int(p[9 + a.rmax])
        if not (pxn and all(k == ks[0] for k in ks) and g3bad == 0 and degen == 0):
            bad.append({"a0": a0, "a1": a1, "w": w, "K": ks, "pxn": pxn, "g3bad": g3bad, "degenerate": degen,
                        "kj_scaled": [kj_scaled(a0, a.n, r2), kj_scaled(a1, a.n, r2)],
                        "kj_last": [kj_last(a0, r1), kj_last(a1, r1)], "near_inverse": near_inverse(a0, a1, a.n),
                        "not_symmetric": not_symmetric(a0, a1, w, a.n)})
    summ["bad"] = len(bad)
    summ["bad_under_kj_scaled"] = sum(all(b["kj_scaled"]) for b in bad)
    summ["bad_under_kj_scaled_not_symmetric_word"] = sum(all(b["kj_scaled"]) and not b["not_symmetric"] for b in bad)
    summ["symmetric_words"] = sum(not_symmetric(j[0], j[1], j[2:], a.n) for j in jobs)
    summ["symmetric_words_bad"] = sum(b["not_symmetric"] for b in bad)
    summ["symmetric_words_K"] = sorted({tuple(b["K"]) for b in bad if b["not_symmetric"]})[:20]
    summ["bad_under_kj_last_and_not_near_inverse"] = sum(all(b["kj_last"]) and not b["near_inverse"] for b in bad)
    summ["bad_under_kj_last"] = sum(all(b["kj_last"]) for b in bad)
    summ["bad_cases"] = bad[:20000]
    (out / f"{tag}.json").write_text(json.dumps(summ, indent=1))
    print(f"{tag}: {summ['reported']}/{len(jobs)} reported, {len(bad)} with K_R != K_1 = P/X/N, of which "
          f"{summ['bad_under_kj_scaled']} pass the scaled KJ ({summ['bad_under_kj_scaled_not_symmetric_word']} at a word "
          f"that isn't NOT-symmetric); NOT-symmetric words {summ['symmetric_words_bad']}/{summ['symmetric_words']} bad; "
          f"{summ['seconds']} s", file=sys.stderr, flush=True)
    return 0 if summ["reported"] == len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
