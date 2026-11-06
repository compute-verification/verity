#!/usr/bin/env python3
"""An independent check of the depth-R binding over the whole word space.

The binding's own check (`benchmarks/pouw/edited_gemm/egemm_depthr_check.py`) tests a family of 5n + 2 perturbations of the
committed word's four limbs. This one enumerates every one of the 2^(4n) words at width n (`fold_kernel.c`), so it finds
the whole all-state kernel K_R of the same-word R-fold, R = 1..5, at each key pair it is given.

What it establishes:
  (1) The one-round kernel K_1 is never trivial: the P/X/N group (2^4 elements at S = 2 half-rounds) is in
      K_R at every key and every R, so the depth-R criterion "no nonzero change leaves the R-fold fixed" is false as stated.
      Shown at 32 bits on the real rotations (13, 19) and key_ok_r11 keys too.
  (2) Whether repetition adds kernel elements: K_R == K_1 at each tested key and R, or the extra elements found.
  (3) Whether every kernel element respects (G₃) (read at width n: each XOR limb's low three
      bits are kept or complemented, each add limb's bit 0 is kept).
Cross-checks: K_1 at two keys equals exact_kit.search.meet_kernel's meet-in-the-middle answer, and every reported element
is re-verified in numpy on every state.

  uv run python benchmarks/pouw/theory/fold_kernel_check.py [--quick] [--verdict PATH]
"""
from __future__ import annotations

import collections
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from exact_kit import search, wrap
from exact_kit.checks import Checks

HERE = Path(__file__).resolve().parent


def build(tmp: Path) -> Path:
    exe = tmp / "fold_kernel"
    subprocess.run(["gcc", "-O2", "-fopenmp", "-o", str(exe), str(HERE / "fold_kernel.c")], check=True)
    return exe


def enumerate_kernels(exe: Path, n, r1, r2, rmax, word, keys):
    """{(a0, a1): {R: set of words w' != word in K_R}} from the C enumerator."""
    inp = "".join(f"{a0} {a1}\n" for a0, a1 in keys)
    out = subprocess.run([str(exe), str(n), str(r1), str(r2), str(rmax), *(str(x) for x in word)],
                         input=inp, capture_output=True, text=True, check=True).stdout
    res = {(a0 & ((1 << n) - 1), a1 & ((1 << n) - 1)): {r: set() for r in range(1, rmax + 1)} for a0, a1 in keys}
    summary = {}
    for line in out.splitlines():
        p = line.split()
        if p[0] == "K":
            res[(int(p[1]), int(p[2]))][int(p[3])].add(tuple(int(x) for x in p[4:8]))
        elif p[0] == "S":
            summary[(int(p[1]), int(p[2]))] = [int(x) for x in p[3:]]
    for k, counts in summary.items():
        assert [len(res[k][r]) for r in range(1, rmax + 1)] == counts, k
    return res


def fold(y, word, a0, a1, n, r1, r2, rounds):
    m = np.uint64((1 << n) - 1)
    d0, d1, d2, d3 = (np.asarray(x, dtype=np.uint64) for x in word)
    y = np.asarray(y, dtype=np.uint64)
    a0, a1 = np.uint64(a0), np.uint64(a1)
    for _ in range(rounds):
        y = (((wrap.rotl(y, r1, n) ^ d0) * a0) + d1) & m
        y = (((wrap.rotl(y, r2, n) ^ d2) * a1) + d3) & m
    return y


def g3_ok(word, other, n):
    """(G₃) at width n: XOR limbs d0, d2 keep or complement their low three bits; add limbs d1, d3 keep bit 0."""
    return all(((word[i] ^ other[i]) & 7) in (0, 7) for i in (0, 2)) and all(((word[i] ^ other[i]) & 1) == 0 for i in (1, 3))


def kj19_bad(a, n):
    t = 1 << (n - 2)
    return (a % t) in (1, t - 1)


def near_inverse(a0, a1, n):
    m = (1 << n) - 1
    prod = (a0 * a1) & m

    def v2(x):
        return n if x == 0 else (x & -x).bit_length() - 1
    return v2((prod - 1) & m) >= n - 5 or v2((prod + 1) & m) >= n - 5


def key_ok_r11(a0, a1, n):
    """The reduced key_ok_r11 (egemm_depthr_check.key_ok_r11)."""
    return bool(a0 & 1 and a1 & 1 and not kj19_bad(a0, n) and not kj19_bad(a1, n) and not near_inverse(a0, a1, n))


def key_sets(n, rng, per):
    """Structured key pairs at width n: KJ19-bad, g3-chains' open-shape condition (a1 = +-1 mod 2^(r+2), r the smaller
    rotation, which the reduced KJ19 at mod 2^(n-2) still admits when r + 2 < n - 2), key_ok_r11, near-inverse."""
    m = (1 << n) - 1
    odd = lambda: int(rng.integers(0, 1 << n)) | 1
    t = 1 << (n - 2)
    near1 = sorted({(s * t + e) & m for s in range(4) for e in (1, -1)})
    sets = {"kj19-bad": [], "open-shape a1": [], "key_ok_r11": [], "near-inverse": []}
    for a in near1[:per]:
        sets["kj19-bad"].append((a, odd()))
        sets["kj19-bad"].append((odd(), a))
    q = 1 << 5
    cand = sorted({(s * q + e) & m for s in range((1 << n) // q) for e in (1, -1)} - set(near1))
    for a1 in cand[:per]:
        sets["open-shape a1"].append((odd(), a1))
    while len(sets["key_ok_r11"]) < per:
        a0, a1 = odd(), odd()
        if key_ok_r11(a0, a1, n):
            sets["key_ok_r11"].append((a0, a1))
    while len(sets["near-inverse"]) < per:
        a0 = odd()
        sets["near-inverse"].append((a0, pow(a0, -1, 1 << n) ^ (2 << int(rng.integers(3, n - 1)))))
    return sets


def mitm_k1(word, a0, a1, n, r1, r2):
    """K_1 by exact_kit.search.meet_kernel: G_{d0',d1'}(y) == H_{d2',d3'}^{-1}(F_w(y)) on every state."""
    m = (1 << n) - 1
    xs = np.arange(1 << n, dtype=np.uint64)
    target = fold(xs, word, a0, a1, n, r1, r2, 1)
    inv = np.uint64(pow(a1, -1, 1 << n))
    mm = np.uint64(m)

    def g(y, a):
        return (((wrap.rotl(y, r1, n) ^ (a & mm)) * np.uint64(a0)) + (a >> np.uint64(n))) & mm

    def h(y, b):
        z = target[y.astype(np.int64)]
        return wrap.rotr((((z - (b >> np.uint64(n))) & mm) * inv & mm) ^ (b & mm), r2, n)

    A = np.arange(1 << (2 * n), dtype=np.uint64)
    pairs = search.meet_kernel(g, h, xs, A, A)
    return {(a & m, a >> n, b & m, b >> n) for a, b in pairs} - {tuple(word)}


def main() -> int:
    checks = Checks()
    quick = "--quick" in checks.args
    rng = np.random.default_rng(20261017)

    # (1) at 32 bits on the real rotations: the P/X/N group's generators collapse the R-fold on every sampled state.
    m32 = (1 << 32) - 1
    word32 = tuple(int(x) for x in rng.integers(0, 1 << 32, 4))
    keys32 = []
    while len(keys32) < 4:
        a = int(rng.integers(0, 1 << 32)) | 1
        if (a & 0x1FFF) not in (1, 0x1FFF):
            keys32.append(a)
    keys32 = [(keys32[0], keys32[1]), (keys32[2], keys32[3])]
    T = 1 << 31
    gens = {
        "P0: d0 ^ 2^31, d1 + 2^31": lambda w, a0, a1: (w[0] ^ T, (w[1] + T) & m32, w[2], w[3]),
        "P1: d2 ^ 2^31, d3 + 2^31": lambda w, a0, a1: (w[0], w[1], w[2] ^ T, (w[3] + T) & m32),
        "X: d1 + 2^31, d2 ^ 2^18 (bit 31 rotated by 19)": lambda w, a0, a1: (w[0], (w[1] + T) & m32, w[2] ^ (1 << 18), w[3]),
        # ~(u ^ d0) * a0 = -a0 - (u ^ d0) * a0, so d1' = a0 - 1 - d1 makes the half-round output ~g, and ~d2 undoes it
        "N: d0 ^ ~0, d1' = a0 - 1 - d1, d2 ^ ~0": lambda w, a0, a1: (w[0] ^ m32, (a0 - 1 - w[1]) & m32, w[2] ^ m32, w[3]),
    }
    states = rng.integers(0, 1 << 32, 1 << 14, dtype=np.uint64)
    for name, gen in gens.items():
        for a0, a1 in keys32:
            w2 = gen(word32, a0, a1)
            same = all(bool((fold(states, word32, a0, a1, 32, 13, 19, R) == fold(states, w2, a0, a1, 32, 13, 19, R)).all())
                       for R in range(1, 6))
            checks.check(f"32-bit (13,19), key {a0:#x},{a1:#x}: {name} leaves the R-fold fixed, R = 1..5 (2^14 states)",
                         same, f"changed limbs {[i for i in range(4) if w2[i] != word32[i]]}; (G3) kept: {g3_ok(word32, w2, 32)}")

    with tempfile.TemporaryDirectory() as tmp:
        exe = build(Path(tmp))
        configs = [(6, 2, 4, 5, (0x1E, 0x37, 0x39, 0x2B), None), (8, 3, 5, 5, (0x9E, 0x37, 0x79, 0xB9), 6 if quick else 16)]
        for n, r1, r2, rmax, word, per in configs:
            if per is None:
                keys = {"all odd pairs": [(a0, a1) for a0 in range(1, 1 << n, 2) for a1 in range(1, 1 << n, 2)]}
            else:
                keys = key_sets(n, rng, per)
            allkeys = [k for ks in keys.values() for k in ks]
            res = enumerate_kernels(exe, n, r1, r2, rmax, word, allkeys)
            xs = np.arange(1 << n, dtype=np.uint64)

            # a sample of the reported elements re-verified in numpy on every state, all at once
            flat = [(k, r, w2) for k, byr in res.items() for r, ws in byr.items() for w2 in sorted(ws)]
            pick = [flat[i] for i in rng.choice(len(flat), size=min(len(flat), 4000), replace=False)] if flat else []
            bad = 0
            for r in range(1, rmax + 1):
                sel = [(k, w2) for k, rr, w2 in pick if rr == r]
                if not sel:
                    continue
                A0 = np.array([k[0] for k, _ in sel], dtype=np.uint64)[:, None]
                A1 = np.array([k[1] for k, _ in sel], dtype=np.uint64)[:, None]
                W2 = [np.array([w2[i] for _, w2 in sel], dtype=np.uint64)[:, None] for i in range(4)]
                mm = np.uint64((1 << n) - 1)
                y0 = np.broadcast_to(xs, (len(sel), xs.size)).astype(np.uint64)
                y1 = y0.copy()
                for _ in range(r):
                    y0 = (((wrap.rotl(y0, r1, n) ^ np.uint64(word[0])) * A0) + np.uint64(word[1])) & mm
                    y0 = (((wrap.rotl(y0, r2, n) ^ np.uint64(word[2])) * A1) + np.uint64(word[3])) & mm
                    y1 = (((wrap.rotl(y1, r1, n) ^ W2[0]) * A0) + W2[1]) & mm
                    y1 = (((wrap.rotl(y1, r2, n) ^ W2[2]) * A1) + W2[3]) & mm
                bad += int((y0 != y1).any(axis=1).sum())
            checks.check(f"n{n}: enumerated kernel elements re-verified on all {1 << n} states in numpy", bad == 0,
                         f"{len(pick)} sampled of {len(flat)} (key, R, word) elements, {bad} failures")

            # K_1 against exact_kit's meet in the middle, at two keys
            for a0, a1 in allkeys[:2]:
                k = (a0 & ((1 << n) - 1), a1 & ((1 << n) - 1))
                mk = mitm_k1(word, k[0], k[1], n, r1, r2)
                checks.check(f"n{n} key {k}: K_1 from the enumerator == exact_kit.search.meet_kernel", mk == res[k][1],
                             f"{len(res[k][1])} vs {len(mk)} non-identity elements")

            for label, ks in keys.items():
                ks = [(a0 & ((1 << n) - 1), a1 & ((1 << n) - 1)) for a0, a1 in ks]
                size1 = collections.Counter(len(res[k][1]) + 1 for k in ks)
                extra = collections.Counter()
                extra_keys = collections.defaultdict(list)
                g3bad = collections.Counter()
                r11 = sum(key_ok_r11(*k, n) for k in ks)
                for k in ks:
                    for r in range(1, rmax + 1):
                        e = res[k][r] - res[k][1]
                        if e:
                            extra[r] += len(e)
                            extra_keys[r].append((k, key_ok_r11(*k, n), kj19_bad(k[0], n) or kj19_bad(k[1], n)))
                        g3bad[r] += sum(not g3_ok(word, w2, n) for w2 in res[k][r])
                r11keys = [k for k in ks if key_ok_r11(*k, n)]
                size1_r11 = collections.Counter(len(res[k][1]) + 1 for k in r11keys)
                g3bad_r11 = sum(not g3_ok(word, w2, n) for k in r11keys for r in range(1, rmax + 1) for w2 in res[k][r])
                checks.check(f"n{n} {label} ({len(ks)} keys, {r11} key_ok_r11): |K_1| = 16 (the P/X/N group, 3S-2 = 4 "
                             "involutions) at every key_ok_r11 key", set(size1_r11) <= {16},
                             f"|K_1| histogram, key_ok_r11 keys {dict(size1_r11)}, all keys {dict(size1)}")
                odd_extra = sum(extra[r] for r in range(3, rmax + 1, 2))
                checks.check(f"n{n} {label}: odd R adds no kernel element at any key (K_3 = K_5 = K_1)", odd_extra == 0,
                             f"extra at odd R: { {r: extra[r] for r in range(3, rmax + 1, 2)} }; keys "
                             f"{ {r: extra_keys[r][:4] for r in range(3, rmax + 1, 2) if extra_keys[r]} }")
                r11_even = [(r, x) for r in range(2, rmax + 1, 2) for x in extra_keys[r] if x[1]]
                checks.check(f"n{n} {label}: even R adds no kernel element at key_ok_r11 keys", not r11_even,
                             f"extra at even R (all keys): { {r: extra[r] for r in range(2, rmax + 1, 2)} } on "
                             f"{ {r: len(extra_keys[r]) for r in range(2, rmax + 1, 2)} } keys, of which KJ19-bad "
                             f"{ {r: sum(x[2] for x in extra_keys[r]) for r in range(2, rmax + 1, 2)} }; "
                             f"key_ok_r11 cases {r11_even[:4]}")
                checks.check(f"n{n} {label}: every kernel element at key_ok_r11 keys respects (G3) at width n",
                             g3bad_r11 == 0, f"{g3bad_r11} violating at key_ok_r11 keys; all keys by R {dict(g3bad)}")

    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
