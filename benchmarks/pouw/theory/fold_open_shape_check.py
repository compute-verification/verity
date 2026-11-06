#!/usr/bin/env python3
"""Two n = 8 keys that the reduced key_ok_r11 (egemm_depthr_check) admits and at which K_R = K_1 = P/X/N fails
(the corrected depth-R criterion).

A sweep found them among 400 random key_ok_r11 keys; this checks them independently. At (97, 179) with the
word (255, 48, 87, 75), K_1 has 64 elements (meet in the middle), all fixing the R-fold for R = 1..5 on every state, 48
of them outside (G₃). At (83, 225) with the word (144, 166, 44, 160), K_1 is the 16-element group, but changes outside it
fix the 2-fold and the 4-fold on every state. At n = 10, (321, 411) has a 32-element K_1. All three keys are
≡ 1 mod 2^(r1+2). The reduced KJ19 rejects only
±1 mod 2^(n−2) = 64, which is looser than its 32-bit KJ19 (±1 mod 2^13 = 2^r1, `key_ok_r11_32`) would be at this width.
Scaled as ±1 mod 2^r1, the filter rejects all three, but at n = 8 it rejects every key, so width-8 evidence can't speak
for the 32-bit filter; n = 10 is the smallest width that can.

  uv run python benchmarks/pouw/theory/fold_open_shape_check.py [--verdict PATH]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fold_kernel_check as fk
from exact_kit.checks import Checks

N, R1, R2 = 8, 3, 5
BIG = (97, 179), (255, 48, 87, 75)
EVEN = (83, 225), (144, 166, 44, 160)
EVEN_EXTRA = (16, 38, 60, 160)
N10, R1_10, R2_10 = 10, 4, 6
BIG10 = (321, 411), (827, 208, 250, 366)


def fixes(word, other, key, rounds, n=N, r1=R1, r2=R2):
    xs = np.arange(1 << n, dtype=np.uint64)
    return bool(np.array_equal(fk.fold(xs, word, *key, n, r1, r2, rounds), fk.fold(xs, other, *key, n, r1, r2, rounds)))


def kj19_scaled(a, e=R1):
    return a % (1 << e) in (1, (1 << e) - 1)


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "edited_gemm"))
    import egemm_depthr_check as fold_ref

    checks = Checks()
    keys = (BIG[0], EVEN[0])
    checks.check("both keys pass the reduced key_ok_r11 (egemm_depthr_check) and are ≡ 1 mod 2^(r1+2) = 32, not ±1 "
                 "mod 2^(n−2) = 64", all(fold_ref.key_ok_r11(*k, N) for k in keys)
                 and all(k[0] % 32 == 1 or k[1] % 32 == 1 for k in keys), str(keys))
    (key, word) = BIG
    k1 = fk.mitm_k1(word, *key, N, R1, R2)
    bad = sorted(w for w in k1 if not fk.g3_ok(word, w, N))
    checks.check("(97, 179), word (255, 48, 87, 75): K_1 has 64 elements by meet in the middle, every one fixing the "
                 "R-fold on every state for R = 1..5, and 48 break (G₃)", len(k1) + 1 == 64 and len(bad) == 48
                 and all(fixes(word, w, key, r) for w in k1 for r in range(1, 6)), f"e.g. {bad[:2]}")
    (key, word) = EVEN
    k1 = fk.mitm_k1(word, *key, N, R1, R2)
    checks.check("(83, 225), word (144, 166, 44, 160): K_1 is 16 elements, and (16, 38, 60, 160), outside it, fixes the "
                 "2-fold and the 4-fold on every state but not the 1- or 3-fold, keeping (G₃)",
                 len(k1) + 1 == 16 and EVEN_EXTRA not in k1 and fixes(word, EVEN_EXTRA, key, 2)
                 and fixes(word, EVEN_EXTRA, key, 4) and not fixes(word, EVEN_EXTRA, key, 1)
                 and not fixes(word, EVEN_EXTRA, key, 3) and fk.g3_ok(word, EVEN_EXTRA, N))
    (key, word) = BIG10
    k1 = fk.mitm_k1(word, *key, N10, R1_10, R2_10)
    checks.check("n = 10, rotations (4, 6): (321, 411) passes the reduced key_ok_r11, is ≡ 1 mod 2^(r1+2) = 64, and with "
                 "the word (827, 208, 250, 366) has a 32-element K_1, every element fixing the R-fold on every state for "
                 "R = 1..3", fold_ref.key_ok_r11(*key, N10) and key[0] % 64 == 1 and len(k1) + 1 == 32
                 and all(fixes(word, w, key, r, N10, R1_10, R2_10) for w in k1 for r in range(1, 4)),
                 f"(G₃) breaks at {sum(not fk.g3_ok(word, w, N10) for w in k1)}")
    rng = np.random.default_rng(1)
    near = [(s + (int(k) << 15)) & 0xFFFFFFFF for k in rng.integers(0, 1 << 17, 64) for s in (1, -1)]
    checks.check("KJ19 scaled as the 32-bit filter is (a ≢ ±1 mod 2^r1, as 13 at 32 bits) rejects all three keys, and "
                 "the 32-bit key_ok_r11 rejects the keys ≡ ±1 mod 2^(13+2) (128 sampled)",
                 all(kj19_scaled(k[0]) or kj19_scaled(k[1]) for k in keys) and kj19_scaled(BIG10[0][0], R1_10)
                 and not any(fold_ref.key_ok_r11_32(a) for a in near))
    admitted = {n: sum(fold_ref.key_ok_r11(a, b, n) and not kj19_scaled(a, e) and not kj19_scaled(b, e)
                       for a in range(1, 1 << n, 2) for b in range(1, 1 << n, 2)) for n, e in ((N, R1), (N10, R1_10))}
    checks.check("so rescaled, the width-8 filter admits no key (3 and 5 mod 8 multiply to ±1 mod 8, which the "
                 "near-inverse clause rejects) and width-8 evidence can't speak for the 32-bit filter; at n = 10 it "
                 "admits 122,880 of 262,144 odd pairs", admitted == {N: 0, N10: 122880}, str(admitted))
    return checks.finish()


if __name__ == "__main__":
    sys.exit(main())
