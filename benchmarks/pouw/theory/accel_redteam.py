"""ACCEL O-REDTEAM's exact CPU checks of attacks on Dither under the draft accelerator model, on the H200's data-sheet rates.

At A = B = 0 (admissible), A~ = R in {0,1}^{n x n} and B~ = S with S[t, j] = U[j, t mod L], U in {+-1..+-64}, L = 32: each
committed word T[l][i][j] is the prefix sum over t < L (l + 1) of R[i, t] S[t, j]. Checks:

- b1: 14 AND.POPC columns a word (7 magnitude planes x 2 signs) plus shift-adds reconstruct each segment dot exactly;
- FFMA: non-negative s32 below 2^23, read as floats (x 2^-149), add exactly and their sum's bit pattern is the integer sum
  below 2^24, but a negative result's pattern is sign-magnitude, and every small negative two's-complement pattern is a NaN;
- Strassen: one level per segment (i, j and the segment's k split in halves), each product accumulated across segments in
  its own accumulator, reproduces every word, with operand combinations in s8 apart from +128;
- biases: no non-negative accumulator bias vanishes in all four Strassen blocks except zero;
- IMAD packing: how many exact terms one 32 x 32 -> 64 IMAD.WIDE can carry.

Each route is priced per committed word as its load on tensor, issue, ALU and IMAD over the data-sheet rate
(`machine_model/data/devices/sm_90/sheet.toml`), against honest `dither-1s-li-l32` (32 s8 MACs, LI at 0.5 ALU + 0.5 IMAD).
Exits 1 if any exactness check fails. Note: compute-research/20261026T0124Z-draft-accel-o-redteam.
"""

import math
import random
import sys

import numpy as np

L = 32
S8_MAC, ALU, IMAD, ISSUE_LANE = 4096, 64, 64, 128
BMMA_BITS = 16 * 8 * 256
FAIL: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        FAIL.append(what)


def time_of(load: dict[str, float]) -> tuple[float, str]:
    r = max(load, key=load.get)
    return load[r], r


HONEST = {"tensor": L / S8_MAC, "issue": 1 / ISSUE_LANE, "ALU": 0.5 / ALU, "IMAD": 0.5 / IMAD}


def b1(rng: random.Random) -> None:
    def route(r, u):
        acc = 0
        for p in range(7):
            cp = sum(a & (1 if x > 0 and (x >> p) & 1 else 0) for a, x in zip(r, u))
            cn = sum(a & (1 if x < 0 and (-x >> p) & 1 else 0) for a, x in zip(r, u))
            acc += (cp - cn) << p
        return acc

    vals = [x for x in range(-64, 65) if x]
    ok = all(
        route(r, u) == sum(a * x for a, x in zip(r, u))
        for r, u in (([rng.randint(0, 1) for _ in range(L)], [rng.choice(vals) for _ in range(L)]) for _ in range(20000))
    )
    check(ok, "b1: 14 AND.POPC columns and 7 shift-adds reconstruct the 1-bit x 7-bit segment dot (20000 segments)")
    recon = 7
    load = {"tensor": 0.0, "issue": (1 + recon) / ISSUE_LANE + 14 * L / BMMA_BITS / 4,
            "ALU": (0.5 + recon) / ALU, "IMAD": 0.5 / IMAD}
    t, r = time_of(load)
    print(f"     b1 per word {t:.6f} SM-clk on {r}: {t / time_of(HONEST)[0]:.1f}x honest (honest {time_of(HONEST)[0]:.6f}, balanced)")


def ffma(rng: random.Random) -> None:
    def as_f(x: int) -> np.float32:
        return np.array([x], dtype=np.uint32).view(np.float32)[0]

    def bits(f: np.float32) -> int:
        return int(np.array([f], dtype=np.float32).view(np.uint32)[0])

    ok = True
    for _ in range(20000):
        x = rng.randrange(0, 1 << 23)
        y = rng.randrange(0, (1 << 24) - x)
        ok &= bits(as_f(x) + as_f(y)) == x + y
    check(ok, "FFMA: s32 patterns in [0, 2^23) add exactly as floats, the sum's pattern the integer sum below 2^24")
    x, y = 5, 9
    neg = bits(as_f(x) - as_f(y))
    check(neg == 0x80000000 | (y - x) and neg != (x - y) % 2**32,
          f"FFMA: a negative result's pattern is sign-magnitude ({neg:#010x}), not two's complement ({(x - y) % 2**32:#010x})")
    check(all(math.isnan(as_f((-k) % 2**32)) for k in [1, 2, 1000, (1 << 23) - 1]),
          "FFMA: every two's-complement pattern of -k, 0 < k < 2^23, is a NaN (so no FP32 arithmetic emits it)")


STRASSEN = {  # Ci = sum of coefficient * Pp
    "C11": {1: 1, 4: 1, 5: -1, 7: 1},
    "C12": {3: 1, 5: 1},
    "C21": {2: 1, 4: 1},
    "C22": {1: 1, 2: -1, 3: 1, 6: 1},
}


def strassen(rng: np.random.Generator, n: int = 64) -> None:
    h, k = n // 2, L // 2
    R = rng.integers(0, 2, (n, n), dtype=np.int64)
    vals = np.array([x for x in range(-64, 65) if x], dtype=np.int64)
    U = rng.choice(vals, (n, L))
    S = U[:, np.arange(n) % L].T
    P = {p: np.zeros((h, h), dtype=np.int64) for p in range(1, 8)}
    ok, plus128, lo, hi, macs = True, 0, 0, 0, 0
    acc = np.zeros((n, n), dtype=np.int64)
    for seg in range(n // L):
        t0 = seg * L
        a = {(r_, c): R[r_ * h:(r_ + 1) * h, t0 + c * k:t0 + (c + 1) * k] for r_ in range(2) for c in range(2)}
        b = {(r_, c): S[t0 + r_ * k:t0 + (r_ + 1) * k, c * h:(c + 1) * h] for r_ in range(2) for c in range(2)}
        A11, A12, A21, A22 = a[0, 0], a[0, 1], a[1, 0], a[1, 1]
        B11, B12, B21, B22 = b[0, 0], b[0, 1], b[1, 0], b[1, 1]
        ops = {1: (A11 + A22, B11 + B22), 2: (A21 + A22, B11), 3: (A11, B12 - B22), 4: (A22, B21 - B11),
               5: (A11 + A12, B22), 6: (A21 - A11, B11 + B12), 7: (A12 - A22, B21 + B22)}
        for p, (x, y) in ops.items():
            lo, hi = min(lo, int(x.min()), int(y.min())), max(hi, int(x.max()), int(y.max()))
            plus128 += int((y == 128).sum())
            P[p] += x @ y
            macs += x.shape[0] * x.shape[1] * y.shape[1]
        acc += R[:, t0:t0 + L] @ S[t0:t0 + L, :]
        blocks = {"C11": acc[:h, :h], "C12": acc[:h, h:], "C21": acc[h:, :h], "C22": acc[h:, h:]}
        for c, co in STRASSEN.items():
            ok &= bool((sum(v * P[p] for p, v in co.items()) == blocks[c]).all())
    check(ok, f"Strassen: one level a segment, products accumulated across segments, reproduces every word (n = {n})")
    print(f"     operand combinations in [{lo}, {hi}]; +128 (outside s8) {plus128} times; MACs {macs / (n * n * n):.4f} of naive")
    iadd3 = sum(math.ceil((len(co) - 1) / 2) for co in STRASSEN.values()) / 4
    fed = sum(sum(1 for co in STRASSEN.values() if p in co) - 1 for p in range(1, 8))
    print(f"     extra deliveries a word {fed / 4:.2f}; post-add IADD3 a word {iadd3:.2f}")
    load = {"tensor": 7 / 8 * L / S8_MAC, "issue": (1 + iadd3) / ISSUE_LANE, "ALU": (0.5 + iadd3) / ALU, "IMAD": 0.5 / IMAD}
    t, r = time_of(load)
    saved = L / 8
    print(f"     Strassen per word {t:.6f} SM-clk on {r}: {t / time_of(HONEST)[0]:.1f}x honest; "
          f"{iadd3 * S8_MAC / ALU / saved:.0f} MAC-times of ALU a MAC saved")


def biases() -> None:
    zero: set[int] = set()
    changed = True
    while changed:
        changed = False
        for co in STRASSEN.values():
            live = {p: v for p, v in co.items() if p not in zero}
            if live and all(v > 0 for v in live.values()) or live and all(v < 0 for v in live.values()):
                zero |= set(live)
                changed = True
    check(zero == set(range(1, 8)), "biases: the only non-negative accumulator biases cancelling in C11, C12, C21, C22 are 0")
    Sb = np.array([x for x in range(-64, 65) if x]) + 63
    check(int(2 * Sb.max()) > 255 or int(2 * Sb.min()) < 0,
          f"biases: at B = 63 Strassen's B-side sums reach [{2 * Sb.min()}, {2 * Sb.max()}], outside s8 and u8")


def imad_packing() -> None:
    def kmax(bound_per_term: int) -> int:
        k = 1
        while (k + 1) * (math.ceil(math.log2(bound_per_term * (k + 1) + 1)) + 1) <= 32:
            k += 1
        return k

    k = kmax(64)
    print(f"     IMAD.WIDE packing: at most {k} exact terms an instruction (slots hold 64 K with sign); "
          f"{math.ceil(L / k)} a word against honest's 0.5 IMAD: {math.ceil(L / k) / 0.5:.0f}x")
    check(k == 3, "IMAD packing: 3 exact terms fit a 32-bit operand's slots, 4 don't")


def main() -> int:
    b1(random.Random(1))
    ffma(random.Random(2))
    strassen(np.random.default_rng(3))
    biases()
    imad_packing()
    print("all checks pass" if not FAIL else f"{len(FAIL)} checks FAIL")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
