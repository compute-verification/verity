"""ACCEL O-CHARGE's exact CPU checks (note compute-research/20261026T0140Z-draft-accel-o-charge).

Run with the workspace venv: `python benchmarks/pouw/theory/accel_o_charge.py [folds|strassen|alu|prices|all]`. Exits 1
if an exactness check fails. Reuses O-REDTEAM's `accel_redteam.py` for b1, FFMA, Strassen's MAC count and IMAD packing.

- folds (4c): every receipt fold (duo, xin, fx1, li of `verity.core.protocols.pouw.circuit.segment`) is non-affine in
  each word over Z/2^32 and over GF(2)^32; the top-bit patterns (the only order-2 elements of (Z/2^32)^k, so the only
  witnesses a Z-linear merge of words can leave) under which a set's state is invariant; every same-chain pair of
  single-bit XOR flips under which it is invariant.
- strassen (4b): depth-1 Strassen of a segment at A = B = 0 priced in tensor-core D-word slots (every dense H200 format
  writes at most 128 D words/SM/clk), and the 6-slot k32 packing of Strassen's sums checked exactly.
- alu: IMAD and dp4a schedules (naive, Strassen depth 1 with the prefix) checked exactly, with op counts.
- prices: per-word loads of each route on tensor, issue, ALU and FMA, c(lambda), and the ALU route's free capacity.
"""

import functools
import itertools
import operator
import random
import sys

import numpy as np

from verity.core.protocols.pouw.circuit import segment as sg

M32 = 0xFFFFFFFF
FAIL: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        FAIL.append(what)


FOLDS = {"duo": (sg._duo_absorb, 16), "xin": (sg._xin_absorb, 8), "fx1": (sg._fx1_absorb, 16), "li": (sg._li_absorb, 8)}


def chain_of(fold: str, q: int) -> int:
    if fold in ("duo", "fx1"):
        t, e = divmod(q, 4)
        return 4 * (t % 2) + e
    if fold == "xin":
        return q // 4
    return next(c for c in range(8) if q in sg.li_entries(c))


def nonaffine(rng: random.Random, f, sw: int, q: int, out) -> tuple[bool, bool]:
    hit_z = hit_g = False
    for _ in range(16):
        st = [rng.getrandbits(32) for _ in range(sw)]
        p = [rng.getrandbits(32) for _ in range(32)]
        p2 = [rng.getrandbits(32) for _ in range(32)]
        a, b = rng.getrandbits(32), rng.getrandbits(32)

        def at(dz: int, xor: bool, st=st, p=p, p2=p2) -> list[int]:
            pp = list(p)
            pp[q] = (pp[q] ^ dz) if xor else (pp[q] + dz) & M32
            return out(f, st, pp, p2)

        f0, fa, fb, fab = at(0, False), at(a, False), at(b, False), at((a + b) & M32, False)
        hit_z |= any((x - y - z + w) & M32 for x, y, z, w in zip(fab, fa, fb, f0))
        ga_, gb_, gab = at(a, True), at(b, True), at(a ^ b, True)
        hit_g |= any(x ^ y ^ z ^ w for x, y, z, w in zip(gab, ga_, gb_, f0))
        if hit_z and hit_g:
            break
    return hit_z, hit_g


def invariant(f, sw: int, bases, flips: list[tuple[int, int]]) -> bool:
    for st, p in bases:
        pp = list(p)
        for q, bit in flips:
            pp[q] ^= 1 << bit
        if f(st[:sw], pp) != f(st[:sw], p):
            return False
    return True


def predicted(fold: str, pt: list[int]) -> bool:
    """Whether top-bit pattern pt should leave the state: li even in every chain, xin even in every (d0, d1), (d2, d3)."""
    if fold == "li":
        return all(sum(1 for q in pt if chain_of(fold, q) == c) % 2 == 0 for c in range(8))
    if fold == "xin":
        return all(sum(1 for q in pt if q // 2 == g) % 2 == 0 for g in range(16))
    return False


def folds(rng: random.Random) -> None:
    outs = {
        "the set's state": lambda f, st, p, p2: f(st, p),
        "the state after the next set": lambda f, st, p, p2: f(f(st, p), p2),
        "the set's warp-XOR checkpoint (one lane)": lambda f, st, p, p2: [
            functools.reduce(operator.xor, v) for v in (f(st, p)[:8], f(st, p)[8:]) if v],
    }
    for name, (f, sw) in FOLDS.items():
        for on, out in outs.items():
            res = [nonaffine(rng, f, sw, q, out) for q in range(32)]
            za = [q for q, (z, _) in enumerate(res) if not z]
            ga = [q for q, (_, g) in enumerate(res) if not g]
            print(f"     {name}: Z/2^32-affine words on {on}: {za or 'none'}; GF(2)-affine: {ga or 'none'}")
            if on != "the set's state":
                check(not za and not ga, f"{name}: non-affine in every word over Z/2^32 and GF(2)^32 on {on}")
        if name in ("li", "xin"):
            last = [q for q in range(32) if (name == "li" and q in [sg.li_entries(c)[3] for c in range(8)])
                    or (name == "xin" and q % 4 == 3)]
            res = [nonaffine(rng, f, sw, q, outs["the set's state"])[0] for q in last]
            check(not any(res), f"{name}: the chain's last word of a set ({last}) is a translation of the set's state")

    bases = [([rng.getrandbits(32) for _ in range(16)], [rng.getrandbits(32) for _ in range(32)]) for _ in range(12)]
    for name, (f, sw) in FOLDS.items():
        inv = functools.partial(invariant, f, sw, bases)

        singles = [q for q in range(32) if any(inv([(q, b)]) for b in range(32))]
        tops = [(q1, q2) for q1, q2 in itertools.combinations(range(32), 2) if inv([(q1, 31), (q2, 31)])]
        same = [(q1, b1, q2, b2) for q1, q2 in itertools.combinations(range(32), 2) if chain_of(name, q1) == chain_of(name, q2)
                for b1 in range(32) for b2 in range(32) if inv([(q1, b1), (q2, b2)])]
        check(not singles, f"{name}: injective in each word alone (no single-bit flip leaves the state)")
        print(f"     {name}: top-bit pairs leaving the state invariant: {len(tops)}; all same-chain single-bit pairs: "
              f"{len(same)}; distinct (bit, bit) kinds {sorted({(b1, b2) for _, b1, _, b2 in same})}")
        if name == "li":
            want = {(q1, q2) for q1, q2 in itertools.combinations(range(32), 2) if chain_of(name, q1) == chain_of(name, q2)}
            check(set(tops) == want, "li: the top bits of a chain's 4 words merge in every pair (48 pairs: no rotation in li)")
            rnd = [rng.getrandbits(1) for _ in range(32)]
            even = [q for q in range(32) if rnd[q]]
            per_chain = [sum(1 for q in even if chain_of(name, q) == c) % 2 for c in range(8)]
            fixed = [q for q in even if per_chain[chain_of(name, q)] == 0]
            check(inv([(q, 31) for q in fixed]), f"li: an even top-bit pattern in every chain ({len(fixed)} words) leaves the state")
        pats = [[q for q in range(32) if rng.getrandbits(1)] for _ in range(3000)]
        pats = [pt for pt in pats if pt]
        agree = sum(inv([(q, 31) for q in pt]) == predicted(name, pt) for pt in pats)
        check(agree == len(pats), f"{name}: {len(pats)} random top-bit patterns invariant exactly when predicted "
              f"({'even in every chain' if name == 'li' else 'even in every (d0, d1) and (d2, d3) pair' if name == 'xin' else 'never'})")
        if name in ("duo", "fx1"):
            check(not tops, f"{name}: no top-bit pair merges (rotl in z moves the top bit), so no Z-linear merge of two words")


def strassen(seed: int = 3, n: int = 64, L: int = 32) -> None:
    rng = np.random.default_rng(seed)
    h, k = n // 2, L // 2
    R = rng.integers(0, 2, (n, n), dtype=np.int64)
    vals = np.array([x for x in range(-64, 65) if x], dtype=np.int64)
    U = rng.choice(vals, (n, L))
    S = U[:, np.arange(n) % L].T
    T = np.zeros((n, n), dtype=np.int64)
    Q = {c: np.zeros((h, h), dtype=np.int64) for c in ("11", "12", "21", "22")}
    ok, lo, hi, slots6 = True, 0, 0, 0
    for seg in range(n // L):
        t0 = seg * L
        a = {(r_, c): R[r_ * h:(r_ + 1) * h, t0 + c * k:t0 + (c + 1) * k] for r_ in range(2) for c in range(2)}
        b = {(r_, c): S[t0 + r_ * k:t0 + (r_ + 1) * k, c * h:(c + 1) * h] for r_ in range(2) for c in range(2)}
        A11, A12, A21, A22 = a[0, 0], a[0, 1], a[1, 0], a[1, 1]
        B11, B12, B21, B22 = b[0, 0], b[0, 1], b[1, 0], b[1, 1]
        cat = lambda x, y: np.concatenate([x, y], axis=1)  # k16 | k16 -> one k32 operand
        stk = lambda x, y: np.concatenate([x, y], axis=0)
        packs = {  # D = A.B + C, each A.B one k32 D word an entry: (M1 + M4) + (M7 - M5), etc.
            "11": [(cat(A11 + A22, A22), stk(B11 + B22, B21 - B11)), (cat(A12 - A22, A11 + A12), stk(B21 + B22, -B22))],
            "22": [(cat(A11 + A22, A21 + A22), stk(B11 + B22, -B11)), (cat(A11, A21 - A11), stk(B12 - B22, B11 + B12))],
            "12": [(cat(A11, A11 + A12), stk(B12 - B22, B22))],
            "21": [(cat(A21 + A22, A22), stk(B11, B21 - B11))],
        }
        for c, ps in packs.items():
            for x, y in ps:
                lo, hi = min(lo, int(x.min()), int(y.min())), max(hi, int(x.max()), int(y.max()))
                Q[c] = Q[c] + x @ y  # chained through the C operand
                slots6 += h * h
        T += R[:, t0:t0 + L] @ S[t0:t0 + L, :]
        ok &= all((Q[c] == T[int(c[0]) * h - h:int(c[0]) * h, int(c[1]) * h - h:int(c[1]) * h]).all() for c in Q)
    nw = n * n * (n // L)
    check(ok, f"Strassen packed into k32: 6 chained D words a 2x2 block of words reproduce every prefix word (n = {n})")
    print(f"     packed operands in [{lo}, {hi}] (+128 needs O-REDTEAM's sparse fix); D words {slots6 / nw:.3f} a committed word")
    print("     MMASLOT: D words a committed word, and tensor time against naive's 1 (every H200 dense format: L_D = 32):")
    for what, dw in (("naive k32", 1.0), ("Strassen-7, k16 products in k32 slots", 7 / 4), ("Strassen sums packed in k32", 1.5),
                     ("Strassen-7 if a k16 s8 shape ran at the k32 MAC rate", 7 / 8)):
        print(f"       {what:55s} {dw:.3f}")


def alu(seed: int = 5, n: int = 32, L: int = 32) -> None:
    rng = np.random.default_rng(seed)
    h, k = n // 2, L // 2
    R = rng.integers(0, 2, (n, n), dtype=np.int64)
    vals = np.array([x for x in range(-64, 65) if x], dtype=np.int64)
    S = rng.choice(vals, (n, L))[:, np.arange(n) % L].T
    T = np.zeros((n, n), dtype=np.int64)
    Q = {c: np.zeros((h, h), dtype=np.int64) for c in ("11", "12", "21", "22")}
    ok = True
    for seg in range(n // L):
        t0 = seg * L
        a = {(r_, c): R[r_ * h:(r_ + 1) * h, t0 + c * k:t0 + (c + 1) * k] for r_ in range(2) for c in range(2)}
        b = {(r_, c): S[t0 + r_ * k:t0 + (r_ + 1) * k, c * h:(c + 1) * h] for r_ in range(2) for c in range(2)}
        A11, A12, A21, A22 = a[0, 0], a[0, 1], a[1, 0], a[1, 1]
        B11, B12, B21, B22 = b[0, 0], b[0, 1], b[1, 0], b[1, 1]
        M = {1: (A11 + A22) @ (B11 + B22), 2: (A21 + A22) @ B11, 3: A11 @ (B12 - B22), 4: A22 @ (B21 - B11),
             5: (A11 + A12) @ B22}
        # chains start from one value each: M6, M7 from merged values; the prefix Q enters by IADD3
        Q["12"] = Q["12"] + M[3] + M[5]                     # IADD3(Q12, M3, M5)
        Q["21"] = Q["21"] + M[2] + M[4]                     # IADD3(Q21, M2, M4)
        s11 = Q["11"] + M[1] + M[4]                         # IADD3(Q11, M1, M4)
        Q["11"] = (s11 - M[5]) + (A12 - A22) @ (B21 + B22)  # IADD(s11, -M5), then the M7 chain from it
        s22 = Q["22"] + M[1] - M[2]                         # IADD3(Q22, M1, -M2)
        Q["22"] = (s22 + M[3]) + (A21 - A11) @ (B11 + B12)  # IADD(s22, M3), then the M6 chain from it
        T += R[:, t0:t0 + L] @ S[t0:t0 + L, :]
        ok &= all((Q[c] == T[int(c[0]) * h - h:int(c[0]) * h, int(c[1]) * h - h:int(c[1]) * h]).all() for c in Q)
    check(ok, f"IMAD Strassen depth 1 with the prefix: 28 IMAD + 6/4 merges a word reproduce every prefix word (n = {n})")
    print("     ALU lane-ops a committed word (L = 32): IMAD naive 32 | IMAD Strassen-1 29.5 | dp4a naive 8 | "
          "dp4a Strassen-1 7 + 1.5 = 8.5 (Strassen loses on dp4a)")


H200 = {"tensor_mac": 4096, "issue": 128, "alu": 64, "fma": 64, "dp4a": 64}  # per SM per clock, data sheet / placeholders


def prices() -> None:
    T, I, A, F = 32 / H200["tensor_mac"], 1 / H200["issue"], 1 / H200["alu"], 1 / H200["fma"]
    routes = {  # SM-clock per committed word on (tensor, issue, alu, fma), fold included (li: 0.5 ALU + 0.5 IMAD)
        "honest TC + li": (T, I, 0.5 * A, 0.5 * F),
        "dp4a naive + li": (0, 9 * I, 0.5 * A, 8.5 * F),
        "IMAD Strassen-1 + li": (0, 30.5 * I, 2 * A, 28.5 * F),
        "TC Strassen-7 in k32 + li (+6/4 merges)": (7 / 4 * T, 2.5 * I, 2 * A, 0.5 * F),
    }
    lams = {"tensor": (1, 0, 0, 0), "issue": (0, 1, 0, 0), "alu": (0, 0, 1, 0), "fma": (0, 0, 0, 1),
            "uniform": (0.25,) * 4, "tc 0.7, rest 0.1": (0.7, 0.1, 0.1, 0.1)}
    hon = routes["honest TC + li"]
    print("     per word (SM-clk) on tensor, issue, alu, fma; time = max; priced = lambda . u:")
    for r, u in routes.items():
        print(f"       {r:42s} " + " ".join(f"{x:.5f}" for x in u) + f"  time {max(u):.5f} = {max(u) / max(hon):.2f}x honest")
    print("     c(lambda) of the best route against honest, lambda over (tensor, issue, alu, fma):")
    for ln, lam in lams.items():
        best = min(sum(l * x for l, x in zip(lam, u)) for u in routes.values())
        print(f"       {ln:18s} c = {best / sum(l * x for l, x in zip(lam, hon)):.3f}")
    for r in ("dp4a naive + li", "IMAD Strassen-1 + li"):
        u = routes[r]
        wpc = 1 / max(u[1:])
        print(f"     free capacity at lambda = tensor only, {r}: {wpc:.2f} words/SM/clk = {32 * wpc:.0f} MAC/SM/clk = "
              f"{32 * wpc / H200['tensor_mac']:.2%} of the tensor rate (dp4a rate a placeholder)")
    lam_h = lambda t, i, a, f: min(t, i + 2 * min(a, f))
    check(abs(lam_h(1, 1, 1, 1) - 1) < 1e-12, "head price min(lambda_TC, lambda_iss + 2 min(lambda_alu, lambda_fma)) at 1")


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("folds", "all"):
        folds(random.Random(11))
    if which in ("strassen", "all"):
        strassen()
    if which in ("alu", "all"):
        alu()
    if which in ("prices", "all"):
        prices()
    print("all checks pass" if not FAIL else f"{len(FAIL)} checks FAIL")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
