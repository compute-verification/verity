"""red-team-leaf: independent security estimate for the Ajtai / Ring-SIS leaf (lane ajtai-design, 2a61a1f).

Instance: h = sum_i b_i(X) g_i(X) in F_p[X]/(X^n + 1), p = 2^31 - 2^27 + 1, w = 256 ring elements b_i, g_i with 0/1
coefficients of degree < steps <= n.  A collision is a nonzero ternary gamma in Z^m (m = w * steps) with A gamma = 0 mod p,
A the n x m unrolled matrix.  This script re-derives every number of the design note from scratch (no import of the lane's
estimator) and adds the checks the design did not make:

  1. the dimension below which NO ternary kernel vector exists (3^d < p^n) -- M1's optimum d* lies there, so M1 is a
     lower bound on the pure-lattice attack, not the attack;
  2. the pure-lattice attack restricted to dimensions where ternary solutions exist, with the *actual* norm of the
     sparsest ternary solution as the target (entropy counting), i.e. what BKZ must reach to output a collision;
  3. the CRT / ideal sub-instance trade-off (X^n + 1 splits completely mod p): zeroing s CRT slots by forcing every
     gamma_i into the ideal I_S costs p^s of the per-polynomial input entropy while removing s log p bits of constraint --
     shown to be monotone non-improving for the generalised birthday attack and volume-neutral for the lattice attack;
  4. Wagner's k-tree with the exact list-size feasibility per relation (fp8 / bf16 / fp4);
  5. the SWIFFT / Buchmann-Lindner "calibration" the design leans on: their 68-bit figure is for PSEUDO-collisions
     (l2 <= sqrt m), i.e. an M0-type target, under the 2009 Lenstra-Verheul BKZ extrapolation -- not Core-SVP and not a
     ternary target; the agreement with M1 is a coincidence of models and is not evidence for M1.

Run: `python redteam_ajtai_estimate.py` (~5 s).  Every number in the red-team note's F-findings comes from here.
"""
from __future__ import annotations

import math

P = 15 * 2**27 + 1
LOG2P = math.log2(P)
C_CLS, C_QNT, C_PLS = 0.292, 0.265, 0.2075


def delta_bkz(b: float) -> float:
    return ((math.pi * b) ** (1.0 / b) * b / (2 * math.pi * math.e)) ** (1.0 / (2.0 * b - 2.0))


def h2(x: float) -> float:
    """binary entropy"""
    if x <= 0 or x >= 1:
        return 0.0
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def log2_ternary_count(d: int, w: int) -> float:
    """log2 of the number of ternary vectors in Z^d with exactly w nonzero entries: C(d, w) 2^w."""
    if w <= 0 or w > d:
        return 0.0 if w == 0 else -math.inf
    return math.lgamma(d + 1) / math.log(2) - math.lgamma(w + 1) / math.log(2) - math.lgamma(d - w + 1) / math.log(2) + w


def sparsest_ternary_weight(d: int, n: int, log2q: float = LOG2P) -> int | None:
    """Smallest weight w such that the expected number of weight-w ternary kernel vectors in d coordinates, C(d,w) 2^w / q^n,
    is >= 1 (the kernel of a random n x d matrix mod q behaves like a random code of co-dimension n log q bits)."""
    need = n * log2q
    lo, hi = 1, d
    if log2_ternary_count(d, d) + 0 < need and math.log2(3) * d < need:
        return None
    # cumulative count sum_{w' <= w} C(d,w') 2^w'  ~ dominated by the largest term below d*2/3; use the max-term proxy
    for w in range(1, d + 1):
        if log2_ternary_count(d, w) >= need:
            return w
    return None


def d_min_ternary(n: int, log2q: float = LOG2P) -> int:
    """Below this dimension no ternary kernel vector is expected at all (3^d < q^n)."""
    return math.ceil(n * log2q / math.log2(3))


def bkz_norm_log2(b: int, d: int, n: int, log2q: float = LOG2P) -> float:
    """log2 of the norm BKZ-b reaches in a d-dim q-ary kernel lattice of volume q^n (d > n): delta^d q^(n/d)."""
    return d * math.log2(delta_bkz(b)) + n * log2q / d


def m0_blocksize(n: int, m: int, log2q: float = LOG2P) -> int:
    """LaBRADOR / Micciancio-Regev proof-norm rule: vector of norm sqrt(m) in the best dimension."""
    lb2 = 0.5 * math.log2(m)
    for b in range(60, 8000):
        if 2 * math.sqrt(n * log2q * math.log2(delta_bkz(b))) <= lb2:
            return b
    return 8000


def m1_blocksize(n: int, m: int, log2q: float = LOG2P) -> tuple[int, int]:
    """Design's M1: exists d <= m with delta^d q^(n/d) <= sqrt(d) (max l2 norm of a ternary vector on d coordinates)."""
    for b in range(60, 8000):
        ld = math.log2(delta_bkz(b))
        d0 = int(math.sqrt(n * log2q / ld))
        for d in sorted(set([n + 1, m] + [max(n + 1, min(m, d0 + k)) for k in range(-200, 201)])):
            if bkz_norm_log2(b, d, n, log2q) <= 0.5 * math.log2(d):
                return b, d
    return 8000, 0


def m1_ternary_exists_blocksize(n: int, m: int, log2q: float = LOG2P, target: str = "sparsest") -> tuple[int, int, float]:
    """Pure-lattice attack that can actually output a collision: smallest b such that for some d in [d_min, m] BKZ-b reaches
    the norm of the sparsest ternary kernel vector expected in d coordinates ("sparsest"), or sqrt(2d/3) (a typical ternary
    vector, "typical")."""
    dmin = d_min_ternary(n, log2q)
    cache: dict[int, float] = {}

    def tgt(d: int) -> float:
        if d not in cache:
            if target == "sparsest":
                w = sparsest_ternary_weight(d, n, log2q)
                cache[d] = 0.5 * math.log2(w) if w else -math.inf
            else:
                cache[d] = 0.5 * math.log2(2 * d / 3)
        return cache[d]

    ds = sorted(set(list(range(dmin, min(m, dmin + 4000), 8)) + [m]))
    for b in range(60, 8000):
        best = None
        for d in ds:
            if bkz_norm_log2(b, d, n, log2q) <= tgt(d):
                best = d
                break
        if best is not None:
            return b, best, tgt(best)
    return 8000, 0, 0.0


def wagner(n: int, m: int, log2q: float = LOG2P, log2_alphabet: float = math.log2(3)) -> tuple[int, float, float]:
    """k-tree: 2^k lists, each of (2 beta + 1)^(m / 2^k) elements; each level zeroes n log q / (k + 1) bits.  Returns
    (k, log2 cost, slack bits = available list entropy - required)."""
    best = (0, math.inf, 0.0)
    for k in range(1, 40):
        need = n * log2q / (k + 1)
        avail = (m / 2**k) * log2_alphabet
        if avail < need:
            continue
        cost = k + need
        if cost < best[1]:
            best = (k, cost, avail - need)
    return best


def crt_ideal_tradeoff(n: int, steps: int, w: int, log2q: float = LOG2P):
    """Force every gamma_i into I_S = {f : f(zeta_k) = 0 for k in S}, |S| = s.  Ternary polynomials of degree < steps in I_S:
    3^steps / q^s of them (the evaluation map at s <= steps points is onto F_q^s).  Remaining constraint: (n - s) log q bits.
    Generalised birthday over w lists (one per bit position), each list = the ternary polynomials in I_S, merged k-tree style
    with 2^k groups of w / 2^k lists each (a group's list = products of its members' lists)."""
    rows = []
    for s in range(0, min(steps, n) + 1):
        ent = steps * math.log2(3) - s * log2q      # log2 |ternary ∩ I_S| per polynomial
        if ent < 0:
            break
        need_bits = (n - s) * log2q
        best = (0, math.inf)
        for k in range(1, int(math.log2(w)) + 1):
            avail = (w / 2**k) * ent
            need = need_bits / (k + 1)
            if avail < need:
                continue
            cost = k + need
            if cost < best[1]:
                best = (k, cost)
        # plus the cost of enumerating I_S ∩ ternary per polynomial, ~ 2^ent (a lattice/ISD enumeration, cheap if ent small)
        rows.append((s, ent, need_bits, best[0], best[1]))
    return rows


def main() -> None:
    print(f"p = {P}, log2 p = {LOG2P:.3f}")
    print()
    print("== A. delta anchors and the design's M0 / M1 (re-derived) ==")
    for b in (341, 391, 438, 976, 1925):
        print(f"  BKZ-{b}: delta = {delta_bkz(b):.5f}  log2 delta = {math.log2(delta_bkz(b)):.5f}  cls {C_CLS*b:.0f} / qnt {C_QNT*b:.0f}")
    for n, m, rel in ((64, 12288, "fp8"), (64, 6912, "fp4"), (128, 24576, "bf16"), (128, 12288, "fp8@n128")):
        b0 = m0_blocksize(n, m)
        b1, d1 = m1_blocksize(n, m)
        print(f"  n={n:3d} m={m:5d} ({rel:8s}): M0 b={b0:4d} -> {C_CLS*b0:.0f}/{C_QNT*b0:.0f};  M1 b={b1:4d} d*={d1:4d} -> {C_CLS*b1:.0f}/{C_QNT*b1:.0f}")

    print()
    print("== B. where ternary kernel vectors exist at all ==")
    for n in (64, 128):
        dmin = d_min_ternary(n)
        b1, d1 = m1_blocksize(n, 24576)
        print(f"  n={n}: 3^d >= p^n needs d >= {dmin};  M1's optimum d* = {d1} {'<' if d1 < dmin else '>='} d_min "
              f"-> M1 {'declares success in a dimension with NO ternary solution' if d1 < dmin else 'is in the feasible regime'}")
        for d in (dmin + 8, dmin + 250, dmin + 750, 2000, 3000, 6000, 12288):
            w = sparsest_ternary_weight(d, n)
            gh = 0.5 * math.log2(d / (2 * math.pi * math.e)) + n * LOG2P / d
            ws = f"~{w} (norm 2^{0.5*math.log2(w):.2f})" if w else "none"
            print(f"     d={d:5d}: sparsest ternary weight {ws}, typical ternary norm 2^{0.5*math.log2(2*d/3):.2f}, "
                  f"GH lambda1 2^{gh:.2f}")

    print()
    print("== C. pure-lattice attack that outputs a collision (d >= d_min, target = sparsest / typical ternary norm) ==")
    for n, m in ((64, 12288), (64, 24576), (128, 24576)):
        for tgt in ("sparsest", "typical"):
            b, d, lt = m1_ternary_exists_blocksize(n, m, target=tgt)
            print(f"  n={n:3d} m={m:5d} target={tgt:8s}: BKZ-{b} in d={d} (target norm 2^{lt:.2f}) -> {C_CLS*b:.0f} cls / {C_QNT*b:.0f} qnt / {C_PLS*b:.0f} pls")

    print()
    print("== D. Wagner k-tree per relation ==")
    for n, m, rel in ((64, 12288, "fp8"), (64, 6912, "fp4"), (64, 24576, "bf16@n64 (forbidden: steps>n)"), (128, 24576, "bf16"),
                      (128, 12288, "fp8@n128"), (32, 12288, "n32 fp8")):
        k, c, slack = wagner(n, m)
        print(f"  n={n:3d} m={m:5d} {rel:28s}: k={k} cost 2^{c:.0f}  (list slack {slack:.0f} bits)")

    print()
    print("== E. CRT / ideal sub-instance trade-off (n=64, steps=48, w=256): zero s slots by gamma_i in I_S ==")
    for s, ent, need, k, cost in crt_ideal_tradeoff(64, 48, 256):
        print(f"  s={s}: per-poly ternary entropy in I_S = {ent:6.1f} bits, remaining constraint {need:7.1f} bits, best k={k}, cost 2^{cost:.1f}")
    print("  (s = 0 is the plain k-tree; the cost never decreases with s: the ideal restriction buys nothing)")
    print("  Lattice view: the kernel sublattice on d coordinates has volume p^n whatever the CRT structure (rank n whenever the d")
    print("  columns span F_p^n, which two bit positions' 96 rotations already do); BKZ sees the same lattice. No subring attack.")

    print()
    print("== F. SWIFFT (64, m=1024, q=257) and Buchmann-Lindner (96, m=1728, q=389) under each rule ==")
    for n, m, q, label in ((64, 1024, 257, "SWIFFT   "), (96, 1728, 389, "BL-2009  ")):
        lq = math.log2(q)
        b0 = m0_blocksize(n, m, lq)
        b1, d1 = m1_blocksize(n, m, lq)
        bt, dt, lt = m1_ternary_exists_blocksize(n, m, lq, "sparsest")
        k, c, _ = wagner(n, m, lq)
        print(f"  {label}: M0 (pseudo-collision sqrt m in best d) b={b0} -> {C_CLS*b0:.0f} cls;  M1 b={b1} d*={d1} -> {C_CLS*b1:.0f} cls;  "
              f"ternary-exists b={bt} d={dt} -> {C_CLS*bt:.0f} cls;  Wagner k={k} 2^{c:.0f};  d_min={d_min_ternary(n, lq)}")
    print("  BL09 report 68 bits for SWIFFT pseudo-collisions (l2 <= 32 = sqrt 1024, NOT ternary) from BKZ runs + Lenstra-Verheul")
    print("  extrapolation; the M0-sublattice target under Core-SVP is ~40 bits, M1 71 bits.  M1 'matching' 68 is a coincidence of two")
    print("  unrelated models (different target norm, different cost model); it is not a calibration of M1.")


if __name__ == "__main__":
    main()
