"""Ajtai / Ring-SIS operand-commitment leaf over BabyBear: seeded matrix derivation and security estimates.

The leaf (`leaf/ajtai.py`, lane ajtai-leaf) hashes a committed operand row as

    h = sum_{j < steps} X^(steps-1-j) * B * s_j   in  R_p = F_p[X] / (X^n + 1),   p = 2^31 - 2^27 + 1,

where `s_j` in {0,1}^w are the `w` operand bits of chain column `j` (w = 256 for the fp8 and bf16 relations) and
`B = (b_0, ..., b_{w-1})` are `w` seeded ring elements.  Equivalently `h = sum_i b_i(X) g_i(X)` with
`g_i(X) = sum_j X^(steps-1-j) s_{j,i}` a polynomial with 0/1 coefficients of degree < steps <= n, i.e. Ring-SIS in
the SWIFFT shape (Lyubashevsky-Micciancio-Peikert-Rosen, FSE 2008) with binary inputs.  A collision is a nonzero
`gamma = s - s'` in {-1,0,1}^(w*steps) with `A gamma = 0 mod p` for the n x (w*steps) matrix
`A = [X^(steps-1) B | ... | X B | B]` (negacyclic rotations of the b_i): an SIS_{n, m = w*steps, p} solution with
`||gamma||_inf <= 1`, `||gamma||_2 <= sqrt(m)`.

Estimators (all Core-SVP: classical 2^(0.292 b), quantum 2^(0.265 b), "plausible" floor 2^(0.2075 b), BKZ block size b,
root-Hermite factor delta_b = ((pi b)^(1/b) b / (2 pi e))^(1/(2(b-1)))):

  M0  proof-norm rule: the MR formula with the l2 bound of the *reduction*, beta_2 = sqrt(m): the attack succeeds at b iff
      2 sqrt(n log2 p log2 delta_b) <= log2 beta_2  (Micciancio-Regev 2009 Sec. 3.5; verbatim the `sis_secure` of the
      lattice-dogs LaBRADOR/Greyhound implementation with their delta = 1.00444).  Over-conservative for a ternary target.
  M1  attack-dimension l2 rule: the attacker zeroes all but d coordinates and needs a vector of l2 norm <= beta_inf sqrt(d)
      there (the largest l2 norm a [-beta, beta] vector on d coordinates can have): success at b iff exists d <= m with
      d log2 delta_b + n log2 p / d <= log2 beta_inf + log2(d) / 2.
  M2  the Dilithium l_inf model (Ducas et al., Dilithium spec App. C.3; pq-crystals/security-estimates MSIS_security.py
      `SIS_linf_cost`, ported verbatim): randomized BKZ shape, coordinates of the found vector Gaussian with
      sigma = l / sqrt(d), probability that all d are <= beta_inf; cost = 2^(c b) / (probability * 2^(0.2075 b)).
  Wagner  generalised birthday (Wagner 2002; the SWIFFT paper's own estimate): 2^k lists of (2 beta + 1)^(m / 2^k)
      ternary combinations, cost ~ 2^(k + n log2 p / (k + 1)).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

P = 15 * 2**27 + 1
LOG2P = math.log2(P)

C_CLASSICAL = math.log2(math.sqrt(3 / 2))     # 0.2925  Becker-Ducas-Gama-Laarhoven 2016
C_QUANTUM = math.log2(math.sqrt(13 / 9))      # 0.2653  Laarhoven 2015
C_PLAUSIBLE = math.log2(math.sqrt(4 / 3))     # 0.2075  the sphere-covering floor (Dilithium spec 6.1)

__all__ = ["P", "LOG2P", "C_CLASSICAL", "C_QUANTUM", "C_PLAUSIBLE", "Estimate", "delta_bkz", "derive_ring_element",
           "derive_B", "estimate", "m0_blocksize", "m1_blocksize", "m2_cost", "wagner_bits", "lhl_random_bits"]


# ------------------------------------------------------------------------------------------------------
# Seeded key derivation
# ------------------------------------------------------------------------------------------------------

DOMAIN = b"verity/ajtai-babybear/B/v1"


def derive_ring_element(i: int, n: int, domain: bytes = DOMAIN) -> list[int]:
    """Coefficients of b_i: the SHAKE-256 XOF of ``domain || LE32(i)`` read as LE32 words w, v = w & 0x7FFF_FFFF,
    rejecting v >= p (acceptance 15/16); coefficient k is the k-th accepted value.  Prefix-consistent: the b_i for a
    smaller n is a prefix of the b_i for a larger n, and B for fewer bits per column is a prefix of B for more."""
    xof = hashlib.shake_256(domain + int(i).to_bytes(4, "little"))
    out: list[int] = []
    chunk = 0
    while len(out) < n:
        chunk += 1
        buf = xof.digest(4 * 8 * n * chunk)[4 * 8 * n * (chunk - 1):]
        for k in range(0, len(buf), 4):
            v = int.from_bytes(buf[k:k + 4], "little") & 0x7FFF_FFFF
            if v < P:
                out.append(v)
                if len(out) == n:
                    break
    return out


def derive_B(w: int, n: int, domain: bytes = DOMAIN) -> list[list[int]]:
    """The w ring elements (each n coefficients) of the leaf: B[i] = derive_ring_element(i, n)."""
    return [derive_ring_element(i, n, domain) for i in range(w)]


# ------------------------------------------------------------------------------------------------------
# BKZ model
# ------------------------------------------------------------------------------------------------------

def delta_bkz(b: float) -> float:
    """Root-Hermite factor of BKZ-b (Chen 2013 / pq-crystals model_BKZ.delta_BKZ)."""
    return ((math.pi * b) ** (1.0 / b) * b / (2 * math.pi * math.e)) ** (1.0 / (2.0 * b - 2.0))


def m0_blocksize(n: int, m: int, beta_inf: int = 1, log2q: float = LOG2P) -> int:
    """M0: smallest b with 2 sqrt(n log2q log2 delta_b) <= log2(beta_inf sqrt(m))."""
    lb2 = math.log2(beta_inf) + 0.5 * math.log2(m)
    for b in range(60, 6000):
        if 2 * math.sqrt(n * log2q * math.log2(delta_bkz(b))) <= lb2:
            return b
    return 6000


def m1_blocksize(n: int, m: int, beta_inf: int = 1, log2q: float = LOG2P) -> tuple[int, int]:
    """M1: smallest b such that some d <= m has delta_b^d q^(n/d) <= beta_inf sqrt(d); returns (b, d)."""
    lbi = math.log2(beta_inf)
    for b in range(60, 6000):
        ld = math.log2(delta_bkz(b))
        # d log delta + n log q / d - log(d)/2 is convex in d on [n+1, m]; check around its minimiser and the ends
        d0 = int(math.sqrt(n * log2q / ld))
        cands = sorted(set([n + 1, m] + [max(n + 1, min(m, d0 + k)) for k in range(-64, 65)]))
        for d in cands:
            if ld * d + n * log2q / d <= lbi + 0.5 * math.log2(d):
                return b, d
    return 6000, 0


def _bkz_shape_randomized(q: float, nq: int, n1: int, b: int):
    glv = nq * math.log(q)
    d = nq + n1
    L: list[float] = []
    slope = -2 * math.log(delta_bkz(b))
    li = 0.0
    lv = 0.0
    for _ in range(d):
        li -= slope
        lv += li
        if lv > glv:
            break
        L = [li] + L
    B = len(L)
    L += (d - B) * [0.0]
    diff = sum(L) - glv
    for i in range(B):
        L[i] -= diff / B
    return (0, B, L)


def m2_cost(n: int, m: int, beta_inf: int, b: int, c_svp: float, q: int = P) -> float:
    """M2 (pq-crystals SIS_linf_cost): log2 cost of the l_inf attack with BKZ-b on the n x m instance."""
    (i, j, L) = _bkz_shape_randomized(q, n, m - n, b)
    l = math.exp(L[i])
    d = j - i + 1
    sigma = l / math.sqrt(d)
    p_middle = math.erf(beta_inf / (sigma * math.sqrt(2.0)))
    if p_middle <= 0:
        return 9999.0
    log2_eps = d * math.log2(p_middle)
    log2_R = max(0.0, -log2_eps - C_PLAUSIBLE * b)
    return c_svp * b + log2_R


def m2_optimize(n: int, m: int, beta_inf: int, c_svp: float, q: int = P) -> tuple[int, float]:
    best = (0, 9999.0)
    for b in range(60, min(m, 6000)):
        if c_svp * b > best[1]:
            break
        cost = m2_cost(n, m, beta_inf, b, c_svp, q)
        if cost <= best[1]:
            best = (b, cost)
    return best


def wagner_bits(n: int, m: int, beta_inf: int = 1, log2q: float = LOG2P) -> tuple[int, float]:
    """Generalised birthday: min over k of k + n log2q/(k+1) s.t. (2 beta + 1)^(m/2^k) >= 2^(n log2q/(k+1))."""
    out_bits = n * log2q
    best = (0, 9999.0)
    for k in range(1, 40):
        list_bits = out_bits / (k + 1)
        avail = (m / 2**k) * math.log2(2 * beta_inf + 1)
        if avail < list_bits:
            continue
        cost = k + list_bits
        if cost < best[1]:
            best = (k, cost)
    return best


def lhl_random_bits(n: int, lam: int = 128, log2q: float = LOG2P) -> int:
    """Leftover-hash-lemma randomness for the hiding variant h = A s + B_r r: statistical distance <= 2^-lam needs
    m_r >= n log2 q + 2 lam uniform bits of r."""
    return math.ceil(n * log2q + 2 * lam)


# ------------------------------------------------------------------------------------------------------
# Report
# ------------------------------------------------------------------------------------------------------

@dataclass
class Estimate:
    n: int
    m: int
    beta: int
    m0_b: int
    m1_b: int
    m1_d: int
    m2_b: int
    m2_classical: float
    m2_quantum: float
    m2_plausible: float
    wagner_k: int
    wagner: float

    @property
    def digest_bits(self) -> float:
        return self.n * LOG2P

    def bits(self, model: str, c: float) -> float:
        b = {"m0": self.m0_b, "m1": self.m1_b}[model]
        return c * b

    def row(self) -> str:
        return (f"| {self.beta} | {self.n} | {self.m} | {self.m0_b} | {C_CLASSICAL*self.m0_b:.0f} / {C_QUANTUM*self.m0_b:.0f} / {C_PLAUSIBLE*self.m0_b:.0f} "
                f"| {self.m1_b} ({self.m1_d}) | {C_CLASSICAL*self.m1_b:.0f} / {C_QUANTUM*self.m1_b:.0f} / {C_PLAUSIBLE*self.m1_b:.0f} "
                f"| {self.m2_b} | {self.m2_classical:.0f} / {self.m2_quantum:.0f} / {self.m2_plausible:.0f} | {self.wagner:.0f} (k={self.wagner_k}) "
                f"| {self.n * 4} |")


def estimate(n: int, m: int, beta: int = 1) -> Estimate:
    b0 = m0_blocksize(n, m, beta)
    b1, d1 = m1_blocksize(n, m, beta)
    bc, cc = m2_optimize(n, m, beta, C_CLASSICAL)
    _, cq = m2_optimize(n, m, beta, C_QUANTUM)
    _, cp = m2_optimize(n, m, beta, C_PLAUSIBLE)
    k, wg = wagner_bits(n, m, beta)
    return Estimate(n, m, beta, b0, b1, d1, bc, cc, cq, cp, k, wg)


HEADER = ("| beta | n | m | M0 b | M0 cls/qnt/pls | M1 b (d*) | M1 cls/qnt/pls | M2 b | M2 cls/qnt/pls | Wagner | digest B |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|")


def main() -> None:
    print(f"p = {P} = 2^{LOG2P:.3f}")
    for b in (300, 350, 400, 438, 484, 600):
        print(f"  BKZ-{b}: delta={delta_bkz(b):.5f} log2delta={math.log2(delta_bkz(b)):.5f} "
              f"cls={C_CLASSICAL*b:.0f} qnt={C_QUANTUM*b:.0f} pls={C_PLAUSIBLE*b:.0f}")
    print()
    print("### beta = 1 (bits): FP8 m = 256 x 48 = 12288, BF16 m = 256 x 96 = 24576, FP4 m = 6912")
    print(HEADER)
    for m in (12288, 24576, 6912):
        for n in (32, 48, 64, 96, 128):
            print(estimate(n, m, 1).row())
    print()
    print("### nibbles (beta = 15, m = 3072) and bytes (beta = 255, m = 1536), FP8 row")
    print(HEADER)
    for beta, m in ((15, 3072), (255, 1536)):
        for n in (64, 128, 192, 256, 320):
            print(estimate(n, m, beta).row())
    print()
    print("### SWIFFT cross-check: n = 64, m = 1024, q = 257 (paper: generalised birthday 2^106, lattice attacks 'out of reach')")
    lq = math.log2(257)
    b1, d1 = m1_blocksize(64, 1024, 1, lq)
    bc, cc = m2_optimize(64, 1024, 1, C_CLASSICAL, 257)
    k, wg = wagner_bits(64, 1024, 1, lq)
    print(f"  M0 b={m0_blocksize(64, 1024, 1, lq)}; M1 b={b1} d*={d1} cls={C_CLASSICAL*b1:.0f}; M2 b={bc} cls={cc:.0f}; "
          f"Wagner k={k} {wg:.0f} bits (paper: k=4 with 2^102 lists, 2^106)")
    print()
    print("### hiding randomness (LHL, 2^-128)")
    for n in (32, 48, 64, 96, 128):
        r = lhl_random_bits(n)
        print(f"  n={n}: m_r >= {r} bits = {math.ceil(r / 48)} bit rows per fp8 column, {math.ceil(r / 96)} per bf16 column")


if __name__ == "__main__":
    main()
