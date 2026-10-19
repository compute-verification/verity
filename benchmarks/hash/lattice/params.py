"""SIS parameters for a tensor-core commitment hash h(x) = A x mod q, and what each encoding costs.

A Core-SVP estimate, not the lattice-estimator (malb/lattice-estimator imports Sage). Three attack models, the ones
ajtai-design used (note:ajtai-design 20260923T1630Z, its section 2.2), re-implemented here and held to that note's
published numbers by `crosscheck()`:

- M1, the l2 rule: BKZ-b succeeds if some sub-dimension d <= m has delta_b^d q^(n/d) <= beta sqrt(d) (and < q), i.e. the
  reduced vector is no longer than the longest vector with |x_i| <= beta. Conservative for the defender: it counts any
  such vector as a collision. It reproduces Buchmann-Lindner's SWIFFT cryptanalysis within ~3-9 bits.
- M2, Dilithium's l_inf model (pq-crystals/security-estimates, MSIS_security.SIS_linf_cost): a GSA profile over the
  kernel lattice, the found vector's coordinates Gaussian, success probability erf(beta / (sigma sqrt 2))^d, a sieve
  call yields 2^(0.2075 b) vectors, repeated until one is short in l_inf. Also the Z-shaped profile with q-vectors at
  the head (each head coordinate uniform mod q), the attacker taking the cheaper.
- Wagner's k-tree over lists of partial inputs (each coordinate one of beta + 1 values).

Cost is Core-SVP: 0.292 b classical, 0.265 b quantum. It ignores polynomial factors and memory (2^(0.2075 b) vectors),
so it under-counts real cost by 2^10 or more at these block sizes: conservative for the defender.

Everything is over the collision difference: inputs with |x_i - x'_i| <= beta (bytes: beta = 255; nibbles 15; bits 1).
q must exceed 2 beta, else every kernel vector reduced mod q into (-q/2, q/2] is a collision (a total break).

    python3 params.py            # the tables (markdown) and params.json beside --out
    python3 params.py --check    # the cross-checks only
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

C_CLASSICAL = math.log2(math.sqrt(3 / 2))  # 0.2925
C_QUANTUM = math.log2(math.sqrt(13 / 9))  # 0.2652
C_SIEVE_VECS = math.log2(math.sqrt(4 / 3))  # 0.2075
INF = 10**9


def log2_delta(b: int) -> float:
    """log2 of BKZ-b's root Hermite factor (Chen 2013)."""
    return math.log2(((math.pi * b) ** (1 / b) * b / (2 * math.pi * math.e)) ** (1 / (2 * (b - 1))))


def m1_breaks(b: int, nbits: float, logq: float, beta: float, m: int) -> bool:
    """M1: does BKZ-b find, in some d <= m, a vector of l2 norm <= beta sqrt(d) that is shorter than q?"""
    ld = log2_delta(b)
    d_opt = math.sqrt(nbits / ld)
    for d in {int(d_opt), int(d_opt) + 1, m} | set(range(max(2, int(d_opt * 0.5)), min(m, int(d_opt * 1.5)) + 1, 8)):
        if d > m or d < 2:
            continue
        ll = d * ld + nbits / d
        if ll < logq and ll <= math.log2(beta * math.sqrt(d)):
            return True
    return False


def m1_block(nbits: float, logq: float, beta: float, m: int) -> int:
    """The least BKZ block size M1 says breaks the instance (binary search; success is monotone in b)."""
    lo, hi = 50, 6000
    if not m1_breaks(hi, nbits, logq, beta, m):
        return INF
    while lo < hi:
        mid = (lo + hi) // 2
        if m1_breaks(mid, nbits, logq, beta, m):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _log2_erf_weight(beta: float, sigma: float) -> float:
    p = math.erf(beta / (sigma * math.sqrt(2)))
    return math.log2(p) if p > 0 else -INF


def m2_cost(b: int, nbits: float, logq: float, beta: float, m: int, c_svp: float) -> float:
    """M2 at block size b: Core-SVP cost plus log2 of the repetitions to hit an l_inf-short vector."""
    s = 2 * log2_delta(b)  # log2 slope of the GSA profile
    best = INF
    # randomized basis (no q-vectors): a GSA line of B vectors over the kernel lattice's volume q^n, unit vectors after
    B = int(math.sqrt(2 * nbits / s))
    if B <= m:
        l0 = s * (B - 1) / 2 + nbits / B  # log2 of the first vector: the line's top
        sigma = 2**l0 / math.sqrt(B)
        log_eps = B * _log2_erf_weight(beta, sigma)
        best = min(best, c_svp * b + max(0.0, -log_eps - C_SIEVE_VECS * b))
    # Z-shape: `a` q-vectors at the head (coordinates uniform mod q), then the slope down to 1, then unit vectors
    nq = nbits / logq
    Bs = int(logq / s)  # vectors on the slope from q to 1
    for a in range(0, int(nq) + 1, max(1, int(nq) // 64)):
        # sloped part must carry volume nbits - a logq with B vectors from log q down: sum = B logq - s B(B-1)/2
        rest = nbits - a * logq
        if rest <= 0:
            break
        # B vectors on a slope ending at 0: B = sqrt(2 rest / s), top = s (B - 1); capped at Bs (top <= log q)
        Bz = int(math.sqrt(2 * rest / s))
        if Bz > Bs or a + Bz > m:
            continue
        top = s * (Bz - 1) / 2 + rest / Bz
        sigma = 2**top / math.sqrt(Bz)
        log_eps = Bz * _log2_erf_weight(beta, sigma) + a * math.log2(min(1.0, (2 * beta + 1) / 2**logq))
        best = min(best, c_svp * b + max(0.0, -log_eps - C_SIEVE_VECS * b))
    return best


def m2_bits(nbits: float, logq: float, beta: float, m: int, c_svp: float = C_CLASSICAL) -> tuple[float, int]:
    best, bb = INF, 0
    for b in range(50, 6000, 2):
        if c_svp * b > best:
            break
        c = m2_cost(b, nbits, logq, beta, m, c_svp)
        if c < best:
            best, bb = c, b
    return best, bb


def wagner_bits(nbits: float, beta: float, m: int) -> float:
    """Wagner's k-tree: 2^k lists of (beta+1)^(m / 2^k) partial inputs, nbits / (k+1) bits zeroed a level."""
    best = INF
    for k in range(1, 40):
        if (m / 2**k) * math.log2(beta + 1) >= nbits / (k + 1):
            best = min(best, k + nbits / (k + 1))
    return best


@dataclass
class Security:
    nbits: float
    logq: float
    beta: float
    m: int
    m1_b: int
    m1_cls: float
    m1_qnt: float
    m2_cls: float
    m2_b: int
    wagner: float

    @property
    def floor_cls(self) -> float:
        return min(self.m1_cls, self.m2_cls, self.wagner)


def security(nbits: float, logq: float, beta: float, m: int) -> Security:
    assert 2**logq > 2 * beta, "q <= 2 beta: every kernel vector reduced mod q is a collision"
    b1 = m1_block(nbits, logq, beta, m)
    m2c, m2b = m2_bits(nbits, logq, beta, m)
    return Security(nbits, logq, beta, m, b1, C_CLASSICAL * b1, C_QUANTUM * b1, m2c, m2b, wagner_bits(nbits, beta, m))


def crosscheck() -> list[str]:
    """ajtai-design's published numbers (its section 2.3-2.4), reproduced; returns the failures."""
    p = math.log2(2013265921)
    out = []
    cases = [
        # (what, n, logq, beta, m, M1 b published, M2 classical published or None)
        ("ajtai n=64 babybear beta=1", 64, p, 1, 12288, 976, 357),
        ("ajtai n=128 babybear beta=1", 128, p, 1, 24576, 1925, 676),
        ("ajtai n=32 babybear beta=1", 32, p, 1, 12288, 485, 188),
        ("SWIFFT (64, 1024, 257)", 64, math.log2(257), 1, 1024, 242, 102),
        ("Buchmann-Lindner (96, 18*96, 389)", 96, math.log2(389), 1, 1728, 402, None),
    ]
    for what, n, lq, beta, m, b_pub, m2_pub in cases:
        b = m1_block(n * lq, lq, beta, m)
        m2c, _ = m2_bits(n * lq, lq, beta, m)
        ok1 = abs(b - b_pub) <= max(6, 0.02 * b_pub)
        ok2 = m2_pub is None or abs(m2c - m2_pub) <= 0.06 * m2_pub
        line = f"{what}: M1 b={b} (published {b_pub}) M2 {m2c:.0f} (published {m2_pub})"
        out.append(("ok   " if ok1 and ok2 else "FAIL ") + line)
    return out


# --- encodings ---------------------------------------------------------------------------------------------------

@dataclass
class Encoding:
    name: str
    unit: str  # the tensor-core operand type
    bx: int  # bits of an input digit (8: bytes, 4: nibbles, 2, 1: bits)
    logq: int
    ba: int  # bits of an A digit (a limb)
    exact: str  # what keeps the product exact

    @property
    def beta(self) -> int:
        return 2**self.bx - 1

    @property
    def limbs(self) -> int:
        return math.ceil(self.logq / self.ba)


ENCODINGS = [
    Encoding("int8-q32", "INT8 IMMA", 8, 32, 8, "s8 x s8 -> s32; K <= 2^17 a limb never overflows; limbs combined mod 2^32 in u32"),
    Encoding("int8-q16", "INT8 IMMA", 8, 16, 8, "as int8-q32, two limbs, combined mod 2^16"),
    Encoding("int8-q8-nibble", "INT8 IMMA", 4, 8, 8, "bytes split to nibbles (2x K); one limb; low byte of the s32 sum"),
    Encoding("int8-q16-nibble", "INT8 IMMA", 4, 16, 8, "nibbles, two limbs"),
    Encoding("int8-q8-bit", "INT8 IMMA", 1, 8, 8, "bytes split to bits (8x K); one limb"),
    Encoding("int8-q16-bit", "INT8 IMMA", 1, 16, 8, "bits, two limbs"),
    Encoding("fp8-q16-nibble", "FP8 E4M3", 4, 16, 4, "digits in [-8,7] exact in E4M3; products <= 64; exact only while every "
             "FP32 partial sum stays inside the tensor core's accumulator width (Hopper FP8 keeps ~14 bits: chunk K <= 2^8)"),
    Encoding("fp8-q8-nibble", "FP8 E4M3", 4, 8, 4, "as fp8-q16-nibble, two limbs"),
    Encoding("fp4-q8-2bit", "FP4 E2M1", 2, 8, 2, "digits in [-2,1] exact in E2M1; four limbs; same accumulator caveat"),
]


def min_nbits(enc: Encoding, target: float, m_bytes: int, model: str = "floor") -> tuple[int, Security]:
    """The least number of output words n (a multiple of 8) whose instance reaches `target` classical bits under the
    model (m1: M1; floor: the least of M1, M2 and Wagner)."""
    m = m_bytes * 8 // enc.bx
    n = 8
    while True:
        s = security(n * enc.logq, enc.logq, enc.beta, m)
        bits = s.m1_cls if model == "m1" else s.floor_cls
        if bits >= target:
            return n, s
        n += 8 if n < 1024 else 32


def ands_per_byte(enc: Encoding, n: int) -> float:
    """ANDs per input byte to recompute A x mod q in a Boolean circuit with free XOR, A public.

    Each input bit x_t times the public constant A_ji 2^t contributes its (logq - t) low bits, half of them ones on
    average: a dot each. A carry-save tree of full adders (one AND each) consumes one dot per adder, so ANDs ~ dots.
    Per input digit of bx bits and output word: sum_t (logq - t)/2; per byte: (8/bx) of those, times n words."""
    per_digit = sum((enc.logq - t) / 2 for t in range(enc.bx))
    return 8 / enc.bx * per_digit * n


SHA512_ANDS_PER_BLOCK = 58120  # C-Flock's count (lane brief); 454 a byte


def rows(m_bytes: int, targets=(128, 96, 80)):
    out = []
    for enc in ENCODINGS:
        for t in targets:
            n, s = min_nbits(enc, t, m_bytes)
            digest = n * enc.logq / 8
            macs = (8 / enc.bx) * n * enc.limbs
            out.append(dict(
                encoding=enc.name, unit=enc.unit, target=t, beta=enc.beta, logq=enc.logq, n=n, nbits=n * enc.logq,
                digest_bytes=digest, limbs=enc.limbs, macs_per_byte=macs, row_bytes=m_bytes,
                compression=m_bytes / digest, key_bytes_unstructured=n * (m_bytes * 8 // enc.bx) * enc.logq / 8,
                key_bytes_ring=(m_bytes * 8 // enc.bx) * enc.logq / 8,  # negacyclic blocks of size n: one row of A a block
                ands_per_byte=ands_per_byte(enc, n),
                ands_per_byte_with_sha_over_digest=ands_per_byte(enc, n) + digest / m_bytes * 454,
                m1_b=s.m1_b, m1_cls=s.m1_cls, m1_qnt=s.m1_qnt, m2_cls=s.m2_cls, wagner=s.wagner, exact=enc.exact,
            ))
    return out


def markdown(rs) -> str:
    h = ("| encoding | unit | target | beta | q | n | digest B | x row (4 KiB) | MACs/B | ANDs/B | M1 b | M1 cls/qnt | M2 cls "
         "| Wagner | key (unstr.) | key (ring) |\n|" + "---|" * 16 + "\n")
    for r in rs:
        h += (f"| {r['encoding']} | {r['unit']} | {r['target']} | {r['beta']} | 2^{r['logq']} | {r['n']} | {r['digest_bytes']:.0f} "
              f"| {r['compression']:.1f}x | {r['macs_per_byte']:.0f} | {r['ands_per_byte']:.0f} | {r['m1_b']} "
              f"| {r['m1_cls']:.0f}/{r['m1_qnt']:.0f} | {r['m2_cls']:.0f} | {r['wagner']:.0f} "
              f"| {r['key_bytes_unstructured'] / 2**20:.1f} MiB | {r['key_bytes_ring'] / 2**10:.0f} KiB |\n")
    return h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--row-bytes", type=int, default=4096)
    ap.add_argument("--out")
    a = ap.parse_args()
    cc = crosscheck()
    print("\n".join(cc))
    if a.check:
        raise SystemExit(any(c.startswith("FAIL") for c in cc))
    rs = rows(a.row_bytes)
    print(markdown(rs))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(crosscheck=cc, rows=rs, models=asdict(Encoding("x", "", 8, 32, 8, ""))), f, indent=1)


if __name__ == "__main__":
    main()
