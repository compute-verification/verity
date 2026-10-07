"""The verifier's public decode and the prover's hints for the NVFP4 unit (numpy, vectorised over the columns), the model as the
oracle (``BLACKWELL_SM120_NVF4.step``), the edge families of the honest sweep, and a torch-free evaluator of a compiled
``System`` (``build_witness_np`` / ``check_constraints_np``) for the laptop tests -- the pod uses ``..witness`` (torch).

Every hint here mirrors one constraint of :mod:`relation`; a wrong hint fails the prover's self-check, never soundness.
"""
from __future__ import annotations

import numpy as np
from verity.ml.circuits.silicon.models import BLACKWELL_SM120_NVF4

from ..compile import P, Expr, System
from .relation import (
    ACC_BITS,
    ACC_CAND_OFF,
    ACC_E_OFF,
    E_FIELD_MAX,
    FRAC_BITS,
    GRID_FLOOR,
    GROUP,
    GROUPS,
    K,
    MAG_BITS,
    NUMERATOR,
    POSITIONS,
    RADIX,
    SCALES,
    WORDS,
    shifter_plan,
)

__all__ = ["FAMILIES", "build_witness_np", "check_constraints_np", "eval_expr_np", "hints_fp4", "oracle_words",
           "public_vectors_fp4", "random_units", "unit_hints", "words_from_components"]

MODEL = BLACKWELL_SM120_NVF4
_NUM = np.asarray(NUMERATOR, dtype=np.int64)


# ---------------------------------------------------------------------------------------------------------------------
# The verifier's decode (pins)
# ---------------------------------------------------------------------------------------------------------------------

def _numerators(codes: np.ndarray) -> np.ndarray:
    mag = _NUM[codes & 7]
    return np.where((codes & 8) != 0, -mag, mag)


def _scales(sc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e, m = (sc >> 3) & 0xF, sc & 7
    return np.where(e == 0, m, 8 + m), np.where(e == 0, -9, e - 10)


def _check_domain(a: np.ndarray, b: np.ndarray, c: np.ndarray | None, y: np.ndarray | None) -> None:
    bad = []
    for name, w in (("a", a), ("b", b)):
        if w.ndim != 2 or w.shape[1] != WORDS:
            raise ValueError(f"{name}: operands are (B, {WORDS}) = {K} E2M1 codes + {SCALES} UE4M3 scale bytes")
        codes, sc = w[:, :K], w[:, K:]
        if ((codes < 0) | (codes >= 16)).any():
            bad.append(f"{name}: E2M1 code outside 4 bits")
        if ((sc < 0) | (sc >= 128)).any():
            bad.append(f"{name}: UE4M3 scale byte with the padding bit set (PTX ISA: invalid)")
        if (sc == 0x7F).any():
            bad.append(f"{name}: UE4M3 NaN scale")
    for name, w in (("c", c), ("y", y)):
        if w is None:
            continue
        if w.shape != (a.shape[0],):
            raise ValueError(f"{name}: one FP32 word per unit")
        if ((w < 0) | (w >= 2**32)).any():
            bad.append(f"{name}: not a 32-bit word")
        if (((w >> 23) & 0xFF) == 0xFF).any():
            bad.append(f"{name}: non-finite FP32 word (NaN / inf are outside the model)")
    if bad:
        raise ValueError("public inputs rejected: " + "; ".join(bad))


def public_vectors_fp4(a, b, c=None, y=None) -> dict[str, np.ndarray]:
    """The verifier's per-unit decode: ``a``/``b`` (B, 68) int -> the pinned numerators ``a[i].n`` / ``b[i].n``, per group
    ``g[g].sm`` (scale mantissa product), ``g[g].X`` (unit weight exponent), ``g[g].part`` (participates), and the group
    anchor ``anc.G``; with ``c``/``y`` (B,) FP32 words (unit mode) also ``c.s / c.t / c.f`` and ``y.s / y.t / y.f``.
    Raises ``ValueError`` off the domain."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    c = None if c is None else np.asarray(c, dtype=np.int64)
    y = None if y is None else np.asarray(y, dtype=np.int64)
    _check_domain(a, b, c, y)
    n_a, n_b = _numerators(a[:, :K]), _numerators(b[:, :K])
    ma, ea = _scales(a[:, K:])
    mb, eb = _scales(b[:, K:])
    prod = (n_a * n_b).reshape(-1, GROUPS, GROUP)
    out: dict[str, np.ndarray] = {}
    for i in range(K):
        out[f"a[{i}].n"] = n_a[:, i]
        out[f"b[{i}].n"] = n_b[:, i]
    part = (prod != 0).any(2) & (ma != 0) & (mb != 0)
    X = -2 + ea + eb
    for g in range(GROUPS):
        out[f"g[{g}].sm"] = ma[:, g] * mb[:, g]
        out[f"g[{g}].X"] = X[:, g]
        out[f"g[{g}].part"] = part[:, g].astype(np.int64)
    out["anc.G"] = np.where(part, X - FRAC_BITS, GRID_FLOOR).max(1)
    if c is not None:
        out["c.s"], out["c.t"], out["c.f"] = c >> 31, (c >> 23) & 0xFF, c & (2**23 - 1)
    if y is not None:
        out["y.s"], out["y.t"], out["y.f"] = y >> 31, (y >> 23) & 0xFF, y & (2**23 - 1)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# The prover's hints
# ---------------------------------------------------------------------------------------------------------------------

def _bit_length(v: np.ndarray, max_bits: int) -> np.ndarray:
    out = np.zeros_like(v)
    for k in range(max_bits + 1):
        out += (v >= (np.int64(1) << np.int64(k))).astype(np.int64)
    return out


def _digits_of(x: np.ndarray, nbits: int) -> list[np.ndarray]:
    return [(x >> (RADIX * c)) & (2**RADIX - 1) for c in range(0, -(-nbits // RADIX))]


#: Wrong rules the negatives are generated under (``unit_hints(..., variant=...)``): each changes one line of the model;
#: the relation must reject every witness they produce whose word differs from the model's (``relation_test``).
VARIANTS = ("frac26", "frac28", "window35", "window37", "nearest-align", "floor-align", "negzero", "scale-sub-as-normal",
            "scale-exp+1", "rne-norm", "participate-zero-scale", "acc-exp+1", "wrong-sign-acc", "drop-group3")


def _shift_hints(H: dict, path: str, d: np.ndarray, mag: np.ndarray, *, bits: int, n_j: int, exact: bool, dead: bool,
                 nearest: bool = False, floor: bool = False, sign: np.ndarray | None = None, strict: bool = True):
    """Hints of one shifter (the constraints of ``relation._shifter``) and the digit vector it places: (l, POSITIONS)."""
    l = d.shape[0]
    is_dead = d < -bits
    if not dead:
        assert not is_dead.any(), f"{path}: shift below -{bits} without a dead path"
    u = np.where(is_dead, bits, np.maximum(-d, 0))
    L = np.where(is_dead, 0, np.maximum(d, 0))
    q = mag >> u
    if nearest:                                                            # a wrong rule: round the dropped bits to nearest
        q = (mag + np.where(u > 0, np.int64(1) << np.maximum(u - 1, 0), 0)) >> u
    if floor and sign is not None:                                         # a wrong rule: negative terms rounded toward -inf
        q = np.where(sign == 1, (mag + (np.int64(1) << u) - 1) >> u, q)
    H[f"{path}.u"] = u
    if bits <= MAG_BITS:
        H[f"{path}.q"] = q
        parts = [(q, 0)]
    else:
        H[f"{path}.qh"], H[f"{path}.ql"] = q >> RADIX, q & (2**RADIX - 1)
        parts = [(q & (2**RADIX - 1), 0), (q >> RADIX, 1)]
    if not exact:
        H[f"{path}.rem"] = mag - (q << u)
    elif strict:
        assert ((q << u) == mag).all(), f"{path}: inexact shift where the relation requires an exact one"
    j, r = L // RADIX, L % RADIX
    assert (j < n_j).all(), f"{path}: coarse shift {j.max()} beyond {n_j - 1}"
    H[f"{path}.rlo"], H[f"{path}.rhi"] = r % 4, r // 4
    for jj in range(n_j):
        H[f"{path}.sel{jj}"] = (j == jj).astype(np.int64)
    if dead:
        H[f"{path}.dead"] = is_dead.astype(np.int64)
        H[f"{path}.k"] = np.where(is_dead, -d - bits, 1)
    D = np.zeros((l, POSITIONS), dtype=np.int64)
    for qp, off in parts:
        x2 = qp << r
        for c, dig in enumerate(_digits_of(x2, (bits if len(parts) == 1 else RADIX) + 11)):
            pos = c + off + j                                     # one-hot placement: position c + off + j
            np.add.at(D, (np.arange(l), pos), dig)
    return D


def unit_hints(c, a, b, variant: str | None = None) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Every hint of the unit by name (l,) int64, and the output FP32 words the hints encode (== ``oracle_words`` when
    ``variant`` is None).  A ``variant`` (one of ``VARIANTS``) generates the witness a prover following a WRONG rule would
    produce; the relation must reject it wherever its word differs from the model's."""
    c, a, b = (np.asarray(v, dtype=np.int64) for v in (c, a, b))
    v = variant
    if v is not None and v not in VARIANTS:
        raise ValueError(f"unknown variant {v!r}")
    l = c.shape[0]
    frac_bits = FRAC_BITS + (-1 if v == "frac26" else 1 if v == "frac28" else 0)
    cand_off = ACC_CAND_OFF + (-1 if v == "window35" else 1 if v == "window37" else 0)
    H: dict[str, np.ndarray] = {}
    # the accumulator
    s, t, f = c >> 31, (c >> 23) & 0xFF, c & (2**23 - 1)
    nz = (t != 0).astype(np.int64)
    z = ((t == 0) & (f == 0)).astype(np.int64)
    if v == "wrong-sign-acc":
        s = 1 - s
    H.update({"acc.s": s, "acc.t": t, "acc.f": f, "acc.nz": nz, "acc.z": z, "acc.fh": f >> RADIX, "acc.fl": f & (2**RADIX - 1)})
    M = f + nz * 2 ** (ACC_BITS - 1)
    tp = t + 1 - nz
    acc_cand = np.where(z == 1, GRID_FLOOR, tp - cand_off)
    acc_e = tp - ACC_E_OFF + (1 if v == "acc-exp+1" else 0)
    # the operands
    n_a, n_b = _numerators(a[:, :K]), _numerators(b[:, :K])
    ma, ea = _scales(a[:, K:])
    mb, eb = _scales(b[:, K:])
    if v == "scale-sub-as-normal":                                         # the subnormal scale field read as a normal one
        ma, mb = np.where(ea == -9, 8 + ma, ma), np.where(eb == -9, 8 + mb, mb)
        ea, eb = np.where(ea == -9, -10, ea), np.where(eb == -9, -10, eb)
    if v == "scale-exp+1":
        ea = ea + 1
    prod = (n_a * n_b).reshape(l, GROUPS, GROUP)
    gd = prod.sum(2)
    part = (prod != 0).any(2) & (ma != 0) & (mb != 0)
    if v == "participate-zero-scale":
        part = (prod != 0).any(2)
    X = -2 + ea + eb
    mant = gd * ma * mb
    if v == "drop-group3":
        mant[:, 3] = 0
    G = np.where(part, X - frac_bits, GRID_FLOOR).max(1)
    lsb = np.maximum(G, acc_cand)
    H["anc.lsb"] = lsb
    plan = shifter_plan()
    Psum = np.zeros((l, POSITIONS), dtype=np.int64)
    for g in range(GROUPS):
        sgn = (mant[:, g] < 0).astype(np.int64)
        H[f"g{g}.sgn"] = sgn
        d = np.where(part[:, g] | (mant[:, g] != 0), X[:, g] - lsb, 0)
        bits, n_j, exact, dead = plan[f"g{g}.shift"]
        D = _shift_hints(H, f"g{g}.shift", d, np.abs(mant[:, g]), bits=bits, n_j=n_j, exact=exact, dead=dead,
                         nearest=(v == "nearest-align"), floor=(v == "floor-align"), sign=sgn)
        Psum += (1 - 2 * sgn)[:, None] * D
    d_acc = np.where(z == 1, 0, acc_e - lsb)
    bits, n_j, exact, dead = plan["accterm.shift"]
    D = _shift_hints(H, "accterm.shift", d_acc, M, bits=bits, n_j=n_j, exact=exact, dead=dead, nearest=(v == "nearest-align"),
                     floor=(v == "floor-align"), sign=s)
    Psum += (1 - 2 * s)[:, None] * D
    # the exact sum and its RZ normalisation
    T = (Psum * (np.int64(1) << (RADIX * np.arange(POSITIONS, dtype=np.int64)))[None, :]).sum(1)
    sgn_t = (T < 0).astype(np.int64)
    magT = np.abs(T)
    nzM = (magT != 0).astype(np.int64)
    bl = _bit_length(magT, 50)
    e_val = bl - 1 + lsb
    normal = e_val >= -126
    grid = np.where(normal, e_val - 23, -149)
    sigma = np.where(nzM == 1, grid - lsb, 0)
    Mo = np.where(sigma >= 0, magT >> np.maximum(sigma, 0), magT << np.maximum(-sigma, 0))
    if v == "rne-norm":                                                    # round the dropped bits to nearest (ties up)
        half = np.where(sigma > 0, np.int64(1) << np.maximum(sigma - 1, 0), 0)
        Mo = np.where(sigma >= 0, (magT + half) >> np.maximum(sigma, 0), Mo)
    Mo = np.where(nzM == 1, Mo, 0)
    R = np.where(sigma >= 0, magT - (Mo << np.maximum(sigma, 0)), 0)
    t_o = np.where((nzM == 1) & normal, e_val + 127, 0)
    hid = ((nzM == 1) & normal).astype(np.int64)
    if variant is None:
        assert (t_o <= E_FIELD_MAX).all(), "the sum leaves binary32 (outside the model: D20)"
        assert ((Mo >> (ACC_BITS - 1)) == hid).all()
    hid = (Mo >> (ACC_BITS - 1)) & 1                                       # (a variant's M may carry into bit 23: keep the hint consistent)
    f_o = Mo & (2 ** (ACC_BITS - 1) - 1)
    sign_out = sgn_t * nzM
    if v == "negzero":                                                     # a zero result keeps the sign of the operands' sum
        sign_out = np.where(nzM == 1, sgn_t, np.where(z == 1, s, np.where(T == 0, s, sgn_t)))
        sgn_t = np.where(nzM == 1, sgn_t, sign_out)
    H.update({"out.sgn": sgn_t, "out.t": t_o, "out.hid": hid, "out.fh": f_o >> RADIX, "out.fl": f_o & (2**RADIX - 1), "out.nzM": nzM})
    bits, n_j, exact, dead = plan["out.shift"]
    DM = _shift_hints(H, "out.shift", sigma, Mo, bits=bits, n_j=n_j, exact=exact, dead=dead, strict=variant is None)
    for pn in range(n_j):
        H[f"out.r{pn}"] = (R >> (RADIX * pn)) & (2**RADIX - 1)
    if variant is None:
        assert (R >> (RADIX * n_j) == 0).all()
    carry = np.zeros(l, dtype=np.int64)
    for pn in range(POSITIONS):
        rp = H[f"out.r{pn}"] if pn < n_j else 0
        S = (1 - 2 * sgn_t) * Psum[:, pn] - DM[:, pn] - rp + carry
        if variant is None:
            assert (S & (2**RADIX - 1) == 0).all(), "carry chain does not close"
        carry = S >> RADIX
        if pn < POSITIONS - 1:
            H[f"out.c{pn + 1}"] = carry
    if variant is None:
        assert (carry == 0).all(), "the digit identity does not close"
    y = (sign_out << 31) | (t_o << 23) | f_o
    return H, y



def hints_fp4(sys: System, c, a, b) -> np.ndarray:
    """(n_hints, l) int64 in the order of ``sys.hints`` (values may be negative: the witness builder reduces mod p)."""
    H, _ = unit_hints(c, a, b)
    return np.stack([H[n] for _, n in sys.hints])


def words_from_components(s, t, f) -> np.ndarray:
    return (np.asarray(s, dtype=np.int64) << 31) | (np.asarray(t, dtype=np.int64) << 23) | np.asarray(f, dtype=np.int64)


def oracle_words(c, a, b) -> np.ndarray:
    """``BLACKWELL_SM120_NVF4.step`` per unit (host Python; the reference)."""
    c, a, b = np.asarray(c), np.asarray(a), np.asarray(b)
    return np.asarray([MODEL.step(int(c[i]), [int(x) for x in a[i]], [int(x) for x in b[i]]) for i in range(len(c))], dtype=np.int64)


# ---------------------------------------------------------------------------------------------------------------------
# Edge families (the honest sweep) -- operands as (n, 68) words, accumulators as FP32 words
# ---------------------------------------------------------------------------------------------------------------------

def _scale_bytes(rng: np.random.Generator, shape, kind: str) -> np.ndarray:
    if kind == "uniform":
        w = rng.integers(0, 0x7F, size=shape, dtype=np.int64)              # 0x00 .. 0x7E: every finite UE4M3 byte
    elif kind == "zero":
        w = np.zeros(shape, dtype=np.int64)
    elif kind == "subnormal":
        w = rng.integers(0, 8, size=shape, dtype=np.int64)                 # exponent field 0: mantissa m at 2^-9
    elif kind == "top":
        w = (15 << 3) | rng.integers(0, 7, size=shape, dtype=np.int64)     # exponent 15 (the hot-clip regime), never NaN
    elif kind == "one":
        w = np.full(shape, 0x38, dtype=np.int64)                           # 1.0
    elif kind == "span":                                                   # one tiny and one huge scale per unit
        w = np.where(rng.random(shape) < 0.5, rng.integers(1, 16, size=shape), rng.integers(0x68, 0x7F, size=shape)).astype(np.int64)
    else:
        raise ValueError(kind)
    return w


def _codes(rng: np.random.Generator, shape, kind: str) -> np.ndarray:
    if kind == "uniform":
        return rng.integers(0, 16, size=shape, dtype=np.int64)
    if kind == "sparse":                                                   # mostly zeros: few nonzero products
        w = rng.integers(0, 16, size=shape, dtype=np.int64)
        return np.where(rng.random(shape) < 0.9, 0, w)
    if kind == "max":                                                      # +-6 everywhere: the widest group dots
        return 7 | (rng.integers(0, 2, size=shape, dtype=np.int64) << 3)
    if kind == "zero":
        return np.zeros(shape, dtype=np.int64)
    if kind == "half":                                                     # the subnormal +-0.5 everywhere
        return 1 | (rng.integers(0, 2, size=shape, dtype=np.int64) << 3)
    raise ValueError(kind)


def _fp32(rng: np.random.Generator, n: int, kind: str) -> np.ndarray:
    s = rng.integers(0, 2, size=n, dtype=np.int64)
    f = rng.integers(0, 2**23, size=n, dtype=np.int64)
    if kind == "zero":
        return np.zeros(n, dtype=np.int64)
    if kind == "negzero":
        return np.full(n, 0x8000_0000, dtype=np.int64)
    if kind == "subnormal":
        return (s << 31) | np.maximum(f, 1)
    if kind == "tiny":
        t = rng.integers(1, 40, size=n, dtype=np.int64)
    elif kind == "small":
        t = rng.integers(60, 100, size=n, dtype=np.int64)
    elif kind == "near":                                                   # within the window of a unit-weight product
        t = rng.integers(100, 140, size=n, dtype=np.int64)
    elif kind == "large":
        t = rng.integers(140, 200, size=n, dtype=np.int64)
    elif kind == "huge":
        t = rng.integers(200, 255, size=n, dtype=np.int64)
    elif kind == "any":
        t = rng.integers(0, 255, size=n, dtype=np.int64)
    else:
        raise ValueError(kind)
    return (s << 31) | (t << 23) | f


#: name -> (codes kind, scale kind, accumulator kind); ``mirror`` flips B's signs against A for cancellation.
FAMILIES: dict[str, tuple[str, str, str]] = {
    "random": ("uniform", "uniform", "any"),
    "random-near": ("uniform", "uniform", "near"),
    "zero-acc": ("uniform", "uniform", "zero"),
    "negzero-acc": ("uniform", "uniform", "negzero"),
    "subnormal-acc": ("uniform", "uniform", "subnormal"),
    "tiny-acc": ("uniform", "uniform", "tiny"),
    "huge-acc": ("uniform", "uniform", "huge"),
    "large-acc": ("uniform", "uniform", "large"),
    "zero-scales": ("uniform", "zero", "any"),
    "subnormal-scales": ("uniform", "subnormal", "near"),
    "top-scales": ("uniform", "top", "large"),
    "top-scales-zero-acc": ("max", "top", "zero"),
    "span-scales": ("uniform", "span", "near"),
    "sparse": ("sparse", "uniform", "any"),
    "sparse-subnormal-acc": ("sparse", "subnormal", "subnormal"),
    "all-zero-codes": ("zero", "uniform", "any"),
    "all-zero-codes-negzero-acc": ("zero", "uniform", "negzero"),
    "half-codes": ("half", "subnormal", "tiny"),
    "max-codes": ("max", "uniform", "any"),
    "cancel": ("uniform", "one", "zero"),                                  # mirrored: the dot cancels exactly
    "cancel-acc": ("uniform", "uniform", "near"),                          # mirrored operands, accumulator alone survives
}
_MIRRORED = {"cancel", "cancel-acc"}


def random_units(rng: np.random.Generator, n: int, family: str = "random") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``n`` units ``(c, a, b)`` of an edge family: FP32 words (n,), operands (n, 68)."""
    ck, sk, ak = FAMILIES[family]
    a = np.concatenate([_codes(rng, (n, K), ck), _scale_bytes(rng, (n, SCALES), sk)], axis=1)
    b = np.concatenate([_codes(rng, (n, K), ck), _scale_bytes(rng, (n, SCALES), sk)], axis=1)
    if family in _MIRRORED:
        # groups 2, 3 repeat groups 0, 1 (codes and scales) with B's signs flipped: mant_2 = -mant_0, mant_3 = -mant_1 at
        # the same unit weights, so the four group terms cancel exactly and only the accumulator survives
        half, shalf = K // 2, SCALES // 2
        a[:, half:K] = a[:, :half]
        a[:, K + shalf:] = a[:, K:K + shalf]
        b[:, :half] = a[:, :half]
        b[:, half:K] = a[:, :half] ^ 8
        b[:, K:] = a[:, K:]
    c = _fp32(rng, n, ak)
    return c, a, b


# ---------------------------------------------------------------------------------------------------------------------
# A torch-free evaluator of a compiled system (the laptop tests; the pod uses ..witness)
# ---------------------------------------------------------------------------------------------------------------------

def eval_expr_np(e: Expr, W: np.ndarray) -> np.ndarray:
    acc = np.full(W.shape[1], e.c % P, dtype=np.int64)
    for j, v in e.lin.items():
        acc = (acc + (v % P) * W[j]) % P
    return acc


def _signed(x: np.ndarray) -> np.ndarray:
    return np.where(x > P // 2, x - P, x)


def build_witness_np(sys: System, pub: dict[str, np.ndarray], hints: np.ndarray) -> np.ndarray:
    """W (m, l) int64 in [0, p) by running ``sys.program`` (the same ops as ``..witness._run_program``)."""
    l = hints.shape[1]
    W = np.zeros((sys.m, l), dtype=np.int64)
    hpos = {n: i for i, (_, n) in enumerate(sys.hints)}
    for op in sys.program:
        kind = op[0]
        if kind == "pin":
            W[op[1]] = np.asarray(pub[op[2]], dtype=np.int64) % P
        elif kind == "hint":
            W[op[1]] = hints[hpos[op[2]]] % P
        elif kind == "prod":
            W[op[1]] = (eval_expr_np(op[2], W) * eval_expr_np(op[3], W)) % P
        elif kind == "bits":
            _, row0, nbits, e = op
            x = eval_expr_np(e, W)
            for i in range(nbits):
                W[row0 + i] = (x >> i) & 1
        elif kind == "sels":
            _, row0, k0s, k1s, key = op
            x = _signed(eval_expr_np(key, W))
            for i, (k0, k1) in enumerate(zip(k0s, k1s)):
                W[row0 + i] = ((x >= k0) & (x <= k1)).astype(np.int64)
        elif kind == "inv":
            x = eval_expr_np(op[2], W)
            W[op[1]] = np.asarray([pow(int(v), P - 2, P) for v in x], dtype=np.int64)
        else:
            raise ValueError(kind)
    return W


def check_constraints_np(sys: System, W: np.ndarray, pub: dict[str, np.ndarray]) -> list[str]:
    """The names of the violated constraints (with the first offending column), or [] when W satisfies the system."""
    bad = []
    for lin in sys.linear:
        v = eval_expr_np(Expr(None, lin.lin, lin.c), W)
        target = np.asarray(pub[lin.pub], dtype=np.int64) % P if lin.pub else 0
        off = np.nonzero(v != target)[0]
        if off.size:
            bad.append(f"linear {lin.name} @{int(off[0])}")
    for q in sys.quadratic:
        v = (eval_expr_np(q.a, W) * eval_expr_np(q.b, W)) % P
        off = np.nonzero(v != eval_expr_np(q.c, W))[0]
        if off.size:
            bad.append(f"quadratic {q.name} @{int(off[0])}")
    return bad
