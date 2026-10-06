"""z3 soundness of the *AIR-side* checker-v2 gadgets over BabyBear (lane b-air-v2, `archive/direct/V2_AIR.md`).

The B v2 AIR realises checker v2's gadgets with tuple lookups (every tuple component bound to a table column),
chunked shift tables, three 10-bit chunks of the sum magnitude, the product form of `attained`, and the FP32
word as two 16-bit limbs.  Each realisation is checked here at the real widths:

    integer      hints are unbounded integers, tables are their symbolic functions: "accepted and the output
                 differs from the mathematical reference" must be UNSAT;
    wrap-free    hints are field elements in [0, p), every identity / boolean / key congruence carries a fresh
                 multiplier of p: "accepted with some multiplier != 0" must be UNSAT, in which case the field
                 reading of the gadget is its integer reading on the same values and the integer UNSAT transfers
                 (the `wrapfree_queries_v2` argument of checker_min.md 3.2);
    direct       "accepted over F_p and the output residue differs from the reference's": the composite query,
                 where it finishes.

    python -m verity_numerical.redteam.z3_air_v2 --json backends/numerical/reports/b_air_v2_z3.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from ..checker.params import REAL, Params, Reject
from ..checker.v2.params import V2

BABYBEAR = 2**31 - 2**27 + 1
DT = 26                    # d~ = min(d, DT): every term is 0 once d >= DT (V2.dt at REAL)
DSP_ROWS = 2 * (DT + 1)    # dsp = 2 d~ + sp
K_ROWS = 32                # koff in [0, 32): k in [-7, 24]

__all__ = ["BABYBEAR", "GADGETS", "main", "run_all"]


# ---------------------------------------------------------------------------------------------------------------------
# Tuple tables: a list of key ranges, an integer function and the same function on z3 expressions
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TupleTable:
    name: str
    keys: tuple[tuple[int, int], ...]                          # (lo, hi) per key component
    outs: tuple[str, ...]
    fn: Callable[..., tuple[int, ...]]
    sym: Callable[..., tuple[object, tuple[object, ...]]]      # sym(z3, *keys) -> (in_domain, outs)
    domain: Callable[..., bool] | None = None

    def row(self, *keys: int) -> tuple[int, ...]:
        for k, (lo, hi) in zip(keys, self.keys):
            if type(k) is not int or not lo <= k < hi:
                raise KeyError(keys)
        if self.domain is not None and not self.domain(*keys):
            raise KeyError(keys)
        return self.fn(*keys)


def _ifchain(z3, key, cases, default):
    out = default
    for k, v in reversed(cases):
        out = z3.If(key == k, v, out)
    return out


def _shift_fn(scale: int):
    def fn(x: int, dsp: int) -> tuple[int]:
        dt, s = divmod(dsp, 2)
        v = (x << scale) >> dt
        return (-v if s else v,)
    return fn


def _shift_sym(scale: int):
    def sym(z3, x, dsp):
        dt, s = dsp / 2, dsp % 2
        cases = [(d, x * 2 ** (scale - d) if d <= scale else x / 2 ** (d - scale)) for d in range(DT + 1)]
        v = _ifchain(z3, dt, cases, z3.IntVal(0))
        return True, (z3.If(s == 1, -v, v),)
    return sym


def shift_table(name: str, bits: int, scale: int) -> TupleTable:
    return TupleTable(name, ((0, 2**bits), (0, DSP_ROWS)), ("term",), _shift_fn(scale), _shift_sym(scale))


SHIFT_HI = shift_table("SHIFT_HI", 8, 18)
SHIFT_LO = shift_table("SHIFT_LO", 8, 10)
SSHIFT_2 = shift_table("SSHIFT_2", 8, 17)
SSHIFT_1 = shift_table("SSHIFT_1", 8, 9)
SSHIFT_0 = shift_table("SSHIFT_0", 8, 1)


def _lead_fn(x: int) -> tuple[int, int]:
    return int(x != 0), x.bit_length()


def _lead_sym(z3, x):
    out = z3.IntVal(10)
    for l in range(9, -1, -1):
        out = z3.If(x < 2**l, l, out)
    return True, (z3.If(x == 0, 0, 1), out)


LEAD10 = TupleTable("LEAD10", ((0, 2**10),), ("nz", "l"), _lead_fn, _lead_sym)


def _align_fn(d: int, s4: int) -> tuple[int]:
    return (2 * min(d, DT) + int(s4 in (1, 2)),)


def _align_sym(z3, d, s4):
    return True, (2 * z3.If(d < DT, d, DT) + z3.If(z3.Or(s4 == 1, s4 == 2), 1, 0),)


def align4_table(p: Params) -> TupleTable:
    return TupleTable("ALIGN4", ((0, V2.shift_rows(p)), (0, 4)), ("dsp",), _align_fn, _align_sym)


def leadnorm_table(p: Params) -> TupleTable:
    """(l, E) -> (koff, A1, e_hat, z, ovf, isfloor); A1 = 2^(10 + k) for k <= 13 else 0 (rows with k > 13 have
    l <= 10, so the high chunks that A1 multiplies are zero)."""
    cl = p.acc_sig - p.width

    def fn(l: int, E: int) -> tuple[int, ...]:
        k = min(p.acc_sig - l, E - p.e_min + cl)
        e1 = E - k + cl
        z = l == 0 or l + k <= 0
        return (k - p.k_lo, 2 ** (10 + k) if k <= 13 else 0, p.floor if z else e1, int(z),
                int(not z and e1 > p.e_max), int(E == p.floor))

    def sym(z3, l, E):
        ku, kc = p.acc_sig - l, E - p.e_min + cl
        k = z3.If(ku < kc, ku, kc)
        e1 = E - k + cl
        z = z3.Or(l == 0, l + k <= 0)
        A1 = _ifchain(z3, k, [(kk, z3.IntVal(2 ** (10 + kk) if kk <= 13 else 0)) for kk in range(p.k_lo, p.k_hi + 1)], z3.IntVal(0))
        return True, (k - p.k_lo, A1, z3.If(z, p.floor, e1), z3.If(z, 1, 0),
                      z3.If(z3.And(z3.Not(z), e1 > p.e_max), 1, 0), z3.If(E == p.floor, 1, 0))

    return TupleTable("LEADNORM", ((0, p.l_hi + 1), (p.floor, V2.last(p) + 1)),
                      ("koff", "A1", "e_hat", "z", "ovf", "isfloor"), fn, sym)


def tnorm10_table(p: Params) -> TupleTable:
    """(v0, koff) -> floor(v0 2^k), clamped to 0 where it would reach 2^24 (rows no table-consistent witness uses)."""
    def fn(v0: int, koff: int) -> tuple[int]:
        k = koff + p.k_lo
        t = (v0 << k) if k >= 0 else (v0 >> -k)
        return (t if t < 2**p.acc_sig else 0,)

    def sym(z3, v0, koff):
        cases = []
        for ko in range(K_ROWS):
            k = ko + p.k_lo
            t = v0 * 2**k if k >= 0 else v0 / 2 ** (-k)
            cases.append((ko, z3.If(t < 2**p.acc_sig, t, 0)))
        return True, (_ifchain(z3, koff, cases, z3.IntVal(0)),)

    return TupleTable("TNORM10", ((0, 2**10), (0, K_ROWS)), ("t",), fn, sym)


def hdr_table(p: Params) -> TupleTable:
    emask = 2**p.exp - 1

    def fn(h: int) -> tuple[int, int, int]:
        s, t = h >> p.exp, h & emask
        return s, int(t != 0), max(t, 1) - p.bias

    def dom(h: int) -> bool:
        return h & emask != p.t_finite

    def sym(z3, h):
        s, t = h / 2**p.exp, h % 2**p.exp
        return t != p.t_finite, (s, z3.If(t == 0, 0, 1), z3.If(t == 0, 1, t) - p.bias)

    return TupleTable("T_HDR", ((0, 2 ** (p.exp + 1)),), ("s", "nz", "e"), fn, sym, domain=dom)


# ---------------------------------------------------------------------------------------------------------------------
# Contexts
# ---------------------------------------------------------------------------------------------------------------------

class IntCtx:
    """Reference interpreter on Python ints (used to check the gadget code against the reference functions)."""

    def rng(self, name, x, lo, hi):
        if type(x) is not int or not lo <= x < hi:
            raise Reject(f"{name}: {x!r} not in [{lo}, {hi})")

    def eq(self, name, a, b):
        if a != b:
            raise Reject(f"{name}: {a!r} != {b!r}")

    def boolean(self, name, b):
        if b not in (0, 1) or type(b) is not int:
            raise Reject(f"{name}: {b!r} is not a bit")

    def lookup_tuple(self, name, table: TupleTable, keys):
        try:
            return table.row(*keys)
        except KeyError:
            raise Reject(f"{name}: {keys!r} off table {table.name}") from None

    def product_zero(self, name, factors):
        if all(f != 0 for f in factors):
            raise Reject(f"{name}: no factor vanishes")


class ZCtx:
    """Integer z3 context: predicates over Z; a tuple lookup is a symbolic row whose key components equal the
    query's components."""

    def __init__(self) -> None:
        import z3
        self.z3 = z3
        self.solver = z3.Solver()
        self.count = 0
        self.wraps: list = []

    def fresh(self, name):
        self.count += 1
        return self.z3.Int(f"{name}#{self.count}")

    def _add(self, c):
        if isinstance(c, bool):
            c = self.z3.BoolVal(c)
        self.solver.add(c)

    def _cong(self, a, b):
        return a == b

    def rng(self, name, x, lo, hi):
        if isinstance(x, int):
            self._add(lo <= x < hi)
            return
        rep = self.fresh(f"{name}.rep")
        self._add(self.z3.And(rep >= lo, rep < hi, self._cong(x, rep)))

    def eq(self, name, a, b):
        if isinstance(a, int) and isinstance(b, int):
            self._add(a == b)
            return
        self._add(self._cong(a, b))

    def boolean(self, name, b):
        if isinstance(b, int):
            self._add(b in (0, 1))
            return
        t = self.fresh(f"{name}.bit")
        self._add(self.z3.And(self.z3.Or(t == 0, t == 1), self._cong(b, t)))

    def lookup_tuple(self, name, table: TupleTable, keys):
        z3 = self.z3
        kk = [self.fresh(f"{name}.k{i}") for i in range(len(keys))]
        in_dom, vals = table.sym(z3, *kk)
        conds = [z3.And(k >= lo, k < hi) for k, (lo, hi) in zip(kk, table.keys)]
        conds += [self._cong(q, k) for q, k in zip(keys, kk)]
        outs = tuple(self.fresh(f"{name}.{o}") for o in table.outs)
        self._add(z3.And(*conds, in_dom, *[o == v for o, v in zip(outs, vals)]))
        return outs

    def product_zero(self, name, factors):
        """The AIR identity `prod_i f_i = 0`.  Over Z and over the field F_p alike this is exactly "some factor is
        zero" (F_p is a field: a product vanishes iff a factor does), so it is modelled as the disjunction of the
        factor identities over F_p (one selector, one shared multiplier), no product of degree 10 for the solver."""
        z3 = self.z3
        j = self.fresh(f"{name}.which")
        zero = self.fresh(f"{name}.zero")          # the vanishing factor's integer value: congruent to 0
        self._add(z3.Or([z3.And(j == i, f == zero) for i, f in enumerate(factors)]))
        self._add(self._cong(zero, 0))


class ModCtx(ZCtx):
    """Predicates over F_p: every identity / boolean / key match is a congruence `a - b = m p` with a fresh
    multiplier `m` (recorded).  Table outputs are the integer values of the matched row (a field element's
    signed representative); hints are field elements in [0, p)."""

    def __init__(self, modulus: int) -> None:
        super().__init__()
        self.modulus = modulus

    def _cong(self, a, b):
        m = self.fresh("wrap")
        self.wraps.append(m)
        return a - b == m * self.modulus

    def rng(self, name, x, lo, hi):
        # a range check is a range-table lookup: the canonical representative must be in [lo, hi)
        if isinstance(x, int):
            self._add(lo <= x % self.modulus < hi)
            return
        rep = self.fresh(f"{name}.rep")
        self._add(self.z3.And(rep >= lo, rep < hi, self._cong(x, rep)))

    def eq(self, name, a, b):
        if isinstance(a, int) and isinstance(b, int):
            self._add((a - b) % self.modulus == 0)
            return
        self._add(self._cong(a, b))

    def boolean(self, name, b):
        if isinstance(b, int):
            self._add(b % self.modulus in (0, 1))
            return
        t = self.fresh(f"{name}.bit")
        self._add(self.z3.And(self.z3.Or(t == 0, t == 1), self._cong(b, t)))


# ---------------------------------------------------------------------------------------------------------------------
# The AIR gadgets (mirroring `archive/direct/src/program_v2.rs`)
# ---------------------------------------------------------------------------------------------------------------------

def product_shift(ctx, p: Params, u, dsp, h: dict, tag="pshift"):
    u_hi, u_lo = h["u_hi"], h["u_lo"]
    ctx.eq(f"{tag}.split", u_hi * 2**8 + u_lo, u)
    (t_hi,) = ctx.lookup_tuple(f"{tag}.hi", SHIFT_HI, (u_hi, dsp))
    (t_lo,) = ctx.lookup_tuple(f"{tag}.lo", SHIFT_LO, (u_lo, dsp))
    return t_hi + t_lo


def product_shift_gen(p: Params, u: int, dsp: int) -> dict:
    return {"u_hi": u >> 8, "u_lo": u & 255}


def state_shift(ctx, p: Params, M, dsp, h: dict, tag="sshift"):
    M2, M1, M0 = h["M2"], h["M1"], h["M0"]
    ctx.eq(f"{tag}.split", M2 * 2**16 + M1 * 2**8 + M0, M)
    (t2,) = ctx.lookup_tuple(f"{tag}.t2", SSHIFT_2, (M2, dsp))
    (t1,) = ctx.lookup_tuple(f"{tag}.t1", SSHIFT_1, (M1, dsp))
    (t0,) = ctx.lookup_tuple(f"{tag}.t0", SSHIFT_0, (M0, dsp))
    return t2 + t1 + t0


def state_shift_gen(p: Params, M: int, dsp: int) -> dict:
    return {"M2": M >> 16, "M1": (M >> 8) & 255, "M0": M & 255}


def signed_sum(ctx, p: Params, terms: Sequence, h: dict, tag="sum"):
    """Returns (sgn, v2, v1, v0, l); `l` is the materialised bit length."""
    sgn, v2, v1, v0 = h["sgn"], h["v2"], h["v1"], h["v0"]
    ctx.boolean(f"{tag}.sgn", sgn)
    S = sum(terms)
    v = S - 2 * sgn * S
    ctx.eq(f"{tag}.split", v2 * 2**20 + v1 * 2**10 + v0, v)
    nz2, l2 = ctx.lookup_tuple(f"{tag}.lead2", LEAD10, (v2,))
    nz1, l1 = ctx.lookup_tuple(f"{tag}.lead1", LEAD10, (v1,))
    _nz0, l0 = ctx.lookup_tuple(f"{tag}.lead0", LEAD10, (v0,))
    l_expr = nz2 * (20 + l2) + (1 - nz2) * (nz1 * (10 + l1) + (1 - nz1) * l0)
    l = h.get("l")
    if l is None:                     # the materialised copy (a column bound by an identity)
        l = l_expr
    else:
        ctx.eq(f"{tag}.l", l, l_expr)
    return sgn, v2, v1, v0, l


def signed_sum_gen(p: Params, terms: Sequence[int]) -> dict:
    S = sum(terms)
    v = abs(S)
    return {"sgn": int(S < 0), "v2": v >> 20, "v1": (v >> 10) & 1023, "v0": v & 1023, "l": v.bit_length()}


def normalise(ctx, p: Params, E, l, v2, v1, v0, tag="norm"):
    koff, A1, e_hat, z, ovf, isfloor = ctx.lookup_tuple(f"{tag}.leadnorm", leadnorm_table(p), (l, E))
    (t,) = ctx.lookup_tuple(f"{tag}.tnorm", tnorm10_table(p), (v0, koff))
    M = (v2 * 2**10 + v1) * A1 + t
    return e_hat, M, z, ovf, isfloor


def group_max(ctx, p: Params, cands: Sequence, s4s: Sequence, l, h: dict, tag="max", attained: bool = True):
    """E hint, one ALIGN4 tuple per candidate, the E range through LEADNORM (l is a free input as in the
    composition), attained as the product identity."""
    E = h["E"]
    dsps = [ctx.lookup_tuple(f"{tag}[{i}]", align4_table(p), (E - c, s))[0] for i, (c, s) in enumerate(zip(cands, s4s))]
    ctx.lookup_tuple(f"{tag}.leadnorm", leadnorm_table(p), (l, E))
    if attained:
        ctx.product_zero(f"{tag}.attained", [E - c for c in cands] + [E - p.floor])
    return (E, *dsps)


def pack_limbs(ctx, p: Params, state, ovfs: Sequence, h: dict, tag="pack"):
    """(s, e_hat, M, z) and the groups' (ovf, sgn) -> (y_hi, y_lo): M = M_hi8 2^16 + M_lo16; the finite high limb
    is the expression s 2^15 + fld 2^7 + M_hi8."""
    s, e_hat, M, z = state
    M_hi8, M_lo16 = h["M_hi8"], h["M_lo16"]
    ctx.rng(f"{tag}.M_hi8", M_hi8, 0, 2**8)
    ctx.rng(f"{tag}.M_lo16", M_lo16, 0, 2**16)
    ctx.eq(f"{tag}.M", M_hi8 * 2**16 + M_lo16, M)
    fld = e_hat + (p.bias - 1) + z * (p.e_min - p.floor)
    y_hi = s * 2**15 + fld * 2**7 + M_hi8
    y_lo = M_lo16
    any_ovf, inf_sign, none_yet = 0, 0, 1
    for ovf_g, sgn_g in ovfs:
        first = none_yet * ovf_g
        any_ovf = any_ovf + first
        inf_sign = inf_sign + first * sgn_g
        none_yet = none_yet - first
    inf_hi = inf_sign * 2**15 + p.t_finite * 2**7
    return y_hi + any_ovf * (inf_hi - y_hi), y_lo - any_ovf * y_lo


def pack_gen(p: Params, M: int) -> dict:
    return {"M_hi8": M >> 16, "M_lo16": M & 0xFFFF}


def epilogue_limbs(ctx, p: Params, y_hi, y_lo, h: dict, tag="cast"):
    hi2, lsb, carry, rem = h["hi2"], h["lsb"], h["carry"], h["rem"]
    ctx.rng(f"{tag}.hi", y_hi, 0, 2**16)
    ctx.rng(f"{tag}.lo", y_lo, 0, 2**16)
    ctx.rng(f"{tag}.hi2", hi2, 0, 2**15)
    ctx.boolean(f"{tag}.lsb", lsb)
    ctx.boolean(f"{tag}.carry", carry)
    ctx.rng(f"{tag}.rem", rem, 0, 2**16)
    ctx.eq(f"{tag}.lsb_split", 2 * hi2 + lsb, y_hi)
    ctx.eq(f"{tag}.round", carry * 2**16 + rem, y_lo + lsb + (2**15 - 1))
    y16 = y_hi + carry
    ctx.rng(f"{tag}.y16", y16, 0, 2**16)
    return y16


def epilogue_gen(p: Params, y_hi: int, y_lo: int) -> dict:
    lsb = y_hi & 1
    t = y_lo + lsb + (2**15 - 1)
    return {"hi2": y_hi >> 1, "lsb": lsb, "carry": t >> 16, "rem": t & 0xFFFF}


def state_decode_limbs(ctx, p: Params, c_hi, c_lo, h: dict, tag="acc"):
    """The word-in unit's decode on limbs (not in the state-chained VU; listed for completeness): c_hi = hdr 2^7 +
    f_hi with hdr through T_HDR, M = nz 2^23 + f_hi 2^16 + c_lo, z with z M = 0 and M + z - 1 in [0, 2^24)."""
    hdr, f_hi, z, mz_hi = h["hdr"], h["f_hi"], h["z"], h["mz_hi"]
    ctx.rng(f"{tag}.f_hi", f_hi, 0, 2**7)
    ctx.eq(f"{tag}.word_hi", hdr * 2**7 + f_hi, c_hi)
    s, nz, e = ctx.lookup_tuple(f"{tag}.hdr", hdr_table(p), (hdr,))
    ctx.boolean(f"{tag}.z", z)
    M = nz * 2**23 + f_hi * 2**16 + c_lo
    ctx.eq(f"{tag}.z0", z * M, 0)
    # M + z - 1 in [0, 2^24): the low 16 bits are c_lo + z - 1 ... realised as a split range (mz_hi < 2^8)
    ctx.rng(f"{tag}.mz_hi", mz_hi, 0, 2**8)
    mz_lo = M + z - 1 - mz_hi * 2**16
    ctx.rng(f"{tag}.mz_lo", mz_lo, 0, 2**16)
    e_hat = e + z * (p.floor - e)
    return s, e_hat, M, z


def state_decode_gen(p: Params, c_hi: int, c_lo: int) -> dict:
    c = c_hi * 2**16 + c_lo
    f = c & (2**23 - 1)
    t = (c >> 23) & 255
    M = f + (2**23 if t else 0)
    z = int(M == 0)
    return {"hdr": c_hi >> 7, "f_hi": c_hi & 127, "z": z, "mz_hi": (M + z - 1) >> 16}


# ---------------------------------------------------------------------------------------------------------------------
# References and specs
# ---------------------------------------------------------------------------------------------------------------------

def _signed_shift(x: int, scale: int, dsp: int) -> int:
    dt, s = divmod(dsp, 2)
    v = (x << scale) >> dt
    return -v if s else v


def _zshift(z3, x, scale: int, dsp):
    dt, s = dsp / 2, dsp % 2
    out = z3.IntVal(0)
    for d in range(DT, -1, -1):
        out = z3.If(dt == d, x * 2 ** (scale - d) if d <= scale else x / 2 ** (d - scale), out)
    return z3.If(s == 1, -out, out)


def _zbitlen(z3, v, hi: int):
    out = z3.IntVal(hi)
    for l in range(hi - 1, -1, -1):
        out = z3.If(v < 2**l, l, out)
    return out


@dataclass
class Spec:
    name: str
    inputs: Callable[[Params], dict[str, tuple[int, int]]]
    hints: tuple[str, ...]
    call: Callable                # call(ctx, p, x, h) -> outs
    gen: Callable                 # gen(p, x) -> honest hints (ints)
    ref: Callable                 # ref(p, x) -> outs (ints)
    zref: Callable                # zref(z3, p, x) -> outs (z3)
    admissible: Callable | None = None       # admissible(p, x) -> bool  (ints)
    zadmissible: Callable | None = None      # zadmissible(z3, p, x)
    note: str = ""
    signed_hints: tuple[str, ...] = ()       # hints read as the signed representative in [-(p-1)/2, (p-1)/2]


def _m(p):
    return p.n + 1


# product_shift
def _ps_inputs(p):
    return {"u": (0, 2 ** (2 * p.sig)), "dsp": (0, DSP_ROWS)}


# state_shift
def _ss_inputs(p):
    return {"M": (0, 2**p.acc_sig), "dsp": (0, DSP_ROWS)}


# signed_sum
def _sum_inputs(p):
    W, B = p.acc_bits, p.prod_bits
    return {"t[0]": (-(2**W) + 1, 2**W), **{f"t[{i}]": (-(2**B) + 1, 2**B) for i in range(1, _m(p))}}


def _terms(p, x):
    return [x[f"t[{i}]"] for i in range(_m(p))]


def _nz(x):
    if isinstance(x, int):
        return int(x != 0)
    import z3
    return z3.If(x != 0, 1, 0)


def _sum_call(ctx, p, x, h):
    sgn, v2, v1, v0, l = signed_sum(ctx, p, _terms(p, x), h)
    return (sgn * _nz(v2 + v1 + v0), v2, v1, v0, l)


def _sum_ref(p, x):
    S = sum(_terms(p, x))
    v = abs(S)
    return (int(S < 0), v >> 20, (v >> 10) & 1023, v & 1023, v.bit_length())


def _sum_zref(z3, p, x):
    S = sum(_terms(p, x))
    v = z3.If(S < 0, -S, S)
    return (z3.If(S < 0, 1, 0), v / 2**20, (v / 2**10) % 1024, v % 1024, _zbitlen(z3, v, p.l_hi))


# normalise
def _norm_inputs(p):
    return {"E": (p.floor, V2.last(p) + 1), "l": (0, p.l_hi + 1), "v2": (0, 1024), "v1": (0, 1024), "v0": (0, 1024)}


def _norm_v(x):
    return x["v2"] * 2**20 + x["v1"] * 2**10 + x["v0"]


def _norm_ref(p, x):
    E, l, v = x["E"], x["l"], _norm_v(x)
    e1 = max(E + l - p.width, p.e_min)
    k = E - e1 + p.acc_sig - p.width
    M = (v << k) if k >= 0 else (v >> -k)
    z = int(M == 0)
    return (p.floor if z else e1, M, z, int(not z and e1 > p.e_max), int(E == p.floor))


def _norm_zref(z3, p, x):
    E, l, v = x["E"], x["l"], _norm_v(x)
    e_raw = E + l - p.width
    e1 = z3.If(e_raw < p.e_min, p.e_min, e_raw)
    k = E - e1 + p.acc_sig - p.width
    M = z3.IntVal(-1)
    for kk in range(p.k_lo, p.k_hi + 1):
        M = z3.If(k == kk, v * 2**kk if kk >= 0 else v / 2 ** (-kk), M)
    z = M == 0
    return (z3.If(z, p.floor, e1), M, z3.If(z, 1, 0), z3.If(z3.And(z3.Not(z), e1 > p.e_max), 1, 0), z3.If(E == p.floor, 1, 0))


# group_max
def _max_inputs(p):
    lo, last = V2.cand_lo(p), V2.last(p)
    return {**{f"cand[{i}]": (lo, last + 1) for i in range(_m(p))}, **{f"s4[{i}]": (0, 4) for i in range(_m(p))},
            "l": (0, p.l_hi + 1)}


def _cands(p, x):
    return [x[f"cand[{i}]"] for i in range(_m(p))]


def _s4s(p, x):
    return [x[f"s4[{i}]"] for i in range(_m(p))]


def _max_call(ctx, p, x, h):
    return group_max(ctx, p, _cands(p, x), _s4s(p, x), x["l"], h)


def _max_call_noatt(ctx, p, x, h):
    return group_max(ctx, p, _cands(p, x), _s4s(p, x), x["l"], h, attained=False)


def _max_ref(p, x):
    E = max(p.floor, *_cands(p, x))
    return (E, *[2 * min(E - c, DT) + int(s in (1, 2)) for c, s in zip(_cands(p, x), _s4s(p, x))])


def _max_zref(z3, p, x):
    E = z3.IntVal(p.floor)
    for c in _cands(p, x):
        E = z3.If(c > E, c, E)
    return (E, *[2 * z3.If(E - c < DT, E - c, DT) + z3.If(z3.Or(s == 1, s == 2), 1, 0) for c, s in zip(_cands(p, x), _s4s(p, x))])


# pack + epilogue on limbs
def _pack_inputs(p):
    G = len(p.groups)
    return {"s": (0, 2), "e_hat": (p.floor, p.exp_hi + 1), "M": (0, 2**p.acc_sig), "z": (0, 2),
            **{f"ovf[{g}]": (0, 2) for g in range(G)}, **{f"sgn[{g}]": (0, 2) for g in range(G)}}


def _pack_call(ctx, p, x, h):
    G = len(p.groups)
    y_hi, y_lo = pack_limbs(ctx, p, (x["s"], x["e_hat"], x["M"], x["z"]), [(x[f"ovf[{g}]"], x[f"sgn[{g}]"]) for g in range(G)], h)
    y16 = epilogue_limbs(ctx, p, y_hi, y_lo, h)
    return (y_hi, y_lo, y16)


def _pack_word(p, x) -> int:
    G = len(p.groups)
    for g in range(G):
        if x[f"ovf[{g}]"]:
            return (x[f"sgn[{g}]"] << 31) | (p.t_finite << 23)
    if x["z"]:
        return 0
    fld = x["e_hat"] + p.bias - 1 + (x["M"] >> 23)
    return (x["s"] << 31) + fld * 2**23 + (x["M"] & (2**23 - 1))


def _pack_ref(p, x):
    y = _pack_word(p, x)
    hi, lo = y >> 16, y & 0xFFFF
    return (hi, lo, (((y >> 16) & 1) + y + (2**15 - 1)) >> 16)


def _pack_zref(z3, p, x):
    G = len(p.groups)
    hidden = x["M"] / 2**23
    fld = x["e_hat"] + p.bias - 1 + hidden
    fin = x["s"] * 2**31 + fld * 2**23 + x["M"] % 2**23
    y = z3.If(x["z"] == 1, 0, fin)
    for g in range(G - 1, -1, -1):
        y = z3.If(x[f"ovf[{g}]"] == 1, x[f"sgn[{g}]"] * 2**31 + p.t_finite * 2**23, y)
    return (y / 2**16, y % 2**16, ((y / 2**16) % 2 + y + (2**15 - 1)) / 2**16)


def _pack_admissible(p, x):
    if x["z"] != int(x["M"] == 0):
        return False
    if x["z"]:
        return x["s"] == 0 and x["e_hat"] == p.floor
    if not (x["e_hat"] >= p.e_min and (x["M"] >> 23 == 1 or x["e_hat"] == p.e_min)):
        return False
    # a finite non-overflowing state has e_hat <= e_max; the overflow flag says otherwise
    G = len(p.groups)
    any_ovf = any(x[f"ovf[{g}]"] for g in range(G))
    return any_ovf or x["e_hat"] <= p.e_max


def _pack_zadmissible(z3, p, x):
    G = len(p.groups)
    any_ovf = z3.Or([x[f"ovf[{g}]"] == 1 for g in range(G)])
    return z3.And(x["z"] == z3.If(x["M"] == 0, 1, 0),
                  z3.Implies(x["z"] == 1, z3.And(x["s"] == 0, x["e_hat"] == p.floor)),
                  z3.Implies(x["z"] == 0, z3.And(x["e_hat"] >= p.e_min, z3.Or(x["M"] >= 2**23, x["e_hat"] == p.e_min))),
                  z3.Or(any_ovf, x["e_hat"] <= p.e_max))


def _pack_gen(p, x):
    y = _pack_word(p, x)
    return {**pack_gen(p, x["M"]), **epilogue_gen(p, y >> 16, y & 0xFFFF)}


# state_decode on limbs
def _sd_inputs(p):
    return {"c_hi": (0, 2**16), "c_lo": (0, 2**16)}


def _sd_call(ctx, p, x, h):
    return state_decode_limbs(ctx, p, x["c_hi"], x["c_lo"], h)


def _sd_ref(p, x):
    c = x["c_hi"] * 2**16 + x["c_lo"]
    t = (c >> 23) & 255
    M = (c & (2**23 - 1)) + (2**23 if t else 0)
    return (c >> 31, p.floor if M == 0 else max(t, 1) - p.bias, M, int(M == 0))


def _sd_zref(z3, p, x):
    c = x["c_hi"] * 2**16 + x["c_lo"]
    t = (c / 2**23) % 256
    M = c % 2**23 + z3.If(t == 0, 0, 2**23)
    return (c / 2**31, z3.If(M == 0, p.floor, z3.If(t == 0, 1, t) - p.bias), M, z3.If(M == 0, 1, 0))


def _sd_admissible(p, x):
    return ((x["c_hi"] >> 7) & 255) != p.t_finite


GADGETS: dict[str, Spec] = {
    "product_shift_chunked": Spec(
        "product_shift_chunked", _ps_inputs, ("u_hi", "u_lo"),
        lambda ctx, p, x, h: (product_shift(ctx, p, x["u"], x["dsp"], h),),
        lambda p, x: product_shift_gen(p, x["u"], x["dsp"]),
        lambda p, x: (_signed_shift(x["u"], p.prod_shift, x["dsp"]),),
        lambda z3, p, x: (_zshift(z3, x["u"], p.prod_shift, x["dsp"]),),
        note="SHIFT (3.5M rows) as SHIFT_HI + SHIFT_LO on the bytes of u"),
    "state_shift_chunked": Spec(
        "state_shift_chunked", _ss_inputs, ("M2", "M1", "M0"),
        lambda ctx, p, x, h: (state_shift(ctx, p, x["M"], x["dsp"], h),),
        lambda p, x: state_shift_gen(p, x["M"], x["dsp"]),
        lambda p, x: (_signed_shift(x["M"], p.acc_shift, x["dsp"]),),
        lambda z3, p, x: (_zshift(z3, x["M"], p.acc_shift, x["dsp"]),),
        note="SSHIFT_HI/LO (2 x 221k rows) as three byte tables"),
    "signed_sum_chunked": Spec(
        "signed_sum_chunked", _sum_inputs, ("sgn", "v2", "v1", "v0", "l"), _sum_call,
        lambda p, x: signed_sum_gen(p, _terms(p, x)), _sum_ref, _sum_zref,
        note="v = (1 - 2 sgn) S as three 10-bit chunks through LEAD10; l materialised"),
    "normalise_chunked": Spec(
        "normalise_chunked", _norm_inputs, (),
        lambda ctx, p, x, h: normalise(ctx, p, x["E"], x["l"], x["v2"], x["v1"], x["v0"]),
        lambda p, x: {}, _norm_ref, _norm_zref,
        admissible=lambda p, x: x["l"] == _norm_v(x).bit_length(),
        zadmissible=lambda z3, p, x: x["l"] == _zbitlen(z3, _norm_v(x), p.l_hi),
        note="M' = (v2 2^10 + v1) A1 + TNORM10(v0, k); every product < 2^24"),
    "group_max_product": Spec(
        "group_max_product", _max_inputs, ("E",), _max_call,
        lambda p, x: {"E": max(p.floor, *_cands(p, x))}, _max_ref, _max_zref,
        note="attained as prod_i (E - cand_i) (E - floor) = 0; dsp packed output", signed_hints=("E",)),
    "group_max_product:without-attained": Spec(
        "group_max_product", _max_inputs, ("E",), _max_call_noatt,
        lambda p, x: {"E": max(p.floor, *_cands(p, x))}, _max_ref, _max_zref,
        note="control: dropping the product identity must be SAT", signed_hints=("E",)),
    "pack_epilogue_limbs": Spec(
        "pack_epilogue_limbs", _pack_inputs, ("M_hi8", "M_lo16", "hi2", "lsb", "carry", "rem"), _pack_call, _pack_gen,
        _pack_ref, _pack_zref, admissible=_pack_admissible, zadmissible=_pack_zadmissible,
        note="the F4 word class repaired: M = M_hi8 2^16 + M_lo16, y_hi an expression < 2^16, epilogue per limb"),
    "state_decode_limbs": Spec(
        "state_decode_limbs", _sd_inputs, ("hdr", "f_hi", "z", "mz_hi"), _sd_call,
        lambda p, x: state_decode_gen(p, x["c_hi"], x["c_lo"]), _sd_ref, _sd_zref, admissible=_sd_admissible,
        note="word-in unit only (absent from the state-chained VU AIR)"),
}


# ---------------------------------------------------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class QueryResult:
    gadget: str
    query: str
    status: str
    seconds: float
    model: dict | None = None
    note: str = ""


def _model_dict(ctx, named: dict) -> dict:
    m = ctx.solver.model()
    out = {}
    for k, v in named.items():
        if isinstance(v, int):
            out[k] = v
        else:
            val = m.eval(v, model_completion=True)
            out[k] = val.as_long() if hasattr(val, "as_long") else str(val)
    return out


def _setup(spec: Spec, p: Params, ctx, modulus: int | None):
    x = {}
    for name, (lo, hi) in spec.inputs(p).items():
        v = ctx.fresh(f"in.{name}")
        ctx.solver.add(v >= lo, v < hi)
        x[name] = v
    h = {}
    for name in spec.hints:
        v = ctx.fresh(f"hint.{name}")
        if modulus is not None:
            # a field element read as an integer: the canonical [0, p) representative, or the signed one for hints
            # whose integer meaning is signed (the exponent E); both are bijections F_p -> Z
            if name in spec.signed_hints:
                ctx.solver.add(v >= -(modulus - 1) // 2, v <= (modulus - 1) // 2)
            else:
                ctx.solver.add(v >= 0, v < modulus)
        h[name] = v
    return x, h


def query(spec: Spec, p: Params, kind: str, *, modulus: int | None = None, timeout_ms: int = 60_000) -> QueryResult:
    """kind: 'integer' (differs-from-reference over Z), 'wrapfree' (some multiplier != 0 over F_p), 'direct'
    (residue differs from the reference over F_p)."""
    ctx = ModCtx(modulus) if kind != "integer" else ZCtx()
    z3 = ctx.z3
    x, h = _setup(spec, p, ctx, modulus if kind != "integer" else None)
    try:
        outs = spec.call(ctx, p, x, h)
    except Reject as exc:
        return QueryResult(spec.name, kind, "rejected-statically", 0.0, note=str(exc))
    if spec.zadmissible is not None:
        ctx.solver.add(spec.zadmissible(z3, p, x))
    refs = spec.zref(z3, p, x)
    label = kind
    if kind == "integer":
        ctx.solver.add(z3.Or([o != r for o, r in zip(outs, refs)]))
        note = "over Z"
    elif kind == "wrapfree":
        ctx.solver.add(z3.Or([w != 0 for w in ctx.wraps]))
        note = f"mod {modulus}, wrap-free ({len(ctx.wraps)} congruences)"
    else:
        ctx.solver.add(z3.Or([(o - r) % modulus != 0 for o, r in zip(outs, refs)]))
        note = f"mod {modulus}, direct"
    ctx.solver.set("timeout", timeout_ms)
    t0 = time.perf_counter()
    status = str(ctx.solver.check())
    seconds = time.perf_counter() - t0
    model = None
    if status == "sat":
        model = {"inputs": _model_dict(ctx, x), "hints": _model_dict(ctx, h),
                 "outputs": _model_dict(ctx, {f"out[{i}]": o for i, o in enumerate(outs)}),
                 "reference": _model_dict(ctx, {f"ref[{i}]": r for i, r in enumerate(refs)})}
        if kind == "wrapfree":
            model["wraps"] = _model_dict(ctx, {f"w{i}": w for i, w in enumerate(ctx.wraps)})
    return QueryResult(spec.name, label, status, round(seconds, 3), model, note + ("; " + spec.note if spec.note else ""))


def selfcheck(spec: Spec, p: Params, n: int = 2000, seed: int = 0) -> dict:
    """The gadget code with honest hints reproduces the reference on random admissible inputs (IntCtx)."""
    rng = random.Random(seed)
    ok = bad = skipped = 0
    for _ in range(n):
        x = {k: rng.randrange(lo, hi) for k, (lo, hi) in spec.inputs(p).items()}
        if spec.admissible is not None and not spec.admissible(p, x):
            skipped += 1
            continue
        if spec.name == "normalise_chunked":
            x["l"] = _norm_v(x).bit_length()
        h = spec.gen(p, x)
        try:
            outs = tuple(spec.call(IntCtx(), p, x, h))
        except Reject as exc:
            bad += 1
            if bad <= 3:
                print(f"  selfcheck {spec.name}: honest hints rejected: {exc}")
            continue
        if tuple(outs) == tuple(spec.ref(p, x)):
            ok += 1
        else:
            bad += 1
            if bad <= 3:
                print(f"  selfcheck {spec.name}: {x} -> {outs} != {spec.ref(p, x)}")
    return {"gadget": spec.name, "ok": ok, "bad": bad, "skipped": skipped}


def table_selfcheck(p: Params, sample: int = 3000, seed: int = 0) -> list[dict]:
    import z3
    rng = random.Random(seed)
    out = []
    for t in (SHIFT_HI, SHIFT_LO, SSHIFT_2, SSHIFT_1, SSHIFT_0, LEAD10, align4_table(p), leadnorm_table(p), tnorm10_table(p), hdr_table(p)):
        rows = 1
        for lo, hi in t.keys:
            rows *= hi - lo
        bad = 0
        n = 0
        for _ in range(min(sample, rows)):
            keys = [rng.randrange(lo, hi) for lo, hi in t.keys]
            n += 1
            in_dom, vals = t.sym(z3, *[z3.IntVal(k) for k in keys])
            in_dom_b = in_dom if isinstance(in_dom, bool) else z3.is_true(z3.simplify(in_dom))
            dom_int = t.domain is None or t.domain(*keys)
            if in_dom_b != dom_int:
                bad += 1
                continue
            if dom_int and tuple(z3.simplify(v).as_long() for v in vals) != t.fn(*keys):
                bad += 1
        out.append({"table": t.name, "rows": rows, "checked": n, "mismatches": bad})
    return out


def run_all(p: Params = REAL, *, timeout_s: float = 60.0, only: Sequence[str] | None = None,
            kinds: Sequence[str] = ("integer", "wrapfree", "direct")) -> dict:
    T = int(timeout_s * 1000)
    report: dict = {"params": repr(p), "modulus": BABYBEAR, "tables": table_selfcheck(p), "selfcheck": [], "z3": []}
    for r in report["tables"]:
        print(f"table {r['table']}: rows={r['rows']} checked={r['checked']} mismatches={r['mismatches']}", flush=True)
    for key, spec in GADGETS.items():
        if only and key not in only:
            continue
        if "without" not in key:
            sc = selfcheck(spec, p)
            report["selfcheck"].append(sc)
            print(f"selfcheck {key}: ok={sc['ok']} bad={sc['bad']} skipped={sc['skipped']}", flush=True)
        for kind in kinds:
            r = query(spec, p, kind, modulus=BABYBEAR if kind != "integer" else None, timeout_ms=T)
            r.gadget = key
            print(f"{key:36s} {kind:9s} {r.status:8s} {r.seconds}s  {r.note}", flush=True)
            report["z3"].append(asdict(r))
    return report


def verdicts(report: dict) -> dict[str, str]:
    """Per gadget: 'UNSAT' if integer UNSAT and (wrap-free UNSAT or direct UNSAT), else the first offending status."""
    out: dict[str, str] = {}
    by = {}
    for r in report["z3"]:
        by.setdefault(r["gadget"], {})[r["query"]] = r["status"]
    for g, qs in by.items():
        if "without" in g:
            out[g] = "SAT (expected)" if qs.get("integer") == "sat" else f"unexpected {qs}"
            continue
        integer = qs.get("integer")
        field_ok = qs.get("wrapfree") == "unsat" or qs.get("direct") == "unsat"
        if integer == "unsat" and field_ok:
            out[g] = "UNSAT (Z and F_babybear)"
        elif integer == "unsat":
            out[g] = f"UNSAT over Z; field {qs.get('wrapfree')}/{qs.get('direct')}"
        else:
            out[g] = f"{integer} over Z"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", default=None, help="comma-separated gadget keys")
    ap.add_argument("--kinds", default="integer,wrapfree,direct")
    args = ap.parse_args(argv)
    report = run_all(timeout_s=args.timeout, only=args.only.split(",") if args.only else None, kinds=tuple(args.kinds.split(",")))
    report["verdicts"] = verdicts(report)
    for g, v in report["verdicts"].items():
        print(f"{g:36s} {v}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1, default=str)
        print(f"wrote {args.json}")
    bad = [r for r in report["z3"] if r["status"] == "sat" and "without" not in r["gadget"] and r["query"] != "wrapfree"]
    bad += [r for r in report["z3"] if r["status"] == "sat" and r["query"] == "wrapfree" and "without" not in r["gadget"]]
    return 1 if bad else 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
