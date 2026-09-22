"""The gadgets: each is ``contract(p)`` (its ranges as data), ``gen(p, ...)`` (the honest hints) and
``check(ctx, p, ..., h)`` (its predicates, returning its outputs as expressions).

A *hint* is a witness column the checker cannot compute by ring arithmetic from earlier values: a bit of a
decomposition, a flag, a selector, a quotient or remainder, a leading-bit position.  Everything else (products,
sums, masked exponents, the packed word) is an expression the checker forms itself.  ``check`` assumes its inputs
lie in the contract's input ranges (they are public words checked once by the step, or outputs of earlier gadgets
whose contracts guarantee them) and enforces every hint range, every checked expression and every identity.

Ranges are half-open ``(lo, hi)``.  Names in ``checked`` are expressions of hints/inputs whose range the checker
enforces although they are not columns; a field embedding must range-check them as well.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .params import Params, Reject
from .tables import align_table, lead_table, norm_table, pow_table

__all__ = [
    "AlignProduct",
    "AlignState",
    "Bf16Decode",
    "Contract",
    "Epilogue",
    "Fp32Decode",
    "GroupMax",
    "LeadingBit",
    "Normalise",
    "Pack",
    "Product",
    "SignedSum",
    "StateOut",
]

Range = tuple[int, int]


@dataclass(frozen=True)
class Contract:
    gadget: str
    inputs: dict[str, Range]
    hints: dict[str, Range]
    checked: dict[str, Range] = field(default_factory=dict)
    outputs: dict[str, Range] = field(default_factory=dict)
    tables: tuple[str, ...] = ()


def _bit() -> Range:
    return (0, 2)


def _bits(n: int) -> Range:
    return (0, 2**n)


# ---------------------------------------------------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------------------------------------------------

class Bf16Decode:
    """Operand word w = s 2^(exp+frac) + t 2^frac + f  ->  m = f + 2^frac [t != 0],  e = max(t, 1) - bias.

    The flag nz = [t != 0] is proved by (1 - nz) t = 0 and t - nz in [0, t_finite - 1); the range of t itself
    excludes the all-ones field (infinities and NaN are outside the domain, so the word is rejected)."""

    name = "bf16_decode"

    @staticmethod
    def contract(p: Params) -> Contract:
        return Contract(
            "bf16_decode",
            inputs={"w": _bits(p.word_bits)},
            hints={"s": _bit(), "t": (0, p.t_finite), "f": _bits(p.frac), "nz": _bit()},
            checked={"t-nz": (0, p.t_finite - 1)},
            outputs={"m": _bits(p.sig), "e": (p.e_min, p.e_max + 1)},
        )

    @staticmethod
    def gen(p: Params, w: int) -> dict:
        t = (w >> p.frac) & (2**p.exp - 1)
        return {"s": w >> (p.word_bits - 1), "t": t, "f": w & (2**p.frac - 1), "nz": int(t != 0)}

    @staticmethod
    def check(ctx, p: Params, w, h: dict, tag: str = "bf16"):
        s, t, f, nz = h["s"], h["t"], h["f"], h["nz"]
        ctx.boolean(f"{tag}.s", s)
        ctx.rng(f"{tag}.t", t, 0, p.t_finite)
        ctx.rng(f"{tag}.f", f, 0, 2**p.frac)
        ctx.boolean(f"{tag}.nz", nz)
        ctx.eq(f"{tag}.word", s * 2 ** (p.word_bits - 1) + t * 2**p.frac + f, w)
        ctx.eq(f"{tag}.nz0", (1 - nz) * t, 0)
        ctx.rng(f"{tag}.nz1", t - nz, 0, p.t_finite - 1)
        m = f + nz * 2**p.frac
        e = t - nz - (p.bias - 1)          # max(t, 1) - bias, since max(t, 1) = t + (1 - nz)
        return s, m, e


class Fp32Decode:
    """Accumulator word c -> the state (s, e_hat, M): M = f + 2^acc_frac [t != 0], e = max(t, 1) - bias, and
    the zero flag z = [M = 0] masking the exponent to the floor (a zero term never raises the group maximum)."""

    name = "fp32_decode"

    @staticmethod
    def contract(p: Params) -> Contract:
        return Contract(
            "fp32_decode",
            inputs={"c": _bits(p.acc_word_bits)},
            hints={"s": _bit(), "t": (0, p.t_finite), "f": _bits(p.acc_frac), "nz": _bit(), "z": _bit()},
            checked={"t-nz": (0, p.t_finite - 1), "M+z-1": _bits(p.acc_sig)},
            outputs={"s": _bit(), "e_hat": (p.exp_lo, p.e_max + 1), "M": _bits(p.acc_sig)},
        )

    @staticmethod
    def gen(p: Params, c: int) -> dict:
        t = (c >> p.acc_frac) & (2**p.exp - 1)
        f = c & (2**p.acc_frac - 1)
        return {"s": c >> (p.acc_word_bits - 1), "t": t, "f": f, "nz": int(t != 0), "z": int(t == 0 and f == 0)}

    @staticmethod
    def check(ctx, p: Params, c, h: dict, tag: str = "acc"):
        s, t, f, nz, z = h["s"], h["t"], h["f"], h["nz"], h["z"]
        ctx.boolean(f"{tag}.s", s)
        ctx.rng(f"{tag}.t", t, 0, p.t_finite)
        ctx.rng(f"{tag}.f", f, 0, 2**p.acc_frac)
        ctx.boolean(f"{tag}.nz", nz)
        ctx.boolean(f"{tag}.z", z)
        ctx.eq(f"{tag}.word", s * 2 ** (p.acc_word_bits - 1) + t * 2**p.acc_frac + f, c)
        ctx.eq(f"{tag}.nz0", (1 - nz) * t, 0)
        ctx.rng(f"{tag}.nz1", t - nz, 0, p.t_finite - 1)
        M = f + nz * 2**p.acc_frac
        ctx.eq(f"{tag}.z0", z * M, 0)
        ctx.rng(f"{tag}.z1", M + z - 1, 0, 2**p.acc_sig)
        e = t - nz - (p.bias - 1)
        e_hat = e + z * (p.floor - e)
        return s, e_hat, M


# ---------------------------------------------------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------------------------------------------------

class Product:
    """u = m_a m_b (< 2^(2 sig)), e_p = e_a + e_b, s_p = s_a xor s_b; the zero flag z = [u = 0] masks the
    exponent of a zero product to the floor."""

    name = "product"

    @staticmethod
    def contract(p: Params) -> Contract:
        return Contract(
            "product",
            inputs={
                "s_a": _bit(), "m_a": _bits(p.sig), "e_a": (p.e_min, p.e_max + 1),
                "s_b": _bit(), "m_b": _bits(p.sig), "e_b": (p.e_min, p.e_max + 1),
            },
            hints={"z": _bit()},
            checked={"u+z-1": _bits(2 * p.sig)},
            outputs={"u": _bits(2 * p.sig), "s_p": _bit(), "e_hat": (p.exp_lo, p.ep_hi + 1)},
        )

    @staticmethod
    def gen(p: Params, m_a: int, m_b: int) -> dict:
        return {"z": int(m_a * m_b == 0)}

    @staticmethod
    def check(ctx, p: Params, a, b, h: dict, tag: str = "prod"):
        (s_a, m_a, e_a), (s_b, m_b, e_b) = a, b
        z = h["z"]
        ctx.boolean(f"{tag}.z", z)
        u = m_a * m_b
        ctx.eq(f"{tag}.z0", z * u, 0)
        ctx.rng(f"{tag}.z1", u + z - 1, 0, 2 ** (2 * p.sig))
        e_p = e_a + e_b
        s_p = s_a + s_b - 2 * s_a * s_b
        e_hat = e_p + z * (p.floor - e_p)
        return u, s_p, e_hat


# ---------------------------------------------------------------------------------------------------------------------
# The group maximum
# ---------------------------------------------------------------------------------------------------------------------

class GroupMax:
    """E = max(floor, cand_0, ..., cand_{m-1}), proved by upper bounds (every d_i = E - cand_i >= 0 and
    E - floor >= 0, each below shift_rows) plus a one-hot selector over the m candidates and the floor that
    attains E.  E above the maximum is attained by nothing; E below it makes some d_i negative."""

    name = "group_max"

    @staticmethod
    def contract(p: Params, m: int | None = None) -> Contract:
        m = p.n + 1 if m is None else m
        last = p.e_hi(len(p.groups) - 1)
        return Contract(
            "group_max",
            inputs={f"cand[{i}]": (p.exp_lo, last + 1) for i in range(m)},
            hints={"E": (p.floor, last + 1), **{f"sel[{i}]": _bit() for i in range(m + 1)}},
            checked={**{f"d[{i}]": (0, p.shift_rows) for i in range(m)}, "d_floor": (0, p.shift_rows)},
            outputs={"E": (p.floor, last + 1), **{f"d[{i}]": (0, p.shift_rows) for i in range(m)}},
        )

    @staticmethod
    def gen(p: Params, cands: list[int]) -> dict:
        E = max(p.floor, *cands)
        options = [*cands, p.floor]
        first = options.index(E)
        return {"E": E, "sel": [int(i == first) for i in range(len(options))]}

    @staticmethod
    def check(ctx, p: Params, cands: list, h: dict, tag: str = "max"):
        E, sel = h["E"], h["sel"]
        if len(sel) != len(cands) + 1:
            raise Reject(f"{tag}.sel: {len(sel)} selectors for {len(cands)} candidates")
        ctx.rng(f"{tag}.E", E, p.floor, p.e_hi(len(p.groups) - 1) + 1)
        for i, b in enumerate(sel):
            ctx.boolean(f"{tag}.sel[{i}]", b)
        ctx.eq(f"{tag}.onehot", sum(sel), 1)
        ctx.eq(f"{tag}.attained", sum(b * c for b, c in zip(sel, [*cands, p.floor])), E)
        ds = [E - c for c in cands]
        for i, d in enumerate(ds):
            ctx.rng(f"{tag}.d[{i}]", d, 0, p.shift_rows)
        ctx.rng(f"{tag}.d_floor", E - p.floor, 0, p.shift_rows)
        return E, ds


# ---------------------------------------------------------------------------------------------------------------------
# Alignment: truncating right shifts as quotient/remainder with tight bounds
# ---------------------------------------------------------------------------------------------------------------------

class AlignState:
    """The state term P = M 2^acc_shift (< 2^W) shifted right by d: P = 2^d q + r with 0 <= r < 2^d and the
    tight quotient bound 0 <= q < 2^(W - d).  Both are stated for d~ = min(d, W) through the POW table:
    r * 2^(W-d~) < 2^W and q * 2^d~ < 2^W; d~ = W forces q = 0, d = 0 forces r = 0."""

    name = "align_state"

    @staticmethod
    def contract(p: Params) -> Contract:
        return Contract(
            "align_state",
            inputs={"M": _bits(p.acc_sig), "d": (0, p.shift_rows)},
            hints={"q": _bits(p.acc_bits), "r": _bits(p.acc_bits)},
            checked={"r*ipow": _bits(p.acc_bits), "q*pow": _bits(p.acc_bits)},
            outputs={"q": _bits(p.acc_bits)},
            tables=("POW",),
        )

    @staticmethod
    def gen(p: Params, M: int, d: int) -> dict:
        P = M << p.acc_shift
        dd = min(d, p.acc_bits)
        q = P >> dd
        return {"q": q, "r": P - (q << dd)}

    @staticmethod
    def check(ctx, p: Params, M, d, h: dict, tag: str = "align_state"):
        q, r = h["q"], h["r"]
        W = p.acc_bits
        ctx.rng(f"{tag}.q", q, 0, 2**W)
        ctx.rng(f"{tag}.r", r, 0, 2**W)
        pow_, ipow = ctx.lookup(f"{tag}.pow", pow_table(p), d)
        ctx.eq(f"{tag}.div", q * pow_ + r, M * 2**p.acc_shift)
        ctx.lt_pow(f"{tag}.rem", r, ipow, W)           # r < 2^min(d, W)
        ctx.lt_pow(f"{tag}.quot", q, pow_, W)          # q < 2^(W - min(d, W)): the tight bound
        return q


class AlignProduct:
    """The product term P = u 2^prod_shift shifted right by d, specialised to its trailing zeros: only u is
    divided.  With (sl, sr, isr) = ALIGN[d]: u = sr q + r, r < sr (r * isr < 2^(2 sig)), q < 2^(2 sig) / sr
    (q * sr < 2^(2 sig), tight), and the aligned term is q * sl.  d <= prod_shift: sr = 1, r = 0, aligned =
    u 2^(prod_shift - d); prod_shift < d < prod_bits: aligned = floor(u / 2^(d - prod_shift)); d >= prod_bits: q = 0."""

    name = "align_product"

    @staticmethod
    def contract(p: Params) -> Contract:
        U = 2 * p.sig
        return Contract(
            "align_product",
            inputs={"u": _bits(U), "d": (0, p.shift_rows)},
            hints={"q": _bits(U), "r": _bits(U)},
            checked={"r*isr": _bits(U), "q*sr": _bits(U)},
            outputs={"aligned": _bits(p.prod_bits)},
            tables=("ALIGN",),
        )

    @staticmethod
    def gen(p: Params, u: int, d: int) -> dict:
        if d <= p.prod_shift:
            return {"q": u, "r": 0}
        if d < p.prod_bits:
            q = u >> (d - p.prod_shift)
            return {"q": q, "r": u - (q << (d - p.prod_shift))}
        return {"q": 0, "r": u}

    @staticmethod
    def check(ctx, p: Params, u, d, h: dict, tag: str = "align_prod"):
        q, r = h["q"], h["r"]
        U = 2 * p.sig
        ctx.rng(f"{tag}.q", q, 0, 2**U)
        ctx.rng(f"{tag}.r", r, 0, 2**U)
        sl, sr, isr = ctx.lookup(f"{tag}.align", align_table(p), d)
        ctx.eq(f"{tag}.div", q * sr + r, u)
        ctx.lt_pow(f"{tag}.rem", r, isr, U)            # r < sr
        ctx.lt_pow(f"{tag}.quot", q, sr, U)            # q < 2^(2 sig) / sr: the tight bound
        return q * sl


# ---------------------------------------------------------------------------------------------------------------------
# The exact signed sum
# ---------------------------------------------------------------------------------------------------------------------

class SignedSum:
    """S = sum_i (1 - 2 s_i) q_i exactly; then S = (1 - 2 sgn) v with v in [0, 2^s_bits), and z = [v = 0] with
    sgn z = 0 (the zero sum has the positive sign).  The premise |S| < 2^s_bits is a theorem of the interface
    bounds (one state term below 2^acc_bits, n product terms below 2^prod_bits): ``premise`` recomputes it
    from the contract and the checker refuses to run if it fails."""

    name = "signed_sum"

    @staticmethod
    def premise(p: Params, m: int | None = None) -> None:
        m = p.n if m is None else m
        bound = (2**p.acc_bits - 1) + m * (2**p.prod_bits - 1)
        if bound >= 2**p.s_bits:
            raise Reject(f"signed_sum premise: |S| <= {bound} is not below 2^{p.s_bits}")

    @staticmethod
    def contract(p: Params, m: int | None = None) -> Contract:
        m = p.n if m is None else m
        SignedSum.premise(p, m)
        return Contract(
            "signed_sum",
            inputs={
                "s[0]": _bit(), "q[0]": _bits(p.acc_bits),
                **{f"s[{i}]": _bit() for i in range(1, m + 1)},
                **{f"q[{i}]": _bits(p.prod_bits) for i in range(1, m + 1)},
            },
            hints={"sgn": _bit(), "z": _bit()},
            checked={"v": _bits(p.s_bits), "v+z-1": _bits(p.s_bits)},
            outputs={"sgn": _bit(), "v": _bits(p.s_bits)},
        )

    @staticmethod
    def gen(p: Params, terms: list[tuple[int, int]]) -> dict:
        S = sum(q if not s else -q for s, q in terms)
        return {"sgn": int(S < 0), "z": int(S == 0)}

    @staticmethod
    def check(ctx, p: Params, terms: list, h: dict, tag: str = "sum"):
        SignedSum.premise(p, len(terms) - 1)
        sgn, z = h["sgn"], h["z"]
        ctx.boolean(f"{tag}.sgn", sgn)
        ctx.boolean(f"{tag}.z", z)
        S = sum(q - 2 * s * q for s, q in terms)
        v = S - 2 * sgn * S
        ctx.rng(f"{tag}.v", v, 0, 2**p.s_bits)
        ctx.eq(f"{tag}.z0", z * v, 0)
        ctx.rng(f"{tag}.z1", v + z - 1, 0, 2**p.s_bits)
        ctx.eq(f"{tag}.zero_sign", sgn * z, 0)
        return sgn, v


# ---------------------------------------------------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------------------------------------------------

class LeadingBit:
    """l with 2^(l-1) <= v < 2^l (l = 0 iff v = 0), proved through LEAD[l] = (2^(l-1), 2^(s_bits-l+1)):
    (v - 2^(l-1)) * 2^(s_bits-l+1) in [0, 2^s_bits)."""

    name = "leading_bit"

    @staticmethod
    def contract(p: Params) -> Contract:
        return Contract(
            "leading_bit",
            inputs={"v": _bits(p.s_bits)},
            hints={"l": (0, p.l_hi + 1)},
            checked={"(v-lo)*ispan": _bits(p.s_bits)},
            outputs={"l": (0, p.l_hi + 1)},
            tables=("LEAD",),
        )

    @staticmethod
    def gen(p: Params, v: int) -> dict:
        return {"l": v.bit_length()}

    @staticmethod
    def check(ctx, p: Params, v, h: dict, tag: str = "lead"):
        l = h["l"]
        lo, ispan = ctx.lookup(f"{tag}.lead", lead_table(p), l)
        ctx.lt_pow(f"{tag}.window", v - lo, ispan, p.s_bits)   # 0 <= v - 2^(l-1) < 2^(l-1)
        return l


class Normalise:
    """e' = max(E + l - W, e_min) by a clamp flag, and M' = floor(v 2^k) with k = E - e' + acc_sig - W (the
    closed form of the 25-bit normalisation followed by the subnormal shift and the rescale to acc_sig bits) as
    div M' + r = v mul, r < div, with (mul, div, idiv) = NORM[k]."""

    name = "normalise"

    @staticmethod
    def contract(p: Params) -> Contract:
        last = p.e_hi(len(p.groups) - 1)
        return Contract(
            "normalise",
            inputs={"E": (p.floor, last + 1), "l": (0, p.l_hi + 1), "v": _bits(p.s_bits)},
            hints={"clamp": _bit(), "M": _bits(p.acc_sig), "r": _bits(p.norm_rem_bits)},
            checked={"clamp_proof": (0, p.cmp_rows), "r*idiv": _bits(p.norm_rem_bits), "k": (p.k_lo, p.k_hi + 1)},
            outputs={"e1": (p.e_min, p.exp_hi + 1), "M": _bits(p.acc_sig)},
            tables=("NORM",),
        )

    @staticmethod
    def gen(p: Params, E: int, l: int, v: int) -> dict:
        e_raw = E + l - p.width
        e1 = max(e_raw, p.e_min)
        k = E - e1 + p.acc_sig - p.width
        if k >= 0:
            return {"clamp": int(e_raw < p.e_min), "M": v << k, "r": 0}
        M = v >> -k
        return {"clamp": int(e_raw < p.e_min), "M": M, "r": v - (M << -k)}

    @staticmethod
    def check(ctx, p: Params, E, l, v, h: dict, tag: str = "norm"):
        clamp, M, r = h["clamp"], h["M"], h["r"]
        ctx.boolean(f"{tag}.clamp", clamp)
        ctx.rng(f"{tag}.M", M, 0, 2**p.acc_sig)
        ctx.rng(f"{tag}.r", r, 0, 2**p.norm_rem_bits)
        e_raw = E + l - p.width
        ctx.rng(f"{tag}.clamp_proof", clamp * (p.e_min - 1 - e_raw) + (1 - clamp) * (e_raw - p.e_min), 0, p.cmp_rows)
        e1 = e_raw + clamp * (p.e_min - e_raw)
        k = E - e1 + p.acc_sig - p.width
        mul, div, idiv = ctx.lookup(f"{tag}.norm", norm_table(p), k)
        ctx.eq(f"{tag}.div", div * M + r, v * mul)
        ctx.lt_pow(f"{tag}.rem", r, idiv, p.norm_rem_bits)      # r < div
        return e1, M


class StateOut:
    """The group's result as the next state: z' = [M' = 0] (the zero cases: S = 0, or truncation to zero) gives
    the unsigned zero at the floor exponent; ovf = [e' > e_max] with a nonzero magnitude is the saturation flag
    (sticky through the remaining groups, as silicon's early return)."""

    name = "state_out"

    @staticmethod
    def contract(p: Params) -> Contract:
        return Contract(
            "state_out",
            inputs={"sgn": _bit(), "e1": (p.e_min, p.exp_hi + 1), "M": _bits(p.acc_sig)},
            hints={"z": _bit(), "ovf": _bit()},
            checked={"M+z-1": _bits(p.acc_sig), "ovf_proof": (0, p.cmp_rows)},
            outputs={"s": _bit(), "e_hat": (p.exp_lo, p.exp_hi + 1), "M": _bits(p.acc_sig), "ovf_eff": _bit(), "z": _bit()},
        )

    @staticmethod
    def gen(p: Params, e1: int, M: int) -> dict:
        return {"z": int(M == 0), "ovf": int(e1 > p.e_max)}

    @staticmethod
    def check(ctx, p: Params, sgn, e1, M, h: dict, tag: str = "out"):
        z, ovf = h["z"], h["ovf"]
        ctx.boolean(f"{tag}.z", z)
        ctx.boolean(f"{tag}.ovf", ovf)
        ctx.eq(f"{tag}.z0", z * M, 0)
        ctx.rng(f"{tag}.z1", M + z - 1, 0, 2**p.acc_sig)
        ctx.rng(f"{tag}.ovf_proof", ovf * (e1 - p.e_max - 1) + (1 - ovf) * (p.e_max - e1), 0, p.cmp_rows)
        ovf_eff = ovf - ovf * z
        s = sgn - sgn * z
        e_hat = e1 + z * (p.floor - e1)
        return s, e_hat, M, ovf_eff, z


# ---------------------------------------------------------------------------------------------------------------------
# Packing and the epilogue
# ---------------------------------------------------------------------------------------------------------------------

class Pack:
    """The FP32 word of the final state: +0 when zero; the infinity of the first overflowing group's sum sign
    when any group saturated; otherwise sign, field e' + bias - 1 + hidden and the low acc_frac bits of M
    (hidden = 0 only at e' = e_min, where the field is 0: a subnormal).

    ``e1`` is ``StateOut``'s ``e_hat``, which sits at the floor ``exp_lo`` when ``z = 1`` (red team wave 2, U4): the
    finite word ``fin`` is then out of range and is discarded by the ``(1 - z)`` mask, never by a range check.  An
    implementation must keep that mask and must **not** range-check ``e1`` to ``[e_min, exp_hi)`` (it would reject
    every honest zero); the declared input range below is the range actually fed."""

    name = "pack"

    @staticmethod
    def contract(p: Params) -> Contract:
        G = len(p.groups)
        return Contract(
            "pack",
            inputs={
                "s": _bit(), "e1": (p.exp_lo, p.exp_hi + 1), "M": _bits(p.acc_sig), "z": _bit(),
                **{f"ovf[{g}]": _bit() for g in range(G)}, **{f"sgn[{g}]": _bit() for g in range(G)},
            },
            hints={"hidden": _bit(), "low": _bits(p.acc_frac)},
            outputs={"y": _bits(p.acc_word_bits)},
        )

    @staticmethod
    def gen(p: Params, M: int) -> dict:
        return {"hidden": M >> p.acc_frac, "low": M & (2**p.acc_frac - 1)}

    @staticmethod
    def check(ctx, p: Params, state, ovfs: list, h: dict, tag: str = "pack"):
        s, e1, M, z = state
        hidden, low = h["hidden"], h["low"]
        ctx.boolean(f"{tag}.hidden", hidden)
        ctx.rng(f"{tag}.low", low, 0, 2**p.acc_frac)
        ctx.eq(f"{tag}.M", hidden * 2**p.acc_frac + low, M)
        fld = e1 + (p.bias - 1) + hidden
        fin = s * 2 ** (p.acc_word_bits - 1) + fld * 2**p.acc_frac + low
        any_ovf, inf_sign, none_yet = 0, 0, 1
        for ovf_g, sgn_g in ovfs:
            first = none_yet * ovf_g
            any_ovf = any_ovf + first
            inf_sign = inf_sign + first * sgn_g
            none_yet = none_yet - none_yet * ovf_g
        inf = inf_sign * 2 ** (p.acc_word_bits - 1) + p.t_finite * 2**p.acc_frac
        return any_ovf * inf + (1 - any_ovf) * (1 - z) * fin


class Epilogue:
    """f32_to_bf16, round to nearest even, as integer arithmetic on the word: y32 = hi 2^L + lo (L = acc_frac -
    frac), hi = 2 hi2 + lsb, lo + lsb + (2^(L-1) - 1) = carry 2^L + rem, y16 = hi + carry < 2^word_bits.  A NaN
    input (the 0x7FC0 case of the target) cannot arise: the word comes from ``Pack`` on the finite domain."""

    name = "epilogue"

    @staticmethod
    def contract(p: Params) -> Contract:
        L = p.acc_frac - p.frac
        return Contract(
            "epilogue",
            inputs={"y32": _bits(p.acc_word_bits)},
            hints={"hi": _bits(p.word_bits), "lo": _bits(L), "hi2": _bits(p.word_bits - 1), "lsb": _bit(),
                   "carry": _bit(), "rem": _bits(L)},
            checked={"hi+carry": _bits(p.word_bits)},
            outputs={"y16": _bits(p.word_bits)},
        )

    @staticmethod
    def gen(p: Params, y32: int) -> dict:
        L = p.acc_frac - p.frac
        hi, lo = y32 >> L, y32 & (2**L - 1)
        lsb = hi & 1
        t = lo + lsb + (2 ** (L - 1) - 1)
        return {"hi": hi, "lo": lo, "hi2": hi >> 1, "lsb": lsb, "carry": t >> L, "rem": t & (2**L - 1)}

    @staticmethod
    def check(ctx, p: Params, y32, h: dict, tag: str = "cast"):
        L = p.acc_frac - p.frac
        hi, lo, hi2, lsb, carry, rem = (h[k] for k in ("hi", "lo", "hi2", "lsb", "carry", "rem"))
        ctx.rng(f"{tag}.hi", hi, 0, 2**p.word_bits)
        ctx.rng(f"{tag}.lo", lo, 0, 2**L)
        ctx.rng(f"{tag}.hi2", hi2, 0, 2 ** (p.word_bits - 1))
        ctx.boolean(f"{tag}.lsb", lsb)
        ctx.boolean(f"{tag}.carry", carry)
        ctx.rng(f"{tag}.rem", rem, 0, 2**L)
        ctx.eq(f"{tag}.split", hi * 2**L + lo, y32)
        ctx.eq(f"{tag}.lsb_split", 2 * hi2 + lsb, hi)
        ctx.eq(f"{tag}.round", carry * 2**L + rem, lo + lsb + (2 ** (L - 1) - 1))
        y16 = hi + carry
        ctx.rng(f"{tag}.y16", y16, 0, 2**p.word_bits)
        return y16
