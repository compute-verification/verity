"""The exact Ada E4M3 transition unit (``silicon.tc_dot(ADA_E4M3_M16N8K32, c, a, b)``) as a unit-uniform gadget system.

The step: 32 E4M3 products (4-bit significands, exact 8-bit products ``u``, exponents ``e_a + e_b`` with ``e = max(t, 1) - 7``),
summed with the incoming FP32 accumulator in two groups of 16 on a **14-bit** truncating adder (``width = 14``, exponent floor
``-139``).  Everything the BF16 unit does (``unit.py``: group maximum, alignment by truncating shifts, exact signed sum,
leading bit, normalisation, overflow / zero flags, component-wise pack) is reused unchanged through the checker's gadgets on
:class:`FP8Params`, an *adder-side* view of the pipeline: the adder state is ``acc_sig = 14`` bits, so a product ``u`` enters as
``u 2^7`` (``prod_shift = width - 2 sig + 1``) and a state ``M14`` as ``M14 2^0``, exactly ``silicon.group_sum``'s
``scaled = magnitude >> 10`` on the 25-bit grid.  ``Normalise`` with ``acc_sig = width`` gives ``M14 = floor(v 2^(E - e'))``,
which is silicon's 14-bit normalisation followed by the subnormal shift; the FP32 word carries ``M14 << 10``.

The one structural novelty vs BF16 is the **lossy rescale** of the incoming accumulator: silicon truncates its 24-bit
significand to the 14-bit adder (``M >> 10``) *before* the grouped add, while its exponent still enters the group maximum and
its zero flag is that of the full 24 bits.  Here it costs no gadget class of its own: the accumulator's fraction ``f`` is
range-checked by 23 bit rows anyway, and ``Mt = sum_{i >= 10} 2^(i-10) f_i + nz 2^13`` is an affine expression of those rows
(the ten dropped bits are the ten low rows).  Unit mode pins ``Mt`` directly (the verifier computes it from the public word).

Domain: finite E4M3 operands (``0x7f`` / ``0xff`` are NaN: rejected by the verifier's decode) and finite FP32 accumulators;
a group whose result leaves the FP32 range gives the sticky infinity exactly as ``silicon.tc_dot`` (``StateOut.ovf``).
"""
from __future__ import annotations

from dataclasses import dataclass

from verity_numerical.checker.gadgets import (
    AlignProduct,
    AlignState,
    GroupMax,
    LeadingBit,
    Normalise,
    Product,
    SignedSum,
    StateOut,
)
from verity_numerical.checker.params import Params
from verity_numerical.checker.tables import norm_table

from ..compile import Expr, LigeroCtx, System

__all__ = ["ADA", "FP8Params", "K", "STEPS", "compile_fp8_unit", "pack_public", "public_vectors_fp8", "unpack_public"]

K = 32                       # operand pairs per step (mma.m16n8k32)
STEPS = 1536 // K            # 48 chained steps for GemmCoordinate<1536>
ACC_FRAC = 23                # FP32 fraction bits
Y_PACK_BITS = 22             # the claimed FP32 word, low 10 (always-zero) bits dropped: s 2^21 + t 2^13 + f >> 10


@dataclass(frozen=True)
class FP8Params(Params):
    """Adder-side widths of ``ADA_E4M3_M16N8K32``.  ``exp = 8`` is the FP32 accumulator's exponent field (bias 127: the
    ``e_min`` / ``e_max`` of ``Normalise`` and ``StateOut``); the operand format is E4M3 (``op_*``); ``acc_sig = width = 14``
    is the adder's state width (the 24-bit FP32 significand truncated by ``rescale = 10`` bits)."""

    sig: int = 4
    exp: int = 8
    acc_sig: int = 14
    width: int = 14
    floor: int = -139
    groups: tuple[int, ...] = (16, 16)
    op_exp: int = 4

    @classmethod
    def from_model(cls, model) -> FP8Params:
        """The adder-side widths of an E4M3 :class:`verity.ml.circuits.silicon.models.GroupSum`: ``groups`` and the adder ``width`` /
        ``zero_exponent`` are the model's, the operand format is the model's dtype.  ``FP8Params.from_model(ADA_E4M3_M16N8K32)
        == ADA``; ``HOPPER_E4M3_K32`` gives one group of 32 on the same 14-bit adder (``--relation fp8-hopper``).  The
        22-bit public packing (``pack_public``) assumes the 14-bit adder (``rescale == 10``), so other widths are refused."""
        from verity.ml.circuits.silicon.models import GroupSum
        from verity.ml.circuits.silicon.term import FORMATS

        if not isinstance(model, GroupSum):
            raise ValueError(f"{getattr(model, 'name', model)!r} is not a GroupSum")
        fmt = FORMATS[model.dtype]
        if fmt.nan_rule != "e4m3":
            raise ValueError(f"{model.name}: the FP8 relation decodes E4M3 operands, not {fmt.name}")
        p = cls(sig=fmt.sig_bits, op_exp=fmt.exp_bits, acc_sig=model.width, width=model.width, floor=model.zero_exponent,
                groups=tuple(model.groups))
        if p.rescale != ACC_FRAC + 1 - 14:
            raise ValueError(f"{model.name}: pack_public drops 10 bits, which is the 14-bit adder's rescale; got width {model.width}")
        return p

    @property
    def op_bias(self) -> int:
        return 2 ** (self.op_exp - 1) - 1                      # 7

    @property
    def op_e_min(self) -> int:                                 # -6: the exponent of every E4M3 subnormal
        return 1 - self.op_bias

    @property
    def op_e_max(self) -> int:                                 # 8: E4M3's all-ones exponent field is finite (except the NaN code)
        return 2**self.op_exp - 1 - self.op_bias

    @property
    def op_frac(self) -> int:
        return self.sig - 1

    @property
    def ep_lo(self) -> int:                                    # smallest nonzero product exponent
        return 2 * self.op_e_min

    @property
    def ep_hi(self) -> int:
        return 2 * self.op_e_max

    @property
    def rescale(self) -> int:                                  # dropped low bits of the FP32 significand: 24 - 14
        return ACC_FRAC + 1 - self.acc_sig

    @property
    def shift_rows(self) -> int:
        """Alignment shifts d in [0, 512): the exact bound (273) rounded up to a power of two so that every ``rng`` on a
        shift costs one 9-bit decomposition instead of two (the tables cover the widened range; conservative, sound)."""
        exact = self.e_hi(len(self.groups) - 1) - self.exp_lo + 1
        return 1 << (exact - 1).bit_length()


ADA = FP8Params()


def _pack_expr(ys, yt, yf_hi):
    return ys * 2 ** (Y_PACK_BITS - 1) + yt * 2 ** (Y_PACK_BITS - 1 - 8) + yf_hi


def pack_public(y: int) -> int:
    """The claimed FP32 word as the field element the chain end tests (``Y_PACK_BITS`` bits): its low 10 bits must be zero
    (every honest word is ``M14 << 10``; the caller rejects otherwise)."""
    if y & (2**10 - 1):
        raise ValueError("claimed FP32 word has nonzero low bits: not a tensor-core accumulator")
    return y >> 10


def unpack_public(yp: int) -> int:
    return yp << 10


def compile_fp8_unit(p: FP8Params = ADA, chain: bool = False) -> System:
    """``chain=False``: public accumulator (pinned ``c.s, c.e_hat, c.Mt``) and public output components ``y.s, y.t, y.f``.
    ``chain=True``: the accumulator's word components are hint rows decoded in-field (with the lossy rescale), the output
    components are expressions and ``sys.chain`` names what the chain test links (``c_in[j+1] = c_out[j]``, ``c_in = 0`` at a
    start, packed word = public at an end)."""
    ctx = LigeroCtx(p)
    k = sum(p.groups)
    W_, U_ = p.acc_bits, 2 * p.sig
    ctx.whitelist_product("align_state", "lt_pow(q, pow, W) gives q < 2^(W - min(d, W)), so q * pow < 2^W", 0, 2**W_)
    ctx.whitelist_product("align_prod", "lt_pow(q, sr, 2 sig) gives q < 2^(2 sig) / sr, so q * sr < 2^(2 sig)", 0, 2**U_)
    ctx.whitelist_product("norm", "driver-added check v < 2^min(s_bits, acc_sig - k) bounds v * mul < 2^max(s_bits, acc_sig)",
                          0, 2 ** max(p.s_bits, p.acc_sig))

    # --- the accumulator -------------------------------------------------------------------------------------------
    ctx.scope = "x"
    if not chain:
        s_c = ctx.pin("c.s", 0, 2)
        e_c = ctx.pin("c.e_hat", p.exp_lo, p.e_max + 1)
        Mt = ctx.pin("c.Mt", 0, 2**p.acc_sig)
    else:
        hc = ctx.hints("acc", ["s", "t", "f", "nz", "z"])
        s_c, t_c, f_c, nz, zc = hc["s"], hc["t"], hc["f"], hc["nz"], hc["z"]
        ctx.boolean("acc.s", s_c)
        ctx.boolean("acc.nz", nz)
        ctx.boolean("acc.z", zc)
        ctx.rng("acc.nz1", t_c - nz, 0, p.t_finite - 1)      # nz = 0 -> t = 0; nz = 1 -> t in [1, 254]: finite (defines t)
        ctx.eq("acc.nz0", (1 - nz) * t_c, 0)
        f_rows = ctx._bits(f_c, ACC_FRAC, "acc.f")            # f in [0, 2^23): the 23 bit rows, reused by the rescale
        ctx._learn(f_c, 0, 2**ACC_FRAC)
        M_c = f_c + nz * 2**ACC_FRAC                          # the 24-bit significand (never truncated for the zero flag)
        ctx.eq("acc.z0", zc * M_c, 0)
        ctx.inverse_nonzero("acc.z1", M_c + zc)               # z = [M = 0]
        # the lossy rescale: the adder sees the top 14 bits of the 24-bit significand (silicon: magnitude >> 10)
        Mt = Expr(ctx, {f_rows[i]: 2 ** (i - p.rescale) for i in range(p.rescale, ACC_FRAC)}, 0) + nz * 2 ** (p.acc_sig - 1)
        ctx._learn(Mt, 0, 2**p.acc_sig)
        e = t_c - nz - (p.bias - 1)
        e_c = e + zc * (p.floor - e)
    state = (s_c, e_c, Mt)
    dec_a, dec_b = [], []
    for name, dec in (("a", dec_a), ("b", dec_b)):
        for i in range(k):
            s = ctx.pin(f"{name}[{i}].s", 0, 2)
            m = ctx.pin(f"{name}[{i}].m", 0, 2**p.sig)
            e = ctx.pin(f"{name}[{i}].e", p.op_e_min, p.op_e_max + 1)
            dec.append((s, m, e))

    # --- products --------------------------------------------------------------------------------------------------
    prods = []
    for i in range(k):
        ctx.scope = f"prod[{i}]"
        h = ctx.hints(f"prod[{i}]", ["z"])
        prods.append(Product.check(ctx, p, dec_a[i], dec_b[i], h, f"prod[{i}]"))

    # --- groups (identical to unit.py on the adder-side params) ------------------------------------------------------
    ovfs = []
    start = 0
    z = 0
    for g, size in enumerate(p.groups):
        group = prods[start:start + size]
        start += size
        tag = f"g{g}"
        cands = [state[1]] + [e for _, _, e in group]
        ctx.scope = f"{tag}.max"
        hmax = {"E": ctx.hint(f"{tag}.max.E"), "sel": [ctx.hint(f"{tag}.max.sel[{j}]") for j in range(size + 2)]}
        E, ds = GroupMax.check(ctx, p, cands, hmax, f"{tag}.max")
        ctx.scope = f"{tag}.align_state"
        q_acc = AlignState.check(ctx, p, state[2], ds[0], ctx.hints(f"{tag}.acc_align", ["q", "r"]), f"{tag}.align_state")
        q_prods = []
        for i, ((u, _, _), d) in enumerate(zip(group, ds[1:])):
            ctx.scope = f"{tag}.align_prod[{i}]"
            hq = ctx.hints(f"{tag}.prod_align[{i}]", ["q", "r"])
            q_prods.append(AlignProduct.check(ctx, p, u, d, hq, f"{tag}.align_prod[{i}]"))
        ctx.scope = f"{tag}.sum"
        terms = [(state[0], q_acc)] + [(s, q) for (_, s, _), q in zip(group, q_prods)]
        sgn, v = SignedSum.check(ctx, p, terms, ctx.hints(f"{tag}.sum", ["sgn", "z"]), f"{tag}.sum")
        ctx.scope = f"{tag}.lead"
        l = LeadingBit.check(ctx, p, v, ctx.hints(f"{tag}.lead", ["l"]), f"{tag}.lead")
        ctx.scope = f"{tag}.norm"
        hn = ctx.hints(f"{tag}.norm", ["clamp", "M", "r"])
        ctx.boolean(f"{tag}.norm.clamp", hn["clamp"])
        ctx.rng(f"{tag}.norm.M", hn["M"], 0, 2**p.acc_sig)
        e_raw = E + l - p.width
        e1 = e_raw + hn["clamp"] * (p.e_min - e_raw)
        kk = E - e1 + p.acc_sig - p.width
        mul, _div, _idiv = ctx.lookup(f"{tag}.norm.norm", norm_table(p), kk)
        vbound = ctx.table_by_key(mul, lambda kv: 2 ** min(p.s_bits, p.acc_sig - kv))
        ctx.lt_table(f"{tag}.norm.vbound", v, vbound, p.s_bits)        # theorem: v < 2^(acc_sig - k) whenever k >= 0
        e1_, M = Normalise.check(ctx, p, E, l, v, hn, f"{tag}.norm")
        ctx.scope = f"{tag}.out"
        s_out, e_out, M_out, ovf_eff, z = StateOut.check(ctx, p, sgn, e1_, M, ctx.hints(f"{tag}.out", ["z", "ovf"]), f"{tag}.out")
        ovfs.append((ovf_eff, sgn))
        state = (s_out, e_out, M_out)

    # --- pack: the FP32 word's components from the 14-bit state (fraction = low13 << 10) -----------------------------
    ctx.scope = "pack"
    hp = ctx.hints("pack", ["hidden", "low"])
    s, e1, M = state
    hidden, low = hp["hidden"], hp["low"]
    ctx.boolean("pack.hidden", hidden)
    ctx.rng("pack.low", low, 0, 2 ** (p.acc_sig - 1))
    ctx.eq("pack.M", hidden * 2 ** (p.acc_sig - 1) + low, M)
    fld = e1 + (p.bias - 1) + hidden
    any_ovf, inf_sign, none_yet = 0, 0, 1
    for ovf_g, sgn_g in ovfs:
        first = none_yet * ovf_g
        any_ovf = any_ovf + first
        inf_sign = inf_sign + first * sgn_g
        none_yet = none_yet - none_yet * ovf_g
    live = (1 - any_ovf) * (1 - z)
    ys, yt, yf_hi = any_ovf * inf_sign + live * s, any_ovf * p.t_finite + live * fld, live * low
    yf = yf_hi * 2**p.rescale
    ctx.sys.outputs = {"y.s": ys, "y.t": yt, "y.f": yf}
    if not chain:
        ctx.eq_pub("y.s", ys, "y.s")
        ctx.eq_pub("y.t", yt, "y.t")
        ctx.eq_pub("y.f", yf, "y.f")
        return ctx.sys
    ypk = _pack_expr(ys, yt, yf_hi)
    ctx.sys.outputs["y16"] = ypk
    names = {n: i for i, n in ctx.sys.hints}
    ctx.sys.chain = {"c": [names["acc.s"], names["acc.t"], names["acc.f"]], "y": [ys, yt, yf], "y16": ypk}
    return ctx.sys


# ---------------------------------------------------------------------------------------------------------------------
# The verifier's public work
# ---------------------------------------------------------------------------------------------------------------------

def public_vectors_fp8(p: FP8Params, c, a, b, y, device="cpu"):
    """The verifier's per-unit decode of the public words (``unit.public_vectors`` for E4M3 / FP32): ``a``/``b`` (B, 32)
    E4M3 bytes -> pinned (s, m, e); unit mode also ``c`` (B,) -> (s, e_hat, Mt) and ``y`` (B,) -> components.  Raises
    ``ValueError`` off the domain: a NaN byte (``0x7f`` / ``0xff``), a non-finite accumulator, a claimed word with nonzero
    low bits."""
    import torch

    a = torch.as_tensor(a, dtype=torch.int64, device=device)
    b = torch.as_tensor(b, dtype=torch.int64, device=device)
    k = sum(p.groups)
    if a.ndim != 2 or a.shape[1] != k or b.shape != a.shape:
        raise ValueError("shape mismatch")
    bad = []
    if ((a < 0) | (a >= 256) | (b < 0) | (b >= 256)).any():
        bad.append("x.a/x.b range")
    if (((a & 0x7F) == 0x7F) | ((b & 0x7F) == 0x7F)).any():
        bad.append("E4M3 NaN operand")
    out: dict = {}
    if c is not None:
        c = torch.as_tensor(c, dtype=torch.int64, device=device)
        y = torch.as_tensor(y, dtype=torch.int64, device=device)
        if c.shape != (a.shape[0],) or y.shape != c.shape:
            raise ValueError("shape mismatch")
        if ((c < 0) | (c >= 2**32)).any():
            bad.append("x.c range")
        if ((y < 0) | (y >= 2**32)).any():
            bad.append("y range")
        t = (c >> ACC_FRAC) & 0xFF
        f = c & (2**ACC_FRAC - 1)
        if (t == 0xFF).any():
            bad.append("acc.t off the finite range")
        nz = (t != 0).to(torch.int64)
        zz = ((t == 0) & (f == 0)).to(torch.int64)
        M = f + nz * 2**ACC_FRAC
        e = t - nz - (p.bias - 1)
        out["c.s"] = c >> 31
        out["c.e_hat"] = e + zz * (p.floor - e)
        out["c.Mt"] = M >> p.rescale
        if (y & (2**p.rescale - 1)).any():
            bad.append("claimed word has nonzero low bits")
        out["y.s"] = y >> 31
        out["y.t"] = (y >> ACC_FRAC) & 0xFF
        out["y.f"] = y & (2**ACC_FRAC - 1)
    for name, w in (("a", a), ("b", b)):
        t = (w >> p.op_frac) & (2**p.op_exp - 1)
        f = w & (2**p.op_frac - 1)
        nz = (t != 0).to(torch.int64)
        m = f + nz * 2**p.op_frac
        e = t - nz - (p.op_bias - 1)
        s = w >> 7
        for i in range(k):
            out[f"{name}[{i}].s"] = s[:, i]
            out[f"{name}[{i}].m"] = m[:, i]
            out[f"{name}[{i}].e"] = e[:, i]
    if bad:
        raise ValueError("public inputs rejected: " + "; ".join(bad))
    return out
