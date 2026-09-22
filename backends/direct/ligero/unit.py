"""The transition unit as a Ligero system: ``compile_unit`` runs the checker's gadgets on :class:`LigeroCtx` in the
order of ``verity_numerical.checker.step.check_step_expr``; ``flatten_hints`` names ``gen_step``'s hints as rows;
``public_vectors`` is the verifier's O(1)-per-unit work on the public words (decode + output components).

Departures from ``check_step_expr`` (each argued in PROTOCOL.md, "relation unchanged"):

* the decodes of the public words (``Fp32Decode`` of ``c``, ``Bf16Decode`` of ``a[i]``, ``b[i]``) are *pinned rows*
  (s, m, e) / (s, e_hat, M): the verifier computes them from the words with the gadgets' own ``gen`` + ``check`` on
  integers and the prover commits them with an equality constraint.  The word identities ``s 2^31 + t 2^23 + f = c``
  do not fit BabyBear (2^32 > p) and are not needed once the components are public;
* ``Pack``'s output word is compared to the claimed ``y`` component-wise (sign / exponent field / fraction), never as
  the 32-bit word (2^32 > p) -- equivalent because whenever the finite branch is live, ``e1 <= e_max`` (StateOut's
  ``ovf_proof``), so every component is in its field's range;
* two range checks that are theorems of the relation bound the products of ``Normalise`` below p:
  ``v < 2^min(s_bits, acc_sig - k)`` and ``M < 2^min(acc_sig, s_bits + k)``.
"""
from __future__ import annotations

from verity_numerical.checker.gadgets import (
    AlignProduct,
    AlignState,
    Bf16Decode,
    Epilogue,
    Fp32Decode,
    GroupMax,
    LeadingBit,
    Normalise,
    Product,
    SignedSum,
    StateOut,
)
from verity_numerical.checker.ctx import IntCtx
from verity_numerical.checker.params import REAL, Params
from verity_numerical.checker.tables import norm_table

from .compile import Expr, LigeroCtx, System

__all__ = ["compile_unit", "flatten_hints", "public_vectors", "HINT_NAMES", "PIN_NAMES", "PUB_NAMES"]


def _k(p: Params) -> int:
    return sum(p.groups)


def compile_unit(p: Params = REAL, chain: bool = False) -> System:
    """``chain=False``: the unit with a public accumulator word ``c`` and public output components (scope ``unit``).
    ``chain=True``: the accumulator's word components ``c.s, c.t, c.f`` are *hint rows* decoded in-field
    (``Fp32Decode`` minus the word identity, which is what the components are), the output components stay
    expressions (``sys.outputs``), and the ``Epilogue`` (RN-even cast to BF16) is constrained in every column on the
    unit's own output word; ``sys.chain`` names the rows the protocol's chain test links across columns
    (``c_in[j+1] = c_out[j]``, ``c_in = 0`` at a chain start, ``y16 = public`` at a chain end)."""
    ctx = LigeroCtx(p)
    k = _k(p)
    W_, U_ = p.acc_bits, 2 * p.sig
    ctx.whitelist_product("align_state", "lt_pow(q, pow, W) gives q < 2^(W - min(d, W)), so q * pow < 2^W", 0, 2**W_)
    ctx.whitelist_product("align_prod", "lt_pow(q, sr, 2 sig) gives q < 2^(2 sig) / sr, so q * sr < 2^(2 sig)", 0, 2**U_)
    ctx.whitelist_product("norm", "driver-added checks v < 2^min(s_bits, acc_sig - k) and M < 2^min(acc_sig, s_bits + k) "
                          "bound v * mul < 2^acc_sig and div * M < 2^s_bits", 0, 2 ** max(p.s_bits, p.acc_sig))

    # --- the accumulator: pinned decode (unit) or in-field decode of hint components (chain) ------------------------
    ctx.scope = "x"
    if not chain:
        s_c = ctx.pin("c.s", 0, 2)
        e_c = ctx.pin("c.e_hat", p.exp_lo, p.e_max + 1)
        M_c = ctx.pin("c.M", 0, 2**p.acc_sig)
    else:
        hc = ctx.hints("acc", ["s", "t", "f", "nz", "z"])
        s_c, t_c, f_c, nz, zc = hc["s"], hc["t"], hc["f"], hc["nz"], hc["z"]
        # Fp32Decode.check without ``acc.word``: (s, t, f) *are* the word here (c = s 2^31 + t 2^23 + f > p is
        # never formed; the components' ranges below make the word unique).
        ctx.boolean("acc.s", s_c)
        ctx.rng("acc.t", t_c, 0, p.t_finite)
        ctx.rng("acc.f", f_c, 0, 2**p.acc_frac)
        ctx.boolean("acc.nz", nz)
        ctx.boolean("acc.z", zc)
        ctx.eq("acc.nz0", (1 - nz) * t_c, 0)
        ctx.rng("acc.nz1", t_c - nz, 0, p.t_finite - 1)
        M_c = f_c + nz * 2**p.acc_frac
        ctx.eq("acc.z0", zc * M_c, 0)
        ctx.rng("acc.z1", M_c + zc - 1, 0, 2**p.acc_sig)
        e = t_c - nz - (p.bias - 1)
        e_c = e + zc * (p.floor - e)
    state = (s_c, e_c, M_c)
    dec_a, dec_b = [], []
    for name, dec in (("a", dec_a), ("b", dec_b)):
        for i in range(k):
            s = ctx.pin(f"{name}[{i}].s", 0, 2)
            m = ctx.pin(f"{name}[{i}].m", 0, 2**p.sig)
            e = ctx.pin(f"{name}[{i}].e", p.e_min, p.e_max + 1)
            dec.append((s, m, e))

    # --- products ----------------------------------------------------------------------------------------------------
    prods = []
    for i in range(k):
        ctx.scope = f"prod[{i}]"
        h = ctx.hints(f"prod[{i}]", ["z"])
        prods.append(Product.check(ctx, p, dec_a[i], dec_b[i], h, f"prod[{i}]"))

    # --- groups ------------------------------------------------------------------------------------------------------
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
        # The two theorem checks (the products' integer bounds); the lookup is memoised and reused by Normalise.check.
        ctx.boolean(f"{tag}.norm.clamp", hn["clamp"])
        ctx.rng(f"{tag}.norm.M", hn["M"], 0, 2**p.acc_sig)
        e_raw = E + l - p.width
        e1 = e_raw + hn["clamp"] * (p.e_min - e_raw)
        kk = E - e1 + p.acc_sig - p.width
        mul, div, idiv = ctx.lookup(f"{tag}.norm.norm", norm_table(p), kk)
        # derived columns: vbound(k) = 2^min(s_bits, acc_sig - k), mbound(k) = 2^min(acc_sig, s_bits + k)
        vbound = ctx.table_by_key(mul, lambda kv: 2 ** min(p.s_bits, p.acc_sig - kv))
        mbound = ctx.table_by_key(mul, lambda kv: 2 ** min(p.acc_sig, p.s_bits + kv))
        ctx.lt_table(f"{tag}.norm.vbound", v, vbound, p.s_bits)
        ctx.lt_table(f"{tag}.norm.mbound", hn["M"], mbound, p.acc_sig)
        e1_, M = Normalise.check(ctx, p, E, l, v, hn, f"{tag}.norm")

        ctx.scope = f"{tag}.out"
        s_out, e_out, M_out, ovf_eff, z = StateOut.check(ctx, p, sgn, e1_, M, ctx.hints(f"{tag}.out", ["z", "ovf"]), f"{tag}.out")
        ovfs.append((ovf_eff, sgn))
        state = (s_out, e_out, M_out)

    # --- pack, component-wise --------------------------------------------------------------------------------------
    ctx.scope = "pack"
    hp = ctx.hints("pack", ["hidden", "low"])
    s, e1, M = state
    hidden, low = hp["hidden"], hp["low"]
    ctx.boolean("pack.hidden", hidden)
    ctx.rng("pack.low", low, 0, 2**p.acc_frac)
    ctx.eq("pack.M", hidden * 2**p.acc_frac + low, M)
    fld = e1 + (p.bias - 1) + hidden
    any_ovf, inf_sign, none_yet = 0, 0, 1
    for ovf_g, sgn_g in ovfs:
        first = none_yet * ovf_g
        any_ovf = any_ovf + first
        inf_sign = inf_sign + first * sgn_g
        none_yet = none_yet - none_yet * ovf_g
    live = (1 - any_ovf) * (1 - z)
    ys, yt, yf = any_ovf * inf_sign + live * s, any_ovf * p.t_finite + live * fld, live * low
    ctx.sys.outputs = {"y.s": ys, "y.t": yt, "y.f": yf}
    if not chain:
        ctx.eq_pub("y.s", ys, "y.s")
        ctx.eq_pub("y.t", yt, "y.t")
        ctx.eq_pub("y.f", yf, "y.f")
        return ctx.sys

    # --- epilogue (chain): RN-even cast of this unit's output word, on its components ------------------------------
    # Epilogue.check with y32 = hi 2^L + lo: since y32 = s 2^31 + t 2^23 + f and L = 16 < 23, the split is
    # f = fhi 2^L + lo (fhi < 2^(acc_frac - L)) and hi = s 2^15 + t 2^7 + fhi -- the same integers, never a word > p.
    ctx.scope = "epi"
    L = p.acc_frac - p.frac
    he = ctx.hints("epi", ["fhi", "lo", "hi2", "lsb", "carry", "rem"])
    fhi, lo, hi2, lsb, carry, rem = (he[n] for n in ("fhi", "lo", "hi2", "lsb", "carry", "rem"))
    ctx.rng("epi.fhi", fhi, 0, 2 ** (p.acc_frac - L))
    ctx.rng("epi.lo", lo, 0, 2**L)
    ctx.rng("epi.hi2", hi2, 0, 2 ** (p.word_bits - 1))
    ctx.boolean("epi.lsb", lsb)
    ctx.boolean("epi.carry", carry)
    ctx.rng("epi.rem", rem, 0, 2**L)
    ctx.eq("epi.split", fhi * 2**L + lo, yf)
    hi = ys * 2 ** (p.word_bits - 1) + yt * 2 ** (p.word_bits - 1 - p.exp) + fhi
    ctx.eq("epi.lsb_split", 2 * hi2 + lsb, hi)
    ctx.eq("epi.round", carry * 2**L + rem, lo + lsb + (2 ** (L - 1) - 1))
    y16 = hi + carry
    ctx.sys.outputs["y16"] = y16
    names = {n: i for i, n in ctx.sys.hints}
    ctx.sys.chain = {"c": [names["acc.s"], names["acc.t"], names["acc.f"]], "y": [ys, yt, yf], "y16": y16}
    return ctx.sys


# ---------------------------------------------------------------------------------------------------------------------
# Hints and public vectors
# ---------------------------------------------------------------------------------------------------------------------

def flatten_hints(p: Params, h: dict, y: int | None = None) -> dict[str, int]:
    """``gen_step``'s hint dict under the row names ``compile_unit`` uses.  ``y`` (the unit's output word) adds the
    chain mode's rows: the accumulator components and the epilogue's hints (``Epilogue.gen`` on ``y``)."""
    out: dict[str, int] = {}
    if y is not None:
        for n in ("s", "t", "f", "nz", "z"):
            out[f"acc.{n}"] = h["acc"][n]
        L = p.acc_frac - p.frac
        ep = Epilogue.gen(p, y)
        for n in ("lo", "hi2", "lsb", "carry", "rem"):
            out[f"epi.{n}"] = ep[n]
        out["epi.fhi"] = (y & (2**p.acc_frac - 1)) >> L
    for i, hp in enumerate(h["prod"]):
        out[f"prod[{i}].z"] = hp["z"]
    for g, hg in enumerate(h["groups"]):
        tag = f"g{g}"
        out[f"{tag}.max.E"] = hg["max"]["E"]
        for j, b in enumerate(hg["max"]["sel"]):
            out[f"{tag}.max.sel[{j}]"] = b
        out[f"{tag}.acc_align.q"], out[f"{tag}.acc_align.r"] = hg["acc_align"]["q"], hg["acc_align"]["r"]
        for i, hq in enumerate(hg["prod_align"]):
            out[f"{tag}.prod_align[{i}].q"], out[f"{tag}.prod_align[{i}].r"] = hq["q"], hq["r"]
        out[f"{tag}.sum.sgn"], out[f"{tag}.sum.z"] = hg["sum"]["sgn"], hg["sum"]["z"]
        out[f"{tag}.lead.l"] = hg["lead"]["l"]
        out[f"{tag}.norm.clamp"], out[f"{tag}.norm.M"], out[f"{tag}.norm.r"] = hg["norm"]["clamp"], hg["norm"]["M"], hg["norm"]["r"]
        out[f"{tag}.out.z"], out[f"{tag}.out.ovf"] = hg["out"]["z"], hg["out"]["ovf"]
    out["pack.hidden"], out["pack.low"] = h["pack"]["hidden"], h["pack"]["low"]
    return out


def HINT_NAMES(sys: System) -> list[str]:
    return [n for _, n in sys.hints]


def PIN_NAMES(sys: System) -> list[str]:
    return [n for _, n in sys.pins]


PUB_NAMES = ("y.s", "y.t", "y.f")


def public_vectors(p: Params, c, a, b, y, device="cpu"):
    """The verifier's per-unit public work, vectorised: ``c`` (B,), ``a``/``b`` (B, k), ``y`` (B,) integer tensors ->
    dict name -> int64 tensor (B,) of pinned values (as signed integers; the caller reduces mod p) and the output
    components.  Raises ``Reject``-equivalent ``ValueError`` when a word is off the finite domain (t = all ones) or
    out of range: the checker's ``x.*`` range checks and the decodes' ``rng(t, 0, t_finite)``.
    ``c=None`` (chain mode): only the operand decodes; the accumulator and the output are not public per unit."""
    import torch

    a = torch.as_tensor(a, dtype=torch.int64, device=device)
    b = torch.as_tensor(b, dtype=torch.int64, device=device)
    k = _k(p)
    if a.ndim != 2 or a.shape[1] != k or b.shape != a.shape:
        raise ValueError("shape mismatch")
    bad = []
    if ((a < 0) | (a >= 2**p.word_bits) | (b < 0) | (b >= 2**p.word_bits)).any():
        bad.append("x.a/x.b range")
    out: dict = {}
    chain = c is None
    if not chain:
        c = torch.as_tensor(c, dtype=torch.int64, device=device)
        y = torch.as_tensor(y, dtype=torch.int64, device=device)
        if c.shape != (a.shape[0],) or y.shape != c.shape:
            raise ValueError("shape mismatch")
        if ((c < 0) | (c >= 2**p.acc_word_bits)).any():
            bad.append("x.c range")
        if ((y < 0) | (y >= 2**p.acc_word_bits)).any():
            bad.append("y range")
        # Fp32Decode of c
        t = (c >> p.acc_frac) & (2**p.exp - 1)
        f = c & (2**p.acc_frac - 1)
        if (t == p.t_finite).any():
            bad.append("acc.t off the finite range")
        nz = (t != 0).to(torch.int64)
        zz = ((t == 0) & (f == 0)).to(torch.int64)
        M = f + nz * 2**p.acc_frac
        e = t - nz - (p.bias - 1)
        e_hat = e + zz * (p.floor - e)
        out["c.s"] = c >> (p.acc_word_bits - 1)
        out["c.e_hat"] = e_hat
        out["c.M"] = M
    for name, w in (("a", a), ("b", b)):
        t = (w >> p.frac) & (2**p.exp - 1)
        f = w & (2**p.frac - 1)
        if (t == p.t_finite).any():
            bad.append(f"{name}.t off the finite range")
        nz = (t != 0).to(torch.int64)
        m = f + nz * 2**p.frac
        e = t - nz - (p.bias - 1)
        s = w >> (p.word_bits - 1)
        for i in range(k):
            out[f"{name}[{i}].s"] = s[:, i]
            out[f"{name}[{i}].m"] = m[:, i]
            out[f"{name}[{i}].e"] = e[:, i]
    if not chain:
        out["y.s"] = y >> (p.acc_word_bits - 1)
        out["y.t"] = (y >> p.acc_frac) & (2**p.exp - 1)
        out["y.f"] = y & (2**p.acc_frac - 1)
    if bad:
        raise ValueError("public inputs rejected: " + "; ".join(bad))
    return out


def reference_decode(p: Params, c: int, a: list[int], b: list[int]) -> dict[str, int]:
    """The same pinned values by the gadgets themselves on integers (a cross-check of ``public_vectors``)."""
    ctx = IntCtx()
    out = {}
    s, e_hat, M = Fp32Decode.check(ctx, p, c, Fp32Decode.gen(p, c))
    out["c.s"], out["c.e_hat"], out["c.M"] = s, e_hat, M
    for name, ws in (("a", a), ("b", b)):
        for i, w in enumerate(ws):
            s, m, e = Bf16Decode.check(ctx, p, w, Bf16Decode.gen(p, w))
            out[f"{name}[{i}].s"], out[f"{name}[{i}].m"], out[f"{name}[{i}].e"] = s, m, e
    return out
