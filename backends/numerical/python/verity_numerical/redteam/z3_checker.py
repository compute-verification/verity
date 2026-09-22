"""z3 against the WP2 checker contracts: the two UNKNOWNs of the checker's own search re-encoded, the U4 range-
containment obligation as a mechanical check, and the gadgets over Goldilocks / Mersenne-31 with case splits.

Re-encodings (the predicates are the gadgets' own ``check`` code run on a different context; nothing is restated):

    split          the Boolean hints (``sgn``, ``z``, ``clamp``, ``ovf``, ``nz`` ...) fixed per case: 2^k linear
                   sub-problems instead of one nonlinear one (``signed_sum`` over F_(2^31-1): U3)
    bitvector      every value a 128-bit vector; the four tables become shift expressions (``pow = 1 << min(d, W)``
                   ...), which ``bv_tables_agree`` checks row by row against ``tables.py``; the symbolic full group
                   step (U2) then has no 512-row disjunctions
    containment    U4: every edge of ``step.py``'s data flow, output range of the producer vs. input range of the
                   consumer, at real widths, with the per-group exponent ceiling ``e_hi(g)`` the contracts assume
"""
from __future__ import annotations

import itertools
import time
from typing import Any

import z3

from verity_numerical.checker import REAL, Params, Reject
from verity_numerical.checker.ctx import Table, Z3Ctx
from verity_numerical.checker.gadgets import (
    AlignProduct,
    AlignState,
    Bf16Decode,
    Epilogue,
    Fp32Decode,
    GroupMax,
    LeadingBit,
    Normalise,
    Pack,
    Product,
    SignedSum,
    StateOut,
)
from verity_numerical.checker.search import (
    GOLDILOCKS,
    SPECS,
    ModZ3Ctx,
    Spec,
    _model_dict,
    _symbolic_hints,
)
from verity_numerical.checker.tables import (
    align_table,
    lead_table,
    norm_table,
    pow_table,
)

__all__ = ["BVCtx", "bv_group_step", "bv_tables_agree", "containment", "gadget_split", "gadgets_over_field"]

BV = 128


# ---------------------------------------------------------------------------------------------------------------------
# Case-split gadget queries (U3)
# ---------------------------------------------------------------------------------------------------------------------

def _boolean_hints(contract) -> list[str]:
    return [n for n, (lo, hi) in contract.hints.items() if (lo, hi) == (0, 2)]


def gadget_split(spec: Spec, p: Params = REAL, *, modulus: int | None = None, timeout_ms: int = 20_000,
                 ideal_bounds: bool = False, drop: tuple[str, ...] = ()) -> dict[str, Any]:
    """``search.z3_gadget`` with every Boolean hint fixed per case.  A field hint that must be a bit is a bit
    (``boolean`` over F_p with a canonical representative), so the split loses nothing.  Returns the worst status
    over the cases (``sat`` if any case is SAT, else ``unknown`` if any, else ``unsat``) with the per-case table."""
    contract = spec.contract(p)
    bools = _boolean_hints(contract)
    rows, t_all = [], time.perf_counter()
    worst, model = "unsat", None
    for bits in itertools.product((0, 1), repeat=len(bools)):
        ctx = ModZ3Ctx(modulus, ideal_bounds=ideal_bounds) if modulus else Z3Ctx()
        if drop:
            _drop(ctx, set(drop))
        x = {}
        for name, (lo, hi) in contract.inputs.items():
            v = ctx.fresh(f"in.{name}")
            ctx.solver.add(v >= lo, v < hi)
            x[name] = v
        h = _symbolic_hints(ctx, contract, modulus)
        for name, b in zip(bools, bits):
            ctx.solver.add(h[name] == b)
        try:
            outs = spec.call(ctx, p, x, h)
        except Reject:
            rows.append({"case": dict(zip(bools, bits)), "status": "rejected-statically"})
            continue
        refs = spec.zref(z3, p, x)
        if spec.zadmissible is not None:
            ctx.solver.add(spec.zadmissible(z3, p, x))
        if modulus:
            ctx.solver.add(z3.Or([(o - r) % modulus != 0 for o, r in zip(outs, refs)]))
        else:
            ctx.solver.add(z3.Or([o != r for o, r in zip(outs, refs)]))
        ctx.solver.set("timeout", timeout_ms)
        t0 = time.perf_counter()
        st = str(ctx.solver.check())
        dt = round(time.perf_counter() - t0, 3)
        row = {"case": dict(zip(bools, bits)), "status": st, "seconds": dt}
        if st == "sat":
            row["model"] = {"inputs": _model_dict(ctx, x), "hints": _model_dict(ctx, h),
                            "outputs": _model_dict(ctx, {f"out[{i}]": o for i, o in enumerate(outs)}),
                            "reference": _model_dict(ctx, {f"ref[{i}]": r for i, r in enumerate(refs)})}
            worst, model = "sat", row["model"]
        elif st == "unknown" and worst != "sat":
            worst = "unknown"
        rows.append(row)
    return {"gadget": spec.name, "modulus": modulus, "ideal_bounds": ideal_bounds, "drop": list(drop),
            "status": worst, "cases": rows, "seconds": round(time.perf_counter() - t_all, 3), "model": model}


def _drop(ctx, drop: set[str]) -> None:
    orig = {"rng": ctx.rng, "eq": ctx.eq, "lt_pow": ctx.lt_pow}
    ctx.rng = lambda name, x, lo, hi: None if name.split(".")[-1] in drop else orig["rng"](name, x, lo, hi)
    ctx.eq = lambda name, a, b: None if name.split(".")[-1] in drop else orig["eq"](name, a, b)
    ctx.lt_pow = lambda name, x, ipow, bits: None if name.split(".")[-1] in drop else orig["lt_pow"](name, x, ipow, bits)


def gadgets_over_field(modulus: int = GOLDILOCKS, p: Params = REAL, *, timeout_ms: int = 20_000,
                       names: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """Every gadget as stated (product ``lt_pow``), read over F_p, Boolean hints split."""
    return [gadget_split(SPECS[n], p, modulus=modulus, timeout_ms=timeout_ms) for n in (names or SPECS)]


# ---------------------------------------------------------------------------------------------------------------------
# Bit-vector encoding (U2)
# ---------------------------------------------------------------------------------------------------------------------

class BVCtx(Z3Ctx):
    """The five predicates on 128-bit signed bit-vectors.  Tables are shift expressions of the key (checked against
    ``tables.py`` by :func:`bv_tables_agree`); the key's membership in the table's domain is asserted."""

    def __init__(self, p: Params) -> None:
        super().__init__()
        self.p = p

    def fresh(self, name: str) -> object:
        self.count += 1
        return z3.BitVec(f"{name}#{self.count}", BV)

    @staticmethod
    def _bv(v) -> object:
        return z3.BitVecVal(v, BV) if isinstance(v, int) else v

    def rng(self, name, x, lo, hi) -> None:
        if isinstance(x, int):
            self._add(lo <= x < hi)
            return
        self._add(z3.And(x >= self._bv(lo), x < self._bv(hi)))

    def eq(self, name, a, b) -> None:
        self._add(self._bv(a) == self._bv(b))

    def boolean(self, name, b) -> None:
        if isinstance(b, int):
            self._add(b in (0, 1))
            return
        self._add(z3.Or(b == self._bv(0), b == self._bv(1)))

    def lt_pow(self, name, x, ipow, bits) -> None:
        self.rng(name, self._bv(x) * self._bv(ipow), 0, 2**bits)

    def lookup(self, name, table: Table, key):
        if isinstance(key, int):
            return super().lookup(name, table, key)
        p = self.p
        one = self._bv(1)
        self._add(z3.And(key >= self._bv(table.lo), key < self._bv(table.hi)))
        if table.name == "POW":
            W = p.acc_bits
            dd = z3.If(key >= self._bv(W), self._bv(W), key)
            return one << dd, one << (self._bv(W) - dd)
        if table.name == "ALIGN":
            U, ps, pb = 2 * p.sig, p.prod_shift, p.prod_bits
            low, mid = key <= self._bv(ps), key < self._bv(pb)
            sl = z3.If(low, one << (self._bv(ps) - key), one)
            sr = z3.If(low, one, z3.If(mid, one << (key - self._bv(ps)), self._bv(2**U)))
            isr = z3.If(low, self._bv(2**U), z3.If(mid, one << (self._bv(pb) - key), one))
            return sl, sr, isr
        if table.name == "LEAD":
            B = p.s_bits
            z0 = key == self._bv(0)
            return z3.If(z0, self._bv(0), one << (key - one)), z3.If(z0, self._bv(2**B), one << (self._bv(B + 1) - key))
        if table.name == "NORM":
            R = p.norm_rem_bits
            neg = z3.If(key < self._bv(0), -key, self._bv(0))
            pos = z3.If(key > self._bv(0), key, self._bv(0))
            return one << pos, one << neg, one << (self._bv(R) - neg)
        raise KeyError(table.name)


def bv_tables_agree(p: Params = REAL) -> dict[str, bool]:
    """Row by row: the shift encoding equals ``tables.py``."""
    out = {}
    for t in (pow_table(p), align_table(p), lead_table(p), norm_table(p)):
        ok = True
        for k in range(t.lo, t.hi):
            ctx = BVCtx(p)
            key = ctx.fresh("k")
            ctx.solver.add(key == k)
            cols = ctx.lookup("t", t, key)
            ctx.solver.add(z3.Or([c != z3.BitVecVal(v, BV) for c, v in zip(cols, t.fn(k))]))
            if ctx.solver.check() != z3.unsat:
                ok = False
                break
        out[t.name] = ok
    return out


def bv_group_step(p: Params = REAL, *, timeout_ms: int = 60_000, concrete: dict | None = None,
                  fix_selectors: tuple[int, int] | None = None) -> dict[str, Any]:
    """``search.z3_group_step`` on :class:`BVCtx`: symbolic state and n products, two independent hint copies,
    outputs required to differ.  ``fix_selectors=(i, j)`` fixes which candidate each copy's one-hot selects (a
    case of the exhaustive split over (n+2)^2 selector pairs)."""
    ctx = BVCtx(p)
    n = p.n
    last = p.e_hi(len(p.groups) - 1)

    def sym(name, lo, hi):
        v = ctx.fresh(name)
        ctx.rng(name, v, lo, hi)
        return v

    s_c, e_c, M_c = sym("s_c", 0, 2), sym("e_c", p.exp_lo, last + 1), sym("M_c", 0, 2**p.acc_sig)
    prods = [(sym(f"u{i}", 0, 2 ** (2 * p.sig)), sym(f"sp{i}", 0, 2), sym(f"ep{i}", p.exp_lo, p.ep_hi + 1)) for i in range(n)]
    x = {"s_c": s_c, "e_c": e_c, "M_c": M_c, **{f"u{i}": u for i, (u, _, _) in enumerate(prods)},
         **{f"sp{i}": s for i, (_, s, _) in enumerate(prods)}, **{f"ep{i}": e for i, (_, _, e) in enumerate(prods)}}
    if concrete:
        for k, v in concrete.items():
            ctx.solver.add(x[k] == z3.BitVecVal(v, BV))

    def copy_(tag, sel_fixed):
        h = {"max": {"E": ctx.fresh(f"{tag}.E"), "sel": [ctx.fresh(f"{tag}.sel{i}") for i in range(n + 2)]},
             "acc_align": {"q": ctx.fresh(f"{tag}.qa"), "r": ctx.fresh(f"{tag}.ra")},
             "prod_align": [{"q": ctx.fresh(f"{tag}.q{i}"), "r": ctx.fresh(f"{tag}.r{i}")} for i in range(n)],
             "sum": {"sgn": ctx.fresh(f"{tag}.sgn"), "z": ctx.fresh(f"{tag}.zs")},
             "lead": {"l": ctx.fresh(f"{tag}.l")},
             "norm": {"clamp": ctx.fresh(f"{tag}.clamp"), "M": ctx.fresh(f"{tag}.M"), "r": ctx.fresh(f"{tag}.rn")},
             "out": {"z": ctx.fresh(f"{tag}.z"), "ovf": ctx.fresh(f"{tag}.ovf")}}
        if sel_fixed is not None:
            for i, b in enumerate(h["max"]["sel"]):
                ctx.solver.add(b == z3.BitVecVal(int(i == sel_fixed), BV))
        cands = [e_c] + [e for _, _, e in prods]
        E, ds = GroupMax.check(ctx, p, cands, h["max"], f"{tag}.max")
        q_acc = AlignState.check(ctx, p, M_c, ds[0], h["acc_align"], f"{tag}.align_state")
        qs = [AlignProduct.check(ctx, p, u, d, hq, f"{tag}.align_prod[{i}]")
              for i, ((u, _, _), d, hq) in enumerate(zip(prods, ds[1:], h["prod_align"]))]
        terms = [(s_c, q_acc)] + [(s, q) for (_, s, _), q in zip(prods, qs)]
        sgn, v = SignedSum.check(ctx, p, terms, h["sum"], f"{tag}.sum")
        l = LeadingBit.check(ctx, p, v, h["lead"], f"{tag}.lead")
        e1, M = Normalise.check(ctx, p, E, l, v, h["norm"], f"{tag}.norm")
        s_out, e_out, M_out, ovf, z = StateOut.check(ctx, p, sgn, e1, M, h["out"], f"{tag}.out")
        return (s_out, e_out, M_out, ovf, z), h

    out1, _h1 = copy_("h1", fix_selectors[0] if fix_selectors else None)
    out2, _h2 = copy_("h2", fix_selectors[1] if fix_selectors else None)
    ctx.solver.add(z3.Or([a != b for a, b in zip(out1, out2)]))
    ctx.solver.set("timeout", timeout_ms)
    t0 = time.perf_counter()
    status = str(ctx.solver.check())
    seconds = round(time.perf_counter() - t0, 3)
    model = None
    if status == "sat":
        m = ctx.solver.model()
        model = {k: m.eval(v, model_completion=True).as_signed_long() for k, v in x.items()}
    return {"query": "bv group step, two witnesses two outputs" + (" (concrete x)" if concrete else " (symbolic x)")
            + (f" selectors {fix_selectors}" if fix_selectors else ""), "status": status, "seconds": seconds, "model": model}


def int_group_step_split(p: Params = REAL, *, timeout_ms: int = 60_000, selectors: tuple[int, int] = (1, 1),
                         concrete: dict | None = None) -> dict[str, Any]:
    """``search.z3_group_step`` (Int encoding) with each copy's one-hot selector fixed: one case of the
    (n+2)^2 split.  UNSAT on every case would be the symbolic step result U2 asks for."""
    ctx = Z3Ctx()
    n = p.n
    last = p.e_hi(len(p.groups) - 1)

    def sym(name, lo, hi):
        v = ctx.fresh(name)
        ctx.solver.add(v >= lo, v < hi)
        return v

    s_c, e_c, M_c = sym("s_c", 0, 2), sym("e_c", p.exp_lo, last + 1), sym("M_c", 0, 2**p.acc_sig)
    prods = [(sym(f"u{i}", 0, 2 ** (2 * p.sig)), sym(f"sp{i}", 0, 2), sym(f"ep{i}", p.exp_lo, p.ep_hi + 1)) for i in range(n)]
    x = {"s_c": s_c, "e_c": e_c, "M_c": M_c, **{f"u{i}": u for i, (u, _, _) in enumerate(prods)},
         **{f"sp{i}": s for i, (_, s, _) in enumerate(prods)}, **{f"ep{i}": e for i, (_, _, e) in enumerate(prods)}}
    if concrete:
        for k, v in concrete.items():
            ctx.solver.add(x[k] == v)

    def copy_(tag, k):
        h = {"max": {"E": ctx.fresh(f"{tag}.E"), "sel": [z3.IntVal(int(i == k)) for i in range(n + 2)]},
             "acc_align": {"q": ctx.fresh(f"{tag}.qa"), "r": ctx.fresh(f"{tag}.ra")},
             "prod_align": [{"q": ctx.fresh(f"{tag}.q{i}"), "r": ctx.fresh(f"{tag}.r{i}")} for i in range(n)],
             "sum": {"sgn": ctx.fresh(f"{tag}.sgn"), "z": ctx.fresh(f"{tag}.zs")},
             "lead": {"l": ctx.fresh(f"{tag}.l")},
             "norm": {"clamp": ctx.fresh(f"{tag}.clamp"), "M": ctx.fresh(f"{tag}.M"), "r": ctx.fresh(f"{tag}.rn")},
             "out": {"z": ctx.fresh(f"{tag}.z"), "ovf": ctx.fresh(f"{tag}.ovf")}}
        cands = [e_c] + [e for _, _, e in prods]
        E, ds = GroupMax.check(ctx, p, cands, h["max"], f"{tag}.max")
        q_acc = AlignState.check(ctx, p, M_c, ds[0], h["acc_align"], f"{tag}.align_state")
        qs = [AlignProduct.check(ctx, p, u, d, hq, f"{tag}.align_prod[{i}]")
              for i, ((u, _, _), d, hq) in enumerate(zip(prods, ds[1:], h["prod_align"]))]
        terms = [(s_c, q_acc)] + [(s, q) for (_, s, _), q in zip(prods, qs)]
        sgn, v = SignedSum.check(ctx, p, terms, h["sum"], f"{tag}.sum")
        l = LeadingBit.check(ctx, p, v, h["lead"], f"{tag}.lead")
        e1, M = Normalise.check(ctx, p, E, l, v, h["norm"], f"{tag}.norm")
        return StateOut.check(ctx, p, sgn, e1, M, h["out"], f"{tag}.out")

    out1 = copy_("h1", selectors[0])
    out2 = copy_("h2", selectors[1])
    ctx.solver.add(z3.Or([a != b for a, b in zip(out1, out2)]))
    ctx.solver.set("timeout", timeout_ms)
    t0 = time.perf_counter()
    status = str(ctx.solver.check())
    seconds = round(time.perf_counter() - t0, 3)
    model = _model_dict(ctx, x) if status == "sat" else None
    return {"query": f"int group step, selectors fixed {selectors}" + (" (concrete x)" if concrete else " (symbolic x)"),
            "status": status, "seconds": seconds, "model": model}


# ---------------------------------------------------------------------------------------------------------------------
# U4: range containment along the real data flow
# ---------------------------------------------------------------------------------------------------------------------

def _sub(inner: tuple[int, int], outer: tuple[int, int]) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def containment(p: Params = REAL) -> dict[str, Any]:
    """Every producer-output -> consumer-input edge of ``step.py`` (and the VU chain), checked as interval
    containment at ``p``.  Group-indexed edges use the contracts' per-group ceilings (``GroupMax`` inputs are
    bounded by ``e_hi(last)``; a ``StateOut`` output of group ``g`` reaches ``e_hi(g) + growth``, which is the
    *next* group's ``e_hi``); the one edge that fails when read contract-to-contract (``StateOut.e_hat`` of the last
    group is above ``GroupMax``'s input ceiling) has no consumer in the data flow."""
    G = len(p.groups)
    edges: list[dict[str, Any]] = []
    fp, bf, pr, mx = Fp32Decode.contract(p), Bf16Decode.contract(p), Product.contract(p), GroupMax.contract(p)
    ast, apr, ssm, lb = AlignState.contract(p), AlignProduct.contract(p), SignedSum.contract(p), LeadingBit.contract(p)
    nm, so, pk, ep = Normalise.contract(p), StateOut.contract(p), Pack.contract(p), Epilogue.contract(p)

    def edge(name, out_rng, in_rng, note=""):
        edges.append({"edge": name, "output": out_rng, "input": in_rng, "ok": _sub(out_rng, in_rng), "note": note})

    edge("x.c -> fp32_decode.c", (0, 2**p.acc_word_bits), fp.inputs["c"])
    edge("x.a -> bf16_decode.w", (0, 2**p.word_bits), bf.inputs["w"])
    edge("bf16_decode.s (hint) -> product.s_a", bf.hints["s"], pr.inputs["s_a"])
    for k in ("m", "e"):
        edge(f"bf16_decode.{k} -> product.{k}_a", bf.outputs[k], pr.inputs[f"{k}_a"])
    edge("fp32_decode.e_hat -> group_max.cand[0] (g=0)", fp.outputs["e_hat"], mx.inputs["cand[0]"])
    edge("product.e_hat -> group_max.cand[i]", pr.outputs["e_hat"], mx.inputs["cand[1]"])
    edge("fp32_decode.M -> align_state.M", fp.outputs["M"], ast.inputs["M"])
    edge("group_max.d[0] -> align_state.d", mx.outputs["d[0]"], ast.inputs["d"])
    edge("group_max.d[i] -> align_product.d", mx.outputs["d[1]"], apr.inputs["d"])
    edge("product.u -> align_product.u", pr.outputs["u"], apr.inputs["u"])
    edge("align_state.q -> signed_sum.q[0]", ast.outputs["q"], ssm.inputs["q[0]"])
    edge("align_product.aligned -> signed_sum.q[i]", apr.outputs["aligned"], ssm.inputs["q[1]"])
    edge("fp32_decode.s -> signed_sum.s[0]", fp.outputs["s"], ssm.inputs["s[0]"])
    edge("product.s_p -> signed_sum.s[i]", pr.outputs["s_p"], ssm.inputs["s[1]"])
    edge("signed_sum.v -> leading_bit.v", ssm.outputs["v"], lb.inputs["v"])
    edge("signed_sum.v -> normalise.v", ssm.outputs["v"], nm.inputs["v"])
    edge("leading_bit.l -> normalise.l", lb.outputs["l"], nm.inputs["l"])
    edge("group_max.E -> normalise.E", mx.outputs["E"], nm.inputs["E"])
    edge("signed_sum.sgn -> state_out.sgn", ssm.outputs["sgn"], so.inputs["sgn"])
    edge("normalise.e1 -> state_out.e1", nm.outputs["e1"], so.inputs["e1"])
    edge("normalise.M -> state_out.M", nm.outputs["M"], so.inputs["M"])
    for g in range(G - 1):
        # the state of group g feeds group g+1: its exponent is at most e_hi(g) + growth = e_hi(g+1) <= e_hi(last)
        e_state_g = (p.floor, p.e_hi(g) + p.growth + 1)
        edge(f"state_out.e_hat (g={g}) -> group_max.cand[0] (g={g + 1})", e_state_g, mx.inputs["cand[0]"],
             note="per-group ceiling e_hi(g)+growth, not the contract's exp_hi")
        edge(f"state_out.e_hat (g={g}) contract range -> group_max.cand[0]", so.outputs["e_hat"], mx.inputs["cand[0]"],
             note="contract-to-contract reading: FAILS by `growth` -- sound only through the per-group argument")
        edge(f"state_out.M (g={g}) -> align_state.M (g={g + 1})", so.outputs["M"], ast.inputs["M"])
        edge(f"state_out.s (g={g}) -> signed_sum.s[0] (g={g + 1})", so.outputs["s"], ssm.inputs["s[0]"])
    for k in ("s", "M", "z"):
        edge(f"state_out.{k} (last) -> pack.{k}", so.outputs[k], pk.inputs[k])
    edge("state_out.e_hat (last) -> pack.e1", so.outputs["e_hat"], pk.inputs["e1"])
    edge("state_out.ovf_eff -> pack.ovf[g]", so.outputs["ovf_eff"], pk.inputs["ovf[0]"])
    edge("signed_sum.sgn -> pack.sgn[g]", ssm.outputs["sgn"], pk.inputs["sgn[0]"])
    edge("pack.y -> fp32_decode.c (next unit)", pk.outputs["y"], fp.inputs["c"])
    edge("pack.y -> epilogue.y32", pk.outputs["y"], ep.inputs["y32"])
    # the derived bounds the contracts rely on, recomputed
    derived = {
        "signed_sum.premise": (2**p.acc_bits - 1) + p.n * (2**p.prod_bits - 1) < 2**p.s_bits,
        "group_max.d_max < shift_rows": p.e_hi(G - 1) - p.exp_lo < p.shift_rows,
        "group_max.d_floor_max < shift_rows": p.e_hi(G - 1) - p.floor < p.shift_rows,
        "normalise.k in NORM": all(p.k_lo <= (E - max(E + l - p.width, p.e_min) + p.acc_sig - p.width) <= p.k_hi
                                   for E in range(p.floor, p.e_hi(G - 1) + 1) for l in range(p.l_hi + 1)),
        "normalise.clamp_proof < cmp_rows": all(abs(E + l - p.width - p.e_min) + 1 <= p.cmp_rows
                                                for E in range(p.floor, p.e_hi(G - 1) + 1) for l in range(p.l_hi + 1)),
        "state_out.ovf_proof < cmp_rows": all(abs(e1 - p.e_max) + 1 <= p.cmp_rows for e1 in range(p.e_min, p.exp_hi + 1)),
        "normalise.M' < 2^acc_sig": all((v << k if k >= 0 else v >> -k) < 2**p.acc_sig
                                        for l in range(p.l_hi + 1) for v in ({0} if l == 0 else {2**(l - 1), 2**l - 1})
                                        for k in {p.acc_sig - l}),
        "pack.fld < t_finite when not ovf": p.e_max + p.bias - 1 + 1 < p.t_finite,
        "epilogue.y16 of an infinity fits": ((p.t_finite << p.frac) + 0) < 2**p.word_bits,
    }
    failing = [e for e in edges if not e["ok"] and "FAILS" not in e["note"]]
    return {"params": repr(p), "edges": edges, "derived": derived,
            "status": "PASS" if not failing and all(derived.values()) else "FAIL",
            "failing_edges": failing, "failing_derived": [k for k, v in derived.items() if not v]}
