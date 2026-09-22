"""z3 soundness of the v2 gadgets at the real widths (``checker.search`` is the harness; this module supplies the
v2 specs, the two-witness group step, the modular queries and the table self-check).

    SPECS_V2                   one Spec per v2 gadget: symbolic inputs in the contract, symbolic hints, outputs
                               required to differ from the mathematical reference (SAT = finding)
    z3_group_step_v2(p)        one full group step, two hint copies, two output states required to differ
    modular_queries_v2(p)      the same over F_m31 / F_goldilocks (which bounds the field embedding needs)
    table_selfcheck(p)         the symbolic table functions (``Table.sym``) agree with the integer rows: every row
                               of the small tables, a sample of the large ones (the z3 proofs are about ``sym``)

``python -m verity_numerical.checker.v2.search --json PATH`` runs everything and prints the table.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict

from ..ctx import Table, Z3Ctx
from ..gadgets import Epilogue
from ..params import REAL, Params
from ..search import (
    GOLDILOCKS,
    M31,
    ModZ3Ctx,
    QueryResult,
    Spec,
    _cast_admissible,
    _cast_ref,
    _cast_zref,
    _flatten,
    _model_dict,
    render,
    z3_gadget,
)
from .gadgets import (
    GroupMax,
    Normalise,
    OpDecode,
    Pack,
    Product,
    ProductShift,
    SignedSum,
    StateDecode,
    StateShift,
)
from .params import V2
from .tables import all_tables, leadnorm_table

__all__ = ["SPECS_V2", "main", "modular_queries_v2", "table_selfcheck", "z3_group_step_v2"]


def _nz(x):
    """[x != 0] on an int or a z3 expression (a spec-side projection, not a checker predicate)."""
    if isinstance(x, int):
        return int(x != 0)
    import z3
    return z3.If(x != 0, 1, 0)


def _is_int(x) -> bool:
    return isinstance(x, int)


# ---------------------------------------------------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------------------------------------------------

def _op_ref(p, x):
    w = x["w"]
    t = (w >> p.frac) & (2**p.exp - 1)
    m = (w & (2**p.frac - 1)) + (2**p.frac if t else 0)
    return (w >> (p.word_bits - 1), m, V2.e_zero(p) if m == 0 else max(t, 1) - p.bias)


def _op_zref(z3, p, x):
    w = x["w"]
    t = (w / 2**p.frac) % 2**p.exp
    m = w % 2**p.frac + z3.If(t == 0, 0, 2**p.frac)
    return (w / 2 ** (p.word_bits - 1), m, z3.If(m == 0, V2.e_zero(p), z3.If(t == 0, 1, t) - p.bias))


def _state_ref(p, x):
    c = x["c"]
    t = (c >> p.acc_frac) & (2**p.exp - 1)
    M = (c & (2**p.acc_frac - 1)) + (2**p.acc_frac if t else 0)
    return (c >> (p.acc_word_bits - 1), p.floor if M == 0 else max(t, 1) - p.bias, M, int(M == 0))


def _state_zref(z3, p, x):
    c = x["c"]
    t = (c / 2**p.acc_frac) % 2**p.exp
    M = c % 2**p.acc_frac + z3.If(t == 0, 0, 2**p.acc_frac)
    return (c / 2 ** (p.acc_word_bits - 1), z3.If(M == 0, p.floor, z3.If(t == 0, 1, t) - p.bias), M, z3.If(M == 0, 1, 0))


def _prod_call(ctx, p, x, h):
    return Product.check(ctx, p, (x["s_a"], x["m_a"], x["e_a"]), (x["s_b"], x["m_b"], x["e_b"]), h)


def _prod_ref(p, x):
    return (x["m_a"] * x["m_b"], x["e_a"] + x["e_b"], x["s_a"] + 2 * x["s_b"])


def _m(p):
    return p.n + 1


def _cands(p, x):
    return [x[f"cand[{i}]"] for i in range(_m(p))]


def _s4s(p, x):
    return [x[f"s4[{i}]"] for i in range(_m(p))]


def _max_call(ctx, p, x, h):
    """GroupMax with ``isfloor`` taken from a LEADNORM lookup on (l, E) as the composition does (any l in range)."""
    E = h["E"]
    rows = [GroupMax.align(ctx, p, E, c, s, f"max[{i}]") for i, (c, s) in enumerate(zip(_cands(p, x), _s4s(p, x)))]
    *_rest, isfloor = ctx.lookup("norm.leadnorm", leadnorm_table(p), V2.leadnorm_key(p, x["l"], E))
    GroupMax.attained(ctx, p, [r[2] for r in rows], isfloor, "max")
    return (E, *[r[0] for r in rows], *[r[1] for r in rows])


def _max_ref(p, x):
    E = max(p.floor, *_cands(p, x))
    DT = V2.dt(p)
    return (E, *[min(E - c, DT) for c in _cands(p, x)], *[int(s in (1, 2)) for s in _s4s(p, x)])


def _max_zref(z3, p, x):
    E = z3.IntVal(p.floor)
    for c in _cands(p, x):
        E = z3.If(c > E, c, E)
    DT = V2.dt(p)
    return (E, *[z3.If(E - c < DT, E - c, DT) for c in _cands(p, x)],
            *[z3.If(z3.Or(s == 1, s == 2), 1, 0) for s in _s4s(p, x)])


def _max_contract(p):
    c = GroupMax.contract(p)
    inputs = {k: v for k, v in c.inputs.items() if k != "isfloor"}
    inputs["l"] = (0, p.l_hi + 1)
    return type(c)(c.gadget, inputs=inputs, hints=c.hints, checked=c.checked, outputs=c.outputs,
                   tables=c.tables + ("LEADNORM",))


def _signed_shift(x: int, scale: int, d: int, s) -> int:
    v = (x << scale) >> d
    return -v if s else v


def _zshift(z3, x, scale: int, d, s, dt_rows: int):
    out = z3.IntVal(0)
    for k in range(dt_rows - 1, -1, -1):
        out = z3.If(d == k, x * 2 ** (scale - k) if k <= scale else x / 2 ** (k - scale), out)
    return (z3.If(s == 1, -out, out),)


def _pshift_ref(p, x):
    return (_signed_shift(x["u"], p.prod_shift, x["dt"], x["sp"]),)


def _pshift_zref(z3, p, x):
    return _zshift(z3, x["u"], p.prod_shift, x["dt"], x["sp"], V2.dt_rows(p))


def _sshift_ref(p, x):
    return (_signed_shift(x["M"], p.acc_shift, x["dt"], x["s"]),)


def _sshift_zref(z3, p, x):
    return _zshift(z3, x["M"], p.acc_shift, x["dt"], x["s"], V2.dt_rows(p))


def _terms(p, x):
    return [x[f"t[{i}]"] for i in range(_m(p))]


def _sum_call(ctx, p, x, h):
    sgn, v_hi, v_lo, l = SignedSum.check(ctx, p, _terms(p, x), h)
    return (sgn * _nz(v_hi + v_lo), v_hi, v_lo, l)      # the sign is only meaningful when the sum is nonzero


def _sum_ref(p, x):
    S = sum(_terms(p, x))
    v, vb = abs(S), V2.vb(p)
    return (int(S < 0), v >> vb, v & (2**vb - 1), v.bit_length())


def _zbitlen(z3, v, hi: int):
    out = z3.IntVal(hi)
    for l in range(hi - 1, -1, -1):
        out = z3.If(v < 2**l, l, out)
    return out


def _sum_zref(z3, p, x):
    S = sum(_terms(p, x))
    v, vb = z3.If(S < 0, -S, S), V2.vb(p)
    return (z3.If(S < 0, 1, 0), v / 2**vb, v % 2**vb, _zbitlen(z3, v, p.l_hi))


def _norm_v(p, x):
    return x["v_hi"] * 2 ** V2.vb(p) + x["v_lo"]


def _norm_ref(p, x):
    E, l, v = x["E"], x["l"], _norm_v(p, x)
    e1 = max(E + l - p.width, p.e_min)
    k = E - e1 + p.acc_sig - p.width
    M = (v << k) if k >= 0 else (v >> -k)
    z = int(M == 0)
    return (p.floor if z else e1, M, z, int(not z and e1 > p.e_max), int(E == p.floor))


def _norm_zref(z3, p, x):
    E, l, v = x["E"], x["l"], _norm_v(p, x)
    e_raw = E + l - p.width
    e1 = z3.If(e_raw < p.e_min, p.e_min, e_raw)
    k = E - e1 + p.acc_sig - p.width
    M = z3.IntVal(-1)
    for kk in range(p.k_lo, p.k_hi + 1):
        M = z3.If(k == kk, v * 2**kk if kk >= 0 else v / 2 ** (-kk), M)
    z = M == 0
    return (z3.If(z, p.floor, e1), M, z3.If(z, 1, 0), z3.If(z3.And(z3.Not(z), e1 > p.e_max), 1, 0),
            z3.If(E == p.floor, 1, 0))


def _norm_inputs(p, x):
    return {**x, "l": _norm_v(p, x).bit_length()}


def _pack_call(ctx, p, x, h):
    G = len(p.groups)
    return (Pack.check(ctx, p, (x["s"], x["e_hat"], x["M"], x["z"]), [(x[f"ovf[{g}]"], x[f"sgn[{g}]"]) for g in range(G)], h),)


def _pack_ref(p, x):
    G = len(p.groups)
    for g in range(G):
        if x[f"ovf[{g}]"]:
            return ((x[f"sgn[{g}]"] << (p.acc_word_bits - 1)) | (p.t_finite << p.acc_frac),)
    if x["z"]:
        return (0,)
    hidden = x["M"] >> p.acc_frac
    fld = x["e_hat"] + p.bias - 1 + hidden
    return ((x["s"] << (p.acc_word_bits - 1)) + fld * 2**p.acc_frac + (x["M"] & (2**p.acc_frac - 1)),)


def _pack_zref(z3, p, x):
    G = len(p.groups)
    hidden = x["M"] / 2**p.acc_frac
    fld = x["e_hat"] + p.bias - 1 + hidden
    fin = x["s"] * 2 ** (p.acc_word_bits - 1) + fld * 2**p.acc_frac + x["M"] % 2**p.acc_frac
    y = z3.If(x["z"] == 1, 0, fin)
    for g in range(G - 1, -1, -1):
        y = z3.If(x[f"ovf[{g}]"] == 1, x[f"sgn[{g}]"] * 2 ** (p.acc_word_bits - 1) + p.t_finite * 2**p.acc_frac, y)
    return (y,)


def _pack_admissible(p, x):
    """The state invariants a group output satisfies: z = [M = 0]; a zero state has s = 0 and e_hat = floor; a
    nonzero state without hidden bit sits at e_min; a nonzero state has e_hat >= e_min."""
    if x["z"] != int(x["M"] == 0):
        return False
    if x["z"]:
        return x["s"] == 0 and x["e_hat"] == p.floor
    return x["e_hat"] >= p.e_min and (x["M"] >> p.acc_frac == 1 or x["e_hat"] == p.e_min)


def _pack_zadmissible(z3, p, x):
    return z3.And(x["z"] == z3.If(x["M"] == 0, 1, 0),
                  z3.Implies(x["z"] == 1, z3.And(x["s"] == 0, x["e_hat"] == p.floor)),
                  z3.Implies(x["z"] == 0, z3.And(x["e_hat"] >= p.e_min,
                                                 z3.Or(x["M"] >= 2**p.acc_frac, x["e_hat"] == p.e_min))))


SPECS_V2: dict[str, Spec] = {
    "op_decode": Spec("op_decode", OpDecode.contract, lambda ctx, p, x, h: OpDecode.check(ctx, p, x["w"], h),
                      lambda p, x: {}, _op_ref, _op_zref,
                      admissible=lambda p, x: (x["w"] >> p.frac) & (2**p.exp - 1) != p.t_finite),
    "state_decode": Spec("state_decode", StateDecode.contract, lambda ctx, p, x, h: StateDecode.check(ctx, p, x["c"], h),
                         lambda p, x: StateDecode.gen(p, x["c"]), _state_ref, _state_zref,
                         admissible=lambda p, x: (x["c"] >> p.acc_frac) & (2**p.exp - 1) != p.t_finite),
    "product": Spec("product", Product.contract, _prod_call, lambda p, x: {}, _prod_ref, lambda z3, p, x: _prod_ref(p, x)),
    "group_max": Spec("group_max", _max_contract, _max_call, lambda p, x: GroupMax.gen(p, _cands(p, x)), _max_ref, _max_zref),
    "product_shift": Spec("product_shift", ProductShift.contract,
                          lambda ctx, p, x, h: (ProductShift.check(ctx, p, x["u"], x["dt"], x["sp"], h),),
                          lambda p, x: {}, _pshift_ref, _pshift_zref),
    "state_shift": Spec("state_shift", StateShift.contract,
                        lambda ctx, p, x, h: (StateShift.check(ctx, p, x["M"], x["dt"], x["s"], h),),
                        lambda p, x: StateShift.gen(p, x["M"]), _sshift_ref, _sshift_zref),
    "signed_sum": Spec("signed_sum", SignedSum.contract, _sum_call, lambda p, x: SignedSum.gen(p, _terms(p, x)),
                       _sum_ref, _sum_zref),
    "normalise": Spec("normalise", Normalise.contract,
                      lambda ctx, p, x, h: Normalise.check(ctx, p, x["E"], x["l"], x["v_hi"], x["v_lo"], h),
                      lambda p, x: {}, _norm_ref, _norm_zref,
                      admissible=lambda p, x: x["l"] == _norm_v(p, x).bit_length(), inputs_of=_norm_inputs, derived=("l",),
                      zadmissible=lambda z3, p, x: x["l"] == _zbitlen(z3, _norm_v(p, x), p.l_hi)),
    "pack": Spec("pack", Pack.contract, _pack_call, lambda p, x: {}, _pack_ref, _pack_zref,
                 admissible=_pack_admissible, zadmissible=_pack_zadmissible),
    "epilogue": Spec("epilogue", Epilogue.contract, lambda ctx, p, x, h: (Epilogue.check(ctx, p, x["y32"], h),),
                     lambda p, x: Epilogue.gen(p, x["y32"]), _cast_ref, _cast_zref, admissible=_cast_admissible,
                     zadmissible=lambda z3, p, x: _cast_zref(z3, p, x)[0] < 2**p.word_bits),
}


# ---------------------------------------------------------------------------------------------------------------------
# Field embedding: symbolic tables over F_p
# ---------------------------------------------------------------------------------------------------------------------

class ModZ3CtxV2(ModZ3Ctx):
    """``ModZ3Ctx`` whose lookups use the symbolic table function: the key's residue must equal some row key
    ``kk`` in ``[lo, hi)`` (in the domain), and the outputs are ``sym(kk)`` -- as a lookup argument binds them."""

    def lookup(self, name, table: Table, key):
        z3 = self.z3
        if isinstance(key, int) or table.sym is None:
            return super().lookup(name, table, key)
        kk = self.fresh(f"{name}.key")
        in_domain, vals = table.sym(z3, kk)
        outs = tuple(self.fresh(f"{name}.{o}") for o in table.outs)
        self._add(z3.And(kk >= table.lo, kk < table.hi, in_domain, self._cong(key, kk),
                         *[o == v for o, v in zip(outs, vals)]))
        return outs


def _z3_gadget_mod(spec: Spec, p: Params, modulus: int, *, timeout_ms: int, label: str) -> QueryResult:
    import verity_numerical.checker.search as S
    saved = S.ModZ3Ctx
    S.ModZ3Ctx = ModZ3CtxV2
    try:
        return z3_gadget(spec, p, timeout_ms=timeout_ms, modulus=modulus, label=label)
    finally:
        S.ModZ3Ctx = saved


def modular_queries_v2(p: Params = REAL, *, timeout_ms: int = 20_000) -> list[QueryResult]:
    """The gadgets that multiply or split witnesses, over a 31-bit field and over Goldilocks.  Expected: everything
    UNSAT over Goldilocks; over M31 ``normalise`` (v_hi A up to 2^54) and ``signed_sum`` ((1 - 2 sgn) S with |S| <
    2^30 is fine, the split v_hi 2^15 + v_lo is fine) are the ones to watch."""
    out = []
    for name in ("state_decode", "group_max", "state_shift", "signed_sum", "normalise", "pack", "epilogue"):
        for modulus, lab in ((M31, "F_m31"), (GOLDILOCKS, "F_goldilocks")):
            out.append(_z3_gadget_mod(SPECS_V2[name], p, modulus, timeout_ms=timeout_ms, label=f"{lab}, as stated"))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# The full group step
# ---------------------------------------------------------------------------------------------------------------------

def z3_group_step_v2(p: Params = REAL, *, timeout_ms: int = 60_000, concrete: dict | None = None) -> QueryResult:
    """One full v2 group step: a state and n products (symbolic unless ``concrete``), two independent hint copies,
    the two output states (s, e_hat, M, z, ovf_eff, sgn (1 - z)) required to differ."""
    ctx = Z3Ctx()
    z3 = ctx.z3
    n = p.n
    last = V2.last(p)
    ez = V2.e_zero(p)

    def sym(name, lo, hi):
        v = ctx.fresh(name)
        ctx.solver.add(v >= lo, v < hi)
        return v

    s_c, e_c, M_c = sym("s_c", 0, 2), sym("e_c", p.floor, last + 1), sym("M_c", 0, 2**p.acc_sig)
    z_c = z3.If(M_c == 0, 1, 0)
    prods = [(sym(f"u{i}", 0, 2 ** (2 * p.sig)), sym(f"ep{i}", 2 * ez, p.ep_hi + 1), sym(f"s4_{i}", 0, 4)) for i in range(n)]
    x = {"s_c": s_c, "e_c": e_c, "M_c": M_c, **{f"u{i}": u for i, (u, _, _) in enumerate(prods)},
         **{f"ep{i}": e for i, (_, e, _) in enumerate(prods)}, **{f"s4_{i}": s for i, (_, _, s) in enumerate(prods)}}
    if concrete:
        for k, v in concrete.items():
            ctx.solver.add(x[k] == v)

    from .step import _group

    def copy_(tag):
        h = {"E": ctx.fresh(f"{tag}.E"), "acc_split": {"M_hi": ctx.fresh(f"{tag}.M_hi"), "M_lo": ctx.fresh(f"{tag}.M_lo")},
             "sum": {"sgn": ctx.fresh(f"{tag}.sgn"), "v_hi": ctx.fresh(f"{tag}.v_hi"), "v_lo": ctx.fresh(f"{tag}.v_lo")}}
        (s_o, e_o, M_o, z_o), (ovf, sgn) = _group(ctx, p, (s_c, e_c, M_c, z_c), prods, h, tag)
        return (s_o, e_o, M_o, z_o, ovf, sgn - sgn * z_o), h

    out1, h1 = copy_("h1")
    out2, h2 = copy_("h2")
    ctx.solver.add(z3.Or([a != b for a, b in zip(out1, out2)]))
    ctx.solver.set("timeout", timeout_ms)
    t0 = time.perf_counter()
    status = str(ctx.solver.check())
    seconds = time.perf_counter() - t0
    model = None
    if status == "sat":
        model = {"inputs": _model_dict(ctx, x),
                 "h1": _model_dict(ctx, dict(_flatten(h1))), "h2": _model_dict(ctx, dict(_flatten(h2))),
                 "out1": _model_dict(ctx, {f"o{i}": o for i, o in enumerate(out1)}),
                 "out2": _model_dict(ctx, {f"o{i}": o for i, o in enumerate(out2)})}
    return QueryResult("group_step_v2", "two-witnesses-two-outputs" + (" (concrete x)" if concrete else " (symbolic x)"),
                       status, round(seconds, 3), model)


def concrete_group_inputs_v2() -> dict:
    x = {"s_c": 0, "e_c": 3, "M_c": (1 << 23) | 0x123456}
    for i in range(8):
        x[f"u{i}"] = (128 + 3 * i) * (128 + 5 * (i + 1))
        x[f"s4_{i}"] = [0, 1, 2, 3, 1, 0, 2, 3][i]
        x[f"ep{i}"] = [0, 3, -5, 3, 7, -20, 3, 1][i]
    return x


# ---------------------------------------------------------------------------------------------------------------------
# The tables' symbolic functions agree with their rows
# ---------------------------------------------------------------------------------------------------------------------

def table_selfcheck(p: Params = REAL, *, full_below: int = 20_000, sample: int = 3_000, seed: int = 0) -> list[dict]:
    """For every v2 table: ``sym`` evaluated (by z3 simplification) on concrete keys equals ``fn``, and the symbolic
    domain predicate equals the integer one; exhaustive below ``full_below`` rows, else a random sample plus the
    key-space boundaries."""
    import z3
    rng = random.Random(seed)
    out = []
    for t in all_tables(p):
        t0 = time.perf_counter()
        span = t.hi - t.lo
        if span <= full_below:
            keys = range(t.lo, t.hi)
        else:
            keys = {t.lo, t.hi - 1, *rng.sample(range(t.lo, t.hi), sample)}
        bad = 0
        n = 0
        for k in keys:
            n += 1
            in_dom, vals = t.sym(z3, z3.IntVal(k))
            in_dom_v = z3.simplify(in_dom) if not isinstance(in_dom, bool) else in_dom
            in_dom_b = bool(in_dom_v) if isinstance(in_dom_v, bool) else z3.is_true(in_dom_v)
            dom_int = t.domain is None or t.domain(k)
            if in_dom_b != dom_int:
                bad += 1
                continue
            if dom_int:
                got = tuple(z3.simplify(v).as_long() for v in vals)
                if got != t.fn(k):
                    bad += 1
        out.append({"table": t.name, "rows": t.rows, "checked": n, "mismatches": bad,
                    "seconds": round(time.perf_counter() - t0, 2)})
    return out


# ---------------------------------------------------------------------------------------------------------------------

def integer_queries_v2(p: Params = REAL, *, timeout_ms: int = 20_000) -> list[QueryResult]:
    out = [z3_gadget(spec, p, timeout_ms=timeout_ms) for spec in SPECS_V2.values()]
    # each dropped predicate must turn the UNSAT into a SAT (the bound is load-bearing)
    out.append(z3_gadget(SPECS_V2["group_max"], p, timeout_ms=timeout_ms, drop=("attained",), label="excessive E: without `attained`"))
    out.append(z3_gadget(SPECS_V2["state_shift"], p, timeout_ms=timeout_ms, drop=("split",), label="halves: without `split`"))
    out.append(z3_gadget(SPECS_V2["signed_sum"], p, timeout_ms=timeout_ms, drop=("split",), label="halves: without `split`"))
    out.append(z3_gadget(SPECS_V2["state_decode"], p, timeout_ms=timeout_ms, drop=("z1",), label="zero flag: without `z1`"))
    out.append(z3_gadget(SPECS_V2["state_decode"], p, timeout_ms=timeout_ms, drop=("f",), label="fraction: without `f`"))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--step-timeout", type=float, default=600.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip-modular", action="store_true")
    ap.add_argument("--skip-step", action="store_true")
    ap.add_argument("--skip-selfcheck", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated gadget names")
    args = ap.parse_args(argv)
    report: dict = {"params": repr(REAL), "tables": [], "z3": []}
    if not args.skip_selfcheck:
        report["tables"] = table_selfcheck()
        for r in report["tables"]:
            print(f"table {r['table']}: rows={r['rows']} checked={r['checked']} mismatches={r['mismatches']} {r['seconds']}s", flush=True)
    T = int(args.timeout * 1000)
    if args.only:
        rows = [z3_gadget(SPECS_V2[n], timeout_ms=T) for n in args.only.split(",")]
    else:
        rows = integer_queries_v2(timeout_ms=T)
    for r in rows:
        print(f"{r.gadget:14s} {r.query:45s} {r.status:8s} {r.seconds}s {r.note}", flush=True)
    if not args.skip_modular and not args.only:
        more = modular_queries_v2(timeout_ms=T)
        for r in more:
            print(f"{r.gadget:14s} {r.query:45s} {r.status:8s} {r.seconds}s {r.note}", flush=True)
        rows += more
    if not args.skip_step and not args.only:
        for r in (z3_group_step_v2(timeout_ms=int(args.step_timeout * 1000), concrete=concrete_group_inputs_v2()),
                  z3_group_step_v2(timeout_ms=int(args.step_timeout * 1000))):
            print(f"{r.gadget:14s} {r.query:45s} {r.status:8s} {r.seconds}s", flush=True)
            rows.append(r)
    report["z3"] = [asdict(r) for r in rows]
    print(render(rows), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1, default=str)
        print(f"wrote {args.json}")
    bad = [r for r in rows if r.status == "sat" and not (r.note.startswith("mod") or "without" in r.note)]
    return 1 if bad else 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
