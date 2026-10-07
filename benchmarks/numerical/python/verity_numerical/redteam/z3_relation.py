"""The predicates of ``verity.ml.circuits.silicon.relation`` transcribed for z3, over the integers or over a prime field F_p.

Transcription rules (faithful to ``check_step``, nothing added):

* every witness column is an ``Int``; over F_p it is constrained to ``[0, p)`` (a canonical representative);
* ``mul`` / ``lin``: ``a * b = c`` and ``sum = 0`` -- exact over the integers, ``= q * p`` with a fresh ``q`` over F_p;
* ``lookup``: the disjunction over the table's rows (``T_pow``: 512 rows, ``T_norm``: B+1, ``T_exp``: 512, ``T_any``);
  a lookup on a *public* key (the BF16 words, the class tuple) pins its outputs to the row;
* a ``w``-bit range check pins the canonical value to ``[0, 2^w)`` (the 16-bit chunk split is exact when ``2^w < p``);
* the "E* attained" chain is read as the module says it must be: a field identity, ``E* in {Z} u {e_i}``;
* a ``mul`` whose one factor is a table output is attached to that lookup's row disjunction (``aligned * pow`` is
  ``aligned * 2^min(k, A)`` in the row-``k`` disjunct), so the whole finite path is linear integer arithmetic once the
  solver picks the rows -- the reason a group query terminates.  Over F_p the quotient of each identity is bounded
  by ``2^48`` (every side is a product of canonical values and constants below ``2^40``; the bound is slack).

Queries return ``(status, model, seconds)``; ``SAT`` means an accepted assignment with the forbidden property exists.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import z3

from verity.ml.circuits.silicon import relation as R
from verity.ml.circuits.silicon.silicon import Pipeline

from .harness import z3_query

__all__ = ["align_query", "estar_query", "final_query", "group_query", "norm_query", "sign_query", "step_query"]

Q_BOUND = 1 << 48


class _M:
    """Constraint builder: integer or F_p semantics."""

    def __init__(self, s: z3.Solver, field: int | None):
        self.s, self.p, self.n = s, field, 0

    def c(self, v: int) -> int:
        return v % self.p if self.p else v

    def var(self, name: str) -> z3.ArithRef:
        self.n += 1
        v = z3.Int(f"{name}~{self.n}")
        if self.p:
            self.s.add(v >= 0, v < self.p)
        return v

    def _eq(self, lhs: Any, rhs: Any) -> Any:
        if self.p:
            self.n += 1
            q = z3.Int(f"q~{self.n}")
            bound = (self._abs_bound(lhs) + self._abs_bound(rhs)) // self.p + 1
            self.s.add(q >= -bound, q <= bound)
            return lhs - rhs == q * self.p
        return lhs == rhs

    def _abs_bound(self, e: Any) -> int:
        """An upper bound on |e| given every variable is canonical in [0, p) (constants exact; sums/products bounded)."""
        if isinstance(e, int):
            return abs(e)
        if z3.is_int_value(e):
            return abs(e.as_long())
        if z3.is_const(e):
            return self.p - 1
        k = e.decl().kind()
        args = [self._abs_bound(a) for a in e.children()]
        if k in (z3.Z3_OP_ADD, z3.Z3_OP_SUB):
            return sum(args)
        if k == z3.Z3_OP_MUL:
            b = 1
            for a in args:
                b *= a
            return b
        if k == z3.Z3_OP_UMINUS:
            return args[0]
        if k == z3.Z3_OP_ITE:
            return max(args[1:])
        return Q_BOUND * self.p

    def eq(self, lhs: Any, rhs: Any) -> None:
        self.s.add(self._eq(lhs, rhs))

    def rng(self, x: Any, bits: int) -> z3.ArithRef:
        """``x in [0, 2^bits)`` on the canonical value; returns the ranged variable."""
        if self.p and not (z3.is_const(x) and x.decl().kind() == z3.Z3_OP_UNINTERPRETED):
            y = self.var("rng")
            self.eq(y, x)
            x = y
        self.s.add(x >= 0, x < (1 << bits))
        return x

    def boolean(self, x: Any) -> None:
        self.s.add(z3.Or(x == 0, x == 1))

    def bmul(self, flag: Any, x: Any, name: str = "bm") -> Any:
        """``flag * x`` for a ``flag`` the relation itself pins to {0, 1} (a BOOL, a table output, or a product/sum of
        such): the two cases, so the identity stays linear.  Constant flags multiply directly."""
        if isinstance(flag, int):
            return flag * x
        res = self.var(name)
        self.s.add(z3.Or(z3.And(flag == 0, self._eq(res, 0)), z3.And(flag == 1, self._eq(res, x))))
        return res

    def lookup(self, key: Any, rows: dict[int, Sequence[int]], outs: Sequence[Any],
               products: Sequence[tuple[Any, int, Any]] = ()) -> None:
        """``(key, outs) in rows``; ``products`` are ``(x, column, result)`` triples meaning ``x * outs[column] = result``,
        attached per row so that they stay linear."""
        disj = []
        for k, vals in rows.items():
            conj = [key == self.c(k)] + [o == self.c(v) for o, v in zip(outs, vals)]
            conj += [self._eq(x * vals[col], res) for x, col, res in products]
            disj.append(z3.And(*conj))
        self.s.add(z3.Or(disj))


def _pow_rows(p: R.Params) -> dict[int, tuple[int, int]]:
    t = R.T_pow(p)
    return {d: t(d) for d in range(512)}


def _norm_rows(p: R.Params) -> dict[int, tuple[int, int]]:
    t = R.T_norm(p)
    return {L: t(L) for L in range(p.B + 1)}


def _exp_rows(p: R.Params) -> dict[int, tuple[int, int, int, int]]:
    t = R.T_exp(p)
    return {i: t(i) for i in range(512)}


# ---------------------------------------------------------------------------------------------------------------------
# gadget queries
# ---------------------------------------------------------------------------------------------------------------------

def align_query(pipe: Pipeline, field: int | None, d: int, scaled: int, *, timeout_ms: int = 20_000):
    """Truncating alignment of one term with ``d`` and ``scaled`` pinned: another ``aligned`` than ``scaled >> d``?"""
    p = R.params(pipe)
    pw, ipw = R.T_pow(p)(d)
    honest = scaled // pw

    def build(s):
        m = _M(s, field)
        aligned, r, rinv = m.var("aligned"), m.var("r"), m.var("rinv")
        m.eq(aligned * pw, scaled - r)
        m.rng(aligned, p.A)
        m.eq(r * ipw, rinv)
        m.rng(rinv, p.A)
        s.add(aligned != honest)
        return {"aligned": aligned, "r": r, "rinv": rinv, "honest_aligned": z3.IntVal(honest)}
    return z3_query(build, timeout_ms=timeout_ms)


def final_query(pipe: Pipeline, field: int | None, magW: int, e_out: int, *, timeout_ms: int = 20_000):
    """Final shift with ``magW`` and ``e_out`` pinned: another ``mag24`` than ``magW >> sh``?"""
    p = R.params(pipe)
    _, pow_s, invpow_s, _ = R.T_exp(p)(e_out + R.E_OFF)
    honest = magW // pow_s

    def build(s):
        m = _M(s, field)
        mag24, r_s, rsinv = m.var("mag24"), m.var("r_s"), m.var("rsinv")
        m.eq(mag24 * pow_s, magW - r_s)
        m.rng(mag24, 24)
        m.eq(r_s * invpow_s, rsinv)
        m.rng(rsinv, p.K)
        s.add(mag24 != honest)
        return {"mag24": mag24, "r_s": r_s, "rsinv": rsinv, "honest_mag24": z3.IntVal(honest)}
    return z3_query(build, timeout_ms=timeout_ms)


def norm_query(pipe: Pipeline, field: int | None, mag_t: int, *, timeout_ms: int = 20_000):
    """Normalisation with ``mag_t`` pinned: another leading-bit position ``L`` than ``mag_t.bit_length()``?"""
    p = R.params(pipe)
    honest = mag_t.bit_length()
    rows = _norm_rows(p)

    def build(s):
        m = _M(s, field)
        L, normpow, z_t, normB, magW, r_n = (m.var(n) for n in ("L", "normpow", "z_t", "normB", "magW", "r_n"))
        m.lookup(L, rows, (normpow, z_t), products=[(z3.IntVal(mag_t), 0, normB)])
        m.rng(normB - (1 - z_t) * (1 << (p.B - 1)), p.B - 1)
        m.eq(magW * (1 << (p.B - p.W)) + r_n, normB)
        m.rng(r_n, p.B - p.W)
        s.add(L != honest)
        return {"L": L, "normB": normB, "magW": magW, "r_n": r_n, "honest_L": z3.IntVal(honest)}
    return z3_query(build, timeout_ms=timeout_ms)


def estar_query(pipe: Pipeline, field: int | None, exps: Sequence[int], *, mode: str = "any", timeout_ms: int = 20_000):
    """The group maximum with the masked exponents pinned: ``E*`` other than ``max(Z, e_i)``; ``mode='above'``
    asks for a larger one, ``'nonmax'`` for E* equal to a non-maximal term's exponent (the selector attack)."""
    p = R.params(pipe)
    honest = max([p.Z, *exps])
    rows = _pow_rows(p)
    cands = sorted({p.Z, *exps})          # the chain pins E* here; only these rows of T_pow are reachable (implied)

    def build(s):
        m = _M(s, field)
        Estar = m.var("Estar")
        ds = []
        for e in exps:
            d, pw, ipw = m.var("d"), m.var("pow"), m.var("invpow")
            m.eq(d, Estar - m.c(e))
            m.lookup(d, {k: rows[k] for k in sorted({c_ - e for c_ in cands}) if 0 <= k < 512}, (pw, ipw))
            ds.append(d)
        s.add(z3.Or([Estar == m.c(p.Z), *[Estar == m.c(e) for e in exps]]))       # the attained chain, as a field identity
        if mode == "above":
            s.add(Estar > m.c(honest) if not field else z3.And(Estar != m.c(honest), Estar < 1 << 12))
        elif mode == "nonmax":
            s.add(z3.Or([Estar == m.c(e) for e in exps if e != honest]) if any(e != honest for e in exps) else False)
        else:
            s.add(Estar != m.c(honest))
        return {"Estar": Estar, "honest": z3.IntVal(honest), **{f"d[{i}]": d for i, d in enumerate(ds)}}
    return z3_query(build, timeout_ms=timeout_ms)


def sign_query(pipe: Pipeline, field: int | None, total: int, *, timeout_ms: int = 20_000):
    """Sign of the group total pinned by ``(1 - 2 neg_t) * total = mag_t``, ``mag_t`` ranged: the other sign?"""
    p = R.params(pipe)
    honest = int(total < 0)

    def build(s):
        m = _M(s, field)
        neg_t, mag_t = m.var("neg_t"), m.var("mag_t")
        m.boolean(neg_t)
        m.eq((1 - 2 * neg_t) * m.c(total), mag_t)
        m.rng(mag_t, p.B)
        s.add(neg_t != honest)
        return {"neg_t": neg_t, "mag_t": mag_t}
    return z3_query(build, timeout_ms=timeout_ms)


# ---------------------------------------------------------------------------------------------------------------------
# a whole group / a whole step
# ---------------------------------------------------------------------------------------------------------------------

def _group(m: _M, pipe: Pipeline, gi: int, st: dict[str, Any], prods: list[dict], align: str,
           extra_ranges: dict[str, int] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every predicate of one group of ``check_step``; ``st`` maps nan/inf/neg/e/M to constants or z3 terms.
    ``extra_ranges`` adds ``RANGE(column, bits)`` predicates the relation does *not* state (to test a repair).
    Returns ``(output-state terms, every column term by its witness name)``."""
    p = R.params(pipe)
    Z = m.c(p.Z)
    size = len(prods)
    pow_rows, norm_rows, exp_rows = _pow_rows(p), _norm_rows(p), _exp_rows(p)
    cols: dict[str, Any] = {}

    def v(name: str) -> z3.ArithRef:
        cols[name] = m.var(f"g{gi}.{name}")
        return cols[name]

    # the terms: (neg, e, scaled) of the accumulator and each product
    ins = [(st["neg"], st["e"], st["M"] * (1 << p.rho))] + [(q["sp"], q["em"], q["P"] << (9 + p.rho)) for q in prods]
    Estar = v("Estar")
    aligneds, signeds = [], []
    m.s.add(z3.Or([Estar == Z, *[Estar == e for _, e, _ in ins]]))                # attained: field identity
    # Rows reachable under the relation's own constraints (implied, so the solution set is unchanged): the chain
    # pins E* to {Z} u {e_i}, hence each d_i and the T_exp index to a few hundred values instead of 512 x 9 x 512.
    exact = all(isinstance(e, int) for _, e, _ in ins)
    cands = sorted({p.Z, *[_signed(m, e) for _, e, _ in ins]}) if exact else None
    for i, (neg, e, scaled) in enumerate(ins):
        d, pw, ipw = v(f"terms.{i}.d"), v(f"terms.{i}.pow"), v(f"terms.{i}.invpow")
        m.eq(d, Estar - e)
        rows_i = pow_rows if cands is None else {k: pow_rows[k] for k in sorted({c_ - _signed(m, e) for c_ in cands}) if 0 <= k < 512}
        aligned, signed = v(f"terms.{i}.aligned"), v(f"terms.{i}.signed")
        if i == 0 or align == R.ALIGN_DIV:
            r, rinv = v(f"terms.{i}.r"), v(f"terms.{i}.rinv")
            m.lookup(d, rows_i, (pw, ipw), products=[(aligned, 0, scaled - r), (r, 1, rinv)])
            m.rng(aligned, p.A)
            m.rng(rinv, p.A)
        else:
            m.lookup(d, rows_i, (pw, ipw))
            P = prods[i - 1]["P"]
            m.lookup(d, {k: (((P << (9 + p.rho)) >> min(k, p.A)),) for k in rows_i}, (aligned,))
        m.eq(m.bmul(neg, aligned), signed)
        aligneds.append(aligned)
        signeds.append(signed)
    total = v("total")
    m.eq(total, z3.Sum([a - 2 * sg for a, sg in zip(aligneds, signeds)]))
    neg_t, mag_t = v("neg_t"), v("mag_t")
    m.boolean(neg_t)
    m.eq(total - 2 * m.bmul(neg_t, total), mag_t)                                   # (1 - 2 neg_t) * total
    m.rng(mag_t, p.B)
    L, normpow, z_t, normB = v("L"), v("normpow"), v("z_t"), v("normB")
    m.lookup(L, norm_rows, (normpow, z_t), products=[(mag_t, 0, normB)])
    m.rng(normB - (1 - z_t) * (1 << (p.B - 1)), p.B - 1)
    magW, r_n = v("magW"), v("r_n")
    m.eq(magW * (1 << (p.B - p.W)) + r_n, normB)
    m.rng(r_n, p.B - p.W)
    e_out = v("e_out")
    m.eq(e_out, Estar + L - p.W)
    idx = m.var(f"g{gi}.idx")
    m.eq(idx, e_out + R.E_OFF)
    e_fin, pow_s, invpow_s, ovf = v("e_fin"), v("pow_s"), v("invpow_s"), v("ovf")
    mag24, r_s, rsinv = v("mag24"), v("r_s"), v("rsinv")
    if cands is not None:
        reach = sorted({c_ + L_ - p.W + R.E_OFF for c_ in cands for L_ in range(p.B + 1)})
        exp_rows = {k: exp_rows[k] for k in reach if 0 <= k < 512}
    m.lookup(idx, exp_rows, (e_fin, pow_s, invpow_s, ovf), products=[(mag24, 1, magW - r_s), (r_s, 2, rsinv)])
    m.rng(mag24, 24)
    m.rng(rsinv, p.K)
    # ISZERO(mag24, 24): two 12-bit chunks, is-zero lookups, and
    c0, c1, z_out = m.var(f"g{gi}.c0"), m.var(f"g{gi}.c1"), v("z_out")
    m.rng(c0, 12)
    m.rng(c1, 12)
    m.eq(c0 + c1 * 4096, mag24)
    m.eq(z_out, z3.If(z3.And(c0 == 0, c1 == 0), 1, 0))
    # special values (every flag below is pinned to {0, 1} by a BOOL, a table row, or products/sums of those)
    pinf0, ninf0 = v("pinf0"), v("ninf0")
    m.eq(m.bmul(st["inf"], 1 - st["neg"]), pinf0)
    m.eq(m.bmul(st["inf"], st["neg"]), ninf0)
    anys = {}
    any_rows = {s_: (int(s_ != 0),) for s_ in range(size + 2)}
    for name, base, key in (("nan", st["nan"], "nan"), ("pinf", pinf0, "pinf"), ("ninf", ninf0, "ninf")):
        S = v(f"S_{name}")
        m.eq(S, base + sum(q[key] for q in prods))
        anys[name] = v(f"any_{name}")
        m.lookup(S, any_rows, (anys[name],))
    ovf_eff = v("ovf_eff")
    m.eq(m.bmul(ovf, 1 - z_t), ovf_eff)
    both = m.var(f"g{gi}.both")
    m.eq(m.bmul(anys["pinf"], anys["ninf"]), both)
    any_inf = v("any_inf")
    m.eq(any_inf, anys["pinf"] + anys["ninf"] - both)
    nan_out = v("nan_out")
    m.eq(both - m.bmul(anys["nan"], both), nan_out - anys["nan"])                   # (1 - any_nan) * both
    fin_inf = m.var(f"g{gi}.fin_inf")
    m.eq(ovf_eff - m.bmul(any_inf, ovf_eff), fin_inf - any_inf)                     # (1 - any_inf) * ovf_eff
    inf_out = v("inf_out")
    m.eq(fin_inf - m.bmul(nan_out, fin_inf), inf_out)                               # (1 - nan_out) * fin_inf
    neg_inf_a = m.var(f"g{gi}.neg_inf_a")
    m.eq(m.bmul(any_inf, anys["ninf"]), neg_inf_a)
    neg_inf = v("neg_inf")
    m.eq(neg_t - m.bmul(any_inf, neg_t), neg_inf - neg_inf_a)                       # (1 - any_inf) * neg_t
    fin = v("fin")
    m.eq(fin, 1 - nan_out - inf_out)
    neg_fin = v("neg_fin")
    m.eq(m.bmul(neg_t, 1 - z_out), neg_fin)
    e_state = v("e_state")
    m.eq(m.bmul(z_out, Z - e_fin), e_state - e_fin)
    out = {k: v(f"out.{k}") for k in ("nan", "inf", "neg", "e", "M")}
    out_neg_a = m.var(f"g{gi}.out_neg_a")
    m.eq(m.bmul(inf_out, neg_inf), out_neg_a)
    m.eq(m.bmul(fin, neg_fin), out["neg"] - out_neg_a)
    m.eq(m.bmul(fin, e_state - Z), out["e"] - Z)
    m.eq(m.bmul(fin, mag24), out["M"])
    m.eq(out["nan"], nan_out)
    m.eq(out["inf"], inf_out)
    for name, bits in (extra_ranges or {}).items():
        m.rng(cols[name], bits)
    return out, cols


def _signed(m: _M, v: int) -> int:
    """The small signed integer a canonical constant stands for."""
    return v if not m.p or v < m.p // 2 else v - m.p


def _forbid(m: _M, out: dict[str, Any], honest: R.State) -> None:
    m.s.add(z3.Or(out["nan"] != m.c(honest.nan), out["inf"] != m.c(honest.inf), out["neg"] != m.c(honest.neg),
                  out["e"] != m.c(honest.e), out["M"] != m.c(honest.M)))


def _state_consts(m: _M, st: R.State) -> dict[str, int]:
    return {"nan": m.c(st.nan), "inf": m.c(st.inf), "neg": m.c(st.neg), "e": m.c(st.e), "M": m.c(st.M)}


def _prods(m: _M, prods: list[dict]) -> list[dict]:
    return [{k: m.c(q[k]) for k in ("sp", "em", "P", "nan", "pinf", "ninf")} for q in prods]


def group_query(pipe: Pipeline, field: int | None, w: dict, gi: int, *, timeout_ms: int = 60_000,
                extra_ranges: dict[str, int] | None = None):
    """Group ``gi`` of the honest witness ``w`` with its input state and products pinned: another output state?
    A SAT model's columns are named ``g{gi}.<witness key>`` and can be replayed with :func:`witness_from_model`."""
    start = sum(pipe.groups[:gi])
    prods = w["prod"][start:start + pipe.groups[gi]]
    st = w["in"] if gi == 0 else w["groups"][gi - 1]["out"]
    honest = w["groups"][gi]["out"]

    def build(s):
        m = _M(s, field)
        out, cols = _group(m, pipe, gi, _state_consts(m, st), _prods(m, prods), w.get("align", R.ALIGN_DIV), extra_ranges)
        _forbid(m, out, honest)
        return {f"g{gi}.{k}": v for k, v in cols.items()}
    return z3_query(build, timeout_ms=timeout_ms)


def step_query(pipe: Pipeline, field: int | None, w: dict, *, timeout_ms: int = 120_000,
               extra_ranges: dict[str, int] | None = None):
    """The whole step: groups chained through their output-state terms; another final state than the honest one?"""
    def build(s):
        m = _M(s, field)
        st: dict[str, Any] = _state_consts(m, w["in"])
        start = 0
        rep = {}
        for gi, size in enumerate(pipe.groups):
            prods = w["prod"][start:start + size]
            start += size
            out, cols = _group(m, pipe, gi, st, _prods(m, prods), w.get("align", R.ALIGN_DIV), extra_ranges)
            rep.update({f"g{gi}.{k}": v for k, v in cols.items()})
            st = out
        _forbid(m, st, w["groups"][-1]["out"])
        return rep
    return z3_query(build, timeout_ms=timeout_ms)


def witness_from_model(pipe: Pipeline, w: dict, model: dict[str, int], field: int | None) -> dict:
    """The honest witness ``w`` with every group column present in a SAT ``model`` replaced by the model's value
    (signed for the exponent-like columns, canonical otherwise); ``out`` states rebuilt.  The result is what the
    adversary would hand to ``check_step``."""
    import copy

    w = copy.deepcopy(w)
    signed_cols = {"Estar", "e_out", "e_fin", "e_state", "total", "out.e"}
    for key, val in model.items():
        gi_s, _, rest = key.partition(".")
        if not gi_s.startswith("g") or not gi_s[1:].isdigit():
            continue
        g = w["groups"][int(gi_s[1:])]
        if field and (rest in signed_cols or rest.startswith("terms.") and rest.endswith(".d")) and val > field // 2:
            val -= field
        parts = rest.split(".")
        if parts[0] == "terms":
            g["terms"][int(parts[1])][parts[2]] = val
        elif parts[0] == "out":
            g["out"] = R.State(**{**g["out"].__dict__, parts[1]: val})
        else:
            g[parts[0]] = val
    w["out"] = w["groups"][-1]["out"]
    return w
