"""Compile the WP2 checker's predicates for one transition unit into a Ligero constraint system.

The gadget code of ``verity_numerical.checker.gadgets`` runs unchanged on :class:`LigeroCtx`; the values it
manipulates are :class:`Expr` -- affine combinations (with Python-integer coefficients) of *rows* of the witness
matrix W.  The context realises the five predicate kinds as

    rng(x, lo, hi)      bit rows of x - lo (and, for a non-power-of-two width, of hi - 1 - x); memoised
    eq(a, b)            one linear constraint  a - b = 0
    boolean(b)          one quadratic constraint  b * b = b
    lookup(T, key)      one-hot selector rows over the *distinct rows* of the table, tied to the key
    lt_pow(x, ipow, B)  the additive form  x + 2^B - 2^B/ipow  in [0, 2^B)   (see PROTOCOL.md: no product, so
                        no wrap-around; requires the memoised range x in [0, 2^B), which the gadgets establish)

A product of two non-constant expressions creates a *product row* z with the quadratic constraint x * y = z.  Every
product and every linear constraint is audited over the integers (``System.audit``): the interval of each side, from
the ranges the constraints themselves establish, must fit in (-p/2, p/2) so that the field identity implies the
integer identity.  The two products the audit cannot bound from the gadget's own ranges (``normalise.div*M`` and
``normalise.v*mul``) are bounded by two extra range checks the driver adds (theorems of the relation; PROTOCOL.md).

Rows fall in four classes, and the witness of each row is computable from the public inputs and the checker's own
hints (``gen_step``): ``pin`` (a decode of a public word, also constrained equal to it), ``hint`` (a value of
``gen_step``), ``prod``/``bit``/``sel``/``inv`` (derived).  ``System.program`` lists them in dependency order.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from verity_numerical.checker.ctx import Table
from verity_numerical.checker.params import Params, Reject

P = 15 * 2**27 + 1          # BabyBear

__all__ = ["CompileError", "Expr", "LigeroCtx", "P", "System", "TableOut"]


class CompileError(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------------------------------------------------

class Expr:
    """c + sum_j lin[j] * W_j with Python-int coefficients (exact; reduced mod p only when materialised)."""

    __slots__ = ("c", "ctx", "lin")

    def __init__(self, ctx, lin: dict[int, int] | None = None, c: int = 0):
        self.ctx = ctx
        self.lin = {k: v for k, v in (lin or {}).items() if v != 0}
        self.c = c

    def key(self) -> tuple:
        return (tuple(sorted(self.lin.items())), self.c)

    def is_const(self) -> bool:
        return not self.lin

    def _coerce(self, o):
        if isinstance(o, Expr):
            return o
        if isinstance(o, int):
            return Expr(self.ctx, {}, o)
        return NotImplemented

    def __add__(self, o):
        o = self._coerce(o)
        if o is NotImplemented:
            return o
        lin = dict(self.lin)
        for k, v in o.lin.items():
            lin[k] = lin.get(k, 0) + v
        return Expr(self.ctx, lin, self.c + o.c)

    __radd__ = __add__

    def __neg__(self):
        return Expr(self.ctx, {k: -v for k, v in self.lin.items()}, -self.c)

    def __sub__(self, o):
        o = self._coerce(o)
        if o is NotImplemented:
            return o
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        o = self._coerce(o)
        if o is NotImplemented:
            return o
        if o.is_const():
            return Expr(self.ctx, {k: v * o.c for k, v in self.lin.items()}, self.c * o.c)
        if self.is_const():
            return o * self.c
        return self.ctx.product(self, o)

    __rmul__ = __mul__

    def __pow__(self, n):
        assert n == 2
        return self * self

    def __repr__(self):
        terms = [f"{v}*r{k}" for k, v in sorted(self.lin.items())]
        if self.c or not terms:
            terms.append(str(self.c))
        return " + ".join(terms)


class TableOut(Expr):
    """A table output: affine in the selector rows, remembering the per-group values so that ``lt_pow`` can form
    2^bits / value as another affine expression of the same selectors."""

    __slots__ = ("keys", "sel_rows", "values")

    def __init__(self, ctx, sel_rows: list[int], values: list[int], keys: list[int] | None = None):
        super().__init__(ctx, {r: v for r, v in zip(sel_rows, values)}, 0)
        self.sel_rows = list(sel_rows)
        self.values = list(values)
        self.keys = list(keys) if keys is not None else []


# ---------------------------------------------------------------------------------------------------------------------
# The system
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Row:
    idx: int
    kind: str                    # pin | hint | prod | bit | sel | inv
    name: str
    lo: int | None = None        # integer interval established for the row (inclusive lo, exclusive hi)
    hi: int | None = None
    spec: tuple = ()             # kind-specific generation data


@dataclass
class Linear:
    """sum coef_j W_j + c = pub  (pub: name of a public per-column vector, or None for 0)."""
    name: str
    lin: dict[int, int]
    c: int
    pub: str | None = None


@dataclass
class Quadratic:
    """A * B = C, each affine."""
    name: str
    a: Expr
    b: Expr
    c: Expr


@dataclass
class System:
    p: Params
    rows: list[Row] = field(default_factory=list)
    linear: list[Linear] = field(default_factory=list)
    quadratic: list[Quadratic] = field(default_factory=list)
    program: list[tuple] = field(default_factory=list)      # (op, ...) in dependency order
    pins: list[tuple[int, str]] = field(default_factory=list)
    hints: list[tuple[int, str]] = field(default_factory=list)
    outputs: dict[str, Expr] = field(default_factory=dict)
    audit_log: list[str] = field(default_factory=list)
    chain: dict | None = None                                # chain mode: {"c": [3 row idx], "y": [3 Expr], "y16": Expr}
    wide: dict = field(default_factory=dict)                 # wide groups (wide.py): group tag -> partition of the terms

    @property
    def m(self) -> int:
        return len(self.rows)

    def summary(self) -> dict:
        kinds: dict[str, int] = {}
        for r in self.rows:
            kinds[r.kind] = kinds.get(r.kind, 0) + 1
        return {"rows": self.m, "by_kind": kinds, "linear": len(self.linear), "quadratic": len(self.quadratic),
                "pins": len(self.pins), "hints": len(self.hints)}


# ---------------------------------------------------------------------------------------------------------------------
# The context
# ---------------------------------------------------------------------------------------------------------------------

class LigeroCtx:
    HALF = P // 2

    def __init__(self, p: Params):
        self.p = p
        self.sys = System(p)
        self.scope = ""
        self._bits_memo: dict[tuple, list[int]] = {}
        self._prod_memo: dict[tuple, int] = {}
        self._lookup_memo: dict[tuple, tuple] = {}
        self._known: dict[tuple, tuple[int, int]] = {}       # expr key -> integer interval [lo, hi)
        self._product_whitelist: dict[str, str] = {}

    # --- rows --------------------------------------------------------------------------------------------------------
    def _row(self, kind: str, name: str, lo=None, hi=None, spec=()) -> Expr:
        r = Row(len(self.sys.rows), kind, f"{self.scope}.{name}" if self.scope else name, lo, hi, spec)
        self.sys.rows.append(r)
        e = Expr(self, {r.idx: 1}, 0)
        if lo is not None:
            self._known[e.key()] = (lo, hi)
        return e

    def pin(self, name: str, lo: int, hi: int) -> Expr:
        """A row the verifier knows (a decode of a public word): constrained equal to the public vector ``name``."""
        e = self._row("pin", name, lo, hi, (name,))
        idx = next(iter(e.lin))
        self.sys.pins.append((idx, name))
        self.sys.program.append(("pin", idx, name))
        self.sys.linear.append(Linear(f"pin:{name}", {idx: 1}, 0, name))
        return e

    def hint(self, name: str) -> Expr:
        """A hint row (a value of ``gen_step``, flattened under ``name``); its range comes from the gadget's checks."""
        e = self._row("hint", name, spec=(name,))
        idx = next(iter(e.lin))
        self.sys.hints.append((idx, name))
        self.sys.program.append(("hint", idx, name))
        return e

    def hints(self, prefix: str, names: list[str]) -> dict[str, Expr]:
        return {n: self.hint(f"{prefix}.{n}") for n in names}

    # --- intervals ---------------------------------------------------------------------------------------------------
    def interval(self, e: Expr) -> tuple[int, int] | None:
        """Integer interval [lo, hi] (inclusive) of an affine expression from the rows' established ranges: the
        intersection of (a) the masked-product analysis and (b) the split along known compound sub-expressions."""
        k = self._known.get(e.key())
        if k is not None:
            return (k[0], k[1] - 1)
        r1 = self._interval_peel(e)
        r2 = self._interval_split(e)
        if r1 is None:
            return r2
        if r2 is None:
            return r1
        return (max(r1[0], r2[0]), min(r1[1], r2[1]))

    def _interval_peel(self, e: Expr) -> tuple[int, int] | None:
        # The masked pattern  k*X + c*(bit*X)  =  (k + c*bit) * X  (SignedSum's (1 - 2 s) q and v = S - 2 sgn S,
        # StateOut's e_hat = e + z (floor - e), ...): the hull of k*I(X) and (k + c)*I(X), which interval arithmetic
        # on the product row alone cannot see.  Each such group is peeled off exactly; the remainder is naive.
        lin = dict(e.lin)
        lo = hi = e.c
        prods = [j for j in lin if self.sys.rows[j].kind == "prod"]
        prods.sort(key=lambda j: -max(len(self.sys.rows[j].spec[0].lin), len(self.sys.rows[j].spec[1].lin)))
        for j in prods:
            if j not in lin:
                continue
            cj = lin[j]
            pa, pb = self.sys.rows[j].spec
            for B, X in ((pa, pb), (pb, pa)):
                if X.is_const() or B.is_const():
                    continue
                g = 0
                for v in list(B.lin.values()) + ([B.c] if B.c else []):
                    g = math.gcd(g, abs(v))
                if g == 0:
                    continue
                Bn = Expr(self, {k: v // g for k, v in B.lin.items()}, B.c // g)
                if not self.known_within(Bn, 0, 2):
                    continue
                j0 = next(iter(X.lin))
                if j0 not in lin or lin[j0] % X.lin[j0]:
                    continue
                kf = lin[j0] // X.lin[j0]
                if kf == 0 or any(lin.get(jj) != kf * v for jj, v in X.lin.items()):
                    continue
                ix = self.interval(X)
                if ix is None:
                    continue
                d = 0
                if len(Bn.lin) == 1 and Bn.c == 0:
                    (jb, vb), = Bn.lin.items()
                    if vb == 1 and jb in lin and jb not in X.lin and jb != j:
                        d = lin.pop(jb)
                y0, y1 = ix[0] - X.c, ix[1] - X.c
                cg = cj * g
                c0 = [kf * y0, kf * y1]
                c1 = [(kf + cg) * y0 + cg * X.c + d, (kf + cg) * y1 + cg * X.c + d]
                lo += min(c0 + c1)
                hi += max(c0 + c1)
                for jj in X.lin:
                    del lin[jj]
                del lin[j]
                break
        lo_, hi_ = lo, hi
        for j, v in lin.items():
            r = self.sys.rows[j]
            if r.lo is None:
                return None
            a, b = v * r.lo, v * (r.hi - 1)
            lo_ += min(a, b)
            hi_ += max(a, b)
        return (lo_, hi_)

    def _interval_split(self, e: Expr) -> tuple[int, int] | None:
        base = (e.c, e.c)
        e = Expr(self, e.lin, 0)
        # A compound sub-expression with an established range (e.g. v after rng(v)): split it off.
        best = None
        for (klin, kc), kiv in self._known.items():
            if len(klin) < 2 or (best is not None and len(klin) <= len(best[0])):
                continue
            j0, v0 = klin[0]
            if j0 not in e.lin or e.lin[j0] % v0:
                continue
            c = e.lin[j0] // v0
            if c != 0 and all(e.lin.get(j) == c * v for j, v in klin):
                best = (klin, kc, kiv, c)
        if best is not None:
            klin, kc, kiv, c = best
            rest = Expr(self, {j: v for j, v in e.lin.items() if j not in dict(klin)}, e.c - c * kc)
            riv = self.interval(rest)
            if riv is None:
                return None
            a, b = c * kiv[0], c * (kiv[1] - 1)
            return (base[0] + min(a, b) + riv[0], base[1] + max(a, b) + riv[1])
        return None


    def _learn(self, e: Expr, lo: int, hi: int) -> None:
        """Record lo <= e < hi; a single-row expression tightens the row's interval."""
        self._known[e.key()] = (lo, hi)
        if len(e.lin) == 1 and e.c == 0:
            (j, v), = e.lin.items()
            if v == 1:
                r = self.sys.rows[j]
                if r.lo is None or (hi - lo) < (r.hi - r.lo):
                    r.lo, r.hi = lo, hi

    def known_within(self, e: Expr, lo: int, hi: int) -> bool:
        iv = self.interval(e)
        return iv is not None and iv[0] >= lo and iv[1] < hi

    def learn(self, e: Expr, lo: int, hi: int) -> None:
        """A driver-asserted bound ``lo <= e < hi`` that is a theorem of constraints already stated (a partial sum
        of a bit decomposition, say); logged, as the product whitelist is."""
        self.sys.audit_log.append(f"learned in {self.scope}: {e!r} in [{lo}, {hi})")
        self._learn(e, lo, hi)

    def bit_rows(self, x: Expr, nbits: int, name: str) -> list[int]:
        """The rows of ``x``'s ``nbits``-bit decomposition (the one ``rng(x, 0, 2^nbits)`` made, or a fresh one)."""
        return self._bits(x, nbits, name)

    # --- products ----------------------------------------------------------------------------------------------------
    def whitelist_product(self, scope_suffix: str, reason: str, lo: int, hi: int) -> None:
        """Products formed in scopes ending with ``scope_suffix`` are accepted with the declared integer interval
        [lo, hi) -- the driver must add the range checks that justify it (recorded in the audit log)."""
        self._product_whitelist[scope_suffix] = (reason, lo, hi)

    def product(self, a: Expr, b: Expr) -> Expr:
        """A product row z = a * b.  Its integer interval is the product of the operands' intervals, or the tighter
        interval a whitelisted scope declares (justified by a range check the gadget states elsewhere; logged).  No
        soundness condition is imposed here: the field identity of every *linear* constraint that uses z is what
        must be an integer identity, and ``_linear`` checks that from these intervals."""
        ka, kb = a.key(), b.key()
        mk = (ka, kb) if ka <= kb else (kb, ka)
        if mk in self._prod_memo:
            return Expr(self, {self._prod_memo[mk]: 1}, 0)
        ia, ib = self.interval(a), self.interval(b)
        lo = hi = None
        if ia is not None and ib is not None:
            prods = [ia[0] * ib[0], ia[0] * ib[1], ia[1] * ib[0], ia[1] * ib[1]]
            lo, hi = min(prods), max(prods) + 1
        for suffix, (reason, wlo, whi) in self._product_whitelist.items():
            if suffix in self.scope and (lo is None or (lo <= wlo and hi >= whi and hi - lo >= P)):
                self.sys.audit_log.append(f"product in {self.scope}: {a!r} * {b!r}: naive interval "
                                          f"{None if lo is None else (lo, hi - 1)} replaced by [{wlo}, {whi}): {reason}")
                lo, hi = wlo, whi
        z = self._row("prod", f"prod{len(self._prod_memo)}", lo, hi, (a, b))
        idx = next(iter(z.lin))
        self._prod_memo[mk] = idx
        self.sys.program.append(("prod", idx, a, b))
        self.sys.quadratic.append(Quadratic(f"{self.scope}.prod{idx}", a, b, z))
        return z

    # --- predicates --------------------------------------------------------------------------------------------------
    def _bits(self, e: Expr, nbits: int, name: str) -> list[int]:
        key = (e.key(), nbits)
        if key in self._bits_memo:
            return self._bits_memo[key]
        rows = []
        for i in range(nbits):
            r = self._row("bit", f"{name}.b{i}", 0, 2, (e, i))
            j = next(iter(r.lin))
            rows.append(j)
            self.sys.quadratic.append(Quadratic(f"{self.scope}.{name}.bool{i}", r, r, r))
        self.sys.program.append(("bits", rows[0], nbits, e))
        lin = {j: -(2**i) for i, j in enumerate(rows)}
        for j, v in e.lin.items():
            lin[j] = lin.get(j, 0) + v
        # A first range check of a hint row *defines* its integer: the row's field value is congruent to exactly one
        # integer of the decomposition's interval, and every later constraint reads it as that integer.
        unknown = [j for j in e.lin if self.sys.rows[j].lo is None]
        if len(unknown) == 1 and abs(e.lin[unknown[0]]) == 1:
            j = unknown[0]
            rest = Expr(self, {k: v for k, v in e.lin.items() if k != j}, e.c)
            iv = self.interval(rest)
            if iv is not None:
                # e = sign*W_j + rest in [0, 2^nbits)  =>  W_j = sign*(e - rest)
                sgn = e.lin[j]
                cands = [sgn * (0 - iv[1]), sgn * (0 - iv[0]), sgn * (2**nbits - 1 - iv[1]), sgn * (2**nbits - 1 - iv[0])]
                self.sys.rows[j].lo, self.sys.rows[j].hi = min(cands), max(cands) + 1
        self._linear(f"{name}.decomp", Expr(self, lin, e.c))
        self._bits_memo[key] = rows
        self._learn(e, 0, 2**nbits)
        return rows

    def _linear(self, name: str, e: Expr, pub: str | None = None) -> None:
        """e = 0 (or e = pub).  Audited: the integer interval of e must fit in (-p, p)."""
        iv = self.interval(e)
        if iv is None:
            raise CompileError(f"linear {self.scope}.{name}: integer interval unknown: {e!r}")
        if iv[0] <= -P or iv[1] >= P:
            raise CompileError(f"linear {self.scope}.{name}: interval {iv} not inside (-p, p): the field identity "
                               f"would not be an integer identity")
        self.sys.linear.append(Linear(f"{self.scope}.{name}", dict(e.lin), e.c, pub))

    def rng(self, name: str, x, lo: int, hi: int) -> None:
        if isinstance(x, int):
            if not lo <= x < hi:
                raise Reject(f"{name}: constant {x} not in [{lo}, {hi})")
            return
        y = x - lo
        N = hi - lo
        if N <= 0:
            raise CompileError(f"{name}: empty range")
        if self.known_within(y, 0, N):
            return
        b = max(1, math.ceil(math.log2(N)))
        self._bits(y, b, name)
        if N != 2**b:
            self._bits((N - 1) - y, b, name + ".upper")
        self._learn(y, 0, N)
        self._learn(x, lo, hi)

    def eq(self, name: str, a, b) -> None:
        if isinstance(a, int) and isinstance(b, int):
            if a != b:
                raise Reject(f"{name}: {a} != {b}")
            return
        e = (a - b) if isinstance(a, Expr) else (b - a)
        if e.is_const():
            if e.c != 0:
                raise Reject(f"{name}: constant {e.c} != 0")
            return
        self._linear(name, e)

    def eq_pub(self, name: str, a: Expr, pub: str) -> None:
        """a = the public per-column vector ``pub`` (the output word's components)."""
        self._linear(name, a, pub)

    def boolean(self, name: str, b) -> None:
        if isinstance(b, int):
            if b not in (0, 1):
                raise Reject(f"{name}: {b} is not a bit")
            return
        if self.known_within(b, 0, 2):
            return
        self.sys.quadratic.append(Quadratic(f"{self.scope}.{name}", b, b, b))
        self._learn(b, 0, 2)

    def lookup(self, name: str, table: Table, key) -> tuple:
        if isinstance(key, int):
            return table.row(key)
        memo = (table.name, key.key())
        if memo in self._lookup_memo:
            return self._lookup_memo[memo]
        # Group the keys by identical output rows; each group must be a contiguous key interval.
        groups: list[tuple[int, int, tuple[int, ...]]] = []
        for k in range(table.lo, table.hi):
            v = table.fn(k)
            if groups and groups[-1][2] == v and groups[-1][1] == k - 1:
                groups[-1] = (groups[-1][0], k, v)
            else:
                groups.append((k, k, v))
        self.rng(f"{name}.key", key, table.lo, table.hi)
        sel_rows = []
        for gi, (k0, k1, _) in enumerate(groups):
            r = self._row("sel", f"{name}.sel{gi}", 0, 2, (key, k0))
            j = next(iter(r.lin))
            sel_rows.append(j)
            self.sys.quadratic.append(Quadratic(f"{self.scope}.{name}.selbool{gi}", r, r, r))
        self.sys.program.append(("sels", sel_rows[0], [g[0] for g in groups], [g[1] for g in groups], key))
        self._linear(f"{name}.onehot", Expr(self, {j: 1 for j in sel_rows}, -1))
        kk = TableOut(self, sel_rows, [g[0] for g in groups], [g[0] for g in groups])
        intervals = [gi for gi, (k0, k1, _) in enumerate(groups) if k1 > k0]
        if not intervals:
            self._linear(f"{name}.keysel", key - kk)
        else:
            for gi in intervals:
                if groups[gi][1] != table.hi - 1:
                    raise CompileError(f"{name}: interval group of {table.name} does not end at the table's top")
            # key - kk >= 0 always; key - kk = 0 unless an interval group is selected (then key < hi bounds it).
            diff = key - kk
            span = max(groups[gi][1] - groups[gi][0] for gi in intervals) + 1
            self.rng(f"{name}.keydiff", diff, 0, 2 ** math.ceil(math.log2(span)))
            not_interval = Expr(self, {sel_rows[gi]: -1 for gi in intervals}, 1)
            self.sys.quadratic.append(Quadratic(f"{self.scope}.{name}.keysel", diff, not_interval, Expr(self)))
        outs = tuple(TableOut(self, sel_rows, [g[2][o] for g in groups], [g[0] for g in groups]) for o in range(len(table.outs)))
        for o, out in enumerate(outs):
            self._known[out.key()] = (min(out.values), max(out.values) + 1)
        self._lookup_memo[memo] = outs
        return outs

    def table_derived(self, out: TableOut, fn) -> TableOut:
        """Another column of the same lookup: ``fn(value)`` per group (a power of two for the additive tests)."""
        t = TableOut(self, out.sel_rows, [fn(v) for v in out.values], out.keys)
        self._known[t.key()] = (min(t.values), max(t.values) + 1)
        return t

    def table_by_key(self, out: TableOut, fn) -> TableOut:
        """Another column of the same lookup as a function of the group's (first) key."""
        t = TableOut(self, out.sel_rows, [fn(k) for k in out.keys], out.keys)
        self._known[t.key()] = (min(t.values), max(t.values) + 1)
        return t

    def lt_pow(self, name: str, x, ipow, bits: int) -> None:
        """0 <= x < 2^bits / ipow, as  pw - 1 - x in [0, 2^b)  with pw = 2^bits / ipow, given x in [0, 2^bits)."""
        if isinstance(ipow, int):
            self.rng(name, x, 0, 2**bits // ipow)
            return
        if not isinstance(ipow, TableOut):
            raise CompileError(f"{name}: lt_pow needs a table output as the power")
        for v in ipow.values:
            if v <= 0 or (2**bits) % v:
                raise CompileError(f"{name}: {v} does not divide 2^{bits}")
        pw = self.table_derived(ipow, lambda v: 2**bits // v)
        self.lt_table(name, x, pw, bits)

    def lt_table(self, name: str, x: Expr, bound: TableOut, xbits: int) -> None:
        """x < bound for a table column of powers of two <= 2^xbits:  bound - 1 - x in [0, 2^b), 2^b >= max bound
        (the upper side holds automatically from x >= 0).  When 2^xbits + 2^b would reach p, the check is split over
        the two halves of x's (memoised) bit decomposition so that every linear constraint stays inside (-p, p)."""
        for v in bound.values:
            if v <= 0 or v & (v - 1) or v > 2**xbits:
                raise CompileError(f"{name}: bound {v} is not a power of two <= 2^{xbits}")
        if not self.known_within(x, 0, 2**xbits):
            self.rng(f"{name}.nonneg", x, 0, 2**xbits)
        b = max(1, math.ceil(math.log2(max(bound.values))))
        if 2**xbits + 2**b < P // 2:
            self.rng(name, bound - 1 - x, 0, 2**b)
            return
        rows = self._bits(x, xbits, name + ".split")
        h = xbits // 2
        xl = Expr(self, {rows[i]: 2**i for i in range(h)}, 0)
        xh = Expr(self, {rows[i]: 2 ** (i - h) for i in range(h, xbits)}, 0)
        self._learn(xl, 0, 2**h)
        self._learn(xh, 0, 2 ** (xbits - h))
        bh = self.table_derived(bound, lambda v: 2 ** max(v.bit_length() - 1 - h, 0))
        bl = self.table_derived(bound, lambda v: 2 ** min(v.bit_length() - 1, h))
        self.rng(name + ".hi", bh - 1 - xh, 0, 2 ** (xbits - h))
        if min(bl.values) < 2**h:
            self.rng(name + ".lo", bl - 1 - xl, 0, 2**h)

    def inverse_nonzero(self, name: str, x: Expr) -> None:
        """x != 0 by a hint row inv with x * inv = 1."""
        inv = self._row("inv", f"{name}.inv", spec=(x,))
        idx = next(iter(inv.lin))
        self.sys.program.append(("inv", idx, x))
        self.sys.quadratic.append(Quadratic(f"{self.scope}.{name}.nonzero", x, inv, Expr(self, {}, 1)))
