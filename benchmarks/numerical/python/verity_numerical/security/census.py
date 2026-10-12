"""Census of the frozen target's checker, counted from the WP2 gadget contracts (``verity_numerical.checker``), not
from ``verity.ml.circuits.silicon.relation`` (whose ``census``/``vu_census`` count none of the repairs the red team and WP2 found:
RANGE(magW), direct remainder ranges, the tight quotient bound, the decode of an arbitrary incoming accumulator).

Method.  ``step.py``'s composed unit (``check_step_expr``) and VU (``check_vu_expr``) are executed once on *symbolic*
values (:class:`Sym`) through a counting context (:class:`CensusCtx`) that implements the checker's five predicates:

    boolean(b)            1 field multiplication  b (1 - b) = 0
    eq(a, b)              1 linear identity (the multiplications inside ``a``, ``b`` are counted when formed)
    rng(x, lo, hi)        a w-bit range check, w = bits of (hi - lo - 1): recorded by width; realised as
                          ceil(w/16) lookups into 16-bit range tables (+ 1 linear split when w > 16), the convention of
                          ``relation.Census.RANGE``; when ``x`` is not a single column it is a *materialised* expression
    lookup(TABLE, key)    1 query into a fixed table (POW, ALIGN, LEAD, NORM); the row's columns become witness columns
    lt_pow(x, ipow, B)    the reference realisation rng(x * ipow, 0, 2^B): 1 multiplication + one B-bit range check
                          (sound over Goldilocks, p > 2^60 > 2^(2B): checker docstring F1; a small field needs the
                          direct realisation instead, not counted here)

A multiplication is counted whenever two non-constant expressions are multiplied (constant coefficients are free);
each product is a column (the ``z`` of a Ligero triple, a gate of a GKR layer) and its multiplicative depth is
recorded, which gives the layered shape A-GKR's data-parallel GKR needs.  Identical products formed twice are
counted twice (an upper count; a compiler may share them).

Witness columns, with bit widths from ``contract().hints``: the public words (``c``, 32 operand words), every hint,
every looked-up row column.  Products and materialised range expressions are further columns for a backend that
commits every value (B-Ligero); A-GKR commits only inputs + hints + lookup outputs and computes the rest.

Two unit variants are counted (``archive/AGENTS.md``: the backend transition unit is one 16-product update with
arbitrary finite ``c``; the VU is the chain from ``c_0 = 0``):

    unit    ``check_step_expr`` with symbolic ``c``: the decode of an arbitrary accumulator word and the pack of the
            output word are inside the unit, plus the unit's output identity ``y = word``;
    vu      ``check_vu_expr`` for K/16 word-chained units from the constant 0 plus the epilogue, exactly as
            ``step.py`` composes it (the first unit decodes a constant; every other unit decodes the previous word).

``diff_vs_relation`` puts the two side by side with ``verity.ml.circuits.silicon.relation.census`` / ``vu_census``.

Per-gadget breakdown (``UnitCensus.by_gadget``): every gadget's ``check`` (and the free-standing pieces v2's
composition calls directly) is wrapped for the duration of a census so that the predicates and multiplications it
forms are attributed to its ``name``; hints are attributed to the gadget whose contract declares them.

The v2 arithmetisation (``verity_numerical.checker.v2``) is counted by ``census_unit_v2`` / ``census_vu_v2`` with the
same interpreter; its fixed tables are ``fixed_tables_v2``.  With the environment variable
``VERITY_CHECKER_VARIANT=v2`` the public ``census_unit`` / ``census_vu`` / ``fixed_tables`` return the v2 counts, so
that every consumer of the census (``security.accounting``, ``explore.fields``, ``explore.tensor``, ``explore.vole``)
re-prices on v2 in a subprocess without a code change (``CHECKER_VARIANT`` names the active variant).
"""
from __future__ import annotations

import contextlib
import math
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from verity_numerical.checker import REAL, Params
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
from verity_numerical.checker.step import (
    check_step_expr,
    check_vu_expr,
    gen_step,
    gen_vu,
)
from verity_numerical.checker.tables import (
    align_table,
    lead_table,
    norm_table,
    pow_table,
)
from verity_numerical.checker.v2 import gadgets as g2
from verity_numerical.checker.v2 import step as step2
from verity_numerical.checker.v2.tables import all_tables as v2_tables

__all__ = [
    "RANGE_CHUNK",
    "CensusCtx",
    "Sym",
    "UnitCensus",
    "CHECKER_VARIANT",
    "census_unit",
    "census_unit_v1",
    "census_unit_v2",
    "census_vu",
    "census_vu_v1",
    "census_vu_v2",
    "diff_vs_relation",
    "fixed_tables",
    "fixed_tables_v1",
    "fixed_tables_v2",
    "gadget_table",
    "layer_shape",
    "lookups_by_table",
]

V1_GADGETS = (Fp32Decode, Bf16Decode, Product, GroupMax, AlignState, AlignProduct, SignedSum, LeadingBit, Normalise,
              StateOut, Pack, Epilogue)
V2_GADGETS = (g2.StateDecode, g2.OpDecode, g2.Product, g2.GroupMax, g2.StateShift, g2.ProductShift, g2.SignedSum,
              g2.Normalise, g2.Pack, Epilogue)
_SECTION_METHODS = ("check", "align", "attained")     # the entry points that take ``ctx`` first

RANGE_CHUNK = 16     # the largest range table (relation.RANGE_CHUNK): a w-bit check is ceil(w/16) lookups
CHECKER_VARIANT = os.environ.get("VERITY_CHECKER_VARIANT", "v1")
if CHECKER_VARIANT not in ("v1", "v2"):
    raise ValueError(f"VERITY_CHECKER_VARIANT must be v1 or v2, got {CHECKER_VARIANT!r}")


def _width(lo: int, hi: int) -> int:
    return max(1, (hi - lo - 1).bit_length())


class Sym:
    """A symbolic checker value: ``const`` when it is an integer constant; otherwise ``atom`` is the column id when
    the value is exactly one column (coefficient 1, no offset) and ``depth`` its multiplicative depth."""

    __slots__ = ("atom", "const", "ctx", "depth")

    def __init__(self, ctx: CensusCtx, const: int | None = None, atom: int | None = None, depth: int = 0):
        self.ctx, self.const, self.atom, self.depth = ctx, const, atom, depth

    # -- arithmetic ------------------------------------------------------------------------------------------------
    def _lift(self, other) -> Sym:
        if isinstance(other, Sym):
            return other
        if isinstance(other, int):
            return Sym(self.ctx, const=other)
        return NotImplemented

    def _linear(self, other: Sym) -> Sym:
        if self.const is not None and other.const is not None:
            return Sym(self.ctx, const=0)          # value irrelevant: constants never reach a counted predicate
        return Sym(self.ctx, depth=max(self.depth, other.depth))

    def __add__(self, other):
        o = self._lift(other)
        return o if o is NotImplemented else self._linear(o)

    __radd__ = __add__

    def __sub__(self, other):
        o = self._lift(other)
        return o if o is NotImplemented else self._linear(o)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __neg__(self):
        return Sym(self.ctx, const=0) if self.const is not None else Sym(self.ctx, depth=self.depth)

    def __mul__(self, other):
        o = self._lift(other)
        if o is NotImplemented:
            return o
        if self.const is not None and o.const is not None:
            return Sym(self.ctx, const=0)
        if self.const is not None or o.const is not None:       # constant coefficient: linear, free
            nz = self if self.const is None else o
            return Sym(self.ctx, depth=nz.depth)
        return self.ctx.product(self, o)

    __rmul__ = __mul__

    def __pow__(self, k):
        raise TypeError("the checker never exponentiates a witness")

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass
class UnitCensus:
    """The counts.  ``columns_*`` are witness columns; ``range_by_width`` maps bit width -> number of range checks;
    ``lookups`` maps fixed-table name -> queries; ``mul_by_depth`` maps multiplicative depth -> products."""

    variant: str
    columns_inputs: int = 0
    columns_hints: int = 0
    columns_lookup_outputs: int = 0
    columns_products: int = 0
    columns_materialised: int = 0
    hint_bits: Counter = field(default_factory=Counter)          # width -> number of hint columns
    mul: int = 0
    lin: int = 0
    lin_range_splits: int = 0
    booleans: int = 0
    range_by_width: Counter = field(default_factory=Counter)
    lookups: Counter = field(default_factory=Counter)
    mul_by_depth: Counter = field(default_factory=Counter)
    columns_in_mul: int = 0                                       # columns that are a direct operand or product
    queries_on_computed_wires: int = 0                            # range checks / lookup keys that are not columns of depth 0
    query_depth: Counter = field(default_factory=Counter)         # multiplicative depth -> number of queries on it
    by_gadget: dict = field(default_factory=dict)                 # gadget name -> Counter(mul, lin, range, fixed, hints)

    def _g(self, section: str) -> Counter:
        return self.by_gadget.setdefault(section, Counter())

    # -- derived -----------------------------------------------------------------------------------------------------
    @property
    def range_lookups(self) -> int:
        """Range checks realised as 16-bit chunk lookups (``relation.RANGE`` convention)."""
        return sum(-(-w // RANGE_CHUNK) * n for w, n in self.range_by_width.items())

    @property
    def fixed_lookups(self) -> int:
        return sum(self.lookups.values())

    @property
    def lookup(self) -> int:
        return self.range_lookups + self.fixed_lookups

    @property
    def lin_total(self) -> int:
        """Identities including the chunk splits of range checks wider than 16 bits."""
        return self.lin + self.lin_range_splits

    @property
    def columns_committed_a(self) -> int:
        """A-GKR commits inputs, hints and looked-up row columns; products and materialised expressions are wires."""
        return self.columns_inputs + self.columns_hints + self.columns_lookup_outputs

    @property
    def columns_all(self) -> int:
        return self.columns_committed_a + self.columns_products + self.columns_materialised

    @property
    def columns_committed_b(self) -> int:
        """B-Ligero (Ligero): every product occupies an aligned triple (x, y, z: 3 slots, operands copied), every
        other column one slot.  ESTIMATE of the layout, upper count."""
        return 3 * self.mul + (self.columns_all - self.columns_in_mul)

    @property
    def depth(self) -> int:
        return max(self.mul_by_depth) if self.mul_by_depth else 0

    @property
    def max_query_depth(self) -> int:
        """Deepest checker layer that holds a lookup query wire (0 = every query is an input-layer value)."""
        return max(self.query_depth) if self.query_depth else 0

    def summary(self) -> dict:
        return {
            "variant": self.variant,
            "columns": {
                "inputs": self.columns_inputs, "hints": self.columns_hints, "lookup_outputs": self.columns_lookup_outputs,
                "products": self.columns_products, "materialised": self.columns_materialised,
                "committed_A": self.columns_committed_a, "all": self.columns_all, "committed_B_estimate": self.columns_committed_b,
            },
            "hint_bits": dict(sorted(self.hint_bits.items())),
            "mul": self.mul, "booleans_in_mul": self.booleans, "lin": self.lin, "lin_range_splits": self.lin_range_splits,
            "lookup": self.lookup, "range_lookups": self.range_lookups, "fixed_lookups": self.fixed_lookups,
            "range_by_width": dict(sorted(self.range_by_width.items())),
            "lookups_by_table": dict(sorted(self.lookups.items())),
            "mul_by_depth": dict(sorted(self.mul_by_depth.items())), "depth": self.depth,
            "queries_on_computed_wires": self.queries_on_computed_wires,
            "query_depth": dict(sorted(self.query_depth.items())), "max_query_depth": self.max_query_depth,
            "by_gadget": gadget_table(self),
        }


def gadget_table(c: UnitCensus) -> dict[str, dict[str, int]]:
    """Per gadget: multiplications, linear identities, lookups (range-check chunks + fixed-table queries), hint
    columns; sorted by lookups + mul descending."""
    rows = {}
    for name, k in c.by_gadget.items():
        lookups = k["range_lookups"] + k["fixed"]
        rows[name] = {"mul": k["mul"], "lin": k["lin"], "lookup": lookups, "range": k["range_lookups"],
                      "fixed": k["fixed"], "hints": k["hints"]}
    return dict(sorted(rows.items(), key=lambda kv: -(kv[1]["lookup"] + kv[1]["mul"])))


class CensusCtx:
    """The counting interpreter of the checker's predicates (see the module docstring)."""

    def __init__(self, variant: str):
        self.c = UnitCensus(variant)
        self._next = 0
        self.sections: list[str] = ["composition"]     # the innermost gadget whose entry point is running

    @property
    def section(self) -> Counter:
        return self.c._g(self.sections[-1])

    def column(self) -> int:
        self._next += 1
        return self._next

    def const(self, v: int) -> Sym:
        return Sym(self, const=v)

    def input(self, bits: int) -> Sym:
        self.c.columns_inputs += 1
        return Sym(self, atom=self.column())

    def hint(self, bits: int) -> Sym:
        self.c.columns_hints += 1
        self.c.hint_bits[bits] += 1
        return Sym(self, atom=self.column())

    def product(self, a: Sym, b: Sym) -> Sym:
        self.c.mul += 1
        self.section["mul"] += 1
        for x in (a, b):
            if x.atom is not None:
                self.c.columns_in_mul += 1
        d = 1 + max(a.depth, b.depth)
        self.c.mul_by_depth[d] += 1
        self.c.columns_products += 1
        self.c.columns_in_mul += 1
        return Sym(self, atom=self.column(), depth=d)

    def _materialise(self, x: Sym) -> None:
        if x.atom is None:
            self.c.columns_materialised += 1

    def _query(self, x: Sym) -> None:
        """A lookup query (range check or table key) on ``x``: on a computed wire unless ``x`` is a depth-0 column."""
        self._materialise(x)
        self.c.query_depth[x.depth] += 1
        if x.atom is None or x.depth > 0:
            self.c.queries_on_computed_wires += 1

    # -- the five predicates ---------------------------------------------------------------------------------------------
    def rng(self, name: str, x, lo: int, hi: int) -> None:
        if isinstance(x, int) or x.const is not None:
            return
        w = _width(lo, hi)
        self._query(x)
        self.c.range_by_width[w] += 1
        self.section["range_lookups"] += -(-w // RANGE_CHUNK)
        if w > RANGE_CHUNK:
            self.c.lin_range_splits += 1
            self.section["lin"] += 1

    def eq(self, name: str, a, b) -> None:
        if (isinstance(a, int) or a.const is not None) and (isinstance(b, int) or b.const is not None):
            return
        self.c.lin += 1
        self.section["lin"] += 1

    def boolean(self, name: str, b) -> None:
        if isinstance(b, int) or b.const is not None:
            return
        self.c.mul += 1
        self.section["mul"] += 1
        self.c.booleans += 1
        self.c.mul_by_depth[1 + b.depth] += 1
        if b.atom is not None:
            self.c.columns_in_mul += 1

    def lookup(self, name: str, table, key) -> tuple:
        if isinstance(key, int) or key.const is not None:
            return tuple(self.const(v) for v in table.row(key if isinstance(key, int) else 0))
        self._query(key)
        self.c.lookups[table.name] += 1
        self.section["fixed"] += 1
        self.c.columns_lookup_outputs += len(table.outs)
        return tuple(Sym(self, atom=self.column(), depth=key.depth) for _ in table.outs)

    def lt_pow(self, name: str, x, ipow, bits: int) -> None:
        self.rng(name, x * ipow, 0, 2**bits)


@contextlib.contextmanager
def _sectioned(gadgets: Sequence[type]):
    """Wrap every gadget entry point (``check``, and v2's ``align``/``attained``) so that the predicates it forms are
    attributed to the gadget's name in the census context (the first positional argument)."""
    saved = []
    for g in gadgets:
        for meth in _SECTION_METHODS:
            fn = g.__dict__.get(meth)
            if fn is None:
                continue
            raw = fn.__func__ if isinstance(fn, staticmethod) else fn

            def wrapped(ctx, *args, _raw=raw, _name=g.name, **kw):
                if isinstance(ctx, CensusCtx):
                    ctx.sections.append(_name)
                    try:
                        return _raw(ctx, *args, **kw)
                    finally:
                        ctx.sections.pop()
                return _raw(ctx, *args, **kw)

            saved.append((g, meth, fn))
            setattr(g, meth, staticmethod(wrapped))
    try:
        yield
    finally:
        for g, meth, fn in saved:
            setattr(g, meth, fn)


# ---------------------------------------------------------------------------------------------------------------------
# Symbolic hints from the gadget contracts
# ---------------------------------------------------------------------------------------------------------------------

def _hint_widths(contract) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, (lo, hi) in contract.hints.items():
        base = name.split("[")[0]
        out[base] = max(out.get(base, 0), _width(lo, hi))
    return out


def _sym_hints(ctx: CensusCtx, p: Params, shape, widths: dict[str, int], gadget: str, key: str | None = None):
    if shape is None:
        return None
    if isinstance(shape, dict):
        return {k: _sym_hints(ctx, p, v, widths, gadget, k) for k, v in shape.items()}
    if isinstance(shape, list):
        return [_sym_hints(ctx, p, v, widths, gadget, key) for v in shape]
    if isinstance(shape, int):
        ctx.c._g(gadget)["hints"] += 1
        return ctx.hint(widths[key])
    raise TypeError(shape)


def _sym_step_hints(ctx: CensusCtx, p: Params, shape: dict) -> dict:
    W = {g.name: _hint_widths(g.contract(p)) for g in V1_GADGETS}

    def H(sh, g):
        return _sym_hints(ctx, p, sh, W[g], g)

    h = {
        "acc": H(shape["acc"], "fp32_decode"),
        "a": [H(x, "bf16_decode") for x in shape["a"]],
        "b": [H(x, "bf16_decode") for x in shape["b"]],
        "prod": [H(x, "product") for x in shape["prod"]],
        "groups": [],
        "pack": H(shape["pack"], "pack"),
    }
    for g in shape["groups"]:
        h["groups"].append({
            "max": H(g["max"], "group_max"),
            "acc_align": H(g["acc_align"], "align_state"),
            "prod_align": [H(x, "align_product") for x in g["prod_align"]],
            "sum": H(g["sum"], "signed_sum"),
            "lead": H(g["lead"], "leading_bit"),
            "norm": H(g["norm"], "normalise"),
            "out": H(g["out"], "state_out"),
        })
    return h


def _sym_step_hints_v2(ctx: CensusCtx, p: Params, shape: dict, first: bool = False) -> dict:
    """``first``: the unit takes the constant zero state (the VU's first unit): its acc_split hints are unused."""
    W = {g.name: _hint_widths(g.contract(p)) for g in V2_GADGETS}
    h: dict = {"groups": []}
    if "acc" in shape:
        h["acc"] = _sym_hints(ctx, p, shape["acc"], W["state_decode"], "state_decode")
    for i, g in enumerate(shape["groups"]):
        h["groups"].append({
            "E": _sym_hints(ctx, p, g["E"], W["group_max"], "group_max", "E"),
            "acc_split": None if first and i == 0 else _sym_hints(ctx, p, g["acc_split"], W["state_shift"], "state_shift"),
            "sum": _sym_hints(ctx, p, g["sum"], W["signed_sum"], "signed_sum"),
        })
    return h


def _shape_inputs(p: Params) -> tuple[int, list[int], list[int]]:
    k = sum(p.groups)
    return 0, [0] * k, [0] * k          # any finite words at any width: the hint *shape* is instance-independent


# ---------------------------------------------------------------------------------------------------------------------
# The two variants
# ---------------------------------------------------------------------------------------------------------------------

def census_unit(p: Params = REAL) -> UnitCensus:
    """One transition unit with an arbitrary (symbolic) incoming accumulator word, output identity included
    (the active ``CHECKER_VARIANT``)."""
    return census_unit_v2(p) if CHECKER_VARIANT == "v2" else census_unit_v1(p)


def census_vu(K: int, p: Params = REAL) -> UnitCensus:
    """The proof unit of the active ``CHECKER_VARIANT`` (v1: word-chained; v2: state-chained)."""
    return census_vu_v2(K, p) if CHECKER_VARIANT == "v2" else census_vu_v1(K, p)


def fixed_tables(p: Params = REAL) -> dict[str, tuple[int, int]]:
    """The fixed tables' (rows, columns including the key) of the active ``CHECKER_VARIANT``, plus the 16-bit range table."""
    return fixed_tables_v2(p) if CHECKER_VARIANT == "v2" else fixed_tables_v1(p)


def census_unit_v1(p: Params = REAL) -> UnitCensus:
    ctx = CensusCtx("unit")
    c0, a0, b0 = _shape_inputs(p)
    shape = gen_step(p, c0, a0, b0)
    h = _sym_step_hints(ctx, p, shape)
    c = ctx.input(p.acc_word_bits)
    a = [ctx.input(p.word_bits) for _ in a0]
    b = [ctx.input(p.word_bits) for _ in b0]
    y = ctx.input(p.acc_word_bits)
    with _sectioned(V1_GADGETS):
        got = check_step_expr(ctx, p, c, a, b, h)
    ctx.eq("y", got, y)
    return ctx.c


def census_unit_v2(p: Params = REAL) -> UnitCensus:
    """The v2 transition unit (word in, word out, symbolic ``c``), output identity included."""
    ctx = CensusCtx("unit-v2")
    c0, a0, b0 = _shape_inputs(p)
    shape = step2.gen_step(p, c0, a0, b0)
    h = _sym_step_hints_v2(ctx, p, shape)
    c = ctx.input(p.acc_word_bits)
    a = [ctx.input(p.word_bits) for _ in a0]
    b = [ctx.input(p.word_bits) for _ in b0]
    y = ctx.input(p.acc_word_bits)
    with _sectioned(V2_GADGETS):
        got = step2.check_step_expr(ctx, p, c, a, b, h)
    ctx.eq("y", got, y)
    return ctx.c


def census_vu_v2(K: int, p: Params = REAL) -> UnitCensus:
    """The v2 proof unit: K/16 state-chained units from the constant zero state, one Pack, the epilogue."""
    if K % sum(p.groups):
        raise ValueError("K must be a multiple of the unit's product count")
    ctx = CensusCtx(f"vu-v2(K={K})")
    _c0, a0, b0 = _shape_inputs(p)
    steps = K // sum(p.groups)
    shape = step2.gen_vu(p, a0 * steps, b0 * steps)
    h = {"steps": [_sym_step_hints_v2(ctx, p, s, first=(i == 0)) for i, s in enumerate(shape["steps"])],
         "epilogue": _sym_hints(ctx, p, shape["epilogue"], _hint_widths(Epilogue.contract(p)), "epilogue")}
    a = [ctx.input(p.word_bits) for _ in range(K)]
    b = [ctx.input(p.word_bits) for _ in range(K)]
    y16 = ctx.input(p.word_bits)
    with _sectioned(V2_GADGETS):
        got = step2.check_vu_expr(ctx, p, a, b, h)
    ctx.eq("y16", got, y16)
    return ctx.c


def fixed_tables_v2(p: Params = REAL) -> dict[str, tuple[int, int]]:
    """The v2 fixed tables' (rows, columns including the key) and the 16-bit range table (6 queries per unit)."""
    out = {t.name: (t.rows, 1 + len(t.outs)) for t in v2_tables(p)}
    out[f"T_range{RANGE_CHUNK}"] = (2**RANGE_CHUNK, 1)
    return out


def census_vu_v1(K: int, p: Params = REAL) -> UnitCensus:
    """The v1 proof unit: K/16 word-chained units from ``c_0 = 0`` plus the epilogue, as ``step.py`` composes it."""
    if K % sum(p.groups):
        raise ValueError("K must be a multiple of the unit's product count")
    ctx = CensusCtx(f"vu(K={K})")
    _c0, a0, b0 = _shape_inputs(p)
    steps = K // sum(p.groups)
    shape = gen_vu(p, a0 * steps, b0 * steps)
    h = {"steps": [_sym_step_hints(ctx, p, s) for s in shape["steps"]],
         "epilogue": _sym_hints(ctx, p, shape["epilogue"], _hint_widths(Epilogue.contract(p)), "epilogue")}
    a = [ctx.input(p.word_bits) for _ in range(K)]
    b = [ctx.input(p.word_bits) for _ in range(K)]
    y16 = ctx.input(p.word_bits)
    with _sectioned(V1_GADGETS):
        got = check_vu_expr(ctx, p, a, b, h)
    ctx.eq("y16", got, y16)
    return ctx.c


def fixed_tables_v1(p: Params = REAL) -> dict[str, tuple[int, int]]:
    """The v1 fixed tables' (rows, columns including the key) and the 16-bit range table."""
    out = {}
    for t in (pow_table(p), align_table(p), lead_table(p), norm_table(p)):
        out[t.name] = (t.rows, 1 + len(t.outs))
    out[f"T_range{RANGE_CHUNK}"] = (2**RANGE_CHUNK, 1)
    return out


def lookups_by_table(c: UnitCensus, consolidate_ranges: bool = True) -> dict[str, int]:
    """Queries per table.  ``consolidate_ranges``: every range check goes through the one 16-bit table (a w-bit value
    with w < 16 is checked as ``x * 2^(16-w)``, a constant coefficient); otherwise one table per width as in
    ``relation`` (``T_range{w}``)."""
    out = Counter(c.lookups)
    for w, n in c.range_by_width.items():
        full, rest = divmod(w, RANGE_CHUNK)
        out[f"T_range{RANGE_CHUNK}"] += full * n
        if rest:
            out[f"T_range{RANGE_CHUNK}" if consolidate_ranges else f"T_range{rest}"] += n
    return dict(sorted(out.items()))


def layer_shape(c: UnitCensus) -> list[dict]:
    """A-GKR's layered quad circuit (libzk 6.3: strictly layered, values used later are carried by copy gates)
    from the census, output layer first.  ESTIMATE of the widths: layer at multiplicative depth d holds its products
    plus every value of lower depth (inputs and earlier products), an upper bound on what must be carried; the
    assertion layer holds the linear identities.  Each entry: ``{"depth", "products", "width", "s_out", "s_in",
    "has_queries"}`` with ``width`` the carried count and ``s_*`` the log2 widths rounded up."""
    inputs = c.columns_committed_a
    depths = sorted(c.mul_by_depth)
    carried = {0: inputs}
    for d in depths:
        carried[d] = carried[d - 1] + c.mul_by_depth[d]
    log2 = lambda n: max(1, math.ceil(math.log2(max(2, n))))
    layers = [{"depth": "assert", "products": 0, "width": c.lin_total, "s_out": log2(c.lin_total),
               "s_in": log2(carried[depths[-1]]) if depths else log2(inputs), "has_queries": False}]
    for d in reversed(depths):
        layers.append({"depth": d, "products": c.mul_by_depth[d], "width": carried[d], "s_out": log2(carried[d]),
                       "s_in": log2(carried[d - 1]), "has_queries": c.query_depth.get(d, 0) > 0})
    return layers


def diff_vs_relation(K: int = 1536, relation_unit: dict | None = None, relation_vu: dict | None = None) -> dict:
    """Checker census beside ``verity.ml.circuits.silicon.relation.census``/``vu_census`` (Ampere pipeline) for the unit and the VU:
    mul, lin, lookup and the lookup tables; ``delta`` = checker - relation."""
    if relation_unit is None or relation_vu is None:
        from verity.ml.circuits.silicon.relation import census as rel_census
        from verity.ml.circuits.silicon.relation import vu_census as rel_vu_census
        from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16
        relation_unit = relation_unit or rel_census(AMPERE_BF16_M16N8K16)
        relation_vu = relation_vu or rel_vu_census(AMPERE_BF16_M16N8K16, K)
    u, v = census_unit(), census_vu(K)
    rows = {}
    for label, chk, rel in (("unit", u, relation_unit), ("vu", v, relation_vu["total"])):
        rows[label] = {
            "relation": {"mul": rel["mul"], "lin": rel["lin"], "lookup": rel["lookup"]},
            "checker": {"mul": chk.mul, "lin": chk.lin_total, "lookup": chk.lookup},
            "delta": {"mul": chk.mul - rel["mul"], "lin": chk.lin_total - rel["lin"], "lookup": chk.lookup - rel["lookup"]},
        }
    rows["unit"]["relation_tables"] = relation_unit["lookups_by_table"]
    rows["unit"]["checker_tables"] = lookups_by_table(u)
    rows["unit"]["checker_tables_per_width"] = lookups_by_table(u, consolidate_ranges=False)
    rows["unit"]["checker_columns"] = u.summary()["columns"]
    rows["vu"]["checker_columns"] = v.summary()["columns"]
    rows["not_in_relation"] = [
        "RANGE of every remainder (align_state.r, align_product.r, normalise.r) and of magW-equivalents (normalise.M)",
        "tight quotient bounds (align_state.quot, align_product.quot) as lt_pow products",
        "decode of an arbitrary incoming accumulator word (fp32_decode with hints s, t, f, nz, z) in every unit",
        "pack of the output word in every unit (the relation packs once per VU)",
        "operand decode by hints + range checks (no T_bf16 table); no T_pcls / T_exp / T_iszero tables",
    ]
    return rows


def _fmt(n: float) -> str:
    return f"{n:,}" if isinstance(n, int) else f"{n:,.1f}"


def main(argv: Sequence[str] | None = None) -> None:
    import json
    import sys

    K = int((argv or sys.argv[1:] or ["1536"])[0])
    print(json.dumps({"unit": census_unit_v1().summary(), "vu": census_vu_v1(K).summary(), "tables": fixed_tables_v1(),
                      "unit_v2": census_unit_v2().summary(), "vu_v2": census_vu_v2(K).summary(),
                      "tables_v2": fixed_tables_v2(), "diff": diff_vs_relation(K)}, indent=1))


if __name__ == "__main__":       # pragma: no cover
    main()
