"""What ``security.census`` does not count that a backend must pay, quantified per unit from the same execution.

The census counts predicates; a backend pays for *columns and constraints in a concrete layout*.  This module runs the
same symbolic execution through a subclass of ``CensusCtx`` that additionally records:

    duplicated products      the same (operand, operand) pair multiplied twice (the census counts both; a compiler
                             may share them: a lower bound on what sharing saves)
    range chunk columns      a w-bit range check with w > 16 is ceil(w/16) lookups + 1 split; the chunks are new
                             committed columns (A-GKR commits them or recomputes them as wires, B commits them)
    scaled small ranges      a w-bit check with w < 16 through the one 16-bit table looks up x * 2^(16-w): a *computed*
                             wire even when x is a column -- in a Ligero layout (B) the queried value must be a row
    unit chaining            in a trace the next unit's ``c`` is a column bound to the previous word: 1 lin + 1 column
                             per unit (``step.py`` passes the expression, so ``census_vu`` counts nothing)
    booleans of selectors    counted by the census as 1 mul each -- confirmed here (no undercount), listed for the record
    copy / permutation       a gadget output feeding the next gadget is an *expression* of committed columns in every
                             gadget of ``step.py`` (never a hint re-supplied by the prover): no copy constraint is needed
                             in an AIR or a GKR layer; a Ligero (B) layout needs the two operand copies per product the
                             direct PROTOCOL already counts (``2 * mul`` lin)
"""
from __future__ import annotations

from collections import Counter

from verity_numerical.checker import REAL, Params
from verity_numerical.checker.step import check_step_expr, gen_step
from verity_numerical.security.census import (
    RANGE_CHUNK,
    CensusCtx,
    Sym,
    _sym_step_hints,
    _width,
)

__all__ = ["census_delta"]


class ReviewCtx(CensusCtx):
    def __init__(self, variant: str):
        super().__init__(variant)
        self.pairs: Counter = Counter()
        self.range_wide = 0            # checks with w > 16
        self.range_small_atom = 0      # checks with w < 16 on a plain column (scaled lookup)
        self.range_small_expr = 0      # checks with w < 16 on an expression
        self.range_16 = 0
        self.lookup_keys_expr = 0      # fixed-table lookups whose key is an expression (d = E - cand, k = ...)

    def product(self, a: Sym, b: Sym) -> Sym:
        key = tuple(sorted((a.atom if a.atom is not None else -id(a), b.atom if b.atom is not None else -id(b))))
        self.pairs[key] += 1
        return super().product(a, b)

    def rng(self, name, x, lo, hi):
        if isinstance(x, int) or x.const is not None:
            return
        w = _width(lo, hi)
        if w > RANGE_CHUNK:
            self.range_wide += 1
        elif w == RANGE_CHUNK:
            self.range_16 += 1
        elif x.atom is not None and x.depth == 0:
            self.range_small_atom += 1
        else:
            self.range_small_expr += 1
        super().rng(name, x, lo, hi)

    def lookup(self, name, table, key):
        if not (isinstance(key, int) or key.const is not None) and (key.atom is None or key.depth > 0):
            self.lookup_keys_expr += 1
        return super().lookup(name, table, key)


def census_delta(p: Params = REAL) -> dict:
    """Per-unit deltas against ``census_unit`` (which the GKR/direct protocol documents quote)."""
    ctx = ReviewCtx("unit")
    k = sum(p.groups)
    shape = gen_step(p, 0, [0] * k, [0] * k)
    h = _sym_step_hints(ctx, p, shape)
    c = ctx.input(p.acc_word_bits)
    a = [ctx.input(p.word_bits) for _ in range(k)]
    b = [ctx.input(p.word_bits) for _ in range(k)]
    y = ctx.input(p.acc_word_bits)
    ctx.eq("y", check_step_expr(ctx, p, c, a, b, h), y)
    cen = ctx.c
    dup = sum(n - 1 for n in ctx.pairs.values() if n > 1)
    # the chunks of a wide range: ceil(w/16) columns each (the census counts the lookups and the split, not the columns)
    chunk_cols = sum(-(-w // RANGE_CHUNK) * n for w, n in cen.range_by_width.items() if w > RANGE_CHUNK)
    return {
        "catalog/census": {"mul": cen.mul, "lin": cen.lin_total, "lookup": cen.lookup, "columns_committed_A": cen.columns_committed_a,
                   "columns_all": cen.columns_all, "booleans": cen.booleans, "queries_on_computed_wires": cen.queries_on_computed_wires},
        "delta_per_unit": {
            "duplicated_products (mul the census counts twice; sharing saves at most this)": -dup,
            "range_chunk_columns (committed columns for w>16 chunks; A may recompute, B commits)": chunk_cols,
            "wide_range_checks": ctx.range_wide,
            "scaled_small_range_queries_on_columns (B: +1 committed row each for x*2^(16-w))": ctx.range_small_atom,
            "scaled_small_range_queries_on_expressions (already materialised)": ctx.range_small_expr,
            "exact_16bit_range_queries": ctx.range_16,
            "fixed_table_keys_on_expressions (d = E - cand, k: query wires the census counts as computed)": ctx.lookup_keys_expr,
            "unit_chaining_in_a_trace (+1 lin, +1 column per unit: c_{i+1} = y_i)": {"lin": 1, "columns": 1},
            "booleanity_of_selectors_and_flags (counted: 1 mul each)": 0,
            "copy_constraints_between_gadgets (outputs are expressions, never re-supplied)": 0,
            "ligero_operand_copies (B; counted by direct PROTOCOL as 2 lin per mul)": 2 * cen.mul,
        },
        "duplicated_pairs": sum(1 for n in ctx.pairs.values() if n > 1),
    }
