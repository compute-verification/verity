"""The WP2-checker census (``security.census``): pinned counts per transition unit and per K = 1536 VU, the layered
shape A-GKR's accountant derives from it, and the diff against ``verity.ml.circuits.silicon.relation.vu_census``."""

from __future__ import annotations

import pytest
from verity_numerical.checker import REAL, TINY
from verity_numerical.security import accounting
from verity_numerical.security.census import (
    RANGE_CHUNK,
    CensusCtx,
    census_unit,
    census_vu,
    diff_vs_relation,
    fixed_tables,
    layer_shape,
    lookups_by_table,
)

UNIT = census_unit()
VU = census_vu(1536)


def test_unit_columns_from_contracts():
    """34 public words (c, 32 operands, y) + 225 hints + 62 looked-up row columns are what A-GKR commits."""
    c = UNIT.summary()["columns"]
    assert c["inputs"] == 34 and c["hints"] == 225 and c["lookup_outputs"] == 62
    assert c["committed_A"] == 321
    assert c["products"] == 244 and c["materialised"] == 100 and c["all"] == 665
    assert c["committed_B_estimate"] == 1064 == 3 * UNIT.mul + (UNIT.columns_all - UNIT.columns_in_mul)
    # hint widths come from contract().hints: 114 bits/flags/selectors, 32 sixteen-bit ... 4 twenty-five-bit q/r
    assert UNIT.hint_bits == {1: 114, 5: 2, 7: 34, 8: 33, 9: 2, 16: 32, 23: 2, 24: 2, 25: 4}
    # hints by gadget: fp32 5 + bf16 4 x 32 + product 16 + 2 groups x (11 + 2 + 16 + 2 + 1 + 3 + 2) + pack 2
    assert 5 + 4 * 32 + 16 + 2 * (11 + 2 + 16 + 2 + 1 + 3 + 2) + 2 == 225


def test_unit_predicates():
    assert UNIT.mul == 358 and UNIT.booleans == 114
    assert UNIT.lin == 115 and UNIT.lin_range_splits == 22 and UNIT.lin_total == 137
    assert UNIT.range_by_width == {7: 36, 8: 66, 9: 26, 16: 112, 23: 2, 24: 5, 25: 8, 30: 6, 32: 1}
    assert UNIT.range_lookups == 284 and UNIT.fixed_lookups == 22 and UNIT.lookup == 306
    assert UNIT.lookups == {"ALIGN": 16, "LEAD": 2, "NORM": 2, "POW": 2}
    # the repairs the relation lacks are inside these counts: remainders r (25-bit x2 per group, 16-bit x8 per
    # group, 7-bit x1 per group) are range-checked directly, and the tight quotient bounds are the lt_pow products
    assert UNIT.range_by_width[25] == 2 * 4        # align_state: q, r, r*ipow, q*pow per group
    assert UNIT.range_by_width[16] == 32 + 16 + 2 * 8 * 4   # operands, u+z-1, align_product q, r, r*isr, q*sr
    assert UNIT.mul_by_depth == {1: 212, 2: 114, 3: 22, 4: 4, 5: 6} and UNIT.depth == 5
    assert UNIT.queries_on_computed_wires == 140 and UNIT.max_query_depth == 5


def test_vu_is_word_chained_units_plus_epilogue():
    """``check_vu_expr`` decodes and packs the word in every unit; the VU census is 96 units + the epilogue."""
    steps = 1536 // 16
    assert VU.mul == steps * UNIT.mul + 2                   # epilogue booleans lsb, carry
    # first unit: c_0 = 0 is a constant (no 32-bit range check: -2 chunks); epilogue: hi, lo, hi2, rem, y16 ranges (+5)
    assert VU.lookup == steps * UNIT.lookup - 2 + 5
    assert VU.columns_hints == steps * UNIT.columns_hints + 6
    assert VU.columns_committed_a == 30_631 and VU.columns_committed_b == 102_059
    assert VU.lin_total == 10_948 + 2_111 == 13_059
    assert VU.summary()["columns"]["all"] == 63_751


def test_diff_vs_relation_shows_the_undercount():
    d = diff_vs_relation(1536)
    assert d["unit"]["relation"] == {"mul": 146, "lin": 104, "lookup": 172}
    assert d["unit"]["checker"] == {"mul": 358, "lin": 137, "lookup": 306}
    assert d["unit"]["delta"] == {"mul": 212, "lin": 33, "lookup": 134}
    assert d["vu"]["relation"] == {"mul": 14_027, "lin": 9_992, "lookup": 16_521}
    assert d["vu"]["checker"] == {"mul": 34_370, "lin": 13_059, "lookup": 29_379}
    assert d["vu"]["delta"] == {"mul": 20_343, "lin": 3_067, "lookup": 12_858}
    assert set(d["unit"]["relation_tables"]) >= {"T_bf16", "T_pcls", "T_exp25"}        # tables the checker does not use
    assert d["unit"]["checker_tables"] == {"ALIGN": 16, "LEAD": 2, "NORM": 2, "POW": 2, "T_range16": 284}
    assert len(d["not_in_relation"]) == 5


def test_tables_and_consolidation():
    assert fixed_tables() == {"POW": (512, 3), "ALIGN": (512, 4), "LEAD": (31, 3), "NORM": (32, 4), "T_range16": (65536, 1)}
    per_width = lookups_by_table(UNIT, consolidate_ranges=False)
    assert per_width["T_range16"] == 135 and per_width["T_range7"] == 38 and per_width["T_range8"] == 71
    assert sum(v for k, v in per_width.items() if k.startswith("T_range")) == 284
    assert RANGE_CHUNK == 16


def test_layer_shape_and_accountant_defaults():
    shape = layer_shape(UNIT)
    assert [L["depth"] for L in shape] == ["assert", 5, 4, 3, 2, 1]
    assert [L["width"] for L in shape] == [137, 679, 673, 669, 647, 533]
    assert [(L["s_out"], L["s_in"]) for L in shape] == [(8, 10), (10, 10), (10, 10), (10, 10), (10, 10), (10, 9)]
    assert all(L["has_queries"] for L in shape[1:]) and not shape[0]["has_queries"]
    # the accountant's defaults are these numbers
    assert accounting.CandidateA.inputs_per_unit == 321 and accounting.CandidateA.query_wires_per_unit == 140
    assert accounting.CandidateB.witness_per_unit == 1064
    assert accounting.CandidateB.linear_per_unit == 137 + 2 * 358 and accounting.CandidateB.quadratic_per_unit == 358
    assert [(t.name, t.rows, t.queries_per_unit, t.columns) for t in accounting.CHECKER_TABLES] == [
        ("ALIGN", 512, 16, 4), ("LEAD", 31, 2, 3), ("NORM", 32, 2, 4), ("POW", 512, 2, 3), ("T_range16", 65536, 284, 1)]
    assert sum(t.rows for t in accounting.CHECKER_TABLES) == 66_623            # multiplicity columns (was 162,715)


def test_census_is_instance_independent_and_width_parametric():
    assert census_unit(REAL).summary() == UNIT.summary()
    tiny = census_unit(TINY)
    assert 0 < tiny.mul < UNIT.mul and tiny.columns_hints < UNIT.columns_hints
    with pytest.raises(ValueError):
        census_vu(1000)


def test_ctx_counts_one_multiplication_per_witness_product_only():
    ctx = CensusCtx("t")
    x, y = ctx.hint(8), ctx.hint(8)
    _ = 3 * x + 2                    # constant coefficients: free
    _ = x * y                        # one product
    ctx.boolean("b", x)              # one product
    ctx.rng("r", x + y, 0, 2**17)    # a materialised 17-bit expression: 2 chunks + 1 split
    ctx.eq("e", x, y)
    assert ctx.c.mul == 2 and ctx.c.columns_materialised == 1 and ctx.c.range_lookups == 2 and ctx.c.lin_range_splits == 1
    assert ctx.c.lin == 1 and ctx.c.columns_products == 1
