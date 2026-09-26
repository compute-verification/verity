"""Alternative-field cost models (``explore.fields``): the Goldilocks baseline is reproduced from the census, the
small-prime re-derivation under the wrap rule and the binary-tower GF(2) census are pinned, and every row carries
the contract's column names."""

from __future__ import annotations

import pytest
from verity_numerical.bench import contract
from verity_numerical.explore import fields as f
from verity_numerical.security import accounting as acc
from verity_numerical.security.census import census_unit

UNIT = census_unit()


def test_goldilocks_reproduces_the_census_and_the_baseline():
    sfc = f.small_field_census(f.GOLDILOCKS.trace)
    assert sfc.deltas == [] and sfc.lt_pow_direct == {}
    assert (sfc.mul, sfc.lin_total, sfc.lookup) == (UNIT.mul, UNIT.lin_total, UNIT.lookup) == (358, 137, 306)
    assert sfc.lt_pow_by_width == {7: 2, 16: 32, 25: 4, 30: 2}          # 40 lt_pow per unit
    assert sfc.columns_committed_a == 321 and sfc.columns_committed_b == 1064
    a = f.prime_row(f.GOLDILOCKS, -128, "A", B=4096)
    b = f.prime_row(f.GOLDILOCKS, -128, "B", B=4096)
    assert a["census.committed_elements"] == 181_353_221 and b["census.committed_elements"] == 779_620_604
    assert a["soundness"]["extension_degree"] == 3 and a["soundness"]["opened_columns"] == 191
    assert a["rounds.sequential_depth"] == 381 and b["rounds.sequential_depth"] == 3


def test_small_prime_wrap_rule_direct_lt_pow():
    """F1: every lt_pow whose product reaches 2^(2B) >= p is realised directly: 38 of 40 (B = 16, 25, 30; B = 7 stays)."""
    m = f.small_field_census(f.MERSENNE31.trace)
    assert m.lt_pow_direct == {16: 32, 25: 4, 30: 2}
    assert m.base.mul == 358 - 38                                        # the direct realisation drops the product
    assert m.base.lookup == UNIT.lookup                                  # and keeps the B-bit range check
    assert m.base.lin_total == UNIT.lin_total


def test_small_prime_repairs_and_totals():
    m = f.small_field_census(f.MERSENNE31.trace)
    bb = f.small_field_census(f.BABYBEAR.trace)
    assert [d.note[:2] for d in m.deltas] == ["F3", "F4"]
    assert [d.note[:2] for d in bb.deltas] == ["F3", "F4", "si"]         # BabyBear also tightens the signed sum
    # F3 per group: 5 hints, 5 ranges, 4 lin, 2 mul; F4 per unit: 2 hints, 2 inputs, 3 lin, 3 mul
    assert (m.mul, m.lin_total, m.lookup) == (358 - 38 + 2 * 2 + 3, 145, 316)
    assert (bb.mul, bb.lin_total, bb.lookup) == (327, 147, 320)
    assert m.extra_hints == 2 * 5 + 2 == 12                              # F3 per group (M0, M1, v0, v1, carry) + F4 (f_hi, low_hi)
    assert m.extra_inputs == 2 and m.columns_committed_a == 321 + 14 and bb.columns_committed_a == 335
    assert 32 not in m.range_by_width                                    # the word is two 16-bit limbs
    assert m.range_by_width[16] == 116 and m.range_by_width[30] == 6 and bb.range_by_width[30] == 8
    assert m.range_by_width[12] == 4 and m.range_by_width[15] == 4 and m.range_by_width[8] == 68
    assert m.columns_committed_b == 1110 and bb.columns_committed_b == 1112
    assert {t.name for t in m.tables()} == {"ALIGN", "LEAD", "NORM", "POW", "T_range16"}
    assert next(t for t in m.tables() if t.name == "T_range16").queries_per_unit == m.range_lookups


@pytest.mark.parametrize("opt", [f.MERSENNE31, f.BABYBEAR])
def test_small_prime_extension_degree_per_operating_point(opt):
    """2^-80: degree 4 (2^124) at every B.  2^-128: degree 5 (2^155) at B = 1 and 64, degree 6 at B = 4096 -- the
    T_range16 identity term (N = 290 x 393,216) alone is 2^-127.9 in 2^155."""
    for B in (1, 64, 4096):
        for cand in ("A", "B"):
            d80, _, _ = f.first_small_field_configuration(-80, opt, cand, B=B)
            assert d80 == 4, (opt.name, B, cand)
            d128, t, bud = f.first_small_field_configuration(-128, opt, cand, B=B)
            assert d128 == (6 if B == 4096 else 5) and t == 191, (opt.name, B, cand, d128)
            assert bud.meets(-128)
    with pytest.raises(acc.AccountingError, match="lookup"):
        f.first_small_field_configuration(-128, opt, "A", B=4096, extension_degrees=(5,))
    # two lookup repetitions rescue degree 5 for A-GKR only: B's powers-combination term floors at 2^-126.6
    d, _, _ = f.first_small_field_configuration(-128, opt, "A", B=4096, extension_degrees=(5,), lookup_repetitions=2)
    assert d == 5
    with pytest.raises(acc.AccountingError):
        f.first_small_field_configuration(-128, opt, "B", B=4096, extension_degrees=(5,), lookup_repetitions=2)
    with pytest.raises(acc.AccountingError):
        f.first_small_field_configuration(-128, opt, "A", B=4096, extension_degrees=(4,), lookup_repetitions=2)


def test_small_prime_rows_halve_bytes_and_raise_field_ops():
    g = f.prime_row(f.GOLDILOCKS, -128, "A", B=4096)
    m = f.prime_row(f.MERSENNE31, -128, "A", B=4096)
    assert m["soundness"]["extension_degree"] == 6
    assert m["estimate.committed_bytes"] == m["census.committed_elements"] * 4
    assert m["census.bytes_encoded"] == pytest.approx(g["census.bytes_encoded"] * (187_265_483 / 181_353_221) / 2, rel=0.01)
    assert m["census.field_ops"] > 2.5 * g["census.field_ops"]           # degree-6 extension: 17 vs 6 base mults
    assert m["estimate.field_ops_goldilocks_equiv"] > 1.3 * g["census.field_ops"]


def test_binary_tower_census_per_unit():
    """GF(2) semantics: 40 lt_pow and every power-of-two range on a column are free, booleans are free; the integer
    identities become 712 ripple-carry adders whose carries are half of all committed bits."""
    base, bc = f.binary_census()
    assert (base.mul, base.lin_total, base.lookup) == (358 - 40, 137 - 6, 306 - 46)   # what the prime census would say
    s = bc.summary()
    assert s["bits"] == {"inputs": 576, "hints": 1350, "lookup_outputs": 1028, "products": 6511, "materialised": 2244,
                         "carries": 11534, "committed": 23243}
    assert bc.adders == 712 and bc.bits_carries == 11_534 and bc.ands_adders == 3 * 11_534
    assert bc.ands == 43_592 and bc.lookups == 40 and bc.lookups_mul_table == 18
    assert bc.range_free == 123 and bc.range_compare == 139 and bc.booleans_free == 114 and bc.lt_pow_direct == 40
    assert bc.bits_carries / bc.bits_committed == pytest.approx(0.496, abs=0.005)
    assert bc.bits_committed > 321 * 64                                  # more bits than Goldilocks A commits


def test_binary_tower_soundness_and_row():
    r = f.binary_row(-128, B=4096)
    s = r["soundness"]
    assert s["reachable"] is False and -105 < s["total_log2"] < -104 and s["meets_-80"] is True
    assert "char(F) > total lookups" in s["logup_refused"]
    assert r["census.committed_elements"] == 4096 * 96 * 23_243 + 16 * (512 + 512 + 31 + 32 + 65536)
    assert r["carry_cost_per_unit"]["carry_bits"] == 11_534
    assert f.binary_row(-80, B=1)["soundness"]["reachable"] is True


def test_rows_use_contract_column_names():
    rows = f.compare(B_values=(1,), targets=(-80,))["rows"]
    assert len(rows) == 2 * len(f.OPTIONS) + 1
    for r in rows:
        for name in r:
            if name.startswith(("census.", "rounds.", "transcript.", "t.")):
                assert name in contract.UNITS, name
        assert {"census.committed_elements", "census.bytes_encoded", "census.field_ops", "rounds.sequential_depth",
                "transcript.bytes", "estimate.peak_bytes"} <= set(r)


def test_report_shape():
    rep = f.report()
    assert rep["meta"]["goldilocks_baseline_reproduced"] == {
        "A_committed_elements_B4096_2^-128": 181_353_221, "B_committed_elements_B4096_2^-128": 779_620_604}
    assert set(rep) >= {"rows", "unit_census", "citations", "assumptions", "what_would_kill_each_option",
                        "carry_cost_binary_per_unit", "small_field_repairs"}
    assert all("url" in c for c in rep["citations"].values())
