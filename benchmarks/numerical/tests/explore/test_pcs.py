"""Pins for the PCS cost model: column names, the objects taken from the calculator, and the headline numbers."""
from __future__ import annotations

from fractions import Fraction

import pytest
from verity_numerical.bench import contract
from verity_numerical.explore import pcs
from verity_numerical.security.accounting import GOLDILOCKS, AccountingError, Field

GL = Field.prime("GL", GOLDILOCKS)


@pytest.fixture(scope="module")
def objs128():
    return pcs.committed_objects(-128)


@pytest.fixture(scope="module")
def rows128(objs128):
    return pcs.table(-128, objects=objs128)


def test_objects_come_from_the_calculator(objs128):
    a, b = objs128["A"], objs128["B"]
    assert a.elements == 181_353_221 and b.elements == 779_620_604          # PROTOCOL.md 9.4: 181M / 780M
    assert a.ext == b.ext == 3 and a.ligero.t == b.ligero.t == 191          # first configuration at 2^-128
    assert a.other_depth == 379 and b.other_depth == 0                      # 381 - 2 Ligero rounds; 3 - 3
    o80 = pcs.committed_objects(-80)
    assert o80["A"].ext == 2 and o80["A"].ligero.t == 120 and o80["B"].elements == 659_229_885


def test_census_columns_are_contract_names(rows128):
    for r in rows128:
        for k in r.values:
            if k.startswith(("census.", "rounds.")):
                assert k in contract.UNITS, k
    for k in ("proof_bytes", "verifier_hashes", "verifier_field_ops", "rounds.sequential_depth", "total.sequential_depth"):
        assert k in pcs.COLUMNS


def test_ligero_rows_reproduce_the_protocol_numbers(objs128):
    a = pcs.ligero_row(objs128["A"]).values
    b = pcs.ligero_row(objs128["B"]).values
    assert a["census.bytes_encoded"] == 47_231 * 2**14 * 8 and 6.1e9 < a["census.bytes_encoded"] < 6.3e9
    assert 26.5e9 < b["census.bytes_encoded"] < 26.7e9
    assert a["rounds.sequential_depth"] == 2 and a["total.sequential_depth"] == 381
    assert b["rounds.sequential_depth"] == 3 and b["total.sequential_depth"] == 3
    assert 73e6 < a["proof_bytes"] < 74e6 and 310e6 < b["proof_bytes"] < 320e6
    assert a["queries"] == 191 and -130 < a["soundness.log2"] < -129


def test_fri_query_counts_per_regime():
    Fc = GL.extension(3)
    n = 2**22
    s_udr, _, _ = pcs.fri_queries(-128, Fraction(1, 4), "proven_udr", n, Fc, applications=21)
    s_ldr, m, _ = pcs.fri_queries(-128, Fraction(1, 4), "proven_ldr", n, Fc, applications=21)
    s_conj, _, _ = pcs.fri_queries(-128, Fraction(1, 4), "conjectured", n, Fc, applications=21)
    assert s_udr == 189 and s_ldr == 166 and m == 3 and s_conj == 64
    assert pcs.fri_queries(-128, Fraction(1, 2), "conjectured", n, Fc, 21)[0] == 128
    assert pcs.fri_queries(-128, Fraction(1, 8), "conjectured", n, Fc, 21)[0] == 43
    # the proven list-decoding term n^2 / |F| does not fit in GL^3 at rate 1/8 and n = 2^23
    with pytest.raises(AccountingError):
        pcs.fri_queries(-128, Fraction(1, 8), "proven_ldr", 2**23, Fc, applications=21)
    assert pcs.fri_queries(-128, Fraction(1, 8), "proven_ldr", 2**23, GL.extension(4), 21)[0] == 86


def test_fri_rows_headline(objs128):
    a = pcs.fri_row(objs128["A"], -128, Fraction(1, 2), "conjectured").values
    assert a["params"].startswith("rows=2^20 cols=173 n=2^21")
    assert a["queries"] == 128 and a["rounds.sequential_depth"] == 22 and a["total.sequential_depth"] == 401
    assert 2.8e9 < a["census.bytes_encoded"] < 3.0e9 and 1.1e6 < a["proof_bytes"] < 1.3e6
    assert a["adapter.sequential_depth"] == 0
    b = pcs.fri_row(objs128["B"], -128, Fraction(1, 2), "conjectured").values
    assert b["params"].startswith("rows=2^20 cols=744")
    assert b["adapter.sequential_depth"] == 31 and b["total.sequential_depth"] == 52     # B loses its depth-3 advantage
    assert 12e9 < b["census.bytes_encoded"] < 13e9
    # the committed-size ratio survives every PCS: encoding is linear in the elements
    assert 4.2 < b["census.bytes_encoded"] / a["census.bytes_encoded"] < 4.4


def test_stir_whir_fewer_verifier_hashes_but_deeper(objs128):
    a = objs128["A"]
    fri = pcs.fri_row(a, -128, Fraction(1, 4), "conjectured").values
    stir = pcs.stir_row(a, -128, Fraction(1, 4), "conjectured").values
    whir = pcs.whir_row(a, -128, Fraction(1, 4), "conjectured").values
    assert stir["verifier_hashes"] < fri["verifier_hashes"] / 3 and stir["proof_bytes"] < fri["proof_bytes"] / 2
    assert stir["rounds.sequential_depth"] == 16 and whir["rounds.sequential_depth"] == 31 > fri["rounds.sequential_depth"] == 22
    assert whir["queries"] == stir["queries"] == 132


def test_basefold_and_brakedown(objs128):
    a = objs128["A"]
    bf = pcs.basefold_row(a, -128, Fraction(1, 2), "proven_udr").values
    fri = pcs.fri_row(a, -128, Fraction(1, 2), "proven_udr").values
    assert bf["queries"] == fri["queries"] == 309 and bf["census.field_ops"] > fri["census.field_ops"]
    bd = pcs.brakedown_row(a, -128).values
    assert bd["queries"] == 3788 and bd["rounds.sequential_depth"] == 2
    assert bd["census.field_ops"] < fri["census.field_ops"]                # linear-time encoding, no NTT
    assert bd["proof_bytes"] > 10 * fri["proof_bytes"]                    # and much larger proofs


def test_table_shape_and_raised_extensions(rows128):
    assert len(rows128) == 2 * (1 + 1 + 3 * 3 * 4 + 1)
    raised = [r for r in rows128 if r.values.get("ext_raised")]
    assert raised and all(r["regime"].startswith("proven_ldr") for r in raised)
    assert not [r for r in rows128 if str(r.values.get("params", "")).startswith("UNREACHABLE")]
    for r in rows128:
        if r.values.get("soundness.log2") is not None:
            assert r["soundness.log2"] <= -128 + 1e-9, (r["pcs"], r["regime"], r["soundness.log2"])
