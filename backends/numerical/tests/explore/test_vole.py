"""Pins the VOLE-ZK cost model's numbers (they follow from the census and the cited anchors; a change here means the
census or the model changed) and checks that every contract-prefixed column is a contract name."""

from __future__ import annotations

import pytest
from verity_numerical.bench import contract
from verity_numerical.explore import vole
from verity_numerical.security import census


@pytest.fixture(scope="module")
def unit():
    return census.census_unit()


@pytest.fixture(scope="module")
def vu():
    return census.census_vu(1536)


def test_unit_embedding_logup(unit):
    e = vole.embed(unit, "logup")
    assert (e.inputs, e.product_outputs, e.helpers) == (321, 244, 306)
    assert e.vole == 871 and e.mul_checks == 358 + 306 == 664
    assert e.lin_free == 137 and e.lookups == 306
    assert e.range_bits == 3384
    assert e.p2v_bytes == 871 * 8


def test_unit_embedding_rom_and_bits(unit):
    rom, bits = vole.embed(unit, "rom"), vole.embed(unit, "bits")
    assert rom.vole == 871 and rom.mul_checks == 358 + 2 * 306
    assert bits.helpers == 3384 + 22 and bits.vole == 321 + 244 + 3406 == 3971
    assert bits.mul_checks == 358 + 3406
    with pytest.raises(ValueError):
        vole.embed(unit, "plookup")


def test_vu_embedding(vu):
    e = vole.embed(vu, "logup")
    assert e.vole == 83_434 and e.mul_checks == 63_749 and e.lin_free == 13_059
    assert e.p2v_bytes == 667_472
    assert e.range_bits == 324_911


def test_first_extension_matches_candidates_pattern():
    """r=3 (~2^183 MACs) first meets 2^-128 at every B; r=2 meets 2^-80 but not 2^-96 at B=4096 -- the same
    quadratic/cubic split as the Goldilocks candidates."""
    for B in (1, 64, 4096):
        r, bud = vole.first_extension(-128, vole._make(1536, B, "logup", "quicksilver_arith_1t"))
        assert r == 3 and bud.meets(-128)
        r80, _ = vole.first_extension(-80, vole._make(1536, B, "logup", "quicksilver_arith_1t"))
        assert r80 == 2
    r96, _ = vole.first_extension(-96, vole._make(1536, 4096, "logup", "quicksilver_arith_1t"))
    assert r96 == 3
    with pytest.raises(vole.accounting.AccountingError):
        vole.first_extension(-128, vole._make(1536, 4096, "logup", "quicksilver_arith_1t"), degrees=(1, 2))


def test_budget_terms_at_b4096():
    bud = vole.account_vole(vole.VoleParams(1536, 4096, 3))
    comps = {c: round(vole.log2(v), 1) for c, v in bud.by_component().items() if v}
    assert comps["batching"] == -155.0 and comps["lookup"] == -155.9 and comps["commitment_hash"] == -136.0
    assert bud.parameters["mul_checks"] == 63_749 * 4096
    assert round(vole.log2(vole.account_vole(vole.VoleParams(1536, 4096, 2)).total()), 1) == -93.4


def test_census_columns_b4096():
    est = vole.census_estimate_vole(vole.VoleParams(1536, 4096, 3))
    assert est["vole.correlations"] == 83_434 * 4096 + 66_623 == 341_812_287
    assert est["vole.mul_checks"] == 63_749 * 4096
    assert est["census.committed_elements"] == est["vole.correlations"]
    assert 2.73e9 < est["transcript.bytes"] < 2.74e9            # 37x A-GKR (74.7 MB), 8.6x B-Ligero (318 MB)
    assert est["rounds.sequential_depth"] == 2 and est["rounds.protocol"] == 2
    assert est["census.hash_bytes"] == 0
    assert 200 < est["estimate.prover_seconds"] < 240              # 1 thread, r=3: 602M gate-eq / 4.8M/s x 3.5
    assert est["t.e2e.rtt_50ms"] - est["t.e2e.rtt_0ms"] == pytest.approx(2 * 0.05)
    assert est["estimate.transfer_seconds_1gbps"] == pytest.approx(est["transcript.bytes"] / 125e6)


def test_census_columns_b1_and_bits_and_antman():
    b1 = vole.census_estimate_vole(vole.VoleParams(1536, 1, 3))
    assert b1["vole.correlations"] == 83_434 + 66_623            # multiplicities dominate a single VU
    assert 0.07 < b1["estimate.prover_seconds"] < 0.09
    bits = vole.census_estimate_vole(vole.VoleParams(1536, 4096, 3, range_mode="bits"))
    assert 12.4e9 < bits["transcript.bytes"] < 12.6e9 and bits["rounds.sequential_depth"] == 1
    ant = vole.census_estimate_vole(vole.VoleParams(1536, 4096, 3, anchor="antman_simd_16t", simd=True))
    assert 20e6 < ant["transcript.bytes"] < 22e6                  # below A's 74.7 MB; B-independent past B ~ 13
    ant64 = vole.census_estimate_vole(vole.VoleParams(1536, 64, 3, anchor="antman_simd_16t", simd=True))
    assert ant64["transcript.bytes"] == ant["transcript.bytes"]


def test_column_names_follow_the_contract():
    est = vole.census_estimate_vole(vole.VoleParams(1536, 64, 3))
    reserved = ("t.", "vu.", "mem.", "fixed.", "bytes.", "transcript.", "verifier.", "rounds.", "census.")
    for name, value in est.items():
        if name.startswith(reserved):
            assert name in contract.UNITS, name
            contract.measurement(name, value)                    # unit supplied and checked
        else:
            assert name.startswith(("vole.", "estimate.")), name
    for col in ("census.witness_elements", "census.committed_elements", "census.bytes_encoded", "census.hash_bytes",
                "census.field_ops", "census.memory_passes", "census.bytes_moved", "rounds.protocol",
                "rounds.sequential_depth", "transcript.bytes", "verifier.seconds",
                *(f"t.e2e.rtt_{r}ms" for r in ("0", "0.1", "1", "10", "50"))):
        assert col in est


def test_anchors_cite_paper_table_hardware():
    for a in vole.ANCHORS.values():
        assert a.paper and a.where and a.hardware and a.url.startswith("https://eprint.iacr.org/")
        assert a.rate_per_s > 0
    assert vole.ANCHORS["quicksilver_arith_1t"].rate_per_s == 4.8e6
    assert vole.ANCHORS["quicksilver_bool_1t"].rate_per_s == 7.6e6


@pytest.mark.slow
def test_table_and_report_shape():
    rows = vole.table(1536, batches=(1, 64), targets=(-128,))
    assert [r["B"] for r in rows] == [1, 64] and all(r["mac_extension"] == 3 for r in rows)
    rep = vole.report(1536)
    assert set(rep["tables"]) >= set(vole.RANGE_MODES)
    assert rep["comparison_B4096"]["vole_logup_transcript_bytes"] > rep["comparison_B4096"]["B_transcript_bytes"]
    assert rep["comparison_B4096"]["vole_antman_transcript_bytes"] < rep["comparison_B4096"]["A_transcript_bytes"]
    assert "designated-verifier" in rep["proof_class"]
