"""Pinned properties of the privacy cost model (``security.zk_cost``) and the ``zk`` knob of ``explore.tensor``.
Everything here is a property of the stated cost model, not a measurement."""

from __future__ import annotations

import pytest
from verity_numerical.explore import tensor as t
from verity_numerical.security import accounting as acc
from verity_numerical.security import zk_cost as z


def test_hvzk_is_what_tensor_md_priced():
    """The default level reproduces tensor.md's rows (up to the PRG bucket, < 1%): the accountant's rows and pads
    already contained the honest-verifier layer."""
    for dn, field, pinned in (("A", "babybear", 1.07e7), ("B", "babybear", 1.43e7), ("A", "goldilocks", 1.25e7)):
        p = t.price_design(t.DESIGNS[dn], t.FIELDS[field], -128)          # zk="hvzk"
        tm = p.times(t.GPUS["H100"], "H100", 0.172, 50e9, fs=t.FIELDS[field])
        assert abs(tm["overhead.vs_native_peak"] / pinned - 1) < 0.01
        assert 0 < tm["t.prg"] < 0.01 * tm["t.total"]


def test_zk_levels_are_ordered_and_small():
    """none < hvzk <= malicious in time; the whole privacy layer is below 10% for every Ligero design (the
    '--breakthrough under 10%' criterion of the lane brief), and step 0 adds exactly one depth and < 0.1%."""
    for dn in ("A", "A-hintfree-depth630", "B"):
        for field in ("babybear", "goldilocks"):
            r = z.design_zk_rows(dn, field)
            lv = r["levels"]
            for uk in lv["none"]["times"]:
                t0, t1, t2 = (lv[k]["times"][uk]["t.total"] for k in t.ZK_LEVELS)
                assert t0 < t1 <= t2
                assert r["deltas"]["hvzk"]["times"][uk]["relative"] < 0.10
                assert (t2 - t1) / t1 < 1e-3
            assert r["deltas"]["malicious"]["rounds.sequential_depth"] == 1
            assert r["deltas"]["hvzk"]["rounds.sequential_depth"] == 0
            assert r["deltas"]["hvzk"]["ligero_rows"] > 3                    # k/l - 1 = 6.7% more rows + 3 blinding rows


def test_a_vs_b_hvzk_cost_shape():
    """A's privacy is cheaper than B's: A's rows are 26% of B's and its pads are 14k elements against 188M committed;
    B's is the Ligero payload loss (k/l) on an encoding/hashing-bound design (~7%)."""
    a = z.design_zk_rows("A", "babybear")["deltas"]["hvzk"]["times"]["H100@0.172"]["relative"]
    b = z.design_zk_rows("B", "babybear")["deltas"]["hvzk"]["times"]["H100@0.172"]["relative"]
    assert 0.005 < a < 0.03
    assert 0.06 < b < 0.08
    pa = t.price_design(t.DESIGNS["A"], t.FIELDS["babybear"], -128, zk="none")
    pb = t.price_design(t.DESIGNS["A"], t.FIELDS["babybear"], -128, zk="hvzk")
    assert 0 < pb.committed_elements - pa.committed_elements < 50_000                 # the pads (x extension degree)


def test_air_fri_zk_requirements():
    a = z.air_fri_zk("census_lookup", "goldilocks")
    req = a["zk_requirements"]
    assert req["padding_covers"] is True                                              # 131,072 padding rows >> 2 + 2 s
    assert req["random_rows_needed"] == 2 + 2 * a["fri"]["queries"]
    assert req["quotient_chunks_before"] == 4 and req["quotient_chunks_after"] == 8   # degree 5 -> 6 with the selector
    assert 0 < a["relative"]["committed"] < 0.02 and 0 < a["relative"]["hash"] < 0.02
    assert a["depth"]["fiat_shamir_plonky3"] == 1 and a["depth"]["interactive_malicious"] == a["depth"]["interactive_hvzk"] + 1
    assert 180 <= a["fri"]["queries"] <= 200                                            # (5/8)^s at rate 1/4, proven UDR


def test_fri_zk_requirements_function():
    r = acc.fri_zk_requirements(19, 393_216, 189, ood_points=2)
    assert r["padding_rows"] == 131_072 and r["padding_covers"]
    small = acc.fri_zk_requirements(10, 1000, 189, ood_points=2)
    assert not small["padding_covers"] and small["extra_trace_rows_if_not"] == 1024
    with pytest.raises(acc.AccountingError):
        acc.fri_zk_requirements(10, 2000, 1, 0)


def test_soundness_terms_leave_the_operating_point():
    """Coin commitments (R x 2^-160) and the libra rho term (instances / |F|) are far below the Ligero query term:
    2^-128 holds at Goldilocks^3 / BabyBear^6 with t = 191 for every ZK variant, and the padded-transcript default
    adds no soundness term at all."""
    s = z.soundness_with_zk()
    for key, rows in s.items():
        for name, r in rows.items():
            assert r["meets_2^-128"], (key, name)
            assert abs(r["total_log2"] + 129.5) < 0.2
    a = s["A/goldilocks^3/t=191"]
    assert a["hvzk"]["depth"] == 381 and a["malicious"]["depth"] == 382
    assert a["malicious+libra"]["pads"] > a["malicious"]["pads"]
    assert acc.zk_sumcheck_masking_error(0, acc.Field.prime("p", 97)) == 0


def test_libra_masking_option_pinned():
    GL = acc.Field.prime("GL", acc.GOLDILOCKS)
    Fc = GL.extension(3)
    lig = acc.LigeroParams(n=2**14, k=2**12, l=3840, t=192)

    def mk(**kw):
        return acc.account_candidate_a(acc.CandidateA(K=1536, B=64, trace_field=GL, challenge_field=Fc, ligero=lig,
                                                      hash_queries=2**60, **kw))

    base, lib = mk(), mk(sumcheck_masking="libra")
    assert base.parameters["masked_sumcheck_instances"] == lib.parameters["masked_sumcheck_instances"] > 6
    assert any("libra" in tm.name for tm in lib.terms) and not any("libra" in tm.name for tm in base.terms)
    assert lib.total() > base.total() and lib.meets(-128) and base.meets(-128)
    with pytest.raises(acc.AccountingError):
        mk(sumcheck_masking="cfs17")


def test_fiat_shamir_costs_parameters():
    fs = z.fiat_shamir_reparameterisation("goldilocks")
    assert fs["ligero_t_fiat_shamir"] > fs["ligero_t_interactive"]
    assert fs["extension_degree_fiat_shamir"] > fs["extension_degree_interactive"]
    assert fs["ligero_t_fiat_shamir"] > 2**12 - 3840                                  # does not fit the ZK payload


def test_report_shape_and_vole_row():
    rep = z.report()
    assert {s["design"] for s in rep["summary"]} == {"A", "A-hintfree-depth630", "B"}
    assert rep["vole"]["zk_delta"] == 0.0 and rep["vole"]["zk_mode"] == "designated-verifier"
    assert len(rep["air_fri"]) == 4
    with pytest.raises(ValueError):
        t.price_design(t.DESIGNS["A"], t.FIELDS["babybear"], -128, zk="perfect")
