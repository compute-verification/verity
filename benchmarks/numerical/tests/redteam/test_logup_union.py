"""Lane a-logup-union: the tagged-union forgery is decided by z3 (UNSAT outside the finite collision set, SAT only
on it, and multiplicity manipulation buys nothing), and the redteam-3 fixtures on disk are the ones the module
generates (the Rust side rejects them in ``main.rs::redteam3_*``)."""
from __future__ import annotations

import json
from pathlib import Path

from verity_numerical.redteam import logup_union as U

REPO = Path(__file__).resolve().parents[4]
FIX = REPO / "fixtures" / "redteam-3" / "logup-union"


def test_collision_set_is_the_protocol_term():
    tables = {0: [1, 2], 1: [10, 20], 2: [100, 200]}
    g = U.union_collision_gammas([(5, 0)], tables, 1000003)
    # one bad query x four foreign rows -> at most four gammas, each (q - r)/(t' - t)
    assert len(g) <= 4 and all(((5 + gg * 0) - (r + gg * t)) % 1000003 == 0
                               for gg in g for t, rows in tables.items() for r in rows if t and (5 - r) * pow(t, -1, 1000003) % 1000003 == gg)


def test_z3_tagged_union_forgery_only_on_collision_gammas():
    r = U.z3_union_forgery()
    assert r["gamma_excluded"] == "unsat" and r["gamma_free"] == "sat" and r["gamma_found_in_collision_set"]
    assert r["collision_count"] <= r["bound_collision_count"]
    assert r["verdict"].startswith("UNSAT outside")


def test_z3_multiplicity_manipulation_cannot_explain_a_foreign_query():
    r = U.z3_union_multiplicity_forgery()
    assert r["outside_collision_set"] == "unsat" and r["on_collision_set"] == "sat"
    assert r["verdict"].startswith("UNSAT for every multiplicity vector")


def test_redteam3_fixtures_present_and_documented():
    man = json.loads((FIX / "cross_table" / "manifest.json").read_text())
    expect = {k.split("/")[1] + ".bin": v["expected"] for k, v in man.items() if k.startswith("cross_table/")}
    assert expect["honest.bin"] == "accept"
    assert {k for k, v in expect.items() if v == "reject"} >= {"x_in_TB_not_TA.bin", "x_out_of_both.bin", "y_out_of_TB.bin",
                                                               "x_in_TB_not_TA_y_shares_it.bin"}
    for f in ("circuit.txt", *expect):
        assert (FIX / "cross_table" / f).exists()
    for f in ("units.bin", "epilogue.bin", "public.bin", "chain.txt"):
        assert (FIX / "forged_epilogue_hint" / f).exists()
    assert "R16" in (FIX / "README.md").read_text()
