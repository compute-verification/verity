"""Wave 2: the WP2 checker contracts under the harness (fast subset of ``campaign --target checker``)."""
from __future__ import annotations

import random

import pytest
from verity_numerical.checker import REAL
from verity_numerical.checker import search as S
from verity_numerical.redteam import census_review as CR
from verity_numerical.redteam import checker_target as CT
from verity_numerical.redteam import harness as H
from verity_numerical.redteam import z3_checker as ZC

CASES = CT.step_cases(4)


@pytest.mark.parametrize("field", [None, CT.GOLDILOCKS])
def test_checker_unit_completeness_and_mutations(field) -> None:
    t = CT.bound_checker_target(REAL, field)
    assert H.run_completeness(t, CASES).status == "PASS"
    f = H.run_mutations(t, CASES[:6], rng=random.Random(0))
    assert f.status == "PASS", f.detail
    assert H.run_coordinated(t, CASES[:4], rounds=40, rng=random.Random(1)).status == "PASS"


def test_checker_columns_cover_every_hint() -> None:
    from verity_numerical.checker.search import _flatten

    x = CASES[0][1]
    h = CT.bound_checker_target().gen(x)
    names = {c.name for c in CT.checker_columns(REAL, h)}
    leaves = {path.replace("[", ".").replace("]", "") for path, _ in _flatten(h)}
    assert leaves == names


def test_field_ctx_reads_the_word_mod_p() -> None:
    """F4 in the harness: over F_(2^31-1) the honest witness also proves y = 0x7fffffff for a zero result."""
    t = CT.bound_checker_target(REAL, CT.M31)
    x = dict(CASES)["all_zero"]
    h = t.gen(x)
    assert t.check(x, 0, h) and t.check(x, 0x7FFFFFFF, h)
    assert not CT.bound_checker_target(REAL, CT.GOLDILOCKS).check(x, 0x7FFFFFFF, h)


def test_vu_target() -> None:
    from verity_numerical.redteam import cases as C

    golden, _ = C.golden_cases()
    x = (list(golden[0][1][1]) + list(golden[3][1][1]), list(golden[0][1][2]) + list(golden[3][1][2]))
    tv = CT.checker_vu_target()
    assert H.run_completeness(tv, [("vu2", x)]).status == "PASS"
    assert H.run_mutations(tv, [("vu2", x)], limit_per_case=40).status == "PASS"


def test_u3_signed_sum_over_m31_is_unsat_with_the_boolean_split() -> None:
    r = ZC.gadget_split(S.SPECS["signed_sum"], REAL, modulus=S.M31, timeout_ms=20_000)
    assert r["status"] == "unsat" and len(r["cases"]) == 4


def test_split_finds_the_known_field_wraps() -> None:
    r = ZC.gadget_split(S.SPECS["leading_bit"], REAL, modulus=S.P40, timeout_ms=20_000)
    assert r["status"] == "sat"
    assert ZC.gadget_split(S.SPECS["leading_bit"], REAL, modulus=S.GOLDILOCKS, timeout_ms=20_000)["status"] == "unsat"


def test_bv_tables_agree_with_tables_py() -> None:
    assert all(ZC.bv_tables_agree(REAL).values())


def test_u4_containment() -> None:
    c = ZC.containment(REAL)
    assert all(c["derived"].values()), c["failing_derived"]
    bad = {e["edge"] for e in c["edges"] if not e["ok"]}
    # pack.e1 now declares the range it is fed (exp_lo: the floor at z = 1, masked by (1 - z)); the group-to-group
    # edge is contained only through the per-group ceiling e_hi(g) + growth, which the contracts do not state
    assert bad == {"state_out.e_hat (g=0) contract range -> group_max.cand[0]"}


def test_census_delta_numbers() -> None:
    d = CR.census_delta(REAL)
    assert d["catalog/census"]["mul"] == 358 and d["catalog/census"]["lin"] == 137 and d["catalog/census"]["lookup"] == 306
    dl = d["delta_per_unit"]
    assert dl["range_chunk_columns (committed columns for w>16 chunks; A may recompute, B commits)"] == 44
    assert dl["wide_range_checks"] == 22
