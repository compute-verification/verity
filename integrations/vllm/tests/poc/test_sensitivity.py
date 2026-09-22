"""D5 sensitivity script: cost re-summation must agree with the exported omega_op; scenarios are monotone."""
from __future__ import annotations

from pathlib import Path

import pytest

from frontier import sensitivity as S

ROOT = Path(__file__).resolve().parents[2]
B0 = ROOT / "docs/data/frontier/aggregate-export/u_table.json"


@pytest.fixture(scope="module")
def rows():
    if not B0.exists():
        pytest.skip("B0 aggregate u_table not present")
    return S.load_rows(B0, 1_000_000, None, None)


def test_baseline_resum_matches_exported_omega_op(rows):
    for r in rows:
        assert S.omega(r, {}, None) == pytest.approx(r["omega_op"], rel=1e-9)


def test_scaling_a_component_to_zero_never_raises_cost(rows):
    for comp in S.COMPONENTS:
        for r in rows:
            assert S.omega(r, {comp: 0.0}, None) <= r["omega_op"] + 1e-12


def test_certificate_cost_is_monotone_in_target_and_in_proof_scale(rows):
    for plan in {r["plan"] for r in rows}:
        R = [r for r in rows if r["plan"] == plan]
        c_nt = S.cert_cost(R, {}, None, None)
        c_64 = S.cert_cost(R, {}, None, 64.0)
        assert c_nt is not None and c_64 is not None and c_nt[0] <= c_64[0]
        c_half = S.cert_cost(R, {"proof_sp1": 0.5}, None, None)
        assert c_half[0] <= c_nt[0]


def test_d1_overrides_touch_only_their_plan(rows):
    sc = {name: (scale, ov) for name, _, scale, ov in S.scenarios(0.0047)}
    _, ov_whole = sc["D1 whole: recording = recorder delta"]
    _, ov_op = sc["D1 operator: full store always-on"]
    for r in rows:
        if r["cu"] == "operator":
            assert ov_whole(r) is None
            o = ov_op(r)
            assert o["acquisition_commit"] == 0.0 and o["recording_always_on"] >= r["cost_decomposition"]["acquisition_commit"]
        else:
            assert ov_op(r) is None and ov_whole(r) == {"recording_always_on": 0.0047}


def test_frontier_u_infeasible_below_cheapest_cell(rows):
    cheapest = min(r["omega_op"] for r in rows)
    assert S.frontier_u(rows, {}, None, cheapest * 0.5) is None
    assert S.frontier_u(rows, {}, None, cheapest) is not None
