"""P2 v1's scale sweep (`benchmarks/pous/p2_v1/sweep.py`): the seed-to-cell order, the Clopper-Pearson interval and the late
answers' attribution, which its table reports."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p2_v1"))

import sweep  # noqa: E402


def test_seeds_fill_every_t_of_one_weight_set_before_the_next_and_every_weight_set_before_the_next_rep():
    cells = [sweep.cell_of(s) for s in range(2 * 39)]
    assert [c["tokens"] for c in cells[:13]] == list(sweep.TOKENS)
    assert {c["weight_set"] for c in cells[:13]} == {sweep.WEIGHT_SETS[0]} and cells[13]["weight_set"] == sweep.WEIGHT_SETS[1]
    assert {c["rep"] for c in cells[:39]} == {0} and cells[39] == {**cells[0], "seed": 39, "rep": 1}
    assert len({(c["weight_set"], c["tokens"], c["rep"]) for c in cells}) == 78


def test_clopper_pearson_matches_the_beta_quantiles():
    lo, hi = sweep.clopper_pearson(18, 20)          # scipy.stats.beta.ppf(0.025, 18, 3), .ppf(0.975, 19, 2)
    assert abs(lo - 0.683017) < 1e-5 and abs(hi - 0.987651) < 1e-5
    lo, hi = sweep.clopper_pearson(0, 10)
    assert lo == 0.0 and abs(hi - 0.308497) < 1e-5
    lo, hi = sweep.clopper_pearson(500, 500)
    assert hi == 1.0 and abs(lo - 0.025 ** (1 / 500)) < 1e-12


def test_a_late_answer_is_the_part_furthest_over_its_median_under_load_when_a_neighbour_was_busy():
    cell = {"verifier_core": [10, 11], "responder_core": [12, 13], "numa_cpus": list(range(16))}
    quiet = {"t": 0.0, "loadavg": ["1", "1", "1"], "busy": [100.0 if c in (10, 12) else 0.0 for c in range(16)]}
    busy_sibling = {**quiet, "busy": [100.0 if c in (10, 11, 12) else 0.0 for c in range(16)]}
    limit, medians = 540_000, {"device": 5_000, "responder": 15_000, "verifier": 30_000}
    device = {"elapsed_ns": 600_000, "server_ns": 520_000, "device_ns": 500_000}
    responder = {"elapsed_ns": 600_000, "server_ns": 570_000, "device_ns": 5_000}
    verifier = {"elapsed_ns": 600_000, "server_ns": 20_000, "device_ns": 5_000}
    assert sweep.attribute(device, limit, medians, quiet, cell)["cause"] == "device, quiet"
    assert sweep.attribute(device, limit, medians, busy_sibling, cell)["cause"] == "device, load"
    assert sweep.attribute(responder, limit, medians, busy_sibling, cell)["cause"] == "responder, load"
    assert sweep.attribute(verifier, limit, medians, quiet, cell)["cause"] == "verifier, quiet"
    assert sweep.attribute(verifier, limit, medians, None, cell)["cause"] == "verifier, no sample"


def test_a_cell_whose_only_failed_gate_is_timed_audit_succeeds_and_any_other_failure_keeps_pod_shs_rc(tmp_path):
    import json
    result = tmp_path / "bench-result.json"
    assert sweep.cell_rc(1, result) == 1
    result.write_text(json.dumps({"gates": {"operating_point": {"passed": True}, "timed_audit": {"passed": False}}}))
    assert sweep.cell_rc(1, result) == 0 and sweep.cell_rc(0, result) == 0
    result.write_text(json.dumps({"gates": {"operating_point": {"passed": False}, "timed_audit": {"passed": False}}}))
    assert sweep.cell_rc(1, result) == 1
    result.write_text(json.dumps({"gates": {"operating_point": {"passed": True}}}))
    assert sweep.cell_rc(1, result) == 1
    over_cap = {"passed": False, "k": True, "rtt_allowance": False, "deadline": True}
    result.write_text(json.dumps({"gates": {"operating_point": over_cap, "timed_audit": {"passed": False}}}))
    assert sweep.cell_rc(1, result) == 0
    result.write_text(json.dumps({"gates": {"operating_point": {**over_cap, "k": False}, "timed_audit": {"passed": True}}}))
    assert sweep.cell_rc(1, result) == 1
