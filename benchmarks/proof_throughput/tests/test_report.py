import copy

import pytest

from verity_proof_throughput import report as R
from verity_proof_throughput import result as RS
from verity_proof_throughput.workloads import workload

W = workload("bf16-k6144")
SW = {"backend": {"name": "flock-circuit", "commit": "0" * 40}, "toolchain": {"python": "3.12"}, "verifier": {}}
LEAN = [{"session": "l0001-1", "accepted": True, "why": None, "wall_s": 2.0, "cpu_s": 1.5, "max_rss_kb": 1000}]


def result(record, hardware, B, run, scale=1.0):
    """The record's prover, ``scale`` times as long."""
    rec = copy.deepcopy(record)
    lv = rec["live"]
    for k in ("prove_total_s", "witness_s", "e2e_s"):
        lv[k] *= scale
    lv["buckets"] = [{k: v * scale if isinstance(v, float) else v for k, v in r.items()} for r in lv["buckets"]]
    return RS.point(W, B, rec, run_id=run, hardware=hardware, software=SW, gpu_peak_mib=None, lean=LEAN)


def test_compare_accepts_one_identity_and_reports_ratios(record, hardware):
    out = R.compare(result(record, hardware, 16, "a"), result(record, hardware, 16, "b", scale=0.5))
    assert out["measurements"]["t.total"]["ratio"] == pytest.approx(0.5)
    assert out["measurements"]["vu.per_second"]["ratio"] > 1


def test_compare_refuses_results_whose_identities_differ(record, hardware):
    a = result(record, hardware, 16, "a")
    with pytest.raises(R.Refused, match="B differs"):
        R.compare(a, result(record, hardware, 32, "b"))
    assert R.compare(a, result(record, hardware, 32, "b"), allow=["B"])["allow"] == ["B"]
    other = copy.deepcopy(hardware)
    other["gpu"]["name"] = "NVIDIA H100 80GB HBM3"
    with pytest.raises(R.Refused, match="hardware differs"):
        R.compare(a, result(record, other, 16, "b"))


def test_compare_refuses_a_result_that_did_not_pass(record, hardware):
    bad = copy.deepcopy(record)
    bad["live"] |= {"accepted": False, "why": "prove exited 1"}
    with pytest.raises(R.Refused, match="did not pass"):
        R.compare(result(record, hardware, 16, "a"), result(bad, hardware, 16, "b"))


def test_the_frontier_is_one_row_per_b_and_refuses_a_mixed_sweep(record, hardware):
    rows = R.frontier([result(record, hardware, b, f"r{b}", scale=b / 16) for b in (16, 1, 4)])
    assert [r["B"] for r in rows] == [1, 4, 16] and all(r["dominant"] for r in rows)
    assert R.best(rows)["B"] == 1
    table = R.markdown(rows, W.name)
    assert table.count("\n| ") == 4 and "overhead vs H200 peak" in table
    other = copy.deepcopy(hardware)
    other["cpu"]["count"] = 128
    with pytest.raises(R.Refused, match="not the same sweep"):
        R.frontier([result(record, hardware, 1, "a"), result(record, other, 2, "b")])
