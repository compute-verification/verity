import json

import pytest

from verity_proof_throughput import workloads as W


def test_cap_is_the_largest_power_of_two_a_statement_holds():
    w = W.workload("bf16-k6144")
    assert w.cap == 1024 == W.MAX_STATEMENT_BITS >> 25
    assert W.WORKLOADS["fp8-coord-k256"].cap == 4096   # no k_log yet: the instance set's size bounds it


def test_ladder_runs_over_powers_of_two():
    assert W.ladder(16) == [1, 2, 4, 8, 16]
    assert W.ladder(1024, lo=64) == [64, 128, 256, 512, 1024]
    for bad in ((12, 1), (16, 3), (8, 16)):
        with pytest.raises(ValueError):
            W.ladder(*bad)


def test_definitions_are_one_synthetic_call_of_the_vu():
    w = W.workload("bf16-k6144")
    (src, count), = w.definitions().items()
    assert count == 1 and src == 'Gemm_v2{K=6144,N=1024,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}'
    assert json.loads(W.defs_json(w)) == w.definitions()
    assert "K=6144" in w.vu and w.extra["dot"] in w.vu


def test_served_unit_workloads_are_not_staged_by_the_class_sweep():
    with pytest.raises(ValueError, match="served-unit"):
        W.WORKLOADS["fp8-coord-k2048"].definitions()


def test_every_workload_names_a_contract_profile():
    from verity_numerical.bench import contract

    assert all(w.profile in contract.PROFILES for w in W.WORKLOADS.values())
    with pytest.raises(ValueError, match="unknown workload"):
        W.workload("bf16-k4096")
