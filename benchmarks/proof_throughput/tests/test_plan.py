from pathlib import Path

import pytest

from verity_proof_throughput import plan as P
from verity_proof_throughput import tool as T
from verity_proof_throughput.workloads import workload

PINS = P.Pins("0" * 40, "1" * 64)
W = workload("bf16-k6144")


def test_identity_is_deterministic_and_covers_every_pin():
    pt = P.Point(W.name, "plain", 64)
    assert P.identity(PINS, pt) == P.identity(PINS, P.Point(W.name, "plain", 64))
    others = [P.identity(P.Pins("2" * 40, PINS.uv_lock_sha256), pt), P.identity(P.Pins(PINS.commit, "3" * 64), pt),
              P.identity(PINS, P.Point(W.name, "plain", 128)), P.identity(PINS, P.Point(W.name, "plain", 64, runs=5))]
    assert len({P.identity(PINS, pt), *others}) == 5


def test_plan_covers_the_ladder_and_finds_passing_runs():
    seen = []

    def lookup(identity, commit, tool):
        seen.append((commit, tool))
        return "r20261017-000000-abcd" if identity == P.identity(PINS, P.Point(W.name, "plain", 4)) else None

    items = P.plan(W, PINS, lo=1, hi=16, lookup=lookup)
    assert [i.point.B for i in items] == [1, 2, 4, 8, 16]
    assert [i.passing_run is not None for i in items] == [False, False, True, False, False]
    assert set(seen) == {(PINS.commit, "proof_throughput_prover")}
    assert len(P.plan(W, PINS, lookup=lambda *_: None)) == 11


def test_configuration_two_is_refused_until_it_exists():
    with pytest.raises(ValueError, match="stubbed"):
        P.points(W, config="recursive-zk")


def test_dispatch_launches_only_points_without_a_passing_run():
    items = P.plan(W, PINS, lo=1, hi=4, lookup=lambda i, *_: "r1" if i == P.identity(PINS, P.Point(W.name, "plain", 2)) else None)
    launched = []
    assert P.dispatch(Path("/r"), items, lambda cmd: launched.append(cmd) or 0) == 0
    assert [T.parse_point(c)["b"] for c in launched] == ["1", "4"]


def test_the_prover_point_is_a_gpu_host_job_on_node_one_and_records_its_identity():
    item = P.plan(W, PINS, lo=8, hi=8, lookup=lambda *_: None)[0]
    cmd = P.prover_command(Path("/r"), item)
    flags = cmd[:cmd.index("--")]
    assert flags[flags.index("--on") + 1] == "vy-nebius-1" and "--gpu-host" in flags and "--queue" in flags
    assert int(flags[flags.index("--cpus") + 1]) <= 32
    job = cmd[cmd.index("--") + 1:]
    assert job[:2] == ["bash", "benchmarks/proof_throughput/job.sh"]
    assert T.parse_point(job) == {"workload": W.name, "b": "8", "warm": "1", "runs": "3", "identity": item.identity}


def test_the_verifier_point_runs_in_the_proofs_pool_or_queued_on_the_gpu_node_never_cpu_one():
    cmd = P.verifier_command(Path("/r"), "art:abc", "r20261017-000000-abcd")
    flags = cmd[:cmd.index("--")]
    assert flags[flags.index("--on") + 1] == "vy-nebius-cpu-3" and flags[flags.index("--pool") + 1] == "proofs"
    assert flags[flags.index("--cwd") + 1] == "clone"
    assert "vy-nebius-cpu-1" not in P.VERIFIER_NODES
    assert T.parse_point(cmd) == {"bundle": "art:abc", "prove_run": "r20261017-000000-abcd"}
    flags = P.verifier_command(Path("/r"), "art:abc", "r1", node=P.PROVER_NODE)
    flags = flags[:flags.index("--")]
    assert "--queue" in flags and "--gpu-host" not in flags and "--pool" not in flags and int(flags[flags.index("--cpus") + 1]) <= 4


def test_tools_parse_only_their_keys():
    assert T.parse_point(["bash", "x.sh", "B=4", "FOO=1", "IDENTITY=ab", "noise"]) == {"b": "4", "identity": "ab"}
    assert T.PROVER_POINT.key_params == ["identity"] and T.VERIFIER_POINT.key_params == ["bundle"]
