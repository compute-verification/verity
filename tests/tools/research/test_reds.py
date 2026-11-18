"""What a failed step names and whom its red rings (`research.reds`), which the verified queue's loop, drive and the
failed-checks monitor share: the ratchet gate's failed merges, circuit-check's failed targets, a job's owner in ci.toml, and
the order a red rings its owners in."""
from __future__ import annotations

from research.reds import (CIRCUIT_STEP, RATCHET_STEP, circuit_failed, job_owner, named, ratchet_failed, ratchet_prs,
                           red_owners)

#: a ratchet-gate log as `ratchet_gate.render` writes it: two merges failed, one passed, and the commits above the last merge
RATCHET_LOG = ("ratchet gate 0123456789ab..dddddddddddd: 4 change(s), 3 failed\n"
               "  #12 (aaaaaaaaaaaa): failed: grows p01.json 2 -> 3 entries\n"
               "      shrink p01.json\n"
               "  #13 (bbbbbbbbbbbb): passed: grows nothing\n"
               "  #14 (cccccccccccc): failed: adds an env read\n"
               "  cccccccccccc..dddddddddddd (dddddddddddd): failed: grows p02.json\n")

#: a circuit-check log: a pin mismatch, a target whose process died, one the checks raised on, a failure with no detail, a
#: template with parameters, and lines that name no target (`--batch`'s templates, a bare check, a warning, the verdict)
CIRCUIT_LOG = ("FAIL lowering/pin-mismatch NetworkHeadCheck_v2{L=6,T=32,COUNTS=0} [definitions]: and 807 -> 808\n"
               "FAIL circuit-check/killed Big_v1{N=4}: its process died (SIGKILL: the OOM killer?) at a peak of 30 GiB\n"
               "FAIL circuit-check/error Bad_v1: ValueError: no\n"
               "FAIL lowering/dead Callee_v1\n"
               "FAIL circuit-check/killed rope-head:D=128: its process died\n"
               "FAIL rope-head: declared 3, got 4\n"
               "FAIL gemm-coordinate/k2048/sm80-mma-bf16: declared 3, got 4\n"
               "FAIL structural/mismatch\n"
               "unchecked lowering/vectors: 2 over 1 target(s)\n"
               "circuit-check: 120 targets, 5 failed\n")


def test_ratchet_failed_reads_each_failed_line():
    assert ratchet_failed(RATCHET_LOG) == [("#12", "aaaaaaaaaaaa", "grows p01.json 2 -> 3 entries"),
                                           ("#14", "cccccccccccc", "adds an env read"),
                                           ("cccccccccccc..dddddddddddd", "dddddddddddd", "grows p02.json")]
    assert ratchet_failed("#15 (eeeeeeeeeeee): failed: not the gate's indent\n  #16 (eeeeeeeeeeee): passed: x\n") == []


def test_ratchet_prs_are_the_failed_merges_once_each():
    """A range of commits is no PR, nor is a name that only starts like one; a merge named twice is one PR."""
    assert ratchet_prs(RATCHET_LOG) == [12, 14]
    assert ratchet_prs(RATCHET_LOG + "  #12 (aaaaaaaaaaaa): failed: again\n  #7x (eeeeeeeeeeee): failed: x\n") == [12, 14]


def test_circuit_failed_reads_each_target():
    assert circuit_failed(CIRCUIT_LOG) == [
        ("lowering/pin-mismatch", "NetworkHeadCheck_v2{L=6,T=32,COUNTS=0}", "[definitions]: and 807 -> 808"),
        ("circuit-check/killed", "Big_v1{N=4}", "its process died (SIGKILL: the OOM killer?) at a peak of 30 GiB"),
        ("circuit-check/error", "Bad_v1", "ValueError: no"),
        ("lowering/dead", "Callee_v1", ""),
        ("circuit-check/killed", "rope-head:D=128", "its process died")]


def test_named_keys_each_failure_by_step():
    """ratchet-gate's by the merge's name, circuit-check's by CHECK/CODE TARGET, each with its line; another step's log names
    nothing, whatever it holds."""
    assert named(RATCHET_STEP, RATCHET_LOG) == {
        "#12": "#12 (aaaaaaaaaaaa): grows p01.json 2 -> 3 entries", "#14": "#14 (cccccccccccc): adds an env read",
        "cccccccccccc..dddddddddddd": "cccccccccccc..dddddddddddd (dddddddddddd): grows p02.json"}
    assert named(CIRCUIT_STEP, CIRCUIT_LOG) == {
        "lowering/pin-mismatch NetworkHeadCheck_v2{L=6,T=32,COUNTS=0}":
            "lowering/pin-mismatch NetworkHeadCheck_v2{L=6,T=32,COUNTS=0} [definitions]: and 807 -> 808",
        "circuit-check/killed Big_v1{N=4}":
            "circuit-check/killed Big_v1{N=4} its process died (SIGKILL: the OOM killer?) at a peak of 30 GiB",
        "circuit-check/error Bad_v1": "circuit-check/error Bad_v1 ValueError: no",
        "lowering/dead Callee_v1": "lowering/dead Callee_v1",
        "circuit-check/killed rope-head:D=128": "circuit-check/killed rope-head:D=128 its process died"}
    assert named(RATCHET_STEP, CIRCUIT_LOG) == {} and named(CIRCUIT_STEP, RATCHET_LOG) == {}
    assert named("lean-build", RATCHET_LOG + CIRCUIT_LOG) == {}


def test_job_owner_is_ci_tomls():
    jobs = [{"name": "pytest", "owner": "ci"}, "not a table", {"owner": "nobody"}, {"name": "lean-build", "owner": 3},
            {"name": "ratchet-gate", "owner": "vllm"}]
    assert job_owner(jobs, "ratchet-gate") == "vllm" and job_owner(jobs, "pytest") == "ci"
    assert job_owner(jobs, "lean-build") is None and job_owner(jobs, "preflight") is None and job_owner([], "pytest") is None


def test_red_owners_ring_who_acts_first_once_each():
    assert red_owners(["lane-a", None, "lane-b"], [None, "lane-a", "vllm"], ["circuits"], ["ci", "top"]) == \
        ("lane-a", "lane-b", "vllm", "circuits", "ci", "top")
    assert red_owners([], ["vllm"]) == ("vllm",) and red_owners([None], [None]) == (None,) and red_owners([], []) == (None,)
