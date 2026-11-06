"""The Flock backend's pinned lowering, its support declaration and its instance statement."""
from __future__ import annotations

import json

import pytest
from verity.ml.circuits.targets.target import BF16_HOPPER, FIRST, FP8_ADA_MMA, FP8_HOPPER_WGMMA, NVFP4_SM120

from verity_flock import instances, lowering
from verity_flock.backend import FLOCK_FRAME_V3_BLAKE3 as FLOCK, GemmSub, Supported, Unavailable, Unsupported


@pytest.mark.parametrize("relation", sorted(lowering.PIPES))
def test_lowering_is_the_pinned_one(relation):
    assert lowering.digest(relation) == lowering.PINS[relation]


@pytest.mark.parametrize("relation", ["bf16-hopper", "fp8-ada"])
@pytest.mark.slow
def test_lowered_unit_matches_the_reference(relation):
    assert lowering.self_check(relation, n=40) == 0


def test_non_finite_accumulator_leaves_a_row_unsatisfied():
    pipe = lowering.PIPES["bf16-hopper"]
    x = [0x3F80] * pipe.k
    _, _, ok = lowering.evaluate("bf16-hopper", x, x, 0x7F800000)
    assert not ok
    _, _, ok = lowering.evaluate("bf16-hopper", x, x, 0x3F800000)
    assert ok


def test_io_words_are_aligned_and_zero_padded():
    head = lowering.netlist("bf16-hopper").split("\n", 1)[0].split()
    assert head[0] == lowering.FORMAT and head[6:11] == ["0", "1", "2", "3", "4"]
    ins, outs = int(head[5]), [int(c) for c in head[12:12 + int(head[11])]]
    assert ins == 5 and len(outs) == 2 and int(head[2]) <= 1 << lowering.K_LOG


def test_supports():
    for t in (BF16_HOPPER, FP8_ADA_MMA, FP8_HOPPER_WGMMA):
        assert isinstance(FLOCK.supports(GemmSub(t)), (Supported, Unavailable))
    assert isinstance(FLOCK.supports(GemmSub(FIRST)), (Supported, Unavailable))
    assert isinstance(FLOCK.supports(GemmSub(NVFP4_SM120)), Unsupported)
    assert isinstance(FLOCK.supports(GemmSub(BF16_HOPPER, 1000)), Unsupported)
    assert isinstance(FLOCK.supports(GemmSub(FP8_ADA_MMA, 512)), Unsupported)


def test_lower_refuses_an_unpinned_lowering(monkeypatch):
    if isinstance(FLOCK.supports(GemmSub(BF16_HOPPER)), Unavailable):
        monkeypatch.setattr("shutil.which", lambda _: "/bin/true")
    cs = FLOCK.lower(GemmSub(BF16_HOPPER))
    assert cs["sha256"] == lowering.PINS["bf16-hopper"] and cs["units"] == 96 and cs["row_bytes"] == 3072
    monkeypatch.setitem(lowering.PINS, "bf16-hopper", "0" * 64)
    with pytest.raises(ValueError, match="not the pinned"):
        FLOCK.lower(GemmSub(BF16_HOPPER))


def test_record_names_only_standard_hashes():
    rec = FLOCK.record()
    assert {h["class"] for h in rec["hashes"]} == {"standard"}
    assert rec["zk_class"] == "NON_ZK_PROOF"


@pytest.mark.parametrize("relation", ["bf16-hopper", "fp8-ada"])
def test_instance_file_and_statement(tmp_path, relation):
    h = instances.write(tmp_path / "i.bin", relation, 4)
    assert h["instances"]["manifest_sha256"] == instances.instances_digest(relation, 4)
    blob = (tmp_path / "i.bin").read_bytes()
    head = json.loads(blob.split(b"\n", 1)[0])
    body = len(blob) - len(blob.split(b"\n", 1)[0]) - 1
    assert body == 4 * (2 * head["row_bytes"] + 4 * head["units"] + 8)
    assert head["roots"] == instances.statement(relation, *instances.draw(relation, 4))["roots"]


def test_frozen_set_digests_match_the_contract():
    from verity_numerical.bench import contract

    assert instances.instances_digest("bf16-hopper", 4096) == contract.BF16_HOPPER_INSTANCES_DIGEST
    assert instances.instances_digest("fp8-ada", 4096) == contract.FP8_ADA_INSTANCES_DIGEST
    assert instances.instances_digest("fp8-hopper", 4096) == contract.FP8_HOPPER_INSTANCES_DIGEST


def test_register_adds_the_union_bound_and_the_verifier(tmp_path):
    import argparse
    from verity_flock.register import patch

    doc = {"workload_fingerprint": {"N_subbatches": 4, "security": {"achieved_log2": -195.5}, "software": {"backend": {}}},
           "measurements": [{"name": "net.rtt_ms", "value": 1.42, "unit": "ms"}], "interaction": {"rtt_method": "coin wait"},
           "validation": {}}
    a = argparse.Namespace(rtt_ms=0.68, rtt_method="tcp", loopback_seconds=0.00083, loopback_method="BORROWED from r1", supersedes="art:x", lane="some-lane")
    out = patch(doc, {"run": "p", "commit": "c1", "binary_sha256": "s1", "host": "h"}, {"run": "v", "commit": "c2", "binary_sha256": "s2", "host": "h2"}, a)
    assert out["workload_fingerprint"]["security"]["achieved_log2"] == -193.44
    ms = {m["name"]: m["value"] for m in out["measurements"]}
    assert ms["net.rtt_ms"] == 0.68 and ms["live.coin_wait_per_round_ms"] == 1.42 and ms["live.loopback_round_seconds"] == 0.00083
    assert out["workload_fingerprint"]["software"]["verifier"]["binary_sha256"] == "s2"
    assert out["interaction"]["loopback_method"].startswith("BORROWED")
    assert out["lane"] == out["registered_by"] == "some-lane"
