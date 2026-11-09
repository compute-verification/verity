"""The Flock bench's view of the backend: a frozen set's instance file and statement, the frozen sets' digests against the
bench contract's, and registration's patch of a bench result."""
from __future__ import annotations

import json

import pytest

from verity_flock import instances


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
