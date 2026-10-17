import hashlib
import json

import pytest

from verity_proof_throughput import cli
from verity_proof_throughput import instances as I
from verity_proof_throughput.workloads import ladder, workload

W = workload("bf16-k6144")


def test_the_manifest_pins_its_ladder_from_its_lanes():
    m = I.load(W)
    assert (m["schema"], m["dataset"], m["tier"], m["n"]) == (I.SCHEMA, I.DATASET, W.name, W.n_max)
    assert len(m["lanes"]) == W.n_max and m["recipe"]["definitions"] == W.definitions()
    assert m["prefixes"] == {str(b): I.prefix_digest(m["lanes"], b) for b in ladder(W.n_max)}


def test_a_recomputed_prefix_matches_the_manifest():
    got = I.check_prefix(W, 4)
    assert got["ok"] and got["prefix"] == I.load(W)["prefixes"]["4"]
    assert got["manifest_sha256"] == hashlib.sha256(I.manifest_path(W).read_bytes()).hexdigest()


def test_lane_digests_cover_every_word_in_order():
    a = I.lane_digests([[1, 2], [3, 4]], [[5, 6]])
    assert a[0] == hashlib.sha256(I.LANE_TAG + bytes([1, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0])).hexdigest()
    assert a != I.lane_digests([[1, 2], [3, 4]], [[5, 7]])
    assert I.prefix_digest(a, 1) != I.prefix_digest(a, 2)


def test_a_result_names_a_prefix_of_the_tier():
    ref = I.instances_ref(W, 64)
    assert ref == {"dataset": I.DATASET, "tier": W.name, "range": [0, 64], "manifest_sha256": I.manifest_sha256(W)}


#: what the staged bf16-k6144 circuit's META binds (run r20261017-071310-138c)
PARTITION_DIGEST = ("acfd881d5a925fde936cdca0df5f49e52ffb5627d157f3a96e457fec4dbb39d8"
                    "0aa6ae853a3f59cc0a6f318183abd43224a9aaf117f2cd922bae1b6c31d62953")
PROGRAM_SHA512 = ("fe1abab7f2143194e42a87208aa777c6714df349d527a5e89772c784056d1689"
                  "0d58329f1c37677ec3ab2bb99b0b72c595c206c4405e9d4913f768cda7de32ca")


def test_the_verifiers_partition_object_is_the_one_the_circuit_binds(tmp_path, capsys):
    meta = {"partition": {"digest": PARTITION_DIGEST, "rule": "verity/partition/v1"}, "program_sha512": PROGRAM_SHA512}
    obj = I.partition_object(W, meta)
    assert obj["program"] == PROGRAM_SHA512 and obj["query"]["name"] == "Q_word"
    with pytest.raises(ValueError, match="binds partition"):
        I.partition_object(W, meta | {"program_sha512": "0" * 128})

    circuit = tmp_path / "circuit.txt"
    circuit.write_text("flock-circuit unit-class/x\nMETA " + json.dumps(meta) + "\nGATES\n")
    assert cli.main(["partition", "--workload", W.name, "--circuit", str(circuit), "--out", str(tmp_path / "p.json")]) == 0
    assert json.loads((tmp_path / "p.json").read_text()) == obj and capsys.readouterr().out.strip() == PARTITION_DIGEST
    circuit.write_text("flock-circuit unit-class/x\nMETA " + json.dumps(meta | {"program_sha512": "0" * 128}) + "\n")
    assert cli.main(["partition", "--workload", W.name, "--circuit", str(circuit), "--out", str(tmp_path / "q.json")]) == 1
