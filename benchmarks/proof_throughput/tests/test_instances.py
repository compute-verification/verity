import hashlib
import json

import pytest
from verity.ml.circuits.codec import program_digest

from verity_proof_throughput import cli
from verity_proof_throughput import instances as I
from verity_proof_throughput.workloads import workload

W = workload("bf16-k6144")
TEMPLATE = 'GemmCoordinate_v2{K=6144,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}'


def test_the_manifest_pins_its_population_from_its_lanes():
    m = I.load(W)
    assert (m["schema"], m["dataset"], m["tier"], m["n"]) == (I.SCHEMA, I.DATASET, W.name, W.n_max)
    assert len(m["lanes"]) == W.n_max and m["population_digest"] == I.population_digest(m["lanes"])
    assert m["recipe"]["partition"] == I.PARTITION and m["recipe"]["population"] == f"population_program({W.vu}, {W.n_max})"


def test_recomputed_lanes_match_the_manifest():
    got = I.class_lanes(W, 4)
    assert I.lane_digests(got["ins"], got["want"]) == I.load(W)["lanes"][:4]
    assert got["program_digest"] == I.load(W)["recipe"]["program_digest"]


def test_lane_digests_cover_every_word_in_order():
    a = I.lane_digests([[1, 2], [3, 4]], [[5, 6]])
    assert a[0] == hashlib.sha256(I.LANE_TAG + bytes([1, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0])).hexdigest()
    assert a != I.lane_digests([[1, 2], [3, 4]], [[5, 7]])
    assert I.population_digest(a) != I.population_digest(a[::-1])


def test_a_result_names_the_population_its_draws_are_taken_from():
    assert I.population_ref(W) == {"dataset": I.DATASET, "tier": W.name, "range": [0, W.n_max],
                                   "manifest_sha256": I.manifest_sha256(W)}


def test_the_law_is_one_stratum_of_every_unit():
    assert I.law(W, 16) == {"law": "stratified", "k": 16, "strata": [{"template": TEMPLATE, "units": [[0, 1024]], "k": 16}]}
    assert I.law(W, 1024)["strata"][0]["k"] == 1024


def test_the_population_command_writes_the_law_and_names_the_class_sweeps_program(tmp_path, capsys):
    assert cli.main(["population", "--workload", W.name, "--B", "8", "--out", str(tmp_path)]) == 0
    assert capsys.readouterr().out.split() == [W.vu, str(W.n_max)]
    assert json.loads((tmp_path / "law.json").read_text()) == I.law(W, 8)
    assert cli.main(["population", "--workload", W.name, "--B", "2048", "--out", str(tmp_path)]) == 1


#: the population's program and its Q_template_instance partition object, as the staged circuit's META binds them
PARTITION_DIGEST = ("d299900a9cc0344297d35733b612e76c654329977eb6d3185d7eb535e58413d3"
                    "e8389e0f0b8a381f80150eb8742a802c294818eb1ecbb513bafc38592626d0f8")
PROGRAM_SHA512 = ("7984ebfb4f2a801208943042c48b4be318fa2c2dad283263e1aa2140e9e5e103"
                  "b8a78eff760855ce50e9748eabeb755b75bb28ade6a233c9db2f203af68870a8")


def test_the_verifiers_partition_and_program_are_the_ones_the_circuit_binds(tmp_path, capsys):
    meta = {"partition": {"digest": PARTITION_DIGEST, "rule": "verity/partition/v1"}, "program_sha512": PROGRAM_SHA512}
    obj = I.partition_object(W, meta)
    assert obj["program"] == PROGRAM_SHA512 and obj["query"] == {"name": "Q_template_instance", "version": 0,
                                                               "params": {"template": TEMPLATE}}
    assert program_digest(json.loads(I.program_file(W))) == PROGRAM_SHA512
    with pytest.raises(ValueError, match="binds partition"):
        I.partition_object(W, meta | {"program_sha512": "0" * 128})

    circuit = tmp_path / "circuit.txt"
    circuit.write_text("flock-circuit unit-class/x\nMETA " + json.dumps(meta) + "\nGATES\n")
    assert cli.main(["partition", "--workload", W.name, "--circuit", str(circuit), "--out", str(tmp_path / "p.json"),
                     "--program-out", str(tmp_path / "program.json")]) == 0
    assert json.loads((tmp_path / "p.json").read_text()) == obj and capsys.readouterr().out.strip() == PARTITION_DIGEST
    assert (tmp_path / "program.json").read_bytes() == I.program_file(W)
    circuit.write_text("flock-circuit unit-class/x\nMETA " + json.dumps(meta | {"program_sha512": "0" * 128}) + "\n")
    assert cli.main(["partition", "--workload", W.name, "--circuit", str(circuit), "--out", str(tmp_path / "q.json")]) == 1
