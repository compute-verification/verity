"""`verity bench prove lean-verify`, the Lean verifier's runner, with a stand-in verifier: each run on a kept proof
recorded, a refusal ending it, each file found by name and the block taken from the flags, and what an accept
holds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from prove_helpers import (
    ACCEPTED,
    LEAN_HOST,
    PARTITION,
    PROGRAM,
    PROGRAM_FILE,
    PROGRAM_SHA512,
    RMSNORM,
    STAGE,
    UNIT_SHA512,
    VERIFIER,
    Verifier,
    _outside_a_research_run,  # noqa: F401  (autouse)
    _sha512,
    circuit_text,
    kept_proof,
)
from verity_bench import lean_verify as L
from verity_bench import prove as P
from verity_bench import prove_flock as PF
from verity_bench import prove_records as R
from verity_bench.cli import main


def test_the_lean_verifiers_runner_records_each_run_of_the_verifier_on_a_kept_proof(tmp_path, monkeypatch, capsys):
    """The runner on a kept proof, the verifier a fake: the statement's files go to the verifier's archive, each run
    is one `verify` on the session with its own seconds, and the record names the block type, the proof's bytes, the
    runs and the host. `prove` reads the record it wrote."""

    tree, out = kept_proof(tmp_path, stage=STAGE), tmp_path / "out"
    verifier = Verifier(monkeypatch, [(ACCEPTED, 12.5), (ACCEPTED, 10.0), (ACCEPTED, 11.0)])
    assert main(["lean-verify", "--lean", VERIFIER, "--proof", str(tree), "--out", str(out)]) == 0
    doc = json.loads((out / L.RECORD).read_text())
    put, *verifies = verifier.calls
    archive = put[2]
    assert put[:2] == [VERIFIER, "archive-put"] and [Path(x).name for x in put[3:]] == ["circuit.txt", "pub-256.bin", "program-word.json"]
    assert not Path(archive).exists()                                # the verifier's archive is scratch, outside the record
    assert verifies == 3 * [[VERIFIER, "verify", "--statement", "verity/flock-circuit", "--archive", archive,
                             "--circuit", "key-circuit.txt", "--public", "key-pub-256.bin", "--program", "key-program-word.json",
                             "--zk", "--session", str(tree / "session")]]
    assert doc["schema"] == L.SCHEMA and doc["accepted"] is True and doc["host"] == LEAN_HOST
    assert doc["block"] == {"word": RMSNORM, "definition": "Staged_v2{N=1}", "k_log": 27, "blocks": 256, "full": 256}
    assert doc["proof"] == {"files": [{"name": "circuit.rep0.bin", "bytes": 300}, {"name": "circuit.rep1.bin", "bytes": 200}],
                            "bytes": 500, "record_bytes": 2}
    assert doc["statement"]["files"] == [{"name": "circuit.txt", "bytes": 4000}, {"name": "pub-256.bin", "bytes": 64},
                                         {"name": "program-word.json", "bytes": len(PROGRAM_FILE)}]
    assert (doc["statement"]["zk"], doc["statement"]["circuit_sha512"]) == (True, "key-circuit.txt")
    # what the circuit states, and the program handed: its digest is not the circuit's, and nothing holds it to the circuit
    assert doc["statement"]["bound"] == {
        "program_sha512": UNIT_SHA512, "held": [], "withheld": [],
        "partition": {"rule": L.PARTITION_COVER, "digest": _sha512(L.PARTITION_COVER.encode() + b"\0", PARTITION)}}
    assert doc["statement"]["program"] == {"sha512": PROGRAM_SHA512, "fn": RMSNORM} and PROGRAM_SHA512 != UNIT_SHA512
    assert doc["label"] == "accepted on the provisional unit cover, units as stated, the Program not held to the circuit"
    v = doc["verifier"]
    assert (v["archive_put_s"], v["repeat"], [r["wall_s"] for r in v["runs"]]) == (0.25, 3, [12.5, 10.0, 11.0])
    assert v["command"][:2] == ["lean-verifier", "verify"]                 # the executable by its file's name, not its path
    assert v["runs"][0] | {"verdict_sha512": None} == {
        "wall_s": 12.5, "cpu_s": 25.0, "user_s": 12.5, "maxrss_gib": 1.5, "rc": 0, "accepted": True, "why": None,
        "verdict_sha512": None, "phases_ms": {"inputs": 40, "setup": 900, "session": 2000}}
    assert capsys.readouterr().out.startswith('LEAN-VERIFY {"accepted": true, ')

    # the record is the one `prove` reads
    read = R.lean_verify(doc, "r20261110-000000-0001")
    assert read == PF.LeanVerify(P.definition_id(RMSNORM), 27, 256, 256, (12.5, 10.0, 11.0), 500, "r20261110-000000-0001",
                                "cpu-node-0 (64 cores, A CPU 9000)", L.COVER_LABEL, L.STATEMENT_ID)
    assert read.statement_seconds == 11.0 and read.bits == 2**35


def test_a_refusal_ends_the_lean_verifiers_runner_with_no_further_run(tmp_path, monkeypatch, capsys):
    """The verifier refuses the second of three runs: the third is never started, the record says refused with the
    verifier's reason, the command exits 1, and `prove` takes no figure from it. A run that prints no verdict is a
    refusal too."""

    tree, out = kept_proof(tmp_path, stage=STAGE), tmp_path / "out"
    verifier = Verifier(monkeypatch, [(ACCEPTED, 12.0), ({"accepted": False, "why": "table circuit: lincheck"}, 3.0),
                                      (ACCEPTED, 12.0)])
    assert main(["lean-verify", "--lean", VERIFIER, "--proof", str(tree), "--out", str(out)]) == 1
    doc = json.loads((out / L.RECORD).read_text())
    assert len(verifier.calls) == 3 and len(verifier.verdicts) == 1          # archive-put and two runs: no third
    assert doc["accepted"] is False and [(r["accepted"], r["why"], r["rc"]) for r in doc["verifier"]["runs"]] == [
        (True, None, 0), (False, "table circuit: lincheck", 1)]
    assert "the Lean verifier refused run 2 of 3: table circuit: lincheck" in capsys.readouterr().out
    assert R.lean_verify(doc, "r20261110-000000-0001") == "the Lean verifier refused its statement: table circuit: lincheck"

    Verifier(monkeypatch, [(None, 1.0)])
    assert main(["lean-verify", "--lean", VERIFIER, "--proof", str(tree), "--out", str(out), "--repeat", "1"]) == 1
    assert json.loads((out / L.RECORD).read_text())["verifier"]["runs"][0]["why"] == "no verdict (rc 1)"


def test_the_lean_verifiers_runner_finds_each_file_by_name_and_takes_the_block_from_the_flags(tmp_path, monkeypatch):
    def measured(*argv: str) -> dict:
        parser = argparse.ArgumentParser()
        L.add_arguments(parser)
        return L.measure(parser.parse_args(["--lean", VERIFIER, "--proof", str(tree), "--repeat", "1", *argv]), tmp_path / "out")

    tree = kept_proof(tmp_path, rule=L.PARTITION_V1)
    verifier = Verifier(monkeypatch, [(ACCEPTED, 1.0)] * 3)
    with pytest.raises(ValueError, match="no block type: the proof holds no stage.json"):
        measured()
    with pytest.raises(ValueError, match=r"the statement's `k_log` is None: give --k-log"):
        measured("--definition", "Staged_v2{N=1}")
    assert verifier.calls == []
    # the flags name the block; a full statement's blocks are 2^35 witness bits' worth
    doc = measured("--definition", "Staged_v2{N=1}", "--k-log", "30", "--blocks", "16", "--plain", "--verify-arg=--coins",
                   "--verify-arg=tree")
    assert doc["block"] == {"word": None, "definition": "Staged_v2{N=1}", "k_log": 30, "blocks": 16, "full": 32}
    assert verifier.calls[-1][-6:] == ["--program", "key-program-word.json", "--coins", "tree", "--session", str(tree / "session")]
    assert doc["label"] == f"accepted as a plain session, the circuit's partition under {L.PARTITION_V1}"
    assert R.lean_verify(doc, "r1").definition == "Staged_v2{N=1}"

    # a name two files carry is refused, and its flag says which; partition.json and registered.json go where present
    (tree / "other").mkdir()
    (tree / "other" / "circuit.txt").write_text("another")
    (tree / "stage" / "partition.json").write_text(json.dumps(PARTITION))
    with pytest.raises(ValueError, match=r"2 paths under .* are named `circuit.txt` .*: say which with --circuit"):
        measured("--definition", "D", "--k-log", "30", "--blocks", "16")
    doc = measured("--definition", "D", "--k-log", "30", "--blocks", "16", "--circuit", str(tree / "stage" / "circuit.txt"))
    assert "--partition" in verifier.calls[-1] and [f["name"] for f in doc["statement"]["files"]] == [
        "circuit.txt", "pub-256.bin", "partition.json", "program-word.json"]
    assert doc["label"] == f"accepted at --zk, the circuit's partition under {L.PARTITION_V1}"
    (tree / "other" / "program.json").write_text("{}")
    with pytest.raises(ValueError, match=r"2 paths under .* are named `program\*.json` .*: say which with --program"):
        measured("--definition", "D", "--k-log", "30", "--blocks", "16", "--circuit", str(tree / "stage" / "circuit.txt"))
    (tree / "other" / "program.json").unlink()

    (tree / "session" / "circuit.rep0.bin").unlink()
    (tree / "session" / "circuit.rep1.bin").unlink()
    with pytest.raises(ValueError, match="holds no proof file"):
        measured("--definition", "D", "--k-log", "30", "--blocks", "16", "--circuit", str(tree / "stage" / "circuit.txt"))
    with pytest.raises(ValueError, match="archive-put` exited 1"):
        Verifier(monkeypatch, [], put_rc=1)
        (tree / "session" / "circuit.rep0.bin").write_bytes(b"p")
        measured("--definition", "D", "--k-log", "30", "--blocks", "16", "--circuit", str(tree / "stage" / "circuit.txt"))


def test_the_lean_verifiers_runner_holds_the_partition_and_the_circuit_and_labels_what_an_accept_holds(tmp_path, monkeypatch):
    """The circuit's second line binds a partition, and states a `program_sha512` that is no program file's digest.
    A partition file that isn't the bound object, or that the circuit binds nothing for, ends the command before any
    verifier process starts; under the provisional cover the verifier takes no partition object, so the bound cover is
    checked and not handed to it. The program is held to nothing the circuit states: the record says which was handed,
    and only one that names another Definition than `stage.json`'s `word` is refused; but a held partition under
    `PARTITION_V1` names its Program, and the program is then held to that digest instead. An accept on the provisional
    cover is labelled as holding no Program to the circuit. The circuit is held to `stage.json`'s `circuit_sha512`."""

    def measured(*argv: str) -> dict:
        parser = argparse.ArgumentParser()
        L.add_arguments(parser)
        return L.measure(parser.parse_args(["--lean", VERIFIER, "--proof", str(tree), "--repeat", "1", "--definition", "D",
                                            "--k-log", "30", "--blocks", "16", *argv]), tmp_path / "out")

    def refused(match: str) -> None:
        calls = len(verifier.calls)
        with pytest.raises(ValueError, match=match):
            measured()
        assert len(verifier.calls) == calls                                  # no verifier process, archive-put included

    tree = kept_proof(tmp_path, rule=L.PARTITION_V1)
    program, partition, circuit, stage = (tree / "stage" / n for n in ("program-word.json", "partition.json", "circuit.txt", "stage.json"))
    verifier = Verifier(monkeypatch, [(ACCEPTED, 1.0)] * 8)

    # the program: any descriptor is handed on, since the circuit's `program_sha512` is no digest of it; the record
    # says which, by the digest the verifier computes (annotations in none) and the Definition it names
    other = {"schema": "another", "definitions": PROGRAM["definitions"]}
    program.write_text(json.dumps(other | {"annotations": {"theta": [9, 9]}}, separators=(",", ":")))
    doc = measured()
    assert doc["statement"]["program"] == {"sha512": _sha512(b"", other), "fn": None} and doc["statement"]["bound"]["held"] == []
    assert verifier.calls[-1][verifier.calls[-1].index("--program") + 1] == "key-program-word.json"
    program.write_text("not a descriptor")
    refused(r"program-word\.json \(--program\) is not JSON .*: no verifier was started")
    # the stage's word is the Definition the program names, or the command ends: the verifier would accept either
    stage.write_text(json.dumps({"word": "Another_v1{N=2}"}))
    program.write_bytes(PROGRAM_FILE)
    refused(r"program-word\.json is the program of another Definition than the statement's: its `program_fn` is RMSNorm.*, and "
            r".*stage\.json names the `word` Another_v1\{N=2\}: no verifier was started")
    stage.write_text(json.dumps({"word": RMSNORM}))
    assert measured()["statement"]["program"] == {"sha512": PROGRAM_SHA512, "fn": RMSNORM}
    stage.unlink()

    # the bound partition object goes to the verifier; another is refused
    partition.write_text(json.dumps(PARTITION, indent=2))
    doc = measured()
    assert doc["statement"]["bound"]["held"] == ["partition", "program"] and doc["statement"]["bound"]["withheld"] == []
    # under it the program is held to the one the partition object names, not to the stage's word: a bound statement's
    # population Program names its population, and the verifier refuses any program but the partition's
    stage.write_text(json.dumps({"word": "Another_v1{N=2}"}))
    assert measured()["statement"]["bound"]["held"] == ["partition", "program"]
    stage.unlink()
    program.write_text(json.dumps(other))
    refused(r"program-word\.json is not the Program the partition object names: its digest is .*: no verifier was started")
    program.write_bytes(PROGRAM_FILE)
    assert doc["label"] == f"accepted at --zk, the circuit's partition under {L.PARTITION_V1}"
    assert verifier.calls[-1][verifier.calls[-1].index("--partition") + 1] == "key-partition.json"
    partition.write_text(json.dumps(PARTITION | {"program": "q"}))
    refused(rf"partition\.json is not the partition the circuit binds: its digest under {L.PARTITION_V1} is .*: no verifier was started")

    # the provisional cover: the bound object is checked and withheld, since the verifier takes none under that rule
    circuit.write_bytes(circuit_text(L.PARTITION_COVER))
    refused(rf"partition\.json is not the partition the circuit binds: its digest under {L.PARTITION_COVER} is ")
    partition.write_text(json.dumps(PARTITION))
    doc = measured()
    put, verify = verifier.calls[-2:]
    assert [Path(x).name for x in put[3:]] == ["circuit.txt", "pub-256.bin", "program-word.json"] and "--partition" not in verify
    assert [x["name"] for x in doc["statement"]["files"]] == ["circuit.txt", "pub-256.bin", "program-word.json"]
    assert doc["statement"]["bound"] == {
        "program_sha512": UNIT_SHA512, "held": ["partition"], "withheld": ["partition"],
        "partition": {"rule": L.PARTITION_COVER, "digest": _sha512(L.PARTITION_COVER.encode() + b"\0", PARTITION)}}
    # the label of an accept there, word for word; a plain session's accept is not that one
    assert doc["label"] == L.COVER_LABEL == "accepted on the provisional unit cover, units as stated, the Program not held to the circuit"
    assert measured("--plain")["label"] == f"accepted as a plain session, the circuit's partition under {L.PARTITION_COVER}"

    # the circuit is the one stage.json names
    (tree / "stage" / "stage.json").write_text(json.dumps({"circuit_sha512": "key-circuit.txt"}))
    assert measured()["accepted"] is True
    (tree / "stage" / "stage.json").write_text(json.dumps({"circuit_sha512": "another"}))
    calls = len(verifier.calls)
    with pytest.raises(ValueError, match=r"circuit\.txt is not the circuit .*stage\.json names: its SHA-512 is key-circuit\.txt"):
        measured()
    assert [c[1] for c in verifier.calls[calls:]] == ["archive-put"]         # the verifier hashed it, and verified nothing

    # a circuit that binds nothing holds no file to it
    (tree / "stage" / "stage.json").unlink()
    circuit.write_bytes(b"x" * 4000)
    refused(r"circuit\.txt binds no partition under a rule this command hashes \(its rule is None; ")
    partition.unlink()
    doc = measured()
    assert doc["statement"]["bound"] == {"program_sha512": None, "partition": None, "held": [], "withheld": []}
    assert doc["label"] == "accepted at --zk, the circuit's partition under no rule"
    circuit.write_bytes(b"flock-circuit unit/made-up\nMETA {not json\n")
    refused(r"circuit\.txt: its second line is not `META` and a JSON object")
    assert verifier.verdicts == []
