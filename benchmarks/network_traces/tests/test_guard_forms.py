"""The record's leak guard on the forms other files write their records in: the verifier's registered-reads files
(`network_attester_rows.reads_text`) and the network attester's usage record (`runner_parts._usage`) are written so
that none of their bytes reads as a piece of a gate run's witness, which the guard would hold though nothing of the
witness is in them."""
from __future__ import annotations

import json
import os

from benchmarks.network_traces import leak_guard as LG
from benchmarks.network_traces.tests.gate_runs import _gate, _put
from benchmarks.network_traces.tests.test_leak_guard import _guard
from verity.protocols.resource_accounting.network.attester import runner_parts as rp
from verity.protocols.resource_accounting.network.verifier import network_attester_rows as WR


def test_a_ports_reads_written_compactly_are_no_piece_of_a_records_counts(tmp_path):
    """A record head's bucket counts are witness, and a port's reads are positions, protocol structure. Where both
    begin with a run of zeros, a registered-reads file in JSON's default form holds a piece of the witness line, a
    key's end and ten zeros, and the guard holds it though nothing of the witness is in it. The verifier's
    registered-reads files are written without spaces (`network_attester_rows.reads_text`), and the same reads are then no piece
    of the witness. The counts as the witness writes them are a find wherever they are written."""
    g = _gate(tmp_path / "G" / "g1")
    line = json.dumps({"salt_c": "ab" * 24, "head": {"window": 0, "counts": [0] * 16, "root": "ef" * 64}})
    at = line.index('": [0, 0')
    pad = json.dumps({"pad": "x" * (-(len(json.dumps({"pad": ""})) + 1 + at) % LG.TEXT)}) + "\n"
    witness = (pad + line + "\n").encode()
    piece = witness[len(pad) + at:len(pad) + at + LG.TEXT]
    assert (len(pad) + at) % LG.TEXT == 0 and piece == b'": [' + b"0, " * 9 + b"0" and len(set(piece)) >= LG.DISTINCT
    (g / "witness" / "certified.jsonl").write_bytes(witness)
    reads = WR.reads_file({"prog0": {"leaves": 4, "reads": [0] * 16, "root": "cd" * 64}})
    assert json.loads(WR.reads_text(reads)) == reads and " " not in WR.reads_text(reads)
    run = tmp_path / "run"
    _put(run, {"out/scan/honest/registered.json": WR.reads_text(reads).encode(),
               "out/scan/honest/spaced.json": json.dumps(reads, sort_keys=True).encode(),
               "out/copy.json": json.dumps({"counts": [0] * 16}).encode()})
    rc, report, _ = _guard(tmp_path, run, "--dry")
    assert rc == 3 and sorted(d["path"] for d in report["found"]) == ["out/copy.json", "out/scan/honest/spaced.json"]


def test_a_list_of_consecutive_integers_is_no_piece_of_the_certifiers_usage_record(tmp_path):
    """The certifier's run record, a witness file written indented, holds its processes' CPUs. Written as a list of
    integers they are the same bytes as any indented list of consecutive integers at their depth, so a verifier
    outcome's proved units were a find though nothing of the witness is in them. The usage record holds them as one
    string of ranges (`runner_parts._usage`), and under the declared data as shipped the outcome is no find and nothing
    is excused. The CPUs written as a list are a find again: the guard is as it was."""
    record = _gate(tmp_path / "G" / "g1") / "witness" / "warden.json"
    usage = rp._usage()
    assert isinstance(usage["cpus"], str) and usage["cpus"] == rp.cpu_ranges(os.sched_getaffinity(0))
    ranged = {k: {**usage, "cpus": rp.cpu_ranges(range(32))} for k in ("usage", "publisher_usage")}
    record.write_text(json.dumps(ranged, indent=1))
    run = tmp_path / "run"
    outcome = {"audit": {"proved_units": list(range(32)), "verdict": "accept"}}
    _put(run, {"out/tie/verifier/w000-outcome.json": json.dumps(outcome, indent=1).encode()})
    rc, report, _ = _guard(tmp_path, run, "--dry")
    assert rc == 0 and report["found"] == [] and report["excused"] == []
    listed = {k: {**v, "cpus": list(range(32))} for k, v in ranged.items()}
    record.write_text(json.dumps(listed, indent=1))
    rc, report, _ = _guard(tmp_path, run, "--dry")
    assert rc == 3 and [d["path"] for d in report["found"]] == ["out/tie/verifier/w000-outcome.json"]
    assert report["excused"] == []
