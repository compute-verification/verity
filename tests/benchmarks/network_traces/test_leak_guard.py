"""`leak_guard`: before custody publishes a run dir, each file in it that holds a gate run's witness (under a witness/
directory or by a declared witness name, a witness piece raw or as hex at any offset, an archive member that does, or an
archive the guard can't read, and with ``--public`` a file of a path the record doesn't declare public) is moved to the
quarantine outside the run dir, and the report names it by path alone; the gate's public files, their JSON written
again and a witness file equal to a public one are no finds, and a declared allowlist entry excuses its piece (with
``*``, any piece of its witness file) only in the records it names, never by a key path."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import random
import signal
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from benchmarks.network_traces import leak_guard as LG
from verity.protocols.resource_accounting.network.attester import runner_parts as rp
from tests.verity.protocols.resource_accounting.network.prover.test_released_keys import GOOD, LIST, _declared, _put as _put_json
from verity.protocols.resource_accounting.network.prover import released_keys as RK
from tests.benchmarks.network_traces.gate_runs import (
    CERTIFIED, DEPLOYMENT, GATE, INDENTED, LINES, SALTS, _gate, _put, _sig_piece, _tar)


def _guard(tmp: Path, run: Path, *extra: str) -> tuple[int, dict, dict]:
    rc = LG.main([str(run), str(tmp / "q"), "--gates", str(tmp / "G"), "--report", str(tmp / "report.json"), *extra])
    return rc, json.loads((tmp / "report.json").read_text()), json.loads((tmp / "q" / "leak-guard.json").read_text())


def test_every_file_holding_the_witness_is_moved_out_and_named_by_its_path_alone(tmp_path):
    """A piece of the certified lines at an odd offset, one salt as upper-case hex or raw, a declared witness name, a
    witness/ directory, a sent gate tarball, an archive member, a compressed stream, a damaged or unreadable archive and
    an archive inside one are each moved to the quarantine (0700), with nothing else; the report holds their paths and
    counts, never a piece's bytes or digest, which the quarantine's report holds."""
    _gate(tmp_path / "G" / "g1")
    piece, _ = _sig_piece()
    salt = SALTS[16:32]
    inner = _tar(tmp_path / "inner.tgz", {"a.txt": b"clean"}).read_bytes()
    zipped = io.BytesIO()
    with zipfile.ZipFile(zipped, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("a/notes.txt", b"salt " + salt.hex().encode() + b"\n")
    run = tmp_path / "run"
    _put(run, {"stdout.log": b"chain: started\n", "job.json": b"{}",
               "out/report.json": json.dumps([{"window": d["window"], "head": d["head"]} for d in LINES], indent=2).encode(),
               "out/leak.txt": b"xxxxx" + piece + b"y", "out/salt.json": json.dumps({"s": salt.hex().upper()}).encode(),
               "out/raw.bin": bytes(3) + SALTS[32:48] + b"\1", "out/n2/certified.jsonl": b"{}",
               "out/scan/witness/x.bin": b"nothing", "out/blob.zst": b"\x28\xb5\x2f\xfd" + bytes(40),
               "out/bad.gz": b"\x1f\x8b" + bytes(40), "out/members.zip": zipped.getvalue(),
               "out/stream.gz": gzip.compress(b"log: " + piece + b"\n"), "out/plain.gz": gzip.compress(b"clean\n")})
    _tar(run / "inputs" / "gate.tgz", {f"{GATE}/end.json": b"{}", f"{GATE}/witness/cells.salts": SALTS})
    _tar(run / "inputs" / "clean.tgz", {f"{GATE}/out/deployment.json": DEPLOYMENT})
    _tar(run / "out" / "nested.tar", {"inner.tgz": inner}, mode="w")

    rc, report, held = _guard(tmp_path, run)
    moved = {"out/leak.txt", "out/salt.json", "out/raw.bin", "out/n2/certified.jsonl", "out/scan/witness/x.bin",
             "out/blob.zst", "out/bad.gz", "out/members.zip", "out/stream.gz", "out/nested.tar", "inputs/gate.tgz"}
    assert rc == 3 and report["verdict"] == "refused" and {d["path"] for d in report["found"]} == moved
    assert all(not (run / p).exists() and (tmp_path / "q" / "files" / p).is_file() for p in moved)
    assert sorted(p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()) == [
        "inputs/clean.tgz", "job.json", "out/plain.gz", "out/report.json", "stdout.log"]
    assert stat.S_IMODE((tmp_path / "q").stat().st_mode) == 0o700
    why = {d["path"]: d["why"] for d in report["found"]}
    assert why["out/leak.txt"] == ["1 witness pieces"] and why["out/n2/certified.jsonl"] == ["a witness path"]
    assert why["inputs/gate.tgz"] == [f"a witness-named member, {GATE}/witness/cells.salts"]
    assert why["out/blob.zst"] == ["a zstd file, which the guard can't read"]
    assert why["out/nested.tar"] == ["an archive inside an archive (inner.tgz)"]
    assert why["out/bad.gz"][0].startswith("a file the guard can't read through")

    text = (tmp_path / "report.json").read_text()
    secrets = [piece, salt, SALTS[32:48]]
    for x in secrets:
        for form in (x.hex(), x.hex().upper(), hashlib.sha256(x).hexdigest(), x.decode("latin-1")):
            assert form not in text
    found = {d["path"]: d["pieces_found"] for d in held["found"]}
    assert [(p["sha256"], p["witness"], p["form"]) for p in found["out/leak.txt"]] == [
        (hashlib.sha256(piece).hexdigest(), f"{GATE}/witness/certified.jsonl", "raw")]
    assert found["out/leak.txt"][0]["at"] == 5 and found["out/salt.json"][0]["form"] == "hex"
    assert found["out/members.zip"][0]["member"] == "a/notes.txt"


def test_the_gates_public_files_their_reprints_and_a_public_twin_are_no_finds(monkeypatch, tmp_path):
    """A record holding the gate's public files as they are, a certificate written again indented (a witness file holds
    it so, public only through `reprinted`) and the witness's copy of a public file is clean; without the reprints the
    indented certificate is a find. A public line that isn't JSON is counted, not written again. A run dir with no gate
    run is read for witness paths alone."""
    g = _gate(tmp_path / "G" / "g1")
    (g / "out" / "torn.jsonl").write_bytes(b'{"window": 0}\n\n{"window": \n')
    run = tmp_path / "run"
    _put(run, {"out/gates/certificates.jsonl": (g / "out" / "certificates.jsonl").read_bytes(),
               "out/verifier/w000-certificates.json": INDENTED, "out/deployment.json": DEPLOYMENT})
    rc, report, _ = _guard(tmp_path, run)
    assert (rc, report["verdict"], report["found"]) == (0, "clean", [])
    assert report["gate_runs"] == [GATE] and report["pieces"] > 0 and report["public_dropped"] > 0
    assert report["public_unparsed"] == 1
    assert LG.main([str(run), str(tmp_path / "q2")]) == 0
    monkeypatch.setattr(LG, "reprinted", lambda p: (b"", 0))
    rc, report, _ = _guard(tmp_path, run, "--dry")
    assert rc == 3 and [d["path"] for d in report["found"]] == ["out/verifier/w000-certificates.json"]


def test_an_allowlist_entry_excuses_its_piece_only_where_it_names(tmp_path):
    """A declared entry (the piece's sha256, the witness file it is cut from, the records it may be in) keeps its piece's
    file in the record at a path its globs match, and the report counts it under the entry's index, without its digest;
    the same piece at another path, or under an entry naming another witness file, is a find."""
    _gate(tmp_path / "G" / "g1")
    piece, _ = _sig_piece()
    entry = {"sha256": hashlib.sha256(piece).hexdigest(), "witness": "certified.jsonl",
             "records": ["out/n2/*/verifier/w*-certificates.json"], "why": "the certificates' shared structure"}
    declared = json.loads(LG.DECLARED.read_text())
    (tmp_path / "ok.json").write_text(json.dumps({**declared, "allow": [entry]}))
    (tmp_path / "other.json").write_text(json.dumps({**declared, "allow": [{**entry, "witness": "cells.salts"}]}))
    run = tmp_path / "run"
    _put(run, {"out/n2/honest/verifier/w008-certificates.json": b"[" + piece + b"]", "out/other.json": piece})
    rc, report, _ = _guard(tmp_path, run, "--declared", str(tmp_path / "ok.json"), "--dry")
    assert rc == 3 and [d["path"] for d in report["found"]] == ["out/other.json"]
    assert report["excused"] == [{"entry": 0, "witness": "certified.jsonl", "pieces": 1,
                                  "records": ["out/n2/honest/verifier/w008-certificates.json"]}]
    rc, report, _ = _guard(tmp_path, run, "--declared", str(tmp_path / "other.json"), "--dry")
    assert len(report["found"]) == 2 and report["excused"] == []
    assert (run / "out" / "other.json").is_file()


def test_a_wildcard_entry_excuses_every_piece_of_its_witness_file_in_its_records_alone(capsys, tmp_path):
    """An entry whose sha256 is ``*`` (a record that observes what the witness file records, as the runner's
    resources.jsonl does the stall probe's cgroup counters) excuses any of that witness file's pieces in the records its
    globs match, and the report and stdout count them without a digest; another witness file's piece there, or its own
    piece in another record, is still a find."""
    _gate(tmp_path / "G" / "g1")
    piece, o = _sig_piece()
    pair = [piece, CERTIFIED[o + LG.TEXT:o + 2 * LG.TEXT]]
    salt = SALTS[16:32]
    entry = {"sha256": "*", "witness": "certified.jsonl", "records": ["resources.jsonl"], "why": "the same counters"}
    declared = json.loads(LG.DECLARED.read_text())
    (tmp_path / "star.json").write_text(json.dumps({**declared, "allow": [entry]}))
    run = tmp_path / "run"
    _put(run, {"resources.jsonl": b"{" + pair[0] + b"," + pair[1] + b"}\n", "out/resources.jsonl": piece})
    rc, report, _ = _guard(tmp_path, run, "--declared", str(tmp_path / "star.json"))
    assert rc == 3 and [d["path"] for d in report["found"]] == ["out/resources.jsonl"]
    assert report["excused"] == [{"entry": 0, "witness": "certified.jsonl", "pieces": 2, "records": ["resources.jsonl"]}]
    assert "2 excused by the allowlist" in capsys.readouterr().out and (run / "resources.jsonl").is_file()
    assert not {hashlib.sha256(x).hexdigest() for x in pair} & set((tmp_path / "report.json").read_text().split('"'))
    _put(run, {"resources.jsonl": piece + salt})
    rc, report, held = _guard(tmp_path, run, "--declared", str(tmp_path / "star.json"), "--dry")
    assert rc == 3 and [d["path"] for d in report["found"]] == ["resources.jsonl"] and report["excused"][0]["pieces"] == 1
    assert [p["witness"] for p in held["found"][0]["pieces_found"]] == [f"{GATE}/witness/cells.salts"]


def test_the_guard_refuses_a_quarantine_the_record_would_take(tmp_path):
    """A quarantine inside the run dir, a missing gates directory, or a declared entry without its reason, with a
    sha256 that is neither a digest nor ``*``, or with anything beside its four fields (the key paths an entry once
    narrowed itself by among them) refuses the guard before it moves anything."""
    run = tmp_path / "run"
    _put(run, {"out/n2/certified.jsonl": b"{}"})
    assert LG.main([str(run), str(run / "out" / "q")]) == 2
    assert LG.main([str(run), str(tmp_path / "q"), "--gates", str(tmp_path / "none")]) == 2
    assert (run / "out" / "n2" / "certified.jsonl").is_file()
    ok = {"sha256": "0" * 64, "witness": "x", "records": ["*"], "why": "y"}
    for bad in ({**ok, "why": ""}, {**ok, "sha256": "A" * 64}, {**ok, "sha256": "**"}, {**ok, "keys": []},
                {**ok, "keys": ["usage/cpus"]}, {**ok, "kind": "structure"}):
        (tmp_path / "bad.json").write_text(json.dumps({"schema": LG.DECLARED_SCHEMA, "names": [], "allow": [ok, bad]}))
        with pytest.raises(SystemExit, match="entry 1 needs"):
            LG.main([str(run), str(tmp_path / "q"), "--declared", str(tmp_path / "bad.json")])
    assert (run / "out" / "n2" / "certified.jsonl").is_file()


@pytest.mark.parametrize("numpy", [True, False])
def test_the_finder_finds_a_needle_at_any_offset_and_across_chunks(monkeypatch, numpy):
    """`Finder.stream` finds each needle at any offset, one straddling two chunks among them, and of needles sharing their
    first 8 bytes the one there, with numpy and in pure Python alike, among enough needles to size its table past its
    least; and nothing that only shares a needle's first 8 bytes."""
    if not numpy:
        monkeypatch.setattr(LG, "np", None)
    monkeypatch.setattr(LG, "CHUNK", 64)
    rnd = random.Random(3)
    a, b, c = rnd.randbytes(32), rnd.randbytes(16), rnd.randbytes(16)
    twin = a[:8] + rnd.randbytes(24)
    data = rnd.randbytes(50) + a + rnd.randbytes(77) + b + c[:8] + rnd.randbytes(9) + twin + rnd.randbytes(31)
    f = LG.Finder([a, b, c, twin, *(rnd.randbytes(32) for _ in range(1 << 14))])
    named = lambda ix: {f.items[i] for i in ix}  # noqa: E731
    assert named(f.stream(io.BytesIO(data))) == {a, b, twin} == named(f.hits(data))
    assert named(f.stream(io.BytesIO(data[:51] + data[52:]))) == {b, twin}


def _held(tmp: Path) -> tuple[Path, Path, Path]:
    """A run dir, its out/ and a hold beside it: a clean output, and one planting a salt as hex."""
    run, hold = tmp / "run", tmp / "hold"
    (run / "out").mkdir(parents=True)
    _put(hold, {"report.json": b'{"verdict": "accepted"}', "n2/planted.json": json.dumps({"s": SALTS[16:32].hex()}).encode()})
    return run, run / "out", hold


def test_a_held_output_reaches_the_record_only_once_the_guard_passes_it(tmp_path):
    """With --hold, the guard reads the held outputs with the run dir and moves to out/ only what it passes: a planted
    witness value is refused (its file to the quarantine under its record path, never in out/), the clean output is
    released, and the report says so by counts and paths."""
    _gate(tmp_path / "G" / "g1")
    run, out, hold = _held(tmp_path)
    rc, report, _ = _guard(tmp_path, run, "--hold", str(hold))
    assert rc == 3 and [(d["path"], d["held"]) for d in report["found"]] == [("out/n2/planted.json", True)]
    assert (report["held"], report["released"]) == (2, 1)
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == ["report.json"]
    assert (tmp_path / "q" / "files" / "out" / "n2" / "planted.json").is_file()
    assert not any(p.is_file() for p in hold.rglob("*"))
    assert SALTS[16:32].hex() not in (tmp_path / "report.json").read_text()


def test_a_record_that_declares_what_is_public_holds_nothing_else(tmp_path):
    """With ``--public``, a file whose record path no glob declares, in the run dir or held, is a find whatever it holds
    (here a window's counts, which no witness piece shows), and a held one is never released; a declared one is still
    read for the witness. Without it, the counts are no find."""
    _gate(tmp_path / "G" / "g1")
    piece, _ = _sig_piece()
    counts = b'{"requests": 3, "steps": 9, "outputs": 12}'
    run, out, hold = tmp_path / "run", tmp_path / "run" / "out", tmp_path / "hold"
    _put(run, {"out/piece.log": b"started\n", "out/certs.json": counts})
    _put(hold, {"answer.json": b'{"verdict": "accept"}', "chain/report/report.json": b'{"verdict": "accepted"}',
                "chain/endpoint/report.json": counts, "chain/scan/answers.json": piece})
    rc, report, _ = _guard(tmp_path, run, "--hold", str(hold), "--dry")
    assert rc == 3 and [d["path"] for d in report["found"]] == ["out/chain/scan/answers.json"]
    declared = ("out/answer.json", "out/piece.log", "out/chain/report/*", "out/chain/scan/answers.json")
    rc, report, _ = _guard(tmp_path, run, "--hold", str(hold), *(x for g in declared for x in ("--public", g)))
    assert rc == 3 and {d["path"]: d["why"] for d in report["found"]} == {
        "out/certs.json": ["not declared public (--public)"],
        "out/chain/endpoint/report.json": ["not declared public (--public)"],
        "out/chain/scan/answers.json": ["1 witness pieces"]}
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == [
        "answer.json", "chain/report/report.json", "piece.log"]
    assert (tmp_path / "q" / "files" / "out" / "chain" / "endpoint" / "report.json").read_bytes() == counts
    assert not any(p.is_file() for p in hold.rglob("*"))


def test_the_guard_refuses_a_hold_custody_would_take(tmp_path):
    """A hold inside the run dir, or one where a held file would replace a record file that isn't a log, refuses the guard
    with nothing released; a held log is appended to the log already there."""
    _gate(tmp_path / "G" / "g1")
    run, out, hold = _held(tmp_path)
    assert LG.main([str(run), str(tmp_path / "q"), "--hold", str(out / "h")]) == 2
    _put(run, {"out/report.json": b"{}", "out/job.log": b"before\n"})
    _put(hold, {"job.log": b"after\n"})
    assert LG.main([str(run), str(tmp_path / "q"), "--hold", str(hold)]) == 2
    assert (hold / "report.json").is_file() and (out / "report.json").read_bytes() == b"{}"
    (out / "report.json").unlink()
    (hold / "n2" / "planted.json").unlink()
    assert LG.main([str(run), str(tmp_path / "q"), "--hold", str(hold), "--gates", str(tmp_path / "G")]) == 0
    assert (out / "job.log").read_bytes() == b"before\nafter\n"


def test_public_bundle_files_are_matched_by_sha256_and_not_scanned(monkeypatch, tmp_path):
    """A record holding the gate's public bundle (the fetch less witness/, as a gate publishes it to any key) and a copy
    of one of its files: each member and the copy equal to a public file is public by its sha256 and never scanned, so
    the bytes scanned are the other records' alone; a member that differs from every public file is scanned and found."""
    g = _gate(tmp_path / "G" / "g1")
    bundle = {f"{GATE}/{p.relative_to(g).as_posix()}": p.read_bytes() for p in sorted(g.rglob("*"))
              if p.is_file() and "witness" not in p.relative_to(g).parts}
    run = tmp_path / "run"
    _tar(run / "out" / "bundle.tgz", bundle)
    _put(run, {"out/gates/certificates.jsonl": (g / "out" / "certificates.jsonl").read_bytes(),
               "out/start.json": b'{"run": "c0"}'})
    scanned = []
    hits = LG.Finder.hits
    monkeypatch.setattr(LG.Finder, "hits", lambda self, data: scanned.append(len(data)) or hits(self, data))
    rc, report, _ = _guard(tmp_path, run)
    assert (rc, report["found"], report["public_files"]) == (0, [], len(bundle) + 1)
    assert sum(scanned) == len(b'{"run": "c0"}')
    _tar(run / "out" / "bundle.tgz", {**bundle, f"{GATE}/out/extra.json": b"[" + _sig_piece()[0] + b"]"})
    rc, report, _ = _guard(tmp_path, run, "--dry")
    assert rc == 3 and [d["path"] for d in report["found"]] == ["out/bundle.tgz"] and report["public_files"] == len(bundle) + 1


def test_the_guard_stops_past_its_budget_and_releases_nothing(monkeypatch, tmp_path):
    """Past --budget-s on the guard's clock (a fake one here, a minute a reading) the guard stops (exit 2): nothing is
    released and the held outputs stay in the hold, out of the record."""
    _gate(tmp_path / "G" / "g1")
    run, out, hold = _held(tmp_path)
    ticks = iter(range(0, 10**6, 60))
    monkeypatch.setattr(LG, "clock", lambda: next(ticks))
    assert _guard_rc(tmp_path, run, "--hold", str(hold), "--budget-s", "100") == 2
    assert not list(out.iterdir()) and (hold / "report.json").is_file() and (hold / "n2" / "planted.json").is_file()


def _guard_rc(tmp: Path, run: Path, *extra: str) -> int:
    return LG.main([str(run), str(tmp / "q"), "--gates", str(tmp / "G"), *extra])


def test_a_guard_ended_by_sigterm_leaves_out_without_the_held_files(tmp_path):
    """A slow guard (here held reading its gates, a FIFO the test closes once it has signalled, so a signal that lands
    just before the guard's read is seen when the read returns) ended by SIGTERM, as a job's time limit ends it, exits
    143 and releases nothing: out/ holds none of the held outputs, which stay in the hold. Without the signal the
    empty gate would be refused (exit 2)."""
    _gate(tmp_path / "G" / "g1")
    run, out, hold = _held(tmp_path)
    fifo = tmp_path / "fetch.tgz"
    os.mkfifo(fifo)
    p = subprocess.Popen([sys.executable, LG.__file__, str(run), str(tmp_path / "q"), "--hold",
                          str(hold), "--gates", str(tmp_path / "G"), "--gates", str(fifo)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(fifo, "wb"):
        p.send_signal(signal.SIGTERM)
    said, _ = p.communicate()
    assert p.returncode == 143, said
    assert "SIGTERM ended the guard: nothing more released" in said
    assert not list(out.iterdir()) and (hold / "report.json").is_file() and (hold / "n2" / "planted.json").is_file()


def test_a_ports_reads_written_compactly_are_no_piece_of_a_records_counts(tmp_path):
    """A record head's bucket counts are witness, and a port's reads are positions, protocol structure. Where both
    begin with a run of zeros, a registered-reads file in JSON's default form holds a piece of the witness line, a
    key's end and ten zeros, and the guard holds it though nothing of the witness is in it. The verifier's
    registered-reads files are written without spaces (`network_attester_rows.reads_text`), and the same reads are then no piece
    of the witness. The counts as the witness writes them are a find wherever they are written."""
    from verity.protocols.resource_accounting.network.verifier import network_attester_rows as WR

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
    """The attester's run record, a witness file written indented, holds its processes' CPUs. Written as a list of
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


def test_the_declared_data_is_well_formed():
    """leak_guard.json, the guard's declared names and allowlist, loads (so none of its entries excuses a piece by a
    key path, which the loader refuses), and its names are the gates' witness files that no record file of the chain
    is named after."""
    names, allow = LG.load_declared(LG.DECLARED)
    assert {"cells.salts", "certified.jsonl", "requests.records", "responses.records"} <= names
    assert not names & {"end.json", "request.json", "deployment.json", "cells.jsonl", "certificates.jsonl", "serve.log",
                        "check.json", "resources.jsonl"}
    assert all(e["witness"] in names for e in allow)


CERTIFICATE = {"key_id": "k", "link": 0, "direction": "ingress", "window": 0, "digest": "00", "signature": "11"}


def _keys_guard(tmp: Path, *extra: str, declared: dict | None = None) -> tuple[int, dict | None, dict | None]:
    """The guard on run dir tmp/run with ``--keys``: its exit, its report and the quarantine's (None if unwritten)."""
    keys = ["--keys"]
    if declared is not None:
        (tmp / "declared.json").write_text(json.dumps(declared))
        keys.append(str(tmp / "declared.json"))
    report = tmp / "report.json"
    if "--report" in extra:
        report = Path(extra[extra.index("--report") + 1])
        extra = tuple(x for x in extra if x not in ("--report", str(report)))
    report.unlink(missing_ok=True)
    rc = LG.main([str(tmp / "run"), str(tmp / "q"), *keys, "--report", str(report), *extra])
    held = tmp / "q" / "leak-guard.json"
    return (rc, json.loads(report.read_text()) if report.is_file() else None,
            json.loads(held.read_text()) if held.is_file() else None)


def test_a_released_file_with_one_undeclared_key_fails_the_record(tmp_path):
    """The control: a record whose released files hold declared keys alone passes the guard with its counts; the same
    record with one undeclared key in one released file fails (exit 3), the file moved to the quarantine. The report
    that is released names the file and how many keys, the quarantine's names the key's path with the fix, and a file
    outside out/ is counted and not read by key."""
    run = tmp_path / "run"
    certs = "out/gates/c0/r0/out/certificates.jsonl"
    _put_json(run, {certs: (json.dumps({"certificate": CERTIFICATE}) + "\n").encode(),
               "out/n2/outcome.json": {"verdict": "accept"}, "out/n2/prove.log": b"text", "resources.jsonl": b"{}\n"})
    rc, report, _ = _keys_guard(tmp_path)
    assert rc == 0 and report["verdict"] == "clean" and report["found"] == []
    assert report["keys"] == {"declared": RK.SCHEMA, "checked": 1, "not_yet": {"n2": 1}, "no_keys": {".log": 1},
                              "outside": 1, "undeclared": 0}
    _put_json(run, {certs: {"certificate": {**CERTIFICATE, "undeclared_key": "a value of any length " * 40}}})
    rc, report, held = _keys_guard(tmp_path)
    assert rc == 3 and report["verdict"] == "refused"
    assert report["found"] == [{"path": certs, "bytes": (tmp_path / "q" / "files" / certs).stat().st_size,
                                "why": ["1 key not declared public (--keys)"], "pieces": 0}]
    assert report["keys"]["undeclared"] == 1 and not (run / certs).exists()
    assert "undeclared_key" not in (tmp_path / "report.json").read_text()
    [bad] = held["found"][0]["keys_undeclared"]
    assert bad["key"] == "certificate/undeclared_key" and bad["count"] == 1
    assert "is not declared public" in bad["fix"] and "auditor.py" in bad["fix"]
    rc, report, _ = _keys_guard(tmp_path)
    assert rc == 0 and report["keys"]["checked"] == 0
    rc, report, _ = LG.main([str(run), str(tmp_path / "q"), "--report", str(tmp_path / "report.json")]), json.loads(
        (tmp_path / "report.json").read_text()), None
    assert rc == 0 and "keys" not in report


def test_a_gate_runs_file_that_says_an_execution_phase_was_refused_fails_the_record(tmp_path):
    """Which execution phases the signer refused depends on what the node sent and when, and no list declares the
    key that says it. So a record of a gate run with one refused line fails (exit 3) on that file alone, by the key
    and its count, and a record is released only of a run whose every execution phase was certified."""
    run = tmp_path / "run"
    files = [f"out/gates/c0/r0/out/{name}" for name in ("certificates.jsonl", "cells-certificates.jsonl",
                                                         "rows-certificates.jsonl", "cells-registrations.jsonl")]
    refused = (json.dumps({"refused": ["l", "egress", 1]}) + "\n").encode()
    _put_json(run, {files[0]: (json.dumps({"certificate": CERTIFICATE}) + "\n").encode(), **dict.fromkeys(files[1:], b"")})
    rc, report, _ = _keys_guard(tmp_path)
    assert rc == 0 and report["keys"]["checked"] == 4 and report["keys"]["undeclared"] == 0
    for path in files:
        (run / path).write_bytes((run / path).read_bytes() + refused)
    rc, report, held = _keys_guard(tmp_path)
    assert rc == 3 and report["verdict"] == "refused" and report["keys"]["undeclared"] == 4
    assert sorted(f["path"] for f in report["found"]) == sorted(files)
    for found in held["found"]:
        [bad] = found["keys_undeclared"]
        assert (bad["key"], bad["count"]) == ("refused", 1) and "is not declared public" in bad["fix"]
    assert not any((run / path).exists() for path in files)


def test_a_held_file_with_an_undeclared_key_is_not_released(tmp_path, capsys):
    """With --hold, a held file is read by key at the record path it is bound for: one with an undeclared key or under
    no list goes to the quarantine and never to out/, the rest is released, and the guard's line says how many files
    it checked by key and how many wait."""
    run, hold = tmp_path / "run", tmp_path / "hold"
    (run / "out").mkdir(parents=True)
    _put_json(hold, {"step/a.json": GOOD, "step/b.json": {**GOOD, "rounds": [{"salt": "00"}, {"salt": "01"}]},
                "scan/summary.json": {"admitted": 3}, "other/x.json": {"k": 1}, "step/prove.log": b"text"})
    rc, report, held = _keys_guard(tmp_path, "--hold", str(hold), declared=_declared(LIST, not_yet=("scan/**",)))
    assert rc == 3 and (report["held"], report["released"]) == (5, 3)
    assert {d["path"]: d["why"] for d in report["found"]} == {
        "out/other/x.json": ["1 key not declared public (--keys)"],
        "out/step/b.json": ["1 key not declared public (--keys)"]}
    assert sorted(p.relative_to(run / "out").as_posix() for p in (run / "out").rglob("*") if p.is_file()) == [
        "scan/summary.json", "step/a.json", "step/prove.log"]
    assert report["keys"] == {"declared": RK.SCHEMA, "checked": 2, "not_yet": {"scan": 1}, "no_keys": {".log": 1},
                              "outside": 0, "undeclared": 2}
    by_path = {d["path"]: d["keys_undeclared"] for d in held["found"]}
    assert [(b["key"], b["count"]) for b in by_path["out/step/b.json"]] == [("rounds/*/salt", 2)]
    assert by_path["out/other/x.json"][0]["key"] == "(the file)"
    assert "2 files checked by key, 2 with a key not declared public; 1 not yet checked (scan 1); 1 with no keys" in (
        capsys.readouterr().out)


def test_the_guards_own_report_is_read_by_key_before_anything_is_released(tmp_path, capsys):
    """A report the guard writes under out/ is a released file too: where its list doesn't declare each key it holds,
    or no list and no ``not_yet`` glob names it, the guard is refused (exit 2) with nothing released or moved and no
    report written; where its list isn't written yet it is counted with the files that wait. Declared data that
    doesn't load refuses the guard too."""
    run, hold = tmp_path / "run", tmp_path / "hold"
    (run / "out").mkdir(parents=True)
    _put_json(hold, {"step/a.json": GOOD})
    own = {"name": "the guard's report", "writer": "leak_guard.py", "files": ["leak-guard.json"], "form": "json",
           "keys": {"schema": "structure", "verdict": "output"}}
    at = ("--hold", str(hold), "--report", str(run / "out" / "leak-guard.json"))
    for declared, why in ((_declared(LIST, own), "holds keys the list \"the guard's report\""),
                          (_declared(LIST), "no list of released_keys.json names this file"),
                          ({**_declared(LIST), "allow": []}, "is not network-traces/released-keys/v1")):
        rc, report, _ = _keys_guard(tmp_path, *at, declared=declared)
        assert rc == 2 and report is None and why in capsys.readouterr().err
        assert (hold / "step" / "a.json").is_file() and list((run / "out").iterdir()) == []
    rc, report, _ = _keys_guard(tmp_path, *at, declared=_declared(LIST, not_yet=("leak-guard.json",)))
    assert rc == 0 and report["keys"]["checked"] == 1 and report["keys"]["not_yet"] == {RK.TOP: 1}
    assert report["released"] == 1 and (run / "out" / "step" / "a.json").is_file()
