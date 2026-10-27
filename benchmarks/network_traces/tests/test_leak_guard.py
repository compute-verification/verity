"""`leak_guard`: before custody publishes a run dir, each file in it that holds a gate run's witness (under a witness/
directory or by a declared witness name, a witness piece raw or as hex at any offset, an archive member that does, or an
archive the guard can't read) is moved to the quarantine outside the run dir, and the report names it by path alone; the
gate's public files, their JSON written again and a witness file equal to a public one are no finds, and a declared
allowlist entry excuses its piece (with ``*``, any piece of its witness file) only in the records it names. chain_record.sh moves a gate sent under inputs/ to the
quarantine once unpacked, runs the guard on every exit and is refused by it (exit 3), and withholds out/, inputs/ and
the logs whole on a signal or when the guard can't run."""
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
import tarfile
import zipfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import leak_guard as LG  # noqa: E402

GATE = "c20261027T000000Z-0000"
_rnd = random.Random(7)
SALTS = _rnd.randbytes(16 * 8)
LINES = [{"window": w, "head": _rnd.randbytes(32).hex(), "sig": _rnd.randbytes(96).hex()} for w in range(4)]
CERTIFIED = "".join(json.dumps(d) + "\n" for d in LINES).encode()
DEPLOYMENT = json.dumps({"session": "s0", "windows": 4}).encode()


def _sig_piece() -> tuple[bytes, int]:
    """An aligned piece of certified.jsonl inside the first line's signature, which no public file holds."""
    at = CERTIFIED.index(LINES[0]["sig"].encode())
    o = -(-at // LG.TEXT) * LG.TEXT
    return CERTIFIED[o:o + LG.TEXT], o


#: A witness file holding a public certificate as a verifier writes it, indented: public only through `reprinted`.
INDENTED = json.dumps({"window": LINES[0]["window"], "head": LINES[0]["head"]}, indent=1).encode()


def _gate(root: Path) -> Path:
    """A gate run: its witness (the certified lines, the cells' salts, a certificate indented, a deployment equal to the
    public one) beside its public files (the certificates, each line's window and head, the deployment, end.json and
    request.json)."""
    g = root / GATE
    for name, b in {"witness/certified.jsonl": CERTIFIED, "witness/cells.salts": SALTS, "witness/check.json": INDENTED,
                    "witness/deployment.json": DEPLOYMENT, "out/deployment.json": DEPLOYMENT,
                    "out/certificates.jsonl": "".join(json.dumps({"window": d["window"], "head": d["head"]}) + "\n"
                                                      for d in LINES).encode(),
                    "end.json": b'{"files": {}}', "request.json": b'{"minutes": 1}'}.items():
        (g / name).parent.mkdir(parents=True, exist_ok=True)
        (g / name).write_bytes(b)
    return g


def _tar(path: Path, members: dict[str, bytes], mode: str = "w:gz") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode) as t:
        for name, b in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(b)
            t.addfile(info, io.BytesIO(b))
    return path


def _put(run: Path, files: dict[str, bytes]) -> None:
    for name, b in files.items():
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_bytes(b)


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
    """A quarantine inside the run dir, a missing gates directory, or a declared entry without its reason or with a
    sha256 that is neither a digest nor ``*`` refuses the guard before it moves anything."""
    run = tmp_path / "run"
    _put(run, {"out/n2/certified.jsonl": b"{}"})
    assert LG.main([str(run), str(run / "out" / "q")]) == 2
    assert LG.main([str(run), str(tmp_path / "q"), "--gates", str(tmp_path / "none")]) == 2
    assert (run / "out" / "n2" / "certified.jsonl").is_file()
    ok = {"sha256": "0" * 64, "witness": "x", "records": ["*"], "why": "y"}
    for bad in ({**ok, "why": ""}, {**ok, "sha256": "A" * 64}, {**ok, "sha256": "**"}):
        (tmp_path / "bad.json").write_text(json.dumps({"schema": LG.DECLARED_SCHEMA, "names": [], "allow": [ok, bad]}))
        with pytest.raises(SystemExit, match="entry 1 needs"):
            LG.main([str(run), str(tmp_path / "q"), "--declared", str(tmp_path / "bad.json")])
    assert (run / "out" / "n2" / "certified.jsonl").is_file()


@pytest.mark.parametrize("numpy", [True, False])
def test_the_finder_finds_a_needle_at_any_offset_and_across_chunks(monkeypatch, numpy):
    """`Finder.stream` finds each needle at any offset, one straddling two chunks among them, with numpy and in pure
    Python alike, and nothing that only shares a needle's first 8 bytes."""
    if not numpy:
        monkeypatch.setattr(LG, "np", None)
    monkeypatch.setattr(LG, "CHUNK", 64)
    rnd = random.Random(3)
    a, b, c = rnd.randbytes(32), rnd.randbytes(16), rnd.randbytes(16)
    data = rnd.randbytes(50) + a + rnd.randbytes(77) + b + c[:8] + rnd.randbytes(40)
    f = LG.Finder({a, b, c})
    assert f.stream(io.BytesIO(data)) == {a, b} == f.find(data)
    assert f.stream(io.BytesIO(data[:51] + data[52:])) == {b}


def test_the_declared_data_is_well_formed():
    """leak_guard.json, the guard's declared names and allowlist, loads, and its names are the gates' witness files that
    no record file of the chain is named after."""
    names, allow = LG.load_declared(LG.DECLARED)
    assert {"cells.salts", "certified.jsonl", "requests.records", "responses.records"} <= names
    assert not names & {"end.json", "request.json", "deployment.json", "cells.jsonl", "certificates.jsonl", "serve.log",
                        "check.json", "resources.jsonl"}
    assert all(e["witness"] in names for e in allow)


def _chain(tmp: Path, gates: str, *, py: str = sys.executable, **kw) -> subprocess.Popen:
    """chain_record.sh on run dir tmp/runs/r1 (its quarantine by default tmp/quarantine/r1), its gates `gates`."""
    env = {"PATH": os.defpath, "HOME": str(tmp), "TMPDIR": str(tmp / "tmp"), "RESEARCH_RUN_DIR": str(tmp / "runs/r1"),
           "CR_MEM_GB": "1", "CR_GATES": gates, "CR_CACHE": str(tmp / "cache"), "PY": py}
    (tmp / "tmp").mkdir(exist_ok=True)
    return subprocess.Popen(["bash", "benchmarks/network_traces/chain_record.sh"], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)


def _sent(tmp: Path) -> Path:
    """A run dir as `research run --send gate.tgz` leaves it, with the runner's logs and a record that leaked a piece."""
    run = tmp / "runs" / "r1"
    g = _gate(tmp / "src")
    _tar(run / "inputs" / "gate.tgz", {f"{GATE}/{p.relative_to(g).as_posix()}": p.read_bytes()
                                       for p in sorted(g.rglob("*")) if p.is_file()})
    _put(run, {"stdout.log": b"", "stderr.log": b"", "out/n2/leak.json": json.dumps({"x": _sig_piece()[0].decode()}).encode()})
    return run


def test_chain_record_moves_a_sent_gate_out_and_the_guard_refuses_the_run(tmp_path):
    """The gate sent under inputs/ goes to the quarantine once unpacked; the job's exit (here at the next gate, which
    isn't there) runs the guard, which moves the leaked record there too and refuses the run by name, exit 3, with its
    scratch (the unpacked gates) removed."""
    run = _sent(tmp_path)
    p = _chain(tmp_path, "gate.tgz absent.tgz")
    out, _ = p.communicate()
    q = tmp_path / "quarantine" / "r1"
    assert p.returncode == 3, out
    assert "gate 1's inputs/gate.tgz, which holds its witness, moved to the quarantine" in out
    assert "leak guard: refused" in out and "out/n2/leak.json: 1 witness pieces" in out
    assert (q / "files" / "inputs" / "gate.tgz").is_file() and (q / "files" / "out" / "n2" / "leak.json").is_file()
    assert not (run / "inputs" / "gate.tgz").exists() and not (run / "out" / "n2" / "leak.json").exists()
    report = json.loads((run / "out" / "leak-guard.json").read_text())
    assert report["gate_runs"] == [GATE] and [d["path"] for d in report["found"]] == ["out/n2/leak.json"]
    assert list((tmp_path / "tmp").iterdir()) == []


def test_chain_record_withholds_the_record_when_the_guard_cant_run(tmp_path):
    """A guard that can't run (here its interpreter fails) withholds out/, inputs/ and the logs whole: none is left for
    custody, all are in the quarantine, and the run fails."""
    run = _sent(tmp_path)
    p = _chain(tmp_path, "gate.tgz absent.tgz", py="false")
    out, _ = p.communicate()
    assert p.returncode == 3, out
    assert sorted(x.name for x in run.iterdir()) == []
    assert sorted(x.name for x in (tmp_path / "quarantine" / "r1" / "files").iterdir()) == [
        "inputs", "out", "stderr.log", "stdout.log"]


def test_chain_record_withholds_the_record_on_a_signal(tmp_path):
    """A SIGTERM (a timeout's, which leaves the job seconds) withholds out/, inputs/ and the logs whole rather than start a
    scan it can't finish. The job is held at its first gate, a FIFO it reads until the test closes it."""
    run = _sent(tmp_path)
    fifo = tmp_path / "gate.fifo"
    os.mkfifo(fifo)
    p = _chain(tmp_path, str(fifo))
    with open(fifo, "wb"):
        p.send_signal(signal.SIGTERM)
    out, _ = p.communicate()
    assert p.returncode == 143, out
    assert "SIGTERM ended the run before the leak guard read it" in out
    assert sorted(x.name for x in run.iterdir()) == []
    assert (tmp_path / "quarantine" / "r1" / "files" / "inputs" / "gate.tgz").is_file()
