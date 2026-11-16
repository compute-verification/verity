"""chain_record.sh and the leak guard: it moves a gate sent under inputs/ to the quarantine once unpacked, runs the
guard on every exit and is refused by it (exit 3), and withholds out/, inputs/ and the logs whole on a signal or when
the guard can't run."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from tests.benchmarks.network_traces.gate_runs import GATE, _gate, _put, _sig_piece, _tar

ROOT = Path(__file__).resolve().parents[3]


def _chain(tmp: Path, gates: str, *, py: str = sys.executable, **kw) -> subprocess.Popen:
    """chain_record.sh on run dir tmp/runs/r1 (its quarantine by default tmp/quarantine/r1), its gates `gates`."""
    env = {"PATH": os.defpath, "HOME": str(tmp), "TMPDIR": str(tmp / "tmp"), "RESEARCH_RUN_DIR": str(tmp / "runs/r1"),
           "CR_MEM_GB": "1", "CR_GATES": gates, "CR_CACHE": str(tmp / "cache"), "PY": py}
    (tmp / "tmp").mkdir(exist_ok=True)
    return subprocess.Popen(["bash", "benchmarks/network_traces/chain_record.sh"], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)


def _sent(tmp: Path) -> Path:
    """A run dir as a job sent gate.tgz leaves it, with the runner's logs and a record that leaked a piece."""
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
    assert "leak guard: refused" in out and "out/n2/leak.json: 1 witness pieces" in out
    log = (run / "out" / "job.log").read_text()
    assert "gate 1's inputs/gate.tgz, which holds its witness, moved to the quarantine" in log
    assert "chain: no tarball absent.tgz" in (run / "out" / "stdout.log").read_text()
    assert (q / "files" / "inputs" / "gate.tgz").is_file() and (q / "files" / "out" / "n2" / "leak.json").is_file()
    assert not (run / "inputs" / "gate.tgz").exists() and not (run / "out" / "n2" / "leak.json").exists()
    report = json.loads((run / "out" / "leak-guard.json").read_text())
    assert report["gate_runs"] == [GATE] and [d["path"] for d in report["found"]] == ["out/n2/leak.json"]
    assert report["held"] == report["released"] == 2 and list((q / "held").iterdir()) == []
    assert list((tmp_path / "tmp").iterdir()) == []


def test_chain_record_fails_on_a_released_file_with_an_undeclared_key(tmp_path):
    """chain_record.sh's guard reads by key what it releases (released_keys.py): the record's files with declared
    keys alone pass it (the job itself still fails, at the gate that isn't there), counted beside the files whose
    lists aren't written yet and those with no keys; with one undeclared key in one of them the guard refuses the
    record, the file in the quarantine and named in the job's log with how many keys, the key's own name in the
    quarantine's report alone."""
    run = _sent(tmp_path)
    (run / "out" / "n2" / "leak.json").unlink()
    certs = "out/gates/c0/r0/out/certificates.jsonl"
    line = {"certificate": {"key_id": "k", "link": 0, "direction": "ingress", "window": 0, "digest": "0", "signature": "1"}}
    _put(run, {certs: (json.dumps(line) + "\n").encode(), "out/n2/outcome.json": b'{"verdict": "accept"}'})
    p = _chain(tmp_path, "gate.tgz absent.tgz")
    out, _ = p.communicate()
    assert "leak guard: clean" in out and (run / certs).is_file(), out
    assert "1 files checked by key, 0 with a key not declared public; 2 not yet checked ((top level) 1, n2 1)" in out
    assert json.loads((run / "out" / "leak-guard.json").read_text())["keys"]["no_keys"] == {".log": 2}
    for f in ("job.log", "stdout.log", "leak-guard.json"):
        (run / "out" / f).unlink()
    _tar(run / "inputs" / "gate.tgz", {f"{GATE}/{p.relative_to(tmp_path / 'src' / GATE).as_posix()}": p.read_bytes()
                                       for p in sorted((tmp_path / "src" / GATE).rglob("*")) if p.is_file()})
    line["certificate"]["undeclared_key"] = 7
    _put(run, {certs: (json.dumps(line) + "\n").encode()})
    p = _chain(tmp_path, "gate.tgz absent.tgz")
    out, _ = p.communicate()
    q = tmp_path / "quarantine" / "r1"
    assert p.returncode == 3, out
    assert "leak guard: refused" in out and f"{certs}: 1 key not declared public (--keys)" in out
    assert (q / "files" / certs).is_file() and not (run / certs).exists()
    assert "undeclared_key" not in out + (run / "out" / "leak-guard.json").read_text()
    [found] = json.loads((q / "leak-guard.json").read_text())["found"]
    assert [b["key"] for b in found["keys_undeclared"]] == ["certificate/undeclared_key"]


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


def test_chain_record_releases_nothing_when_sigterm_ends_its_guard(tmp_path):
    """A timeout's SIGTERM to the job's process group while the guard reads (held here unpacking one more gate, a FIFO the
    test closes once it has signalled) ends the guard and the job (exit 143) with nothing released: the run dir is left
    empty, and the job's outputs and logs stay in the hold, out of the record."""
    run = _sent(tmp_path)
    fifo = tmp_path / "slow.tgz"
    os.mkfifo(fifo)
    py = tmp_path / "py"
    py.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@" --gates "{fifo}"\n')
    py.chmod(0o755)
    p = _chain(tmp_path, "gate.tgz absent.tgz", py=str(py), start_new_session=True)
    with open(fifo, "wb"):
        os.killpg(p.pid, signal.SIGTERM)
    out, _ = p.communicate()
    q = tmp_path / "quarantine" / "r1"
    assert p.returncode == 143, out
    assert "SIGTERM ended the leak guard (exit 143)" in out and "nothing released from" in out
    assert sorted(x.name for x in run.iterdir()) == []
    assert sorted(x.name for x in (q / "held").iterdir()) == ["job.log", "stdout.log"]
    assert not (q / "files" / "out" / "job.log").exists()
