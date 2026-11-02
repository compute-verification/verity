"""wipe_gate.sh against a stub gate: the lane's own `fetch` (with witness/) is kept 0700 under KEEP, outside the run dir,
and $OUT, which custody publishes, holds only the bundle less witness/, which `wipe_egress.read_run` still reads; the
gate's start reply, which names the unit the witness records, stays in KEEP/RUN; a KEEP inside the run dir is refused
before anything is sent. Every output is held under KEEP until the leak guard passes it: one holding a witness value is
refused, and a SIGTERM releases nothing."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
RUN = "c20261027T160500Z-0000"
#: a witness value the stub's status plants (STUB_NOTE), 64 hex characters: two of the guard's text pieces
PLANTED = hashlib.sha256(b"a certified line").hexdigest()
#: the warden's stall probe's record of its cgroup, as the gate's unit runs it: the unit's name begins one of the guard's
#: aligned text pieces (`leak_guard.TEXT` bytes)
_CGROUP = '{"cgroups": {"warden": {"path": "/sys/fs/cgroup/system.slice/system-vy\\\\x2dcertifier.slice/'
STALLS = " " * (-len(_CGROUP) % 32) + _CGROUP + f'vy-certifier@{RUN}.service"}}}}}}\n'

STUB = r'''#!/usr/bin/env python3
import io, json, os, signal, sys, tarfile
cmd = sys.argv[-1]
if cmd.startswith("warden-rehearse "):
    sys.stdin.read()
    if os.environ.get("STUB_REFUSE"):
        print(json.dumps({"refused": "another rehearsal still runs"}))
        sys.exit(1)
    run = os.environ.get("STUB_RUN", "%(run)s")
    print(json.dumps({"run": run, "started": True, "unit": f"vy-certifier@{run}.service"}))
elif cmd.startswith("status "):
    if os.environ.get("STUB_FIFO"):
        open(os.environ["STUB_FIFO"], "w")
        signal.pause()
    print(json.dumps({"run": "%(run)s", "note": os.environ.get("STUB_NOTE"),
                      "end": {"run": "%(run)s", "rc": 0, "ended_utc": "2026-10-27T17:05:00Z",
                              "verdict": {"accepted": True}}}))
elif cmd == "fetch %(run)s":
    files = {"end.json": b"{}", "request.json": b"{}", "out/certificates.jsonl": b"",
             "out/deployment.json": json.dumps({"windows": 3, "origin_unix": 1.5,
                                                "params": {"window": 32, "bucket_ms": 100}}).encode(),
             "witness/certified.jsonl": b"%(planted)s\n", "witness/prompts.json": b"[]",
             "witness/stalls.json": %(stalls)r.encode()}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in ("", "/out", "/witness"):
            info = tarfile.TarInfo("%(run)s" + d)
            info.type = tarfile.DIRTYPE
            tar.addfile(info)
        for name, data in files.items():
            info = tarfile.TarInfo("%(run)s/" + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    sys.stdout.buffer.write(buf.getvalue())
else:
    sys.exit(f"stub gate: no verb {cmd!r}")
''' % {"run": RUN, "planted": PLANTED, "stalls": STALLS}


def _gate(tmp_path: Path, popen: bool = False, **env: str):
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    (bin_ / "ssh").write_text(STUB.replace("/usr/bin/env python3", sys.executable, 1))
    (bin_ / "ssh").chmod(0o755)
    out = tmp_path / "run" / "out"
    out.mkdir(parents=True, exist_ok=True)
    env = {"PATH": f"{bin_}{os.pathsep}{os.path.dirname(sys.executable)}{os.pathsep}{os.defpath}",
           "HOME": str(tmp_path), "OUT": str(out), "RESEARCH_RUN_DIR": str(tmp_path / "run"), "RESEARCH_RUN_ID": "r1",
           "GATE_KEY": str(tmp_path / "key"), "REQUESTS": "4", "PY": sys.executable, **env}
    if popen:
        return subprocess.Popen(["bash", str(HERE / "wipe_gate.sh")], cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=env, start_new_session=True)
    return subprocess.run(["bash", str(HERE / "wipe_gate.sh")], cwd=ROOT, capture_output=True, text=True, env=env)


def _published(tmp_path: Path) -> list[str]:
    return sorted(p.relative_to(tmp_path / "run" / "out").as_posix()
                  for p in (tmp_path / "run" / "out").rglob("*") if p.is_file())


def test_record_keeps_no_witness(tmp_path):
    keep = tmp_path / "keep"
    got = _gate(tmp_path, KEEP=str(keep))
    assert got.returncode == 0, got.stdout + got.stderr
    assert json.loads(got.stdout.splitlines()[-1])["windows"] == 3
    kept = keep / RUN / "fetch.tar.gz"
    assert (keep.stat().st_mode & 0o777, (keep / RUN).stat().st_mode & 0o777, kept.stat().st_mode & 0o077) == \
        (0o700, 0o700, 0)
    with tarfile.open(kept) as tar:
        assert f"{RUN}/witness/certified.jsonl" in tar.getnames()
    assert _published(tmp_path) == ["bundle.tgz", "leak-guard.json", "status.json", "summary.json", "wipe-gate.log"]
    report = json.loads((tmp_path / "run" / "out" / "leak-guard.json").read_text())
    assert (report["verdict"], report["gate_runs"], report["held"], report["released"]) == ("clean", [RUN], 4, 4)
    assert list((keep / "held" / "r1").iterdir()) == []
    with tarfile.open(tmp_path / "run" / "out" / "bundle.tgz") as tar:
        names = tar.getnames()
    assert not [n for n in names if "witness" in n.split("/")]
    assert f"{RUN}/out/deployment.json" in names and f"{RUN}/request.json" in names
    sys.path.insert(0, str(HERE))
    import wipe_egress
    assert sorted(wipe_egress.read_run(tmp_path / "run" / "out" / "bundle.tgz")) == sorted(wipe_egress.FILES)


def test_the_gates_start_reply_stays_in_keep_since_it_names_the_unit_the_warden_records(tmp_path):
    """The gate's start reply names the run's systemd unit, vy-certifier@RUN.service, as the warden's stall probe
    records its cgroup in the witness (stalls.json), and no public file of the run holds it: in the hold, the leak guard
    holds the reply (and the log it was echoed to) as a piece of the witness, and the job exited 3. The reply goes to
    KEEP/RUN alone (0600), so the job exits on the gate's rc."""
    keep = tmp_path / "keep"
    got = _gate(tmp_path, KEEP=str(keep))
    assert got.returncode == 0, got.stdout + got.stderr
    reply, out = keep / RUN / "start.json", tmp_path / "run" / "out"
    assert json.loads(reply.read_text())["unit"] == f"vy-certifier@{RUN}.service"
    assert reply.stat().st_mode & 0o777 == 0o600 and "start.json" not in _published(tmp_path)
    assert "vy-certifier@" not in (out / "wipe-gate.log").read_text()
    assert json.loads((out / "leak-guard.json").read_text())["found"] == []
    published, q = tmp_path / "published", tmp_path / "q"
    published.mkdir()
    shutil.copyfile(reply, published / "start.json")
    held = subprocess.run([sys.executable, str(HERE / "leak_guard.py"), str(published), str(q), "--gates",
                           str(keep / RUN / "fetch.tar.gz"), "--dry", "--report", str(tmp_path / "report.json")],
                          capture_output=True, text=True)
    assert held.returncode == 3, held.stdout + held.stderr
    (piece,) = [p for f in json.loads((q / "leak-guard.json").read_text())["found"] for p in f["pieces_found"]]
    assert (piece["witness"], piece["keypath"]) == (f"{RUN}/witness/stalls.json", ["unit"])


def test_a_refused_start_is_a_visible_failure_and_keeps_nothing(tmp_path):
    keep = tmp_path / "keep"
    got = _gate(tmp_path, KEEP=str(keep), STUB_REFUSE="1")
    assert got.returncode == 1, got.stdout + got.stderr
    log = (tmp_path / "run" / "out" / "wipe-gate.log").read_text()
    assert "the gate didn't start the session (exit 1)" in log and "another rehearsal still runs" in log
    assert not (keep / RUN).exists() and "start.json" not in _published(tmp_path)


def test_a_start_naming_no_certifier_run_writes_nothing_under_its_name(tmp_path):
    """The run names a directory under KEEP, so the job holds it to the auditor's `CERTIFIER_RUN`, as the front does."""
    sys.path.insert(0, str(HERE.parent / "live_challenger"))
    import auditor as A
    assert f'=~ ^{A.CERTIFIER_RUN.pattern.replace(r"\d", "[0-9]")}$ ]]' in (HERE / "wipe_gate.sh").read_text()
    keep = tmp_path / "keep"
    got = _gate(tmp_path, KEEP=str(keep), STUB_RUN="../escape")
    assert got.returncode == 1, got.stdout + got.stderr
    assert "the gate didn't start the session (exit 0)" in (tmp_path / "run" / "out" / "wipe-gate.log").read_text()
    assert not (tmp_path / "escape").exists() and not list(keep.glob("*/start.json"))


def test_keep_inside_the_run_dir_is_refused(tmp_path):
    got = _gate(tmp_path, KEEP=str(tmp_path / "run" / "gate"))
    assert got.returncode == 2 and "inside the run dir" in got.stdout
    assert not (tmp_path / "run" / "out" / "start.json").exists()


def test_an_output_holding_a_witness_value_is_refused(tmp_path):
    """The gate's status planting a witness value: the guard moves status.json to the quarantine and fails the job (exit
    3); the rest of the hold is released."""
    keep = tmp_path / "keep"
    got = _gate(tmp_path, KEEP=str(keep), STUB_NOTE=PLANTED)
    assert got.returncode == 3, got.stdout + got.stderr
    assert "leak guard: refused" in got.stdout and "out/status.json: 2 witness pieces" in got.stdout
    assert "status.json" not in _published(tmp_path) and "bundle.tgz" in _published(tmp_path)
    assert (keep / "quarantine" / "r1" / "files" / "out" / "status.json").is_file()
    assert PLANTED not in "".join(p.read_text(errors="replace") for p in (tmp_path / "run").rglob("*") if p.is_file()
                                  and p.suffix != ".tgz")


def test_sigterm_mid_session_releases_nothing(tmp_path):
    """A timeout's SIGTERM to the job's process group while it polls the gate (its `status` held on a FIFO the test opens)
    ends the job (exit 143) with nothing released: $OUT stays empty and the outputs stay in the hold."""
    keep, fifo = tmp_path / "keep", tmp_path / "status.fifo"
    os.mkfifo(fifo)
    p = _gate(tmp_path, popen=True, KEEP=str(keep), STUB_FIFO=str(fifo))
    with open(fifo, "rb"):
        os.killpg(p.pid, signal.SIGTERM)
        said, _ = p.communicate()
    assert p.returncode == 143, said
    assert "SIGTERM ended the run before the leak guard read it: nothing released" in said
    assert _published(tmp_path) == []
    assert sorted(x.name for x in (keep / "held" / "r1").iterdir()) == ["status.json", "wipe-gate.log"]
    assert (keep / RUN / "start.json").is_file()
