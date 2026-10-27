"""wipe_gate.sh against a stub gate: the lane's own `fetch` (with witness/) is kept 0700 under KEEP, outside the run dir,
and $OUT, which custody publishes, holds only the bundle less witness/, which `wipe_egress.read_run` still reads; a KEEP
inside the run dir is refused before anything is sent."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
RUN = "c20261027T160500Z-0000"

STUB = r'''#!/usr/bin/env python3
import io, json, sys, tarfile
cmd = sys.argv[-1]
if cmd.startswith("warden-rehearse "):
    sys.stdin.read()
    print(json.dumps({"run": "%(run)s", "started": True}))
elif cmd == "status %(run)s":
    print(json.dumps({"run": "%(run)s", "end": {"run": "%(run)s", "rc": 0, "ended_utc": "2026-10-27T17:05:00Z",
                                              "verdict": {"accepted": True}}}))
elif cmd == "fetch %(run)s":
    files = {"end.json": b"{}", "request.json": b"{}", "out/certificates.jsonl": b"",
             "out/deployment.json": json.dumps({"windows": 3, "origin_unix": 1.5,
                                                "params": {"window": 32, "bucket_ms": 100}}).encode(),
             "witness/certified.jsonl": b"salt", "witness/prompts.json": b"[]"}
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
''' % {"run": RUN}


def _gate(tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    (bin_ / "ssh").write_text(STUB.replace("/usr/bin/env python3", sys.executable, 1))
    (bin_ / "ssh").chmod(0o755)
    out = tmp_path / "run" / "out"
    out.mkdir(parents=True, exist_ok=True)
    return subprocess.run(["bash", str(HERE / "wipe_gate.sh")], cwd=ROOT, capture_output=True, text=True,
                          env={"PATH": f"{bin_}{os.pathsep}{os.path.dirname(sys.executable)}{os.pathsep}{os.defpath}",
                               "HOME": str(tmp_path), "OUT": str(out), "RESEARCH_RUN_DIR": str(tmp_path / "run"),
                               "GATE_KEY": str(tmp_path / "key"), "REQUESTS": "4", **env})


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
    published = sorted(p.relative_to(tmp_path / "run" / "out").as_posix()
                       for p in (tmp_path / "run" / "out").rglob("*") if p.is_file())
    assert published == ["bundle.tgz", "start.json", "status.json"]
    with tarfile.open(tmp_path / "run" / "out" / "bundle.tgz") as tar:
        names = tar.getnames()
    assert not [n for n in names if "witness" in n.split("/")]
    assert f"{RUN}/out/deployment.json" in names and f"{RUN}/request.json" in names
    sys.path.insert(0, str(HERE))
    import wipe_egress
    assert sorted(wipe_egress.read_run(tmp_path / "run" / "out" / "bundle.tgz")) == sorted(wipe_egress.FILES)


def test_keep_inside_the_run_dir_is_refused(tmp_path):
    got = _gate(tmp_path, KEEP=str(tmp_path / "run" / "gate"))
    assert got.returncode == 2 and "inside the run dir" in got.stdout
    assert not (tmp_path / "run" / "out" / "start.json").exists()
