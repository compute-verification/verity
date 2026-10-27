"""wipe_gate_guard.sh against wipe_gate.sh's stub gate: on its way out the leak guard holds the run dir against the kept
fetch's witness, unpacked 0700 under KEEP. A clean record keeps $OUT whole with the guard's report beside it; a record
whose gate answer carries a piece of the witness has that file moved to KEEP's quarantine, and the run exits 3."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_wipe_gate as WG  # noqa: E402

SALT = "".join(f"{(i * 37 + 11) % 256:02x}" for i in range(64))


def _stub(leak: bool) -> str:
    stub = WG.STUB.replace('"witness/certified.jsonl": b"salt"',
                           f'"witness/certified.jsonl": b\'{{"salt": "{SALT}"}}\\n\'')
    assert stub != WG.STUB
    if leak:
        stub = stub.replace('"verdict": {"accepted": True}}', f'"verdict": {{"accepted": True}}, "note": "{SALT}"}}')
    return stub


def _guarded(tmp_path: Path, leak: bool) -> tuple[subprocess.CompletedProcess, Path, Path]:
    bin_, out, keep = tmp_path / "bin", tmp_path / "run" / "out", tmp_path / "keep"
    bin_.mkdir()
    (bin_ / "ssh").write_text(_stub(leak).replace("/usr/bin/env python3", sys.executable, 1))
    (bin_ / "ssh").chmod(0o755)
    out.mkdir(parents=True)
    got = subprocess.run(["bash", str(HERE / "wipe_gate_guard.sh")], cwd=ROOT, capture_output=True, text=True,
                         env={"PATH": f"{bin_}{os.pathsep}{os.path.dirname(sys.executable)}{os.pathsep}{os.defpath}",
                              "HOME": str(tmp_path), "OUT": str(out), "RESEARCH_RUN_DIR": str(tmp_path / "run"),
                              "GATE_KEY": str(tmp_path / "key"), "REQUESTS": "4", "KEEP": str(keep)})
    return got, out, keep / WG.RUN


def test_a_clean_record_keeps_its_bundle_and_the_guards_report(tmp_path):
    got, out, kept = _guarded(tmp_path, leak=False)
    assert got.returncode == 0, got.stdout + got.stderr
    assert "leak guard: clean" in got.stdout
    assert sorted(p.name for p in out.iterdir()) == ["bundle.tgz", "leak-guard.json", "start.json", "status.json"]
    report = json.loads((out / "leak-guard.json").read_text())
    assert report["verdict"] == "clean" and report["gate_runs"] == [WG.RUN] and report["pieces"] > 0
    assert SALT not in (out / "leak-guard.json").read_text()
    assert ((kept / "gate").stat().st_mode & 0o777, (kept / "quarantine").stat().st_mode & 0o777) == (0o700, 0o700)


def test_a_record_holding_a_piece_of_the_witness_is_refused_and_the_file_quarantined(tmp_path):
    got, out, kept = _guarded(tmp_path, leak=True)
    assert got.returncode == 3, got.stdout + got.stderr
    assert "leak guard: refused" in got.stdout and SALT not in got.stdout
    assert not (out / "status.json").exists() and (kept / "quarantine" / "files" / "out" / "status.json").exists()
    assert json.loads((out / "leak-guard.json").read_text())["found"][0]["path"] == "out/status.json"
