"""The hourly Lean prune (`vy-lean-prune`): the files deploy.toml installs for it are all it needs to run."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NEBIUS = REPO / "tools/research/src/research/pods/nebius"
LIB = "/usr/local/lib/vy-monitoring/lean-audit/"


def test_the_files_deploy_toml_installs_run_the_hourly_lean_prune(tmp_path):
    """Catches lean_audit.py loading a module deploy.toml doesn't install beside it: every hourly prune on every node then fails
    at import, and the records trees it caps fill the nodes' disks."""
    manifest = tomllib.loads((NEBIUS / "deploy.toml").read_text())
    files = {e["path"]: e for e in manifest["file"] if e["path"].startswith(LIB)}
    argv = re.search(r"^ExecStart=(.+)$", (NEBIUS / "vy-lean-prune.service").read_text(), re.M)[1].split()
    assert argv[1:] == [LIB + "tools/verity/lean/lean_audit.py", "--prune"] and argv[1] in files
    service = next(e for e in manifest["file"] if e["path"] == "/etc/systemd/system/vy-lean-prune.service")
    assert all(e["nodes"] == service["nodes"] for e in files.values()), "every node that runs the prune has all its files"
    lib = tmp_path / "lib"
    for path, e in files.items():
        dest = lib / path[len(LIB):]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO / e["src"]).read_bytes())
    records = tmp_path / "cache" / "lean-records"
    for name in ("a", "b", "c", "d"):
        (records / name).mkdir(parents=True)
        (records / name / ".lock").touch()
        os.utime(records / name / ".lock", (1, ord(name)))
    r = subprocess.run([sys.executable, "-S", str(lib / argv[1][len(LIB):]), "--prune", "--cache", str(tmp_path / "cache" / "lean-audit")],
                       capture_output=True, text=True, cwd=tmp_path,
                       env={"HOME": str(tmp_path), "LEAN_SLOTS": str(tmp_path / "no-slots")})
    assert r.returncode == 0, r.stderr
    assert "1 from lean-records" in r.stdout and sorted(p.name for p in records.iterdir()) == ["b", "c", "d"], r.stdout
