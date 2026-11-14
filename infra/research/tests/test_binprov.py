"""Installed binaries name the commit that built them (`research.binprov`): recorded at build time beside the binary, read into
job.json `run.binaries` by every run, and a binary replaced without recording shows as `stale` / `missing` instead of passing as
the run's commit."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from research import binprov as BP
from research.cli import main


def _bin(d: Path, name: str, data: bytes) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(data)
    (d / name).chmod(0o755)
    return d / name


def test_record_reads_the_shipped_trees_ready_json_and_snapshot_tells_recorded_stale_missing(tmp_path):
    tree = tmp_path / "src" / "c0ffee"
    (tree / "archive" / "sp1").mkdir(parents=True)
    (tree / "READY.json").write_text(json.dumps({"identity": "c" * 40, "commit": "c" * 40, "tree": "t" * 40}))
    bins = tmp_path / "bin"
    host = _bin(bins, "veritor-zk-host-cuda", b"build of commit c")
    doc = BP.record(host, tree / "archive" / "sp1")
    assert doc["source"] == {"commit": "c" * 40, "tree": "t" * 40, "dirty": False, "dir": str(tree.resolve()), "via": "READY.json"}
    assert json.loads((bins / "veritor-zk-host-cuda.provenance.json").read_text()) == doc
    other = _bin(bins, "ligero-verify", b"hand build")
    rows = {Path(r["path"]).name: r for r in BP.snapshot([bins])}
    assert set(rows) == {"veritor-zk-host-cuda", "ligero-verify"}
    assert rows["veritor-zk-host-cuda"]["provenance"] == "recorded" and rows["veritor-zk-host-cuda"]["source"]["commit"] == "c" * 40
    assert rows["ligero-verify"] == {"path": str(other), "sha256": BP.sha256_file(other), "bytes": 10, "provenance": "missing"}
    host.write_bytes(b"a later build installed over it, unrecorded")
    row = next(r for r in BP.snapshot([bins]) if r["path"] == str(host))
    assert row["provenance"] == "stale" and row["recorded_sha256"] == doc["sha256"] != row["sha256"]


def test_record_from_a_git_worktree_says_dirty_and_from_a_pods_sync_tree_reads_its_stamp(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.rs").write_text("fn main() {}\n")
    g = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*g, "init", "-q"], check=True)
    subprocess.run([*g, "add", "."], check=True)
    subprocess.run([*g, "commit", "-qm", "one"], check=True)
    head = subprocess.run([*g, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    b = _bin(tmp_path / "bin", "x", b"x")
    assert BP.record(b, repo)["source"] | {"dir": None} == {"commit": head, "tree": BP.source_identity(repo)["tree"], "dirty": False,
                                                            "dir": None, "via": "git"}
    (repo / "a.rs").write_text("fn main() { edited }\n")
    assert BP.record(b, repo)["source"]["dirty"] is True
    synced = tmp_path / "synced"
    synced.mkdir()
    (synced / ".research-source.json").write_text(json.dumps({"commit": "d" * 40, "tree": "e" * 40, "dirty": True}))
    assert BP.source_identity(synced)["commit"] == "d" * 40 and BP.source_identity(synced)["via"] == ".research-source.json"
    assert BP.source_identity(tmp_path / "nowhere-in-git")["via"] is None


def test_the_cli_records_and_every_run_carries_run_binaries(tmp_path, monkeypatch):
    bins = tmp_path / "bin"
    b = _bin(bins, "tool", b"t")
    r = subprocess.run([sys.executable, "-m", "research.binprov", "record", str(b), "--source", str(tmp_path)],
                       capture_output=True, text=True, check=True)
    assert r.stdout.startswith(f"provenance {b}: sha256 ")
    monkeypatch.setenv(BP.BIN_DIRS_ENV, str(bins))
    runs = tmp_path / "runs"
    assert main(["run", "--runs-dir", str(runs), "--project", "t", "--no-sampler", "--", "true"]) == 0
    job = json.loads(next(runs.glob("*/job.json")).read_text())
    assert [(Path(x["path"]).name, x["provenance"]) for x in job["run"]["binaries"]] == [("tool", "recorded")]
    monkeypatch.setenv(BP.BIN_DIRS_ENV, str(tmp_path / "absent"))
    assert main(["run", "--runs-dir", str(runs), "--project", "t", "--no-sampler", "--", "true"]) == 0
    assert all("binaries" not in json.loads(p.read_text())["run"] for p in runs.glob("*/job.json")
               if json.loads(p.read_text())["run"]["id"] != job["run"]["id"]), "no bin dir: nothing added"
