"""`research pods ssh` / `research pods sync` (pods/connect.py): target resolution by machine name, pod id or pod name; the files a
sync ships (git-visible only); prune-what-we-shipped; the source stamp.  The sync runs a real rsync into a local directory through
`LocalTransport`; no network, no pod."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from research import remote
from research.pods import connect as C

PODS = [{"id": "abc123def45678", "name": "vy-lane-a-veritor-campaign"},
        {"id": "zzz999yyy88877", "name": "vy-twin-veritor-campaign"},
        {"id": "qqq111www22233", "name": "vy-twin-veritor-campaign"},
        {"id": "mmm000nnn11122", "name": "plain-name"},
        {"id": "vvv222uuu33344", "name": "vy-lane-b-verity"}]


def test_resolve_pod_by_machine_name_id_and_pod_name():
    machines = {"vy-control": {"pod_id": "ctl000ctl00000"}}
    listing = lambda: PODS  # noqa: E731
    assert C.resolve_pod("vy-control", machines=machines, pods=listing) == ("ctl000ctl00000", "vy-control")
    assert C.resolve_pod("abc123def45678", machines=machines, pods=listing) == ("abc123def45678", "vy-lane-a-veritor-campaign")
    assert C.resolve_pod("vy-lane-a", machines=machines, pods=listing)[0] == "abc123def45678"
    assert C.resolve_pod("vy-lane-a-veritor-campaign", machines=machines, pods=listing)[0] == "abc123def45678"
    assert C.resolve_pod("plain-name", machines=machines, pods=listing)[0] == "mmm000nnn11122"
    assert C.resolve_pod("vy-lane-b", machines=machines, pods=listing)[0] == "vvv222uuu33344"
    assert C.resolve_pod("vy-lane-b-verity", machines=machines, pods=listing)[0] == "vvv222uuu33344"
    with pytest.raises(remote.Refused, match="names 2 pods"):
        C.resolve_pod("vy-twin", machines=machines, pods=listing)
    with pytest.raises(remote.Refused, match="no machine or pod named 'vy-nope'.*vy-control.*vy-lane-a.*vy-lane-b,"):
        C.resolve_pod("vy-nope", machines=machines, pods=listing)
    assert C.resolve_pod("ffffeeeedddd11", machines=machines, pods=lambda: []) == ("ffffeeeedddd11", "ffffeeeedddd11")


def test_ssh_print_uses_the_looked_up_port(tmp_path, monkeypatch, capsys):
    (tmp_path / "machines.toml").write_text('[machines.vy-x]\nprovider = "runpod"\npod_id = "abc123def45678"\nproject = "verity"\n')
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "machines.toml"))
    seen = {}

    def fake_target(machine):
        seen.update(machine)
        return {"host": "203.0.113.7", "port": 40123}

    monkeypatch.setattr(remote, "pod_target", fake_target)
    assert C.main_ssh(["vy-x", "--print", "--", "nvidia-smi", "-L"]) == 0
    out = capsys.readouterr().out.strip()
    assert seen["pod_id"] == "abc123def45678" and "-p 40123" in out and out.endswith("root@203.0.113.7 nvidia-smi -L")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path) -> Path:
    r = tmp_path / "wt"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    files = {".gitignore": "build/\n*.log\n.venv/\n", "a.py": "A = 1\n", "pkg/b.py": "B = 2\n", "gone.py": "x\n", "docs/big.bin": "0" * 100}
    for rel, text in files.items():
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        (r / rel).write_text(text)
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "init")
    (r / "build").mkdir()
    (r / "build/out.o").write_text("obj")
    (r / "run.log").write_text("log")
    (r / ".venv").mkdir()
    (r / ".venv/python").write_text("py")
    return r


def test_git_visible_files_are_tracked_existing_plus_unignored_untracked(repo):
    (repo / "new.py").write_text("N = 3\n")
    (repo / "gone.py").unlink()
    assert C.git_visible_files(repo) == [".gitignore", "a.py", "docs/big.bin", "new.py", "pkg/b.py"]
    assert C.git_visible_files(repo, excludes=["docs", "*.gitignore"]) == ["a.py", "new.py", "pkg/b.py"]


@pytest.mark.parametrize("rsync", [True, False], ids=["rsync", "tar-stream"])
def test_sync_ships_prunes_only_its_own_files_and_stamps_the_source(repo, tmp_path, rsync):
    dest = tmp_path / "pod" / "src"
    t = C.LocalTransport(rsync=rsync)
    rep = C.sync(repo, str(dest), t)
    assert rep["files"] == 5 and rep["pruned"] == [] and rep["via"] == ("rsync" if rsync else "tar")
    assert (dest / "pkg/b.py").read_text() == "B = 2\n" and not (dest / "build").exists() and not (dest / "run.log").exists() and not (dest / ".venv").exists()
    stamp = json.loads((dest / C.STAMP).read_text())
    assert stamp["schema"] == C.STAMP_SCHEMA and stamp["commit"] == _git(repo, "rev-parse", "HEAD") and stamp["dirty"] is False and stamp["files"] == 5

    (dest / ".venv").mkdir()                                   # things the pod made itself: never touched
    (dest / ".venv/lib.so").write_text("so")
    (dest / "results.json").write_text("{}")
    _git(repo, "rm", "-q", "gone.py")
    _git(repo, "commit", "-q", "-m", "drop gone.py")
    (repo / "new.py").write_text("N = 3\n")
    (repo / "link.py").symlink_to("a.py")

    dry = C.sync(repo, str(dest), t, dry_run=True)
    assert dry["pruned"] == ["gone.py"] and (dest / "gone.py").exists() and not (dest / "new.py").exists()

    rep2 = C.sync(repo, str(dest), t)
    assert rep2["pruned"] == ["gone.py"] and not (dest / "gone.py").exists() and (dest / "new.py").exists()
    assert (dest / ".venv/lib.so").exists() and (dest / "results.json").exists() and not (dest / ".research-sync-prune").exists()
    stamp2 = json.loads((dest / C.STAMP).read_text())
    assert stamp2["commit"] == _git(repo, "rev-parse", "HEAD") != stamp["commit"] and stamp2["dirty"] is True and stamp2["dirty_digest"]
    assert (dest / "link.py").is_symlink() and (dest / "link.py").read_text() == "A = 1\n"
    assert (dest / C.SHIPPED).read_text().split() == [".gitignore", "a.py", "docs/big.bin", "link.py", "new.py", "pkg/b.py"]


def test_sync_refuses_a_relative_destination(repo):
    with pytest.raises(remote.Refused, match="absolute"):
        C.sync(repo, "workspace/src", C.LocalTransport())
