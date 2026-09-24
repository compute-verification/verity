"""`research notes sync` / `scan`: several machines write one notes repo through a remote; nobody loses a write, a conflict drops
nothing, key material and big files never leave the machine."""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import pytest

from research import notes
from research import notes_sync as NS

DOC = "kb/doc.md"
SECTION = "\n".join(f"filler {i}" for i in range(12))


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def _clone(origin: Path, dest: Path) -> Path:
    subprocess.run(["git", "clone", "-q", str(origin), str(dest)], check=True, capture_output=True)
    _git(dest, "config", "user.name", dest.name)
    _git(dest, "config", "user.email", f"{dest.name}@t")
    return dest


@pytest.fixture
def machines(tmp_path):
    """A bare origin with one commit on main, and two clones of it (machine a, machine b)."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    seed = _clone(origin, tmp_path / "seed")
    (seed / "kb").mkdir()
    (seed / DOC).write_text(f"# doc\n## a\n<!-- a -->\n{SECTION}\n## b\n<!-- b -->\n{SECTION}\n")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-qm", "seed")
    _git(seed, "push", "-q", "origin", "HEAD:main")
    return origin, _clone(origin, tmp_path / "a"), _clone(origin, tmp_path / "b")


def _insert_after(p: Path, marker: str, line: str) -> None:
    p.write_text(p.read_text().replace(marker, f"{marker}\n{line}", 1))


def test_a_sync_commits_pushes_and_the_other_machine_picks_it_up(machines):
    origin, a, b = machines
    (a / "lanes" / "x").mkdir(parents=True)
    (a / "lanes" / "x" / "report.md").write_text("CHECKPOINT none (20:00Z) [open] started\n")
    rep = NS.sync(a, secrets=[])
    assert (rep["rc"], rep["pushed"], rep["paths"]) == (0, True, 1)
    assert _git(origin, "show", "main:lanes/x/report.md").startswith("CHECKPOINT")
    rep = NS.sync(b, secrets=[])
    assert rep["rc"] == 0 and rep["commit"] is None and (b / "lanes" / "x" / "report.md").is_file()
    assert NS.sync(b, secrets=[]) == {**rep, "note": "already on the remote"}


def test_a2_two_machines_20_times_in_the_same_minute_lose_no_write(machines):
    origin, a, b = machines
    errors: list = []

    def write_and_sync(repo: Path, who: str, i: int) -> None:
        _insert_after(repo / DOC, f"<!-- {who} -->", f"{who} write {i}")
        d = repo / "lanes" / "shared"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"checkpoint-{who}-{i}.md").write_text(f"{who} {i}\n")
        rep = NS.sync(repo, retries=20, secrets=[])
        if rep["rc"]:
            errors.append((who, i, rep))

    for i in range(20):
        ts = [threading.Thread(target=write_and_sync, args=(m, w, i)) for m, w in ((a, "a"), (b, "b"))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    assert errors == []
    doc = _git(origin, "show", f"main:{DOC}")
    assert all(f"{w} write {i}" in doc for w in "ab" for i in range(20))
    files = _git(origin, "ls-tree", "--name-only", "main", "lanes/shared/").split()
    assert len(files) == 40


def test_a3_a_same_hunk_conflict_names_the_file_and_loses_nothing(machines):
    origin, a, b = machines
    _insert_after(a / DOC, "<!-- a -->", "a says yes")
    assert NS.sync(a, secrets=[])["pushed"]
    _insert_after(b / DOC, "<!-- a -->", "b says no")
    rep = NS.sync(b, secrets=[])
    assert rep["rc"] == 3 and rep["conflict"] == [DOC] and DOC in NS.render_sync(rep)
    assert "b says no" in _git(b, "show", f"HEAD:{DOC}") and "b says no" in (b / DOC).read_text()
    assert "a says yes" in _git(origin, "show", f"main:{DOC}") and "b says no" not in _git(origin, "show", f"main:{DOC}")
    assert not (b / ".git" / "rebase-merge").exists()


def test_key_material_is_refused_before_anything_is_pushed(machines):
    origin, a, _ = machines
    before = _git(origin, "rev-parse", "main")
    (a / "kb" / "creds.md").write_text("aws key AKIA" + "ABCDEFGHIJKLMNOP" + " oops\n")
    rep = NS.sync(a, secrets=[])
    assert rep["rc"] == 4 and rep["keys"] == ["kb/creds.md: aws-access-key-id"] and not rep["pushed"]
    assert _git(origin, "rev-parse", "main") == before and rep["commit"]
    (a / "kb" / "creds.md").write_text("the apikey field lives in ~/.runpod/config.toml\n")   # talking about keys is fine
    _git(a, "reset", "-q", "--soft", "HEAD~1")
    assert NS.sync(a, secrets=[])["rc"] == 0
    (a / "kb" / "env.md").write_text("value: s3cr3t-value-of-the-env-var\n")
    assert NS.sync(a, secrets=["s3cr3t-value-of-the-env-var"])["keys"] == ["kb/env.md: env-secret"]


def test_big_files_and_steward_state_stay_out_and_no_remote_commits_locally(tmp_path, machines):
    _, a, _ = machines
    (a / "lanes" / "x").mkdir(parents=True)
    (a / "lanes" / "x" / "dump.bin").write_bytes(b"\1" * (NS.MAX_BYTES + 1))
    (a / "steward-state.json").write_text("{}")
    (a / "lanes" / "x" / "r.md").write_text("r\n")
    rep = NS.sync(a, secrets=[])
    assert rep["paths"] == 1 and rep["skipped"] == ["lanes/x/dump.bin"] and "over 1 MB" in NS.render_sync(rep)
    lone = tmp_path / "lone"
    _git(tmp_path, "init", "-q", "-b", "main", str(lone))
    _git(lone, "config", "user.name", "t")
    _git(lone, "config", "user.email", "t@t")
    (lone / "n.md").write_text("n\n")
    rep = NS.sync(lone, secrets=[])
    assert rep["rc"] == 0 and rep["commit"] and rep["note"] == "no remote 'origin': committed locally only"


def test_a5_scan_finds_big_blobs_and_keys_anywhere_in_history(machines):
    _, a, _ = machines
    assert NS.scan(a, secrets=[]) == {"blobs": 1, "big": [], "keys": []}
    (a / "big.bin").write_bytes(b"x" * 2000)
    (a / "k.md").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n")
    _git(a, "add", "-A")
    _git(a, "commit", "-qm", "bad")
    _git(a, "rm", "-q", "k.md", "big.bin")
    _git(a, "commit", "-qm", "gone, but still in history")
    rep = NS.scan(a, max_bytes=1000, secrets=[])
    assert rep["big"] == [{"path": "big.bin", "bytes": 2000}] and rep["keys"] == ["k.md: private-key"]


def test_checkpoint_syncs_its_lane_only_when_asked(machines, monkeypatch):
    origin, a, _ = machines
    monkeypatch.delenv(NS.SYNC_ENV, raising=False)
    assert notes.main(["checkpoint", "x", "open", "started", "--commit", "none", "--root", str(a)]) == 0
    assert "lanes/x" not in _git(origin, "ls-tree", "-r", "--name-only", "main")
    monkeypatch.setenv(NS.SYNC_ENV, "1")
    (a / "kb" / "other.md").write_text("someone else's edit on this machine\n")
    assert notes.main(["checkpoint", "x", "open", "measuring", "--commit", "none", "--root", str(a)]) == 0
    tree = _git(origin, "ls-tree", "-r", "--name-only", "main").split()
    assert any(p.startswith("lanes/x/") for p in tree) and "kb/other.md" not in tree
    assert notes.main(["sync", "--root", str(a)]) == 0 and "kb/other.md" in _git(origin, "ls-tree", "-r", "--name-only", "main")
