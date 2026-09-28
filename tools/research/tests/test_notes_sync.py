"""`research notes sync` / `scan`: several machines write one notes repo through a remote; nobody loses a write, a conflict drops
nothing, key material and big files never leave the machine."""
from __future__ import annotations

import datetime as dt
import subprocess
import threading
from pathlib import Path

import pytest

from research import notes
from research import notes_sync as NS

DOC = "kb/doc.md"
SECTION = "\n".join(f"filler {i}" for i in range(12))
NOW = dt.datetime(2026, 9, 28, 6, 42, tzinfo=dt.timezone.utc)


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


@pytest.mark.slow
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


def test_the_watcher_reports_a_failed_sync_as_sync_failed_with_its_lag_every_pass(machines):
    origin, a, b = machines
    now = dt.datetime(2026, 9, 26, 4, 50, tzinfo=dt.timezone.utc)
    for i in range(3):
        _insert_after(a / DOC, "<!-- a -->", f"a says yes {i}")
        assert NS.sync(a, secrets=[])["pushed"]
    _insert_after(b / DOC, "<!-- a -->", "b says no")
    rep = NS.sync(b, secrets=[])
    assert rep["rc"] == 3 and NS.lag(b) == (1, 3)
    (line,) = notes.sync_lines(b, rep, now)
    assert line.startswith("04:50Z SYNC-FAILED ahead 1 behind 3: ") and DOC in line
    assert notes.sync_lines(b, NS.sync(b, secrets=[]), now)[0].startswith("04:50Z SYNC-FAILED")     # again, while it lasts
    assert notes.sync_lines(a, NS.sync(a, secrets=[]), now) == []                                    # a clean pass says nothing


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
    (a / "k.md").write_text("-----BEGIN OPENSSH " "PRIVATE KEY-----\n")   # split, so no scanner reads a key here
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


def _note(repo: Path, rel: str, created: str | None = None) -> Path:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text((f"---\nlane: {p.parent.name}\ncreated: {created}\n---\n\n" if created else "") + "a note\n")
    return p


def _tree(origin: Path) -> list[str]:
    return _git(origin, "ls-tree", "-r", "--name-only", "main").split()


def test_a_lane_note_stamped_over_5_min_ahead_is_left_out_and_the_rest_sync(machines):
    origin, a, _ = machines
    six, five, created = ("lanes/x/20260928T0648Z-report-x.md", "lanes/y/20260928T0647Z-report-y.md",
                          "lanes/z/20260928T0640Z-handoff-z.md")
    for rel in (six, five):
        _note(a, rel)
    _note(a, created, created="2026-09-28T07:00:00Z")
    _git(a, "add", six)                                                          # staged, and still left out
    (a / "kb" / "other.md").write_text("an edit\n")
    rep = NS.sync(a, secrets=[], now=NOW)
    assert (rep["rc"], rep["pushed"], rep["paths"]) == (NS.FUTURE_RC, True, 2)
    assert rep["future"] == [
        f"{six}: stamp 20260928T0648Z is 6 min ahead; use 20260928T0642Z (date -u +%Y%m%dT%H%MZ)",
        f"{created}: created: 2026-09-28T07:00:00Z is 18 min ahead; use 2026-09-28T06:42Z (date -u +%Y-%m-%dT%H:%MZ)"]
    tree = _tree(origin)
    assert five in tree and "kb/other.md" in tree and six not in tree and created not in tree
    assert f"?? {six}" in _git(a, "status", "--porcelain", "--untracked-files=all")
    assert notes.sync_lines(a, rep, NOW) == [
        f"06:42Z notes sync: committed {rep['commit']} (2 paths); pushed; left out 2 stamped ahead of the clock (SYNC-REFUSED)",
        *(f"06:42Z SYNC-REFUSED {x}" for x in rep["future"])]
    later = NS.sync(a, secrets=[], now=NOW + dt.timedelta(minutes=13))           # the clock caught up; nothing was recorded
    assert (later["rc"], later["future"]) == (0, []) and {six, created} <= set(_tree(origin))


def test_notes_already_committed_are_not_rechecked_but_a_changed_created_is(machines):
    origin, a, _ = machines
    old = "lanes/w/20260928T0800Z-report-w.md"
    _note(a, old, created="2026-09-28T08:00Z")
    _git(a, "add", "-A")
    _git(a, "commit", "-qm", "tonight's history")
    _git(a, "push", "-q", "origin", "HEAD:main")
    with (a / old).open("a") as f:
        f.write("CHECKPOINT none (06:42Z) [open] still here\n")
    rep = NS.sync(a, secrets=[], now=NOW)
    assert (rep["rc"], rep["pushed"], rep["future"]) == (0, True, []) and "still here" in _git(origin, "show", f"main:{old}")
    (a / old).write_text((a / old).read_text().replace("created: 2026-09-28T08:00Z", "created: 2026-09-28T09:00Z"))
    rep = NS.sync(a, secrets=[], now=NOW)
    assert rep["rc"] == NS.FUTURE_RC and rep["future"] == [
        f"{old}: created: 2026-09-28T09:00Z is 138 min ahead; use 2026-09-28T06:42Z (date -u +%Y-%m-%dT%H:%MZ)"]


def test_sync_and_checkpoint_exit_5_naming_the_note_and_the_stamp_to_use(machines, monkeypatch, capsys):
    origin, a, _ = machines
    monkeypatch.setattr(NS, "_utcnow", lambda: NOW)
    monkeypatch.setenv(NS.SYNC_ENV, "1")
    bad = "lanes/x/20260928T0700Z-report-x.md"
    _note(a, bad, created="2026-09-28T07:00Z")
    assert notes.main(["checkpoint", "x", "open", "measuring", "--commit", "none", "--root", str(a)]) == NS.FUTURE_RC
    assert (f"SYNC-REFUSED {bad}: stamp 20260928T0700Z is 18 min ahead; use 20260928T0642Z (date -u +%Y%m%dT%H%MZ); "
            f"created: 2026-09-28T07:00Z is 18 min ahead; use 2026-09-28T06:42Z (date -u +%Y-%m-%dT%H:%MZ)") in capsys.readouterr().err
    assert "CHECKPOINT none" in (a / bad).read_text() and bad not in _tree(origin)
    assert notes.main(["sync", "--root", str(a)]) == NS.FUTURE_RC and f"SYNC-REFUSED {bad}: stamp" in capsys.readouterr().err
    fixed = a / "lanes/x/20260928T0642Z-report-x.md"
    (a / bad).rename(fixed)
    fixed.write_text(fixed.read_text().replace("created: 2026-09-28T07:00Z", "created: 2026-09-28T06:42Z"))
    assert notes.main(["sync", "--root", str(a)]) == 0 and "lanes/x/20260928T0642Z-report-x.md" in _tree(origin)


def test_created_is_read_from_the_front_matter_with_or_without_seconds():
    utc = dt.timezone.utc
    for value, when in (("2026-09-28T07:00Z", dt.datetime(2026, 9, 28, 7, 0, tzinfo=utc)),
                        ("2026-09-28T07:00:30Z", dt.datetime(2026, 9, 28, 7, 0, 30, tzinfo=utc)),
                        ('"2026-09-28 07:00"', dt.datetime(2026, 9, 28, 7, 0, tzinfo=utc)),
                        ("2026-09-28T09:00+02:00", dt.datetime(2026, 9, 28, 7, 0, tzinfo=utc))):
        assert NS.created_at(f"---\nlane: x\ncreated: {value}\n---\nbody\n")[1] == when, value
    assert NS.created_at("created: 2026-09-28T07:00Z\n") is None                  # not in front matter
    assert NS.created_at("---\ncreated: 2026-09-28\n---\n") is None               # no time of day
