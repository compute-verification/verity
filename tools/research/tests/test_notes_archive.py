"""`research notes archive`: a lane's older notes move into month (or day) folders in one commit; the working set, reports,
mail the lane has not been shown and anything uncommitted stay; nothing is overwritten, and another machine's edit survives."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pytest

from research import notes
from research import notes_archive as NA
from research import notes_sync as NS

import test_notes_sync
from test_notes_sync import NOW, _git, _tree

machines = test_notes_sync.machines
C = "lanes/coordinator"


def _write(repo: Path, rel: str, text: str = "a note\n") -> Path:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def _commit_push(repo: Path, msg: str = "notes") -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", msg)
    _git(repo, "push", "-q", "origin", "HEAD:main")


def _all_shown(repo: Path, lane: str = "coordinator") -> None:
    """The lane's inbox has shown it everything in this clone: its `.inbox-seen` is past every mtime, and git ignores it as the
    notes repo does."""
    (repo / ".gitignore").write_text(f"{notes.SEEN}\n")
    _write(repo, f"lanes/{lane}/{notes.SEEN}", "9e9\n")


def test_archive_moves_old_stamped_notes_and_keeps_the_working_set(machines, capsys, monkeypatch):
    origin, a, _ = machines
    monkeypatch.setattr(NA, "_utcnow", lambda: NOW)
    old = {f"{C}/20260920T0100Z-handoff-from-x.md": f"{C}/2026-09/20260920T0100Z-handoff-from-x.md",
           f"{C}/20260921T0200Z-answer-y.md": f"{C}/2026-09/20260921T0200Z-answer-y.md"}
    stay = [f"{C}/20260920T0000Z-report-coordinator.md", f"{C}/STATE.md", f"{C}/.20260920T0100Z-x.md.gLZFHZ", f"{C}/evidence/pass.sh",
            f"{C}/20260928T0600Z-handoff-from-z.md", f"{C}/20260919T0000Z-note-w.md"]
    for rel in [*old, *stay]:
        _write(a, rel)
    _all_shown(a)
    _commit_push(a)
    _write(a, f"{C}/20260919T0000Z-note-w.md", "a note\nchanged since the last commit\n")
    _write(a, f"{C}/20260918T0000Z-untracked.md")
    moves, kept = NA.plan(a, "coordinator", NOW)
    assert moves == sorted(old.items())
    assert {k: v for k, v in kept.items() if v} == {
        "report": ["20260920T0000Z-report-coordinator.md"], "unstamped": ["STATE.md"], "dotfile": [".20260920T0100Z-x.md.gLZFHZ", notes.SEEN],
        "directory": ["evidence"], "untracked": ["20260918T0000Z-untracked.md"], "changed": ["20260919T0000Z-note-w.md"],
        "recent": ["20260928T0600Z-handoff-from-z.md"]}
    status = _git(a, "status", "--porcelain")
    assert notes.main(["archive", "coordinator", "--root", str(a)]) == 0 and _git(a, "status", "--porcelain") == status
    assert "2 notes stamped before" in capsys.readouterr().out
    assert notes.main(["archive", "coordinator", "--apply", "--root", str(a)]) == 0
    assert "moved 2 notes into <YYYY-MM>/ folders" in capsys.readouterr().out
    assert [x for x in _git(a, "show", "--format=", "--name-status", "-M", "HEAD").splitlines() if x] == [
        f"R100\t{src}\t{dst}" for src, dst in sorted(old.items())]
    assert _git(a, "status", "--porcelain") == status                   # the uncommitted edit and the untracked note stay as they were
    assert NA.plan(a, "coordinator", NOW)[0] == []                     # idempotent
    rep = NS.sync(a, secrets=[], now=NOW)
    assert (rep["rc"], rep["pushed"]) == (0, True)
    assert set(old.values()) <= set(_tree(origin)) and not set(old) & set(_tree(origin))
    assert [src for src, _ in NA.plan(a, "coordinator", NOW)[0]] == [f"{C}/20260918T0000Z-untracked.md", f"{C}/20260919T0000Z-note-w.md"]
    assert notes.report_for(a, "coordinator").name == "20260920T0000Z-report-coordinator.md"
    assert [p.name for p in notes.mail(a, "coordinator")] == ["20260928T0600Z-handoff-from-z.md"]


def test_archive_by_day_while_another_machine_edits_and_writes_loses_nothing(machines, monkeypatch):
    origin, a, b = machines
    monkeypatch.setattr(NA, "_utcnow", lambda: NOW)
    one, two = f"{C}/20260920T0100Z-handoff-from-x.md", f"{C}/20260921T2300Z-answer-y.md"
    for rel in (one, two):
        _write(a, rel)
    _all_shown(a)
    _commit_push(a)
    assert NS.sync(b, secrets=[])["rc"] == 0
    with (b / one).open("a") as f:
        f.write("b's reply, written before it saw the move\n")
    _write(b, f"{C}/20260928T0640Z-handoff-from-b.md")
    assert notes.main(["archive", "coordinator", "--bucket", "day", "--apply", "--root", str(a)]) == 0
    assert NS.sync(a, secrets=[])["pushed"]
    rep = NS.sync(b, secrets=[])
    assert (rep["rc"], rep["pushed"]) == (0, True)
    tree = _tree(origin)
    assert {f"{C}/2026-09/20/20260920T0100Z-handoff-from-x.md", f"{C}/2026-09/21/20260921T2300Z-answer-y.md",
            f"{C}/20260928T0640Z-handoff-from-b.md"} <= set(tree) and one not in tree and two not in tree
    assert "b's reply" in _git(origin, "show", f"main:{C}/2026-09/20/20260920T0100Z-handoff-from-x.md")


def test_archive_overwrites_nothing_and_undoes_itself_on_failure(machines, capsys, monkeypatch):
    origin, a, _ = machines
    monkeypatch.setattr(NA, "_utcnow", lambda: NOW)
    note, moved = f"{C}/20260920T0100Z-handoff-from-x.md", f"{C}/2026-09/20260920T0100Z-handoff-from-x.md"
    _write(a, note)
    _write(a, f"{C}/20260921T0100Z-answer-y.md")
    _all_shown(a)
    _commit_push(a)
    assert notes.main(["archive", "coordinator", "--apply", "--root", str(a)]) == 0
    _write(a, note, "the mirror sent it back\n")
    _commit_push(a, "a copy sent back")
    moves, kept = NA.plan(a, "coordinator", NOW)
    assert (moves, kept["already archived"]) == ([], ["20260920T0100Z-handoff-from-x.md"])
    capsys.readouterr()
    assert notes.main(["archive", "coordinator", "--root", str(a)]) == 0
    assert "1 names are in their folder already" in capsys.readouterr().out and (a / moved).read_text() == "a note\n"
    late = f"{C}/20260922T0100Z-answer-z.md"
    _write(a, late)
    _commit_push(a)
    (a / "kb" / "doc.md").write_text("staged elsewhere\n")
    _git(a, "add", "kb/doc.md")
    assert notes.main(["archive", "coordinator", "--apply", "--root", str(a)]) == 1
    assert "FAILED, nothing moved: the index holds staged changes" in capsys.readouterr().err and (a / late).is_file()
    _git(a, "reset", "-q")
    head, status = _git(a, "rev-parse", "HEAD"), _git(a, "status", "--porcelain")
    lock = a / ".git" / "index.lock"
    lock.write_text("")                                                 # another writer holds the index: `git add` fails
    with pytest.raises(RuntimeError, match="git add failed"):
        NA.apply(a, "coordinator", NA.plan(a, "coordinator", NOW)[0], "m")
    lock.unlink()
    assert (a / late).is_file() and not (a / f"{C}/2026-09/20260922T0100Z-answer-z.md").exists()
    assert (_git(a, "rev-parse", "HEAD"), _git(a, "status", "--porcelain")) == (head, status)


def test_archive_keeps_mail_the_lane_has_not_been_shown(machines, capsys, monkeypatch):
    origin, a, _ = machines
    monkeypatch.setattr(NA, "_utcnow", lambda: NOW)
    shown, unseen = "lanes/x/20260920T0100Z-handoff-from-y.md", "lanes/x/20260920T0200Z-handoff-from-z.md"
    for rel in (shown, unseen):
        _write(a, rel)
    _commit_push(a)
    moves, kept = NA.plan(a, "x", NOW)                                  # a clone without .inbox-seen cannot tell: no mail moves
    assert moves == [] and kept["unseen mail"] == [Path(shown).name, Path(unseen).name]
    assert notes.main(["archive", "x", "--apply", "--root", str(a)]) == 0
    out = capsys.readouterr().out
    assert f"this clone has no lanes/x/{notes.SEEN}, so none of its 2 mail files moves" in out and "nothing stamped before" in out
    t = dt.datetime(2026, 9, 27, tzinfo=dt.timezone.utc).timestamp()
    os.utime(a / shown, (t - 60, t - 60))
    os.utime(a / unseen, (t + 60, t + 60))
    (a / "lanes" / "x" / notes.SEEN).write_text(f"{t}\n")
    moves, kept = NA.plan(a, "x", NOW)
    assert moves == [(shown, "lanes/x/2026-09/20260920T0100Z-handoff-from-y.md")] and kept["unseen mail"] == [Path(unseen).name]


def test_the_old_by_spelling_still_buckets_and_says_to_use_bucket(tmp_path, capsys):
    (tmp_path / "lanes" / "x").mkdir(parents=True)
    assert NA.main(["x", "--by", "day", "--root", str(tmp_path)], tmp_path) == 0
    out = capsys.readouterr()
    assert "--by is --bucket now" in out.err and "nothing stamped before" in out.out
