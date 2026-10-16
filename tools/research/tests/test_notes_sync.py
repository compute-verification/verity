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
from research.store.remote import FsRemote

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




def _fail_once(monkeypatch, verb: str, stderr: str) -> list:
    real, failed = NS._run, []

    def run(root, *args, **kw):
        if verb in args and not failed:
            failed.append(args)
            return subprocess.CompletedProcess(["git", *args], 128, "", stderr)
        return real(root, *args, **kw)
    monkeypatch.setattr(NS, "_run", run)
    return failed


@pytest.mark.parametrize("verb, stderr", [
    ("push", "error: cannot lock ref 'refs/heads/main': Unable to create '/srv/notes.git/refs/heads/main.lock': File exists.\n"
             " ! [remote rejected] HEAD -> main (failed to lock)"),
    ("fetch", "fatal: the remote end hung up unexpectedly"),
])
def test_a_fetch_or_push_that_fails_under_load_is_retried(machines, monkeypatch, verb, stderr):
    """Two machines' syncs under load 156 (vy-mig-check-20, 6 Oct) failed on their first attempt with an error that was not a
    lost push race, and gave up there; a lock or transfer error clears on the next attempt."""
    origin, a, _ = machines
    failed = _fail_once(monkeypatch, verb, stderr)
    _insert_after(a / DOC, "<!-- a -->", "a under load")
    rep = NS.sync(a, secrets=[])
    assert failed and (rep["rc"], rep["pushed"], rep["attempts"]) == (0, True, 2), rep
    assert "a under load" in _git(origin, "show", f"main:{DOC}")


def test_a_refused_fetch_is_not_retried(machines, monkeypatch):
    _, a, _ = machines
    _fail_once(monkeypatch, "fetch", "remote: Permission to danielreuter/research-notes.git denied to cursor[bot].\n"
                                     "fatal: unable to access 'https://github.com/x': The requested URL returned error: 403")
    _insert_after(a / DOC, "<!-- a -->", "a refused")
    rep = NS.sync(a, secrets=[])
    assert (rep["rc"], rep["attempts"]) == (1, 1) and "git fetch origin failed" in rep["error"] and "403" in rep["error"]


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


def test_a_detached_head_commits_nothing(machines):
    """The steward's clone, 6 Oct 16:05Z: once HEAD was detached, every pass committed onto it and none could push (nine syncs
    unpushed by 17:20Z); a sync on a detached HEAD now leaves the work uncommitted for the branch to take."""
    _, a, _ = machines
    _git(a, "checkout", "-q", "--detach")
    _insert_after(a / DOC, "<!-- a -->", "a while detached")
    before = _git(a, "rev-parse", "HEAD")
    rep = NS.sync(a, secrets=[])
    assert (rep["rc"], rep["commit"], rep["pushed"]) == (2, None, False) and "detached" in NS.render_sync(rep)
    assert _git(a, "rev-parse", "HEAD") == before and "a while detached" in (a / DOC).read_text()


def test_a_rebase_whose_abort_lost_its_state_goes_back_to_the_branch(machines, monkeypatch):
    """The same clone, 6 Oct 16:05Z: on NFS the rebase lost `.git/rebase-merge` ("could not lock .../git-rebase-todo"), so its
    abort did nothing and HEAD stayed detached at origin/main; the sync puts the branch back, with the local commit on it."""
    origin, a, b = machines
    _insert_after(a / DOC, "<!-- b -->", "a first")
    assert NS.sync(a, secrets=[])["pushed"]
    _insert_after(b / DOC, "<!-- a -->", "b local")
    real = NS._run

    def run(root, *args, **kw):
        if args[:1] == ("rebase",) and "--abort" not in args:
            real(root, "checkout", "-q", "--detach", args[-1])
            return subprocess.CompletedProcess(["git", *args], 1, "",
                                               "error: could not lock '.git/rebase-merge/git-rebase-todo': No such file or directory")
        if args == ("rebase", "--abort"):
            return subprocess.CompletedProcess(["git", *args], 128, "", "fatal: no rebase in progress?")
        return real(root, *args, **kw)
    monkeypatch.setattr(NS, "_run", run)
    rep = NS.sync(b, secrets=[])
    assert rep["rc"] == 3 and _git(b, "symbolic-ref", "--short", "HEAD").strip() == "main"
    assert "b local" in _git(b, "show", f"HEAD:{DOC}") and "b local" in (b / DOC).read_text()
    monkeypatch.setattr(NS, "_run", real)
    assert NS.sync(b, secrets=[])["pushed"] and "b local" in _git(origin, "show", f"main:{DOC}")


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


@pytest.mark.slow
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


def test_evidence_and_renders_stay_out_of_the_notes_and_their_removal_syncs(machines):
    """A file added or changed under renders/, campaigns/*/assets/ or lanes/*/evidence/ is left out, staged or not, and named with
    where it belongs; the rest syncs, and deleting one that history already holds is pushed like any change."""
    origin, a, _ = machines
    (a / "lanes" / "x" / "evidence").mkdir(parents=True)
    (a / "lanes" / "x" / "evidence" / "old.json").write_text("{}\n")
    _git(a, "add", "-A")
    _git(a, "commit", "-qm", "history from before the rule")
    _git(a, "push", "-q", "origin", "HEAD:main")
    ev = ["campaigns/c/assets/x/run.json", "lanes/x/evidence/run.log", "renders/daily/20260928T0642Z-tables.json"]
    for rel in ev:
        (a / rel).parent.mkdir(parents=True, exist_ok=True)
        (a / rel).write_text("output\n")
    _git(a, "add", "renders")
    (a / "lanes" / "x" / "20260928T0640Z-report-x.md").write_text("CHECKPOINT none (06:40Z) [open] ran\n")
    rep = NS.sync(a, secrets=[], now=NOW)
    assert (rep["rc"], rep["pushed"], rep["paths"], sorted(rep["evidence"])) == (0, True, 1, sorted(ev))
    assert set(_tree(origin)) == {DOC, "lanes/x/evidence/old.json", "lanes/x/20260928T0640Z-report-x.md"}
    assert "left out 3 evidence/render files (the evidence store holds them: research data put" in NS.render_sync(rep)
    (a / "lanes" / "x" / "evidence" / "old.json").unlink()
    rep = NS.sync(a, secrets=[], now=NOW)
    assert rep["pushed"] and "lanes/x/evidence/old.json" not in _tree(origin) and len(rep["evidence"]) == 3


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


@pytest.mark.slow
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


def test_a_commit_whose_diff_is_not_utf8_still_syncs(machines):
    """The key scan reads `git log -p` of the outgoing commits: removing a Latin-1 or binary file must not crash it."""
    origin, a, _ = machines
    (a / "kb" / "latin1.txt").write_bytes(b"caf\xe9 \x93quoted\x94\n")
    _git(a, "add", "-A")
    _git(a, "commit", "-qm", "before")
    _git(a, "push", "-q", "origin", "HEAD:main")
    (a / "kb" / "latin1.txt").unlink()
    rep = NS.sync(a, secrets=[], now=NOW)
    assert (rep["rc"], rep["pushed"]) == (0, True) and "kb/latin1.txt" not in _tree(origin)
# ---- front matter ---------------------------------------------------------------------------------------------------------------------

CURSOR_HEAD = '---\ncursor:\n  subagentId: "bc-0000"\n---\n\n# a Cursor agent\'s note\n'


def test_header_issues_refuse_broken_front_matter_and_warn_on_what_disagrees_with_the_path():
    ok = "---\nlane: coordinator\nkind: handoff\nfrom: x\ncreated: 2026-09-28T06:40Z\nto:\n  - a\n- b\n# a comment\n\n---\n# title\n"
    hand = "lanes/coordinator/20260928T0640Z-handoff-from-x.md"
    assert NS.header_issues(hand, ok, NOW) == ([], [])
    assert NS.header_issues(hand, CURSOR_HEAD, NOW) == ([], [])                     # Cursor's attribution block, nothing else
    assert NS.header_issues(hand, "# no front matter at all\n", NOW) == ([], [])    # lane, kind, time are in the path
    use = "use 2026-09-28T06:42Z (date -u +%Y-%m-%dT%H:%MZ)"
    for head, want in (("---\nlane: coordinator\n# title\nbody\n", ["front matter opened with --- is never closed: add a --- line under the header"]),
                       ("---\nlane: coordinator\nthis line is prose\n---\n", ["front matter line 3 is not `key: value`: 'this line is prose'"]),
                       ("---\ncreated: now\n---\n", [f"created: 'now' is not a UTC time; {use}"]),
                       ("---\ncreated: 2026-09-24\n---\n", [f"created: '2026-09-24' is not a UTC time; {use}"]),
                       ("---\ncreated: 20260926T1344Z\n---\n", [f"created: '20260926T1344Z' is not a UTC time; {use}"])):
        assert NS.header_issues(hand, head, NOW) == (want, []), head
    assert NS.header_issues(hand, ok.replace("lane: coordinator", "lane: x"), NOW) == (
        [], ["lane: x but the note is in lanes/coordinator/ (a handoff's lane is its recipient)"])
    assert NS.header_issues("lanes/y/20260926T0640Z-report-y.md", "---\nlane: y\ncreated: 2026-09-28T06:40Z\n---\n", NOW) == (
        [], ["created: 2026-09-28T06:40Z is 48 h from the file name's stamp 20260926T0640Z"])


def test_sync_leaves_out_a_note_with_broken_front_matter_and_warns_on_the_rest(machines):
    origin, a, _ = machines
    bad, off, fine = ("lanes/coordinator/20260928T0640Z-handoff-from-x.md", "lanes/coordinator/20260928T0641Z-handoff-from-y.md",
                      "lanes/coordinator/20260928T0642Z-handoff-from-z.md")
    for rel, text in ((bad, "---\nlane: coordinator\ncreated: now\n---\n# x\n"), (off, "---\nlane: y\nkind: handoff\n---\n# y\n"),
                      (fine, CURSOR_HEAD)):
        (a / rel).parent.mkdir(parents=True, exist_ok=True)
        (a / rel).write_text(text)
    rep = NS.sync(a, secrets=[], now=NOW)
    assert (rep["rc"], rep["pushed"], rep["paths"], rep["future"]) == (NS.FUTURE_RC, True, 2, [])
    assert rep["header"] == [f"{bad}: created: 'now' is not a UTC time; use 2026-09-28T06:42Z (date -u +%Y-%m-%dT%H:%MZ)"]
    assert rep["warn"] == [f"{off}: lane: y but the note is in lanes/coordinator/ (a handoff's lane is its recipient)"]
    assert {off, fine} <= set(_tree(origin)) and bad not in _tree(origin)
    assert NS.refused_lines(rep) == [f"SYNC-REFUSED {rep['header'][0]}", f"SYNC-WARN {rep['warn'][0]}"]
    assert "left out 1 with broken front matter (SYNC-REFUSED); 1 warning (SYNC-WARN)" in NS.render_sync(rep)
    (a / bad).write_text("---\nlane: coordinator\ncreated: 2026-09-28T06:40Z\n---\n# x\n")
    assert NS.sync(a, secrets=[], now=NOW)["rc"] == 0 and bad in _tree(origin)


def test_front_matter_is_checked_only_for_new_content_and_new_problems(machines):
    """History is never re-judged: a problem HEAD's copy already had does not stop a checkpoint, a move (month folders) carries a
    note whatever its header, and a moved note that is refused keeps its old path instead of vanishing."""
    origin, a, _ = machines
    old, moved = "lanes/w/20260920T0800Z-report-w.md", "lanes/w/2026-09/20260920T0800Z-report-w.md"
    (a / "lanes" / "w").mkdir(parents=True)
    (a / old).write_text("---\nlane: w\ncreated: now\n---\nCHECKPOINT none (08:00Z) [open] started\n")
    _git(a, "add", "-A")
    _git(a, "commit", "-qm", "last week's note")
    _git(a, "push", "-q", "origin", "HEAD:main")
    with (a / old).open("a") as f:
        f.write("CHECKPOINT none (06:42Z) [open] still here\n")
    rep = NS.sync(a, secrets=[], now=NOW)
    assert (rep["rc"], rep["header"], rep["warn"]) == (0, [], []) and "still here" in _git(origin, "show", f"main:{old}")
    (a / "lanes" / "w" / "2026-09").mkdir()
    _git(a, "mv", old, moved)
    rep = NS.sync(a, secrets=[], now=NOW)
    assert (rep["rc"], rep["header"]) == (0, []) and moved in _tree(origin) and old not in _tree(origin)
    again = "lanes/w/20260920T0800Z-report-w.md"
    _git(a, "mv", moved, again)
    (a / again).write_text("---\nlane: w\n# the closing line went missing\n")
    rep = NS.sync(a, secrets=[], now=NOW)
    assert rep["rc"] == NS.FUTURE_RC and rep["header"] == [f"{again}: front matter opened with --- is never closed: add a --- line under the header"]
    assert moved in _tree(origin) and again not in _tree(origin)


# ---- citations ------------------------------------------------------------------------------------------------------------------------

ART = "cc59294a" + "0" * 56
RUN, LABELLED, UNKNOWN = "r20260928-060054-e5c1", "r20260928-042319-9aac", "r20260928-000000-0000"


def _vault(tmp_path: Path, monkeypatch=None) -> Path:
    """A fs remote holding ART's manifest, RUN's attempt and a label on LABELLED; with `monkeypatch`, the store's configured remote."""
    vault = tmp_path / "vault"
    for key in (f"manifests/{ART}.json", f"attempts/{RUN}.json", f"labels/{LABELLED}/{'a' * 64}.json"):
        (vault / key).parent.mkdir(parents=True, exist_ok=True)
        (vault / key).write_text("{}\n")
    if monkeypatch:
        (tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{vault}"\n')
        monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    return vault


def test_outgoing_citations_are_the_ids_that_the_pushed_lines_add_to_notes(machines):
    origin, a, _ = machines
    note = "lanes/x/20260928T0640Z-report-x.md"
    _note(a, note)
    with (a / note).open("a") as f:
        f.write(f"gate {RUN} passed; art:cc59294a and art:{ART}, again {RUN}\n")
    (a / "kb" / "k.md").write_text(f"see {UNKNOWN}\n")
    (a / "lanes" / "x" / "log.txt").write_text(f"{LABELLED}\n")          # not a note
    (a / "README.md").write_text(f"{LABELLED}\n")                          # not under lanes/ or kb/
    _git(a, "add", "-A")
    _git(a, "commit", "-qm", "cite")
    assert NS.outgoing_citations(a, "refs/remotes/origin/main") == {note: ["art:cc59294a", f"art:{ART}", RUN], "kb/k.md": [UNKNOWN]}
    assert NS.outgoing_citations(a, None) == NS.outgoing_citations(a, "refs/remotes/origin/main")     # a first push: the last commit
    _git(a, "push", "-q", "origin", "HEAD:main")
    _git(a, "fetch", "-q", "origin")
    (a / "lanes" / "x" / "2026-09").mkdir()
    _git(a, "mv", note, "lanes/x/2026-09/20260928T0640Z-report-x.md")
    _git(a, "commit", "-qm", "move")
    assert NS.outgoing_citations(a, "refs/remotes/origin/main") == {}                                 # a move adds no lines


def test_missing_citations_asks_the_remote_and_a_label_accounts_for_a_run(tmp_path):
    remote = FsRemote(_vault(tmp_path))
    cited = ["art:cc59294a", f"art:{ART}", "art:deadbeef", RUN, LABELLED, UNKNOWN]
    assert NS.missing_citations(cited, remote) == ({"art:deadbeef", UNKNOWN}, None)
    assert NS.missing_citations(cited) == (set(), None)                  # no remote configured: no check

    class Down:
        def list(self, prefix):
            raise OSError("unreachable")

        head = list

    assert NS.missing_citations(cited, Down()) == (set(), "the store's remote did not answer for 6 cited ids (OSError)")

    release = threading.Event()

    class Hangs:
        def list(self, prefix):
            release.wait()
            return []

        head = list

    assert NS.missing_citations(cited, Hangs(), wait_s=0) == (set(), "6 cited ids not answered within 0 s")   # returns while they hang
    release.set()


def test_citation_warnings_name_each_note_and_how_to_publish(tmp_path, monkeypatch):
    remote = FsRemote(_vault(tmp_path))
    cites = {"lanes/x/a.md": [RUN, UNKNOWN, "art:deadbeef"], "kb/b.md": [LABELLED]}
    assert NS.citation_warnings(cites, remote) == [
        f"lanes/x/a.md: cites {UNKNOWN}, art:deadbeef, which the evidence store does not have "
        f"({NS.CITE_HINTS['run']}; {NS.CITE_HINTS['art']})"]
    many = [f"r20260928-0000{i:02d}-0000" for i in range(NS.CITE_MAX + 6)]
    assert NS.citation_warnings({"kb/c.md": many}, remote) == [
        f"kb/c.md: cites {', '.join(many[:NS.CITE_MAX])}, which the evidence store does not have ({NS.CITE_HINTS['run']})",
        f"citation check: 6 more cited ids not looked up (at most {NS.CITE_MAX} a sync)"]
    monkeypatch.setenv(NS.CITE_ENV, "0")
    assert NS.citation_warnings(cites, remote) == []


def test_sync_warns_once_pushed_on_a_citation_the_store_lacks(machines, tmp_path, monkeypatch):
    origin, a, _ = machines
    _vault(tmp_path, monkeypatch)
    note = "lanes/x/20260928T0640Z-report-x.md"
    _note(a, note)
    with (a / note).open("a") as f:
        f.write(f"gate {RUN} passed; train {UNKNOWN} passed but unrecorded; {LABELLED} waived\n")
    rep = NS.sync(a, secrets=[], now=NOW)
    assert (rep["rc"], rep["pushed"]) == (0, True) and note in _tree(origin)
    assert rep["warn"] == [f"{note}: cites {UNKNOWN}, which the evidence store does not have ({NS.CITE_HINTS['run']})"]
    assert NS.refused_lines(rep) == [f"SYNC-WARN {rep['warn'][0]}"] and "1 warning (SYNC-WARN)" in NS.render_sync(rep)
    with (a / note).open("a") as f:
        f.write("CHECKPOINT none (06:42Z) [open] no new citation\n")
    assert NS.sync(a, secrets=[], now=NOW)["warn"] == []                  # what was pushed is not re-checked
    monkeypatch.setenv(NS.CITE_ENV, "0")
    with (a / note).open("a") as f:
        f.write(f"again {UNKNOWN}\n")
    assert NS.sync(a, secrets=[], now=NOW)["warn"] == []


# ---- RESEARCH_NOTES_TOKEN: past a cloud VM's global insteadOf --------------------------------------------------------------------------

TOKEN = "dummy-notes-token"
EXPLICIT = "https://x-access-token@github.com/compute-verification/research-notes.git"


@pytest.fixture
def cloud_vm(machines, tmp_path, monkeypatch):
    """A cloud VM's global git config: `https://github.com/` rewritten to the bot's credentials (here a path with no repository, as
    the bot is refused), and GitHub (the explicit-user URL) standing in as the local origin; any other explicit-user URL has no
    repository either, and git may not leave the machine.  Clone a's origin is the plain notes URL."""
    origin, a, b = machines
    cfg = tmp_path / "gitconfig"
    for url, prefix in ((tmp_path / "bot", "https://github.com/"), (tmp_path / "nowhere", "https://x-access-token@github.com/"),
                        (origin, EXPLICIT)):
        _git(tmp_path, "config", "--file", str(cfg), f"url.{url}{'' if url == origin else '/'}.insteadOf", prefix)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    _git(a, "remote", "set-url", "origin", NS.NOTES_URL)
    return origin, a, b


def _argv_log(monkeypatch) -> list[list[str]]:
    seen: list[list[str]] = []
    run = subprocess.run

    def spy(cmd, *args, **kw):
        seen.append(list(cmd))
        return run(cmd, *args, **kw)

    monkeypatch.setattr(NS.subprocess, "run", spy)
    return seen


def test_token_url_names_the_github_repo_with_an_explicit_user_and_never_the_token(monkeypatch):
    assert NS.token_url(NS.NOTES_URL) is None, "unset: nothing changes"
    monkeypatch.setenv(NS.TOKEN_ENV, TOKEN)
    for url in (NS.NOTES_URL, "https://github.com/compute-verification/research-notes", "https://github.com/danielreuter/research-notes",
                "git@github.com:danielreuter/research-notes.git",
                "ssh://git@github.com/danielreuter/research-notes.git", "https://someone@github.com/danielreuter/research-notes.git"):
        opts, where = NS.token_url(url)
        assert where == EXPLICIT and not where.startswith("https://github.com/"), url
        assert opts == ["-c", "credential.helper=", "-c", f"credential.helper={NS.CREDENTIAL_HELPER}"] and TOKEN not in " ".join(opts)
    assert NS.token_url("/srv/notes.git") is None and NS.token_url("https://gitlab.com/a/b.git") is None


def test_the_credential_helper_answers_the_token_from_the_environment(cloud_vm, tmp_path, monkeypatch):
    monkeypatch.setenv(NS.TOKEN_ENV, TOKEN)
    opts, _ = NS.token_url(NS.NOTES_URL)
    r = subprocess.run(["git", *opts, "credential", "fill"], input="protocol=https\nhost=github.com\n\n", capture_output=True,
                       text=True, check=True, cwd=tmp_path)
    assert "username=x-access-token\n" in r.stdout and f"password={TOKEN}\n" in r.stdout


def test_sync_with_the_token_goes_past_the_bots_insteadof(cloud_vm, monkeypatch):
    origin, a, b = cloud_vm
    (a / "lanes" / "x").mkdir(parents=True)
    (a / "lanes" / "x" / "report.md").write_text("CHECKPOINT none (20:00Z) [open] started\n")
    rep = NS.sync(a, secrets=[])
    assert rep["rc"] == 1 and "git fetch origin failed" in rep["error"], "unset: the plain URL is the bot's, refused as before"
    assert f"{NS.TOKEN_ENV} is empty in this shell" in rep["error"] and "Cloud Agents > Secrets" in rep["error"], rep["error"]
    monkeypatch.setenv("CLOUD_AGENT_INJECTED_SECRET_NAMES", f"OTHER,{NS.TOKEN_ENV}")
    err = NS.sync(a, secrets=[])["error"]
    assert NS.FRESH_SHELL in err and "Cloud Agents" not in err, "a secret this VM injects is lost from the shell, not unconfigured"
    monkeypatch.setenv(NS.TOKEN_ENV, TOKEN)
    seen = _argv_log(monkeypatch)
    rep = NS.sync(a, secrets=[])
    assert (rep["rc"], rep["pushed"]) == (0, True) and _git(origin, "show", "main:lanes/x/report.md").startswith("CHECKPOINT")
    assert _git(a, "rev-parse", "refs/remotes/origin/main") == _git(origin, "rev-parse", "main")
    net = [c for c in seen if {"fetch", "push"} & set(c)]
    assert net and all(EXPLICIT in c for c in net) and not any(TOKEN in " ".join(c) for c in seen)
    assert _git(a, "config", "remote.origin.url").strip() == NS.NOTES_URL, "the clone's config is not touched"


def test_a_refused_fetch_with_the_token_says_so_without_the_token(cloud_vm, tmp_path, monkeypatch):
    _, a, _ = cloud_vm
    monkeypatch.setenv(NS.TOKEN_ENV, TOKEN)
    _git(tmp_path, "config", "--file", str(tmp_path / "gitconfig"), "--unset-all", f"url.{cloud_vm[0]}.insteadOf")
    (a / "kb" / "new.md").write_text("new\n")
    rep = NS.sync(a, secrets=[])
    assert rep["rc"] == 1 and "git fetch origin failed" in rep["error"] and TOKEN not in rep["error"]
    assert f"{NS.TOKEN_ENV} is empty" not in rep["error"], "set but refused: not the empty-token hint"


def test_the_standby_pull_uses_the_token_and_updates_the_tracking_branch(cloud_vm, monkeypatch):
    origin, a, b = cloud_vm
    (b / "kb" / "from-b.md").write_text("b\n")
    assert NS.sync(b, secrets=[])["pushed"]
    monkeypatch.setenv(NS.TOKEN_ENV, TOKEN)
    assert NS.pull(a).returncode == 0 and (a / "kb" / "from-b.md").is_file()
    assert _git(a, "rev-parse", "refs/remotes/origin/main") == _git(origin, "rev-parse", "main")


def test_budgets_are_fetched_from_the_notes_origin_with_the_token(cloud_vm, monkeypatch):
    from research.pods import budgets as B

    origin, a, b = cloud_vm
    (b / B.FILE).write_text('[budgets]\n"vy-a-" = { cap_usd = 3, expires = "2026-09-24T20:00Z", max_pod_hours = 4 }\n')
    assert NS.sync(b, secrets=[])["pushed"]
    with pytest.raises(OSError):
        B.read(a)
    monkeypatch.setenv(NS.TOKEN_ENV, TOKEN)
    assert "vy-a-" in B.read(a)[0]


def test_a_missing_notes_clone_is_cloned_once_the_token_is_set(cloud_vm, tmp_path, monkeypatch):
    origin = cloud_vm[0]
    root = tmp_path / "home" / ".research" / "notes"
    assert NS.ensure_clone(root) is None and not root.exists(), "unset: nothing to clone with"
    monkeypatch.setenv(NS.TOKEN_ENV, TOKEN)
    seen = _argv_log(monkeypatch)
    assert NS.ensure_clone(root) == f"notes: cloned {NS.NOTES_URL} into {root}"
    assert (root / DOC).is_file() and _git(root, "config", "remote.origin.url").strip() == NS.NOTES_URL
    assert _git(root, "rev-parse", "HEAD") == _git(origin, "rev-parse", "main") and not any(TOKEN in " ".join(c) for c in seen)
    assert NS.ensure_clone(root) is None, "there already"
    assert [p.name for p in root.parent.iterdir()] == ["notes"]


def test_a_failed_clone_leaves_nothing_and_says_why_without_the_token(cloud_vm, tmp_path, monkeypatch):
    monkeypatch.setenv(NS.TOKEN_ENV, TOKEN)
    _git(tmp_path, "config", "--file", str(tmp_path / "gitconfig"), "--unset-all", f"url.{cloud_vm[0]}.insteadOf")
    root = tmp_path / "home" / "notes"
    line = NS.ensure_clone(root)
    assert line.startswith(f"notes: clone of {NS.NOTES_URL} into {root} failed: ") and TOKEN not in line
    assert list(root.parent.iterdir()) == []


def test_machine_resolution_clones_the_registry_on_a_fresh_vm(cloud_vm, tmp_path, monkeypatch):
    from research import remote
    from research.pods import registry

    origin, _, b = cloud_vm
    (b / "machines.d").mkdir()
    (b / "machines.d" / "vy-nebius-1.toml").write_text('provider = "ssh"\nhost = "203.0.113.7"\nproject = "vy"\n')
    assert NS.sync(b, secrets=[])["pushed"]
    root = tmp_path / "fresh" / "notes"
    monkeypatch.setenv("RESEARCH_NOTES", str(root))
    monkeypatch.setenv("RESEARCH_MACHINES", str(tmp_path / "absent-machines.toml"))
    monkeypatch.delenv(registry.REGISTRY_ENV)
    assert "vy-nebius-1" not in remote.load_machines() and not root.exists(), "unset: no clone, no registry"
    monkeypatch.setenv(NS.TOKEN_ENV, TOKEN)
    assert remote.resolve_machine("vy-nebius-1", "vy")["host"] == "203.0.113.7" and (root / "machines.d").is_dir()
