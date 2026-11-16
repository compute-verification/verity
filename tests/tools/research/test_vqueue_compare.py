"""`research.vqueue_compare`: a decision log beside a repository whose `checked` is a chain of drive's `Queue: #N` merges,
compared over a window: each line of the report, each kind of disagreement, the parsers, and the command's exit codes."""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from research import vqueue_compare as vc
from research.queue import label_target
from research.store import labels_sync, vocab
from research.store.local import LocalStore
from research.store.model import Label

DAY = "2030-01-07"
C1, C2, C3, C4, C5, C6 = (f"c{i}" + "0" * 38 for i in range(1, 7))
EARLY = "b0" + "0" * 38


def at(hm: str) -> str:
    return f"{DAY}T{hm}:00Z"


def hm(s: str) -> str:
    return f"01-07 {s}Z"


def git(repo: Path, *args: str, stamp: str | None = None) -> str:
    """``git args`` in ``repo``; a commit it makes is dated ``stamp`` (`HH:MM` on the fixture's day)."""
    env = None
    if stamp:
        t = f"@{int(vc.when(at(stamp)).timestamp())} +0000"
        env = dict(os.environ, GIT_AUTHOR_DATE=t, GIT_COMMITTER_DATE=t)
    return subprocess.run(["git", *args], cwd=repo, env=env, input="", capture_output=True, text=True, check=True).stdout.strip()


def decision(stamp: str, tick: int, action: str, main: str, **kw) -> dict:
    return {"at": at(stamp), "tick": tick, "shadow": True, "main": main, "action": action, "sha": None, "base": None,
            "members": [], "heads": [], "priority": None, "number": None, "why": "", "outcome": None, "failed": [], **kw}


@dataclass
class World:
    """Drive's old path, in git: `checked` lands 101, 102, 103, 104 and 106, then reverts 103 and regenerates the lock
    records; `main` holds up to 104.  The shadow's log: C1 = {101, 102} on M0 and C2 = {103} on C1 pass and advance; C3 =
    {104} fails and ejects 104, with a retry and a ring first; 105 never merges onto a build; C4 = {107} advances after the
    old path ejected 107; C5 = {108} goes stale; C6 = {109} is still pending at the window's end, as a drop of it that failed
    doesn't end it."""
    tmp: Path
    repo: Path
    m0: str
    heads: dict[int, str]
    q: dict[int, str]
    revert: str
    labels: dict[tuple[int, str], list[Label]]
    log: Path
    moves: Path
    reds: Path

    def read(self, heads: list[tuple[int, str]]) -> dict[tuple[int, str], list[Label]]:
        return {k: self.labels.get(k, []) for k in heads}

    def argv(self, *extra: str) -> list[str]:
        return ["--repo", str(self.repo), "--log", str(self.log), "--since", at("14:00"), "--until", at("16:00"), *extra]


SKIP_105 = "doesn't merge onto c2: a.txt"
NO_BUILD = "no member merges onto its base"
PUSH_FAILED = "git push: connection reset"
EJECT_104 = "pytest failed on c3"
EJECTED_107 = "the train's check failed pytest"


@pytest.fixture
def world(tmp_path, monkeypatch) -> World:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    tree = git(repo, "mktree")

    def commit(message: str, *parents: str, stamp: str) -> str:
        return git(repo, "commit-tree", tree, *(x for p in parents for x in ("-p", p)), "-m", message, stamp=stamp)

    m0 = commit("the base", stamp="13:00")
    heads = {n: commit(f"change {n}", m0, stamp="13:30") for n in range(100, 110)}
    q, tip = {}, m0
    for n, stamp in ((101, "14:10"), (102, "14:11"), (103, "14:12"), (104, "14:13"), (106, "14:20")):
        tip = q[n] = commit(f"Queue: #{n} change {n}\n\nPR: #{n}\nHead: {heads[n]}\nOwner: lane-{n}", tip, heads[n], stamp=stamp)
    revert = commit(f'Revert "Queue: #103 change 103"\n\nThis reverts commit {q[103]}, reversing\nchanges made to {q[102]}.',
                    tip, stamp="14:50")
    lock = commit("Lock records regenerated for #103", revert, stamp="14:51")
    git(repo, "update-ref", "refs/remotes/origin/checked", lock)
    git(repo, "update-ref", "refs/remotes/origin/main", q[104])

    def mark(n: int, stamp: str, value: bool = True, by: str = "", ref: str | None = None) -> Label:
        return Label(target=label_target(n, heads[n]), key=vocab.READY, value=value, by=by or f"lane-{n}", ts=at(stamp), ref=ref)

    labels = {(n, heads[n]): [mark(n, s)] for n, s in ((101, "14:00"), (102, "14:01"), (103, "14:02"), (104, "14:03"),
                                                        (105, "14:04"), (106, "14:15"), (108, "14:40"), (109, "14:58"))}
    labels[(107, heads[107])] = [mark(107, "14:05"), mark(107, "14:50", False, "ci", EJECTED_107)]

    def members(*prs: int) -> dict:
        return {"members": list(prs), "heads": [heads[n] for n in prs]}

    def outcome(state: str, *prs: int) -> dict:
        return {**members(*prs), "state": state, "suspects": list(prs) if state == "fail" else []}

    log = [
        decision("13:00", 0, "build", m0, sha=EARLY, base=m0, **members(100)),
        decision("13:30", 0, "drop", m0, sha=EARLY, why="stale"),
        {**decision("14:05", 1, "build", m0, sha=C1, base=m0, priority=0, **members(101, 102)), "later": {"a key": "ignored"}},
        decision("14:06", 2, "build", m0, sha=C2, base=C1, **members(103)),
        decision("14:07", 3, "build", m0, sha=C3, base=C2, **members(104)),
        decision("14:07", 3, "skip", m0, number=105, heads=[heads[105]], why=SKIP_105),
        decision("14:20", 4, "retry", m0, sha=C3, why="graph2 lost its worker", enqueued=True),
        decision("14:21", 4, "retry", m0, sha=C3, why="its retry's results are pending", enqueued=False),
        decision("14:25", 5, "ring", m0, sha=C3, why="infra three times"),
        decision("14:25", 5, "build", m0, base=C3, why=NO_BUILD),
        decision("14:30", 6, "advance", m0, sha=C1, base=m0, **members(101, 102)),
        decision("14:30", 6, "drop", m0, sha=C1, outcome=outcome("pass", 101, 102), why="passed"),
        decision("14:40", 7, "advance", C1, sha=C2),
        decision("14:40", 7, "drop", C1, sha=C2, outcome=outcome("pass", 103), why="passed"),
        decision("14:41", 8, "build", C2, sha=C4, base=C2, **members(107)),
        decision("14:42", 8, "build", C2, sha=C5, base=C4, **members(108)),
        decision("14:43", 9, "drop", C2, sha=C5, why="stale"),
        decision("14:45", 9, "drop", C2, sha=C3, outcome=outcome("fail", 104), why="failed"),
        decision("14:45", 9, "eject", C2, number=104, heads=[heads[104]], why=EJECT_104,
                 failed=[{"task": "pytest", "key": "k104"}]),
        decision("14:55", 10, "advance", C2, sha=C4, base=C2, **members(107)),
        decision("14:55", 10, "drop", C2, sha=C4, outcome=outcome("pass", 107), why="passed"),
        decision("15:00", 11, "build", C4, sha=C6, base=C4, **members(109)),
        decision("15:05", 12, "drop", C4, sha=C6, why="a second candidate on c4", error=PUSH_FAILED),
        decision("16:30", 13, "advance", C4, sha=C6, base=C4, **members(109)),
    ]
    files = tmp_path / "in"
    files.mkdir()
    (path := files / "decisions.jsonl").write_text("\n".join(json.dumps(d) for d in log[:4]) + '\n{"at": "' + at("14:06")
                                                  + '", "act\n\n' + "\n".join(json.dumps(d) for d in log[4:]) + "\n")
    (moves := files / "main-moves").write_text(f"# where main was as the window opened, then each move\n{at('14:00')} {m0}\n"
                                               f"{at('15:00')} {q[102][:12]}\n\n{at('15:30')} {q[104]}\nnot a move\n"
                                               f"{at('15:45')} {'f' * 12}\n")
    (reds := files / "old-reds.json").write_text(json.dumps({q[102][:12]: ["pytest"], q[104]: ["lean-build"], "e" * 12: ["x"]}))
    w = World(tmp_path, repo, m0, heads, q, revert, labels, path, moves, reds)
    monkeypatch.setattr(vc, "store_labels", lambda root: w.read)
    return w


def table(text: str) -> dict[str, list[str]]:
    """The per-PR table's rows by PR, each split into its cells."""
    return {cells[0]: cells for line in text.splitlines() if line.startswith("  #")
            for cells in [re.split(r"\s{2,}", line.strip())]}


def test_the_report_line_by_line(world, capsys):
    w, h, q = world, world.heads, world.q
    assert vc.main(w.argv("--main-moves", str(w.moves), "--old-reds", str(w.reds))) == 1
    text = capsys.readouterr().out
    lines = text.splitlines()
    assert lines[0] == (f"vqueue compare {hm('14:00')} to {hm('16:00')}: 21 decision(s) in 12 tick(s) (3 advance, 7 build, "
                        f"6 drop, 1 eject, 2 retry, 1 ring, 1 skip); 9 PR(s) seen")
    assert re.split(r"\s{2,}", lines[1].strip()) == ["PR", "head", "marked", "first build", "shadow", "old path", "main"]
    assert table(text) == {
        "#101": ["#101", h[101][:12], hm("14:00"), hm("14:05"), f"advance {hm('14:30')} {C1[:12]}",
                 f"landed {hm('14:10')} {q[101][:12]}", hm("15:00")],
        "#102": ["#102", h[102][:12], hm("14:01"), hm("14:05"), f"advance {hm('14:30')} {C1[:12]}",
                 f"landed {hm('14:11')} {q[102][:12]}", hm("15:00")],
        "#103": ["#103", h[103][:12], hm("14:02"), hm("14:06"), f"advance {hm('14:40')} {C2[:12]}",
                 f"reverted {hm('14:50')} {w.revert[:12]}", "-"],
        "#104": ["#104", h[104][:12], hm("14:03"), hm("14:07"), f"eject {hm('14:45')}: {EJECT_104}",
                 f"landed {hm('14:13')} {q[104][:12]}", hm("15:30")],
        "#105": ["#105", h[105][:12], hm("14:04"), "-", f"skip {hm('14:07')}: {SKIP_105}", "-", "-"],
        "#106": ["#106", h[106][:12], hm("14:15"), "-", "-", f"landed {hm('14:20')} {q[106][:12]}", "-"],
        "#107": ["#107", h[107][:12], hm("14:05"), hm("14:41"), f"advance {hm('14:55')} {C4[:12]}", f"ejected {hm('14:50')}",
                 "-"],
        "#108": ["#108", h[108][:12], hm("14:40"), hm("14:42"), f"built {C5[:12]}", "-", "-"],
        "#109": ["#109", h[109][:12], hm("14:58"), hm("15:00"), f"built {C6[:12]}", "-", "-"],
    }
    rest = lines[11:]
    assert rest[:4] == [
        f"{'mark -> shadow advance':28} n=4  p50 0.57 h  p90 0.83 h",
        f"{'mark -> old-path main':28} n=3  p50 1.00 h  p90 1.45 h",
        "on main by the shadow only: 2 (#103 #107); by the old path only: 1 (#104)",
        "disagreements: 6",
    ]
    assert rest[4:10] == [
        f"  (a) #104 at {h[104][:12]}: the shadow ejected it {hm('14:45')} ({EJECT_104}; failed: pytest (k104)), and the old "
        f"path landed it {hm('14:13')} as {q[104][:12]}",
        f"  (b) #103 at {h[103][:12]}: the shadow advanced {C2[:12]} with it {hm('14:40')}, and the old path reverted it "
        f"{hm('14:50')} ({w.revert[:12]} undoes {q[103][:12]})",
        f"  (b) #107 at {h[107][:12]}: the shadow advanced {C4[:12]} with it {hm('14:55')}, and the old path ejected it "
        f"{hm('14:50')} by ci: {EJECTED_107}",
        f"  (c) #105: ready (marked {hm('14:04')}) and never batched; the last reason: its skip {hm('14:07')}: {SKIP_105}",
        f"  (c) #106: ready (marked {hm('14:15')}) and never batched; the last reason: no build {hm('14:25')}: {NO_BUILD}",
        f"  (d) {C1[:12]} passed {hm('14:30')} with #101 #102 on {w.m0[:12]}, and the old path's check of {q[102][:12]}, the "
        f"same PRs on it, was red: pytest",
    ]
    assert rest[10:14] == [
        f"longest pending: 1.00 h, {C6[:12]} (#109) built {hm('15:00')}, still pending",
        "builds dropped stale: 1; infra retries: 1; rings: 1; failed actions: 1",
        f"  drop {C6[:12]} {hm('15:05')} failed: {PUSH_FAILED}",
        "soundness: every advance (3) was to a candidate the log built on the logged main, none with a failed outcome (3 with a "
        "logged pass)",
    ]
    warnings = rest[14:]
    assert len(warnings) == 4 and all(x.startswith("warning: ") for x in warnings)
    assert warnings[0].startswith("warning: log line 5 isn't a decision (JSONDecodeError: ") and warnings[0].endswith(": left out")
    assert warnings[1:] == [
        "warning: --main-moves line 6: 'not a move': Invalid isoformat string: 'not'; want `<UTC ISO> <commit>`: left out",
        f"warning: --main-moves line 7: '{at('15:45')} {'f' * 12}' names no one commit of `main`'s first-parent line: left out",
        f"warning: --old-reds: {'e' * 12} names no one commit of `checked`'s first-parent line: left out",
    ]


def test_the_json_report(world, capsys):
    w, h, q = world, world.heads, world.q
    assert vc.main(w.argv("--main-moves", str(w.moves), "--old-reds", str(w.reds), "--json")) == 1
    rep = json.loads(capsys.readouterr().out)
    assert rep["window"] == {"since": at("14:00"), "until": at("16:00")} and rep["main_moves"] is True
    assert rep["log"]["decisions"] == 21 and rep["log"]["not_shadow"] == 0 and [u["line"] for u in rep["log"]["unreadable"]] == [5]
    rows = {r["pr"]: r for r in rep["prs"]}
    assert sorted(rows) == list(range(101, 110))
    assert rows[104] == {
        "pr": 104, "head": h[104], "marked": at("14:03"),
        "first_build": {"at": at("14:07"), "sha": C3, "head": h[104]}, "advance": None,
        "eject": {"at": at("14:45"), "head": h[104], "why": EJECT_104, "failed": [{"task": "pytest", "key": "k104"}]},
        "skip": None, "landed": {"at": at("14:13"), "commit": q[104], "head": h[104]}, "reverted": None, "ejected": None,
        "main": at("15:30"), "on_main": True}
    assert rows[103]["reverted"] == {"at": at("14:50"), "commit": w.revert, "landed": q[103], "head": h[103]}
    assert rows[103]["landed"] is None and rows[103]["main"] is None
    assert rows[107]["ejected"] == {"at": at("14:50"), "by": "ci", "why": EJECTED_107, "head": h[107]}
    agg = rep["aggregates"]
    assert agg["mark_to_advance"] == {"n": 4, "p50_h": pytest.approx(34 / 60), "p90_h": pytest.approx(50 / 60)}
    assert agg["mark_to_main"] == {"n": 3, "p50_h": pytest.approx(1.0), "p90_h": pytest.approx(1.45)}
    assert (agg["shadow_only"], agg["old_only"]) == ([103, 107], [104])
    kinds = [(d["kind"], d["pr"]) for d in rep["disagreements"]]
    assert sorted(kinds, key=lambda k: (k[0], k[1] or 0)) == [("a", 104), ("b", 103), ("b", 107), ("c", 105), ("c", 106),
                                                             ("d", None)]
    d = next(d for d in rep["disagreements"] if d["kind"] == "d")
    assert (d["sha"], d["base"], d["prs"], d["old"]) == (C1, w.m0, [101, 102], {"tip": q[102], "red": ["pytest"]})
    live = rep["liveness"]
    assert live["longest_pending"] == {"hours": 1.0, "sha": C6, "members": [109], "built": at("15:00"), "ended": None}
    assert live["dropped_stale"] == [{"at": at("14:43"), "sha": C5, "why": "stale"}]
    assert live["retries"] == [{"at": at("14:20"), "sha": C3, "why": "graph2 lost its worker"}]
    assert live["rings"] == [{"at": at("14:25"), "sha": C3, "why": "infra three times"}]
    assert live["failed_actions"] == [{"at": at("15:05"), "action": "drop", "sha": C6, "error": PUSH_FAILED}]
    assert rep["soundness"] == {"advances": 3, "passed": 3, "violations": [], "ok": True}


def test_without_main_moves_the_old_paths_main_is_main_now(world, capsys):
    """Without `--main-moves`, when `main` reached a commit is unknown: no mark -> main spread, and the old path's PRs on `main`
    are those on it now.  Without `--old-reds` no candidate is compared with a red."""
    assert vc.main(world.argv()) == 1
    text = capsys.readouterr().out
    assert f"{'mark -> old-path main':28} unknown: no --main-moves" in text.splitlines()
    assert ("on main by the shadow only: 2 (#103 #107); by the old path only: 1 (#104) (the old path's: on `main` now)"
            in text.splitlines())
    rows = table(text)
    assert rows["#101"][-1] == rows["#104"][-1] == "yes, time unknown" and rows["#106"][-1] == "-"
    assert "disagreements: 5" in text and "(d)" not in text


def test_the_window_picks_what_is_reported_from_everything_logged_before_its_end(world, capsys):
    """A window that opens after C1 advanced and closes before C6 does.  C2's advance in it names no members, and its build
    before the window says they are 103's; 101 and 102, which C1 put on `main` before the window and the old path's `main`
    reached in it, are seen and disagree on nothing; C1, which ended before the window, isn't pending in it."""
    argv = ["--repo", str(world.repo), "--log", str(world.log), "--since", at("14:35"), "--until", at("15:10"),
            "--main-moves", str(world.moves), "--json"]
    assert vc.main(argv) == 1
    rep = json.loads(capsys.readouterr().out)
    assert rep["log"]["decisions"] == 11 and rep["log"]["ticks"] == 6
    rows = {r["pr"]: r for r in rep["prs"]}
    assert sorted(rows) == [101, 102, 103, 104, 107, 108, 109]
    assert rows[101]["advance"] == {"at": at("14:30"), "sha": C1, "head": world.heads[101]} and rows[101]["main"] == at("15:00")
    assert rows[103]["advance"] == {"at": at("14:40"), "sha": C2, "head": world.heads[103]}
    agg = rep["aggregates"]
    assert agg["mark_to_advance"] == {"n": 2, "p50_h": pytest.approx(44 / 60), "p90_h": pytest.approx(50 / 60)}
    assert agg["mark_to_main"] == {"n": 2, "p50_h": pytest.approx(119 / 120), "p90_h": pytest.approx(1.0)}
    assert (agg["shadow_only"], agg["old_only"]) == ([103, 107], [])
    assert sorted((d["kind"], d["pr"]) for d in rep["disagreements"]) == [("a", 104), ("b", 103), ("b", 107)]
    live = rep["liveness"]
    assert live["longest_pending"] == {"hours": pytest.approx(38 / 60), "sha": C3, "members": [104], "built": at("14:07"),
                                       "ended": {"at": at("14:45"), "action": "drop"}}
    assert ([d["sha"] for d in live["dropped_stale"]], live["retries"], live["rings"]) == ([C5], [], [])
    assert [f["sha"] for f in live["failed_actions"]] == [C6]
    assert rep["soundness"] == {"advances": 2, "passed": 2, "violations": [], "ok": True}
    assert rep["warnings"][0].startswith("log line 5 isn't a decision") and rep["warnings"][1:] == [
        "--main-moves line 6: 'not a move': Invalid isoformat string: 'not'; want `<UTC ISO> <commit>`: left out",
        f"--main-moves line 7: '{at('15:45')} {'f' * 12}' names no one commit of `main`'s first-parent line: left out"]


def test_agreement_exits_zero(tmp_path, monkeypatch, capsys):
    """The shadow advances what the old path lands and `main` reaches: no disagreement, a sound advance, exit 0.  `--fetch`
    reads origin's `main` and `checked` as they are now, not as the clone last fetched them; without `--since` and `--until`
    the window runs from the log's first decision to the injected clock."""
    origin, repo = tmp_path / "origin", tmp_path / "repo"
    origin.mkdir()
    git(origin, "init", "-q")
    tree = git(origin, "mktree")
    m0 = git(origin, "commit-tree", tree, "-m", "the base", stamp="13:00")
    head = git(origin, "commit-tree", tree, "-p", m0, "-m", "change 1", stamp="13:10")
    q1 = git(origin, "commit-tree", tree, "-p", m0, "-p", head, "-m", f"Queue: #1 change\n\nPR: #1\nHead: {head}\nOwner: o",
             stamp="14:20")
    for ref in ("checked", "main"):
        git(origin, "update-ref", f"refs/heads/{ref}", m0)
    git(tmp_path, "clone", "-q", "--no-checkout", str(origin), str(repo))
    for ref in ("checked", "main"):
        git(origin, "update-ref", f"refs/heads/{ref}", q1)
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps(d) for d in (
        decision("14:05", 1, "build", m0, sha=C1, base=m0, members=[1], heads=[head]),
        decision("14:15", 2, "advance", m0, sha=C1, base=m0, members=[1], heads=[head]),
        decision("14:15", 2, "drop", m0, sha=C1, outcome={"members": [1], "heads": [head], "state": "pass", "suspects": []}))))
    labels = {(1, head): [Label(label_target(1, head), vocab.READY, True, "o", at("14:00"))]}
    monkeypatch.setattr(vc, "store_labels", lambda root: lambda heads: {k: labels.get(k, []) for k in heads})
    monkeypatch.setattr(vc, "clock", lambda: vc.when(at("15:00")))
    assert vc.main(["--repo", str(repo), "--log", str(log)]) == 0          # the clone's `main` and `checked` are the base
    assert "on main by the shadow only: 1 (#1); by the old path only: 0" in capsys.readouterr().out
    assert vc.main(["--repo", str(repo), "--log", str(log), "--fetch"]) == 0
    text = capsys.readouterr().out
    assert text.startswith(f"vqueue compare {hm('14:05')} to {hm('15:00')}: 3 decision(s) in 2 tick(s)")
    assert "disagreements: 0\n" in text and "on main by the shadow only: 0; by the old path only: 0" in text
    assert f"longest pending: 0.17 h, {C1[:12]} (#1) built {hm('14:05')}, advance {hm('14:15')}" in text


def test_an_input_that_cant_be_read_exits_two(world, capsys):
    assert vc.main(world.argv("--checked", "origin/nowhere")) == 2
    assert "research vqueue-compare: git log" in capsys.readouterr().err
    assert vc.main(["--repo", str(world.repo), "--log", str(world.tmp / "absent.jsonl")]) == 2
    assert "absent.jsonl" in capsys.readouterr().err
    assert vc.main(world.argv("--until", "yesterday")) == 2


# ---- soundness and liveness, over parsed decisions ------------------------------------------------------------------------------

M0, X1, X2, X3, X4, X5 = ("a0" + "0" * 38, *(f"d{i}" + "0" * 38 for i in range(1, 6)))


def test_an_advance_off_the_logged_main_or_of_a_failed_candidate_is_unsound():
    """Two advances in one tick are sound when the second's base is the first; an advance whose base isn't the logged `main`,
    one whose candidate's logged outcome failed, and one that names no base each violate it."""
    fail = {"members": [4], "heads": [], "state": "fail", "suspects": [4]}
    entries, bad = vc.parse_log("\n".join(json.dumps(d) for d in (
        decision("10:00", 1, "build", M0, sha=X1, base=M0, members=[1]),
        decision("10:00", 1, "build", M0, sha=X2, base=X1, members=[2]),
        decision("10:10", 2, "advance", M0, sha=X1, base=M0),
        decision("10:10", 2, "drop", M0, sha=X1, why="on main"),
        decision("10:10", 2, "advance", M0, sha=X2, base=X1),
        decision("10:15", 3, "build", X2, sha=X3, base=M0, members=[3]),
        decision("10:30", 4, "advance", X2, sha=X3, base=M0),
        decision("10:40", 5, "build", X3, sha=X4, base=X3, members=[4]),
        decision("10:50", 6, "drop", X3, sha=X4, outcome=fail),
        decision("10:50", 6, "advance", X3, sha=X4, base=X3),
        decision("11:00", 7, "advance", X4, sha=X5),
    )))
    assert bad == []
    rep = vc.compare(entries, vc.parse_chain(""), [], None, {}, lambda heads: {}, vc.when(at("10:00")), vc.when(at("12:00")))
    assert rep["soundness"]["ok"] is False and rep["soundness"]["advances"] == 5 and rep["soundness"]["passed"] == 0
    assert [(v["sha"], v["why"]) for v in rep["soundness"]["violations"]] == [
        (X3, "its base isn't the logged `main`"), (X4, "the outcome logged for it failed"), (X5, "it names no base")]
    text = vc.render(rep)
    assert "soundness: VIOLATED by 3 of 5 advance(s):" in text
    assert f"  {X3[:12]} {hm('10:30')}: its base isn't the logged `main` (base {M0[:12]}, main {X2[:12]})" in text
    assert rep["warnings"] == [f"advance of {X5} at {at('11:00')} names no members, and the log has no build of it: its PRs "
                               f"aren't counted"]
    assert f"warning: {rep['warnings'][0]}" in text.splitlines()
    assert rep["liveness"]["longest_pending"]["sha"] == X3 and rep["liveness"]["longest_pending"]["hours"] == 0.25
    assert rep["liveness"]["dropped_stale"] == []                      # a drop `on main` isn't stale


def test_an_advance_of_a_sha_the_log_never_built_is_unsound_on_the_logged_main_too():
    """`main` moves only to a candidate the loop built and judged: an advance on the logged `main` naming its members, of a sha
    no build line names, violates soundness."""
    entries, bad = vc.parse_log("\n".join(json.dumps(d) for d in (
        decision("10:00", 1, "build", M0, sha=X1, base=M0, members=[1]),
        decision("10:10", 2, "advance", M0, sha=X1, base=M0),
        decision("10:20", 3, "advance", X1, sha=X2, base=X1, members=[2]),
    )))
    assert bad == []
    rep = vc.compare(entries, vc.parse_chain(""), [], None, {}, lambda heads: {}, vc.when(at("10:00")), vc.when(at("12:00")))
    assert rep["soundness"]["ok"] is False and rep["soundness"]["advances"] == 2
    assert [(v["sha"], v["base"], v["why"]) for v in rep["soundness"]["violations"]] == [
        (X2, X1, "the log holds no build of it")]
    assert f"  {X2[:12]} {hm('10:20')}: the log holds no build of it (base {X1[:12]}, main {X1[:12]})" in vc.render(rep)


def test_an_advance_that_lists_no_heads_holds_its_members_at_the_heads_built():
    """The old path landed 1 at one head and reverted it; the shadow built and advanced 1 at another, its advance listing
    members but no heads: no disagreement.  At the reverted head it is (b)."""
    chain = vc.parse_chain(_chain_text((A, "", "10:00", "root"), (Q1, f"{A} {HEAD[1]}", "10:01", _queue(1)),
                                       (R1, Q1, "10:02", f'Revert "Queue: #1 change"\n\nThis reverts commit {Q1}.')))
    for built, kinds in ((HEAD[2], []), (HEAD[1], [("b", 1)])):
        entries, _ = vc.parse_log("\n".join(json.dumps(d) for d in (
            decision("10:05", 1, "build", A, sha=X1, base=A, members=[1], heads=[built]),
            decision("10:10", 2, "advance", A, sha=X1, base=A, members=[1]))))
        rep = vc.compare(entries, chain, [A], None, {}, lambda heads: {}, vc.when(at("10:00")), vc.when(at("11:00")))
        assert [(d["kind"], d["pr"]) for d in rep["disagreements"]] == kinds


def test_a_build_logged_with_an_error_after_its_push_is_a_candidate_still_and_a_failed_action():
    """The loop pushed X1 and its enqueue failed, so its build line names its sha beside the error; it was enqueued again and
    advanced.  A build that failed before its push names no sha and is no candidate."""
    entries, bad = vc.parse_log("\n".join(json.dumps(d) for d in (
        decision("10:05", 1, "build", M0, sha=X1, base=M0, members=[1], heads=[HEAD[1]], enqueued=False, error="graph2 is down"),
        decision("10:06", 2, "retry", M0, sha=X1, base=M0, members=[1], heads=[HEAD[1]], enqueued=True),
        decision("10:10", 3, "advance", M0, sha=X1, base=M0, members=[1], heads=[HEAD[1]]),
        decision("10:11", 4, "build", X1, base=X1, members=[2], error="git push: connection reset"))))
    s = vc.shadow(entries)
    assert not bad and list(s.cands) == [X1] and s.cands[X1].passed is entries[2] and s.built[1] == (entries[0], HEAD[1])
    assert [(e.action, e.sha) for e in s.errors] == [("build", X1), ("build", None)] and s.nobuild == []


def test_unread_labels_and_an_empty_window_are_warned():
    entries, _ = vc.parse_log(json.dumps(decision("10:00", 1, "build", M0, sha=X1, base=M0, members=[1], heads=["1" * 40])))
    rep = vc.compare(entries, vc.parse_chain(""), [], None, {}, lambda heads: {}, vc.when(at("09:00")), vc.when(at("11:00")))
    assert rep["warnings"] == [f"the `ready` labels of 1 head(s) weren't read, so their marks are unknown: pr:1@{'1' * 12}"]
    assert rep["prs"][0]["marked"] is None
    later = vc.compare(entries, vc.parse_chain(""), [], None, {}, lambda heads: {}, vc.when(at("12:00")), vc.when(at("13:00")))
    assert later["warnings"] == ["no decision in the window"] and later["prs"] == []
    assert f"longest pending: 3.00 h, {X1[:12]} (#1) built {hm('10:00')}, still pending" in vc.render(later).splitlines()
    empty = vc.compare([], vc.parse_chain(""), [], None, {}, lambda heads: {}, vc.when(at("12:00")), vc.when(at("13:00")))
    assert "longest pending: no candidate built" in vc.render(empty).splitlines()


# ---- the parsers ----------------------------------------------------------------------------------------------------------------

def test_parse_log_orders_decisions_and_reports_each_other_line():
    good = decision("10:05", 2, "build", M0, sha=X1, members=[1, 2], heads=["1" * 40, "2" * 40], later=1)
    text = "\n".join([json.dumps(good), "", '{"at": "x', "[1]", json.dumps({"at": at("10:00"), "tick": 1}),
                      json.dumps({**good, "outcome": "pass"}), json.dumps({**good, "at": "noon"}),
                      json.dumps(decision("10:00", 1, "skip", M0, number=3, heads=["3" * 40]))])
    entries, bad = vc.parse_log(text)
    assert [(e.line, e.action) for e in entries] == [(8, "skip"), (1, "build")]
    assert [i for i, _ in bad] == [3, 4, 5, 6, 7]
    assert bad[1][1] == "TypeError: not a JSON object" and bad[2][1] == "KeyError: 'action'"
    skip, build = entries
    assert skip.prs == (3,) and skip.head(3) == "3" * 40 and build.prs == (1, 2) and build.head(2) == "2" * 40


def _chain_text(*recs: tuple[str, str, str, str]) -> str:
    """`git log --first-parent --format=CHAIN_FORMAT` of ``recs`` (commit, parents, HH:MM, message), given oldest first."""
    return "".join(f"{c}\x00{p}\x00{int(vc.when(at(t)).timestamp())}\x00{body}\n\x1e\n" for c, p, t, body in reversed(recs))


def _sha(tag: str) -> str:
    return tag.encode().hex().ljust(40, "0")[:40]


A, Q1, R1, RR, Q2, R2, LK, Q3, RA, Q4 = map(_sha, ("A", "Q1", "R1", "RR", "Q2", "R2", "LK", "Q3", "RA", "Q4"))
HEAD = {n: _sha(f"h{n}") for n in (1, 2, 3, 4)}


def _queue(n: int) -> str:
    return f"Queue: #{n} change\n\nPR: #{n}\nHead: {HEAD[n]}\nOwner: o"


def test_parse_chain_reads_landings_and_the_reverts_that_undo_them():
    """A revert of a revert relands; a `Revert "Queue: #N` subject naming no commit undoes N's newest landing; a revert of a
    commit that isn't a landing undoes none."""
    chain = vc.parse_chain(_chain_text(
        (A, "", "10:00", "root"),
        (Q1, f"{A} {HEAD[1]}", "10:01", _queue(1)),
        (R1, Q1, "10:02", f'Revert "Queue: #1 change"\n\nThis reverts commit {Q1}, reversing\nchanges made to {A}.'),
        (RR, R1, "10:03", f'Revert "Revert "Queue: #1 change""\n\nThis reverts commit {R1}.'),
        (Q2, f"{RR} {HEAD[2]}", "10:04", _queue(2)),
        (R2, Q2, "10:05", 'Revert "Queue: #2 change"'),
        (LK, R2, "10:06", "Lock records regenerated for #2"),
        (Q3, f"{LK} {HEAD[3]}", "10:07", _queue(3)),
        (RA, Q3, "10:08", f'Revert "the root"\n\nThis reverts commit {A}.'),
        (Q4, f"{RA} {HEAD[4]}", "10:09", _queue(4)),
    ))
    assert chain.commits == [A, Q1, R1, RR, Q2, R2, LK, Q3, RA, Q4] and chain.index[Q2] == 4
    assert {c: (x.pr, x.head, x.undone) for c, x in chain.landings.items()} == {
        Q1: (1, HEAD[1], None), Q2: (2, HEAD[2], R2), Q3: (3, HEAD[3], None), Q4: (4, HEAD[4], None)}
    assert chain.landings[Q2].undone_at == vc.when(at("10:05")) and chain.landings[Q2].at == vc.when(at("10:04"))
    assert chain.undoes == {R1: (Q1,), RR: (R1,), R2: (Q2,), RA: (A,)}


def test_same_prs_nets_reverts_and_refuses_another_tree():
    chain = vc.parse_chain(_chain_text(
        (A, "", "10:00", "root"),
        (Q1, f"{A} {HEAD[1]}", "10:01", _queue(1)),
        (Q2, f"{Q1} {HEAD[2]}", "10:02", _queue(2)),
        (R1, Q2, "10:03", f'Revert "Queue: #1 change"\n\nThis reverts commit {Q1}.'),
        (LK, R1, "10:04", "Lock records regenerated"),
        (Q3, f"{LK} {HEAD[3]}", "10:05", _queue(3)),
        (RA, Q3, "10:06", f"Revert the root\n\nThis reverts commit {A}."),
        (Q4, f"{RA} {HEAD[4]}", "10:07", _queue(4)),
    ))
    red = ["pytest"]
    assert vc.same_prs(chain, A, {2: HEAD[2]}, {LK: red}) == LK                       # 1 landed and was reverted below the tip
    assert vc.same_prs(chain, A, {2: None}, {LK: red}) == LK                          # a head the log didn't give matches any
    assert vc.same_prs(chain, A, {2: HEAD[1]}, {LK: red}) is None                     # another head
    assert vc.same_prs(chain, A, {1: HEAD[1], 2: HEAD[2]}, {Q2: red, LK: red}) == Q2   # the first tip that matches
    assert vc.same_prs(chain, Q1, {2: HEAD[2]}, {Q2: red}) == Q2
    assert vc.same_prs(chain, Q1, {2: HEAD[2], 3: HEAD[3]}, {Q3: red}) is None         # R1 reverts a commit below the base
    assert vc.same_prs(chain, LK, {3: HEAD[3], 4: HEAD[4]}, {Q4: red}) is None        # RA too
    assert vc.same_prs(chain, LK, {3: HEAD[3]}, {Q3: red, A: red}) == Q3              # a red below the base isn't read


def test_parse_moves_and_arrivals():
    line = [A, Q1, Q2, Q3, Q4]
    text = "\n".join(["# main's moves", "", f"{at('10:00')} {Q1}", f"{at('11:00')} {Q4[:7]}", f"{at('10:30')} {Q3.upper()}",
                      f"{at('10:40')} {Q2[:6]}", f"noon {Q2}", f"{at('10:50')}", f"{at('10:50')} {R1}"])
    moves, warn = vc.parse_moves(text, line)
    assert moves == [(vc.when(at("10:00")), Q1), (vc.when(at("11:00")), Q4), (vc.when(at("10:30")), Q3)]
    assert [w.split(":")[0] for w in warn] == ["--main-moves line 6", "--main-moves line 7", "--main-moves line 8",
                                               "--main-moves line 9"]
    assert "want `<UTC ISO> <commit>`" in warn[1]
    assert vc.arrivals(line, moves) == {Q2: vc.when(at("10:30")), Q3: vc.when(at("10:30")), Q4: vc.when(at("11:00"))}
    assert vc.arrivals(line, []) == {}
    off, warn = vc.parse_moves(f"{at('10:00')} {A}\n{at('10:20')} {LK}\n{at('10:40')} {RA}\n", line, {LK: Q2}.get)
    assert (off, len(warn)) == ([(vc.when(at("10:00")), A), (vc.when(at("10:20")), Q2)], 1)   # LK is off the line, and holds Q2
    assert vc.arrivals(line, off) == {Q1: vc.when(at("10:20")), Q2: vc.when(at("10:20"))}


def test_a_move_off_mains_first_parent_line_reads_as_the_newest_commit_of_it_held(world):
    """`main` moved to a commit `checked` merged later as a second parent, such as a lock commit: it brought the commits of
    `main`'s first-parent line it holds."""
    line = git(world.repo, "rev-list", "--first-parent", "--reverse", "refs/remotes/origin/main").split()
    side = git(world.repo, "commit-tree", git(world.repo, "mktree"), "-p", world.q[102], "-m", "Lock records regenerated",
               stamp="14:30")
    onto = vc._onto(world.repo, line)
    assert onto(side) == onto(side[:10]) == world.q[102] and onto(world.q[103]) == world.q[103]
    assert onto("f" * 40) is None


def test_parse_reds_resolves_tips_on_checked():
    chain = vc.parse_chain(_chain_text((A, "", "10:00", "root"), (Q1, f"{A} {HEAD[1]}", "10:01", _queue(1))))
    assert vc.parse_reds({Q1[:8]: ["lean"], A: "pytest", "f" * 9: ["x"]}, chain) == (
        {Q1: ["lean"], A: ["pytest"]}, [f"--old-reds: {'f' * 9} names no one commit of `checked`'s first-parent line: left out"])
    assert vc.parse_reds([Q1], chain) == ({}, ["--old-reds isn't a JSON object {tip: [step, ...]}: no red is compared"])


# ---- the labels -----------------------------------------------------------------------------------------------------------------

def test_marks_and_ejections_from_labels():
    head = "1" * 40

    def lab(t: str, value, ref=None):
        return Label(label_target(1, head), vocab.READY, value, "o", at(t), ref)

    labs = [lab("10:00", True), lab("11:00", False, "pytest"), lab("12:00", "true")]
    assert vc.marked(labs) == vc.when(at("12:00")) and vc.marked(labs, vc.when(at("11:59"))) == vc.when(at("10:00"))
    assert vc.marked(labs, vc.when(at("09:00"))) is None and vc.marked([]) is None
    assert vc.ejection(labs) is None and vc.ejection(labs[:2]).ref == "pytest"
    assert vc.ejection([lab("10:00", True), lab("11:00", False)]) is None     # a mark withdrawn with no reason isn't an ejection


def test_store_labels_reads_ready_labels_from_the_stores_remote(tmp_path, monkeypatch, capsys):
    """The labels of each head, pulled from the store's remote; a store with no remote reads none, and says so."""
    head, other = "1" * 40, "2" * 40
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "absent.toml"))
    assert vc.store_labels(str(tmp_path / "bare"))([(1, head)]) == {}
    assert "the `ready` labels of 1 of 1 head(s) weren't read" in capsys.readouterr().err
    (cfg := tmp_path / "store.toml").write_text(f'[remote]\ntype = "fs"\npath = "{tmp_path / "vault"}"\n')
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(cfg))
    lane = LocalStore(tmp_path / "lane")
    lane.add_label(label_target(1, head), vocab.READY, True, "lane-1", ts=at("10:00"))
    assert labels_sync.push_targets(lane, [label_target(1, head)])["failed"] == []
    got = vc.store_labels(str(tmp_path / "reader"))([(1, head), (2, other)])
    assert [(lab.value, lab.ts) for lab in got[(1, head)]] == [(True, at("10:00"))] and got[(2, other)] == []
