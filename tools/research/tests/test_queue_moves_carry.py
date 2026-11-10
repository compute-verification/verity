"""A mark and a land verdict at a PR's old head carry to the head `tools/move/restack.py` makes of it across a move `main`
landed (`Queue.mapped_change`, `StoreLabels`'s ``moved``): the new head's own change is the old one's with the move's map
applied, as the restack applies it.  The same restack with one line edited besides keeps neither, and a PR the map doesn't
touch keeps both by its plain own change, as before.  The scripts are this checkout's, copied into a throwaway origin with a
word map that moves `old_proto/` to `new_proto/` and rewrites the word, and the move lands as the loop lands a PR: a merge
whose second parent is the move."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import research.queue as Q
from research.queue import StoreLabels, label_target
from research.store import vocab
from test_queue import World, _git

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ("tools/move/rename.py", "tools/move/layout.py", "tools/move/restack.py")
MAP = "tools/move/demo_map.toml"
A = "def f():\n    return 1\n\n\ndef g():\n    return 2\n"
B = "import old_proto.a\n\nX = old_proto.a.f()\n\n\nZ = 1\n"
ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """An origin whose `main` holds the move scripts, the map and `old_proto/`; `move` is the map's commit, which `land` lands."""
    monkeypatch.setattr(Q, "MERGED_ROOT", tmp_path / "scratch")
    w = World(tmp_path)
    w.with_store_remote()
    _push_main(w, w.change({**{p: (ROOT / p).read_text() for p in SCRIPTS}, "tools/verity/check/pins.py": "PINS = ()\n",
                            "tools/research/src/research/store/moves.json": '{"moves": []}\n',
                            MAP: '[rewrite]\nexclude = ["tools/move/**", "tools/research/src/research/store/moves.json"]\n\n'
                                 '[words]\n"old_proto" = "new_proto"\n',
                            "old_proto/__init__.py": "", "old_proto/a.py": A, "b.py": B}))
    return w


def _push_main(world: World, commit: str) -> None:
    _git(world.dev, "push", "-q", "origin", f"{commit}:refs/heads/main")
    _git(world.dev, "fetch", "-q", "origin")


def _move(world: World) -> str:
    """The map's commit on `main` as it is: what `rename.py` writes there, with its move line, as `stage.py` commits it."""
    _git(world.dev, "checkout", "-q", "--detach", "origin/main")
    subprocess.run([sys.executable, "tools/move/rename.py", "--map", MAP], cwd=world.dev, env=ENV, check=True,
                   capture_output=True)
    _git(world.dev, "add", "-A")
    _git(world.dev, "commit", "-q", "-m", f"Move by {MAP}\n\nMove: {MAP}\n")
    return _git(world.dev, "rev-parse", "HEAD")


def _land(world: World, move: str) -> str:
    """``move`` landed as the loop lands a PR, after another PR moved `main`: `main` merges it as its second parent."""
    _push_main(world, other := world.change({"c.txt": "c landed meanwhile\n"}))
    tree = _git(world.dev, "merge-tree", "--write-tree", other, move)
    _push_main(world, landed := _git(world.dev, "commit-tree", tree, "-p", other, "-p", move, "-m", "Queue: #9 the move"))
    return landed


def _restack(world: World, head: str) -> str:
    """``head`` carried across `main`'s moves by `tools/move/restack.py`, as its owner runs it, and pushed."""
    r = subprocess.run([sys.executable, str(ROOT / "tools/move/restack.py"), head, "--main", "origin/main", "--json"],
                       cwd=world.dev, env=ENV, capture_output=True, text=True)
    (got,) = json.loads(r.stdout)
    assert got["status"] == "clean", got
    _git(world.dev, "push", "-q", "origin", f"{got['result']}:refs/heads/restacked-{head[:8]}")
    return got["result"]


def _mark(world: World, n: int, head: str) -> None:
    world.store.add_label(label_target(n, head), vocab.READY, True, f"lane-{n}", ts="2026-11-10T10:00:00Z")
    world.store.add_label(label_target(n, head), vocab.REVIEWED, "land", "reviewer", ts="2026-11-10T11:00:00Z")


def _queue(world: World) -> Q.Queue:
    q = world.queue()
    q._clone()
    q.fetch(world.gh.open_prs())
    return q


def _kept(labels: StoreLabels, n: int, head: str) -> tuple[bool, list]:
    return labels.ready(n, head), [(lab.value, lab.by) for lab in labels.reviews(n, head)]


def test_a_restack_across_a_move_keeps_the_mark_and_verdict_and_one_more_edit_or_a_plain_merge_loses_both(world):
    """#1 edits the file the map moves and adds a line with the map's word; restacked across the landed move by `restack.py`, its
    own change is no longer its old head's (the file moved, the word changed), but it is that change with the map applied, so
    the new head takes the old head's mark and verdict, and the log says so.  The same restack with one more line edited takes
    neither, and so does a plain merge of `main`, which leaves the old word in the line #1 added."""
    world.gh.put(1, old := world.change({"old_proto/a.py": A.replace("return 2", "return 3"),
                                         "b.py": B + "Y = old_proto.a.g()\n"}), ready=False)
    _mark(world, 1, old)
    _land(world, _move(world))
    world.gh.put(1, new := _restack(world, old), ready=False)
    q = _queue(world)
    assert _git(q.workdir, "show", f"{new}:b.py").endswith("Y = new_proto.a.g()")
    assert q.own_change(new) not in (None, q.own_change(old))
    labels = StoreLabels(world.store, log=world.logs.append, change=q.own_change, moved=q.mapped_change)
    assert _kept(labels, 1, new) == (True, [("land", "reviewer")])
    assert labels.ready_by(1, new) == "lane-1" and labels.marked_at(1, new) == "2026-11-10T10:00:00Z"
    assert [s for s in world.logs if "takes the" in s] == [
        f"research queue: #1 at {new[:8]} takes the `{key}` labels at {old[:8]}: none of its own counts, and its own change is "
        f"theirs across the moves `main` landed since, as a restack carries it (patch-id {q.own_change(new)[:12]})"
        for key in (vocab.READY, vocab.REVIEWED)]
    assert not any(Path(world.tmp, "scratch").iterdir())                 # each script's scratch worktree is gone

    extra = world.change({"new_proto/a.py": _git(world.dev, "show", f"{new}:new_proto/a.py").replace("return 1", "return 11")},
                         base=new)
    _git(world.dev, "checkout", "-q", "--detach", old)
    _git(world.dev, "merge", "-q", "--no-edit", "origin/main")
    plain = _git(world.dev, "rev-parse", "HEAD")
    for head in (extra, plain):
        _git(world.dev, "push", "-q", "origin", f"{head}:refs/heads/other-{head[:8]}")
        world.gh.put(1, head, ready=False)
        q.fetch(world.gh.open_prs())
        fresh = StoreLabels(world.store, change=q.own_change, moved=q.mapped_change)
        assert q.mapped_change(old, head) == q.own_change(new) != q.own_change(head)
        assert _kept(fresh, 1, head) == (False, [])


def test_a_pr_the_map_doesnt_touch_keeps_both_by_its_plain_own_change_and_a_head_with_no_move_between_maps_to_none(world):
    """#2 changes a file the map doesn't name: restacked across the move, its own change is its old head's, so it keeps the mark
    and verdict as before the move rule, by its plain own change.  A head crossing no move has no mapped change."""
    world.gh.put(2, old := world.change({"a.txt": "a changed\n"}), ready=False)
    _mark(world, 2, old)
    _land(world, _move(world))
    world.gh.put(2, new := _restack(world, old), ready=False)
    q = _queue(world)
    assert q.own_change(new) == q.own_change(old) is not None
    labels = StoreLabels(world.store, log=world.logs.append, change=q.own_change, moved=q.mapped_change)
    assert _kept(labels, 2, new) == (True, [("land", "reviewer")])
    assert [s.split(": none of its own counts, ")[1] for s in world.logs if "takes the" in s] == [
        f"and its own change is theirs (patch-id {q.own_change(new)[:12]})"] * 2
    assert q.mapped_change(new, new) is None and q.mapped_change(old, old) is None


def test_with_mapped_jobs_a_head_waits_until_work_out_mapped_has_mapped_it_and_the_mapping_is_kept(world):
    """With `mapped_jobs` (the loop's), `mapped_change` only asks: the new head takes nothing until `work_out_mapped` maps the
    old head, which the loop does at the end of a tick; then it takes both, and the mapping is kept in `mapped-changes.json`, so
    a queue started afresh reads it there without running a script.  A budget used up starts nothing more."""
    world.gh.put(1, old := world.change({"old_proto/a.py": A.replace("return 2", "return 3")}), ready=False)
    _mark(world, 1, old)
    onto = _land(world, _move(world))
    world.gh.put(1, new := _restack(world, old), ready=False)
    q = _queue(world)
    q.mapped_jobs = 2
    labels = StoreLabels(world.store, change=q.own_change, moved=q.mapped_change)
    assert _kept(labels, 1, new) == (False, [])
    assert q.work_out_mapped(budget=0.0) == 0 and q.work_out_mapped() == 0     # a used-up budget drops the wanted heads
    assert _kept(StoreLabels(world.store, change=q.own_change, moved=q.mapped_change), 1, new) == (False, [])
    assert q.work_out_mapped() == 1 and q.work_out_mapped() == 0
    assert _kept(StoreLabels(world.store, change=q.own_change, moved=q.mapped_change), 1, new) == (True, [("land", "reviewer")])
    kept = json.loads((Path(_git(q.workdir, "rev-parse", "--absolute-git-dir")) / "mapped-changes.json").read_text())
    assert kept == {f"{old} {onto}": q.own_change(new)}

    again = _queue(world)
    again.mapped_jobs = 2
    again._crossed = {(onto, old): [("0" * 40, ["0" * 40])]}                  # a script run now would fail
    assert again.mapped_change(old, new) == q.own_change(new)


def test_an_old_head_that_doesnt_merge_with_the_moves_base_or_whose_script_fails_maps_to_none_and_says_why(world, monkeypatch):
    """`mapped_change` maps nothing it can't: a head whose merge with the move's base conflicts, and one the move's script
    fails on, each logged once with why, and neither takes a label."""
    world.gh.put(1, old := world.change({"c.txt": "c from the PR\n"}), ready=False)
    _mark(world, 1, old)
    _push_main(world, world.change({"c.txt": "c on main\n"}))
    move = _move(world)
    _push_main(world, move)
    _git(world.dev, "push", "-q", "origin", f"{old}:refs/heads/old")
    world.gh.put(1, new := world.change({"c.txt": "c merged\n"}), ready=False)
    q = _queue(world)
    assert q.mapped_change(old, new) is None
    assert [s for s in world.logs if "isn't mapped" in s] == [
        f"research queue: {old[:8]}'s own change isn't mapped across the moves {_git(q.workdir, 'merge-base', q.base, new)[:8]} "
        f"has: it doesn't merge with {_git(q.workdir, 'rev-parse', f'{move}^')[:12]}, the base of the move {move[:12]}: c.txt"]

    monkeypatch.setattr(Q, "_MAPPED", "import sys; sys.exit('the script fails')")
    world.gh.put(2, two := world.change({"a.txt": "a from #2\n"}, base=f"{move}^"), ready=False)
    _git(world.dev, "push", "-q", "origin", f"{two}:refs/heads/two")
    q.fetch(world.gh.open_prs())
    assert q.mapped_change(two, new) is None
    assert world.logs[-1].endswith(f"the move {move[:12]}'s script fails on it: exit 1: the script fails")
