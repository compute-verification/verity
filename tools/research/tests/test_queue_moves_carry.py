"""A mark and a land verdict at a PR's old head carry to the head `tools/move/restack.py` makes of it across a move `main`
landed (`Queue.mapped_change`, `StoreLabels`'s ``moved``): the new head's own change is the old one's with the move's map
applied, as the restack applies it.  The same restack with one line edited besides keeps neither, and a PR the map doesn't
touch keeps both by its plain own change, as before.  The scripts are this checkout's, copied into a throwaway origin with a
word map that moves `old_proto/` to `new_proto/` and rewrites the word, and the move lands as the loop lands a PR: a merge
whose second parent is the move."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import research.queue as Q
from research.queue import StoreLabels, label_target
from research.store import vocab
from test_queue import World, _git
from waits import Gate, wait_exit

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ("tools/move/rename.py", "tools/move/layout.py", "tools/move/restack.py")
MAP, PATHS, LIB = "tools/move/demo_map.toml", "tools/move/demo_paths_map.toml", "tools/move/demo_lib_map.toml"
A = "def f():\n    return 1\n\n\ndef g():\n    return 2\n"
B = "import old_proto.a\n\nX = old_proto.a.f()\n\n\nZ = 1\n"
ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
#: the start of a `pins.py` that `rename.py` imports from the tree it runs on: the git directory of that worktree, the clone's
GIT_DIR = ("import json, pathlib, subprocess\n"
           "c = pathlib.Path(subprocess.run(['git', 'rev-parse', '--git-common-dir'], capture_output=True, text=True)"
           ".stdout.strip()).resolve()\n")
#: one that plants a hook there, which the clone's next fetch would run with the queue's credentials
PLANT = (GIT_DIR + "(c / 'hooks').mkdir(exist_ok=True)\n"
         "(c / 'hooks' / 'reference-transaction').write_text('#!/bin/sh\\ncat > /dev/null\\n')\nPINS = ()\n")


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
                            PATHS: '[roots]\nold = { old_proto = "old_proto" }\nnew = { new_proto = "new_proto" }\n\n'
                                   '[paths]\n"old_proto/" = "new_proto/"\n\n[modules]\n"old_proto" = "new_proto"\n\n'
                                   '[rewrite]\nexclude = ["tools/move/**"]\n',
                            LIB: '[roots]\nold = { new_proto = "new_proto" }\nnew = { new_proto = "lib/new_proto" }\n\n'
                                 '[paths]\n"new_proto/" = "lib/new_proto/"\n\n[rewrite]\nexclude = ["tools/move/**"]\n',
                            "old_proto/__init__.py": "", "old_proto/a.py": A, "b.py": B}))
    return w


def _push_main(world: World, commit: str) -> None:
    _git(world.dev, "push", "-q", "origin", f"{commit}:refs/heads/main")
    _git(world.dev, "fetch", "-q", "origin")


def _move(world: World, path: str = MAP, on: str = "origin/main") -> str:
    """A map's commit on ``on`` (`main` as it is), as `stage.py` commits it: what `rename.py` writes there for the word map, with
    its `Move:` line, or what `layout.py` writes for a path map, with its `Layout-move:` line."""
    _git(world.dev, "checkout", "-q", "--detach", on)
    script, args, line = (("tools/move/rename.py", ["--map", MAP], f"Move: {MAP}") if path == MAP
                          else ("tools/move/layout.py", ["--map", path, "--no-lock"], f"Layout-move: {path} --no-lock"))
    subprocess.run([sys.executable, script, *args], cwd=world.dev, env=ENV, check=True, capture_output=True)
    _git(world.dev, "add", "-A")
    _git(world.dev, "commit", "-q", "-m", f"Move by {args[1]}\n\n{line}\n")
    return _git(world.dev, "rev-parse", "HEAD")


def _land(world: World, move: str) -> str:
    """``move`` landed as the loop lands a PR, after another PR moved `main`: `main` merges it as its second parent."""
    _push_main(world, other := world.change({"c.txt": f"c landed beside {move[:8]}\n"}))
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
    neither, and so does a plain merge of `main`, which leaves the old word in the line #1 added.  A commit whose line only
    looks like a move's is no move, to either; and labels the store's authentication refuses carry to no head."""
    world.gh.put(1, old := world.change({"old_proto/a.py": A.replace("return 2", "return 3"),
                                         "b.py": B + "Y = old_proto.a.g()\n"}), ready=False)
    _mark(world, 1, old)
    _land(world, _move(world))
    _push_main(world, _git(world.dev, "commit-tree", "origin/main^{tree}", "-p", "origin/main", "-m",
                           f"Notes\n\nMove: {MAP} ran here\n"))
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

    class Unsigned:
        def why(self, lab, beside):
            return "unsigned"

    said: list[str] = []
    refused = StoreLabels(world.store, log=said.append, verify=Unsigned(), change=q.own_change, moved=q.mapped_change)
    assert _kept(refused, 1, new) == (False, []) and not [s for s in said if "takes the" in s]

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


def test_a_restack_across_a_path_move_keeps_the_mark_and_verdict_and_one_more_edit_loses_both(world):
    """The same across a path map, a commit ending in `Layout-move:` that `layout.py` wrote, as the layout's moves are staged:
    #1's edit of the file the map moves and its new import of the old module, restacked, are its old change with the map
    applied, so the new head takes both labels; with one more line edited it takes neither."""
    world.gh.put(1, old := world.change({"old_proto/a.py": A.replace("return 2", "return 3"),
                                         "c.py": "import old_proto.a\n\nW = 1\n"}), ready=False)
    _mark(world, 1, old)
    _land(world, _move(world, PATHS))
    world.gh.put(1, new := _restack(world, old), ready=False)
    q = _queue(world)
    assert _git(q.workdir, "show", f"{new}:c.py").startswith("import new_proto.a")
    assert _git(q.workdir, "show", f"{new}:new_proto/a.py").endswith("return 3")
    assert q.own_change(new) not in (None, q.own_change(old))
    assert _kept(StoreLabels(world.store, change=q.own_change, moved=q.mapped_change), 1, new) == (True, [("land", "reviewer")])

    world.gh.put(1, extra := world.change({"c.py": "import new_proto.a\n\nW = 2\n"}, base=new), ready=False)
    q.fetch(world.gh.open_prs())
    assert q.mapped_change(old, extra) == q.own_change(new) != q.own_change(extra)
    assert _kept(StoreLabels(world.store, change=q.own_change, moved=q.mapped_change), 1, extra) == (False, [])


def test_a_pr_based_before_main_moved_on_keeps_both_across_two_moves_landed_as_one_pr(world):
    """As the layout lands: a chain of moves staged on `main`'s tip, a word map and then a path map on what the first made,
    landed as one PR's merge.  #1 is based before `main` moved on, so the restack first merges the first move's base into it,
    and its own change across both is its old one with both maps applied: the new head takes both labels."""
    world.gh.put(1, old := world.change({"old_proto/a.py": A.replace("return 2", "return 3")}), ready=False)
    _mark(world, 1, old)
    _push_main(world, world.change({"d.txt": "main moved on\n"}))
    onto = _land(world, _move(world, LIB, on=(first := _move(world))))
    world.gh.put(1, new := _restack(world, old), ready=False)
    q = _queue(world)
    assert _git(q.workdir, "show", f"{new}:lib/new_proto/a.py").endswith("return 3")
    second = _git(q.workdir, "rev-parse", f"{onto}^2")
    assert [m for m, _ in Q.moves_between(q.workdir, onto, old)] == [first, second]
    assert q.own_change(new) not in (None, q.own_change(old))
    assert _kept(StoreLabels(world.store, change=q.own_change, moved=q.mapped_change), 1, new) == (True, [("land", "reviewer")])
    assert q._kept_mapped() == {f"{old} {hashlib.sha512(f'{first} {second}'.encode()).hexdigest()}": q.own_change(new)}


def test_a_pr_based_after_one_move_landed_crosses_only_the_next_and_keeps_both(world):
    """Moves that landed one at a time, as drive's did: #1 is based on `main` after the first landed, so it crosses only the
    second, and keeps both labels across it."""
    _land(world, _move(world))
    world.gh.put(1, old := world.change({"new_proto/a.py": A.replace("return 2", "return 3")}), ready=False)
    _mark(world, 1, old)
    onto = _land(world, second := _move(world, LIB))
    world.gh.put(1, new := _restack(world, old), ready=False)
    q = _queue(world)
    assert [m for m, _ in Q.moves_between(q.workdir, onto, old)] == [second]
    assert _git(q.workdir, "show", f"{new}:lib/new_proto/a.py").endswith("return 3")
    assert _kept(StoreLabels(world.store, change=q.own_change, moved=q.mapped_change), 1, new) == (True, [("land", "reviewer")])


def test_a_pr_whose_merge_of_the_moves_base_changes_its_own_change_keeps_neither(world):
    """`main` changed a line beside #1's since its base, so merging the move's base changes #1's own change, as a plain merge
    of `main` would, which keeps no label either: the restack is clean, but `mapped_change` maps #1 to nothing, and says why."""
    D = "".join(f"{i}\n" for i in range(1, 10))
    _push_main(world, world.change({"d.txt": D}))
    world.gh.put(1, old := world.change({"d.txt": D.replace("3\n", "three\n")}), ready=False)
    _mark(world, 1, old)
    _push_main(world, base := world.change({"d.txt": D.replace("5\n", "five\n")}))
    _land(world, move := _move(world))
    world.gh.put(1, new := _restack(world, old), ready=False)
    q = _queue(world)
    assert _kept(StoreLabels(world.store, log=world.logs.append, change=q.own_change, moved=q.mapped_change), 1, new) == \
        (False, [])
    assert world.logs[-1].endswith(f"its merge with {base[:12]}, the base of the move {move[:12]}, changes its own change")


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


def _drop(q: Q.Queue) -> Path:
    return Path(_git(q.workdir, "rev-parse", "--absolute-git-dir")) / "mapped-changes.d"


def _marks(q: Q.Queue) -> dict[str, dict]:
    """Each running mapping job's marker in `mapped-changes.d/` (`MAPPING_MARK`), by its old head's start."""
    return {p.name[:12]: json.loads(p.read_text()) for p in sorted(_drop(q).glob(f"*{Q.MAPPING_MARK}"))}


def _tick(world: World, olds: list[str], new: str, jobs: int = 2, now: float | None = None) -> tuple[Q.Queue, list, int]:
    """One of the loop's `--once` ticks, each a process of its own: a queue started afresh asks for each old head's mapping
    for ``new`` (`mapped_change`), then ends by working them out (`work_out_mapped`)."""
    q = _queue(world)
    q.mapped_jobs = jobs
    if now is not None:
        q.now = lambda: now
    got = [q.mapped_change(old, new) for old in olds]
    return q, got, q.work_out_mapped()


@pytest.fixture
def jobs(world, monkeypatch) -> Path:
    """Mapping jobs whose launches go under the test's directory, run by a stand-in `MAP_JOB`: it waits on a gate at
    GATES/HEAD (`Gate`) when there is one, then writes `write_mapped`'s record of the head, `patch-` and its start, or a
    refusal, `boom`, when GATES/HEAD.refuse exists, or exits 3 with none written when GATES/HEAD.exit does.  GATES is returned."""
    monkeypatch.setattr(tempfile, "tempdir", str(world.tmp / "tmp"))
    (gates := world.tmp / "gates").mkdir()
    monkeypatch.setattr(Q, "MAP_JOB", "import json, os, sys; from pathlib import Path\n"
                        "sys.path.insert(0, sys.argv[1]); import research.queue as Q\n"
                        "repo, head, onto, moves, out, root = sys.argv[2:]\n"
                        f"gate = os.path.join({str(gates)!r}, head)\n"
                        "if os.path.exists(gate): open(gate, 'rb').read(1)\n"
                        "if os.path.exists(gate + '.exit'): sys.exit(3)\n"
                        "Q.write_mapped(Path(out), Q.mapped_key(head, json.loads(moves)), "
                        "None if os.path.exists(gate + '.refuse') else 'patch-' + head[:8], 'boom')\n")
    return gates


def _olds(world: World, n: int) -> tuple[list[str], str]:
    """``n`` old heads based before a move `main` lands, and a new head after it."""
    _push_main(world, base := world.change({"d.txt": "d\n"}))
    olds = []
    for k in range(1, n + 1):
        olds.append(old := world.change({f"f{k}.txt": f"{k}\n"}, base=base))
        _git(world.dev, "push", "-q", "origin", f"{old}:refs/heads/old-{k}")
    _land(world, _move(world))
    return olds, world.change({"g.txt": "g\n"})


def test_with_mapped_jobs_a_tick_detaches_a_mapping_that_outlives_it_and_a_later_tick_carries_the_mark(world, monkeypatch):
    """With `mapped_jobs` (the loop's), `mapped_change` only asks: the new head takes nothing until `work_out_mapped`, at the
    end of a tick, has detached a job mapping the old head (`MAP_JOB`, `landing.detach`), and the tick ends.  The job, a
    process of its own, maps the old head for real, its scratch under the loop's `MERGED_ROOT`, and writes the mapping into
    `mapped-changes.d/` by `mapped_key`; a later tick, a queue started afresh as each of the loop's `--once` ticks is, reads
    it there without running a script, so the new head takes both labels, for the head merged with a later `main` too.  A
    mapping that dies with the tick's process, as one on a thread of its own does, reaches no later tick."""
    monkeypatch.setattr(tempfile, "tempdir", str(world.tmp / "tmp"))
    world.gh.put(1, old := world.change({"old_proto/a.py": A.replace("return 2", "return 3")}), ready=False)
    _mark(world, 1, old)
    _land(world, move := _move(world))
    world.gh.put(1, new := _restack(world, old), ready=False)
    q = _queue(world)
    q.mapped_jobs = 1
    assert _kept(StoreLabels(world.store, change=q.own_change, moved=q.mapped_change), 1, new) == (False, [])
    assert q.work_out_mapped() == 0
    assert [m["at"] for m in _marks(q).values()] == [key := Q.mapped_key(old, [(move, [])])]
    assert world.logs[-1] == (f"research queue: mapping {old[:8]} across the 1 move(s) `main` landed since, in a detached job "
                              "(1 of 1 running)")
    (job,) = _marks(q).values()
    del q
    wait_exit(job["pid"])
    assert not any(Path(world.tmp, "scratch").iterdir())

    monkeypatch.setattr(Q, "mapped_tree", lambda *a, **k: pytest.fail("a script ran"))
    later = _queue(world)
    later.mapped_jobs = 1
    assert _kept(StoreLabels(world.store, change=later.own_change, moved=later.mapped_change), 1, new) == \
        (True, [("land", "reviewer")])
    assert json.loads((_drop(later) / f"{Q.mapped_name(key)}.json").read_text()) == {key: later.own_change(new)}
    assert later.work_out_mapped() == 1 and not _marks(later) and not Path(job["dir"]).exists()
    assert later.work_out_mapped() == 0

    _push_main(world, world.change({"e.txt": "main moved on\n"}))
    _git(world.dev, "checkout", "-q", "--detach", new)
    _git(world.dev, "merge", "-q", "--no-edit", "origin/main")
    _git(world.dev, "push", "-q", "origin", f"{(merged := _git(world.dev, 'rev-parse', 'HEAD'))}:refs/heads/later")
    world.gh.put(1, merged, ready=False)
    again = _queue(world)
    again.mapped_jobs = 1
    assert again.mapped_change(old, new) == again.mapped_change(old, merged) == again.own_change(new) == \
        again.own_change(merged)
    assert not _marks(again)


def test_at_most_mapped_jobs_map_at_once_and_while_a_heads_job_runs_no_tick_starts_another(world, jobs):
    """Of the heads `mapped_change` defers, `work_out_mapped` starts jobs while fewer than `mapped_jobs` run, counting the
    markers earlier ticks left: a later tick, started afresh, starts no second job for a head whose job runs, nor a third
    while two run; it starts the third once a job has ended, and reads what each wrote.  With no marker a later tick
    reads, each tick would start every head again."""
    olds, new = _olds(world, 3)
    gates = {old: Gate(jobs / old) for old in olds}

    def started() -> list[str]:
        return [s.split()[3] for s in world.logs if s.startswith("research queue: mapping ")]

    try:
        q, got, done = _tick(world, olds, new)
        assert (got, done, started()) == ([None] * 3, 0, [o[:8] for o in olds[:2]])
        first = _marks(q)
        assert sorted(first) == sorted(o[:12] for o in olds[:2])
        q, got, done = _tick(world, olds, new)
        assert (got, done, _marks(q), started()) == ([None] * 3, 0, first, [o[:8] for o in olds[:2]])

        gates[olds[0]].release()
        wait_exit(first[olds[0][:12]]["pid"])
        q, got, done = _tick(world, olds, new)
        assert (got, done, started()) == ([f"patch-{olds[0][:8]}", None, None], 1, [o[:8] for o in olds])
        assert sorted(_marks(q)) == sorted(o[:12] for o in olds[1:])
        for old in olds[1:]:
            gates[old].release()
            wait_exit(_marks(q)[old[:12]]["pid"])
        q, got, done = _tick(world, olds, new)
        assert (got, done, _marks(q), len(started())) == ([f"patch-{old[:8]}" for old in olds], 2, {}, 3)
    finally:
        for g in gates.values():
            g.close()


def test_a_stale_marker_whose_job_is_gone_with_no_result_starts_its_head_again(world, jobs):
    """A marker whose process is gone with no exit written (a restart of the pod) and no result holds nothing: the tick says so
    and starts the head again, and that job's mapping is read.  So does one with no result past the time a job may take
    (`_mapped_lost`), whose pid may be another process's after a restart."""
    olds, new = _olds(world, 1)
    (old,) = olds
    q, t = _queue(world), 1000.0
    key = Q.mapped_key(old, moves := q._moves_crossed(_git(q.workdir, "merge-base", q.base, new), old))
    (_drop(q) / "gone").mkdir(parents=True)
    mark = _drop(q) / f"{Q.mapped_name(key)}{Q.MAPPING_MARK}"
    mark.write_text(json.dumps({"dir": str(_drop(q) / "gone"), "pid": 2**31 - 1, "out": str(_drop(q)), "at": key,
                                "until": 10.0 ** 12}))
    q, got, done = _tick(world, olds, new, now=t)
    assert (got, done) == ([None], 0)
    assert world.logs[-2:] == [
        f"research queue: the mapping job of {mark.stem} ended with no result (its process is gone with no exit written (a "
        "restart of the pod)): a later tick that wants it starts it again",
        f"research queue: mapping {old[:8]} across the 1 move(s) `main` landed since, in a detached job (1 of 2 running)"]
    (job,) = _marks(q).values()
    assert job["pid"] != 2**31 - 1 and not (_drop(q) / "gone").exists()
    assert job["until"] == t + Q._mapped_lost(moves) == t + Q.MAPPED_TIMEOUT_S + Q.MAPPED_SLACK_S
    wait_exit(job["pid"])
    assert _tick(world, olds, new, now=t)[1:] == ([f"patch-{old[:8]}"], 1)

    (_drop(q) / f"{Q.mapped_name(key)}.json").unlink()
    mark.write_text(json.dumps({"dir": str(_drop(q) / "gone"), "pid": os.getpid(), "out": str(_drop(q)), "at": key,
                                "until": t}))
    q, got, done = _tick(world, olds, new, now=t)
    assert (got, done, list(_marks(q))) == ([None], 0, [old[:12]])
    assert _marks(q)[old[:12]]["pid"] == os.getpid()                        # within its time: it runs
    q, got, done = _tick(world, olds, new, now=t + 1)
    assert world.logs[-2].startswith(f"research queue: the mapping job of {mark.stem} ended with no result (none written by ")
    (job,) = _marks(q).values()
    assert job["pid"] != os.getpid()
    wait_exit(job["pid"])
    assert _tick(world, olds, new, now=t + 1)[1:] == ([f"patch-{old[:8]}"], 1)


def test_a_refused_mapping_is_kept_so_no_tick_maps_that_head_again_but_a_new_head_is_mapped(world, jobs):
    """A job's refusal is written into `mapped-changes.d/` beside its mappings, `{key: {"refused": why}}`, and logged once, when
    a tick reads its end; no later tick starts that head across those moves again, and none restarts a job that exited with no
    result written, whose exit is kept as its refusal.  A new head of the PR is another `mapped_key`, mapped afresh, and a
    mapping `research queue map-moves` made elsewhere and copied in by hand outweighs a refusal.  A refusal kept only in the
    tick's process would map its head again every tick."""
    olds, new = _olds(world, 3)
    refused, exited, fresh = olds
    (jobs / f"{refused}.refuse").touch()
    (jobs / f"{exited}.exit").touch()
    q, got, done = _tick(world, olds[:2], new)
    for job in _marks(q).values():
        wait_exit(job["pid"])
    q, got, done = _tick(world, olds[:2], new)
    assert (got, done, _marks(q)) == ([None, None], 2, {})
    moves = q._moves_crossed(_git(q.workdir, "merge-base", q.base, new), refused)
    said = sorted(s for s in world.logs if "isn't mapped" in s)
    assert said == sorted([f"research queue: {refused[:8]}'s own change isn't mapped across the moves `main` landed since: boom",
                    f"research queue: {exited[:8]}'s own change isn't mapped across the moves `main` landed since: its mapping "
                    "job exited 3 with none written"])
    assert json.loads((_drop(q) / f"{Q.mapped_name(Q.mapped_key(refused, moves))}.json").read_text()) == \
        {Q.mapped_key(refused, moves): {"refused": "boom"}}

    before = len(world.logs)
    q, got, done = _tick(world, olds[:2], new)
    assert (got, done, _marks(q), world.logs[before:]) == ([None, None], 0, {}, [])

    q, got, done = _tick(world, olds, new)
    assert (got, done, list(_marks(q))) == ([None, None, None], 0, [fresh[:12]])
    wait_exit(_marks(q)[fresh[:12]]["pid"])
    assert _tick(world, olds, new)[1:] == ([None, None, f"patch-{fresh[:8]}"], 1)

    (_drop(q) / "by-hand.json").write_text(json.dumps({Q.mapped_key(refused, moves): "patch-by-hand"}))
    assert _tick(world, olds, new)[1] == ["patch-by-hand", None, f"patch-{fresh[:8]}"]


def test_research_queue_map_moves_maps_old_heads_elsewhere_and_the_loop_reads_them_from_mapped_changes_d(world, monkeypatch,
                                                                                                         capsys):
    """`research queue map-moves`, in a clone of its own, maps each old head across the moves `main` has as the loop would,
    and writes what it worked out by `mapped_key`, saying so for each head, and why for a head crossing no move, missing, or
    marked with no land verdict of its own (the store's labels), whose tree no script runs on; each is given as N@HEAD.
    Copied into `mapped-changes.d/` beside the loop's `mapped-changes.json`, the file serves the loop's queue without a script:
    read when the queue first looks, and again by `work_out_mapped` once a file there has changed; one that doesn't parse is
    said and skipped."""
    world.gh.put(1, old := world.change({"old_proto/a.py": A.replace("return 2", "return 3")}), ready=False)
    _mark(world, 1, old)
    world.gh.put(3, bare := world.change({"old_proto/a.py": A.replace("return 1", "return 4")}), ready=False)
    world.store.add_label(label_target(3, bare), vocab.READY, True, "lane-3", ts="2026-11-10T10:00:00Z")
    _git(world.tmp, "clone", "-q", str(world.origin), str(elsewhere := world.tmp / "elsewhere"))
    landed = _land(world, move := _move(world))
    world.gh.put(1, new := _restack(world, old), ready=False)
    world.gh.put(2, late := world.change({"x.txt": "x\n"}, base=f"{landed}^1"), ready=False)
    _mark(world, 2, late)
    out, missing = world.tmp / "mapped.json", "f" * 40
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(world.tmp / "store.toml"))
    store = ["--store", str(world.tmp / "store")]
    pools: list[int] = []

    class Pool(concurrent.futures.ThreadPoolExecutor):
        def __init__(self, workers):
            pools.append(workers)
            super().__init__(workers)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", Pool)
    monkeypatch.chdir(elsewhere)
    assert Q.main(["map-moves", f"1@{old}", f"1@{new}", f"2@{late}", f"3@{bare}", f"4@{missing}", "--jobs", "2",
                   "--out", str(out), *store]) == 0
    q = _queue(world)
    assert json.loads(out.read_text()) == {Q.mapped_key(old, [(move, [])]): q.own_change(new)}
    assert capsys.readouterr().err.splitlines() == [
        f"research queue map-moves: {old[:12]}: {q.own_change(new)}",
        f"research queue map-moves: {new[:12]}: it crosses no move",
        f"research queue map-moves: {late[:12]}: it has {_git(q.workdir, 'rev-parse', f'{landed}^1')[:12]}, which came after "
        f"{_git(q.workdir, 'rev-parse', f'{move}^')[:12]}, the base of the move {move[:12]}: restack.py refuses it (based-after)",
        f"research queue map-moves: {bare[:12]}: {Q.UNLANDED}",
        f"research queue map-moves: {missing[:12]}: no such commit here or on origin"]
    assert Q.main(["map-moves", f"1@{new}", "--jobs", "0", *store]) == 0 and capsys.readouterr().out == "{}\n" and pools == [2, 1]
    with pytest.raises(SystemExit, match=f"{old}: give each old head as N@HEAD"):
        Q.main(["map-moves", old, *store])

    monkeypatch.setattr(Q, "mapped_tree", lambda *a, **k: pytest.fail("a script ran"))
    q.mapped_jobs = 1
    assert q.mapped_change(old, new) is None
    (drop := Path(_git(q.workdir, "rev-parse", "--absolute-git-dir")) / "mapped-changes.d").mkdir()
    (drop / "bad.json").write_text("{")
    (drop / "list.json").write_text("[1]")
    (drop / "dir.json").mkdir()
    (drop / "layout.json").write_text(out.read_text())
    assert q.work_out_mapped() == 0
    assert [s.split(": ")[1:3] for s in world.logs if "isn't read" in s] == [
        [f"{drop / name} isn't read", kind] for name, kind in (("bad.json", "JSONDecodeError"), ("dir.json", "IsADirectoryError"),
                                                                ("list.json", "TypeError"))]
    assert q.work_out_mapped() == 0 and sum("isn't read" in s for s in world.logs) == 3     # read again only once changed
    assert _kept(StoreLabels(world.store, change=q.own_change, moved=q.mapped_change), 1, new) == (True, [("land", "reviewer")])
    again = _queue(world)
    again.mapped_jobs = 1
    assert again.mapped_change(old, new) == q.own_change(new)
    assert not (Path(_git(q.workdir, "rev-parse", "--absolute-git-dir")) / "mapped-changes.json").exists()


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
    world.gh.put(3, again := world.change({"c.txt": "c merged again\n"}), ready=False)
    q.fetch(world.gh.open_prs())
    assert q.mapped_change(old, again) is None                                # the same old head on the same `main`: not again
    assert [s for s in world.logs if "isn't mapped" in s] == [
        f"research queue: {old[:8]}'s own change isn't mapped across the moves `main` landed since: it doesn't merge with "
        f"{_git(q.workdir, 'rev-parse', f'{move}^')[:12]}, the base of the move {move[:12]}: c.txt"]

    monkeypatch.setattr(Q, "_MAPPED", "import sys; sys.exit('the script fails')")
    world.gh.put(2, two := world.change({"a.txt": "a from #2\n"}, base=f"{move}^"), ready=False)
    _git(world.dev, "push", "-q", "origin", f"{two}:refs/heads/two")
    q.fetch(world.gh.open_prs())
    assert q.mapped_change(two, new) is None
    assert world.logs[-1].endswith(f"the move {move[:12]}'s script fails on it: exit 1: the script fails")

    onto = _git(q.workdir, "merge-base", q.base, new)
    assert Q.mapped_change(q.workdir, two, onto, [(move, [])], Path("unused")) == (None, f"the move {move[:12]} has 0 parents")
    monkeypatch.setattr(Q, "mapped_tree", lambda *a, **k: ("0" * 40, ""))      # a tree that isn't there: git refuses it
    got, why = q._mapped_one(two, onto, moves := Q.moves_between(q.workdir, onto, two))
    assert got is None and why.startswith("QueueError: ")
    q._mapped_done(Q.mapped_key(two, moves), None, "")
    assert world.logs[-1] == (f"research queue: {two[:8]}'s own change isn't mapped across the moves `main` landed since: it "
                              "has no own change once mapped")


def test_a_moves_script_runs_in_a_directory_of_its_own_under_merged_root_and_one_killed_there_leaves_no_worktree(world,
                                                                                                                 monkeypatch):
    """The script's temporary directory is one of this process's under `MERGED_ROOT` (the loop's --merged-root), as
    `merged_worktree` names its own, so a killed loop's is swept; a run killed at its time (`run_group` gives None) leaves a
    worktree half made there, which goes, with the clone's record of it, and the head maps to nothing, saying why."""
    q, scratch = _queue(world), world.tmp / "scratch"
    head = _git(q.workdir, "rev-parse", q.base)
    monkeypatch.setattr(Q, "_MAPPED", "import os, sys; sys.exit(os.environ['TMPDIR'])")
    tree, why = Q.mapped_tree(q.workdir, head, head, Path("unused"))
    assert tree is None and why.startswith(f"exit 1: {scratch}/{Q.MERGED_PREFIX}{os.getpid()}-")
    monkeypatch.setattr(Q, "_MAPPED", "print('mapped ' + 'a' * 40); raise SystemExit(3)")
    assert Q.mapped_tree(q.workdir, head, head, Path("unused")) == (None, f"exit 3: mapped {'a' * 40}")

    def killed(argv, cwd, left, env=None):
        _git(cwd, "worktree", "add", "-q", "--detach", str(Path(env["TMPDIR"], "restack-x", "tree")), head)
        return None, ""

    monkeypatch.setattr(Q, "run_group", killed)
    assert Q.mapped_tree(q.workdir, head, head, Path("unused")) == (None, f"ran over {Q.MAPPED_TIMEOUT_S:.0f} s")
    assert not any(scratch.iterdir())
    assert _git(q.workdir, "worktree", "list", "--porcelain").count("worktree ") == 1

    copied: list[tuple[str, bool]] = []
    monkeypatch.setattr(Q, "mapped_change", lambda repo, h, onto, moves, restack, env=None: copied.append(
        (str(restack), (restack / "restack.py").is_file())) or ("p", ""))
    assert Q.map_one(q.workdir, head, head, []) == ("p", "") and not any(scratch.iterdir())
    assert copied[0][0].startswith(f"{scratch}/{Q.MERGED_PREFIX}{os.getpid()}-") and copied[0][1]


def test_a_moves_script_runs_without_the_queues_credentials(world, monkeypatch):
    """The script runs the head's own code (`rename.py` runs the tree's `pins.py`, `layout.py` its `uv lock`), which needs none
    of the queue's GitHub and Slack credentials, so it sees none, and `gh` finds no stored login."""
    q = _queue(world)
    head = _git(q.workdir, "rev-parse", q.base)
    for k in Q.SCRIPT_UNSET:
        monkeypatch.setenv(k, "dummy-" + k.lower())
    monkeypatch.setattr(Q, "_MAPPED", "import os, sys; gh = os.environ.get('GH_CONFIG_DIR', ''); sys.exit("
                        "','.join(sorted(k for k, v in os.environ.items() if v.startswith('dummy-'))) + '|' + "
                        "str(gh.startswith(os.environ['TMPDIR']) and not os.path.exists(gh)))")
    assert Q.mapped_tree(q.workdir, head, head, Path("unused")) == (None, "exit 1: |True")


def test_only_an_old_head_a_reviewer_passed_runs_a_moves_script_so_a_head_marked_alone_plants_no_hook(world):
    """Mapping an old head runs a move's script on its tree, which runs the tree's own code (`rename.py` imports its
    `pins.py`) as the queue's user in a worktree of the queue's clone, so `StoreLabels` maps only an old head whose own land
    verdict counts (`lands`).  #2's old head, marked on ci's word and passed by nobody, has a `pins.py` that plants a hook in
    the clone, which the queue's next fetch would run with its credentials: no script runs on it, so nothing is planted, and
    its mark carries to no head across the move.  Its owner's own land verdict changes nothing (``owners``); a reviewer's
    lets the script run, and the mapping, not the new head's own change, carries nothing."""
    world.gh.put(2, old := world.change({"tools/verity/check/pins.py": PLANT, "b.py": B + "W = 1\n"}), ready=False)
    world.store.add_label(label_target(2, old), vocab.READY, True, "ci", ts="2026-11-10T10:00:00Z")
    _land(world, _move(world))
    world.gh.put(2, new := world.change({"d.txt": "d\n"}), ready=False)
    q = _queue(world)
    hook = Path(_git(q.workdir, "rev-parse", "--absolute-git-dir")) / "hooks" / "reference-transaction"

    def kept() -> tuple[bool, list]:
        return _kept(StoreLabels(world.store, change=q.own_change, moved=q.mapped_change, owners=lambda n: {f"lane-{n}"}), 2, new)

    assert kept() == (False, []) and not q._mapped
    q.fetch(world.gh.open_prs())
    assert not hook.exists()
    world.store.add_label(label_target(2, old), vocab.REVIEWED, "land", "lane-2", ts="2026-11-10T11:00:00Z")
    assert kept() == (False, []) and not q._mapped and not hook.exists()
    world.store.add_label(label_target(2, old), vocab.REVIEWED, "land", "reviewer", ts="2026-11-10T12:00:00Z")
    assert kept() == (False, []) and list(q._mapped) == [(old, new)] and hook.exists()


def test_a_head_marked_alone_runs_no_script_so_its_code_forges_no_mapping_that_carries_another_prs_verdict(world):
    """Nor can such a head's code write `mapped-changes.d/`, which the loop reads as mappings worked out elsewhere: #2's
    `pins.py` would write there that #1's passed old head maps to the own change of #1's new head, which adds a file nobody
    reviewed.  No script runs on #2's heads, so nothing is written there, and #1's new head takes no label: #1's old head,
    mapped for real, isn't its own change."""
    world.gh.put(1, h0 := world.change({"b.py": B + "Y = 1\n"}), ready=False)
    _mark(world, 1, h0)
    before = world.change({"e.txt": "e\n"})
    _land(world, _move(world))
    world.gh.put(1, h2 := world.change({"evil.py": "import os\n"}), ready=False)
    q = _queue(world)
    onto = _git(q.workdir, "merge-base", q.base, h2)
    key, target = Q.mapped_key(h0, Q.moves_between(q.workdir, onto, h0)), q.own_change(h2)
    forge = (GIT_DIR + "d = c / 'mapped-changes.d'\nd.mkdir(exist_ok=True)\n"
             f"(d / 'x.json').write_text(json.dumps({{{key!r}: {target!r}}}))\nPINS = ()\n")
    world.gh.put(2, k1 := world.change({"tools/verity/check/pins.py": forge}, base=f"{before}^"), ready=False)
    world.store.add_label(label_target(2, k1), vocab.READY, True, "lane-2", ts="2026-11-10T10:00:00Z")
    q.fetch(world.gh.open_prs())
    world.gh.put(2, k2 := world.change({"f.txt": "f\n"}), ready=False)
    q.fetch(world.gh.open_prs())
    labels = StoreLabels(world.store, change=q.own_change, moved=q.mapped_change, owners=lambda n: {f"lane-{n}"})
    assert _kept(labels, 2, k2) == (False, [])
    q.work_out_mapped()
    assert not (Path(_git(q.workdir, "rev-parse", "--absolute-git-dir")) / "mapped-changes.d").exists()
    assert _kept(labels, 1, h2) == (False, []) and q.mapped_change(h0, h2) not in (None, target)


def test_a_head_based_after_a_move_that_isnt_the_last_is_mapped_across_the_rest_as_a_crossing_naming_them_carries_it(world):
    """#1 is based on a commit `main` took after the first move's base, before that move landed: `restack.py` refuses to carry
    it across that move (`based-after`), so its owner names the move after it (`--move-commit`).  The mapping skips the first
    move as that crossing does and maps #1 across the second, so the crossed head keeps the old head's mark and verdict."""
    first = _move(world)
    _push_main(world, other := world.change({"c.txt": "c landed meanwhile\n"}))
    world.gh.put(1, old := world.change({"new_proto/extra.py": "X = 1\n"}), ready=False)
    _mark(world, 1, old)
    tree = _git(world.dev, "merge-tree", "--write-tree", other, first)
    _push_main(world, _git(world.dev, "commit-tree", tree, "-p", other, "-p", first, "-m", "Queue: #9 the move"))
    onto = _land(world, second := _move(world, LIB))
    r = subprocess.run([sys.executable, str(ROOT / "tools/move/restack.py"), old, "--move-commit", second, "--onto",
                        "origin/main", "--json"], cwd=world.dev, env=ENV, capture_output=True, text=True)
    (got,) = json.loads(r.stdout)
    assert got["status"] == "clean", got
    _git(world.dev, "push", "-q", "origin", f"{(new := got['result'])}:refs/heads/crossed")
    world.gh.put(1, new, ready=False)
    q = _queue(world)
    assert [m for m, _ in Q.moves_between(q.workdir, onto, old)] == [first, second]
    assert _git(q.workdir, "show", f"{new}:lib/new_proto/extra.py") == "X = 1" and q.own_change(new) != q.own_change(old)
    assert _kept(StoreLabels(world.store, change=q.own_change, moved=q.mapped_change), 1, new) == (True, [("land", "reviewer")])


def test_a_head_based_after_the_moves_base_maps_to_none_as_restack_refuses_it(world):
    """A PR based on a commit `main` took after the move's base, before the move landed, is one `restack.py` refuses
    (`based-after`: its merge of the move's base would bring that commit in a second time), and its owner merges `main`
    instead: `mapped_change` maps it to nothing either, and says why."""
    move = _move(world)
    _push_main(world, other := world.change({"c.txt": "c landed meanwhile\n"}))
    world.gh.put(1, old := world.change({"b.py": B + "Y = old_proto.a.g()\n"}), ready=False)
    _git(world.dev, "push", "-q", "origin", f"{old}:refs/heads/old")
    tree = _git(world.dev, "merge-tree", "--write-tree", other, move)
    _push_main(world, _git(world.dev, "commit-tree", tree, "-p", other, "-p", move, "-m", "Queue: #9 the move"))
    r = subprocess.run([sys.executable, str(ROOT / "tools/move/restack.py"), old, "--main", "origin/main", "--json"],
                       cwd=world.dev, env=ENV, capture_output=True, text=True)
    assert json.loads(r.stdout)[0]["status"] == "based-after"
    world.gh.put(1, new := world.change({"c.txt": "c merged\n"}), ready=False)
    q = _queue(world)
    assert q.mapped_change(old, new) is None
    assert world.logs[-1].endswith(f"it has {other[:12]}, which came after {_git(world.dev, 'rev-parse', f'{move}^')[:12]}, "
                                   f"the base of the move {move[:12]}: restack.py refuses it (based-after)")
