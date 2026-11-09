"""Moves through the verified queue (`research.vqueue_loop`'s "Moves"): a PR whose body says `Move: tools/move/MAP.toml` is
built as its move regenerated on its candidate's base by its map's script (`rename.py` for a word map, `layout.py` for a path
map) and certified by `move_check.py`, all as `main` has them, and lands as that commit on a land verdict; one whose
regeneration fails is refused with why, its owner told once a head.  A PR whose head lacks a move below it is merged across
it, restacked by `restack.py` in the background, which runs only when a test runs it (`jobs`).  The scripts are this
checkout's, on `test_vqueue_loop`'s throwaway origin, with maps that move `old_proto/` to `new_proto/`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.executor import GRAPH_V2
from research.queue import RESTACKING
from test_drive_moves import jobs  # noqa: F401 -- the restack jobs, run when a test runs them
from test_drive_review import _move_main
from test_queue import _git
from test_vqueue_loop import Env, _acts, policy

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ("tools/move/rename.py", "tools/move/layout.py", "tools/move/restack.py", "tools/verity/check/move_check.py")
WORDS, PATHS = "tools/move/word_map.toml", "tools/move/path_map.toml"
BAD, ODD = "tools/move/bad_map.toml", "tools/move/odd_map.toml"
MOVES = "tools/research/src/research/store/moves.json"
EXCLUDE = f'[rewrite]\nexclude = ["tools/move/**", "{MOVES}"]\n\n'
WORD_MAP = '[words]\n"old_proto" = "new_proto"\n'
PATH_MAP = ('[roots]\nold = { old_proto = "old_proto" }\nnew = { new_proto = "new_proto" }\n\n[paths]\n"old_proto/" = "new_proto/"\n'
            '\n[modules]\n"old_proto" = "new_proto"\n')
B = "from old_proto.a import f\n\nX = f()\n"
#: a change to the line the move rewrites: it conflicts with the move as `git merge-tree` merges it, and not as restacked
TWO = {"b.py": "from old_proto.a import f as g\n\nX = g()\n"}


@pytest.fixture
def env(tmp_path, monkeypatch) -> Env:
    e = Env(tmp_path, monkeypatch)
    _move_main(e.o, {**{p: (ROOT / p).read_text() for p in SCRIPTS}, "tools/verity/check/pins.py": "PINS = ()\n",
                     MOVES: '{"moves": []}\n', WORDS: EXCLUDE + WORD_MAP, PATHS: EXCLUDE + PATH_MAP,
                     BAD: EXCLUDE + '[words]\n"banana" = "cherry"\n', "old_proto/__init__.py": "",
                     "old_proto/a.py": "def f():\n    return 1\n", "b.py": B, "fruit.py": "banana = 1\n"})
    e.graph2.list_graph(f"ready:{e.main()}", {"schema": GRAPH_V2, "commit": e.main(), "tasks": []})
    e.o.gh.closed = []
    e.o.gh.close = lambda n, comment: e.o.gh.closed.append((n, comment))
    return e


def _move_pr(env: Env, n: int, map: str = WORDS) -> str:
    """A move PR whose head is the request: the move as its owner made it, on `main` as it was."""
    head = env.o.change({"old_proto/__init__.py": None, "old_proto/a.py": None, "new_proto/__init__.py": "",
                         "new_proto/a.py": "def f():\n    return 1\n", "b.py": B.replace("old_proto", "new_proto")})
    env.o.gh.put(n, head, body=f"Lane `lane-{n}`.\nMove: {map}\n")
    return head


def _pass(env: Env, sha: str) -> None:
    keys = env.listed(sha, {"pytest:x": ["x.txt"]})
    env.result(keys["pytest:x"], "pytest:x", "pass")


def _show(env: Env, ref: str, path: str) -> str:
    return _git(env.o.origin, "show", f"{ref}:{path}")


def _files(env: Env, ref: str) -> list[str]:
    return _git(env.o.origin, "ls-tree", "-r", "--name-only", ref).split()


def _parents(env: Env, ref: str) -> list[str]:
    return _git(env.o.origin, "rev-list", "--parents", "-n", "1", ref).split()[1:]


def test_a_move_pr_is_built_as_its_move_regenerated_on_its_candidate_s_base_and_lands_as_that_one_commit(env):
    """#1 is below the move in its candidate, so the move carries a file its request lacks.  Once landed, the move PR is closed
    with what landed, and its head, which `main` doesn't contain, is landed by its trailers: never admitted again."""
    env.put(1, {"old_proto/c.py": "C = 1\n"})
    head = _move_pr(env, 2)
    loop = env.loop(policy)
    (built,) = loop.tick()
    assert (built["action"], built["members"], built["heads"][1]) == ("build", [1, 2], head), built
    move = built["sha"]
    (below,) = _parents(env, move)                                  # one commit on #1's merge: the move regenerated there
    assert _parents(env, below)[0] == env.main()
    body = _git(env.o.origin, "log", "-1", "--format=%B", move)
    assert body.startswith("Queue: #2 ") and f"Head: {head}" in body and body.endswith(f"Move: {WORDS}"), body
    assert f"`python3 tools/move/rename.py --map {WORDS}` at the root of {below}" in body, body
    files = _files(env, move)
    assert "new_proto/c.py" in files and not [f for f in files if f.startswith("old_proto/")], files
    assert _show(env, move, "b.py") == B.replace("old_proto", "new_proto").rstrip("\n")
    assert any(f"#2's move ({WORDS}) regenerated on {below[:12]} as {move[:12]}" in x for x in env.out), env.out
    _pass(env, move)
    assert _acts(loop.tick()) == ["advance", "on_main"] and env.main() == move
    assert env.o.gh.closed == [(2, f"Landed in `main` at {move[:12]}: its move ({WORDS}), regenerated by the queue as {move[:12]}.")]
    assert _acts(loop.tick()) == ["drop"] and loop.tick() == []


@pytest.mark.parametrize("shadow", [False, True])
def test_a_move_that_doesnt_regenerate_on_main_is_refused_never_built_and_its_owner_told_why_once_a_head(env, shadow):
    bad = env.o.change({"fruit.py": "banana = 2\n"})
    env.o.gh.put(1, bad, body=f"Lane `lane-1`.\nMove: {BAD}\n")
    env.put(2, {"c.txt": "2\n"})
    loop = env.loop(policy, shadow=shadow)
    refused, built = loop.tick()
    assert (refused["action"], refused["number"], refused["heads"], refused["shadow"]) == ("refuse", 1, [bad], shadow), refused
    assert "isn't certified" in refused["why"] and "move_check.py" in refused["why"], refused
    assert f"#1's move ({BAD}) at {bad[:12]} doesn't regenerate on `main`" in refused["text"], refused
    assert built["action"] == "build" and built["members"] == [2], built
    assert [p.endswith(refused["text"]) for p in env.posts] == ([] if shadow else [True]), env.posts
    assert loop.tick() == [] and len(env.posts) == (0 if shadow else 1)
    again = env.o.change({"fruit.py": "banana = 3\n"})
    env.o.gh.put(1, again, body=f"Lane `lane-1`.\nMove: {BAD}\n")
    (told,) = loop.tick()
    assert (told["action"], told["heads"]) == ("refuse", [again]), told


def test_a_path_map_s_move_is_layout_py_s_and_ends_in_its_layout_move_line(env):
    """`rename.py` reads a word map's `[words]`, so a path map's move is `layout.py`'s (`--no-lock` on a tree with no
    `uv.lock`), and its commit ends in the line `restack.py` reruns it by, which the queue reads as a move it made."""
    head = _move_pr(env, 1, PATHS)
    loop = env.loop(policy)
    (built,) = loop.tick()
    move = built["sha"]
    assert built["members"] == [1] and _parents(env, move) == [env.main()], built
    body = _git(env.o.origin, "log", "-1", "--format=%B", move)
    assert f"`python3 tools/move/layout.py --map {PATHS} --no-lock` at the root of {env.main()}" in body, body
    assert body.endswith(f"Head: {head}\nOwner: lane-1\nLayout-move: {PATHS} --no-lock"), body
    assert _show(env, move, "b.py") == B.replace("old_proto", "new_proto").rstrip("\n")
    assert json.loads(_show(env, move, MOVES))["moves"] == [{"paths": {"old_proto/": "new_proto/"}}]
    _pass(env, move)
    assert _acts(loop.tick()) == ["advance", "on_main"]
    assert env.o.gh.closed == [(1, f"Landed in `main` at {move[:12]}: its move ({PATHS}), regenerated by the queue as {move[:12]}.")]


@pytest.mark.parametrize("tables, kind", [(WORD_MAP + "\n" + PATH_MAP, "both [words] and [paths]"),
                                          ("", "neither [words] nor [paths]")], ids=["both", "neither"])
def test_a_map_with_both_tables_or_neither_is_refused_naming_its_tables(env, tables, kind):
    _move_main(env.o, {ODD: EXCLUDE + tables})
    _move_pr(env, 1, ODD)
    (refused,) = env.loop(policy).tick()
    assert refused["action"] == "refuse" and f"its map {ODD} has {kind}" in refused["why"], refused


def test_a_move_pr_needs_a_land_verdict_where_another_pr_lands_on_its_mark_alone(env):
    _move_main(env.o, {"tools/verity/check/drive.toml": 'lowrisk_excludes = ["*.lean"]\n'})
    env.put(1, {"c.txt": "1\n"})
    _move_pr(env, 2)
    env.labels.reviews = lambda n, head: []                         # no verdict, and no hold: the low-risk rule's
    (built,) = env.loop(policy).tick()
    assert built["members"] == [1], built


def test_a_pr_whose_only_conflict_is_a_move_that_landed_under_it_is_judged_through_its_restack_and_built_across_it(env, jobs):
    """`git merge-tree` alone judges it conflicting (it edits a line the move rewrote), so it would never be built.  Judged
    through the restack, it waits untold while `restack.py` reruns the move (a path map's, by its move line) on its head, and
    is then built merged across the move.  The restack commits on its head, so the PR sweep closes it once it lands."""
    _move_pr(env, 1, PATHS)
    old = env.main()
    loop = env.loop(policy)
    (built,) = loop.tick()
    _pass(env, moved := built["sha"])
    assert _acts(loop.tick()) == ["advance", "on_main"]
    two = env.o.change(TWO, base=old)
    env.o.gh.put(2, two)
    assert _acts(loop.tick()) == ["drop"]
    (job,) = jobs
    assert job.head == two and loop.tick() == []                    # waits off the pool while its restack runs
    job.run()
    (again,) = loop.tick()
    assert (again["action"], again["members"], again["base"]) == ("build", [2], moved), again
    merge = again["sha"]
    assert _parents(env, merge)[0] == moved and _parents(env, merge)[1] != two
    assert _show(env, merge, "b.py") == TWO["b.py"].replace("old_proto", "new_proto").rstrip("\n")
    _pass(env, merge)
    assert _acts(loop.tick()) == ["advance", "on_main"] and two in _git(env.o.origin, "rev-list", env.main()).split()
    assert not [p for p in env.posts if "#2 at" in p], env.posts


@pytest.mark.parametrize("map", [WORDS, PATHS])
def test_a_member_above_a_move_in_its_candidate_waits_untold_on_its_restack_and_lands_across_it_restacked_once(env, jobs, map):
    """#3 adds a file that imports what the move moves, outside its paths: a plain merge is clean and leaves it importing the
    old name.  Its change names a word, path or module of the map, so it is restacked across the move, waiting off the pool,
    untold.  The restack is kept while its move is on a candidate, so #3 merges across the move once that lands, with no second
    restack."""
    env.put(1, {"c.txt": "1\n"})
    _move_pr(env, 2, map)
    three = env.put(3, {"e.py": "from old_proto.a import f\n"})
    loop = env.loop(policy)
    wait, built = loop.tick()
    assert (wait["action"], wait["number"], wait["heads"]) == ("wait", 3, [three]), wait
    assert wait["why"].startswith(RESTACKING) and built["members"] == [1, 2], (wait, built)
    (job,) = jobs
    assert job.head == three and loop.tick() == []
    assert {x.number: x.conflict for x in loop.world.ready(env.main())}[3], "off the pool while its restack runs"
    job.run()
    _pass(env, built["sha"])
    assert _acts(loop.tick()) == ["advance", "on_main"]
    assert _acts(loop.tick()) == ["drop"]
    (again,) = loop.tick()
    assert (again["action"], again["members"], again["base"]) == ("build", [3], built["sha"]), again
    assert _show(env, again["sha"], "e.py") == "from new_proto.a import f" and len(jobs) == 1
    assert not [p for p in env.posts if "#3" in p], env.posts


def test_the_moves_work_is_kept_in_the_loop_s_cache_across_a_restart_never_in_drive_s_state_and_a_dry_run_starts_none(
        env, jobs, tmp_path):
    cache = tmp_path / "cache"
    _move_pr(env, 1)
    old = env.main()
    loop = env.loop(policy, cache=cache)
    (built,) = loop.tick()
    _pass(env, built["sha"])
    loop.tick()
    env.o.gh.put(2, two := env.o.change(TWO, base=old))
    kept = (cache / "moves.json").read_text()
    assert env.loop(policy, cache=cache, dry_run=True).tick() == [] and jobs == []
    assert (cache / "moves.json").read_text() == kept
    loop.tick()
    (job,) = jobs
    assert list(json.loads((cache / "moves.json").read_text())["restacking"]) == [f"{two}:{built['sha']}"]
    assert not {"moves_made", "restacks", "restacking"} & set(loop.world.q.load_state())
    assert env.loop(policy, cache=cache).tick() == [] and len(jobs) == 1     # restarted: its restack still runs
