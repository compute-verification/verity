"""`research queue cross` carries the open PRs whose heads predate a move `main` landed across it with `main`'s own
`tools/move/restack.py`, as each owner would: a dry run crosses and reports, and pushes nothing; `--push --from DIR` pushes
each clean crossing a run left in DIR to its PR's branch with a lease on the old head, where its mark and land verdict carry
by the loop's own equality.  A crossing that conflicts is refused with why, a head that has the moves is skipped, an owner's
push since wins over the lease, and a stack crosses lowest first, each onto the crossing below it, or is reported.  The
throwaway origin and its move are `test_queue_moves_carry`'s."""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

import research.queue as Q
from research.clone import RECORD_DRIVER
from research.queue import StoreLabels, label_target
from research.store import vocab
from test_queue import World, _git
from test_queue_moves_carry import LIB, A, B, _kept, _land, _mark, _move, _push_main, _queue, world  # noqa: F401


def _open(world: World, n: int, head: str) -> None:
    """PR ``n`` open at ``head`` from its own branch `lane-N`, pushed there."""
    world.gh.put(n, head, ready=False)
    world.gh.prs[n] = dataclasses.replace(world.gh.prs[n], branch=f"lane-{n}")
    _git(world.dev, "push", "-q", "-f", "origin", f"{head}:refs/heads/lane-{n}")


def _branch(world: World, n: int) -> str:
    return _git(world.origin, "rev-parse", f"refs/heads/lane-{n}")


def _cross(world: World, monkeypatch, *args: str, where: str = "elsewhere") -> int:
    """`research queue cross ARGS` in a clone of the origin of its own (``where``), as on a node or ci's VM, with the store's
    labels and the world's GitHub."""
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(world.tmp / "store.toml"))
    monkeypatch.setattr(Q, "CROSS_KEPT", False)
    monkeypatch.setattr(Q, "GitHub", lambda slug: world.gh)
    monkeypatch.setattr(Q, "PublicGitHub", lambda slug: world.gh)
    if not (clone := world.tmp / where).exists():
        _git(world.tmp, "clone", "-q", str(world.origin), str(clone))
    monkeypatch.chdir(clone)
    return Q.main(["cross", "--remote", str(world.origin), "--repo", Q.REPO, "--store", str(world.tmp / "store"), *args])


def _report(out: Path) -> dict:
    return json.loads((out / Q.CROSS_REPORT).read_text())


def test_a_clean_crossing_is_pushed_only_with_push_and_its_mark_and_verdict_carry_by_its_mapped_change(world, monkeypatch,
                                                                                                      capsys):
    """#1, marked and passed, edits the file the move moves and adds a line with the move's word.  The dry run crosses it,
    reports the new head and that its verdict carries, writes the bundle and its mapped change, and pushes nothing.  `--push
    --from` in another clone, as on ci's VM, fetches the bundle and pushes the new head to #1's branch with a lease on the old
    one; the loop, reading the mapped change from `mapped-changes.d/` with no script of its own, carries the mark and verdict
    to it."""
    _open(world, 1, old := world.change({"old_proto/a.py": A.replace("return 2", "return 3"),
                                         "b.py": B + "Y = old_proto.a.g()\n"}))
    _mark(world, 1, old)
    _land(world, move := _move(world))
    out = world.tmp / "out"
    assert _cross(world, monkeypatch, "--out", str(out)) == 0
    report = _report(out)
    (c,) = report["crossed"]
    assert (c["pr"], c["old"], c["src"], c["branch"], report["refused"], report["moves"]) == (1, old, old, "lane-1", [],
                                                                                              [move])
    assert _branch(world, 1) == old and (out / Q.CROSS_BUNDLE).is_file() and not (out / Q.CROSS_PUSHED).exists()
    assert capsys.readouterr().out.strip() == Q.cross_table(report)
    assert f"crossed #1 {old[:12]} -> {c['new'][:12]}  carries its mark and verdict" in Q.cross_table(report)

    assert _cross(world, monkeypatch, "--push", "--from", str(out), where="pusher") == 0
    new = c["new"]
    assert _branch(world, 1) == new
    assert json.loads((out / Q.CROSS_PUSHED).read_text()) == {"1": f"pushed {new[:12]} to lane-1"}
    world.gh.put(1, new, ready=False)
    q = _queue(world)
    assert _git(q.workdir, "show", f"{new}:b.py").endswith("Y = new_proto.a.g()")
    assert json.loads((out / Q.CROSS_MAPPED).read_text()) == {Q.mapped_key(old, [(move, [])]): q.own_change(new)}
    monkeypatch.setattr(Q, "mapped_tree", lambda *a, **k: pytest.fail("a script ran"))
    (drop := Path(_git(q.workdir, "rev-parse", "--absolute-git-dir")) / "mapped-changes.d").mkdir()
    (drop / "cross.json").write_text((out / Q.CROSS_MAPPED).read_text())
    q.mapped_jobs = 1
    q.work_out_mapped()
    labels = StoreLabels(world.store, change=q.own_change, moved=q.mapped_change, owners=lambda n: {f"lane-{n}"})
    assert _kept(labels, 1, new) == (True, [("land", "reviewer")])


def test_a_head_lacking_an_earlier_move_crosses_the_later_one_named_and_its_mark_and_verdict_carry(world, monkeypatch):
    """#1 is based on a commit `main` took after the first move's base, before that move landed, so it lacks that move, which
    restack.py won't cross (`based-after`).  Naming only the second, the step lets restack.py cross that one, and the loop's
    mapping skips the first as that crossing does: the crossed head's own change is the old one mapped across the second, so
    it crosses clean, keyed by both moves as the loop keys it, and its mark and verdict carry."""
    first = _move(world)
    _push_main(world, other := world.change({"c.txt": "c landed meanwhile\n"}))
    _open(world, 1, old := world.change({"new_proto/extra.py": "X = 1\n"}))
    _mark(world, 1, old)
    tree = _git(world.dev, "merge-tree", "--write-tree", other, first)
    _push_main(world, _git(world.dev, "commit-tree", tree, "-p", other, "-p", first, "-m", "Queue: #9 the move"))
    _land(world, second := _move(world, LIB))
    out = world.tmp / "out"
    assert _cross(world, monkeypatch, "--move-commit", second, "--out", str(out)) == 0
    report = _report(out)
    assert report["refused"] == [] and [(c["pr"], c["old"], c["src"]) for c in report["crossed"]] == [(1, old, old)]
    new = report["crossed"][0]["new"]
    assert _cross(world, monkeypatch, "--push", "--from", str(out)) == 0 and _branch(world, 1) == new
    world.gh.put(1, new, ready=False)
    q = _queue(world)
    assert _git(q.workdir, "show", f"{new}:lib/new_proto/extra.py") == "X = 1" and q.own_change(new) != q.own_change(old)
    assert json.loads((out / Q.CROSS_MAPPED).read_text()) == {Q.mapped_key(old, [(first, []), (second, [])]): q.own_change(new)}
    labels = StoreLabels(world.store, change=q.own_change, moved=q.mapped_change, owners=lambda n: {f"lane-{n}"})
    assert _kept(labels, 1, new) == (True, [("land", "reviewer")])


def test_a_crossing_that_conflicts_is_refused_with_why_and_a_head_with_the_moves_or_no_verdict_is_skipped(world, monkeypatch):
    """#2 edits a line `main` edited again after the move landed: restack.py conflicts there, and the crossing is refused,
    naming the file.  #3, based after the move, has it: skipped as already crossed.  #4, marked with no land verdict, is
    skipped (its mark wouldn't carry, and restack.py would run its own code).  #5 isn't marked: skipped."""
    _open(world, 2, two := world.change({"old_proto/a.py": A.replace("return 2", "return 3")}))
    _mark(world, 2, two)
    _open(world, 4, four := world.change({"b.py": B + "W = old_proto.a.g()\n"}))
    world.store.add_label(label_target(4, four), vocab.READY, True, "lane-4", ts="2026-11-10T10:00:00Z")
    _open(world, 5, five := world.change({"e.txt": "e\n"}))
    _land(world, move := _move(world))
    _push_main(world, world.change({"new_proto/a.py": A.replace("return 2", "return 5")}))
    _open(world, 3, three := world.change({"f.txt": "f\n"}))
    _mark(world, 3, three)
    out = world.tmp / "out"
    assert _cross(world, monkeypatch, "--move-commit", move, "--out", str(out)) == 0
    report = _report(out)
    assert report["crossed"] == [] and not (out / Q.CROSS_BUNDLE).exists()
    (r,) = report["refused"]
    assert r["pr"] == 2 and r["old"] == two and "conflicts at its" in r["why"] and "new_proto/a.py" in r["why"]
    assert {s["pr"]: (s["head"], s["kind"]) for s in report["skipped"]} == {
        3: (three, "already crossed"), 4: (four, "no land verdict"), 5: (five, "not marked")}
    assert {_branch(world, n) for n in (2, 4)} == {two, four}


def test_a_head_with_no_land_verdict_is_never_crossed_so_none_of_its_code_runs_in_the_clone_the_push_runs_from(world,
                                                                                                             monkeypatch):
    """restack.py runs a crossed head's own code and the root suite its tests, in worktrees of the clone `--push` pushes from
    with its hooks and credential config.  #4, marked with no land verdict of its own, adds a root conftest that writes into
    that clone's git directory, where a hook would go: no run crosses it (`UNLANDED`), `--unlanded` is no option, and nothing
    of it runs there.  Once a reviewer passes it, the same run crosses it and its conftest runs, so the check isn't vacuous."""
    monkeypatch.setattr(Q, "CROSS_ROOT_SYNC", None)
    monkeypatch.setattr(Q, "CROSS_ROOT_PYTHON", sys.executable)
    monkeypatch.setattr(Q, "CROSS_ROOT_SUITE", ["tests", "-m", "not slow"])
    plant = ("import pathlib\nimport subprocess\n\n"
             "git = subprocess.run(['git', 'rev-parse', '--git-common-dir'], capture_output=True, text=True, check=True)\n"
             "pathlib.Path(git.stdout.strip(), 'planted').write_text('a hook could go here\\n')\n")
    _open(world, 4, four := world.change({"tests/conftest.py": plant, "tests/test_toy.py": "def test_toy():\n    pass\n"}))
    world.store.add_label(label_target(4, four), vocab.READY, True, "lane-4", ts="2026-11-10T10:00:00Z")
    _land(world, _move(world))
    planted = world.tmp / "elsewhere" / ".git" / "planted"
    assert _cross(world, monkeypatch, "--out", str(out := world.tmp / "out")) == 0
    assert [(s["pr"], s["kind"], s["why"]) for s in _report(out)["skipped"]] == [(4, "no land verdict", Q.CROSS_UNLANDED)]
    with pytest.raises(SystemExit) as refused:
        _cross(world, monkeypatch, "--unlanded", "--out", str(world.tmp / "unlanded"))
    assert refused.value.code == 2 and not planted.exists()

    world.store.add_label(label_target(4, four), vocab.REVIEWED, "land", "reviewer", ts="2026-11-10T11:00:00Z")
    assert _cross(world, monkeypatch, "--out", str(passed := world.tmp / "passed")) == 0
    assert [c["pr"] for c in _report(passed)["crossed"]] == [4] and planted.exists()


def test_an_owners_push_since_the_crossing_wins_over_its_lease(world, monkeypatch):
    """Between the dry run and the push, #6's owner pushes a commit of their own, and GitHub still shows the old head: the lease
    on the old head refuses the push, reported as the owner's, and their commit stays.  Once GitHub shows it, the push isn't
    tried."""
    _open(world, 6, old := world.change({"old_proto/a.py": A.replace("return 2", "return 3")}))
    _mark(world, 6, old)
    _land(world, _move(world))
    out = world.tmp / "out"
    assert _cross(world, monkeypatch, "--out", str(out)) == 0
    assert [c["pr"] for c in _report(out)["crossed"]] == [6]
    theirs = world.change({"g.txt": "g\n"}, base=old)
    _git(world.dev, "push", "-q", "origin", f"{theirs}:refs/heads/lane-6")
    assert _cross(world, monkeypatch, "--push", "--from", str(out)) == 0
    assert json.loads((out / Q.CROSS_PUSHED).read_text()) == {"6": "not pushed: owner pushed since"}
    assert _branch(world, 6) == theirs
    world.gh.put(6, theirs, ready=False)
    world.gh.prs[6] = dataclasses.replace(world.gh.prs[6], branch="lane-6")
    assert _cross(world, monkeypatch, "--push", "--from", str(out)) == 0
    assert json.loads((out / Q.CROSS_PUSHED).read_text()) == {"6": f"not pushed: owner pushed since ({theirs[:12]})"}


def test_restack_runs_with_mains_drivers_and_without_the_queues_credentials_or_the_stores_keys(world, monkeypatch):
    """restack.py runs each head's own code, which needs none of the queue's GitHub and Slack credentials nor the store's keys a
    node's run reads marks with: it sees none of them, and `gh` finds no stored login.  It merges with `main`'s drivers, the
    environment `cross_drivers` gave `cross_one`."""
    for k in (*Q.SCRIPT_UNSET, "AWS_ACCESS_KEY_ID", "RESEARCH_STORE_CONFIG"):
        monkeypatch.setenv(k, "dummy-" + k.lower())
    seen: dict[str, str] = {}
    monkeypatch.setattr(Q, "run_group", lambda argv, cwd, left, env=None: seen.update(env or {}) or (1, "refused"))
    head = _git(world.dev, "rev-parse", "origin/main")
    drivers = {"GIT_CONFIG_PARAMETERS": "'merge.x.driver'='python3 /d/x.py %O %A %B %P'"}
    assert Q.cross_one(world.dev, head, head, [], drivers=drivers) == {"why": f"{Q.RESTACK} exit 1: refused"}
    assert not [k for k, v in seen.items() if v.startswith("dummy-")]
    assert seen["GH_CONFIG_DIR"].startswith(seen["TMPDIR"]) and not Path(seen["GH_CONFIG_DIR"]).exists()
    assert seen["GIT_CONFIG_PARAMETERS"] == drivers["GIT_CONFIG_PARAMETERS"]


def test_the_push_never_pushes_main_checked_or_next_whatever_a_pr_names(tmp_path):
    """A crossing whose PR names `main`, `checked`, `next` or a full ref as its branch is never pushed, before any git runs."""
    report = {"main": "0" * 40, "moves": [], "crossed": [
        {"pr": n, "old": "1" * 40, "new": "2" * 40, "own": "3", "branch": branch, "below": None, "lands": True}
        for n, branch in enumerate(("main", "checked", "next", "refs/heads/x"))]}
    said = Q.cross_push(tmp_path, report, "origin", log=lambda s: None)
    assert said == {str(n): f"not pushed: its PR names the branch {b}, never pushed"
                    for n, b in enumerate(("main", "checked", "next", "refs/heads/x"))}


def test_a_stack_crosses_lowest_first_and_what_restack_wont_cross_onto_the_crossing_below_is_reported(world, monkeypatch):
    """#8 is stacked on #7: its head holds #7's.  Both marked and passed, #7 crosses onto `main` first, and #8 is tried onto
    #7's crossing, which restack.py refuses (#7's old head is a commit after the move's base that #8 shares with it): #8 is
    reported as a stack for its owners, and only #7 is pushed.  With #7 not marked, #8 isn't tried: reported likewise."""
    _open(world, 7, low := world.change({"old_proto/a.py": A.replace("return 2", "return 3")}))
    _open(world, 8, high := world.change({"b.py": B + "Y = old_proto.a.g()\n"}, base=low))
    for n, head in ((7, low), (8, high)):
        _mark(world, n, head)
    _land(world, _move(world))
    out = world.tmp / "out"
    tried: list[tuple[str, str | None]] = []
    real_one = Q.cross_one
    monkeypatch.setattr(Q, "cross_one", lambda repo, head, *a, **k: tried.append((head, k.get("onto"))) or real_one(
        repo, head, *a, **k))
    assert _cross(world, monkeypatch, "--out", str(out)) == 0
    report = _report(out)
    (c,) = report["crossed"]
    assert (c["pr"], c["below"], report["refused"]) == (7, None, [])
    assert tried == [(low, None), (high, c["new"])]
    assert [(s["pr"], s["kind"]) for s in report["skipped"]] == [(8, "stack")]
    assert "won't cross it onto #7's crossing, whose old head it holds (based-after)" in report["skipped"][0]["why"]
    pushed: list[str] = []
    real = Q._git

    def spy(repo, *args, **kw):
        if "push" in args:
            pushed.append(args[-1])
        return real(repo, *args, **kw)

    monkeypatch.setattr(Q, "_git", spy)
    assert _cross(world, monkeypatch, "--push", "--from", str(out)) == 0
    assert pushed == [f"{c['new']}:refs/heads/lane-7"] and (_branch(world, 7), _branch(world, 8)) == (c["new"], high)

    monkeypatch.setattr(Q, "_git", real)
    _open(world, 7, low)
    world.store.add_label(label_target(7, low), vocab.READY, False, "ci", ts="2026-11-10T12:00:00Z")
    assert _cross(world, monkeypatch, "--out", str(again := world.tmp / "again")) == 0
    report = _report(again)
    assert report["crossed"] == report["refused"] == []
    assert {s["pr"]: s["kind"] for s in report["skipped"]} == {7: "not marked", 8: "stack"}
    assert f"skipped #8 {high[:12]}: it holds #7's head, which this run doesn't cross: not marked ready at its head" in (
        Q.cross_table(report))


NEW_DRIVER, OLD_DRIVER = "adapters/vllm/tests/lint/merge_allowlist.py", "integrations/vllm/tests/lint/merge_allowlist.py"


@pytest.mark.parametrize("driver", [f"python3 {NEW_DRIVER} %O %A %B %P",
                                    f'f={NEW_DRIVER}; [ -f "$f" ] || f={OLD_DRIVER}; python3 "$f" %O %A %B %P'],
                         ids=["python3-path", "shell-command"])
def test_a_crossing_merges_with_mains_drivers_where_the_tree_it_merges_keeps_them_at_their_old_path(tmp_path, monkeypatch,
                                                                                                     driver):
    """Every git the queue runs registers the merge drivers (`git_env`), which may name a path a move changed: merging two
    trees from before the move, as a crossing's base merge does, finds no `python3 PATH` driver at `main`'s path and conflicts,
    unless its environment runs `main`'s copy (`cross_drivers`), as the carry check's merge's does (`mapped_change`).  A driver
    that is a shell command finding its script at either place is left as it is, and merges there itself."""
    monkeypatch.delenv("GIT_CONFIG_PARAMETERS", raising=False)
    monkeypatch.setitem(RECORD_DRIVER, "merge.vllm-allowlist.driver", driver)
    r = tmp_path / "r"
    _git(tmp_path, "init", "-q", "-b", "main", str(r))
    (r / OLD_DRIVER).parent.mkdir(parents=True)
    (r / OLD_DRIVER).write_text("import sys\na, b = (open(p).read().splitlines() for p in sys.argv[2:4])\n"
                                "open(sys.argv[2], 'w').write(''.join(f'{x}\\n' for x in a + [x for x in b if x not in a]))\n")
    (r / ".gitattributes").write_text("*.lst merge=vllm-allowlist\n")
    (r / "x.lst").write_text("a\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "base")
    (r / "x.lst").write_text("a\nb\n")
    _git(r, "commit", "-q", "-am", "theirs")
    base = _git(r, "rev-parse", "HEAD")
    (r / NEW_DRIVER).parent.mkdir(parents=True)
    _git(r, "mv", OLD_DRIVER, NEW_DRIVER)
    _git(r, "commit", "-q", "-m", "Move: tools/move/ integrations/vllm -> adapters/vllm")
    main = _git(r, "rev-parse", "HEAD")
    _git(r, "checkout", "-q", "-b", "head", "HEAD~2")
    (r / "x.lst").write_text("a\nc\n")
    _git(r, "commit", "-q", "-am", "ours")
    _git(r, "config", "merge.abs.driver", "python3 /nowhere/drv.py %O %A %B %P")

    env = Q.cross_drivers(r, main, tmp_path / "d")
    assert "nowhere" not in env.get("GIT_CONFIG_PARAMETERS", "")
    if driver.startswith("python3 "):
        assert Q._git(r, "merge-tree", "--write-tree", "HEAD", base, check=False).returncode == 1
        assert "doesn't merge" in Q.mapped_change(r, "head", main, [(main, [base])], tmp_path)[1]
        assert "doesn't merge" not in Q.mapped_change(r, "head", main, [(main, [base])], tmp_path, env)[1]
        assert list(env) == ["GIT_CONFIG_PARAMETERS"]
        assert f"'merge.vllm-allowlist.driver'='python3 {tmp_path / 'd'}/" in env["GIT_CONFIG_PARAMETERS"]
    else:
        assert "vllm-allowlist" not in env.get("GIT_CONFIG_PARAMETERS", "")
    merged = Q._git(r, "merge-tree", "--write-tree", "HEAD", base, env=env).stdout.strip()
    assert _git(r, "show", f"{merged}:x.lst") == "a\nc\nb"
    assert not subprocess.run([sys.executable, "-c", "import subprocess, sys; sys.exit(subprocess.run(['git', '-c', "
                               "'core.hooksPath=/dev/null', 'merge-tree', '--write-tree', 'HEAD', sys.argv[1]]).returncode)",
                               base], cwd=r, env={**Q._script_env(), **env}, capture_output=True).returncode


def test_a_crossing_whose_tree_fails_the_root_suite_is_refused_with_the_test_and_never_pushed(world, monkeypatch):
    """#1, marked and passed, adds a root test that reads a path the move retires through a string the map leaves alone.  It
    crosses clean and its mark would carry, but the root suite fails on the crossed tree, as it would in the batch it joins,
    so it is refused with that test and `--push` leaves its branch alone.  `main`'s root suite passes."""
    monkeypatch.setattr(Q, "CROSS_ROOT_SYNC", None)
    monkeypatch.setattr(Q, "CROSS_ROOT_PYTHON", sys.executable)
    monkeypatch.setattr(Q, "CROSS_ROOT_SUITE", ["tests", "-m", "not slow"])
    toy = "from pathlib import Path\n\n\ndef test_the_proto_is_there():\n    assert Path('old' + '_proto', 'a.py').is_file()\n"
    _open(world, 1, old := world.change({"tests/test_toy.py": toy}))
    _mark(world, 1, old)
    _land(world, _move(world))
    out = world.tmp / "out"
    assert _cross(world, monkeypatch, "--out", str(out)) == 0
    report = _report(out)
    (r,) = report["refused"]
    assert (report["crossed"], r["pr"], r["root"]) == ([], 1, ["tests/test_toy.py::test_the_proto_is_there"])
    assert r["why"] == (f"crossed clean as {r['new'][:12]}, but the root suite fails on the crossed tree: "
                        "tests/test_toy.py::test_the_proto_is_there")
    assert _cross(world, monkeypatch, "--push", "--from", str(out)) == 0 and _branch(world, 1) == old
