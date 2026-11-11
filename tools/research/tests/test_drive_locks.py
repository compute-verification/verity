"""The lock run after a Lean landing on `checked` (drive.toml's `regenerate_locks`).  A member that changes Lean, a lock or
the spec lands on its verdict like any other, with no Lean check before it, and drive runs lean-audit alone on `checked`'s tip,
which regenerates each stale lock record.  Passed, its records are committed on that tip as `Lock records regenerated for
#N...`, which the merge gate vouches for against the run, and the after-landing run checks that commit, so `main` moves only
to a commit whose records are the regenerated ones; failed, its culprit is fixed forward or reverted on `checked` as any red's
is."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research import merge
from research.drive import FIX_FORWARD_S, LAUNCH_RING_EVERY_S, LEAN_STEP, LOST_AFTER_S
from research.queue import PR
from test_drive_checked import _checked, _pass
from test_drive_review import FLOCK_LOCK, Review, _detached, _main, _move_main, _rings, _subject
from test_queue import World, _git
from test_queue_drive import _failed_run, _live_run, _steps_run

#: a change to `pkg`'s Lean and its lock
LEAN = {"pkg/A.lean": "theorem a : True := trivial\n", "pkg/lean-audit.json": '{"records": 1}\n'}
#: lean-audit's log of a build that fails in `LEAN`'s Lean file
BUILD_FAILED = "AUDIT /x/lean-audit-scratch-1/pkg: FAIL\n  build: exit 1\n    error: A.lean:1:8: unknown identifier 'x'\n"


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _locks(world: World, monkeypatch, **kw) -> Review:
    """A review landing onto `checked` whose drive commits its lock runs' records."""
    h = Review(world, monkeypatch, land_into="checked", **kw)
    h.d.regenerate_locks = True
    return h


def _lands(h: Review, changes: dict[int, dict[str, str]]) -> str:
    """Each PR of ``changes`` (its files) lands on its verdict, in one tick: `checked`'s tip after it."""
    for n, files in changes.items():
        h.world.gh.put(n, h.world.change(files))
        h.review(n)
    assert h.tick().startswith(f"landed #{max(changes)}"), h.world.logs
    return _checked(h.world)


def _passes(h: Review, rid: str, tip: str, records: dict[str, str | None]) -> None:
    """Lock run ``rid`` of ``tip`` passed, having regenerated ``records`` ({package: its lock's text})."""
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(h.world.tmp, rid, {LEAN_STEP: "passed"})]
    h.d.fetch = _live_run(h.world.tmp, rid, tip, records)[1]


def _vouched(h: Review, lock: str, rid: str) -> str | None:
    """The run the merge gate vouches for lock commit ``lock`` against (`merge.regenerated_parent`)."""
    return merge.regenerated_parent(h.world.origin, lock, h.q.store, tool="check", kinds=merge.produced(h.q.workdir, "check"),
                                    command=merge.declared_command(h.q.workdir, "check"), deferred=rid, fetch=h.d.fetch)


def _failed_audit(tmp_path: Path, rid: str, log: str | None) -> Path:
    """Lock run ``rid``'s run dir: its lean-audit failed with ``log``, or passed when ``log`` is None."""
    d = tmp_path / rid
    d.mkdir()
    st = {"step": LEAN_STEP, "status": "passed"} if log is None else {"step": LEAN_STEP, "status": "failed", "rc": 1,
                                                                       "log": "lean-audit.log"}
    (d / "steps.jsonl").write_text(json.dumps(st) + "\n")
    if log is not None:
        (d / "lean-audit.log").write_text(log)
    return d


def test_a_lean_member_lands_on_its_verdict_and_main_moves_only_to_its_lock_commit(world, monkeypatch):
    """A Lean member lands with no Lean check before it, and the lock run of `checked`'s tip starts that tick.  Nothing checks
    what landed until the run's records are committed on that tip, as the lock commit the merge gate vouches for against the
    run; the after-landing run then checks the lock commit (`--no-regenerate`), and `main` moves to it once that passed."""
    h = _locks(world, monkeypatch)
    main = _main(world)
    tip = _lands(h, {1: LEAN, 2: {"b.txt": "2\n"}})
    (rid,) = h.lock_ids()
    assert h.launched == [(tip, "m1", False)] and h.posted == [], world.logs
    assert h.postland()["locks"]["tip"] == tip and h.postland()["locks"]["state"] is None
    h.tick()
    assert len(h.launched) == 1 and h.posted == [] and _main(world) == main   # one lock run; nothing below it to check
    _passes(h, rid, tip, {"pkg": '{"records": 2}\n'})
    h.tick()
    lock = _checked(world)
    assert _subject(world, "checked") == "Lock records regenerated for #1 #2" and _git(world.origin, "rev-parse", "checked^") == tip
    assert _git(world.origin, "show", "checked:pkg/lean-audit.json") == '{"records": 2}'
    body = _git(world.origin, "log", "-1", "--format=%B", "checked")
    assert f"Checked: {rid} (the lock run of `checked` {tip[:12]}, passed by regenerating them)" in body, body
    assert "Regenerated: pkg/lean-audit.json" in body and _vouched(h, lock, rid) == rid, body
    assert h.postland()["fresh"] == lock and h.postland()["locks"]["state"] == "committed"
    assert [c for c, *_ in h.posted] == [lock] and h.posted[0][2]["no_regenerate"], world.logs
    assert _main(world) == main
    _pass(h)
    h.tick()
    assert _main(world) == lock and h.after[-1][0] == main and len(h.launched) == 1, world.logs


def test_the_bookkeeping_a_lock_run_repaired_is_committed_with_its_records_and_told(world, monkeypatch):
    """Drive's lock run under `--bookkeeping warn`: a file of `merge.REPAIRED` its Lean report's `repaired`
    names is committed at its repository path beside the regenerated locks, which the merge gate vouches for against the run
    as it does the locks; its warnings are listed in the lock commit and told to the Lean lane and each owner they name, once."""
    h = _locks(world, monkeypatch)
    tip = _lands(h, {1: LEAN})
    (rid,) = h.lock_ids()
    _passes(h, rid, tip, {"pkg": '{"records": 2}\n'})
    d = h.world.tmp / "live" / rid
    caps = "UNWITNESSED = {\"pkg\": 2}\n"
    (d / "lean-audit" / "repaired" / "tests").mkdir(parents=True)
    (d / "lean-audit" / "repaired" / "tests" / "test_lean_packages.py").write_text(caps)
    audit = json.loads((d / merge.AUDIT).read_text())
    warned = ["pkg: bookkeeping: P.Sound (@proofs): listed under `open` on LEAN-186, due 2026-11-14 (open.unlisted)"]
    audit |= {"repaired": {"tests/test_lean_packages.py":
                           f"/workspace/research/runs/{rid}/lean-audit/repaired/tests/test_lean_packages.py"},
              "bookkeeping": warned}
    (d / merge.AUDIT).write_text(json.dumps(audit))
    h.tick()
    lock = _checked(world)
    assert _git(world.origin, "show", "checked:tests/test_lean_packages.py") == caps.strip()
    body = _git(world.origin, "log", "-1", "--format=%B", "checked")
    assert "Regenerated: pkg/lean-audit.json, tests/test_lean_packages.py" in body and _vouched(h, lock, rid) == rid, body
    assert f"  {warned[0]}\n" in body and body.index(warned[0]) < body.index("Checked:"), body
    (told,) = [p for p in h.posts if "repaired bookkeeping" in p]
    assert told.startswith("@lean @proofs queue: ") and warned[0] in told and "#1" in told, told
    h.tick()
    assert len([p for p in h.posts if "repaired bookkeeping" in p]) == 1


def test_a_lean_report_naming_a_repaired_file_outside_repaired_is_unreadable(tmp_path):
    with pytest.raises(merge.Unreadable, match="no lock run repairs"):
        merge._regenerated_in("r1", {"regenerated": {}, "repaired": {"AGENTS.md": "/x/lean-audit/repaired/AGENTS.md"}})
    assert merge._regenerated_in("r1", {"repaired": {"tests/test_lean_packages.py": "/r/lean-audit/repaired/t.py"}}) == {
        "tests/test_lean_packages.py": "lean-audit/repaired/t.py"}


def test_a_lock_run_that_regenerated_nothing_leaves_its_tip_for_the_after_landing_run(world, monkeypatch):
    h = _locks(world, monkeypatch)
    tip = _lands(h, {1: LEAN})
    _passes(h, h.lock_ids()[0], tip, {})
    h.tick()
    assert _checked(world) == tip and h.postland()["fresh"] == tip and [c for c, *_ in h.posted] == [tip], world.logs
    assert any("regenerated no record" in x for x in world.logs), world.logs


def test_the_after_landing_run_checks_what_landed_below_the_first_lean_change_while_the_lock_run_runs(world, monkeypatch):
    """A landing below the first Lean change since `fresh` needs no regenerated record: the after-landing run checks it while
    the lock run runs, and `main` moves to it; the Lean landing above it waits for its lock commit."""
    h = _locks(world, monkeypatch)
    tip = _lands(h, {1: {"a.txt": "1\n"}, 2: LEAN})
    below = _git(world.origin, "rev-parse", "checked^1")
    assert h.launched == [(tip, "m1", False)] and [c for c, *_ in h.posted] == [below], world.logs
    _pass(h)
    h.tick()
    assert _main(world) == below and len(h.posted) == 1, world.logs


def test_a_lock_commit_below_a_moved_checked_is_merged_onto_it_and_main_moves_to_the_merge(world, monkeypatch):
    """Landing goes on while the lock run runs, and one lock run runs at a time.  Its records are committed on the tip it ran
    on, where the gate vouches for them, and merged onto `checked`'s tip under the same subject, taking each record no landing
    since changed; the after-landing run checks the merge."""
    h = _locks(world, monkeypatch)
    one = _lands(h, {1: LEAN})
    (rid,) = h.lock_ids()
    two = _lands(h, {2: {"b.txt": "2\n"}})
    assert len(h.launched) == 1 and h.posted == [], world.logs
    _passes(h, rid, one, {"pkg": '{"records": 2}\n'})
    h.tick()
    merged, lock = _checked(world), _git(world.origin, "rev-parse", "checked^2")
    assert _git(world.origin, "rev-parse", "checked^1") == two and _git(world.origin, "rev-parse", f"{lock}^") == one
    assert _subject(world, "checked") == _subject(world, lock) == "Lock records regenerated for #1"
    body = _git(world.origin, "log", "-1", "--format=%B", "checked")
    assert f"merged onto {two[:12]} with each record no landing since changed)" in body, body
    assert "Regenerated: pkg/lean-audit.json" in body and _vouched(h, lock, rid) == rid, body
    assert _git(world.origin, "show", "checked:pkg/lean-audit.json") == '{"records": 2}'
    assert _git(world.origin, "show", "checked:b.txt") == "2"
    assert [c for c, *_ in h.posted] == [merged] and len(h.launched) == 1, world.logs
    _pass(h)
    h.tick()
    assert _main(world) == merged, world.logs


def test_a_lean_landing_during_a_lock_run_gets_the_next_lock_run_and_main_may_move_to_the_first_lock_commit(world, monkeypatch):
    """A Lean landing while a lock run runs waits for it; once its records are merged onto the moved `checked`, the next lock
    run checks the merge, and the after-landing run meanwhile checks the first lock commit, whose records are the
    regenerated ones for all it holds, so `main` moves to it."""
    h = _locks(world, monkeypatch)
    one = _lands(h, {1: LEAN})
    (rid,) = h.lock_ids()
    _lands(h, {2: {"pkg/B.lean": "theorem b : True := trivial\n"}})
    assert len(h.launched) == 1 and h.posted == [], world.logs
    _passes(h, rid, one, {"pkg": '{"records": 2}\n'})
    h.tick()
    merged, lock = _checked(world), _git(world.origin, "rev-parse", "checked^2")
    assert h.launched[1:] == [(merged, "m1", False)] and [c for c, *_ in h.posted] == [lock], world.logs
    _pass(h)
    h.tick()
    assert _main(world) == lock, world.logs


def test_a_lock_run_that_fails_to_build_a_landings_lean_names_it_and_reverts_it_on_checked(world, monkeypatch, tmp_path):
    """A Lean failure is a red like any other: the lock run's lean-audit names the landing whose Lean doesn't build, `main`
    stays, and the culprit is reverted on `checked` once `FIX_FORWARD_S` passed with no fix marked.  The revert leaves no
    Lean change for a lock run, so the after-landing run checks it, and `main` moves to it."""
    h = _locks(world, monkeypatch)
    main = _main(world)
    tip = _lands(h, {1: LEAN, 2: {"b.txt": "2\n"}})
    (rid,) = h.lock_ids()
    h.evidence = [_failed_audit(tmp_path, rid, BUILD_FAILED)]
    h.run_state[rid] = ("done", 1, None)
    h.tick()
    (ring,) = _rings(h)
    assert f"step {LEAN_STEP} is red on `checked` {tip[:12]} (run {rid})" in ring and "Culprit: #1" in ring, ring
    assert _checked(world) == tip
    h.clock += FIX_FORWARD_S
    h.tick()
    reverted = _checked(world)
    assert _git(world.origin, "rev-parse", "checked^") == tip and _subject(world, "checked").startswith("Revert")
    assert "pkg/A.lean" not in _git(world.origin, "ls-tree", "-r", "--name-only", "checked").split()
    assert any("#1 reverted on `checked`" in p for p in h.posts), h.posts
    assert h.postland()["locks"]["state"] == "failed" and _main(world) == main
    assert len(h.launched) == 1 and [c for c, *_ in h.posted] == [reverted], world.logs
    _pass(h)
    h.tick()
    assert _main(world) == reverted, world.logs


#: a Lean file of `pkg` declaring `Verity.Core.coinBits`, and one declaring a `coinBits` in another namespace
COINS = "namespace Verity.Core\n\n@[simp] def coinBits : Nat := 3\n\nend Verity.Core\n"
OTHER = "namespace Other\n\ndef coinBits : Nat := 0\n\nend Other\n"


@pytest.mark.parametrize("change", ["edits", "removes"])
def test_a_lean_failure_naming_a_declaration_names_the_landing_that_touched_the_file_declaring_it(world, monkeypatch, tmp_path,
                                                                                                   change):
    """A Lean build breaks downstream of the declaration a landing changed, in a file no landing touched, so its error's file
    names no culprit and the Lean package's inputs name every Lean landing.  The failure names the declaration: drive takes
    the files that declare it (`git grep` for `KIND NAME` at the red's tip and where it last passed, the last part of the name
    in a file opening its namespace), so the landing that edited or removed the declaring file is the culprit, not #2, whose
    Lean declares a `coinBits` in another namespace."""
    _move_main(world, {"pkg/Coins.lean": COINS})
    h = _locks(world, monkeypatch)
    one = {"pkg/Coins.lean": COINS.replace(":= 3", ":= 4") if change == "edits" else None}
    tip = _lands(h, {1: one, 2: {"pkg/Other.lean": OTHER}, 3: {"b.txt": "3\n"}})
    (rid,) = h.lock_ids()
    log = ("AUDIT /x/lean-audit-scratch-1/pkg: FAIL\n  build: exit 1\n"
           "    error: Sampling.lean:12:4: unknown constant 'Verity.Core.coinBits'\n")
    h.evidence = [_failed_audit(tmp_path, rid, log)]
    h.run_state[rid] = ("done", 1, None)
    h.tick()
    (ring,) = _rings(h)
    assert "Culprit: #1 (its change touches pkg/Coins.lean), of the 3 landing(s)" in ring, ring
    assert h.postland()["red"][f"step {LEAN_STEP}"]["paths"][:2] == ["pkg/Coins.lean", "pkg/Sampling.lean"], h.postland()
    assert h.d._lean_declared(log, (tip,)) == ({"pkg/Coins.lean"} if change == "edits" else set())


def test_a_culprit_whose_records_a_lock_run_committed_is_reverted_with_its_lock_commit(world, monkeypatch, tmp_path):
    """A landing's lock commit rewrites the lock the landing's revert restores: reverted alone, the revert conflicts or keeps
    the records of reverted Lean.  A red after-landing run's culprit is reverted with the lock commits regenerated for it,
    and the next lock run checks what the revert leaves."""
    h = _locks(world, monkeypatch)
    main = _main(world)
    tip = _lands(h, {1: {**LEAN, "pkg/tests/test_x.py": "def test_y():\n    assert False\n"}})
    _passes(h, h.lock_ids()[0], tip, {"pkg": '{"records": 2}\n'})
    h.tick()
    lock = _checked(world)
    assert [c for c, *_ in h.posted] == [lock], world.logs
    h.run_state["r-post-0"] = ("done", 1, None)
    h.evidence = [_failed_run(tmp_path, "tests/test_x.py::test_y", "tests/test_x.py:2")]
    h.tick()
    (ring,) = _rings(h)
    assert f"suite pkg is red on `checked` {lock[:12]}" in ring and "Culprit: #1" in ring, ring
    h.clock += FIX_FORWARD_S
    h.tick()
    (told,) = [p for p in h.posts if "reverted on `checked`" in p]
    assert f"#1 reverted on `checked`, with the lock commit {lock[:12]} regenerated for it" in told, told
    assert _git(world.origin, "show", "checked:pkg/lean-audit.json") == _git(world.origin, "show", f"{main}:pkg/lean-audit.json")
    assert "pkg/A.lean" not in _git(world.origin, "ls-tree", "-r", "--name-only", "checked").split()
    assert h.launched[-1] == (_checked(world), "m1", False) and _main(world) == main, world.logs


@pytest.mark.parametrize("fault", ["enospc", "after-audit"])
def test_a_lock_run_failed_by_its_machine_runs_again_once_on_the_other_machines(world, monkeypatch, tmp_path, fault):
    """A lock run whose step ran out of disk, or that failed after its lean-audit passed, says nothing of what landed: it runs
    again once, on the machines it didn't run on.  Failed by its machine again, it is red as any failure is: named when its
    step names a culprit, which is fixed forward or reverted, else told to `CI`, since nothing then reverts and the Lean
    landings wait."""
    h = _locks(world, monkeypatch)
    h.d.on = "vy-a,vy-b,vy-c"
    tip = _lands(h, {1: {"pkg/lean-audit.json": "{}"}})
    (rid,) = h.lock_ids()
    log = "AUDIT /x/lean-audit-scratch-1/pkg: PASS\nOSError: [Errno 28] No space left on device\n"
    h.evidence = [_failed_audit(tmp_path, rid, log if fault == "enospc" else None)]
    h.run_state[rid] = ("done", 1, None, None, "vy-b")
    h.tick()
    assert h.launched == [(tip, "vy-a,vy-b,vy-c", False), (tip, "vy-a,vy-c", False)], world.logs
    assert any("a machine's fault: again, once, on vy-a,vy-c" in x for x in world.logs), world.logs
    h.run_state[h.lock_ids()[1]] = ("done", 1, None, None, "vy-a")
    h.tick()
    assert len(h.launched) == 2 and h.postland()["locks"]["state"] == "failed"
    if fault == "enospc":
        (ring,) = _rings(h)
        assert f"step {LEAN_STEP} is red on `checked` {tip[:12]}" in ring and "Culprit: #1" in ring, ring
        h.clock += FIX_FORWARD_S
        h.tick()
        assert _subject(world, "checked").startswith("Revert") and [c for c, *_ in h.posted] == [_checked(world)], world.logs
    else:
        (post,) = [p for p in h.posts if "named nothing red" in p]
        assert post.startswith("@ci queue: the lock run") and "rc=1" in post, post
        assert h.posted == [] and _checked(world) == tip, world.logs


def test_a_killed_lock_run_runs_again_once_and_one_lost_past_the_bound_too(world, monkeypatch):
    """A lock run the machine killed, or whose state can't be read for hours, said nothing of what landed: it runs again once.
    Lost again, it failed, naming nothing red, which `CI` hears."""
    h = _locks(world, monkeypatch)
    tip = _lands(h, {1: {"pkg/lean-audit.json": "{}"}})
    h.run_state[h.lock_ids()[0]] = ("done", 137, None)
    h.tick()
    assert h.launched == [(tip, "m1", False)] * 2, world.logs
    h.run_state[h.lock_ids()[1]] = ("unknown", None, None)
    h.tick()
    assert len(h.launched) == 2 and h.postland()["locks"]["state"] is None  # within the bound it still waits
    h.clock += LOST_AFTER_S + 1
    h.tick()
    assert len(h.launched) == 2 and h.postland()["locks"]["state"] == "failed"
    (post,) = [p for p in h.posts if "lost on its resend" in p]
    assert post.startswith("@ci queue:") and "named nothing red" in post, post


@pytest.mark.parametrize("part, files, lock_run", [
    ("proofs", {"security/Proofs/Y.lean": "theorem Y : True := trivial\n"}, False),
    ("spec", {"security/Definitions/Claims/Sampling.lean": "def s := 1\n"}, True),
    ("code", {"verity/protocols/flock/verifier/Flock/X.lean": "def x := Nat.gcd 4 6\n"}, True)])
def test_a_two_demos_lean_change_lands_on_its_verdict_and_a_lock_run_follows_it_unless_it_is_proofs_alone(
        world, monkeypatch, part, files, lock_run):
    """A "two demos" PR lands on its verdict as any other, its title exempting nothing: one that changes the spec, or Lean that
    runs as code, is followed by a lock run before the after-landing run checks it; one of `security/` proofs alone changes
    no statement and no lock, and the after-landing run checks it at once."""
    if part == "code":
        _move_main(world, FLOCK_LOCK)
    h = _locks(world, monkeypatch)
    world.gh.put(1, head := world.change(files))
    world.gh.prs[1] = PR(1, head, f"two demos: {part}", "Lane `lane-1`.")
    h.review(1)
    assert h.tick().startswith("landed #1"), world.logs
    tip = _checked(world)
    assert h.launched == ([(tip, "m1", False)] if lock_run else []), world.logs
    assert [c for c, *_ in h.posted] == ([] if lock_run else [tip]), world.logs


def test_a_lock_run_launches_detached_and_a_later_tick_reads_its_run_ids(world, monkeypatch, tmp_path):
    """A tick that waited on the lock run's train.sh for its run ids would take minutes.  The launch returns at once, and a
    later tick reads the ids."""
    h = _locks(world, monkeypatch)
    starts = _detached(h, tmp_path, locks=True)
    tip = _lands(h, {1: LEAN})
    rec = h.postland()["locks"]
    assert starts == [tip] and rec["ids"] == [] and rec["launch"]["dir"] == str(tmp_path / "launch-1") and rec["state"] is None
    h.tick()
    assert starts == [tip] and h.posted == [] and "launch" in h.postland()["locks"]   # still launching: it waits
    rid = "r20261106-120000-beef"
    h.printed[rec["launch"]["dir"]] = ([rid], f"RUN {rid}")
    h.run_state[rid] = ("running", None, None)
    h.clock += 600
    h.tick()
    rec = h.postland()["locks"]
    assert rec["ids"] == [rid] and "launch" not in rec and rec["at"] == h.clock and rec["state"] is None
    _passes(h, rid, tip, {})
    h.tick()
    assert h.postland()["locks"]["state"] == "committed" and starts == [tip] and [c for c, *_ in h.posted] == [tip], world.logs


def test_a_lock_run_launch_that_registered_no_run_rings_ci_and_top_and_again_only_an_hour_later(world, monkeypatch, tmp_path):
    h = _locks(world, monkeypatch)
    starts = _detached(h, tmp_path, locks=True)
    tip = _lands(h, {1: LEAN})
    h.d.launched = lambda launch: ([], "train.sh: refused: its settings check failed (exit 1)")

    def rings() -> list[str]:
        return [p for p in h.posts if "registered no run" in p]

    h.tick()
    (ring,) = rings()
    assert ring.startswith(f"@ci @top queue: train.sh {tip[:12]} m1 --deferred {LEAN_STEP} registered no run 1 time"), ring
    assert starts == [tip] * 2                                              # dropped, and launched again in the same tick
    h.clock += LAUNCH_RING_EVERY_S - 1
    h.tick()
    assert len(rings()) == 1 and starts == [tip] * 3
    h.clock += 1
    h.tick()
    assert len(rings()) == 2 and "registered no run 3 times since" in rings()[1], rings()


def test_a_launched_lock_run_survives_a_tick_that_fails_after_it(world, monkeypatch):
    """A tick that fails after launching the lock run keeps its record, so the next tick follows that run rather than
    launching another."""
    h = _locks(world, monkeypatch)
    world.gh.put(1, world.change(LEAN))
    h.review(1)
    real = h.d.start_postland

    def dies(drive):
        if (drive.get("postland") or {}).get("locks"):
            raise RuntimeError("the after-landing launch died")
        real(drive)

    h.d.start_postland = dies
    with pytest.raises(RuntimeError):
        h.tick()
    del h.d.start_postland
    h.tick()
    assert len(h.launched) == 1 and h.postland()["locks"]["state"] is None, world.logs
