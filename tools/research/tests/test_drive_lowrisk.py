"""The low-risk lane (drive.toml's `lowrisk_lane`, `Driver.land_lowrisk`): after the reviewed prefix, a tick lands up to
`lowrisk_per_tick` members of `next` onto `checked`, never `main`, on their ready mark alone, each whose merge onto `checked`'s
tip changes nothing `lowrisk_excludes` names, merges with no conflict, contains no unlanded PR and compiles."""
from __future__ import annotations

import pytest

from research.drive import LEAN_STEP
from research.queue import lowrisk
from test_drive_checked import _checked
from test_drive_review import Review, _main, _subject
from test_queue import World, _git
from test_queue_drive import _steps_run, _tip

#: drive.toml's `lowrisk_excludes`
EXCLUDES = ("*.lean", "lean-audit.json", "lakefile.*", "lean-toolchain", "lake-manifest.json", "TRUSTED.sha256",
            "verity/core/", "verity/ml/circuits/", "verity/ml/kernels/", "verity/ml/commitments/", "verity/ml/crypto/",
            "verity/ml/randomness/")
TAG = "(low-risk lane: no Lean, lock or soundness-path change)"


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _lane(world: World, monkeypatch, per_tick: int = 20, **kw) -> Review:
    into = kw.pop("land_into", "checked")
    if into == "checked":
        _git(world.origin, "update-ref", "refs/heads/checked", _main(world))
    h = Review(world, monkeypatch, land_into=into, **kw)
    h.d.lowrisk_lane, h.d.lowrisk_per_tick, h.d.lowrisk_excludes = True, per_tick, EXCLUDES
    return h


def _landed(world: World) -> list[str]:
    return [x for x in world.logs if x.endswith(TAG)]


def _skipped(world: World) -> list[str]:
    return [x for x in world.logs if "low-risk lane skipped" in x]


@pytest.mark.parametrize("files, low", [
    (["tools/research/src/research/drive.py", "README.md"], True),
    (["integrations/vllm/verity_vllm/serve.py", "verity/ml/flock/x.py", "verity/ml/definitions/y.py"], True),
    (["Security/Proofs/Core/X.lean"], False),
    (["docs/a.md", "verity/core/protocols/pous/lean/lean-audit.json"], False),
    (["verity/core/service/flock/lakefile.toml"], False),
    (["x/lean-toolchain"], False), (["x/lake-manifest.json"], False), (["x/TRUSTED.sha256"], False),
    (["verity/core/claims.py"], False),
    (["verity/ml/circuits/ops/add.py"], False), (["verity/ml/kernels/k.cu"], False),
    (["verity/ml/commitments/c.py"], False), (["verity/ml/crypto/c.py"], False), (["verity/ml/randomness/r.py"], False),
    (["verity/ml/circuitsx/a.py"], True),                   # a directory entry is a whole path component
    ([], False),                                            # nothing to land
])
def test_the_class_is_no_file_drive_toml_excludes(files, low):
    assert lowrisk(files, EXCLUDES) is low


def test_no_excludes_is_no_low_risk_change():
    assert not lowrisk(["README.md"], ())


def test_a_tick_lands_up_to_lowrisk_per_tick_low_risk_members_in_next_order_onto_checked_never_main(world, monkeypatch):
    h = _lane(world, monkeypatch, per_tick=2)
    main = _main(world)
    world.gh.put(1, world.change({"verity/core/v.py": "1\n"}))            # a verifier: waits for its review
    world.gh.put(2, world.change({"x.txt": "2\n"}))
    world.gh.put(3, world.change({"y.txt": "3\n"}))
    world.gh.put(4, world.change({"z.txt": "4\n"}))
    assert h.tick().endswith("landed #2 #3 on the low-risk lane"), world.logs
    tip = _checked(world)
    assert _subject(world, "checked") == "Queue: #3 change 3" and _subject(world, "checked^1") == "Queue: #2 change 2"
    assert _git(world.origin, "rev-parse", "checked^1^1") == main and _main(world) == main
    assert _landed(world) == [f"research queue drive: landed #{n} on `checked` at {c[:12]} {TAG}"
                              for n, c in ((2, _git(world.origin, "rev-parse", "checked^1")), (3, tip))]
    assert h.postland()["pending"] == [_git(world.origin, "rev-parse", "checked^1"), tip]
    h.tick()                                                    # `next` rebuilt on `checked` without them; #4 lands
    assert _subject(world, "checked") == "Queue: #4 change 4" and _main(world) == main
    h.tick()                                                    # and rebuilt again without #4
    assert world.on_next(h.q) == [1] and _subject(world, "checked") == "Queue: #4 change 4"


def test_off_drive_lands_only_on_review(world, monkeypatch):
    h = _lane(world, monkeypatch)
    h.d.lowrisk_lane = False
    world.gh.put(1, world.change({"x.txt": "1\n"}))
    checked = _checked(world)
    assert h.tick().startswith("nothing reviewed lands"), world.logs
    assert _checked(world) == checked and not _landed(world) and not _skipped(world)
    h.review(1)
    assert h.tick().startswith("landed #1") and not _landed(world)


def test_a_member_containing_an_unlanded_pr_waits_and_lands_once_that_pr_lands_in_the_same_tick(world, monkeypatch):
    h = _lane(world, monkeypatch)
    world.gh.put(1, one := world.change({"verity/core/v.py": "1\n"}))
    world.gh.put(2, world.change({"x.txt": "2\n"}, base=one))           # its own change is low-risk; it holds #1's
    h.tick()
    assert not _landed(world)
    assert _skipped(world)[-1] == "research queue drive: low-risk lane skipped #2 (contains #1, which isn't landed)"
    h.review(1)
    assert h.tick() == "landed #1; landed #2 on the low-risk lane", world.logs
    assert _subject(world, "checked") == "Queue: #2 change 2" and _subject(world, "checked^1") == "Queue: #1 change 1"


def test_a_conflict_on_checkeds_tip_is_skipped_and_the_rest_land(world, monkeypatch):
    """`checked` moved under `next` within the tick (here between the reviewed landing and the lane): #1 no longer merges onto
    its tip, and is skipped; #2 lands."""
    h = _lane(world, monkeypatch)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"x.txt": "2\n"}))
    real = h.d._stack_lean

    def move(drive, dry_run):
        _git(world.dev, "push", "-q", "origin", f"{world.change({'a.txt': 'other\n'}, base='origin/checked')}:refs/heads/checked")
        h.q.fetch()
        return real(drive, dry_run)

    monkeypatch.setattr(h.d, "_stack_lean", move)
    assert h.tick().endswith("landed #2 on the low-risk lane"), world.logs
    assert _skipped(world) == ["research queue drive: low-risk lane skipped #1 (conflict in a.txt)"]


def test_a_member_that_doesnt_compile_is_skipped(world, monkeypatch):
    h = _lane(world, monkeypatch)
    world.gh.put(1, world.change({"tools/x.py": "def (:\n"}))
    world.gh.put(2, world.change({"tools/y.py": "y = 1\n"}))
    assert h.tick().endswith("landed #2 on the low-risk lane"), world.logs
    (line,) = _skipped(world)
    assert line.startswith("research queue drive: low-risk lane skipped #1 (compile: tools/x.py: SyntaxError")


def test_a_hold_verdict_or_a_withdrawn_mark_keeps_a_member_off_the_lane(world, monkeypatch):
    h = _lane(world, monkeypatch)
    world.gh.put(1, world.change({"x.txt": "1\n"}))
    world.gh.put(2, world.change({"y.txt": "2\n"}))
    world.gh.put(3, three := world.change({"z.txt": "3\n"}))
    h.review(1, "hold", ref="wait")
    real = h.d._stack_lean

    def withdraw(drive, dry_run):
        world.gh.ready.discard((3, three))                     # after the sync that kept it on `next`
        return real(drive, dry_run)

    monkeypatch.setattr(h.d, "_stack_lean", withdraw)
    assert h.tick().endswith("landed #2 on the low-risk lane"), world.logs
    assert _skipped(world) == [f"research queue drive: low-risk lane skipped #1 (held by reviewer (wait)), "
                               f"#3 (no ready mark at {three[:8]})"]


def _carried(world: World) -> list[str]:
    return [x for x in world.logs if "(the low-risk lane's), and trees that differ in no file `lowrisk_excludes` names" in x]


def _lean(h: Review) -> dict:
    return h.d.q.load_state()["drive"]["lean"]


def test_a_running_lean_check_moves_across_a_low_risk_landing_and_lands_there(world, monkeypatch):
    """#2 lands off `next` under the running Lean check of #1: `next` is rebuilt on `checked` and the check holds #1's new
    commit, whose tree differs from the one it checks by #2's file alone, so it isn't cancelled or launched again."""
    h = _lane(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/T.lean": "theorem T : True := trivial\n"}))
    h.review(1)
    h.tick()
    one = _tip(world)
    assert [c for c, *_ in h.launched] == [one], world.logs
    world.gh.put(2, world.change({"x.txt": "2\n"}))
    assert h.tick().endswith("landed #2 on the low-risk lane"), world.logs
    assert _subject(world, "checked") == "Queue: #2 change 2" and world.on_next(h.q) == [1]
    new = _tip(world)
    assert _git(world.origin, "rev-parse", f"{new}^1") == _checked(world)
    (rid,) = h.lean_ids()
    assert _carried(world) == [f"research queue drive: the Lean check {rid} of {one[:12]} (running) holds {new[:12]} of `next` "
                               f"rebuilt on `checked` {_checked(world)[:12]}: the same PRs up to #1 at the same heads less #2 (the "
                               "low-risk lane's), and trees that differ in no file `lowrisk_excludes` names"]
    rec = _lean(h)[new]
    assert list(_lean(h)) == [new] and rec["members"] == [[1, world.gh.prs[1].head]] and rec["rekeyed_from"] == one
    assert not any("low-risk lane waits" in x for x in world.logs)
    h.tick()
    assert h.cancelled == [] and len(h.launched) == 1 and list(_lean(h)) == [new], world.logs
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #1"), world.logs
    assert _checked(world) == new and len(h.launched) == 1


def _moved_regenerating(world: World, monkeypatch, tmp_path) -> tuple[Review, str, str, str]:
    """#1's running Lean check moved across #2's low-risk landing, then passed by regenerating #1's lock: (the harness, its run,
    the commit it ran on, the commit it holds)."""
    from test_queue_drive import _live_run

    h = _lane(world, monkeypatch)
    h.d.regenerate_locks = True
    world.gh.put(1, world.change({"pkg/T.lean": "theorem T : True := trivial\n", "pkg/lean-audit.json": '{"records": 1}\n'}))
    h.review(1)
    h.tick()
    one, (rid,) = _tip(world), h.lean_ids()
    world.gh.put(2, world.change({"x.txt": "2\n"}))
    assert h.tick().endswith("landed #2 on the low-risk lane"), world.logs
    new = _tip(world)
    assert _lean(h)[new]["rekeyed_from"] == one and len(_carried(world)) == 1, world.logs
    h.d.fetch = _live_run(tmp_path, rid, one, {"pkg": '{"records": 2}\n'})[1]
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {LEAN_STEP: "passed"})]
    return h, rid, one, new


def test_a_moved_lean_check_that_passes_by_regenerating_records_lands_them_on_the_moved_commit(world, monkeypatch, tmp_path):
    """What the Lean check reads is the same in the tree it ran on and in the moved commit's, which differ by #2's file alone:
    its regenerated records land on the moved commit, which the lock gate vouches for (`merge.regenerated_parent` with
    `rekeyed_from`), and nothing is checked again."""
    h, rid, one, new = _moved_regenerating(world, monkeypatch, tmp_path)
    assert h.tick().startswith("landed #1"), world.logs
    tip = _checked(world)
    assert _git(world.origin, "rev-parse", f"{tip}^") == new and len(h.launched) == 1 and h.cancelled == [], world.logs
    assert _git(world.origin, "diff", "--name-only", new, tip) == "pkg/lean-audit.json"
    assert _git(world.origin, "show", f"{tip}:pkg/lean-audit.json") == '{"records": 2}'
    assert f"Checked: {rid} (the Lean check of {one[:12]}, passed" in _git(world.origin, "log", "-1", "--format=%B", tip)


def test_a_moved_lean_check_whose_trees_differ_in_an_excluded_file_is_checked_again(world, monkeypatch, tmp_path):
    """`lowrisk_excludes` came to name #2's file after the move: the records its check regenerated no longer hold on the moved
    commit by the rule the lock gate applies, so it isn't landed (the gate would refuse it at every tick) but checked again."""
    h, rid, one, new = _moved_regenerating(world, monkeypatch, tmp_path)
    checked = _checked(world)
    h.d.lowrisk_excludes = (*EXCLUDES, "x.txt")
    assert not h.tick().startswith("landed"), world.logs
    assert _checked(world) == checked and len(h.launched) == 2 and h.launched[1][0] == new, h.launched
    assert any(f"the Lean check {rid} moved to {new[:12]} regenerated lock records, which don't hold there (the trees of "
               f"{one[:12]} and {new[:12]} differ in x.txt, which a Lean check reads or `lowrisk_excludes` names)" in x
               for x in world.logs), world.logs


def test_a_passed_lean_check_awaiting_its_verdict_moves_too_and_the_lane_doesnt_wait(world, monkeypatch):
    h = _lane(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/T.lean": "theorem T : True := trivial\n"}))
    h.tick()
    one = _tip(world)
    h.d.hand_lean(one, "ci")                                    # drive's own record: it moves
    (rid,) = h.lean_ids()
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {LEAN_STEP: "passed"})]
    world.gh.put(2, world.change({"x.txt": "2\n"}))
    assert h.tick().endswith("landed #2 on the low-risk lane"), world.logs
    new = _tip(world)
    assert len(_carried(world)) == 1 and _lean(h)[new]["state"] == "passed" and list(_lean(h)) == [new], world.logs
    h.review(1)
    assert h.tick().startswith("landed #1"), world.logs
    assert _checked(world) == new and len(h.launched) == 1


def test_a_lean_check_isnt_moved_across_a_change_lowrisk_excludes_names_and_the_lane_waits(world, monkeypatch):
    """(b): `checked` moved within the tick by a verifier's file (`verity/core/`), so #1's commit rebuilt on the lane's
    landing differs from the one its check runs on by more than the lane's change."""
    h = _lane(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/T.lean": "theorem T : True := trivial\n"}))
    h.review(1)
    h.tick()
    one = _tip(world)
    world.gh.put(2, world.change({"x.txt": "2\n"}))
    real, moved = h.d._stack_lean, []

    def move(drive, dry_run):
        _git(world.dev, "push", "-q", "origin",
             f"{world.change({'verity/core/v.py': 'v = 1\n'}, base='origin/checked')}:refs/heads/checked")
        h.q.fetch()
        moved.append(_checked(world))
        return real(drive, dry_run)

    monkeypatch.setattr(h.d, "_stack_lean", move)
    h.tick()
    assert _checked(world) == moved[0] and not _landed(world) and not _carried(world), world.logs
    assert list(_lean(h)) == [one] and _git(world.origin, "rev-parse", "next^1") == one
    (line,) = [x for x in world.logs if "low-risk lane waits" in x]
    assert line.startswith(f"research queue drive: low-risk lane waits: a landing off `next` rebuilds it, and (b) can't be "
                           f"shown for the Lean check {h.lean_ids()[0]} of {one[:12]}: its tree and that of "), line
    assert line.endswith("differ in verity/core/v.py, which `lowrisk_excludes` names"), line


def test_the_lanes_merge_rebuilds_conflicted_pins_as_the_sync_does(world, monkeypatch):
    """The lane merges each member again (onto `checked`'s tip, and above a Lean check it moves) as the sync merges it: a merge
    whose only conflicts are in pins keeps each pin rebuilt from the merged tree.  Refusing it stopped every landing of a tick
    on one member's PoUS pin, which the sync had merged."""
    from test_queue import EXFIL, POUS, TRUSTED, UP, _pin_main, _pinned, _trusted

    tree = _pin_main(world)
    h = _lane(world, monkeypatch)
    one = {**tree, f"{POUS}/lakefile.toml": "l1\n"}
    two = {**tree, f"{POUS}/lean-toolchain": "t1\n"}
    paths = ["lean-toolchain", "lakefile.toml", "Old.lean", UP]
    exfil = ["Security/Proofs/lakefile.toml", "x.lean"]
    a, b = _pinned(world, one, paths, exfil), _pinned(world, two, paths, exfil)
    _git(h.d.repo, "fetch", "-q", "origin", f"refs/dev/{a}", f"refs/dev/{b}")
    merged, why = h.d._clean_merge(a, b)
    assert merged and why == "", why
    assert _git(h.d.repo, "show", f"{merged}:{TRUSTED}") + "\n" == _trusted(paths, {**one, f"{POUS}/lean-toolchain": "t1\n"})
    assert _git(h.d.repo, "show", f"{merged}:{EXFIL}")
    d, e = world.change({"q.txt": "1\n"}), world.change({"q.txt": "2\n"})
    _git(h.d.repo, "fetch", "-q", "origin", f"refs/dev/{d}", f"refs/dev/{e}")
    assert h.d._clean_merge(d, e) == (None, "conflict in q.txt")


def test_the_lanes_merge_ports_the_lean_a_member_adds_as_the_sync_does(world, monkeypatch, tmp_path):
    """Above a Lean check it moves, the lane merges each member again: once the tree has Lean's module system, the merged
    tree's `.lean` files that aren't modules are ported, as the sync's merge ported them, so the moved check holds the tree it
    checked."""
    from test_queue_ported import _port, _tooling

    h = _lane(world, monkeypatch)
    a = world.change(_tooling(tmp_path / "argv"))
    b = world.change({"Lib/New.lean": "import Lib.Old\n"}, base=a)
    _git(h.d.repo, "fetch", "-q", "origin", f"refs/dev/{a}", f"refs/dev/{b}")
    merged, why = h.d._clean_merge(a, b)
    assert merged and why == "", why
    assert _git(h.d.repo, "show", f"{merged}:Lib/New.lean") + "\n" == _port("import Lib.Old\n")


def test_the_lane_waits_while_a_lean_check_drive_didnt_launch_runs(world, monkeypatch):
    """A hand check's run is of the commit it runs on, which a landing off `next` takes off it: (a) can't be shown for it."""
    h = _lane(world, monkeypatch)
    h.d.lowrisk_lane = False
    world.gh.put(1, world.change({"x.txt": "1\n"}))
    h.tick()
    tip = _git(world.origin, "rev-parse", "next")
    h.hand[tip] = ["r-hand"]                                    # ci's hand Lean check of `next`'s tip
    h.run_state["r-hand"] = ("running", None, None)
    h.d.lowrisk_lane = True
    checked = _checked(world)
    h.tick()
    assert _checked(world) == checked and not _landed(world), world.logs
    assert f"research queue drive: low-risk lane waits: a landing off `next` rebuilds it, and the Lean check of {tip[:12]}, " \
           "which drive didn't launch, runs, and its run is of that commit" in world.logs


#: drive.toml's `lowrisk_excludes` in full: the guards and the landing rules too
FULL = (*EXCLUDES, "tools/verity/lean/", "tests/", "tools/verity/check/", ".gitattributes", "drive.py", "queue.py")
REVIEWED_TAG = ("(low-risk lane: reviewed, no Lean change and nothing a Lean check reads, above the prefix a Lean check of "
                "`next` holds)")


def _running_lean(world: World, monkeypatch) -> tuple[Review, str]:
    """A Lean check of #1, a reviewed Lean change, running; (the harness, its commit)."""
    h = _lane(world, monkeypatch)
    h.d.lowrisk_excludes = FULL
    world.gh.put(1, world.change({"pkg/T.lean": "theorem T : True := trivial\n"}))
    h.review(1)
    h.tick()
    assert [c for c, *_ in h.launched] == [one := _tip(world)], world.logs
    return h, one


def test_a_reviewed_non_lean_fix_above_a_running_lean_check_lands_below_it_and_the_check_moves(world, monkeypatch):
    """#2, reviewed, changes no Lean and nothing a Lean check reads, but a file `lowrisk_excludes` names (a test): it lands on
    `checked` in the tick it comes, below the running check of #1, which moves to #1's commit on it without a relaunch, and
    lands there on its pass."""
    h, one = _running_lean(world, monkeypatch)
    world.gh.put(2, world.change({"tests/test_x.py": "x = 2\n"}))
    h.review(2)
    assert h.tick().endswith("landed #2 on the low-risk lane"), world.logs
    assert _subject(world, "checked") == "Queue: #2 change 2" and world.on_next(h.q) == [1]
    assert any(x.endswith(REVIEWED_TAG) and x.startswith("research queue drive: landed #2 on `checked`") for x in world.logs)
    new = _tip(world)
    (rid,) = h.lean_ids()
    assert list(_lean(h)) == [new] and _lean(h)[new]["rekeyed_from"] == one and _lean(h)[new]["state"] is None
    assert any(f"the Lean check {rid} of {one[:12]} (running) holds {new[:12]}" in x and "but a reviewed landing's" in x
               for x in world.logs), world.logs
    h.tick()
    assert h.cancelled == [] and len(h.launched) == 1 and list(_lean(h)) == [new], world.logs
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(world.tmp, rid, {LEAN_STEP: "passed"})]
    assert h.tick().startswith("landed #1"), world.logs
    assert _checked(world) == new and len(h.launched) == 1


@pytest.mark.parametrize("files, reviewed", [({"tools/verity/lean/check.py": "x = 2\n"}, True),
                                             ({"tests/test_x.py": "x = 2\n"}, False)],
                         ids=["reviewed, touching what a Lean check reads", "unreviewed"])
def test_a_member_above_a_running_lean_check_that_touches_a_lean_input_or_has_no_verdict_waits(world, monkeypatch, files,
                                                                                                reviewed):
    h, one = _running_lean(world, monkeypatch)
    checked = _checked(world)
    world.gh.put(2, world.change(files))
    if reviewed:
        h.review(2)
    h.tick()
    h.tick()
    assert _checked(world) == checked and not any("landed #2" in x for x in world.logs), world.logs
    assert world.on_next(h.q) == [1, 2] and list(_lean(h)) == [one] and len(h.launched) == 1 and h.cancelled == []


def test_a_reviewed_members_files_dont_let_a_check_move_across_another_change_to_an_excluded_file(world, monkeypatch):
    """(b) takes the reviewed landing's own files only: `checked` moved within the tick by a verifier's file, so the lane waits,
    and lands nothing that tick."""
    h, one = _running_lean(world, monkeypatch)
    world.gh.put(2, world.change({"tests/test_x.py": "x = 2\n"}))
    h.review(2)
    real = h.d._stack_lean

    def move(drive, dry_run):
        _git(world.dev, "push", "-q", "origin",
             f"{world.change({'verity/core/v.py': 'v = 1\n'}, base='origin/checked')}:refs/heads/checked")
        h.q.fetch()
        return real(drive, dry_run)

    monkeypatch.setattr(h.d, "_stack_lean", move)
    h.tick()
    assert not any("landed #2" in x for x in world.logs) and list(_lean(h)) == [one], world.logs
    (line,) = [x for x in world.logs if "low-risk lane waits" in x]
    assert line.endswith("differ in verity/core/v.py, which `lowrisk_excludes` names"), line


def test_the_lane_never_lands_on_main(world, monkeypatch):
    h = _lane(world, monkeypatch, land_into="main")
    main = _main(world)
    world.gh.put(1, world.change({"x.txt": "1\n"}))
    h.tick()
    assert _main(world) == main and not _landed(world)
