"""The low-risk lane (drive.toml's `lowrisk_lane`, `Driver.land_lowrisk`): after the reviewed prefix, a tick lands up to
`lowrisk_per_tick` members of `next` onto `checked`, never `main`, on their ready mark alone, each whose merge onto `checked`'s
tip changes nothing `lowrisk_excludes` names, merges with no conflict, contains no unlanded PR and compiles."""
from __future__ import annotations

import pytest

from research.queue import lowrisk
from test_drive_checked import _checked
from test_drive_review import Review, _main, _subject
from test_queue import World, _git

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
    real = h.d.land_reviewed

    def move(drive, dry_run):
        out = real(drive, dry_run)
        _git(world.dev, "push", "-q", "origin", f"{world.change({'a.txt': 'other\n'}, base='origin/checked')}:refs/heads/checked")
        h.q.fetch()
        return out

    monkeypatch.setattr(h.d, "land_reviewed", move)
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
    real = h.d.land_reviewed

    def withdraw(drive, dry_run):
        world.gh.ready.discard((3, three))                     # after the sync that kept it on `next`
        return real(drive, dry_run)

    monkeypatch.setattr(h.d, "land_reviewed", withdraw)
    assert h.tick().endswith("landed #2 on the low-risk lane"), world.logs
    assert _skipped(world) == [f"research queue drive: low-risk lane skipped #1 (held by reviewer (wait)), "
                               f"#3 (no ready mark at {three[:8]})"]


<<<<<<< 99c8914848
def _carried(world: World) -> list[str]:
    return [x for x in world.logs if "(the low-risk lane's), and trees that differ in no file `lowrisk_excludes` names" in x]


def _lean(h: Review) -> dict:
    return h.d.q.load_state()["drive"]["lean"]


def _rebuilds(world: World) -> list[str]:
    return [x for x in world.logs if x.startswith("research queue drive: low-risk rebuild of `next`")]


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
    merge = h.d._clean_merge

    def slow(tip, head):                                        # each merge takes 1.5 s of drive's clock
        h.clock += 1.5
        return merge(tip, head)

    monkeypatch.setattr(h.d, "_clean_merge", slow)
    assert h.tick().endswith("landed #2 on the low-risk lane"), world.logs
    assert _subject(world, "checked") == "Queue: #2 change 2" and world.on_next(h.q) == [1]
    assert _rebuilds(world) == [f"research queue drive: low-risk rebuild of `next` on `checked` {_checked(world)[:12]}: 1 of 1 "
                                "member(s) merged again in 1.5 s, kept: 1 Lean check(s) move to it"]
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
    (line,) = _rebuilds(world)
    assert line.endswith(": 1 of 1 member(s) merged again in 0.0 s, dropped: the lane waits"), line


||||||| merged common ancestors
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


=======
>>>>>>> 7af4d24fca3e
def test_the_lanes_merge_rebuilds_conflicted_pins_as_the_sync_does(world, monkeypatch):
    """The lane merges each member again onto `checked`'s tip as the sync merges it: a merge whose only conflicts are in pins
    keeps each pin rebuilt from the merged tree.  Refusing it stopped every landing of a tick
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
    """The lane merges each member again onto `checked`'s tip: once the tree has Lean's module system, the merged tree's
    `.lean` files that aren't modules are ported, as the sync's merge ported them."""
    from test_queue_ported import _port, _tooling

    h = _lane(world, monkeypatch)
    a = world.change(_tooling(tmp_path / "argv"))
    b = world.change({"Lib/New.lean": "import Lib.Old\n"}, base=a)
    _git(h.d.repo, "fetch", "-q", "origin", f"refs/dev/{a}", f"refs/dev/{b}")
    merged, why = h.d._clean_merge(a, b)
    assert merged and why == "", why
    assert _git(h.d.repo, "show", f"{merged}:Lib/New.lean") + "\n" == _port("import Lib.Old\n")


def test_the_lanes_merge_takes_a_members_deletion_of_lean_the_tip_only_ported_as_the_sync_does(world, monkeypatch, tmp_path):
    """A member made before the port that deletes a `.lean` file the tip changed only by porting it merges again with the
    file deleted, as the sync merged it (`queue.port_deletions`); one the tip changed beyond porting still conflicts."""
    from test_queue_ported import _port, _tooling

    h = _lane(world, monkeypatch)
    old = "-- made before the port\nimport Init\n"
    base = world.change({"Lib/Gone.lean": old, "Lib/Edited.lean": old})
    gone, edited = (world.change({f: None}, base=base) for f in ("Lib/Gone.lean", "Lib/Edited.lean"))
    tip = world.change({**_tooling(tmp_path / "argv"), "Lib/Gone.lean": _port(old),
                        "Lib/Edited.lean": _port(old) + "\ndef y := 2\n"}, base=base)
    _git(h.d.repo, "fetch", "-q", "origin", *(f"refs/dev/{c}" for c in (gone, edited, tip)))
    merged, why = h.d._clean_merge(tip, gone)
    assert merged and why == "", why
    assert _git(h.d.repo, "ls-tree", "--name-only", merged, "--", "Lib/Gone.lean") == ""
    assert _git(h.d.repo, "show", f"{merged}:Lib/Edited.lean") + "\n" == _port(old) + "\ndef y := 2\n"
    assert h.d._clean_merge(tip, edited) == (None, "conflict in Lib/Edited.lean")


def test_the_lanes_merge_is_the_syncs_record_and_a_tick_keeps_its_own_for_the_next(world, monkeypatch):
    """The lane merges onto `checked`'s tip through the sync's merge records (`Queue._recorded`): `next`'s lowest member, which
    the sync merged onto that tip, lands with no merge of its own, and `next` rebuilt the next tick on the landing, a commit
    with the tree of that merge, merges nothing again.  A merge the lane makes and leaves out (it doesn't compile) is kept in
    the queue's state by drive's tick, and the next tick's lane, in a new process, takes it."""
    from test_queue import _merge_trees

    h = _lane(world, monkeypatch, per_tick=1)
    world.gh.put(1, h1 := world.change({"x.txt": "1\n"}))
    world.gh.put(2, h2 := world.change({"y.txt": "2\n"}))
    calls = _merge_trees(monkeypatch)
    assert h.tick().endswith("landed #1 on the low-risk lane"), world.logs
    assert calls == [h1, h2]
    calls.clear()
    h.q._merge_records = {}
    assert h.tick().endswith("landed #2 on the low-risk lane"), world.logs
    assert calls == [] and _subject(world, "checked") == "Queue: #2 change 2"

    world.gh.put(3, h3 := world.change({"verity/core/v.py": "3\n"}))     # a verifier: waits for its review
    world.gh.put(4, h4 := world.change({"bad.py": "def (\n"}))
    h.tick()
    assert calls == [h3, h4, h4], calls                         # the sync's two, and #4 onto `checked`'s tip, under #3
    assert _skipped(world)[-1].startswith("research queue drive: low-risk lane skipped #4 (compile: bad.py"), world.logs
    calls.clear()
    h.q._merge_records = {}
    h.tick()
    assert h4 not in calls and _skipped(world)[-1].startswith("research queue drive: low-risk lane skipped #4 (compile"), calls


def test_the_lane_never_lands_on_main(world, monkeypatch):
    h = _lane(world, monkeypatch, land_into="main")
    main = _main(world)
    world.gh.put(1, world.change({"x.txt": "1\n"}))
    h.tick()
    assert _main(world) == main and not _landed(world)


def test_a_rebuild_cut_at_a_member_that_doesnt_merge_again_is_completed_by_the_next_sync(world, monkeypatch):
    """`checked` moved within the tick by a file #3 changes too: the lane lands #2, its rebuild of `next` stops at #3, which no
    longer merges, and pushes `next` cut there with #1's moved check below it.  The next sync appends #4 again and leaves
    #3 out as a conflict, told as any; nobody's ready mark is withdrawn."""
    h = _lane(world, monkeypatch)
    world.gh.put(1, world.change({"pkg/T.lean": "theorem T : True := trivial\n"}))
    h.review(1)
    h.tick()
    world.gh.put(2, world.change({"x.txt": "2\n"}))
    world.gh.put(3, three := world.change({"a.txt": "3\n"}))
    world.gh.put(4, four := world.change({"verity/core/v.py": "v = 4\n"}))      # not the lane's
    real, moved = h.d._stack_lean, []

    def move(drive, dry_run):
        if not moved:
            _git(world.dev, "push", "-q", "origin",
                 f"{world.change({'a.txt': 'other\n'}, base='origin/checked')}:refs/heads/checked")
            h.q.fetch()
            moved.append(_checked(world))
        return real(drive, dry_run)

    monkeypatch.setattr(h.d, "_stack_lean", move)
    assert h.tick().endswith("landed #2 on the low-risk lane"), world.logs
    assert world.on_next(h.q) == [1] and len(_carried(world)) == 1, world.logs
    (line,) = _rebuilds(world)
    assert line.endswith(": 1 of 3 member(s) merged again in 0.0 s, kept: 1 Lean check(s) move to it"), line
    h.tick()
    assert world.on_next(h.q) == [1, 4], world.logs
    assert (3, three) in world.gh.ready and (4, four) in world.gh.ready
    assert "3" in h.q.load_state()["conflicts"], world.logs
    assert h.cancelled == [] and len(h.launched) == 1
    assert any(f"#3 at {three[:8]} is out of `next`: conflicts with `next` in a.txt" in p for p in h.posts), h.posts
