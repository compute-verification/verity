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


def test_the_lane_waits_while_a_lean_check_of_next_runs(world, monkeypatch):
    """A landing off `next` rebuilds all of it on the next sync, which would take the commit a Lean check runs on off `next`."""
    h = _lane(world, monkeypatch)
    h.d.lowrisk_lane = False
    world.gh.put(1, world.change({"x.txt": "1\n"}))
    h.tick()
    h.hand[_git(world.origin, "rev-parse", "next")] = ["r-hand"]          # ci's hand Lean check of `next`'s tip
    h.run_state["r-hand"] = ("running", None, None)
    h.d.lowrisk_lane = True
    checked = _checked(world)
    h.tick()
    assert _checked(world) == checked and not _landed(world), world.logs
    assert any(x.startswith("research queue drive: low-risk lane waits: a landing off `next` would rebuild it under the Lean "
                            "check of ") for x in world.logs)


def test_the_lane_never_lands_on_main(world, monkeypatch):
    h = _lane(world, monkeypatch, land_into="main")
    main = _main(world)
    world.gh.put(1, world.change({"x.txt": "1\n"}))
    h.tick()
    assert _main(world) == main and not _landed(world)
