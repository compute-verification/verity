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
            "verity/core/", "verity/circuits/", "verity/definitions/kernels/", "verity/commitments/", "verity/commitments/crypto/",
            "verity/commitments/randomness/")
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
    (["adapters/vllm/verity_vllm/serve.py", "verity/ml/flock/x.py", "verity/ml/definitions/y.py"], True),
    (["Security/Proofs/Core/X.lean"], False),
    (["docs/a.md", "verity/protocols/pous/lean/lean-audit.json"], False),
    (["verity/protocols/flock/verifier/lakefile.toml"], False),
    (["x/lean-toolchain"], False), (["x/lake-manifest.json"], False), (["x/TRUSTED.sha256"], False),
    (["verity/core/claims.py"], False),
    (["verity/definitions/ops/add.py"], False), (["verity/definitions/kernels/k.cu"], False),
    (["verity/ml/commitments/c.py"], False), (["verity/commitments/crypto/c.py"], False), (["verity/commitments/randomness/r.py"], False),
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
