"""The verdict carry in drive (drive.toml's `verdict_carry`, `Driver._carry_verdicts`): a member with no land verdict at its head
takes its PR's land verdict from an older head when its commits since only merge trusted commits as the queue would
(`research.carry`, whose own cases are `test_carry`).  "shadow" logs what it would carry and writes nothing; "on" writes the
land verdict at the head as ci's, naming both heads, and the landing takes it."""
from __future__ import annotations

import pytest

from research.queue import label_target
from research.store import vocab
from test_drive_review import Review, _main, _move_main, _subject
from test_queue import World, _git, _merge


#: when `_reviewed`'s reviewer landed #1's older head, so a later verdict is newer whatever the clock says
LANDED_AT = "2026-11-01T00:00:00Z"


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _reviewed(world: World, monkeypatch, mode: str, body: str | None = None) -> tuple[Review, str, str]:
    """(a drive with `verdict_carry` ``mode`` that saw #1 at ``old`` with no verdict before a reviewer's land there, ``old``,
    `main`, which moved after ``old`` was made)."""
    old = world.change({"a.txt": "1\n"})
    _move_main(world, {"m.txt": "moved\n"})
    h = Review(world, monkeypatch)
    h.d.verdict_carry = mode
    world.gh.put(1, old, body=body)
    assert "no land verdict" in h.tick(), world.logs
    world.store.add_label(label_target(1, old), vocab.REVIEWED, "land", "reviewer", ts=LANDED_AT)
    return h, old, _main(world)


def _verdicts(world: World, pr: int, head: str) -> list:
    return world.store.labels(label_target(pr, head), key=vocab.REVIEWED)


def _said(world: World, head: str) -> list[str]:
    return [x for x in world.logs if "carr" in x and f"to {head[:12]} (" in x]


def test_shadow_logs_the_carry_once_beside_the_landings_own_decision_and_writes_no_label(world, monkeypatch):
    h, old, main = _reviewed(world, monkeypatch, "off")
    world.gh.put(1, new := _merge(world, old, main))
    assert h.tick().startswith(f"nothing reviewed lands: #1 at {new[:8]}: no land verdict"), world.logs
    assert _said(world, new) == []
    h.d.verdict_carry = "shadow"
    for _ in range(2):
        assert h.tick().startswith(f"nothing reviewed lands: #1 at {new[:8]}: no land verdict"), world.logs
    assert _said(world, new) == [f"research queue drive: would carry #1's land from {old[:12]} to {new[:12]} (its 1 commit(s) "
                                 f"since only merge {main[:12]} as the queue would)"]
    assert _verdicts(world, 1, new) == []


def test_on_writes_cis_land_verdict_naming_both_heads_and_the_member_lands_on_it(world, monkeypatch):
    h, old, main = _reviewed(world, monkeypatch, "on")
    world.gh.put(1, new := _merge(world, old, main))
    assert h.tick().startswith("landed #1"), world.logs
    assert _subject(world, "main").startswith("Queue: #1")
    (lab,) = _verdicts(world, 1, new)
    assert (lab.by, lab.value) == ("ci", "land")
    assert lab.ref == f"carried from {old} to {new}: its 1 commit(s) since only merge {main[:12]} as the queue would"
    (line,) = _said(world, new)
    assert line.startswith(f"research queue drive: carried #1's land from {old[:12]} to {new[:12]} (its 1 commit(s) since "
                           f"only merge {main[:12]} as the queue would); written as ci ({label_target(1, new)}; "), line


def test_a_hold_at_the_new_head_wins(world, monkeypatch):
    h, old, main = _reviewed(world, monkeypatch, "on")
    world.gh.put(1, new := _merge(world, old, main))
    h.review(1, "hold", by="other", ref="the merge breaks x")
    out = h.tick()
    assert out.startswith(f"nothing reviewed lands: #1 at {new[:8]}: held by other (the merge breaks x)"), out
    assert _said(world, new) == [f"research queue drive: did not carry #1's land from {old[:12]} to {new[:12]} (held by other "
                                 f"(the merge breaks x) at {new[:12]})"]
    assert [lab.value for lab in _verdicts(world, 1, new)] == ["hold"]


def test_a_hold_at_an_older_head_newer_than_the_land_keeps_it(world, monkeypatch):
    """A reviewer held the merge at ``mid`` after the land at ``old``: a merge of `main` above ``mid`` doesn't carry that land."""
    h, old, main = _reviewed(world, monkeypatch, "shadow")
    world.gh.put(1, mid := _merge(world, old, main))
    h.tick()
    assert _said(world, mid)[0].startswith(f"research queue drive: would carry #1's land from {old[:12]} to {mid[:12]} (")
    world.store.add_label(label_target(1, mid), vocab.REVIEWED, "hold", "other", ref="the merge breaks x",
                          ts="2026-11-02T00:00:00Z")
    _move_main(world, {"n.txt": "moved again\n"})
    world.gh.put(1, new := _merge(world, mid, _main(world)))
    h.tick()
    assert _said(world, new) == [f"research queue drive: would not carry #1's land from {old[:12]} to {new[:12]} (other's hold at "
                                 f"{mid[:12]} (the merge breaks x) is newer)"]


def test_a_hold_standing_at_the_older_head_keeps_it_though_the_land_there_is_newer(world, monkeypatch):
    """Another reviewer's hold at ``old`` stands beside the land there, so ``old`` doesn't land on its verdicts, nor does the
    head the land would carry to."""
    h, old, main = _reviewed(world, monkeypatch, "on")
    world.store.add_label(label_target(1, old), vocab.REVIEWED, "hold", "other", ref="x is wrong", ts="2026-10-31T00:00:00Z")
    world.gh.put(1, new := _merge(world, old, main))
    assert h.tick().startswith(f"nothing reviewed lands: #1 at {new[:8]}: no land verdict"), world.logs
    assert _said(world, new) == [f"research queue drive: did not carry #1's land from {old[:12]} to {new[:12]} (held by other "
                                 f"(x is wrong) at {old[:12]})"]
    assert _verdicts(world, 1, new) == []


@pytest.mark.parametrize("marked, land, hold", [(False, True, False), (True, False, False), (True, True, True),
                                                (True, True, False)],
                         ids=["unmarked", "marked with no land verdict", "marked with a land verdict and a hold",
                              "marked with a land verdict"])
def test_a_merged_in_head_of_another_pr_carries_only_while_it_is_marked_with_a_land_verdict_and_no_hold(world, monkeypatch,
                                                                                                         marked, land, hold):
    """#1 merges #2's head: only marked, with a land verdict and no hold, is it trusted.  Unmarked, #2 is #1's own lane's, so
    #1 carries its change onto `next` (`Queue._carried`) though no one marked it; marked, #2 is below #1 in `next`."""
    h = Review(world, monkeypatch)
    h.d.verdict_carry = "on"
    head, old = world.change({"b.txt": "2\n"}), world.change({"a.txt": "1\n"})
    world.gh.put(2, head, body="Lane `lane-x`.", ready=marked)
    world.gh.put(1, old, body="Lane `lane-x`.")
    assert "no land verdict" in h.tick(), world.logs
    h.review(1)
    if land:
        h.review(2)
    if hold:
        h.review(2, "hold", by="other", ref="b is wrong")
    world.gh.put(1, new := _merge(world, old, head), body="Lane `lane-x`.")
    out = h.tick()
    (line,) = _said(world, new)
    if marked and land and not hold:
        assert out.startswith("landed #1"), out
        assert any(x.startswith("research queue drive: landed #2 #1 on `main`") for x in world.logs), world.logs
        assert line.startswith(f"research queue drive: carried #1's land from {old[:12]} to {new[:12]} (its 1 commit(s) since "
                               f"only merge {head[:12]} as the queue would); written as ci "), line
        assert [lab.by for lab in _verdicts(world, 1, new)] == ["ci"]
    else:
        assert out.startswith("nothing reviewed lands: "), out
        assert line == (f"research queue drive: did not carry #1's land from {old[:12]} to {new[:12]} (untrusted-parent: "
                        f"{new[:12]} merges {head[:12]}, which is neither landed nor another PR's marked head with a land "
                        "verdict and no hold)")
        assert _verdicts(world, 1, new) == []


@pytest.mark.parametrize("where", ["a commit after the merge", "the merge itself"])
def test_a_new_head_that_changes_a_file_beyond_what_it_merges_carries_nothing(world, monkeypatch, where):
    h, old, main = _reviewed(world, monkeypatch, "on")
    merged = _merge(world, old, main)
    if where == "a commit after the merge":
        new = world.change({"a.txt": "2\n"}, base=merged)
        why = f"authored: {new[:12]} changes a.txt"
    else:
        _git(world.dev, "checkout", "-q", "--detach", merged)
        (world.dev / "a.txt").write_text("2\n")
        _git(world.dev, "commit", "-q", "--amend", "--no-edit", "-a")
        new = _git(world.dev, "rev-parse", "HEAD")
        _git(world.dev, "push", "-q", "origin", f"{new}:refs/dev/{new}")
        why = f"edited-merge: {new[:12]} differs from the queue's merge of its parents in a.txt"
    world.gh.put(1, new)
    assert h.tick().startswith(f"nothing reviewed lands: #1 at {new[:8]}: no land verdict"), world.logs
    assert _said(world, new) == [f"research queue drive: did not carry #1's land from {old[:12]} to {new[:12]} ({why})"]
    assert _verdicts(world, 1, new) == []


def test_a_pr_ci_owns_carries_nothing_as_cis_verdict_wouldnt_count(world, monkeypatch):
    h, old, main = _reviewed(world, monkeypatch, "on", body="Lane `ci`.")
    world.gh.put(1, new := _merge(world, old, main), body="Lane `ci`.")
    assert "no land verdict" in h.tick(), world.logs
    assert _said(world, new) == [f"research queue drive: did not carry #1's land from {old[:12]} to {new[:12]} (@ci is among its "
                                 "owners, so its verdict wouldn't count)"]
    assert _verdicts(world, 1, new) == []
