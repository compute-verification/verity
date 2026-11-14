"""A capture map that leaves out a field it may leave out (`verity_vllm.commit.capture_map`): the pinned checkpoint, a tap's
`commit`, the row counts.  Each reader takes the field absent, null and empty alike, as the map naming none."""
from __future__ import annotations

import json
from typing import Any

import pytest

from verity_vllm.commit import capture_map as CM

ABSENT = object()
PROFILE = "0123456789ab"
ENV = {"tp": 1, "L": 2}
STEP: dict[str, Any] = {"profile": PROFILE, "request": "r0", "t": 0, "phase": "decode"}
UNPINNED = [ABSENT, None, {}, {"checkpoint": None}, {"checkpoint": {}}, {"checkpoint": {"revision": "main"}}]
UNBOUND = [ABSENT, None, {}, {"name": "free"}, {"name": "free", "binder": None}]
NO_ROW_COUNTS = [ABSENT, None, {}, {"ratios": None}, {"ratios": {}}, {"positions": "positions"}]


def given(**fields: Any) -> dict[str, Any]:
    """`fields` without the ones held as `ABSENT`."""
    return {k: v for k, v in fields.items() if v is not ABSENT}


def tap(tid: str, **more: Any) -> dict[str, Any]:
    """A model-level tap of the sampler, with `more` of its fields."""
    return given(id=tid, layer="sampler", op=f"sampler.{tid}", port=0, program="p", **more)


def bound(tid: str, binder: str = "moe", **more: Any) -> dict[str, Any]:
    """A tap the Commit binds under its own id as the tensor's name."""
    return tap(tid, commit={"name": tid, "binder": binder}, **more)


def tiny(*taps: dict[str, Any], **more: Any) -> dict[str, Any]:
    return given(schema=CM.SCHEMA, id="tiny", taps=list(taps), layer_sets={}, **more)


def value_id(tid: str) -> str:
    return f"p={PROFILE}/r=r0/t=0/ph=decode/l=sampler/op=sampler.{tid}/port=0"


@pytest.mark.parametrize("pins", UNPINNED)
def test_a_map_that_pins_no_checkpoint_is_no_checkpoints_map(tmp_path, pins):
    """`of_checkpoint` passes over a map with no `pins`, a null or empty one, or one that names no repo, without reading its
    target, and finds the map that pins the repo at the world asked for."""
    (tmp_path / "a.json").write_text(json.dumps(given(schema=CM.SCHEMA, id="unpinned", pins=pins)))
    pinned = {"schema": CM.SCHEMA, "id": "pinned", "pins": {"checkpoint": {"repo": "org/model"}}, "target": {"tensor_parallel_size": 2}}
    (tmp_path / "b.json").write_text(json.dumps(pinned))
    assert CM.of_checkpoint("org/model", 2, tmp_path) == pinned
    assert CM.of_checkpoint("org/model", 1, tmp_path) is None
    assert CM.of_checkpoint("org/other", 2, tmp_path) is None


@pytest.mark.parametrize("commit", UNBOUND)
def test_a_tap_the_commit_does_not_bind_is_no_binders_tap(commit):
    """With `binders`, a step's taps are the ones one of them binds: a tap with no `commit`, a null or empty one, or one that
    names no binder is none of theirs.  Without `binders` it is a tap of the step like any other."""
    cmap = tiny(tap("free", commit=commit), bound("gate"), bound("route", binder="router"))

    def ids(**which: Any) -> list[str]:
        return CM.identities(cmap, ENV, **STEP, **which)

    assert ids() == [value_id("free"), value_id("gate"), value_id("route")]
    assert ids(binders=("moe",)) == [value_id("gate")]
    assert ids(binders=("router", "moe")) == [value_id("gate"), value_id("route")]
    assert ids(binders=()) == []
    assert CM.committer_names(cmap, ENV, binders=("moe",)) == {"gate": (cmap["taps"][1], None)}


@pytest.mark.parametrize("row_counts", NO_ROW_COUNTS)
def test_a_map_without_row_counts_counts_no_rows(row_counts):
    """A map with no `row_counts`, a null or empty one, or one without ratios has no row count: no tap's leading dimension is
    one, a step has none, writes a row under every name, and no committed rows differ; `bind_names` asks for neither the
    step's positions nor its leading dimensions."""
    lead, flat = bound("lead", shape=["n4", 8]), bound("flat")
    cmap = tiny(lead, flat, row_counts=row_counts)
    names = CM.committer_names(cmap, ENV)
    assert CM.row_count_of(cmap, lead) is None and CM.row_count_of(cmap, flat) is None
    assert CM.row_counts(cmap, ENV, [3, 7]) == {}
    assert CM.empty_names(cmap, ENV, [0, 1], names) == set()
    assert CM.rows_differ(cmap, ENV, [3, 7], names, {"lead": 5}) == {}
    assert CM.bind_names(cmap, ENV, ["lead", "stray"], **STEP) == {
        "ids": [value_id("lead")], "unknown": ["stray"], "missing": ["flat"], "empty": [], "rows_differ": {}}


def test_a_map_with_row_counts_holds_a_step_to_them():
    """The same map with a ratio: the tap's leading dimension is the row count, which a step's positions give; a step that
    commits the name commits that many rows, and one whose count is 0 commits nothing under it."""
    lead, flat = bound("lead", shape=["n4", 8]), bound("flat", shape=[4, 8])
    cmap = tiny(lead, flat, row_counts={"positions": "positions", "ratios": {"n4": "4"}})
    names = CM.committer_names(cmap, ENV)
    assert CM.row_count_of(cmap, lead) == "n4" and CM.row_count_of(cmap, flat) is None
    assert CM.row_counts(cmap, ENV, [2, 3, 7]) == {"n4": 2} and CM.row_counts(cmap, ENV, [0, 1]) == {"n4": 0}
    assert CM.empty_names(cmap, ENV, [0, 1], names) == {"lead"} and CM.empty_names(cmap, ENV, [3], names) == set()
    assert CM.rows_differ(cmap, ENV, [3, 7], names, {"lead": 1}) == {"lead": (None, 1, 2)}
    assert CM.rows_differ(cmap, ENV, [3, 7], names, {}) == {"lead": (None, -1, 2)}
    with pytest.raises(CM.MapError, match="needs its positions"):
        CM.bind_names(cmap, ENV, ["lead", "flat"], **STEP)
    with pytest.raises(CM.MapError, match="needs their leading dimensions"):
        CM.bind_names(cmap, ENV, ["lead", "flat"], **STEP, positions=[3, 7])
    both = [value_id("flat"), value_id("lead")]
    assert CM.bind_names(cmap, ENV, ["lead", "flat"], **STEP, positions=[3, 7], rows={"lead": 2}) == {
        "ids": both, "unknown": [], "missing": [], "empty": [], "rows_differ": {}}
    assert CM.bind_names(cmap, ENV, ["lead", "flat"], **STEP, positions=[3, 7], rows={"lead": 1}) == {
        "ids": both, "unknown": [], "missing": [], "empty": [], "rows_differ": {"lead": (None, 1, 2)}}
    assert CM.bind_names(cmap, ENV, ["flat"], **STEP, positions=[0, 1]) == {
        "ids": [value_id("flat")], "unknown": [], "missing": [], "empty": [], "rows_differ": {}}
    assert CM.bind_names(cmap, ENV, ["lead", "flat"], **STEP, positions=[0, 1]) == {
        "ids": both, "unknown": [], "missing": [], "empty": ["lead"], "rows_differ": {}}
