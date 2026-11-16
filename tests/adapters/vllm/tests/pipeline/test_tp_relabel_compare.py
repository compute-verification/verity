"""`verity_vllm.pipeline.tp.relabel_build` where a rank's directory and its relabel differ otherwise than in one value of one
file: a subdirectory on both sides, a value of another kind, a key one side lacks, a list one side cuts short and a key that holds
a slash.  Then a Build in which no request of another rank can come from the template's: every derive runs as without the
relabel, and no relabel check is made.  CPU-only: the directories are written by hand, and the row records what it is asked
to derive."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verity_vllm.pipeline.tp import relabel_build as RB

REQ = "build_request_LP4_T2"


def _dirs(tmp_path: Path, real: dict[str, Any], relabel: dict[str, Any]) -> tuple[str, str]:
    """A rank's directory and its relabel, each from ``{relative path: bytes, or a value written as JSON}``."""
    for side, files in (("real", real), ("relabel", relabel)):
        (tmp_path / side).mkdir(exist_ok=True)
        for name, body in files.items():
            (tmp_path / side / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / side / name).write_bytes(body if isinstance(body, bytes) else json.dumps(body).encode())
    return str(tmp_path / "real"), str(tmp_path / "relabel")


def _one(tmp_path: Path, name: str, real: Any, relabel: Any) -> dict | None:
    return RB.compare_dirs(*_dirs(tmp_path, {name: real}, {name: relabel}))


# ---------------------------------------------------------------------------------------------------------------- subdirectories


def test_a_subdirectory_on_both_sides_is_compared_file_by_file_and_equal_ones_are_no_difference(tmp_path):
    files = {"sub/x.json": {"k": 1}, "sub/deep/w.bin": b"abc", "top.bin": b"same"}
    assert RB.compare_dirs(*_dirs(tmp_path, files, files)) is None


def test_a_difference_inside_a_subdirectory_is_named_by_its_path_from_the_ranks_directory(tmp_path):
    real = {"sub/x.json": {"k": 1}, "sub/deep/w.bin": b"abc", "top.bin": b"same"}
    assert RB.compare_dirs(*_dirs(tmp_path, real, {**real, "sub/deep/w.bin": b"abd"})) == {
        "file": "sub/deep/w.bin", "offset": 2, "real": "abc", "relabel": "abd", "files": ["sub/deep/w.bin"]}


def test_a_subdirectorys_difference_comes_in_its_names_place_among_the_files_and_before_the_summaries(tmp_path):
    real = {"a.bin": b"same", "sub/x.json": {"k": [1, 2]}, "zz.bin": b"one", "result.json": {"ok": True}, "artifact.json": {"n": 1}}
    relabel = {"a.bin": b"same", "sub/x.json": {"k": [1, 3]}, "zz.bin": b"two", "result.json": {"ok": False}, "artifact.json": {"n": 2}}
    assert RB.compare_dirs(*_dirs(tmp_path, real, relabel)) == {
        "file": "sub/x.json", "path": "/k/1", "real": "2", "relabel": "3", "files": ["sub/x.json", "zz.bin", "artifact.json", "result.json"]}


def test_a_file_one_side_lacks_inside_a_subdirectory_is_named_with_the_side_that_has_it(tmp_path):
    real = {"sub/x.json": {"k": 1}}
    assert RB.compare_dirs(*_dirs(tmp_path, real, {**real, "sub/more.bin": b"m"})) == {
        "file": "sub/more.bin", "only_in": "relabel", "files": ["sub/more.bin"]}


@pytest.mark.parametrize("side", ["real", "relabel"])
def test_a_subdirectory_one_side_lacks_is_named_whole_with_the_side_that_has_it(tmp_path, side):
    both, more = {"x.json": {"k": 1}}, {"x.json": {"k": 1}, "sub/w.bin": b"w", "sub/deep/v.bin": b"v"}
    real, relabel = (more, both) if side == "real" else (both, more)
    assert RB.compare_dirs(*_dirs(tmp_path, real, relabel)) == {"file": "sub", "only_in": side, "files": ["sub"]}


# --------------------------------------------------------------------------------------------------- the first differing JSON path


@pytest.mark.parametrize("real, relabel, path, said", [
    ({"k": [1]}, {"k": {"0": 1}}, "/k", ("[1]", '{"0": 1}')),
    ([1], {"a": 1}, "/", ("[1]", '{"a": 1}')),
    ({"n": None}, {"n": 0}, "/n", ("null", "0")),
    ({"t": {"n": [0, {"v": 1.5}]}}, {"t": {"n": [0, {"v": "1.5x"}]}}, "/t/n/1/v", ("1.5", "1.5x")),
    ({"a": True, "b": 1}, {"a": 1, "b": 2}, "/a", ("true", "1")),
], ids=["a list and a mapping", "the whole file", "null and a number", "a number and text, nested", "true and 1 before a later difference"])
def test_a_value_of_another_kind_is_the_first_difference_at_its_own_path(tmp_path, real, relabel, path, said):
    """The path of the value itself, not of anything inside it; the whole file's is ``/``."""
    assert _one(tmp_path, "descriptor.json", real, relabel) == {
        "file": "descriptor.json", "path": path, "real": said[0], "relabel": said[1], "files": ["descriptor.json"]}


@pytest.mark.parametrize("name", ["descriptor.json", "result.json"], ids=["a content file", "a file compared without its telemetry"])
@pytest.mark.parametrize("real, relabel, path, said", [
    ({"a": 1, "b": 2}, {"a": 1}, "/b", ("2", "<absent>")),
    ({"a": 1}, {"a": 1, "c": [3]}, "/c", ("<absent>", "[3]")),
    ({"z": 1, "a": 2}, {"a": 3, "y": 1}, "/z", ("1", "<absent>")),
    ({"a": 2}, {"y": 1, "a": 2, "x": 0}, "/y", ("<absent>", "1")),
    ({"t": {"x": {"p": 1}}}, {"t": {"x": {}}}, "/t/x/p", ("1", "<absent>")),
    ({"t": [{"p": 1}, {}]}, {"t": [{"p": 1}, {"q": None}]}, "/t/1/q", ("<absent>", "null")),
], ids=["the relabel lacks a key", "the real one lacks a key", "the real side's keys first, in its order", "then the relabel's own",
        "a key lacked under two others", "a key lacked inside a list"])
def test_a_key_one_side_lacks_is_the_first_difference_and_reads_absent_on_that_side(tmp_path, name, real, relabel, path, said):
    assert _one(tmp_path, name, real, relabel) == {"file": name, "path": path, "real": said[0], "relabel": said[1], "files": [name]}


@pytest.mark.parametrize("real, relabel, path, said", [
    ({"k": [1, 2]}, {"k": [1]}, "/k/1", ("2", "<absent>")),
    ({"k": []}, {"k": [[7]]}, "/k/0", ("<absent>", "[7]")),
], ids=["the relabel's list is cut short", "the real one's is empty"])
def test_a_list_one_side_cuts_short_differs_at_the_first_index_that_side_lacks(tmp_path, real, relabel, path, said):
    assert _one(tmp_path, "tp_links.json", real, relabel) == {
        "file": "tp_links.json", "path": path, "real": said[0], "relabel": said[1], "files": ["tp_links.json"]}


@pytest.mark.parametrize("real, relabel, path, said", [
    ({"shards": {"org/model": [1, 2]}}, {"shards": {"org/model": [1, 3]}}, "/shards/org/model/1", "<absent>"),
    ({"org": 5, "org/model": 1}, {"org": 5, "org/model": 2}, "/org/model", "5"),
], ids=["no key is the part before the slash", "another key is"])
def test_a_difference_under_a_key_that_holds_a_slash_is_named_by_its_path_and_its_values_are_not_the_ones_shown(tmp_path, real, relabel,
                                                                                                             path, said):
    """Today's answer: the path joins the keys with slashes and is read back split at every slash, so both sides show what the
    part before the slash names: nothing, or another key's value.  The difference is still one, in the right file at the right
    path; what should change is that the two differing values are shown."""
    assert _one(tmp_path, "weights.json", real, relabel) == {
        "file": "weights.json", "path": path, "real": said, "relabel": said, "files": ["weights.json"]}


# ------------------------------------------------------------------------------------------ a Build the relabel has nothing to do in


def _row(tmp_path: Path, world: int) -> SimpleNamespace:
    """What `derive` reads of a `TpRow` before it relabels anything: the world, the Build's dir and the row's own derives."""
    row = SimpleNamespace(W=world, bdir=str(tmp_path / "build"), rounds=[])
    row.derive_all = lambda todo, jobs: row.rounds.append((todo, jobs))
    return row


class _Asked(Exception):
    """The row was asked for a step's derive: the relabel has begun."""


def _job(row: SimpleNamespace, rank: int, name: str) -> tuple[Any, ...]:
    return (rank, f"{row.bdir}/rank{rank}/{name}", "--wrapper", name)


@pytest.mark.parametrize("world, jobs", [
    (2, [(0, "build_step"), (1, "build_step")]),
    (2, [(0, "build_step"), (0, REQ)]),
    (4, [(0, REQ), (2, REQ), (3, REQ)]),
    (4, [(1, REQ), (0, "build_request_LP8_T2"), (2, "build_request_LP8_T2")]),
    (2, []),
], ids=["steps alone", "the template's request alone", "a shape the template has no request of", "the template's shape is its alone",
        "nothing to derive"])
def test_a_build_with_no_request_of_another_rank_to_relabel_derives_everything_as_without_the_relabel(tmp_path, world, jobs):
    """One round of the row's derives, the jobs as given and as wide as asked; an earlier Build's relabel check is left as it was."""
    row = _row(tmp_path, world)
    todo = [_job(row, rank, name) for rank, name in jobs]
    (tmp_path / "build" / "relabel_check").mkdir(parents=True)
    (tmp_path / "build" / "relabel_check" / "guard.json").write_text("an earlier Build's")
    assert RB.derive(row, todo, 3) is None
    assert row.rounds == [(todo, 3)]
    assert [p.name for p in (tmp_path / "build").iterdir()] == ["relabel_check"]
    assert (tmp_path / "build" / "relabel_check" / "guard.json").read_text() == "an earlier Build's"


@pytest.mark.parametrize("world, jobs, t", [(2, [(0, REQ), (1, REQ)], 0), (3, [(1, REQ), (2, REQ)], 1), (4, [(1, REQ), (3, REQ)], 1)],
                         ids=["a world of two: rank 0", "a world of three: rank 1", "a world of four: rank 1"])
def test_a_request_shape_the_template_shares_with_another_rank_starts_the_relabel_check_anew(tmp_path, world, jobs, t):
    """The other side of that test: the earlier check is removed and the template's dir made, and the row is asked for the template's
    one more step into it before any derive runs.  The stand-in stops there."""
    row = _row(tmp_path, world)

    def step(rank: int, out: str) -> tuple[Any, ...]:
        raise _Asked(rank, out)

    row.step = step
    (tmp_path / "build" / "relabel_check" / "old").mkdir(parents=True)
    with pytest.raises(_Asked) as asked:
        RB.derive(row, [_job(row, rank, name) for rank, name in jobs], 3)
    assert asked.value.args == (t, f"{row.bdir}/relabel_check/rank{t}/build_step") and row.rounds == []
    assert [p.name for p in (tmp_path / "build" / "relabel_check").iterdir()] == [f"rank{t}"]
