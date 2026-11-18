"""`a5.warm`: the longest an earlier warm measured first, no compose started past the memory bound by earlier peaks, each
template's row handed on as it finishes with its place, and what it measured kept in the cache for the next warm."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "benchmarks" / "one_stage"))
import a5 as A5  # noqa: E402

#: the stand-in compose's log: one line as each compose starts, naming the composes then running
RUNNING = "running"


def _compose(t: str) -> dict:
    """`a5._warm_one`'s stand-in: notes the composes running beside it (one file each under the cache's ``running``) as it
    starts, and its own peak as the cache's earlier measurement says."""
    cache = Path(A5._WARM["a"].compose_cache)
    (run := cache / RUNNING).mkdir(exist_ok=True)
    (run / t).touch()
    with open(cache / "starts.jsonl", "a") as f:
        f.write(json.dumps(sorted(p.name for p in run.iterdir())) + "\n")
    (run / t).unlink()
    return {"template": t, "seconds": float(len(t)), "kept": False, "pin": t, "k_log": 3, "peak_gib": 1.0}


def _window(templates: list[str]) -> tuple[dict, SimpleNamespace]:
    def compose(out, partition, program):
        raise AssertionError("the stand-in composes")
    compose.inputs = ["unit/window"]
    sets = {t: (compose, None) for t in templates}
    mine = SimpleNamespace(members=[SimpleNamespace(template=t) for t in templates], partition_digest="pd",
                           partition={"program": "pg"})
    return sets, mine


def test_the_longest_measured_go_first_and_each_row_is_handed_on_as_it_finishes(tmp_path, monkeypatch):
    """With an earlier warm's seconds in the cache, an unmeasured template goes first, then the longest; ``done`` gets
    each row as it finishes with ``n`` of ``of``, and the cache keeps this warm's seconds and peaks."""
    monkeypatch.setattr(A5, "_warm_one", _compose)
    (tmp_path / A5.WARM_SECONDS).write_text(json.dumps({"short": {"seconds": 5}, "long": {"seconds": 600}}))
    seen: list[dict] = []
    got = A5.warm(SimpleNamespace(compose_cache=str(tmp_path)), *_window(["short", "long", "new"]), done=seen.append)
    assert [(r["template"], r["n"], r["of"]) for r in seen] == [("new", 1, 3), ("long", 2, 3), ("short", 3, 3)]
    assert got == seen
    assert json.loads((tmp_path / A5.WARM_SECONDS).read_text()) == {
        t: {"seconds": float(len(t)), "peak_gib": 1.0, "bytes": None} for t in ("short", "long", "new")}


def test_a_template_the_cache_held_keeps_its_measured_seconds(tmp_path, monkeypatch):
    """A warm that finds a template kept takes no time over it, and leaves the seconds its compose was measured at for the
    next warm's order."""
    monkeypatch.setattr(A5, "_warm_one", lambda t: {**_compose(t), "seconds": 0.2, "kept": True})
    (tmp_path / A5.WARM_SECONDS).write_text(json.dumps({"long": {"seconds": 600, "peak_gib": 9}}))
    A5.warm(SimpleNamespace(compose_cache=str(tmp_path)), *_window(["long"]))
    assert json.loads((tmp_path / A5.WARM_SECONDS).read_text()) == {"long": {"seconds": 600, "peak_gib": 9}}


def test_no_compose_starts_past_the_memory_bound_by_earlier_peaks(tmp_path, monkeypatch):
    """Two composes an earlier warm measured at 20 GiB each never run together under a 30 GiB bound, though there are
    jobs for both; the small ones run beside either. One past the bound alone still runs, alone."""
    monkeypatch.setattr(A5, "_warm_one", _compose)
    (tmp_path / A5.WARM_SECONDS).write_text(json.dumps({
        "big1": {"seconds": 900, "peak_gib": 20}, "big2": {"seconds": 800, "peak_gib": 20},
        "huge": {"seconds": 1000, "peak_gib": 64}, **{f"s{i}": {"seconds": 1, "peak_gib": 1} for i in range(4)}}))
    seen: list[dict] = []
    A5.warm(SimpleNamespace(compose_cache=str(tmp_path)), *_window(["big1", "big2", "huge", "s0", "s1", "s2", "s3"]),
            jobs=3, mem_gib=30, done=seen.append)
    assert sorted(r["template"] for r in seen) == ["big1", "big2", "huge", "s0", "s1", "s2", "s3"]
    assert [(r["n"], r["of"]) for r in seen] == [(n, 7) for n in range(1, 8)]
    starts = [json.loads(x) for x in (tmp_path / "starts.jsonl").read_text().splitlines()]
    assert not any({"big1", "big2"} <= set(s) for s in starts), starts
    assert all(s == ["huge"] for s in starts if "huge" in s), starts


def _free(monkeypatch, gib: float) -> None:
    monkeypatch.setattr(A5.shutil, "disk_usage", lambda p: SimpleNamespace(free=int(gib * 2**30)))


def test_a_warm_under_its_free_space_floor_starts_nothing_and_says_so(tmp_path, monkeypatch):
    """With less free than ``min_free_gib`` on the cache's file system, no compose starts and the refusal names the
    free space, the floor and the templates still to go."""
    monkeypatch.setattr(A5, "_warm_one", lambda t: (_ for _ in ()).throw(AssertionError("composed under the floor")))
    _free(monkeypatch, 40)
    try:
        A5.warm(SimpleNamespace(compose_cache=str(tmp_path)), *_window(["a", "b"]), min_free_gib=50)
    except RuntimeError as e:
        assert "2 of 2 templates to go" in str(e) and "40.0 GiB free and the floor is 50 GiB" in str(e)
    else:
        raise AssertionError("a warm under its floor started")


def test_a_warm_keeping_circuits_refuses_when_what_earlier_warms_measured_would_cross_the_floor(tmp_path, monkeypatch):
    """Keeping circuits, the floor counts the bytes earlier warms measured for the templates to come, and the refusal says
    the projected total; keeping records alone, the same free space passes."""
    monkeypatch.setattr(A5, "_warm_one", _compose)
    (tmp_path / A5.WARM_SECONDS).write_text(json.dumps({"a": {"seconds": 1, "bytes": 60 * 2**30},
                                                        "b": {"seconds": 1, "bytes": 30 * 2**30}}))
    _free(monkeypatch, 120)
    try:
        A5.warm(SimpleNamespace(compose_cache=str(tmp_path), keep_circuits=True), *_window(["a", "b"]), min_free_gib=50)
    except RuntimeError as e:
        assert "take 90.0 GiB more (by earlier warms), 90.0 GiB in all" in str(e)
    else:
        raise AssertionError("a warm keeping circuits past its floor started")
    got = A5.warm(SimpleNamespace(compose_cache=str(tmp_path)), *_window(["a", "b"]), min_free_gib=50)
    assert sorted(r["template"] for r in got) == ["a", "b"]
