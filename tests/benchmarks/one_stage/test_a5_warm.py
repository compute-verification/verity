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
        t: {"seconds": float(len(t)), "peak_gib": 1.0} for t in ("short", "long", "new")}


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
    starts = [json.loads(x) for x in (tmp_path / "starts.jsonl").read_text().splitlines()]
    assert not any({"big1", "big2"} <= set(s) for s in starts), starts
    assert all(s == ["huge"] for s in starts if "huge" in s), starts
