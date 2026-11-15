"""Mark to `main` (`landing.py`) on a small decision log: only the loop's done, non-shadow advances in the window count, a PR
once a head at its first landing; the mark is the one in force when `main` moved, carried across a restack; a PR with none
is left out; each class's p90, median, count and share within 2 h are exact; and a run that can't read an input writes no
measurement.

Nothing is read from the control pod, GitHub, the store or git: `collect` takes stand-ins for each."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.landing import landing as L  # noqa: E402

RUN = "r20000101-000000-aaaa"
STAMP = {"card": "none", "gpus": [], "driver": None, "cuda": None, "node": "test", "commit": "d" * 40}
T0 = L.Q.ts("2026-01-10T00:00:00Z")
START, END = T0 - 86400, T0 + 86400
H1, H2, H6, H6_OLD, H7, H5 = ("1" * 40, "2" * 40, "6" * 40, "e" * 40, "7" * 40, "5" * 40)


def at(hours: float) -> str:
    return L.iso(T0 + hours * 3600)


def advance(hours: float, members: list[int], heads: list[str], **kw: object) -> dict:
    return {"action": "advance", "at": at(hours), "sha": "c" * 40, "base": "b" * 40, "members": members, "heads": heads,
            "done": True, "shadow": False, **kw}


DECISIONS = [
    {"action": "build", "at": at(0), "sha": "c" * 40, "members": [1, 2], "heads": [H1, H2]},
    advance(1, [1, 2], [H1, H2]),
    advance(2, [3], ["3" * 40], shadow=True),          # the shadow's own `main`
    advance(3, [4], ["4" * 40], done=False),           # a push that failed
    advance(-30, [5], [H5]),                           # before the window
    advance(5, [6], [H6]),                             # a head a restack made from H6_OLD
    advance(6, [1], [H1]),                             # the same head again: counted at its first landing
    advance(7, [7], [H7]),                             # marked only after it landed
]


def ready(hours: float, value: bool = True) -> dict:
    return {"key": "ready", "value": "true" if value else "false", "by": "x", "ts": at(hours)}


LABELS = {f"1@{H1}": [ready(-0.5), ready(5.5)],                     # 1.5 h
          f"2@{H2}": [ready(-4), ready(-3, False), ready(-2)],      # withdrawn, then marked again: 3 h
          f"6@{H6_OLD}": [ready(4)], f"6@{H6}": [],                 # carried across the restack: 1 h
          f"7@{H7}": [ready(8)]}
LEAN = {H1: False, H2: True, H6: False, H7: False}


def collect(**kw: object) -> dict:
    seen: list = []

    def labels(pairs: list[tuple[str, str]]) -> dict:
        seen.extend(pairs)
        return {f"{pr}@{h}": LABELS[f"{pr}@{h}"] for pr, h in pairs}

    stand_ins = {"decisions": lambda start: DECISIONS, "restacks": lambda heads: ({H6: H6_OLD}, []), "labels": labels,
                 "lean": lambda repo, base, head: LEAN[head]}
    doc = L.collect(ROOT, START, END, **{**stand_ins, **kw})
    doc["asked"] = seen
    return doc


def test_only_the_loops_advances_in_the_window_count_once_a_head() -> None:
    got = L.landings(DECISIONS, START, END)
    assert [(x["pr"], x["head"], x["landed"]) for x in got] == [
        ("1", H1, T0 + 3600), ("2", H2, T0 + 3600), ("6", H6, T0 + 5 * 3600), ("7", H7, T0 + 7 * 3600)]


def test_the_mark_in_force_when_main_moved_carried_across_a_restack() -> None:
    doc = collect()
    assert doc["errors"] == []
    assert ("6", H6_OLD) in doc["asked"]
    assert [(r["pr"], r["hours"], r["class"]) for r in doc["rows"]] == [
        ("1", 1.5, "plain"), ("2", 3.0, "lean"), ("6", 1.0, "plain"), ("7", None, "plain")]
    assert doc["rows"][3]["mark"] is None


def test_each_class_numbers() -> None:
    s = collect()["summary"]
    assert s["unmarked"] == 1
    assert s["plain"] == {"n": 2, "median_h": 1.25, "p90_h": pytest.approx(1.45), "within_2h": 1.0}
    assert s["lean"] == {"n": 1, "median_h": 3.0, "p90_h": 3.0, "within_2h": 0.0}


def test_the_result_holds_both_metrics_and_their_companions(tmp_path: Path) -> None:
    w = L.kit.Writer(tmp_path, RUN, STAMP)
    assert L.write(w, collect(), tmp_path / "out", 1)
    result = json.loads((tmp_path / L.kit.RESULT).read_text())
    got = {m["name"]: (m["value"], m["unit"], m["verdict"]) for m in result["measurements"]}
    assert got == {
        "ci/plain-mark-to-main-p90": (1.45, "hours", "within target"),
        "ci/plain-mark-to-main-median": (1.25, "hours", None),
        "ci/plain-landed": (2, "PRs", None),
        "ci/plain-within-2h": (1.0, "share of the PRs landed", None),
        "ci/lean-mark-to-main-p90": (3.0, "hours", "over target"),
        "ci/lean-mark-to-main-median": (3.0, "hours", None),
        "ci/lean-landed": (1, "PRs", None),
        "ci/lean-within-2h": (0.0, "share of the PRs landed", None)}
    assert result["validation"]["status"] == "passed"
    assert result["measurement_files"] == ["out/landing.json"]
    assert json.loads((tmp_path / "out" / L.DOC).read_text())["schema"] == L.SCHEMA


def test_a_class_with_no_landing_says_why(tmp_path: Path) -> None:
    w = L.kit.Writer(tmp_path, RUN, STAMP)
    assert L.write(w, collect(lean=lambda repo, base, head: False), tmp_path / "out", 1)
    got = {m["name"]: m for m in json.loads((tmp_path / L.kit.RESULT).read_text())["measurements"]}
    assert got["ci/lean-landed"]["value"] == 0
    assert got["ci/lean-mark-to-main-p90"]["value"] is None
    assert got["ci/lean-mark-to-main-p90"]["why"] == "no PR changing Lean landed in the window"


@pytest.mark.parametrize("broken", ["decisions", "restacks", "labels", "lean"])
def test_an_input_not_read_writes_no_measurement(tmp_path: Path, broken: str) -> None:
    def unread(*args: object) -> None:
        raise L.Unread("the stand-in is down")

    def lost(*args: object) -> None:
        raise RuntimeError("no such object")

    kw = {"decisions": unread, "restacks": lambda heads: ({}, ["GitHub: the commits of 4 heads"]), "labels": unread,
          "lean": lost}
    doc = collect(**{broken: kw[broken]})
    assert doc["errors"] and doc["rows"] == []
    assert not L.write(L.kit.Writer(tmp_path, RUN, STAMP), doc, tmp_path / "out", 1)
    result = json.loads((tmp_path / L.kit.RESULT).read_text())
    assert result["measurements"] == [] and result["validation"]["status"] == "failed"
