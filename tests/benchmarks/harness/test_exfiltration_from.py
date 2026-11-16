"""`verity bench exfiltration --from DIR`: one run's record directory read in place of the store's runs of record, with
no store, and printed through the kit's report; the made-up record in the tree printed as made up."""

from __future__ import annotations

import json
import runpy
import shutil
import sys

import pytest
from exfiltration_helpers import REPO, YEAR_GB, recorded

from verity_bench import exfiltration as E
from verity_bench import jobs
from verity_bench import kit_report as kit

RUNNER = REPO / "benchmarks" / "exfiltration" / "run.py"
MADE_UP = REPO / "benchmarks" / "exfiltration" / "made_up"


@pytest.fixture
def no_store(monkeypatch):
    """A store the command must not open: opening it fails the test. No place to write is set."""

    def opened(*a, **kw):
        raise AssertionError("--from opened the store")

    monkeypatch.setattr(jobs, "StoreCompletion", opened)
    for name in ("RESEARCH_RUN_DIR", "OUT"):
        monkeypatch.delenv(name, raising=False)


def run(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["run.py", *argv])
    with pytest.raises(SystemExit) as done:
        runpy.run_path(str(RUNNER), run_name="__main__")
    return done.value.code


def test_the_made_up_record_prints_the_tables_as_made_up_with_no_store(monkeypatch, capsys, no_store):
    assert run(monkeypatch, "--from", str(MADE_UP)) == 0
    out = capsys.readouterr().out.rstrip("\n")
    text, line = out.rsplit("\n", 1)
    first, *rest = text.splitlines()
    assert first.startswith("# verity bench exfiltration: ")
    assert rest[:3] == [kit.NOT_STATED, kit.NOT_AN_AUDIT, kit.MADE_UP]
    assert rest[3].startswith("  what the record says of itself: written by hand from the tests' fixture runs")
    assert "- **the compute audit's part: 14,240 GB** (modeled)" in text and "## Inputs" in text
    doc = json.loads(line)
    assert (doc["made_up"], doc["audit"], doc["ran_with"], doc["runs"]) == (True, False, None, ["made_up"])
    assert doc["value"] == pytest.approx(YEAR_GB) and doc["schema"] == E.SCHEMA


def test_with_a_place_to_write_each_figure_is_written_modeled_and_made_up(monkeypatch, capsys, no_store, tmp_path):
    monkeypatch.setenv("OUT", str(tmp_path / "out"))
    assert run(monkeypatch, "--from", str(MADE_UP)) == 0
    capsys.readouterr()
    got = json.loads((tmp_path / "out" / "result.json").read_text())["measurements"]
    metrics = runpy.run_path(str(RUNNER))["METRICS"]
    assert {m["name"] for m in got} == set(metrics)
    assert all(m["mark"] == kit.MADE_UP_MARK and m["made_up"] is True for m in got)
    year = {m["name"]: m for m in got}
    assert (year[next(iter(metrics))]["period"], year[next(iter(metrics))]["over"]) == ("year", "world")
    assert json.loads((tmp_path / "out" / "exfiltration.json").read_text())["made_up"] is True


def test_a_record_directory_gives_the_figures_the_stores_runs_of_record_give(tmp_path):
    """The made-up record holds the fixture's two runs of record in one directory: the same figures, its one run read."""

    mine = E.build(record=MADE_UP, root=REPO)
    theirs = E.build(store=recorded(tmp_path).fresh(), root=REPO)
    assert mine["runs"] == ["made_up"] and mine["refused"] == [] and "stop_class" not in mine
    assert mine["value"] == theirs["value"] == pytest.approx(YEAR_GB)
    assert runless(mine["periods"]) == runless(theirs["periods"])
    assert runless(mine["compute_audit"]) == runless(theirs["compute_audit"])


def runless(x):
    if isinstance(x, dict):
        return {k: runless(v) for k, v in x.items() if k != "run"}
    return [runless(v) for v in x] if isinstance(x, list) else x


def test_a_record_that_says_nothing_of_itself_is_not_called_made_up(monkeypatch, capsys, no_store, tmp_path):
    shutil.copytree(MADE_UP, tmp_path / "r", ignore=shutil.ignore_patterns("job.json"))
    assert run(monkeypatch, "--from", str(tmp_path / "r")) == 0
    text, line = capsys.readouterr().out.rstrip("\n").rsplit("\n", 1)
    assert kit.MADE_UP not in text and "made_up" not in json.loads(line) and json.loads(line)["runs"] == ["r"]


@pytest.mark.parametrize("given, said", [
    ("nowhere", "{at}: no such directory, and not a run id: give --from the directory of a run's record"),
    ("empty", "no run of record holds the planner's rows: empty holds no planner rows (`out/rows.jsonl`, "
              "`out/three/rows.jsonl`) and no plan (`out/mix.json`) (give --from a run's record that does)"),
])
def test_a_from_that_names_no_record_of_rows_is_refused_in_one_line(monkeypatch, capsys, no_store, tmp_path,
                                                                    given, said):
    (tmp_path / "empty").mkdir()
    assert run(monkeypatch, "--from", str(tmp_path / given)) == 2
    err = capsys.readouterr().err
    assert err == f"verity bench exfiltration: {said.format(at=tmp_path / given)}\n"


def test_from_is_the_runners_and_takes_no_json(monkeypatch, capsys, no_store):
    with pytest.raises(SystemExit) as done:
        E.main(["--from", str(MADE_UP)])
    assert done.value.code == 2 and "run `verity bench exfiltration --from DIR`" in capsys.readouterr().err
    assert run(monkeypatch, "--from", str(MADE_UP), "--json") == 2
