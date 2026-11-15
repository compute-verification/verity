"""`kit.report`, the one way `verity bench compute` and `verity bench memory` print and write their number: the text,
then one line of JSON whose first keys are value, unit, commit and run; and, where there is a place to write
(`$RESEARCH_RUN_DIR/out` under a research run, else `$OUT`), that line as `<name>.json` there and the measurements in
the run's result, with nothing written where there is none. Each command is run on its own suite's fixture record and
stand-in store, and prints the same bytes wherever it writes."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from verity_bench import kit

BENCHMARKS = Path(__file__).resolve().parents[2]
BENCH = "r20000101-000009-ffff"
STAMP = {"card": "none", "gpus": [], "driver": None, "cuda": None, "node": "test", "commit": "d" * 40}


def loaded(name: str, path: Path, **beside: ModuleType) -> ModuleType:
    """The module at `path` under `name`, with `beside` importable by their names while it loads."""
    held = {k: sys.modules.get(k) for k in beside}
    sys.modules.update(beside)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for k, v in held.items():
            if v is None:
                sys.modules.pop(k)
            else:
                sys.modules[k] = v


COMPUTE = loaded("compute_suite", BENCHMARKS / "compute" / "tests" / "test_compute.py",
                 run=loaded("compute_run", BENCHMARKS / "compute" / "run.py"))
MEMORY = loaded("memory_suite", BENCHMARKS / "memory" / "tests" / "test_memory.py")


def compute(tmp_path: Path, monkeypatch) -> int:
    store = COMPUTE.FakeStore(tmp_path / "store")
    store.add(COMPUTE.RUN, COMPUTE.a_record(tmp_path))
    return COMPUTE.bench(store, "--from", COMPUTE.RUN)


def memory(tmp_path: Path, monkeypatch) -> int:
    references = tmp_path / "references.json"
    tmp_path.mkdir()
    references.write_text("{}")
    monkeypatch.setattr(MEMORY.M, "REFERENCES", references)
    store = MEMORY.FakeStore(tmp_path / "store")
    store.add(MEMORY.RUN, {"out/verifier.json": MEMORY.a_record(MEMORY.with_amounts(MEMORY.a_session()))})
    return MEMORY.bench(store, "--from", MEMORY.RUN)


COMMANDS = {"compute": compute, "memory": memory}


@pytest.fixture
def place(tmp_path: Path, monkeypatch) -> Path:
    """A directory the place to write is put under, the command's working directory beside it, no research run
    around the command, and the kit's writer stamping with `STAMP`."""
    for name in ("RESEARCH_RUN_DIR", "OUT", "RESEARCH_RUN_ID", "RESEARCH_SOURCE_SHA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(kit, "host_stamp", lambda: dict(STAMP))
    (tmp_path / "cwd").mkdir()
    monkeypatch.chdir(tmp_path / "cwd")
    return tmp_path / "place"


def held(place: Path) -> list[str]:
    return sorted(p.relative_to(place).as_posix() for p in place.rglob("*") if p.is_file()) if place.exists() else []


@pytest.mark.parametrize("name", COMMANDS)
def test_each_command_prints_one_line_of_json_and_writes_it_and_its_result_only_where_it_has_a_place(
        name, place, tmp_path, monkeypatch, capsys):
    run = COMMANDS[name]
    assert run(tmp_path / "alone", monkeypatch) == 0
    alone = capsys.readouterr()
    line = alone.out.splitlines()[-1]
    doc = json.loads(line)
    assert list(doc)[:4] == ["value", "unit", "commit", "run"] and isinstance(doc["value"], float)
    assert held(place) == [] and held(tmp_path / "cwd") == []

    monkeypatch.setenv("OUT", str(place / "out"))
    assert run(tmp_path / "out", monkeypatch) == 0
    assert capsys.readouterr() == alone
    assert held(place) == sorted(["out/result.json", f"out/{name}.json"])
    assert (place / "out" / f"{name}.json").read_text() == line + "\n"
    outside = json.loads((place / "out" / "result.json").read_text())

    monkeypatch.setenv("RESEARCH_RUN_DIR", str(place / "run"))
    monkeypatch.setenv("OUT", str(place / "run" / "out"))
    monkeypatch.setenv("RESEARCH_RUN_ID", BENCH)
    assert run(tmp_path / "run", monkeypatch) == 0
    assert capsys.readouterr() == alone
    assert held(place / "run") == [f"out/{name}.json", "result.json"]
    assert (place / "run" / "out" / f"{name}.json").read_text() == line + "\n"
    result = json.loads((place / "run" / "result.json").read_text())
    assert result["run_id"] == BENCH
    measured = result["measurements"]
    assert [m["name"] for m in measured] == [m["name"] for m in outside["measurements"]]
    assert all((m["run"], m["host"], m["mark"]) == (BENCH, STAMP, "measured") for m in measured)
    assert held(tmp_path / "cwd") == []


def test_report_prints_the_text_then_the_line_and_writes_only_what_it_is_given(place, monkeypatch, capsys):
    doc = {"value": 0.5, "unit": "share", "commit": "c" * 40, "run": "r"}
    m = {"name": "a/metric", "value": 0.5, "unit": "share"}
    assert kit.report("x", "x: 0.5", doc, [m]) is None
    assert capsys.readouterr().out == f"x: 0.5\n{json.dumps(doc)}\n" and held(place) == []
    monkeypatch.setenv("OUT", str(place))
    assert kit.report("x", "x: 0.5", doc) == place and held(place) == ["x.json"]
    assert kit.report("y", "y: 0.5", doc, [m], document=False) == place and held(place) == ["result.json", "x.json"]
    assert [w["name"] for w in json.loads((place / "result.json").read_text())["measurements"]] == ["a/metric"]
