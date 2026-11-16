"""`kit.report`, the one way `verity bench compute`, `verity bench memory` and `verity bench inference_only` print and
write their number: the text, then one line of JSON whose first keys are value, unit, commit and run; and, where there
is a place to write (a command's `--out`, else `$RESEARCH_RUN_DIR/out` under a research run, else `$OUT`; an empty one
is none), that line as `<name>.json` there and the measurements in the run's result, with nothing written where there
is none. compute and memory run on their own suites' fixture records and stand-in stores, inference_only on its
directory's stated policy; each prints the same text wherever it writes."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from verity_bench import kit

BENCHMARKS = (Path(__file__).resolve().parents[3] / "benchmarks")
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


COMPUTE = loaded("compute_suite", BENCHMARKS.parent / "tests" / "benchmarks" / "compute" / "test_compute.py",
                 run=loaded("compute_run", BENCHMARKS / "compute" / "run.py"))
MEMORY = loaded("memory_suite", BENCHMARKS.parent / "tests" / "benchmarks" / "memory" / "test_memory.py")


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
    store.add(BENCH, {"out/verifier.json": MEMORY.a_record(MEMORY.with_amounts(MEMORY.a_session()))})
    monkeypatch.setenv("RESEARCH_RUN_ID", BENCH)
    return MEMORY.bench(store, "--from", BENCH)


def inference_only(tmp_path: Path, monkeypatch, *argv: str) -> int:
    """The command as a person runs it, `python benchmarks/inference_only/run.py --stated-only`, on its directory's
    stated policy (`served.toml`): its modules import their siblings by bare name, so it runs in a process of its own,
    and its host stamp is its own."""
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", "0" * 40)
    got = subprocess.run([sys.executable, str(BENCHMARKS / "inference_only" / "run.py"), "--stated-only", *argv],
                         capture_output=True, text=True)
    sys.stdout.write(got.stdout)
    return got.returncode


COMMANDS = {"compute": compute, "memory": memory, "inference_only": inference_only}
#: the mark each command's measurements carry: inference_only's number is priced from a stated policy
MARK = {"compute": "measured", "memory": "measured", "inference_only": "modeled"}
#: the commands whose JSON's run is the research run around them, not a recorded run they read
OWN_RUN = {"inference_only"}


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


def printed(capsys) -> tuple[list[str], dict, str]:
    """What a command printed: its text lines, its JSON document and that document's line."""
    lines = capsys.readouterr().out.splitlines()
    return lines[:-1], json.loads(lines[-1]), lines[-1]


@pytest.mark.parametrize("name", COMMANDS)
def test_each_command_prints_one_line_of_json_and_writes_it_and_its_result_only_where_it_has_a_place(
        name, place, tmp_path, monkeypatch, capsys):
    run = COMMANDS[name]
    assert run(tmp_path / "alone", monkeypatch) == 0
    text, doc, _ = printed(capsys)
    assert list(doc)[:4] == ["value", "unit", "commit", "run"] and isinstance(doc["value"], (int, float))
    assert held(place) == [] and held(tmp_path / "cwd") == []

    monkeypatch.setenv("OUT", str(place / "out"))
    assert run(tmp_path / "out", monkeypatch) == 0
    assert printed(capsys)[:2] == (text, doc)
    assert held(place) == sorted(["out/result.json", f"out/{name}.json"])
    assert json.loads((place / "out" / f"{name}.json").read_text()) == doc
    outside = json.loads((place / "out" / "result.json").read_text())

    monkeypatch.setenv("RESEARCH_RUN_DIR", str(place / "run"))
    monkeypatch.setenv("OUT", str(place / "run" / "out"))
    monkeypatch.setenv("RESEARCH_RUN_ID", BENCH)
    assert run(tmp_path / "run", monkeypatch) == 0
    got, under, line = printed(capsys)
    assert (got, under) == (text, doc | ({"run": BENCH} if name in OWN_RUN else {}))
    assert held(place / "run") == [f"out/{name}.json", "result.json"]
    assert (place / "run" / "out" / f"{name}.json").read_text() == line + "\n"
    result = json.loads((place / "run" / "result.json").read_text())
    assert result["run_id"] == BENCH
    measured = result["measurements"]
    assert [m["name"] for m in measured] == [m["name"] for m in outside["measurements"]] and measured
    # a made-up record's measurements (compute's fixture) are figures no run measured, whatever the command's mark
    mark = kit.kit_report.MADE_UP_MARK if doc.get("made_up") else MARK[name]
    assert all((m["run"], m["mark"]) == (BENCH, mark) for m in measured)
    assert all(m["host"] == STAMP for m in measured) or name == "inference_only"
    assert held(tmp_path / "cwd") == []


def test_an_empty_out_names_no_place_for_inference_only_either(place, tmp_path, monkeypatch, capsys):
    """Its metric's command passes `--out "$OUT"`, so an empty `OUT` reaches it as an empty `--out`."""
    monkeypatch.setenv("OUT", "")
    monkeypatch.setenv("RESEARCH_RUN_DIR", "")
    assert inference_only(tmp_path, monkeypatch, "--out", "") == 0
    assert list(printed(capsys)[1])[:4] == ["value", "unit", "commit", "run"]
    assert held(tmp_path / "cwd") == [] and held(place) == []


def test_report_prints_the_text_then_the_line_and_writes_only_what_it_is_given(place, monkeypatch, capsys):
    doc = {"value": 0.5, "unit": "share", "commit": "c" * 40, "run": "r"}
    m = {"name": "a/metric", "value": 0.5, "unit": "share"}
    assert kit.report("x", "x: 0.5", doc, [m]) == 0
    assert capsys.readouterr().out == f"x: 0.5\n{json.dumps(doc)}\n" and held(place) == []
    monkeypatch.setenv("OUT", str(place))
    assert kit.report("x", "x: 0.5", doc) == 0 and held(place) == ["x.json"]
    assert kit.report("y", "y: 0.5", doc, [m], document=False) == 0 and held(place) == ["result.json", "x.json"]
    assert [w["name"] for w in json.loads((place / "result.json").read_text())["measurements"]] == ["a/metric"]
    # a place the caller names, its --out, is the place, whatever the environment says
    assert kit.report("z", "z: 0.5", doc, out=place / "named") == 0
    assert held(place / "named") == ["z.json"]


@pytest.mark.parametrize("empty", [("OUT",), ("RESEARCH_RUN_DIR",), ("OUT", "RESEARCH_RUN_DIR")])
def test_an_empty_variable_names_no_place(empty, place, tmp_path, monkeypatch, capsys):
    """An empty `OUT` or `RESEARCH_RUN_DIR` is no place, as unset: nothing goes to the working directory."""
    for name in empty:
        monkeypatch.setenv(name, "")
    m = {"name": "a/metric", "value": 0.5, "unit": "share"}
    assert kit.report("x", "x: 0.5", {"value": 0.5}, [m]) == 0
    assert held(tmp_path / "cwd") == [] and held(place) == []
