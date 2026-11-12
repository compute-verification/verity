"""`nci-record-check`, the record form (`NciCheck.Record`, `NciCheck.RecordCount`) on one case, and
`verity/protocols/inference_only/verifier/nci/lean/scripts/served_record_case.py`, which writes that case from a
served run's files with the reference's verdicts on it.

A made run's files stand in for a served run's: the record difftest's vectors' served class's shape at two slots and
three positions, its Program, circuit record, registration log, limits and count's limits each written as a file the
way a serve writes them. The first test is that the script reads them back as the vectors' own case, pins
included, and that its verdicts are the reference's. The second is each option's case. The third is that the
reference form-only, which the script runs, gives the whole reference's verdict on every shaped case and walks no
leaf. The fourth runs the compiled command on the script's cases and on a refusal of each of the form's two
expanded rules, and holds its two lines to the reference's verdicts on every field.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import verity.circuits.layout

LEAN = Path(__file__).resolve().parents[4] / "inference_only" / "verifier" / "nci" / "lean"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, LEAN / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vectors = _load("record_difftest_vectors")
script = _load("served_record_case")
ref = script.ref
SERVED = "2x3 registered"
#: a key of a record's `registered` that the form reads nothing of and refuses, as a served record's key it does not
#: know: `--drop` takes it out
MORE = "unknown"
#: the two refusals the reference states as the Lean's rule and then expands leaf by leaf, unless form-only
EXPANDED = ("own-rows", "leaf-widths")
#: the shaped cases the compiled command is run on beside the script's: a refusal of each expanded rule, with its message
REFUSED = ("a leaf that reads sixteen gates through one row", "2x3 one input run handed to two rows")
LINE = re.compile(r"\[\d+ s\] (check|count): (.*)")


@pytest.fixture(scope="module")
def shaped() -> dict[str, dict]:
    return {m["shape"]: m for m in vectors.shaped()}


def _files(case: dict, where: Path) -> list[str]:
    """`case` as a run's files under `where`, and the script's arguments that read them."""
    where.mkdir(parents=True, exist_ok=True)
    inst, tr = case["instance"], case["trusted"]
    (where / "program.json").write_text(inst["program"])
    (where / "circuit.json").write_text(inst["record"])
    logs = []
    for k, rec in enumerate(tr["log"]):
        logs.append(str(where / f"log{k}.json"))
        Path(logs[-1]).write_text(json.dumps(rec))
    (where / "limits.json").write_text(json.dumps({"alpha": inst["alpha"], "gamma": inst["gamma"]}))
    (where / "count.json").write_text(json.dumps(case["count"]))
    return ["--program", str(where / "program.json"), "--record", str(where / "circuit.json"), "--log", *logs,
            "--limits", str(where / "limits.json"), "--count", str(where / "count.json")]


def _more(case: dict) -> dict:
    """`case` with a key `MORE` in its record's `registered`, the run's record committing that text."""
    doc = json.loads(case["instance"]["record"])
    doc["registered"][MORE] = {}
    record = script.circuit_bytes(doc).decode()
    log = [*case["trusted"]["log"][:-1], {**case["trusted"]["log"][-1]}]
    log[-1]["window"] = {**log[-1]["window"], "circuit": vectors.digest_of(record)}
    return {**case, "instance": {**case["instance"], "record": record}, "trusted": {**case["trusted"], "log": log}}


def _written(args: list[str], out: Path, name: str, *more: str) -> tuple[dict, dict]:
    assert script.main([*args, "--out", str(out), "--name", name, *more]) == 0
    return (json.loads((out / f"{name}.case.json").read_text()),
            json.loads((out / f"{name}.reference.json").read_text()))


def test_the_script_reads_a_runs_files_as_the_case_they_make(shaped, tmp_path):
    made = shaped[SERVED]["case"]
    assert "order" not in made["instance"] and made["trusted"]["run"] == ref.reference.record_digest(made["trusted"]["log"][-1])
    case, got = _written(_files(made, tmp_path / "run"), tmp_path / "out", "served")
    assert case == {**made, "spot": None, "rows": []}
    assert case["instance"]["pins"] and case["count"]["table"]
    inst, tr = made["instance"], made["trusted"]
    assert got == {"check": ref.check_detail(inst, tr), "count": ref.count(inst, tr, made["count"])}
    assert got["check"]["complies"] and got["count"]["complies"]
    assert script.main([*_files(made, tmp_path / "run"), "--out", str(tmp_path / "only"), "--name", "n", "--case-only"]) == 0
    assert [p.name for p in (tmp_path / "only").iterdir()] == ["n.case.json"]


def test_each_option_makes_the_case_it_says(shaped, tmp_path):
    """`--unregister` leaves one weight the record reads out of the log's first record, the chain made again, and
    both rules refuse its fresh bits; `--no-pins` gives no pin, so the opening's ports are fresh too; a record with
    a key of `registered` the form refuses is the made case again once `--drop registered.KEY` takes the key out and
    points the run's record at the new text; and a key or a value the files lack is refused by name."""
    made = shaped[SERVED]["case"]
    args = _files(made, tmp_path / "run")
    _, held = _written(args, tmp_path / "out", "served")
    first = {e["value"]: e for e in made["trusted"]["log"][0]["registered"]}
    params = json.loads(made["instance"]["record"])["registered"]["params"]
    smallest = min((n, p) for p, n in params.items() if p in first)[1]
    largest = max((n, p) for p, n in params.items() if p in first)[1]
    for how, value in (("smallest", smallest), ("largest", largest), (largest, largest)):
        case, got = _written(args, tmp_path / "out", f"less-{how.replace('/', '-')}", "--unregister", how)
        log = case["trusted"]["log"]
        assert [e["value"] for e in log[0]["registered"]] == [v for v in first if v != value]
        assert [r["previous"] for r in log] == [None, *map(ref.reference.record_digest, log[:-1])]
        assert case["trusted"]["run"] == ref.reference.record_digest(log[-1])
        assert (got["check"]["failure"], got["count"]["failure"]) == ("nci", "fresh"), how
        assert got["check"]["unregistered"] > held["check"]["unregistered"], how
    case, got = _written(args, tmp_path / "out", "no-pins", "--no-pins")
    assert case["instance"]["pins"] == [] and got["check"]["openings"] > 0 == held["check"]["openings"]
    more = _more(made)
    case, got = _written(_files(more, tmp_path / "more"), tmp_path / "out", "more")
    assert got == {"check": {"error": "inputs"}, "count": {"error": "inputs"}}
    case, got = _written(_files(more, tmp_path / "more"), tmp_path / "out", "less-more", "--drop", f"registered.{MORE}")
    assert case == {**made, "spot": None, "rows": []} and got == held
    inst, log = made["instance"], made["trusted"]["log"]
    limits = {"alpha": inst["alpha"], "gamma": inst["gamma"]}
    with pytest.raises(ValueError, match="registered has no such key"):
        script.served_case(inst["program"], inst["record"], log, limits, drop=("registered.nothing",))
    with pytest.raises(ValueError, match="registers no such value"):
        script.served_case(inst["program"], inst["record"], log, limits, unregister="weights/nothing")


def test_form_only_is_the_whole_references_verdict_and_walks_no_leaf(shaped, monkeypatch):
    """On every shaped case, and on the random cases broken in each way of the two expanded rules (`EXPANDED`) beside
    the first unbroken ones, the reference form-only gives the whole reference's two verdicts, refusals and their
    messages included; and it gives them with every leaf-by-leaf walk (`hands_on`, `lands_whole`, the root's
    gates through `resolve_gate`) made to fail if called."""
    turns = vectors._turns()
    ways = {s: vectors.WAYS[k % len(vectors.WAYS)] for s, k in turns.items()}
    seeds = [s for s, (stage, _) in ways.items() if stage in EXPANDED] + [s for s in range(40) if s not in turns]
    made = {**shaped, **{f"random {s}": vectors.case(s) for s in seeds}}
    assert {ways[s] for s in seeds if s in ways} == {(stage, how) for stage in EXPANDED for how in vectors.BROKEN[stage]}
    whole = {what: (ref.check_detail(m["case"]["instance"], m["case"]["trusted"]),
                    ref.count(m["case"]["instance"], m["case"]["trusted"], m["case"]["count"]))
             for what, m in made.items()}
    assert {c.get("error") for c, _ in whole.values()} >= {None, *EXPANDED}

    def walked(*_):
        raise AssertionError("form-only walked a leaf")
    for name in ("hands_on", "lands_whole"):
        monkeypatch.setattr(ref, name, walked)
    monkeypatch.setattr(verity.circuits.layout, "resolve_gate", walked)
    for what, m in made.items():
        inst, tr = m["case"]["instance"], m["case"]["trusted"]
        got = (ref.check_detail(inst, tr, form_only=True), ref.count(inst, tr, m["case"]["count"], form_only=True))
        assert got == whole[what], what


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_compiled_command_gives_the_references_verdicts(shaped, tmp_path):
    """`lake exe nci-record-check` on the script's cases, as written and with each option, and on the shaped
    refusals of `leaf-widths` and `own-rows`: its two lines are the reference's verdicts on every field."""
    import verity_difftest as D

    exe = D.build(LEAN, "nci-record-check")
    args = _files(shaped[SERVED]["case"], tmp_path / "run")
    cases = {}
    for name, more in (("served", ()), ("less-smallest", ("--unregister", "smallest")), ("no-pins", ("--no-pins",)),
                       ("less-openings", ("--drop", "registered.openings"))):
        _, cases[name] = _written(args, tmp_path / "out", name, *more)
    for k, what in enumerate(REFUSED):
        name = f"refused-{k}"
        (tmp_path / "out" / f"{name}.case.json").write_text(script.line(shaped[what]["case"]))
        cases[name] = script.verdicts(shaped[what]["case"])
    assert {cases[f"refused-{k}"]["check"]["error"] for k in range(len(REFUSED))} == {"leaf-widths", "own-rows"}
    for name, want in cases.items():
        p = subprocess.run([str(exe), str(tmp_path / "out" / f"{name}.case.json")], capture_output=True, text=True)
        assert p.returncode == 0, (name, p.stderr)
        got = dict(m.groups() for m in map(LINE.fullmatch, p.stdout.splitlines()) if m)
        assert {k: json.loads(v) for k, v in got.items()} == want, name
