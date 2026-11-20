"""`verity_bench.settings`: every environment variable the harness reads declared once (a type, a role, a doc), the point's record
of the value settings, the refusal of a ``VERITY_BENCH_*`` variable nothing reads, and the ratchet on direct environment uses."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
from bench_standins import on_stand_in_machine
from verity_bench import jobs, settings
from verity_bench.cli import main
from verity_config.settings import UnknownSetting, direct_uses

MEMBER = (Path(__file__).resolve().parents[3] / "benchmarks" / "harness")
REGISTRY_FILE = "verity_bench/settings.py"
#: The files of this member that still use the environment directly, and how often (`direct_uses`).  It only shrinks.
BASELINE: dict[str, int] = {}


def test_the_variables_and_the_result_form_the_harness_shares_with_research_are_researchs() -> None:
    from research import result
    from verity_bench import kit

    assert {result.ENV_RUN_ID, result.ENV_RUN_DIR, result.ENV_SOURCE_SHA} <= settings.SETTINGS.keys()
    assert kit.RESULT_SCHEMA == result.RESULT_SCHEMA


def test_an_unknown_verity_bench_variable_is_refused_by_name() -> None:
    assert settings.unknown({"VERITY_BENCH_VENV": "/v", "VERITY_BENCH_FULL": "1", "VERITY_RETAIN": "none", "PATH": "/b"}) == []
    with pytest.raises(UnknownSetting, match="unknown setting VERITY_BENCH_VNEV: nothing in the tree reads it"):
        settings.validate({"VERITY_BENCH_VNEV": "/v"}, of_record=False)


def test_verity_bench_refuses_an_unknown_setting_at_startup(monkeypatch: pytest.MonkeyPatch,
                                                           capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setenv("VERITY_BENCH_VNEV", "/v")
    with pytest.raises(SystemExit) as e:
        main(["plan", "llama31-8b-fp8-1gpu", "--local-only"])
    assert e.value.code == 2 and "unknown setting VERITY_BENCH_VNEV" in capsys.readouterr().err


def test_every_prefixed_name_the_member_spells_is_declared() -> None:
    """Python and shell alike: a script sets what the harness reads, and the harness forwards what a script reads."""
    names: dict[str, set[str]] = {}
    for p in sorted(MEMBER.rglob("*")):
        rel = p.relative_to(MEMBER)
        if p.suffix in (".py", ".sh", ".md") and not {"__pycache__", "tests"} & set(rel.parts):
            for n in re.findall(r"\bVERITY_BENCH_[A-Z0-9_]*[A-Z0-9]\b", p.read_text()):
                names.setdefault(n, set()).add(str(rel))
    bad = {n: sorted(f) for n, f in names.items() if not settings.REGISTRY.known(n)}
    assert not bad, f"declare it in {REGISTRY_FILE}, or name it as read elsewhere: {bad}"
    assert {"VERITY_BENCH_VENV", "VERITY_BENCH_SETUP", "VERITY_BENCH_LIB_CACHE"} <= set(names)


def test_direct_environment_uses_only_shrink() -> None:
    found = {}
    for p in sorted(MEMBER.rglob("*.py")):
        rel = str(p.relative_to(MEMBER))
        if "__pycache__" not in p.parts and rel != REGISTRY_FILE and (uses := direct_uses(p.read_text())):
            found[rel] = len(uses)
    grown = {f: n for f, n in found.items() if n > BASELINE.get(f, 0)}
    assert not grown, (f"read and write the environment through {REGISTRY_FILE} (declare the variable there; a vLLM variable "
                       f"through verity_vllm.engine.env's apply_target), not os.environ: {grown}")
    shrunk = {f: found.get(f, 0) for f, n in BASELINE.items() if found.get(f, 0) < n}
    assert not shrunk, f"fewer direct uses than BASELINE allows: lower it in tests/test_settings.py to {shrunk}"


def test_a_point_writes_its_settings_record_beside_its_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from verity_bench import cli

    root = Path(__file__).resolve().parents[3]
    plan = jobs.make_plan("llama31-8b-fp8-1gpu", root=root, commit="a" * 40, lookup=lambda identity, commit: None)
    item = plan.points[0]
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", "a" * 40)
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261027-000000-0000")
    monkeypatch.delenv("OUT", raising=False)
    monkeypatch.chdir(root)
    on_stand_in_machine(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "run_benchmark", lambda *a, **k: {"gates": [], "points": []})
    monkeypatch.setattr(cli, "write", lambda doc, path: path.write_text(json.dumps(doc)))
    monkeypatch.setattr(cli, "render", lambda doc: "")
    card = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    monkeypatch.setattr(cli.kit, "host_stamp", lambda: {"commit": "abc", "card": card})
    assert cli.main(["_vllm-point", "--spec", "llama31-8b-fp8-1gpu", "--point", item.point.id, "--identity", item.identity]) == 0
    record = json.loads((tmp_path / "out" / "settings.json").read_text())
    assert record == settings.record() and record["values"] == {"RESEARCH_SOURCE_SHA": "a" * 40}
    envelope = json.loads((tmp_path / "result.json").read_text())
    assert envelope["run_id"] == "r20261027-000000-0000" and "out/settings.json" in envelope["artifacts"]


@pytest.mark.parametrize("steady, says", [("spec", "is a steady-state spec"), ("point", "is a steady-state point")])
def test_a_point_refuses_a_steady_state_spec_or_point(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steady: str,
                                                      says: str) -> None:
    """`_vllm-point` runs a vLLM spec's points: a steady-state spec's plan, or a steady-state point in a plan, is
    `steady_plan`'s, and is refused with TypeError before anything runs."""
    from verity_bench import cli, steady_plan

    root = Path(__file__).resolve().parents[3]
    held = steady_plan.make_plan("glm53-flock-steady-head-block", root=root,
                                 lookup=lambda spec, item, identity: (None, None))
    plan = held
    if steady == "point":
        served = jobs.make_plan("llama31-8b-fp8-1gpu", root=root, commit="a" * 40, lookup=lambda identity, commit: None)
        plan = replace(served, points=(replace(served.points[0], point=held.points[0].point),))
    item = plan.points[0]
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("RESEARCH_SOURCE_SHA", "a" * 40)
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261027-000000-0000")
    monkeypatch.chdir(root)
    monkeypatch.setattr(jobs, "make_plan", lambda *a, **k: plan)
    monkeypatch.setattr(cli, "run_benchmark", lambda *a, **k: pytest.fail("the point ran"))
    with pytest.raises(TypeError, match=says):
        cli.main(["_vllm-point", "--spec", "s", "--point", item.point.id, "--identity", item.identity])
