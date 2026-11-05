"""`verity_bench.settings`: every environment variable the harness reads declared once (a type, a role, a doc), the point's record
of the value settings, the refusal of a ``VERITY_BENCH_*`` variable nothing reads, and the ratchet on direct environment uses."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from verity_bench import jobs, settings
from verity_bench.cli import main
from verity_bench.settings import check, direct_uses
from verity_config.settings import DEBUG, REDACTED, VALUE, WHERE, DebugOnRecord, Registry, Setting, UnknownSetting

MEMBER = Path(__file__).resolve().parents[1]
REGISTRY_FILE = "verity_bench/settings.py"
#: The files of this member that still use the environment directly, and how often (`direct_uses`).  It only shrinks.
#: runner.py reads the integration's own VERITY_INDUCTOR_DIR_PER_PROCESS, which outside code can read only once
#: ``verity_vllm.settings`` is a public interface (``tests/test_vllm_public_api.py``).
BASELINE = {"verity_bench/vllm/runner.py": 1}


def test_every_setting_has_a_type_a_role_and_a_doc() -> None:
    for name, s in settings.SETTINGS.items():
        assert s.name == name and callable(s.type) and s.role in (VALUE, WHERE, DEBUG) and s.doc.strip(), s
        assert name.startswith(settings.REGISTRY.prefixes) or not name.startswith("VERITY_"), name


def test_a_read_parses_the_environment_at_the_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)
    assert settings.RESEARCH_RUN_ID == "local" and not settings.is_set("RESEARCH_RUN_ID")
    monkeypatch.setenv("RESEARCH_RUN_ID", "r1")
    assert settings.RESEARCH_RUN_ID == "r1" == settings.get("RESEARCH_RUN_ID") and settings.raw("RESEARCH_RUN_ID") == "r1"
    assert settings.get("RESEARCH_RUN_ID", {"RESEARCH_RUN_ID": ""}) == "local"
    with pytest.raises(AttributeError):
        _ = settings.VERITY_BENCH_NOT_A_SETTING
    with pytest.raises(AttributeError, match="not a declared setting"):
        settings.put("VERITY_BENCH_NOT_A_SETTING", "1")
    numbers = Registry("x.py", ("X_",), Setting("X_N", int, 3, VALUE, "a number"), shared=False)
    assert numbers.get("X_N", {}) == 3 and numbers.get("X_N", {"X_N": "4"}) == 4
    with pytest.raises(ValueError, match="X_N='four'"):
        numbers.get("X_N", {"X_N": "four"})
    with pytest.raises(ValueError, match="declared twice"):
        Registry("x.py", ("X_",), *[Setting("X_N", int, 3, VALUE, "a")] * 2, shared=False)


def test_put_setdefault_and_unset_write_only_declared_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERITY_BENCH_VENV", raising=False)
    assert settings.setdefault("VERITY_BENCH_VENV", "/v") == "/v" == settings.VERITY_BENCH_VENV
    settings.put("VERITY_BENCH_VENV", "/w")
    env = settings.environ()
    env["VERITY_BENCH_VENV"] = "/x"
    assert settings.VERITY_BENCH_VENV == "/w"
    settings.unset("VERITY_BENCH_VENV")
    assert settings.VERITY_BENCH_VENV is None


def test_the_record_carries_the_value_settings_in_force() -> None:
    env = {"VERITY_BENCH_SETUP": "benchmarks/harness/setup/pearl_c.sh", "RESEARCH_SOURCE_SHA": "a" * 40,
           "VERITY_BENCH_VENV": "/v", "RESEARCH_RUN_ID": "r", "PATH": "/bin"}
    rec = settings.record(env)
    assert rec["schema"] == "verity-config/settings/v1"
    assert rec["values"] == {"RESEARCH_SOURCE_SHA": "a" * 40, "VERITY_BENCH_SETUP": "benchmarks/harness/setup/pearl_c.sh"}
    assert rec["digest"] == ("8f3621ccf6b149240eb7d51f56bc8150b8aaa1c2eb1edb70dbfbd678417f3c6b"
                             "da0c4cde48e9f6b5600736fe733aa02bf35754807ec6f56335e92f97a7a3b691")
    assert settings.record({})["values"] == {}
    keyed = Registry("x.py", ("X_",), Setting("X_API_KEY", str, None, VALUE, "a credential"), shared=False)
    assert keyed.record({"X_API_KEY": "dummy-not-a-real-key"})["values"] == {"X_API_KEY": REDACTED}


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


def test_check_refuses_a_value_that_doesnt_parse_and_a_debug_setting_on_a_run_of_record() -> None:
    r = Registry("x.py", ("X_",), Setting("X_TRACE", str, None, DEBUG, "a trace file"),
                 Setting("X_N", int, 3, VALUE, "a number"), shared=False)
    check(r, {"X_TRACE": "/t"}, of_record=False)
    check(r, {"X_TRACE": ""}, of_record=True)
    with pytest.raises(DebugOnRecord, match="X_TRACE: debug setting, refused on a run of record"):
        check(r, {"X_TRACE": "/t"}, of_record=True)
    with pytest.raises(ValueError, match="X_N='x'"):
        check(r, {"X_N": "x"}, of_record=False)


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


def test_direct_uses_sees_each_form() -> None:
    src = ("import os\nimport os as o\nfrom os import environ, getenv as g\n"
           "a = os.environ.get('A')\nb = o.environ['B']\nc = environ.get('C')\nd = g('D')\nos.environ['E'] = '1'\n"
           "e = {**os.environ}\nf = os.getpid()\nx.os.environ\n")
    assert direct_uses(src) == [(4, "os.environ"), (5, "os.environ"), (6, "os.environ"), (7, "os.getenv"), (8, "os.environ"),
                                (9, "os.environ")]


def test_direct_environment_uses_only_shrink() -> None:
    found = {}
    for p in sorted(MEMBER.rglob("*.py")):
        rel = str(p.relative_to(MEMBER))
        if "__pycache__" not in p.parts and rel != REGISTRY_FILE and (uses := direct_uses(p.read_text())):
            found[rel] = len(uses)
    grown = {f: n for f, n in found.items() if n > BASELINE.get(f, 0)}
    assert not grown, (f"read and write the environment through {REGISTRY_FILE} (declare the variable there; a vLLM variable "
                       f"through verity_vllm.engine.env), not os.environ: {grown}")
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
    monkeypatch.setattr(cli, "run_benchmark", lambda *a, **k: {"gates": []})
    monkeypatch.setattr(cli, "write", lambda doc, path: path.write_text(json.dumps(doc)))
    monkeypatch.setattr(cli, "render", lambda doc: "")
    assert cli.main(["_vllm-point", "--spec", "llama31-8b-fp8-1gpu", "--point", item.point.id, "--identity", item.identity]) == 0
    record = json.loads((tmp_path / "out" / "settings.json").read_text())
    assert record == settings.record() and record["values"] == {"RESEARCH_SOURCE_SHA": "a" * 40}
    envelope = json.loads((tmp_path / "result.json").read_text())
    assert envelope["run_id"] == "r20261027-000000-0000" and "out/settings.json" in envelope["artifacts"]
