"""theory/run.py: it finds every script on the registry, runs each with the flags it takes, and one that fails fails
the run."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

THEORY = Path(__file__).resolve().parents[1] / "theory"


@pytest.fixture(scope="module")
def run():
    spec = importlib.util.spec_from_file_location("theory_run_under_test", THEORY / "run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_checks_are_the_scripts_that_make_a_registry(run):
    """Every script here that makes a `Checks` is found, the runner itself and the scripts that make none left out, and
    the four whose arguments `--quick` shortens take it."""
    found = run.checks()
    made = sorted(p.stem for p in THEORY.glob("*.py") if p.name != "run.py" and "Checks()" in p.read_text(encoding="utf-8"))
    assert found == made and len(found) == 17 and "run" not in found and "kappa_count" not in found
    assert [n for n in found if run.takes_quick(n)] == ["fold_kernel_check", "fold_root_check", "fold_wide32",
                                                        "zero_routes_check"]


def test_a_check_gets_only_the_flags_it_takes(run, tmp_path):
    script = [sys.executable, "-B"]
    assert run.command("nosum", True, False, None) == [*script, str(THEORY / "nosum.py")]
    assert run.command("zero_routes_check", True, True, tmp_path / "z.json") == [
        *script, str(THEORY / "zero_routes_check.py"), "--quick", "--mutations", "--verdict", str(tmp_path / "z.json")]
    assert "--quick" not in run.command("zero_routes_check", False, True, None)


def test_a_check_runs_and_leaves_its_verdict_beside_the_runs(run, tmp_path, capfd):
    """zero_routes_check, the fastest: the run's verdict has one check for it with the script's own summary line, and
    the script's verdict is beside it."""
    code = run.main(["--quick", "--verdict", str(tmp_path / "run.verdict.json"), "zero_routes_check"])
    out = capfd.readouterr().out
    assert code == 0, out[-4000:]
    doc = json.loads((tmp_path / "run.verdict.json").read_text(encoding="utf-8"))
    own = json.loads((tmp_path / "zero_routes_check.verdict.json").read_text(encoding="utf-8"))
    n = own["counts"]["checks"]
    assert own["ok"] and n > 0 and own["args"] == ["--quick"]
    assert doc["ok"] and doc["args"] == ["--quick", "zero_routes_check"]
    assert doc["checks"] == [{"name": "zero_routes_check.py", "ok": True, "detail": f"{n}/{n} checks OK"}]
    assert out.rstrip().endswith("1/1 checks OK")


def test_a_script_that_fails_fails_the_run_and_the_rest_still_run(run, tmp_path, monkeypatch, capfd):
    (tmp_path / "bad.py").write_text(f"# {run.REGISTRY}\nimport sys\nprint('0/1 checks OK  <-- FAILURES')\nsys.exit(1)\n",
                                     encoding="utf-8")
    (tmp_path / "good.py").write_text(f"# {run.REGISTRY}\nprint('1/1 checks OK')\nprint()\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("print('no registry')\n", encoding="utf-8")
    monkeypatch.setattr(run, "HERE", tmp_path)
    code = run.main(["--verdict", str(tmp_path / "v" / "run.verdict.json")])
    capfd.readouterr()
    doc = json.loads((tmp_path / "v" / "run.verdict.json").read_text(encoding="utf-8"))
    assert code == 1 and not doc["ok"]
    assert [(c["name"], c["ok"], c["detail"]) for c in doc["checks"]] == [
        ("bad.py", False, "exit 1: 0/1 checks OK  <-- FAILURES"), ("good.py", True, "1/1 checks OK")]


def test_a_name_that_is_no_check_is_refused(run):
    with pytest.raises(SystemExit, match="no check named kappa_count"):
        run.main(["kappa_count"])
