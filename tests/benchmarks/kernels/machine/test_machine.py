"""machine.py: the CUDA toolkit a kernels runner takes from its machine's record at the version it needs, and its card's
`arch` from the card records, each refused by name; CUDA, where given, wins."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "benchmarks" / "kernels" / "machine.py"
ONE_CARD = ROOT / "tests" / "benchmarks" / "harness" / "fixtures" / "machines" / "one-card.toml"
OTHERS = '[{ directory = "/usr/local/cuda-12.9", version = "12.9" }]'
NAME = "h100-like"


@pytest.fixture
def m() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("machine_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(monkeypatch, run_dir: Path, name: str = NAME, toolkit: str | None = None) -> Path:
    """A run `research run --on` launched onto machine `name`: its copy of the machine's [properties] in inputs/, like
    vy-h100-1's (CUDA 13.0 the toolkit's own, with 12.9 and 13.3 beside it), or with `toolkit`'s value."""
    text = ONE_CARD.read_text()
    assert OTHERS in text
    text = text.replace(OTHERS, '[{ directory = "/usr/local/cuda-12.9", version = "12.9" }, '
                                '{ directory = "/usr/local/cuda-13.3", version = "13.3" }]')
    if toolkit is not None:
        start = text.index("toolkit = { value = ")
        end = text.index(', read = "readlink', start)
        text = text[:start] + f"toolkit = {{ value = {toolkit}" + text[end:]
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "inputs" / f"{NAME}.toml").write_text(text)
    monkeypatch.setenv("RESEARCH_MACHINE", name)
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    return run_dir


def test_the_toolkit_is_the_runs_machines_at_the_version_asked_not_the_first_listed(m, monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path)
    assert [m.toolkit(v, {}) for v in ("13.0", "13.3", "12.9")] == [
        "/usr/local/cuda-13.0", "/usr/local/cuda-13.3", "/usr/local/cuda-12.9"]


def test_cuda_where_given_wins_with_no_machine(m):
    assert m.toolkit("13.3", {"CUDA": "/opt/cuda-13.3"}) == "/opt/cuda-13.3"


def test_a_run_no_launcher_named_a_machine_is_refused_naming_what_to_set(m):
    with pytest.raises(SystemExit, match=r"^no CUDA toolkit 13\.0: set CUDA to its directory, or run on a machine "
                                         r"`research run --on` names: no machine named: RESEARCH_MACHINE is unset"):
        m.toolkit("13.0", {})


def test_a_machine_whose_record_has_no_properties_is_refused_by_name(m, monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, name="vy-h100-1")
    with pytest.raises(SystemExit, match=r"^no CUDA toolkit 13\.0: .*machine 'vy-h100-1': .*inputs/vy-h100-1\.toml is not there"):
        m.toolkit("13.0", {})


def test_an_unread_toolkit_is_refused_naming_the_property_and_the_machine(m, monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, toolkit='"unread"')
    with pytest.raises(SystemExit, match=rf"^machine '{NAME}': its \[properties\] have toolkit unread: read it there, or set CUDA"):
        m.toolkit("13.0", {})


@pytest.mark.parametrize(("toolkit", "lists"), [(None, "13.0, 12.9, 13.3"), ('"none"', "none")])
def test_a_version_the_machine_does_not_list_is_named(m, monkeypatch, tmp_path, toolkit, lists):
    _run(monkeypatch, tmp_path, toolkit=toolkit)
    with pytest.raises(m.NotListed, match=rf"list no CUDA toolkit 13\.1 under toolkit \(they list {lists}\): set CUDA"):
        m.toolkit("13.1", {})


def test_the_command_prints_the_directory_and_exits_3_on_a_version_not_listed(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path)
    env = {**os.environ, "CUDA": ""}
    run = lambda *a: subprocess.run([sys.executable, str(SCRIPT), *a], env=env, capture_output=True, text=True)  # noqa: E731
    ok, absent, refused = run("toolkit", "13.3"), run("toolkit", "13.1"), run("arch", "NVIDIA A100-SXM4-80GB")
    assert (ok.returncode, ok.stdout) == (0, "/usr/local/cuda-13.3\n")
    assert absent.returncode == 3 and absent.stdout == "" and "list no CUDA toolkit 13.1" in absent.stderr
    assert refused.returncode == 1 and refused.stderr.startswith("0 card records under ")


@pytest.mark.parametrize(("name", "arch"), [("NVIDIA H200", "sm_90a"), ("NVIDIA H100 80GB HBM3", "sm_90a")])
def test_the_arch_is_the_card_records_whose_driver_holds_the_reported_name(m, name, arch):
    assert m.arch(name) == arch


def test_a_card_no_record_holds_is_refused_by_name(m):
    with pytest.raises(SystemExit, match=r"^0 card records under .* have 'NVIDIA A100-SXM4-80GB'"):
        m.arch("NVIDIA A100-SXM4-80GB")
