"""env.sh, which every launcher sources, puts the tree's packages on PYTHONPATH for a pinned-vLLM environment made without
the workspace, and `lake` on the PATH for the in-process challenger's Lean coins."""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
BASH = shutil.which("bash")


def _tools(tmp_path: Path) -> str:
    """A PATH of none of this machine's programs (env.sh runs only the launch's Python), so no `lake` installed here is
    on it."""
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    return str(tools)


def _sourced(tmp_path: Path, *names: str, path: str | None = None) -> dict[str, str]:
    path = path or _tools(tmp_path)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    if not (venv / "bin" / "python").exists():
        (venv / "bin" / "python").symlink_to(sys.executable)
    env = {"PATH": path, "HOME": str(tmp_path), "VERITY_BENCH_VENV": str(venv), "CUDA_HOME": str(tmp_path)}
    printed = " ".join(f'"${n}"' for n in names)
    ran = subprocess.run([BASH, "-c", f". benchmarks/harness/env.sh && printf '%s\\n' {printed}"], cwd=ROOT, env=env,
                         capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr
    return dict(zip(names, ran.stdout.splitlines()))


def _members() -> list[str]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]["workspace"]["members"]


def _first_party() -> dict[str, Path]:
    """Every top-level package the workspace builds, by name: `verity`, and each that a member's wheel names."""
    found = {"verity": ROOT / "verity"}
    for m in _members():
        wheel = tomllib.loads((ROOT / m / "pyproject.toml").read_text()).get("tool", {}).get("hatch", {}).get(
            "build", {}).get("targets", {}).get("wheel", {}).get("packages", [])
        found |= {Path(p).name: ROOT / m / p for p in wheel}
    return found


def _imported(package: Path) -> set[str]:
    names = set()
    for f in package.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(), str(f))):
            if isinstance(node, ast.Import):
                names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names.add(node.module.split(".")[0])
    return names


def test_every_workspace_members_import_root_is_on_the_path(tmp_path):
    """A member added to the workspace is on the path with no line in env.sh: `adapters/torch` was added after env.sh's
    list was written, and a served PoUW spec stopped at `No module named verity_torch`."""
    path = _sourced(tmp_path, "PYTHONPATH")["PYTHONPATH"].split(":")
    for name, package in _first_party().items():
        assert str(package.parent) in path, f"{name} ({package.relative_to(ROOT)}) is on no entry of env.sh's PYTHONPATH"
    assert path[-2:] == [str(ROOT), str(ROOT / "benchmarks")]


def test_every_first_party_package_the_harness_and_the_adapter_import_is_reachable(tmp_path):
    path = [Path(p) for p in _sourced(tmp_path, "PYTHONPATH")["PYTHONPATH"].split(":")]
    first_party = _first_party()
    importers = (ROOT / "adapters/vllm/verity_vllm", ROOT / "benchmarks/harness/verity_bench")
    imported = set().union(*map(_imported, importers)) & first_party.keys()
    assert {"verity", "verity_vllm", "verity_torch", "verity_config", "research"} <= imported
    unreachable = sorted(n for n in imported if not any((p / n / "__init__.py").is_file() for p in path))
    assert not unreachable, f"verity_vllm and verity_bench import {unreachable}, which env.sh's PYTHONPATH doesn't reach"


def test_the_python_is_the_launchs_and_the_toolkit_the_machines_with_no_path_of_a_node_named(tmp_path):
    """env.sh names no node's directory: the Python is the launch's or this user's record of one, else refused, and
    CUDA_HOME, where none is given, is the toolkit the run's machine's [properties] list first."""
    script = (ROOT / "benchmarks/harness/env.sh").read_text()
    assert "/workspace" not in script and "/usr/local" not in script
    ran = subprocess.run([BASH, "-c", ". benchmarks/harness/env.sh"], cwd=ROOT, env={"PATH": _tools(tmp_path)},
                         capture_output=True, text=True)
    assert ran.returncode == 2 and "else the path recorded in" in ran.stderr and "(none of them set)" in ran.stderr
    venv = tmp_path / "py"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").symlink_to(sys.executable)
    (tmp_path / "inputs").mkdir()
    machine = (ROOT / "tests/benchmarks/harness/fixtures/machines/one-card.toml").read_text()
    (tmp_path / "inputs" / "one-card.toml").write_text('machine = "one-card"\n\n[properties]'
                                                       + machine.split("[properties]", 1)[1])
    env = {"PATH": _tools(tmp_path), "HOME": str(tmp_path), "VERITY_BENCH_VENV": str(venv),
           "RESEARCH_MACHINE": "one-card", "RESEARCH_RUN_DIR": str(tmp_path)}
    ran = subprocess.run([BASH, "-c", '. benchmarks/harness/env.sh && printf "%s\\n" "$CUDA_HOME"'], cwd=ROOT, env=env,
                         capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == "/usr/local/cuda-13.0"



def test_without_verity_bench_venv_the_python_is_the_environment_the_launch_runs_in(tmp_path):
    """A launch inside an environment (`uv run`, an activated one) needs no VERITY_BENCH_VENV: env.sh takes $VIRTUAL_ENV,
    VERITY_BENCH_VENV wins where both are set, and one that holds no Python is refused by its path."""

    def venv(name: str) -> Path:
        (tmp_path / name / "bin").mkdir(parents=True)
        (tmp_path / name / "bin" / "python").symlink_to(sys.executable)
        return tmp_path / name

    def sourced(**env: str) -> subprocess.CompletedProcess:
        return subprocess.run([BASH, "-c", '. benchmarks/harness/env.sh && printf "%s\\n" "$VENV"'], cwd=ROOT,
                              env={"PATH": _tools(tmp_path), "HOME": str(tmp_path), "CUDA_HOME": str(tmp_path), **env},
                              capture_output=True, text=True)

    uv, given = venv("uv-env"), venv("given")
    ran = sourced(VIRTUAL_ENV=str(uv))
    assert ran.returncode == 0 and ran.stdout.strip() == str(uv), ran.stderr
    ran = sourced(VIRTUAL_ENV=str(uv), VERITY_BENCH_VENV=str(given))
    assert ran.returncode == 0 and ran.stdout.strip() == str(given), ran.stderr
    ran = sourced(VIRTUAL_ENV=str(tmp_path / "empty"))
    assert ran.returncode == 2 and f"({tmp_path}/empty)" in ran.stderr


def test_a_launch_that_names_no_environment_takes_the_one_the_bootstrap_recorded(tmp_path):
    """A launch line (`research run ... -- bash SCRIPT.sh`) gives neither VERITY_BENCH_VENV nor VIRTUAL_ENV: env.sh takes
    the environment pod_bootstrap.sh recorded under this user's state directory, both of the launch's win over it, a
    record of an environment that holds no Python is refused by its path, and the refusal names the command that records
    one a machine already has."""
    bootstrap = ROOT / "adapters/vllm/verity_vllm/ops/pod_bootstrap.sh"
    home, state = tmp_path / "home", tmp_path / "state"
    for name in ("pinned", "uv-env", "given"):
        (tmp_path / name / "bin").mkdir(parents=True)
        (tmp_path / name / "bin" / "python").symlink_to(sys.executable)

    def sourced(**env: str) -> subprocess.CompletedProcess:
        return subprocess.run([BASH, "-c", '. benchmarks/harness/env.sh && printf "%s\\n" "$VENV"'], cwd=ROOT,
                              env={"PATH": _tools(tmp_path), "HOME": str(home), "CUDA_HOME": str(tmp_path), **env},
                              capture_output=True, text=True)

    ran = sourced()
    assert ran.returncode == 2 and f"{home}/.local/state/verity-bench/venv" in ran.stderr
    assert "pod_bootstrap.sh --record-venv PATH" in ran.stderr
    recorded = subprocess.run([BASH, str(bootstrap), "--record-venv", str(tmp_path / "pinned")],
                              env={"PATH": "/usr/bin:/bin", "HOME": str(home)}, capture_output=True, text=True)
    assert recorded.returncode == 0, recorded.stderr
    assert (home / ".local/state/verity-bench/venv").read_text() == f"{(tmp_path / 'pinned').resolve()}\n"
    ran = sourced()
    assert ran.returncode == 0 and ran.stdout.strip() == str((tmp_path / "pinned").resolve()), ran.stderr
    ran = sourced(VIRTUAL_ENV=str(tmp_path / "uv-env"))
    assert ran.returncode == 0 and ran.stdout.strip() == str(tmp_path / "uv-env"), ran.stderr
    ran = sourced(VIRTUAL_ENV=str(tmp_path / "uv-env"), VERITY_BENCH_VENV=str(tmp_path / "given"))
    assert ran.returncode == 0 and ran.stdout.strip() == str(tmp_path / "given"), ran.stderr

    recorded = subprocess.run([BASH, str(bootstrap), "--record-venv", str(tmp_path / "uv-env")],
                              env={"PATH": "/usr/bin:/bin", "HOME": str(home), "XDG_STATE_HOME": str(state)},
                              capture_output=True, text=True)
    assert recorded.returncode == 0 and (state / "verity-bench/venv").is_file(), recorded.stderr
    ran = sourced(XDG_STATE_HOME=str(state))
    assert ran.returncode == 0 and ran.stdout.strip() == str((tmp_path / "uv-env").resolve()), ran.stderr
    (state / "verity-bench/venv").write_text(f"{tmp_path / 'gone'}\n")
    ran = sourced(XDG_STATE_HOME=str(state))
    assert ran.returncode == 2 and f"({tmp_path}/gone)" in ran.stderr

@pytest.mark.parametrize("script", ["benchmarks/harness/overhead_node.sh",
                                    *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "benchmarks/harness/setup").glob("*.sh"))])
def test_a_point_script_names_no_nodes_directory_and_no_card_by_name(script):
    """The toolkit is the kit's (`machine.py toolkit`), the card's arch its file's (`machine.py arch`), the Python the
    launch's, and a cache the user's: a script that names one of our nodes' paths, or tells a card by its name, refuses
    every other machine; one that branches on an architecture's name holds a toolkit no machine's record gives."""
    text = (ROOT / script).read_text()
    assert "/workspace" not in text and "/usr/local/cuda" not in text
    assert not re.search(r"\*(H100|H200|\"RTX PRO 6000\")\*", text)
    assert not re.search(r"(==|!=)\s*\"?sm_\d", text)


def test_lake_is_on_the_path_from_elan_when_nothing_else_puts_it_there(tmp_path):
    elan = tmp_path / ".elan" / "bin"
    lake = elan / "lake"
    assert _sourced(tmp_path, "PATH")["PATH"].split(":")[-1] == _tools(tmp_path)
    assert str(elan) not in _sourced(tmp_path, "PATH")["PATH"].split(":"), "no elan, so nothing added"
    elan.mkdir(parents=True)
    lake.write_text("#!/bin/sh\n")
    lake.chmod(0o755)
    assert _sourced(tmp_path, "PATH")["PATH"].split(":")[0] == str(elan)
    other = tmp_path / "bin"
    other.mkdir()
    (other / "lake").write_text("#!/bin/sh\n")
    (other / "lake").chmod(0o755)
    assert str(elan) not in _sourced(tmp_path, "PATH", path=f"{other}:{_tools(tmp_path)}")["PATH"].split(":")
