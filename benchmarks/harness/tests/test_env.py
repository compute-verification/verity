"""env.sh, which every launcher sources, puts the tree's packages on PYTHONPATH for a pinned-vLLM environment made without
the workspace, and `lake` on the PATH for the in-process challenger's Lean coins."""

from __future__ import annotations

import ast
import shutil
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASH = shutil.which("bash")


def _tools(tmp_path: Path) -> str:
    """A PATH of only the programs env.sh runs, so no `lake` installed on this machine is on it."""
    tools = tmp_path / "tools"
    if not tools.is_dir():
        tools.mkdir()
        for name in ("awk", "tr"):
            (tools / name).symlink_to(shutil.which(name))
    return str(tools)


def _sourced(tmp_path: Path, *names: str, path: str | None = None) -> dict[str, str]:
    path = path or _tools(tmp_path)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "bin" / "python").write_text("#!/bin/sh\n")
    (venv / "bin" / "python").chmod(0o755)
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
