"""verity_vllm/ops/serve_fresh.sh: serving from a clone with everything under one directory.

The serve itself needs a Hopper card and cannot run here; what is checked is the dry run (every path under the directory, whatever
the caller's environment names, the steps in order, nothing made), each refusal before anything is built, and that the script
names no shared path."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

VLLM = (Path(__file__).resolve().parents[5] / "adapters" / "vllm")
SCRIPT = VLLM / "verity_vllm" / "ops" / "serve_fresh.sh"
BASH = shutil.which("bash") or "/bin/bash"
TOOLS = ("realpath", "dirname", "cut", "awk", "tr", "df", "tail", "mkdir", "bash", "date", "setsid")
HOSTILE = {"WORK": "/workspace", "VENV": "/workspace/venv312", "HF_HOME": "/workspace/hf", "FA2": "/workspace/cp/fa2",
           "UV_CACHE_DIR": "/root/.cache/uv", "PIP_CACHE_DIR": "/root/.cache/pip", "HOME": "/root",
           "XDG_CACHE_HOME": "/root/.cache", "TRITON_CACHE_DIR": "/workspace/cache/triton"}


def _run(cwd: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([BASH, str(SCRIPT), *args], cwd=cwd, capture_output=True, text=True,
                          env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **(env or {})})


def _paths(out: str) -> dict[str, str]:
    return dict(line[len("path "):].split("=", 1) for line in out.splitlines() if line.startswith("path ") and "=" in line)


def test_dry_run_puts_every_path_under_the_work_dir_whatever_the_environment_names(tmp_path: Path) -> None:
    r = _run(tmp_path, "--dry-run", env=HOSTILE)
    assert r.returncode == 0, r.stderr
    work = str(tmp_path / ".verity-work")
    paths = _paths(r.stdout)
    assert {"WORK", "VENV", "HF_HOME", "FA2", "NT", "RT", "HOME", "XDG_CACHE_HOME", "UV_PYTHON_INSTALL_DIR",
            "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "CUDA_CACHE_PATH", "NATIVE_COLLECT_BUILD"} <= set(paths)
    assert all(p == work or p.startswith(work + "/") for p in paths.values()), paths
    assert f"uv and pip caches: {work}/.cache/" in r.stdout
    tree = str(VLLM.parents[1])
    roots = next(line for line in r.stdout.splitlines() if line.startswith("pythonpath ")).split(" ", 1)[1].split(":")
    assert {f"{tree}/adapters/vllm", f"{tree}/adapters/torch", f"{tree}/tools/config", f"{tree}/tools/research/src"} <= set(roots)
    out = r.stdout.replace(work, "WORK").replace(tree, "TREE")   # either may sit under /workspace (a pod's checkout, a node's TMPDIR)
    assert "/workspace" not in out and "/root" not in out
    steps = [line.split(":", 1)[0][len("step "):] for line in r.stdout.splitlines() if line.startswith("step ")]
    assert steps == ["bootstrap", "workload", "predict", "build", "serve", "request", "stop"]
    build = next(line for line in r.stdout.splitlines() if line.startswith("build: "))
    jobs = re.search(r"MAX_JOBS=(\d+) ", build)
    assert "head size 64 only" in build and jobs and 1 <= int(jobs[1]) <= (os.cpu_count() or 1), build
    assert not (tmp_path / ".verity-work").exists(), "a dry run makes nothing"


def test_dry_run_takes_the_directory_the_caller_names(tmp_path: Path) -> None:
    r = _run(tmp_path, "--dry-run", "--work", str(tmp_path / "w"))
    assert r.returncode == 0, r.stderr
    assert _paths(r.stdout)["HF_HOME"] == str(tmp_path / "w" / "hf")
    assert not (tmp_path / "w").exists()


def _machine(tmp_path: Path, cards: str | None, *, nvcc: bool = True, uv: bool = True) -> dict[str, str]:
    """A PATH holding only the tools the checks use, a fake `nvidia-smi` printing `cards`, a fake `uv`, and a CUDA_HOME
    whose bin/nvcc exists or not."""
    b, cuda = tmp_path / "bin", tmp_path / "cuda"
    b.mkdir(exist_ok=True)
    (cuda / "bin").mkdir(parents=True, exist_ok=True)
    for t in TOOLS:
        found = shutil.which(t)
        assert found, f"{t} is not on PATH"
        (b / t).symlink_to(found)
    fakes = {"nvidia-smi": f"#!/bin/sh\nprintf '{cards}\\n'\n" if cards is not None else None, "uv": "#!/bin/sh\n" if uv else None}
    if nvcc:
        fakes[str(cuda / "bin" / "nvcc")] = "#!/bin/sh\n"
    for name, text in fakes.items():
        if text is not None:
            f = b / name if "/" not in name else Path(name)
            f.write_text(text)
            f.chmod(0o755)
    return {"PATH": str(b), "CUDA_HOME": str(cuda)}


@pytest.mark.parametrize(("cards", "extra", "args", "says"), [
    (None, {}, (), "no nvidia-smi"),
    ("0, 8.9, NVIDIA L40S", {}, (), "no Hopper card (compute capability 9.0)"),
    ("0, 9.0, NVIDIA H200\\n1, 8.9, NVIDIA L40S", {}, ("--gpu", "1"), "no Hopper card (compute capability 9.0) at index 1"),
    ("0, 9.0, NVIDIA H200", {"nvcc": False}, (), "no nvcc under CUDA_HOME="),
    ("0, 9.0, NVIDIA H200", {"uv": False}, (), "no uv: install it"),
    ("0, 9.0, NVIDIA H200", {"DISK_GB": "999999999"}, (), "this needs: free room there or pass --work DIR"),
])
def test_a_machine_that_lacks_something_is_refused_in_one_line_before_anything_is_built(
        tmp_path: Path, cards: str | None, extra: dict, args: tuple[str, ...], says: str) -> None:
    env = _machine(tmp_path, cards, nvcc=extra.get("nvcc", True), uv=extra.get("uv", True))
    env.update({k: v for k, v in extra.items() if k == "DISK_GB"})
    r = _run(tmp_path, "--work", str(tmp_path / "w"), *args, env=env)
    assert r.returncode == 3, (r.stdout, r.stderr)
    assert len(r.stderr.strip().splitlines()) == 1 and says in r.stderr, r.stderr
    assert r.stdout.strip().splitlines()[-1].startswith("SERVE-FRESH FAILED check: ")
    assert not (tmp_path / "w" / "venv312").exists() and not (tmp_path / "w" / "logs").exists()


def test_a_signal_to_the_script_stops_the_step_it_is_running_and_everything_that_step_started(tmp_path: Path) -> None:
    """A stopped run once left its server holding the card: the step's whole process group must go with the script."""
    ops = tmp_path / "tree" / "adapters" / "vllm" / "verity_vllm" / "ops"
    ops.mkdir(parents=True)
    (tmp_path / "tree" / "pyproject.toml").write_text("")
    script = ops / SCRIPT.name
    shutil.copy(SCRIPT, script)
    fifo = tmp_path / "held"
    os.mkfifo(fifo)
    # the stand-in bootstrap says it is up, then a grandchild keeps the FIFO open until it is stopped
    (ops / "pod_bootstrap.sh").write_text('exec 3>"$HELD"; echo up >&3; bash -c "exec tail -f /dev/null" 3>&3\n')
    env = {**_machine(tmp_path, "0, 9.0, NVIDIA H200"), "DISK_GB": "0", "HELD": str(fifo)}
    p = subprocess.Popen([BASH, str(script), "--work", str(tmp_path / "w")], cwd=tmp_path, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with open(fifo) as held:
        assert held.readline() == "up\n"
        p.send_signal(signal.SIGTERM)
        assert held.read() == ""   # end of file: every holder of the FIFO, the grandchild included, is gone
    out, err = p.communicate()
    assert p.returncode == 143, (out, err)
    assert out.strip().splitlines()[-1] == "SERVE-FRESH FAILED bootstrap: stopped by a signal"


def test_the_script_names_no_shared_path_and_no_cache_of_the_caller() -> None:
    text = SCRIPT.read_text()
    assert "/workspace" not in text and "/root" not in text and "~/" not in text
    assert "unset UV_CACHE_DIR PIP_CACHE_DIR" in text
    assert "gpu-lease" not in text and "research run" not in text
