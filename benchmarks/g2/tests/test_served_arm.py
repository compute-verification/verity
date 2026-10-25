"""G2's served arm (`benchmarks/served_commit/served_arm.py`, run by `served_arm.sh`): the compiled Commit names its execution, so
the harness builds it at vLLM's serving defaults compiled beside an eager spec's arms (`verity_bench.vllm.arms.experimental_arm`),
each arm with its precision, token control and retention; the eager Commit is the harness's `commit` with every chunk constant at
4 KiB before that arm makes its committer; and `served_arm.sh` runs a seed pass into the job's one Inductor dir before the measured
pass, or seeds it from a Build of record and checks it after (`served-pin`), listing the autotune winners after each. The arms these
build on are stand-ins (the harness's suite tests them), and so are `run.sh`, `nvidia-smi` and the venv's `served-pin`."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SERVED_COMMIT = Path(__file__).resolve().parents[2] / "served_commit"


def _module(name: str, **attrs) -> ModuleType:
    mod = ModuleType(name)
    vars(mod).update(attrs)
    return mod


@pytest.fixture
def served_arm(monkeypatch):
    """served_arm.py over stand-ins: `prof_arm.Rows4kArm`, the harness's `CommitArm` (its ``ready`` records the chunk constants
    it sees) and the integration's chunk constants (256 B, and step rows' at 0 so that setting it shows)."""
    step_rows = _module("verity_vllm.commit.step_rows", ROW_BYTES=0)
    chunks = {n: _module(f"verity_vllm.commit.committer.{n}", GPU_CHUNK_BYTES=256)
              for n in ("host_retention", "native_collect", "native_host")}

    class Rows4kArm:
        def __init__(self, id: str, retain: str) -> None:
            self.id, self.retain = id, retain

    class CommitArm:
        execution = None

        def __init__(self, id: str) -> None:
            self.id, self.seen = id, None

        def ready(self, target) -> None:
            self.seen = (step_rows.ROW_BYTES, *(chunks[n].GPU_CHUNK_BYTES for n in sorted(chunks)))

        def gate(self, target) -> dict:
            return {"passed": True, "commit_scheme": "rows-sha512"}

    for mod in (_module("served_commit"), _module("served_commit.prof_arm", Rows4kArm=Rows4kArm), _module("verity_bench"),
                _module("verity_bench.vllm"), _module("verity_bench.vllm.commit_arm", CommitArm=CommitArm),
                _module("verity_vllm"), _module("verity_vllm.commit", step_rows=step_rows), step_rows,
                _module("verity_vllm.commit.committer", **chunks), *chunks.values()):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    spec = importlib.util.spec_from_file_location("served_commit.served_arm", SERVED_COMMIT / "served_arm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_compiled_commit_names_its_execution_precision_control_and_retention(served_arm):
    assert [(a.id, a.execution, a.precision, a.control, a.retain) for a in (served_arm.ROWS_4K, served_arm.ROWS_4K_FP8)] == [
        ("served-commit-rows-4k", "compiled", "bfloat16", "stock@compiled", "none"),
        ("served-commit-rows-4k", "compiled", "fp8", "stock-fp8@compiled", "none")]


def test_the_eager_commit_is_the_harness_commit_with_every_chunk_constant_at_4k(served_arm):
    arms = (served_arm.EAGER_ROWS_4K, served_arm.EAGER_ROWS_4K_FP8)
    assert [(a.id, a.execution, a.precision, a.control) for a in arms] == [
        ("commit-rows-4k-eager", None, "bfloat16", "stock"), ("commit-rows-4k-eager", None, "fp8", "stock-fp8")]
    arm = served_arm.EAGER_ROWS_4K
    arm.ready(SimpleNamespace())
    assert arm.seen == (4096, 4096, 4096, 4096)
    assert arm.gate(None) == {"passed": True, "commit_scheme": "rows-sha512", "row_bytes": 4096}


RUN_SH = """#!/usr/bin/env bash
i=$(find "$TORCHINDUCTOR_CACHE_DIR" -name '*.best_config' | wc -l)
touch "$TORCHINDUCTOR_CACHE_DIR/w$i.best_config"
echo "$OUT|$TORCHINDUCTOR_CACHE_DIR|$NATIVE_COLLECT_BUILD|$VLLM_CACHE_ROOT|$VERITY_INDUCTOR_DIR_PER_PROCESS|${VLLM_DISABLE_COMPILE_CACHE:-}|$*" >> "$LOG"
[ "${OUT##*/}" = seed ] || exit "${RC:-0}"
"""
#: the venv's python, for `-m verity_vllm.pipeline.cli served-pin CMD --build B --inductor-dir D`: seed adds a winner, check passes
PYTHON = """#!/usr/bin/env bash
echo "pin|$4|$6|$8" >> "$LOG"
[ "$4" = seed ] && touch "$8/b0.best_config"
[ "$4" = check ] && echo "[served-pin] $8: winners equal to artifact 533721fc9d37833a's"
exit 0
"""


def _run_served_arm(tmp_path: Path, **env: str) -> tuple[subprocess.CompletedProcess, Path, list[list[str]]]:
    """served_arm.sh with `--spec s.toml` in a tree whose `benchmarks/served_commit/run.sh` records each pass (its OUT, Inductor
    dir, build and cache dirs, the per-process mark, whether compile caches are off, and its arguments) and adds one autotune
    winner to the Inductor dir, and whose venv's python records each `served-pin` call."""
    tree, out, log = tmp_path / "tree", tmp_path / "out", tmp_path / "passes.log"
    (tree / "benchmarks" / "served_commit").mkdir(parents=True)
    (tree / "benchmarks" / "harness").mkdir()
    (tree / "bin").mkdir()
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    (tree / "benchmarks" / "served_commit" / "run.sh").write_text(RUN_SH)
    (tree / "benchmarks" / "harness" / "env.sh").write_text('VENV=$VERITY_BENCH_VENV\n')
    (tmp_path / "venv" / "bin" / "python").write_text(PYTHON)
    (tmp_path / "venv" / "bin" / "python").chmod(0o755)
    (tree / "bin" / "nvidia-smi").write_text("#!/bin/sh\necho '0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 0 MiB'\n")
    (tree / "bin" / "nvidia-smi").chmod(0o755)
    run = subprocess.run(["bash", str(SERVED_COMMIT / "served_arm.sh"), "--spec", "s.toml"], cwd=tree, capture_output=True,
                         text=True, check=False, env={"PATH": f"{tree / 'bin'}:{os.environ['PATH']}", "OUT": str(out), "LOG": str(log),
                                                      "VERITY_BENCH_VENV": str(tmp_path / "venv"), **env})
    return run, out, [line.split("|") for line in log.read_text().splitlines()] if log.exists() else []


def test_served_arm_sh_seeds_the_jobs_inductor_dir_before_the_measured_pass(tmp_path):
    run, out, passes = _run_served_arm(tmp_path, SEED_ARGS="--spec s.toml --arms stock@compiled", RC="3")
    assert run.returncode == 3 and "NVIDIA RTX PRO 6000" in run.stdout
    (seed_out, ind, build, cache, mark, fresh, seed_args), (measured_out, ind2, *_, args) = passes
    assert fresh == ""
    assert (seed_out, seed_args, measured_out, args) == (f"{out}/seed", "--spec s.toml --arms stock@compiled --reps 1", str(out),
                                                         "--spec s.toml")
    job = Path(ind).parent
    assert ind == ind2 and (build, cache, mark) == (f"{job}/nc_build", f"{job}/vllm_cache", "1") and not job.exists()
    assert (out / "inductor_winners.txt").read_text() == "seed 1\nmeasured 2\n"
    assert (out / "seed_winners.txt").read_text() == "./w0.best_config\n"
    assert (out / "measured_winners.txt").read_text() == "./w0.best_config\n./w1.best_config\n"


def test_served_arm_sh_without_a_seed_runs_the_measured_pass_alone(tmp_path):
    run, out, passes = _run_served_arm(tmp_path)
    assert run.returncode == 0 and [p[0] for p in passes] == [str(out)]
    assert (out / "inductor_winners.txt").read_text() == "measured 1\n" and not (out / "seed_winners.txt").exists()


def test_served_arm_sh_pins_the_inductor_dir_to_a_build_of_record_and_checks_it_after_the_run(tmp_path):
    run, out, passes = _run_served_arm(tmp_path, PIN_BUILD_NAME="build.json", RESEARCH_RUN_DIR=str(tmp_path / "run"), RC="3")
    build = str(tmp_path / "run" / "inputs" / "build.json")
    (_, seed, built, ind), (measured_out, ind2, *_, fresh, args), (_, check, built2, ind3) = passes
    assert run.returncode == 3 and (seed, check, built, built2) == ("seed", "check", build, build)
    assert (measured_out, args, fresh) == (str(out), "--spec s.toml", "1") and ind == ind2 == ind3
    assert (out / "inductor_winners.txt").read_text() == "seed 1\nmeasured 2\n"
    assert (out / "pin_check.txt").read_text() == f"[served-pin] {ind}: winners equal to artifact 533721fc9d37833a's\n"


def test_served_arm_sh_refuses_a_seed_pass_beside_a_build_of_record(tmp_path):
    run, _, passes = _run_served_arm(tmp_path, PIN_BUILD="b.json", SEED_ARGS="--arms stock@compiled")
    assert run.returncode == 2 and passes == [] and "drop SEED_ARGS" in run.stderr
