"""`serve.sh`'s refusal of a run directory that already holds a record, before it touches the machine or writes a file:
a share read after a failed run would otherwise be an earlier run's number. And its serve's card under a caller's
gpu-lease: the lease's, with no lease of its own, as gpu-lease isn't re-entrant. And the Lean verdict's limit, which
grows as the cores fall."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

SERVE = Path(__file__).resolve().parents[3] / "benchmarks" / "compute" / "serve.sh"


@pytest.mark.parametrize("held", ["steps.txt", "compute-run.json", "logs", "pieces", "record", "registered"])
def test_a_run_dir_holding_a_record_is_refused_before_any_step(tmp_path, held):
    out = tmp_path / "run" / "out"
    out.mkdir(parents=True)
    if "." in held:
        (out / held).write_text("an earlier run's\n")
    else:
        (out / held).mkdir()
    before = sorted(p.name for p in out.iterdir())
    got = subprocess.run(["bash", str(SERVE)], env={**os.environ, "RUN_DIR": str(tmp_path / "run")},
                         capture_output=True, text=True)
    assert got.returncode == 3 and got.stdout == ""
    assert f"{out} already holds a run's record ({out / held})" in got.stderr and "give RUN_DIR a new directory" in got.stderr
    assert sorted(p.name for p in out.iterdir()) == before


def served(tmp_path, **env) -> list[str]:
    """What serve.sh's card choice and its `lease` print of a serve, with a `gpu-lease` on PATH that only says how it was
    called: serve.sh's own lines, run alone."""
    text = SERVE.read_text()
    card = re.search(r'^\[ -z "\$\{GPU_LEASE_UUID:-\}" \] \|\| GPU=.*$', text, re.M)
    lease = re.search(r"^lease\(\) \{.*?^\}$", text, re.M | re.S)
    assert card and lease, "serve.sh's card line under a lease or its lease() moved: this test reads them"
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "gpu-lease").write_text('#!/bin/sh\necho "gpu-lease $*"\n')
    (bin_ / "gpu-lease").chmod(0o755)
    script = f'GPU=${{GPU-0}}\n{card[0]}\n{lease[0]}\necho "GPU=$GPU"\nlease echo "served on ${{CUDA_VISIBLE_DEVICES:-none}}"\n'
    keep = {k: v for k, v in os.environ.items() if k not in ("GPU", "GPU_LEASE_UUID", "CUDA_VISIBLE_DEVICES")}
    got = subprocess.run(["bash", "-c", script], env={**keep, "PATH": f"{bin_}:{keep['PATH']}", **env},
                         capture_output=True, text=True, check=True)
    return got.stdout.splitlines()


def test_outside_a_lease_the_serve_takes_one_of_the_card_gpu_names(tmp_path):
    assert served(tmp_path) == ["GPU=0", 'gpu-lease 1 --wait --on 0 --max-min 30 -- echo served on none']


def test_under_a_callers_lease_the_serve_runs_on_its_card_and_takes_no_lease(tmp_path):
    """`research run --gpus 1`, or a caller's own `gpu-lease 1 --cores ...`, holds the card: a second lease of it would wait
    forever, so the serve runs on the lease's card (CUDA_VISIBLE_DEVICES, GPU its UUID for the up-front check)."""
    got = served(tmp_path, GPU_LEASE_UUID="GPU-67cd8250,GPU-099c8642", CUDA_VISIBLE_DEVICES="3,4", GPU="0")
    assert got == ["GPU=GPU-67cd8250", "served on 3,4"]


@pytest.mark.parametrize(("cores", "given", "limit"), [
    (96, None, "3600"), (192, None, "3600"), (36, None, "9600"), (20, None, "17280"), (20, "5000", "5000")])
def test_the_lean_verdicts_limit_grows_as_the_cores_fall(tmp_path, cores, given, limit):
    """On 20 cores the Lean verdict ran past a flat 3,600 s and the run failed with Python's accept in hand: the limit
    is 3,600 s on 96 cores and as much longer as there are fewer, and LEAN_TIMEOUT stands where given."""
    line = re.search(r"^CORES=\$\(nproc\) && LEAN_TIMEOUT=.*$", SERVE.read_text(), re.M)
    assert line, "serve.sh's LEAN_TIMEOUT line moved: this test reads it"
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "nproc").write_text(f"#!/bin/sh\necho {cores}\n")
    (bin_ / "nproc").chmod(0o755)
    keep = {k: v for k, v in os.environ.items() if k != "LEAN_TIMEOUT"}
    env = {**keep, "PATH": f"{bin_}:{keep['PATH']}", **({"LEAN_TIMEOUT": given} if given else {})}
    got = subprocess.run(["bash", "-c", f'{line[0]}\necho "$LEAN_TIMEOUT"'], env=env, capture_output=True, text=True,
                         check=True)
    assert got.stdout.strip() == limit
