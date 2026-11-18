"""The target check of dc's GPU scripts, run on a CPU with stand-ins for nvidia-smi and nvcc: each takes its target from
the card the run opened (its card record's `arch`), goes on to the build on a card whose target it has forms for (which
the stand-in nvcc refuses), and stops with exit 2 before anything is built on any other. With no card, a build takes the
card file's sku and is refused without one. The jobs themselves need an H200 or a B200."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DC = "calibration/pouw/dc"
#: the name nvidia-smi gives a card of each compute capability, and the card file's sku
NAMES = {"9.0": "NVIDIA H200", "10.0": "NVIDIA B200", "12.0": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}
SKUS = {"9.0": "h200-sxm-141gb", "10.0": "b200-sxm-180gb", "12.0": "rtx-pro-6000-server"}
#: (script, the compute capabilities whose targets it has forms for, its exit when the build is refused)
JOBS = [("run.sh", ["9.0", "10.0"], 1), ("uni.sh", ["9.0", "10.0"], 1), ("pipe.sh", ["10.0"], 1), ("ffill.sh", ["9.0"], 1),
        ("s8_mma.sh", ["9.0", "10.0"], 4)]
OWN = [(s, cc, refused) for s, ccs, refused in JOBS for cc in ccs]
OTHER = [(s, cc) for s, ccs, _ in JOBS for cc in NAMES if cc not in ccs]
TARGET = {"9.0": "sm_90a", "10.0": "sm_100a", "12.0": "sm_120a"}


def _start(tmp_path: Path, script: str, args: list[str], reported: str | None, **env_: str) -> subprocess.CompletedProcess:
    """The script with a nvidia-smi that reports a card of compute capability `reported` (none: no GPU) and a nvcc that
    compiles nothing."""
    bin_ = tmp_path / "bin"
    bin_.mkdir(parents=True)
    (bin_ / "nvidia-smi").write_text(
        '#!/bin/sh\n[ -n "$STAND_IN_NAME" ] || exit 9\ncase "$*" in *--query-gpu=name,*) echo "$STAND_IN_NAME, GPU-0, 580.0, 1980, 2619" ;;\n'
        '  *--query-gpu=name*) echo "$STAND_IN_NAME" ;; *--query-gpu=*) echo "$STAND_IN_CC" ;; *) echo stand-in ;; esac\n')
    (bin_ / "nvcc").write_text('#!/bin/sh\n[ "$1" = --version ] && { echo "nvcc: stand-in"; exit 0; }\nexit 1\n')
    for tool in bin_.iterdir():
        tool.chmod(0o755)
    run_dir = tmp_path / "run"
    (run_dir / "out").mkdir(parents=True)
    env = {"PATH": f"{bin_}{os.pathsep}{os.defpath}", "NVCC": str(bin_ / "nvcc"), "STAND_IN_CC": reported or "",
           "STAND_IN_NAME": NAMES.get(reported or "", ""), "OUT": str(run_dir / "out"), "RESEARCH_RUN_DIR": str(run_dir),
           "TMPDIR": str(tmp_path), **env_}
    return subprocess.run(["bash", f"{DC}/{script}", *args], cwd=ROOT, env=env, capture_output=True, text=True)


@pytest.mark.parametrize("script,cc,refused", OWN, ids=[f"{s} {TARGET[cc]}" for s, cc, _ in OWN])
def test_a_card_whose_target_it_has_forms_for_reaches_the_build(tmp_path, script, cc, refused):
    done = _start(tmp_path, script, [], cc)
    assert done.returncode == refused, (done.stdout, done.stderr)
    assert "nothing built" not in done.stderr


@pytest.mark.parametrize("script,cc", OTHER, ids=[f"{s} {TARGET[cc]}" for s, cc in OTHER])
def test_any_other_card_is_refused_before_anything_is_built(tmp_path, script, cc):
    done = _start(tmp_path, script, [], cc)
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert f"the card's target is {TARGET[cc]}, and " in done.stderr and "nothing built" in done.stderr
    assert not (tmp_path / "run" / "out" / "env" / "nvcc.txt").exists()


@pytest.mark.parametrize("script", ["run.sh", "uni.sh", "pipe.sh", "ffill.sh"])
def test_with_no_card_a_build_takes_the_card_files_sku_and_is_refused_without_one(tmp_path, script):
    cc = next(ccs for s, ccs, _ in JOBS if s == script)[0]
    done = _start(tmp_path / "none", script, [], None, DC_BUILD_ONLY="1")
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert "this machine has no GPU: pass --sku SKU" in done.stderr
    done = _start(tmp_path / "sku", script, [], None, DC_BUILD_ONLY="1", DC_SKU=SKUS[cc])
    assert done.returncode == 1 and "nothing built" not in done.stderr, (done.stdout, done.stderr)


def test_s8_mma_builds_for_the_sku_it_is_given_on_a_cpu_node(tmp_path):
    done = _start(tmp_path / "sku", "s8_mma.sh", ["build", SKUS["10.0"]], None)
    assert done.returncode == 4, (done.stdout, done.stderr)
    done = _start(tmp_path / "rtx", "s8_mma.sh", ["build", SKUS["12.0"]], None)
    assert done.returncode == 2 and "the card's target is sm_120a" in done.stderr, (done.stdout, done.stderr)
    done = _start(tmp_path / "none", "s8_mma.sh", ["build"], None)
    assert done.returncode != 0 and "usage: s8_mma.sh build SKU" in done.stderr, (done.stdout, done.stderr)


def test_emp_builds_for_the_sku_it_is_given_and_refuses_without_one(tmp_path):
    done = _start(tmp_path / "none", "emp.sh", ["build"], None)
    assert done.returncode != 0 and "usage: emp.sh build SKU" in done.stderr, (done.stdout, done.stderr)
    done = _start(tmp_path / "sku", "emp.sh", ["build", SKUS["9.0"], f"{DC}/probes/scalar.cu"], None)
    assert done.returncode == 1 and "build failed: " in done.stderr and "nothing built" in done.stderr, done.stderr


EMP_RUNS = [(s, cc) for s in ("h32", "emp") for cc in NAMES]


@pytest.mark.parametrize("probe_set,cc", EMP_RUNS, ids=[f"emp.sh run {s} {TARGET[cc]}" for s, cc in EMP_RUNS])
def test_emp_run_builds_its_set_for_the_card_it_has_sources_for_and_refuses_any_other(tmp_path, probe_set, cc):
    done = _start(tmp_path, "emp.sh", ["run", probe_set], cc)
    if cc == "9.0":
        assert done.returncode == 1 and f"build failed: {ROOT}/{DC}/probes/sm90_{probe_set}" in done.stderr, done.stderr
        assert "the card's target is" not in done.stderr
    else:
        assert done.returncode == 2, (done.stdout, done.stderr)
        assert f"the card's target is {TARGET[cc]}, and " in done.stderr and "nothing built" in done.stderr
        assert not (tmp_path / "run" / "out" / "build.log").exists()
