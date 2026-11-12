"""The verifier-cost benchmarks still build against the checkout's verifiers.

A benchmark's Lean here is one file, `<name>/Main.lean`, which its script copies into a scratch Lake package on a node and
builds against a verifier's package. Nothing else reads those files, so a change to a function one of them calls leaves its
benchmark unable to build, unnoticed until somebody next runs it. So each main is elaborated here (`lake env lean`, which
writes nothing) in the package that holds the libraries it imports, once `lake build` has built there the modules it imports;
and whole_run.py, whose inputs come from sampled proofs' difftest module, still writes a run that its own check accepts.

An elaboration compiles and links nothing and runs nothing: that a benchmark's executable still gives the numbers its script
reads is its run's to show.
"""
from __future__ import annotations

import fcntl
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from verity_difftest import BUILD_LOCK

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
#: each Lake package a main is elaborated in, with the libraries `lake env` gives there (its own, and Flock's where it
#: requires Flock's package by path). A main goes in the first that holds every library it imports, so a main that reads
#: Flock alone builds neither of the others.
PACKAGES = (("verity/protocols/flock/verifier", {"Flock"}),
            ("verity/protocols/inference_only/verifier/nci/lean", {"NciCheck", "Flock"}),
            ("verity/protocols/spot/verifier/lean", {"Spot", "Flock"}))
MAINS = sorted(p.parent.name for p in HERE.glob("*/Main.lean"))
assert MAINS, f"no <name>/Main.lean under {HERE}: this suite would hold nothing"


def lake(package: Path, *args: str) -> subprocess.CompletedProcess:
    """`lake ARGS` in `package`, one at a time with every suite's Lake build (the difftest harness's lock): Lake builds a
    path requirement in place, and these packages share Flock's with the suites `check` runs beside this one."""
    BUILD_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(BUILD_LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return subprocess.run(["lake", *args], cwd=package, capture_output=True, text=True)


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
@pytest.mark.parametrize("name", MAINS)
def test_main_elaborates_against_the_checkouts_verifier(name):
    main = HERE / name / "Main.lean"
    modules = re.findall(r"^import\s+(\S+)", main.read_text(), re.M)
    libraries = {m.split(".")[0] for m in modules}
    package = next((ROOT / p for p, holds in PACKAGES if libraries <= holds), None)
    assert package is not None, (
        f"{name}/Main.lean imports {sorted(libraries)}, which no package of PACKAGES holds together: add the package its "
        f"script builds it against to PACKAGES here, and that package's lakefile to `inputs` in this suite's pyproject.toml")
    where = package.relative_to(ROOT)
    built = lake(package, "build", *(f"+{m}" for m in modules))
    assert built.returncode == 0, (
        f"`lake build` of {' '.join(modules)} in {where} failed ({built.returncode}), so {name}/Main.lean was not read:\n"
        f"{built.stdout[-3000:]}{built.stderr[-3000:]}")
    got = lake(package, "env", "lean", str(main))
    assert got.returncode == 0, (
        f"benchmarks/verifier_cost/{name}/Main.lean no longer elaborates against {where}, so no script here that builds it "
        f"can run. Fix the main in the change that moved what it calls:\n{got.stdout[-3000:]}{got.stderr[-3000:]}")


def test_whole_run_writes_a_run_its_check_accepts(tmp_path):
    """whole_run.py on the toy at whole_run.sh's smallest size: `--check` exits nonzero unless the Python reference
    (`test_difftest.run`) accepts the run it wrote."""
    got = subprocess.run([sys.executable, str(HERE / "whole_run.py"), str(tmp_path), "toy", "146", "--check"], cwd=ROOT,
                         capture_output=True, text=True)
    assert got.returncode == 0 and (tmp_path / "run-toy-146.json").is_file(), (
        f"whole_run.py toy 146 --check failed ({got.returncode}), so whole_run.sh has no input to time: fix the generator "
        f"in the change that moved what it calls:\n{got.stdout[-3000:]}{got.stderr[-3000:]}")
