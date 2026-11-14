"""The front door (``run.py``): which runner a NAME starts, what it hands it, and what it says to a NAME it has none for.
And the runners of ``commit_gpu/`` and ``harness/``: the job each starts, and the names ``harness/``'s refuses."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

from conftest import load

R = load("run.py")


def test_a_directorys_runner_is_its_run_py_else_its_run_sh(tmp_path):
    for file in ("job/run.sh", "script/run.py", "both/run.sh", "both/run.py", "neither/tool.py", "run.sh"):
        (tmp_path / file).parent.mkdir(exist_ok=True)
        (tmp_path / file).touch()
    assert R.runners(tmp_path) == {"both": [sys.executable, str(tmp_path / "both/run.py")],
                                   "job": ["bash", str(tmp_path / "job/run.sh")],
                                   "script": [sys.executable, str(tmp_path / "script/run.py")]}


def test_every_name_it_takes_is_a_directory_here_with_its_runner_in_it():
    names = R.runners()
    assert names and all((R.HERE / name).is_dir() for name in names)
    assert all(Path(command[1]).parent == R.HERE / name for name, command in names.items())


def _front_door(tmp_path: Path, *argv: str) -> subprocess.CompletedProcess:
    """A copy of the front door beside one benchmark, ``echo``, whose runner prints its arguments and exits 7."""
    shutil.copy(R.__file__, tmp_path / "run.py")
    (tmp_path / "echo").mkdir(exist_ok=True)
    (tmp_path / "echo" / "run.py").write_text("import sys\nprint(sys.argv[1:])\nsys.exit(7)\n")
    return subprocess.run([sys.executable, str(tmp_path / "run.py"), *argv], capture_output=True, text=True)


def test_it_hands_the_runner_the_rest_and_ends_as_the_runner_does(tmp_path):
    got = _front_door(tmp_path, "echo", "--flag", "a b")
    assert (got.returncode, got.stdout) == (7, "['--flag', 'a b']\n"), got.stderr


def test_a_name_with_no_runner_is_refused_with_the_names_that_have_one(tmp_path):
    for argv in ((), ("gone",), ("run",)):
        got = _front_door(tmp_path, *argv)
        assert got.returncode == 1 and got.stdout == ""
        assert got.stderr.strip() == "benchmarks/pouw/run.py NAME [ARGS] (a - in NAME is a _): NAME is one of echo"


def test_commit_gpus_runner_starts_the_job_beside_it_and_ends_as_it_does(tmp_path):
    """``commit_gpu/run.sh`` beside a stand-in ``gpu.sh``: whatever the working directory, the job gets the arguments
    and the caller's variables, and its exit code is the runner's. The directory holds the job the runner names."""
    assert R.runners()["commit_gpu"] == ["bash", str(R.HERE / "commit_gpu" / "run.sh")]
    assert (R.HERE / "commit_gpu" / "gpu.sh").is_file()
    shutil.copy(R.HERE / "commit_gpu" / "run.sh", tmp_path / "run.sh")
    (tmp_path / "gpu.sh").write_text('printf "%s|" "$@" "$BUILDS"; exit 7\n')
    got = subprocess.run(["bash", str(tmp_path / "run.sh"), "a b", "c"], capture_output=True, text=True, cwd="/",
                         env={"PATH": "/usr/bin:/bin", "BUILDS": "a0r0l"})
    assert (got.returncode, got.stdout, got.stderr) == (7, "a b|c|a0r0l|", "")


#: the harness's jobs, each with the script the runner starts for it
HARNESS_JOBS = {"build": "native/build.sh", "pin_target": "jobs/pin_target.sh",
                **{job: f"{job}.py" for job in ("bench", "characterize", "verify", "ledger", "price_twins", "ieee_pin")}}


def _harness(tmp_path: Path, *argv: str, **env: str) -> subprocess.CompletedProcess:
    """A copy of ``harness/run.sh`` beside a stand-in for each job's script, which prints its job, its arguments and
    ``$BUILDS`` and exits 7, started from ``/`` with only ``env`` and a ``PATH`` that holds this Python's ``python3``."""
    shutil.copy(R.HERE / "harness" / "run.sh", tmp_path / "run.sh")
    for job, file in HARNESS_JOBS.items():
        (tmp_path / file).parent.mkdir(exist_ok=True)
        (tmp_path / file).write_text(f'printf "%s|" {job} "$@" "$BUILDS"; exit 7\n' if file.endswith(".sh") else
                                     f'import os, sys\nprint("{job}", *sys.argv[1:], os.environ["BUILDS"], sep="|", '
                                     'end="|")\nsys.exit(7)\n')
    return subprocess.run(["bash", str(tmp_path / "run.sh"), *argv], capture_output=True, text=True, cwd="/",
                          env={"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin", "BUILDS": "a0r0l", **env})


def test_the_harnesss_runner_starts_the_job_it_is_asked_for_and_ends_as_it_does(tmp_path):
    """Whatever the working directory, each job's script gets the arguments and the caller's variables, and its exit
    code is the runner's; a Python job runs under ``$PY`` where the caller has set it. The front door takes the runner,
    and the directory holds the script of every job."""
    assert R.runners()["harness"] == ["bash", str(R.HERE / "harness" / "run.sh")]
    assert all((R.HERE / "harness" / file).is_file() for file in HARNESS_JOBS.values())
    for job in HARNESS_JOBS:
        got = _harness(tmp_path, job, "a b", "c")
        assert (got.returncode, got.stdout, got.stderr) == (7, f"{job}|a b|c|a0r0l|", ""), job
    (tmp_path / "py").write_text('printf "%s|" py "$@"; exit 9\n')
    (tmp_path / "py").chmod(0o755)
    got = _harness(tmp_path, "bench", "--plan", PY=str(tmp_path / "py"))
    assert (got.returncode, got.stdout, got.stderr) == (9, f"py|{tmp_path}/bench.py|--plan|", "")
    got = _harness(tmp_path, "build", PY=str(tmp_path / "py"))
    assert (got.returncode, got.stdout) == (7, "build|a0r0l|")


def test_the_harnesss_runner_refuses_a_name_that_is_no_job_and_its_readme_names_each_job(tmp_path):
    """A name that is no job starts nothing and gets the usage, which names every job and no other; so does the
    README's paragraph on the runner."""
    for argv in ((), ("gone",), ("run",), ("sass_gate",), ("bench.py",), ("../harness/bench",), ("bench ",)):
        got = _harness(tmp_path, *argv)
        assert (got.returncode, got.stdout) == (2, ""), argv
        usage, _, rest = got.stderr.strip().partition("   JOB: ")
        names, _, said = rest.rpartition(" (got ")
        assert usage == "usage: run.sh JOB [ARGS...]" and said == f"'{''.join(argv)}')"
        assert sorted(names.replace(" or ", ", ").split(", ")) == sorted(HARNESS_JOBS)
    paragraph = next(p for p in (R.HERE / "harness" / "README.md").read_text().split("\n\n")
                     if p.startswith("The runner is `run.sh JOB [ARGS]`"))
    jobs = paragraph.partition("JOB is ")[2].partition(", the Python script")[0]
    assert sorted(re.findall(r"`([a-z_]+)`", jobs)) == sorted(HARNESS_JOBS)
