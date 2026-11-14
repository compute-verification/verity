"""benchmarks/pouw/pearl_c_vllm/run.sh, the directory's runner: its jobs are the job scripts beside it, every one, and
the README's table has a row a job.  A name that is no job is refused with the usage, before anything starts."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "pearl_c_vllm"
#: sourced by the jobs, never started
SOURCED = {"flashinfer_env.sh"}


def jobs() -> list[str]:
    (case,) = re.findall(r'^case "\$job" in ([a-z_|]+)\)$', (HERE / "run.sh").read_text(), re.M)
    return case.split("|")


def strays(d: Path, names: list[str]) -> set[str]:
    """What keeps the jobs `names` from being `d`'s job scripts: a job with no script, and a script that is no job."""
    named = {j + ".sh" for j in names}
    scripts = {p.name for p in d.glob("*.sh")} - SOURCED - {"run.sh"}
    return named ^ scripts


def test_the_runner_names_every_job_script_and_only_those():
    assert not strays(HERE, jobs()) and len(set(jobs())) == len(jobs())
    assert 'exec bash "$HERE/$job.sh" "$@"' in (HERE / "run.sh").read_text()


def test_a_script_that_is_no_job_and_a_job_with_no_script_are_strays(tmp_path):
    for name in [j + ".sh" for j in jobs()] + ["run.sh", *SOURCED]:
        (tmp_path / name).touch()
    assert not strays(tmp_path, jobs())
    (tmp_path / "other.sh").touch()
    (tmp_path / "verify.sh").unlink()
    assert strays(tmp_path, jobs()) == {"other.sh", "verify.sh"}


def test_the_readme_has_a_row_for_every_job():
    rows = re.findall(r"^\| `([a-z_]+)[ `]", (HERE / "README.md").read_text(), re.M)
    assert [r for r in rows if r in jobs()] == jobs()


def test_a_name_that_is_no_job_is_refused_with_the_usage():
    for args in ([], ["e2e"], ["flashinfer_env"]):
        r = subprocess.run(["bash", str(HERE / "run.sh"), *args], capture_output=True, text=True)
        assert r.returncode == 2 and r.stderr.startswith("usage: run.sh JOB") and not r.stdout


def test_a_job_gets_the_arguments_the_environment_and_the_exit_code(tmp_path):
    """The runner beside a stand-in job script: the job sees its arguments as given and the caller's variables, and its
    exit code is the runner's."""
    shutil.copy(HERE / "run.sh", tmp_path / "run.sh")
    (tmp_path / "verify.sh").write_text('printf "%s|" "$@" "$MODE"; exit 7\n')
    r = subprocess.run(["bash", str(tmp_path / "run.sh"), "verify", "a run", "48"], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "MODE": "smoke"})
    assert (r.returncode, r.stdout, r.stderr) == (7, "a run|48|smoke|", "")
