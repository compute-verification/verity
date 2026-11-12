"""benchmarks/pouw/served_h1/run.sh, the directory's runner: a job is a script beside it, named without `.sh`, every one
but the one the jobs source, and the README's table has a row a job.  A name that is no job is refused with the usage,
before anything starts."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "served_h1"
#: sourced by the jobs, never started
SOURCED = {"arch"}


def jobs() -> list[str]:
    return sorted(p.stem for p in HERE.glob("*.sh") if p.stem not in SOURCED | {"run"})


def test_the_readme_has_a_row_for_every_job_and_no_other():
    rows = re.findall(r"^\| `([a-z0-9_]+)` \|", (HERE / "README.md").read_text(), re.M)
    assert sorted(rows) == jobs() and len(set(rows)) == len(rows)


def test_the_usage_names_every_job():
    r = subprocess.run(["bash", str(HERE / "run.sh")], capture_output=True, text=True)
    assert r.returncode == 2 and r.stderr.split("JOB: one of ")[1].split(" (got")[0].split() == jobs()


def test_a_name_that_is_no_job_is_refused_with_the_usage():
    for args in ([], ["window"], ["arch"], ["run"], ["../served_h1/serve"], ["serve.sh"]):
        r = subprocess.run(["bash", str(HERE / "run.sh"), *args], capture_output=True, text=True)
        assert r.returncode == 2 and r.stderr.startswith("usage: run.sh JOB") and not r.stdout


def test_a_job_gets_the_arguments_the_environment_and_the_exit_code(tmp_path):
    """The runner beside a stand-in job script: the job sees its arguments as given and the caller's variables, and its
    exit code is the runner's."""
    shutil.copy(HERE / "run.sh", tmp_path / "run.sh")
    (tmp_path / "serve.sh").write_text('printf "%s|" "$@" "$MODE"; exit 7\n')
    r = subprocess.run(["bash", str(tmp_path / "run.sh"), "serve", "--steps 4", "16"], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "MODE": "smoke"})
    assert (r.returncode, r.stdout, r.stderr) == (7, "--steps 4|16|smoke|", "")
