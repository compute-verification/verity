"""Each directory's runner, `run.sh JOB [ARGS...]`: a job is a script beside it, named without `.sh`, every one but those
the jobs source, and the directory's README has a table row a job.  A name that is no job is refused with the usage,
before anything starts."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

POUW = Path(__file__).resolve().parents[1]
#: each directory with a runner -> the scripts beside it its jobs source, never started
RUNNERS = {"served_h1": {"arch"}, "served_zk": set(), "hidden_zk": set()}


def jobs(d: str) -> list[str]:
    return sorted(p.stem for p in (POUW / d).glob("*.sh") if p.stem not in RUNNERS[d] | {"run"})


@pytest.mark.parametrize("d", RUNNERS)
def test_the_readme_has_a_row_for_every_job_and_no_other(d):
    rows = re.findall(r"^\| `([a-z0-9_]+)` \|", (POUW / d / "README.md").read_text(), re.M)
    assert sorted(rows) == jobs(d) and len(set(rows)) == len(rows)


@pytest.mark.parametrize("d", RUNNERS)
def test_the_usage_names_every_job(d):
    r = subprocess.run(["bash", str(POUW / d / "run.sh")], capture_output=True, text=True)
    assert r.returncode == 2 and r.stderr.split("JOB: one of ")[1].split(" (got")[0].split() == jobs(d)


@pytest.mark.parametrize("d", RUNNERS)
def test_a_name_that_is_no_job_is_refused_with_the_usage(d):
    job = jobs(d)[0]
    for args in ([], ["window"], ["run"], *([s] for s in RUNNERS[d]), [f"../{d}/{job}"], [f"{job}.sh"]):
        r = subprocess.run(["bash", str(POUW / d / "run.sh"), *args], capture_output=True, text=True)
        assert r.returncode == 2 and r.stderr.startswith("usage: run.sh JOB") and not r.stdout


@pytest.mark.parametrize("d", RUNNERS)
def test_a_job_gets_the_arguments_the_environment_and_the_exit_code(d, tmp_path):
    """The runner beside a stand-in job script: the job sees its arguments as given and the caller's variables, and its
    exit code is the runner's."""
    shutil.copy(POUW / d / "run.sh", tmp_path / "run.sh")
    (tmp_path / "stand_in.sh").write_text('printf "%s|" "$@" "$MODE"; exit 7\n')
    r = subprocess.run(["bash", str(tmp_path / "run.sh"), "stand_in", "--steps 4", "16"], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "MODE": "smoke"})
    assert (r.returncode, r.stdout, r.stderr) == (7, "--steps 4|16|smoke|", "")
