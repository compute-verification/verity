"""Each directory's runner, `run.sh JOB [ARGS...]`: a job is a script beside it, named without `.sh`, every one but those
the jobs source, and the directory's README has a table row a job, but a job whose name holds a term the Glossary
retires, which it describes under the table, and a cost job that is ahead of the README (`AHEAD`).  A name that is no job
is refused with the usage, before anything starts."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

POUW = Path(__file__).resolve().parents[1]
#: each directory with a runner -> the scripts beside it its jobs source, never started
RUNNERS = {"served_h1": {"arch"}, "served_zk": set(), "hidden_zk": set(), "h1_prove": set(), "h1_lanes": set()}
#: the directories whose `run.py` is a step of their jobs, beside the runner
STEP_BESIDE = ("h1_prove", "h1_lanes")
#: each directory -> how many of its jobs the README describes without a row
NO_ROW = {"served_h1": 1, "h1_prove": 1}
#: the directories where a job named `*_cost` may be beside the runner before the README describes it: a branch that adds
#: one need not hold the README.  The README's test fails once it describes a job this still excuses
AHEAD = ("h1_prove", "h1_lanes")


def jobs(d: str) -> list[str]:
    return sorted(p.stem for p in (POUW / d).glob("*.sh") if p.stem not in RUNNERS[d] | {"run"})


def described(d: str) -> list[str]:
    """The jobs `d`'s README has to describe."""
    return [j for j in jobs(d) if not (d in AHEAD and j.endswith("_cost"))]


@pytest.mark.parametrize("d", RUNNERS)
def test_the_readme_has_a_row_for_every_job_and_no_other(d):
    text = (POUW / d / "README.md").read_text()
    rows = re.findall(r"^\| `([a-z0-9_]+)` \|", text, re.M)
    assert set(rows) <= set(described(d)) and len(set(rows)) == len(rows)
    assert len(described(d)) - len(rows) == NO_ROW.get(d, 0) == text.count("One more job has no row")


@pytest.mark.parametrize("d", RUNNERS)
def test_the_usage_names_every_job(d):
    r = subprocess.run(["bash", str(POUW / d / "run.sh")], capture_output=True, text=True)
    assert r.returncode == 2 and sorted(r.stderr.split("JOB: one of ")[1].split(" (got")[0].split()) == jobs(d)


@pytest.mark.parametrize("d", AHEAD)
def test_a_cost_job_may_be_ahead_of_the_readme_and_no_other_job(d, tmp_path, monkeypatch):
    """A copy of the directory's README and scripts with one more job, `JOB_cost`, that the README has no row for: the
    README's test and the usage's still pass, whichever of the two names the shell lists first.  With another job the
    README has no row for, the README's test fails."""
    (tmp_path / d).mkdir()
    for p in (POUW / d / "README.md", *(POUW / d).glob("*.sh")):
        shutil.copy(p, tmp_path / d)
    monkeypatch.setattr(sys.modules[__name__], "POUW", tmp_path)
    (tmp_path / d / f"{jobs(d)[0]}_cost.sh").write_text("exit 0\n")
    test_the_readme_has_a_row_for_every_job_and_no_other(d)
    test_the_usage_names_every_job(d)
    (tmp_path / d / "other.sh").write_text("exit 0\n")
    with pytest.raises(AssertionError):
        test_the_readme_has_a_row_for_every_job_and_no_other(d)


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


@pytest.mark.parametrize("d", STEP_BESIDE)
def test_the_runner_execs_only_a_job_script_beside_it(d, tmp_path):
    """The runner's one command is the exec of `JOB.sh` beside it, so a job runs its own script and nothing else, and
    the `run.py` beside it is never started, under any name."""
    lines = (POUW / d / "run.sh").read_text().splitlines()
    code = [s.strip() for s in lines if s.strip() and not s.lstrip().startswith("#")]
    starts = re.compile(r"(?<![\w./$-])(exec|eval|source|bash|sh|python3?|uv)\b|^\. ")
    assert [s for s in code if starts.search(s)] == ['exec bash "$HERE/$job.sh" "$@"']
    shutil.copy(POUW / d / "run.sh", tmp_path / "run.sh")
    (tmp_path / "run.py").write_text("import pathlib; pathlib.Path(__file__).with_name('started').touch()\n")
    (tmp_path / "job.sh").write_text("exit 0\n")
    for name in ("run", "run.py", "run_py"):
        r = subprocess.run(["bash", str(tmp_path / "run.sh"), name], capture_output=True, text=True)
        assert r.returncode == 2 and r.stderr.startswith("usage: run.sh JOB")
    assert not (tmp_path / "started").exists()
