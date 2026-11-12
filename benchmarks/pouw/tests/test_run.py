"""The front door (``run.py``): which runner a NAME starts, what it hands it, and what it says to a NAME it has none for."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from conftest import load

R = load("run.py")


def test_a_directorys_runner_is_its_run_sh_else_its_run_py(tmp_path):
    for file in ("job/run.sh", "script/run.py", "both/run.sh", "both/run.py", "neither/tool.py", "run.sh"):
        (tmp_path / file).parent.mkdir(exist_ok=True)
        (tmp_path / file).touch()
    assert R.runners(tmp_path) == {"both": ["bash", str(tmp_path / "both/run.sh")],
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
