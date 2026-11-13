"""`run.sh`, the directory's runner: a name runs that name's script with the arguments after it, and any other name is
refused with the usage."""

import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
USAGE = "usage: verity bench bandwidth chip|write_gpu|write_host [ARGS]\n"
SCRIPTS = {"chip": "run_chip.sh", "write_gpu": "run_write_gpu.sh", "write_host": "run_write_host.sh"}


def run(directory: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(directory / "run.sh"), *args], capture_output=True, text=True)


@pytest.fixture
def stand_in(tmp_path: Path) -> Path:
    """The runner beside scripts that only say what they were handed: no card and no compiler."""
    shutil.copy(HERE / "run.sh", tmp_path / "run.sh")
    for name, script in SCRIPTS.items():
        (tmp_path / script).write_text(f'echo "{name}: $*"\n')
    return tmp_path


@pytest.mark.parametrize("name", SCRIPTS)
def test_a_name_runs_its_script_with_the_arguments_after_it(stand_in, name):
    done = run(stand_in, name, "--one", "two words")
    assert (done.returncode, done.stdout, done.stderr) == (0, f"{name}: --one two words\n", "")


def test_the_runner_exits_with_its_scripts_status(stand_in):
    (stand_in / "run_chip.sh").write_text("exit 3\n")
    assert run(stand_in, "chip").returncode == 3


@pytest.mark.parametrize("args", [(), ("run_chip.sh",), ("write",)])
def test_no_name_or_another_name_is_refused_with_the_usage(stand_in, args):
    done = run(stand_in, *args)
    assert (done.returncode, done.stdout, done.stderr) == (2, "", USAGE)


def test_every_name_of_the_usage_has_its_script_in_the_directory():
    assert run(HERE).stderr == USAGE
    assert USAGE.split()[4].split("|") == list(SCRIPTS)
    assert all((HERE / script).is_file() for script in SCRIPTS.values())
