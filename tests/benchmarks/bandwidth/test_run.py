"""`run.sh`, the directory's runner: a name runs that name's script with the arguments after it, and any other name is
refused with the usage."""

import shutil
import subprocess
from pathlib import Path

import pytest

HERE = (Path(__file__).resolve().parent.parent.parent.parent / "benchmarks" / "bandwidth")
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


ROOT = HERE.parent.parent
#: run_write_gpu.sh's build on an RTX PRO 6000 before ARCH, word for word but the build directory
RTX_BUILD = ["-O3", "-std=c++17", "-arch=sm_120", "-o", "B/bwwrite", str(HERE / "write.cu")]


def build_line(tmp_path: Path, card: str, **env: str) -> tuple[subprocess.CompletedProcess, list[str] | None]:
    """run_write_gpu.sh with `nvcc` and `nvidia-smi` stand-ins (exported functions, ahead of any on PATH): the card
    `nvidia-smi` names, and the arguments nvcc was handed (None when it wasn't), its build failing so nothing runs."""
    argv = tmp_path / "nvcc.argv"
    fns = {"BASH_FUNC_nvcc%%": f'() {{ printf "%s\\n" "$@" > {argv}; return 1; }}',
           "BASH_FUNC_nvidia-smi%%": f'() {{ echo "{card}"; }}'}
    done = subprocess.run(["bash", str(HERE / "run_write_gpu.sh")], cwd=ROOT, capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "OUT": str(tmp_path / "out"), "NVCC": "nvcc", **fns, **env})
    if not argv.is_file():
        return done, None
    got = argv.read_text().splitlines()
    return done, [("B/" + Path(a).name if a.endswith("/bwwrite") else a) for a in got]


@pytest.mark.parametrize("card,env", [("NVIDIA RTX PRO 6000 Blackwell Server Edition", {}),
                                      ("NVIDIA H200", {"ARCH": "120a"})])
def test_the_build_at_120a_is_the_rtx_pro_6000_build_word_for_word(tmp_path, card, env):
    done, got = build_line(tmp_path, card, **env)
    assert (done.returncode, got) == (4, RTX_BUILD)


@pytest.mark.parametrize("card,env", [("NVIDIA H200", {}), ("NVIDIA H100 80GB HBM3", {}),
                                      ("NVIDIA RTX PRO 6000 Blackwell Server Edition", {"ARCH": "90a"})])
def test_hopper_or_arch_90a_builds_for_sm_90(tmp_path, card, env):
    done, got = build_line(tmp_path, card, **env)
    assert (done.returncode, got) == (4, [a.replace("sm_120", "sm_90") for a in RTX_BUILD])


def test_a_card_the_name_does_not_tell_is_refused_before_the_build(tmp_path):
    done, got = build_line(tmp_path, "NVIDIA L40S")
    assert (done.returncode, got) == (3, None)
    assert "write_gpu: ARCH (90a, 120a) is not given and the card's name does not tell it" in done.stderr
    assert (tmp_path / "out" / "rc.txt").read_text() == "3\n"
