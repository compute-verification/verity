"""`run.sh`, the directory's one runner (`verity calibrate bandwidth chip|write_gpu|write_host`): a name runs that
probe, any other name is refused with the usage, and `write_gpu`'s build is the card's."""

import subprocess
from pathlib import Path

import pytest

HERE = (Path(__file__).resolve().parent.parent.parent.parent / "calibration" / "bandwidth")
USAGE = "usage: verity calibrate bandwidth chip|write_gpu|write_host\n"
NAMES = ("chip", "write_gpu", "write_host")


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(HERE / "run.sh"), *args], capture_output=True, text=True)


@pytest.mark.parametrize("args", [(), ("run_chip.sh",), ("write",)])
def test_no_name_or_another_name_is_refused_with_the_usage(args):
    done = run(*args)
    assert (done.returncode, done.stdout, done.stderr) == (2, "", USAGE)


def test_the_usages_names_are_the_probes_and_the_old_scripts_are_gone():
    assert USAGE.split()[4].split("|") == list(NAMES)
    assert all(f"{name}() {{" in (HERE / "run.sh").read_text() for name in NAMES)
    assert not [p.name for p in HERE.glob("run_*.sh")]


def test_each_probe_writes_its_result_through_result_py_and_chip_leases_through_run_sh():
    text = (HERE / "run.sh").read_text()
    assert 'result.py chip "$O" "$rc"' in text and 'result.py" write_gpu "$O" "$rc"' in text
    assert 'result.py" write_host "$O"' in text and 'bash "$PWD/run.sh" chip leased "$S"' in text


ROOT = HERE.parent.parent
#: `write_gpu`'s build on an RTX PRO 6000 before ARCH, word for word but the build directory
RTX_BUILD = ["-O3", "-std=c++17", "-arch=sm_120", "-o", "B/bwwrite", str(HERE / "write.cu")]


def build_line(tmp_path: Path, card: str, **env: str) -> tuple[subprocess.CompletedProcess, list[str] | None]:
    """`run.sh write_gpu` with `nvcc` and `nvidia-smi` stand-ins (exported functions, ahead of any on PATH): the card
    `nvidia-smi` names, and the arguments nvcc was handed (None when it wasn't), its build failing so nothing runs."""
    argv = tmp_path / "nvcc.argv"
    fns = {"BASH_FUNC_nvcc%%": f'() {{ printf "%s\\n" "$@" > {argv}; return 1; }}',
           "BASH_FUNC_nvidia-smi%%": f'() {{ echo "{card}"; }}'}
    done = subprocess.run(["bash", str(HERE / "run.sh"), "write_gpu"], cwd=ROOT, capture_output=True, text=True,
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


def test_write_gpu_with_no_toolkit_13_0_is_refused_before_the_build(tmp_path):
    done, got = build_line(tmp_path, "NVIDIA H200", NVCC="", CUDA_HOME="")
    assert (done.returncode, got) == (3, None)
    assert "no CUDA toolkit 13.0: set CUDA_HOME" in done.stderr
    assert (tmp_path / "out" / "rc.txt").read_text() == "3\n"


def test_write_gpu_takes_the_toolkit_it_is_given(tmp_path):
    """With no NVCC, the compiler is `$CUDA_HOME/bin/nvcc`: a stand-in there is what the build runs."""
    home = tmp_path / "toolkit"
    (home / "bin").mkdir(parents=True)
    nvcc = home / "bin" / "nvcc"
    nvcc.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {tmp_path}/given.argv\nexit 1\n')
    nvcc.chmod(0o755)
    done, _ = build_line(tmp_path, "NVIDIA H200", NVCC="", CUDA_HOME=str(home))
    assert done.returncode == 4 and "-arch=sm_90" in (tmp_path / "given.argv").read_text().splitlines()


@pytest.mark.parametrize("env", [{}, {"CUDA_HOME": ""}])
def test_chip_with_no_toolkit_given_is_refused_in_one_line_before_anything_is_built(tmp_path, env):
    done = subprocess.run(["bash", str(HERE / "run.sh"), "chip"], cwd=ROOT, capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "RESEARCH_RUN_DIR": str(tmp_path), **env})
    assert (done.returncode, done.stdout) == (3, "")
    assert done.stderr == ("REFUSED: chip: no CUDA toolkit given: set CUDA_HOME to its directory (the machine's own: "
                           "python3 benchmarks/kernels/machine.py toolkit)\n")
    assert not list((tmp_path / "out").iterdir())


def test_the_script_writes_no_toolkit_path_of_its_own():
    assert "/usr/local" not in (HERE / "run.sh").read_text()
