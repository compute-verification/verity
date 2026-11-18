"""`run.sh`, the directory's one runner (`verity calibrate rate_squaring sq|ifma`): a name runs that form, any other
name or more arguments are refused with the usage, and `sq`'s build is the card's."""

import subprocess
from pathlib import Path

import pytest

HERE = (Path(__file__).resolve().parent.parent.parent.parent / "calibration" / "rate_squaring")
USAGE = "usage: verity calibrate rate_squaring sq|ifma\n"
NAMES = ("sq", "ifma")


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(HERE / "run.sh"), *args], capture_output=True, text=True)


@pytest.mark.parametrize("args", [(), ("sq.sh",), ("gamma",), ("sq", "--one"), ("ifma", "two")])
def test_no_name_another_name_or_more_arguments_are_refused_with_the_usage(args):
    done = run(*args)
    assert (done.returncode, done.stdout, done.stderr) == (2, "", USAGE)


def test_the_usages_names_are_the_forms_and_the_old_scripts_are_gone():
    assert USAGE.split()[4].split("|") == list(NAMES)
    assert all(f"{name}() {{" in (HERE / "run.sh").read_text() for name in NAMES)
    assert not (HERE / "sq.sh").exists() and not (HERE / "ifma.sh").exists()


def test_each_form_writes_its_result_through_result_py():
    text = (HERE / "run.sh").read_text()
    assert 'result.py sq "$O"' in text and 'result.py ifma "$O"' in text


ROOT = HERE.parent.parent
#: `sq`'s build on an RTX PRO 6000 before ARCH, word for word but the build directory
RTX_BUILD = ["-O3", "-gencode", "arch=compute_120a,code=sm_120a", "-std=c++17", "-Xptxas", "-v", "-DSQ_BITS=512",
             "-DSQ_C=569", "-o", "B/sq", "calibration/rate_squaring/sq.cu"]


def build_line(tmp_path: Path, card: str, **env: str) -> tuple[subprocess.CompletedProcess, list[str] | None]:
    """`run.sh sq` with `nvcc` and `nvidia-smi` stand-ins (exported functions, ahead of the toolkit it puts on PATH) and
    CUDA_HOME a stand-in toolkit, as on a machine with no record: the card `nvidia-smi` names, and the arguments nvcc
    was handed (None when it wasn't), its build failing so nothing runs."""
    argv = tmp_path / "nvcc.argv"
    fns = {"BASH_FUNC_nvcc%%": f'() {{ printf "%s\\n" "$@" > {argv}; return 1; }}',
           "BASH_FUNC_nvidia-smi%%": f'() {{ echo "{card}"; }}'}
    env = {"CUDA_HOME": str(tmp_path / "toolkit"), **env}
    done = subprocess.run(["bash", str(HERE / "run.sh"), "sq"], cwd=ROOT, capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "RESEARCH_RUN_DIR": str(tmp_path), **fns, **env})
    if not argv.is_file():
        return done, None
    return done, [("B/sq" if a.endswith("/sq") else a) for a in argv.read_text().splitlines()]


@pytest.mark.parametrize("card,env", [("NVIDIA RTX PRO 6000 Blackwell Server Edition", {}),
                                      ("NVIDIA H200", {"ARCH": "120a"})])
def test_the_build_at_120a_is_the_rtx_pro_6000_build_word_for_word(tmp_path, card, env):
    done, got = build_line(tmp_path, card, **env)
    assert (done.returncode, got) == (4, RTX_BUILD)


@pytest.mark.parametrize("card,env", [("NVIDIA H200", {}), ("NVIDIA H100 80GB HBM3", {}),
                                      ("NVIDIA RTX PRO 6000 Blackwell Server Edition", {"ARCH": "90a"})])
def test_hopper_or_arch_90a_builds_for_sm_90a(tmp_path, card, env):
    done, got = build_line(tmp_path, card, **env)
    assert (done.returncode, got) == (4, [a.replace("120a", "90a") for a in RTX_BUILD])


def test_no_toolkit_13_0_is_refused_before_the_build(tmp_path):
    done, got = build_line(tmp_path, "NVIDIA H200", CUDA_HOME="")
    assert (done.returncode, got) == (3, None)
    assert "no CUDA toolkit 13.0: set CUDA_HOME" in done.stderr


def test_the_script_writes_no_toolkit_path_of_its_own():
    assert "/usr/local" not in (HERE / "run.sh").read_text()


def test_a_card_the_name_does_not_tell_is_refused_before_the_build(tmp_path):
    done, got = build_line(tmp_path, "NVIDIA L40S")
    assert (done.returncode, got) == (3, None)
    assert "sq: ARCH (90a, 120a) is not given and the card's name does not tell it" in done.stderr
