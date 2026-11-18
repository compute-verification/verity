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
#: `sq`'s build on an RTX PRO 6000 when the card's name chose it, word for word but the build directory
RTX_BUILD = ["-O3", "-gencode", "arch=compute_120a,code=sm_120a", "-std=c++17", "-Xptxas", "-v", "-DSQ_BITS=512",
             "-DSQ_C=569", "-o", "B/sq", "calibration/rate_squaring/sq.cu"]


def build_line(tmp_path: Path, card: str | None, **env: str) -> tuple[subprocess.CompletedProcess, list[str] | None]:
    """`run.sh sq` for CARD ``card`` (None: not given) with an `nvcc` stand-in (an exported function, ahead of the
    toolkit it puts on PATH) and CUDA_HOME a stand-in toolkit, as on a machine with no record: the arguments nvcc was
    handed (None when it wasn't), its build failing so nothing runs."""
    argv = tmp_path / "nvcc.argv"
    fns = {"BASH_FUNC_nvcc%%": f'() {{ printf "%s\\n" "$@" > {argv}; return 1; }}'}
    env = {"CUDA_HOME": str(tmp_path / "toolkit"), **({} if card is None else {"CARD": card}), **env}
    done = subprocess.run(["bash", str(HERE / "run.sh"), "sq"], cwd=ROOT, capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "RESEARCH_RUN_DIR": str(tmp_path), **fns, **env})
    if not argv.is_file():
        return done, None
    return done, [("B/sq" if a.endswith("/sq") else a) for a in argv.read_text().splitlines()]


@pytest.mark.parametrize("card,target", [("rtx-pro-6000", "120a"), ("h200", "90a"), ("h100", "90a")])
def test_each_cards_build_is_the_one_its_name_chose_word_for_word(tmp_path, card, target):
    """The -gencode the card's file gives through the kit is the string the script wrote for that card by its name."""
    done, got = build_line(tmp_path, card, ARCH="77a")
    assert (done.returncode, got) == (4, [a.replace("120a", target) for a in RTX_BUILD])


def test_a_card_whose_file_states_an_architecture_the_kit_knows_is_built_for_it(tmp_path):
    """The script has no list of cards: a B200, which the name branches refused, builds for its file's sm_100a."""
    done, got = build_line(tmp_path, "b200")
    assert (done.returncode, got) == (4, [a.replace("120a", "100a") for a in RTX_BUILD])


def test_no_toolkit_13_0_is_refused_before_the_build(tmp_path):
    done, got = build_line(tmp_path, "h200", CUDA_HOME="")
    assert (done.returncode, got) == (3, None)
    assert "no CUDA toolkit 13.0: set CUDA_HOME" in done.stderr


def test_the_script_writes_no_toolkit_path_of_its_own():
    assert "/usr/local" not in (HERE / "run.sh").read_text()


def test_no_card_or_a_short_name_no_card_file_has_is_refused_before_the_build(tmp_path):
    done, got = build_line(tmp_path, None)
    assert (done.returncode, got) == (3, None)
    assert "\nsq: CARD is not given: the short name of the card's file under calibration/hardware/cards\n" in done.stderr
    done, got = build_line(tmp_path, "l40s")
    assert (done.returncode, got) == (3, None)
    assert "0 GPU card files under " in done.stderr and "have `short` 'l40s'" in done.stderr


def test_the_script_reads_no_cards_name_and_takes_no_arch():
    text = (HERE / "run.sh").read_text()
    assert "query-gpu=name" not in text and "${ARCH" not in text
