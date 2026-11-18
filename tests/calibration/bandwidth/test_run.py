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
#: `write_gpu`'s build on an RTX PRO 6000 when the card's name chose it, word for word but the build directory
RTX_BUILD = ["-O3", "-std=c++17", "-arch=sm_120", "-o", "B/bwwrite", str(HERE / "write.cu")]
#: each card file's short name with the plain architecture the scripts wrote for that card by its name
WROTE = [("rtx-pro-6000", "sm_120"), ("h200", "sm_90"), ("h100", "sm_90")]


def build_line(tmp_path: Path, card: str | None, **env: str) -> tuple[subprocess.CompletedProcess, list[str] | None]:
    """`run.sh write_gpu` for CARD ``card`` (None: not given) with an `nvcc` stand-in (an exported function, ahead of
    any on PATH): the arguments nvcc was handed (None when it wasn't), its build failing so nothing runs."""
    argv = tmp_path / "nvcc.argv"
    fns = {"BASH_FUNC_nvcc%%": f'() {{ printf "%s\\n" "$@" > {argv}; return 1; }}'}
    env = {"NVCC": "nvcc", **({} if card is None else {"CARD": card}), **env}
    done = subprocess.run(["bash", str(HERE / "run.sh"), "write_gpu"], cwd=ROOT, capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "OUT": str(tmp_path / "out"), **fns, **env})
    if not argv.is_file():
        return done, None
    got = argv.read_text().splitlines()
    return done, [("B/" + Path(a).name if a.endswith("/bwwrite") else a) for a in got]


@pytest.mark.parametrize("card,plain", WROTE)
def test_write_gpus_build_is_the_one_the_cards_name_chose_word_for_word(tmp_path, card, plain):
    """The card file's `arch` less its `a` is the plain architecture the script wrote for that card by its name."""
    done, got = build_line(tmp_path, card, ARCH="77a")
    assert (done.returncode, got) == (4, [a.replace("sm_120", plain) for a in RTX_BUILD])


def test_the_plain_architecture_of_every_card_file_is_its_arch_less_the_a_and_never_the_a_target():
    """The one rule, over every GPU card file there is: the three the name branches knew give what was written."""
    from verity.device import load_card, skus

    cards = [load_card(sku, directory=ROOT / "calibration" / "hardware" / "cards")
             for sku in skus("gpu", directory=ROOT / "calibration" / "hardware" / "cards")]
    plain = {card["short"]: card["arch"].removesuffix("a") for card in cards if card.get("short")}
    assert all(plain[short] == wrote for short, wrote in WROTE)
    assert all(card["arch"].endswith("a") and not plain[card["short"]].endswith("a") for card in cards if card.get("short"))


def test_write_gpu_with_no_card_or_a_short_name_no_card_file_has_is_refused_before_the_build(tmp_path):
    done, got = build_line(tmp_path, None)
    assert (done.returncode, got) == (3, None)
    assert "write_gpu: CARD is not given: the short name of the card's file under calibration/hardware/cards\n" in done.stderr
    assert (tmp_path / "out" / "rc.txt").read_text() == "3\n"
    done, got = build_line(tmp_path, "l40s")
    assert (done.returncode, got) == (3, None)
    assert "have `short` 'l40s'" in done.stderr and (tmp_path / "out" / "rc.txt").read_text() == "3\n"


def test_write_gpu_with_no_toolkit_13_0_is_refused_before_the_build(tmp_path):
    done, got = build_line(tmp_path, "h200", NVCC="", CUDA_HOME="")
    assert (done.returncode, got) == (3, None)
    assert "no CUDA toolkit 13.0: set CUDA_HOME" in done.stderr
    assert (tmp_path / "out" / "rc.txt").read_text() == "3\n"


def toolkit(tmp_path: Path) -> Path:
    """A stand-in toolkit whose nvcc writes its arguments to given.argv and fails."""
    home = tmp_path / "toolkit"
    (home / "bin").mkdir(parents=True)
    nvcc = home / "bin" / "nvcc"
    nvcc.write_text(f'#!/bin/sh\n[ "$1" = --version ] && exit 0\nprintf "%s\\n" "$@" > {tmp_path}/given.argv\nexit 1\n')
    nvcc.chmod(0o755)
    return home


def test_write_gpu_takes_the_toolkit_it_is_given(tmp_path):
    """With no NVCC, the compiler is `$CUDA_HOME/bin/nvcc`: a stand-in there is what the build runs."""
    done, _ = build_line(tmp_path, "h200", NVCC="", CUDA_HOME=str(toolkit(tmp_path)))
    assert done.returncode == 4 and "-arch=sm_90" in (tmp_path / "given.argv").read_text().splitlines()


def chip(tmp_path: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(HERE / "run.sh"), "chip", *args], cwd=ROOT, capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "RESEARCH_RUN_DIR": str(tmp_path), **env})


@pytest.mark.parametrize("env", [{}, {"CUDA_HOME": ""}])
def test_chip_with_no_toolkit_given_is_refused_in_one_line_before_anything_is_built(tmp_path, env):
    done = chip(tmp_path, CARD="rtx-pro-6000", **env)
    assert (done.returncode, done.stdout) == (3, "")
    assert done.stderr == ("REFUSED: chip: no CUDA toolkit given: set CUDA_HOME to its directory (the machine's own: "
                           "python3 benchmarks/kernels/machine.py toolkit)\n")
    assert not list((tmp_path / "out").iterdir())


def test_chip_with_no_card_or_a_short_name_no_card_file_has_is_refused_before_anything_is_built(tmp_path):
    home = str(toolkit(tmp_path))
    done = chip(tmp_path, CUDA_HOME=home)
    assert (done.returncode, done.stdout) == (3, "")
    assert done.stderr == "REFUSED: chip: CARD is not given: the short name of the card's file under calibration/hardware/cards\n"
    done = chip(tmp_path, CUDA_HOME=home, CARD="l40s")
    assert done.returncode == 3 and "have `short` 'l40s'" in done.stderr
    assert not list((tmp_path / "out").iterdir())


def test_chips_build_on_an_rtx_pro_6000_is_the_plain_sm_120_it_wrote(tmp_path):
    done = chip(tmp_path, CUDA_HOME=str(toolkit(tmp_path)), CARD="rtx-pro-6000", ARCH="77a")
    assert done.returncode == 3
    assert (tmp_path / "given.argv").read_text().splitlines()[:5] == ["-O3", "-std=c++17", "-arch=sm_120", "-Xptxas", "-v"]
    assert "flags: -O3 -std=c++17 -arch=sm_120 -Xptxas -v\n" in (tmp_path / "out" / "build.txt").read_text()


@pytest.mark.parametrize("name,said", [("NVIDIA H200", "whose card file states sm_90a, not the sm_120a of CARD=rtx-pro-6000"),
                                       ("NVIDIA L40S", "0 GPU card files under ")])
def test_a_leased_card_of_another_architecture_or_of_no_card_file_is_refused_before_bwchip_runs(tmp_path, name, said):
    """What the probe needs of the leased card is the architecture bwchip was built for, read from the card's file."""
    fns = {"BASH_FUNC_nvidia-smi%%": f'() {{ echo "{name}"; }}'}
    done = chip(tmp_path, "leased", str(tmp_path / "none"), CARD="rtx-pro-6000", ARCH="sm_120a", GPU_LEASE_UUID="GPU-0", **fns)
    assert done.returncode == 3 and done.stdout.startswith(f"REFUSED: GPU GPU-0 is '{name}'") and said in done.stdout
    assert not (tmp_path / "out" / "ur.jsonl").exists()


def test_the_script_writes_no_toolkit_path_of_its_own_reads_no_cards_name_to_choose_and_writes_no_architecture():
    text = (HERE / "run.sh").read_text()
    assert "/usr/local" not in text and "-arch=sm_" not in text and "*H200*" not in text and '"RTX PRO 6000"*' not in text
