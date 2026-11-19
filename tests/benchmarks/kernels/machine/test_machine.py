"""machine.py: the kit's toolkit and card-architecture readers printed for a shell script, exit 3 where the machine's
`[properties]` list no toolkit at the version, and a refusal by name otherwise (the readers' own tests are the kit's)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "benchmarks" / "kernels" / "machine.py"
ONE_CARD = ROOT / "tests" / "benchmarks" / "harness" / "fixtures" / "machines" / "one-card.toml"


def test_the_command_prints_the_directory_exits_3_on_a_version_not_listed_and_refuses_an_unknown_card(tmp_path):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "m.toml").write_text(ONE_CARD.read_text())
    env = {k: v for k, v in os.environ.items() if k != "CUDA_HOME"} | {"RESEARCH_RUN_DIR": str(tmp_path),
                                                                        "RESEARCH_MACHINE": "m"}
    run = lambda *a: subprocess.run([sys.executable, str(SCRIPT), *a], env=env, capture_output=True, text=True)  # noqa: E731
    ok, absent = run("toolkit", "13.0"), run("toolkit", "13.1")
    arch, refused, own = run("arch", "NVIDIA H200"), run("arch", "NVIDIA A100-SXM4-80GB"), run("arch")
    gencode, no_target = run("gencode", "sm_120a"), run("gencode", "sm_77a")
    short, no_short = run("card-arch", "rtx-pro-6000"), run("card-arch", "l40s")
    assert (ok.returncode, ok.stdout) == (0, "/usr/local/cuda-13.0\n")
    assert absent.returncode == 3 and absent.stdout == "" and "list no CUDA toolkit 13.1" in absent.stderr
    assert (arch.returncode, arch.stdout) == (0, "sm_90a\n")
    assert (own.returncode, own.stdout) == (0, "sm_90a\n")
    assert refused.returncode == 1 and refused.stderr.startswith("0 GPU card files under ")
    assert (short.returncode, short.stdout) == (0, "sm_120a\n")
    assert no_short.returncode == 1 and no_short.stderr.startswith("0 GPU card files under ") and "`short` 'l40s'" in no_short.stderr
    assert (gencode.returncode, gencode.stdout) == (0, "arch=compute_120a,code=sm_120a\n")
    assert no_target.returncode == 1 and no_target.stderr.startswith("target 'sm_77a': no architecture 'sm_77a'")


def test_a_private_run_directory_without_the_record_is_told_the_file_looked_for_and_to_carry_it(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "CUDA_HOME"} | {"RESEARCH_RUN_DIR": str(tmp_path),
                                                                        "RESEARCH_MACHINE": "m"}
    run = lambda *a: subprocess.run([sys.executable, str(SCRIPT), *a], env=env, capture_output=True, text=True)  # noqa: E731
    for r in (run("toolkit", "13.0"), run("toolkit"), run("arch")):
        assert r.returncode == 1 and r.stdout == ""
        assert "$RESEARCH_RUN_DIR/inputs/$RESEARCH_MACHINE.toml" in r.stderr and str(tmp_path / "inputs" / "m.toml") in r.stderr
        assert "a private run directory must carry the record" in r.stderr
    other, usage, card_file = run("arch", "NVIDIA A100-SXM4-80GB"), run("no-such-reader"), run("gencode", "sm_90a")
    assert other.returncode == 1 and usage.returncode == 1 and "private run directory" not in other.stderr
    assert "machine.py: looked for the run's machine record" not in usage.stderr
    assert (card_file.returncode, card_file.stdout) == (0, "arch=compute_90a,code=sm_90a\n")
