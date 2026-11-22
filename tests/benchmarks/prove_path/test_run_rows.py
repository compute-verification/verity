"""run.sh refuses a run that names no arm before anything is written, and outside a research run writes under the
clone."""
from __future__ import annotations

import subprocess
from pathlib import Path

RUN_SH = Path(__file__).resolve().parents[3] / "benchmarks" / "prove_path" / "run.sh"
REPO = RUN_SH.parents[2]


def test_a_run_naming_no_arm_is_refused_before_anything_is_written(tmp_path):
    got = subprocess.run(["env", "-u", "PP_TP", "-u", "PP_WEIGHTS_TIE", "-u", "OUT", f"RESEARCH_RUN_DIR={tmp_path}", "bash",
                          str(RUN_SH)], cwd=REPO, capture_output=True, text=True)
    assert got.returncode == 2 and "give PP_WEIGHTS_TIE" in got.stdout
    assert list(tmp_path.iterdir()) == []


def test_outside_a_research_run_the_outputs_go_under_the_clone_and_say_so(tmp_path):
    """With neither OUT nor RESEARCH_RUN_DIR set, run.sh's outputs go to out/prove_path under the clone (git ignores out/), said
    in one line, and no variable is unbound."""
    lines = RUN_SH.read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith("REPO=$(pwd)"))
    j = next(k for k in range(i, len(lines)) if lines[k].startswith("mkdir -p $O"))
    got = subprocess.run(["env", "-u", "OUT", "-u", "RESEARCH_RUN_DIR", "bash", "-c",
                          "set -u\n" + "\n".join(lines[i:j + 1]) + '\necho "O=$O"'], cwd=tmp_path, capture_output=True, text=True)
    assert got.returncode == 0, got.stderr
    assert got.stdout.splitlines() == [f"OUT unset outside a research run: the outputs go to {tmp_path}/out/prove_path",
                                       f"O={tmp_path}/out/prove_path"]
    assert (tmp_path / "out" / "prove_path").is_dir()


