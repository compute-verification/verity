"""run.sh's single-GPU Commit (``PP_LLAMA``) is of the row ``LROW`` names, the bf16 RTX PRO 6000 B4 row unless it names
another, and its workload is the stored one of that name, in whichever stored-workload directory's SHA256SUMS lists it:
the served run's for the bf16 row, the integration's for the int8 H200 row `a5_llama1b_h200.sh` leaves. A name neither
lists is refused before anything is written, as an art: row dir is."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

RUN_SH = Path(__file__).resolve().parents[1] / "run.sh"
REPO = RUN_SH.parents[2]
BF16 = "llama32-1b__bf16__rtxpro6000__tp1__b4__i256__o32__mixed__greedy__bi-eager"
INT8_H200 = "llama32-1b__int8__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager"


def lookup() -> str:
    """run.sh's lines from ``LROW=`` through the refusal's ``fi``."""
    lines = RUN_SH.read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith("LROW="))
    j = next(k for k in range(i, len(lines)) if lines[k].startswith("fi"))
    return "\n".join(lines[i:j + 1]) + "\n"


@pytest.mark.parametrize(("row", "workload"), [
    (None, f"adapters/vllm/tools/served_run/workloads/{BF16}.json"),
    (INT8_H200, f"adapters/vllm/workloads/{INT8_H200}.json"),
])
def test_a_rows_workload_is_the_stored_one_of_its_name(row, workload):
    env = ["env", "-u", "LROW", "-u", "PP_TP", "-u", "PP_SERVE_RUN", "PP_LLAMA=/row", *([f"LROW={row}"] if row else [])]
    got = subprocess.run([*env, "bash", "-c", lookup() + 'echo "$LWL"'], cwd=REPO, capture_output=True, text=True)
    assert got.returncode == 0, got.stdout + got.stderr
    assert got.stdout.strip() == workload


def test_a_row_no_stored_workload_names_is_refused_before_anything_is_written(tmp_path):
    unset = [a for v in ("PP_TP", "PP_SERVE_RUN", "PP_SERVE_ROWS", "OUT") for a in ("-u", v)]
    got = subprocess.run(["env", *unset, f"RESEARCH_RUN_DIR={tmp_path}", f"PP_LLAMA={tmp_path}",
                          f"LROW={INT8_H200.replace('h200', 'b300')}", "bash", str(RUN_SH)],
                         cwd=REPO, capture_output=True, text=True)
    assert got.returncode == 2
    assert "no stored workload of that name" in got.stdout and "give LROW the row id of PP_LLAMA's Commit" in got.stdout
    assert list(tmp_path.iterdir()) == []
