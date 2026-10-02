"""`test_native_jit_load.py` leaves its pytest process as it found it: run first in one process, the `exp` reference against torch after it
still passes."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SUITE = Path(__file__).resolve().parents[2]
EXP = "tests/program/test_ref_prims.py::test_transcendentals_vs_torch_and_float64[F32ExpRn_v1-f32_exp_rn_bits-exp]"


@pytest.mark.slow
def test_the_exp_reference_passes_after_the_native_jit_tests_in_one_process():
    pytest.importorskip("torch")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider", "-p", "no:xdist",
                        "tests/commit/test_native_jit_load.py", EXP], cwd=SUITE, capture_output=True, text=True)
    assert r.returncode == 0 and " passed" in r.stdout and "skipped" not in r.stdout and "deselected" not in r.stdout, \
        r.stdout[-6000:] + r.stderr[-3000:]
