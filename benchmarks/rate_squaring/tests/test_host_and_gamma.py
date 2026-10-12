"""`ifma.c`'s gate on this machine's CPU (skipped without gcc or AVX-512 IFMA), and `gamma.py`'s formula against rows of
ra-rate-redteam's `node_gamma.py` table (its report's 18:00Z γ table), so the two price the same thing."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _has_ifma() -> bool:
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return "avx512ifma" in flags


@pytest.mark.skipif(shutil.which("gcc") is None or not _has_ifma(), reason="needs gcc and AVX-512 IFMA")
@pytest.mark.parametrize("g", [1, 2])
def test_ifma_gate(tmp_path: Path, g: int) -> None:
    exe = tmp_path / f"ifma-{g}"
    subprocess.run(["gcc", "-O2", "-march=native", f"-DG={g}", "-o", str(exe), str(HERE / "ifma.c"), "-lpthread"],
                   check=True)
    lines = []
    for k in (1, 777):
        out = subprocess.run([str(exe), "gate", str(k), str(16 * g)], check=True, capture_output=True, text=True).stdout
        lines += out.splitlines()
    assert len(lines) == 2 * 16 * g
    for line in lines:
        _, k, bits, c, x0, xk = line.split()
        p = 2 ** int(bits) - int(c)
        assert pow(int(x0, 16) % p, 1 << int(k), p) == int(xk, 16), line


@pytest.mark.parametrize(("s", "host", "x1", "low", "high"), [
    (1.0, 128 * 1.12e7, 175, 0.74, 0.90),   # measured, GMP
    (1.0, 128 * 2.5e8, 164, 0.77, 0.93),    # measured, IFMA bound
    (1.2, 128 * 1.12e7, 145, 0.82, 1.00),   # pipe at 100%, GMP
    (1.45, 128 * 2.5e8, 113, 0.94, 1.18),   # issue bound, IFMA bound
])
def test_gamma_matches_node_gamma(s: float, host: float, x1: int, low: float, high: float) -> None:
    gamma = _load("gamma")
    h = host / (gamma.GPUS * gamma.GPU_MAX)
    x = gamma.X1 / (s * (1 + h))
    assert abs(x - x1) < 1
    assert [round(g, 2) for g in gamma.gamma(x, gamma.K)] == [low, high]


def test_measured_rates_are_clock_matched() -> None:
    gamma = _load("gamma")
    assert gamma.s_of(gamma.REF * gamma.MHZ / gamma.REF_MHZ) == pytest.approx(1.0)
    assert round(gamma.X1 / gamma.GPU["FOLD 5, column-chained (measured best)"], 1) == 138.0
