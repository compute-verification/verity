"""`verity.circuits.laying.zk_min_units` is `flock-circuit zk-min-units` (``FLOCK_CIRCUIT``), and the least n whose m
reaches the zero-knowledge floor."""

from __future__ import annotations

import subprocess

import pytest
import settings

from verity.circuits.laying import ZK_M_MIN, zk_min_units

GRID = [(k, g) for k in (0, 1, 7, 12, 18, 20, 21, 22, 23, 26) for g in (1, 2, 3, 8, 64)]


def m(k_log: int, n: int, g: int) -> int:
    return k_log + max(3, (-(-n // g) - 1).bit_length())


@pytest.mark.parametrize("k_log,g", GRID)
def test_it_is_the_least_n_at_the_floor(k_log, g):
    n = zk_min_units(k_log, g)
    assert m(k_log, n, g) >= ZK_M_MIN
    assert n == 1 or m(k_log, n - 1, g) < ZK_M_MIN


@pytest.mark.skipif(settings.flock_circuit() is None, reason="FLOCK_CIRCUIT names no flock-circuit build")
def test_python_is_flock_circuits():
    exe = settings.flock_circuit()
    rust = [int(subprocess.run([str(exe), "zk-min-units", str(k), str(g)], capture_output=True, text=True,
                               check=True).stdout) for k, g in GRID]
    assert rust == [zk_min_units(k, g) for k, g in GRID]
