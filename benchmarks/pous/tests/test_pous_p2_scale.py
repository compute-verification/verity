"""`p2_v1/scale.py`'s CPU pieces: its special-form and powmod roots are `reference.rho`'s, a label it encodes decodes back,
and a path it gathers has log2 B siblings of 64 bytes."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

gmpy2 = pytest.importorskip("gmpy2")

BENCH = Path(__file__).resolve().parents[1]


def _scale():
    spec = importlib.util.spec_from_file_location("pous_p2_scale", BENCH / "p2_v1" / "scale.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path.insert(0, str(BENCH))
    from p2_v1 import reference
    return mod, reference


def test_chain_and_powmod_roots_are_reference_rho():
    S, R = _scale()
    P = R.DEPLOYMENT
    for roots in (S.Chain(), S.Powmod()):
        for y in (1, 2, 3, P.p - 1, P.p - 2, 21065, (1 << 16447) + 12345):
            assert roots.rho(y % P.p) == R.rho(P, y % P.p)


def test_chain_encode_round_trips():
    S, R = _scale()
    P, m, k, t = S._block_inputs(0, 70_000)
    c = S.Chain().encode(P, m, k, t)
    assert c == R.encode(P, m, k, t)
    assert R.decode(P, c, k, t) == m


def test_path_cost_shape():
    S, _ = _scale()
    rec = S.path_cost(1 << 10, 64, 0)
    assert rec["siblings"] == 10 and rec["path_bytes"] == 640
    assert rec["tree_bytes"] == sum((1 << 10) >> lv for lv in range(10)) * 64 + 64
