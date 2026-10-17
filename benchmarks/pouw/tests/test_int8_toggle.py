"""int8_toggle: the zero instance's operand patterns (the GPU timing itself runs only on the node)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location("pouw_int8_toggle", Path(__file__).parents[1] / "int8_toggle.py")
T = importlib.util.module_from_spec(spec)
sys.modules["pouw_int8_toggle"] = T
spec.loader.exec_module(T)


def test_each_operand_fills_its_range():
    gen = torch.Generator().manual_seed(0)
    for kind, (lo, hi) in T.RANGES.items():
        x = T.operand(kind, (64, 64), gen, "cpu")
        assert x.dtype == torch.int8 and int(x.min()) == lo and int(x.max()) == hi - 1, kind


def test_the_patterns_cover_each_designs_zero_instance():
    assert T.PATTERNS["bit"] == ("bit", "bit")              # A: both operands bits
    assert T.PATTERNS["u-bit"] == ("wide", "bit")           # B: U on the peeled side, a bit on the served side
    assert set(T.PATTERNS) >= {"wide", "zero"}
