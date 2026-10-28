"""The γ accounting of Pearl's FP8 scheme under W1."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("pouw_gamma", Path(__file__).parents[1] / "gamma.py")
G = importlib.util.module_from_spec(spec)
sys.modules["pouw_gamma"] = G
spec.loader.exec_module(G)


def test_pearl_has_a_floor():
    assert float(G.row(2048, 2048, 2048).gamma_floor.rstrip("%")) > 1.0


def test_uncovered_lm_head():
    import json
    model = json.loads(G.WORKLOAD.read_text())["model"]
    assert 0.09 < G.uncovered_fraction(model) < 0.11
