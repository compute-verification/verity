"""Verification plans over IR programs: prescribed inputs are sampled gates (IR §E)."""

from __future__ import annotations

import importlib.util
import sys
from fractions import Fraction
from pathlib import Path

from verity.ir import query
from verity.verification.plan import Obligation, weight_binding_plan


def _running_example():
    """The IR acceptance test's running example (``tests/ir/test_running_example.py``).

    Its definitions register in the global registry at import, so it must be executed once: reuse pytest's
    import when the IR tests are in the same session, else load it by path."""
    path = Path(__file__).resolve().parents[1] / "ir" / "test_running_example.py"
    for name in ("test_running_example", "ir_running_example"):
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve() == path:
            return module
    spec = importlib.util.spec_from_file_location("ir_running_example", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ir_running_example"] = module
    spec.loader.exec_module(module)
    return module


def test_plan_input_gates_are_sampled_gates():
    P = _running_example().make_program()
    C = P.circuit
    # W (param 1) at rate 1 = WB's model binding; everything else sampled at 1/4
    plan = weight_binding_plan(P, [1], other_rate=Fraction(1, 4))
    assert plan.rate_of(C[4]) == 1 and plan.rate_of(C[13]) == 1  # W Input gates
    assert plan.rate_of(C[0]) == Fraction(1, 4)  # X Input gates are sampled like any other gate
    assert plan.rate_of(C[14]) == Fraction(1, 4)
    assert plan.expected_checked(C, C[0:54]) == 10 + Fraction(44, 4)
    # the same plan with sampled weights
    plan2 = weight_binding_plan(P, [1], other_rate=Fraction(1, 4), weight_rate=Fraction(1, 2))
    assert plan2.rate_of(C[4]) == Fraction(1, 2)
    row = query(C, "//Row")[0]
    ob = Obligation(row, plan, {g: None for g in row.in_}, {g: None for g in row.out})
    assert ob.to_json()["function_call"]["args"] == [0, 1, 4, 5]
