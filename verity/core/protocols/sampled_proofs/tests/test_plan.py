"""Verification plans over IR programs: prescribed inputs are sampled gates (IR §E), under the two-stage law."""

from __future__ import annotations

import importlib.util
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from verity.ml.circuits import query
from verity.ml.circuits.layout import FunctionCall
from verity.core.protocols.sampled_proofs.two_stage import OwnRandomness, Stage, select_replay_units
from verity.core.protocols.sampled_proofs.plan import DEFAULT_CLASS, Obligation, UnconditionalObligation, VerificationPlan, weight_binding_plan

RUNNING_EXAMPLE = Path(__file__).resolve().parents[4] / "ml" / "circuits" / "tests" / "test_running_example.py"


def _running_example():
    """The IR acceptance test's running example.  Its definitions register in the global registry at import, so it
    must be executed once: reuse pytest's import when the IR tests are in the same session, else load it by path."""
    for name in ("test_running_example", "ir_running_example"):
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve() == RUNNING_EXAMPLE:
            return module
    spec = importlib.util.spec_from_file_location("ir_running_example", RUNNING_EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ir_running_example"] = module
    spec.loader.exec_module(module)
    return module


def test_plan_input_gates_are_sampled_gates():
    P = _running_example().make_program()
    C = P.circuit
    # W (param 1) at p = 1 = WB's model binding; everything else sampled at 1/4, each gate its own replay unit
    plan = weight_binding_plan(P, [1], other_rate=Fraction(1, 4))
    assert plan.check_probability(C[4]) == 1 and plan.check_probability(C[13]) == 1  # W Input gates
    assert plan.check_probability(C[0]) == Fraction(1, 4)  # X Input gates are sampled like any other gate
    assert plan.check_probability(C[14]) == Fraction(1, 4)
    assert plan.stage_of(C[14]) == Stage(Fraction(1, 4), 1) and plan.class_of(C[14]) == DEFAULT_CLASS
    assert plan.expected_checked(C[0:54]) == 10 + Fraction(44, 4)
    # the same plan with sampled weights
    plan2 = weight_binding_plan(P, [1], other_rate=Fraction(1, 4), weight_rate=Fraction(1, 2))
    assert plan2.check_probability(C[4]) == Fraction(1, 2)
    row = query(C, "//Row")[0]
    ob = Obligation(row, plan, {g: None for g in row.in_}, {g: None for g in row.out})
    assert ob.to_json()["function_call"]["args"] == [0, 1, 4, 5]


def test_a_gate_level_plan_samples_through_the_law():
    P = _running_example().make_program()
    C = P.circuit
    plan = weight_binding_plan(P, [1], other_rate=Fraction(1, 4))
    units = plan.units(C[0:54])
    picked = {u.index for u in select_replay_units(OwnRandomness(random.Random(1).randbytes), plan.law, units)}
    assert set(range(4, 14)) <= picked                 # every weight Input gate, at p = 1
    assert picked == {u.index for u in select_replay_units(OwnRandomness(random.Random(1).randbytes), plan.law, units)}
    assert picked < set(range(54))
    assert plan.to_json()["classes"][0]["stage"] == {"p": "1", "k": 1} and plan.to_json()["default"] == {"p": "1/4", "k": 1}


def test_unconditional_obligations_are_the_covered_families():
    P = _running_example().make_program()
    wb = weight_binding_plan(P, [1], other_rate=Fraction(1, 4))
    weights = wb.classes[0][0]
    plan = VerificationPlan(wb.classes, wb.default, unconditional=(UnconditionalObligation(weights, "weights of record"),))
    assert plan.covered() == frozenset({weights.name}) and wb.covered() == frozenset()
    assert plan.law.profile(DEFAULT_CLASS, n_r=44, n_v=1, delta=0.01, covered=plan.covered()).covered == {weights.name}


def test_class_names_are_distinct():
    P = _running_example().make_program()
    wb = weight_binding_plan(P, [1], other_rate=Fraction(1, 4))
    with pytest.raises(ValueError):
        VerificationPlan(wb.classes * 2, wb.default)


def test_plans_never_read_the_rescanning_boundary(monkeypatch):
    """`FunctionCall.out` rescans the circuit per call (the prototype measured 0.6 s per call at 260k gates): the
    protocol builds obligations from the declared interface."""
    P = _running_example().make_program()
    C = P.circuit
    rows = query(C, "//Row")

    def refuse(self):
        raise AssertionError("FunctionCall.out read by sampled proofs")

    monkeypatch.setattr(FunctionCall, "out", property(refuse))
    plan = weight_binding_plan(P, [1], other_rate=Fraction(1, 4))
    obs = [Obligation.of_call(r, plan) for r in rows]
    assert [o.to_json()["function_call"]["returns"] for o in obs][:2] == [[list(rows[0].returns)[0]], [list(rows[1].returns)[0]]]
    assert plan.expected_checked(C[0:54]) == 10 + Fraction(44, 4)
