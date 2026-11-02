"""te_core.capture.circuit: the training formalism's quantities on recorded/islands/toy/train.py (V = 16, H = 8, fp32,
4 x 8 tokens a step, measured at step 1): a dense update's u steps up by each matrix as one token's work first reads it
(emb, fc, head), however the program writes the update; a is a token's hidden state; Δ_grad is the fp32 gradient and the
state the forward doesn't read; frozen layers above the update are its tail; MeZO's gradient is one fp32 difference."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from te_core.capture import script
from te_core.capture.circuit import StepCircuit, measure, vertex_cut

PROGRAM = Path(__file__).parent / "recorded" / "islands" / "toy" / "train.py"
V, H, B, CTX = 16, 8, 4, 8
T = B * CTX
EMB, FC, HEAD = V * H, H * H, V * H
VALUES = EMB + FC + HEAD


@pytest.fixture
def starts(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "STREAM_TOKENS", 10 ** 5)
    shapes = {"emb": [V, H], "fc": [H, H], "head": [V, H], "frozen": [V, H]}
    (tmp_path / "toy.start.json").write_text(json.dumps({"config": {"vocab": V, "d_model": H},
                                                         "tensors": {n: ["float32", s] for n, s in shapes.items()}}))
    return tmp_path


def recorded(starts, *args):
    return script.record(PROGRAM, ["--batch", str(B), "--update", *args], step=1, starts=starts, wall=[0.1] * 8,
                         step_tokens=T)


@pytest.mark.parametrize("update,state", [("inplace", 0), ("rebind", 0), ("adamw", 64 * VALUES),
                                          ("sign-ef", 32 * VALUES), ("adamw --micro 2", 64 * VALUES)])
def test_a_dense_update(starts, update, state):
    """τ: a token's id and its target's, 9 positions of each row per 8 consumed tokens. The token ops are every op but
    the optimizer's, all dependent; dW and the head's input gradient read no update value (t), and no frozen layer sits
    above the head. ``--micro 2``: one token's work is its micro-batch's, half the step's."""
    f = recorded(starts, *update.split()).formalism
    assert f["tokens"] == T and f["token_factor"] == 1 and f["tau_ids"] == 9 / 8 and f["tau_other_bits"] == 0
    assert f["a_bits"] == 32 * H
    assert [b for _, b in f["u"]] == [32 * b for b in (0, EMB, EMB, EMB + FC, EMB + FC, VALUES, VALUES)]
    assert f["u"][1][0] == 0 and f["u"][-1][0] == 1
    assert f["u1_bits"] == f["update_cut_bits"] == 32 * VALUES
    assert f["grad_cut_bits"] == 32 * VALUES and f["state_bits"] == state and f["delta_grad_bits"] == 32 * VALUES + state
    assert f["phi"] == f["sigma"] and f["phi"] > 0.9 and f["dep_share"] == 1
    assert f["tail_share"] == pytest.approx(1 / 3, abs=0.01) and f["frozen_tail_share"] == 0
    if "--micro" in update:
        assert f["group_gates"] == pytest.approx(recorded(starts, "adamw").formalism["group_gates"] / 2)


def test_frozen_layers_above_the_update_are_its_tail(starts):
    """With the head frozen, u(1) is emb's and fc's bits; the head's forward and input gradient are the frozen tail, and
    t adds fc's dW."""
    f = recorded(starts, "inplace", "--freeze", "head").formalism
    assert f["u1_bits"] == f["grad_cut_bits"] == 32 * (EMB + FC)
    work = FC + HEAD + HEAD + FC + FC
    assert f["frozen_tail_share"] == pytest.approx(2 * HEAD / work, abs=0.01)
    assert f["tail_share"] == pytest.approx((2 * HEAD + FC) / work, abs=0.01)


def test_a_mezo_step(starts):
    """Both perturbed passes are the token's work, reading the update's perturbed copies (u(1) = 32 |θ'|) with nothing
    free of it; a is both passes' hidden states; Δ_grad is the one fp32 difference of the losses."""
    f = recorded(starts, "mezo").formalism
    assert f["u1_bits"] == 32 * VALUES and f["tail_share"] == 0
    assert f["a_bits"] == 2 * 32 * H
    assert f["delta_grad_bits"] == f["grad_cut_bits"] == 32 and f["state_bits"] == 0


@pytest.mark.parametrize("source", ["theta", "trained"])
def test_a_cache_rewritten_each_step(starts, source):
    """The side GEMM reads θ's table scaled by the buffer: outside Dep when the buffer comes from θ, and a second path a
    token's state is handed on when it comes from the trained embedding."""
    plain = recorded(starts, "inplace").formalism
    f = recorded(starts, "inplace", "--cache", source).formalism
    assert f["u1_bits"] == 32 * VALUES
    if source == "theta":
        assert f["phi"] < plain["phi"] and f["a_bits"] == 32 * H
    else:
        assert f["phi"] >= plain["phi"] and f["a_bits"] == 2 * 32 * H


def test_a_saved_circuit_measures_alike(starts, tmp_path):
    g = recorded(starts, "adamw")
    g.circuit.save(tmp_path / "c.npz")
    assert measure(StepCircuit.load(tmp_path / "c.npz")) == g.formalism


def test_a_vertex_cut():
    """0 -> {1, 2} -> 3: the narrowest set meeting both paths is {0} or {3}, or {1, 2}; an uncuttable sink and source
    leave none."""
    preds = [[], [0], [0], [1, 2]]
    assert vertex_cut(preds, {0: 10, 1: 3, 2: 4, 3: 9}, [0], [3])[0] == 7
    assert vertex_cut(preds, {0: None, 1: 3, 2: None, 3: None}, [0], [3]) == (None, [])
