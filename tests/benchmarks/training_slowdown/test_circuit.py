"""training_slowdown.capture.circuit: the training formalism's quantities on a saved step circuit, and the cut they are
measured with."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from training_slowdown.capture.circuit import StepCircuit, measure, vertex_cut
from training_slowdown.formalism import READ_BACKS

ADAPTER = Path(__file__).parent / "fixtures" / "wildcard_adapter.npz"


def test_a_grown_adapters_step():
    """A grown block's 2-bit adapter on a tiny start (`fixtures/wildcard_adapter.npz`, measured at step 1): the frozen
    layers are θ's function of the tokens, so a token's hidden state where it meets θ' is one 32-bit value, and nearly
    all of its affected work reads no update value."""
    f = measure(StepCircuit.load(ADAPTER))
    assert f["a_bits"] == 32 and f["tail_share"] == pytest.approx(0.989, abs=0.001)


def test_a_saved_circuit_measures_alike(tmp_path):
    """A circuit measures alike once saved and loaded, read-back lists included; one saved without the lists measures
    without them, which training_slowdown.formalism refuses as unchecked."""
    c = StepCircuit.load(ADAPTER)
    want = measure(c)
    c.save(tmp_path / "c.npz")
    assert measure(StepCircuit.load(tmp_path / "c.npz")) == want and all(k in want for k in READ_BACKS)
    replace(c, read_backs=None).save(tmp_path / "old.npz")
    old = measure(StepCircuit.load(tmp_path / "old.npz"))
    assert old == {k: v for k, v in want.items() if k not in READ_BACKS}


def test_a_vertex_cut():
    """0 -> {1, 2} -> 3: the narrowest set meeting both paths is {0} or {3}, or {1, 2}; an uncuttable sink and source
    leave none."""
    preds = [[], [0], [0], [1, 2]]
    assert vertex_cut(preds, {0: 10, 1: 3, 2: 4, 3: 9}, [0], [3])[0] == 7
    assert vertex_cut(preds, {0: None, 1: 3, 2: None, 3: None}, [0], [3]) == (None, [])
