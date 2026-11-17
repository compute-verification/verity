"""training_slowdown.capture.circuit: what the grade and the library read of a saved step circuit."""
from __future__ import annotations

from pathlib import Path

from training_slowdown.capture.circuit import StepCircuit, measure

ADAPTER = Path(__file__).parent / "fixtures" / "wildcard_adapter.npz"


def test_a_grown_adapters_step():
    """A grown block's 2-bit adapter on a tiny start (`fixtures/wildcard_adapter.npz`, measured at step 1): its gates,
    the share of them an update reaches, and τ's token positions and other bits."""
    f = measure(StepCircuit.load(ADAPTER))
    assert set(f) == {"gates", "affected_share", "tau_ids", "tau_other_bits"}
    assert f["gates"] > 0 and 0 < f["affected_share"] < 1 and f["tau_ids"] >= 1


def test_a_saved_circuit_measures_alike(tmp_path):
    """A circuit measures alike once saved and loaded, and keeps its read-back lists."""
    c = StepCircuit.load(ADAPTER)
    c.save(tmp_path / "c.npz")
    back = StepCircuit.load(tmp_path / "c.npz")
    assert measure(back) == measure(c) and back.read_backs == c.read_backs is not None
