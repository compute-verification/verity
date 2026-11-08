"""`capture.py` records an `OUTSIDE_MODE` op (FA3's ``fwd``) with the event its recorder would write, but runs the op with the
Python dispatch modes off: a ``Tensor[]`` return holding an undefined tensor reads None in Python, and C++ refuses a None in a
``Tensor[]`` handed back by a mode."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch.utils._python_dispatch import _get_current_dispatch_mode  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
LIB = torch.library.Library("spotcap_test", "DEF")
LIB.define("fwd(Tensor q, Tensor k, Tensor(a!)? out=None) -> Tensor[]")


def _fwd(q, k, out=None):
    if out is not None:
        out.copy_(q - k)
    return [q + k, q * k]


LIB.impl("fwd", _fwd, "CPU")


def _capture(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("spot_capture_outside_mode", HERE / "capture.py")
    capture = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, capture)
    spec.loader.exec_module(capture)
    S = capture._State({"capture": str(tmp_path), "steps": [3], "mla_tokens": 0, "open": {}, "keep": "opened"})
    monkeypatch.setattr(capture, "S", S)
    monkeypatch.setattr(capture, "OUTSIDE_MODE", (("spotcap_test", "fwd"),))
    monkeypatch.setattr(torch.ops.spotcap_test, "fwd", torch.ops.spotcap_test.fwd)
    return capture, S


def test_an_op_run_outside_the_mode_records_the_event_the_recorder_writes(tmp_path, monkeypatch) -> None:
    capture, S = _capture(tmp_path, monkeypatch)
    q, k = torch.arange(4.0), torch.full((4,), 2.0)

    def recorded():
        S.events.clear()
        out = torch.zeros(4)
        with capture.Recorder():
            got = torch.ops.spotcap_test.fwd(q, k, out=out)
        capture._drain()
        return got, out, list(S.events)

    want, want_out, want_events = recorded()
    capture._install_outside_mode()
    got, got_out, events = recorded()
    assert len(events) == 1 and events == want_events and events[0]["op"] == "spotcap_test::fwd"
    assert set(events[0]["out"]) == {"ret0", "ret1", "out"}
    assert all(torch.equal(a, b) for a, b in zip(got, want)) and torch.equal(got_out, want_out)


def test_an_undefined_tensor_in_the_list_passes_and_is_recorded_as_none(tmp_path, monkeypatch) -> None:
    capture, S = _capture(tmp_path, monkeypatch)

    class Fa3Fwd:
        """FA3's ``fwd`` when a call doesn't split: C++ leaves the accumulators undefined, which Python reads as None."""
        default = torch.ops.spotcap_test.fwd.default

        def __call__(self, q, k, out=None):
            assert _get_current_dispatch_mode() is None
            return [q + k, None]

    monkeypatch.setattr(torch.ops.spotcap_test, "fwd", Fa3Fwd())
    capture._install_outside_mode()
    q = torch.ones(2)
    with capture.Recorder():
        got = torch.ops.spotcap_test.fwd(q, q)
    assert torch.equal(got[0], torch.full((2,), 2.0)) and got[1] is None
    (e,) = S.events
    assert e["out"]["ret1"] is None and e["out"]["ret0"]["shape"] == [2]
    assert torch.ops.spotcap_test.fwd(q, q)[1] is None and len(S.events) == 1


def test_an_op_the_engine_never_loaded_is_noted(tmp_path, monkeypatch, capsys) -> None:
    capture, _ = _capture(tmp_path, monkeypatch)
    monkeypatch.setattr(capture, "OUTSIDE_MODE", (("spotcap_test", "not_loaded"),))
    capture._install_outside_mode()
    assert "no spotcap_test::not_loaded" in capsys.readouterr().err
