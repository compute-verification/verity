"""`verity_vllm.observe.capture`'s `Capture.install`: it writes the run header, registers the roots and the runner's external
buffers and wraps the runner, in that order and once; and only a recording capture names the modules, hooks the root's
forward, installs the Triton adapter and, where snapshot steps are listed, hands the recorder its two snapshot callbacks.
Each step `install` takes is stood in by an entry in a list, so the test holds the order and the conditions, not the steps."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from verity_vllm.observe import capture as C
from verity_vllm.observe.capture import Capture, ObserverConfig


def _capture(monkeypatch, config: ObserverConfig, header: object = None) -> tuple[Capture, list]:
    capture = Capture.__new__(Capture)
    capture.installed, capture.cfg, capture.namer = False, config, None
    monkeypatch.setattr(capture, "header", header, raising=False)
    monkeypatch.setattr(capture, "runner", "the runner", raising=False)
    monkeypatch.setattr(capture, "model", "the model", raising=False)
    monkeypatch.setattr(capture, "rec", SimpleNamespace(after_call=None, after_launch=None), raising=False)
    steps: list = []

    def namer(rec, model):
        steps.append(("module namer", rec is capture.rec, model))
        return "the namer"

    monkeypatch.setattr(capture, "_write", lambda event: steps.append(("header", event)))
    monkeypatch.setattr(capture, "_install_roots", lambda: steps.append("roots"))
    monkeypatch.setattr(capture, "_install_external_buffers", lambda: steps.append("external buffers"))
    monkeypatch.setattr(capture, "_install_root_phase_hooks", lambda: steps.append("root phase hooks"))
    monkeypatch.setattr(capture, "_wrap_runner", lambda: steps.append("runner"))
    monkeypatch.setattr(C, "ModuleNamer", namer)
    monkeypatch.setattr(C.triton_adapter, "install", lambda rec: steps.append(("triton", rec is capture.rec)))
    return capture, steps


@pytest.mark.parametrize("mode", ["none", "empty"])
def test_a_capture_that_does_not_record_registers_and_wraps_only_and_installs_once(monkeypatch, mode):
    capture, steps = _capture(monkeypatch, ObserverConfig(mode=mode, snapshot_steps=frozenset({2})), header="the header")
    capture.install()
    assert steps == [("header", "the header"), "roots", "external buffers", "runner"] and capture.installed is True
    assert capture.namer is None and capture.rec.after_call is None and capture.rec.after_launch is None
    capture.install()
    assert len(steps) == 4


def test_a_recording_capture_names_modules_hooks_the_root_and_installs_triton_before_it_wraps_the_runner(monkeypatch):
    capture, steps = _capture(monkeypatch, ObserverConfig())
    capture.install()
    assert steps == ["roots", "external buffers", ("module namer", True, "the model"), "root phase hooks", ("triton", True), "runner"]
    assert capture.namer == "the namer" and capture.installed is True
    assert capture.rec.after_call is None and capture.rec.after_launch is None


def test_a_recording_capture_with_snapshot_steps_hands_the_recorder_its_two_callbacks(monkeypatch):
    capture, steps = _capture(monkeypatch, ObserverConfig(module_hooks=False, triton=False, snapshot_steps=frozenset({2})))
    capture.install()
    assert steps == ["roots", "external buffers", "runner"] and capture.namer is None
    assert capture.rec.after_call == capture._after_call and capture.rec.after_launch == capture._after_launch
