"""The workload's execution label decides enforce_eager -- never a harness literal.

Torch-free part (run_config's laptop mirror) always runs; the adapter part (execution_of_workload / build_engine refusal) needs torch and
runs on a pod (`/workspace/venv-cu129/bin/python -m pytest tests/engine/test_execution_label.py`).  The build_engine
refusal is exercised WITHOUT a GPU: the mismatch is raised before LLM() (monkeypatched to fail loudly if reached).
"""
from __future__ import annotations
import json
import pytest

pytest.importorskip("verity_vllm.pipeline.match")
from verity_vllm.pipeline import match as rc

EAGER_ROW = "llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager"
COMPILED_ROW = "llama32-1b__bf16__l40s__tp1__b8__i1024__o128__mixed__greedy__bi-compiled"


def _wl(tmp_path, row_id: str | None, execution: str | None = None) -> str:
    doc = {"schema": "verity-vllm/workload/v2", "requests": [], "sampling": {"variant": "greedy"}, "variants": {"greedy": {"temperature": 0.0}}}
    if row_id:
        doc["sweep"] = {"row_id": row_id}
    if execution:
        doc["execution"] = execution
    p = tmp_path / "wl.json"
    p.write_text(json.dumps(doc))
    return str(p)


def test_run_config_mirror_reads_row_suffix(tmp_path):
    assert rc._workload_execution(_wl(tmp_path, EAGER_ROW)) == "bi-eager"
    assert rc._workload_enforce_eager(_wl(tmp_path, EAGER_ROW)) is True
    assert rc._workload_execution(_wl(tmp_path, COMPILED_ROW)) == "bi-compiled"
    assert rc._workload_enforce_eager(_wl(tmp_path, COMPILED_ROW)) is False


def test_run_config_mirror_explicit_field_wins_and_undeclared_defaults_eager(tmp_path):
    assert rc._workload_execution(_wl(tmp_path, EAGER_ROW, execution="bi-compiled")) == "bi-compiled"
    assert rc._workload_execution(_wl(tmp_path, None)) is None
    assert rc._workload_enforce_eager(_wl(tmp_path, None)) is True


def test_run_config_mirror_refuses_unknown_label(tmp_path):
    with pytest.raises(SystemExit):
        rc._workload_enforce_eager(_wl(tmp_path, "x__bi-quantum"))


# --------------------------------------------------------------------------------------------------------------- adapter (torch)
def _va():
    return pytest.importorskip("verity_vllm.engine.vllm_adapter")   # per test: the laptop (no torch) still runs the mirror tests above


def test_adapter_execution_of_workload_and_labels():
    va = _va()
    assert va.execution_of_workload({"sweep": {"row_id": EAGER_ROW}}) == "bi-eager"
    assert va.execution_of_workload({"sweep": {"row_id": COMPILED_ROW}}) == "bi-compiled"
    assert va.execution_of_workload({"execution": "bi-compiled", "sweep": {"row_id": EAGER_ROW}}) == "bi-compiled"
    assert va.execution_of_workload({"requests": []}) is None
    assert va.execution_of_workload(None) is None
    assert va.execution_label_of(True) == "bi-eager" and va.execution_label_of(False) == "bi-compiled" and va.execution_label_of(None) is None
    assert va.execution_label_of(True, False) == "eager" and va.execution_label_of(False, False) == "compiled"
    assert va.execution_label_of(False, None) is None
    assert va.EXECUTION_ENFORCE_EAGER is rc.ENFORCE_EAGER          # the laptop's table is the pod's


def test_load_workload_carries_row_id_and_execution(tmp_path):
    va = _va()
    p = _wl(tmp_path, COMPILED_ROW)
    wl = va.load_workload(p)
    assert wl["row_id"] == COMPILED_ROW and wl["execution"] == "bi-compiled"


def _no_llm(monkeypatch):
    """LLM() must never be reached by the refusal paths: make the vllm import site explode if it is."""
    _va()
    import types, sys
    fake = types.ModuleType("vllm")

    class LLM:
        def __init__(self, **kw):
            raise AssertionError("LLM() was reached: the execution check did not fail closed")
    fake.LLM = LLM
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")             # build_engine applies the label's; the test's process gets its own back
    from verity_vllm.engine import engine_profile as prof
    monkeypatch.setattr(prof, "apply_env", lambda: {})
    monkeypatch.setattr(prof, "checkpoint_for", lambda case, ck: {"local_path": "/nonexistent", "repo": "x", "revision": "y"})


def test_build_engine_refuses_unknown_label(monkeypatch):
    va = _va()
    _no_llm(monkeypatch)
    with pytest.raises(RuntimeError, match="not modelled"):
        va.build_engine(case="B0", execution="bi-quantum")


def test_build_engine_refuses_contradicting_engine_arg(monkeypatch):
    va = _va()
    _no_llm(monkeypatch)
    with pytest.raises(RuntimeError, match="execution mismatch"):
        va.build_engine(case="B0", execution="bi-eager", engine_args={"enforce_eager": False})
    with pytest.raises(RuntimeError, match="execution mismatch"):
        va.build_engine(case="B0", execution="bi-compiled", engine_args={"enforce_eager": True})


def test_build_engine_sets_enforce_eager_from_label(monkeypatch):
    """The kwargs handed to LLM() carry the label's enforce_eager (True for bi-eager, explicit False for bi-compiled)."""
    va = _va()
    _no_llm(monkeypatch)
    seen = {}
    import sys

    class LLM:
        def __init__(self, **kw):
            seen.update(kw)
            raise KeyboardInterrupt  # stop right after the kwargs are observed (no GPU here)
    sys.modules["vllm"].LLM = LLM
    for label, want in (("bi-eager", True), ("bi-compiled", False)):
        seen.clear()
        with pytest.raises(KeyboardInterrupt):
            va.build_engine(case="B0", execution=label)
        assert seen.get("enforce_eager") is want, (label, seen)
