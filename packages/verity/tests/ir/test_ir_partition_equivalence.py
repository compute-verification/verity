"""The vLLM integration's copy of the partition validator against `verity.ir.partition`, on the same inputs.

Every test of `test_ir_partition.py` runs again here with the validator's entry points replaced by differential
stand-ins: each call runs both copies (the integration's through its own boundary module) and asserts equal verdicts
(or the same exception) before the test sees core's.  Skipped without `verity_vllm`; removed with the integration
copy."""
from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
import types

import pytest

from verity.ir import partition as CP
from verity.ir.parts import Unsupported

IP = pytest.importorskip("verity_vllm.query.partition")
pytest.importorskip("verity_vllm.query.boundary")

_OPAQUE = object()


def canon(x):
    if x is None or isinstance(x, (bool, int, float, str, Unsupported)):
        return x
    if isinstance(x, (frozenset, set)):
        return frozenset(x)
    if isinstance(x, (list, tuple)):
        return tuple(canon(y) for y in x)
    if isinstance(x, dict):
        return {k: canon(v) for k, v in x.items()}
    if hasattr(x, "to_json"):
        return (type(x).__name__, canon(x.to_json()))
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return (type(x).__name__,) + tuple(canon(getattr(x, f.name)) for f in dataclasses.fields(x))
    return _OPAQUE


def _stub_calls():
    """Snapshot of an injected stub boundary's call counters, restored after the integration's run so a test that
    counts boundary calls sees core's alone."""
    stub = CP._BOUNDARY
    saved = dict(stub.calls) if hasattr(stub, "calls") else None

    def restore() -> None:
        if saved is not None:
            stub.calls.clear()
            stub.calls.update(saved)
    return restore


def differential(core, other, name: str):
    def run(*args, **kw):
        restore = _stub_calls()
        try:
            rb = ("ok", other(*args, **kw))
        except Exception as e:  # noqa: BLE001
            rb = ("raise", type(e).__name__, str(e))
        restore()
        try:
            ra = core(*args, **kw)
        except Exception as e:
            assert rb == ("raise", type(e).__name__, str(e)), (name, rb, e)
            raise
        assert rb[0] == "ok", (name, rb)
        if rb[1] is not ra:
            ca = canon(ra)
            assert ca is _OPAQUE or canon(rb[1]) == ca, (name, ca, canon(rb[1]))
        return ra
    return run


class Differential:
    """A module stand-in: function attributes run both modules, classes and constants are core's, setting sets both."""

    def __init__(self, core, other) -> None:
        object.__setattr__(self, "_pair", (core, other))

    def __getattr__(self, name: str):
        core, other = self._pair
        a = getattr(core, name)
        if isinstance(a, (type, types.ModuleType)) or not callable(a):
            return a
        return differential(a, getattr(other, name), name)

    def __setattr__(self, name: str, value) -> None:
        for m in self._pair:
            setattr(m, name, value)


def _load(stem: str):
    spec = importlib.util.spec_from_file_location(f"_differential_{stem}", pathlib.Path(__file__).with_name(f"{stem}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_pmod = _load("test_ir_partition")
_pmod.partition = Differential(CP, IP)
for _name in ("validate_partition", "validate_width", "readiness_32bit", "gate_width_readiness", "validate_vu_family",
              "validate_vu_partition", "vu_status"):
    setattr(_pmod, _name, differential(getattr(CP, _name), getattr(IP, _name), _name))
for _k, _v in vars(_pmod).items():
    if _k.startswith("test_") or _k in ("gemm", "synth", "oracle"):
        globals()[_k] = _v
