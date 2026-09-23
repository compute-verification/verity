"""`vllm_meta.ensure_distributed` retries the store bind when a concurrent derive took the port (EADDRINUSE), and only that."""
from __future__ import annotations

import contextlib
import sys
import types

import pytest

from verity_capture.experimental.cb_a import vllm_meta


def _stub_vllm(monkeypatch, init):
    cfg = types.ModuleType("vllm.config")
    cfg.set_current_vllm_config = lambda _c: contextlib.nullcontext()
    dist = types.ModuleType("vllm.distributed")
    dist.init_distributed_environment = init
    dist.ensure_model_parallel_initialized = lambda *_a: None
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", cfg)
    monkeypatch.setitem(sys.modules, "vllm.distributed", dist)
    monkeypatch.setattr(vllm_meta, "_DIST_READY", False)


def test_eaddrinuse_retried_with_a_fresh_port(monkeypatch):
    methods = []

    def init(**kw):
        methods.append(kw["distributed_init_method"])
        if len(methods) < 3:
            raise RuntimeError("The server socket has failed to listen on any local network address. port: 44193, code: -98, name: EADDRINUSE")

    ports = iter([44193, 44193, 51000])
    monkeypatch.setattr(vllm_meta, "_free_port", lambda: next(ports))
    _stub_vllm(monkeypatch, init)
    assert vllm_meta.ensure_distributed(None) == "gloo"
    assert methods == ["tcp://127.0.0.1:44193", "tcp://127.0.0.1:44193", "tcp://127.0.0.1:51000"]
    assert vllm_meta._DIST_READY


def test_other_errors_and_exhaustion_raise(monkeypatch):
    def other(**_kw):
        raise RuntimeError("gloo: no such device")

    _stub_vllm(monkeypatch, other)
    with pytest.raises(RuntimeError, match="no such device"):
        vllm_meta.ensure_distributed(None)
    assert not vllm_meta._DIST_READY

    calls = []

    def always_taken(**_kw):
        calls.append(1)
        raise RuntimeError("EADDRINUSE")

    _stub_vllm(monkeypatch, always_taken)
    with pytest.raises(RuntimeError, match="EADDRINUSE"):
        vllm_meta.ensure_distributed(None)
    assert len(calls) == vllm_meta._PORT_TRIES
