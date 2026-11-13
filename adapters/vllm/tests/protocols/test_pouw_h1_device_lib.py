"""The CUDA row-table library as H1's `tables_device.device_lib` declares it: what it asks the build for, the argument and
result types of its two entry points, and its refusal on a node with no nvcc. The build is a stand-in, as a node without
CUDA has no nvcc to make one: what is held here is the declaration, not the kernel."""
from __future__ import annotations

import ctypes
from types import SimpleNamespace
from typing import Any

import pytest

import verity_vllm.protocols.pouw.h1.tables_device as TD

_vp = ctypes.c_void_p


def _library(monkeypatch: pytest.MonkeyPatch, tool: str | None) -> tuple[Any, list[tuple[Any, ...]]]:
    """`device_lib` for sm_120 on a node whose nvcc is `tool`, uncached, with each build it asked for."""
    asked: list[tuple[Any, ...]] = []

    def build(*args: Any) -> SimpleNamespace:
        asked.append(args)
        return SimpleNamespace(rt_midstates=SimpleNamespace(), rt_step=SimpleNamespace())

    monkeypatch.setattr(TD, "nvcc", lambda: tool)
    monkeypatch.setattr(TD.RTD, "_build", build)
    return TD.device_lib.__wrapped__("120"), asked


def test_the_device_library_is_one_build_for_the_arch_with_its_two_entry_points_declared(monkeypatch):
    dll, asked = _library(monkeypatch, "/opt/cuda/bin/nvcc")
    assert asked == [("/opt/cuda/bin/nvcc", TD.RTD.DEVICE_SOURCES,
                      (*TD.RTD.NVCC_FLAGS, "-gencode=arch=compute_120,code=sm_120"), "librow_table_cuda.so")]
    assert dll.rt_midstates.argtypes == [_vp, _vp, _vp]
    assert dll.rt_step.argtypes == [_vp, _vp, _vp, ctypes.c_uint64, ctypes.c_uint32, _vp, _vp,
                                    ctypes.POINTER(TD.RTD.Args), _vp, _vp, _vp, _vp, _vp]
    assert dll.rt_midstates.restype is ctypes.c_int and dll.rt_step.restype is ctypes.c_int


def test_a_node_with_no_nvcc_builds_no_device_library(monkeypatch):
    with pytest.raises(RuntimeError, match="no nvcc on the path or configured"):
        _library(monkeypatch, None)
