"""The CUDA row-table library as H1's `tables_device.device_lib` declares it: what it asks the build for, the argument and
result types of its two entry points, and its refusal on a node with no nvcc. The build is a stand-in, as a node without
CUDA has no nvcc to make one: what is held here is the declaration, not the kernel."""
from __future__ import annotations

import ctypes
from types import SimpleNamespace
from typing import Any

import pytest

import verity_torch.pouw.h1.tables_device as TD
import verity_torch.pouw.row_table_device as RTL

_vp = ctypes.c_void_p


def _library(monkeypatch: pytest.MonkeyPatch, tool: str | None) -> tuple[Any, list[tuple[Any, ...]]]:
    """`device_lib` for an RTX PRO 6000 (capability 12.0, sm_120a) on a node whose nvcc is `tool`, uncached, with each build it asked for."""
    asked: list[tuple[Any, ...]] = []

    def build(*args: Any) -> SimpleNamespace:
        asked.append(args)
        return SimpleNamespace(rt_midstates=SimpleNamespace(), rt_step=SimpleNamespace())

    monkeypatch.setattr(TD, "nvcc", lambda: tool)
    monkeypatch.setattr(TD.RTD, "_build", build)
    return TD.device_lib.__wrapped__(RTL.arch(12, 0)), asked


def test_the_device_library_is_one_build_for_the_arch_with_its_two_entry_points_declared(monkeypatch):
    dll, asked = _library(monkeypatch, "/opt/cuda/bin/nvcc")
    assert asked == [("/opt/cuda/bin/nvcc", TD.RTD.DEVICE_SOURCES,
                      (*TD.RTD.NVCC_FLAGS, "-gencode=arch=compute_120a,code=sm_120a"), "librow_table_cuda.so")]
    assert dll.rt_midstates.argtypes == [_vp, _vp, _vp]
    assert dll.rt_step.argtypes == [_vp, _vp, _vp, ctypes.c_uint64, ctypes.c_uint32, _vp, _vp,
                                    ctypes.POINTER(TD.RTD.Args), _vp, _vp, _vp, _vp, _vp]
    assert dll.rt_midstates.restype is ctypes.c_int and dll.rt_step.restype is ctypes.c_int


def test_a_node_with_no_nvcc_builds_no_device_library(monkeypatch):
    with pytest.raises(RuntimeError, match="no nvcc on the path or configured"):
        _library(monkeypatch, None)


def test_a_device_builds_for_its_pinned_arch_specific_target_and_any_other_capability_is_refused():
    """A library is built for the arch-specific target whose ptxas the SASS gate's pins cover (sm_90a, sm_120a), never the
    plain sm_90 or sm_120, whose integer-division sequences the gate passes nowhere; another capability is refused by name."""
    assert (RTL.arch(9, 0), RTL.arch(12, 0)) == ("90a", "120a")
    for major, minor in ((8, 9), (9, 1), (10, 0)):
        with pytest.raises(RuntimeError, match=rf"compute capability {major}\.{minor} has no arch-specific target"):
            RTL.arch(major, minor)
