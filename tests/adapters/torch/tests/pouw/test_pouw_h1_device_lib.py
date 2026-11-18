"""The CUDA row-table library as H1's `tables_device.device_lib` declares it: what it asks the build for, the argument and
result types of its two entry points, and its refusal on a node with no nvcc. The build is a stand-in, as a node without
CUDA has no nvcc to make one: what is held here is the declaration, not the kernel."""
from __future__ import annotations

import ctypes
from types import SimpleNamespace
from typing import Any

import pytest
from verity.errors import InvalidArtifact

import verity_torch.pouw.h1.tables_device as TD
import verity_torch.pouw.row_table_device as RTL

_vp = ctypes.c_void_p


def _library(monkeypatch: pytest.MonkeyPatch, tool: str | None) -> tuple[Any, list[tuple[Any, ...]]]:
    """`device_lib` for an RTX PRO 6000 (its card file's arch, sm_120a) on a node whose nvcc is `tool`, uncached, with each build it asked for."""
    asked: list[tuple[Any, ...]] = []

    def build(*args: Any) -> SimpleNamespace:
        asked.append(args)
        return SimpleNamespace(rt_midstates=SimpleNamespace(), rt_step=SimpleNamespace())

    monkeypatch.setattr(TD, "nvcc", lambda: tool)
    monkeypatch.setattr(TD.RTD, "_build", build)
    return TD.device_lib.__wrapped__(RTL.arch("NVIDIA RTX PRO 6000 Blackwell Server Edition")), asked


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


def test_a_device_builds_for_its_card_files_arch_and_a_device_no_card_file_names_is_refused(monkeypatch):
    """A library is built for the target its device's card file states (`verity.device.gpu_card_reporting`'s ``arch``): the
    arch-specific one, sm_90a or sm_120a, never the plain sm_90 or sm_120, whose integer-division sequences the SASS gate
    passes nowhere. A device no card file names is refused by the reader, naming what to add; so is a card whose arch is no
    target."""
    assert [RTL.arch(n) for n in ("NVIDIA H100 80GB HBM3", "NVIDIA H200", "NVIDIA RTX PRO 6000 Blackwell Server Edition")] == [
        "90a", "90a", "120a"]
    with pytest.raises(KeyError, match=r"0 GPU card files under .* have 'NVIDIA A100-SXM4-80GB'.* add \{ value = 'NVIDIA A100"):
        RTL.arch("NVIDIA A100-SXM4-80GB")
    cards = {"a card": {"kind": "gpu", "sku": "a-card", "arch": "sm_90a"}, "no target": {"kind": "gpu", "sku": "x", "arch": "90a"}}

    def reporting(name, *, directory):
        assert directory == RTL.CARD_FILES
        if name not in cards:
            raise KeyError(f"no card file has {name!r}")
        return cards[name]

    monkeypatch.setattr(RTL, "gpu_card_reporting", reporting)
    assert RTL.arch("a card") == "90a"
    with pytest.raises(KeyError, match="no card file has 'a device'"):
        RTL.arch("a device")
    with pytest.raises(InvalidArtifact, match=r"arch = '90a'; a GPU card states sm_"):
        RTL.arch("no target")
