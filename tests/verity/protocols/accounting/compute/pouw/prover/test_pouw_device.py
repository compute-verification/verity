"""`pouw_device`, the one toolchain reader: the configured C++ compiler and nvcc, else the path's, and the row-table kernels
build with them."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from verity.protocols.accounting.compute.pouw.prover.serving import device as PD
from verity.protocols.accounting.compute.pouw.prover.serving import row_table_kernels as RTK


def test_the_toolchain_is_the_configured_one_else_the_paths(monkeypatch):
    path = {"clang++": "/p/clang++", "nvcc": "/p/nvcc"}
    monkeypatch.setattr(PD.shutil, "which", path.get)
    monkeypatch.setattr(PD, "TOOLCHAIN", PD.Toolchain())
    assert (PD.compiler(), PD.nvcc()) == ("clang++", "/p/nvcc")
    PD.configure(cxx="/t/g++", nvcc="/t/nvcc")
    assert (PD.compiler(), PD.nvcc()) == ("/t/g++", "/t/nvcc")
    PD.configure(cxx=None)
    assert PD.compiler() == "/t/g++"
    path.clear()
    monkeypatch.setattr(PD, "TOOLCHAIN", PD.Toolchain())
    assert (PD.compiler(), PD.nvcc()) == (None, None)
    with pytest.raises(TypeError, match="no toolchain field 'arch'"):
        PD.configure(arch="sm_90a")


def test_the_row_table_libraries_build_with_the_configured_toolchain(monkeypatch):
    built: list[str] = []

    def build(tool, sources, flags, name):
        built.append(tool)
        return SimpleNamespace(rt_step_host=SimpleNamespace(), rt_midstates=SimpleNamespace(), rt_step=SimpleNamespace())

    monkeypatch.setattr(PD.TOOLCHAIN, "cxx", "/opt/toolchain/bin/g++")
    monkeypatch.setattr(PD.TOOLCHAIN, "nvcc", "/opt/toolchain/bin/nvcc")
    monkeypatch.setattr(RTK, "_build", build)
    RTK.host_lib.__wrapped__()
    RTK.device_lib.__wrapped__("120")
    assert built == ["/opt/toolchain/bin/g++", "/opt/toolchain/bin/nvcc"]
    assert RTK.nvcc() == PD.nvcc() == "/opt/toolchain/bin/nvcc"


@pytest.mark.parametrize("lib, what", [(lambda: RTK.host_lib.__wrapped__(), "C\\+\\+ compiler"),
                                       (lambda: RTK.device_lib.__wrapped__("120"), "nvcc")])
def test_no_toolchain_and_no_path_builds_no_row_table_library_and_names_the_missing_tool(monkeypatch, lib, what):
    monkeypatch.setattr(PD, "TOOLCHAIN", PD.Toolchain())
    monkeypatch.setattr(PD.shutil, "which", lambda name: None)
    monkeypatch.setattr(RTK, "_build", lambda *a: pytest.fail("built with no toolchain"))
    with pytest.raises(RuntimeError, match=f"no {what} configured or on the path"):
        lib()
