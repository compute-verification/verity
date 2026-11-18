from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import verity_torch.pouw.h1.receipts as RC
import verity_torch.pouw.h1.registered_device as RD
from verity.protocols.resource_accounting.compute.pouw.prover.serving import device as PD


def _entry() -> SimpleNamespace:
    return SimpleNamespace(argtypes=None, restype=None)


def test_the_registered_rows_host_library_is_built_with_the_configured_compiler(monkeypatch):
    asked: list[tuple[Any, ...]] = []

    def build(*args: Any) -> SimpleNamespace:
        asked.append(args)
        return SimpleNamespace(rr_commit_host=_entry(), rr_public_salts_host=_entry(), rr_ok_salts_host=_entry())

    monkeypatch.setattr(PD.TOOLCHAIN, "cxx", "/opt/toolchain/bin/g++")
    monkeypatch.setattr(RD.RTD, "_build", build)
    RD.host_lib.__wrapped__()
    assert [a[0] for a in asked] == ["/opt/toolchain/bin/g++"]


@pytest.mark.parametrize("lib", [RD.host_lib, RD.zy_host_lib])
def test_no_compiler_builds_no_h1_host_library(monkeypatch, lib):
    monkeypatch.setattr(PD, "compiler", lambda: None)
    monkeypatch.setattr(RD.RTD, "_build", lambda *a: pytest.fail("built with no compiler"))
    with pytest.raises(RuntimeError, match="no C\\+\\+ compiler configured or on the path"):
        lib.__wrapped__()


def test_the_receipts_noise_kernel_is_built_with_the_configured_nvcc(monkeypatch):
    asked: list[tuple[Any, ...]] = []

    def built(*args: Any, **kw: Any) -> Path:
        asked.append(args)
        return Path("noise.cubin")

    monkeypatch.setattr(PD.TOOLCHAIN, "nvcc", "/opt/toolchain/bin/nvcc")
    monkeypatch.setattr(RD.RTD, "built", built)
    assert RC.noise_cubin("120") == Path("noise.cubin")
    assert [a[0] for a in asked] == ["/opt/toolchain/bin/nvcc"]


def test_no_nvcc_builds_no_noise_kernel(monkeypatch):
    monkeypatch.setattr(PD, "nvcc", lambda: None)
    monkeypatch.setattr(RD.RTD, "built", lambda *a, **k: pytest.fail("built with no nvcc"))
    with pytest.raises(RuntimeError, match="no nvcc configured or on the path"):
        RC.noise_cubin("120")
