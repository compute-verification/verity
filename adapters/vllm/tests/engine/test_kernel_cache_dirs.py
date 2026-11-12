"""Where `verity_vllm.engine.kernel_identity` looks for a process's generated kernels: Inductor's cache directory (the
environment's, else torch's default) and Triton's (the environment's, else none).  torch's default is stood in through
``sys.modules``, so nothing here needs an Inductor."""
from __future__ import annotations

import sys
from types import SimpleNamespace as NS

import pytest

from verity_vllm.engine import kernel_identity as KI

INDUCTOR, TRITON = "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"


@pytest.fixture
def torch_default(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """torch's own answer is ``/torch/default``; the list holds one entry for each time it was asked."""
    asked: list[str] = []

    def cache_dir() -> str:
        asked.append("asked")
        return "/torch/default"

    monkeypatch.setitem(sys.modules, "torch._inductor.runtime.runtime_utils", NS(cache_dir=cache_dir))
    return asked


def test_inductors_directory_is_the_environments_where_it_names_one(monkeypatch: pytest.MonkeyPatch, torch_default: list[str]) -> None:
    for held in ("/pod/inductor", "relative/dir", " "):
        monkeypatch.setenv(INDUCTOR, held)
        assert KI.inductor_cache_dir() == held
    assert torch_default == [], "torch is not asked while the environment names a directory"


def test_inductors_directory_is_torchs_default_where_the_environment_names_none(monkeypatch: pytest.MonkeyPatch,
                                                                                 torch_default: list[str]) -> None:
    monkeypatch.delenv(INDUCTOR, raising=False)
    assert KI.inductor_cache_dir() == "/torch/default"
    monkeypatch.setenv(INDUCTOR, "")
    assert KI.inductor_cache_dir() == "/torch/default", "an empty value names none"
    assert torch_default == ["asked", "asked"]


def test_tritons_directory_is_the_environments_string_or_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRITON, raising=False)
    assert KI.triton_cache_dir() is None
    for held in ("/pod/triton", "relative/dir", ""):
        monkeypatch.setenv(TRITON, held)
        assert KI.triton_cache_dir() == held


def test_neither_directory_is_read_from_the_others_variable(monkeypatch: pytest.MonkeyPatch, torch_default: list[str]) -> None:
    monkeypatch.setenv(TRITON, "/pod/triton")
    monkeypatch.delenv(INDUCTOR, raising=False)
    assert KI.inductor_cache_dir() == "/torch/default"
    monkeypatch.setenv(INDUCTOR, "/pod/inductor")
    monkeypatch.delenv(TRITON, raising=False)
    assert KI.triton_cache_dir() is None
