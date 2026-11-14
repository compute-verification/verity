"""The two modules that patch a FlashAttention tree for the tap, `verity_vllm.acquire.fa2_tap_src.apply_fa2_tap` and
`verity_vllm.acquire.fa3_tap_src.apply_fa3_tap`: an import of one does nothing, and its `main` does the work.

Torch-free, and with no FlashAttention tree: a `main` is stopped at its first step that needs one.
"""

from __future__ import annotations

import importlib
import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

FA2 = "verity_vllm.acquire.fa2_tap_src.apply_fa2_tap"
FA3 = "verity_vllm.acquire.fa3_tap_src.apply_fa3_tap"


def _dir(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin is not None
    return Path(spec.origin).parent


class _Ran(Exception):
    """Raised in place of a child process, so that no test starts one."""


@pytest.fixture
def ran(monkeypatch):
    calls: list[tuple[tuple, dict]] = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        raise _Ran

    monkeypatch.setattr(subprocess, "run", run)
    return calls


@pytest.mark.parametrize("name", [FA2, FA3])
@pytest.mark.parametrize("arguments", [0, 2])
def test_an_import_does_nothing(name, arguments, tmp_path, monkeypatch, capsys, ran):
    """Without arguments or with two directories, the module's statements read no argument, write no file, start no
    process and print nothing."""
    tree = tmp_path / "tree"
    tree.mkdir()
    monkeypatch.setattr(sys, "argv", ["x", str(tree), str(_dir(name))][:1 + arguments])
    runpy.run_module(name)
    assert ran == [] and list(tree.iterdir()) == [] and capsys.readouterr().out == ""


def test_fa2_applies_its_patch_first_and_takes_this_directory_by_default(tmp_path, ran):
    apply_fa2_tap = importlib.import_module(FA2)
    src = tmp_path / "csrc" / "flash_attn" / "src"
    src.mkdir(parents=True)
    with pytest.raises(_Ran):
        apply_fa2_tap.main(apply_fa2_tap.Options(src=str(src)))
    assert ran == [((["git", "apply", str(_dir(FA2) / "verity_tap_v7.patch")],), {"cwd": str(tmp_path), "check": True})]


def test_fa3_copies_its_headers_first_and_takes_this_directory_by_default(tmp_path, ran):
    apply_fa3_tap = importlib.import_module(FA3)
    hop = tmp_path / "hopper"
    hop.mkdir()
    with pytest.raises(FileNotFoundError, match="softmax.h"):
        apply_fa3_tap.main(apply_fa3_tap.Options(hopper=str(hop)))
    headers = ["verity_tap_fa3.h", "verity_tap_params.h"]
    assert sorted(p.name for p in hop.iterdir()) == headers and ran == []
    assert all((hop / h).read_bytes() == (_dir(FA3) / h).read_bytes() for h in headers)
