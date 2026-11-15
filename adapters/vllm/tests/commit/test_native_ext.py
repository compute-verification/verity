"""`verity_vllm.commit.committer.native_ext`: the collector's extension loads once a process through `native_jit.jit_load`, logs the
source digest and the directory it was built in, and is refused and not kept when it lacks a name its source binds.  The loader and
the binding check are stand-ins: no compiler and no torch."""
from __future__ import annotations

import logging
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from verity_vllm.commit.committer import native_ext, native_jit

DIGEST = "0123456789abcdef" * 4


@pytest.fixture
def loads(monkeypatch) -> tuple[list[dict[str, Any]], SimpleNamespace, list[str]]:
    """Each `jit_load` call's arguments, the module the stand-in returns, and the names the stand-in binding check reports missing."""
    calls: list[dict[str, Any]] = []
    built = SimpleNamespace()
    missing: list[str] = []

    def jit_load(name: str, sources: list[str], **kw: Any) -> tuple[SimpleNamespace, str, str]:
        calls.append({"name": name, "sources": list(sources), **kw})
        return built, f"{kw['parent']}/{DIGEST[:12]}", DIGEST

    monkeypatch.setattr(native_ext, "_EXT", None)
    monkeypatch.setattr(native_jit, "jit_load", jit_load)
    monkeypatch.setattr(native_jit, "missing_bindings", lambda module, cpp_path: list(missing))
    return calls, built, missing


def test_the_extension_loads_once_and_logs_where_it_was_built(loads, caplog, capsys, tmp_path):
    calls, built, _missing = loads
    caplog.set_level(logging.INFO, logger="verity_vllm")
    assert native_ext.ext(build_dir=str(tmp_path)) is built
    assert calls == [{"name": "verity_native_collect", "sources": native_jit.source_paths(native_ext.HERE), "parent": str(tmp_path),
                      "cflags": native_jit.CFLAGS, "cuda_cflags": native_jit.CUDA_CFLAGS, "verbose": False}]
    (record,) = caplog.records
    assert (record.name, record.levelname) == ("verity_vllm.commit.committer.native_ext", "INFO")
    assert record.getMessage() == f"[jit] native collector source digest 0123456789ab -> {tmp_path}/0123456789ab"
    assert native_ext.ext() is built and native_ext.ext(build_dir="elsewhere") is built
    assert len(calls) == 1 and len(caplog.records) == 1, "one load a process"
    assert capsys.readouterr().out == "", "nothing on stdout"


def test_an_extension_that_lacks_a_bound_name_is_refused_and_not_kept(loads, caplog, tmp_path):
    calls, built, missing = loads
    caplog.set_level(logging.INFO, logger="verity_vllm")
    missing.append("commit_rows")
    with pytest.raises(RuntimeError) as e:
        native_ext.ext(build_dir=str(tmp_path))
    said = str(e.value)
    assert f"native collector .so in {tmp_path}/0123456789ab does not expose ['commit_rows']" in said
    assert "a stale build (source digest 0123456789ab); remove the directory and rebuild" in said
    assert native_ext._EXT is None and not caplog.records
    missing.clear()
    assert native_ext.ext(build_dir=str(tmp_path)) is built and len(calls) == 2, "the refused module was not kept"


def test_importing_the_module_loads_neither_numpy_nor_torch_nor_the_core():
    """`verity_vllm.commit.committer.native_ext`: the loader (`native_jit`), which reads the kernels' headers from the core `verity`, is
    read when `ext` builds, not when this module is imported."""
    code = ("import sys; before = set(sys.modules); import verity_vllm.commit.committer.native_ext; "
            "print(sorted({m.split('.')[0] for m in set(sys.modules) - before} & {'numpy', 'torch', 'verity'}))")
    child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "[]"
