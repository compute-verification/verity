"""`_jit.build_dir`: the C++ twin libraries build in a per-user cache outside the checkout, keyed by the sources and the compiler
flags; building them writes nothing under the repository; concurrent first users of one library resolve one path, compile it
once and load the whole library."""
from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import shutil
from pathlib import Path

import pytest

from verity.ml.definitions.numerics import _jit

REPO = Path(__file__).resolve().parents[4]
TWINS = (("libtc_model.so", "tc_model.cpp"), ("librms_triton_model.so", "rms_triton_model.cpp"), ("libfa2_model.so", "fa2_model.cpp"))
needs_cxx = pytest.mark.skipif(shutil.which(os.environ.get("CXX", "g++")) is None, reason="no C++ compiler")


def _files(root: Path) -> dict[str, int]:
    return {str(p.relative_to(root)): p.stat().st_size for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


def test_build_dir_is_a_user_cache_keyed_by_the_sources_and_the_compiler_flags(tmp_path, monkeypatch):
    monkeypatch.delenv("VERITY_NUMERICS_BUILD_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    key = _jit.source_digest()
    assert _jit.build_dir() == (tmp_path / "xdg").resolve() / "verity" / "numerics" / key
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert _jit.build_dir() == (tmp_path / "home").resolve() / ".cache" / "verity" / "numerics" / key

    monkeypatch.setenv("CXX", "clang++")
    assert _jit.source_digest() != key
    monkeypatch.delenv("CXX")
    monkeypatch.setattr(_jit, "CXX_FLAGS", _jit.CXX_FLAGS + ("-O3",))
    assert _jit.source_digest() != key
    monkeypatch.undo()

    copy = tmp_path / "cpp"
    shutil.copytree(_jit.CPP_DIR, copy, ignore=shutil.ignore_patterns("build"))
    monkeypatch.setattr(_jit, "CPP_DIR", copy)
    assert _jit.source_digest() == key
    (copy / "tc_model.cpp").write_text((copy / "tc_model.cpp").read_text() + "\n// edited\n")
    assert _jit.source_digest() != key

    monkeypatch.setenv("VERITY_NUMERICS_BUILD_DIR", str(tmp_path / "override"))
    assert _jit.build_dir() == (tmp_path / "override").resolve()


def test_no_twin_library_resolves_under_the_checkout():
    from verity.ml.definitions.numerics import cpu_model, fa2_model, rms_triton_model
    for lib in (cpu_model.LIB, fa2_model.LIB, rms_triton_model.LIB):
        assert not Path(lib).resolve().is_relative_to(REPO.resolve()), f"{lib} is inside the checkout {REPO}"


@needs_cxx
def test_building_every_twin_writes_nothing_under_the_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv("VERITY_NUMERICS_BUILD_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    kernels = _jit.CPP_DIR.parent
    before = _files(kernels)
    for name, src in TWINS:
        s = _jit.CPP_DIR / src
        lib = _jit.atomic_build(_jit.lib_path(name), [s], lambda out, s=s: _jit.cxx_compile(out, s, cwd=s.parent))
        assert lib.parent == _jit.build_dir() and lib.is_file()
        ctypes.CDLL(str(lib))
    assert _files(kernels) == before
    assert sorted(p.name for p in _jit.build_dir().iterdir() if not p.name.endswith(".lock")) == sorted(n for n, _ in TWINS)


@needs_cxx
def test_concurrent_first_users_resolve_one_path_compile_once_and_load_it(tmp_path, monkeypatch):
    monkeypatch.delenv("VERITY_NUMERICS_BUILD_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    src = _jit.CPP_DIR / "tc_model.cpp"
    compiles = tmp_path / "compiles"
    ctx = mp.get_context("fork")
    go, loaded = ctx.Event(), ctx.SimpleQueue()

    def compile_to(out: Path) -> None:
        with open(compiles, "ab") as fh:
            fh.write(b"1")
        _jit.cxx_compile(out, src)

    def first_user() -> None:
        go.wait()
        lib = _jit.atomic_build(_jit.lib_path("libtc_model.so"), [src], compile_to)
        ctypes.CDLL(str(lib)).tc_records
        loaded.put(str(lib))

    users = [ctx.Process(target=first_user) for _ in range(6)]
    for u in users:
        u.start()
    go.set()
    paths = [loaded.get() for _ in users]
    for u in users:
        u.join()
        assert u.exitcode == 0
    assert set(paths) == {str(_jit.build_dir() / "libtc_model.so")}
    assert compiles.read_bytes() == b"1", "exactly one compile across 6 concurrent first users"
    assert not list(_jit.build_dir().glob("*.tmp*")), "no temp file left behind"
