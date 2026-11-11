"""`_jit.build_dir`: the C++ twin libraries build in a per-user cache outside the checkout, keyed by the sources and the compiler
flags; building them writes nothing under the repository; concurrent first users of one library resolve one path, compile it
once and load the whole library; each loads as a `_jit.Model` naming every file its build compiles."""
from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import re
import shutil
import sys
import types
from pathlib import Path

import pytest

from verity.definitions.numerics import _jit

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


def test_every_cpp_model_loads_as_a_model_naming_every_file_its_build_compiles(monkeypatch):
    """circuit-check keys a check that called a C++ model on the files its `_jit.Model` names (`circuit_check.cache.MODEL`),
    wherever the library was built.  So each model's library is a `Model` naming its build's sources, those hold every file a
    source includes, every file of `cpp/` is some model's, and nothing here loads a library with ctypes but as a `_jit.Model`."""
    from verity.definitions.numerics import cpu_model, dense_rows, fa2_model, fa3_model, rms_triton_model

    built: dict[str, tuple[Path, ...]] = {}

    def atomic_build(lib, sources, compile_to, *, force=False):
        built[Path(lib).name] = tuple(Path(s) for s in sources)
        return Path(lib)

    class Model(types.SimpleNamespace):
        def __init__(self, path, sources):
            super().__init__(path=Path(path), sources=tuple(Path(s) for s in sources))

        def __getattr__(self, name):                   # a function, which the loaders give argtypes and a restype
            setattr(self, name, fn := types.SimpleNamespace())
            return fn

    monkeypatch.setattr(_jit, "atomic_build", atomic_build)
    monkeypatch.setattr(_jit, "Model", Model)
    for m in (cpu_model, fa2_model, fa3_model, rms_triton_model):
        monkeypatch.setattr(m, "_lib", None)
    dense_rows.k16_lib.cache_clear()
    try:
        libs = [cpu_model.lib(), fa2_model.lib(), fa3_model.lib(), rms_triton_model.lib(), dense_rows.k16_lib()]
    finally:
        dense_rows.k16_lib.cache_clear()
    assert all(isinstance(lib, Model) for lib in libs)
    assert {lib.path.name: lib.sources for lib in libs} == built, "a model names the sources its build compiles"
    for lib in libs:
        for s in lib.sources:
            for inc in re.findall(r'^\s*#\s*include\s+"([^"]+)"', s.read_text(), re.M):
                assert (s.parent / inc).resolve() in {x.resolve() for x in lib.sources}, f"{s.name} includes {inc}: name it"
    assert {s.name for lib in libs for s in lib.sources} == {p.name for p in _jit.CPP_DIR.glob("*.cpp")}
    for f in sorted(_jit.CPP_DIR.parent.rglob("*.py")):
        assert not re.search(r"\b(?:CDLL|PyDLL|LoadLibrary)\s*\(|\b(?:cdll|pydll)\.", f.read_text()), \
            f"{f.name} loads a library with ctypes itself: load a model's as `_jit.Model(path, sources)`"


@needs_cxx
def test_a_models_every_call_raises_its_sources_whatever_its_build_directory(tmp_path, monkeypatch):
    """`_jit.CALLED` with the model's sources, absolute, at each call of a function of its library, once it was loaded: a
    library is loaded once a process, by whichever check first runs it, and its build directory need not be keyed on them."""
    monkeypatch.setenv("VERITY_NUMERICS_BUILD_DIR", str(tmp_path / "libs"))
    src = tmp_path / "m.cpp"
    src.write_text('extern "C" int answer(int x) { return x + 41; }\n')
    lib = _jit.atomic_build(_jit.lib_path("libanswer.so"), [src], lambda out: _jit.cxx_compile(out, src))
    model = _jit.Model(lib, [src])
    seen: list[tuple] = []
    on = [True]

    def hook(event, args):                             # an audit hook stays for the process: this one only while the test runs
        if on and event == _jit.CALLED:
            seen.append(args)

    sys.addaudithook(hook)
    try:
        model.answer.restype = ctypes.c_int
        assert seen == []
        assert [model.answer(1), model.answer(2)] == [42, 43]
    finally:
        on.clear()
    assert seen == [((str(src.resolve()),),)] * 2


def test_no_twin_library_resolves_under_the_checkout():
    from verity.definitions.numerics import cpu_model, fa2_model, rms_triton_model
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
