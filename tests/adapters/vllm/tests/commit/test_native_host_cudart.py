"""`NativeHostCommitter._load_cudart` finds the CUDA runtime of the fast copy: the libraries beside torch, then the ones the
`nvidia.cuda_runtime` package ships, then three sonames, in that order.  The first that loads and has `cudaMemcpyAsync` and
`cudaMemsetAsync` is the one, returned with the place it came from; a place that gives none is passed over.  When no place
gives one, the error names every place it asked and what each raised.

No runtime is loaded here: `ctypes.CDLL`, `glob.glob` and the package are stand-ins, so the test reads the same on a host with a
card and on one without."""
from __future__ import annotations

import ctypes
import glob
import importlib
import os
import sys
import types

import pytest

from verity_vllm.commit.committer import native_host as NH

BESIDE_TORCH = os.path.join(os.path.dirname(NH.torch.__file__), "lib", "libcudart*.so*")
PACKAGE = "/site/nvidia/cuda_runtime/__init__.py"
IN_PACKAGE = "/site/nvidia/cuda_runtime/lib/libcudart.so*"
SONAMES = ["libcudart.so", "libcudart.so.12", "libcudart.so.13"]
TORCHS, SHIPPED = "/torch/lib/libcudart.so.12", "/site/nvidia/cuda_runtime/lib/libcudart.so.12"


def runtime(*functions: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(**{f: types.SimpleNamespace() for f in functions or ("cudaMemcpyAsync", "cudaMemsetAsync")})


def places(monkeypatch, *, beside_torch=(), shipped=None, loads=None, package_raises=None) -> list[str]:
    """Stand-ins for the three places: the files beside torch, the files of the `nvidia.cuda_runtime` package (None: no such
    package) and what `ctypes.CDLL` gives for a name (`loads`; any other name is no library).  Returns the names `CDLL` is
    asked, in order, as they come."""
    asked: list[str] = []
    found = {BESIDE_TORCH: list(beside_torch), IN_PACKAGE: list(shipped or ())}
    real = glob.glob

    def cdll(name: str):
        asked.append(name)
        if name in (loads or {}):
            return loads[name]
        raise OSError(f"{name}: cannot open shared object file: No such file or directory")

    class Package:                                   # stands for the package's module, of which the loader reads `__file__` alone
        @property
        def __file__(self) -> str:                   # what the loader reads to find the package's `lib`
            if package_raises is not None:
                raise package_raises
            return PACKAGE

    monkeypatch.setattr(ctypes, "CDLL", cdll)
    monkeypatch.setattr(glob, "glob", lambda pattern, **kw: found[pattern] if "libcudart" in str(pattern) else real(pattern, **kw))
    if shipped is None and package_raises is None:
        monkeypatch.setitem(sys.modules, "nvidia", None)
    else:
        package = Package()
        monkeypatch.setitem(sys.modules, "nvidia", types.SimpleNamespace(cuda_runtime=package))
        monkeypatch.setitem(sys.modules, "nvidia.cuda_runtime", package)
    return asked


@pytest.mark.parametrize("the_one, before", [
    (TORCHS, []), (SHIPPED, [TORCHS]), ("libcudart.so", [TORCHS, SHIPPED]), ("libcudart.so.12", [TORCHS, SHIPPED, "libcudart.so"]),
    ("libcudart.so.13", [TORCHS, SHIPPED, "libcudart.so", "libcudart.so.12"])])
def test_the_first_place_that_loads_is_the_runtime_and_the_places_are_asked_in_order(monkeypatch, the_one, before):
    rt = runtime()
    asked = places(monkeypatch, beside_torch=[TORCHS], shipped=[SHIPPED], loads={the_one: rt})
    got, where = NH.NativeHostCommitter._load_cudart()
    assert got is rt and where == the_one and asked == [*before, the_one]
    v, n, i = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    assert (rt.cudaMemcpyAsync.argtypes, rt.cudaMemcpyAsync.restype) == ([v, v, n, i, v], i)
    assert (rt.cudaMemsetAsync.argtypes, rt.cudaMemsetAsync.restype) == ([v, i, n, v], i)


def test_with_no_package_the_sonames_follow_the_files_beside_torch(monkeypatch):
    rt = runtime()
    asked = places(monkeypatch, beside_torch=["/torch/lib/libcudart.so.11", TORCHS], loads={"libcudart.so.12": rt})
    assert NH.NativeHostCommitter._load_cudart() == (rt, "libcudart.so.12")
    assert asked == ["/torch/lib/libcudart.so.11", TORCHS, "libcudart.so", "libcudart.so.12"]


def test_a_package_that_cannot_say_where_it_is_is_passed_over(monkeypatch):
    rt = runtime()
    asked = places(monkeypatch, package_raises=TypeError("no file"), loads={"libcudart.so": rt})
    assert NH.NativeHostCommitter._load_cudart() == (rt, "libcudart.so") and asked == ["libcudart.so"]


@pytest.mark.parametrize("has", [("cudaMemcpyAsync",), ("cudaMemsetAsync",)])
def test_a_library_without_both_functions_is_not_the_runtime(monkeypatch, has):
    rt = runtime()
    asked = places(monkeypatch, beside_torch=[TORCHS], loads={TORCHS: runtime(*has), "libcudart.so.13": rt})
    assert NH.NativeHostCommitter._load_cudart() == (rt, "libcudart.so.13") and asked == [TORCHS, *SONAMES]


def no_library(name: str) -> str:
    return f"{name} (OSError: {name}: cannot open shared object file: No such file or directory)"


def test_no_runtime_anywhere_is_an_error_that_names_each_place_and_what_it_raised(monkeypatch):
    asked = places(monkeypatch, beside_torch=[TORCHS], shipped=[SHIPPED], loads={SHIPPED: runtime("cudaMemcpyAsync")})
    with pytest.raises(RuntimeError) as e:
        NH.NativeHostCommitter._load_cudart()
    assert asked == [TORCHS, SHIPPED, *SONAMES]
    assert str(e.value) == "libcudart not found; tried " + ", ".join([
        no_library(TORCHS), f"{SHIPPED} (AttributeError: 'types.SimpleNamespace' object has no attribute 'cudaMemsetAsync')",
        *map(no_library, SONAMES)])


@pytest.mark.parametrize("package_raises, said", [(None, None), (TypeError("no file"), "TypeError: no file")], ids=["no-package", "no-file"])
def test_the_error_names_the_package_first_when_it_gave_no_place(monkeypatch, package_raises, said):
    asked = places(monkeypatch, package_raises=package_raises)
    if said is None:
        with pytest.raises(ImportError) as absent:
            importlib.import_module("nvidia.cuda_runtime")
        said = f"{type(absent.value).__name__}: {absent.value}"
    with pytest.raises(RuntimeError) as e:
        NH.NativeHostCommitter._load_cudart()
    assert asked == SONAMES
    assert str(e.value) == "libcudart not found; tried " + ", ".join([f"nvidia.cuda_runtime ({said})", *map(no_library, SONAMES)])
