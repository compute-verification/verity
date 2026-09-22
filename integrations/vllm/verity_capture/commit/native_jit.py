"""The native collector's JIT build is keyed by WHAT IT COMPILES, never by mtimes (COORD R17-41 (1), stoch M-0756 / rel M-0803).

torch.utils.cpp_extension.load() decides "up to date" by comparing the .so's mtime with the sources'.  record_v5/ship.sh PRESERVES source
mtimes, and /workspace/cp/nc_build is one pod-wide directory: on cp-big an .so built 22:15:25Z from the previous tree's sources was adopted
by the r17-int-3 ship (its native_collect.cpp is dated 21:35:09Z) -- every Commit there died at the first plan-hit step
(`Collector.plan_version` absent, rc 134) while the B0 canary, which hits no plan, passed.  Two rules, both torch-free so the laptop and the
pod CPU gate test them:

  * `source_set_digest()`   sha256 over the source files (bytes, in a fixed order), the compiler flags and the torch/CUDA version strings;
    `keyed_build_dir(bd, digest)` = `<bd>/<digest12>` -- a different source set can never find another's .so, shared dirs are safe,
    mtimes are irrelevant.  The digest is recorded beside the .so (RELEASE.json artifacts.verity_native_collect.so.source_digest).
  * `binding_names(cpp)` / `missing_bindings(module, cpp)`   every pybind `.def*("name"` of the source must be an attribute of the loaded
    module or of one of its classes; the pod readiness FAILS on a missing one (a stale .so is no longer canary-blind).
"""
from __future__ import annotations

import hashlib
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ("native_collect.cpp", "native_gather.cu", "native_tree.cu")
CFLAGS = ("-O3",)
CUDA_CFLAGS = ("-O3", "--use_fast_math")

_DEF = re.compile(r'\.def(?:_readonly|_readwrite|_property_readonly|_property|_static)?\(\s*"([A-Za-z_][A-Za-z0-9_]*)"')
_CLASS = re.compile(r'py::class_<[^>]+>\(\s*m\s*,\s*"(\w+)"')


def source_paths(here: str = HERE) -> list[str]:
    return [os.path.join(here, s) for s in SOURCES]


def source_set_digest(paths: list[str] | None = None, *, cflags=CFLAGS, cuda_cflags=CUDA_CFLAGS, torch_version: str = "",
                      cuda_version: str = "") -> str:
    """sha256 of the source SET: each file's basename + bytes in the fixed SOURCES order, then the flags and the toolchain versions."""
    h = hashlib.sha256()
    for p in paths if paths is not None else source_paths():
        h.update(os.path.basename(p).encode())
        h.update(b"\0")
        with open(p, "rb") as f:
            h.update(f.read())
        h.update(b"\0")
    h.update(("cflags=" + " ".join(cflags) + "|cuda_cflags=" + " ".join(cuda_cflags)).encode())
    h.update(("|torch=" + torch_version + "|cuda=" + cuda_version).encode())
    return h.hexdigest()


def keyed_build_dir(build_dir: str, digest: str) -> str:
    """`<build_dir>/<digest12>`: the JIT dir a caller names is the PARENT; the source set picks the child."""
    return os.path.join(build_dir, digest[:12])


def binding_names(cpp_path: str) -> set[str]:
    """Every pybind `.def*("name"` in the source (methods, properties, module functions)."""
    with open(cpp_path, encoding="utf-8", errors="replace") as f:
        return set(_DEF.findall(f.read()))


def class_names(cpp_path: str) -> set[str]:
    with open(cpp_path, encoding="utf-8", errors="replace") as f:
        return set(_CLASS.findall(f.read()))


def missing_bindings(module, cpp_path: str) -> list[str]:
    """The names the source binds that the LOADED module does not expose (on the module or on any of its classes) -- [] when the .so was
    compiled from this source.  Dunder names (`__call__`) are checked on the classes like any other."""
    have: set[str] = set(dir(module))
    for c in class_names(cpp_path):
        cls = getattr(module, c, None)
        if cls is not None:
            have |= set(dir(cls))
    missing = sorted(n for n in binding_names(cpp_path) if n not in have)
    missing_cls = sorted(c for c in class_names(cpp_path) if not hasattr(module, c))
    return missing_cls + missing
