"""The native collector's JIT build is keyed by WHAT IT COMPILES, never by mtimes.

torch.utils.cpp_extension.load() decides "up to date" by comparing the .so's mtime with the sources'.  A ship that PRESERVES source
mtimes into one pod-wide build directory (/workspace/cp/nc_build) lets an .so built from another tree's sources look newer than this
tree's sources, and load() adopts it: every Commit then dies at the first plan-hit step (`Collector.plan_version` absent, rc 134) while a canary that hits no plan passes.  Two rules,
both torch-free so the laptop and the pod CPU gate test them:

  * `source_set_digest()`   sha256 over the source files (bytes, in a fixed order), the compiler flags and the torch/CUDA version strings;
    `keyed_build_dir(bd, digest)` = `<bd>/<digest12>` -- a different source set can never find another's .so, shared dirs are safe,
    mtimes are irrelevant.  The digest is recorded beside the .so (RELEASE.json artifacts.verity_native_collect.so.source_digest).
  * `binding_names(cpp)` / `missing_bindings(module, cpp)`   every pybind `.def*("name"` of the source must be an attribute of the loaded
    module or of one of its classes; the pod readiness FAILS on a missing one (a stale .so is no longer canary-blind).

`jit_load()` is how every torch JIT extension of the Commit loads (the collector, `hidden_gpu_tree`, `verity_leafhash`), safe for several
jobs on one host at once: the source set keys the build dir, and builders serialise on an fcntl lock beside it (`build_lock`).  The kernel
drops that lock however its holder exits, so under it a torch FileBaton `lock` left in the dir can only be a killed build's, and it is
removed -- never guessed from its age (a build on a loaded host can outlast any age limit).
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import re
import sys
from collections.abc import Callable, Sequence
from typing import Any

from verity.ml.kernels import SHA512

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ("native_collect.cpp", "native_gather.cu", "native_tree.cu", "native_rows.cu")
#: headers a source includes by name from outside the sources' directory: the GPU's one SHA-512
SHARED = {"sha512.cuh": str(SHA512)}
CFLAGS = ("-O3",)
CUDA_CFLAGS = ("-O3", "--use_fast_math")

_DEF = re.compile(r'\.def(?:_readonly|_readwrite|_property_readonly|_property|_static)?\(\s*"([A-Za-z_][A-Za-z0-9_]*)"')
_CLASS = re.compile(r'py::class_<[^>]+>\(\s*m\s*,\s*"(\w+)"')


_INCLUDE = re.compile(rb'^[ \t]*#[ \t]*include[ \t]*"([^"]+)"', re.M)


def source_paths(here: str = HERE) -> list[str]:
    return [os.path.join(here, s) for s in SOURCES]


def local_headers(sources: Sequence[str], shared: dict[str, str] = SHARED) -> list[tuple[str, str]]:
    """(path relative to the sources' directory, absolute path) of every header the sources include by a quoted path that exists beside
    the including file, transitively, sorted.  A name of `shared` that a file in the sources' directory includes and finds nowhere
    beside it is that header, staged at that name.  Any other quoted include found nowhere beside it is the compiler's (-I) business
    and is skipped.  Refused: sources with one basename twice, sources in different directories that include headers, and a header
    outside the sources' directory -- a staged copy (`<build_dir>/src/`) could not hold either faithfully."""
    names = [os.path.basename(p) for p in sources]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"JIT sources share basenames {dup}: their staged copies in one src/ would overwrite each other")
    found: dict[str, str] = {}
    todo = [os.path.abspath(p) for p in sources]
    roots = {os.path.dirname(p) for p in todo}
    seen: set[str] = set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        with open(p, "rb") as f:
            body = f.read()
        for inc in _INCLUDE.findall(body):
            name = inc.decode()
            h = os.path.normpath(os.path.join(os.path.dirname(p), name))
            rel = None
            if not os.path.isfile(h):
                if name not in shared or os.path.dirname(p) not in roots:
                    continue
                rel, h = name, shared[name]
            if len(roots) > 1:
                raise ValueError(f"JIT sources in several directories {sorted(roots)} include local header {h}: cannot stage one src/ for them")
            if rel is None:
                rel = os.path.relpath(h, next(iter(roots)))
                if rel.startswith(os.pardir + os.sep):
                    raise ValueError(f"JIT source {p} includes {h}, outside the sources' directory: its staged copy could not resolve it")
            found[rel] = h
            todo.append(h)
    return sorted(found.items())


def source_set_digest(paths: list[str] | None = None, *, cflags=CFLAGS, cuda_cflags=CUDA_CFLAGS, torch_version: str = "",
                      cuda_version: str = "") -> str:
    """sha256 of the source SET: each file's basename + bytes in the fixed SOURCES order, then each local header's relative path + bytes
    (`local_headers`; nothing when there is none, so a header-free set keeps its digest), then the flags and the toolchain versions."""
    h = hashlib.sha256()
    paths = paths if paths is not None else source_paths()
    for p in paths:
        h.update(os.path.basename(p).encode())
        h.update(b"\0")
        with open(p, "rb") as f:
            h.update(f.read())
        h.update(b"\0")
    for rel, p in local_headers(paths):
        h.update(b"header:" + rel.encode() + b"\0")
        with open(p, "rb") as f:
            h.update(f.read())
        h.update(b"\0")
    h.update(("cflags=" + " ".join(cflags) + "|cuda_cflags=" + " ".join(cuda_cflags)).encode())
    h.update(("|torch=" + torch_version + "|cuda=" + cuda_version).encode())
    return h.hexdigest()


def keyed_build_dir(build_dir: str, digest: str) -> str:
    """`<build_dir>/<digest12>`: the JIT dir a caller names is the PARENT; the source set picks the child."""
    return os.path.join(build_dir, digest[:12])


def default_parent(name: str) -> str:
    """torch's default extension cache for `name`, the directory `load(name=...)` uses when it is given none."""
    from torch.utils.cpp_extension import _get_build_directory

    return _get_build_directory(name, verbose=False)


def staged_sources(build_dir: str, sources: Sequence[str]) -> list[str]:
    """The sources copied into `<build_dir>/src/` (rewritten only when their bytes differ), the paths torch compiles.  torch writes the
    source paths into build.ninja and ninja rebuilds when that file changes, so compiling each tree's own copy rebuilt the shared,
    digest-keyed dir for every job of another tree (~2 min of nvcc under the build lock per GPU job).  A digest-keyed
    dir holds one source set, so its staged copy is the same file for every tree.  The local headers they include (`local_headers`) are
    staged at their relative paths beside them, so the copies compile as the originals do."""
    sd = os.path.join(build_dir, "src")
    os.makedirs(sd, exist_ok=True)
    out = []
    headers = local_headers(sources)
    for rel, p in [(os.path.basename(s), s) for s in sources] + headers:
        with open(p, "rb") as f:
            body = f.read()
        dst = os.path.join(sd, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            with open(dst, "rb") as f:
                same = f.read() == body
        except FileNotFoundError:
            same = False
        if not same:
            tmp = f"{dst}.{os.getpid()}.tmp"
            with open(tmp, "wb") as f:
                f.write(body)
            os.replace(tmp, dst)
        out.append(dst)
    return out[:len(sources)]


@contextlib.contextmanager
def build_lock(build_dir: str):
    """Exclusive fcntl lock on `<build_dir>.build-lock`, beside the build dir: one builder at a time, released by the kernel when its holder
    exits, however it exits."""
    fd = os.open(build_dir + ".build-lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def jit_load(name: str, sources: Sequence[str], *, parent: str | None = None, cflags: Sequence[str] = (), cuda_cflags: Sequence[str] = (),
             with_cuda: bool = True, verbose: bool = False, loader: Callable[..., Any] | None = None, torch_version: str | None = None,
             cuda_version: str | None = None) -> tuple[Any, str, str]:
    """torch's `cpp_extension.load` for one extension into `<parent>/<digest12>` (`parent`: the caller's, else torch's default for
    `name`), under `build_lock`, compiled from the dir's staged copy of `sources` (`staged_sources`); a torch `lock` found in the dir while
    holding it is removed.  `loader` and the toolchain versions are
    torch's unless given (tests).  Returns (module, build dir, source-set digest)."""
    if loader is None or torch_version is None or cuda_version is None:
        import torch
        from torch.utils.cpp_extension import load

        loader = loader or load
        torch_version = str(torch.__version__) if torch_version is None else torch_version
        cuda_version = str(torch.version.cuda) if cuda_version is None else cuda_version
    digest = source_set_digest(list(sources), cflags=tuple(cflags), cuda_cflags=tuple(cuda_cflags), torch_version=torch_version,
                               cuda_version=cuda_version)
    bd = keyed_build_dir(parent or default_parent(name), digest)
    os.makedirs(bd, exist_ok=True)
    with build_lock(bd):
        stale = os.path.join(bd, "lock")
        if os.path.exists(stale):
            print(f"[jit] removing {stale}: no live build holds {bd}.build-lock, so a killed build left it", file=sys.stderr, flush=True)
            with contextlib.suppress(FileNotFoundError):
                os.remove(stale)
        module = loader(name=name, sources=staged_sources(bd, sources), extra_cflags=list(cflags), extra_cuda_cflags=list(cuda_cflags),
                        with_cuda=with_cuda, build_directory=bd, verbose=verbose)
    return module, bd, digest


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
