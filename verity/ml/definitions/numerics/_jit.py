"""Atomic, serialised JIT builds of the C++ twin libraries (``lib*.so``).

The models compile their ``.so`` on first use.  Concurrent first users -- forked replay
evaluators after a fresh install, two Commits on one pod -- used to race: one process' ``g++ -o lib.so`` was still writing while a
sibling ``dlopen``'ed the path (``OSError: lib.so: file too short``) and the sibling's proof blocks came back 'not evaluated'.  Here the compile
goes to a private temp path and is published by ``os.replace`` (atomic: a reader sees the old complete file or the new complete one,
never a partial), and builders serialise on an ``fcntl`` lock beside the library, re-checking staleness once they hold it (the second
builder finds the first's output fresh and returns).  Processes that already mapped an older file keep their mapping (POSIX unlink
semantics), so a rebuild never disturbs a running evaluator either.

The libraries are build products, not sources: they land in :func:`build_dir`, a per-user cache outside the checkout keyed by the
sources and the compiler flags (override with ``VERITY_NUMERICS_BUILD_DIR``), and are compiled on first use by whoever needs them.
Nothing is ever written under the repository: a build into ``cpp/build/`` left a ``lib*.so.*.tmp`` in the tree while a concurrent
check's repository guard looked.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

CPP_DIR = Path(__file__).resolve().parent / "cpp"
CXX_FLAGS = ("-O2", "-std=c++17", "-fPIC", "-shared", "-ffp-contract=off")


def cxx() -> str:
    """The C++ compiler the twins build with: ``$CXX``, else ``g++``."""
    return os.environ.get("CXX", "g++")


def source_digest() -> str:
    """sha256 over every file in ``cpp/`` (name and bytes), the compiler (:func:`cxx`) and ``CXX_FLAGS``: the key of the build cache."""
    h = hashlib.sha256()
    h.update(cxx().encode() + b"\0" + " ".join(CXX_FLAGS).encode() + b"\0")
    for p in sorted(CPP_DIR.iterdir()):
        if p.is_file():
            h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def build_dir() -> Path:
    """Where the JIT-built ``lib*.so`` live: ``$VERITY_NUMERICS_BUILD_DIR``, else
    ``${XDG_CACHE_HOME:-~/.cache}/verity/numerics/<source_digest()>/``.  Not created here."""
    env = os.environ.get("VERITY_NUMERICS_BUILD_DIR")
    if env:
        return Path(env).expanduser().resolve()
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache").expanduser()
    return cache.resolve() / "verity" / "numerics" / source_digest()


def lib_path(name: str) -> Path:
    """``build_dir() / name`` -- the publish path of one twin library (``libfa2_model.so``, ...)."""
    return build_dir() / name


def stale(lib: Path, *sources: Path) -> bool:
    if not lib.exists():
        return True
    try:
        m = lib.stat().st_mtime
        return any(m < s.stat().st_mtime for s in sources if s.exists())
    except FileNotFoundError:
        return True


@contextlib.contextmanager
def build_lock(lib: Path):
    """Exclusive advisory lock on ``<lib>.lock`` (fcntl where available; a no-op elsewhere -- the atomic publish still holds)."""
    lock = lib.with_name(lib.name + ".lock")
    fd = None
    try:
        fd = os.open(str(lock), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        yield
    finally:
        if fd is not None:
            try:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
            os.close(fd)


def atomic_build(lib: Path, sources: Sequence[Path], compile_to: Callable[[Path], None], *, force: bool = False) -> Path:
    """Publish ``lib`` from ``compile_to(tmp_path)`` atomically, serialised with other builders; skip when fresh (unless ``force``)."""
    lib = Path(lib)
    if not force and not stale(lib, *sources):
        return lib
    lib.parent.mkdir(parents=True, exist_ok=True)
    with build_lock(lib):
        if not force and not stale(lib, *sources):         # a sibling built it while we waited for the lock
            return lib
        fd, tmp = tempfile.mkstemp(prefix=lib.name + ".", suffix=f".tmp{os.getpid()}", dir=str(lib.parent))
        os.close(fd)
        tmp_p = Path(tmp)
        try:
            compile_to(tmp_p)
            os.replace(str(tmp_p), str(lib))                 # atomic publish: never a half-written library at ``lib``
        finally:
            with contextlib.suppress(FileNotFoundError):
                tmp_p.unlink()
    return lib


def cxx_compile(out: Path, src: Path, *, cwd: Path | None = None, extra: Sequence[str] = ()) -> None:
    """:func:`cxx` with ``CXX_FLAGS`` and OpenMP when the compiler has it, serial otherwise."""
    base = [cxx(), *CXX_FLAGS, *extra, "-o", str(out), str(src)]
    try:
        subprocess.run(base[:1] + ["-fopenmp"] + base[1:], check=True, cwd=str(cwd) if cwd else None, capture_output=True)
    except subprocess.CalledProcessError:
        subprocess.run(base + ["-Wno-unknown-pragmas"], check=True, cwd=str(cwd) if cwd else None)
