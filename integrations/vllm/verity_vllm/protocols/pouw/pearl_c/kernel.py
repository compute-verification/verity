"""The Pearl-C device path's kernel (`pouw_pearl_c_device`): GPU 1's pipeline loaded from its directory (``run.py``,
``fixture.py`` and the cubin), once per process per (sources, cubin), and the digests that name what ran.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load(name: str, path: Path) -> tuple[Any, str]:
    """`path` as a module and its bytes' SHA-256.  The module is named by the digest, so two kernel directories in one process
    never share a module, and the same bytes loaded twice are one module."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    key = f"{name}_{digest[:16]}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        mod = sys.modules[key] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[key], digest


def _the_cubin(directory: Path) -> Path:
    """A kernel directory's one cubin, when the caller names none."""
    found = sorted(directory.glob("*.cubin"))
    if len(found) != 1:
        raise ValueError(f"{directory}: {len(found)} cubins, not one: name the cubin")
    return found[0]


def source_digest(run_sha256: str, fixture_sha256: str) -> str:
    """The kernel sources' digest: `run.py`'s and `fixture.py`'s, in that order."""
    return hashlib.sha256(bytes.fromhex(run_sha256) + bytes.fromhex(fixture_sha256)).hexdigest()


@dataclass
class Kernel:
    """GPU 1's pipeline, loaded once per process per (sources, cubin): ``run.py`` (as `run`), ``fixture.py`` (as `fixture`), the
    device (`dev`), and the digests that name what ran (the manifest records both)."""

    run: Any
    fixture: Any
    dev: Any
    cubin_sha256: str
    source_sha256: str | None = None

    @classmethod
    def load(cls, directory: str, cubin: str | None = None) -> Kernel:
        d = Path(directory)
        run, run_sha = _load("pearl_c_run", d / "run.py")
        fixture, fixture_sha = _load("pearl_c_fixture", d / "fixture.py")
        raw = Path(cubin or _the_cubin(d)).read_bytes()
        key = (source_digest(run_sha, fixture_sha), hashlib.sha256(raw).hexdigest())
        if key not in _KERNELS:
            dev = run.Dev(raw)
            dev.open()
            _KERNELS[key] = cls(run, fixture, dev, key[1], key[0])
        return _KERNELS[key]


#: (source_sha256, cubin_sha256) -> the loaded kernel
_KERNELS: dict[tuple[str, str], Kernel] = {}
