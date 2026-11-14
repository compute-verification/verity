"""The Pearl-C device path's kernel (`pouw_pearl_c_device`): GPU 1's pipeline loaded from its directory (``run.py`` and the
cubin), once per process per (sources, cubin), and the digests that name what ran.  A unit's keys are the scheme's
(`pearl_c.keys_of`), so the loader executes no other file of the directory but ``grouped.py``, which `grouped.module` loads.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from importlib.abc import Loader
    from importlib.machinery import ModuleSpec

#: the files of a kernel directory the device path executes, when present: ``run.py`` at `Kernel.load`, ``grouped.py`` at
#: `grouped.module`
SOURCES = ("run.py", "grouped.py")


def _load(name: str, path: Path) -> tuple[Any, str]:
    """`path` as a module and its bytes' SHA-256.  The module is named by the digest, so two kernel directories in one process
    never share a module, and the same bytes loaded twice are one module."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    key = f"{name}_{digest[:16]}"
    if key not in sys.modules:
        spec = cast("ModuleSpec", importlib.util.spec_from_file_location(key, path))   # None for a path that is no module's
        mod = sys.modules[key] = importlib.util.module_from_spec(spec)
        cast("Loader", spec.loader).exec_module(mod)
    return sys.modules[key], digest


def _the_cubin(directory: Path) -> Path:
    """A kernel directory's one cubin, when the caller names none."""
    found = sorted(directory.glob("*.cubin"))
    if len(found) != 1:
        raise ValueError(f"{directory}: {len(found)} cubins, not one: name the cubin")
    return found[0]


def source_digest(files: dict[str, str]) -> str:
    """The kernel sources' digest: SHA-256 over each file's name, a zero byte and its SHA-256, in name order."""
    return hashlib.sha256(b"".join(n.encode() + b"\0" + bytes.fromhex(files[n]) for n in sorted(files))).hexdigest()


@dataclass
class Kernel:
    """GPU 1's pipeline, loaded once per process per (sources, cubin): ``run.py`` (as `run`), the device (`dev`), and what
    names what ran, which the manifest records: the cubin's SHA-256, each of `SOURCES` the directory holds by its SHA-256
    (`files`), and their `source_digest`."""

    run: Any
    dev: Any
    cubin_sha256: str
    source_sha256: str | None = None
    files: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, directory: str, cubin: str | None = None) -> Kernel:
        d = Path(directory)
        run, run_sha = _load("pearl_c_run", d / "run.py")
        files = {"run.py": run_sha} | {n: hashlib.sha256((d / n).read_bytes()).hexdigest()
                                       for n in SOURCES[1:] if (d / n).is_file()}
        raw = Path(cubin or _the_cubin(d)).read_bytes()
        key = (source_digest(files), hashlib.sha256(raw).hexdigest())
        if key not in _KERNELS:
            dev = run.Dev(raw)
            dev.open()
            _KERNELS[key] = cls(run, dev, key[1], key[0], files)
        return _KERNELS[key]


#: (source_sha256, cubin_sha256) -> the loaded kernel
_KERNELS: dict[tuple[str, str], Kernel] = {}
