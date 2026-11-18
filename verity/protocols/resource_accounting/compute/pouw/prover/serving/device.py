"""The one toolchain reader of PoUW's builds: the C++ compiler (`compiler`) and nvcc (`nvcc`) every kernel library under
`pouw_cuda/` and the device adapters' H1 libraries build with, each `TOOLCHAIN`'s when `configure` set it, else the first on
the path. Each build takes its target from the device it runs on."""
from __future__ import annotations

import shutil
from dataclasses import dataclass


@dataclass
class Toolchain:
    """The C++ compiler and nvcc the builds use (None: the first on the path)."""

    cxx: str | None = None
    nvcc: str | None = None


TOOLCHAIN = Toolchain()


def configure(**changes) -> Toolchain:
    """Set `TOOLCHAIN`'s fields (``cxx``, ``nvcc``; None leaves one as it is)."""
    for key, value in changes.items():
        if not hasattr(TOOLCHAIN, key):
            raise TypeError(f"no toolchain field {key!r}")
        if value is not None:
            setattr(TOOLCHAIN, key, value)
    return TOOLCHAIN


def compiler() -> str | None:
    """`TOOLCHAIN`'s C++ compiler, or the first of g++, c++ and clang++ on the path."""
    return TOOLCHAIN.cxx or next((c for c in ("g++", "c++", "clang++") if shutil.which(c)), None)


def nvcc() -> str | None:
    """`TOOLCHAIN`'s nvcc, or nvcc on the path."""
    return TOOLCHAIN.nvcc or shutil.which("nvcc")


__all__ = ["TOOLCHAIN", "Toolchain", "compiler", "configure", "nvcc"]
