"""How memory accounting's benchmarks open a built CUDA library: `load(path, functions, optional)` is `ctypes.CDLL(path)`
with each of `functions`' argument and result types set, and each of `optional`'s where the library exports it. A library
that lacks one of `functions` raises ctypes' AttributeError, which names it. A signature is a list of argument types (an
int result, ctypes' default) or a pair (argument types, result type), the result None for void."""
from __future__ import annotations

import ctypes
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

Signature = Sequence[Any] | tuple[Sequence[Any], Any]


def load(path: str | Path, functions: Mapping[str, Signature],
         optional: Mapping[str, Signature] | None = None) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    for name, sig in functions.items():
        bind(getattr(lib, name), sig)
    for name, sig in (optional or {}).items():
        if hasattr(lib, name):
            bind(getattr(lib, name), sig)
    return lib


def bind(fn: Any, sig: Signature) -> None:
    args, result = sig if isinstance(sig, tuple) else (sig, ctypes.c_int)
    fn.argtypes, fn.restype = list(args), result
