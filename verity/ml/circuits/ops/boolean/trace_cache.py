"""Traced circuits a build reuses across processes. A prover's or developer's build turns the cache on (`use`, or `use_env` from
``$VERITY_TRACE_CACHE``); a verifier never does, and nothing in `verity.core` names this module (``PROVER_ONLY`` in
``tests/test_boundaries.py``).

`trace.trace` with a key, as `trace.definition` and `elementwise`'s `specialized`, `shared_specialized` and `shared_operand` give
it (the Definition's name, version and bindings), looks the circuit up in the directory `use` named before it runs the builder,
and writes the circuit it traced there. An entry is named by the SHA-512 of that key with the parameter and return types, ``cse``,
Python's and numpy's versions, and a digest of the code that could change the gates (`code_digest`): every Python and JSON file
of `verity` but tests, every file of `verity.ml.circuits` but its tests, Markdown and TOML (so its tables), and the builder's
source file when it is elsewhere. So an edit anywhere in `verity` makes every entry a miss. What the digest can't see (a builder
outside `verity` that calls a helper from another module there, a monkeypatched builder) can make an entry stale; a stale or
tampered entry gives other gates, and so a Program digest unequal to one built without the cache. An entry whose bytes don't
match the SHA-512 its header records, or whose header doesn't hold its key, is traced again and rewritten.

An entry is ``MAGIC``, a line of JSON (the key, the circuit's sizes and counts, its arrays' type code and byte order, the payload's
SHA-512), then the payload: each gate's kind as one byte (`trace._KINDS`), each gate's first and second operand wires (a NOT's
second 0), and the output wires, as arrays of 4- or 8-byte unsigned ints.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import sys
from array import array
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

import verity.ml.circuits as _circuits
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.codec import _static_json, canonical_json
from verity.ml.circuits.types import Type

MAGIC = b"verity-trace-cache v1\n"
ENV = "VERITY_TRACE_CACHE"
_KIND = {k: i for i, k in enumerate(TR._KINDS)}
_CIRCUITS = Path(_circuits.__file__).resolve().parent
_VERITY = _CIRCUITS.parents[1]
#: directories `code_digest` never enters, besides hidden ones (`.lake`)
_SKIP = {"tests", "__pycache__", "target"}


def _covered(path: Path) -> bool:
    """Whether `code_digest` reads the file ``path`` (resolved)."""
    if not path.is_relative_to(_VERITY):
        return False
    if any(d in _SKIP or d.startswith(".") for d in path.relative_to(_VERITY).parts[:-1]):
        return False
    return path.suffix in (".py", ".json") or (path.is_relative_to(_CIRCUITS) and path.suffix not in (".md", ".toml"))


@functools.cache
def code_digest() -> str:
    """SHA-512 over the files `_covered` names, by path and bytes."""
    h = hashlib.sha512()
    for top, dirs, files in os.walk(_VERITY):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP and not d.startswith("."))
        for name in sorted(files):
            p = Path(top, name)
            if _covered(p):
                h.update(p.relative_to(_VERITY).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


@functools.cache
def _file_digest(path: str) -> str:
    return hashlib.sha512(Path(path).read_bytes()).hexdigest()


def _source(build: Callable) -> str | None:
    """The builder's source file when `code_digest` doesn't cover it (outside `verity`, or in tests), "" when it does, None when
    the builder has no code to name one."""
    while isinstance(build, functools.partial):
        build = build.func
    code = getattr(build, "__code__", None)
    if code is None or not os.path.isfile(code.co_filename):
        return None
    return "" if _covered(Path(code.co_filename).resolve()) else code.co_filename


class TraceCache:
    """The entries under ``directory``; ``hits``, ``misses`` and ``bad`` (entries traced again) count this process's lookups."""

    def __init__(self, directory: str | os.PathLike) -> None:
        self.dir = Path(directory)
        self.hits = self.misses = self.bad = 0

    def key(self, key: tuple, build: Callable, params: Sequence[Type], ret: Type, cse: bool) -> dict | None:
        """The entry's key, None for a builder or binding it can't name (traced, not cached)."""
        name, version, bindings = key
        src = _source(build)
        if src is None:
            return None
        try:
            statics = {k: _static_json(v) for k, v in bindings.items()}
        except TypeError:
            return None
        return {"name": name, "version": version, "bindings": statics, "params": [t.to_json() for t in params],
                "ret": ret.to_json(), "cse": cse, "code": code_digest(), "builder": _file_digest(src) if src else "",
                "python": "%d.%d" % sys.version_info[:2], "numpy": np.__version__}

    def path(self, k: dict) -> Path:
        d = hashlib.sha512(canonical_json(k)).hexdigest()
        return self.dir / d[:2] / f"{d}.trace"

    def trace(self, key: tuple, build: Callable, params: Sequence[Type], ret: Type, cse: bool) -> TR.Traced:
        k = self.key(key, build, params, ret, cse)
        if k is None:
            return TR.trace(build, params, ret, cse=cse)
        p = self.path(k)
        if p.exists():
            T = load(p, k)
            if T is not None:
                self.hits += 1
                return T
            self.bad += 1
        self.misses += 1
        T = TR.trace(build, params, ret, cse=cse)
        store(p, k, T)
        return T


def store(path: Path, key: dict, T: TR.Traced) -> None:
    """Write ``T`` under ``key`` at ``path``, whole or not at all (a temporary file, then a rename)."""
    tc = "I" if T.first_gate + len(T.gates) <= 1 << 32 else "Q"
    a = array(tc, (ops[0] for _, ops in T.gates))
    b = array(tc, (ops[1] if len(ops) > 1 else 0 for _, ops in T.gates))
    payload = b"".join((bytes(_KIND[g] for g, _ in T.gates), a.tobytes(), b.tobytes(), array(tc, T.outputs).tobytes()))
    head = {"key": key, "n_inputs": T.n_inputs, "gates": len(T.gates), "outputs": len(T.outputs), "counts": list(T.counts.items()),
            "type": tc, "byteorder": sys.byteorder, "sha512": hashlib.sha512(payload).hexdigest()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(MAGIC + canonical_json(head) + b"\n" + payload)
    os.replace(tmp, path)


def load(path: Path, key: dict) -> TR.Traced | None:
    """The circuit at ``path`` if it is a whole entry for ``key`` in this machine's byte order, else None."""
    try:
        with open(path, "rb") as f:
            if f.readline() != MAGIC:
                return None
            head = json.loads(f.readline())
            payload = f.read()
        if head["key"] != key or head["byteorder"] != sys.byteorder or hashlib.sha512(payload).hexdigest() != head["sha512"]:
            return None
        n, tc = head["gates"], head["type"]
        w = array(tc).itemsize
        if len(payload) != n + (2 * n + head["outputs"]) * w:
            return None
        mv, cols = memoryview(payload), []
        for lo, hi in ((n, n + n * w), (n + n * w, n + 2 * n * w), (n + 2 * n * w, len(payload))):
            a = array(tc)
            a.frombytes(mv[lo:hi])
            cols.append(a.tolist())
    except (OSError, ValueError, KeyError, TypeError):
        return None
    kinds, NOT = TR._KINDS, _KIND["not"]
    gates = [(kinds[g], (x,) if g == NOT else (x, y)) for g, x, y in zip(payload[:n], cols[0], cols[1])]
    return TR.Traced(head["n_inputs"], gates, cols[2], dict(head["counts"]))


def use(directory: str | os.PathLike | None) -> TraceCache | None:
    """Reuse traced circuits through ``directory`` in this process; None: trace every time."""
    TR.CACHE = TraceCache(directory) if directory else None
    return TR.CACHE


def use_env() -> TraceCache | None:
    """`use` the directory ``$VERITY_TRACE_CACHE`` names, if it names one: for a prover's or developer's entry point."""
    return use(os.environ.get(ENV) or None)

