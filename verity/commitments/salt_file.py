"""A prover's salt file: each registered parameter's 192-byte row salts, by parameter name, in a directory beside its
registration, readable by its owner alone and never part of a record or an uploaded output.

``<R>.salts/`` beside the registration ``<R>.json`` (`beside`) holds ``index.json`` (``{"schema", "params": {name: {"file",
"rows"}}}``) and, per parameter, ``<i>.bin``: its rows' salts in row order, row j at bytes [192 j, 192 j + 192).
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

__all__ = ["INDEX", "SALT_BYTES", "SCHEMA", "beside", "params", "read", "write"]

SALT_BYTES = 192
INDEX = "index.json"
SCHEMA = "verity/salt-file/v1"


def beside(registration: str | Path) -> Path:
    """The salt file of the registration at ``registration``: ``<R>.salts`` for ``<R>.json``."""
    return Path(registration).with_suffix(".salts")


def _private(path: Path, data: bytes) -> None:
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
        os.fchmod(f.fileno(), 0o600)
        f.write(data)


def write(path: str | Path, salts: Mapping[str, bytes]) -> Path:
    """``salts[name]`` (its rows' salts, concatenated) written as the salt file ``path``, replacing one there."""
    d = Path(path)
    for name, s in salts.items():
        if not s or len(s) % SALT_BYTES:
            raise ValueError(f"{name}: {len(s)} bytes of salts, not a positive multiple of {SALT_BYTES}")
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    for old in d.glob("*.bin"):
        old.unlink()
    index = {}
    for i, name in enumerate(sorted(salts)):
        _private(d / f"{i}.bin", bytes(salts[name]))
        index[name] = {"file": f"{i}.bin", "rows": len(salts[name]) // SALT_BYTES}
    _private(d / INDEX, (json.dumps({"schema": SCHEMA, "params": index}, indent=1, sort_keys=True) + "\n").encode())
    return d


def params(path: str | Path) -> dict[str, int]:
    """Each parameter of the salt file ``path`` with its row count."""
    doc = json.loads((Path(path) / INDEX).read_text())
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} salt file")
    return {name: int(e["rows"]) for name, e in doc["params"].items()}


def read(path: str | Path, param: str, rows: Iterable[int] | None = None) -> bytes:
    """Parameter ``param``'s salts from the salt file ``path``: every row's, or ``rows``' in the order given, concatenated."""
    doc = json.loads((Path(path) / INDEX).read_text())
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} salt file")
    if param not in doc["params"]:
        raise ValueError(f"{path}: no salts for {param!r}")
    e = doc["params"][param]
    n = int(e["rows"])
    with open(Path(path) / e["file"], "rb") as f:
        if rows is None:
            data = f.read()
            if len(data) != n * SALT_BYTES:
                raise ValueError(f"{path}: {param!r} holds {len(data)} bytes, not {n} rows of {SALT_BYTES}")
            return data
        out = []
        for j in rows:
            if not 0 <= j < n:
                raise ValueError(f"{path}: {param!r} has {n} rows, not row {j}")
            f.seek(j * SALT_BYTES)
            out.append(f.read(SALT_BYTES))
        return b"".join(out)
