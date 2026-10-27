"""An H1 window's private prover state (DRAWN1): what opening a drawn proof block's committed values needs after the
window's close, kept in a directory of the prover's own (`pouw_h1.H1Run`'s ``prover``), owner-only (0700, each file 0600),
never under a run's ``$OUT`` or any published path. The serving node keeps it (`research keep`), and its holder alone reads it.

* **What it keeps.** `keep_key`: the run's hm96 salt key (`KEY`, 32 bytes), from which the salt of every hiding leaf of
  the window is drawn: its request, slot, A and z tables' (`pouw_h1_requests.hiding_rows`) and its served trees' rows
  (`audit.hm96_salt` at `pouw_h1_served.nonce`). `keep_call`: each served call's m s8 rows, their f32 scales and its z/y
  records (`pouw_h1_records`, one a proof block), as ``calls/<index>.npz``. An empty call keeps nothing: its A table is T zero
  rows and its z table zero records. A handed-off window (`pouw_h1_handoff`) also keeps its saved state in `STATE`: its
  round log, request entries, slot owners, and each call's A codes and each linear's weight codes (the window's ``a/<K>``
  and ``b/<K>`` rows).
* **Opening a drawn proof block.** Proof block u of the window is proof block u - base of the call whose manifest row has base ≤ u < base +
  units (`locate`). `open_call` rebuilds that call's A table and z table from the kept rows, scales and records under the
  key, the Program salt (from the record's salt coin, `pouw_h1_challenger.program_salt_of`) and the call's
  nonces, and refuses either whose root isn't the one its manifest row carries, which its round's root commits. Then
  ``z.open(u - base)`` is the proof block's record (z then y) with its salt and path, and ``a.open(i)`` slot i's A row and scale
  (the proof block's row tile is slots 32·r .. 32·r + 31). Its ``a/<K>`` rows are its call's block of the window's A value
  (`pouw_h1_served.layout`), its codes the A table's padded to the Program's slot, and its ``b/<K>`` rows its linear's weight
  rows, each salted at `pouw_h1_served.nonce`; their trees are the ones the record's ``served`` entries name.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

KEY = "key"
CALLS = "calls"
STATE = "state"


def private(directory: str | Path) -> Path:
    """`directory`, created if missing and made owner-only (0700)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(d, 0o700)
    return d


def _write(path: Path, data: bytes) -> None:
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
        f.write(data)


def keep_key(directory: str | Path, key: bytes) -> Path:
    """The run's salt key, kept as `KEY` (0600) in the prover's directory."""
    path = private(directory) / KEY
    _write(path, bytes(key))
    return path


def key_of(directory: str | Path) -> bytes:
    return (Path(directory) / KEY).read_bytes()


def keep_call(directory: str | Path, index: int, a: np.ndarray, s_a: np.ndarray, records: np.ndarray | None) -> Path:
    """Call `index`'s served A rows (int8 [m, k]), their f32 scales [m] and its z/y records (or none), as committed."""
    d = private(Path(directory) / CALLS)
    path = d / f"{int(index)}.npz"
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
        np.savez(f, a=np.asarray(a, np.int8), s_a=np.asarray(s_a, "<f4").reshape(-1),
                 records=np.zeros((0, 0), np.uint8) if records is None else np.asarray(records))
    return path


def locate(rows: Sequence[Mapping[str, Any]], unit: int) -> tuple[int, int]:
    """Proof block `unit` of the window as (its call's index, its leaf in that call's z table), from the manifest `rows`."""
    for row in rows:
        if row["base"] <= unit < row["base"] + row["units"]:
            return int(row["index"]), unit - int(row["base"])
    raise ValueError(f"unit {unit} is no call's: the window's units are its calls' [base, base + units)")


@dataclass(frozen=True)
class Opened:
    """A call's A table and z table (`commitments.MerkleTree`s, None for a call that committed no z), rebuilt."""

    index: int
    a: Any
    z: Any


def open_call(directory: str | Path, record: Mapping[str, Any], index: int, salt: bytes) -> Opened:
    """Call `index` of the window `record` (`pouw_h1.CALLS_FILE`), its tables rebuilt from the prover's directory under
    the window's Program salt `salt` (`pouw_h1_challenger.program_salt_of` its commitment): refused (ValueError) when a
    rebuilt table's root isn't the one the call's manifest row carries."""
    from verity.core.protocols.pouw.protocol import tagged

    from verity_vllm.protocols.pouw.h1 import records as ZR
    from verity_vllm.protocols.pouw.h1 import requests as RQ

    row = record["calls"][index]
    key, t, k, m = key_of(directory), int(row["rows"]), int(row["k"]), int(row["served"])
    path = Path(directory) / CALLS / f"{index}.npz"
    if m:
        with np.load(path) as kept:
            a, s_a, records = kept["a"], kept["s_a"], kept["records"]
    else:
        a, s_a, records = np.zeros((0, k), np.int8), np.zeros(0, "<f4"), np.zeros((int(row["units"]), ZR.ZY_BYTES), np.uint8)

    def table(name: str, leaves: list[bytes]) -> Any:
        tree = RQ.hiding_rows(tagged(RQ.TABLE_TAGS[name], salt, index), index, leaves, key,
                              RQ.TABLES_NONCE + 3 * index + RQ.TABLE_NONCES[name])
        if tree.commitment.root.hex() != (row.get("tables") or {}).get(name):
            raise ValueError(f"call {index}'s {name} table, rebuilt from the prover's directory, is not the one its "
                             "manifest row carries")
        return tree

    at = table("a", RQ.table_leaves(t, k + 4, m, a.reshape(m, k), s_a.reshape(m, 1)))
    z = None if not records.size else table("z", [r.tobytes() for r in records])
    return Opened(index, at, z)


__all__ = ["CALLS", "KEY", "Opened", "STATE", "keep_call", "keep_key", "key_of", "locate", "open_call", "private"]
