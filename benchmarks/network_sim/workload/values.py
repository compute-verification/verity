"""Each unit's commitment to the values it holds: the values placed at its leaves, as hiding rows.

A unit's values are the gates the placement puts at its leaves, in canonical order. Row i of its commitment holds the bits of
values [i·k, (i + 1)·k) packed least significant first, each in a slot of the model's widest value (`value_bits`: 1 for a
Boolean Program, 32 for the served Program's words), k = `ROW_BITS` / width, under
`verity.ml.commitments.advice` (hm96-sha512 leaves in a frame-v3-sha512 tree) with the binding SHA-512(`LABEL` ‖ the unit's name).
The root is public; the rows and salts are the unit's.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

from verity.ml.commitments import advice

LABEL = b"verity/netsim/values/v1/"
ROW_BITS = 256


def binding(unit: str) -> bytes:
    return hashlib.sha512(LABEL + unit.encode()).digest()


@dataclass(frozen=True)
class UnitValues:
    unit: str
    gates: tuple[int, ...]
    width: int
    registration: advice.Committed

    @property
    def root(self) -> bytes:
        return self.registration.root


def rows(gates: Sequence[int], value: dict[int, int], width: int) -> list[bytes]:
    per = ROW_BITS // width
    out = []
    for i in range(0, len(gates), per):
        acc = 0
        for j, g in enumerate(gates[i:i + per]):
            acc |= (value[g] & ((1 << width) - 1)) << (j * width)
        out.append(acc.to_bytes(ROW_BITS // 8, "little"))
    return out


def commit(unit: str, gates: Sequence[int], value: dict[int, int], width: int = 1,
           salts: Sequence[bytes] | None = None) -> UnitValues:
    gates = tuple(sorted(gates))
    if not gates:
        raise ValueError(f"unit {unit} holds no values")
    return UnitValues(unit, gates, width, advice.commit(rows(gates, value, width), ROW_BITS, binding(unit), salts))
