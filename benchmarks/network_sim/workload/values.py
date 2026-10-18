"""Each unit's commitment to the values it holds: the values placed at its leaves, as hiding rows.

A unit's values are the gates the placement puts at its leaves, in canonical order. Row i of its commitment holds the bits of
values [i·k, (i + 1)·k) packed least significant first (k = `ROW_BITS` / width for a Boolean Program's 1-bit values), under
`verity.ml.commitments.advice` (hm96-sha512 leaves in a frame-v3-sha512 tree) with the binding SHA-512(`LABEL` ‖ the unit's name).
The root is public; the rows and salts are the unit's. A crossing value is opened at its source unit's commitment
(`UnitValues.open`, checked by `verify`): refinement puts every crossing value in the committed set.
"""
from __future__ import annotations

import bisect
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
    registration: advice.Registration

    @property
    def per_row(self) -> int:
        return ROW_BITS // self.width

    @property
    def root(self) -> bytes:
        return self.registration.root

    def locate(self, gate: int) -> tuple[int, int]:
        """(row, bit offset) of a value the unit holds."""
        i = _index(self.gates, gate)
        return i // self.per_row, (i % self.per_row) * self.width

    def open(self, gate: int) -> advice.Opening:
        return self.registration.open(self.locate(gate)[0])


def _index(gates: Sequence[int], gate: int) -> int:
    i = bisect.bisect_left(gates, gate)
    if i == len(gates) or gates[i] != gate:
        raise KeyError(f"gate {gate} is not one of the unit's values")
    return i


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
    return UnitValues(unit, gates, width, advice.register(rows(gates, value, width), ROW_BITS, binding(unit), salts))


def verify(root: bytes, unit: str, count: int, width: int, opening: advice.Opening, row: int, offset: int) -> int | None:
    """The value at (row, offset) of `opening`, when the opening is row `row` of the `count` values `unit` committed under
    `root`; else None."""
    n_rows = -(-count // (ROW_BITS // width))
    if opening.index != row or not advice.verify(root, n_rows, ROW_BITS, binding(unit), opening):
        return None
    return (int.from_bytes(opening.row, "little") >> offset) & ((1 << width) - 1)
