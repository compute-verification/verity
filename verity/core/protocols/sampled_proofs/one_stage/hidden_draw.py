"""A sample over hidden positions, the Python mirror of `SampledProofs.HiddenDraw` (held to it by the sampled-proofs
difftest's `hidden_offsets`, `hidden_locate` and `hidden_witness` pieces).

A stratum's population is `M` positions whose layout is private. Before the sampling coin the prover commits a calls table,
call `c` an offset and its live rows, each row `width` proof blocks (column tiles) wide; the verifier samples positions `r < M`
from its own coins. A drawn position is proved by a witness `(c, j, t)` that the opening checks inside the proof: call
`c` covers `r` (`offset <= r < offset + rows * width`) and `r - offset = j * width + t` (`witness`).

The table needs no proof that it is the honest prover's (`table`, over `offsets`): whatever call the prover names for each
position, two positions never share a witness (`witness_inj` in the Lean), so the positions proved inject into the
table's (call, row, tile) triples. `locate` is the honest prover's witness, the first call that covers the position.
"""

from __future__ import annotations

from typing import NamedTuple


class Call(NamedTuple):
    offset: int
    rows: int


Witness = tuple[int, int, int]


def _check_width(width: int) -> None:
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        raise ValueError(f"width {width!r} is not a positive integer")


def covers(width: int, call: Call, r: int) -> bool:
    return call.offset <= r < call.offset + call.rows * width


def witness(width: int, calls: list[Call], c: int, r: int) -> Witness | None:
    """The witness `(c, j, t)` for position `r` when the prover names call `c`, or None when `c` is no call or doesn't
    cover `r`."""
    _check_width(width)
    if not 0 <= c < len(calls) or not covers(width, calls[c], r):
        return None
    j, t = divmod(r - calls[c].offset, width)
    return c, j, t


def offsets(width: int, rows: list[int]) -> list[int]:
    """The honest prover's offsets: call `c` starts after the positions of the calls before it."""
    _check_width(width)
    out, at = [], 0
    for m in rows:
        out.append(at)
        at += m * width
    return out


def total(width: int, rows: list[int]) -> int:
    """The positions the honest table covers, `M = width * sum(rows)`."""
    _check_width(width)
    return width * sum(rows)


def table(width: int, rows: list[int]) -> list[Call]:
    """The honest prover's table over the calls' live rows."""
    return [Call(o, m) for o, m in zip(offsets(width, rows), rows)]


def locate(width: int, calls: list[Call], r: int) -> Witness | None:
    """The first call that covers `r`, with its row and tile."""
    _check_width(width)
    for c, call in enumerate(calls):
        if covers(width, call, r):
            return witness(width, calls, c, r)
    return None
