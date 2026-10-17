"""Shared-memory bank conflicts of one warp-wide access, from its lanes' byte offsets.

sm_120's shared memory has 32 banks of 4 bytes: byte offset a is in bank (a // 4) % 32. An access is served in wavefronts.
A wavefront serves at most one 4-byte word in each bank, and every lane that reads one word shares it (a broadcast). The
lanes of an access fall into phases, and each phase moves at most 128 bytes:

- LDS, STS and ATOMS move 4, 8 or 16 bytes a lane: 32, 16 or 8 lanes a phase.
- LDSM and STSM move 16-byte matrix rows: each register a lane holds is one phase of 8 rows, whose addresses lanes 8p to
  8p + 7 give (x1: lanes 0-7, x2: 0-15, x4: 0-31; an m16n16 form holds two registers a matrix).

A phase whose busiest bank holds n distinct words takes n wavefronts: an n-way conflict. `degree` is the largest n over the
phases, and 1 means conflict-free.
"""
from __future__ import annotations

from collections import defaultdict

BANKS, BANK_BYTES, PHASE_BYTES = 32, 4, 128


def lane_bytes(op: str) -> int:
    """The bytes one lane moves (LDS/STS/ATOMS), or one matrix row (LDSM/STSM): 16 for a .128 access or a matrix row, 8 for
    .64, else 4 (a sub-word access still occupies its word's bank)."""
    o = op.split(".")
    if o[0] in ("LDSM", "STSM"):
        return 16
    return 16 if "128" in o[1:] else 8 if "64" in o[1:] else 4


def matrix_registers(op: str) -> int:
    """Registers a lane holds in an LDSM/STSM: its .2/.4 suffix (1 without one), twice that for an m16n16 (`1616`) form."""
    o = op.split(".")
    n = int(o[-1]) if o[-1] in ("2", "4") else 1
    return 2 * n if any("1616" in s for s in o[1:]) else n


def phases(op: str, lanes: list[int | None]) -> list[list[int]]:
    """The access's phases, each the byte offsets (one a lane, or one a matrix row) one wavefront group serves."""
    if len(lanes) != 32:
        raise ValueError(f"{op}: {len(lanes)} lane offsets, not 32")
    if op.split(".")[0] in ("LDSM", "STSM"):
        return [[a for a in lanes[8 * p:8 * p + 8] if a is not None] for p in range(matrix_registers(op))]
    per = PHASE_BYTES // lane_bytes(op)
    return [[a for a in lanes[i:i + per] if a is not None] for i in range(0, 32, per)]


def phase_degree(op: str, offsets: list[int]) -> int:
    """One phase's wavefronts: the most distinct 4-byte words one bank serves."""
    width = lane_bytes(op)
    words: dict[int, set[int]] = defaultdict(set)
    for a in offsets:
        for w in range(a // BANK_BYTES, (a + width - 1) // BANK_BYTES + 1):
            words[w % BANKS].add(w)
    return max((len(s) for s in words.values()), default=1)


def worst(op: str, lanes: list[int | None]) -> tuple[int, list[int]]:
    """The access's conflict degree (1: none) and the offsets of the phase that has it."""
    return max(((phase_degree(op, ph), ph) for ph in phases(op, lanes)), key=lambda d: d[0], default=(1, []))


def degree(op: str, lanes: list[int | None]) -> int:
    return worst(op, lanes)[0]
