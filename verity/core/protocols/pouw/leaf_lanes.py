"""A drawn H1 leaf proved as its lanes and their combine (PRF-134, option A): the statement set, the values each statement
reads and at which row, and the lane partials' layout, the after-draw value the statements share.

Serving keeps one CP-W receipt a 32 × 32 leaf. A drawn leaf's proof is its 32 lane statements and one combine statement
(`circuit.h1zy`'s `PouwH1LaneZy_v1{TILE}`, lane l by its public index, and `PouwH1Combine_v1{TILE}`), each proved under the
firewall on its own. The statements share the leaf's lane partials (``partials``), which the prover recomputes after the
draw from what it committed before it and commits once under hiding leaves (`sampled_proofs.one_stage.produced`). A
partial is never an output: outputs are public and at most 16 bits, and a partial is 2 (KS + m) u32. Only the
statements' ok bits leave.

**The lane geometry** (`lane_rows`, `lane_cols`, `key_block`), as serving's kernels write a leaf's lanes (``kh_lanes``):
an R × C leaf has G Q lanes, G = R / 4 and Q = C / 8 (at 32 × 32 a warp, lane l = 4 g + q). Lane l = Q g + q holds the
leaf's rows g + G s (row slot s < 4) and columns 2 q + 2 Q (t >> 1) + (t & 1) (column slot t < 8), and its 16 fold keys are
block l of σ_R's stream.

**The partials' layout** (`word_order`): row l of ``partials`` is lane l's partial. Its 2 (KS + m) u32 words are, for each
set the lane absorbs in order (T_0 .. T_(KS-1) the prefix sets, O', then D_1 .. D_(m-1) the digits, m = `n_digits`(KS)),
the XOR of its 8 chains' y after that set and then of their z: word 2 s is set s's y, word 2 s + 1 its z. That is 260
words at K = 4096. The combine checks that the rows' XOR, word by word, is the leaf's committed receipt.

**The statements and their reads** (`statements`, `reads`): each a C-Flock statement of one unit, its ports these:

* lane l: ``a0`` .. ``a3`` the leaf's A rows at its row slots, ``b0`` .. ``b7`` its B rows at its column slots, ``zy``
  (its z/y record) and ``partials`` (row l); ``admit`` (the leaf's R admit bits) and a row ``pub`` of the leaf's salt,
  position and roots and the lane index l (`PUBLIC`, l a 16-bit word); its output ``o0`` is ok.
* the combine: ``p0`` .. ``p31`` (rows 0 .. 31 of ``partials``) and ``receipt`` (the leaf's); its output ``o0`` is ok.

Every port of every statement, its output row included, is a port of the verifier's own registered reads (`own`,
`registered_file`), never a row the public file states: a port the verifier doesn't register is free in the relation it
checks.

* **The reads** (`reads`) open the registration's and the produced record's roots at the rows the verifier states here, so
  a partial swapped between rows opens at no read's position, and one from another leaf's tree opens under no root the
  produced record names. Every session's opened rows are the one opening the record keeps (`produced.shared`).
* **The known ports** (`known`): ``pub``, ``admit`` and ``o0`` the verifier registers at their known content
  (`produced.known_port`): lane l's ``pub`` row (value ``pub<l>``) holds index l, so a lane whose index isn't the
  verifier's l opens no root; every statement's ``o0`` (value ``ok<i>``, i its place in `statements`) is the row 1.

STAND-IN: the rows of ``a``, ``b``, ``zy`` and ``receipt`` are leaf-local (the leaf's own registration); in a served
window they are the window's table rows (`circuit.h1window`), which this module then takes as an offset.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from verity.core.protocols.sampled_proofs.one_stage import produced as PR
from verity.core.protocols.sampled_proofs.one_stage import registered as G

PARTIALS = "partials"
A, B, ADMIT, ZY, RECEIPT = "a", "b", "admit", "zy", "receipt"
#: the lane statement's public row's fields, in order
PUBLIC = ("salt", "position", "roots", "lane")
PUB, OK = "pub", "o0"
LANE_ROWS, LANE_COLS = 4, 8
LANE, COMBINE = "lane", "combine"


def n_digits(ks: int) -> int:
    """H1's digit count m at KS segments: the least m with -128^m <= -128 KS and 127 KS // 128^(m - 1) <= 127."""
    m = 1
    while not (-(128 ** m) <= -128 * ks and 127 * ks // 128 ** (m - 1) <= 127):
        m += 1
    return m


def _grid(r: int, c: int) -> tuple[int, int]:
    if r <= 0 or c <= 0 or r % LANE_ROWS or c % LANE_COLS:
        raise ValueError(f"a leaf's lanes tile it by 4 rows and 8 columns, not {r} × {c}")
    return r // LANE_ROWS, c // LANE_COLS


def lanes(r: int, c: int) -> int:
    g, q = _grid(r, c)
    return g * q


def lane_rows(r: int, c: int, ln: int) -> tuple[int, ...]:
    """Lane ln's leaf rows, by row slot."""
    G_, Q = _grid(r, c)
    if not 0 <= ln < G_ * Q:
        raise ValueError(f"an {r} × {c} leaf has {G_ * Q} lanes, not lane {ln}")
    return tuple(ln // Q + G_ * s for s in range(LANE_ROWS))


def lane_cols(r: int, c: int, ln: int) -> tuple[int, ...]:
    """Lane ln's leaf columns, by column slot."""
    G_, Q = _grid(r, c)
    if not 0 <= ln < G_ * Q:
        raise ValueError(f"an {r} × {c} leaf has {G_ * Q} lanes, not lane {ln}")
    q = ln % Q
    return tuple(2 * q + 2 * Q * (t >> 1) + (t & 1) for t in range(LANE_COLS))


def key_block(ln: int) -> int:
    """Lane ln's block of σ_R's stream: its 16 fold keys."""
    return int(ln)


def word_order(k: int) -> list[str]:
    """A partial's words at contraction K, by name: ``"<set>.y"`` then ``"<set>.z"`` for each set in absorption order,
    ``T<s>`` (s < KS), ``O'``, then ``D<d>`` (1 <= d < m)."""
    if k <= 0 or k % 32:
        raise ValueError(f"K is a positive multiple of 32, not {k}")
    ks = k // 32
    sets = [f"T{s}" for s in range(ks)] + ["O'"] + [f"D{d}" for d in range(1, n_digits(ks))]
    return [f"{s}.{h}" for s in sets for h in ("y", "z")]


def words(k: int) -> int:
    """A partial's, or the receipt's, u32 words: 2 (KS + m)."""
    return len(word_order(k))


def statements(r: int, c: int) -> list[tuple[str, int | None]]:
    """The leaf's statements in order: ``("lane", l)`` for each lane, then ``("combine", None)``."""
    return [(LANE, ln) for ln in range(lanes(r, c))] + [(COMBINE, None)]


def reads(r: int, c: int, kind: str, ln: int | None = None) -> dict[str, tuple[str, int]]:
    """A statement's registered ports, each ``(value, row)``: the value it reads and the row the verifier reads it at."""
    if kind == LANE:
        out = {f"a{s}": (A, i) for s, i in enumerate(lane_rows(r, c, int(ln)))}
        out.update({f"b{t}": (B, j) for t, j in enumerate(lane_cols(r, c, int(ln)))})
        return {**out, ZY: (ZY, 0), PARTIALS: (PARTIALS, int(ln))}
    if kind == COMBINE and ln is None:
        return {**{f"p{x}": (PARTIALS, x) for x in range(lanes(r, c))}, RECEIPT: (RECEIPT, 0)}
    raise ValueError(f"a leaf's statement is a lane with its index or the combine, not {(kind, ln)!r}")


def known(r: int, c: int, kind: str, ln: int | None = None) -> dict[str, str]:
    """A statement's ports the verifier registers at known content, each its value: lane l's ``pub`` (``pub<l>``) and
    ``admit``, and every statement's ``o0`` (``ok<i>``, i its place in `statements`)."""
    reads(r, c, kind, ln)
    i = statements(r, c).index((kind, None if ln is None else int(ln)))
    return {PUB: f"pub{int(ln)}", ADMIT: ADMIT, OK: f"ok{i}"} if kind == LANE else {OK: f"ok{i}"}


def own(r: int, c: int, kind: str, ln: int | None = None) -> set[str]:
    """Every port the verifier registers for a statement: its `reads` and its `known` ports."""
    return set(reads(r, c, kind, ln)) | set(known(r, c, kind, ln))


def registered_file(ports: Mapping[str, tuple[str, int]], entries: Mapping[str, Mapping[str, Any]],
                    roots: Mapping[str, str], schemas: Mapping[str, str] | None = None,
                    known_ports: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The Lean verifier's ``--registered`` file of one statement (one unit) from its `reads`: each port's value's entry
    (a registered value's from the accepted registration, or ``partials``' from `produced.entry`) and root (the accepted
    registration's, or the accepted produced record's), and its `known` ports' (`produced.known_port`'s)."""
    out = {}
    for port, (value, row) in ports.items():
        if value not in entries or value not in roots:
            raise ValueError(f"port {port!r} reads {value!r}, whose entry and root the verifier does not hold")
        out[port] = PR.port(value, entries[value], roots[value], [row], (schemas or {}).get(value, G.SCHEMA))
    if set(out) & set(known_ports or {}):
        raise ValueError(f"ports {sorted(set(out) & set(known_ports or {}))} are both reads and known")
    return {"format": G.READS, "ports": {**out, **(known_ports or {})}}


__all__ = [
    "ADMIT",
    "COMBINE",
    "LANE",
    "LANE_COLS",
    "LANE_ROWS",
    "OK",
    "PARTIALS",
    "PUB",
    "PUBLIC",
    "RECEIPT",
    "ZY",
    "A",
    "B",
    "key_block",
    "known",
    "lane_cols",
    "lane_rows",
    "lanes",
    "n_digits",
    "own",
    "reads",
    "registered_file",
    "statements",
    "word_order",
    "words",
]
