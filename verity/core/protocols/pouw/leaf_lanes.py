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
checks. The file says so (``complete``), and the Lean verifier then refuses a statement any port of whose circuit, an
input its ok's cone reads (the lane index's ``pub`` among them) or an output row, is not one of the file's ports
(Flock's `Registered.checkComplete`, `checkComplete_ok`).

* **The reads** (`reads`) open the registration's and the produced record's roots at the rows the verifier states here, so
  a partial swapped between rows opens at no read's position, and one from another leaf's tree opens under no root the
  produced record names. Every session's opened rows are the one opening the record keeps (`produced.shared`).
* **The known ports** (`known`): ``pub``, ``admit`` and ``o0`` the verifier registers at their known content
  (`produced.known_port`): lane l's ``pub`` row (value ``pub<l>``) holds index l, so a lane whose index isn't the
  verifier's l opens no root; every statement's ``o0`` (value ``ok<i>``, i its place in `statements`) is the row 1.
* **Their rows** (`known_rows`, the served path's one producer of them): lane l's ``pub`` (`pub_words`: the unit's salt,
  position and roots as its seed reads them, `seeds.leaves` and `seeds.position_leaves`, 100 words, then l as the 101st),
  ``admit`` (the leaf's R admit bits, `bits_words`) and ``o0`` (`OK_ROW`); the salts they're committed under are the
  caller's. Read least significant bit first, as `Accounting.TS.Lanes.decLane` reads a field, the salt is its 16
  big-endian 32-bit halves with half 0 the least significant, and the position its low half above its high one. Lean's
  is proofs' `PouwWindow.knownRows`, the rows `LeafLanesSound` takes, whose difftest reads this through h1_lanes'
  `known_rows` (`benchmarks/pouw/h1_lanes/run.py`), a wrapper over it.

STAND-IN: the rows of ``a``, ``b``, ``zy`` and ``receipt`` are leaf-local (the leaf's own registration); in a served
window they are the window's table rows (`circuit.h1window`), which this module then takes as an offset.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from verity.core.protocols.pouw.circuit import seeds as SD
from verity.core.protocols.sampled_proofs.one_stage import produced as PR
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.core.protocols.sampled_proofs.one_stage import registration as R

PARTIALS = "partials"
A, B, ADMIT, ZY, RECEIPT = "a", "b", "admit", "zy", "receipt"
#: the lane statement's public row's fields, in order
PUBLIC = ("salt", "position", "roots", "lane")
#: their bits: a digest, a u64, two digests and the lane index's u16 word
PUBLIC_BITS = (512, 64, 1024, 16)
PUB, OK = "pub", "o0"
#: every statement's ``o0`` row: ok = 1
OK_ROW = (1,)
#: the ``--registered`` file's key that its ports are every port of the statement (Flock's `Registered.COMPLETE_KEY`)
COMPLETE = "complete"
#: a ``--registered`` port's key that the statement reads its row in full (Flock's `KnownRead.READ_KEY`)
READ_IN_FULL = "read_in_full"
#: the values a statement reads in full (`READ_IN_FULL`): each is read by several statements at one row
FULL = (A, B, PARTIALS)
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


def bits_words(bits: Sequence[int]) -> list[int]:
    """A row's u16 words from its bits: bit t is bit t mod 16 of word t // 16, the last word's missing bits 0."""
    b = [int(x) for x in bits]
    if any(x not in (0, 1) for x in b):
        raise ValueError("a row's bits are 0 or 1")
    return [sum(x << t for t, x in enumerate(b[i:i + 16])) for i in range(0, len(b), 16)]


def pub_words(salt: bytes, position: int, roots: Sequence[bytes], ln: int) -> list[int]:
    """Lane ln's ``pub`` row: the salt's, the position's and the two roots' leaves as the unit's seed reads them
    (`seeds.leaves`, `seeds.position_leaves`: big-endian 32-bit halves, each least significant bit first), 100 words, then
    ln as the 101st."""
    if len(salt) != 64 or len(roots) != 2 or any(len(x) != 64 for x in roots):
        raise ValueError("the salt and the two roots are 64-byte digests")
    if not 0 <= int(ln) < 1 << 16:
        raise ValueError(f"the lane index is a u16 word, not {ln!r}")
    bits = SD.leaves(salt, True) + SD.position_leaves(position, True) + SD.leaves(roots[0], True) + SD.leaves(roots[1], True)
    assert len(bits) == sum(PUBLIC_BITS[:-1])
    return bits_words(bits) + [int(ln)]


def known_rows(r: int, c: int, kind: str, ln: int | None = None, *, admit: Sequence[int] | None = None,
               salt: bytes | None = None, position: int | None = None,
               roots: Sequence[bytes] | None = None) -> dict[str, tuple[str, list[int]]]:
    """A statement's `known` ports' rows, each ``(value, words)``: lane l's ``pub`` (`pub_words` of the unit's ``salt``,
    ``position`` and ``roots``) and ``admit`` (`bits_words` of the leaf's ``admit`` bits, its R), and every statement's
    ``o0`` (`OK_ROW`), the combine's one known port. Lean's is `PouwWindow.knownRows`, with its arguments in this order."""
    vals = known(r, c, kind, ln)
    out = {OK: (vals[OK], list(OK_ROW))}
    if kind == LANE:
        if salt is None or position is None or roots is None or admit is None:
            raise ValueError("a lane's known rows are of the leaf's admit bits and the unit's salt, position and roots")
        out[PUB] = (vals[PUB], pub_words(salt, position, roots, int(ln)))
        out[ADMIT] = (vals[ADMIT], bits_words(admit))
    return out


def verifier_entries(registration: Mapping[str, Any], produced: Mapping[str, Any], lanes_: int,
                     words_: int) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    """The verifier's entries and roots of a drawn leaf's values, from the accepted registration record and the accepted
    produced record alone (Lean's `PouwWindow.Lanes.roots?`): each registered value's entry in the registration's
    ``registered`` (the first of its value) and that entry's root, and ``partials``' own entry (`produced.entry`:
    ``lanes_`` rows of ``words_`` u16 words under the registration's digest) at the produced record's root of
    ``partials``. `registered_file` reads them; `pouw-leaf-own` holds a statement's file to them."""
    ent: dict[str, Mapping[str, Any]] = {}
    roots: dict[str, str] = {}
    for e in registration.get("registered") or ():
        if e["value"] not in ent:
            ent[e["value"]], roots[e["value"]] = e, e["root"]
    got = [e for e in produced["produced"] if e.get("value") == PARTIALS]
    if not got:
        raise ValueError(f"the produced record has no {PARTIALS}")
    ent[PARTIALS] = PR.entry(registration["program"], PARTIALS, lanes_, words_, R.digest(registration))
    roots[PARTIALS] = got[0]["root"]
    return ent, roots


def registered_file(ports: Mapping[str, tuple[str, int]], entries: Mapping[str, Mapping[str, Any]],
                    roots: Mapping[str, str], schemas: Mapping[str, str] | None = None,
                    known_ports: Mapping[str, Mapping[str, Any]] | None = None, complete: bool = True,
                    place: Mapping[str, tuple[str, int]] | None = None) -> dict[str, Any]:
    """The Lean verifier's ``--registered`` file of one statement (one unit) from its `reads`: each port's value's entry
    (a registered value's from the accepted registration, or ``partials``' from `produced.entry`) and root (the accepted
    registration's, or the accepted produced record's), and its `known` ports' (`produced.known_port`'s). ``place``: the
    unit's placement, ``{value: (table, base)}``: port ``(value, row)`` reads row ``base + row`` of ``table``, whose
    entry, root and schema it takes (Lean's `PouwWindow.Lanes.ownOf`); a value it doesn't name is its own table from row 0
    (`selfPlace`), every value so where it is None. ``complete``:
    the ports are every port of the statement (`own`), which the verifier then holds the circuit to
    (`Registered.checkComplete`). A port of an A row, a B row or a partials row (`FULL`: lane l's ``a<s>``, ``b<t>``
    and ``partials``, the combine's ``p<x>``, by the value it reads, whatever table that is placed in) is marked
    ``read_in_full`` (`READ_IN_FULL`): the verifier holds it to one row of the registered row's words, every bit of
    which the statement reads (Flock's `KnownRead.fullRows`), so the statements that read one row (four lanes an A row,
    four a B row, lane l and the combine a partials row) read its same bits (proofs' `PouwWindow.Lanes.readFull`, which
    `LeafLanesSound` takes)."""
    out = {}
    for port, (value, row) in ports.items():
        table, base = (place or {}).get(value, (value, 0))
        if table not in entries or table not in roots:
            raise ValueError(f"port {port} reads {table}, whose entry and root the verifier does not hold")
        out[port] = PR.port(table, entries[table], roots[table], [base + row], (schemas or {}).get(table, G.SCHEMA))
        if value in FULL:
            out[port][READ_IN_FULL] = True
    if set(out) & set(known_ports or {}):
        raise ValueError(f"ports {sorted(set(out) & set(known_ports or {}))} are both reads and known")
    return {"format": G.READS, "ports": {**out, **(known_ports or {})}, **({COMPLETE: True} if complete else {})}


__all__ = [
    "ADMIT",
    "COMBINE",
    "COMPLETE",
    "FULL",
    "LANE",
    "LANE_COLS",
    "LANE_ROWS",
    "OK",
    "OK_ROW",
    "PARTIALS",
    "PUB",
    "PUBLIC",
    "PUBLIC_BITS",
    "READ_IN_FULL",
    "RECEIPT",
    "ZY",
    "A",
    "B",
    "bits_words",
    "key_block",
    "known",
    "known_rows",
    "lane_cols",
    "lane_rows",
    "lanes",
    "n_digits",
    "own",
    "pub_words",
    "reads",
    "registered_file",
    "statements",
    "verifier_entries",
    "word_order",
    "words",
]
