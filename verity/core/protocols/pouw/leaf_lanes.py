"""A drawn H1 leaf proved as its lanes and their combine: the statement set, the values each statement
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

**A J10-R leaf** (``j10r``, the leaf's K: a unit over `PouwH1Tile_v5`) is proved as `h1zy`'s v2 statements, each lane in
n = `weights.chunks`(K) chunks along K (`h1zy.lane_statements`; every served J10-R K is past `CHUNK_MAX`, so n >= 2), then
`PouwH1Combine_v2`, which folds the other statements' oks into its own:

* `statements`: ``("lane", l)`` at n = 1, else ``("chunk", l, c)`` for each chunk c of lane l in chain order, lane by
  lane, then ``("combine", None)``. Statement i < 32 n is lane i // n's chunk i % n.
* `reads`: chunk c of lane l reads ``a<s>`` and ``b<t>`` at its chunk rows, row n i + c of a table that commits row i as n
  chunk rows (`h1zy.chunk_leaves`), ``carry_in`` (c > 0) at row l of ``carry<c - 1>`` and ``carry_out`` (c < n - 1) at row
  l of ``carry<c>`` (`CARRY`, what chunk c hands on, `h1zy.CARRY`), and the last chunk ``zy`` and ``partials`` (row l).
  A lane or chunk's ``o0`` is row i of ``oks``, the statements' ok rows (`OKS`): its output is a committed value, never
  public. The combine reads ``p<x>``, ``receipt`` and ``ok<i>``, row i of ``oks`` for each statement i < 32 n.
* `known`: lane l's ``pub`` (`PUBLIC_J10R`: the unit's call and blocks after its roots, `pub_words`) in each of its
  chunks, ``admit`` in its last, and the combine's ``o0`` (value ``ok<32 n>``), the leaf's one known ok.
* The carries and ``oks`` are after-draw values like the partials, each one table committed once
  (`one_stage.produced`): ``carry<c>`` 32 rows of `carry_words`, ``oks`` 32 n rows of one word (`verifier_entries`).
  A port of a carry or of ``oks`` is read in full (`read_in_full`), but never ``o0``: an output row binds only the bits
  the circuit returns, and the combine holds the rest of each ok row to 0.

STAND-IN: the rows of ``a``, ``b``, ``zy`` and ``receipt`` are leaf-local (the leaf's own registration); in a served
window they are the window's table rows (`circuit.h1window`), which this module then takes as an offset.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from verity.core.protocols.pouw.weights import chunks
from verity.core.protocols.sampled_proofs.one_stage import produced as PR
from verity.core.protocols.sampled_proofs.one_stage import registered as G
from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.ml.commitments.scheme import DIGEST_BYTES

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
#: a J10-R lane's public row's fields (`h1zy.PUBLIC_J10R`): the unit's call and blocks after its roots
PUBLIC_J10R = ("salt", "position", "roots", "call", "blocks", "lane")
#: their bits: a digest, a u64, two digests, a u64, two u64 and the lane index's u16 word
PUBLIC_J10R_BITS = (512, 64, 1024, 64, 128, 16)
#: a J10-R leaf's after-draw ok rows (`h1zy.OK_WORD`), statement i's at row i, and its carries, chunk c's ``carry<c>``
OKS, CARRY = "oks", "carry"
#: the values a statement reads in full (`READ_IN_FULL`): each is read by several statements at one row
FULL = (A, B, PARTIALS, OKS)
LANE_ROWS, LANE_COLS = 4, 8
LANE, CHUNK, COMBINE = "lane", "chunk", "combine"


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


def n_chunks(j10r: int | None) -> int:
    """The statements a lane of the leaf is: 1 for v1's, a J10-R leaf's (``j10r`` its K) `weights.chunks`(K)."""
    if j10r is None:
        return 1
    word_order(int(j10r))
    return chunks(int(j10r))


def carry(ch: int) -> str:
    """The value chunk ``ch`` of each lane hands chunk ``ch`` + 1 (`h1zy.CARRY`): ``carry<ch>``, row l lane l's."""
    return f"{CARRY}{int(ch)}"


def carry_words(k: int, n: int, ch: int) -> int:
    """A ``carry<ch>`` row's u16 words at K in n chunks: `h1zy.CARRY`'s 880 + 8 (ch + 1) KS_c bytes, KS_c = K / 32 n."""
    if not 0 <= ch < n - 1 or k % (32 * n):
        raise ValueError(f"chunk {ch} of {n} at K = {k} hands on no carry: chunks 0 .. {n - 2} do, at 32 n | K")
    return 440 + 4 * (ch + 1) * (k // (32 * n))


def statements(r: int, c: int, *, j10r: int | None = None) -> list[tuple]:
    """The leaf's statements in order: ``("lane", l)`` for each lane, then ``("combine", None)``; a J10-R leaf's lanes in
    n >= 2 chunks each ``("chunk", l, ch)``, lane by lane in chain order."""
    n = n_chunks(j10r)
    if n == 1:
        return [(LANE, ln) for ln in range(lanes(r, c))] + [(COMBINE, None)]
    return [(CHUNK, ln, ch) for ln in range(lanes(r, c)) for ch in range(n)] + [(COMBINE, None)]


def _key(r: int, c: int, kind: str, ln: int | None, ch: int | None, j10r: int | None) -> tuple:
    key = (kind, None if ln is None else int(ln)) + (() if ch is None else (int(ch),))
    if key not in statements(r, c, j10r=j10r):
        if kind in (LANE, CHUNK) and ln is not None and not 0 <= int(ln) < lanes(r, c):
            lane_rows(r, c, int(ln))
        form = "J10-R " if j10r is not None else ""
        raise ValueError(f"a {form}leaf's statement is a lane with its index (a chunk of one past K = 4096, with its chunk) "
                         f"or the combine, not {key!r}")
    return key


def reads(r: int, c: int, kind: str, ln: int | None = None, ch: int | None = None, *,
          j10r: int | None = None) -> dict[str, tuple[str, int]]:
    """A statement's registered ports, each ``(value, row)``: the value it reads and the row the verifier reads it at.
    ``ch``: a J10-R lane chunk's chunk; ``j10r``: a J10-R leaf's K."""
    key, n = _key(r, c, kind, ln, ch, j10r), n_chunks(j10r)
    if kind == COMBINE:
        out = {**{f"p{x}": (PARTIALS, x) for x in range(lanes(r, c))}, RECEIPT: (RECEIPT, 0)}
        return out if j10r is None else {**out, **{f"ok{i}": (OKS, i) for i in range(lanes(r, c) * n)}}
    ln, ch = int(ln), 0 if ch is None else int(ch)
    out = {f"a{s}": (A, n * i + ch) for s, i in enumerate(lane_rows(r, c, ln))}
    out.update({f"b{t}": (B, n * j + ch) for t, j in enumerate(lane_cols(r, c, ln))})
    if ch:
        out["carry_in"] = (carry(ch - 1), ln)
    out.update({ZY: (ZY, 0), PARTIALS: (PARTIALS, ln)} if ch == n - 1 else {"carry_out": (carry(ch), ln)})
    return out if j10r is None else {**out, OK: (OKS, statements(r, c, j10r=j10r).index(key))}


def known(r: int, c: int, kind: str, ln: int | None = None, ch: int | None = None, *,
          j10r: int | None = None) -> dict[str, str]:
    """A statement's ports the verifier registers at known content, each its value: lane l's ``pub`` (``pub<l>``) and
    ``admit``, and every statement's ``o0`` (``ok<i>``, i its place in `statements`); a J10-R leaf's lane's ``pub`` in each
    chunk, ``admit`` in its last, and only the combine's ``o0``."""
    key, n = _key(r, c, kind, ln, ch, j10r), n_chunks(j10r)
    i = statements(r, c, j10r=j10r).index(key)
    if kind == COMBINE:
        return {OK: f"ok{i}"}
    last = {ADMIT: ADMIT} if (0 if ch is None else int(ch)) == n - 1 else {}
    return {PUB: f"pub{int(ln)}", **last, **({OK: f"ok{i}"} if j10r is None else {})}


def own(r: int, c: int, kind: str, ln: int | None = None, ch: int | None = None, *, j10r: int | None = None) -> set[str]:
    """Every port the verifier registers for a statement: its `reads` and its `known` ports."""
    return set(reads(r, c, kind, ln, ch, j10r=j10r)) | set(known(r, c, kind, ln, ch, j10r=j10r))


def read_in_full(value: str) -> bool:
    """Whether a port of ``value`` reads its row in full (`READ_IN_FULL`): `FULL`'s and a J10-R leaf's carries."""
    return value in FULL or (value.startswith(CARRY) and value[len(CARRY):].isdigit())


def bits_words(bits: Sequence[int]) -> list[int]:
    """A row's u16 words from its bits: bit t is bit t mod 16 of word t // 16, the last word's missing bits 0."""
    b = [int(x) for x in bits]
    if any(x not in (0, 1) for x in b):
        raise ValueError("a row's bits are 0 or 1")
    return [sum(x << t for t, x in enumerate(b[i:i + 16])) for i in range(0, len(b), 16)]


def _leaf_bits(data: bytes) -> list[int]:
    """`data`'s bits as the unit's seed reads them (`seeds.leaves`): its big-endian 32-bit halves, each least significant
    bit first."""
    return [(h >> j) & 1 for i in range(0, len(data), 4) for h in [int.from_bytes(data[i:i + 4], "big")] for j in range(32)]


def _u64_bits(x: int) -> list[int]:
    """A u64's bits as `_leaf_bits` reads its 8 big-endian bytes (`seeds.position_leaves`)."""
    if not 0 <= x < 1 << 64:
        raise ValueError(f"a position or a counter is a u64, not {x!r}")
    return _leaf_bits(x.to_bytes(8, "big"))


def pub_words(salt: bytes, position: int, roots: Sequence[bytes], ln: int, *, call: int | None = None,
              blocks: Sequence[int] | None = None) -> list[int]:
    """Lane ln's ``pub`` row: the salt's, the position's and the two roots' leaves as the unit's seed reads them
    (`seeds.leaves`, `seeds.position_leaves`: big-endian 32-bit halves, each least significant bit first), 100 words, then
    ln as the 101st. A J10-R lane's (`PUBLIC_J10R`) has the unit's ``call`` and its two ``blocks`` (each a u64 as the
    position's leaves) after the roots, 112 words, then ln as the 113th."""
    n = DIGEST_BYTES["sha512"]
    if len(salt) != n or len(roots) != 2 or any(len(x) != n for x in roots):
        raise ValueError("the salt and the two roots are 64-byte digests")
    if not 0 <= int(ln) < 1 << 16:
        raise ValueError(f"the lane index is a u16 word, not {ln!r}")
    if (call is None) != (blocks is None) or (blocks is not None and len(blocks) != 2):
        raise ValueError("a J10-R lane's row has the unit's call and its two blocks, a v1 lane's neither")
    bits = _leaf_bits(salt) + _u64_bits(position) + _leaf_bits(roots[0]) + _leaf_bits(roots[1])
    if call is not None:
        bits += _u64_bits(call) + _u64_bits(blocks[0]) + _u64_bits(blocks[1])
    assert len(bits) == sum((PUBLIC_BITS if call is None else PUBLIC_J10R_BITS)[:-1])
    return bits_words(bits) + [int(ln)]


def known_rows(r: int, c: int, kind: str, ln: int | None = None, ch: int | None = None, *,
               admit: Sequence[int] | None = None, salt: bytes | None = None, position: int | None = None,
               roots: Sequence[bytes] | None = None, j10r: int | None = None, call: int | None = None,
               blocks: Sequence[int] | None = None) -> dict[str, tuple[str, list[int]]]:
    """A statement's `known` ports' rows, each ``(value, words)``: lane l's ``pub`` (`pub_words` of the unit's ``salt``,
    ``position`` and ``roots``, and a J10-R leaf's ``call`` and ``blocks``) and ``admit`` (`bits_words` of the leaf's
    ``admit`` bits, its R), and every statement's ``o0`` (`OK_ROW`), the combine's one known port (a J10-R leaf's only
    known ``o0``). Lean's is `PouwWindow.knownRows`, with its arguments in this order."""
    vals = known(r, c, kind, ln, ch, j10r=j10r)
    out = {OK: (vals[OK], list(OK_ROW))} if OK in vals else {}
    if kind != COMBINE:
        if salt is None or position is None or roots is None or (ADMIT in vals and admit is None):
            raise ValueError("a lane's known rows are of the leaf's admit bits and the unit's salt, position and roots")
        if (j10r is None) != (call is None or blocks is None):
            raise ValueError("a J10-R lane's pub row has the unit's call and blocks, a v1 lane's neither")
        out[PUB] = (vals[PUB], pub_words(salt, position, roots, int(ln), call=call, blocks=blocks))
        if ADMIT in vals:
            out[ADMIT] = (vals[ADMIT], bits_words(admit))
    return out


def produced_values(r: int, c: int, k: int, *, j10r: bool = False) -> dict[str, tuple[int, int]]:
    """The drawn leaf's after-draw values (`one_stage.produced`), each ``(rows, u16 words a row)``: ``partials`` (a row a
    lane of 2 `words`(k)), and a J10-R leaf's ``carry<ch>`` (a row a lane, `carry_words`) and ``oks`` (a row a lane or
    chunk statement, one word)."""
    out = {PARTIALS: (lanes(r, c), 2 * words(k))}
    if j10r:
        n = n_chunks(k)
        out.update({carry(ch): (lanes(r, c), carry_words(k, n, ch)) for ch in range(n - 1)})
        out[OKS] = (lanes(r, c) * n, len(OK_ROW))
    return out


def verifier_entries(registration: Mapping[str, Any], produced: Mapping[str, Any], lanes_: int,
                     words_: int, *, values: Mapping[str, tuple[int, int]] | None = None,
                     ) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    """The verifier's entries and roots of a drawn leaf's values, from the accepted registration record and the accepted
    produced record alone (Lean's `PouwWindow.Lanes.roots?`): each registered value's entry in the registration's
    ``registered`` (the first of its value) and that entry's root, and ``partials``' own entry (`produced.entry`:
    ``lanes_`` rows of ``words_`` u16 words under the registration's digest) at the produced record's root of
    ``partials``; with ``values`` (`produced_values`, a J10-R leaf's) each of those after-draw values' the same way.
    `registered_file` reads them; `pouw-leaf-own` holds a statement's file to them."""
    ent: dict[str, Mapping[str, Any]] = {}
    roots: dict[str, str] = {}
    for e in registration.get("registered") or ():
        if e["value"] not in ent:
            ent[e["value"]], roots[e["value"]] = e, e["root"]
    for value, (rows, wds) in (values or {PARTIALS: (lanes_, words_)}).items():
        got = [e for e in produced["produced"] if e.get("value") == value]
        if not got:
            raise ValueError(f"the produced record has no {value}")
        ent[value] = PR.entry(registration["program"], value, rows, wds, R.digest(registration))
        roots[value] = got[0]["root"]
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
    `LeafLanesSound` takes). So is a J10-R leaf's port of a carry or of ``oks`` (`read_in_full`), but never an ``o0``,
    an output row, of which the circuit returns only the ok bit."""
    out = {}
    for port, (value, row) in ports.items():
        table, base = (place or {}).get(value, (value, 0))
        if table not in entries or table not in roots:
            raise ValueError(f"port {port} reads {table}, whose entry and root the verifier does not hold")
        out[port] = PR.port(table, entries[table], roots[table], [base + row], (schemas or {}).get(table, G.SCHEMA))
        if read_in_full(value) and port != OK:
            out[port][READ_IN_FULL] = True
    if set(out) & set(known_ports or {}):
        raise ValueError(f"ports {sorted(set(out) & set(known_ports or {}))} are both reads and known")
    return {"format": G.READS, "ports": {**out, **(known_ports or {})}, **({COMPLETE: True} if complete else {})}


__all__ = [
    "ADMIT",
    "CARRY",
    "CHUNK",
    "COMBINE",
    "COMPLETE",
    "FULL",
    "LANE",
    "LANE_COLS",
    "LANE_ROWS",
    "OK",
    "OKS",
    "OK_ROW",
    "PARTIALS",
    "PUB",
    "PUBLIC",
    "PUBLIC_BITS",
    "PUBLIC_J10R",
    "PUBLIC_J10R_BITS",
    "READ_IN_FULL",
    "RECEIPT",
    "ZY",
    "A",
    "B",
    "bits_words",
    "carry",
    "carry_words",
    "key_block",
    "known",
    "known_rows",
    "lane_cols",
    "lane_rows",
    "lanes",
    "n_chunks",
    "n_digits",
    "own",
    "produced_values",
    "pub_words",
    "read_in_full",
    "reads",
    "registered_file",
    "statements",
    "verifier_entries",
    "word_order",
    "words",
]
