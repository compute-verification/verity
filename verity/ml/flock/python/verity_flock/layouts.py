"""Layouts: how a statement places its circuit types (Project doc `constant-api-public.md` §2.2, settled with the soundness
lane). The verifier derives every row from the types and these records; a statement never carries rows. This module is the
Python reference for parsing and checking them (`Flock/Layout.lean` decides the same `WellFormed`), and for each layout's
own-region size, the count `derive`'s v2 rule makes.

A record, hashed as RFC 8785 JSON under ``"verity/layout/v1"``::

    {"sha512": ..., "type": digest (absent for a generated read), "range_log": s (absent for the unit's layout),
     "in_split": [...], "out_split": [...],
     "own": {"derive": "v2"} | {"gen": "table/v2", "table": sha512, "in_bits": n, "value_bits": v, "lo_bits": lo},
     "calls": [{"item": i, "layout": j, "offset": q} | {"item": i, "layout": j, "inline": true}, ...],
     "reads": [{"item": i, "layout": j, "offset": q} | {"item": i, "inline": {"lo_bits": lo}}, ...]}

Layouts are listed callees first, and the statement names the unit's (``unit_layout``). Rules:

- **An inner layout's own region fills** ``[0, own_size)``:
  - its input port groups, each an aligned power-of-two run of 128-bit words (one word at least), leaving out exported bits;
  - one row per AND item, the rows of inlined callees and of inline reads at their items;
  - its output port groups, aligned the same way;
  - its constant.

  Each placed callee sits at an aligned offset at or above ``own_size``, inside ``2^range_log``.
- **The unit's layout has no range:** the block places its own region and each layout it places in block ranges. Its
  entries' ``offset`` is the call's slot among the unit's placed entries of the same layout, in item order.
- **Exported inputs.** A placed call's input bit is exported when its form is exactly one input wire w of the caller that
  nothing else in the caller reads. The callee's input row is then w's column, and w takes no row in the caller's groups.
- **An inlined call names its callee's layout, which places nothing.** It governs the expansion, so a read inside it gets
  its ``lo_bits`` from it.
- **A read is tied to its type's table entry** ``[sha512, index_bits, value_bits]``: the held table has exactly
  ``2^index_bits`` words, and a placed read's generated layout has that ``in_bits`` and ``value_bits``.
- **A read's rows are the ``table/v2`` generator's,** with product rows for its low ``k`` value bits. For a library table
  ``k`` is one more than its highest varying bit (every higher bit is constant over the table, and wired); a program
  table keeps every bit (``k`` = its value bits), so no shape depends on a program table's values:
  - inline: decoders, ``k·2^(n-lo)`` product rows and ``k`` copies;
  - placed: that behind a 128-bit input word, padded to a word, then a 128-bit output word and the constant.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits import library

from . import circuit_types as CT

FORMAT = "verity/layout/v1"
GEN = "table/v2"
WORD = 128
_SHA512 = re.compile(r"[0-9a-f]{128}")


class InvalidLayout(ValueError):
    """A statement's layouts break a rule."""


def ceil_pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def decode_rows(m: int) -> int:
    """The rows of a one-hot decoder of m bits: each block of two or more bits commits its minterms."""
    return 0 if m <= 1 else (1 << m) + decode_rows(m // 2) + decode_rows(m - m // 2)


def live_bits(words: Sequence[int], value_bits: int) -> int:
    """One more than the highest of the low ``value_bits`` bits that is not constant over the table."""
    return (table_info(words).diff & ((1 << value_bits) - 1)).bit_length()


def read_rows_inline(n: int, lo: int, k: int) -> int:
    return decode_rows(lo) + decode_rows(n - lo) + k * (1 << (n - lo)) + k


def read_rows_placed(n: int, lo: int, k: int) -> int:
    body = WORD + decode_rows(lo) + decode_rows(n - lo) + k * (1 << (n - lo))
    return -(-body // WORD) * WORD + WORD + 1


@dataclass(frozen=True)
class Layout:
    type: str | None
    range_log: int | None
    in_split: tuple[int, ...]
    out_split: tuple[int, ...]
    own: tuple
    calls: tuple
    reads: tuple

    def record(self) -> dict:
        own = {"derive": "v2"} if self.own[0] == "derive" else {"gen": GEN, "table": self.own[1], "in_bits": self.own[2],
                                                               "value_bits": self.own[3], "lo_bits": self.own[4]}
        calls = [{"item": i, "layout": j, "offset": q} if q is not None else {"item": i, "layout": j, "inline": True}
                 for i, j, q in self.calls]
        reads = [{"item": i, "layout": j, "offset": q} if j is not None else {"item": i, "inline": {"lo_bits": q}}
                 for i, j, q in self.reads]
        d = {"in_split": list(self.in_split), "out_split": list(self.out_split), "own": own, "calls": calls, "reads": reads}
        if self.type is not None:
            d["type"] = self.type
        if self.range_log is not None:
            d["range_log"] = self.range_log
        return d

    def digest(self) -> str:
        return hashlib.sha512(FORMAT.encode() + b"\0" + canonical_json(self.record())).hexdigest()

    def to_json(self) -> dict:
        return {**self.record(), "sha512": self.digest()}


def from_json(d: Mapping) -> Layout:
    """The layout a record encodes; its ``sha512`` must be the record's digest."""
    def bad(why):
        raise InvalidLayout(why)

    try:
        own = d["own"]
        if own == {"derive": "v2"}:
            o = ("derive",)
        elif own.get("gen") == GEN and set(own) == {"gen", "table", "in_bits", "value_bits", "lo_bits"}:
            o = ("gen", own["table"], own["in_bits"], own["value_bits"], own["lo_bits"])
        else:
            bad(f"own is neither derive v2 nor gen {GEN}: {own}")
        calls = []
        for e in d["calls"]:
            if set(e) == {"item", "layout", "offset"}:
                calls.append((e["item"], e["layout"], e["offset"]))
            elif set(e) == {"item", "layout", "inline"} and e["inline"] is True:
                calls.append((e["item"], e["layout"], None))
            else:
                bad(f"a call entry is neither placed nor inline: {e}")
        reads = []
        for e in d["reads"]:
            if set(e) == {"item", "layout", "offset"}:
                reads.append((e["item"], e["layout"], e["offset"]))
            elif set(e) == {"item", "inline"} and set(e["inline"]) == {"lo_bits"}:
                reads.append((e["item"], None, e["inline"]["lo_bits"]))
            else:
                bad(f"a read entry is neither placed nor inline: {e}")
        extra = set(d) - {"sha512", "type", "range_log", "in_split", "out_split", "own", "calls", "reads"}
        if extra:
            bad(f"unknown keys {sorted(extra)}")
        lay = Layout(d.get("type"), d.get("range_log"), tuple(d["in_split"]), tuple(d["out_split"]), o, tuple(calls),
                     tuple(reads))
    except (KeyError, TypeError, AttributeError) as e:
        raise InvalidLayout(f"not a layout record: {e!r}") from None
    if d.get("sha512") != lay.digest():
        bad("sha512 is not the record's digest")
    return lay


@dataclass(frozen=True)
class TableInfo:
    """A held table's facts: its word count, and the OR over its words of each word XOR the first."""
    size: int
    diff: int


def table_info(words: Sequence[int]) -> TableInfo:
    if hasattr(words, "dtype"):
        import numpy as np

        arr = np.asarray(words, dtype=np.uint64)
        return TableInfo(len(arr), int(np.bitwise_or.reduce(arr ^ arr[0])) if len(arr) else 0)
    w0, diff = words[0], 0
    for w in words:
        diff |= w ^ w0
    return TableInfo(len(words), diff)


def k_of(sha: str, value_bits: int, ti: TableInfo) -> int:
    """The value bits a read gives product rows: a library table's up to its highest varying bit, a program table's all."""
    if library.table_label(sha) == library.LIBRARY:
        return (ti.diff & ((1 << value_bits) - 1)).bit_length()
    return value_bits


def _ints_ok(xs, lo, hi) -> bool:
    return all(isinstance(x, int) and not isinstance(x, bool) and lo <= x < hi for x in xs) and list(xs) == sorted(set(xs))


def _port_bits(ports) -> list[int]:
    return [n * w for n, w in ports]


def _groups(bits: list[int], split: Sequence[int]) -> list[tuple[int, int]]:
    """Each group's (first bit, bit count) over ports split at the given port indices."""
    cuts, starts, out = [0, *split, len(bits)], [0], []
    for b in bits:
        starts.append(starts[-1] + b)
    for a, b in zip(cuts, cuts[1:]):
        if b > a:
            out.append((starts[a], starts[b] - starts[a]))
    return out


def _uses(t: CT.CircuitType) -> dict[int, int]:
    n: dict[int, int] = {}
    for it in t.items:
        for f in CT._item_forms(it):
            for w in f[0]:
                n[w] = n.get(w, 0) + 1
    for f in t.outputs:
        for w in f[0]:
            n[w] = n.get(w, 0) + 1
    return n


def exported(t: CT.CircuitType, lay: Layout) -> set[int]:
    """The caller's input wires exported to a placed call: a whole single-wire argument that nothing else reads."""
    uses, out, c = _uses(t), set(), 0
    for it in t.items:
        if it[0] != CT.CALL:
            continue
        placed = lay.calls[c][2] is not None
        c += 1
        if placed:
            for f in it[2]:
                if f[1] == 0 and len(f[0]) == 1 and f[0][0] < t.in_bits and uses.get(f[0][0]) == 1:
                    out.add(f[0][0])
    return out


def _own_items_rows(types: Mapping[str, CT.CircuitType], layouts: Sequence[Layout], t: CT.CircuitType, lay: Layout,
                    info: Callable[[str], TableInfo]) -> int:
    n = c = r = 0
    for it in t.items:
        if it[0] == CT.AND:
            n += 1
        elif it[0] == CT.CALL:
            _, j, q = lay.calls[c]
            c += 1
            if q is None:
                sub = layouts[j]
                n += _own_items_rows(types, layouts, types[sub.type], sub, info)
        else:
            _, j, lo = lay.reads[r]
            r += 1
            if j is None:
                sha, ib, vb = t.tables[it[1]]
                n += read_rows_inline(ib, lo, k_of(sha, vb, info(sha)))
    return n


def own_size(types: Mapping[str, CT.CircuitType], layouts: Sequence[Layout], j: int, info: Callable[[str], TableInfo]) -> int:
    """Layout j's own-region rows under derive's v2 rule (a generated read: its whole rows)."""
    lay = layouts[j]
    if lay.own[0] == "gen":
        _, sha, n, vb, lo = lay.own
        return read_rows_placed(n, lo, k_of(sha, vb, info(sha)))
    t = types[lay.type]
    ex = exported(t, lay)
    rows = 0

    def group(nbits):
        nonlocal rows
        span = WORD * ceil_pow2(max(1, -(-nbits // WORD)))
        rows = -(-rows // span) * span + span

    for first, count in _groups(_port_bits(t.ins), lay.in_split):
        group(sum(1 for w in range(first, first + count) if w not in ex))
    rows += _own_items_rows(types, layouts, t, lay, info)
    for _first, count in _groups(_port_bits(t.outs), lay.out_split):
        group(count)
    return rows + 1


def _places_nothing(layouts: Sequence[Layout], j: int) -> bool:
    lay = layouts[j]
    return (lay.own[0] == "derive" and all(q is None for _, _, q in lay.calls) and all(lj is None for _, lj, _ in lay.reads)
            and all(_places_nothing(layouts, jj) for _, jj, _ in lay.calls))


def validate(types: Mapping[str, CT.CircuitType], layouts: Sequence[Layout], unit: int, info: Callable[[str], TableInfo | None]
             ) -> list[int]:
    """Refuse layouts that break a rule; return each layout's own-region size. ``types`` must already be valid
    (`circuit_types.validate`); ``info(sha512)`` describes a held table, or None."""
    def bad(why):
        raise InvalidLayout(why)

    if not 0 <= unit < len(layouts):
        bad("unit_layout names no layout")
    reached = [j == unit for j in range(len(layouts))]
    for j in range(len(layouts) - 1, -1, -1):
        if reached[j]:
            for jj in [e[1] for e in layouts[j].calls] + [e[1] for e in layouts[j].reads if e[1] is not None]:
                if isinstance(jj, int) and 0 <= jj < j:
                    reached[jj] = True
    if not all(reached):
        bad("a layout the unit's does not reach")
    sizes: list[int] = []
    for j, lay in enumerate(layouts):
        is_unit = j == unit
        if (lay.range_log is None) != is_unit or (lay.range_log is not None and not 7 <= lay.range_log <= 40):
            bad(f"layout {j}: a range_log in 7..40 on every layout but the unit's, which has none")
        if lay.own[0] == "gen":
            _, sha, n, vb, lo = lay.own
            ti = info(sha) if isinstance(sha, str) and _SHA512.fullmatch(sha) else None
            if lay.type is not None or is_unit or lay.calls or lay.reads or lay.in_split or lay.out_split:
                bad(f"layout {j}: a generated read has no type, entries or port groups, and isn't the unit's")
            if (ti is None or not (isinstance(n, int) and 1 <= n <= 32) or ti.size != 1 << n
                    or not (isinstance(vb, int) and 1 <= vb <= 32) or not (isinstance(lo, int) and 2 <= lo < n)):
                bad(f"layout {j}: a generated read of a table not held, or of the wrong index width, value width or split")
            if library.table_label(sha) == library.LIBRARY and (library.table(sha).index_bits, library.table(sha).value_bits) != (n, vb):
                bad(f"layout {j}: a generated read of a library table in a shape other than the library's")
            sizes.append(read_rows_placed(n, lo, k_of(sha, vb, ti)))
            if sizes[-1] > 1 << lay.range_log:
                bad(f"layout {j}: its read's {sizes[-1]} rows exceed its range")
            continue
        t = types.get(lay.type)
        if t is None:
            bad(f"layout {j}: its type is not in the statement")
        if not _ints_ok(lay.in_split, 1, len(t.ins)) or not _ints_ok(lay.out_split, 1, len(t.outs)):
            bad(f"layout {j}: port groups are not increasing port indices inside the ports")
        call_items = [i for i, it in enumerate(t.items) if it[0] == CT.CALL]
        read_items = [i for i, it in enumerate(t.items) if it[0] == CT.READ]
        if len(lay.calls) != len(call_items) or len(lay.reads) != len(read_items):
            bad(f"layout {j}: not one entry per call and read item")
        placed: list[tuple[int, int]] = []
        slots: dict[int, int] = {}
        for (i, jj, q), ci in zip(lay.calls, call_items):
            if i != ci or not (isinstance(jj, int) and 0 <= jj < j):
                bad(f"layout {j}: a call entry names the wrong item or a layout not listed before it")
            callee = t.calls[t.items[ci][1]]
            if layouts[jj].own[0] != "derive" or layouts[jj].type != callee:
                bad(f"layout {j}: a call entry's layout is not of the callee's type")
            if q is None:
                if not _places_nothing(layouts, jj):
                    bad(f"layout {j}: an inlined call's layout places something")
            else:
                placed.append((jj, q))
        for (i, jj, q), ri in zip(lay.reads, read_items):
            if i != ri:
                bad(f"layout {j}: a read entry names the wrong item")
            sha, ib, vb = t.tables[t.items[ri][1]]
            ti = info(sha)
            if ti is None or ti.size != 1 << ib:
                bad(f"layout {j}: a read of a table not held, or not of 2^index_bits words")
            if jj is None:
                if not (isinstance(q, int) and 2 <= q < ib):
                    bad(f"layout {j}: an inline read's split is out of range")
            else:
                if not (isinstance(jj, int) and 0 <= jj < j) or layouts[jj].own[:4] != ("gen", sha, ib, vb):
                    bad(f"layout {j}: a placed read's layout is not a generated read of its table, index width and value width")
                placed.append((jj, q))
        size = own_size(types, layouts, j, lambda s: info(s))
        sizes.append(size)
        if is_unit:
            for jj, q in placed:
                if q != slots.get(jj, 0):
                    bad(f"layout {j}: the unit's entries of one layout take slots 0, 1, ... in item order")
                slots[jj] = q + 1
            continue
        R = 1 << lay.range_log
        if size > R:
            bad(f"layout {j}: its own region of {size} rows exceeds its range")
        spans = []
        for jj, q in placed:
            s = 1 << layouts[jj].range_log
            if not (isinstance(q, int) and q >= 0) or q * s < size or (q + 1) * s > R:
                bad(f"layout {j}: a placed range is off its own region or out of its range")
            spans.append((q * s, (q + 1) * s))
        for a in range(len(spans)):
            for b in range(a + 1, len(spans)):
                if spans[a][0] < spans[b][1] and spans[b][0] < spans[a][1]:
                    bad(f"layout {j}: two placed ranges overlap")
    return sizes


__all__ = ["FORMAT", "GEN", "InvalidLayout", "Layout", "TableInfo", "decode_rows", "exported", "from_json", "k_of", "live_bits",
           "own_size", "read_rows_inline", "read_rows_placed", "table_info", "validate"]
