"""`derive`: the rows of a statement's circuit types under their layouts, the prover's mirror of the verifier's Lean
`Flock.Derive` (the soundness lane's #199 for flat types, S3 for the rest; Project doc `constant-api-public.md` §2.2).
The verifier derives every row it folds from the types and layouts, so these rows are never trusted: `check` holds them
to the Lean derivation byte for byte, and a mismatch can only make an honest proof fail.

**Physical rows,** where rows sit:
- **An inner layout's range** holds:
  - its own region: input port groups (aligned word runs, exported bits left out), one row per AND item (inlined
    callees' items and inline reads' rows at their items), output copy rows, and the constant last. That's
    `ir_lower.layout`'s v2 rule, so a flat type's rows are `ir_lower.netlist`'s byte for byte;
  - each placed callee's rows at its aligned offset. Its non-exported input rows become binding copies `form · 1`
    (A = the caller's form, B = the callee's constant), and its constant row copies the caller's (`[c] · [c]`).
- **A read's rows** are `table/v2`'s: `Flock.Lookup.buildV2` (the verifier lane's #202) when placed; inline, the same
  decoders over the index forms, the product rows and copies for the low k bits, and the higher (constant) bits as
  constant wires. A product row's B side is the table's, folded table-direct, so it's listed empty and the read's
  `READ` record names it.
- **The unit's layout is placed by the block.** Its physical rows are its own region alone, constant last as a self
  row, as `ir_lower.netlist` writes a unit. Each placed entry is a part, a slot of its layout, whose rows are that
  layout's own; the block's Δ binds them (`Derived.delta`): each part's non-exported input rows to the unit's forms,
  the same form in A and B, and every slot's constant to the pin, `one`. In the unit coordinates this module uses for
  the logical list, the own region comes first, then each part's range in item order, aligned, then `one`.

**The logical order,** what `Rows.compose` reads, over the rows the block's matrices hold, Δ applied (store note
`internal/lanes/coordinator/20260928T0250Z-note-to-flock-soundness-constant-first-logical-order.md`):
- the unit's inputs, exported bits included;
- then the unit's segment. A segment is its constant copy, its binding copies (one per non-exported input bit), its
  items' rows in item order (a placed call or read being that callee's segment, recursively), and its output copy rows;
- `one` last.

Forced-zero padding rows are in neither list's meaning; a placed read's output rows are all listed, since readers read
them.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from verity.ml.circuits import library

from . import circuit_types as CT
from . import layouts as LY

WORD = 128
KONST = -1
ONE = "one"

Form = tuple[tuple[int, ...], int]


def ceil_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _xor(forms: Sequence[Form], c: int = 0) -> Form:
    cols: set[int] = set()
    for fs, fc in forms:
        cols ^= set(fs)
        c ^= fc
    return tuple(sorted(cols)), c


def _entries(f: Form) -> list:
    return list(f[0]) + ([KONST] if f[1] else [])


@dataclass
class Read:
    """A read's table-direct B side: product row ``prod0 + j·nh + h`` (``j < k``) reads low minterm ``lo_top + l`` exactly
    when bit j of word ``h·2^lo + l`` is set."""
    table: str
    n: int
    lo: int
    k: int
    prod0: int
    lo_top: int

    def shifted(self, d: int) -> "Read":
        return Read(self.table, self.n, self.lo, self.k, self.prod0 + d, self.lo_top + d)


@dataclass
class Derived:
    """A layout's rows in local columns (its range, or the unit's coordinates)."""
    name: str
    rows: dict = field(default_factory=dict)  # column -> (A, B), KONST resolved
    const: int = 0
    in_groups: list = field(default_factory=list)  # (word, words, per-port bits)
    out_groups: list = field(default_factory=list)
    in_cols: list = field(default_factory=list)  # each input bit's column
    out_cols: list = field(default_factory=list)  # each output bit's column (its copy row)
    items: list = field(default_factory=list)  # the item rows' columns, in logical order
    reads: list = field(default_factory=list)
    own: int = 0  # the own region's rows
    size: int = 0  # the physical rows: up to an inner layout's last placed range; the unit's own region
    parts: list = field(default_factory=list)  # the unit's placed entries: (layout index, base in unit coordinates)
    part_rows: list = field(default_factory=list)  # the unit's parts' own derivations with their bases, copied lazily
    delta: list = field(default_factory=list)  # the unit's Δ: (row in unit coordinates, source entries or ONE)
    extent: int = 0  # the unit coordinates' end, where `one` sits


class _Build:
    def __init__(self, name: str):
        self.d = Derived(name)
        self.at = 0

    def row(self, a: list, b: list) -> int:
        c = self.at
        self.d.rows[c] = (a, b)
        self.at += 1
        return c

    def decode(self, leaves: Sequence[Form]) -> tuple[list[Form], int]:
        """``Flock.Lookup.decode``: minterm i of the leaves, committing every block of two or more bits; the first row."""
        if len(leaves) == 1:
            (cols, c), = leaves
            return [(cols, c ^ 1), (cols, c)], self.at
        h = len(leaves) // 2
        lo, _ = self.decode(leaves[:h])
        hi, _ = self.decode(leaves[h:])
        first = self.at
        out = [((self.row(_entries(lo[i % (1 << h)]), _entries(hi[i >> h])),), 0) for i in range(1 << len(leaves))]
        return out, first


def _resolve(rows: dict, const) -> dict:
    return {c: (sorted(x if x != KONST else const for x in a), sorted(x if x != KONST else const for x in b))
            for c, (a, b) in rows.items()}


def gen_read(words: Sequence[int], sha: str, n: int, lo: int, bits: int) -> Derived:
    """``Flock.Lookup.buildV2``: a placed read's rows. Input word (the index bits' self rows), the two decoders, ``k·nh``
    product rows (hi minterm in A, the table's side in B), padding to a word, the output word (bit j < k the XOR of its
    products, a constant bit's row ``[c]·[c]`` for 1 and empty for 0), the constant."""
    assert len(words) == 1 << n and 2 <= lo < n and bits <= 32
    k = LY.k_of(sha, bits, LY.table_info(words))
    B = _Build(f"table/v2 {sha[:16]}")
    for i in range(WORD):
        B.row([i], [i]) if i < n else B.row([], [])
    lo_m, lo_top = B.decode([((i,), 0) for i in range(lo)])
    hi_m, _ = B.decode([((i,), 0) for i in range(lo, n)])
    nh = len(hi_m)
    prod0 = B.at
    for _ in range(k):
        for h in range(nh):
            B.row(_entries(hi_m[h]), [])
    while B.at % WORD:
        B.row([], [])
    out0 = B.at
    for j in range(WORD):
        if j < k:
            B.row([prod0 + j * nh + h for h in range(nh)], [KONST])
        elif j < bits and (words[0] >> j) & 1:
            B.row([KONST], [KONST])
        else:
            B.row([], [])
    const = B.row([KONST], [KONST])
    d = B.d
    d.rows, d.const, d.own, d.size = _resolve(d.rows, const), const, B.at, B.at
    d.in_groups, d.out_groups = [(0, 1, [n])], [(out0 // WORD, 1, [bits])]
    d.in_cols, d.out_cols = list(range(n)), [out0 + j for j in range(bits)]
    d.items = list(range(WORD, prod0 + k * nh))
    d.reads = [Read(sha, n, lo, k, prod0, lo_top)]
    return d


def rows_digest(d: Derived, n: int, lo: int, bits: int) -> str:
    """``Flock.Lookup.rowsDigest`` of a generated read, the digest #202's vectors pin."""
    r = d.reads[0]
    unit_log = max(7, (d.size - 1).bit_length())
    h = bytearray(b"flock-lookup-rows/v2\0")
    for x in (n, lo, bits, r.k, d.size, d.const, r.prod0, r.lo_top, unit_log):
        h += x.to_bytes(4, "little")
    for c in range(d.size):
        for m in d.rows[c]:
            h += len(m).to_bytes(4, "little")
            for x in m:
                h += x.to_bytes(4, "little")
    return hashlib.sha512(bytes(h)).hexdigest()


class Derivation:
    """Every layout of a statement derived once (callees first), and the unit's rows in unit coordinates."""

    def __init__(self, types: Mapping[str, CT.CircuitType], layouts: Sequence[LY.BlockLayout], unit: int,
                 tables: Mapping[str, Sequence[int]]):
        self.types, self.layouts, self.unit_j, self.tables = types, layouts, unit, tables
        self.info = {sha: LY.table_info(ws) for sha, ws in tables.items()}
        LY.validate(types, layouts, unit, self.info.get)
        self.done: dict[int, Derived] = {}
        for j in range(len(layouts)):
            if j != unit:
                self.done[j] = self._derive(j, unit_level=False)
        self.unit = self._derive(unit, unit_level=True)

    # --- one layout ---------------------------------------------------------------------------------------------------
    def _derive(self, j: int, unit_level: bool) -> Derived:
        lay = self.layouts[j]
        if lay.own[0] == "gen":
            _, sha, n, vb, lo = lay.own
            return gen_read(self.tables[sha], sha, n, lo, vb)
        t = self.types[lay.type]
        B = _Build(t.digest()[:16])
        ex = LY.exported(t, lay)
        wire: dict[int, Form] = {}
        # the own region: input groups without exported bits
        bits = [n * w for n, w in t.ins]
        groups = LY._groups(bits, lay.in_split)
        cuts = [0, *lay.in_split, len(bits)]
        port_spans = [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]
        in_cols: list[int | None] = [None] * t.in_bits
        for (first, count), (pa, pb) in zip(groups, port_spans):
            kept = [w for w in range(first, first + count) if w not in ex]
            size = WORD * ceil_pow2(max(1, -(-len(kept) // WORD)))
            start = -(-B.at // size) * size
            while B.at < start:
                B.row([], [])
            for i, w in enumerate(kept):
                in_cols[w] = start + i
            per_port, s = [], sum(bits[:pa])
            for p in range(pa, pb):
                per_port.append(sum(1 for w in range(s, s + bits[p]) if w not in ex))
                s += bits[p]
            B.d.in_groups.append((start // WORD, size // WORD, per_port))
            while B.at < start + size:
                c = B.at
                B.row([c], [c]) if c - start < len(kept) else B.row([], [])
        for w in range(t.in_bits):
            if w not in ex:
                wire[w] = ((in_cols[w],), 0)
        # placed entries: their bases, so exported wires' columns are known before any form reads them
        placed = self._placements(j, t, lay, unit_level)
        for w, col in placed["export_cols"].items():
            in_cols[w] = col
            wire[w] = ((col,), 0)
        # the items
        self._items(t, lay, B, wire, placed)
        # output groups, then the constant
        outs = [_xor([wire[x] for x in f[0]], f[1]) for f in t.outputs]
        obits = [n * w for n, w in t.outs]
        out_cols: list[int] = []
        for first, count in LY._groups(obits, lay.out_split):
            size = WORD * ceil_pow2(max(1, -(-count // WORD)))
            start = -(-B.at // size) * size
            while B.at < start:
                B.row([], [])
            for o in range(first, first + count):
                out_cols.append(B.row(_entries(outs[o]), [KONST]))
            while B.at < start + size:
                B.row([], [])
        B.d.out_groups = self._out_groups(obits, lay.out_split, out_cols)
        const = B.row([KONST], [KONST])
        d = B.d
        d.const, d.own = const, B.at
        d.in_cols, d.out_cols = [c for c in in_cols], out_cols
        # the placed entries' rows: bound here in an inner layout, bound by the block's Δ for the unit
        for inst in placed["insts"]:
            self._instantiate(d, inst, const, unit_level)
        d.rows = _resolve(d.rows, const)
        if unit_level:
            d.size = d.own
            d.extent = max([d.own] + [inst["base"] + (1 << self.layouts[inst["layout"]].range_log) for inst in placed["insts"]])
            d.parts = [(inst["layout"], inst["base"]) for inst in placed["insts"]]
            d.delta = [(const, ONE)] + d.delta
        else:
            d.size = max([d.own] + [inst["base"] + (1 << self.layouts[inst["layout"]].range_log) for inst in placed["insts"]])
        return d

    @staticmethod
    def _out_groups(obits, split, out_cols):
        out, i = [], 0
        cuts = [0, *split, len(obits)]
        for a, b in zip(cuts, cuts[1:]):
            if b > a:
                n = sum(obits[a:b])
                size = WORD * ceil_pow2(max(1, -(-n // WORD)))
                out.append((out_cols[i] // WORD, size // WORD, list(obits[a:b])))
                i += n
        return out

    def _placements(self, j: int, t: CT.CircuitType, lay: LY.BlockLayout, unit_level: bool) -> dict:
        """Each placed call and read: its layout, base and the export locations of its caller's wires. In the unit's
        coordinates, placed entries follow its own region in item order, each aligned to its range."""
        insts, export_cols, end = [], {}, 0
        uses = LY._uses(t)
        c = r = 0
        cursor = LY.own_size(self.types, self.layouts, j, self.info.get)
        for i, it in enumerate(t.items):
            if it[0] == CT.CALL:
                _, jj, q = lay.calls[c]
                c += 1
                if q is None:
                    continue
                sub = self.done[jj]
                s = self.layouts[jj].range_log
                if unit_level:
                    base = -(-cursor // (1 << s)) * (1 << s)
                    cursor = base + (1 << s)
                else:
                    base = q * (1 << s)
                exported = set()
                for bit, f in enumerate(it[2]):
                    if f[1] == 0 and len(f[0]) == 1 and f[0][0] < t.in_bits and uses.get(f[0][0]) == 1:
                        exported.add(bit)
                        export_cols[f[0][0]] = base + sub.in_cols[bit]
                insts.append(dict(item=i, kind="call", layout=jj, sub=sub, base=base, args=it[2], exported=exported))
                end = max(end, base + (1 << s))
            elif it[0] == CT.READ:
                _, jj, q = lay.reads[r]
                r += 1
                if jj is None:
                    continue
                sub = self.done[jj]
                s = self.layouts[jj].range_log
                if unit_level:
                    base = -(-cursor // (1 << s)) * (1 << s)
                    cursor = base + (1 << s)
                else:
                    base = q * (1 << s)
                insts.append(dict(item=i, kind="read", layout=jj, sub=sub, base=base, args=it[2], exported=set()))
                end = max(end, base + (1 << s))
        return dict(insts=insts, export_cols=export_cols, end=end, by_item={x["item"]: x for x in insts})

    def _items(self, t: CT.CircuitType, lay: LY.BlockLayout, B: _Build, wire: dict, placed: dict,
               args: Sequence[Form] | None = None) -> list[Form]:
        """Lay out ``t``'s items under ``lay`` into ``B``'s own region, in item order; ``args`` binds an inlined type's
        inputs. Returns its output forms."""
        if args is not None:
            wire = {w: f for w, f in enumerate(args)}
        n = t.in_bits
        c = r = 0
        for i, it in enumerate(t.items):
            if it[0] == CT.AND:
                col = B.row(_entries(_xor([wire[x] for x in it[1][0]], it[1][1])),
                            _entries(_xor([wire[x] for x in it[2][0]], it[2][1])))
                B.d.items.append(col)
                wire[n] = ((col,), 0)
                n += 1
            elif it[0] == CT.CALL:
                _, jj, q = lay.calls[c]
                c += 1
                u = self.types[t.calls[it[1]]]
                fs = [_xor([wire[x] for x in f[0]], f[1]) for f in it[2]]
                if q is None:
                    outs = self._items(u, self.layouts[jj], B, {}, {"by_item": {}}, args=fs)
                else:
                    inst = placed["by_item"][i]
                    inst["forms"] = fs
                    B.d.items.append(("seg", inst))
                    outs = [((inst["base"] + col,), 0) for col in inst["sub"].out_cols]
                for f in outs:
                    wire[n] = f
                    n += 1
            else:
                _, jj, q = lay.reads[r]
                r += 1
                sha, ib, vb = t.tables[it[1]]
                fs = [_xor([wire[x] for x in f[0]], f[1]) for f in it[2]]
                if jj is None:
                    outs = self._inline_read(B, fs, sha, ib, q, vb)
                else:
                    inst = placed["by_item"][i]
                    inst["forms"] = fs
                    B.d.items.append(("seg", inst))
                    outs = [((inst["base"] + col,), 0) for col in inst["sub"].out_cols]
                for f in outs:
                    wire[n] = f
                    n += 1
        return [_xor([wire[x] for x in f[0]], f[1]) for f in t.outputs]

    def _inline_read(self, B: _Build, idx: Sequence[Form], sha: str, n: int, lo: int, vb: int) -> list[Form]:
        words = self.tables[sha]
        k = LY.k_of(sha, vb, self.info[sha])
        start = B.at
        lo_m, lo_top = B.decode(list(idx[:lo]))
        hi_m, _ = B.decode(list(idx[lo:]))
        nh = len(hi_m)
        prod0 = B.at
        for _ in range(k):
            for h in range(nh):
                B.row(_entries(hi_m[h]), [])
        outs: list[Form] = []
        for j in range(vb):
            if j < k:
                outs.append(((B.row([prod0 + j * nh + h for h in range(nh)], [KONST]),), 0))
            else:
                outs.append(((), (words[0] >> j) & 1))
        B.d.items.extend(range(start, B.at))
        B.d.reads.append(Read(sha, n, lo, k, prod0, lo_top))
        return outs

    def _instantiate(self, d: Derived, inst: dict, const: int, unit_level: bool) -> None:
        """A placed entry's rows at its base. In an inner layout its input rows become binding copies and its constant
        copies the caller's; for the unit, its rows stay its layout's and the bindings go to Δ."""
        sub, base = inst["sub"], inst["base"]
        if unit_level:
            d.part_rows.append((sub, base))
        else:
            for col, (a, b) in sub.rows.items():
                d.rows[base + col] = ([x + base for x in a], [x + base for x in b])
        cb = base + sub.const
        if unit_level:
            d.delta.append((cb, ONE))
        else:
            d.rows[cb] = ([const], [const])
        for bit, f in enumerate(inst["forms"]):
            if bit in inst["exported"]:
                continue
            row = base + sub.in_cols[bit]
            src = [x if x != KONST else const for x in _entries(f)]
            if unit_level:
                d.delta.append((row, sorted(src)))
            else:
                d.rows[row] = (sorted(src), [cb])
        d.reads.extend(r.shifted(base) for r in sub.reads)

    # --- outputs --------------------------------------------------------------------------------------------------------
    def physical(self, j: int | None = None) -> Derived:
        return self.unit if j is None or j == self.unit_j else self.done[j]

    def block_rows(self) -> dict:
        """The unit's rows in unit coordinates as the block's matrices hold them: every part's rows with Δ applied (a
        bound input row reads its source in A and B; a slot's constant reads ``one``)."""
        d = self.unit
        rows = dict(d.rows)
        for sub, base in d.part_rows:
            for col, (a, b) in sub.rows.items():
                rows[base + col] = ([x + base for x in a], [x + base for x in b])
        for c, src in d.delta:
            rows[c] = ([ONE], [ONE]) if src == ONE else (list(src), list(src))
        return rows

    def logical(self) -> list:
        """The unit's rows in logical order, Δ applied: (column, A, B), ``one`` last."""
        d = self.unit
        rows = self.block_rows()
        order = [c for c in d.in_cols] + self._segment_items(d, 0, set(), top=True) + [ONE]
        return [(c, *rows[c]) if c != ONE else (ONE, [ONE], [ONE]) for c in order]

    def _segment_items(self, d: Derived, base: int, exported: set[int], top: bool = False) -> list:
        head = [base + d.const] + ([] if top else [base + c for i, c in enumerate(d.in_cols) if i not in exported])
        body = []
        for x in d.items:
            if isinstance(x, tuple):
                inst = x[1]
                body += self._segment_items(inst["sub"], base + inst["base"], inst["exported"])
            else:
                body.append(base + x)
        return head + body + [base + c for c in d.out_cols]


def netlist(d: Derived, name: str | None = None) -> str:
    """The rows as ``ir_lower.netlist`` writes them (dense, 0 .. size - 1), then one ``READ`` line per read's table-direct
    side in these rows."""
    head = ["flock-ir-unit/v2", name or d.name, d.size, d.const]
    for groups in (d.in_groups, d.out_groups):
        head.append(len(groups))
        for word, words, bits in groups:
            head += [word, words, len(bits), *bits]
    lines = [" ".join(map(str, head))]
    for c in range(d.size):
        a, b = d.rows.get(c, ([], []))
        lines.append(" ".join(map(str, [len(a), *a, len(b), *b])))
    for r in d.reads:
        if r.prod0 < d.size:
            lines.append(f"READ {r.table} {r.n} {r.lo} {r.k} {r.prod0} {r.lo_top}")
    return "\n".join(lines) + "\n"


def delta_text(dv: "Derivation") -> str:
    """The unit's parts and Δ as text: ``PART layout-digest base`` per part, then ``DELTA row |src| src..`` (``one``
    as ``one``)."""
    d = dv.unit
    lines = [f"PART {dv.layouts[j].digest()} {base}" for j, base in d.parts]
    for c, src in d.delta:
        s = ["one"] if src == ONE else src
        lines.append(" ".join(map(str, ["DELTA", c, len(s), *s])))
    return "\n".join(lines) + "\n"


def logical_text(dv: Derivation) -> str:
    """The logical list as text: per row its column, then ``|A| A.. |B| B..`` (``one`` as ``one``)."""
    out = []
    for c, a, b in dv.logical():
        out.append(" ".join(map(str, [c, len(a), *a, len(b), *b])))
    return "\n".join(out) + "\n"


__all__ = ["Derivation", "Derived", "Read", "gen_read", "logical_text", "netlist", "rows_digest"]
