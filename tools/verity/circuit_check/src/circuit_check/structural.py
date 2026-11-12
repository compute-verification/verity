"""``circuit-check --structural UNIT.tmpl TARGET``: a ``flock-ir-tmpl`` template (`verity_flock.tmpl`) accepted as the Flock
lowering of the Definition TARGET, which `verity_flock.ir_lower.lower` lowers flat.

**Syntactic.** Both sides are read into one table of canonical gates: a bit is a GF(2) form over the inputs and the table's
AND nodes, and an AND of two forms is one node, keyed by its operands with their constants taken out and its pair chosen
from the three its operands span, so a gate is the same node whichever way round, or through whichever of the three pairs,
it is written. The template's rows are its unroll's (placed as `tmpl.unroll` places them, which mirrors
``Flock.Tmpl.unroll``) in column order, read off its Definitions copy by copy so the unrolled rows are never held: an input
column is its input, the constant column one, and any other row the AND of its two forms; the lowering's ANDs are
`forms.Circuit`'s. The template is its lowering's gates when each of its output bits is the same form as the lowering's.
That holds exactly when the two are one circuit up to XORs moved across rows, committed forms (a row ``A·1``, which the
template adds past a call's support), constants folded or carried as inputs, dead and duplicated rows, and the order of an
AND's operands, each of which leaves every bit's value as it was: so it proves, by construction, that the template computes
what the lowering computes on every input. It is not complete: one function written as two AND structures is not matched.

A form is held as its fingerprint, the XOR of its nodes' labels, each node's label 512 bits the check samples from the OS's
random source when it makes the node, since a hash proof block's bits are forms of about half its nodes, which no exact
table of them fits. Which forms a check builds does not depend on the labels (a triple's node is its first writer's pair),
and two distinct forms share a fingerprint with probability 2^-512 over labels sampled after both are fixed, so among the M
forms a check builds it identifies two that differ with probability at most M² / 2^513 over its own coins, with no premise
on a hash; short of that the table is the exact one.

**Semantic.** When the forms differ (a builder that chose a different structure for a constant it was given as an input, say),
the report says which output bits differ and both are evaluated, bit-sliced: on every input when the template has at most
`EXHAUSTIVE_BITS` input bits, else on ``vectors`` random inputs, which miss a difference on a fraction ε of the inputs with
probability (1 − ε)^vectors. An exhaustive pass is a proof too; a random one is a warning, ``structural/random-only``.

**A proof block** (`check_unit`): a template on a proof block's u16 row ports, as a structural stager writes it
(`class_statement.UnitStager`'s ``structural``), is held the same way to the flat proof block the flat stager lays out on
those ports
(`class_statement.unit_circuit`), so the class a pin names is accepted as its Definition's lowering on its ports.

Refused by name, since neither comparison covers them: a template one of whose rows, read or not, reads a column at or past
its own that is no input and not the constant (so some column is not a function of the inputs), that writes a column twice
or writes the constant, one of whose rows writes a column a row read before it copy by copy (a callee's column its caller
writes past the copy's first column, an input whose row comes after a reader), whose input columns' rows are not their
own ``(c)·(c)``, whose port groups overlap, reach the constant or are not the Definition's input leaves and output bits, or
hold a column of their words that is no port's bit at other than zero; and a Definition whose lowering asserts, which rows
carry as more than gates. A table read is read as the rows the flat layout writes it as (`ir_lower._Read`), so a template
carrying those rows, as `struct_lower`'s does, is compared gate for gate.
"""
from __future__ import annotations

import math
import operator
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import checks as CK

EXHAUSTIVE_BITS = 16
VECTORS = 4096
WORD = 128
ZERO = 0


class Canon:
    """The canonical gates both sides are read into. A bit is one int, its form's fingerprint shifted up one and its
    constant in bit 0. Nodes ``0 .. inputs - 1`` are the inputs (`inputs`, their bits); each AND node takes the next."""

    def __init__(self, inputs: int) -> None:
        self.next = 0
        self.inputs = [self._node() for _ in range(inputs)]
        self.ands: dict[tuple[int, int], tuple[int, int]] = {}
        self.reads: dict[tuple, list[int]] = {}

    def READ(self, table: str, idx: Sequence[int], bits: int) -> list[int]:
        """The ``bits`` low bits of ``table``'s word at the index whose bits are ``idx``, uninterpreted: one node a bit per
        table and index, so two reads at one index are one value whatever the table holds."""
        key = (table, tuple(idx))
        w = self.reads.get(key)
        if w is None:
            w = self.reads[key] = [self._node() for _ in range(bits)]
        return w

    def _node(self) -> int:
        self.next += 1
        return int.from_bytes(os.urandom(64), "little") << 1

    def AND(self, a: int, b: int) -> int:
        """``a · b``: (f + x)(g + y) = fg + y f + x g + x y. The three forms f, g and f + g have one node, the product of
        the two the first AND of them wrote, and since f (f + g) = f + fg, fg is that node plus the operand it shares
        with another of their pairs."""
        x, y = a & 1, b & 1
        f, g = a ^ x, b ^ y
        if not f:
            return b if x else ZERO
        if not g:
            return a if y else ZERO
        if f == g:
            return a if x == y else ZERO
        h = f ^ g
        lo, mid, _top = sorted((f, g, h))
        v = self.ands.get((lo, mid))
        if v is None:
            fg = self._node()
            self.ands[lo, mid] = (fg, h)
        else:
            n, e = v
            fg = n if e == h else n ^ (g if e == f else f)
        if y:
            fg ^= f
        if x:
            fg ^= g
        return fg | (x & y)


def _xor(bits) -> int:
    acc = 0
    for v in bits:
        acc ^= v
    return acc


class Refused(ValueError):
    """A template neither comparison covers, by name (`code`)."""

    def __init__(self, code: str, what: str) -> None:
        super().__init__(what)
        self.code = code


def ports(groups: Sequence[str]) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """Per input port and per output port of a proof block's header (``n_in (col words n bits..).. n_out (col words n
    bits..)..``), its columns, bit 0 first; and the columns of the groups' words that are no port's bit, which a proof block
    holds at zero."""
    try:
        g = [int(x) for x in groups]
    except ValueError:
        raise Refused("ports", "the header's port groups are not integers") from None
    at, out, pad, spans = 0, [], [], []
    try:
        for _side in range(2):
            side = []
            for _ in range(g[at]):
                col, words, nb = g[at + 1:at + 4]
                bits = g[at + 4:at + 4 + nb]
                lo, hi = col * WORD, (col + words) * WORD
                if len(bits) != nb or min(bits, default=0) < 0 or sum(bits) > hi - lo:
                    raise Refused("ports", f"a port group of {words} words holds the bits {bits}")
                if any(lo < h and l < hi for l, h in spans):
                    raise Refused("ports", f"the port group at word {col} overlaps another")
                spans.append((lo, hi))
                c = lo
                for b in bits:
                    side.append(list(range(c, c + b)))
                    c += b
                pad.extend(range(c, hi))
                at += 3 + nb
            at += 1
            out.append(side)
    except (IndexError, ValueError) as e:
        raise e if isinstance(e, Refused) else Refused("ports", "the header's port groups end early") from None
    if at != len(g):
        raise Refused("ports", f"the header's port groups end at token {at} of {len(g)}")
    return out[0], out[1], pad


def _side(refs: Sequence[int], cols: int, ins: int) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """A row side of a Definition of ``cols`` columns and ``ins`` inputs, each reference named an odd number of times
    (`tmpl.canon`): its own columns, ascending; its inputs; and whether it holds the constant, which any reference past its
    inputs is (as `tmpl.unroll` reads one)."""
    par: dict[int, int] = {}
    for x in refs:
        if x < 0:
            raise Refused("malformed", f"a row reads column {x}")
        x = min(x, cols + ins)
        par[x] = par.get(x, 0) ^ 1
    odd = sorted(x for x, p in par.items() if p)
    return (tuple(x for x in odd if x < cols), tuple(x - cols for x in odd if cols <= x < cols + ins),
            int(cols + ins in odd))


def _walk(t: Any, inputs: Sequence[int], one: int, AND) -> tuple[list, dict]:
    """The value of each of the unrolled template's columns, its rows placed as `tmpl.unroll` places them
    (``Flock.Tmpl.unroll``: each call's copies at their offsets, a reference past a Definition's columns the caller column
    its input is bound to, past its inputs the constant) and evaluated in column order, each AND by ``AND``, the constant
    ``one`` and input port bit ``i`` ``inputs[i]``; and what it took. A Definition's rows are read once however many copies
    place them, so the unrolled rows are never held. Every row is evaluated, read or not, so each is a function of the
    inputs: a row reading a column at or past its own that is no input and not the constant is refused, as is a column
    written twice or the constant written; a column no row writes is zero. A copy's rows are evaluated where the copy
    starts, so a row that writes a column some row read before it in that order (a column a callee leaves to its caller
    past the copy's first column, an input whose own row comes after a row reading it) is refused: that read saw zero
    where column order sees the row's value, and refusing it leaves every read the column's final value, so the walk's
    values are column order's."""
    try:
        n = t.defs[t.root].cols
    except IndexError:
        raise Refused("malformed", f"no root Definition {t.root}") from None
    k = n - 1
    ins, outs, pad = ports(t.groups)
    cols = [c for p in ins for c in p]
    roots = [c for p in outs for c in p]
    if any(c >= k for c in cols + roots + pad):
        raise Refused("ports", f"a port group's words reach the constant's column {k} or past it")
    given = dict(zip(cols, inputs))
    free = {*given, k}
    val: list = [None] * n
    val[k] = one
    prepared: dict[int, tuple] = {}
    early: dict[int, int] = {}
    st = {"evaluated_rows": 0, "ands": 0, "copies": 0}

    def written(C: int) -> None:
        if val[C] is not None:
            raise Refused("malformed", "the constant is written by a row" if C == k else f"column {C} written twice")
        if early and C in early:
            raise Refused("order", f"column {C} is read by the row at column {early[C]} before its own row: a copy's rows "
                                   f"are evaluated where the copy starts, so write the column before its first reader (as "
                                   f"a row of the callee that reads it, or below the copy's first column), and put each "
                                   f"input's row below every row that reads it")

    def prepare(d: int) -> tuple:
        df = t.defs[d]
        sides = [(col, _side(a, df.cols, df.ins), _side(b, df.cols, df.ins)) for col, a, b in df.rows]
        late = [any(x >= col for x in sa[0] + sb[0]) for col, sa, sb in sides]
        order = sorted([(col, 0, i) for i, (col, _a, _b) in enumerate(df.rows)]
                       + [(c.base + j * c.stride, 1, (ci, j)) for ci, c in enumerate(df.calls) for j in range(c.count)])
        return df, sides, late, order, min((col for col, _a, _b in df.rows), default=df.cols)

    def side(s: tuple, o: int, inp: list[int], C: int, checked: bool) -> int:
        loc, ix, cst = s
        v = one if cst else 0
        if checked:
            for r in [o + x for x in loc] + [inp[i] for i in ix]:
                if r >= C and r not in free:
                    raise Refused("order", f"row {C} reads column {r}, at or past its own")
                w = val[r]
                if w:
                    v ^= w
                elif w is None:
                    early.setdefault(r, C)
            return v
        for x in loc:
            w = val[o + x]
            if w:
                v ^= w
            elif w is None:
                early.setdefault(o + x, C)
        for i in ix:
            w = val[inp[i]]
            if w:
                v ^= w
            elif w is None:
                early.setdefault(inp[i], C)
        return v

    def place(d: int, o: int, inp: list[int]) -> None:
        rec = prepared.get(d)
        if rec is None:
            rec = prepared[d] = prepare(d)
        df, sides, late, order, lo = rec
        c0, c1 = df.cols, df.cols + df.ins
        safe = max((r for r in inp if r not in free), default=-1) < o + lo
        for start, kind, p in order:
            if kind:
                ci, j = p
                c = df.calls[ci]
                bound = [o + x if x < c0 else inp[x - c0] if x < c1 else k
                         for ln, at, step, reps, rs in c.segs for r in range(reps)
                         for x in range(at + r * rs + j * step, at + r * rs + j * step + ln)]
                place(c.e, o + start, bound)
                continue
            col, sa, sb = sides[p]
            C = o + col
            if not 0 <= C < n:
                raise Refused("malformed", f"a row at column {C}, outside the unit's {n}")
            written(C)
            if C in given:
                row = tuple(tuple(sorted([o + x for x in s[0]] + [inp[i] for i in s[1]] + [k] * s[2])) for s in (sa, sb))
                if row != ((C,), (C,)):
                    raise Refused("ports", f"input column {C}'s row is {row}, not its own (c)·(c)")
                val[C] = given[C]
                continue
            checked = late[p] or not safe
            va, vb = side(sa, o, inp, C, checked), side(sb, o, inp, C, checked)
            val[C] = AND(va, vb)
            copy = va == one or vb == one
            st["evaluated_rows"] += 1
            st["copies"] += copy
            st["ands"] += bool(va and vb) and not copy

    try:
        place(t.root, 0, [])
    except (IndexError, TypeError) as e:
        raise Refused("malformed", f"its unroll: {type(e).__name__} {e}") from None
    unwritten = [c for c in cols if val[c] is None]
    if unwritten:
        raise Refused("ports", f"input column {unwritten[0]}'s row is ((), ()), not its own (c)·(c)")
    return val, {"rows": n, **st, "input_ports": len(ins), "input_bits": len(cols), "output_bits": len(roots),
                 "input_widths": [len(p) for p in ins], "roots": roots, "pad": pad}


def template_bits(t: Any, K: Canon) -> tuple[list[int], dict]:
    """The canonical bit of each of the template's output port bits (`_walk`), in port order, its inputs read as nodes
    ``0 ..`` in port order; and what it took."""
    val, rec = _walk(t, K.inputs, 1, K.AND)
    roots, pad = rec.pop("roots"), rec.pop("pad")
    held = [c for c in pad if val[c]]
    if held:
        raise Refused("ports", f"column {held[0]} is in a port group's words but no port's bit, and is not zero "
                               f"({len(held)} such)")
    return [val[c] or ZERO for c in roots], rec


def _lowered(target: Any, C: Any = None) -> tuple[Any, list[list], list[list]]:
    from verity_flock import ir_lower as IL

    from verity.circuits.boolean.forms import Circuit

    _prog, root = IL.standalone(target)
    args = list(root.args)
    if len(set(args)) != len(args):
        raise Refused("ports", f"{getattr(target, 'id', target)}: its standalone call repeats an argument leaf")
    C = Circuit() if C is None else C
    ins = [C.input(f"in{a}", IL._width(root.circuit.gate(a).prim)) for a in args]
    words = IL.lower(C, target, ins)
    for j, w in enumerate(words):
        C.output(f"o{j}", w)
    return C, ins, words


def _read_bits(C: Any, g: int, bits: list[int], K: Canon, ev) -> tuple[dict[int, int], int]:
    """Value ``bits`` of ``C``'s table read ``g`` as the canonical bits of the rows the flat layout writes it as
    (`ir_lower._Read`, its index forms' bits by ``ev``), so a template carrying those rows is compared gate for gate; and
    how many rows."""
    from verity_flock import ir_lower as IL

    r = IL._Read(C, g, bits, 0)
    out = {j: 1 if c else ZERO for j, c in r.const_bits.items()}
    if not r.rows:
        return out, 0
    ra: dict[int, list] = {}
    rb: dict[int, list] = {}
    r.fill(ra, rb, lambda f: [("v", ev(f))])
    val = [ZERO] * r.rows

    def side(x) -> int:
        if isinstance(x, np.ndarray):
            return _xor(val[c] for c in x.tolist())
        return _xor(1 if t == "C" else t[1] if isinstance(t, tuple) else val[t] for t in x)

    for c in range(r.rows):
        val[c] = K.AND(side(ra.pop(c)), side(rb.pop(c)))
    out.update((j, val[r.out_col[j]]) for j in r.bits)
    return out, r.rows


def lowering_bits(C: Any, ins: list[list], words: list[list], K: Canon,
                  slots: dict[str, tuple[int, int]] | None = None) -> tuple[list[tuple], dict]:
    """The canonical bit of each of the flat lowering's output bits, its inputs read as nodes ``0 ..`` in order; a table read
    as its rows (`_read_bits`), or, of a table ``slots`` maps to ``(index bits, most value bits)`` at those widths, as a
    lookup slot's word (`Canon.READ`), a library table's bits constant over it constants."""
    from verity_flock import ir_lower as IL

    if C.asserts:
        raise Refused("unsupported", f"the lowering has {len(C.asserts)} assertions, which a template's rows do not carry "
                                     "as gates")
    val: dict[int, int] = {}
    i = 0
    for w in ins:
        for b in w:
            (v,) = tuple(b.s)
            val[v] = K.inputs[i]
            i += 1
    live = sorted(C.live())

    def ev(f) -> int:
        return _xor(val[v] for v in f.s) ^ f.c

    value_bits: dict[int, list[int]] = {}
    for v in live:
        if C.kind[v] == "lookup":
            value_bits.setdefault(C.data[v][0], []).append(C.data[v][1])
    reads: dict[int, dict[int, int]] = {}
    consts: dict[tuple, dict[int, int]] = {}
    ands = read_rows = slot_reads = index_bits = 0
    for v in live:
        if C.kind[v] == "and":
            a, b = C.data[v]
            val[v] = K.AND(ev(a), ev(b))
            ands += 1
        elif C.kind[v] == "lookup":
            g, j = C.data[v]
            if g not in reads:
                table, idx, width = C.lookups[g]
                slot = (slots or {}).get(table)
                if slot is not None and slot[0] == len(idx) and width <= slot[1]:
                    word = K.READ(table, [ev(f) for f in idx], slot[1])
                    reads[g] = {j2: word[j2] for j2 in value_bits[g]}
                    ck = (table, tuple(value_bits[g]))
                    if ck not in consts:
                        consts[ck] = IL.table_const_bits(table, value_bits[g])
                    reads[g].update((j2, 1 if c else ZERO) for j2, c in consts[ck].items())
                    slot_reads += 1
                    index_bits += len(idx)
                else:
                    reads[g], n = _read_bits(C, g, sorted(value_bits[g]), K, ev)
                    read_rows += n
            val[v] = reads[g][j]
        elif C.kind[v] != "in":
            raise Refused("unsupported", f"the lowering's bit {v} is a {C.kind[v]}")
    return [ev(b) for w in words for b in w], {"live_ands": ands + read_rows, "reads": len(reads),
                                                 **({"slot_reads": slot_reads, "index_bits": index_bits} if slots else {}),
                                                 "input_ports": len(ins), "input_bits": sum(map(len, ins)),
                                                 "output_bits": sum(map(len, words))}


def _lanes(n: int, exhaustive: bool, T: int, seed: int) -> list[int]:
    if exhaustive:
        t = np.arange(T, dtype=np.int64)
        bits = [((t >> j) & 1).astype(np.uint8) for j in range(n)]
    else:
        rng = np.random.default_rng(seed)
        bits = [rng.integers(0, 2, T, dtype=np.uint8) for _ in range(n)]
    return [int.from_bytes(np.packbits(b, bitorder="little").tobytes(), "little") for b in bits]


def evaluate_template(t: Any, lanes: list[int], T: int) -> list[int]:
    """The template's output port bits (`_walk`) on ``lanes`` (a T-bit int per input port bit, in port order)."""
    val, rec = _walk(t, lanes, (1 << T) - 1, operator.and_)
    return [val[c] or 0 for c in rec["roots"]]


def evaluate_lowering(C: Any, ins: list[list], words: list[list], lanes: list[int], T: int) -> list[int]:
    it = iter(lanes)
    vals = {nm: [next(it) for _ in bits] for nm, bits in C.inputs.items()}
    outs, _fail = C.evaluate(vals, T)
    return [x for nm in C.outputs for x in outs[nm]]


def error_bound(rec: dict) -> dict:
    """The bound on a wrong verdict, over the check's own coins (the labels): ``M`` counts the forms whose equality it tests
    (each AND's two operands, their sum and its result; the output bits; each slot read's index bits on the lowering's
    side, the template's being its rows'; the empty form), at most 4 · (template evaluated rows + lowering live ANDs) +
    output bits + index bits + 1, and two distinct ones share a fingerprint with probability at most M² / 2^513."""
    M = (4 * (rec["template"]["evaluated_rows"] + rec["lowering"]["live_ands"]) + rec["lowering"]["output_bits"]
         + rec["lowering"].get("index_bits", 0) + 1)
    return {"M": M, "log2": round(2 * math.log2(M) - 513, 1), "statement": "Pr[wrong verdict] <= M^2 / 2^513"}


def check(target: Any, tmpl: Any, *, exhaustive_bits: int = EXHAUSTIVE_BITS, vectors: int = VECTORS,
          seed: int = CK.Options.seed, label: str | None = None) -> dict:
    """``tmpl`` (a `verity_flock.tmpl.Tmpl`, or its text) held to ``target``'s flat lowering: a result as `checks.check_definition` gives
    one (``failures``, ``warnings``), with ``structural``: the verdict (``syntactic``, ``exhaustive``, ``random`` or
    ``refused``), the counts on both sides, the bound on a wrong verdict (`error_bound`), and why the forms differ when they
    do."""
    from verity.circuits.evaluation.reference import resolve

    t = resolve(target)
    return _check(tmpl, lambda: _lowered(t), label or t.id, {"definition": t.id}, ("the Definition's", "input leaves"),
                  exhaustive_bits, vectors, seed)


def check_unit(definition: Any, in_rows: Sequence[tuple[str, Sequence[str]]], tmpl: Any, *, public: Sequence[str] = (),
               name: str = "unit", exhaustive_bits: int = EXHAUSTIVE_BITS, vectors: int = VECTORS,
               seed: int = CK.Options.seed) -> dict:
    """``tmpl`` held, as `check` holds it to a Definition's lowering, to the flat proof block `class_statement.unit_lowering`
    lays ``definition`` out as on the u16 row ports ``in_rows`` (``public`` its public ports): `class_statement.unit_circuit`
    on `class_statement.unit_ports`, input ``in{j}`` and output ``out{j}`` the template's ports in order. So a structural
    stager's proof block (`UnitStager`'s ``structural``, `struct_lower` on those ports) is held to the proof block the flat
    stager lays out."""
    from verity_flock import class_statement as CS

    from verity.circuits.evaluation.reference import resolve

    t = resolve(definition)

    def flat_unit():
        ports, _out_words, _row_words = CS.unit_ports(t, name, in_rows, public)
        C = CS.unit_circuit(t, ports)
        return C, [C.inputs[f"in{j}"] for j in range(len(ports.ins))], [C.outputs[f"out{j}"] for j in range(len(ports.outs))]

    return _check(tmpl, flat_unit, name, {"definition": t.id, "unit": name}, ("the unit's", "input ports"), exhaustive_bits,
                  vectors, seed)


class _Template:
    """The template side of a check: its counts, its inputs, its canonical output bits and its evaluation."""

    slots = None

    def __init__(self, tmpl: Any) -> None:
        from verity_flock import tmpl as TM

        if not isinstance(tmpl, TM.Tmpl):
            try:
                tmpl = TM.parse(tmpl)
            except (AssertionError, IndexError, ValueError) as e:
                raise Refused("malformed", f"not a {TM.FORMAT} text: {type(e).__name__} {e}") from None
        self.t = tmpl
        self.inputs = sum(map(len, ports(tmpl.groups)[0]))

    def counts(self) -> dict:
        t = self.t
        return {"defs": len(t.defs), "template_rows": sum(len(d.rows) for d in t.defs),
                "template_calls": sum(len(d.calls) for d in t.defs)}

    def bits(self, K: Canon) -> tuple[list[int], dict]:
        return template_bits(self.t, K)

    def evaluate(self, lanes: list[int], T: int) -> list[int]:
        return evaluate_template(self.t, lanes, T)


def _check(tmpl: Any, reference, name: str, rec: dict, theirs: tuple[str, str], exhaustive_bits: int, vectors: int,
           seed: int) -> dict:
    """`check`'s verdict on ``tmpl`` (a template, its text, or a function making a side as `_Template` is one) against
    ``reference()``'s ``(C, input words, output words)``, built once the template's side is read and its unroll dropped;
    ``theirs`` names the reference's side and its inputs in a refusal."""
    F = CK.Findings()
    rec = {**rec, "labels": {"source": "os.urandom", "bits": 512}}
    try:
        side = tmpl() if callable(tmpl) else _Template(tmpl)
        rec |= side.counts()
        K = Canon(side.inputs)
        mine, rec["template"] = side.bits(K)
        widths = rec["template"].pop("input_widths")
        C, ins, words = reference()
        if widths != [len(w) for w in ins]:
            raise Refused("ports", f"input ports of widths {widths}, {theirs[0]} {theirs[1]} {[len(w) for w in ins]}")
        bits, rec["lowering"] = lowering_bits(C, ins, words, K, side.slots)
        if len(mine) != len(bits):
            raise Refused("ports", f"{len(mine)} output bits, {theirs[0]} {len(bits)}")
    except Refused as e:
        F.fail("structural", e.code, name, str(e))
        rec["verdict"] = "refused"
        return {"failures": F.failures, "warnings": F.warnings, "structural": rec}
    rec["canonical_nodes"] = K.next
    rec["error_bound"] = error_bound(rec)
    differ = [j for j, (a, b) in enumerate(zip(mine, bits)) if a != b]
    if not differ:
        rec["verdict"] = "syntactic"
        return {"failures": F.failures, "warnings": F.warnings, "structural": rec}
    rec["differing_output_bits"] = len(differ)
    d = differ[0]
    rec["why_not_syntactic"] = (f"{len(differ)} of {len(mine)} output bits are other forms (the first is bit {d}): not one "
                                f"circuit up to the rewrites the forms absorb")
    n = sum(map(len, ins))
    exhaustive = n <= exhaustive_bits
    T = 1 << n if exhaustive else vectors
    lanes = _lanes(n, exhaustive, T, seed)
    got = side.evaluate(lanes, T)
    want = evaluate_lowering(C, ins, words, lanes, T)
    bad = [j for j, (a, b) in enumerate(zip(got, want)) if a != b]
    rec["semantic"] = {"inputs": n, "exhaustive": exhaustive, "vectors": T, **({} if exhaustive else {"seed": seed}),
                       "mismatched_output_bits": len(bad)}
    if bad:
        j = bad[0]
        lane = ((got[j] ^ want[j]) & -(got[j] ^ want[j])).bit_length() - 1
        F.fail("structural", "mismatch", name, f"output bit {j} differs on {'input' if exhaustive else 'random vector'} "
               f"{lane} ({len(bad)} of {len(got)} output bits differ on some of the {T} vectors)",
               output_bits=len(bad), vectors=T, exhaustive=exhaustive)
        rec["verdict"] = "refused"
    elif exhaustive:
        rec["verdict"] = "exhaustive"
    else:
        rec["verdict"] = "random"
        F.warn("structural", "random-only", name, len(differ),
               f"output bits whose forms differ, equal on {T} random vectors (seed {seed}) only: a difference on a "
               f"fraction ε of the {n} input bits' values passes with probability (1 - ε)^{T}")
    return {"failures": F.failures, "warnings": F.warnings, "structural": rec}


def check_file(target: Any, path: Path, **kw) -> dict:
    """`check` on the template at ``path``, or, a directory, `structural_slots.check` on the split it holds."""
    if Path(path).is_dir():
        from . import structural_slots as SS

        return SS.check(target, path, **kw)
    return check(target, Path(path).read_text(), **kw)


__all__ = [
    "EXHAUSTIVE_BITS",
    "VECTORS",
    "Canon",
    "Refused",
    "check",
    "check_file",
    "check_unit",
    "error_bound",
    "evaluate_lowering",
    "evaluate_template",
    "lowering_bits",
    "ports",
    "template_bits",
]
