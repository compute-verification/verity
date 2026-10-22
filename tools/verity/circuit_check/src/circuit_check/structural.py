"""``circuit-check --structural UNIT.tmpl TARGET``: a ``flock-ir-tmpl/v1`` unit (`verity_flock.tmpl`) accepted as the C-Flock
lowering of the Definition TARGET, which `verity_flock.ir_lower.lower` lowers flat.

**Syntactic.** Both sides are read into one table of canonical gates: a bit is a GF(2) form over the inputs and the table's
AND nodes, and an AND of two forms is one node, keyed by its operands with their constants taken out and its pair chosen
from the three its operands span, so a gate is the same node whichever way round, or through whichever of the three pairs,
it is written. The template's rows are its unroll (`tmpl.unroll`, which mirrors ``Flock.Tmpl.unroll``) in column order:
an input column is its input, the constant column one, and any other row the AND of its two forms; the lowering's ANDs are
`forms.Circuit`'s. The template is its lowering's gates when each of its output bits is the same form as the lowering's.
That holds exactly when the two are one circuit up to XORs moved across rows, committed forms (a row ``A·1``, which the
template adds past a call's support), constants folded or carried as inputs, dead and duplicated rows, and the order of an
AND's operands, each of which leaves every bit's value as it was: so it proves, by construction, that the template computes
what the lowering computes on every input. It is not complete: one function written as two AND structures is not matched.

A form is interned, so two forms are one exactly when their ids are, with no premise on a hash: as the one text its set of
nodes decides (`Canon`), its nodes in ascending order or, when that is longer, its bitmap, so a hash unit's state bits,
forms of about half the unit's nodes each, take a bit per node they span.

**Semantic.** When the forms differ (a builder that chose a different structure for a constant it was given as an input, say),
the report says which output bits differ and both are evaluated, bit-sliced: on every input when the template has at most
`EXHAUSTIVE_BITS` input bits, else on ``vectors`` random inputs, which miss a difference on a fraction ε of the inputs with
probability (1 − ε)^vectors. An exhaustive pass is a proof too; a random one is a warning, ``structural/random-only``.

Refused by name, since neither comparison covers them: a template one of whose rows reads a column at or past its own that
is no input and not the constant (so some column is not a function of the inputs), whose input columns' rows are not their
own ``(c)·(c)``, whose port groups overlap, reach the constant or are not the Definition's input leaves and output bits, or
hold a column of their words that is no port's bit at other than zero; and a Definition whose lowering asserts or reads a
table, which rows carry as more than gates.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import checks as CK

EXHAUSTIVE_BITS = 16
VECTORS = 4096
WORD = 128
ZERO = 0
IDS = (1 << 40) - 1
NODE = np.dtype("<i4")
LIMB = np.dtype("<u8")
_ONE = np.uint64(1)


class Canon:
    """The canonical gates both sides are read into. A bit is one int, its form's id shifted up one and its constant in
    bit 0; form 0 is the empty form. Nodes ``0 .. inputs - 1`` are the inputs (`inputs`, their bits); each AND node takes
    the next.

    A form, a set of nodes, is interned as one text (`forms`, by id), which the set alone decides: its nodes in ascending
    order, 4 bytes each, or, when that is longer than its bitmap, the bitmap's first 64-node word index (4 bytes) and its
    words from the first nonzero one to the last, 8 bytes each (`dense` marks those). The table finds a text by its hash and
    compares it byte for byte, so two forms are equal exactly when their ids are."""

    def __init__(self, inputs: int) -> None:
        self.forms: list[bytes] = [b""]
        self.dense = bytearray(1)
        self.ids: tuple[dict[bytes, int], dict[bytes, int]] = ({b"": 0}, {})
        self.ands: dict[int, int] = {}
        self.next = 0
        self.inputs = [self._node() << 1 for _ in range(inputs)]

    def _id(self, text: bytes, dense: int) -> int:
        ids = self.ids[dense]
        f = ids.get(text)
        if f is None:
            f = ids[text] = len(self.forms)
            self.forms.append(text)
            self.dense.append(dense)
        return f

    def _node(self) -> int:
        self.next += 1
        return self._id((self.next - 1).to_bytes(4, "little"), 0)

    def _sparse(self, s: np.ndarray) -> int:
        """The form of the ascending nodes ``s``."""
        if len(s) and 2 * (int(s[-1] >> 6) - int(s[0] >> 6) + 1) + 1 < len(s):
            lo = int(s[0]) >> 6
            w = np.zeros(int(s[-1] >> 6) - lo + 1, LIMB)
            d = s - (lo << 6)
            np.bitwise_xor.at(w, d >> 6, _ONE << (d & 63).astype(LIMB))
            return self._id(lo.to_bytes(4, "little") + w.tobytes(), 1)
        return self._id(s.astype(NODE, copy=False).tobytes(), 0)

    def _words(self, lo: int, w: np.ndarray) -> int:
        """The form of the bitmap ``w`` whose first word is word ``lo``."""
        nz = np.flatnonzero(w)
        if not len(nz):
            return 0
        w, lo = w[nz[0]:nz[-1] + 1], lo + int(nz[0])
        if 2 * len(w) + 1 < int(np.bitwise_count(w).sum()):
            return self._id(lo.to_bytes(4, "little") + w.tobytes(), 1)
        s = np.flatnonzero(np.unpackbits(w.view(np.uint8), bitorder="little")) + (lo << 6)
        return self._id(s.astype(NODE).tobytes(), 0)

    def _bitmap(self, f: int) -> tuple[int, np.ndarray]:
        t = self.forms[f]
        return int.from_bytes(t[:4], "little"), np.frombuffer(t, LIMB, offset=4)

    def nodes(self, bit: int) -> np.ndarray:
        """The nodes of ``bit``'s form, ascending."""
        f = bit >> 1
        if not self.dense[f]:
            return np.frombuffer(self.forms[f], NODE)
        lo, w = self._bitmap(f)
        return np.flatnonzero(np.unpackbits(w.view(np.uint8), bitorder="little")) + (lo << 6)

    def size(self, bit: int) -> int:
        f = bit >> 1
        if self.dense[f]:
            return int(np.bitwise_count(self._bitmap(f)[1]).sum())
        return len(self.forms[f]) // 4

    def _fold(self, ts: list[int]) -> int:
        """The XOR of the forms ``ts``, at least one of them dense."""
        maps = [self._bitmap(t) for t in ts if self.dense[t]]
        s = [np.frombuffer(self.forms[t], NODE) for t in ts if not self.dense[t]]
        s = np.concatenate(s).astype(np.int64) if s else np.zeros(0, np.int64)
        lo = min([m[0] for m in maps] + ([int(s.min()) >> 6] if len(s) else []))
        hi = max([m[0] + len(m[1]) for m in maps] + ([(int(s.max()) >> 6) + 1] if len(s) else []))
        w = np.zeros(hi - lo, LIMB)
        for at, m in maps:
            w[at - lo:at - lo + len(m)] ^= m
        if len(s):
            d = s - (lo << 6)
            np.bitwise_xor.at(w, d >> 6, _ONE << (d & 63).astype(LIMB))
        return self._words(lo, w)

    def _x(self, f: int, g: int) -> int:
        """The form f + g."""
        if f == g:
            return 0
        if not f or not g:
            return f or g
        if self.dense[f] or self.dense[g]:
            return self._fold([f, g])
        a, b = (np.frombuffer(self.forms[t], NODE) for t in (f, g))
        return self._sparse(np.setxor1d(a, b, assume_unique=True))

    def xor(self, a: int, b: int) -> int:
        return self._x(a >> 1, b >> 1) << 1 | ((a ^ b) & 1)

    def xor_all(self, bits) -> int:
        """The XOR of ``bits``: each node in an odd number of their forms."""
        c, ts = 0, []
        for b in bits:
            c ^= b & 1
            if b > 1:
                ts.append(b >> 1)
        if len(ts) < 3:
            return (self._x(*ts) if len(ts) == 2 else ts[0] if ts else 0) << 1 | c
        if any(self.dense[t] for t in ts):
            return self._fold(ts) << 1 | c
        s = np.sort(np.frombuffer(b"".join([self.forms[t] for t in ts]), NODE))
        run = np.flatnonzero(np.diff(s, prepend=-1, append=-1))
        return self._sparse(s[run[:-1][np.diff(run) & 1 == 1]]) << 1 | c

    def AND(self, a: int, b: int) -> int:
        """``a · b``: (f + x)(g + y) = fg + y f + x g + x y. The three forms f, g and f + g have one node, the product of
        the two the first AND of them wrote, and since f (f + g) = f + fg, fg is that node plus the operand it shares
        with another of their pairs: an AND that writes the pair its node was made for, as both sides mostly do, is that
        node alone."""
        x, y = a & 1, b & 1
        f, g = a >> 1, b >> 1
        if not f:
            return b if x else ZERO
        if not g:
            return a if y else ZERO
        if f == g:
            return a if x == y else ZERO
        h = self._x(f, g)
        i, j, _k = sorted((f, g, h))
        v = self.ands.get(i << 40 | j)
        if v is None:
            fg = self._node()
            self.ands[i << 40 | j] = fg << 40 | h
        else:
            n, e = v >> 40, v & IDS
            fg = n if e == h else self._x(n, g if e == f else f)
        if y:
            fg = self._x(fg, f)
        if x:
            fg = self._x(fg, g)
        return fg << 1 | (x & y)


class Refused(ValueError):
    """A template neither comparison covers, by name (`code`)."""

    def __init__(self, code: str, what: str) -> None:
        super().__init__(what)
        self.code = code


def ports(groups: Sequence[str]) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """Per input port and per output port of a unit's header (``n_in (col words n bits..).. n_out (col words n bits..)..``),
    its columns, bit 0 first; and the columns of the groups' words that are no port's bit, which a unit holds at zero."""
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


def _live(n: int, row, roots: Sequence[int], skip: set) -> list[int]:
    seen = np.zeros(n, bool)
    stack = [c for c in roots]
    while stack:
        c = stack.pop()
        if seen[c]:
            continue
        seen[c] = True
        if c in skip:
            continue
        a, b = row(c)
        stack.extend(a)
        stack.extend(b)
    return np.flatnonzero(seen).tolist()


def template_bits(flat: Any, K: Canon) -> tuple[list[tuple], dict]:
    """The canonical bit of each of the unrolled template's output port bits, in port order, its inputs read as nodes
    ``0 ..`` in port order; and what it took."""
    k = flat.const
    n = len(flat.na)
    ins, outs, pad = ports(flat.head[4:])
    cols = [c for p in ins for c in p]
    roots = [c for p in outs for c in p]
    if any(c >= k for c in cols + roots + pad):
        raise Refused("ports", f"a port group's words reach the constant's column {k} or past it")
    if len(flat.refs) and not 0 <= int(flat.refs.min()) <= int(flat.refs.max()) < n:
        raise Refused("malformed", f"a row reads a column outside the unit's {n}")
    node = {c: i for i, c in enumerate(cols)}
    for c in cols:
        if flat.row(c) != ((c,), (c,)):
            raise Refused("ports", f"input column {c}'s row is {flat.row(c)}, not its own (c)·(c)")
    free = np.zeros(n, bool)
    free[cols] = True
    free[k] = True
    at = np.repeat(np.arange(n), np.diff(flat.ptr))
    late = np.flatnonzero((flat.refs >= at) & ~free[flat.refs] & ~free[at])
    if len(late):
        raise Refused("order", f"row {at[late[0]]} reads column {flat.refs[late[0]]}, at or past its own")
    val: dict[int, int] = {k: 1, **{c: K.inputs[i] for c, i in node.items()}}
    ands = copies = 0
    live = _live(n, flat.row, roots + pad, set(val))
    for c in live:
        if c in val:
            continue
        a, b = flat.row(c)
        val[c] = K.AND(K.xor_all([val[r] for r in a]), K.xor_all([val[r] for r in b]))
        copy = (k,) in (a, b)
        copies += copy
        ands += bool(a and b) and not copy
    held = [c for c in pad if val[c] != ZERO]
    if held:
        raise Refused("ports", f"column {held[0]} is in a port group's words but no port's bit, and is not zero "
                               f"({len(held)} such)")
    return [val[c] for c in roots], {"rows": n, "live_rows": len(live), "live_ands": ands, "live_copies": copies,
                                     "input_ports": len(ins),
                                     "input_bits": len(cols), "output_bits": len(roots),
                                     "input_widths": [len(p) for p in ins]}


def _lowered(target: Any) -> tuple[Any, list[list], list[list]]:
    from verity_flock import ir_lower as IL

    from verity.ml.circuits.boolean.forms import Circuit

    _prog, root = IL.standalone(target)
    args = list(root.args)
    if len(set(args)) != len(args):
        raise Refused("ports", f"{getattr(target, 'id', target)}: its standalone call repeats an argument leaf")
    C = Circuit()
    ins = [C.input(f"in{a}", IL._width(root.circuit.gate(a).prim)) for a in args]
    words = IL.lower(C, target, ins)
    for j, w in enumerate(words):
        C.output(f"o{j}", w)
    return C, ins, words


def lowering_bits(C: Any, ins: list[list], words: list[list], K: Canon) -> tuple[list[tuple], dict]:
    """The canonical bit of each of the flat lowering's output bits, its inputs read as nodes ``0 ..`` in order."""
    if C.asserts or C.lookups:
        raise Refused("unsupported", f"the lowering has {len(C.asserts)} assertions and {len(C.lookups)} table reads, "
                                     "which a template's rows do not carry as gates")
    val: dict[int, int] = {}
    i = 0
    for w in ins:
        for b in w:
            (v,) = tuple(b.s)
            val[v] = K.inputs[i]
            i += 1
    live = sorted(C.live())

    def ev(f) -> int:
        return K.xor_all([val[v] for v in f.s]) ^ f.c

    ands = 0
    for v in live:
        if C.kind[v] == "and":
            a, b = C.data[v]
            val[v] = K.AND(ev(a), ev(b))
            ands += 1
        elif C.kind[v] != "in":
            raise Refused("unsupported", f"the lowering's bit {v} is a {C.kind[v]}")
    return [ev(b) for w in words for b in w], {"live_ands": ands, "input_ports": len(ins), "input_bits": sum(map(len, ins)),
                                                 "output_bits": sum(map(len, words))}


def _lanes(n: int, exhaustive: bool, T: int, seed: int) -> list[int]:
    if exhaustive:
        t = np.arange(T, dtype=np.int64)
        bits = [((t >> j) & 1).astype(np.uint8) for j in range(n)]
    else:
        rng = np.random.default_rng(seed)
        bits = [rng.integers(0, 2, T, dtype=np.uint8) for _ in range(n)]
    return [int.from_bytes(np.packbits(b, bitorder="little").tobytes(), "little") for b in bits]


def evaluate_template(flat: Any, lanes: list[int], T: int) -> list[int]:
    """The unrolled template's output port bits on ``lanes`` (a T-bit int per input port bit, in port order)."""
    k = flat.const
    ins, outs, _pad = ports(flat.head[4:])
    cols = [c for p in ins for c in p]
    roots = [c for p in outs for c in p]
    z = {k: (1 << T) - 1, **dict(zip(cols, lanes))}
    for c in _live(len(flat.na), flat.row, roots, set(z)):
        if c in z:
            continue
        a, b = flat.row(c)
        x = y = 0
        for r in a:
            x ^= z[r]
        for r in b:
            y ^= z[r]
        z[c] = x & y
    return [z[c] for c in roots]


def evaluate_lowering(C: Any, ins: list[list], words: list[list], lanes: list[int], T: int) -> list[int]:
    it = iter(lanes)
    vals = {nm: [next(it) for _ in bits] for nm, bits in C.inputs.items()}
    outs, _fail = C.evaluate(vals, T)
    return [x for j in range(len(words)) for x in outs[f"o{j}"]]


def check(target: Any, tmpl: Any, *, exhaustive_bits: int = EXHAUSTIVE_BITS, vectors: int = VECTORS,
          seed: int = CK.Options.seed, label: str | None = None) -> dict:
    """``tmpl`` (a `verity_flock.tmpl.Tmpl`, or its text) held to ``target``'s flat lowering: a result as `checks.check_definition` gives
    one (``failures``, ``warnings``), with ``structural``: the verdict (``syntactic``, ``exhaustive``, ``random`` or
    ``refused``), the counts on both sides, and why the forms differ when they do."""
    from verity_flock import tmpl as TM

    from verity.ml.circuits.evaluation.reference import resolve

    t = resolve(target)
    name = label or t.id
    F = CK.Findings()
    rec: dict[str, Any] = {"definition": t.id}
    try:
        if not isinstance(tmpl, TM.Tmpl):
            try:
                tmpl = TM.parse(tmpl)
            except (AssertionError, IndexError, ValueError) as e:
                raise Refused("malformed", f"not a {TM.FORMAT} text: {type(e).__name__} {e}") from None
        rec |= {"defs": len(tmpl.defs), "template_rows": sum(len(d.rows) for d in tmpl.defs),
                "template_calls": sum(len(d.calls) for d in tmpl.defs)}
        try:
            flat = TM.unroll(tmpl)
        except (AssertionError, IndexError, ValueError) as e:
            raise Refused("malformed", f"its unroll: {type(e).__name__} {e}") from None
        C, ins, words = _lowered(t)
        K = Canon(sum(map(len, ins)))
        mine, rec["template"] = template_bits(flat, K)
        widths = rec["template"].pop("input_widths")
        if widths != [len(w) for w in ins]:
            raise Refused("ports", f"input ports of widths {widths}, the Definition's input leaves {[len(w) for w in ins]}")
        theirs, rec["lowering"] = lowering_bits(C, ins, words, K)
        if len(mine) != len(theirs):
            raise Refused("ports", f"{len(mine)} output bits, the Definition's {len(theirs)}")
    except Refused as e:
        F.fail("structural", e.code, name, str(e))
        rec["verdict"] = "refused"
        return {"failures": F.failures, "warnings": F.warnings, "structural": rec}
    rec["canonical_nodes"] = K.next
    rec["forms"] = len(K.forms)
    differ = [j for j, (a, b) in enumerate(zip(mine, theirs)) if a != b]
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
    got = evaluate_template(flat, lanes, T)
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
    return check(target, Path(path).read_text(), **kw)


__all__ = [
    "EXHAUSTIVE_BITS",
    "VECTORS",
    "Canon",
    "Refused",
    "check",
    "check_file",
    "evaluate_lowering",
    "evaluate_template",
    "lowering_bits",
    "ports",
    "template_bits",
]
