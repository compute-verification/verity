"""``circuit-check --structural UNIT.tmpl TARGET``: a ``flock-ir-tmpl`` template (`verity_flock.tmpl`) accepted as the Flock
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

A form is held as its fingerprint, the XOR of its nodes' labels, each node's label 512 bits the check draws from the OS's
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

Refused by name, since neither comparison covers them: a template one of whose rows reads a column at or past its own that
is no input and not the constant (so some column is not a function of the inputs), whose input columns' rows are not their
own ``(c)·(c)``, whose port groups overlap, reach the constant or are not the Definition's input leaves and output bits, or
hold a column of their words that is no port's bit at other than zero; and a Definition whose lowering asserts or reads a
table, which rows carry as more than gates.
"""
from __future__ import annotations

import math
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
        val[c] = K.AND(_xor(val[r] for r in a), _xor(val[r] for r in b))
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


def _lowered(target: Any, C: Any = None) -> tuple[Any, list[list], list[list]]:
    from verity_flock import ir_lower as IL

    from verity.ml.circuits.boolean.forms import Circuit

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
        return _xor(val[v] for v in f.s) ^ f.c

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
    return [x for nm in C.outputs for x in outs[nm]]


def error_bound(rec: dict) -> dict:
    """The bound on a wrong verdict, over the check's own coins (the labels): ``M`` counts the forms whose equality it tests
    (each AND's two operands, their sum and its result; the output bits; the empty form), at most 4 · (template live rows +
    lowering live ANDs) + output bits + 1, and two distinct ones share a fingerprint with probability at most M² / 2^513."""
    M = 4 * (rec["template"]["live_rows"] + rec["lowering"]["live_ands"]) + rec["lowering"]["output_bits"] + 1
    return {"M": M, "log2": round(2 * math.log2(M) - 513, 1), "statement": "Pr[wrong verdict] <= M^2 / 2^513"}


def check(target: Any, tmpl: Any, *, exhaustive_bits: int = EXHAUSTIVE_BITS, vectors: int = VECTORS,
          seed: int = CK.Options.seed, label: str | None = None) -> dict:
    """``tmpl`` (a `verity_flock.tmpl.Tmpl`, or its text) held to ``target``'s flat lowering: a result as `checks.check_definition` gives
    one (``failures``, ``warnings``), with ``structural``: the verdict (``syntactic``, ``exhaustive``, ``random`` or
    ``refused``), the counts on both sides, the bound on a wrong verdict (`error_bound`), and why the forms differ when they
    do."""
    from verity.ml.circuits.evaluation.reference import resolve

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

    from verity.ml.circuits.evaluation.reference import resolve

    t = resolve(definition)

    def flat_unit():
        ports, _out_words, _row_words = CS.unit_ports(t, name, in_rows, public)
        C = CS.unit_circuit(t, ports)
        return C, [C.inputs[f"in{j}"] for j in range(len(ports.ins))], [C.outputs[f"out{j}"] for j in range(len(ports.outs))]

    return _check(tmpl, flat_unit, name, {"definition": t.id, "unit": name}, ("the unit's", "input ports"), exhaustive_bits,
                  vectors, seed)


def _check(tmpl: Any, reference, name: str, rec: dict, theirs: tuple[str, str], exhaustive_bits: int, vectors: int,
           seed: int) -> dict:
    """`check`'s verdict on ``tmpl`` against ``reference()``'s ``(C, input words, output words)``, built once the
    template's side is read and its unroll dropped; ``theirs`` names the reference's side and its inputs in a refusal."""
    from verity_flock import tmpl as TM

    F = CK.Findings()
    rec = {**rec, "labels": {"source": "os.urandom", "bits": 512}}
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
        K = Canon(sum(map(len, ports(flat.head[4:])[0])))
        mine, rec["template"] = template_bits(flat, K)
        del flat
        widths = rec["template"].pop("input_widths")
        C, ins, words = reference()
        if widths != [len(w) for w in ins]:
            raise Refused("ports", f"input ports of widths {widths}, {theirs[0]} {theirs[1]} {[len(w) for w in ins]}")
        bits, rec["lowering"] = lowering_bits(C, ins, words, K)
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
    flat = TM.unroll(tmpl)
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
    "check_unit",
    "error_bound",
    "evaluate_lowering",
    "evaluate_template",
    "lowering_bits",
    "ports",
    "template_bits",
]
