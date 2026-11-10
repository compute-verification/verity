"""Scratch, never for landing: the rule that a reader's port is compared in its word view only where no leaf of its
Definition's return resolves to a leaf of that port, inverted or not, through a callee or not, run on the Definitions
decoded from the small served phase's Program (tests/program/predict/test_window_circuit.py), not read from their
source.

Per template and port: the return leaves that resolve to a leaf of the port, followed leaf by leaf through callees and
through `Not_v1` (`exact`), and what one pass over each Definition's reference runs says, callees first, never a leaf
(`by runs`, the cost a verifier may pay; it may only say more). Then every wire under the rule as built and under each
reading of the second case, and the form's own summary of the record with the two wires taken: what it charges at
the embedding read's `tok`. Run from integrations/vllm:  uv run python served_port_closed.py"""
import collections
import json
import random
import re
import sys
import time

sys.path[:0] = [".", "../../verity/core/protocols/network/nci/lean/scripts"]
import record_difftest_vectors as g
from tests.program.predict import test_window_circuit as T
from verity.ml.circuits import boolean as B
from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.layout import Scope, _arg_ref
from verity.ml.circuits.refs import runs

ref, IF, codec = g.ref, g.IF, g.codec
T0 = time.monotonic()


def said(*a) -> None:
    print(f"[{time.monotonic() - T0:6.0f} s]", *a, flush=True)


def short(tid: str) -> str:
    return tid.split("{")[0]


def shape(ws) -> str:
    return f"{len(ws)}x{sorted(set(ws))}"


def source(d, r):
    """What reference `r` of Definition `d`'s body is, from `d` alone: `("port", parameter, leaf, inverted)`, a leaf
    of `d`'s own parameter reached through callees' parameters and returns and through `Not_v1`; else `("gate", id)`,
    the primitive that computes it."""
    scope, inv = Scope(d, 0, None, 0, 0), False
    space, idx, leaf = r
    while True:
        if space == "p":
            if scope.parent is None:
                return ("port", idx, leaf, inv)
            up = _arg_ref(scope, idx, leaf)
            if up is None:  # a scan's carry: the iteration before's returned carry
                scope = Scope(scope.spec, 0, scope.parent, scope.node_idx, scope.member - 1)
                space, idx, leaf = scope.spec.body.ret[leaf]
                continue
            (space, idx, leaf), scope = up, scope.parent
            continue
        node = scope.spec.body.nodes[idx]
        fn = node.fn
        if node.form == "primitive":
            if fn.id != "Not_v1":
                return ("gate", fn.id)
            inv = not inv
            space, idx, leaf = node.args[0].refs[0]
            continue
        if node.form == "call":
            scope = Scope(fn, 0, scope, idx, 0)
            space, idx, leaf = fn.body.ret[leaf]
            continue
        if node.form == "batch":
            m, j = divmod(leaf, fn.out_leaves)
            if isinstance(fn, PrimitiveDefinition):
                if fn.id != "Not_v1":
                    return ("gate", fn.id)
                inv = not inv
                space, idx, leaf = _arg_ref(Scope(fn, 0, scope, idx, m), 0, 0)
                continue
            scope = Scope(fn, 0, scope, idx, m)
            space, idx, leaf = fn.body.ret[j]
            continue
        c = node.out.fields[0].leaves
        if leaf < c:
            m, j = node.n - 1, leaf
        else:
            y = node.out.fields[1].elem.leaves
            m, j = divmod(leaf - c, y)
            j += c
        scope = Scope(fn, 0, scope, idx, m)
        space, idx, leaf = fn.body.ret[j]


def exact(d) -> dict[str, collections.Counter]:
    """Per parameter of `d` some return leaf resolves to: how many do, as they are and inverted."""
    params, ret = d.signature()
    out: dict[str, collections.Counter] = {}
    for j in range(ret.leaves):
        s = source(d, d.body.ret[j])
        if s[0] == "port":
            out.setdefault(params[s[1]][0], collections.Counter())["inverted" if s[3] else "as it is"] += 1
    return out


_RUNS: dict[int, frozenset[int]] = {}


def by_runs(d) -> frozenset[int]:
    """The parameters of `d` a return leaf may resolve to, read from reference runs alone, callees first: a run of a
    parameter is that parameter; a run of a node is every argument of the node that its callee's return may resolve
    to, by the same rule; `Not_v1` passes its one argument on and no other primitive does."""
    if isinstance(d, PrimitiveDefinition):
        return frozenset(range(len(d.params))) if d.id == "Not_v1" else frozenset()
    if id(d) not in _RUNS:
        reach: list[frozenset[int]] = []

        def of(refs) -> frozenset[int]:
            got: set[int] = set()
            for space, idx, _, _ in runs(refs):
                got |= {idx} if space == "p" else reach[idx]
            return frozenset(got)

        for n in d.body.nodes:
            reach.append(frozenset().union(*[of(n.args[i].refs) for i in by_runs(n.fn)]))
        _RUNS[id(d)] = of(d.body.ret)
    return _RUNS[id(d)]


BOOLEAN_ID = re.compile(r"^((And|Xor)([2-9]|[1-9][0-9]+)_v1|Not_v1|Const1\[0x[01]\]_v1|Input1_v1|RegisteredInput1_v1)$")


def boolean_by_id(d) -> bool:
    """`boolean.is_boolean`, its primitives known by id: on a decoded Definition `is_boolean` knows a gate only where
    it is the module's own object, which a decoded gate is not."""
    params, ret = d.signature()
    if not all(t.width_set() <= {1} for t in (ret, *(t for _, t in params))):
        return False
    seen, stack = set(), [d]
    while stack:
        x = stack.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        for n in x.body.nodes:
            if isinstance(n.fn, PrimitiveDefinition):
                if not BOOLEAN_ID.fullmatch(n.fn.id):
                    return False
            else:
                stack.append(n.fn)
    return True


def takes(row, out) -> bool:
    return len(set(row)) == 1 and len(set(out)) == 1 and row[0] % out[0] == 0


def study(label: str, ts: list, wires: list) -> dict:
    """`ts` the record's templates (`ref.Template`), `wires` each `(reader, port, writer, out)`."""
    by = {t.tid: t for t in ts}
    readers = {r for r, *_ in wires}
    boolean = {t.tid: boolean_by_id(t.fn) for t in ts}
    theirs = {t.tid: B.is_boolean(t.fn) for t in ts}
    hit = {t.tid: exact(t.fn) for t in ts}
    coarse = {t.tid: {t.fn.signature()[0][i][0] for i in by_runs(t.fn)} for t in ts}
    assert all(set(hit[t.tid]) <= coarse[t.tid] for t in ts), "the runs miss a pass-through the leaves show"
    said(f"== {label}: {len(ts)} templates, {sum(t.units for t in ts)} units, {len(wires)} wires")
    print("templates (B Boolean, its primitives known by id; b Boolean to boolean.is_boolean; R reads a wire), then "
          "each port a return leaf resolves to:")
    for t in ts:
        ports = [f"{p} ({', '.join(f'{n} {how}' for how, n in sorted(c.items()))})" for p, c in sorted(hit[t.tid].items())]
        more = sorted(coarse[t.tid] - set(hit[t.tid]))
        print(f"  {'B' if boolean[t.tid] else '-'}{'b' if theirs[t.tid] else '-'}{'R' if t.tid in readers else '-'} "
              f"{short(t.tid)} x{t.units}: "
              f"exact: {'; '.join(ports) or 'none'}" + (f" | by runs, also: {', '.join(more)}" if more else ""))
        if boolean[t.tid]:
            for name, ty in t.fn.signature()[0]:
                closed = name not in hit[t.tid]
                print(f"       {name}: leaves {shape(t.widths[name])}, elements {shape(B.elements(ty))}, "
                      + ("no return leaf resolves to it" if closed else "PASSED ON")
                      + ("" if closed == (name not in coarse[t.tid]) else " (by runs: may be passed on)"))

    def row(view: str, tid: str, port: str) -> tuple[tuple[int, ...], bool]:
        """The reader's leaves under `view`, and whether the port is taken whatever the writer's."""
        t = by[tid]
        ty = dict(t.fn.signature()[0])[port]
        closed = port not in (coarse[tid] if view.endswith("by runs") else hit[tid])
        bits = set(t.widths[port]) == {1}
        if view.startswith("word view") and boolean[tid] and closed:
            return tuple(B.elements(ty)), False
        if view.startswith("any writer, a port of bits") and bits and closed:
            return t.widths[port], True
        if view.startswith("any writer, any port") and closed:
            return t.widths[port], True
        return t.widths[port], False

    views = ("as built", "word view, a Boolean reader's closed port", "word view, a Boolean reader's closed port, by runs",
             "any writer, a port of bits, closed", "any writer, a port of bits, closed, by runs", "any writer, any port, closed")
    first = None
    for v in views:
        got = []
        for r, p, w, o in wires:
            leaves, free = row(v, r, p)
            got.append(free or takes(leaves, by[w].widths[o]))
        first = first or got
        moved = collections.Counter(
            ("admitted -> REFUSED" if was else "refused -> admitted", short(w), o, shape(by[w].widths[o]), short(r), p,
             shape(row(v, r, p)[0])) for (r, p, w, o), was, now in zip(wires, first, got, strict=True) if was != now)
        print(f"{v}: {sum(got)} admitted, {len(got) - sum(got)} refused")
        for (how, w, o, ow, r, p, rw), n in sorted(moved.items()):
            print(f"   {how}: {w}.{o} {ow} -> {r}.{p} {rw}" + (f" ({n} wires)" if n > 1 else ""))
        for (r, p, w, o), now in zip(wires, got, strict=True):
            if not now:
                print(f"   refused: {short(w)}.{o} {shape(by[w].widths[o])} -> {short(r)}.{p} {shape(row(v, r, p)[0])}")
    return {"boolean": boolean, "hit": hit}


c = T._circuit(T.WINDOW)
program, _ = c.program()
desc = codec.encode_program(program)
text, digest = codec.canonical_json(desc).decode(), codec.program_digest(desc)
doc = json.loads(json.dumps({"format": IF.CIRCUIT, "name": c.name, "program": digest, **c.description(),
                             **({"blocks": c.blocks_record()} if c.blocks is not None else {})}))
record = IF.circuit_bytes(doc).decode()
said(f"the served Program's canonical bytes: {len(text)}; decoding them")
templates = ref.templates_of(ref.reference.desc_program(text), doc, digest)
said("decoded; every Definition below is the decoded one")
found = study("the small served phase's circuit, its Definitions decoded",
              templates, [(w.reader, w.port, w.writer, w.out) for w in c.wires])

def passed_bool(S: int = 2):
    """`record_difftest_vectors.passed_on`, its passing template a Boolean Definition: one parameter of sixteen rows
    of sixteen bits, one output of sixteen bits, leaf `j` its row `j`'s leaf `j`, no gate."""
    from verity.ml.circuits.types import Array, Value
    p = g.Circuit(f"passedbool{S}", S)
    v16, bits = Value(16), Array(16, Value(1))
    gate = [[("op", 16, [(0, 0)])]]
    wide = p.add(g.template("PassWide", [("w", v16)], [v16], gate), [([S, 16], ["slot", "-"], 0)])
    on = p.add(g.template("PassOnBool", [("x", Array(16, bits))], [bits], [("pass", 0, [17 * j for j in range(16)])]),
               [([S], ["slot"], 0)])
    take = p.add(g.template("PassTake", [("v", v16)], [v16], gate), [([S], ["slot"], 0)])
    p.cuts = {on: {"x": 16}}
    p.wires = IF.coalesce([IF.Wire(on, "x", wide, "o0", s, (16,), 16 * s, (1,), (0, 16)) for s in range(S)]
                          + [IF.Wire(take, "v", on, "o0", s, (1,), (s + 1) % S, (0,), (0, 1)) for s in range(S)])
    p.params = {"weights/wide": 16 * S}
    p.reads = IF.coalesce([IF.Read(wide, "w", "weights/wide", u, (1,), u, (0,)) for u in range(16 * S)])
    p.order = p.ordered()
    return p


for label, p in (("the difftest's passed_on (its passing template has a 16-bit gate, so it is not Boolean)", g.passed_on()),
                 ("passed_on, its passing template a Boolean Definition", passed_bool())):
    p_text, p_digest = p.program()
    study(label, ref.templates_of(ref.reference.desc_program(p_text), json.loads(IF.circuit_bytes(p.doc(p_digest))),
                                  p_digest), [(w.reader, w.port, w.writer, w.out) for w in p.wires])

# what the form charges at tok, the two wires taken by the rule as lean states it
rng = random.Random(0)
params = dict(c.registered.params) if c.registered is not None else {}
roots = {q: g._hex(rng) for q in params}
tr = g.trusted(rng, digest, record, {q: (roots[q], n, 1) for q, n in params.items()}, roots)
inst = {"program": text, "record": record, "gamma": g.BIG, "alpha": [g.BIG, 1], "pins": []}
built = ref.leaf_widths


def viewed(ts, wires, cuts):
    for t in ts:
        if found["boolean"][t.tid]:
            for name, ty in t.fn.signature()[0]:
                if name not in found["hit"][t.tid]:
                    t.widths[name] = tuple(B.elements(ty))
    return built(ts, wires, cuts)


ref.leaf_widths = viewed
embed = next(t for t in templates if short(t.tid) == "EmbeddingRead_v1")
rows = [[embed.tid, "tok", u, 0] for u in range(embed.units)]
got = ref.check_detail(inst, tr)
said("the form's reference on the served record, the two wires taken (a policy of no limit, no pin, every parameter "
     "registered: its counts say nothing of a served policy):")
print("  ", {k: got[k] for k in ("error", "message", "blocks", "inputs", "wired_rows", "wired_bits", "cross_wires") if k in got})
kinds = ref.kinds(inst, tr, rows)
said(f"the {len(rows)} rows of EmbeddingRead_v1.tok, {dict(embed.rows and [(q, w) for q, w, _ in embed.rows])['tok']} "
     f"words each, by kind: {dict(collections.Counter(map(str, kinds)))}")
ref.leaf_widths = built
