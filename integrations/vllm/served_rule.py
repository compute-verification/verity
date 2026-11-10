"""Scratch, never for landing: the small served phase's circuit (tests/program/predict/test_window_circuit.py)
through the record form's reference at this checkout, under the closed-row rule as built.

The reference's verdict, word for word; per template what its Definition passes on, by reference parts (the rule,
`ref.passed_on`) and leaf by leaf through callees, batch members and scans with every primitive a gate (`exact`),
and its closed rows; each wire and how the rule takes it (as built, into a closed row, or refused); and each wire's
row bits against its writer's output's bits. Run from integrations/vllm:
    uv run python served_rule.py [--case DIR]   (--case writes the one difftest case for the compiled Lean)."""
import collections
import json
import random
import sys
import time
from pathlib import Path

sys.path[:0] = [".", "../../verity/core/protocols/network/nci/lean/scripts"]
import record_difftest_vectors as g
from tests.program.predict import test_window_circuit as T
from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.layout import Scope, _arg_ref

ref, IF, codec = g.ref, g.IF, g.codec
T0 = time.monotonic()


def said(*a) -> None:
    print(f"[{time.monotonic() - T0:6.0f} s]", *a, flush=True)


def short(tid: str) -> str:
    return tid.split("{")[0]


def shape(ws) -> str:
    return f"{len(ws)}x{sorted(set(ws))}"


def source(d, r):
    """What reference `r` of Definition `d`'s body is, from `d` alone: `("port", parameter, leaf)`, a leaf of `d`'s
    own parameter reached through callees' parameters and returns, batch members and scan iterations; else
    `("gate", id)`, the primitive that computes it. Every primitive is a gate, an inverter included."""
    scope = Scope(d, 0, None, 0, 0)
    space, idx, leaf = r
    while True:
        if space == "p":
            if scope.parent is None:
                return ("port", idx, leaf)
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
            return ("gate", fn.id)
        if node.form == "call":
            scope = Scope(fn, 0, scope, idx, 0)
            space, idx, leaf = fn.body.ret[leaf]
            continue
        if node.form == "batch":
            m, j = divmod(leaf, fn.out_leaves)
            if isinstance(fn, PrimitiveDefinition):
                return ("gate", fn.id)
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


def exact(d) -> collections.Counter:
    """Per parameter of `d`, how many leaves of its return are a leaf of that parameter, by reference."""
    if isinstance(d, PrimitiveDefinition):
        return collections.Counter()
    params, ret = d.signature()
    out: collections.Counter = collections.Counter()
    for j in range(ret.leaves):
        s = source(d, d.body.ret[j])
        if s[0] == "port":
            out[params[s[1]][0]] += 1
    return out


c = T._circuit(T.WINDOW)
program, _ = c.program()
desc = codec.encode_program(program)
text, digest = codec.canonical_json(desc).decode(), codec.program_digest(desc)
doc = json.loads(json.dumps({"format": IF.CIRCUIT, "name": c.name, "program": digest, **c.description(),
                             **({"blocks": c.blocks_record()} if c.blocks is not None else {})}))
record = IF.circuit_bytes(doc).decode()
rng = random.Random(0)
params = dict(c.registered.params) if c.registered is not None else {}
roots = {p: g._hex(rng) for p in params}
tr = g.trusted(rng, digest, record, {p: (roots[p], n, 1) for p, n in params.items()}, roots)
inst = {"program": text, "record": record, "gamma": g.BIG, "alpha": [g.BIG, 1], "pins": []}
said("circuit:", c.name, "| templates", len(c.templates), "| units", sum(n for _, n in c.templates), "| wires",
     len(c.wires), "| Program bytes", len(text))
got = ref.check_detail(inst, tr)
said("THE REFERENCE'S VERDICT, word for word:", json.dumps(got))

decoded = ref.reference.desc_program(text)
templates = ref.templates_of(decoded, doc, digest)
wires, cuts = ref.wires_of(templates, doc, None)
by = {t.tid: t for t in templates}
said("decoded; every Definition below is the decoded one")
print("templates: what each passes on, by parts (the rule) and by leaves (return leaves that are a leaf of the row), "
      "and its closed rows")
for t in templates:
    names = [name for name, _ in t.fn.signature()[0]]
    parts = sorted(names[i] for i in ref.passed_on(t.fn))
    leaves = exact(t.fn)
    assert set(leaves) <= set(parts), (t.tid, "the parts miss a row the leaves show passed on")
    ret = t.fn.signature()[1].leaves
    print(f"  {short(t.tid)} x{t.units}: return leaves {ret} | by parts: {', '.join(parts) or 'none'} | by leaves: "
          + (", ".join(f"{p} {n}" for p, n in sorted(leaves.items())) or "none")
          + f" | closed: {', '.join(t.closed) or 'none'}")

how: collections.Counter = collections.Counter()
kinds: collections.Counter = collections.Counter()
refused = []
for w in wires:
    row, out = by[w["reader"]].widths[w["port"]], by[w["writer"]].widths[w["out"]]
    built = len(set(row)) == 1 and len(set(out)) == 1 and row[0] % out[0] == 0
    shut = w["port"] in by[w["reader"]].closed
    try:
        ref.leaf_widths(decoded, templates, [w], cuts)
        taken = True
    except ref.Malformed:
        taken = False
    assert taken == (built or shut)
    way = "as built" if built else "into a closed row" if shut else "REFUSED"
    how[way] += 1
    key = (way, f"{short(w['writer'])}.{w['out']} {shape(out)}", f"{short(w['reader'])}.{w['port']} {shape(row)}",
           "also closed" if built and shut else "")
    kinds[key] += 1
    if not taken:
        refused.append(key)
print(f"the rule as built: {len(wires)} wires, {how['as built']} taken as built, {how['into a closed row']} taken into "
      f"a closed row, {how['REFUSED']} refused")
for (way, writer, reader, also), n in sorted(kinds.items()):
    if way != "as built":
        print(f"   {way}: {writer} -> {reader} ({n} wires)")
print("wires taken as built whose reader's row is also closed:",
      sum(n for (way, _, _, also), n in kinds.items() if way == "as built" and also))

# each wire's row bits against its writer's output's bits: a cut port's row is its leaves over the cut
short_rows: collections.Counter = collections.Counter()
for w in wires:
    row, out = by[w["reader"]].widths[w["port"]], by[w["writer"]].widths[w["out"]]
    k = int(cuts.get(w["reader"], {}).get(w["port"], 1))
    row_bits, out_bits = sum(row) // k, sum(out)
    assert sum(row) % k == 0 and row_bits >= out_bits and row_bits == 16 * -(-out_bits // 16)
    if row_bits > out_bits:
        short_rows[(f"{short(w['writer'])}.{w['out']}", out_bits, f"{short(w['reader'])}.{w['port']}", row_bits, k)] += 1
print(f"wires whose reader row holds bits its writer's output lacks: {sum(short_rows.values())} of {len(wires)}")
for (writer, ob, reader, rb, k), n in sorted(short_rows.items()):
    print(f"   {writer} ({ob} bits) -> {reader} (a row of {rb} bits" + (f", the port cut {k} ways" if k > 1 else "")
          + f"): {n} wires")

if "--case" in sys.argv:
    out = Path(sys.argv[sys.argv.index("--case") + 1])
    out.mkdir(parents=True, exist_ok=True)
    case = {"instance": inst, "trusted": tr, "spot": None, "rows": [], "count": {"table": {}, "c": 0, "k": 0, "n": 0}}
    (out / "cases.jsonl").write_text(g.line(case))
    (out / "python.jsonl").write_text(g.line(g.results(case)))
    said("wrote", out)
