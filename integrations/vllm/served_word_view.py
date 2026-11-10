"""Scratch, never for landing: the word view of a Boolean reader's ports (`verity.ml.circuits.boolean.word_type`)
under the record form's leaf-widths rule at this checkout, measured and attacked.

On the small served phase's circuit the vllm suite builds (tests/program/predict/test_window_circuit.py): which
templates are Boolean, each one's ports as leaves and as elements, whether any leaf of a port leaves by an output
with no gate, and every wire's verdict under the rule as built and under each reading of the word view. Then one
record the word view admits and that reads sixteen 16-bit gates of another slot through one 16-bit row: the
difftest's `passed_on` with its passing template a Boolean Definition. Run from integrations/vllm:
uv run python served_word_view.py"""
import collections
import random
import sys

sys.path[:0] = [".", "../../verity/core/protocols/network/nci/lean/scripts"]
import record_difftest_vectors as g
from tests.program.predict import test_window_circuit as T
from verity.ml.circuits import boolean as B
from verity.ml.circuits.layout import Scope, resolve_gate
from verity.ml.circuits.program import is_input
from verity.ml.circuits.types import Array, Value

ref, IF, codec = g.ref, g.IF, g.codec


def short(tid: str) -> str:
    return tid.split("{")[0]


def shape(ws) -> str:
    return f"{len(ws)}x{sorted(set(ws))}"


def bit_port(ty) -> bool:
    return set(ty.leaf_widths()) <= {1}


def passed(program, k: int, fn) -> list[tuple[str, int, tuple[str, int]]]:
    """The output leaves of root node `k`'s first unit that are leaves of its own rows, no gate between: each as
    (output, leaf, (port, leaf))."""
    unit = Scope(program.root, 0, None, 0, 0).child(k, 0)
    params, ret = fn.signature()
    own = {}
    for i, (name, ty) in enumerate(params):
        for leaf in range(ty.leaves):
            own[resolve_gate(unit, ("p", i, leaf))[0]] = (name, leaf)
    out, j = [], 0
    for o, ty in enumerate(ref.fields_of(ret)):
        for leaf in range(ty.leaves):
            gate, prim = resolve_gate(unit, fn.body.ret[j])
            j += 1
            if is_input(prim):
                out.append((f"o{o}", leaf, own.get(gate)))
    return out


def takes(row, out) -> bool:
    """The rule as built: each side's leaves of one width, the writer's dividing the reader's."""
    return len(set(row)) == 1 and len(set(out)) == 1 and row[0] % out[0] == 0


def study(label: str, c, program) -> dict:
    ts = [ref.Template(IF.descriptor_id(fn), fn, n) for fn, n in c.templates]
    nodes = [k for k, n in enumerate(program.root.body.nodes) if not is_input(n.fn)]
    assert len(nodes) == len(ts)
    by = {t.tid: t for t in ts}
    boolean = {t.tid: B.is_boolean(t.fn) for t in ts}
    thru = {t.tid: passed(program, k, t.fn) for t, k in zip(ts, nodes, strict=True)}
    readers = {w.reader for w in c.wires}
    print(f"== {label}: {len(ts)} templates, {sum(t.units for t in ts)} units, {len(c.wires)} wires")
    print("templates (B Boolean by is_boolean; b some port all bits; R reads a wire; P a port leaf leaves ungated):")
    elems: dict[tuple[str, str], tuple[int, ...]] = {}
    for t in ts:
        params, ret = t.fn.signature()
        ports = []
        for name, ty in params:
            if bit_port(ty):
                elems[t.tid, name] = tuple(B.elements(ty))
                cut = int(c.cuts.get(t.tid, {}).get(name, 1))
                split = "" if len(elems[t.tid, name]) % cut == 0 else f" CUT {cut} THROUGH AN ELEMENT"
                ports.append(f"{name} {shape(t.widths[name])} -> {shape(elems[t.tid, name])}"
                             + (f" cut {cut}" if cut != 1 else "") + split)
        for j, ty in enumerate(ref.fields_of(ret)):
            if bit_port(ty):
                elems[t.tid, f"o{j}"] = tuple(B.elements(ty))
        flags = ("B" if boolean[t.tid] else "-") + ("b" if ports else "-") + ("R" if t.tid in readers else "-") \
            + ("P" if thru[t.tid] else "-")
        print(f"  {flags} {short(t.tid)} x{t.units}" + (": " + "; ".join(ports) if ports else ""))
        if thru[t.tid]:
            froms = collections.Counter((o, src[0] if src else None) for o, _, src in thru[t.tid])
            print("       ungated:", ", ".join(f"{n} leaves of {o} are {p}'s" for (o, p), n in sorted(froms.items())))

    def widths(view: str, tid: str, port: str, out: bool) -> tuple[int, ...]:
        raw = by[tid].widths[port]
        if (tid, port) not in elems or view == "built":
            return raw
        if out:
            return elems[tid, port] if view == "boolean, ports and outputs" and boolean[tid] else raw
        if view == "any port of bits":
            return elems[tid, port]
        if not boolean[tid] or (view == "boolean, nothing ungated" and thru[tid]):
            return raw
        return elems[tid, port]

    views = ("built", "boolean, ports", "boolean, nothing ungated", "boolean, ports and outputs", "any port of bits")
    verdict = {v: [takes(widths(v, w.reader, w.port, False), widths(v, w.writer, w.out, True)) for w in c.wires]
               for v in views}
    for v in views:
        ok = sum(verdict[v])
        print(f"{v}: {ok} admitted, {len(c.wires) - ok} refused")
        moved = collections.Counter(
            (("admitted -> REFUSED" if was else "refused -> admitted"), short(w.writer), w.out, short(w.reader), w.port,
             shape(widths(v, w.writer, w.out, True)), shape(widths(v, w.reader, w.port, False)))
            for w, was, now in zip(c.wires, verdict["built"], verdict[v], strict=True) if was != now)
        for (how, wt, o, rd, p, ow, rw), n in sorted(moved.items()):
            print(f"   {how}: {wt}.{o} {ow} -> {rd}.{p} {rw}" + (f" ({n} wires)" if n > 1 else ""))
        for w, now in zip(c.wires, verdict[v], strict=True):
            if not now:
                print(f"   refused: {short(w.writer)}.{w.out} {shape(widths(v, w.writer, w.out, True))} -> "
                      f"{short(w.reader)}.{w.port} {shape(widths(v, w.reader, w.port, False))}")
    return {"templates": ts, "boolean": boolean, "thru": thru, "elems": elems}


served = T._circuit(T.WINDOW)
study("the small served phase's circuit", served, served.program()[0])


def passed_bool(S: int = 2):
    """`record_difftest_vectors.passed_on`, its passing template a Boolean Definition: one parameter of sixteen rows
    of sixteen bits, one output of sixteen bits, leaf `j` its row `j`'s leaf `j`, no gate."""
    c = g.Circuit(f"passedbool{S}", S)
    v16, bits = Value(16), Array(16, Value(1))
    gate = [[("op", 16, [(0, 0)])]]
    wide = c.add(g.template("PassWide", [("w", v16)], [v16], gate), [([S, 16], ["slot", "-"], 0)])
    on = c.add(g.template("PassOnBool", [("x", Array(16, bits))], [bits], [("pass", 0, [17 * j for j in range(16)])]),
               [([S], ["slot"], 0)])
    take = c.add(g.template("PassTake", [("v", v16)], [v16], gate), [([S], ["slot"], 0)])
    c.cuts = {on: {"x": 16}}
    c.wires = IF.coalesce([IF.Wire(on, "x", wide, "o0", s, (16,), 16 * s, (1,), (0, 16)) for s in range(S)]
                          + [IF.Wire(take, "v", on, "o0", s, (1,), (s + 1) % S, (0,), (0, 1)) for s in range(S)])
    c.params = {"weights/wide": 16 * S}
    c.reads = IF.coalesce([IF.Read(wide, "w", "weights/wide", u, (1,), u, (0,)) for u in range(16 * S)])
    c.order = c.ordered()
    return c


c = passed_bool()
text, _ = c.program()
found = study("passed_on, its passing template Boolean", c, ref.reference.desc_program(text))
rng = random.Random(50_025)
roots = {p: g._hex(rng) for p in c.params}
made = g.finish(rng, c, {"weights/wide": (roots["weights/wide"], c.params["weights/wide"] // 2, 2)}, roots, [],
                policy={}, refused="leaf-widths")
inst, tr = made["case"]["instance"], made["case"]["trusted"]
print("the form as built:", ref.check_detail(inst, tr).get("error"))
built = ref.leaf_widths


def viewed(gated: bool):
    def rule(templates, wires, cuts):
        for t in templates:
            if B.is_boolean(t.fn) and not (gated and found["thru"][t.tid]):
                for name, ty in t.fn.signature()[0]:
                    t.widths[name] = tuple(B.elements(ty))
        return built(templates, wires, cuts)
    return rule


for what, gated in (("the word view on a Boolean reader's ports", False),
                    ("the same, only where no leaf of a port leaves ungated", True)):
    ref.leaf_widths = viewed(gated)
    got = ref.check_detail(inst, tr)
    print(f"{what}:", {k: got[k] for k in ("error", "complies", "fresh", "cross_wires", "gates", "blocks") if k in got})
ref.leaf_widths = built
slot = c.slot()
wide_id, on_id, take_id = (codec.descriptor_id(fn) for fn, _ in c.templates)
rows = {(w.reader, w.port, u, w.cut[0] + r): (w.writer, v)
        for w in c.wires for u, r, v in ref.entries({**w.doc(), "c0": w.cut[0], "C": w.cut[1]})}
for s in range(c.slots):
    writer, v = rows[take_id, "v", s, 0]
    gates = {rows[on_id, "x", v, j] for j in range(16)}
    print(f"slot {slot[take_id][s]}: Take's one 16-bit row is On unit {v}'s output (slot {slot[on_id][v]}), whose bit j "
          f"is bit j of its row j: {len(gates)} gates of {shape(found['templates'][0].widths['o0'])} bits, "
          f"in slots {sorted({slot[wide_id][u] for _, u in gates})}")
