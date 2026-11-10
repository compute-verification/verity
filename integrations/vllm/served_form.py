"""The TINY served window circuit the vllm suite builds (tests/program/predict/test_window_circuit.py), fed to the
record form's reference at this checkout: the form's verdict on it, each wire under the leaf-widths rule, the root
under the own-rows rule, and the same wires with EmbeddingRead_v1.tok taken as one 32-bit leaf. Run from
integrations/vllm:  uv run python served_form.py [--case DIR]   (--case writes the one difftest case for the Lean)."""
import collections, json, random, sys
from pathlib import Path
sys.path.insert(0, ".")
sys.path.insert(0, "../../verity/core/protocols/network/nci/lean/scripts")
import record_difftest_vectors as g
from tests.program.predict import test_window_circuit as T

ref, IF, codec = g.ref, g.IF, g.codec
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
short = lambda t: t.split("{")[0]
print("circuit:", c.name, "| templates", len(c.templates), "| units", sum(n for _, n in c.templates), "| wires", len(c.wires))
got = ref.check_detail(inst, tr)
print("the form at this checkout:", json.dumps(got))

templates = ref.templates_of(ref.reference.desc_program(text), doc, digest)
try:
    ref.own_rows(ref.reference.desc_program(text))
    print("own-rows: taken")
except ref.Malformed as e:
    print("own-rows: refused", e.code)
wires, cuts = ref.wires_of(templates, doc, None)
by = {t.tid: t for t in templates}


def sweep(label):
    pairs, bad = collections.Counter(), []
    for w in wires:
        key = (short(w["writer"]), w["out"], short(w["reader"]), w["port"])
        ww, rw = by[w["writer"]].widths[w["out"]], by[w["reader"]].widths[w["port"]]
        pairs[(f"{len(ww)}x{sorted(set(ww))}", f"{len(rw)}x{sorted(set(rw))}")] += 1
        try:
            ref.leaf_widths(templates, [w], cuts)
        except ref.Malformed:
            bad.append(key)
    print(f"{label}: {len(wires)} wires, {len(wires) - len(bad)} admitted, {len(bad)} refused")
    for b in bad:
        print("   refused:", "%s.%s -> %s.%s" % b)
    return pairs


pairs = sweep("leaf-widths, the Definitions as served")
print("wires by (writer's leaves, reader's leaves), as count x widths:")
for k, n in sorted(pairs.items()):
    print("  ", k, n)
under = [(short(w["writer"]), w["out"], short(w["reader"]), w["port"]) for w in wires
         if set(by[w["writer"]].widths[w["out"]]) == {1} and set(by[w["reader"]].widths[w["port"]]) != {1}]
print("bits under a wider leaf, admitted today:", len(under))
for u in under:
    print("   %s.%s -> %s.%s" % u)
tok = [t for t in templates if short(t.tid) == "EmbeddingRead_v1"]
for t in tok:
    print("EmbeddingRead_v1.tok leaves as served:", f"{len(t.widths['tok'])} x {sorted(set(t.widths['tok']))} bits")
    into = sorted({(short(w["writer"]), w["out"], f"{len(by[w['writer']].widths[w['out']])}x{sorted(set(by[w['writer']].widths[w['out']]))}")
                   for w in wires if w["reader"] == t.tid and w["port"] == "tok"})
    print("   wires into tok from:", into)
    t.widths["tok"] = (32,)
sweep("leaf-widths, with EmbeddingRead_v1.tok one 32-bit leaf")
if "--case" in sys.argv:
    out = Path(sys.argv[sys.argv.index("--case") + 1])
    out.mkdir(parents=True, exist_ok=True)
    for t in tok:
        t.widths["tok"] = (1,) * 32
    case = {"instance": inst, "trusted": tr, "spot": None, "rows": [], "count": {"table": {}, "c": 0, "k": 0, "n": 0}}
    (out / "cases.jsonl").write_text(g.line(case))
    (out / "python.jsonl").write_text(g.line(g.results(case)))
    print("wrote", out)
