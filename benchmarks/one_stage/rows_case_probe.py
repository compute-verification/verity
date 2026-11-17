"""Scratch: class 4,27,2's record on its population Program as a case of `inference-only-record-check`."""
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "benchmarks/one_stage"), str(ROOT / "benchmarks/prove_path")]
import inference_only as IO  # noqa: E402
import prove_path as PP  # noqa: E402
from verity.circuits import codec  # noqa: E402
from verity.commitments.periods import interface as IF  # noqa: E402
from verity_vllm.program.predict.window_circuit import class_circuit  # noqa: E402
from verity_vllm.program.predict.window_class import PearlTables  # noqa: E402

S = ROOT / "verity/protocols/workload_classification/training_detection/inference_only/verifier/lean/scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, S / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


vectors, script = load("record_difftest_vectors"), load("served_record_case")
ref = script.ref
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
say = lambda *a: print(f"[{time.time() - t0:.0f} s]", *a, flush=True)  # noqa: E731

serve = IO.serve_facts("unsloth/Llama-3.2-1B", "9535bd9b1d1dea6acafbdc4813b728796aeb28da")
circuit = class_circuit(PP.registered_of_serve(serve, None, None, PP.CAP_BITS), IO.window_class("4,27,2"),
                        tables=PearlTables("pearl-c-h100-v1-h3"))
say("circuit", circuit.name, "templates", len(circuit.templates), "wires", len(circuit.wires))
doc = IF.descriptor_record(circuit, 64)
prog = circuit.program()
desc = codec.encode_program(prog[0])
text, digest = codec.canonical_json(desc).decode(), codec.program_digest(desc)
assert doc["program"] == digest, (doc["program"][:16], digest[:16])
record = IF.circuit_bytes(doc).decode()
say("record", len(record), "bytes; program", len(text), "bytes; keys", sorted(doc), "blocks",
    json.dumps(doc.get("blocks"))[:400])
reg = doc.get("registered", {})
params = reg.get("params", {})
formats = reg.get("formats", {})
pins = IO.weight_pins(sorted(params))
rng = random.Random(1)
roots = {p: vectors._hex(rng) for p in params}
held = {}
for p in pins:
    b = ref.width_of(doc, p)
    held[p] = (roots[p], 1, 16 * params[p] // b, *([b] if b != 16 else []))
by_prefix = {}
for p in params:
    by_prefix.setdefault(p.split("/", 1)[0], [0, 0, 0])
    by_prefix[p.split("/", 1)[0]][0] += 1
    by_prefix[p.split("/", 1)[0]][1] += 16 * params[p]
    by_prefix[p.split("/", 1)[0]][2] += p in formats
say("params by prefix [count, bits, with a format]", by_prefix, "held", len(held))
tr = vectors.trusted(rng, digest, record, held, roots)
table, priced, missing = IO.table_of([f for f, _ in circuit.templates])
say("table", len(table), "priced", sorted(priced), "missing", missing)
limits = {"gamma": 10**16, "alpha": [338_992_160, 11_044_318_495_127]}
count = {"table": table, "c": 11_044_318_495_127, "k": 338_992_160, "n": 4}
case, _ = script.served_case(text, record, tr["log"], limits, count)
(out / "pop.case.json").write_text(json.dumps(case))
weight = "weights/model.embed_tokens.weight@2048" if "weights/model.embed_tokens.weight@2048" in held else sorted(held)[0]
neg, changed = script.served_case(text, record, tr["log"], limits, count, unregister=weight)
(out / "pop-unregistered.case.json").write_text(json.dumps(neg))
say("cases written; negative:", changed)
