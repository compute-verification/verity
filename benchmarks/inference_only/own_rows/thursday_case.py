"""A diagnostic: Thursday's served circuit (Llama 3.2 1B, class 4,27,2, four requests, the RTX target, seed 7, no Pearl+
tables: `served_door.py build`'s lowering) as one case of the record form, its record `interface.Window.circuit_record`
writes, its log a deployment record registering every registered parameter but the circuit's run values and the run's
record registering those, every root a stand-in. Then the case with `--nodes` and the reference's verdicts, and the
H200 policy's count limits beside the session policy's own `c` and `k` on this record.

argv: TREE OUT
"""
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

tree, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(tree / "tests/adapters/vllm"))
SCRIPTS = tree / "verity/protocols/workload_classification/training_detection/inference_only/verifier/lean/scripts"
sys.path.insert(0, str(SCRIPTS))
from verity.circuits.codec import descriptor_chunks, program_digest  # noqa: E402
from verity.commitments.periods import interface as IF  # noqa: E402
from verity.protocols.proofs.spot.verifier import service as S  # noqa: E402
from verity_vllm.program.predict import window_circuit as WCirc  # noqa: E402
from verity_vllm.program.predict import window_class as WC  # noqa: E402
from tests.program.predict import test_steps as T  # noqa: E402
import served_record_case as SRC  # noqa: E402

t = time.perf_counter()
cfg = json.loads((tree / "tests/adapters/vllm/tests/program/predict/fixtures/llama32-1b.config.json").read_text())
reg = T.Registered.of(T.Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg}, target=dict(T.RTXPRO6000)),
                      default_seed=7)
circuit = WCirc.lower(reg, WC.WindowClass(4, 27, 2), None, requests=[WC.Request(0, j, 27, 2) for j in range(4)]).circuit
program, query = circuit.program()
desc_bytes = b"".join(descriptor_chunks(program))
print("lowered in", round(time.perf_counter() - t, 1), "s; description sha256", hashlib.sha256(desc_bytes).hexdigest(),
      "bytes", len(desc_bytes), "root nodes", len(program.root.body.nodes), flush=True)
(out / "C.json").write_bytes(desc_bytes)

rule = "rate:1/64"
S.SamplingDesign(rule)
law = dict(S.params_sampling_design(rule, program, query, None)[0])
win = IF.Window(circuit, "00" * 64, law, 64)
record = IF.circuit_bytes(win.circuit_record()).decode()
(out / "record.json").write_text(record)
doc = json.loads(record)
regd = doc.get("registered", {})
params = regd.get("params", {})
formats = regd.get("formats", {})
run_values = {v for v, *_ in IF.program_values(circuit.registered)} if circuit.registered is not None else set()
print("record:", len(record), "bytes;", len(doc["templates"]), "templates;", len(doc.get("wires", [])), "wires;",
      len(params), "registered params;", "run values:", sorted(run_values), flush=True)


def root(p: str) -> str:
    return hashlib.sha512(b"stand-in root " + p.encode()).hexdigest()


def entry(p: str) -> dict:
    b = int((formats.get(p) or {}).get("word_bits", 16))
    return {"value": p, "root": root(p), "leaves": 16 * params[p] // b, "words": 1, "word_bits": b}


first = {"previous": None, "registered": [entry(p) for p in sorted(params) if p not in run_values]}
digest = program_digest(json.loads(desc_bytes))
run = {"program": digest, "window": {"circuit": hashlib.sha512(record.encode()).hexdigest(),
                                      "registered_roots": {p: root(p) for p in params}},
       "registered": [entry(p) for p in sorted(params) if p in run_values]}
log = SRC.rechained([first, run])

pins = {p: root(p) for p in params if p not in run_values}
k_policy = c_policy = None
try:
    sys.path.insert(0, str(tree / "benchmarks/inference_only"))
    import built as B
    pol = B.policy(doc, pins)
    c_policy, k_policy, table = pol["c"], pol["k"], pol["table"]
    print("session policy on this record: c", c_policy, "k", k_policy, "slots", pol["slots"], flush=True)
    div = B.divisions(doc, pins, pol["gates"])
    print("divisions:", json.dumps({n: {k: v for k, v in d.items() if k != "k_by_kind"} for n, d in div.items()}), flush=True)
    print("session k by kind:", json.dumps(div["session"]["k_by_kind"])[:3000], flush=True)
except Exception as e:  # noqa: BLE001
    table = None
    print("policy: not computed:", type(e).__name__, str(e)[:500], flush=True)
limits = {"alpha": [1, 1], "gamma": 10**15}
count = {"table": table or {}, "c": c_policy or 0, "k": k_policy or 0, "n": 31457280}
for name, nodes in (("thursday", False), ("thursday-nodes", True)):
    try:
        case, changed = SRC.served_case(desc_bytes.decode(), record, log, limits, count, nodes=nodes)
    except ValueError as e:
        print(name, "case refused:", e, flush=True)
        continue
    (out / f"{name}.case.json").write_text(SRC.line(case))
    rec2 = json.loads(case["instance"]["record"])
    decl = rec2.get("registered", {}).get("nodes", [])
    print(name, "changed:", changed, "declared nodes:", len(decl),
          dict(collections.Counter(("run" if (d.get("param") in run_values) else "param" if "param" in d else "table")
                                   for d in decl)), flush=True)
    got = SRC.verdicts(case)
    print(name, "check:", json.dumps(got["check"], sort_keys=True), flush=True)
    print(name, "count:", json.dumps(got["count"], sort_keys=True), flush=True)
