# circuits' build command (1:07 AM PST), then a stand-in circuit record and registration log for the same circuit `c`
set -eo pipefail
CLASS=$1
{ [ -L .venv ] && [ ! -e .venv ] && rm .venv; } || true
uv sync -q --all-packages --extra torch-cpu
SCRIPTS=$PWD/verity/protocols/workload_classification/training_detection/inference_only/verifier/lean/scripts
cd tests/adapters/vllm
uv run --extra torch-cpu python - $CLASS "$OUT" "$SCRIPTS" <<'PY'
import gzip, hashlib, importlib.util, json, os, random, sys, time; sys.path.insert(0, os.getcwd())
t0 = time.time()
from tests.program.predict.test_window_circuit_fa3 import H200, LLAMA_1B, PEARL, _registered
from verity_vllm.program.model import INT8_EXACT_QUANT
from verity_vllm.program.predict.window_circuit import class_circuit
from verity_vllm.program.predict.window_class import PearlTables, WindowClass
from verity.circuits import codec
from verity.commitments.periods import interface as IF
name, out, scripts = sys.argv[1:4]; cls, cfg = WindowClass(4, 27, 2), json.loads(LLAMA_1B.read_text())
c = (class_circuit(_registered(config=cfg), cls, tables=PearlTables(PEARL)) if name == "tied" else
 class_circuit(_registered(config=cfg), cls, tables=None) if name == "untied" else
 class_circuit(_registered({**H200, "int8_exact": "linears"}, {**cfg, "quantization_config": dict(INT8_EXACT_QUANT)}), cls, tables=None))
desc = json.loads(codec.canonical_json(codec.encode_program(c.program()[0])))
text, digest = codec.canonical_json(desc).decode(), codec.program_digest(desc)
print(f"BUILT {name}: program {hashlib.sha256(text.encode()).hexdigest()[:16]} in {time.time()-t0:.0f} s", flush=True)
sys.path.insert(0, scripts)
def load(m):
    s = importlib.util.spec_from_file_location(m, f"{scripts}/{m}.py"); mod = importlib.util.module_from_spec(s); s.loader.exec_module(mod); return mod
V, SR = load("record_difftest_vectors"), load("served_record_case")
units = {}
for x in desc["definitions"][desc["root"]]["body"]["nodes"]:
    t = desc["callees"][x["fn"]]
    if not V.INPUT_ID.match(t):
        units[t] = units.get(t, 0) + (x["n"] if x["form"] == "batch" else 1)
doc = {"format": IF.CIRCUIT, "name": c.name, "templates": [list(t) for t in units.items()], "program": digest,
       "wires": [w.doc() for w in c.wires], **c.declared(),
       **({"blocks": c.blocks.doc(sum(c.units.values()), "")} if c.blocks is not None else {})}
record = IF.circuit_bytes(doc).decode()
reg = c.registered
params = dict(reg.params) if reg is not None else {}
width = lambda p: (reg.formats.get(p, {}) if reg is not None else {}).get("word_bits", 16)
roots = {p: hashlib.sha512(f"stand-in root {p}".encode()).hexdigest() for p in params}
held = {p: V._registered(roots[p], n, width(p)) for p, n in params.items()}
tr = V.trusted(random.Random(0), digest, record, held, roots, before=1)
case, _ = SR.served_case(text, record, tr["log"], {"alpha": [1, 1], "gamma": V.BIG})
print(f"RECORD {name}: {len(record)} bytes, {len(params)} registered parameters, "
      f"{len(reg.reads) if reg is not None else 0} registered reads, {len(c.wires)} wires; record keys {sorted(doc)}", flush=True)
with gzip.open(f"{out}/{name}.case.json.gz", "wt") as f:
    json.dump(case, f)
open(f"{out}/{name}.record.json", "w").write(record)
json.dump(tr["log"], open(f"{out}/{name}.log.json", "w"))
print(f"CASE {name} written in {time.time()-t0:.0f} s", flush=True)
t1 = time.time()
try:
    v = SR.verdicts(case)
    print(f"PY_TWIN {name}: check {json.dumps(v['check'])[:600]} count {json.dumps(v['count'])[:300]} in {time.time()-t1:.0f} s", flush=True)
except Exception as e:
    print(f"PY_TWIN {name}: raised {type(e).__name__}: {str(e)[:600]} in {time.time()-t1:.0f} s", flush=True)
PY
