# circuits' build command of 1:07 AM PST, verbatim, then the same circuit's description and cuts dumped (no second build)
set -eo pipefail
CLASS=$1
cd tests/adapters/vllm
uv run --extra torch-cpu python - $CLASS "$OUT/$CLASS.program.json" "$OUT/$CLASS.circuit.json" <<'PY'
import json, os, sys, time; sys.path.insert(0, os.getcwd())
t0 = time.time()
from tests.program.predict.test_window_circuit_fa3 import H200, LLAMA_1B, PEARL, _registered
from verity_vllm.program.model import INT8_EXACT_QUANT
from verity_vllm.program.predict.window_circuit import class_circuit
from verity_vllm.program.predict.window_class import PearlTables, WindowClass
from verity.circuits import codec
name, out = sys.argv[1:3]; cls, cfg = WindowClass(4, 27, 2), json.loads(LLAMA_1B.read_text())
c = (class_circuit(_registered(config=cfg), cls, tables=PearlTables(PEARL)) if name == "tied" else
 class_circuit(_registered(config=cfg), cls, tables=None) if name == "untied" else
 class_circuit(_registered({**H200, "int8_exact": "linears"}, {**cfg, "quantization_config": dict(INT8_EXACT_QUANT)}), cls, tables=None))
open(out, "wb").write(codec.canonical_json(json.loads(codec.canonical_json(codec.encode_program(c.program()[0])))))
json.dump({"ids": list(c.ids), "cuts": {k: dict(v) for k, v in c.cuts.items()}, "external": list(c.external),
           "templates": [[t, n] for t, (_, n) in zip(c.ids, c.templates)], "order": c.order and [list(x) for x in c.order]},
          open(sys.argv[3], "w"))
print(f"BUILT {name} {os.path.getsize(out)} bytes in {time.time()-t0:.0f} s", flush=True)
PY
