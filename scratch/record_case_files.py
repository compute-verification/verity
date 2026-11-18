"""One served class at 4,27,2 built wired, written as the files `served_record_case.py` reads: the Program, two circuit
records (`descriptor_record` at cap 64 as built, `circuit.json`, and a stand-in with its templates in the Program's
batch order, `circuit_po.json`), each with a two-record registration log (the first registering every parameter the
record registers, the run's own committing that record and those roots), and the limits. Run from tests/adapters/vllm
with LEANPKG the inference-only Lean package."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(os.environ["LEANPKG"]) / "scripts"))

name, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)


def build(name):
    from tests.program.predict.test_window_circuit_fa3 import H200, LLAMA_1B, PEARL, _registered
    from verity_vllm.program.model import INT8_EXACT_QUANT
    from verity_vllm.program.predict.window_circuit import class_circuit
    from verity_vllm.program.predict.window_class import PearlTables, WindowClass
    cls, config = WindowClass(4, 27, 2), json.loads(LLAMA_1B.read_text())
    if name == "tiny":
        from tests.program.predict import test_steps as S
        from tests.program.predict.test_window_circuit import CLASS, GRAIN
        return class_circuit(S._registered(), CLASS, None, GRAIN, provable=False)
    if name == "untied":
        return class_circuit(_registered(config=config), cls, tables=None)
    if name == "tied":
        return class_circuit(_registered(config=config), cls, tables=PearlTables(PEARL))
    return class_circuit(_registered({**H200, "int8_exact": "linears"},
                                     {**config, "quantization_config": dict(INT8_EXACT_QUANT)}), cls, tables=None)


import inference_only_record_reference as R  # noqa: E402
import verity.commitments.periods.interface as IF  # noqa: E402
from verity.circuits import codec  # noqa: E402

t0 = time.time()
c = build(name)
t1 = time.time()
prog, _q = c.program()
desc = json.loads(codec.canonical_json(codec.encode_program(prog)))
program = codec.canonical_json(desc).decode()
built = IF.descriptor_record(c, 64)
units = {}
for n in R.reference.desc_program(program).root.body.nodes:
    if not R.is_input(n.fn):
        t = codec.descriptor_id(n.fn)
        units[t] = units.get(t, 0) + (n.n if n.form == "batch" else 1)
if sorted(units.items()) != sorted(map(tuple, built["templates"])):
    raise SystemExit("the Program's templates are not the record's")
ordered = {**built, "templates": [[t, k] for t, k in units.items()]}
params = built.get("registered", {}).get("params", {})
formats = built.get("registered", {}).get("formats", {})


def hx(*parts):
    return hashlib.sha512("/".join(map(str, parts)).encode()).hexdigest()


entries, roots = [], {}
for p, n in sorted(params.items()):
    b = (formats.get(p) or {}).get("word_bits", 16)
    leaves = 16 * n // b
    words = 2 if leaves % 2 == 0 else 1
    roots[p] = hx("root", p)
    entries.append({"value": p, "root": roots[p], "leaves": leaves // words, "words": words, "domain": hx("domain", p),
                    **({"word_bits": b} if b != 16 else {})})
V2 = "verity/registration/v2"
first = {"format": V2, "registered": entries, "previous": None}
(out / "program.json").write_text(program)
(out / "log0.json").write_text(json.dumps(first))
(out / "limits.json").write_text(json.dumps({"alpha": [10 ** 12, 1], "gamma": 10 ** 12}))
for suffix, doc in (("", built), ("_po", ordered)):
    record = IF.circuit_bytes(doc).decode()
    run = {"format": V2, "protocol": "verity/sampled-proofs/one-stage/v1", "program": codec.program_digest(desc),
           "window": {"circuit": hashlib.sha512(record.encode()).hexdigest(), "registered_roots": roots},
           "registered": [], "previous": R.reference.record_digest(first)}
    (out / f"circuit{suffix}.json").write_text(record)
    (out / f"log1{suffix}.json").write_text(json.dumps(run))
print(json.dumps({"class": name, "build_s": round(t1 - t0), "files_s": round(time.time() - t1),
                  "program_bytes": len(program), "record_bytes": len(IF.circuit_bytes(built)),
                  "templates": len(built.get("templates", [])), "in_program_order": built["templates"] == ordered["templates"],
                  "wires": len(built.get("wires", [])), "params": len(params), "formats": len(formats)}), flush=True)
