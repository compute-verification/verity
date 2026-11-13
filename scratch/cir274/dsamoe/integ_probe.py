import sys, collections, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "tests/program/predict"))
import verity_vllm; print("verity_vllm from", verity_vllm.__file__)
from test_node_window import glm_request
from verity_vllm.program.predict import node_window as NW, node_window_sparse as NS, node_window_dense as ND
print("overlap", set(ND.FAMILIES) & set(NS.FAMILIES))
fams = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES}
moe = sys.argv[2] == "moe"
fn = glm_request(int(sys.argv[1]), int(sys.argv[3]) if len(sys.argv) > 3 else 1, moe=moe).program.fn
circuit, w = NW.node_window(fn, families=fams)
cen = w.census(circuit)
print(json.dumps({k: cen[k] for k in ("digest", "free", "unlowered")}, indent=0)[:2500])
mine = ("Moe", "Embedding", "SparseMla", "DsaLatent", "BatchedGemm", "Fp8GroupQuant", "CublasSplitK", "LlBf16")
c = collections.Counter()
for x in circuit.wires:
    a, b = any(k in x.writer for k in mine), any(k in x.reader for k in mine)
    if a != b:
        c[(x.writer[:50], x.out, x.reader[:50], x.port)] += 1
for k, v in sorted(c.items()): print(v, k)
print("free on mine", collections.Counter((f.tid[:50], f.port, f.family) for f in w.free if any(k in f.tid for k in mine)))
