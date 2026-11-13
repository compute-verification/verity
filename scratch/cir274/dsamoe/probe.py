"""CIR-274 DSAMOE: the node window at LP prompt ids and LAYERS layers (dense, or every layer MoE) with the frame's families
and node_window_sparse's, its census and units by template. From adapters/vllm:

    uv run python ../../../verity-scratch/scratch/cir274/dsamoe/probe.py LP LAYERS [moe] [only=Fam,Fam]
"""
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "tests/program/predict"))
from test_node_window import glm_request  # noqa: E402
from verity_vllm.program.predict import node_window as NW  # noqa: E402
from verity_vllm.program.predict import node_window_sparse as NS  # noqa: E402

LP, layers = int(sys.argv[1]), int(sys.argv[2])
moe = "moe" in sys.argv[3:]
only = next((a[5:].split(",") for a in sys.argv[3:] if a.startswith("only=")), None)
fams = {**NW.FAMILIES, **NS.FAMILIES}
if only is not None:
    fams = {k: v for k, v in fams.items() if k in only or k in NW.FAMILIES}
t0 = time.time()
fn = glm_request(LP, layers, moe=moe).program.fn
circuit, w = NW.node_window(fn, families=fams)
census = w.census(circuit)
print(json.dumps({k: census[k] for k in ("digest", "units", "templates", "wire_patterns", "free", "unlowered", "steps")}, indent=1))
for tid, n in sorted(circuit.units.items()):
    print(f"{n:>9} {tid[:160]}")
print("seconds", round(time.time() - t0, 1), "peak GiB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 3))
