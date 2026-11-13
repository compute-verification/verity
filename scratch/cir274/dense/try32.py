"""CIR-274 DENSE: the node window at LP prompt ids, 2 outputs and LAYERS layers (the test's `glm_request`) with the frame's
families and the dense half's, its census and its units by template. Run from adapters/vllm:

    uv run python ../../../verity-dense-scratch/scratch/cir274/dense/try32.py [LP] [LAYERS] [moe]
"""
import json
import resource
import sys
import time

sys.path.insert(0, "tests/program/predict")
from test_node_window import glm_request  # noqa: E402

from verity_vllm.program.predict import node_window as NW, node_window_dense as ND  # noqa: E402

LP = int(sys.argv[1]) if len(sys.argv) > 1 else 32
L = int(sys.argv[2]) if len(sys.argv) > 2 else 1
moe = len(sys.argv) > 3 and sys.argv[3] == "moe"
fn = glm_request(LP, L, moe=moe).program.fn
t0 = time.perf_counter()
circuit, w = NW.node_window(fn, families={**NW.FAMILIES, **ND.FAMILIES})
cen = w.census(circuit)
cen.pop("peers")
print(json.dumps(cen, indent=1))
for t, n in sorted(circuit.units.items()):
    print(f"{n:8d} {t[:150]}  cuts={circuit.cuts.get(t)}")
print("wall", round(time.perf_counter() - t0, 1), "peak GiB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 3))
