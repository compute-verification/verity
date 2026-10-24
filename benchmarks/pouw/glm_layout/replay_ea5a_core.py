"""Replay core credit's ROOTS1 steps (`pouw_h1_gpu.core_verdict`: the manifest under its round roots, `credit._a_link`,
`credit.layout_of`, `credit._arow`) over a served H1 window's record, on CPU, with no GPU and no engine (VLLM-81).

The record is ``out/commit/pouw-h1-calls.json`` of a ``pouw_h1_gpu --workload fill --served --commit`` run, e.g.
r20261023-224227-ea5a's (GLM-5.3 INT8 layers 0-2, which this reproduces: ``research fetch r20261023-224227-ea5a --all``
puts it under ``~/.research/runs/r20261023-224227-ea5a/out/commit/``). The window's declared Program is rebuilt from the
record's shapes and module names (`pouw_h1_served.declared`), as the harness built it, so its descriptor is the one core
read. ROOT is the checkout whose core and vllm integration are replayed (the run's own commit, or a fix branch):

    git worktree add /tmp/root <commit>
    cd /tmp/root && PYTHONPATH=/tmp/root:/tmp/root/integrations/vllm \\
        uv run --extra torch-cpu python benchmarks/pouw/glm_layout/replay_ea5a_core.py /tmp/root RECORD.json

At ea5a's commit (ce0d35319d) it prints the manifest, a-link and arow refusals the run printed; a window that ties prints
``"tied": true``. Rebuilding ea5a's 72-call Program takes about 2 minutes and a few GB (its descriptor is 368 MB).
"""
import json
import sys

root, path = sys.argv[1], sys.argv[2]
sys.path[:0] = [root, f"{root}/integrations/vllm"]

from tests.protocol_options import pouw_h1_gpu as GPU  # noqa: E402
from verity_vllm.protocol_options import pouw_h1_served as SV  # noqa: E402

rec = json.load(open(path))
shapes = [tuple(x) for x in rec["public"]["shapes"]]
modules: dict[int, str] = {}
for c in rec["calls"]:
    modules.setdefault(c["layer"], c["module"])
prog = SV.declared(shapes, modules, rec["public"]["rows"], rec["public"]["steps"])
desc = SV.descriptor(prog)
print(f"declared Program: {len(prog.fn.bindings['CALLS'])} calls, descriptor {len(desc)} bytes")
print("declared order, step 0:", [modules[w] for w in range(len(shapes))])
print("recorded calls:", len(rec["calls"]), "refused at record:", rec["refused"])
print(json.dumps(GPU.core_verdict({"commitment": rec["commitment"], "calls": rec["calls"], "descriptor": desc.hex()}),
                 indent=1, default=str))
