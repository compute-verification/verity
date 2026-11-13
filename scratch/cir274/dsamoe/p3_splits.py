"""CIR-274 DSAMOE: which prompt lengths make a prefill step mixed, some SparseMlaSplit nodes split (NS >= 2) and some not,
by `launch_num_splits` as `step_needs` calls it (the census's 132 SMs, max_seqlen_k the index's top-k 2048, a prefill token
t of one request seeing min(t + 1, 2048) keys). A mixed step's BatchedGemm{K=512} reads x from two gather templates (the
frame's `_whole`), two wires a (head, column) run. Run from the root of the tree:

    .venv/bin/python p3_splits.py OUT MAXLP

Writes OUT/splits.json: each length's split tokens, and the runs of lengths that are mixed."""
import json
import sys
from pathlib import Path

from verity.definitions.registry.sparse_mla_split import launch_num_splits

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
top = int(sys.argv[2])
rows, runs = {}, []
for LP in range(1, top + 1):
    ns = launch_num_splits([min(t + 1, 2048) for t in range(LP)], 132, 2048, 0)
    split = sum(1 for s in ns if s > 1)
    mixed = 0 < split < LP
    rows[LP] = {"split_tokens": split, "max_ns": max(ns), "mixed": mixed}
    if mixed and runs and runs[-1][1] == LP - 1:
        runs[-1][1] = LP
    elif mixed:
        runs.append([LP, LP])
(out / "splits.json").write_text(json.dumps({"mixed_runs": runs, "lengths": rows}) + "\n")
print("DSAMOE splits mixed runs", runs, flush=True)
for LP in (43, 64, 65, 86, 92, 93, 126, 127, 128, 129, 150, 200, 256, 512, 1024):
    if LP <= top:
        print("DSAMOE splits", LP, rows[LP], flush=True)
