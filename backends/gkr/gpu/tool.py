"""Store declaration for Candidate A's GPU prover (research.store README §6.1).

    research run --tool a_gpu_prove --project verity --scratch triton -- python -m gpu.run prove /workspace/bb/pos4096 --vus 4096 --reps 3 --warmup 1
    research run --tool a_gpu_prove --project verity --scratch triton -- python backends/gkr/bench_result.py /workspace/bb/pos4096 --vus 4096 \
        --checker v1 --path graphed --reps 3 --warmup 1 --verifier /workspace/bin/verity-gkr-verify --threads 16

Key params: `dir` (the bb_export / v2 export directory: a path-valued param, replaced by the content identity of the tree, never
the pathname), `vus` (the batch), `checker` (v1 | v2), `path` (graphed | packed | torch: the prover's kernel path, selected by its
own environment switches), `field` (babybear6), `clear`.  `--device`, `--reps`, `--warmup`, `--json`, `--proof-out`, `--verifier`,
`--threads` are recorded but do not enter the derivation (the device is the condition `hw.family`; the rest say how often and where).
`python -m gpu.run prove` has no `--checker` / `--path` flags: for that form `checker` is left to the export's manifest (unset here)
and `path` is read from the environment switches when present, else `graphed`.  Imports nothing but `research.store.tool`.
"""
from __future__ import annotations

import argparse
import os

from research.store.tool import Tool

DEFAULT_FIELD = "babybear6"


def parse(argv: list[str]) -> dict:
    toks = list(argv)
    # the wrapper form: `python backends/gkr/bench_result.py DIR ...` or `python bench_result.py DIR ...`; the module form:
    # `python -m gpu.run prove DIR ...`
    if "prove" in toks and any(t.endswith("gpu.run") for t in toks):
        rest = toks[toks.index("prove") + 1:]
        form = "gpu.run prove"
    else:
        idx = next((i for i, t in enumerate(toks) if t.endswith("bench_result.py")), None)
        if idx is None:
            return {"_unparsed": argv}
        rest = toks[idx + 1:]
        form = "bench_result.py"
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("dir")
    ap.add_argument("--vus", type=int, default=None)
    ap.add_argument("--checker", default=None)
    ap.add_argument("--path", default=None)
    ap.add_argument("--field", default=DEFAULT_FIELD)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--tile", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--proof-out", default=None)
    ap.add_argument("--verifier", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ns, unknown = ap.parse_known_args(rest)
    out = vars(ns)
    out["form"] = form
    if out["path"] is None:
        if os.environ.get("VERITY_GPU_TORCH_ONLY"):
            out["path"] = "torch"
        elif os.environ.get("VERITY_GPU_NO_GRAPHS") or os.environ.get("VERITY_GPU_NO_PACKED"):
            out["path"] = "packed"
        else:
            out["path"] = "graphed"
    if out["checker"] is None:
        out.pop("checker")               # unknown from argv alone (gpu.run prove): the export's manifest decides; not guessed here
    if unknown:
        out["_unknown"] = unknown
    return out


A_GPU_PROVE = Tool(
    name="a_gpu_prove", version="1",
    command=["python", "-m", "gpu.run"],
    closure=["backends/gkr/gpu/**", "backends/gkr/packed/**", "backends/gkr/bench_result.py", "packages/verity/src/verity/**"],
    key_params=["dir", "vus", "checker", "path", "field", "clear"],
    key_conditions=["hw.family"],
    param_artifacts=["dir"],
    caches=["triton"],
    produces=["bench-result/v1", "run-files/v1"],
    parse=parse,
)
