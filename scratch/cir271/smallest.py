"""CIR-271 (REQBLOCK): the smallest class whose node request Program keeps every Definition of the (256, 32) one.

    .venv/bin/python scratch/cir271/smallest.py OUT LAYERS "LP:T LP:T ..."

For each (LP, T) builds GLM-5.3's TP8 class-G node request Program (`build_node_program`, one declared request, eager) and
records its Definitions (template families) with their calls, its template bindings, its primitives by id and each
family's serving cut (`serving_templates`). Writes OUT/smallest.json and one REQBLOCK line a shape.
"""
import importlib.util
import json
import resource
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

out, layers, shapes = Path(sys.argv[1]), (int(sys.argv[2]) if sys.argv[2] else None), sys.argv[3].split()
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("compile_cost", "benchmarks/compile_cost/run.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

from verity.circuits.codec import descriptor_id
from verity_vllm.pipeline import serving_templates as ST
from verity_vllm.program.predict.node import build_node_program
from verity_vllm.program.predict.phi import Phi

#: the 29 Definitions of the full (256, 32) node Program, run r20261112-202908-e9ba
FULL = ["AllGather", "AllReduceMnnvl", "AllReduceMultimem", "AllReduceRmsNormMnnvl", "BatchedGemm", "CublasSplitKGemm",
        "CublasSplitKGemmF32Out", "DeepGemmFp8Linear", "DsaIndexerKRow", "DsaIndexerQ", "DsaMqaLogits", "DsaRMSNorm",
        "DsaTopKShortRow", "EmbeddingShard", "FlashInferFp8Linear", "Fp8GroupQuantUe8m0", "Gemm", "LlBf16DotprodGemm",
        "MoeExpertFp8Gemm", "MoeExpertFp8GemmW", "MoeRouterSigmoidBias", "MoeSum", "RMSNormCudaVector8",
        "RMSNormFusedCudaVector8", "ResidualAdd", "RoPEHeadGptJ", "SiluMul", "SparseMlaSplit", "TokenSelect"]
KINDS = (("CUT", ST.CUT), ("COORDS", ST.COORDS), ("WIDE_COORDS", ST.WIDE_COORDS), ("COLS", ST.COLS), ("SLICES", ST.SLICES),
         ("PARTS", ST.PARTS), ("KCHUNKS", ST.KCHUNKS), ("CHAINS", ST.CHAINS), ("STAGES", ST.STAGES), ("EXPERTS", ST.EXPERTS))


def cuts(family: str) -> list[str]:
    word = f"{family}_v1"
    return [k for k, c in KINDS if any(x.split("_v")[0] == family for x in c)] or ["row"]


reg = cc.registered(0, layers, "eager")
rec: dict = {"layers": layers, "shapes": {}}
for s in shapes:
    LP, T = (int(x) for x in s.split(":"))
    t0 = time.perf_counter()
    try:
        phi = Phi.from_json({**reg.phi.to_json(), "workload": {"requests": [
            {"request_id": "q0", "index": 0, "prompt_len": LP, "max_tokens": T + 1, "arrive_step": 0, "n": 1, "seed": None}]}})
        r = build_node_program(phi, "request", LP=LP, T=T, with_desc=False)
        body = r.node_fn.body.nodes
        fam, tids, prim = Counter(), Counter(), Counter()
        for n in body:
            if n.form in ("call", "batch"):
                f = (getattr(n.fn, "name", None) or n.fn.id.split("{")[0]).split("_v")[0]
                fam[f] += n.n if n.form == "batch" else 1
                tids[descriptor_id(n.fn)] += 1
            elif n.form == "primitive":
                prim[n.fn.id.split("{")[0]] += 1
        got = {"calls": sum(fam.values()), "templates": len(tids), "families": dict(sorted(fam.items())),
               "missing": [f for f in FULL if f not in fam], "extra": sorted(set(fam) - set(FULL)),
               "primitives": dict(prim.most_common(10)), "cuts": {f: cuts(f) for f in sorted(fam)},
               "body_nodes": len(body), "wall_s": round(time.perf_counter() - t0, 1),
               "peak_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 2)}
    except BaseException as e:  # noqa: BLE001 - a refusal is a result
        got = {"error": f"{type(e).__name__}: {e}"[:1200], "trace": traceback.format_exc()[-1500:],
               "wall_s": round(time.perf_counter() - t0, 1)}
    rec["shapes"][s] = got
    (out / "smallest.json").write_text(json.dumps(rec, indent=1) + "\n")
    print("REQBLOCK smallest", layers, s, json.dumps({k: v for k, v in got.items() if k not in ("families", "cuts", "trace")})[:1500],
          flush=True)
