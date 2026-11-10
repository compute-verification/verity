"""cublas_bf16's sidecar binds each route `cublas_routes.json` records for GLM-5.3's two split-K weights on one TP8 rank (the MoE
router's gate, K = 6144, N = 256, f32 out; the indexer's ``wk``, K = 6144, N = 160) at its served statics, so circuit-check's
catalog checks every one: a route the table gains (another split count, or the non-split GEMM) without its binding, or a binding
of a route the table doesn't record, fails here by id."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from verity.circuits.boolean.trace import word_of
from verity.circuits.defs import bind
from verity.definitions.ops import gemm as G
from verity.definitions.ops.boolean import gemm as BG
from verity.definitions.registry import bf16_gemm_check as BC
from verity.definitions.registry import cublas_bf16 as CB

SIDECAR = Path(CB.__file__).with_name("cublas_bf16.circuit-check.py")
#: each weight's `cublas_routes.json` key: (its split-K Definition, its non-split one)
ROUTES = {"router 6144 256": (CB.CublasSplitKGemmF32OutWord, CB.GemmF32OutWord),
          "linear 6144 160": (CB.CublasSplitKGemmWord, G.Gemm)}


def test_every_recorded_glm53_route_has_a_served_binding():
    spec = importlib.util.spec_from_file_location("cublas_bf16_circuit_check", SIDECAR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bound = {make().id for label, make in mod._model_roots() if "/served-" in label}
    dot = word_of(BG.HopperBF16WgmmaDot16)
    want = set()
    for key, (split, whole) in ROUTES.items():
        _phase, K, N = key.split()
        for s in {run[2] for run in BC.CUBLAS_ROUTES["routes"][key]}:
            fn = bind(split, K=int(K), N=int(N), SPLIT_K=s, DOT=dot) if s else bind(whole, K=int(K), N=int(N), DOT=dot)
            want.add(fn.id)
    assert bound == want, (f"bind each of these routes in {SIDECAR.name}'s _model_roots, labelled '<family>/served-<weight>', and "
                           f"drop the rest: missing {sorted(want - bound)}, not recorded {sorted(bound - want)}")
