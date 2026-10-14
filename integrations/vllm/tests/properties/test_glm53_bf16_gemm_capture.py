"""The GLM-5.3 and DeepSeek-V4-Pro bf16 GEMM capture without a GPU: its comparison jobs over operands and words made here (the
kernels' words stood in for by the models), a dotprod whose K loop ends in a tail tile and a ragged round, a flipped word found and
located, the dotprod's contraction decided by the stress rows, the gate's kernel record at each model's gate, each model's cases,
the absorb weights' served layout, the switch's environment, the bindings, the verdict per case and per model, the second process
under the default configuration, and `main` end to end (the GPU run is `tests/properties/glm53_bf16_gemm_capture_gpu.py --out ...`
on an H200)."""
from __future__ import annotations

import copy
import json
import os

import numpy as np
import pytest
import torch

from tests.properties import glm53_bf16_gemm_capture_gpu as X
from verity_vllm.program.registry import bf16_gemm_check as BC
from verity_vllm.program.registry import mla_check as MK
from verity_vllm.properties.record import REGISTRY, digest_of

DOTPROD = {"kind": "dotprod", "BS": 32, "VEC": 8}
SPLITK = {"kind": "splitk", "SPLIT_K": 2, "TILE_K": 128, "WARPS": 2}
CUBLAS = {"kind": "cublas"}
#: at BS 32: one main tile (8 x 32), one tail tile at half width (4 x 32) and two scalar rounds, the last ragged, as DeepSeek-V4-Pro's
#: K 7168 at BS 256 ends in a tail tile
TAIL_K = 8 * 32 + 4 * 32 + 37


def _words(rng, shape, scale: float) -> np.ndarray:
    return (MK.u32(rng.standard_normal(shape).astype(np.float32) * np.float32(scale)) >> 16).astype(np.uint16)


def _x(rng, M: int, K: int, stress: str = "row3") -> np.ndarray:
    """The capture's activation: normal rows, and 2^-122 times a normal row at row 3 (`row3`), at every row (`all`) or nowhere."""
    x = _words(rng, (M, K), 1.0)
    if stress == "all":
        return _words(rng, (M, K), X.STRESS_SCALE)
    if stress == "row3" and M > 3:
        x[3] = _words(rng, (K,), X.STRESS_SCALE)
    return x


def _job(tmp_path, kind: str, *, M: int = 5, fma: int = 1, stress: str = "row3", flip: tuple | None = None) -> dict:
    """One launch's job with the kernel's words made by the model it binds: a GLM-5.3 gate launch on `DOTPROD` / `SPLITK` /
    `CUBLAS`, a DeepSeek-V4-Pro one on `DOTPROD` at `TAIL_K` (`tail`), a linear, a DeepSeek-V4-Pro compressor GEMM (`f32out`), or
    a two-head bmm."""
    rng = np.random.default_rng(3)
    d = tmp_path / f"{kind}_{fma}_{stress}_{flip}"
    if kind == "bmm":
        a, b = _x(rng, 2 * M, 32, stress).reshape(2, M, 32), _words(rng, (2, 32, 3), 2.0 ** -5)
        out = BC.bmm_rows(a, b)
        if flip:
            out[flip] ^= 1
        return {"model": "glm53", "phase": "bmm", "case": "v_absorb", "M": M, "K": 32, "N": 3, "kernel": None,
                "paths": X._save(d, a=a, b=b, out=out)}
    K, N = {"splitk": (512, 16), "tail": (TAIL_K, 16)}.get(kind, (256, 16))
    x, w = _x(rng, M, K, stress), _words(rng, (N, K), 2.0 ** -6)
    dot = lambda: BC.dotprod_rows(x, w, BS=32, VEC=8, FMA=fma)                    # noqa: E731
    out = {"dotprod": dot, "tail": dot, "splitk": lambda: BC.splitk_rows(x, w, 2, 128, 2), "cublas": lambda: BC.f32out_rows(x, w),
           "f32out": lambda: BC.f32out_rows(x, w), "linear": lambda: BC.linear_rows(x, w)}[kind]()
    if flip:
        out[flip] ^= 1
    phase = kind if kind in ("linear", "f32out") else "router"
    return {"model": "dsv4-pro" if kind in ("tail", "f32out") else "glm53", "phase": phase,
            "case": {"linear": "lm_head.shard", "f32out": "compressor.kv_score.c4"}.get(kind, "mlp.gate"), "M": M, "K": K, "N": N,
            "kernel": {"dotprod": DOTPROD, "tail": DOTPROD, "splitk": SPLITK, "cublas": CUBLAS}.get(kind),
            "paths": X._save(d, x=x, w=w, out=out)}


def _total(results: list[dict]) -> dict:
    assert not [r for r in results if "error" in r], results
    return {k: sum(r[k] for r in results) for k in X.TOTALS}


def _sampled(M: int, cols: int) -> int:
    return sum(len({0, cols - 1, (r * 7919) % cols}) for r in sorted({0, 3, M - 1}))


@pytest.mark.parametrize("kind", ["dotprod", "tail", "splitk", "cublas", "linear", "f32out", "bmm"])
def test_each_launch_kind_takes_the_kernel_words(tmp_path, kind: str) -> None:
    jobs = X.jobs_of(_job(tmp_path, kind))
    assert sorted(r for j in jobs for r in j["reference_rows"]) == [0, 3, 4]
    t = _total([X.compare(j) for j in jobs])
    cols = 6 if kind == "bmm" else 16
    per_model = 2 if kind in ("dotprod", "tail") else 1
    assert t["twin_mismatch"] == 0 and t["reference_coords"] == per_model * _sampled(5, cols)
    if kind in ("dotprod", "tail"):
        assert (t["words"], t["dotprod_words"], t["fma1_mismatch"]) == (0, 5 * cols, 0) and t["fma0_mismatch"] == t["fma_differ"] > 0
    else:
        assert (t["words"], t["mismatch"], t["dotprod_words"]) == (5 * cols, 0, 0)
    if kind == "tail":
        assert len(BC.dotprod_rounds(TAIL_K, 32, 8)) == 8 + 4 + 2


@pytest.mark.parametrize("kind,flip,where", [("linear", (2, 4), (2, 4)), ("splitk", (4, 15), (4, 15)), ("f32out", (0, 3), (0, 3)),
                                             ("bmm", (1, 2, 1), (2, 4))])
def test_a_flipped_word_is_found_and_located(tmp_path, kind: str, flip: tuple, where: tuple) -> None:
    res = [X.compare(j) for j in X.jobs_of(_job(tmp_path, kind, flip=flip))]
    assert _total(res)["mismatch"] == 1
    first = next(r["first_mismatch"] for r in res if r["first_mismatch"])
    assert (first["row"], first["column"]) == where


@pytest.mark.parametrize("fma", [0, 1])
def test_the_stress_rows_decide_the_dotprod_contraction(tmp_path, fma: int) -> None:
    for stress in ("row3", "all"):
        t = _total([X.compare(j) for j in X.jobs_of(_job(tmp_path, "dotprod", fma=fma, stress=stress))])
        assert t[f"fma{fma}_mismatch"] == 0 and t[f"fma{1 - fma}_mismatch"] == t["fma_differ"] > 0 and X.fma_of(t) == fma
    t = _total([X.compare(j) for j in X.jobs_of(_job(tmp_path, "dotprod", fma=fma, stress="none"))])
    assert t["fma_differ"] == 0 and X.fma_of(t) is None                        # normal rows can't tell the two apart
    assert X.fma_of({**t, "fma_differ": 3, "fma0_mismatch": 1, "fma1_mismatch": 2}) is None


@pytest.mark.parametrize(("K", "splits"), [(6144, 3), (6144, 4), (7168, 2), (7168, 3), (7168, 4), (7168, 5)])
@pytest.mark.parametrize("output", ["f32", "bf16"])
def test_the_splitk_probe_distinguishes_every_served_hypothesis(K: int, splits: int, output: str) -> None:
    x, w, differ = X.splitk_probe_words(K, splits, output)
    models = BC.cublas_splitk_hypotheses(x[None], w, splits, out=output)
    candidate = models[BC.CUBLAS_SPLITK_HYPOTHESIS]
    assert set(differ) == set(models)
    assert all(not np.array_equal(candidate, words) for name, words in models.items()
               if name != BC.CUBLAS_SPLITK_HYPOTHESIS)
    assert differ[BC.CUBLAS_SPLITK_HYPOTHESIS] == 0
    assert all(v > 0 for k, v in differ.items() if k != BC.CUBLAS_SPLITK_HYPOTHESIS)
    assert ("contiguous-f32-reverse" in models) == (splits > 2)
    assert ("contiguous-f32-tree" in models) == (splits > 3)


def test_the_splitk_probe_unwraps_the_router_output() -> None:
    tensor = object()
    assert X._primary_output(tensor) is tensor
    assert X._primary_output((tensor, {"metadata": True})) is tensor


def test_the_profiler_record_pins_grid_split_count_and_reduction() -> None:
    trace = {"traceEvents": [
        {"cat": "kernel", "name": "nvjet_bz_splitK_TNT",
         "args": {"gridX": 4, "gridY": 3, "gridZ": 8, "blockX": 256, "blockY": 1, "blockZ": 1}},
        {"cat": "kernel", "name": "void cublasLt::splitKreduce_kernel<32>()", "args": {"grid": [12, 1, 1]}},
    ]}
    got = X._profile_record(trace)
    assert got["ok"] and got["split_k"] == 8
    assert got["launches"][0]["grid"] == [4, 3, 8] and got["launches"][0]["block"] == [256, 1, 1]
    assert got["reduction_kernels"][0]["name"].startswith("void cublasLt::splitKreduce")
    assert not X._profile_record({"traceEvents": trace["traceEvents"][:1]})["ok"]


def test_cutedsl_splitk_is_not_a_profiled_cublas_split() -> None:
    trace = {"traceEvents": [
        {"cat": "kernel", "name": "kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_ll_bf16_splitkLLBf16SplitK",
         "args": {"grid": [1, 16, 6]}},
    ]}
    got = X._profile_record(trace)
    assert got["ok"] and got["split_k"] is None and got["reduction_kernels"] == []


def test_cutedsl_dispatch_wins_over_legacy_profile_split_count(tmp_path) -> None:
    job = _job(tmp_path, "splitk")
    job["split_k"] = SPLITK["SPLIT_K"]              # v3 used to mistake this profiler field for a cuBLAS route
    assert X._cublas_split_k(job) is None
    t = _total([X.compare(j) for j in X.jobs_of(job)])
    assert (t["words"], t["mismatch"], t["twin_mismatch"]) == (5 * 16, 0, 0)


def test_the_default_cublas_split_counts_are_pinned_by_shape_and_m() -> None:
    assert [BC.default_cublas_split_k("router", M, 6144, 256) for M in (17, 32, 128, 256)] == [4, 4, 4, 3]
    assert [BC.default_cublas_split_k("router", M, 7168, 384) for M in (17, 32, 128, 256)] == [5, 5, 4, 4]
    assert [BC.default_cublas_split_k("linear", M, 6144, 160) for M in X.LINEAR_M] == [4] * len(X.LINEAR_M)
    assert [BC.default_cublas_split_k("linear", M, 7168, 64) for M in X.LINEAR_M] == [5] * len(X.LINEAR_M)
    assert BC.default_cublas_split_k("f32out", 256, 7168, 2048) is None       # c4 M256 is the non-split kernel


def test_the_default_cublas_route_does_not_confuse_unrecorded_with_non_split() -> None:
    for M in X.LINEAR_M:
        assert BC.default_cublas_route("linear", M, 6144, 160) == ("splitk", {"SPLIT_K": 4})
        assert BC.default_cublas_route("linear", M, 6144, 19360) == ("non_split", {})
    assert BC.default_cublas_route("linear", 5, 6144, 19360) is None
    assert BC.default_cublas_route("linear", 1, 6144, 161) is None


def test_the_default_operand_dump_keeps_reduction_routes_and_deduplicates_weights(tmp_path) -> None:
    profile = {"launches": [{"name": "main_splitK", "grid": [4, 8, 1]}],
               "reduction_kernels": [{"name": "splitKreduce_kernel", "grid": [8, 2, 1]}]}
    jobs, launches = [], []
    for M in (1, 4):
        paths = X._save(tmp_path / "scratch" / f"M{M}", x=np.full((M, 8), M, np.uint16),
                        w=np.arange(24, dtype=np.uint16).reshape(3, 8), out=np.full((M, 3), M, np.uint32))
        jobs.append({"model": "glm53", "phase": "linear", "case": "indexer.wk_weights_proj", "M": M, "K": 8, "N": 3,
                     "kernel": None, "paths": paths})
        launches.append({"model": "glm53", "phase": "linear", "case": "indexer.wk_weights_proj", "M": M, "profile": profile})
    out = tmp_path / "out"
    got = X.dump_default_operands(out, jobs, launches)
    manifest = json.loads((out / got["manifest"]).read_text())
    a, b = (manifest["cases"][f"glm53/linear/indexer.wk_weights_proj/M{M}"] for M in (1, 4))
    assert got["cases"] == 2 and a["paths"]["w"] == b["paths"]["w"]
    assert np.array_equal(np.load(out / b["paths"]["x"]), np.full((4, 8), 4, np.uint16))
    assert len(list((out / "default_operands").rglob("w.npy"))) == 1


def test_the_gate_kernel_record_is_dispatch_sm90s() -> None:
    assert X._kernel_of([]) == {"kind": "cublas"}
    dot = {"backend": "dotprod", "bs": 128, "split_k": 0, "num_stages": 0}
    split = {"backend": "splitk", "bs": 0, "split_k": 8, "num_stages": 2}
    assert X._kernel_of([dot]) == X.expected_kernel(1) == X.expected_kernel(4) == X.expected_kernel(3, "dsv4-pro") == \
        {"kind": "dotprod", "BS": 128, "VEC": 8}
    assert X._kernel_of([{**dot, "bs": 256}]) == X.expected_kernel(2, "dsv4-pro") == {"kind": "dotprod", "BS": 256, "VEC": 8}
    assert X._kernel_of([split]) == X.expected_kernel(9) == X.expected_kernel(11) == X.expected_kernel(6, "dsv4-pro") == \
        {"kind": "splitk", "SPLIT_K": 8, "TILE_K": 256, "WARPS": 4}
    assert X.expected_kernel(5)["SPLIT_K"] == X.expected_kernel(16)["SPLIT_K"] == X.expected_kernel(5, "dsv4-pro")["SPLIT_K"] == 6
    assert X.expected_kernel(7, "dsv4-pro")["SPLIT_K"] == X.expected_kernel(16, "dsv4-pro")["SPLIT_K"] == 7
    assert X.expected_kernel(17) == X.expected_kernel(17, "dsv4-pro") == {"kind": "cublas"}
    for model, m in X.MODELS.items():                     # each model's router M, quick or not, launch every kernel its gate has
        every = {json.dumps(X.expected_kernel(M, model), sort_keys=True) for M in range(1, BC.LL_MAX_M + 2)}
        for ms in (m.router_m, m.quick_router_m):
            assert {json.dumps(X.expected_kernel(M, model), sort_keys=True) for M in ms} == every


def test_each_model_runs_its_own_gemms() -> None:
    every = X.cases(False, set(X.PHASES), list(X.MODELS))
    keys = [X.case_key(*c) for c in every]
    assert len(keys) == len(set(keys)) == sum(len(m.router_m) + 1 + len(X.LINEAR_M) * (len(m.linear) + len(m.f32out) + len(m.bmm))
                                              for m in X.MODELS.values())
    assert {(m, p) for m, p, _, _ in every} == {("glm53", "router"), ("glm53", "linear"), ("glm53", "bmm"), ("dsv4-pro", "router"),
                                                 ("dsv4-pro", "linear"), ("dsv4-pro", "f32out")}
    assert [c for c in every if c[2] == X.PROBE] == [("glm53", "router", X.PROBE, 1), ("dsv4-pro", "router", X.PROBE, 1)]
    assert X.cases(True, {"f32out"}, ["dsv4-pro"]) == [("dsv4-pro", "f32out", n, M) for n, _, _ in X.MODELS["dsv4-pro"].f32out
                                                      for M in X.QUICK_M]


def test_an_unknown_model_or_phase_is_refused(tmp_path) -> None:
    for flag, value in (("--models", "glm53,qwen3"), ("--only", "router,attention")):
        with pytest.raises(SystemExit, match="unknown"):
            X.main(["--out", str(tmp_path / "out"), "--scratch", str(tmp_path), flag, value])


def test_the_absorb_weights_are_the_served_views() -> None:
    P, V, L = X.QK_NOPE, X.V_HEAD, X.KV_LORA
    kv_b = torch.arange(X.NHL * (P + V) * L, dtype=torch.float32).reshape(X.NHL * (P + V), L)
    uk_t, uv = X.absorb_weights(kv_b)
    assert list(uk_t.shape) == [X.NHL, P, L] and list(uv.shape) == [X.NHL, L, V]
    assert [list(uk_t.stride()), list(uv.stride())] == [X.SERVED_STRIDE["q_nope_absorb"], X.SERVED_STRIDE["v_absorb"]]
    h, p, v, ll = 5, 17, 200, 301
    assert uk_t[h, p, ll] == kv_b[h * (P + V) + p, ll] and uv[h, ll, v] == kv_b[h * (P + V) + P + v, ll]


def test_the_switch_environment_is_set_or_cleared(tmp_path, monkeypatch) -> None:
    for k in (*X.SWITCH_ENV, "VLLM_BATCH_INVARIANT", "TRITON_CACHE_DIR", "VLLM_ENABLE_V1_MULTIPROCESSING"):
        monkeypatch.setenv(k, "x")
    env = X.environment(False, tmp_path)
    assert {k: env[k] for k in X.SWITCH_ENV} == X.SWITCH_ENV and env["VLLM_BATCH_INVARIANT"] == "0"
    env = X.environment(True, tmp_path)
    assert not set(X.SWITCH_ENV) & set(env) and all(k not in os.environ for k in X.SWITCH_ENV)


def test_the_programs_bind_at_every_models_shapes() -> None:
    """GLM-5.3: the gate, the dotprod at BS 128 at each FMA, split-K at 6 and 8, two linears and two absorbs; DeepSeek-V4-Pro: the
    gate, the dotprod at BS 128 and 256 at each FMA, split-K at 6, 7 and 8, two linears and three compressor GEMMs."""
    got = X.bind_check()
    assert got["ok"] and len(got["bindings"]) == 9 + 13 and len(X.bind_check(["glm53"])["bindings"]) == 9
    hop = "DOT=HopperBF16WgmmaDot16_v1"
    for key in (f"Gemm_v2{{K=6144,N=19360,{hop}}}", f"LlBf16SplitKGemm_v1{{N=256,K=6144,SPLIT_K=8,TILE_K=256,WARPS=4,{hop}}}",
                f"BatchedGemm_v1{{B=8,K=512,N=256,{hop}}}", "LlBf16DotprodGemm_v1{N=384,K=7168,BS=256,VEC=8,FMA=1}",
                f"LlBf16SplitKGemm_v1{{N=384,K=7168,SPLIT_K=7,TILE_K=256,WARPS=4,{hop}}}", f"GemmF32Out_v1{{K=7168,N=2048,{hop}}}",
                f"Gemm_v2{{K=7168,N=16160,{hop}}}", f"Gemm_v2{{K=7168,N=64,{hop}}}"):
        assert got["bindings"][key], key


TOTALS = {"words": 14, "mismatch": 0, "dotprod_words": 4, "fma1_mismatch": 0, "fma0_mismatch": 2, "fma_differ": 2,
          "reference_coords": 4, "twin_mismatch": 0}
NONE = {k: 0 for k in TOTALS}
F32OUT = "dsv4-pro/f32out/compressor.kv_score.c4/M1"


def _doc() -> dict:
    ok = {"deterministic": True, "operands_ok": True}
    g, d = {"model": "glm53"}, {"model": "dsv4-pro"}
    launches = [{**ok, **g, "phase": "router", "case": "mlp.gate", "M": 1, "kernel": {"kind": "dotprod"}, "dispatch_ok": True, "us": 4.0},
                {**ok, **g, "phase": "linear", "case": "lm_head.shard", "M": 1, "us": 9.0},
                {**ok, **g, "phase": "bmm", "case": "v_absorb", "M": 1, "us": 3.0},
                {**ok, **d, "phase": "f32out", "case": "compressor.kv_score.c4", "M": 1, "us": 5.0}]
    comps = [{**g, "phase": "router", "case": "mlp.gate", "M": 1, **NONE, "dotprod_words": 4, "fma0_mismatch": 2, "fma_differ": 2,
              "reference_coords": 1},
             {**g, "phase": "linear", "case": "lm_head.shard", "M": 1, **NONE, "words": 6, "reference_coords": 1},
             {**g, "phase": "bmm", "case": "v_absorb", "M": 1, **NONE, "words": 4, "reference_coords": 1},
             {**d, "phase": "f32out", "case": "compressor.kv_score.c4", "M": 1, **NONE, "words": 4, "reference_coords": 1}]
    doc = {"schema": X.SCHEMA, "models": list(X.MODELS), "expected": [X._case(x) for x in launches], "gate": {"ok": True}, "errors": {},
           "switch": {"ok": True, "cublas_version": 120900}, "selection": {m: {"ok": True} for m in X.MODELS}, "bind": {"ok": True},
           "launches": launches,
           "comparisons": comps, "totals": dict(TOTALS)}
    doc["fma"] = X.fma_of(doc["totals"])
    doc["cases"] = X.case_verdicts(doc)
    return doc


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("errors", "glm53/router/x"), "boom"), (("switch", "ok"), False),
                                        (("selection", "dsv4-pro", "ok"), False), (("selection",), {}), (("bind", "ok"), False),
                                        (("totals", "mismatch"), 1), (("totals", "reference_coords"), 0),
                                        (("totals", "twin_mismatch"), 1), (("fma",), None), (("launches", 0, "dispatch_ok"), False),
                                        (("launches", 1, "deterministic"), False), (("launches", 3, "operands_ok"), False)])
def test_the_verdict_needs_every_check(path: tuple, value) -> None:
    assert X._verdict(_doc())
    doc = copy.deepcopy(_doc())
    node = doc
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    assert not X._verdict(doc)
    assert not X._verdict({**_doc(), "launches": _doc()["launches"][:3]})      # a run without an expected case is not a GO


def test_each_case_has_its_own_verdict() -> None:
    doc = _doc()
    assert {k: c["ok"] for k, c in doc["cases"].items()} == {"glm53/router/mlp.gate/M1": True, "glm53/linear/lm_head.shard/M1": True,
                                                             "glm53/bmm/v_absorb/M1": True, F32OUT: True}
    doc["comparisons"][1]["mismatch"] = 1
    doc["errors"]["glm53/bmm/q_nope_absorb/M4/compare"] = "boom"
    doc["errors"]["glm53/bmm/v_absorb/M16"] = "boom"                            # M16's error is not M1's
    doc["errors"]["selection/dsv4-pro"] = "boom"                                # not a case
    doc["fma"] = 0                                                              # a dotprod launch whose words are not the run's fma
    got = X.case_verdicts(doc)
    assert {k: c["ok"] for k, c in got.items()} == {"glm53/router/mlp.gate/M1": False, "glm53/linear/lm_head.shard/M1": False,
                                                    "glm53/bmm/v_absorb/M1": True, F32OUT: True, "glm53/bmm/q_nope_absorb/M4": False,
                                                    "glm53/bmm/v_absorb/M16": False}


def test_each_model_has_its_own_verdict() -> None:
    doc = _doc()
    assert X.model_verdicts(doc) == {"glm53": True, "dsv4-pro": True}
    doc["cases"][F32OUT]["ok"] = False
    assert X.model_verdicts(doc) == {"glm53": True, "dsv4-pro": False} and not X._verdict(doc)
    doc = _doc()
    doc["expected"].append("glm53/bmm/q_nope_absorb/M1")                       # expected and never launched
    assert X.model_verdicts(doc) == {"glm53": False, "dsv4-pro": True} and not X._verdict(doc)


def test_the_default_run_is_a_second_process(tmp_path, monkeypatch) -> None:
    """The second process gets `--served-default --no-default`, the same models and its own out directory; its cases that are not
    GO are the cases the default configuration changes, and the switch's cost is each case's time over the default's."""
    seen = {}
    lm, va = "glm53/linear/lm_head.shard/M1", "glm53/bmm/v_absorb/M1"

    def fake_run(argv, capture_output, text, cwd):
        seen["argv"] = argv
        sub = tmp_path / "served_default"
        sub.mkdir()
        cases = {lm: {"ok": False, "us": 6.0}, va: {"ok": True, "us": 3.0}}
        (sub / X.RECORD).write_text(json.dumps({"ok": False, "digest": "d", "errors": {}, "cases": cases, "launches": [],
                                                "switch": {"cublas_version": 120900}, "model_verdicts": {"glm53": False},
                                                "splitk_hypotheses": {}}))
        return type("P", (), {"returncode": 1, "stdout": "out", "stderr": ""})()

    monkeypatch.setattr(X.subprocess, "run", fake_run)
    a = X.argparse.Namespace(workers=3, only="linear,bmm", models="glm53", min_driver=575, want_cc="9.0", want_sms=132, scratch=None,
                             quick=True, dump_operands=True)
    got = X.default_run(a, tmp_path)
    argv = seen["argv"]
    assert {"--served-default", "--no-default", "--quick", "--dump-operands"} <= set(argv)
    assert argv[argv.index("--out") + 1] == str(tmp_path / "served_default") and argv[argv.index("--models") + 1] == "glm53"
    assert got["changed"] == [lm] and got["us"][va] == 3.0
    doc = {"served_default": got, "cases": {lm: {"us": 9.0}, va: {"us": 3.0}}}
    assert X._cost(doc) == {lm: 1.5, va: 1.0}


def test_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys) -> None:
    """`main` with the GPU's pieces stood in for (gate, switch, selection, the launches as kernel words made here, the default run)
    and vLLM's config module stubbed: the comparison jobs through the spawned worker pool, the decided fma, the per-case and
    per-model verdicts, `--models`, the sealed record and the exit status, so a slip in the orchestration shows here and not in the
    H200's hour."""
    import contextlib
    import sys
    import types

    vc = types.ModuleType("vllm.config")
    vc.CompilationConfig = vc.VllmConfig = lambda **kw: kw
    vc.set_current_vllm_config = lambda cfg: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", vc)
    for k in (*X.SWITCH_ENV, "VLLM_BATCH_INVARIANT", "TRITON_CACHE_DIR", "VLLM_ENABLE_V1_MULTIPROCESSING"):
        monkeypatch.setenv(k, "")
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True})
    monkeypatch.setattr(X.G, "_environment", lambda: {})
    monkeypatch.setattr(X, "switch", lambda served_default: {"ok": True, "cublas_version": 120900})
    monkeypatch.setattr(X, "make_gate", lambda cfg, model: object())
    monkeypatch.setattr(X, "selection", lambda gate: {"ok": True})
    todo = [("glm53", "router", "mlp.gate", 5), ("glm53", "router", "mlp.gate", 6), ("glm53", "linear", "lm_head.shard", 5),
            ("glm53", "bmm", "v_absorb", 5), ("dsv4-pro", "router", "mlp.gate", 5), ("dsv4-pro", "f32out", "compressor.kv_score.c4", 5)]
    monkeypatch.setattr(X, "cases", lambda quick, only, models: [c for c in todo if c[0] in models])
    jobs = {("glm53", "router", 5): _job(tmp_path, "dotprod", fma=0), ("glm53", "router", 6): _job(tmp_path, "splitk", M=6),
            ("glm53", "linear", 5): _job(tmp_path, "linear", flip=(1, 2)), ("glm53", "bmm", 5): _job(tmp_path, "bmm"),
            ("dsv4-pro", "router", 5): _job(tmp_path, "tail", fma=0), ("dsv4-pro", "f32out", 5): _job(tmp_path, "f32out")}

    def launch(work, weights, gates, model, phase, name, M, seed, served_default):
        job = jobs[(model, phase, M)]
        rec = {"model": model, "phase": phase, "case": name, "M": M, "K": job["K"], "N": job["N"], "deterministic": True,
               "operands_ok": True, "us": 2.0}
        if phase == "router":
            rec.update(kernel=job["kernel"], dispatch_ok=True)
        return rec, {**job, "case": name}

    monkeypatch.setattr(X, "launch", launch)
    monkeypatch.setattr(X, "default_run", lambda a, outdir: {"ok": False, "changed": ["glm53/router/mlp.gate/M6"],
                                                             "us": {"glm53/router/mlp.gate/M5": 1.0}})
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1"]) == 1
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False and not doc["errors"] and doc["fma"] == 0
    assert {k: c["ok"] for k, c in doc["cases"].items()} == {"glm53/router/mlp.gate/M5": True, "glm53/router/mlp.gate/M6": True,
                                                             "glm53/linear/lm_head.shard/M5": False, "glm53/bmm/v_absorb/M5": True,
                                                             "dsv4-pro/router/mlp.gate/M5": True,
                                                             "dsv4-pro/f32out/compressor.kv_score.c4/M5": True}
    assert doc["model_verdicts"] == {"glm53": False, "dsv4-pro": True} and doc["expected"] == [X.case_key(*c) for c in todo]
    assert doc["totals"]["mismatch"] == 1 and doc["switch_cost"] == {"glm53/router/mlp.gate/M5": 2.0}
    assert doc["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":16:8" and doc["env"]["VLLM_BATCH_INVARIANT"] == "0"
    printed = capsys.readouterr().out
    assert "case glm53/linear/lm_head.shard/M5" in printed and "GLM53-BF16-GEMM NO-GO" in printed and "fma=0" in printed
    assert "glm53=NO-GO dsv4-pro=GO" in printed
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1", "--no-default", "--models", "dsv4-pro"]) == 0
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["models"] == ["dsv4-pro"] and doc["model_verdicts"] == {"dsv4-pro": True} and len(doc["cases"]) == 2
    assert "GLM53-BF16-GEMM GO" in capsys.readouterr().out
    jobs[("glm53", "linear", 5)] = _job(tmp_path, "linear")
    monkeypatch.setattr(X, "default_run", lambda a, outdir: {"ok": True, "changed": [],
                                                             "us": {"glm53/router/mlp.gate/M5": 1.0}})
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1", "--no-default"]) == 0
    assert "GLM53-BF16-GEMM GO" in (printed := capsys.readouterr().out) and "glm53=GO dsv4-pro=GO" in printed


def test_the_record_is_a_registered_property() -> None:
    from verity_vllm.properties.record import seal

    doc = seal(_doc(), X._verdict(_doc()))
    assert REGISTRY["glm53_bf16_gemm_capture"].schema == X.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)
