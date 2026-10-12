"""The GLM-5.3 bf16 GEMM capture without a GPU: its comparison jobs over operands and words made here (the kernels' words stood in
for by the models), a flipped word found and located, the dotprod's contraction decided by the stress rows, the gate's kernel
record, the absorb weights' served layout, the switch's environment, the bindings, the verdict, the second process under the
default configuration, and `main` end to end (the GPU run is `tests/properties/glm53_bf16_gemm_capture_gpu.py --out ...` on an
H200)."""
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
    """One launch's job with the kernel's words made by the model it binds: a gate launch on `DOTPROD` / `SPLITK` / `CUBLAS`, a
    linear, or a two-head bmm."""
    rng = np.random.default_rng(3)
    d = tmp_path / f"{kind}_{fma}_{stress}_{flip}"
    if kind == "bmm":
        a, b = _x(rng, 2 * M, 32, stress).reshape(2, M, 32), _words(rng, (2, 32, 3), 2.0 ** -5)
        out = BC.bmm_rows(a, b)
        if flip:
            out[flip] ^= 1
        return {"phase": "bmm", "case": "v_absorb", "M": M, "K": 32, "N": 3, "kernel": None, "paths": X._save(d, a=a, b=b, out=out)}
    K, N = (512, 16) if kind == "splitk" else (256, 16)
    x, w = _x(rng, M, K, stress), _words(rng, (N, K), 2.0 ** -6)
    k = {"dotprod": DOTPROD, "splitk": SPLITK, "cublas": CUBLAS}.get(kind)
    out = {"dotprod": lambda: BC.dotprod_rows(x, w, BS=32, VEC=8, FMA=fma), "splitk": lambda: BC.splitk_rows(x, w, 2, 128, 2),
           "cublas": lambda: BC.f32out_rows(x, w), "linear": lambda: BC.linear_rows(x, w)}[kind]()
    if flip:
        out[flip] ^= 1
    return {"phase": "linear" if kind == "linear" else "router", "case": "lm_head.shard" if kind == "linear" else "mlp.gate", "M": M,
            "K": K, "N": N, "kernel": k, "paths": X._save(d, x=x, w=w, out=out)}


def _total(results: list[dict]) -> dict:
    assert not [r for r in results if "error" in r], results
    return {k: sum(r[k] for r in results) for k in X.TOTALS}


def _sampled(M: int, cols: int) -> int:
    return sum(len({0, cols - 1, (r * 7919) % cols}) for r in sorted({0, 3, M - 1}))


@pytest.mark.parametrize("kind", ["dotprod", "splitk", "cublas", "linear", "bmm"])
def test_each_launch_kind_takes_the_kernel_words(tmp_path, kind: str) -> None:
    jobs = X.jobs_of(_job(tmp_path, kind))
    assert sorted(r for j in jobs for r in j["reference_rows"]) == [0, 3, 4]
    t = _total([X.compare(j) for j in jobs])
    cols = 6 if kind == "bmm" else 16
    per_model = 2 if kind == "dotprod" else 1
    assert t["twin_mismatch"] == 0 and t["reference_coords"] == per_model * _sampled(5, cols)
    if kind == "dotprod":
        assert (t["words"], t["dotprod_words"], t["fma1_mismatch"]) == (0, 5 * cols, 0) and t["fma0_mismatch"] == t["fma_differ"] > 0
    else:
        assert (t["words"], t["mismatch"], t["dotprod_words"]) == (5 * cols, 0, 0)


@pytest.mark.parametrize("kind,flip,where", [("linear", (2, 4), (2, 4)), ("splitk", (4, 15), (4, 15)), ("bmm", (1, 2, 1), (2, 4))])
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


def test_the_gate_kernel_record_is_dispatch_sm90s() -> None:
    assert X._kernel_of([]) == {"kind": "cublas"}
    dot = {"backend": "dotprod", "bs": 128, "split_k": 0, "num_stages": 0}
    split = {"backend": "splitk", "bs": 0, "split_k": 8, "num_stages": 2}
    assert X._kernel_of([dot]) == X.expected_kernel(1) == X.expected_kernel(4) == {"kind": "dotprod", "BS": 128, "VEC": 8}
    assert X._kernel_of([split]) == X.expected_kernel(9) == X.expected_kernel(11) == {"kind": "splitk", "SPLIT_K": 8, "TILE_K": 256,
                                                                                     "WARPS": 4}
    assert X.expected_kernel(5)["SPLIT_K"] == X.expected_kernel(16)["SPLIT_K"] == 6 and X.expected_kernel(17) == {"kind": "cublas"}
    assert {X.expected_kernel(M)["kind"] for M in X.ROUTER_M} == {"dotprod", "splitk", "cublas"}


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


def test_the_programs_bind_at_every_glm53_shape() -> None:
    got = X.bind_check()
    assert got["ok"] and len(got["bindings"]) == 1 + 2 + 2 + len(X.LINEAR) + len(X.BMM)
    assert got["bindings"]["Gemm_v2{K=6144,N=19360,DOT=HopperBF16WgmmaDot16_v1}"]
    assert got["bindings"]["LlBf16SplitKGemm_v1{N=256,K=6144,SPLIT_K=8,TILE_K=256,WARPS=4,DOT=HopperBF16WgmmaDot16_v1}"]


TOTALS = {"words": 10, "mismatch": 0, "dotprod_words": 4, "fma1_mismatch": 0, "fma0_mismatch": 2, "fma_differ": 2,
          "reference_coords": 3, "twin_mismatch": 0}
NONE = {k: 0 for k in TOTALS}


def _doc() -> dict:
    ok = {"deterministic": True, "operands_ok": True}
    launches = [{**ok, "phase": "router", "case": "mlp.gate", "M": 1, "kernel": {"kind": "dotprod"}, "dispatch_ok": True, "us": 4.0},
                {**ok, "phase": "linear", "case": "lm_head.shard", "M": 1, "us": 9.0},
                {**ok, "phase": "bmm", "case": "v_absorb", "M": 1, "us": 3.0}]
    comps = [{"phase": "router", "case": "mlp.gate", "M": 1, **NONE, "dotprod_words": 4, "fma0_mismatch": 2, "fma_differ": 2,
              "reference_coords": 1},
             {"phase": "linear", "case": "lm_head.shard", "M": 1, **NONE, "words": 6, "reference_coords": 1},
             {"phase": "bmm", "case": "v_absorb", "M": 1, **NONE, "words": 4, "reference_coords": 1}]
    doc = {"schema": X.SCHEMA, "gate": {"ok": True}, "errors": {}, "switch": {"ok": True}, "selection": {"ok": True}, "bind": {"ok": True},
           "launches": launches, "comparisons": comps, "totals": dict(TOTALS)}
    doc["fma"] = X.fma_of(doc["totals"])
    doc["cases"] = X.case_verdicts(doc)
    return doc


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("errors", "router/x"), "boom"), (("switch", "ok"), False),
                                        (("selection", "ok"), False), (("bind", "ok"), False), (("totals", "mismatch"), 1),
                                        (("totals", "reference_coords"), 0), (("totals", "twin_mismatch"), 1), (("fma",), None),
                                        (("launches", 0, "dispatch_ok"), False), (("launches", 1, "deterministic"), False),
                                        (("launches", 2, "operands_ok"), False)])
def test_the_verdict_needs_every_check(path: tuple, value) -> None:
    assert X._verdict(_doc())
    doc = copy.deepcopy(_doc())
    node = doc
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    assert not X._verdict(doc)
    assert not X._verdict({**_doc(), "launches": _doc()["launches"][:2]})      # a run without the bmm phase is not a GO


def test_each_case_has_its_own_verdict() -> None:
    doc = _doc()
    assert {k: c["ok"] for k, c in doc["cases"].items()} == {"router/mlp.gate/M1": True, "linear/lm_head.shard/M1": True,
                                                             "bmm/v_absorb/M1": True}
    doc["comparisons"][1]["mismatch"] = 1
    doc["errors"]["bmm/q_nope_absorb/M4/compare"] = "boom"
    doc["errors"]["bmm/v_absorb/M16"] = "boom"                                  # M16's error is not M1's
    doc["fma"] = 0                                                              # a dotprod launch whose words are not the run's fma
    got = X.case_verdicts(doc)
    assert {k: c["ok"] for k, c in got.items()} == {"router/mlp.gate/M1": False, "linear/lm_head.shard/M1": False,
                                                    "bmm/v_absorb/M1": True, "bmm/q_nope_absorb/M4": False, "bmm/v_absorb/M16": False}


def test_the_default_run_is_a_second_process(tmp_path, monkeypatch) -> None:
    """The second process gets `--served-default --no-default` and its own out directory; its cases that are not GO are the cases
    the default configuration changes, and the switch's cost is each case's time over the default's."""
    seen = {}

    def fake_run(argv, capture_output, text, cwd):
        seen["argv"] = argv
        sub = tmp_path / "served_default"
        sub.mkdir()
        cases = {"linear/lm_head.shard/M1": {"ok": False, "us": 6.0}, "bmm/v_absorb/M1": {"ok": True, "us": 3.0}}
        (sub / X.RECORD).write_text(json.dumps({"ok": False, "digest": "d", "errors": {}, "cases": cases}))
        return type("P", (), {"returncode": 1, "stdout": "out", "stderr": ""})()

    monkeypatch.setattr(X.subprocess, "run", fake_run)
    a = X.argparse.Namespace(workers=3, only="linear,bmm", min_driver=575, want_cc="9.0", want_sms=132, scratch=None, quick=True)
    got = X.default_run(a, tmp_path)
    assert {"--served-default", "--no-default", "--quick"} <= set(seen["argv"]) and seen["argv"][seen["argv"].index("--out") + 1] == \
        str(tmp_path / "served_default")
    assert got["changed"] == ["linear/lm_head.shard/M1"] and got["us"]["bmm/v_absorb/M1"] == 3.0
    doc = {"served_default": got, "cases": {"linear/lm_head.shard/M1": {"us": 9.0}, "bmm/v_absorb/M1": {"us": 3.0}}}
    assert X._cost(doc) == {"linear/lm_head.shard/M1": 1.5, "bmm/v_absorb/M1": 1.0}


def test_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys) -> None:
    """`main` with the GPU's pieces stood in for (gate, switch, selection, the launches as kernel words made here, the default run)
    and vLLM's config module stubbed: the comparison jobs through the spawned worker pool, the decided fma, the per-case verdicts,
    the sealed record and the exit status, so a slip in the orchestration shows here and not in the H200's first hour."""
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
    monkeypatch.setattr(X, "switch", lambda served_default: {"ok": True})
    monkeypatch.setattr(X, "make_gate", lambda cfg: object())
    monkeypatch.setattr(X, "selection", lambda gate: {"ok": True})
    monkeypatch.setattr(X, "cases", lambda quick, only: [("router", "mlp.gate", 5), ("router", "mlp.gate", 6), ("linear", "lm_head.shard", 5),
                                                         ("bmm", "v_absorb", 5)])
    jobs = {("router", 5): _job(tmp_path, "dotprod", fma=0), ("router", 6): _job(tmp_path, "splitk", M=6),
            ("linear", 5): _job(tmp_path, "linear", flip=(1, 2)), ("bmm", 5): _job(tmp_path, "bmm")}

    def launch(work, weights, gate, phase, name, M, seed):
        job = jobs[(phase, M)]
        rec = {"phase": phase, "case": name, "M": M, "K": job["K"], "N": job["N"], "deterministic": True, "operands_ok": True, "us": 2.0}
        if phase == "router":
            rec.update(kernel=job["kernel"], dispatch_ok=True)
        return rec, {**job, "case": name}

    monkeypatch.setattr(X, "launch", launch)
    monkeypatch.setattr(X, "default_run", lambda a, outdir: {"ok": False, "changed": ["router/mlp.gate/M6"], "us": {"router/mlp.gate/M5": 1.0}})
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1"]) == 1
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is False and not doc["errors"] and doc["fma"] == 0
    assert {k: c["ok"] for k, c in doc["cases"].items()} == {"router/mlp.gate/M5": True, "router/mlp.gate/M6": True,
                                                             "linear/lm_head.shard/M5": False, "bmm/v_absorb/M5": True}
    assert doc["totals"]["mismatch"] == 1 and doc["switch_cost"] == {"router/mlp.gate/M5": 2.0}
    assert doc["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":16:8" and doc["env"]["VLLM_BATCH_INVARIANT"] == "0"
    printed = capsys.readouterr().out
    assert "case linear/lm_head.shard/M5" in printed and "GLM53-BF16-GEMM NO-GO" in printed and "fma=0" in printed
    jobs[("linear", 5)] = _job(tmp_path, "linear")
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1", "--no-default"]) == 0
    assert "GLM53-BF16-GEMM GO" in capsys.readouterr().out


def test_the_record_is_a_registered_property() -> None:
    from verity_vllm.properties.record import seal

    doc = seal(_doc(), X._verdict(_doc()))
    assert REGISTRY["glm53_bf16_gemm_capture"].schema == X.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)
