"""spot_proved's staging of a step's bf16 cuBLAS GEMMs (`stage.cublas_instances`) on a fake capture, committed as `capture.py`
commits one: GLM-5.3's router gate (``aten::mm`` against the gate weight's transposed view, f32 out) and its indexer's wk +
weights projection (``aten::linear``, as vLLM's ``F.linear`` records, bf16 out) are staged per token row at the Definition the
Program binds at the call's M rows, and pass the match gate on words the route's numpy model computes; a call whose M is past
the route table is counted, with why, and the step's other calls are still staged; an ``aten::mm`` of two activations and a
biased ``aten::linear`` are counted, not staged."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.definitions.registry import bf16_gemm_check as BC

ROOT = Path(__file__).resolve().parents[3]
#: the gate's call is past the ll_bf16 tier (M > 16): cuBLAS's; T of its M rows are the step's real tokens
LAYER, K, M, T = 3, 6144, 20, 5
GATE, WK = f"model.layers.{LAYER}.mlp.gate.weight", f"model.layers.{LAYER}.self_attn.indexer.wk_weights_proj.weight"


def _stage():
    spec = importlib.util.spec_from_file_location("spot_stage", ROOT / "benchmarks" / "spot_proved" / "stage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bf16(rng, shape) -> np.ndarray:
    f = (rng.standard_normal(shape).astype(np.float32) * np.float32(0.5)).view(np.uint32)
    return ((f + np.uint32(0x8000)) >> np.uint32(16)).astype(np.uint16)


def _blob(d: Path, a: np.ndarray, dtype: str) -> dict:
    raw = np.ascontiguousarray(a).tobytes()
    h = hashlib.sha512(raw).hexdigest()
    p = d / "rank0" / "blobs" / h[:2] / h
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(raw)
    return {"sha512": h, "dtype": dtype, "shape": list(a.shape), "stride": [a.shape[1], 1]}


def _served(x: np.ndarray, w: np.ndarray, phase: str) -> np.ndarray:
    """The words cuBLAS's captured route at x's M rows computes (`bf16_gemm_check`'s models of each route)."""
    kernel, st = BC.default_cublas_route(phase, x.shape[0], x.shape[1], w.shape[0])
    out = "bf16" if phase == "linear" else "f32"
    if kernel == "splitk":
        return BC.cublas_splitk_rows(x, w, st["SPLIT_K"], out=out)
    return BC.linear_rows(x, w) if phase == "linear" else BC.f32out_rows(x, w)


def fake_capture(d: Path, calls: list[tuple[str, int, int]], flip: int | None = None, other: bool = False) -> list[int]:
    """Step 0 of a one-rank capture of ``calls``, each (weight, N, M): M bf16 rows (T of them the step's real tokens) against
    the opened weight, the gate's an ``aten::mm`` against its transposed view and wk's an ``aten::linear``, its output the
    route's words at M (f32 for the gate, bf16 for wk). A call at more M than the route table maps records only shapes.
    ``flip``: the call whose first output word is changed. ``other``: then an ``aten::mm`` of two activations and a biased
    ``aten::linear``, shapes only. Returns each call's seq."""
    rng = np.random.default_rng(11)
    ws = {GATE: _bf16(rng, (256, K)), WK: _bf16(rng, (160, K))}
    (d / "rank0" / "step0" / "open").mkdir(parents=True)
    opened = {}
    for name, w in ws.items():
        raw = w.tobytes()
        (d / "rank0" / "step0" / "open" / f"{name}.bin").write_bytes(raw)
        opened[name] = {"sha512": hashlib.sha512(raw).hexdigest(), "dtype": "torch.bfloat16", "shape": list(w.shape)}
    events = []
    for i, (name, N, M) in enumerate(calls):
        f32 = name == GATE
        dtype = "torch.float32" if f32 else "torch.bfloat16"
        mat2 = {"param": name, "shape": [K, N], "dtype": "torch.bfloat16", "stride": [1, K], "byte_offset": 0}
        if M > BC.CUBLAS_ROUTES["m_max"]:
            x = {"shape": [M, K], "dtype": "torch.bfloat16", "stride": [K, 1], "big": M * K * 2}
            y = {"shape": [M, N], "dtype": dtype, "stride": [N, 1], "big": M * N * (4 if f32 else 2)}
        else:
            xa = _bf16(rng, (M, K))
            ya = _served(xa, ws[name], "router" if f32 else "linear").astype(np.uint32 if f32 else np.uint16)
            if flip == i:
                ya.reshape(-1)[0] ^= 1
            x, y = _blob(d, xa, "torch.bfloat16"), _blob(d, ya, dtype)
        if name == GATE:
            events.append({"seq": i, "layer": LAYER, "parent": None, "op": "aten::mm", "in": {"self": x, "mat2": mat2},
                           "out": {"ret0": y}})
        else:
            weight = {"param": name, "shape": [N, K], "dtype": "torch.bfloat16"}
            events.append({"seq": i, "layer": LAYER, "parent": None, "op": "aten::linear",
                           "in": {"input": x, "weight": weight, "bias": None}, "out": {"ret0": y}})
    if other:
        a = {"shape": [M, 64], "dtype": "torch.bfloat16", "stride": [64, 1], "big": 0}
        events.append({"seq": len(events), "layer": LAYER, "parent": None, "op": "aten::mm",
                       "in": {"self": a, "mat2": {**a, "shape": [64, M], "stride": [1, 64]}}, "out": {"ret0": {**a, "shape": [M, M]}}})
        b = {"shape": [160], "dtype": "torch.bfloat16", "big": 0}
        events.append({"seq": len(events), "layer": LAYER, "parent": None, "op": "aten::linear",
                       "in": {"input": {**a, "shape": [M, K]}, "weight": {"param": WK, "shape": [160, K], "dtype": "torch.bfloat16"},
                              "bias": b}, "out": {"ret0": {**a, "shape": [M, 160]}}})
    events.append({"seq": len(events), "layer": None, "parent": None, "op": "model_output", "in": {},
                   "out": {"hidden": {"shape": [calls[0][2], K]}}})
    sd = d / "rank0" / "step0"
    ops = "".join(json.dumps(e) + "\n" for e in events).encode()
    pj = json.dumps({n: {"sha512": o["sha512"]} for n, o in opened.items()}).encode()
    (sd / "ops.jsonl").write_bytes(ops)
    (d / "rank0" / "params.json").write_bytes(pj)
    ops_sha, params_sha = hashlib.sha512(ops).hexdigest(), hashlib.sha512(pj).hexdigest()
    root = hashlib.sha512(f"verity/spot/rank-root\0{ops_sha}\0{params_sha}".encode()).hexdigest()
    (sd / "meta.json").write_text(json.dumps({"ops_sha512": ops_sha, "params_sha512": params_sha, "rank_root": root,
                                              "scheduled": {"req0": T}, "check": True}))
    step_root = hashlib.sha512(f"verity/spot/step-root\0{root}".encode()).hexdigest()
    (sd / "commit.json").write_text(json.dumps({"root": step_root, "coins": "00" * 64, "rank_roots": [root],
                                                "layers": {"moe": [LAYER]}, "opened": opened}))
    return list(range(len(calls) + 2 * other))


def _label(phase: str, M: int, N: int) -> str:
    """The staged label of a call at M rows: the Definition its route binds, as the Program binds it."""
    kernel, st = BC.default_cublas_route(phase, M, K, N)
    name = {("router", "splitk"): "CublasSplitKGemmF32Out_v1", ("router", "non_split"): "GemmF32Out_v1",
            ("linear", "splitk"): "CublasSplitKGemm_v1", ("linear", "non_split"): "Gemm_v2"}[(phase, kernel)]
    return f"{name}_K{K}_N{N}" + (f"_S{st['SPLIT_K']}" if kernel == "splitk" else "")


def _gate(stage, inst) -> bool:
    """`stage.main`'s match gate on one instance: its Definition evaluated on the captured inputs, against the captured outputs."""
    import a11  # on the path `stage.py` puts benchmarks/one_stage on

    _, spec, _, args_of, want_of = inst
    ev = a11.evaluate_units(spec, spec, [stage.words(a) for a in args_of()], 1, 4_000_000)
    got = np.asarray([int(v) for v in ev["call_out"]], dtype=np.uint64)
    return bool(np.array_equal(got, np.concatenate([stage.words(a) for a in want_of()])))


def _profile():
    from verity_vllm import TargetProfile
    return TargetProfile(compute_capability=(9, 0), batch_invariant=False)


@pytest.mark.slow
def test_the_router_gate_and_wk_stage_per_token_row_at_the_programs_definitions(tmp_path):
    S = _stage()
    fake_capture(tmp_path, [(GATE, 256, M), (WK, 160, M)])
    unstaged: dict = {}
    insts = list(S.cublas_instances(S.Capture(tmp_path, 0), _profile(), unstaged))
    assert unstaged == {} and len(insts) == 2 * T
    by = {i[0]: i for i in insts}
    assert set(by) == {_label("router", M, 256), _label("linear", M, 160)}
    for label, inst in by.items():
        assert inst[1].id.startswith(label.split("_K")[0] + "{") and f"K={K},N={inst[4]()[0].shape[0]}," in inst[1].id
        (x, w), (y,) = inst[3](), inst[4]()
        assert x.shape == (K,) and w.shape[1] == K and y.shape == (w.shape[0],)
        assert _gate(S, inst), label


@pytest.mark.slow
def test_a_flipped_output_word_fails_the_match_gate(tmp_path):
    S = _stage()
    fake_capture(tmp_path, [(WK, 160, M)], flip=0)
    insts = list(S.cublas_instances(S.Capture(tmp_path, 0), _profile()))
    assert [i[2][2] for i in insts] == list(range(T))
    assert not _gate(S, insts[0]) and _gate(S, insts[1])


def test_a_call_past_the_route_table_is_counted_and_the_step_still_stages_its_other_calls(tmp_path):
    S = _stage()
    past = BC.CUBLAS_ROUTES["m_max"] + 1
    seqs = fake_capture(tmp_path, [(WK, 160, M), (GATE, 256, past)])
    unstaged: dict = {}
    insts = list(S.cublas_instances(S.Capture(tmp_path, 0), _profile(), unstaged))
    assert {i[2][1] for i in insts} == {seqs[0]} and len(insts) == T
    (label, rec), = unstaged.items()
    assert label == f"CublasGemmF32Out_K{K}_N256" and rec["calls"] == [{"rank": 0, "seq": seqs[1], "rows": past}]
    assert rec["population"] == T
    assert f"no captured vLLM-default cuBLAS router route for M={past}, K={K}, N=256" in rec["why"]
    assert f"maps M to {BC.CUBLAS_ROUTES['m_max']}" in rec["why"]


@pytest.mark.slow
def test_every_call_gates_each_cublas_call_and_names_any_other_mm(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("spot_stage_cublas_every", ROOT / "benchmarks" / "spot_proved" / "stage.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    monkeypatch.setitem(sys.modules, "spot_stage_cublas_every", S)  # the pool pickles its task by module name
    cap, out = tmp_path / "cap", tmp_path / "out"
    seqs = fake_capture(cap, [(GATE, 256, M), (WK, 160, M)], other=True)
    monkeypatch.setattr(sys, "argv", ["stage.py", "--capture", str(cap), "--step", "0", "--out", str(out),
                                      "--every-call", "1", "--jobs", "1"])
    assert S.main() == 0
    rep = json.loads((out / "gate.json").read_text())
    for label in (_label("router", M, 256), _label("linear", M, 160)):
        t = rep["types"][label]
        assert (t["calls"], t["calls_gated"], t["agree"], t["disagree"], t["errors"]) == (1, 1, 1, 0, 0), label
    assert rep["unstaged"]["AtenMm_other"]["calls"] == [{"rank": 0, "seq": seqs[-2]}]
    assert rep["unstaged"]["AtenLinear_other"]["calls"] == [{"rank": 0, "seq": seqs[-1]}]
    assert rep["read_ops"] == {"aten::linear": 1, "aten::mm": 1}
    assert rep["not_gated_ops"] == {"aten::linear": 1, "aten::mm": 1}
