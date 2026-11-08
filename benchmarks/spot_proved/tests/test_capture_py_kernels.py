"""`capture.py` records the kernels `PY_KERNELS` lists, which vLLM launches from Python past the dispatcher, without changing
what they return: each event binds the returns and the written buffers by SHA-512, and a paged cache only by shape. The
sparse backend's top-k-to-slot conversion is recorded with the inputs and both returns `stage.Wires.conversion` reads."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parents[1]
ATTN = "vllm.models.deepseek_v32.attention"
LL = "vllm.model_executor.kernels.linear.cute_dsl.ll_bf16"


def _rms(x, w, eps):
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def fused_norm_rope(positions, q_c, q_rms_norm_w, q_rms_eps, kv_c, kv_rms_norm_w, kv_rms_eps, k_pe, k_rope_cos_sin_cache,
                    index_k, index_k_layer_norm_w, index_k_layer_norm_bias, index_k_layer_norm_eps,
                    index_k_rope_cos_sin_cache, topk_indices_buffer, slot_mapping=None, indexer_k_cache=None,
                    mla_kv_cache=None, mla_kv_cache_dtype="auto", mla_k_scale=None, has_indexer=True,
                    index_rope_interleave=False, q_c_out=None, kv_c_out=None, k_pe_out=None, index_k_out=None):
    q_c_out = torch.empty_like(q_c) if q_c_out is None else q_c_out
    q_c_out.copy_(_rms(q_c, q_rms_norm_w, q_rms_eps))
    kv_c_out.copy_(_rms(kv_c, kv_rms_norm_w, kv_rms_eps))
    k_pe_out.copy_(k_pe.flip(-1))
    topk_indices_buffer[: q_c.shape[0]] = -1
    if mla_kv_cache is not None:
        mla_kv_cache.view(-1, mla_kv_cache.shape[-1])[slot_mapping] = torch.cat([kv_c_out, k_pe_out.flatten(1)], dim=-1)
    return q_c_out


def ll_bf16_gemm(hidden_states, router_weight, output_dtype=torch.float32):
    return (hidden_states.float() @ router_weight.float().T).to(output_dtype)


def _stub(monkeypatch, *, ll: bool) -> dict:
    mods = {}
    names = ["vllm", "vllm.models", "vllm.models.deepseek_v32", ATTN]
    if ll:
        names += ["vllm.model_executor", "vllm.model_executor.kernels", "vllm.model_executor.kernels.linear",
                  "vllm.model_executor.kernels.linear.cute_dsl", LL]
    for name in names:
        mods[name] = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mods[name])
    mods[ATTN].fused_norm_rope = fused_norm_rope
    if ll:
        mods[LL].ll_bf16_gemm = ll_bf16_gemm
    else:
        monkeypatch.delitem(sys.modules, LL, raising=False)
    return mods


def _capture(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("spot_capture_py_kernels", HERE / "capture.py")
    capture = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, capture)
    spec.loader.exec_module(capture)
    S = capture._State({"capture": str(tmp_path), "steps": [3], "mla_tokens": 0, "open": {}, "keep": "opened"})
    monkeypatch.setattr(capture, "S", S)
    return capture, S


def _sha(t) -> str:
    return hashlib.sha512(t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _norm_rope_call(mods):
    g = torch.Generator().manual_seed(7)
    n = 3
    q_c, kv_c = torch.randn(n, 16, generator=g).bfloat16(), torch.randn(n, 8, generator=g).bfloat16()
    k_pe = torch.randn(n, 1, 4, generator=g).bfloat16()
    cache = torch.zeros(4, 2, 12, dtype=torch.bfloat16)
    kv_c_out, k_pe_out = torch.empty_like(kv_c), torch.empty_like(k_pe)
    topk = torch.arange(8 * 4, dtype=torch.int32).view(8, 4)
    q = mods[ATTN].fused_norm_rope(
        torch.arange(n), q_c, torch.ones(16, dtype=torch.bfloat16) * 1.5, 1e-6, kv_c, torch.ones(8, dtype=torch.bfloat16), 1e-6,
        k_pe, torch.zeros(8, 4), None, None, None, 1e-6, None, topk, slot_mapping=torch.tensor([5, 0, 6]),
        indexer_k_cache=None, mla_kv_cache=cache, mla_kv_cache_dtype="auto", has_indexer=False, kv_c_out=kv_c_out,
        k_pe_out=k_pe_out, index_k_out=None)
    return q, kv_c_out, k_pe_out, topk, cache


def _router_call():
    g = torch.Generator().manual_seed(11)
    x, w = torch.randn(1, 16, generator=g).bfloat16(), torch.randn(5, 16, generator=g).bfloat16()
    return importlib.import_module(LL).ll_bf16_gemm(x, w)


def test_python_launched_kernels_are_recorded_with_their_outputs_and_return_the_same_bits(tmp_path, monkeypatch) -> None:
    mods = _stub(monkeypatch, ll=True)
    want_q, want_kv, want_pe, want_topk, want_cache = _norm_rope_call(mods)
    want_logits = _router_call()
    capture, S = _capture(tmp_path, monkeypatch)
    capture._install_py_kernels()
    assert mods[ATTN].fused_norm_rope is not fused_norm_rope and mods[LL].ll_bf16_gemm is not ll_bf16_gemm

    S.recording, S.layer = True, 4
    q, kv, pe, topk, cache = _norm_rope_call(mods)
    logits = _router_call()
    capture._drain()
    for got, want in ((q, want_q), (kv, want_kv), (pe, want_pe), (topk, want_topk), (cache, want_cache), (logits, want_logits)):
        assert torch.equal(got, want)

    norm, router = S.events
    assert (norm["op"], norm["layer"], router["op"]) == ("fused_norm_rope", 4, "ll_bf16_gemm")
    assert norm["in"]["mla_kv_cache"] == {"shape": [4, 2, 12], "dtype": "torch.bfloat16", "kv_rows": True}
    assert norm["in"]["indexer_k_cache"] is None and norm["in"]["q_rms_eps"] == 1e-6
    assert norm["in"]["slot_mapping"]["sha512"] == _sha(torch.tensor([5, 0, 6]))
    assert norm["out"]["r0"]["sha512"] == _sha(q)
    assert {k: norm["out"][k]["sha512"] for k in ("kv_c_out", "k_pe_out", "topk_indices_buffer")} == {
        "kv_c_out": _sha(kv), "k_pe_out": _sha(pe), "topk_indices_buffer": _sha(topk)}
    assert "index_k_out" not in norm["out"] and "q_c_out" not in norm["out"]
    assert set(router["in"]) == {"hidden_states", "router_weight"}
    assert router["out"]["r0"]["sha512"] == _sha(logits) and router["out"]["r0"]["dtype"] == "torch.float32"


def test_outside_a_recorded_forward_nothing_is_recorded(tmp_path, monkeypatch) -> None:
    mods = _stub(monkeypatch, ll=True)
    want = _norm_rope_call(mods)
    capture, S = _capture(tmp_path, monkeypatch)
    capture._install_py_kernels()
    for got, w in zip(_norm_rope_call(mods), want):
        assert torch.equal(got, w)
    _router_call()
    assert S.events == []


def convert(req_id, block_table, token_indices, BLOCK_SIZE=64, BLOCK_STRIDE_ROWS=None, NUM_TOPK_TOKENS=2048,
            return_valid_counts=False):
    ok = token_indices >= 0
    blk = block_table[req_id.long()[:, None], (token_indices // BLOCK_SIZE).clamp(min=0).long()]
    out = torch.where(ok, blk * (BLOCK_STRIDE_ROWS or BLOCK_SIZE) + token_indices % BLOCK_SIZE, -1).int()
    return (out, ok.sum(1).int()) if return_valid_counts else out


def test_the_sparse_backends_top_k_to_slot_conversion_is_recorded_with_its_inputs_slots_and_counts(tmp_path, monkeypatch) -> None:
    fams = "vllm.v1.attention.backends.mla.flashattn_mla_sparse"
    for name in ("vllm", "vllm.v1", "vllm.v1.attention", "vllm.v1.attention.backends", "vllm.v1.attention.backends.mla", fams):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules[fams].triton_convert_req_index_to_global_index = convert
    req, table = torch.tensor([0, 1, 1], dtype=torch.int32), torch.tensor([[3, 1], [2, 0]], dtype=torch.int32)
    topk = torch.tensor([[0, 5, -1, -1], [1, 0, -1, -1], [2, 1, 0, -1]], dtype=torch.int32)

    def call():
        return sys.modules[fams].triton_convert_req_index_to_global_index(
            req, table, topk, BLOCK_SIZE=4, BLOCK_STRIDE_ROWS=6, NUM_TOPK_TOKENS=4, return_valid_counts=True)
    want = call()
    capture, S = _capture(tmp_path, monkeypatch)
    capture._install_py_kernels()
    S.recording = True
    got = call()
    assert all(torch.equal(g, w) for g, w in zip(got, want))
    (ev,) = S.events
    assert ev["op"] == "triton_convert_req_index_to_global_index"
    assert {k: ev["in"][k]["sha512"] for k in ("req_id", "block_table", "token_indices")} == {
        "req_id": _sha(req), "block_table": _sha(table), "token_indices": _sha(topk)}
    assert (ev["in"]["BLOCK_SIZE"], ev["in"]["BLOCK_STRIDE_ROWS"], ev["in"]["return_valid_counts"]) == (4, 6, True)
    assert (ev["out"]["r0"]["sha512"], ev["out"]["r1"]["sha512"]) == (_sha(want[0]), _sha(want[1]))


def test_a_vllm_without_a_kernels_module_records_the_rest(tmp_path, monkeypatch, capsys) -> None:
    mods = _stub(monkeypatch, ll=False)
    capture, S = _capture(tmp_path, monkeypatch)
    capture._install_py_kernels()
    assert f"no {LL}" in capsys.readouterr().err
    S.recording = True
    _norm_rope_call(mods)
    assert [e["op"] for e in S.events] == ["fused_norm_rope"]
