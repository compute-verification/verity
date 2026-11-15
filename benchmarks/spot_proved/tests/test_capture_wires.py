"""A step `capture.py` records wires ``q_b_proj``'s input in `stage.Wires`: GLM-5.3's ``q_a_layernorm`` runs inside
``fused_norm_rope``, a kernel launched from Python that no dispatcher sees, and its return (the normed query latent) is what
``q_b_proj`` and the indexer's ``wq_b`` read. `PY_KERNELS` records that return, so each reader's input is wired to it (a
``call``: no population gates ``fused_norm_rope``, so it is committed, not proved); without that record it is `Unwired`."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parents[1]
ATTN = "vllm.models.deepseek_v32.attention"


def fused_norm_rope(positions, q_c, q_rms_norm_w, q_rms_eps, kv_c, kv_rms_norm_w, kv_rms_eps, k_pe, k_rope_cos_sin_cache,
                    index_k, index_k_layer_norm_w, index_k_layer_norm_bias, index_k_layer_norm_eps,
                    index_k_rope_cos_sin_cache, topk_indices_buffer, slot_mapping=None, indexer_k_cache=None,
                    mla_kv_cache=None, mla_kv_cache_dtype="auto", mla_k_scale=None, has_indexer=True,
                    index_rope_interleave=False, q_c_out=None, kv_c_out=None, k_pe_out=None, index_k_out=None):
    from torch.utils._python_dispatch import _disable_current_modes
    with _disable_current_modes():
        q = q_c.float()
        q = (q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + q_rms_eps) * q_rms_norm_w.float()).to(q_c.dtype)
        kv_c_out.copy_(kv_c)
        k_pe_out.copy_(k_pe)
        return q


def _module(monkeypatch, name: str, file: str):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "setrecursionlimit", lambda n: None)
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


def _sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


def _record(tmp_path, monkeypatch, *, py_kernels: bool):
    """One forward's slice (the normed query latent, then ``q_b_proj`` and ``wq_b`` reading it) recorded by `capture.py`,
    written as a committed step the way its ``_finish`` writes one."""
    attn = types.ModuleType(ATTN)
    attn.fused_norm_rope = fused_norm_rope
    for name in ("vllm", "vllm.models", "vllm.models.deepseek_v32"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, ATTN, attn)
    capture = _module(monkeypatch, "spot_capture_wires", "capture.py")
    S = capture._State({"capture": str(tmp_path), "steps": [0], "mla_tokens": 0, "open": {}, "keep": "all", "min_free_gib": 0})
    S.step = 0
    monkeypatch.setattr(capture, "S", S)
    if py_kernels:
        capture._install_py_kernels()

    g = torch.Generator().manual_seed(5)
    qkv_a = torch.randn(4, 24, generator=g).bfloat16()
    q_c, kv_c, k_pe = qkv_a[:, :16], qkv_a[:, 16:22], qkv_a[:, 22:].unsqueeze(1)
    q_b, wq_b = (torch.randn(16, 8, generator=g).bfloat16() for _ in range(2))
    S.recording, S.layer = True, 0
    with capture.Recorder():
        q = attn.fused_norm_rope(torch.arange(4), q_c, torch.ones(16, dtype=torch.bfloat16), 1e-6, kv_c,
                                 torch.ones(6, dtype=torch.bfloat16), 1e-6, k_pe, None, None, None, None, 1e-6, None,
                                 torch.full((4, 2), -1, dtype=torch.int32), kv_c_out=torch.empty_like(kv_c),
                                 k_pe_out=torch.empty_like(k_pe))
        torch.mm(q, q_b)
        torch.mm(q, wq_b)
    S.recording = False

    sd = tmp_path / "rank0" / "step0"
    sd.mkdir(parents=True, exist_ok=True)
    ops = "".join(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n" for e in S.events).encode()
    pj = b"{}"
    (sd / "ops.jsonl").write_bytes(ops)
    (sd / "params.json").write_bytes(pj)
    root = _sha(f"verity/spot/rank-root\0{_sha(ops)}\0{_sha(pj)}".encode())
    (sd / "meta.json").write_text(json.dumps({"scheduled": {"req": 4}, "ops_sha512": _sha(ops), "params_sha512": _sha(pj),
                                              "rank_root": root, "check": {}}))
    (sd / "commit.json").write_text(json.dumps({"root": _sha(f"verity/spot/step-root\0{root}".encode()), "coins": "00" * 64,
                                                "rank_roots": [root], "layers": {"dense": [0]}, "opened": {}}))
    stage = _module(monkeypatch, "spot_stage_capture_wires", "stage.py")
    cap = stage.Capture(tmp_path, 0)
    return stage, cap, stage.Wires(cap)


def test_q_b_proj_and_wq_b_read_the_recorded_fused_norm_rope_return(tmp_path, monkeypatch) -> None:
    stage, cap, wires = _record(tmp_path, monkeypatch, py_kernels=True)
    ev = cap.events[0]
    norm = [e for e in ev if e["op"] == "fused_norm_rope"]
    readers = [e for e in ev if e["op"] == "aten::mm"]
    assert len(norm) == 1 and len(readers) == 2
    for e in readers:
        assert wires.source(0, e["in"]["self"]) == {
            "kind": "call", "rank": 0, "seq": norm[0]["seq"], "op": "fused_norm_rope", "out": "r0",
            "into": {"seq": e["seq"], "op": "aten::mm", "in": "self"}}
    assert {"kv_c_out", "k_pe_out"} <= set(norm[0]["out"]), "the step's own latent and RoPE key rows are recorded too"


def test_without_the_fused_norm_rope_record_q_b_projs_input_is_unwired(tmp_path, monkeypatch) -> None:
    stage, cap, wires = _record(tmp_path, monkeypatch, py_kernels=False)
    ev = cap.events[0]
    assert not [e for e in ev if e["op"] == "fused_norm_rope"]
    with pytest.raises(stage.Unwired, match="reads self"):
        wires.source(0, next(e for e in ev if e["op"] == "aten::mm")["in"]["self"])
