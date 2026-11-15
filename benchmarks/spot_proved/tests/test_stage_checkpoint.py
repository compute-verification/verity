"""`stage.py --checkpoint`: a weight the capture didn't keep is composed from the checkpoint and read only when its SHA-512 is the one
the step's ``params.json`` committed (any other composition is an error, never "not kept"), and the DeepGEMM linears and MLA's
absorbed weights take vLLM's load-time rewrite while the routed experts don't."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
GOOD = np.arange(6, dtype=np.uint16).reshape(2, 3)


def _stage(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "setrecursionlimit", lambda n: None)
    spec = importlib.util.spec_from_file_location("spot_stage", HERE / "stage.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cap(stage, composed: np.ndarray, checkpoint: str | None = "ckpt"):
    cap = stage.Capture.__new__(stage.Capture)
    cap.d, cap.step, cap.ranks, cap.checkpoint = Path("/nonexistent"), 7, [0], checkpoint
    cap.commit = {0: {"opened": {}}}
    cap.params = {0: {"w": {"shape": [2, 3], "dtype": "torch.bfloat16", "sha512": hashlib.sha512(GOOD.tobytes()).hexdigest()}}}
    cap._params, cap._ck, cap._lock, cap._composers = {}, None, threading.Lock(), {0: lambda n: composed}
    return cap


def test_a_composed_weight_is_read_only_at_its_committed_digest(monkeypatch) -> None:
    stage = _stage(monkeypatch)
    assert np.array_equal(_cap(stage, GOOD.copy()).param(0, "w"), GOOD)
    with pytest.raises(LookupError, match="not the committed bytes") as bad:
        _cap(stage, GOOD + 1).param(0, "w")
    assert not str(bad.value).startswith("not kept")
    with pytest.raises(LookupError, match="not in an opened layer"):
        _cap(stage, GOOD.copy(), checkpoint=None).param(0, "w")


def test_the_deepgemm_linears_and_mla_absorb_take_the_load_rewrite_and_the_routed_experts_do_not(monkeypatch) -> None:
    stage = _stage(monkeypatch)
    e4, f32, bf = "torch.float8_e4m3fn", "torch.float32", "torch.bfloat16"
    L = "model.layers.3."
    params = {L + "self_attn.q_b_proj.weight": e4, L + "self_attn.q_b_proj.weight_scale_inv": f32,
              L + "self_attn.W_UK_T": bf, L + "self_attn.W_UV": bf, L + "mlp.shared_experts.down_proj.weight": e4,
              L + "mlp.shared_experts.down_proj.weight_scale_inv": f32, L + "mlp.experts.w13_weight": e4,
              L + "mlp.experts.w13_weight_scale_inv": f32, L + "mlp.gate.weight": bf, L + "input_layernorm.weight": bf}
    roles = {k: v["role"] for k, v in stage.load_roles({k: {"dtype": d} for k, d in params.items()}).items()}
    assert roles == {L + "self_attn.q_b_proj.weight": "fp8_w_requant_ue8m0", L + "self_attn.q_b_proj.weight_scale_inv": "fp8_ws_requant_ue8m0",
                     L + "self_attn.W_UK_T": "mla.absorb", L + "self_attn.W_UV": "mla.absorb",
                     L + "mlp.shared_experts.down_proj.weight": "fp8_w_requant_ue8m0",
                     L + "mlp.shared_experts.down_proj.weight_scale_inv": "fp8_ws_requant_ue8m0"}


def _safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    hdr, blobs, off = {}, [], 0
    for name, (dt, a) in tensors.items():
        b = np.ascontiguousarray(a).tobytes()
        hdr[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(hdr).encode()
    path.write_bytes(len(h).to_bytes(8, "little") + h + b"".join(blobs))


def test_a_decode_step_composes_mla_absorb_from_kv_b_proj_it_never_read(monkeypatch, tmp_path) -> None:
    """A decode-only step's parameters hold W_UK_T and W_UV but not kv_b_proj, which only prefill reads: each rank composes them from
    its rows of the checkpoint's kv_b_proj, requantised, at the digests the step committed."""
    from verity_vllm.check.fp8_ue8m0 import absorbed, requant
    stage = _stage(monkeypatch)
    G, world, NH, DN, DV, KV = 2, 2, 2, 2, 2, 4
    rng = np.random.default_rng(7)
    words = (rng.integers(0, 0x7F, (world * NH * (DN + DV), KV)) | rng.integers(0, 2, (world * NH * (DN + DV), KV)) << 7).astype(np.uint8)
    scales = rng.uniform(2 ** -8, 2 ** -4, (words.shape[0] // G, KV // G)).astype(np.float32)
    L = "model.layers.3.self_attn."
    _safetensors(tmp_path / "model.safetensors", {L + "kv_b_proj.weight": ("F8_E4M3", words),
                                                  L + "kv_b_proj.weight_scale_inv": ("F32", scales)})
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"weight_block_size": [G, G]},
                                                      "qk_nope_head_dim": DN, "v_head_dim": DV}))

    def want(rank: int) -> dict[str, np.ndarray]:
        rows = slice(rank * NH * (DN + DV), (rank + 1) * NH * (DN + DV))
        w, s = requant(words[rows], scales[rows.start // G:rows.stop // G].view(np.uint32), G)
        return {f: absorbed(w, s, G, NH, DN, DV, f) for f in ("W_UK_T", "W_UV")}

    rank1 = want(1)
    assert not np.array_equal(rank1["W_UK_T"], want(0)["W_UK_T"])
    bf, e4, f32 = "torch.bfloat16", "torch.float8_e4m3fn", "torch.float32"
    cap = stage.Capture.__new__(stage.Capture)
    cap.d, cap.step, cap.ranks, cap.checkpoint = Path("/nonexistent"), 7, [0, 1], str(tmp_path)
    cap.commit = {1: {"opened": {}}}
    cap.params = {1: {L + f: {"shape": list(a.shape), "dtype": bf, "sha512": hashlib.sha512(a.tobytes()).hexdigest()}
                      for f, a in rank1.items()}}
    cap.params[1] |= {L + "q_b_proj.weight": {"shape": [4, 4], "dtype": e4, "sha512": ""},
                      L + "q_b_proj.weight_scale_inv": {"shape": [2, 2], "dtype": f32, "sha512": ""}}
    cap._params, cap._ck, cap._lock, cap._composers = {}, None, threading.Lock(), {}
    for f, a in rank1.items():
        assert np.array_equal(cap.param(1, L + f), a)
