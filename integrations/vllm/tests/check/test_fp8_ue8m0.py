"""check/fp8_ue8m0.py: the weights of record of vLLM's DeepGEMM linears are its load-time ue8m0 requant of the checkpoint's words, and
MLA's W_UK_T / W_UV the bf16 dequantisation of the requantised kv_b_proj -- each checked word for word against the same formulas in
torch (vLLM's `requant_weight_ue8m0_inplace` / `per_block_cast_to_fp8`, `MLAAttention.process_weights_after_loading`'s split and
permutes), whose float8 and bf16 casts are torch's own; and the Build's roles decide which fields take the rewrite."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity_vllm.check import fp8_ue8m0 as U  # noqa: E402

G = 128


def _checkpoint(rng, N, K):
    w = torch.from_numpy(np.clip(rng.standard_normal((N, K)) * 64, -440, 440).astype(np.float32)).to(torch.float8_e4m3fn)
    s = torch.from_numpy(rng.uniform(1e-4, 3e-3, (-(-N // G), K // G)).astype(np.float32))
    return w, s


def _torch_requant(w, s):
    N, K = w.shape
    full = s.repeat_interleave(G, 0).repeat_interleave(G, 1)[:N, :K]
    pad = torch.zeros((-(-N // G) * G, K), dtype=torch.float32)
    pad[:N] = w.to(torch.float32) * full
    view = pad.view(-1, G, K // G, G)
    amax = view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = torch.pow(2.0, torch.ceil(torch.log2((amax / 448.0).abs())))
    return (view * (1.0 / sf)).to(torch.float8_e4m3fn).view_as(pad)[:N].contiguous(), sf.view(view.size(0), view.size(2))


@pytest.mark.parametrize("N,K", [(256, 512), (2624 // 8, 384), (200, 256)])
def test_the_requant_is_vllms_word_for_word(N, K):
    w, s = _checkpoint(np.random.default_rng(N + K), N, K)
    got_w, got_s = U.requant(w.view(torch.uint8).numpy(), s.numpy().view(np.uint32), G)
    want_w, want_s = _torch_requant(w, s)
    assert np.array_equal(got_w, want_w.view(torch.uint8).numpy()) and np.array_equal(got_s, want_s.numpy().view(np.uint32))
    assert not np.array_equal(got_w, w.view(torch.uint8).numpy())
    assert np.all((got_s & 0x7FFFFF) == 0)


def test_the_absorbed_weights_are_mlas_split_of_the_requantised_kv_b():
    NH, DN, DV, L = 2, 192, 128, 256
    w, s = _checkpoint(np.random.default_rng(7), NH * (DN + DV), L)
    w.view(torch.uint8)[0, :8] = 0x80
    rw, rs = _torch_requant(w, s)
    deq = (rw.to(torch.float32) * rs.repeat_interleave(G, 0).repeat_interleave(G, 1)).to(torch.bfloat16)
    deq = torch.where(deq == 0, torch.zeros_like(deq), deq)
    W_UK, W_UV = deq.T.reshape(L, NH, DN + DV).split([DN, DV], dim=-1)
    qw, qs = U.requant(w.view(torch.uint8).numpy(), s.numpy().view(np.uint32), G)
    for which, want in (("W_UK_T", W_UK.permute(1, 2, 0)), ("W_UV", W_UV.transpose(0, 1))):
        got = U.absorbed(qw, qs, G, NH, DN, DV, which)
        assert got.shape == tuple(want.shape) and np.array_equal(got, want.contiguous().view(torch.int16).numpy().view(np.uint16))
    assert U.absorbed(qw, qs, G, NH, DN, DV, "W_UK_T")[0, 0, :8].tolist() == [0] * 8


def test_the_builds_roles_decide_which_fields_take_the_rewrite():
    L0 = "m.layers.0.self_attn"
    fields = [f"{L0}.q_b_proj.weight", f"{L0}.q_b_proj.weight_scale_inv", f"{L0}.kv_b_proj.weight", f"{L0}.kv_b_proj.weight_scale_inv",
              f"{L0}.W_UK_T", f"{L0}.W_UV", "m.layers.3.mlp.experts.routed_experts.w13_weight"]
    roles = {fields[0]: {"role": "fp8_w_requant_ue8m0"}, fields[1]: {"role": "fp8_ws_requant_ue8m0"}, fields[4]: {"role": "mla.absorb"},
             fields[5]: {"role": "mla.absorb"}, fields[6]: {"role": "moe.expert_weight"}}
    cfg = {"quantization_config": {"weight_block_size": [G, G]}}
    rw = U.Rewrites(fields, roles, lambda f: None, cfg)
    assert [rw.covers(f) for f in fields] == [True] * 6 + [False]
    assert not any(U.Rewrites(fields, {}, lambda f: None, cfg).covers(f) for f in fields)
    assert not U.Rewrites(fields, roles, lambda f: None, {}).covers(fields[0])
