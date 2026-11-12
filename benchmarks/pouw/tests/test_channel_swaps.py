"""`pearl_c_vllm/channel_swaps.py` on stand-ins: a tiny random Llama whose function its swaps keep, the pinned search on
rows with planted massive channels, its float64 screen against the exact per-row rules, the BF16 safetensors views it
writes the permuted checkpoint through, and its tokenizer on a toy byte-level vocabulary."""
from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parents[1] / "pearl_c_vllm" / "channel_swaps.py"
spec = importlib.util.spec_from_file_location("channel_swaps", HERE)
CS = sys.modules.setdefault("channel_swaps", importlib.util.module_from_spec(spec))
spec.loader.exec_module(CS)

SHAPE = CS.Shape(layers=2, hidden=64, heads=4, kv_heads=2, head_dim=16, inter=128, eps=1e-5, theta=500000.0,
                 scaling={"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                          "original_max_position_embeddings": 8192}, tied=True)


def tiny_weights(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    s, w = SHAPE, {}

    def r(*shape):
        return (torch.randn(*shape, generator=g) * 0.2).to(torch.bfloat16)

    w["model.embed_tokens.weight"] = r(100, s.hidden)
    w["model.norm.weight"] = (1 + r(s.hidden)).to(torch.bfloat16)
    for L in range(s.layers):
        p = f"model.layers.{L}."
        w[p + "input_layernorm.weight"] = (1 + r(s.hidden)).to(torch.bfloat16)
        w[p + "post_attention_layernorm.weight"] = (1 + r(s.hidden)).to(torch.bfloat16)
        w[p + "self_attn.q_proj.weight"] = r(s.heads * s.head_dim, s.hidden)
        w[p + "self_attn.k_proj.weight"] = r(s.kv_heads * s.head_dim, s.hidden)
        w[p + "self_attn.v_proj.weight"] = r(s.kv_heads * s.head_dim, s.hidden)
        w[p + "self_attn.o_proj.weight"] = r(s.hidden, s.heads * s.head_dim)
        w[p + "mlp.gate_proj.weight"] = r(s.inter, s.hidden)
        w[p + "mlp.up_proj.weight"] = r(s.inter, s.hidden)
        w[p + "mlp.down_proj.weight"] = r(s.hidden, s.inter)
    return w


DOC = {"shape": SHAPE.__dict__, "spaces": {
    "hidden:-1": {"kind": "hidden", "layer": -1, "swaps": [[0, 3, 8], [0, 61, 16]]},
    "intermediate:1": {"kind": "intermediate", "layer": 1, "swaps": [[0, 5, 40]]},
    "attention:0": {"kind": "attention", "layer": 0, "swaps": [[1, 6, 0], [0, 13, 8]]}}}


def test_the_swaps_keep_the_function_and_move_each_channel():
    w = tiny_weights()
    ids = [1, 7, 42, 9, 3, 77, 15]
    with torch.no_grad():
        before = CS.forward(w, SHAPE, ids, torch.float64)
        cap0: dict = {}
        CS.forward(w, SHAPE, ids, torch.float64, cap0)
        CS.apply(w, SHAPE, DOC)
        after = CS.forward(w, SHAPE, ids, torch.float64)
        cap1: dict = {}
        CS.forward(w, SHAPE, ids, torch.float64, cap1)
    assert float((before - after).abs().max()) <= 1e-12 * float(before.abs().max())
    for kind, L in (("qkv", 0), ("gate_up", 1), ("down", 1), ("o", 0), ("lm_head", -1)):
        p = CS.column_perm(DOC, SHAPE, kind, L)
        assert not np.array_equal(p, np.arange(len(p)))
        np.testing.assert_allclose(cap1[(kind, L)][0], cap0[(kind, L)][0][:, p], rtol=1e-2, atol=1e-2)


def test_a_wrong_swap_changes_the_function():
    w = tiny_weights()
    ids = [1, 7, 42, 9]
    with torch.no_grad():
        before = CS.forward(w, SHAPE, ids, torch.float64)
        t = w["model.layers.0.mlp.down_proj.weight"]
        t[:, [5, 40]] = t[:, [40, 5]].clone()
        after = CS.forward(w, SHAPE, ids, torch.float64)
    assert float((before - after).abs().max()) > 1e-6


def _rows(n: int, k: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, k)).astype(np.float32).astype(np.float64).astype(np.float32)


def test_the_float64_screen_is_the_exact_rules():
    k = 256
    x = _rows(40, k, 1)
    x[:10, 13] = 2000.0           # off the stride, s/ρ over 444: the floor fails
    x[10:20, 16] = 2000.0         # on the stride: ρ sees it
    x[20:30, 1:6] = 20.0          # five loud entries, over k/64: liveness fails
    x = torch.tensor(x).to(torch.bfloat16).float().numpy()
    got = CS.failing(x)
    assert list(got) == [not CS.exact_passes(r) for r in x]
    assert got[:10].all() and not got[10:20].any() and got[20:30].all() and not got[30:].any()


def test_the_search_moves_the_massive_channel_onto_the_stride():
    sp = CS.Space("hidden", -1, 256, 1, 1, 256)
    x = _rows(64, 256, 2)
    x[0::16, 13] = 3000.0
    x[8::16, 101] = -2500.0
    x = torch.tensor(x).to(torch.bfloat16).float().numpy()
    swaps, before = CS.search(sp, x)
    assert before == 8 and {u for _, u, _ in swaps} == {13, 101} and all(t % 8 == 0 for _, _, t in swaps)
    assert not CS.failing(x[:, CS.perm_of(sp, swaps)]).any()
    assert all(CS.exact_passes(r) for r in x[:, CS.perm_of(sp, swaps)])


def test_an_attention_swap_moves_the_dim_in_every_head_of_its_kv_head():
    sp = CS.Space("attention", 0, 4 * 16, 2, 2, 16)
    p = CS.perm_of(sp, [(1, 6, 0)])
    assert [(i, int(p[i])) for i in range(64) if p[i] != i] == [(32, 38), (38, 32), (48, 54), (54, 48)]


def test_the_safetensors_views_write_the_files_bytes(tmp_path):
    a = torch.arange(6, dtype=torch.float32).to(torch.bfloat16).view(2, 3)
    raw = a.view(torch.uint8).numpy().tobytes()
    header = json.dumps({"x": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, len(raw)]}}).encode()
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + raw)
    data, h, base = CS.read_safetensors(path)
    t = CS.tensors(data, h, base)["x"]
    assert torch.equal(t, a)
    t[:, [0, 2]] = t[:, [2, 0]].clone()
    assert torch.equal(CS.tensors(data, h, base)["x"], a[:, [2, 1, 0]])


def test_the_tokenizer_merges_by_rank_and_round_trips(tmp_path):
    chars = CS._byte_chars()
    vocab = {c: i for i, c in enumerate(chars[b] for b in range(256))}
    sp = chars[ord(" ")]
    vocab.update({"ab": 256, sp + "ab": 257, "abc": 258})
    doc = {"model": {"vocab": vocab, "merges": ["a b", sp + " ab", "ab c"], "ignore_merges": True},
           "added_tokens": [{"id": 300, "content": "<|begin_of_text|>"}]}
    (tmp_path / "tokenizer.json").write_text(json.dumps(doc))
    tok = CS.Tokenizer(tmp_path / "tokenizer.json")
    assert tok.encode("abc ab, x") == [300, 258, 257, vocab[","], vocab[sp], vocab["x"]]
    with pytest.raises(ValueError):
        tok.encode("caf\u00e9")
