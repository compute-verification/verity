"""DeepSeek-V2's layers (`dsv2.py`) on a tiny DeepSeek-V2-Lite-shaped model: the bf16 stream is the model (an
independent float64 reference), the arms pass their gates and reach every linear, HF parity, the refusals, and
`scale_error`, the V4 proxy's step."""
import json
import math
import shutil
import types
from pathlib import Path

import dsv2
import gates as G
import matmul
import model as M
import numpy as np
import pytest
import spec
import torch
import torch.nn.functional as F
from conftest import write_safetensors
from test_engine import evaluate, passed, ppl

HID, HEADS, NOPE, ROPE, KVL, VD, INTER, MOE, E, TOPK, SHARED, VOCAB = 128, 4, 32, 16, 64, 32, 256, 64, 8, 2, 2, 256
YARN = {"type": "yarn", "factor": 4.0, "mscale": 0.707, "mscale_all_dim": 0.707, "original_max_position_embeddings": 32,
        "beta_fast": 32, "beta_slow": 1}
ROUTED_SCALE = 1.5


def tiny_dsv2(root: Path, layers: int = 3, seed: int = 0, **config) -> Path:
    """A random DeepSeek-V2-Lite-shaped model: MLA with no q LoRA (4 heads, kv_lora 64, nope 32, rope 16, v 32, so Q
    and K are 48 wide and an arm pads them to 64), YaRN RoPE (factor 4 past 32 positions, Lite's mscales), layer 0
    dense (MLP 256), the rest 8 routed experts top-2 by greedy softmax plus two shared ones (64 each)."""
    g = torch.Generator().manual_seed(seed)

    def w(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale / shape[-1] ** 0.5).to(torch.bfloat16)

    def norm(d):
        return (1 + 0.1 * torch.randn(d, generator=g)).to(torch.bfloat16)

    t = {"model.embed_tokens.weight": torch.randn(VOCAB, HID, generator=g).to(torch.bfloat16),
         "model.norm.weight": torch.ones(HID, dtype=torch.bfloat16), "lm_head.weight": w(VOCAB, HID)}
    t["model.embed_tokens.weight"][7, 3] = 60.0
    for li in range(layers):
        p = f"model.layers.{li}."
        a = p + "self_attn."
        t[p + "input_layernorm.weight"], t[p + "post_attention_layernorm.weight"] = norm(HID), norm(HID)
        t[a + "kv_a_layernorm.weight"] = norm(KVL)
        t[a + "q_proj.weight"] = w(HEADS * (NOPE + ROPE), HID, scale=2.0)
        t[a + "kv_a_proj_with_mqa.weight"] = w(KVL + ROPE, HID, scale=2.0)
        t[a + "kv_b_proj.weight"] = w(HEADS * (NOPE + VD), KVL)
        t[a + "o_proj.weight"] = w(HID, HEADS * VD)
        if li == 0:
            mlps = {"mlp.": INTER}
        else:
            t[p + "mlp.gate.weight"] = w(E, HID, scale=4.0)
            mlps = {**{f"mlp.experts.{e}.": MOE for e in range(E)}, "mlp.shared_experts.": MOE * SHARED}
        for m, n in mlps.items():
            t[p + m + "gate_proj.weight"] = w(n, HID)
            t[p + m + "up_proj.weight"] = w(n, HID)
            t[p + m + "down_proj.weight"] = w(HID, n)
    cfg = {"architectures": ["DeepseekV2ForCausalLM"], "model_type": "deepseek_v2", "hidden_size": HID,
           "num_hidden_layers": layers, "num_attention_heads": HEADS, "num_key_value_heads": HEADS,
           "intermediate_size": INTER, "moe_intermediate_size": MOE, "rms_norm_eps": 1e-6, "vocab_size": VOCAB,
           "tie_word_embeddings": False, "max_position_embeddings": 128, "hidden_act": "silu",
           "torch_dtype": "bfloat16", "q_lora_rank": None, "kv_lora_rank": KVL, "qk_nope_head_dim": NOPE,
           "qk_rope_head_dim": ROPE, "v_head_dim": VD, "n_routed_experts": E, "num_experts_per_tok": TOPK,
           "n_shared_experts": SHARED, "first_k_dense_replace": 1, "moe_layer_freq": 1, "topk_method": "greedy",
           "scoring_func": "softmax", "routed_scaling_factor": ROUTED_SCALE, "norm_topk_prob": False, "n_group": 1,
           "topk_group": 1, "rope_theta": 10000, "rope_scaling": YARN, "attention_bias": False, **config}
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(cfg))
    names = sorted(t)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
    for fn, ns in shards.items():
        write_safetensors(root / fn, {n: t[n] for n in ns})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {n: fn for fn, ns in shards.items() for n in ns}}))
    return root


@pytest.fixture(scope="session")
def v2(tmp_path_factory) -> Path:
    return tiny_dsv2(tmp_path_factory.mktemp("dsv2"))


def yarn_rope(T: int) -> tuple[torch.Tensor, torch.Tensor]:
    """YaRN's angles for positions 0..T-1 (DeepSeek-V2's yarn_find_correction_range, linear ramp), float64."""
    f, base, omax = YARN["factor"], 10000.0, YARN["original_max_position_embeddings"]
    pos = base ** (torch.arange(0, ROPE, 2, dtype=torch.float64) / ROPE)

    def corr(rot):
        return ROPE * math.log(omax / (rot * 2 * math.pi)) / (2 * math.log(base))

    lo, hi = max(math.floor(corr(YARN["beta_fast"])), 0), min(math.ceil(corr(YARN["beta_slow"])), ROPE - 1)
    ramp = ((torch.arange(ROPE // 2, dtype=torch.float64) - lo) / (hi - lo)).clamp(0, 1)
    inv = (1 / (f * pos)) * ramp + (1 / pos) * (1 - ramp)
    ang = torch.arange(T, dtype=torch.float64)[:, None] * inv[None]
    return ang.cos(), ang.sin()


def float64_nll(snap, ids) -> np.ndarray:
    """The model in float64: MLA with dense causal attention at YaRN's scale, RoPE on interleaved pairs, the softmax
    router's top-k weights times the routed scale, the shared experts."""
    W = M.Weights(snap)
    t = {k: v.double() for k, v in W.read(list(W.where)).items()}
    B, T = ids.shape
    x = t["model.embed_tokens.weight"][torch.from_numpy(ids)]
    cos, sin = yarn_rope(T)
    mscale = 0.1 * YARN["mscale_all_dim"] * math.log(YARN["factor"]) + 1
    scale = mscale * mscale / math.sqrt(NOPE + ROPE)

    def norm(h, w, eps=1e-6):
        return w * h / torch.sqrt((h * h).mean(-1, keepdim=True) + eps)

    def rot(v):
        a, b = v[..., 0::2], v[..., 1::2]
        return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2)

    def mlp(p, h):
        return (F.silu(h @ t[p + "gate_proj.weight"].T) * (h @ t[p + "up_proj.weight"].T)) @ t[p + "down_proj.weight"].T

    for li in range(M.Config.read(snap).layers):
        p = f"model.layers.{li}."
        a = p + "self_attn."
        h = norm(x, t[p + "input_layernorm.weight"])
        q = (h @ t[a + "q_proj.weight"].T).view(B, T, HEADS, NOPE + ROPE).transpose(1, 2)
        kva = h @ t[a + "kv_a_proj_with_mqa.weight"].T
        kv = (norm(kva[..., :KVL], t[a + "kv_a_layernorm.weight"]) @ t[a + "kv_b_proj.weight"].T)
        kv = kv.view(B, T, HEADS, NOPE + VD).transpose(1, 2)
        qq = torch.cat([q[..., :NOPE], rot(q[..., NOPE:])], -1)
        kk = torch.cat([kv[..., :NOPE], rot(kva[..., KVL:])[:, None].expand(B, HEADS, T, ROPE)], -1)
        s = (qq @ kk.transpose(-1, -2)) * scale
        s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
        o = (torch.softmax(s, -1) @ kv[..., NOPE:]).transpose(1, 2).reshape(B, T, -1)
        x = x + o @ t[a + "o_proj.weight"].T
        h = norm(x, t[p + "post_attention_layernorm.weight"])
        if li == 0:
            x = x + mlp(p + "mlp.", h)
            continue
        wt, idx = torch.softmax(h @ t[p + "mlp.gate.weight"].T, -1).topk(TOPK, -1)
        out = mlp(p + "mlp.shared_experts.", h)
        for e in range(E):
            out = out + (wt * (idx == e)).sum(-1, keepdim=True) * ROUTED_SCALE * mlp(p + f"mlp.experts.{e}.", h)
        x = x + out
    logits = norm(x, t["model.norm.weight"]) @ t["lm_head.weight"].T
    return F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), torch.from_numpy(ids)[:, 1:].reshape(-1),
                           reduction="none").view(B, -1).numpy()


def test_the_bf16_stream_is_the_model(v2, ids):
    r = evaluate(v2, ids[:4], [])
    want = float64_nll(v2, ids[:4])
    assert r["nll"]["bf16/s0"].shape == (4, 63)
    assert abs(ppl(r["nll"]["bf16/s0"]) / ppl(want) - 1) < 1e-2
    assert 50 < ppl(want) < 2000


ARMS = ["int8-all", "adopted-bd-lm-pv", "adopted-bd-lm-pv1024"]


def test_the_arms_pass_their_gates(v2, ids, calib):
    r = evaluate(v2, ids[:4], ARMS, calib=calib)
    assert passed(r), r["gates"]
    g = r["gates"]
    assert g["int8_range"]["checked"] == g["accumulator"]["checked"] > 0 and g["product_exact"]["checked"] > 0
    assert g["peel"]["checked"] > 0
    bf = ppl(r["nll"]["bf16/s0"])
    for a in ARMS:
        assert abs(ppl(r["nll"][f"{a}/s0"]) / bf - 1) < 0.1, a
        assert r["divergence"][f"{a}/s0"][-1] > 0
    assert not np.array_equal(r["nll"]["int8-all/s0"], r["nll"]["adopted-bd-lm-pv/s0"])
    # 64 keys are one block of 1024
    assert np.array_equal(r["nll"]["adopted-bd-lm-pv/s0"], r["nll"]["adopted-bd-lm-pv1024/s0"])


@pytest.mark.parametrize("arm", ["int8-all", "adopted-bd-lm-pv"])
def test_every_linear_reaches_the_arm(v2, ids, calib, monkeypatch, arm):
    """An arm's steps reach attention's four linears, the dense MLP's three, the shared experts' three and each routed
    expert's gate_up and down (the router stays FP32), and its attention steps run; v3's arm takes lm_head."""
    seen, attention = set(), []
    real, real_attn = dsv2.Dsv2Layer.linear, dsv2.Dsv2Layer.edited_attention

    def linear(self, name, x, run, collect=None):
        if run is not None and run.steps_for("linear", name):
            seen.add((self.li, ".".join(p for p in name.split(".") if not p.isdigit())))
        return real(self, name, x, run, collect)

    def edited(self, query, key, value, run):
        attention.append((self.li, query.shape[-1], key.shape[-1], value.shape[-1]))
        return real_attn(self, query, key, value, run)

    monkeypatch.setattr(dsv2.Dsv2Layer, "linear", linear)
    monkeypatch.setattr(dsv2.Dsv2Layer, "edited_attention", edited)
    evaluate(v2, ids[:1], [arm], calib=calib)
    attn = {f"self_attn.{n}" for n in ("q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj")}
    assert {n for li, n in seen if li == 0} == attn | {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
    for li in (1, 2):
        assert {n for at, n in seen if at == li} == attn | {
            "mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj", "mlp.shared_experts.down_proj",
            "mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
    assert sorted(set(attention)) == [(li, 64, 64, VD) for li in range(3)]
    assert any(s.applies(0, "linear", spec.HEAD) for s in spec.load(arm).linear) == (arm != "int8-all")


def test_hf_parity(v2, ids):
    """The engine's bf16 layers are HF transformers' DeepseekV2DecoderLayer with SDPA and eager experts, bit for bit,
    and its rotary table HF's."""
    pytest.importorskip("transformers.models.deepseek_v2")
    r = evaluate(v2, ids[:4], [], hf_check=True)
    g = r["gates"]["hf_reference"]
    assert g["checked"] == M.Config.read(v2).layers and g["failed"] == 0, g


def test_refusals(v2, ids, calib, tmp_path):
    with pytest.raises(ValueError, match="LUT softmax"):
        evaluate(v2, ids[:1], ["co32c-u16-cohalf-lut"], calib=calib)
    for over, match in (({"q_lora_rank": 32}, "q_lora_rank"), ({"topk_method": "group_limited_greedy"}, "greedy"),
                        ({"scoring_func": "sigmoid"}, "greedy softmax")):
        snap = tmp_path / next(iter(over))
        shutil.copytree(v2, snap)
        c = json.loads((snap / "config.json").read_text())
        (snap / "config.json").write_text(json.dumps({**c, **over}))
        with pytest.raises(ValueError, match=match):
            evaluate(snap, ids[:1], [])


def chain(x, w, kind, *pre):
    """The int8 chain on one matmul with the steps `pre` before it."""
    arm = spec.parse({kind: [*pre, {"step": "int8"}]}, "x")
    run = matmul.Run(types.SimpleNamespace(name="t"), 0, 1, types.SimpleNamespace(li=0), G.Gates(), {}, {}, None)
    steps = arm.linear if kind == "linear" else arm.attention
    if kind == "linear":
        mm = matmul.MM("linear", "mlp.up_proj", run, x, w)
    else:
        s, d = x.shape[-2], x.shape[-1]
        causal = torch.ones(s, s, dtype=torch.bool).tril()
        mm = matmul.MM("qk", "qk", run, x, w, cache={}, extra={"group": 0, "rows": (0, s), "causal": causal,
                                                                "rep": x.shape[0], "s": s, "d": d})
    return matmul.run_chain(steps, mm), run.gates


@pytest.mark.parametrize("kind", ["linear", "attention"])
def test_scale_error_scales_the_rest_of_the_chains_error(kind):
    """scale_error(f) first in a chain is y0 + f (y - y0): y the chain's product, y0 the FP32 product of its operands;
    factor 1 is the chain itself, bit for bit, and a negative factor is refused."""
    g = torch.Generator().manual_seed(4)
    x, w = (torch.randn(16, 64, generator=g), torch.randn(8, 64, generator=g)) if kind == "linear" else \
        (torch.randn(2, 16, 64, generator=g), torch.randn(16, 64, generator=g))
    y0 = x @ w.T
    y, gates = chain(x, w, kind)
    assert gates.passed and not torch.equal(y, y0)
    key = "factor" if kind == "linear" else "qk"
    for f in (0.0, 0.5, 2.5):
        got, _ = chain(x, w, kind, {"step": "scale_error", key: f})
        assert torch.allclose(got - y0, f * (y - y0), atol=1e-4), f
    assert torch.equal(chain(x, w, kind, {"step": "scale_error", key: 1.0})[0], y)
    with pytest.raises(ValueError, match=">= 0"):
        spec.parse({kind: [{"step": "scale_error", key: -1.0}]}, "x")
