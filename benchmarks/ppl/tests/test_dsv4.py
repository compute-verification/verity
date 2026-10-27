"""DeepSeek-V4's layers (`dsv4.py`) on a tiny V4-shaped checkpoint in DeepSeek's own names and formats: block-FP8
linears and FP4 routed experts with UE8M0 scales, one hash-routed and two top-k layers, an HCA, a CSA and a sliding
layer, mHC's four streams. HF parity per layer and end to end (embedding, hc head, final norm), with the indexer's
selection and with it skipped, the config derivations HF's, the dequantization HF's, the arms pass their gates and
reach every linear, and the refusals."""
import json
import shutil
from pathlib import Path

import dsv4
import model as M
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import write_safetensors
from test_engine import evaluate, passed, ppl

HID, HEADS, HD, ROPE, QL, OG, OL, VOCAB = 128, 8, 64, 16, 64, 2, 32, 256
E, TOPK, MOE, IH, ID = 8, 2, 64, 4, 32
CSA, HCA, WIN, BLOCK = 4, 16, 16, 32
KINDS = ["heavily_compressed_attention", "compressed_sparse_attention", "sliding_attention"]
MLPS = ["hash_moe", "moe", "moe"]
YARN = {"type": "yarn", "factor": 4, "original_max_position_embeddings": 16, "beta_fast": 32, "beta_slow": 1}


def tiny_dsv4(root: Path, index_topk: int = 4, seed: int = 0, **config) -> Path:
    """A random DeepSeek-V4-shaped checkpoint as DeepSeek ships it: 8 query heads of 64 on one KV head (RoPE on the
    last 16), q LoRA 64, o_a in 2 groups of 32, sliding window 16, HCA pooling 16 tokens, CSA 4 (its indexer 4 heads of
    32 keeping `index_topk`; at T = 64 its 16 entries are cut to 4 by default), 8 experts top-2 of 64 plus one shared,
    layer 0 hash-routed. FP8 linears in 32 x 32 blocks, FP4 experts in 1 x 32."""
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale=1.0):
        return torch.randn(*shape, generator=g) * scale

    def e8m0(e: torch.Tensor) -> torch.Tensor:
        return (e + 127).to(torch.uint8).view(torch.float8_e8m0fnu)

    def fp8(n, k, scale=1.0) -> tuple[torch.Tensor, torch.Tensor]:
        e = torch.randint(-10, -7, (n // BLOCK, k // BLOCK), generator=g)
        s = e.double().exp2().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
        return (rnd(n, k, scale=scale / k ** 0.5) / s).to(torch.float8_e4m3fn), e8m0(e)

    def fp4(n, k) -> tuple[torch.Tensor, torch.Tensor]:
        nib = torch.randint(0, 16, (n, k), generator=g, dtype=torch.uint8)
        packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).view(torch.int8)
        return packed, e8m0(torch.randint(-7, -4, (n, k // 32), generator=g))

    def bf(*shape, scale=1.0):
        return rnd(*shape, scale=scale / shape[-1] ** 0.5).to(torch.bfloat16)

    def nw(d):
        return (1 + 0.1 * rnd(d)).to(torch.bfloat16)

    t = {"embed.weight": rnd(VOCAB, HID).to(torch.bfloat16), "norm.weight": nw(HID), "head.weight": bf(VOCAB, HID),
         "hc_head_fn": rnd(4, 4 * HID, scale=0.5 / (4 * HID) ** 0.5), "hc_head_base": rnd(4, scale=0.5),
         "hc_head_scale": 1 + 0.1 * rnd(1)}
    t["embed.weight"][7, 3] = 60.0

    def put(name, pair):
        t[name + ".weight"], t[name + ".scale"] = pair

    for li, kind in enumerate(KINDS):
        p = f"layers.{li}."
        a = p + "attn."
        t[p + "attn_norm.weight"], t[p + "ffn_norm.weight"] = nw(HID), nw(HID)
        for b in ("attn", "ffn"):
            t[p + f"hc_{b}_fn"] = rnd(24, 4 * HID, scale=0.5 / (4 * HID) ** 0.5)
            t[p + f"hc_{b}_base"] = rnd(24, scale=0.5)
            t[p + f"hc_{b}_scale"] = 1 + 0.1 * rnd(3)
        t[a + "attn_sink"] = rnd(HEADS)
        t[a + "q_norm.weight"], t[a + "kv_norm.weight"] = nw(QL), nw(HD)
        put(a + "wq_a", fp8(QL, HID, 2.0))
        put(a + "wq_b", fp8(HEADS * HD, QL, 2.0))
        put(a + "wkv", fp8(HD, HID, 2.0))
        put(a + "wo_a", fp8(OG * OL, HEADS * HD // OG))
        put(a + "wo_b", fp8(HID, OG * OL))
        if kind != "sliding_attention":
            csa = kind == "compressed_sparse_attention"
            m, w = (CSA, 2 * HD) if csa else (HCA, HD)
            c = a + "compressor."
            t[c + "ape"] = rnd(m, w, scale=0.5)
            t[c + "wkv.weight"], t[c + "wgate.weight"] = bf(w, HID, scale=2.0), bf(w, HID, scale=2.0)
            t[c + "norm.weight"] = nw(HD)
            if csa:
                ix = a + "indexer."
                t[ix + "compressor.ape"] = rnd(CSA, 2 * ID, scale=0.5)
                t[ix + "compressor.wkv.weight"] = bf(2 * ID, HID, scale=2.0)
                t[ix + "compressor.wgate.weight"] = bf(2 * ID, HID, scale=2.0)
                t[ix + "compressor.norm.weight"] = nw(ID)
                put(ix + "wq_b", fp8(IH * ID, QL, 2.0))
                t[ix + "weights_proj.weight"] = bf(IH, HID, scale=4.0)
        f = p + "ffn."
        t[f + "gate.weight"] = bf(E, HID, scale=4.0)
        if MLPS[li] == "hash_moe":
            t[f + "gate.tid2eid"] = torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(VOCAB)])
        else:
            t[f + "gate.bias"] = rnd(E, scale=0.1)
        for e in range(E):
            put(f + f"experts.{e}.w1", fp4(MOE, HID))
            put(f + f"experts.{e}.w3", fp4(MOE, HID))
            put(f + f"experts.{e}.w2", fp4(HID, MOE))
        put(f + "shared_experts.w1", fp8(MOE, HID))
        put(f + "shared_experts.w3", fp8(MOE, HID))
        put(f + "shared_experts.w2", fp8(HID, MOE))
    cfg = {"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4", "hidden_size": HID,
           "num_hidden_layers": len(KINDS), "num_attention_heads": HEADS, "num_key_value_heads": 1, "head_dim": HD,
           "qk_rope_head_dim": ROPE, "q_lora_rank": QL, "o_groups": OG, "o_lora_rank": OL, "sliding_window": WIN,
           "index_n_heads": IH, "index_head_dim": ID, "index_topk": index_topk, "n_routed_experts": E,
           "num_experts_per_tok": TOPK, "moe_intermediate_size": MOE, "n_shared_experts": 1,
           "scoring_func": "sqrtsoftplus", "norm_topk_prob": True, "routed_scaling_factor": 2.5, "swiglu_limit": 2.0,
           "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "hidden_act": "silu", "rms_norm_eps": 1e-6,
           "vocab_size": VOCAB, "tie_word_embeddings": False,
           "max_position_embeddings": YARN["factor"] * YARN["original_max_position_embeddings"],
           "rope_theta": 10000, "compress_rope_theta": 160000, "rope_scaling": YARN, "layer_types": KINDS,
           "compress_rates": {"compressed_sparse_attention": CSA, "heavily_compressed_attention": HCA},
           "mlp_layer_types": MLPS, "num_nextn_predict_layers": 0, "torch_dtype": "bfloat16", "expert_dtype": "fp4",
           "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                                   "scale_fmt": "ue8m0", "weight_block_size": [BLOCK, BLOCK]}, **config}
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
def v4(tmp_path_factory) -> Path:
    return tiny_dsv4(tmp_path_factory.mktemp("dsv4"))


@pytest.fixture(scope="session")
def v4_all(tmp_path_factory) -> Path:
    """index_topk 16: every CSA query's top-k is every entry it may see, so the indexer is skipped (`compress`)."""
    return tiny_dsv4(tmp_path_factory.mktemp("dsv4_all"), index_topk=16)


def hf_config(snap: Path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("transformers.models.deepseek_v4")
    return transformers.AutoConfig.from_pretrained(str(snap))


@pytest.mark.parametrize("legacy", [False, True])
def test_the_config_derivations_are_hfs(v4, tmp_path, legacy):
    """Layer types, MLP types, compress rates and both RoPE parameter sets as HF's `DeepseekV4Config` derives them, from
    explicit fields or (V4-Pro's own config) the per-layer `compress_ratios` (one past the layers: the MTP's) and
    `num_hash_layers`."""
    snap = tmp_path / "snap"
    shutil.copytree(v4, snap)
    c = json.loads((snap / "config.json").read_text())
    if legacy:
        for k in ("layer_types", "compress_rates", "mlp_layer_types"):
            c.pop(k)
        c.update(num_hidden_layers=5, compress_ratios=[128, 128, 4, 128, 4, 0], num_hash_layers=3)
    (snap / "config.json").write_text(json.dumps(c))
    hc = hf_config(snap)
    f = dsv4.fields(c)
    assert f["layer_types"] == hc.layer_types and f["mlp_layer_types"] == hc.mlp_layer_types
    assert f["compress_rates"] == hc.compress_rates and f["rope_dim"] == hc.qk_rope_head_dim
    for k in ("main", "compress"):
        want = {n: v for n, v in hc.rope_parameters[k].items() if n not in ("partial_rotary_factor", "type")}
        assert {n: v for n, v in f["rope"][k].items() if n != "type"} == want, k


def test_dequant_is_hfs(v4):
    """FP8 E4M3 blocks and packed FP4 with UE8M0 scales, as the checkpoint ships them and the engine reads them (bytes),
    dequantize to HF's `Fp8Dequantize._dequantize_one`, in FP32 and in bf16."""
    pytest.importorskip("transformers")
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize
    W = M.Weights(v4)
    p = "model.layers.1."
    hf = Fp8Dequantize(None)
    for base in ("attn.wq_b", "attn.wo_a", "ffn.shared_experts.w2", "ffn.experts.3.w1", "ffn.experts.5.w2"):
        t = W.read([p + base + ".weight", p + base + ".scale"])
        q, s = t[p + base + ".weight"], t[p + base + ".scale"]
        assert s.dtype == torch.uint8
        hq = q if q.dtype == torch.int8 else q.view(torch.float8_e4m3fn)
        for dt in (torch.float32, torch.bfloat16):
            assert torch.equal(dsv4.dequant(q, s, dt), hf._dequantize_one(hq, s, output_dtype=dt)), (base, dt)
    W.close()


@pytest.mark.parametrize("which", ["v4", "v4_all"])
def test_hf_parity(which, request, ids):
    """Every layer is HF transformers' `DeepseekV4DecoderLayer` with eager attention and eager experts, bit for bit,
    with HF's RoPE tables and mask: with the indexer selecting 4 of 16 CSA entries, and with it skipped."""
    snap = request.getfixturevalue(which)
    hf_config(snap)
    r = evaluate(snap, ids[:2], [], hf_check=True)
    g = r["gates"]["hf_reference"]
    assert g["checked"] == len(KINDS) and g["failed"] == 0, g


def hf_model(snap: Path):
    """HF's `DeepseekV4ForCausalLM` holding the engine's weights (the layers' `hf_state` and the routed experts
    stacked)."""
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as hf
    hc = hf_config(snap)
    hc._attn_implementation = "eager"
    hc._experts_implementation = "eager"
    cfg = M.Config.read(snap)
    W = M.Weights(snap)
    top = W.read(M.top_names(cfg) + ["lm_head.weight"])
    sd = {"model.embed_tokens.weight": top["model.embed_tokens.weight"], "lm_head.weight": top["lm_head.weight"],
          "model.norm.weight": top["model.norm.weight"].float(),
          **{f"model.hc_head.hc_{p}": top[f"model.hc_head.{p}"] for p in ("fn", "base", "scale")}}
    for li in range(cfg.layers):
        layer = dsv4.Dsv4Layer(cfg, li, W.read(W.layer_names(li)), "cpu")
        sd.update({f"model.layers.{li}.{k}": v for k, v in dsv4.hf_state(layer).items()})
        for part in ("gate_up_proj", "down_proj"):
            sd[f"model.layers.{li}.mlp.experts.{part}"] = torch.stack(
                [layer.weight(f"mlp.experts.{e}.{part}") for e in range(E)])
    W.close()
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            m = hf.DeepseekV4ForCausalLM(hc)
    finally:
        torch.set_default_dtype(prev)
    res = m.load_state_dict(sd, strict=False, assign=True)
    assert not res.missing_keys and not res.unexpected_keys, res
    for mod in list(m.modules()):
        if hasattr(mod, "rotary_emb"):
            mod.rotary_emb = hf.DeepseekV4RotaryEmbedding(hc)
    return m.eval()


def test_the_bf16_stream_is_hfs_model(v4, ids):
    """The engine's NLL is HF's whole model's (the embedding's four streams, every layer, the hc head, the final norm,
    lm_head), token for token."""
    m = hf_model(v4)
    x = torch.from_numpy(ids[:2])
    with torch.no_grad():
        logits = m(input_ids=x).logits.float()
    want = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1), reduction="none").view(2, -1)
    r = evaluate(v4, ids[:2], [])
    got = r["nll"]["bf16/s0"]
    assert got.shape == (2, 63)
    assert np.array_equal(got, want.numpy()), float(np.abs(got - want.numpy()).max())
    assert 50 < ppl(got) < 5000


ARMS = ["int8-all", "adopted-bd-lm-pv", "adopted-bd-lm-pv1024"]


@pytest.mark.parametrize("which", ["v4", "v4_all"])
def test_the_arms_pass_their_gates(which, request, ids, calib):
    snap = request.getfixturevalue(which)
    r = evaluate(snap, ids[:4], ARMS, calib=calib)
    assert passed(r), r["gates"]
    g = r["gates"]
    assert g["int8_range"]["checked"] == g["accumulator"]["checked"] > 0 and g["product_exact"]["checked"] > 0
    assert g["peel"]["checked"] > 0
    bf = ppl(r["nll"]["bf16/s0"])
    for a in ARMS:
        assert abs(ppl(r["nll"][f"{a}/s0"]) / bf - 1) < 0.1, a
        assert r["divergence"][f"{a}/s0"][-1] > 0
        assert 0 < r["attn_rel_err"][f"{a}/s0"] < 0.2, a
    assert not np.array_equal(r["nll"]["int8-all/s0"], r["nll"]["adopted-bd-lm-pv/s0"])


@pytest.mark.parametrize("arm", ["int8-all", "adopted-bd-lm-pv"])
def test_every_linear_reaches_the_arm(v4, ids, calib, monkeypatch, arm):
    """An arm's steps reach q_a, q_b, kv, both o_a groups, o_b, the compressors' kv and gate (CSA: the indexer's, and
    its q_b), the shared expert's three and each routed expert's gate_up and down (routers, the indexer's weights_proj
    and mHC stay at the reference's precision); its attention sees each layer's window keys and compressed entries."""
    seen, attention = set(), []
    real, real_attn = dsv4.Dsv4Layer.linear, dsv4.Dsv4Layer.edited_attention

    def linear(self, name, x, run, collect=None):
        if run is not None and run.steps_for("linear", name):
            seen.add((self.li, ".".join(p for p in name.split(".") if not p.isdigit()), name))
        return real(self, name, x, run, collect)

    def edited(self, q, kv, vis, run):
        attention.append((self.li, tuple(q.shape), tuple(kv.shape), tuple(vis.shape)))
        return real_attn(self, q, kv, vis, run)

    monkeypatch.setattr(dsv4.Dsv4Layer, "linear", linear)
    monkeypatch.setattr(dsv4.Dsv4Layer, "edited_attention", edited)
    evaluate(v4, ids[:1], [arm], calib=calib)
    a, c, ix = "self_attn.", "self_attn.compressor.", "self_attn.compressor.indexer."
    base = {a + "q_a_proj", a + "q_b_proj", a + "kv_proj", a + "o_a_proj", a + "o_b_proj",
            "mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj", "mlp.shared_experts.down_proj",
            "mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
    want = {0: base | {c + "kv_proj", c + "gate_proj"},
            1: base | {c + "kv_proj", c + "gate_proj", ix + "kv_proj", ix + "gate_proj", ix + "q_b_proj"},
            2: base}
    for li in want:
        assert {n for at, n, _ in seen if at == li} == want[li], li
        assert {full for at, _, full in seen if at == li and "o_a_proj" in full} == {f"{a}o_a_proj.{j}" for j in range(OG)}
    T = ids.shape[1]
    entries = {0: T // HCA, 1: T // CSA, 2: 0}
    assert sorted(set(attention)) == [(li, (1, HEADS, T, HD), (1, 1, T + entries[li], HD), (T, T + entries[li]))
                                      for li in range(len(KINDS))]


def test_refusals(v4, ids, calib, tmp_path):
    with pytest.raises(ValueError, match="LUT softmax"):
        evaluate(v4, ids[:1], ["co32c-u16-cohalf-lut"], calib=calib)
    for over, match in (({"scoring_func": "sigmoid"}, "sqrtsoftplus"), ({"n_shared_experts": 2}, "shared expert")):
        snap = tmp_path / next(iter(over))
        shutil.copytree(v4, snap)
        c = json.loads((snap / "config.json").read_text())
        (snap / "config.json").write_text(json.dumps({**c, **over}))
        with pytest.raises(ValueError, match=match):
            evaluate(snap, ids[:1], [])
