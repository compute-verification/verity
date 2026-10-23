"""GLM-5.3's layers (`glm.py`) on a tiny block-FP8 GLM: the bf16 stream is the model (an independent float64 reference),
the int8 arm's weight codes are the recipe's, the arms pass their gates, HF parity, and the streamed weights."""
import hashlib
import json
import math
import shutil
from pathlib import Path

import glm
import model as M
import models
import numpy as np
import pytest
import spec
import steps
import stream
import torch
import torch.nn.functional as F
from conftest import write_safetensors
from test_engine import evaluate, passed, ppl

HID, HEADS, QL, KVL, NOPE, ROPE, VD, INTER, MOE, E, TOPK, VOCAB = 256, 4, 128, 128, 32, 32, 32, 256, 128, 8, 2, 256


def fp8_blocks(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w on E4M3 per block of 128 x 128 (scale amax / 448), as a block-FP8 checkpoint holds it: codes and scales."""
    n, k = w.shape
    s = torch.zeros(-(-n // 128), -(-k // 128))
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            s[i, j] = w[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128].abs().amax() / 448
    q = (w / s.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]).to(torch.float8_e4m3fn)
    return q, s


def tiny_glm(root: Path, layers: int = 3, seed: int = 0) -> Path:
    """A random GLM-5.3-shaped model: MLA (4 heads, q_lora 128, kv_lora 128, nope 32, rope 32, v 32), layer 0 dense
    (MLP 256), the rest 8 routed experts top-2 plus a shared one (128 each), every linear block-FP8 (kv_a's 160 rows
    end in a partial block), DSA indexers on layers 0 and 2 (index_topk 64, the tests' T), and an MTP layer the engine
    never reads."""
    g = torch.Generator().manual_seed(seed)

    def w(*shape, scale=1.0):
        return torch.randn(*shape, generator=g) * scale / shape[-1] ** 0.5

    t = {"model.embed_tokens.weight": torch.randn(VOCAB, HID, generator=g).to(torch.bfloat16),
         "model.norm.weight": torch.ones(HID, dtype=torch.bfloat16), "lm_head.weight": w(VOCAB, HID).to(torch.bfloat16)}
    t["model.embed_tokens.weight"][7, 3] = 60.0

    def fp8(name, *shape, scale=1.0):
        t[name + ".weight"], t[name + ".weight_scale_inv"] = fp8_blocks(w(*shape, scale=scale))

    def norm(name, d):
        t[name + ".weight"] = (1 + 0.1 * torch.randn(d, generator=g)).to(torch.bfloat16)

    kinds = ["dense"] + ["sparse"] * (layers - 1)
    indexers = ["full" if i % 2 == 0 else "shared" for i in range(layers)]
    for li in range(layers + 1):
        p = f"model.layers.{li}."
        norm(p + "input_layernorm", HID)
        norm(p + "post_attention_layernorm", HID)
        norm(p + "self_attn.q_a_layernorm", QL)
        norm(p + "self_attn.kv_a_layernorm", KVL)
        fp8(p + "self_attn.q_a_proj", QL, HID)
        fp8(p + "self_attn.q_b_proj", HEADS * (NOPE + ROPE), QL, scale=2.0)
        fp8(p + "self_attn.kv_a_proj_with_mqa", KVL + ROPE, HID, scale=2.0)
        fp8(p + "self_attn.kv_b_proj", HEADS * (NOPE + VD), KVL)
        fp8(p + "self_attn.o_proj", HID, HEADS * VD)
        if li < layers and indexers[li] == "full" or li == layers:
            fp8(p + "self_attn.indexer.wq_b", 2 * 64, QL)
            fp8(p + "self_attn.indexer.wk", 64, HID)
            t[p + "self_attn.indexer.weights_proj.weight"] = w(2, HID).to(torch.bfloat16)
            norm(p + "self_attn.indexer.k_norm", 64)
            t[p + "self_attn.indexer.k_norm.bias"] = (0.1 * torch.randn(64, generator=g)).to(torch.bfloat16)
        if li < layers and kinds[li] == "dense":
            for n, shape in (("gate_proj", (INTER, HID)), ("up_proj", (INTER, HID)), ("down_proj", (HID, INTER))):
                fp8(p + f"mlp.{n}", *shape)
            continue
        t[p + "mlp.gate.weight"] = w(E, HID, scale=4.0).to(torch.bfloat16)
        t[p + "mlp.gate.e_score_correction_bias"] = 0.05 * torch.randn(E, generator=g)
        for ex in [f"experts.{e}" for e in range(E)] + ["shared_experts"]:
            for n, shape in (("gate_proj", (MOE, HID)), ("up_proj", (MOE, HID)), ("down_proj", (HID, MOE))):
                fp8(p + f"mlp.{ex}.{n}", *shape)
    t[f"model.layers.{layers}.enorm.weight"] = torch.ones(HID, dtype=torch.bfloat16)
    cfg = {"architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa", "hidden_size": HID,
           "num_hidden_layers": layers, "num_attention_heads": HEADS, "num_key_value_heads": HEADS, "head_dim": 192,
           "intermediate_size": INTER, "moe_intermediate_size": MOE, "rms_norm_eps": 1e-5, "vocab_size": VOCAB,
           "tie_word_embeddings": False, "max_position_embeddings": 4096, "hidden_act": "silu", "dtype": "bfloat16",
           "q_lora_rank": QL, "kv_lora_rank": KVL, "qk_nope_head_dim": NOPE, "qk_rope_head_dim": ROPE,
           "v_head_dim": VD, "n_routed_experts": E, "num_experts_per_tok": TOPK, "n_group": 1, "topk_group": 1,
           "norm_topk_prob": True, "routed_scaling_factor": 2.5, "scoring_func": "sigmoid", "topk_method": "noaux_tc",
           "n_shared_experts": 1, "first_k_dense_replace": 1, "mlp_layer_types": kinds, "indexer_types": indexers,
           "index_topk": 64, "index_head_dim": 64, "index_n_heads": 2, "attention_bias": False,
           "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"}, "num_nextn_predict_layers": 1,
           "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
                                   "weight_block_size": [128, 128]}}
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
def glm_snap(tmp_path_factory) -> Path:
    return tiny_glm(tmp_path_factory.mktemp("glm"))


def float64_nll(snap, ids) -> np.ndarray:
    """The model in float64 from its FP8 codes and scales: MLA with dense causal attention, interleaved RoPE, the
    sigmoid router with its correction bias, the experts by their routing weights."""
    cfg = M.Config.read(snap)
    c = cfg.glm
    W = M.Weights(snap)
    raw = W.read(list(W.where))
    t = {}
    for n, x in raw.items():
        if n.endswith(".weight_scale_inv"):
            continue
        s = raw.get(n[:-len(".weight")] + ".weight_scale_inv") if n.endswith(".weight") else None
        t[n] = glm.dequant(x, s).double() if s is not None else x.double()
    B, T = ids.shape
    x = t["model.embed_tokens.weight"][torch.from_numpy(ids)]
    inv = 1.0 / (8000000 ** (torch.arange(0, ROPE, 2, dtype=torch.float64) / ROPE))
    f = torch.arange(T, dtype=torch.float64)[:, None] * inv[None]
    cos, sin = f.cos(), f.sin()

    def norm(h, w, eps):
        return w * h / torch.sqrt((h * h).mean(-1, keepdim=True) + eps)

    def rot(v):
        a, b = v[..., 0::2], v[..., 1::2]
        return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)

    def mlp(p, h):
        return (F.silu(h @ t[p + "gate_proj.weight"].T) * (h @ t[p + "up_proj.weight"].T)) @ t[p + "down_proj.weight"].T

    for li in range(cfg.layers):
        p = f"model.layers.{li}."
        a = p + "self_attn."
        h = norm(x, t[p + "input_layernorm.weight"], cfg.eps)
        qr = norm(h @ t[a + "q_a_proj.weight"].T, t[a + "q_a_layernorm.weight"], 1e-6)
        q = (qr @ t[a + "q_b_proj.weight"].T).view(B, T, HEADS, NOPE + ROPE).transpose(1, 2)
        kva = h @ t[a + "kv_a_proj_with_mqa.weight"].T
        kp = norm(kva[..., :KVL], t[a + "kv_a_layernorm.weight"], 1e-6)
        kv = (kp @ t[a + "kv_b_proj.weight"].T).view(B, T, HEADS, NOPE + VD).transpose(1, 2)
        k_rot = rot(kva[..., KVL:])[:, None].expand(B, HEADS, T, ROPE)
        qq = torch.cat([q[..., :NOPE], rot(q[..., NOPE:])], -1)
        kk = torch.cat([kv[..., :NOPE], k_rot], -1)
        s = (qq @ kk.transpose(-1, -2)) / math.sqrt(NOPE + ROPE)
        s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
        o = (torch.softmax(s, -1) @ kv[..., NOPE:]).transpose(1, 2).reshape(B, T, -1)
        x = x + o @ t[a + "o_proj.weight"].T
        h = norm(x, t[p + "post_attention_layernorm.weight"], cfg.eps)
        if c["mlp_layer_types"][li] == "dense":
            x = x + mlp(p + "mlp.", h)
            continue
        sc = torch.sigmoid(h @ t[p + "mlp.gate.weight"].T)
        idx = (sc + t[p + "mlp.gate.e_score_correction_bias"]).topk(TOPK, -1).indices
        wt = sc.gather(-1, idx)
        wt = wt / wt.sum(-1, keepdim=True) * 2.5
        out = mlp(p + "mlp.shared_experts.", h)
        for e in range(E):
            gate = (wt * (idx == e)).sum(-1, keepdim=True)
            out = out + gate * mlp(p + f"mlp.experts.{e}.", h)
        x = x + out
    logits = norm(x, t["model.norm.weight"], cfg.eps) @ t["lm_head.weight"].T
    return F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), torch.from_numpy(ids)[:, 1:].reshape(-1),
                           reduction="none").view(B, -1).numpy()


def test_the_bf16_stream_is_the_model(glm_snap, ids):
    r = evaluate(glm_snap, ids[:4], [])
    want = float64_nll(glm_snap, ids[:4])
    assert r["nll"]["bf16/s0"].shape == (4, 63)
    assert abs(ppl(r["nll"]["bf16/s0"]) / ppl(want) - 1) < 1e-2
    assert 50 < ppl(want) < 2000


def test_the_arms_pass_their_gates(glm_snap, ids):
    arms = ["int8-channel127-v3", "fp8-block", "int8-linears"]
    r = evaluate(glm_snap, ids[:4], arms)
    assert passed(r), r["gates"]
    g = r["gates"]
    assert g["int8_range"]["checked"] == g["accumulator"]["checked"] > 0 and g["product_exact"]["checked"] > 0
    bf = ppl(r["nll"]["bf16/s0"])
    for a in arms:
        assert abs(ppl(r["nll"][f"{a}/s0"]) / bf - 1) < 0.05, a
        assert r["divergence"][f"{a}/s0"][-1] > 0
    assert len({r["nll"][f"{a}/s0"].tobytes() for a in arms}) == 3


@pytest.mark.parametrize("arm, kv_b, head", [("int8-linears", True, False), ("int8-channel127-v3", False, True)])
def test_every_linear_reaches_the_arm(glm_snap, ids, monkeypatch, arm, kv_b, head):
    """An arm's steps reach every FP8 linear a token passes: attention's five, the dense MLP's three, the shared
    expert's three and each routed expert's gate_up and down; the router stays FP32. v3's arm leaves kv_b_proj bf16,
    as v3 stores it, and takes lm_head."""
    seen = set()
    real = glm.GlmLayer.linear

    def linear(self, name, x, run, collect=None):
        if run is not None and run.steps_for("linear", name):
            seen.add((self.li, name.split(".")[0] + "." + ".".join(p for p in name.split(".")[1:] if not p.isdigit())))
        return real(self, name, x, run, collect)

    monkeypatch.setattr(glm.GlmLayer, "linear", linear)
    evaluate(glm_snap, ids[:4], [arm])
    attn = {f"self_attn.{n}" for n in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj")} | (
        {"self_attn.kv_b_proj"} if kv_b else set())
    assert {n for li, n in seen if li == 0} == attn | {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
    assert {n for li, n in seen if li == 1} == attn | {"mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj",
                                                       "mlp.shared_experts.down_proj", "mlp.experts.gate_up_proj",
                                                       "mlp.experts.down_proj"}
    assert any(s.applies(0, "linear", spec.HEAD) for s in spec.load(arm).linear) == head


def v3_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`verity-int8-channel127/v3`'s row rule as vLLM's int8_checkpoint writes it, in numpy on the float64 weight."""
    floor = np.float64(np.float32(1e-30))
    s = (np.maximum(np.abs(w).max(axis=1), floor) / np.float64(127)).astype(np.float32)
    return np.clip(np.rint(w / s.astype(np.float64)[:, None]), -127, 127).astype(np.int8), s


def test_the_int8_codes_are_the_recipe(glm_snap, ids, monkeypatch):
    """The int8 arm's weight on a block-FP8 linear is `verity-int8-channel127/v3`'s: W = e4m3(q) s over the 128 x 128
    blocks, exactly (float64), s_row = RN32(max(amax_row, RN32(1e-30)) / 127), q = clip(rint(W / s_row), +-127); on
    lm_head the bf16 weight widened, the same rule. The arm's chain quantizes exactly those weights."""
    W = M.Weights(glm_snap)
    p = "model.layers.1.self_attn.kv_a_proj_with_mqa"
    raw = W.read([p + ".weight", p + ".weight_scale_inv", "lm_head.weight"])
    codes, s = raw[p + ".weight"], raw[p + ".weight_scale_inv"]
    e4m3 = codes.view(torch.float8_e4m3fn).float().numpy().astype(np.float64)
    w64 = np.empty(codes.shape)
    for i in range(codes.shape[0]):
        for j in range(codes.shape[1]):
            w64[i, j] = e4m3[i, j] * np.float64(s[i // 128, j // 128])
    exact = glm.dequant(codes, s, torch.float64)
    assert np.array_equal(exact.numpy(), w64)
    assert torch.equal(glm.dequant(codes, s), exact.float())
    for w in (exact, raw["lm_head.weight"].double()):
        q, sw = steps.exact_rows(w, 127)
        want_q, want_s = v3_rows(w.numpy())
        assert np.array_equal(q.numpy(), want_q) and np.array_equal(sw.numpy(), want_s) and q.abs().max() == 127
    tie = torch.tensor([[127.0, 0.5, 1.5, 2.5]], dtype=torch.float64)
    assert steps.exact_rows(tie, 127)[0].tolist() == [[127, 0, 2, 2]]

    quantized = []
    real = steps.exact_rows

    def spy(w, top):
        quantized.append(w)
        return real(w, top)

    monkeypatch.setattr(steps, "exact_rows", spy)
    evaluate(glm_snap, ids[:1], ["int8-channel127-v3"])
    assert quantized and all(w.dtype == torch.float64 for w in quantized)
    assert any(w.shape == exact.shape and torch.equal(w, exact) for w in quantized)
    assert any(torch.equal(w, raw["lm_head.weight"].double()) for w in quantized)


def test_g128_is_vllms_group_quant():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(5, 384, generator=g) * torch.tensor([1.0, 30.0, 0.0]).repeat_interleave(128)
    y = steps.fake_quant(x, "e4m3", "g128")
    for j in range(3):
        b = x[:, j * 128:(j + 1) * 128]
        s = b.abs().amax(-1, keepdim=True).clamp(min=1e-10) / 448
        assert torch.equal(y[:, j * 128:(j + 1) * 128], (b / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s)
    with pytest.raises(ValueError, match="fp.scale"):
        spec.parse({"linear": [{"step": "fp", "fmt": "e4m3", "scale": "g64"}]}, "x")


def test_refusals(glm_snap, ids):
    with pytest.raises(ValueError, match="index_topk"):
        evaluate(glm_snap, np.concatenate([ids[:1], ids[:1]], 1), [])
    with pytest.raises(ValueError, match="attention runs in bf16"):
        evaluate(glm_snap, ids[:1], ["int8-attention"])
    with pytest.raises(ValueError, match="int8.exact is v3's"):
        spec.parse({"linear": [{"step": "int8", "exact": True, "gptq": True}]}, "x")
    rotated = spec.parse({"linear": [{"step": "rotate"}, {"step": "int8", "exact": True}]}, "x")
    with pytest.raises(ValueError, match="int8.exact reads"):
        evaluate(glm_snap, ids[:1], [rotated])
    with pytest.raises(ValueError, match="linears are among"):
        spec.parse({"linear": [{"step": "int8", "linears": ["mlp.experts.3.down_proj"]}]}, "x")


def test_hf_parity(glm_snap, ids):
    """The engine's bf16 layers are HF transformers' GlmMoeDsaDecoderLayer with SDPA and eager experts, bit for bit,
    its indexers included (at T <= index_topk they keep every key)."""
    pytest.importorskip("transformers.models.glm_moe_dsa")
    r = evaluate(glm_snap, ids[:4], [], hf_check=True)
    g = r["gates"]["hf_reference"]
    assert g["checked"] == M.Config.read(glm_snap).layers and g["failed"] == 0, g


# ---------------------------------------------------------------- streaming

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def hub(glm_snap, tmp_path, monkeypatch):
    """`glm_snap` served as a pinned, streamed model `tiny-glm` from a local directory in place of the hub, and an
    HF home holding none of it."""
    pin = {"repo": "test/tiny-glm", "revision": "0" * 40, "files": {p.name: sha(p) for p in glm_snap.iterdir()}}
    (tmp_path / "pins").mkdir()
    (tmp_path / "pins" / "tiny-glm.json").write_text(json.dumps(pin))
    monkeypatch.setattr(models, "HERE", tmp_path)
    monkeypatch.setitem(models.MODELS, "tiny-glm", {"repo": pin["repo"], "revision": pin["revision"],
                                                    "tokenizer": "glm5", "gpus": 1, "pin": "tiny-glm.json"})
    pulled = []

    def curl(src, dst, *flags):
        name = src.rsplit("/", 1)[1]
        data = (glm_snap / name).read_bytes()
        if flags:
            lo, hi = map(int, flags[1].split("-"))
            data = data[lo:hi + 1]
        else:
            pulled.append(name)
        Path(dst).write_bytes(data)

    monkeypatch.setattr(stream, "curl", curl)
    home = tmp_path / "hf"
    snap = models.snapshot("tiny-glm", home)
    return pin, snap, pulled


def test_a_streamed_model_is_the_local_one(hub, glm_snap, ids):
    pin, snap, pulled = hub
    stream.ensure(pin, snap)
    assert sorted(p.name for p in snap.iterdir()) == ["config.json", "model.safetensors.index.json"]
    assert sorted(pulled) == ["config.json", "model.safetensors.index.json"]
    pulled.clear()
    assert models.pin_of(snap) == pin
    local = evaluate(glm_snap, ids[:4], ["int8-channel127-v3"])
    streamed = evaluate(snap, ids[:4], ["int8-channel127-v3"])
    for k in local["nll"]:
        assert np.array_equal(local["nll"][k], streamed["nll"][k]), k
    assert sorted(pulled) == sorted(n for n in pin["files"] if n.endswith(".safetensors"))
    assert not any(p.name.endswith(".safetensors") for p in snap.iterdir())


def test_shards_are_deleted_once_read(hub, glm_snap):
    pin, snap, _ = hub
    stream.ensure(pin, snap)
    W = M.Weights(snap, layers=3)
    names = [n for n in W.where if not n.startswith("model.layers.3.")]
    files = sorted(W.left)
    first = [n for n in names if W.where[n][0] == files[0]]
    W.read(first[:1])
    assert (W.hub.dir / files[0]).exists()
    W.read(first[1:])
    assert not (W.hub.dir / files[0]).exists()
    W.close()
    assert not W.hub.dir.exists()


def test_a_shard_off_its_pin_is_refused(hub, glm_snap):
    pin, snap, _ = hub
    stream.ensure(pin, snap)
    fn = min(n for n in pin["files"] if n.endswith(".safetensors"))
    pin["files"][fn] = "0" * 64
    (models.HERE / "pins" / "tiny-glm.json").write_text(json.dumps(pin))
    W = M.Weights(snap)
    n = next(n for n in W.where if W.where[n][0] == fn)
    with pytest.raises(ValueError, match="sha256"):
        W.read([n])
    W.close()


def test_ensure_replaces_a_wrong_small_file(hub, glm_snap):
    pin, snap, _ = hub
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    stream.ensure(pin, snap)
    assert sha(snap / "config.json") == pin["files"]["config.json"]
    shutil.rmtree(snap)
