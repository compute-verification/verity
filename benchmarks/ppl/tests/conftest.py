import json
import struct
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# the tiny model's tensors are too small for threads to pay off
torch.set_num_threads(1)

ST_DTYPES = {torch.bfloat16: "BF16", torch.float32: "F32", torch.float8_e4m3fn: "F8_E4M3"}


def write_safetensors(path: Path, tensors: dict) -> None:
    header, blobs, at = {}, [], 0
    for name, t in tensors.items():
        b = t.contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": ST_DTYPES[t.dtype], "shape": list(t.shape), "data_offsets": [at, at + len(b)]}
        blobs.append(b)
        at += len(b)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for b in blobs:
            f.write(b)


def tiny_snapshot(root: Path, arch: str = "llama", layers: int = 2, seed: int = 0) -> Path:
    """A random Llama-3.1-shaped (llama3 RoPE) or Qwen2.5-shaped (q/k/v biases, no BOS) model: hidden 128, 4 heads on 2
    KV heads of 32, MLP 224 (7 x 32, so the `q` rotation's odd factor is 7), vocab 256, token 7 a massive activation."""
    g = torch.Generator().manual_seed(seed)
    hid, heads, kv, hd, inter, vocab = 128, 4, 2, 32, 224, 256

    def w(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale / shape[-1] ** 0.5).to(torch.bfloat16)

    t = {"model.embed_tokens.weight": (torch.randn(vocab, hid, generator=g)).to(torch.bfloat16),
         "model.norm.weight": torch.ones(hid, dtype=torch.bfloat16), "lm_head.weight": w(vocab, hid)}
    t["model.embed_tokens.weight"][7, 3] = 60.0
    for li in range(layers):
        p = f"model.layers.{li}."
        t[p + "input_layernorm.weight"] = (1 + 0.1 * torch.randn(hid, generator=g)).to(torch.bfloat16)
        t[p + "post_attention_layernorm.weight"] = (1 + 0.1 * torch.randn(hid, generator=g)).to(torch.bfloat16)
        t[p + "self_attn.q_proj.weight"] = w(heads * hd, hid, scale=2.0)
        t[p + "self_attn.k_proj.weight"] = w(kv * hd, hid, scale=2.0)
        t[p + "self_attn.v_proj.weight"] = w(kv * hd, hid)
        t[p + "self_attn.o_proj.weight"] = w(hid, heads * hd)
        t[p + "mlp.gate_proj.weight"] = w(inter, hid)
        t[p + "mlp.up_proj.weight"] = w(inter, hid)
        t[p + "mlp.down_proj.weight"] = w(hid, inter)
        if arch == "qwen2":
            for n, d in (("q", heads * hd), ("k", kv * hd), ("v", kv * hd)):
                t[p + f"self_attn.{n}_proj.bias"] = (torch.randn(d, generator=g) * (2.0 if n == "k" else 0.2)).to(torch.bfloat16)
    cfg = {"architectures": ["LlamaForCausalLM" if arch == "llama" else "Qwen2ForCausalLM"],
           "model_type": arch, "hidden_size": hid, "num_hidden_layers": layers, "num_attention_heads": heads,
           "num_key_value_heads": kv, "head_dim": hd, "intermediate_size": inter, "rms_norm_eps": 1e-5,
           "vocab_size": vocab, "tie_word_embeddings": False, "max_position_embeddings": 4096, "hidden_act": "silu",
           "torch_dtype": "bfloat16"}
    if arch == "llama":
        cfg.update(rope_theta=500000.0, attention_bias=False, mlp_bias=False,
                   rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                 "original_max_position_embeddings": 256})
    else:
        cfg.update(rope_theta=1000000.0, use_sliding_window=False, sliding_window=4096, max_window_layers=layers)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(cfg))
    write_safetensors(root / "model.safetensors", t)
    return root


@pytest.fixture(autouse=True)
def deterministic_cpu():
    """Two of PyTorch's CPU kernels are not deterministic: SDPA's flash kernel, and oneDNN's bf16 GEMM behind F.linear
    (the same call twice can differ, by up to 3 on 2% of a 256 x 224 output; torch 2.9 and 2.14). The tests pin SDPA's
    math backend and turn oneDNN off, which leaves both deterministic. The GPU runs use neither."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH), torch.backends.mkldnn.flags(enabled=False):
        yield


@pytest.fixture(scope="session")
def llama(tmp_path_factory) -> Path:
    return tiny_snapshot(tmp_path_factory.mktemp("llama"), "llama")


@pytest.fixture(scope="session")
def qwen(tmp_path_factory) -> Path:
    return tiny_snapshot(tmp_path_factory.mktemp("qwen"), "qwen2")


@pytest.fixture(scope="session")
def ids():
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 256, (8, 64), generator=g)
    x[:, 0] = 7
    return x.numpy()


@pytest.fixture(scope="session")
def calib():
    g = torch.Generator().manual_seed(2)
    x = torch.randint(0, 256, (4, 64), generator=g)
    x[:, 0] = 7
    return x.numpy()
