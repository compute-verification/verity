"""Small open causal LMs in plain torch, with one coordinate of one activation set to a chosen value (a fault).

GPT-2 (learned positions, LayerNorm, tanh-GELU MLP) and Qwen2 (RoPE, RMSNorm, GQA, SwiGLU), read from a Hugging Face
snapshot's safetensors and pinned by revision (`MODELS`). Each forward takes a KV cache of its prefix, so a batch of
faulted sequences shares one honest prefix, and every activation a fault can hit is a named site:

  emb (layer -1)  the residual stream into layer 0 (token plus position embedding)
  attn  l         layer l's attention output, after its output projection and before the residual add
  mlph  l         layer l's MLP hidden activation, the input of its down (c_proj) projection
  mlp   l         layer l's MLP output, after its down projection and before the residual add
  resid l         the residual stream after layer l
  final (layer L) the final norm's output, the unembedding's input

A `Fault` replaces site[b, pos, coord[b]] by value[b] for each faulted batch row b, in the one forward that computes
position `pos`; everything after it is honest. Rows of one batch may fault different sites. Values are rounded to bf16
first: a fault is 16 bits.
"""
from __future__ import annotations

import json
import math
import os
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

MODELS = {
    "gpt2": {"repo": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e"},
    "qwen2.5-0.5b": {"repo": "Qwen/Qwen2.5-0.5B", "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987"},
}
FILES = ("config.json", "model.safetensors", "tokenizer.json")
SITES = ("emb", "attn", "mlph", "mlp", "resid", "final")
ST_DTYPES = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}


def cache_dir() -> Path:
    return Path(os.environ.get("LEVERAGE_CACHE", Path.home() / ".cache" / "verity-leverage"))


def fetch(name: str) -> Path:
    """The snapshot's files under the cache, downloaded once from the hub at the pinned revision."""
    m = MODELS[name]
    d = cache_dir() / name / m["revision"]
    d.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        p = d / f
        if not p.exists():
            url = f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/{f}"
            tmp = p.with_suffix(p.suffix + ".part")
            with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as out:
                while chunk := r.read(1 << 22):
                    out.write(chunk)
            tmp.rename(p)
    return d


def read_safetensors(path: Path) -> dict[str, torch.Tensor]:
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    base = 8 + n
    out = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        t = torch.frombuffer(bytearray(raw[base + a:base + b]), dtype=ST_DTYPES[v["dtype"]])
        out[k] = t.reshape(v["shape"])
    return out


@dataclass(frozen=True)
class Spec:
    arch: str
    d: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    ff: int
    vocab: int
    eps: float
    rope_theta: float


def spec_of(cfg: dict) -> Spec:
    if cfg["model_type"] == "gpt2":
        d, h = cfg["n_embd"], cfg["n_head"]
        return Spec("gpt2", d, cfg["n_layer"], h, h, d // h, 4 * d, cfg["vocab_size"], cfg["layer_norm_epsilon"], 0.0)
    if cfg["model_type"] == "qwen2":
        d, h = cfg["hidden_size"], cfg["num_attention_heads"]
        if not cfg.get("tie_word_embeddings", False):
            raise ValueError("only tied Qwen2 checkpoints: the unembedding is the token embedding")
        return Spec("qwen2", d, cfg["num_hidden_layers"], h, cfg["num_key_value_heads"], d // h,
                    cfg["intermediate_size"], cfg["vocab_size"], cfg["rms_norm_eps"], cfg["rope_theta"])
    raise ValueError(f"model_type {cfg['model_type']!r}")


@dataclass
class Fault:
    """site[row, pos, coord] := value for each (site, layer) -> (rows, coords, values), in the forward computing pos."""
    pos: int
    rows: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

    @classmethod
    def of(cls, pos: int, faults: list[tuple[str, int, int, float]]) -> "Fault":
        """One fault per batch row: faults[b] = (site, layer, coord, value)."""
        by: dict[tuple[str, int], list[tuple[int, int, float]]] = {}
        for b, (site, layer, coord, value) in enumerate(faults):
            if site not in SITES:
                raise ValueError(f"site {site!r} not in {SITES}")
            by.setdefault((site, layer), []).append((b, coord, value))
        rows = {}
        for key, xs in by.items():
            r, c, v = zip(*xs)
            rows[key] = (torch.tensor(r), torch.tensor(c), torch.tensor(v, dtype=torch.float32).to(torch.bfloat16))
        return cls(pos, rows)


class Cache:
    """Per layer (K, V), each (B, kv_heads, S, head_dim)."""

    def __init__(self, kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None):
        self.kv = kv or []

    @property
    def length(self) -> int:
        return self.kv[0][0].shape[2] if self.kv else 0

    def expand(self, b: int) -> "Cache":
        return Cache([(k.expand(b, -1, -1, -1), v.expand(b, -1, -1, -1)) for k, v in self.kv])

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if layer < len(self.kv):
            k0, v0 = self.kv[layer]
            if k0.shape[0] != k.shape[0]:
                k0, v0 = k0.expand(k.shape[0], -1, -1, -1), v0.expand(k.shape[0], -1, -1, -1)
            k, v = torch.cat([k0, k], 2), torch.cat([v0, v], 2)
            self.kv[layer] = (k, v)
        else:
            self.kv.append((k, v))
        return k, v


class LM:
    def __init__(self, name: str, dtype: torch.dtype = torch.float32):
        d = fetch(name)
        self.name = name
        self.cfg = json.loads((d / "config.json").read_text())
        self.spec = spec_of(self.cfg)
        self.dtype = dtype
        raw = read_safetensors(d / "model.safetensors")
        self.W = {k.removeprefix("transformer.").removeprefix("model."): v.to(dtype) for k, v in raw.items()}
        s = self.spec
        if s.arch == "gpt2":
            self.E = self.W["wte.weight"]
        else:
            self.E = self.W["embed_tokens.weight"]
            self.inv_freq = 1.0 / (s.rope_theta ** (torch.arange(0, s.head_dim, 2, dtype=torch.int64).float() / s.head_dim))
        self.tokenizer_path = d / "tokenizer.json"

    def tokenizer(self):
        from tokenizers import Tokenizer
        return Tokenizer.from_file(str(self.tokenizer_path))

    # -- sites -------------------------------------------------------------------------------------------------------
    @staticmethod
    def _site(name: str, layer: int, x: torch.Tensor, pos0: int, fault: Fault | None, taps: dict | None):
        if fault is not None and (name, layer) in fault.rows and pos0 <= fault.pos < pos0 + x.shape[1]:
            r, c, v = fault.rows[(name, layer)]
            x = x.clone()
            x[r, fault.pos - pos0, c] = v.to(x.dtype)
        if taps is not None and (name, layer) in taps:
            taps[(name, layer)] = x
        return x

    # -- blocks ------------------------------------------------------------------------------------------------------
    def _attend(self, q, k, v, pos0: int) -> torch.Tensor:
        s = self.spec
        t, n = q.shape[2], k.shape[2]
        mask = torch.arange(n)[None, :] <= (pos0 + torch.arange(t))[:, None]
        if s.kv_heads != s.heads:
            r = s.heads // s.kv_heads
            k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    def _rope(self, x: torch.Tensor, pos0: int) -> torch.Tensor:
        t = x.shape[2]
        freqs = (pos0 + torch.arange(t, dtype=torch.float32))[:, None] * self.inv_freq[None, :]
        emb = torch.cat([freqs, freqs], -1)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    def _rms(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.spec.eps)
        return w * xf.to(x.dtype)

    def _gpt2_layer(self, l, x, pos0, cache, fault, taps):
        s, W, p = self.spec, self.W, f"h.{l}."
        b, t, _ = x.shape
        h = F.layer_norm(x, (s.d,), W[p + "ln_1.weight"], W[p + "ln_1.bias"], s.eps)
        q, k, v = (h @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]).split(s.d, -1)
        q, k, v = (z.view(b, t, s.heads, s.head_dim).transpose(1, 2) for z in (q, k, v))
        k, v = cache.append(l, k, v)
        a = self._attend(q, k, v, pos0).transpose(1, 2).reshape(b, t, s.d)
        a = a @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        x = x + self._site("attn", l, a, pos0, fault, taps)
        h = F.layer_norm(x, (s.d,), W[p + "ln_2.weight"], W[p + "ln_2.bias"], s.eps)
        m = F.gelu(h @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"], approximate="tanh")
        m = self._site("mlph", l, m, pos0, fault, taps)
        o = m @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
        x = x + self._site("mlp", l, o, pos0, fault, taps)
        return self._site("resid", l, x, pos0, fault, taps)

    def _qwen2_layer(self, l, x, pos0, cache, fault, taps):
        s, W, p = self.spec, self.W, f"layers.{l}."
        b, t, _ = x.shape
        h = self._rms(x, W[p + "input_layernorm.weight"])
        q = h @ W[p + "self_attn.q_proj.weight"].T + W[p + "self_attn.q_proj.bias"]
        k = h @ W[p + "self_attn.k_proj.weight"].T + W[p + "self_attn.k_proj.bias"]
        v = h @ W[p + "self_attn.v_proj.weight"].T + W[p + "self_attn.v_proj.bias"]
        q = self._rope(q.view(b, t, s.heads, s.head_dim).transpose(1, 2), pos0)
        k = self._rope(k.view(b, t, s.kv_heads, s.head_dim).transpose(1, 2), pos0)
        v = v.view(b, t, s.kv_heads, s.head_dim).transpose(1, 2)
        k, v = cache.append(l, k, v)
        a = self._attend(q, k, v, pos0).transpose(1, 2).reshape(b, t, s.heads * s.head_dim)
        a = a @ W[p + "self_attn.o_proj.weight"].T
        x = x + self._site("attn", l, a, pos0, fault, taps)
        h = self._rms(x, W[p + "post_attention_layernorm.weight"])
        m = F.silu(h @ W[p + "mlp.gate_proj.weight"].T) * (h @ W[p + "mlp.up_proj.weight"].T)
        m = self._site("mlph", l, m, pos0, fault, taps)
        o = m @ W[p + "mlp.down_proj.weight"].T
        x = x + self._site("mlp", l, o, pos0, fault, taps)
        return self._site("resid", l, x, pos0, fault, taps)

    def final_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.spec.arch == "gpt2":
            return F.layer_norm(x, (self.spec.d,), self.W["ln_f.weight"], self.W["ln_f.bias"], self.spec.eps)
        return self._rms(x, self.W["norm.weight"])

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, cache: Cache, fault: Fault | None = None, taps: dict | None = None,
                last_only: bool = False) -> torch.Tensor:
        """Logits (B, T, V) for ids (B, T) at positions cache.length.., appending to `cache`; `taps` (keys (site, layer))
        receive each named site's activation."""
        s, pos0 = self.spec, cache.length
        if s.arch == "gpt2":
            x = self.W["wte.weight"][ids] + self.W["wpe.weight"][pos0:pos0 + ids.shape[1]]
            layer = self._gpt2_layer
        else:
            x = self.E[ids]
            layer = self._qwen2_layer
        x = self._site("emb", -1, x, pos0, fault, taps)
        for l in range(s.layers):
            x = layer(l, x, pos0, cache, fault, taps)
        if last_only:
            x = x[:, -1:]
            pos0 = pos0 + ids.shape[1] - 1
        h = self._site("final", s.layers, self.final_norm(x), pos0, fault, taps)
        return h @ self.E.T


def served_logprobs(logits: torch.Tensor, precision: str) -> torch.Tensor:
    """What a server's log_softmax returns: logits rounded to bf16 first when its unembedding runs in bf16."""
    z = logits.float()
    if precision == "bf16":
        z = z.to(torch.bfloat16).float()
    return torch.log_softmax(z, -1)


def bits_per_token(vocab: int) -> float:
    return math.log2(vocab)
