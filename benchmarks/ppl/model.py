"""Llama-3.1 and Qwen2.5 decoder layers, read one at a time from a snapshot's safetensors, op for op as HF transformers'
`LlamaDecoderLayer` / `Qwen2DecoderLayer` compute them with SDPA attention (the engine's check window compares them bit
for bit on every layer when transformers is importable). The layer's linears and attention dispatch to an arm's chains;
with no arm they are the bf16 reference: `F.linear`, and `F.scaled_dot_product_attention(is_causal=True,
enable_gqa=True)`, PyTorch's SDPA, as HF's `sdpa_attention_forward` calls it. GLM-5.3's layers are `glm.py`'s
(`make_layer`)."""
from __future__ import annotations

import json
import math
import struct
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

import models
from matmul import MM, Run, Softmaxed, run_chain

#: F8_E4M3 is read as its bytes: a block-FP8 layer decodes them with their scales (`glm.dequant`)
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I64": torch.int64, "I32": torch.int32,
          "I8": torch.int8, "U8": torch.uint8, "F8_E4M3": torch.uint8}
ARCHS = {"LlamaForCausalLM": "llama", "Qwen2ForCausalLM": "qwen2", "GlmMoeDsaForCausalLM": "glm_moe_dsa"}
#: GLM-5.3's config fields its layers read (`glm.GlmLayer`)
GLM_KEYS = ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "n_routed_experts",
            "num_experts_per_tok", "moe_intermediate_size", "n_group", "topk_group", "norm_topk_prob",
            "routed_scaling_factor", "scoring_func", "topk_method", "n_shared_experts", "mlp_layer_types", "index_topk",
            "hidden_act")
#: a KV group's s x s work runs on all query rows at once while a (rep, s, s) float64 tensor is within ROWS_WHOLE bytes,
#: else on chunks of ROWS_BYTES (32K), as attention.py
ROWS_WHOLE, ROWS_BYTES = 2 ** 32, 2 ** 31
#: the head and loss run on a whole window up to LOSS_WHOLE tokens, else in chunks of LOSS_ROWS
LOSS_WHOLE, LOSS_ROWS = 8192, 4096


@dataclass(frozen=True)
class Config:
    arch: str
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    eps: float
    vocab: int
    tie: bool
    qkv_bias: bool
    rope: dict
    #: GLM-5.3's own fields (`GLM_KEYS`); its head_dim is the RoPE slice's, qk_rope_head_dim, as HF's config sets it
    glm: dict = field(default_factory=dict)

    @property
    def rep(self) -> int:
        return self.heads // self.kv_heads

    @staticmethod
    def read(snapshot: Path) -> "Config":
        c = json.loads((Path(snapshot) / "config.json").read_text())
        arch = ARCHS.get((c.get("architectures") or [""])[0])
        if arch is None:
            raise ValueError(f"{snapshot}: architectures {c.get('architectures')}; the bench runs {sorted(ARCHS)}")
        if arch == "qwen2" and c.get("use_sliding_window"):
            raise ValueError("full causal attention only")
        rope = dict(c.get("rope_scaling") or c.get("rope_parameters") or {})
        rope.setdefault("rope_theta", c.get("rope_theta", rope.get("rope_theta")))
        rope.setdefault("rope_type", rope.get("type", "default"))
        heads = c["num_attention_heads"]
        glm = {k: c[k] for k in GLM_KEYS} if arch == "glm_moe_dsa" else {}
        head_dim = glm["qk_rope_head_dim"] if glm else c.get("head_dim") or c["hidden_size"] // heads
        return Config(arch, c["hidden_size"], c["num_hidden_layers"], heads, c.get("num_key_value_heads", heads),
                      head_dim, c["intermediate_size"], c["rms_norm_eps"], c["vocab_size"],
                      bool(c.get("tie_word_embeddings", False)), arch == "qwen2", rope, glm)


# ---------------------------------------------------------------- weights

class Weights:
    """Tensors by name from a snapshot's safetensors (read directly: an 8-byte header length, a JSON header, the bytes),
    a layer at a time; a background thread prefetches the next layer.

    A streamed model's shards (`models.pin_of`) that the snapshot lacks are pulled when a read needs them
    (`stream.Hub`) and deleted once every tensor in them that a decoder layer below `layers` or the top reads is read."""

    def __init__(self, snapshot: Path, layers: int | None = None):
        self.snap = Path(snapshot)
        idx = self.snap / "model.safetensors.index.json"
        files = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else ["model.safetensors"]
        pin = models.pin_of(self.snap)
        self.hub = None
        self.headers: dict = {}
        self.where: dict = {}
        for fn in files:
            if (self.snap / fn).is_file():
                with open(self.snap / fn, "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    header = json.loads(f.read(n))
            elif pin is not None:
                if self.hub is None:
                    import stream
                    self.hub = stream.Hub(pin)
                n, header = self.headers[fn] = self.hub.header(fn)
            else:
                raise FileNotFoundError(self.snap / fn)
            for name, meta in header.items():
                if name != "__metadata__":
                    self.where[name] = (fn, 8 + n + meta["data_offsets"][0], meta["data_offsets"][1] - meta["data_offsets"][0],
                                        DTYPES[meta["dtype"]], tuple(meta["shape"]))
        self.left: dict = {fn: set() for fn in self.headers}
        for n, w in self.where.items():
            if w[0] in self.left and self._read_by(n, layers):
                self.left[w[0]].add(n)
        self.lock = threading.Lock()
        self.pending: dict = {}

    @staticmethod
    def _read_by(name: str, layers: int | None) -> bool:
        if layers is None or not name.startswith("model.layers."):
            return True
        return int(name.split(".")[2]) < layers

    def read(self, names) -> dict:
        out = {}
        by_file: dict = {}
        for n in names:
            by_file.setdefault(self.where[n][0], []).append(n)
        for fn, ns in by_file.items():
            path = self.snap / fn if fn not in self.headers else self.hub.path(fn, self.headers[fn])
            with open(path, "rb") as f:
                for n in sorted(ns, key=lambda x: self.where[x][1]):
                    _, off, size, dtype, shape = self.where[n]
                    f.seek(off)
                    buf = bytearray(f.read(size))
                    out[n] = torch.frombuffer(buf, dtype=dtype).view(shape) if size else torch.empty(shape, dtype=dtype)
            if fn in self.headers:
                with self.lock:
                    self.left[fn] -= set(ns)
                    if not self.left[fn]:
                        self.hub.done(fn)
        return out

    def close(self) -> None:
        if self.hub is not None:
            self.hub.close()

    def layer_names(self, li: int) -> list[str]:
        p = f"model.layers.{li}."
        return [n for n in self.where if n.startswith(p)]

    def prefetch(self, li: int) -> None:
        def go():
            self.pending[li] = self.read(self.layer_names(li))
        t = threading.Thread(target=go, daemon=True)
        t.start()
        self.pending[("thread", li)] = t

    def layer(self, li: int) -> dict:
        t = self.pending.pop(("thread", li), None)
        if t is not None:
            t.join()
            return self.pending.pop(li)
        return self.read(self.layer_names(li))


# ---------------------------------------------------------------- the reference ops, as HF computes them

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    h = x.to(torch.float32)
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    return w * h.to(dtype)


def inv_freq(cfg: Config) -> torch.Tensor:
    """HF's default and llama3 RoPE inverse frequencies (modeling_rope_utils), on the CPU in FP32."""
    dim, base = cfg.head_dim, cfg.rope["rope_theta"]
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim))
    kind = cfg.rope["rope_type"]
    if kind == "default":
        return inv
    if kind != "llama3":
        raise ValueError(f"rope {kind}")
    r = cfg.rope
    factor, lo, hi, old = r["factor"], r["low_freq_factor"], r["high_freq_factor"], r["original_max_position_embeddings"]
    low_wl, high_wl = old / lo, old / hi
    wl = 2 * math.pi / inv
    llama = torch.where(wl > low_wl, inv / factor, inv)
    smooth = (old / wl - lo) / (hi - lo)
    smoothed = (1 - smooth) * llama / factor + smooth * llama
    medium = ~(wl < high_wl) * ~(wl > low_wl)
    return torch.where(medium, smoothed, llama)


def rotary(cfg: Config, T: int, device, dtype=torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor]:
    inv = inv_freq(cfg).to(device)
    pos = torch.arange(T, device=device)[None]
    inv_e = inv[None, :, None].expand(1, -1, 1).to(dtype=torch.float, device=device)
    freqs = (inv_e @ pos[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return (emb.cos() * 1.0).to(dtype), (emb.sin() * 1.0).to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def row_chunk(rep: int, s: int) -> int:
    return s if rep * s * s * 8 <= ROWS_WHOLE else max(1, ROWS_BYTES // (rep * s * 8))


# ---------------------------------------------------------------- one layer

class Layer:
    def __init__(self, cfg: Config, li: int, tensors: dict, device):
        self.cfg, self.li = cfg, li
        p = f"model.layers.{li}."
        self.t = {n[len(p):]: t.to(device=device, dtype=torch.bfloat16) for n, t in tensors.items()}
        self.scaling = cfg.head_dim ** -0.5

    def weight(self, name: str) -> torch.Tensor:
        return self.t[f"{name}.weight"]

    def forward(self, h: torch.Tensor, cos, sin, run: Run | None = None, collect=None) -> torch.Tensor:
        c = self.cfg
        res = h
        x = rms_norm(h, self.t["input_layernorm.weight"], c.eps)
        B, T, _ = x.shape
        q = self.linear("self_attn.q_proj", x, run, collect).view(B, T, -1, c.head_dim).transpose(1, 2)
        k = self.linear("self_attn.k_proj", x, run, collect).view(B, T, -1, c.head_dim).transpose(1, 2)
        v = self.linear("self_attn.v_proj", x, run, collect).view(B, T, -1, c.head_dim).transpose(1, 2)
        cu, su = cos.unsqueeze(1), sin.unsqueeze(1)
        q, k = (q * cu) + (rotate_half(q) * su), (k * cu) + (rotate_half(k) * su)
        if collect is not None:
            collect("keys", k)
        a = self.attention(q, k, v, run).reshape(B, T, -1).contiguous()
        h = res + self.linear("self_attn.o_proj", a, run, collect)
        res = h
        x = rms_norm(h, self.t["post_attention_layernorm.weight"], c.eps)
        m = F.silu(self.linear("mlp.gate_proj", x, run, collect)) * self.linear("mlp.up_proj", x, run, collect)
        return res + self.linear("mlp.down_proj", m, run, collect)

    def linear(self, name: str, x: torch.Tensor, run: Run | None, collect=None) -> torch.Tensor:
        w, b = self.t[f"{name}.weight"], self.t.get(f"{name}.bias")
        if collect is not None:
            collect(name, x.reshape(-1, x.shape[-1]).float())
        steps = run.steps_for("linear", name) if run is not None else []
        if not steps:
            return F.linear(x, w, b)
        mm = MM("linear", name, run, x.reshape(-1, x.shape[-1]).float(), w, maps=(lambda t: t.float(),),
                extra={"weight64": w.double})
        y = run_chain(steps, mm)
        if b is not None:
            y = y + b.float()
        return y.to(x.dtype).reshape(*x.shape[:-1], w.shape[0])

    def attention(self, q, k, v, run: Run | None) -> torch.Tensor:
        if run is None or not (run.steps_for("qk", "qk") or run.steps_for("pv", "pv")):
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, scale=self.scaling, is_causal=True,
                                               enable_gqa=True)
            return o.transpose(1, 2).contiguous()
        return self.edited_attention(q, k, v, run)

    def edited_attention(self, query, key, value, run: Run) -> torch.Tensor:
        """Per KV group, in FP32: S = Q K^T through the arm's attention chain, the causal softmax (or the chain's own,
        when it returns `Softmaxed`), O = P V through it; query rows in chunks past 8K (the rows are independent). On the
        check window, the attention's relative error against FP32 attention on the same inputs (a diagnostic)."""
        b, H, s, d = query.shape
        if b != 1:
            raise ValueError("an arm's attention runs one window at a time")
        G = key.shape[1]
        rep = H // G
        dev = query.device
        qk_steps, pv_steps = run.steps_for("qk", "qk"), run.steps_for("pv", "pv")
        keys = torch.arange(s, device=dev)
        rows = row_chunk(rep, s)
        out = torch.empty(b, s, H, d, dtype=query.dtype, device=dev)
        for g in range(G):
            qg = query[0, g * rep:(g + 1) * rep].float()
            kg, vg = key[0, g].float(), value[0, g].float()
            ref = torch.empty(rep, s, d, device=dev) if run.checking else None
            cache: dict = {}
            o = torch.empty(rep, s, d, device=dev)
            for r0 in range(0, s, rows):
                r1 = min(s, r0 + rows)
                causal = keys[None, :] <= torch.arange(r0, r1, device=dev)[:, None]
                if ref is not None:
                    sc = (qg[:, r0:r1] @ kg.T) * self.scaling
                    ref[:, r0:r1] = torch.softmax(sc.masked_fill(~causal, float("-inf")), dim=-1) @ vg
                    del sc
                extra = {"group": g, "rows": (r0, r1), "causal": causal, "rep": rep, "s": s, "d": d}
                scores = run_chain(qk_steps, MM("qk", "qk", run, qg[:, r0:r1], kg, cache=cache, extra=extra))
                if isinstance(scores, Softmaxed):
                    p, pv_extra = scores.p, {**extra, **scores.extra}
                else:
                    p, pv_extra = torch.softmax((scores * self.scaling).masked_fill(~causal, float("-inf")), dim=-1), extra
                del scores
                o[:, r0:r1] = run_chain(pv_steps, MM("pv", "pv", run, p, vg, cache=cache, extra=pv_extra)).float()
                del pv_extra
                del p, causal
            if ref is not None:
                run.attn_err[0] += float(((o - ref) ** 2).sum())
                run.attn_err[1] += float((ref ** 2).sum())
            out[0, :, g * rep:(g + 1) * rep] = o.transpose(0, 1).to(query.dtype)
        return out


def head(cfg: Config, norm_w: torch.Tensor, head_w: torch.Tensor, h: torch.Tensor, ids: torch.Tensor,
         proj: Callable | None = None) -> torch.Tensor:
    """The NLL of every next token of one window (T - 1 of them): the final norm, the head (`proj(rows, chunk)` in
    place of rows @ head_w^T when given), the cross entropy, in chunks of LOSS_ROWS tokens past 8K."""
    T = h.shape[0]
    proj = proj or (lambda x, _c: x @ head_w.T)
    if T <= LOSS_WHOLE:
        logits = proj(rms_norm(h, norm_w, cfg.eps), 0).float()
        return F.cross_entropy(logits[:-1], ids[1:], reduction="none")
    n = rms_norm(h, norm_w, cfg.eps)
    return torch.cat([F.cross_entropy(proj(n[i:min(i + LOSS_ROWS, T - 1)], c).float(), ids[i + 1:i + 1 + LOSS_ROWS],
                                      reduction="none") for c, i in enumerate(range(0, T - 1, LOSS_ROWS))])


def hf_layer(cfg: Config, snapshot: Path, li: int, layer: Layer):
    """HF transformers' own decoder layer with the same weights and SDPA attention, or None without transformers."""
    try:
        import transformers
    except ImportError:
        return None
    hc = transformers.AutoConfig.from_pretrained(str(snapshot))
    hc._attn_implementation = "sdpa"
    mod = {"llama": transformers.models.llama.modeling_llama.LlamaDecoderLayer,
           "qwen2": transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer}[cfg.arch]
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            hl = mod(hc, li)
    finally:
        torch.set_default_dtype(prev)
    dev = layer.t["input_layernorm.weight"].device
    hl = hl.to_empty(device=dev)
    hl.load_state_dict(layer.t, strict=True)
    rot_cls = {"llama": transformers.models.llama.modeling_llama.LlamaRotaryEmbedding,
               "qwen2": transformers.models.qwen2.modeling_qwen2.Qwen2RotaryEmbedding}[cfg.arch]
    return hl.eval(), rot_cls(hc).to(dev)


def make_layer(cfg: Config, li: int, tensors: dict, device):
    if cfg.arch == "glm_moe_dsa":
        import glm
        return glm.GlmLayer(cfg, li, tensors, device)
    return Layer(cfg, li, tensors, device)


def check_reference(gates, cfg: Config, snapshot: Path, layer: Layer, h: torch.Tensor, cos, sin, out: torch.Tensor) -> None:
    """The engine's bf16 layer output equals HF's layer on the same input, bit for bit (and its RoPE tables HF's)."""
    if cfg.arch == "glm_moe_dsa":
        import glm
        return glm.check_reference(gates, cfg, snapshot, layer, h, cos, sin, out)
    hf = hf_layer(cfg, snapshot, layer.li, layer)
    if hf is None:
        gates.skip("hf_reference", "transformers is not importable")
        return
    hl, rot = hf
    pos = torch.arange(h.shape[1], device=h.device)[None]
    with torch.no_grad():
        hcos, hsin = rot(h, pos)
        want = hl(h, attention_mask=None, position_embeddings=(hcos, hsin))
        want = want[0] if isinstance(want, tuple) else want
    ok = bool(torch.equal(hcos, cos) and torch.equal(hsin, sin) and torch.equal(want, out))
    gates.record("hf_reference", ok, f"layer {layer.li}",
                 None if ok else float((want.float() - out.float()).abs().max()))
    del hl
