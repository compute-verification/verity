"""DeepSeek-V4's decoder layers (`deepseek_v4`; DeepSeek-V4-Pro), from its FP8 and FP4 checkpoint, op for op as HF
transformers' `DeepseekV4DecoderLayer` computes them with eager attention and eager experts (`check_reference` compares
them bit for bit on the check window).

The weights are the checkpoint's, dequantized exactly: each FP8 linear's E4M3 codes and each routed expert's packed
E2M1 (FP4) codes times their UE8M0 block scales (`dequant`, HF's `_dequantize_one`), held in bf16 as HF holds them
(exact), in FP32 for an arm's chain and in float64 as `weight64`. HF's `_keep_in_fp32_modules_strict` (the mHC
parameters, the sinks, the compressors' position biases, the router's correction bias and every RMSNorm weight) stay
FP32; everything else is bf16. The checkpoint's names are DeepSeek's (`layers.N.attn.wq_a`, `ffn.experts.E.w1`); the
layer keeps them (`model.checkpoint_name` puts them under `model.layers.N.`) and names its linears as HF does.

The residual is mHC's `hc_mult` streams ([B, T, 4, hidden]: `embed`), mixed into each block and out of it in FP32
(`hc_pre`, Sinkhorn), and collapsed at the end by the hc head (`final_norm`). Attention is multi-query: one 512-wide KV
head, read as both K and V, for 128 query heads, each with its own sink logit; RoPE is interleaved on the trailing 64
channels and taken off the output at the query's position. Its keys are the 128-key sliding window and the
compressor's entries: HCA pools every 128 tokens into one, CSA every 4 (overlapping windows of 8) and lets the
indexer pick at most `index_topk` of them per query. At T <= 4 index_topk every query's top-k is every entry it may
see, so the indexer is skipped and the mask is causal over the entries (`compress`; exact, HF's block bias is the same).
At 8K the reference attention and the indexer's scores run on chunks of query rows (`EAGER_WHOLE`), and the HF check,
whose whole-window eager attention would not fit a GPU, compares one window's first 4224 tokens (`check_prefix`).

An arm's linear steps reach q_a, q_b, kv, each of o_a's 16 groups (`self_attn.o_a_proj.G`, rows G of wo_a, as HF's
grouped bmm reads them), o_b, the compressors' and indexer's kv and gate, the indexer's q_b, the shared expert's gate,
up and down and each routed expert's gate_up (w1's rows, then w3's) and down (`mlp.experts.E.gate_up_proj`,
`mlp.experts.E.down_proj`), each expert on the rows routed to it. The routers, the indexer's weights_proj and scores,
and mHC stay at the reference's precision. Its attention steps see one KV group of 128 heads over the window's keys
then the compressor's entries, with the query's visible ones as the mask (`extra["causal"]`, `general_mask`) and the
sinks (`extra["sinks"]`); the LUT softmax is refused. SmoothQuant's groups (`groups`, `first`) are the inputs linears
share: x (q_a, kv, the compressors'), q_a's normed output (q_b, the indexer's q_b), each o_a group's heads, o_b's
input, the MoE input (the shared expert's gate and up, every routed gate_up), and each down's own. o_a has no
kv_shared form: K and V are one tensor under MQA, so no row of a V projection takes 1 / m."""
from __future__ import annotations

import math
import types

import torch
import torch.nn.functional as F

from matmul import MM, Run, Softmaxed, run_chain
from model import row_chunk

#: E2M1's 16 values by nibble, as HF's `_FP4_E2M1_LUT`
FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
#: HF's `_keep_in_fp32_modules_strict`: a parameter under a module of one of these names stays FP32
FP32_STRICT = ("attn_hc", "ffn_hc", "hc_head", "sinks", "position_bias", "e_score_correction_bias", "q_a_norm",
               "kv_norm", "input_layernorm", "post_attention_layernorm", "norm")
RATIO_TYPES = {0: "sliding_attention", 4: "compressed_sparse_attention", 128: "heavily_compressed_attention"}
#: the bf16 reference's attention runs whole while its FP32 logits ([B, 128, T, keys]) are within EAGER_WHOLE bytes (four
#: windows at 2K; one window's HF-check prefix at 8K), else on query-row chunks of EAGER_BYTES; the indexer's scores
#: ([B, T, 64, entries]) on chunks of INDEX_BYTES
EAGER_WHOLE, EAGER_BYTES, INDEX_BYTES = 16 << 30, 4 << 30, 4 << 30
KEYS = ("q_lora_rank", "o_groups", "o_lora_rank", "sliding_window", "index_n_heads", "index_head_dim", "index_topk",
        "n_routed_experts", "num_experts_per_tok", "moe_intermediate_size", "n_shared_experts", "scoring_func",
        "norm_topk_prob", "routed_scaling_factor", "swiglu_limit", "hc_mult", "hc_sinkhorn_iters", "hc_eps",
        "hidden_act")
#: a layer's checkpoint name (after `layers.N.`) -> HF's, for every tensor but the routed experts'
HF_NAMES = {
    "attn_norm.weight": "input_layernorm.weight", "ffn_norm.weight": "post_attention_layernorm.weight",
    **{f"hc_{b}_{p}": f"{b}_hc.{p}" for b in ("attn", "ffn") for p in ("fn", "base", "scale")},
    "attn.attn_sink": "self_attn.sinks", "attn.wq_a.weight": "self_attn.q_a_proj.weight",
    "attn.q_norm.weight": "self_attn.q_a_norm.weight", "attn.wq_b.weight": "self_attn.q_b_proj.weight",
    "attn.wkv.weight": "self_attn.kv_proj.weight", "attn.kv_norm.weight": "self_attn.kv_norm.weight",
    "attn.wo_a.weight": "self_attn.o_a_proj.weight", "attn.wo_b.weight": "self_attn.o_b_proj.weight",
    "attn.compressor.ape": "self_attn.compressor.position_bias",
    "attn.compressor.wkv.weight": "self_attn.compressor.kv_proj.weight",
    "attn.compressor.wgate.weight": "self_attn.compressor.gate_proj.weight",
    "attn.compressor.norm.weight": "self_attn.compressor.kv_norm.weight",
    "attn.indexer.compressor.ape": "self_attn.compressor.indexer.position_bias",
    "attn.indexer.compressor.wkv.weight": "self_attn.compressor.indexer.kv_proj.weight",
    "attn.indexer.compressor.wgate.weight": "self_attn.compressor.indexer.gate_proj.weight",
    "attn.indexer.compressor.norm.weight": "self_attn.compressor.indexer.kv_norm.weight",
    "attn.indexer.wq_b.weight": "self_attn.compressor.indexer.q_b_proj.weight",
    "attn.indexer.weights_proj.weight": "self_attn.compressor.indexer.scorer.weights_proj.weight",
    "ffn.gate.weight": "mlp.gate.weight", "ffn.gate.bias": "mlp.gate.e_score_correction_bias",
    "ffn.gate.tid2eid": "mlp.gate.tid2eid", "ffn.shared_experts.w1.weight": "mlp.shared_experts.gate_proj.weight",
    "ffn.shared_experts.w3.weight": "mlp.shared_experts.up_proj.weight",
    "ffn.shared_experts.w2.weight": "mlp.shared_experts.down_proj.weight",
}
#: an arm-visible linear (HF's name) -> its checkpoint weight
LINEARS = {"self_attn.q_a_proj": "attn.wq_a", "self_attn.q_b_proj": "attn.wq_b", "self_attn.kv_proj": "attn.wkv",
           "self_attn.o_b_proj": "attn.wo_b", "self_attn.compressor.kv_proj": "attn.compressor.wkv",
           "self_attn.compressor.gate_proj": "attn.compressor.wgate",
           "self_attn.compressor.indexer.kv_proj": "attn.indexer.compressor.wkv",
           "self_attn.compressor.indexer.gate_proj": "attn.indexer.compressor.wgate",
           "self_attn.compressor.indexer.q_b_proj": "attn.indexer.wq_b",
           "self_attn.compressor.indexer.scorer.weights_proj": "attn.indexer.weights_proj",
           "mlp.shared_experts.gate_proj": "ffn.shared_experts.w1", "mlp.shared_experts.up_proj": "ffn.shared_experts.w3",
           "mlp.shared_experts.down_proj": "ffn.shared_experts.w2"}
TOP = {"embed.weight": "model.embed_tokens.weight", "norm.weight": "model.norm.weight", "head.weight": "lm_head.weight",
       "hc_head_fn": "model.hc_head.fn", "hc_head_base": "model.hc_head.base", "hc_head_scale": "model.hc_head.scale"}


def fields(c: dict) -> dict:
    """The config fields the layers read, with HF's `DeepseekV4Config` derivations: layer types (explicit, or from the
    legacy per-layer `compress_ratios`), compress rates, MLP types (the first `num_hash_layers` hash-routed), the RoPE
    width and the two RoPE parameter sets (`main`, plain; `compress`, YaRN at compress_rope_theta, attention_factor 1)."""
    n = c["num_hidden_layers"]
    d = {k: c[k] for k in KEYS}
    if (d["scoring_func"], d["hidden_act"], d["n_shared_experts"], d["norm_topk_prob"]) != ("sqrtsoftplus", "silu", 1, True):
        raise ValueError("DeepSeek-V4: the bench runs sqrtsoftplus routing, normalized top-k and one silu shared expert")
    rates = dict(c.get("compress_rates") or {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128})
    kinds = c.get("layer_types") or ([RATIO_TYPES[r] for r in c["compress_ratios"]] if c.get("compress_ratios") else
                                     ["heavily_compressed_attention"] * min(n, 2) +
                                     ["compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
                                      for i in range(max(n - 2, 0))])
    mlp = c.get("mlp_layer_types") or ["hash_moe"] * min(n, c.get("num_hash_layers", 3)) + ["moe"] * max(0, n - c.get("num_hash_layers", 3))
    rope_dim = int(c["head_dim"] * (c.get("partial_rotary_factor") or c.get("qk_rope_head_dim", 64) / c["head_dim"]))
    yarn = dict(c.get("rope_scaling") or {})
    yarn.setdefault("rope_type", yarn.get("type", "default"))
    compress = {**yarn, "rope_theta": c.get("compress_rope_theta", 160000.0)}
    if compress["rope_type"] == "yarn":
        compress.setdefault("attention_factor", 1.0)
    d.update(layer_types=list(kinds[:n]), compress_rates=rates, mlp_layer_types=list(mlp[:n]), rope_dim=rope_dim,
             rope={"main": {"rope_type": "default", "rope_theta": c.get("rope_theta", 10000.0)}, "compress": compress})
    return d


def checkpoint_name(name: str) -> str | None:
    """A checkpoint tensor's name in the engine's layout (`model.layers.N.` + DeepSeek's own; the top's HF's), or None
    for the MTP layer, which the bench never reads."""
    if name.startswith("mtp."):
        return None
    if name.startswith("layers."):
        return "model." + name
    return TOP.get(name, name)


def hf_name(n: str) -> str:
    return HF_NAMES[n]


def keep32(n: str) -> bool:
    return any(part in FP32_STRICT for part in hf_name(n).split("."))


def e8m0(s: torch.Tensor) -> torch.Tensor:
    return (s.to(torch.float32) - 127.0).exp2()


def dequant(q: torch.Tensor, s: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """FP8 E4M3 codes (their bytes, n x k) or packed E2M1 (int8, n x k / 2, the low nibble first) times their UE8M0
    block scales (one per block of the grid the scale's shape cuts), in `dtype`: exact in FP32 and so in float64 and,
    for these significands and exponents, in bf16."""
    if q.dtype == torch.int8:
        lut = torch.tensor(FP4, dtype=torch.float32, device=q.device)
        u8 = q.view(torch.uint8)
        v = torch.stack([lut[(u8 & 0xF).long()], lut[((u8 >> 4) & 0xF).long()]], dim=-1).reshape(q.shape[0], -1)
    else:
        v = q.view(torch.float8_e4m3fn).to(torch.float32)
    n, k = v.shape
    sr, sc = s.shape
    v = (v.view(sr, n // sr, sc, k // sc) * e8m0(s)[:, None, :, None]).view(n, k)
    return v.to(dtype)


def inv_freq(cfg, kind: str) -> torch.Tensor:
    """HF's inverse frequencies of one RoPE parameter set (`main` or `compress`) on the RoPE's 64 channels."""
    import dsv2
    return dsv2.inv_freq(types.SimpleNamespace(head_dim=cfg.dsv4["rope_dim"], rope=cfg.dsv4["rope"][kind]))[0]


def table(inv: torch.Tensor, pos: torch.Tensor, dtype=torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor]:
    """`DeepseekV4RotaryEmbedding.forward`: cos and sin of pos x inv_freq ([..., 32]), in `dtype`."""
    freqs = pos[..., None].float() * inv.to(device=pos.device, dtype=torch.float)
    return (freqs.cos() * 1.0).to(dtype), (freqs.sin() * 1.0).to(dtype)


def rotary(cfg, T: int, device) -> tuple[dict, None]:
    """Both RoPE tables for positions 0..T-1, {main, compress}: (cos, sin), as the model hands its layers; it stands in
    the engine's (cos, sin) as (tables, None)."""
    pos = torch.arange(T, device=device)[None]
    return {k: table(inv_freq(cfg, k), pos) for k in ("main", "compress")}, None


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    return torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
    """HF's `apply_rotary_pos_emb`: interleaved RoPE on x's trailing 2 cos.shape[-1] channels, in FP32."""
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    d = cos.shape[-1]
    nope, r = x[..., :-d], x[..., -d:]
    return torch.cat([nope, ((r.float() * cos) + (rotate_half(r).float() * sin)).to(x.dtype)], dim=-1)


def norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """`DeepseekV4RMSNorm`: the FP32 weight times the FP32 normalized x, cast once."""
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return (w * h).to(x.dtype)


def unorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """`DeepseekV4UnweightedRMSNorm`."""
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


def repeat_kv(x: torch.Tensor, n: int) -> torch.Tensor:
    b, g, s, d = x.shape
    return x if n == 1 else x[:, :, None].expand(b, g, n, s, d).reshape(b, g * n, s, d)


def sliding(T: int, W: int, device) -> torch.Tensor:
    """Query i sees keys i - W < j <= i (HF's sliding-window causal mask), as a bool [T, T]."""
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None]
    return (j <= i) & (j > i - W)


def embed(cfg, emb: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """The model's input streams: each token's embedding on all hc_mult of them."""
    e = emb[ids]
    return e.unsqueeze(2).expand(*e.shape[:-1], cfg.dsv4["hc_mult"], e.shape[-1]).contiguous()


def final_norm(cfg, top: dict, device):
    """`norm(hc_head(h))`: the hc head's sigmoid collapse of the streams in FP32, then the FP32-weight RMSNorm."""
    fn, base, scale = (top[f"model.hc_head.{p}"].to(device=device, dtype=torch.float32) for p in ("fn", "base", "scale"))
    w = top["model.norm.weight"].to(device=device, dtype=torch.float32)
    g = cfg.dsv4

    def apply(h: torch.Tensor) -> torch.Tensor:
        x = h[None]
        flat = unorm(x.flatten(2).float(), cfg.eps)
        pre = torch.sigmoid(F.linear(flat, fn) * scale + base) + g["hc_eps"]
        return norm((pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype), w, cfg.eps)[0]
    return apply


class Dsv4Layer:
    needs_ids = True

    def __init__(self, cfg, li: int, tensors: dict, device):
        self.cfg, self.li, self.g = cfg, li, cfg.dsv4
        g = self.g
        p = f"model.layers.{li}."
        raw = {n[len(p):]: t for n, t in tensors.items()}
        self.codes: dict = {}
        self.t: dict = {}
        for n, t in raw.items():
            if n.endswith(".scale") and n[:-len(".scale")] + ".weight" in raw:
                continue
            if n.endswith(".weight") and n[:-len(".weight")] + ".scale" in raw:
                self.codes[n[:-len(".weight")]] = (t.to(device), raw[n[:-len(".weight")] + ".scale"].to(device))
            elif t.dtype == torch.int64:
                self.t[n] = t.to(device)
            elif ".experts." in n:
                raise ValueError(f"DeepSeek-V4: routed expert tensor {n} without its scale")
            else:
                self.t[n] = t.to(device=device, dtype=torch.float32 if keep32(n) else torch.bfloat16)
        self.kind = g["layer_types"][li]
        self.hash = g["mlp_layer_types"][li] == "hash_moe"
        self.ratio = g["compress_rates"].get(self.kind, 0)
        self.csa = self.kind == "compressed_sparse_attention"
        self.rope_kind = "main" if self.kind == "sliding_attention" else "compress"
        self.inv_c = inv_freq(cfg, "compress")
        self.scaling = cfg.head_dim ** -0.5
        self.bf16: dict = {}
        self.arm_linears = None
        E = g["n_routed_experts"]
        a, c, ix = "self_attn.", "self_attn.compressor.", "self_attn.compressor.indexer."
        self.groups = {a + "q_a_proj": "x", a + "kv_proj": "x"}
        if self.kind != "sliding_attention":
            self.groups.update({c + "kv_proj": "x", c + "gate_proj": "x"})
        if self.csa:
            self.groups.update({ix + "kv_proj": "x", ix + "gate_proj": "x"})
        self.groups[a + "q_b_proj"] = "qr"
        if self.csa:
            self.groups[ix + "q_b_proj"] = "qr"
        self.groups.update({f"{a}o_a_proj.{j}": f"o_a.{j}" for j in range(g["o_groups"])})
        self.groups[a + "o_b_proj"] = "ob"
        self.groups.update({"mlp.shared_experts.gate_proj": "moe", "mlp.shared_experts.up_proj": "moe",
                            "mlp.shared_experts.down_proj": "shared_down"})
        for e in range(E):
            self.groups[f"mlp.experts.{e}.gate_up_proj"] = "moe"
            self.groups[f"mlp.experts.{e}.down_proj"] = f"down.{e}"
        #: the linear whose input stands for its group's (calibration reads it): the shared expert's gate sees every row
        self.first: dict = {}
        for n, grp in self.groups.items():
            self.first.setdefault(grp, n)

    # ------------------------------------------------------------ weights

    def wq(self, base: str, dtype=torch.float32) -> torch.Tensor:
        if base in self.codes:
            return dequant(*self.codes[base], dtype)
        return self.t[base + ".weight"].to(dtype)

    def weight32(self, name: str, dtype=torch.float32) -> torch.Tensor:
        """A linear's weight in FP32 (`dtype` float64: exact): an expert's gate_up is w1's rows, then w3's; o_a's group
        G is wo_a's rows G o_lora_rank .. (G + 1) o_lora_rank."""
        if name.startswith("mlp.experts."):
            e, part = name.split(".")[2:4]
            p = f"ffn.experts.{e}."
            if part == "gate_up_proj":
                return torch.cat([self.wq(p + "w1", dtype), self.wq(p + "w3", dtype)])
            return self.wq(p + "w2", dtype)
        if name.startswith("self_attn.o_a_proj."):
            j, r = int(name.rsplit(".", 1)[1]), self.g["o_lora_rank"]
            return self.wq("attn.wo_a", dtype)[j * r:(j + 1) * r]
        return self.wq(LINEARS[name], dtype)

    def wb(self, base: str) -> torch.Tensor:
        """A non-expert checkpoint weight in bf16, dequantized once per layer."""
        if base not in self.bf16:
            self.bf16[base] = self.wq(base).to(torch.bfloat16)
        return self.bf16[base]

    def weight(self, name: str) -> torch.Tensor:
        if name.startswith("mlp.experts."):
            return self.weight32(name).to(torch.bfloat16)
        if name.startswith("self_attn.o_a_proj."):
            j, r = int(name.rsplit(".", 1)[1]), self.g["o_lora_rank"]
            return self.wb("attn.wo_a")[j * r:(j + 1) * r]
        return self.wb(LINEARS[name])

    def evict(self, run: Run | None) -> None:
        """Drops the routed experts' cached int8 weights of any linear chain but this run's (they are a third of a GPU
        per chain): runs come arm by arm, so a chain's entries are made once and dropped when the next arm differs."""
        if run is None:
            return
        keys = tuple(s.key for s in run.arm.linear)
        if keys == self.arm_linears:
            return
        self.arm_linears = keys
        own = set(keys)
        for k in [k for k in run.cache if isinstance(k, tuple) and len(k) == 4 and isinstance(k[0], str)
                  and k[0].startswith("mlp.experts.") and not set(k[1]) <= own]:
            del run.cache[k]

    def linear(self, name: str, x: torch.Tensor, run: Run | None, collect=None) -> torch.Tensor:
        if collect is not None:
            collect(name, x.reshape(-1, x.shape[-1]).float())
        steps = run.steps_for("linear", name) if run is not None else []
        if not steps:
            return F.linear(x, self.weight(name))
        mm = MM("linear", name, run, x.reshape(-1, x.shape[-1]).float(), name, maps=(self.weight32,),
                extra={"weight64": lambda: self.weight32(name, torch.float64)})
        y = run_chain(steps, mm)
        return y.to(x.dtype).reshape(*x.shape[:-1], y.shape[-1])

    # ------------------------------------------------------------ the layer

    def forward(self, h: torch.Tensor, cos, sin, run: Run | None = None, collect=None, ids=None) -> torch.Tensor:
        if ids is None:
            raise ValueError("DeepSeek-V4: the layer reads the token ids (hash routing)")
        self.evict(run)
        eps = self.cfg.eps
        post, comb, x = self.hc_pre(h, "attn")
        a = self.attention(norm(x, self.t["attn_norm.weight"], eps), cos, run, collect)
        h = self.hc_post(h, post, comb, a)
        post, comb, x = self.hc_pre(h, "ffn")
        m = self.moe(norm(x, self.t["ffn_norm.weight"], eps), ids, run, collect)
        return self.hc_post(h, post, comb, m)

    def hc_pre(self, h: torch.Tensor, which: str):
        """`DeepseekV4HyperConnection`: post, comb (Sinkhorn) and the streams collapsed by pre, in FP32."""
        g = self.g
        n, eps = g["hc_mult"], g["hc_eps"]
        fn, base, scale = (self.t[f"hc_{which}_{p}"] for p in ("fn", "base", "scale"))
        B, S = h.shape[:2]
        flat = F.linear(unorm(h.view(B, S, -1).float(), self.cfg.eps), fn.float())
        pre_w, post_w, comb_w = flat.split([n, n, n * n], dim=-1)
        pre_b, post_b, comb_b = base.split([n, n, n * n])
        pre_s, post_s, comb_s = scale.unbind(0)
        comb_w = comb_w.view(*comb_w.shape[:-1], n, n)
        pre = torch.sigmoid(pre_w * pre_s + pre_b) + eps
        post = 2 * torch.sigmoid(post_w * post_s + post_b)
        comb = torch.softmax(comb_w * comb_s + comb_b.view(n, n), dim=-1) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(g["hc_sinkhorn_iters"] - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        return post, comb, (pre.unsqueeze(-1) * h).sum(dim=2).to(h.dtype)

    @staticmethod
    def hc_post(h: torch.Tensor, post: torch.Tensor, comb: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        dtype = h.dtype
        return post.to(dtype).unsqueeze(-1) * out.unsqueeze(-2) + torch.matmul(comb.to(dtype).transpose(-1, -2), h)

    def attention(self, x: torch.Tensor, pe: dict, run: Run | None, collect) -> torch.Tensor:
        c, g = self.cfg, self.g
        eps, d = c.eps, c.head_dim
        B, S, _ = x.shape
        cos, sin = pe[self.rope_kind]
        a = "self_attn."
        q_res = norm(self.linear(a + "q_a_proj", x, run, collect), self.t["attn.q_norm.weight"], eps)
        q = self.linear(a + "q_b_proj", q_res, run, collect).view(B, S, -1, d).transpose(1, 2)
        q = rope(unorm(q, eps), cos, sin)
        kv = norm(self.linear(a + "kv_proj", x, run, collect), self.t["attn.kv_norm.weight"], eps)
        kv = rope(kv.view(B, S, -1, d).transpose(1, 2), cos, sin)
        if collect is not None:
            collect("keys", kv)
        win = sliding(S, g["sliding_window"], x.device)
        bias = None
        if self.kind != "sliding_attention":
            ckv, bias = self.compress(x, q_res, pe, run, collect)
            kv = torch.cat([kv, ckv], dim=2)
        edited = run is not None and bool(run.steps_for("qk", "qk") or run.steps_for("pv", "pv"))
        if edited:
            if any(s.kw.get("softmax") == "lut" for s in run.arm.attention):
                raise ValueError("DeepSeek-V4: the LUT softmax has no sink; this attention's every head has one")
            vis = win if bias is None else torch.cat([win, bias[0, 0] == 0], dim=-1)
            o = self.edited_attention(q, kv, vis, run)
        else:
            mask = torch.zeros(B, 1, S, S, dtype=x.dtype, device=x.device).masked_fill(~win, torch.finfo(x.dtype).min)
            if bias is not None:
                mask = torch.cat([mask, bias.to(mask.dtype)], dim=-1)
            o = self.eager(q, kv, mask)
        o = rope(o.transpose(1, 2), cos, -sin).transpose(1, 2)
        y = self.o_a(o.reshape(B, S, g["o_groups"], -1), run, collect).flatten(2)
        return self.linear(a + "o_b_proj", y, run, collect)

    def eager(self, q: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """HF's `eager_attention_forward` with the sinks: bf16 scores plus the mask, the sink logit appended in FP32,
        the max taken off, the FP32 softmax, the sink dropped, bf16 P V. Whole while its FP32 logits fit EAGER_WHOLE,
        else on chunks of query rows (each row's softmax is its own)."""
        B, H, S, _ = q.shape
        k = repeat_kv(kv, H)
        per_row = B * H * k.shape[2] * 4
        if per_row * S <= EAGER_WHOLE:
            return self.eager_rows(q, k, mask)
        rows = max(1, EAGER_BYTES // per_row)
        o = q.new_empty(B, S, H, k.shape[-1])
        for r in range(0, S, rows):
            o[:, r:r + rows] = self.eager_rows(q[:, :, r:r + rows], k, mask[:, :, r:r + rows])
        return o

    def eager_rows(self, q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        w = w + mask
        sinks = self.t["attn.attn_sink"].reshape(1, -1, 1, 1).expand(q.shape[0], -1, q.shape[-2], -1)
        logits = torch.cat([w, sinks], dim=-1)
        del w
        logits = logits - logits.max(dim=-1, keepdim=True).values
        p = F.softmax(logits, dim=-1, dtype=logits.dtype)[..., :-1].to(k.dtype)
        del logits
        return torch.matmul(p, k).transpose(1, 2).contiguous()

    def edited_attention(self, q: torch.Tensor, kv: torch.Tensor, vis: torch.Tensor, run: Run) -> torch.Tensor:
        """In FP32, one window: S = Q K^T through the arm's attention chain, the softmax over each query's visible keys
        and its head's sink (the sink's share dropped), O = P V through the chain, V = K; query rows in chunks. On the
        check window, the attention's relative error against FP32 attention on the same inputs (a diagnostic)."""
        b, H, S, d = q.shape
        if b != 1:
            raise ValueError("an arm's attention runs one window at a time")
        qg, kg = q[0].float(), kv[0, 0].float()
        sk = kg.shape[0]
        sinks = self.t["attn.attn_sink"].float()[:, None, None]
        qk_steps, pv_steps = run.steps_for("qk", "qk"), run.steps_for("pv", "pv")
        rows = row_chunk(H, sk)
        cache: dict = {}
        o = torch.empty(H, S, d, device=q.device)
        ref = torch.empty(H, S, d, device=q.device) if run.checking else None
        for r0 in range(0, S, rows):
            r1 = min(S, r0 + rows)
            vr = vis[r0:r1]
            extra = {"group": 0, "rows": (r0, r1), "causal": vr, "rep": H, "s": sk, "d": d, "sinks": True,
                     "general_mask": True}
            if ref is not None:
                ref[:, r0:r1] = self.sink_softmax((qg[:, r0:r1] @ kg.T) * self.scaling, vr, sinks) @ kg
            scores = run_chain(qk_steps, MM("qk", "qk", run, qg[:, r0:r1], kg, cache=cache, extra=extra))
            if isinstance(scores, Softmaxed):
                raise ValueError("DeepSeek-V4: an integer softmax has no sink")
            p = self.sink_softmax(scores * self.scaling, vr, sinks)
            del scores
            o[:, r0:r1] = run_chain(pv_steps, MM("pv", "pv", run, p, kg, cache=cache, extra=extra)).float()
            del p
        if ref is not None:
            run.attn_err[0] += float(((o - ref) ** 2).sum())
            run.attn_err[1] += float((ref ** 2).sum())
        return o.transpose(0, 1)[None].to(q.dtype)

    @staticmethod
    def sink_softmax(s: torch.Tensor, vis: torch.Tensor, sinks: torch.Tensor) -> torch.Tensor:
        z = torch.cat([s.masked_fill(~vis, float("-inf")), sinks.expand(-1, s.shape[1], 1)], dim=-1)
        return torch.softmax(z, dim=-1)[..., :-1]

    def o_a(self, grouped: torch.Tensor, run: Run | None, collect) -> torch.Tensor:
        """`DeepseekV4GroupedLinear`: group G of the heads through wo_a's rows G, by HF's bmm in the reference, by
        each group's linear in an arm."""
        G = self.g["o_groups"]
        names = [f"self_attn.o_a_proj.{j}" for j in range(G)]
        if run is not None and run.steps_for("linear", names[0]):
            return torch.stack([self.linear(n, grouped[..., j, :], run, collect) for j, n in enumerate(names)], dim=-2)
        hd = grouped.shape[-1]
        if collect is not None:
            for j, n in enumerate(names):
                collect(n, grouped[..., j, :].reshape(-1, hd).float())
        w = self.wb("attn.wo_a").view(G, -1, hd).transpose(1, 2)
        x = grouped.reshape(-1, G, hd).transpose(0, 1)
        return torch.bmm(x, w).transpose(0, 1).reshape(*grouped.shape[:-2], G, -1)

    def pool(self, kv: torch.Tensor, gate: torch.Tensor, ape: torch.Tensor, nw: torch.Tensor, d: int,
             overlap: bool) -> torch.Tensor:
        """A compressor's entries ([B, 1, n, d]): each window of m tokens' kv summed under the softmax of its gate plus
        the position bias (CSA: window w - 1's first half beside window w's second, 2m slots), normed, with RoPE
        (`compress`) at the window's first position."""
        B = kv.shape[0]
        m = ape.shape[0]
        n = kv.shape[1] // m
        ck = kv[:, :n * m].view(B, n, m, -1)
        cg = gate[:, :n * m].view(B, n, m, -1) + ape
        if overlap:
            nk = ck.new_zeros((B, n, 2 * m, d))
            ng = cg.new_full((B, n, 2 * m, d), float("-inf"))
            nk[:, :, m:] = ck[..., d:]
            ng[:, :, m:] = cg[..., d:]
            if n > 1:
                nk[:, 1:, :m] = ck[:, :-1, :, :d]
                ng[:, 1:, :m] = cg[:, :-1, :, :d]
            ck, cg = nk, ng
        c = norm((ck * cg.softmax(dim=2, dtype=torch.float32).to(ck.dtype)).sum(dim=2), nw, self.cfg.eps)
        pos = (torch.arange(n, device=kv.device) * m).unsqueeze(0).expand(B, -1)
        cos, sin = table(self.inv_c, pos, c.dtype)
        return rope(c.unsqueeze(1), cos, sin)

    def compress(self, x: torch.Tensor, q_res: torch.Tensor, pe: dict, run: Run | None, collect):
        """The compressor's entries and their block bias ([B, 1, T, n]: 0 where the query sees the entry, -inf
        elsewhere): entry w covers tokens up to (w + 1) m - 1, so query t sees w < (t + 1) // m, and on CSA only the
        indexer's picks; at n <= index_topk it picks every one (HF's scatter gives this same bias)."""
        B, S, _ = x.shape
        m, d = self.ratio, self.cfg.head_dim
        p = "self_attn.compressor."
        kv = self.linear(p + "kv_proj", x, run, collect)
        gate = self.linear(p + "gate_proj", x, run, collect)
        ckv = self.pool(kv, gate, self.t["attn.compressor.ape"], self.t["attn.compressor.norm.weight"], d, self.csa)
        n = ckv.shape[2]
        thr = (torch.arange(S, device=x.device)[None] + 1) // m
        entries = torch.arange(n, device=x.device)
        if not self.csa or n <= self.g["index_topk"]:
            bias = ckv.new_zeros((B, 1, S, n)).masked_fill(entries.view(1, 1, 1, -1) >= thr.unsqueeze(1).unsqueeze(-1),
                                                          float("-inf"))
            return ckv, bias
        idx = self.indexer(x, q_res, pe, thr, run, collect)
        safe = torch.where(idx >= 0, idx, torch.full_like(idx, n))
        bias = ckv.new_full((B, 1, S, n + 1), float("-inf"))
        bias.scatter_(-1, safe.unsqueeze(1), 0.0)
        return ckv, bias[..., :n]

    def indexer(self, x: torch.Tensor, q_res: torch.Tensor, pe: dict, thr: torch.Tensor, run: Run | None, collect):
        """`DeepseekV4Indexer`: its own 128-wide compressor's entries scored by sum_h w_h relu(q_h . k), the future
        masked, the top index_topk per query (-1 where a pick is still in the future)."""
        B, S, _ = x.shape
        g = self.g
        di, nh = g["index_head_dim"], g["index_n_heads"]
        p = "self_attn.compressor.indexer."
        kv = self.linear(p + "kv_proj", x, run, collect)
        gate = self.linear(p + "gate_proj", x, run, collect)
        ckv = self.pool(kv, gate, self.t["attn.indexer.compressor.ape"], self.t["attn.indexer.compressor.norm.weight"],
                        di, True)[:, 0]
        cos, sin = pe["compress"]
        q = self.linear(p + "q_b_proj", q_res, run, collect).view(B, S, -1, di).transpose(1, 2)
        q = rope(q, cos, sin).transpose(1, 2)
        k = ckv.transpose(-1, -2).float().unsqueeze(1)
        w = F.linear(x, self.weight(p + "scorer.weights_proj")).float() * nh ** -0.5
        n = k.shape[-1]
        rows = S if B * S * nh * n * 4 <= INDEX_BYTES else max(1, INDEX_BYTES // (B * nh * n * 4))
        s = torch.empty(B, S, n, device=x.device)
        for r in range(0, S, rows):
            scores = F.relu(torch.matmul(q[:, r:r + rows].float(), k)) * di ** -0.5
            s[:, r:r + rows] = (scores * w[:, r:r + rows].unsqueeze(-1)).sum(dim=2)
            del scores
        s = s.masked_fill(torch.arange(n, device=x.device).view(1, 1, -1) >= thr.unsqueeze(-1), float("-inf"))
        idx = s.topk(min(g["index_topk"], n), dim=-1).indices
        return torch.where(idx >= thr.unsqueeze(-1), torch.full_like(idx, -1), idx)

    def swiglu(self, gu: torch.Tensor) -> torch.Tensor:
        lim = self.g["swiglu_limit"]
        gate, up = gu.chunk(2, dim=-1)
        return F.silu(gate.clamp(max=lim)) * up.clamp(min=-lim, max=lim)

    def shared(self, x: torch.Tensor, run, collect) -> torch.Tensor:
        lim, p = self.g["swiglu_limit"], "mlp.shared_experts."
        gate = self.linear(p + "gate_proj", x, run, collect).clamp(max=lim)
        up = self.linear(p + "up_proj", x, run, collect).clamp(min=-lim, max=lim)
        return self.linear(p + "down_proj", F.silu(gate) * up, run, collect)

    def route(self, flat: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """HF's routers: sqrtsoftplus scores of the bf16 logits; the experts top-k of scores + the correction bias, or
        (hash layers) the token's tid2eid row; their scores normalized, times routed_scaling_factor."""
        g = self.g
        scores = F.softplus(F.linear(flat, self.t["ffn.gate.weight"])).sqrt()
        if self.hash:
            idx = self.t["ffn.gate.tid2eid"][ids.reshape(-1)].long()
        else:
            idx = torch.topk(scores + self.t["ffn.gate.bias"], g["num_experts_per_tok"], dim=-1, sorted=False).indices
        w = scores.gather(1, idx)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        return idx, w * g["routed_scaling_factor"]

    def moe(self, x: torch.Tensor, ids: torch.Tensor, run, collect) -> torch.Tensor:
        """`DeepseekV4SparseMoeBlock` with eager experts: each hit expert on its rows (gate clamped above and up both
        ways at swiglu_limit), weighted, added in expert order; then the shared expert."""
        B, S, D = x.shape
        flat = x.view(-1, D)
        idx, w = self.route(flat, ids)
        E = self.g["n_routed_experts"]
        out = torch.zeros_like(flat)
        mask = F.one_hot(idx, num_classes=E + 1).permute(2, 1, 0)
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for e in hit:
            e = int(e[0])
            if e == E:
                continue
            pos, tok = torch.where(mask[e])
            cur = self.swiglu(self.linear(f"mlp.experts.{e}.gate_up_proj", flat[tok], run, collect))
            cur = self.linear(f"mlp.experts.{e}.down_proj", cur, run, collect) * w[tok, pos, None]
            out.index_add_(0, tok, cur.to(out.dtype))
        return out.view(B, S, D) + self.shared(x, run, collect)


class LazyExperts:
    """HF's stacked expert parameter, indexed per expert: each read dequantizes one expert in bf16 (the stack would be
    51 GB)."""

    def __init__(self, layer: Dsv4Layer, part: str):
        self.layer, self.part = layer, part

    def __getitem__(self, e) -> torch.Tensor:
        return self.layer.weight(f"mlp.experts.{int(e)}.{self.part}")


def hf_state(layer: Dsv4Layer) -> dict:
    """The layer's tensors as HF's decoder layer holds them (dtypes included), but the routed experts'."""
    sd = {hf_name(n): t for n, t in layer.t.items()}
    for base in layer.codes:
        if ".experts." not in base:
            sd[hf_name(base + ".weight")] = layer.wb(base)
    return sd


def check_prefix(layer: Dsv4Layer, B: int, T: int) -> int | None:
    """None while the engine's reference attention on B windows of T runs whole (and HF's fits beside it); else the
    length of the one window's prefix the check compares instead: the least multiple of the compress rates whose CSA
    entries outnumber index_topk, so the indexer's selection is in it (4224 for V4-Pro)."""
    if B * layer.cfg.heads * T * (T + (T // layer.ratio if layer.ratio else 0)) * 4 <= EAGER_WHOLE:
        return None
    rates = layer.g["compress_rates"]
    m = math.lcm(*rates.values())
    return min(T, (layer.g["index_topk"] * rates.get("compressed_sparse_attention", m) // m + 1) * m)


def check_reference(gates, cfg, snapshot, layer: Dsv4Layer, h: torch.Tensor, cos, sin, out: torch.Tensor,
                    ids: torch.Tensor | None = None) -> None:
    """The engine's bf16 layer output equals HF's `DeepseekV4DecoderLayer` on the same input, ids and weights, bit for
    bit, with HF's RoPE tables and its sliding-window causal mask (and those equal the engine's). Past `check_prefix`
    (8K) both run on the first window's prefix: causal, so the same function of the same tokens."""
    try:
        import transformers
        from transformers.masking_utils import create_sliding_window_causal_mask
        from transformers.models.deepseek_v4 import modeling_deepseek_v4 as hf
    except ImportError:
        gates.skip("hf_reference", "transformers with deepseek_v4 is not importable")
        return
    hc = transformers.AutoConfig.from_pretrained(str(snapshot))
    hc._attn_implementation = "eager"
    hc._experts_implementation = "eager"
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            hl = hf.DeepseekV4DecoderLayer(hc, layer.li)
    finally:
        torch.set_default_dtype(prev)
    res = hl.load_state_dict(hf_state(layer), strict=False, assign=True)
    want_missing = {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
    if set(res.missing_keys) != want_missing or res.unexpected_keys:
        raise ValueError(f"DeepSeek-V4 layer {layer.li}: HF's state: missing {res.missing_keys}, "
                         f"unexpected {res.unexpected_keys}")
    ex = hl.mlp.experts
    for part in ("gate_up_proj", "down_proj"):
        del ex._parameters[part]
        setattr(ex, part, LazyExperts(layer, part))
    dev = h.device
    for mod in list(hl.modules()):
        if hasattr(mod, "rotary_emb"):
            mod.rotary_emb = hf.DeepseekV4RotaryEmbedding(hc).to(dev)
    hl.eval()
    B, T = h.shape[:2]
    where = f"layer {layer.li}"
    P = check_prefix(layer, B, T)
    if P is not None:
        h, ids = h[:1, :P].contiguous(), ids[:1, :P]
        cos = {k: (c[:, :P], s[:, :P]) for k, (c, s) in cos.items()}
        with torch.no_grad():
            out = layer.forward(h, cos, sin, ids=ids)
        B, T, where = 1, P, f"{where}, the first {P} tokens of one window"
    pos = torch.arange(T, device=dev)[None]
    emb = h[:, :, 0]
    rot = hf.DeepseekV4RotaryEmbedding(hc).to(dev)
    with torch.no_grad():
        pe = {k: rot(emb, position_ids=pos, layer_type=k) for k in ("main", "compress")}
        mask = create_sliding_window_causal_mask(config=hc, inputs_embeds=emb, attention_mask=None,
                                                 past_key_values=None, position_ids=pos)
        want = hl(h, input_ids=ids, position_embeddings=pe, position_ids=pos, attention_mask=mask)
    want = want[0] if isinstance(want, tuple) else want
    mine = torch.zeros(1, 1, T, T, dtype=h.dtype, device=dev).masked_fill(~sliding(T, layer.g["sliding_window"], dev),
                                                                         torch.finfo(h.dtype).min)
    tables = all(torch.equal(pe[k][i], cos[k][i]) for k in pe for i in (0, 1))
    ok = bool(tables and torch.equal(mask.expand(B, -1, -1, -1), mine.expand(B, -1, -1, -1)) and torch.equal(want, out))
    gates.record("hf_reference", ok, where,
                 None if ok else {"tables": tables, "max": float((want.float() - out.float()).abs().max())})
    del hl
