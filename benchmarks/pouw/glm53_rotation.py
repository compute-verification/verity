"""GLM-5.3 under Pearl-C's registered stack, the keyed 8-block rotation then the keyed latent rotation
(`registration.ROTATION_8`, `LATENT_ROTATION`), on its real weights: does the registration accept,
does the rotated model compute the same function through every consumer of each rotated tensor (MLA's fused q_a/kv_a,
q_b, kv_b as a linear and as the absorbed W_UK_T / W_UV BMMs, the DSA indexer, the router, the MoE, the MTP layer,
lm_head), and which of the linears Pearl-C admits at TP8 each registered stack never reaches.

The rotation is rung 3 of the approved-weights lane (research store docs/pouw/approved-weights.md §8j; its code,
``aw_fold.block_rotation`` on the archived branch ``cursor/approved-weights-gpu-cc03``): an orthogonal Q on the residual
stream's d channels, the |S| <= d/1024 massive channels unmixed at keyed positions off every 8th, the rest in 8 blocks of
consecutive calibrated activation magnitude, each a keyed Haar rotation onto keyed positions.  Every linear that reads the
residual stream takes W·diag(γ/γ_keep)·Q, every one that writes it Qᵀ·W, the embedding E·Q, and each RMSNorm keeps γ on S and
one scale per block (γ_keep).  S leaves the chain: on a residual-reading linear its |S| columns are the uncredited side
product (Pearl-C's `Split`), so the unit reads the kept channels in order, zero-padded to `k_unit`.  The latent rotation
is the same construction on each MLA latent, keyed per layer: the q latent on q_a's rows, q_a_layernorm folded into q_b's
and wq_b's columns (its S their side product); the kv latent on kv_a's first kv_lora_rank rows, kv_a_layernorm folded into
kv_b's columns, so into the absorbed W_UK_T and W_UV.

The key is `registration.rotation_key` over an epoch started from a challenger's salt coin; the weights' root is a stand-in
(SHA-512 over the checkpoint's index), since this measures the transform and not a commitment.  S and the blocks are
calibrated on `--tokens` tokens of README.md through layers 0 … max(--layers) in FP64.

Checked (all on the checkpoint's dequantized weights, `--layers` and the MTP layer, every TP8 rank's slice):
  registration   `PearlC.check_registration` on every adopted stack and the refused ones
  reach          per linear and registered stack (`STACK`, and ROTATION_8 alone), whether it changes the weights at all,
                 held to `registration.check_linear`
  rows           Pearl-C's per-row rules (noise floor ρ·α >= 1, `live_row`) on every registered B row, vectorized and held
                 to `clear_work.row_passes` (the exact FP32 rule) on sampled and near-threshold rows
  function       FP64 forward under `STACK` against unrotated: every intermediate (the latents back through their
                 rotations), both attention routes (dense-mha with kv_b as a linear; sparse-mqa's absorbed BMMs for the last
                 token over the latent cache), the indexer's scores and top-k, the router's logits and top-k, q and the
                 attention per TP8 rank's heads, the layer's output, the MTP layer and the logits
  r8_only        the attention under ROTATION_8 alone, the other registered stack (the latent rotation leaves the MLP)
  vo             the V/O rotation and head interleave mapped onto MLA on top of `STACK` (kv_b's V rows and o_proj's
                 columns per head; heads permuted within a rank in q_b, kv_b and o_proj)
  cast           how far Pearl-C's v1 cast (e4m3(α·x) per row, both operands, per rank's k on row-parallel linears) moves
                 each layer's update, unrotated and under `STACK` (the measure registration's docstring states: 158% / 4%)
  lm_head        n = 19360 padded to whole 64-column tiles with zero rows, the last tile's rows voluntary: the domain, that
                 no other tile goes undrawn, V-EX's cap, and the credit on the full tiles' real columns
  shares         the credit map's shares (`--map`, glm53_credit_map's JSON), then per registered stack with its unreached
                 linears uncredited under Pearl-C and lm_head padded

  uv run --with tokenizers python benchmarks/pouw/glm53_rotation.py --out OUT [--layers 0,6] [--tokens 256] [--map MAP]
  python benchmarks/pouw/glm53_rotation.py --synthetic --out OUT     # a small random model, for the code path
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "clear"))
sys.path.insert(0, str(ROOT / "verity" / "core" / "protocols" / "pouw" / "tests"))
import clear_work as CW  # noqa: E402
import epoch_coins as EC  # noqa: E402
import run_result  # noqa: E402

from verity.core.protocols.pouw import registration as REG  # noqa: E402
from verity.core.protocols.pouw.audit import Epoch, Workload, weights_root  # noqa: E402
from verity.core.protocols.pouw.protocol import DomainError, Matmul, Shape  # noqa: E402
from verity.ml.protocols.pouw.schemes import pearl_c as C  # noqa: E402
from verity.ml.protocols.pouw.schemes.pearl_c_device import SM120  # noqa: E402
from verity.ml.protocols.pouw.schemes.pearl_c_work import Split, UnitRecord, pad_weight  # noqa: E402

SNAP = "/workspace/hf/hub/models--zai-org--GLM-5.3/snapshots/aca966e4e02791568aa6a4ced368624b3d897f42"
TP = 8
F64 = torch.float64
BLOCKS = 8
RATIO = 20.0
EXACT = 1e-9
SCHEMES = {"pearl-c-h100-v1-h3": C.PearlC(forming="v1", hashing="h3"),
           "pearl-c-sm120-v1-h3": C.PearlC(forming="v1", device=SM120, hashing="h3")}
#: Pearl-C's admitted linears per the credit map (art:5f07f86d…), by this tool's names
ADMITTED = ("qa", "kva", "qb", "o", "wqb", "gate", "up", "down", "router", "sgate", "sup", "egate", "eup", "eh", "lm")
#: the linears that read the residual stream (columns take Q) and write it (rows take Qᵀ)
READS = ("qa", "kva", "wk", "wp", "gate", "up", "router", "sgate", "sup", "egate", "eup", "lm")
WRITES = ("o", "down", "sdown", "edown")
ROW_PARALLEL = ("o", "down", "sdown", "edown")
#: each linear's `registration.check_linear` side; eh_proj reads the MTP layer's two rotated inputs and writes its stream
SIDES = {**{n: REG.READS_RESIDUAL for n in READS + ("eh",)}, **{n: REG.WRITES_RESIDUAL for n in WRITES},
         **{n: REG.READS_LATENT for n in ("qb", "kvb", "wqb")}}
#: the registered stack, and the one before it
STACK, R8_ONLY = (REG.ROTATION_8, REG.LATENT_ROTATION), (REG.ROTATION_8,)
MAP_IDS = {"attn.fused_qkv_a (q_a + kv_a_mqa)": ("qa", "kva"), "attn.q_b": ("qb",), "attn.o": ("o",),
           "indexer.wq_b": ("wqb",), "dense.gate_up": ("gate", "up"), "dense.down": ("down",), "moe.router": ("router",),
           "shared.gate_up": ("sgate", "sup"), "routed.w13": ("egate", "eup"), "lm_head": ("lm",), "eh_proj": ("eh",)}


# ---- the checkpoint ---------------------------------------------------------------------------------------------------

class Checkpoint:
    """Tensors of a safetensors checkpoint by name, FP8 dequantized by its 128 x 128 block scales, as FP64."""

    def __init__(self, snap: str):
        self.snap = Path(snap)
        self.cfg = json.loads((self.snap / "config.json").read_text())
        self.map = json.loads((self.snap / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers: dict[str, tuple[int, dict]] = {}

    def has(self, name: str) -> bool:
        return name in self.map

    def _raw(self, name: str) -> torch.Tensor:
        f = self.map[name]
        if f not in self.headers:
            with open(self.snap / f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self.headers[f] = (8 + n, json.loads(fh.read(n)))
        off, h = self.headers[f]
        e = h[name]
        a, b = e["data_offsets"]
        with open(self.snap / f, "rb") as fh:
            fh.seek(off + a)
            buf = bytearray(fh.read(b - a))
        dt = {"F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16, "F32": torch.float32}[e["dtype"]]
        return torch.frombuffer(buf, dtype=torch.uint8).view(dt).reshape(e["shape"])

    def get(self, name: str) -> torch.Tensor:
        w = self._raw(name)
        if w.dtype == torch.float8_e4m3fn:
            s = self._raw(name + "_scale_inv").to(F64)
            r, c = w.shape
            return w.to(F64) * s.repeat_interleave(128, 0)[:r].repeat_interleave(128, 1)[:, :c]
        return w.to(F64)

    def root(self) -> bytes:
        return hashlib.sha512((self.snap / "model.safetensors.index.json").read_bytes()).digest()


class Synthetic(Checkpoint):
    """A small random model of GLM-5.3's layout, for the code path."""

    def __init__(self):
        self.cfg = {"hidden_size": 512, "num_attention_heads": 8, "q_lora_rank": 256, "kv_lora_rank": 128,
                    "qk_nope_head_dim": 48, "qk_rope_head_dim": 16, "v_head_dim": 64, "index_n_heads": 4,
                    "index_head_dim": 32, "index_topk": 2048, "n_routed_experts": 16, "num_experts_per_tok": 4,
                    "moe_intermediate_size": 64, "intermediate_size": 256, "vocab_size": 1000, "first_k_dense_replace": 1,
                    "num_hidden_layers": 2, "routed_scaling_factor": 2.5, "rms_norm_eps": 1e-5,
                    "rope_parameters": {"rope_theta": 10000}, "indexer_types": ["full", "full"], "n_shared_experts": 1}
        self.cache: dict[str, torch.Tensor] = {}
        self.names = set(self._shapes())

    def _shapes(self) -> dict[str, tuple[int, ...]]:
        c = self.cfg
        d, h, ql, kl = c["hidden_size"], c["num_attention_heads"], c["q_lora_rank"], c["kv_lora_rank"]
        dn, dr, dv = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
        ih, idim, e, mi, i = c["index_n_heads"], c["index_head_dim"], c["n_routed_experts"], c["moe_intermediate_size"], c["intermediate_size"]
        out = {"model.embed_tokens.weight": (c["vocab_size"], d), "lm_head.weight": (c["vocab_size"], d), "model.norm.weight": (d,)}
        for L in range(c["num_hidden_layers"] + 1):
            p = f"model.layers.{L}."
            a = p + "self_attn."
            out.update({p + "input_layernorm.weight": (d,), p + "post_attention_layernorm.weight": (d,),
                        a + "q_a_proj.weight": (ql, d), a + "kv_a_proj_with_mqa.weight": (kl + dr, d),
                        a + "q_a_layernorm.weight": (ql,), a + "kv_a_layernorm.weight": (kl,),
                        a + "q_b_proj.weight": (h * (dn + dr), ql), a + "kv_b_proj.weight": (h * (dn + dv), kl),
                        a + "o_proj.weight": (d, h * dv), a + "indexer.wq_b.weight": (ih * idim, ql),
                        a + "indexer.wk.weight": (idim, d), a + "indexer.k_norm.weight": (idim,),
                        a + "indexer.k_norm.bias": (idim,), a + "indexer.weights_proj.weight": (ih, d)})
            if L < c["first_k_dense_replace"]:
                out.update({p + "mlp.gate_proj.weight": (i, d), p + "mlp.up_proj.weight": (i, d), p + "mlp.down_proj.weight": (d, i)})
            else:
                out.update({p + "mlp.gate.weight": (e, d), p + "mlp.gate.e_score_correction_bias": (e,),
                            p + "mlp.shared_experts.gate_proj.weight": (mi, d), p + "mlp.shared_experts.up_proj.weight": (mi, d),
                            p + "mlp.shared_experts.down_proj.weight": (d, mi)})
                for x in range(e):
                    out.update({p + f"mlp.experts.{x}.gate_proj.weight": (mi, d), p + f"mlp.experts.{x}.up_proj.weight": (mi, d),
                                p + f"mlp.experts.{x}.down_proj.weight": (d, mi)})
            if L == c["num_hidden_layers"]:
                out.update({p + "enorm.weight": (d,), p + "hnorm.weight": (d,), p + "eh_proj.weight": (d, 2 * d),
                            p + "shared_head.norm.weight": (d,)})
        return out

    def has(self, name: str) -> bool:
        return name in self.names

    def get(self, name: str) -> torch.Tensor:
        if name not in self.cache:
            shape = self._shapes()[name]
            g = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little") >> 1)
            if len(shape) == 1:
                t = 1 + 0.3 * torch.rand(shape, generator=g, dtype=F64) if not name.endswith(("bias",)) else 0.1 * torch.randn(shape, generator=g, dtype=F64)
                if name.endswith(("input_layernorm.weight", "post_attention_layernorm.weight")):
                    t[3] = 9.0            # a channel whose scale stands out, as massive channels' norms do
            else:
                t = torch.randn(shape, generator=g, dtype=F64) / math.sqrt(shape[1])
                if name == "model.embed_tokens.weight":
                    t[:, 3] *= 60.0      # a massive residual channel
            self.cache[name] = t
        return self.cache[name]

    def root(self) -> bytes:
        return hashlib.sha512(b"synthetic").digest()


# ---- the model --------------------------------------------------------------------------------------------------------

class Dims:
    def __init__(self, c: dict):
        self.d, self.h, self.ql, self.kl = c["hidden_size"], c["num_attention_heads"], c["q_lora_rank"], c["kv_lora_rank"]
        self.dn, self.dr, self.dv = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
        self.ih, self.idim, self.topk = c["index_n_heads"], c["index_head_dim"], c["index_topk"]
        self.e, self.k = c["n_routed_experts"], c["num_experts_per_tok"]
        self.scale_r, self.eps = c["routed_scaling_factor"], c["rms_norm_eps"]
        self.theta = c.get("rope_parameters", {}).get("rope_theta", c.get("rope_theta", 10000))
        self.fd, self.L = c["first_k_dense_replace"], c["num_hidden_layers"]
        self.indexer = c.get("indexer_types") or ["full"] * self.L
        self.vocab = c["vocab_size"]


def rms(x, g, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g


def rope(x, pos, theta):
    """Interleaved pairs (2i, 2i + 1) at θ^(−2i/d)."""
    d = x.shape[-1]
    f = theta ** (-torch.arange(0, d, 2, dtype=F64) / d)
    a = pos[:, None].to(F64) * f[None, :]
    shape = (len(pos),) + (1,) * (x.dim() - 2) + (d // 2,)
    cos, sin = a.cos().reshape(shape), a.sin().reshape(shape)
    e, o = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2], out[..., 1::2] = e * cos - o * sin, e * sin + o * cos
    return out


def pearl_cast(x: torch.Tensor) -> torch.Tensor:
    """Pearl-C v1's quantizer per row along the last axis, without its noise: e4m3(α·x)/α, α = 448/max(s + 4ρ, 2^-32),
    ρ over every 8th position."""
    s = x.abs().amax(-1, keepdim=True)
    rho = x[..., ::8].pow(2).mean(-1, keepdim=True).sqrt()
    alpha = 448.0 / torch.clamp(s + 4 * rho, min=2.0 ** -32)
    return (x * alpha).to(torch.float32).to(torch.float8_e4m3fn).to(F64) / alpha


def kept_rows(w: torch.Tensor, side: list[int]) -> torch.Tensor:
    """The unit's rows of `w`: its kept channels (all but `side`) in order, zero-padded to `Split.k_unit`."""
    if not side:
        return w
    k = w.shape[-1]
    keep = torch.tensor([j for j in range(k) if j not in set(side)])
    sp = Split(k, tuple(sorted(side))) if len(side) <= k // 100 else None
    unit = sp.k_unit if sp else -(-len(keep) // 128) * 128
    out = torch.zeros(w.shape[:-1] + (unit,), dtype=w.dtype)
    out[..., :len(keep)] = w[..., keep]
    return out


class Run:
    """One model variant: its parameters by layer and the side sets of its residual-reading linears."""

    def __init__(self, dims: Dims, mode: str = "ref"):
        self.D, self.mode = dims, mode

    def lin(self, name: str, x: torch.Tensor, w: torch.Tensor, side: list[int] | None = None) -> torch.Tensor:
        if self.mode == "ref" or name not in ADMITTED:
            return x @ w.T
        side = side or []
        out = x[:, side] @ w[:, side].T if side else 0.0
        xk, wk = kept_rows(x, side), kept_rows(w, side)
        if name in ROW_PARALLEL:
            parts = zip(xk.chunk(TP, -1), wk.chunk(TP, -1))
        else:
            parts = [(xk, wk)]
        for xa, wb in parts:
            out = out + pearl_cast(xa) @ pearl_cast(wb).T
        return out

    def attention(self, P: dict, x: torch.Tensor, pos: torch.Tensor, trace: dict, tag: str) -> torch.Tensor:
        D, T, side = self.D, x.shape[0], P.get("side")
        qa = self.lin("qa", x, P["qa"], side)
        kv = self.lin("kva", x, P["kva"], side)
        ql = rms(qa, P["qa_g"], D.eps)
        cn = rms(kv[:, :D.kl], P["kva_g"], D.eps)
        kpe = rope(kv[:, D.kl:], pos, D.theta)
        q = self.lin("qb", ql, P["qb"], P.get("q_side")).view(T, D.h, D.dn + D.dr)
        qn, qp = q[..., :D.dn], rope(q[..., D.dn:], pos, D.theta)
        kvb = (cn @ P["kvb"].T).view(T, D.h, D.dn + D.dv)
        kn, v = kvb[..., :D.dn], kvb[..., D.dn:]
        scale = (D.dn + D.dr) ** -0.5
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        sc = (torch.einsum("thd,shd->hts", qn, kn) + torch.einsum("thd,sd->hts", qp, kpe)) * scale
        sel = None
        if P.get("wqb") is not None:
            sel = self.indexer(P, x, ql, pos, trace, tag)
            mask = mask & sel
        p = torch.softmax(sc.masked_fill(~mask, float("-inf")), -1)
        o = torch.einsum("hts,shd->thd", p, v).reshape(T, D.h * D.dv)
        # sparse-mqa: the last token over the latent cache, kv_b split into the load-derived W_UK_T and W_UV
        wb = P["kvb"].view(D.h, D.dn + D.dv, D.kl)
        w_uk, w_uv = wb[:, :D.dn], wb[:, D.dn:]
        qlat = torch.einsum("hd,hdc->hc", qn[-1], w_uk)
        sa = (qlat @ cn.T + qp[-1] @ kpe.T) * scale
        if sel is not None:
            sa = sa.masked_fill(~sel[-1][None, :], float("-inf"))
        lat = torch.softmax(sa, -1) @ cn
        oa = torch.einsum("hc,hdc->hd", lat, w_uv).reshape(D.h * D.dv)
        trace[tag + "q_latent"], trace[tag + "kv_latent"], trace[tag + "k_pe"] = ql, cn, kpe
        trace[tag + "q"], trace[tag + "attn"], trace[tag + "attn_absorbed_last"] = q, o, oa
        trace[tag + "absorbed_vs_dense"] = float((oa - o[-1]).norm() / o[-1].norm())
        return self.lin("o", o, P["o"])

    def indexer(self, P, x, ql, pos, trace, tag) -> torch.Tensor:
        D, T = self.D, x.shape[0]
        qi = self.lin("wqb", ql, P["wqb"], P.get("q_side")).view(T, D.ih, D.idim)
        qi = torch.cat([rope(qi[..., :D.dr], pos, D.theta), qi[..., D.dr:]], -1)
        side = P.get("side")
        ki = self.lin("wk", x, P["wk"], side)
        ki = torch.nn.functional.layer_norm(ki, (D.idim,), P["kn_w"], P["kn_b"], D.eps)
        ki = torch.cat([rope(ki[:, :D.dr], pos, D.theta), ki[:, D.dr:]], -1)
        w = self.lin("wp", x, P["wp"], side) * D.ih ** -0.5 * D.idim ** -0.5
        s = torch.einsum("th,ths->ts", w, torch.relu(torch.einsum("thd,sd->ths", qi, ki)))
        causal = torch.ones(T, T, dtype=torch.bool).tril()
        s = s.masked_fill(~causal, float("-inf"))
        top = s.topk(min(D.topk, T), -1).indices
        sel = torch.zeros(T, T, dtype=torch.bool).scatter_(1, top, True) & causal
        trace[tag + "index_scores"], trace[tag + "index_sel"] = s.masked_fill(~causal, 0.0), sel
        return sel

    def mlp(self, P: dict, x2: torch.Tensor, trace: dict, tag: str, experts=None) -> torch.Tensor:
        D, side = self.D, P.get("side")
        if "gate" in P:
            g, u = self.lin("gate", x2, P["gate"], side), self.lin("up", x2, P["up"], side)
            return self.lin("down", torch.nn.functional.silu(g) * u, P["down"])
        logits = self.lin("router", x2, P["router"], side)
        s = torch.sigmoid(logits)
        idx = (s + P["bias"]).topk(D.k, -1).indices
        w = s.gather(1, idx)
        w = w / w.sum(-1, keepdim=True) * D.scale_r
        trace[tag + "router_logits"], trace[tag + "router_sel"] = logits, idx.sort(-1).values
        sg, su = self.lin("sgate", x2, P["sgate"], side), self.lin("sup", x2, P["sup"], side)
        y = self.lin("sdown", torch.nn.functional.silu(sg) * su, P["sdown"])
        for e in sorted(set(idx.flatten().tolist())):
            rows, slot = (idx == e).nonzero(as_tuple=True)
            eg, eu, ed = experts(e)
            xe = x2[rows]
            h = torch.nn.functional.silu(self.lin("egate", xe, eg, side)) * self.lin("eup", xe, eu, side)
            y = y.index_add(0, rows, self.lin("edown", h, ed) * w[rows, slot][:, None])
        return y

    def layer(self, P, h, pos, trace, tag, experts=None):
        a = self.attention(P, rms(h, P["in_g"], self.D.eps), pos, trace, tag)
        h1 = h + a
        m = self.mlp(P, rms(h1, P["post_g"], self.D.eps), trace, tag, experts)
        trace[tag + "update"] = a + m
        return h1 + m


def load_layer(ck: Checkpoint, L: int, D: Dims) -> dict:
    p, a = f"model.layers.{L}.", f"model.layers.{L}.self_attn."
    P = {"in_g": ck.get(p + "input_layernorm.weight"), "post_g": ck.get(p + "post_attention_layernorm.weight"),
         "qa": ck.get(a + "q_a_proj.weight"), "kva": ck.get(a + "kv_a_proj_with_mqa.weight"),
         "qa_g": ck.get(a + "q_a_layernorm.weight"), "kva_g": ck.get(a + "kv_a_layernorm.weight"),
         "qb": ck.get(a + "q_b_proj.weight"), "kvb": ck.get(a + "kv_b_proj.weight"), "o": ck.get(a + "o_proj.weight")}
    if ck.has(a + "indexer.wq_b.weight"):
        P.update(wqb=ck.get(a + "indexer.wq_b.weight"), wk=ck.get(a + "indexer.wk.weight"),
                 kn_w=ck.get(a + "indexer.k_norm.weight"), kn_b=ck.get(a + "indexer.k_norm.bias"),
                 wp=ck.get(a + "indexer.weights_proj.weight"))
    if ck.has(p + "mlp.gate_proj.weight"):
        P.update(gate=ck.get(p + "mlp.gate_proj.weight"), up=ck.get(p + "mlp.up_proj.weight"), down=ck.get(p + "mlp.down_proj.weight"))
    else:
        P.update(router=ck.get(p + "mlp.gate.weight"), bias=ck.get(p + "mlp.gate.e_score_correction_bias"),
                 sgate=ck.get(p + "mlp.shared_experts.gate_proj.weight"), sup=ck.get(p + "mlp.shared_experts.up_proj.weight"),
                 sdown=ck.get(p + "mlp.shared_experts.down_proj.weight"))
    return P


def expert_loader(ck: Checkpoint, L: int):
    p = f"model.layers.{L}.mlp.experts."
    return lambda e: tuple(ck.get(p + f"{e}.{n}_proj.weight") for n in ("gate", "up", "down"))


# ---- the rotation -----------------------------------------------------------------------------------------------------

def haar(n: int, key: bytes, label: str) -> torch.Tensor:
    seed = int.from_bytes(hashlib.sha512(key + label.encode()).digest()[:16], "little")
    g = np.random.Generator(np.random.PCG64(seed)).standard_normal((n, n))
    q, r = np.linalg.qr(g)
    return torch.from_numpy(q * np.sign(np.diag(r)))


class Rotation:
    """Q on d channels (x' = x·Q): S unmixed at keyed positions off every 8th, the rest in `blocks` blocks of consecutive
    calibrated magnitude, each a keyed Haar onto keyed positions (``aw_fold.block_rotation``)."""

    def __init__(self, d: int, s: list[int], prof: list[float], key: bytes, label: str, blocks: int = BLOCKS):
        seed = int.from_bytes(hashlib.sha512(key + f"{label}/positions".encode()).digest()[:16], "little")
        pos = np.random.Generator(np.random.PCG64(seed)).permutation(d).tolist()
        ps = [p for p in pos if p % 8][:len(s)]
        rest = [p for p in pos if p not in set(ps)]
        others = sorted(set(range(d)) - set(s), key=lambda c: prof[c])
        self.d, self.s, self.ps = d, list(s), ps
        self.groups = []                      # (channels, positions, Haar)
        lo = 0
        for b in range(blocks):
            idx = others[b * len(others) // blocks:(b + 1) * len(others) // blocks]
            self.groups.append((torch.tensor(idx), torch.tensor(rest[lo:lo + len(idx)]), haar(len(idx), key, f"{label}/block/{b}")))
            lo += len(idx)

    def right(self, w: torch.Tensor) -> torch.Tensor:
        """w·Q along the last axis (a row vector's basis change, a reading linear's columns)."""
        out = torch.zeros_like(w)
        if self.s:
            out[..., self.ps] = w[..., self.s]
        for c, p, hq in self.groups:
            out[..., p] = w[..., c] @ hq
        return out

    def left_t(self, w: torch.Tensor) -> torch.Tensor:
        """Qᵀ·w along the first axis (a writing linear's rows)."""
        return self.right(w.T).T

    def keep(self, g: torch.Tensor) -> torch.Tensor:
        """γ_keep, by channel: γ on S, each block's RMS of γ on its channels."""
        k = torch.ones_like(g)
        if self.s:
            k[self.s] = g[self.s]
        for c, _, _ in self.groups:
            k[c] = g[c].pow(2).mean().sqrt()
        return k

    def norm(self, g: torch.Tensor) -> torch.Tensor:
        """The rotated norm's scale, by position: γ_keep carried to its positions (constant on each block, so it commutes
        with the block's Haar)."""
        k = self.keep(g)
        out = torch.empty_like(k)
        if self.s:
            out[self.ps] = k[self.s]
        for c, p, _ in self.groups:
            out[p] = k[c[0]]
        return out

    def fold(self, w: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """A reading linear behind the norm of scale γ: W·diag(γ/γ_keep)·Q."""
        return self.right(w * (g / self.keep(g))[None, :])


def rotate_layer(P: dict, Q: Rotation) -> dict:
    """ROTATION_8 on one layer: the residual's readers and writers, the norms; MLA's latent side untouched."""
    R = dict(P)
    R["in_g"], R["post_g"] = Q.norm(P["in_g"]), Q.norm(P["post_g"])
    for n in ("qa", "kva", "wk", "wp"):
        if P.get(n) is not None:
            R[n] = Q.fold(P[n], P["in_g"])
    for n in ("gate", "up", "router", "sgate", "sup"):
        if n in P:
            R[n] = Q.fold(P[n], P["post_g"])
    for n in ("o", "down", "sdown"):
        if n in P:
            R[n] = Q.left_t(P[n])
    R["side"] = Q.ps
    return R


def rotate_experts(experts, Q: Rotation, post_g: torch.Tensor, rows: "Rows | None" = None, prefix: str = ""):
    """The rotated experts; with `rows`, each expert's registered w13 rows (both variants) recorded the first time it is
    rotated, so a pass that rotates them all records them all (`record_rest`)."""
    seen: set[int] = set()

    def f(e):
        g, u, d = experts(e)
        r = Q.fold(g, post_g), Q.fold(u, post_g), Q.left_t(d)
        if rows is not None and e not in seen:
            seen.add(e)
            for n, a, b in ((f"{prefix}egate", g, r[0]), (f"{prefix}eup", u, r[1])):
                rows.add(f"{n}/unrotated", a, sample=4)
                rows.add(f"{n}/rotated", b, Q.ps, sample=4)
        return r

    def record_rest(n_experts: int) -> None:
        for e in range(n_experts):
            if e not in seen:
                f(e)
    f.record_rest = record_rest
    return f


def rotate_latent(P: dict, Rq: Rotation, Rkv: Rotation, D: Dims) -> dict:
    """The latent placement: q latent (q_a's rows, q_a_layernorm, q_b's and wq_b's columns) and kv latent (kv_a's first
    kv_lora_rank rows, kv_a_layernorm, kv_b's columns; the absorbed W_UK_T and W_UV are kv_b's, so they follow)."""
    R = dict(P)
    R["qa"] = Rq.left_t(P["qa"])
    R["qa_g"] = Rq.norm(P["qa_g"])
    R["qb"] = Rq.fold(P["qb"], P["qa_g"])
    if P.get("wqb") is not None:
        R["wqb"] = Rq.fold(P["wqb"], P["qa_g"])
    R["kva"] = torch.cat([Rkv.left_t(P["kva"][:D.kl]), P["kva"][D.kl:]])
    R["kva_g"] = Rkv.norm(P["kva_g"])
    R["kvb"] = Rkv.fold(P["kvb"], P["kva_g"])
    return R


def vo_interleave(P: dict, D: Dims, key: bytes) -> dict:
    """VO_ROTATION and HEAD_INTERLEAVE on MLA: per head a keyed Haar on kv_b's V rows and o_proj's columns; then, within
    each rank's heads, one keyed permutation of q_b's, kv_b's and o_proj's head blocks."""
    R = dict(P)
    hq, hkv = D.dn + D.dr, D.dn + D.dv
    kvb, o = P["kvb"].clone().view(D.h, hkv, D.kl), P["o"].clone().view(P["o"].shape[0], D.h, D.dv)
    for h in range(D.h):
        r = haar(D.dv, key, f"vo/{h}")
        kvb[h, D.dn:] = r.T @ kvb[h, D.dn:]
        o[:, h] = o[:, h] @ r
    per = D.h // TP
    seed = int.from_bytes(hashlib.sha512(key + b"interleave").digest()[:16], "little")
    rng = np.random.Generator(np.random.PCG64(seed))
    perm = torch.tensor([r * per + int(j) for r in range(TP) for j in rng.permutation(per)])
    R["qb"] = P["qb"].view(D.h, hq, -1)[perm].reshape(P["qb"].shape)
    R["kvb"] = kvb[perm].reshape(P["kvb"].shape)
    R["o"] = o[:, perm].reshape(P["o"].shape)
    return R


# ---- the checks -------------------------------------------------------------------------------------------------------

def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.to(F64), b.to(F64)
    n = b.norm()
    return float((a - b).norm() / n) if n > 0 else float((a - b).norm())


def row_stats(w: torch.Tensor) -> dict:
    """Pearl-C's per-row rules on the FP32 rows of `w`, vectorized: (pass mask, noise margin ρ·α, live slack k/64 − dead)."""
    x = w.to(torch.float32).to(F64)
    k = x.shape[-1]
    s = x.abs().amax(-1)
    rho2 = x[..., ::8].pow(2).sum(-1) / len(range(0, k, 8))
    rho = rho2.sqrt()
    alpha = 448.0 / torch.clamp(s + 4 * rho, min=2.0 ** -32)
    noise = rho * alpha
    dead = (x.pow(2) >= 64 * rho2[..., None]).sum(-1)
    slack = k / 64 - dead.to(F64)
    return {"pass": (noise >= 1.0) & (slack >= 0), "noise": noise, "slack": slack}


def exact_pass(row: torch.Tensor) -> bool:
    words = row.to(torch.float32).numpy().view(np.uint32).tolist()
    return CW.row_passes(words)


class Rows:
    """Every registered B row's verdict, per linear and variant, with the exact rule on a sample."""

    def __init__(self, sample: int = 24):
        self.out: dict[str, dict] = {}
        self.sample = sample
        self.mismatch: list[str] = []

    def add(self, key: str, w: torch.Tensor, side: list[int] | None = None, row_parallel: bool = False,
            sample: int | None = None) -> None:
        unit = kept_rows(w, side or [])
        mats = list(unit.chunk(TP, -1)) if row_parallel else list(unit.split(8192, 0))
        o = self.out.setdefault(key, {"rows": 0, "fail": 0, "fail_noise": 0, "fail_live": 0, "min_noise": math.inf,
                                      "min_slack": math.inf, "k": mats[0].shape[-1], "exact_checked": 0})
        n_exact = self.sample if sample is None else sample
        for m in mats:
            st = row_stats(m)
            o["rows"] += m.shape[0]
            o["fail"] += int((~st["pass"]).sum())
            o["fail_noise"] += int((st["noise"] < 1).sum())
            o["fail_live"] += int((st["slack"] < 0).sum())
            o["min_noise"] = min(o["min_noise"], float(st["noise"].min()))
            o["min_slack"] = min(o["min_slack"], float(st["slack"].min()))
            g = torch.Generator().manual_seed(o["rows"])
            near = torch.argsort(torch.minimum((st["noise"] - 1).abs() * 64, st["slack"].abs()))[: n_exact // 2]
            pick = torch.unique(torch.cat([near, torch.randint(m.shape[0], (n_exact // 2,), generator=g)]))
            for i in pick.tolist():
                if exact_pass(m[i]) != bool(st["pass"][i]):
                    self.mismatch.append(f"{key} row {i}")
                o["exact_checked"] += 1


def calibrate(ck: Checkpoint, D: Dims, ids: torch.Tensor, upto: int, log, cap: int | None = None) -> dict:
    """FP64 forward of layers 0 … upto on `ids`: every layer's input, S (channel_scores' rule: largest |h| at a norm's input
    at least RATIO times that norm's median channel's, at most d/1024, largest first), the magnitude profile, and the
    latents' (q_a's and kv_a's outputs before their norms) for the latent placement."""
    T = len(ids)
    pos = torch.arange(T)
    h = ck.get("model.embed_tokens.weight")[ids]
    run = Run(D)
    amax, prof, lat = [], [], {"q": [], "kv": []}
    inputs = {}
    for L in range(upto + 1):
        t0 = time.time()
        P = load_layer(ck, L, D)
        inputs[L] = h
        tr: dict = {}
        x = rms(h, P["in_g"], D.eps)
        a = run.attention(P, x, pos, tr, "")
        h1 = h + a
        for v in (h, h1):
            amax.append(v.abs().amax(0))
            prof.append((v / v.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-30)).abs().mean(0).clamp_min(1e-12).log())
        lat["q"].append(x @ P["qa"].T)
        lat["kv"].append((x @ P["kva"].T)[:, :D.kl])
        h = h1 + run.mlp(P, rms(h1, P["post_g"], D.eps), tr, "", expert_loader(ck, L))
        log(f"calibration layer {L}: {time.time() - t0:.1f}s")
    inputs[upto + 1] = h

    def split(amaxes, cap):
        a = torch.stack(amaxes)
        score = (a / a.median(1, keepdim=True).values.clamp_min(1e-30)).amax(0)
        top = torch.argsort(score, descending=True)
        return [int(c) for c in top[:cap] if score[c] >= RATIO], {int(c): round(float(score[c]), 1) for c in top[:8]}

    s, top = split(amax, D.d // 1024 if cap is None else cap)
    out = {"inputs": inputs, "S": s, "S_scores": top, "profile": torch.stack(prof).mean(0).exp().tolist()}
    for n, dim in (("q", D.ql), ("kv", D.kl)):
        am = [v.abs().amax(0) for v in lat[n]]
        sl, tl = split(am, dim // 1024 if cap is None else cap)
        pr = torch.stack([(v / v.pow(2).mean(-1, keepdim=True).sqrt()).abs().mean(0).clamp_min(1e-12).log() for v in lat[n]])
        out[n] = {"S": sl, "S_scores": tl, "profile": pr.mean(0).exp().tolist()}
    return out


def latent_back(y: torch.Tensor, R: Rotation, g: torch.Tensor) -> torch.Tensor:
    """A rotated latent after its norm, back to the checkpoint's: y·Rᵀ, with γ_keep undone to γ."""
    return unrotate(y, R) * (g / R.keep(g))


def compare(tr0: dict, tr1: dict, Q: Rotation | None, prefix: str, latents: dict | None = None) -> dict:
    """Rotated (tr1) against unrotated (tr0): residual-basis values rotated back, latents (`latents`: trace name ->
    (rotation, norm scale)) back by `latent_back`, the rest as they are, top-k sets equal."""
    out = {}
    for name, a in tr0.items():
        if not name.startswith(prefix) or isinstance(a, float):
            continue
        b = tr1[name]
        short = name[len(prefix):]
        if a.dtype == torch.bool or a.dtype == torch.int64:
            out[short] = {"equal": bool(torch.equal(a, b))}
            continue
        if short == "update" and Q is not None:
            b = unrotate(b, Q)
        if latents and short in latents:
            b = latent_back(b, *latents[short])
        out[short] = {"rel": rel(b, a)}
    return out


def unrotate(y: torch.Tensor, Q: Rotation) -> torch.Tensor:
    """y·Qᵀ: a residual-basis row back to the checkpoint's channels."""
    out = torch.zeros_like(y)
    if Q.s:
        out[..., Q.s] = y[..., Q.ps]
    for c, p, hq in Q.groups:
        out[..., c] = y[..., p] @ hq.T
    return out


def router_margin(tr: dict, tag: str, P: dict, k: int) -> float:
    """The smallest gap between the k-th and (k+1)-th choice score, over tokens: how near a top-k tie the router is."""
    s = torch.sigmoid(tr[tag + "router_logits"]) + P["bias"]
    v = s.topk(k + 1, -1).values
    return float((v[:, k - 1] - v[:, k]).min())


def latents(D: Dims, L: int, cal: dict, key: bytes) -> tuple[Rotation, Rotation]:
    """Layer L's q-latent and kv-latent rotations, keyed per layer, on the latents' calibrated S and profiles."""
    return (Rotation(D.ql, cal["q"]["S"], cal["q"]["profile"], key, f"q-latent/{L}"),
            Rotation(D.kl, cal["kv"]["S"], cal["kv"]["profile"], key, f"kv-latent/{L}"))


def rotate_stack(P: dict, Q: Rotation, Rq: Rotation, Rkv: Rotation, D: Dims) -> dict:
    """The registered stack (`STACK`) on one layer: ROTATION_8, then the latent rotation; q_b and wq_b read the q
    latent's S as their side product."""
    R = rotate_latent(rotate_layer(P, Q), Rq, Rkv, D)
    R["q_side"] = Rq.ps
    return R


def reach(P: dict, R: dict, experts=None, rex=None) -> dict[str, bool]:
    """Per linear, whether the variant changes its weights at all."""
    out = {n: (not torch.equal(P[n], R[n])) for n in P if isinstance(P[n], torch.Tensor) and P[n].dim() == 2}
    if experts:
        for n, a, b in zip(("egate", "eup", "edown"), experts(0), rex(0)):
            out[n] = not torch.equal(a, b)
    return out


def per_rank(tr0: dict, tr1: dict, D: Dims) -> dict:
    """q and the attention output per TP8 rank's heads (h/8 each), the rotated against the unrotated."""
    q0, q1 = tr0["q"].chunk(TP, 1), tr1["q"].chunk(TP, 1)
    a0, a1 = tr0["attn"].chunk(TP, -1), tr1["attn"].chunk(TP, -1)
    return {"q": [rel(b, a) for a, b in zip(q0, q1)], "attn": [rel(b, a) for a, b in zip(a0, a1)]}


def check_layer(ck, D, L, h, pos, Q, key, cal, rows: Rows, log) -> dict:
    t0 = time.time()
    P = load_layer(ck, L, D)
    experts = expert_loader(ck, L) if "router" in P else None
    Rq, Rkv = latents(D, L, cal, key)
    R8 = rotate_layer(P, Q)
    R = rotate_stack(P, Q, Rq, Rkv, D)
    rex = rotate_experts(experts, Q, P["post_g"], rows) if experts else None
    hr = Q.right(h)
    lat_back = {"q_latent": (Rq, P["qa_g"]), "kv_latent": (Rkv, P["kva_g"])}
    res: dict = {"layer": L, "q_S": Rq.s, "kv_S": Rkv.s}
    tr0, tr1 = {}, {}
    y0 = Run(D).layer(P, h, pos, tr0, "", experts)
    y1 = Run(D).layer(R, hr, pos, tr1, "", rex)
    if rex:
        rex.record_rest(D.e)
    res["function"] = compare(tr0, tr1, Q, "", lat_back)
    res["function"]["output"] = {"rel": rel(unrotate(y1, Q), y0)}
    res["per_rank"] = per_rank(tr0, tr1, D)
    res["absorbed_vs_dense"] = tr0["absorbed_vs_dense"]
    if "router" in P:
        res["router_margin"] = router_margin(tr0, "", P, D.k)
    res["reach"] = {"stack": reach(P, R, experts, rex), "r8_only": reach(P, R8, experts, rex)}
    # ROTATION_8 alone, the other registered stack (attention only: the latent rotation leaves the MLP as it has it)
    x1 = rms(hr, R["in_g"], D.eps)
    a0 = P["o"] @ tr0["attn"].T
    tr2: dict = {}
    a2 = Run(D).attention(R8, x1, pos, tr2, "")
    r8 = {"attn_out": {"rel": rel(unrotate(a2, Q), a0.T)}, "q_latent": {"rel": rel(tr2["q_latent"], tr0["q_latent"])},
          "kv_latent": {"rel": rel(tr2["kv_latent"], tr0["kv_latent"])}}
    if "index_sel" in tr0:
        r8["index_sel"] = {"equal": bool(torch.equal(tr2["index_sel"], tr0["index_sel"]))}
    res["r8_only"] = r8
    # the V/O stack with the latent rotation on MLA
    VO = vo_interleave(R, D, key)
    tr3: dict = {}
    a3 = Run(D).attention(VO, x1, pos, tr3, "")
    res["vo"] = {"attn_out": {"rel": rel(unrotate(a3, Q), a0.T)},
                 "attn_absorbed_last_after_o": {"rel": rel(unrotate((VO["o"] @ tr3["attn_absorbed_last"])[None, :], Q)[0],
                                                           P["o"] @ tr0["attn_absorbed_last"])}}
    # the cast: how far it moves the layer's update
    trc0, trc1 = {}, {}
    Run(D, "cast").layer(P, h, pos, trc0, "", experts)
    Run(D, "cast").layer(R, hr, pos, trc1, "", rotate_experts(experts, Q, P["post_g"]) if experts else None)
    res["cast"] = {"unrotated": rel(trc0["update"], tr0["update"]), "rotated": rel(unrotate(trc1["update"], Q), tr0["update"])}
    # every registered row, both variants, every rank's slice
    for n in ("qa", "kva", "qb", "o", "wqb", "gate", "up", "down", "router", "sgate", "sup"):
        if P.get(n) is None:
            continue
        rp = n in ROW_PARALLEL
        rows.add(f"{n}/unrotated", P[n], None, rp)
        rows.add(f"{n}/rotated", R[n], R["side"] if n in READS else R["q_side"] if n in ("qb", "wqb") else None, rp)
    res["seconds"] = round(time.time() - t0, 1)
    log(f"layer {L}: output rel {res['function']['output']['rel']:.2e}, r8 only {r8['attn_out']['rel']:.2e}, "
        f"vo {res['vo']['attn_out']['rel']:.2e}, cast {res['cast']}, {res['seconds']}s")
    return res


def check_mtp(ck, D, L, h, ids, pos, Q, key, cal, rows: Rows, log) -> dict:
    """The MTP layer: enorm(embed(next token)) ‖ hnorm(h) through eh_proj, its decoder layer, shared_head.norm and lm_head
    (rank 0's vocabulary slice).  `h` stands in for the last layer's output."""
    p = f"model.layers.{L}."
    E = ck.get("model.embed_tokens.weight")
    emb = E[torch.cat([ids[1:], ids[-1:]])]
    ge, gh, gs = ck.get(p + "enorm.weight"), ck.get(p + "hnorm.weight"), ck.get(p + "shared_head.norm.weight")
    eh = ck.get(p + "eh_proj.weight")
    d = D.d
    z0 = torch.cat([rms(emb, ge, D.eps), rms(h, gh, D.eps)], -1) @ eh.T
    ehr = Q.left_t(torch.cat([Q.fold(eh[:, :d], ge), Q.fold(eh[:, d:], gh)], -1))
    z1 = torch.cat([rms(Q.right(emb), Q.norm(ge), D.eps), rms(Q.right(h), Q.norm(gh), D.eps)], -1) @ ehr.T
    out = {"eh_proj": {"rel": rel(unrotate(z1, Q), z0)}}
    P = load_layer(ck, L, D)
    experts = expert_loader(ck, L)
    Rq, Rkv = latents(D, L, cal, key)
    R = rotate_stack(P, Q, Rq, Rkv, D)
    rex = rotate_experts(experts, Q, P["post_g"], rows, "mtp.")
    tr0, tr1 = {}, {}
    y0 = Run(D).layer(P, z0, pos, tr0, "", experts)
    y1 = Run(D).layer(R, z1, pos, tr1, "", rex)
    rex.record_rest(D.e)
    out["layer"] = compare(tr0, tr1, Q, "", {"q_latent": (Rq, P["qa_g"]), "kv_latent": (Rkv, P["kva_g"])})
    out["layer"]["output"] = {"rel": rel(unrotate(y1, Q), y0)}
    out["per_rank"] = per_rank(tr0, tr1, D)
    out["reach"] = reach(P, R, experts, rex)
    lm = ck.get("lm_head.weight")[: D.vocab // TP].clone()
    l0 = rms(y0, gs, D.eps) @ lm.T
    l1 = rms(y1, Q.norm(gs), D.eps) @ Q.fold(lm, gs).T
    out["logits"] = {"rel": rel(l1, l0), "argmax_equal": bool(torch.equal(l0.argmax(-1), l1.argmax(-1)))}
    rows.add("eh/unrotated", eh)
    rows.add("eh/rotated", ehr, Q.ps + [d + p_ for p_ in Q.ps])
    for n in ("qa", "kva", "qb", "o", "wqb", "router", "sgate", "sup"):
        if P.get(n) is not None:
            rows.add(f"mtp.{n}/unrotated", P[n], None, n in ROW_PARALLEL)
            rows.add(f"mtp.{n}/rotated", R[n], R["side"] if n in READS else R["q_side"] if n in ("qb", "wqb") else None,
                     n in ROW_PARALLEL)
    log(f"mtp layer {L}: eh_proj {out['eh_proj']['rel']:.2e}, output {out['layer']['output']['rel']:.2e}, logits {out['logits']['rel']:.2e}")
    return out


def check_lm_head(ck, D, h, Q, rows: Rows, log) -> dict:
    """lm_head per rank, rotated with the final norm folded, padded from n = vocab/8 to whole 64-column tiles with zero rows."""
    g = ck.get("model.norm.weight")
    lm = ck.get("lm_head.weight")
    n = D.vocab // TP
    tile = SCHEMES["pearl-c-h100-v1-h3"].tile[1]
    npad = -(-n // tile) * tile
    for r in range(TP):
        rank = lm[r * n:(r + 1) * n]
        rows.add("lm/unrotated", rank)
        lmr = Q.fold(rank, g)
        rows.add("lm/rotated", lmr, Q.ps)
        if r == 0:
            lm0 = lmr
    l0 = rms(h, g, D.eps) @ lm[:n].T
    l1 = rms(Q.right(h), Q.norm(g), D.eps) @ lm0.T
    del lm
    out = {"n": n, "n_padded": npad, "function": {"rel": rel(l1, l0), "argmax_equal": bool(torch.equal(l0.argmax(-1), l1.argmax(-1)))}}
    k = kept_rows(lm0[:1], Q.ps).shape[-1]
    k_kept = D.d - len(Q.ps)
    padded = torch.cat([kept_rows(lm0, Q.ps), torch.zeros(npad - n, k, dtype=F64)])
    st = row_stats(padded)
    out["padded_rows_fail"] = bool((~st["pass"][n:]).all())
    out["padded_rows_exact_fail"] = all(not exact_pass(padded[i]) for i in range(n, npad))
    out["real_rows_fail"] = int((~st["pass"][:n]).sum())
    for name, s in SCHEMES.items():
        dom = {}
        for nn in (n, npad):
            try:
                s.check(Matmul(0, Shape(64, k, nn), 0))
                dom[nn] = "admitted"
            except DomainError as e:
                dom[nn] = str(e)
        _, voluntary = pad_weight([b"\0"] * n, s.tile[1])
        rec = UnitRecord(Matmul(0, Shape(64, k, npad), 0), 64, k_kept, (), (), (), voluntary)
        _, credit = CW.work(s, [rec])
        honest = (1 - s.cap) * Fraction(s.credit_of(Shape(64, k_kept, n)))
        cells = sum(len(r) * len(c) for r, c in (rec.credited(t) for t in s.tiles(rec.matmul.shape)))
        undrawn = [t.cols for _, t, w in CW.tickets(s, [rec]) if w == 0]
        out[name] = {"domain": dom, "padded_tile": [voluntary[0], voluntary[-1]], "credited_cells": cells,
                     "real_cells": 64 * n, "credited_fraction": cells / (64 * n),
                     "undrawn_tiles_are_the_padded_one": bool(undrawn) and all(c == voluntary for c in undrawn),
                     "within_vex_cap": len(voluntary) <= CW.voluntary_cap(npad),
                     "credit_over_real_columns_honest": float(credit / honest)}
    log(f"lm_head: rel {out['function']['rel']:.2e}, padded rows fail {out['padded_rows_fail']}, {out['pearl-c-h100-v1-h3']}")
    return out


def shares(map_doc: dict, unreached: dict[str, set[str]], lm_padded: dict[str, float]) -> dict:
    """The credit map's shares, then, per registered stack (`unreached`: its label -> the admitted linears it leaves
    unreached), Pearl-C's with those uncredited and lm_head credited, where its padded shape is admitted, on the share of
    its MACs its full tiles' real columns hold (`lm_padded`, by scheme), and the mixes."""
    # glm53_credit_map's steps: prefill leaves lm_head out (one row per sequence)
    steps = {"decode": (("both", "decode"), ()), "prefill": (("both", "prefill"), ("lm_head",)),
             "decode+mtp": (("both", "decode", "mtp"), ())}
    pearls = ("pearl-c-h100-v1-h3", "pearl-c-sm120-v1-h3")
    mixes = {"mix H200": ("pearl-c-h100-v1-h3", "h1-sm_90"), "mix RTX PRO 6000": ("pearl-c-sm120-v1-h3", "h1-sm_120")}
    rows = map_doc["linears"]

    def credited(r, scheme, gone) -> float:
        """The share of the linear's MACs credited: lm_head padded earns on the real columns of its full tiles."""
        c = float(r["schemes"][scheme]["credited"])
        if gone is None or scheme not in pearls:
            return c
        base = r["id"].removeprefix("mtp.")
        if base == "lm_head":
            return lm_padded[scheme]
        names = MAP_IDS.get(base, ())
        return 0.0 if any(x in gone for x in names) else c

    out = {}
    for label, gone in (("the map as it stands", None), *((f"under {k}, lm_head padded", v) for k, v in unreached.items())):
        tab = {}
        for step, (routes, left_out) in steps.items():
            rs = [r for r in rows if r["route"] in routes and r["id"] not in left_out]
            tot = sum(r["macs"] for r in rs)
            v = {s: sum(r["macs"] * credited(r, s, gone) for r in rs) / tot for s in rows[0]["schemes"]}
            for m, (a, b) in mixes.items():
                v[m] = sum(r["macs"] * max(credited(r, a, gone), credited(r, b, gone)) for r in rs) / tot
            v["uncredited by Pearl-C (H200)"] = [r["id"] for r in rs if r["schemes"]["pearl-c-h100-v1-h3"]["credited"]
                                                 and credited(r, "pearl-c-h100-v1-h3", gone) == 0]
            tab[step] = v
        out[label] = tab
    out["reproduces_the_map"] = all(
        abs(out["the map as it stands"][step][k] - map_doc["summary"][step][k]) < 1e-12 for step in steps
        for k in ("pearl-c-h100-v1-h3", "h1-sm_90", "pearl-c-sm120-v1-h3", "h1-sm_120", "mix H200", "mix RTX PRO 6000"))
    return out


def registration() -> dict:
    s = SCHEMES["pearl-c-h100-v1-h3"]
    out = {}
    for t, cur in (*((t, False) for t in REG.REGISTRATION_STACKS), ((REG.LATENT_ROTATION,), True), ((), False), ((), True)):
        try:
            s.check_registration(t, cur)
            out[f"{list(t)} curated={cur}"] = "accepted"
        except DomainError as e:
            out[f"{list(t)} curated={cur}"] = f"refused: {e}"
    return out


def registered(name: str, stack: tuple[str, ...]) -> bool:
    """Whether `registration.check_linear` registers the linear under `stack`."""
    try:
        REG.check_linear(stack, SIDES[name])
        return True
    except DomainError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", default=SNAP)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--layers", default="0,6")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--map", default=None, help="glm53_credit_map's credit_map.json")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    def log(m):
        print(f"[{time.time() - t_start:7.1f}s] {m}", flush=True)

    ck = Synthetic() if a.synthetic else Checkpoint(a.snapshot)
    D = Dims(ck.cfg)
    layers = [int(x) for x in a.layers.split(",")] if not a.synthetic else [0, 1]
    if a.synthetic:
        ids = torch.tensor([(7 * i + 3) % D.vocab for i in range(a.tokens)])
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(Path(a.snapshot) / "tokenizer.json"))
        ids = torch.tensor(tok.encode((ROOT / "README.md").read_text()).ids[: a.tokens])
    T = len(ids)
    pos = torch.arange(T)
    cal = calibrate(ck, D, ids, max(layers), log, 2 if a.synthetic else None)
    # the key: rotation_key over an epoch started from a challenger's salt coin (the weights' root a stand-in)
    roots = {0: ck.root()}
    ch = EC.challenger(b"glm53-rotation")
    opening = ch.open("salt", weights_root(roots))
    epoch = Epoch.start(Workload((Matmul(0, Shape(64, 1024, 128), 0),)), roots, ch.commitment, opening)
    key = REG.rotation_key(epoch, opening)
    Q = Rotation(D.d, cal["S"], cal["profile"], key, "residual")
    log(f"S = {cal['S']} (scores {cal['S_scores']}); q-latent S = {cal['q']['S']}, kv-latent S = {cal['kv']['S']}")
    rows = Rows()
    doc: dict = {"tool": "glm53_rotation", "synthetic": a.synthetic, "snapshot": None if a.synthetic else a.snapshot,
                 "tokens": T, "layers": layers, "tp": TP, "blocks": BLOCKS, "S": cal["S"], "S_scores": cal["S_scores"],
                 "latent_S": {"q": cal["q"]["S"], "kv": cal["kv"]["S"]}, "registration": registration()}
    doc["checked"] = [check_layer(ck, D, L, cal["inputs"][L], pos, Q, key, cal, rows, log) for L in layers]
    last = cal["inputs"][max(layers) + 1]
    doc["mtp"] = check_mtp(ck, D, D.L, last, ids, pos, Q, key, cal, rows, log)
    doc["lm_head"] = check_lm_head(ck, D, last, Q, rows, log)
    doc["rows"] = rows.out
    doc["rows_exact_mismatch"] = rows.mismatch
    stacks = {"stack": STACK, "r8_only": R8_ONLY}
    doc["stacks"] = {k_: list(t) for k_, t in stacks.items()}
    doc["reach"], doc["registered"], doc["rule_agrees_with_reach"], doc["admitted_unreached"] = {}, {}, {}, {}
    for k_, t in stacks.items():
        got: dict[str, bool] = {}
        for c in doc["checked"]:
            for n, v in c["reach"][k_].items():
                got[n] = got.get(n, False) or v
        if k_ == "stack":
            for n, v in doc["mtp"]["reach"].items():
                got[n] = got.get(n, False) or v
        reg = {n: registered(n, t) for n in SIDES}
        doc["reach"][k_], doc["registered"][k_] = got, reg
        doc["rule_agrees_with_reach"][k_] = {n: got[n] == reg[n] for n in got}
        doc["admitted_unreached"][k_] = sorted(n for n in ADMITTED if not reg[n])
    unreached = set(doc["admitted_unreached"]["stack"])
    if a.map:
        lm = doc["lm_head"]
        doc["shares"] = shares(json.loads(Path(a.map).read_text()),
                               {"+".join(t): set(doc["admitted_unreached"][k_]) for k_, t in stacks.items()},
                               {s: lm[s]["credited_fraction"] if lm[s]["domain"][lm["n_padded"]] == "admitted" else 0.0
                                for s in SCHEMES})

    worst = max([v["rel"] for c in doc["checked"] for v in c["function"].values() if "rel" in v]
                + [v for c in doc["checked"] for vs in c["per_rank"].values() for v in vs]
                + [v for vs in doc["mtp"]["per_rank"].values() for v in vs]
                + [v["rel"] for v in doc["mtp"]["layer"].values() if "rel" in v]
                + [doc["mtp"]["eh_proj"]["rel"], doc["mtp"]["logits"]["rel"], doc["lm_head"]["function"]["rel"]])
    sets = all(v["equal"] for c in doc["checked"] for v in c["function"].values() if "equal" in v) and \
        all(v["equal"] for v in doc["mtp"]["layer"].values() if "equal" in v) and \
        all(v["equal"] for c in doc["checked"] for v in c["r8_only"].values() if "equal" in v)
    r8_worst = max(v["rel"] for c in doc["checked"] for v in c["r8_only"].values() if "rel" in v)
    vo_worst = max(v["rel"] for c in doc["checked"] for v in c["vo"].values())
    rot_fail = {k_: v["fail"] for k_, v in rows.out.items() if k_.endswith("/rotated") and v["fail"]}
    doc["verdict"] = {"function_worst_rel": worst, "topk_sets_equal": sets, "r8_only_worst_rel": r8_worst,
                      "vo_worst_rel": vo_worst, "rotated_rows_failing": rot_fail,
                      "exact_rule_agrees": not rows.mismatch,
                      "check_linear_agrees_with_reach": all(v for t in doc["rule_agrees_with_reach"].values() for v in t.values()),
                      "registration_accepts_stacks": all(doc["registration"][f"{list(t)} curated=False"] == "accepted"
                                                         for t in stacks.values()),
                      "admitted_unreached_under_stack": sorted(unreached),
                      "lm_head_padding_sound": all(doc["lm_head"][s]["credit_over_real_columns_honest"] <= 1
                                                   and doc["lm_head"][s]["undrawn_tiles_are_the_padded_one"]
                                                   and doc["lm_head"][s]["within_vex_cap"] for s in SCHEMES)}
    v = doc["verdict"]
    ok = (worst < EXACT and sets and r8_worst < EXACT and vo_worst < EXACT and v["exact_rule_agrees"]
          and v["check_linear_agrees_with_reach"] and v["registration_accepts_stacks"] and v["lm_head_padding_sound"])
    doc["seconds"] = round(time.time() - t_start, 1)
    (out / "rotation.json").write_text(json.dumps(doc, indent=1, default=float) + "\n")
    log(json.dumps(doc["verdict"], default=float))
    log(f"admitted but unreached: {doc['admitted_unreached']}")
    if "shares" in doc:
        log(f"reproduces the map: {doc['shares']['reproduces_the_map']}")
        for k_, tab in doc["shares"].items():
            if isinstance(tab, dict):
                log(f"{k_}: " + json.dumps({s: {m: (round(x, 4) if isinstance(x, float) else x) for m, x in t.items()} for s, t in tab.items()}))
    meas = [("function_worst_rel", worst, "1"), ("r8_only_worst_rel", r8_worst, "1"), ("vo_worst_rel", vo_worst, "1")]
    for c in doc["checked"]:
        meas += [(f"cast_update_rel_L{c['layer']}_unrotated", c["cast"]["unrotated"], "1"),
                 (f"cast_update_rel_L{c['layer']}_rotated", c["cast"]["rotated"], "1")]
    run_result.write(out / "rotation.json", ok, "glm53_rotation",
                     f"{'+'.join(STACK)} exact to {worst:.1e}; admitted but unreached under it: {sorted(unreached)}", meas)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
