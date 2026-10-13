"""A CPU stand-in for DeepSeek-V4-Pro served under the capture map `deepseek-v4-eager-hopper-tp8-v0`: every module of the map at its
layers (the map's model class at the root, linears that hold the tensor-parallel size), whose decoder layers, attention layers and
indexers run the pinned vLLM's call structure on small tensors.

`DecoderLayer.forward` is `DeepseekV4DecoderLayer.forward`: layer 0's attention side calls this module's
`mhc_pre_broadcast_tilelang` with the embedding, every other one `mhc_fused_post_pre_tilelang` with `hc_attn_fn`, and every ffn
side `mhc_fused_post_pre_tilelang` with `hc_ffn_fn`, each by the call-time global as vLLM's module does; `Model.forward` runs the
layers, then `mhc_post_tilelang` and `hc_head_fused_kernel_tilelang`.  The mHC functions take vLLM's parameters and return its
shapes, and say whether they took the FMA path (`num_tokens <= 16`, kernels/mhc/tilelang.py:518) in `FMA_CALLS`; a decoder layer
keeps each side's four returns in `seen`.
`Attention.forward` is `DeepseekV4FlashMLAAttention.forward` on small widths (the map's head counts and cache row widths): the
input projections, `fused_q_kv_rmsnorm` by its call-time global, `_prepare_and_attn` (the query insert, which writes the SWA cache
rows, then on a CSA layer the indexer, then the compressor, sequentially as `execute_in_parallel` does without aux streams),
`_sparse_indexer_and_attn` (the indexer op on the scored route, then `forward_mqa`, which calls `_forward_prefill` on the prefill
rows and `_forward_decode`, and so `flash_mla_with_kvcache`, on the decode rows) and `_o_proj` (`deep_gemm_fp8_o_proj`'s
`fused_inv_rope_fp8_quant` and `fp8_einsum`; wo_b is a tanh).  The indexer returns (None, None, None) on its short route, when the
step's longest sequence has at most TOPK * CSA_R tokens, after its compressor and the top-k fill.  A compressor writes its state
rows and, for each token whose position + 1 is a multiple of its ratio and whose slots are set, a compressed row, as the pinned
kernels do (and the step's compressed slots are set at exactly those tokens, as vLLM's are); the caches are paged as vLLM's are (a padded block stride; a byte cache's block holds every slot's data bytes, then
every slot's scale bytes), and the step's per-cache metadata is the forward context's (`CONTEXT`, which `run_step` sets).
Each module keeps what it computed (`seen`) for the tests to compare with what a source bound."""
from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import torch

from verity_vllm.commit import capture_map as CM
from verity_vllm.commit.committer.native_host import ACQUIRE_CLASSES

ROOT = Path(__file__).parents[2]
MAP = ROOT / "manifests" / "capture-maps" / "deepseek-v4-eager-hopper-tp8-v0.json"
CONFIG_PATH = ROOT / "tests" / "check" / "fixtures" / "deepseek_v4" / "config.json"
CMAP = CM.load(MAP)
CONFIG = json.loads(CONFIG_PATH.read_text())
TP = CMAP["target"]["tensor_parallel_size"]
ENV = CM.symbols(CMAP, CONFIG, TP)
L, HC, TOPK, CSA_R = ENV["L"], ENV["HC"], ENV["TOPK"], ENV["CSA_R"]
CSA = set(CM.layer_set(CMAP, "csa", ENV))
#: the stand-in's hidden width (the sources read no width of an mHC value)
H = 8
#: the map's module classes whose class hierarchy holds one of the committer's ACQUIRE_CLASSES on the pinned vLLM
HOOKED = {e["type"] for e in CMAP["modules"] if e["type"] in ACQUIRE_CLASSES} | {"GateLinear"}
#: (function, M, whether it took the FMA path) of every mHC call, in order
FMA_CALLS: list[tuple[str, int, bool]] = []
HCA_R, SW, NHP, NHL, OGL, HPG = ENV["HCA_R"], ENV["SW"], ENV["NHP"], ENV["NHL"], ENV["OGL"], ENV["HPG"]
SWAB, IKB, HD, ID = ENV["SWAB"], ENV["IKB"], ENV["HD"], ENV["ID"]
#: the stand-in's attention widths: head dim, q lora rank, o lora rank (OGL * OLS is H), indexer heads and head dim
HD_S, QLS, OLS, IH_S, ID_S = 8, 6, H // OGL, 4, 8
#: cache blocks, the SWA and compressed caches' block sizes, a cache block's padding, the HCA decode index width, top-k rows
NB, SWA_BS, KV_BS, PAD, NX_S, TOPK_ROWS = 6, 16, 8, 24, 128, 32
#: the forward context: the step's attention metadata, one entry per cache prefix (None outside `run_step`)
CONTEXT = types.SimpleNamespace(attn_metadata=None)


def _paged(blocks: int, block_size: int, width: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
    """A [blocks, block_size, width] cache whose block stride is padded, as a cache sharing pages with another is."""
    return torch.zeros(blocks, block_size * width + PAD, dtype=dtype)[:, : block_size * width].view(blocks, block_size, width)


def write_split(cache: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor, scale_bytes: int) -> None:
    """Write each row at its slot of a paged byte cache: its data bytes among the block's data, its scale bytes after them."""
    bs, width = int(cache.shape[1]), int(cache.shape[2])
    data = width - scale_bytes
    for s, row in zip(slots.tolist(), rows):
        if s < 0:
            continue
        block, o = cache[s // bs].reshape(-1), s % bs
        block[o * data:(o + 1) * data] = row[:data]
        block[bs * data + o * scale_bytes:bs * data + (o + 1) * scale_bytes] = row[data:]


def _bytes(width: int, positions: torch.Tensor, tag: int) -> torch.Tensor:
    return ((torch.arange(width)[None] * 5 + positions.long()[:, None] * 3 + tag) % 256).to(torch.uint8)


def fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps):
    return (qr.float() * q_weight).to(torch.bfloat16), (kv.float() * kv_weight).to(torch.bfloat16)


def fused_indexer_q_rope_quant(positions, index_q, index_q_cos_sin_cache, index_weights, index_weights_softmax_scale,
                               index_weights_head_scale, use_fp4=False):
    q = (torch.tanh(index_q.float() + positions[:, None, None]) * 100).to(torch.float8_e4m3fn)
    return q, index_weights.float() * index_weights_softmax_scale * index_weights_head_scale


def flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens, head_dim_v, tile_scheduler_metadata, num_splits=None,
                           softmax_scale=None, causal=False, is_fp8_kvcache=False, indices=None, attn_sink=None, extra_k_cache=None,
                           extra_indices_in_kvcache=None, topk_length=None, extra_topk_length=None, out=None):
    out.copy_(q * 2)
    return out, None


def fused_inv_rope_fp8_quant(o, positions, cos_sin_cache, n_groups, heads_per_group, nope_dim=448, rope_dim=64, quant_group_size=128,
                             tma_aligned_scales=False):
    """FlashMLA's [T, G, D] e4m3 output and its scales, each a transposed view of a [G, T, .] buffer."""
    g = o.float().reshape(o.shape[0], n_groups, heads_per_group * o.shape[-1])
    scale = g.abs().amax(-1, keepdim=True).clamp(min=1e-4) / 448
    q = (g / scale).to(torch.float8_e4m3fn)
    return q.transpose(0, 1).contiguous().transpose(0, 1), scale.transpose(0, 1).contiguous().transpose(0, 1)


def fp8_einsum(*args, **kwargs):
    eq, (a, sa), (b, sb), out = args[:4]
    out.copy_(torch.einsum(eq, a.float() * sa, b.float() * sb).to(out.dtype))


def deep_gemm_fp8_o_proj(o, positions, cos_sin_cache, wo_a, wo_b, *, n_groups, heads_per_group, nope_dim, rope_dim, o_lora_rank,
                         einsum_recipe, tma_aligned_scales):
    o_fp8, o_scale = fused_inv_rope_fp8_quant(o, positions, cos_sin_cache, n_groups=n_groups, heads_per_group=heads_per_group,
                                              nope_dim=nope_dim, rope_dim=rope_dim, tma_aligned_scales=tma_aligned_scales)
    z = torch.empty((o.shape[0], n_groups, o_lora_rank), dtype=torch.bfloat16)
    fp8_einsum("bhr,hdr->bhd", (o_fp8, o_scale), (wo_a.weight, wo_a.weight_scale_inv), z, recipe=(1, 1, 1))
    wo_a.seen = {"o_fp8": o_fp8, "o_scale": o_scale, "z": z}
    return torch.tanh(z.flatten(1).float()).to(torch.bfloat16)


def _out(*shape: int, seed: float) -> torch.Tensor:
    return torch.full(shape, float(seed), dtype=torch.float32)


def mhc_pre_broadcast_tilelang(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                               sinkhorn_repeat, n_splits=1, norm_weight=None, norm_eps=1e-6, fn_broadcast=None):
    m = residual.shape[0]
    FMA_CALLS.append(("mhc_pre_broadcast_tilelang", m, False))
    s = float(fn.sum())
    return (residual.unsqueeze(1).expand(m, HC, H).to(torch.bfloat16) + s, _out(m, HC, 1, seed=s + 1), _out(m, HC, HC, seed=s + 2),
            (residual + s + 3).to(torch.bfloat16))


def mhc_fused_post_pre_tilelang(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                                hc_post_mult_value, sinkhorn_repeat, n_splits=1, tile_n=1, norm_weight=None, norm_eps=1e-6):
    m = residual.view(-1, HC, H).shape[0]
    FMA_CALLS.append(("mhc_fused_post_pre_tilelang", m, m <= 16))
    s = float(fn.sum()) + float(x.float().sum()) * 1e-3
    return (residual + x.unsqueeze(1) + s, _out(m, HC, 1, seed=s + 1), _out(m, HC, HC, seed=s + 2), (x + s + 3).to(torch.bfloat16))


def mhc_post_tilelang(x, residual, post_layer_mix, comb_res_mix):
    FMA_CALLS.append(("mhc_post_tilelang", residual.shape[0], False))
    return residual + x.unsqueeze(1) * post_layer_mix.to(x.dtype)


def hc_head_fused_kernel_tilelang(hs_flat, fn, hc_scale, hc_base, rms_eps, hc_eps):
    FMA_CALLS.append(("hc_head_fused_kernel_tilelang", hs_flat.shape[0], False))
    return hs_flat.float().mean(1).to(torch.bfloat16) + float(fn.sum())


class Compressor(torch.nn.Module):
    """`DeepseekCompressor.forward`: `save_partial_states`, then (unless an HCA step crosses no 128 boundary) a compressed row for
    each token the compress kernels' mask selects, into the cache its `k_cache_prefix` names."""

    def setup(self, prefix: str, ratio: int, width: int, k_cache_prefix: str, scale_bytes: int, static: dict[str, Any], tag: int) -> None:
        self.prefix, self.compress_ratio, self.k_cache_prefix, self.scale_bytes, self.tag = prefix, ratio, k_cache_prefix, scale_bytes, tag
        self._static_forward_context = static
        self.state_cache.prefix = f"{prefix}.state_cache"
        self.state_cache.kv_cache = _paged(NB, 4 if ratio == CSA_R else 8, width, torch.float32)
        static[self.state_cache.prefix] = self.state_cache
        self.ape = (torch.arange(ratio, dtype=torch.float32) * 0.25)[:, None]
        self.seen: dict[str, Any] = {}

    def forward(self, kv_score, positions, rotary_emb):
        md, self.seen = CONTEXT.attn_metadata, {}
        if not isinstance(md, dict) or self.state_cache.prefix not in md:
            return None
        smd = md[self.state_cache.prefix]
        w = kv_score.shape[-1] // 2
        kv, score = kv_score.split([w, w], dim=-1)
        rows = torch.cat([kv, score + self.ape[positions.long() % self.compress_ratio]], dim=-1)
        cache, bs, slots = self.state_cache.kv_cache, smd.block_size, smd.slot_mapping
        for t, s in enumerate(slots.tolist()):
            if s >= 0:
                cache[s // bs, s % bs] = rows[t]
        self.seen["state"] = rows[slots >= 0]
        if self.compress_ratio == HCA_R and smd.c128_boundary is False:
            return None
        k_cache = self._static_forward_context[self.k_cache_prefix].kv_cache
        k_slots = md[self.k_cache_prefix].slot_mapping
        written = []
        for t in range(min(len(slots), len(positions), len(k_slots))):
            p = int(positions[t])
            if slots[t] < 0 or (p + 1) % self.compress_ratio or k_slots[t] < 0:
                continue
            row = _bytes(int(k_cache.shape[-1]), positions[t:t + 1], self.tag)
            write_split(k_cache, k_slots[t:t + 1], row, self.scale_bytes)
            written.append(row[0])
        self.seen["compressed"] = torch.stack(written) if written else torch.zeros(0, int(k_cache.shape[-1]), dtype=torch.uint8)
        return None


class IndexerOp(torch.nn.Module):
    """`SparseAttnIndexer`: the scored route's top-k, written into the shared `topk_indices_buffer`."""

    def forward(self, hidden_states, q_quant, k, weights):
        m = hidden_states.shape[0]
        buf = self.topk_indices_buffer
        buf[:m] = (torch.arange(TOPK)[None] * 3 + (weights.float().sum(1, keepdim=True) * 7).round().long()) % 4093
        self.seen = {"topk": buf[:m].clone()}


class Indexer(torch.nn.Module):
    """`DeepseekV4Indexer.forward`: the short route (its compressor, then the top-k fill: -1 past (pos + 1) // CSA_R) when the
    step's longest sequence has at most TOPK * CSA_R tokens, else `fused_indexer_q_rope_quant` and its compressor."""

    def setup(self, layer: int, static: dict[str, Any], buf: torch.Tensor) -> None:
        self.prefix = f"model.layers.{layer}.attn.indexer"
        self.k_cache.prefix = f"{self.prefix}.k_cache"
        self.k_cache.kv_cache = _paged(NB, KV_BS, IKB)
        static[self.k_cache.prefix] = self.k_cache
        self.compressor.setup(f"{self.prefix}.compressor", CSA_R, 4 * ID, self.k_cache.prefix, IKB - ID, static, 2 * layer + 1)
        self.topk_indices_buffer = self.indexer_op.topk_indices_buffer = buf

    def forward(self, hidden_states, qr, compressed_kv_score, indexer_weights, positions, rotary_emb, qr_scale=None):
        md = self.md
        m = positions.shape[0]
        if md.max_seq_len // CSA_R <= TOPK:
            self.compressor(compressed_kv_score, positions, rotary_emb)
            fill = torch.arange(TOPK)[None].expand(m, TOPK)
            self.topk_indices_buffer[:m] = torch.where(fill < ((positions.long() + 1) // CSA_R)[:, None], fill, -1)
            self.seen = {"short": True, "topk": self.topk_indices_buffer[:m].clone()}
            return None, None, None
        q = qr.float()[:, :1, None].expand(m, IH_S, ID_S)
        q_quant, weights = fused_indexer_q_rope_quant(positions, q, None, indexer_weights, 0.5, IH_S ** -0.5)
        self.compressor(compressed_kv_score, positions, rotary_emb)
        self.seen = {"short": False, "q": q_quant, "weights": weights}
        return q_quant, None, weights


class Attention(torch.nn.Module):
    """`DeepseekV4FlashMLAAttention`, with FlashMLA's `forward_mqa` split of the step's rows (module docstring)."""

    def setup(self, layer: int, static: dict[str, Any], buf: torch.Tensor) -> None:
        self.layer, self.md, self.seen = layer, None, {}
        self.prefix = f"model.layers.{layer}.attn"
        self.compress_ratio = CSA_R if layer in CSA else HCA_R
        self.n_local_heads, self.padded_heads, self.eps = NHL, NHP, 1e-6
        self.q_norm.weight, self.kv_norm.weight = torch.full((QLS,), 1.5), torch.full((HD_S,), 0.75)
        g = torch.Generator().manual_seed(1000 + layer)
        self.wo_a.weight = (torch.randn(OGL, OLS, HPG * HD_S, generator=g) * 0.1).to(torch.float8_e4m3fn)
        self.wo_a.weight_scale_inv = torch.full((OGL, 1, 1), 0.5)
        self.kv_cache = _paged(NB, KV_BS, SWAB)
        static[self.prefix] = self
        self.swa_cache_layer.prefix = f"{self.prefix}.swa_cache"
        self.swa_cache_layer.kv_cache = _paged(NB, SWA_BS, SWAB)
        static[self.swa_cache_layer.prefix] = self.swa_cache_layer
        self.compressor.setup(f"{self.prefix}.compressor", self.compress_ratio, (4 if layer in CSA else 2) * HD, self.prefix,
                              SWAB - 576, static, 2 * layer)
        self.topk_indices_buffer = buf
        if layer in CSA:
            self.indexer.setup(layer, static, buf)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, llama_4_scaling: Any = None) -> torch.Tensor:
        o_padded = torch.zeros(hidden_states.shape[0], self.padded_heads, HD_S, dtype=hidden_states.dtype)
        self.seen = {"rows": []}
        qr_kv, kv_score, indexer_kv_score, indexer_weights = self._run_parallel_input_projections(hidden_states)
        qr, qr_scale, kv = self._split_qkv_and_norm(qr_kv)
        self._prepare_and_attn(hidden_states, qr, kv, qr_scale, kv_score, indexer_kv_score, indexer_weights, positions, o_padded)
        self.seen["out"] = o_padded
        return self._o_proj(o_padded[:, : self.n_local_heads, :], positions)

    def _run_parallel_input_projections(self, hidden_states):
        x = hidden_states.float()
        m = x.shape[0]
        qr_kv = torch.cat([x[:, :1].expand(m, QLS) * 1.5, x[:, 1:2].expand(m, HD_S) * 0.5], dim=1).to(torch.bfloat16)
        width = (4 if self.layer in CSA else 2) * HD
        kv_score = x.sum(1, keepdim=True) + torch.arange(width, dtype=torch.float32) * 1e-3
        indexer_kv_score = indexer_weights = None
        if self.layer in CSA:
            indexer_kv_score = x[:, :1] - torch.arange(4 * ID, dtype=torch.float32) * 1e-3
            indexer_weights = x[:, :1].expand(m, IH_S) * 0.25
        self.seen.update(kv_score=kv_score, indexer_kv_score=indexer_kv_score)
        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    def _split_qkv_and_norm(self, qr_kv):
        qr, kv = qr_kv.split([QLS, HD_S], dim=-1)
        qr, kv = fused_q_kv_rmsnorm(qr, kv, self.q_norm.weight.data, self.kv_norm.weight.data, self.eps)
        self.seen.update(qr=qr, kv=kv)
        return qr, None, kv

    def _prepare_and_attn(self, hidden_states, qr, kv, qr_scale, kv_score, indexer_kv_score, indexer_weights, positions, o_padded):
        attn_metadata = CONTEXT.attn_metadata
        m = qr.shape[0]
        q = (qr.float()[:, :1] + torch.arange(NHL * HD_S) * 1e-2).view(m, NHL, HD_S).to(torch.bfloat16)
        q = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)
        index_q = index_q_scale = index_weights = None
        if self.layer in CSA:
            index_q, index_q_scale, index_weights = self.indexer(hidden_states, qr, indexer_kv_score, indexer_weights, positions, None,
                                                                 qr_scale)
        self.compressor(kv_score, positions, None)
        self._sparse_indexer_and_attn(hidden_states, index_q, index_q_scale, index_weights, q, kv, positions, o_padded)

    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        padded = torch.zeros(q.shape[0], self.padded_heads, HD_S, dtype=q.dtype)
        padded[:, : self.n_local_heads] = q * 1.25
        self.seen["q"] = padded
        if isinstance(attn_metadata, dict) and self.swa_cache_layer.prefix in attn_metadata:
            rows = _bytes(SWAB, positions, 2 * self.layer)
            write_split(self.swa_cache_layer.kv_cache, attn_metadata[self.swa_cache_layer.prefix].slot_mapping, rows, SWAB - 576)
            self.seen["swa_rows"] = rows
        return padded

    def _sparse_indexer_and_attn(self, hidden_states, index_q, index_q_scale, index_weights, q, kv, positions, out):
        self.seen["positions"] = positions
        if self.layer in CSA and index_q is not None:
            self.indexer.indexer_op(hidden_states, index_q, None, index_weights)
        self.forward_mqa(q, kv, positions, out)

    def _o_proj(self, o, positions):
        return deep_gemm_fp8_o_proj(o, positions, None, self.wo_a, None, n_groups=OGL, heads_per_group=self.n_local_heads // OGL,
                                    nope_dim=HD_S // 2, rope_dim=HD_S // 2, o_lora_rank=OLS, einsum_recipe=(1, 1, 1),
                                    tma_aligned_scales=False)

    def forward_mqa(self, q, kv, positions, output) -> None:
        md = self.md
        if md.num_prefills > 0:
            self._forward_prefill(q[md.num_decode_tokens:], positions[md.num_decode_tokens:], None, None,
                                  output[md.num_decode_tokens:], None, md)
        if md.num_decodes > 0:
            self._forward_decode(q[:md.num_decode_tokens], None, md, None, False, output[:md.num_decode_tokens])

    def _forward_decode(self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output) -> None:
        d = swa_metadata.num_decode_tokens
        if self.compress_ratio == CSA_R:
            buf = self.topk_indices_buffer[:d]
            topk_indices, topk_lens = torch.where(buf >= 0, buf * 2 + 1, -1).view(d, 1, -1), (buf >= 0).sum(1).to(torch.int32)
        else:
            topk_indices, topk_lens = swa_metadata.c128a_global_decode_topk_indices, swa_metadata.c128a_decode_topk_lens
        self.seen.update(swa_indices=swa_metadata.decode_swa_indices, swa_lens=swa_metadata.decode_swa_lens, extra_indices=topk_indices,
                         extra_lens=topk_lens)
        flash_mla_with_kvcache(q=q.unsqueeze(1), k_cache=None, block_table=None, head_dim_v=HD_S, tile_scheduler_metadata=None,
                               cache_seqlens=None, is_fp8_kvcache=True, indices=swa_metadata.decode_swa_indices,
                               topk_length=swa_metadata.decode_swa_lens, softmax_scale=1.0, attn_sink=None, extra_k_cache=kv_cache,
                               extra_indices_in_kvcache=topk_indices, extra_topk_length=topk_lens, out=output.unsqueeze(1))
        self.seen["rows"].append(("decode", q.shape[0]))

    def _forward_prefill(self, q, positions, compressed_k_cache, swa_k_cache, output, attn_metadata, swa_metadata) -> None:
        output.copy_(q * 3)
        self.seen["rows"].append(("prefill", q.shape[0]))


class DecoderLayer(torch.nn.Module):
    """`DeepseekV4DecoderLayer.forward`: the attention side's mHC pre (broadcast on layer 0), attention, the ffn side's."""

    def setup(self, layer: int) -> None:
        g = torch.Generator().manual_seed(layer)
        self.layer = layer
        for side in ("attn", "ffn"):
            setattr(self, f"hc_{side}_fn", torch.nn.Parameter(torch.randn(3, generator=g), requires_grad=False))
            setattr(self, f"hc_{side}_scale", torch.nn.Parameter(torch.ones(3), requires_grad=False))
            setattr(self, f"hc_{side}_base", torch.nn.Parameter(torch.zeros(3), requires_grad=False))

    def forward(self, x, positions, input_ids, post_mix=None, res_mix=None, residual=None):
        if residual is None:
            residual, post_mix, res_mix, x = mhc_pre_broadcast_tilelang(
                x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, 1e-6, 1e-6, 1e-6, 2.0, 20, norm_weight=torch.ones(H),
                norm_eps=1e-6, fn_broadcast=self.hc_attn_fn)
        else:
            residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(
                x, residual, post_mix, res_mix, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, 1e-6, 1e-6, 1e-6, 2.0, 20,
                n_splits=1, tile_n=1, norm_weight=torch.ones(H), norm_eps=1e-6)
        self.seen = {"hc_attn": (residual, post_mix, res_mix, x)}
        x = self.attn(positions, x, None)
        residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(
            x, residual, post_mix, res_mix, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, 1e-6, 1e-6, 1e-6, 2.0, 20,
            n_splits=1, tile_n=1, norm_weight=torch.ones(H), norm_eps=1e-6)
        self.seen["hc_ffn"] = (residual, post_mix, res_mix, x)
        return x * 0.5, residual, post_mix, res_mix


class Model(torch.nn.Module):
    """`DeepseekV4Model.forward` with PP 1, no sequence parallelism and no aux hidden states."""

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        hidden = positions.float().unsqueeze(1).expand(-1, H).to(torch.bfloat16)
        residual = post_mix = res_mix = None
        for layer in self.layers:
            hidden, residual, post_mix, res_mix = layer(hidden, positions, input_ids, post_mix, res_mix, residual)
        hidden = mhc_post_tilelang(hidden, residual, post_mix, res_mix)
        self.seen = {"hc_post": hidden}
        hidden = hc_head_fused_kernel_tilelang(hidden, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, 1e-6, 1e-6)
        self.seen["hc_head"] = hidden
        return hidden


BASES = {"model": Model, "model.layers.{i}": DecoderLayer, "model.layers.{i}.attn": Attention, "model.layers.{i}.attn.indexer": Indexer,
         "model.layers.{i}.attn.compressor": Compressor, "model.layers.{i}.attn.indexer.compressor": Compressor,
         "model.layers.{i}.attn.indexer.indexer_op": IndexerOp}


def dsv4_model() -> torch.nn.Module:
    """A module tree with every module of the map at its layers, the root of the map's model class, the stand-in's decoder layers,
    attention layers and indexers at their patterns, linears that hold the tensor-parallel size, and the config the map's symbols
    read."""
    mod, _, qual = CMAP["model_class"].rpartition(".")
    root = type(qual, (torch.nn.Module,), {"__module__": mod})()
    root.config = types.SimpleNamespace(to_dict=lambda: dict(CONFIG))
    classes: dict[str, type] = {}
    for e in sorted(CMAP["modules"], key=lambda e: e["pattern"].count(".")):
        if not e["pattern"]:
            continue
        layers = CM.layer_set(CMAP, e.get("when", "all"), ENV) if "{i}" in e["pattern"] else [None]
        for i in layers[:1] if e.get("shared") else layers:
            name = CM.at_layer(e["pattern"], i)
            parent = root
            for part in name.split(".")[:-1]:
                parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
            base = BASES.get(e["pattern"], torch.nn.Module)
            cls = classes.setdefault(e["type"], type(e["type"], (base,), {}))
            m = torch.nn.ModuleList() if e["type"] == "ModuleList" else cls()
            if e["type"].endswith("Linear"):
                m.tp_size = TP
            if isinstance(parent, torch.nn.ModuleList):
                parent.append(m)
            else:
                parent.add_module(name.rsplit(".", 1)[-1], m)
    root.static_forward_context = static = {}
    buf = torch.full((TOPK_ROWS, TOPK), -7, dtype=torch.int32)
    for i in range(L):
        root.model.layers[i].setup(i)
        root.model.layers[i].attn.setup(i, static, buf)
    for k in ("fn", "scale", "base"):
        setattr(root.model, f"hc_head_{k}", torch.nn.Parameter(torch.ones(3) * 0.5, requires_grad=False))
    return root


def metadata(*, decodes: int = 0, prefill_tokens: int = 0, max_seq_len: int = 64) -> types.SimpleNamespace:
    """A step's attention metadata as the sources read it: `swa_metadata`'s split, its decode indices (two rows past the decodes,
    as a buffer sized for the batch is), the C128A decode indices, and the indexer metadata's longest sequence."""
    pad = decodes + 2
    return types.SimpleNamespace(num_decodes=decodes, num_decode_tokens=decodes, num_prefills=int(prefill_tokens > 0),
                                 num_prefill_tokens=prefill_tokens, num_actual_tokens=decodes + prefill_tokens, max_seq_len=max_seq_len,
                                 decode_swa_indices=(torch.arange(pad * SW, dtype=torch.int32) % 997).view(pad, 1, SW),
                                 decode_swa_lens=torch.arange(pad, dtype=torch.int32) + 5,
                                 c128a_global_decode_topk_indices=(torch.arange(pad * NX_S, dtype=torch.int32) * 3 % 1009).view(pad, 1, NX_S),
                                 c128a_decode_topk_lens=torch.arange(pad, dtype=torch.int32) + 2)


def step_metadata(model: torch.nn.Module, positions: torch.Tensor, skip: tuple[int, ...] = (),
                  unslotted: tuple[int, ...] = ()) -> dict[str, Any]:
    """The forward context's metadata of one step, per cache prefix (none for a layer in `skip`): the SWA slots, each compressor's
    state slots (and on an HCA layer whether a position + 1 is a multiple of HCA_R) and its compressed cache's slots, which are -1
    wherever the token's position + 1 is not a multiple of the compressor's ratio, as the pinned vLLM's
    `get_compressed_slot_mapping` gives them (v1/attention/backends/mla/compressor_utils.py:59-85), and -1 at the tokens
    `unslotted` too.  The pinned vLLM gives every real token a state slot and every boundary token a compressed slot
    (v1/worker/block_table.py:429-475 pads only past the step's tokens or a token another context-parallel rank holds), so
    `unslotted` is a fault a test injects, not a step that vLLM serves."""
    p = positions.long()
    unset = torch.zeros(len(p), dtype=torch.bool)
    unset[list(unslotted)] = True
    out: dict[str, Any] = {}
    for i in range(L):
        if i in skip:
            continue
        a = model.model.layers[i].attn
        out[a.swa_cache_layer.prefix] = types.SimpleNamespace(slot_mapping=(7 * p + 3) % (NB * SWA_BS), block_size=SWA_BS)
        for comp in (a.compressor, a.indexer.compressor if i in CSA else None):
            if comp is None:
                continue
            bs = int(comp.state_cache.kv_cache.shape[1])
            boundary = bool(((p + 1) % HCA_R == 0).any()) if comp.compress_ratio == HCA_R else None
            out[comp.state_cache.prefix] = types.SimpleNamespace(slot_mapping=(5 * p + 1) % (NB * bs), block_size=bs, c128_boundary=boundary)
            off = unset | ((p + 1) % comp.compress_ratio != 0)
            out[comp.k_cache_prefix] = types.SimpleNamespace(slot_mapping=torch.where(off, -1, (7 * p + 5) % (NB * KV_BS)))
    return out


LONG = 4 * TOPK * CSA_R
#: (attention metadata, the routes the step takes, its phase), the steps of `tests/check/test_capture_map_dsv4.py`
STEPS = [(metadata(decodes=4), ("decode-tokens", "hc-fused", "indexer-short"), "decode"),
         (metadata(decodes=4, max_seq_len=LONG), ("decode-tokens", "hc-fused", "indexer-scored"), "decode"),
         (metadata(prefill_tokens=20, max_seq_len=LONG), ("hc-split", "indexer-scored", "prefill-tokens"), "prefill"),
         (metadata(prefill_tokens=10), ("hc-fused", "indexer-short", "prefill-tokens"), "prefill"),
         (metadata(decodes=2, prefill_tokens=18, max_seq_len=LONG), ("decode-tokens", "hc-split", "indexer-scored", "prefill-tokens"),
          "prefill")]


def run_step(model: torch.nn.Module, md: Any, first_position: int = 0, skip: tuple[int, ...] = (), unslotted: tuple[int, ...] = ()) -> None:
    """The model's forward on one step of `md.num_actual_tokens` tokens at positions from `first_position`; the layers in `skip`
    take a step without metadata (their attention runs no FlashMLA path, writes no cache, and their indexer takes no route), and
    the tokens `unslotted` have no compressed slot (a fault: `step_metadata`)."""
    for i in range(L):
        a = model.model.layers[i].attn
        a.md = md if i not in skip else metadata()
        if i in CSA:
            a.indexer.md = md if i not in skip else types.SimpleNamespace(max_seq_len=md.max_seq_len)
    n = md.num_actual_tokens
    positions = torch.arange(first_position, first_position + n)
    CONTEXT.attn_metadata = step_metadata(model, positions, skip, unslotted)
    try:
        model.model(torch.arange(n), positions)
    finally:
        CONTEXT.attn_metadata = None
