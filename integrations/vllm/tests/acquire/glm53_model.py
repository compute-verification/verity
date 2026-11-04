"""A CPU stand-in for GLM-5.3 served under the capture map `glm-moe-dsa-eager-hopper-tp8-v0`: every module of the map at its layers
(the map's model class at the root, linears that hold the tensor-parallel size), and attention layers that run the pinned vLLM's call
structure on real tensors of the map's per-rank sizes.

`Attention.forward` is `DeepseekV32Attention.forward` without its projections (the hooked linears' outputs are sampled per layer and
step): it calls this module's `fused_norm_rope` with the step's slot mapping and the layer's caches, which writes the MLA cache entry
[kv_c, k_pe] and, on an indexer layer, the indexer key's e4m3 bytes and f32 scale where `_fp8_quant_and_cache_write` puts them
(per block, the block's key rows and then their scales) and clears the layer's top-k rows; then the W_UK_T product, `fused_q`, and
`_sparse_indexer_and_attn`, which scores the indexer (not on a dense-MHA step without decode tokens), asks `_use_sparse_mha` (with
masked MHA unavailable, as on compute capability 9.x), and attends: every row through `impl.forward_mqa` and the W_UV product into
`output`, or through `forward_impl`, which asks again when the step has prefill tokens, sends its decode rows through `forward_mqa`
and `_v_up_proj` and its prefill rows through `impl.forward_mha`.  Each layer keeps what it computed (`seen`) for the tests to compare
with what a source bound."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import json
import types
from pathlib import Path
from typing import Any

import torch

from verity_vllm.commit import capture_map as CM

ROOT = Path(__file__).parents[2]
MAP = ROOT / "manifests" / "capture-maps" / "glm-moe-dsa-eager-hopper-tp8-v0.json"
CONFIG_PATH = ROOT / "tests" / "check" / "fixtures" / "glm_moe_dsa" / "config.json"
CMAP = CM.load(MAP)
CONFIG = json.loads(CONFIG_PATH.read_text())
TP = CMAP["target"]["tensor_parallel_size"]
ENV = CM.symbols(CMAP, CONFIG, TP)
L = ENV["L"]
INDEXER = set(CM.layer_set(CMAP, "indexer", ENV))
QL, KVL, DR, DN, DV, NHL, IH, ID, TOPK = (ENV[k] for k in ("QL", "KVL", "DR", "DN", "DV", "NHL", "IH", "ID", "TOPK"))
BLOCKS, BLOCK = 8, 4
DENSE, SPARSE = ("dense-mha",), ("sparse-mqa",)
#: the map's module classes whose class hierarchy holds one of the committer's ACQUIRE_CLASSES on the pinned vLLM (GateLinear is a
#: ReplicatedLinear, the fused A projection a MergedColumnParallelLinear); ParallelLMHead is skipped
HOOKED = {"ColumnParallelLinear", "RowParallelLinear", "MergedColumnParallelLinear", "ReplicatedLinear", "GateLinear",
          "DeepSeekV2FusedQkvAProjLinear", "RMSNorm", "LayerNorm", "SiluAndMul", "RotaryEmbedding", "VocabParallelEmbedding",
          "LogitsProcessor"}


def _rms(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype)


def index_key(k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One indexer key row as the cache holds it: its e4m3 bytes and its power-of-two f32 scale's bytes."""
    k = _rms(k).float()
    scale = torch.exp2(torch.ceil(torch.log2((k.abs().amax() / 448).clamp(min=1e-10))))
    return (k / scale).to(torch.float8_e4m3fn).view(torch.uint8), scale.reshape(1).view(torch.uint8)


def fused_norm_rope(positions, q_c, q_w, q_eps, kv_c, kv_w, kv_eps, k_pe, cos_sin, index_k, k_norm_w, k_norm_b, k_norm_eps,
                    index_cos_sin, topk_indices_buffer, *, slot_mapping=None, indexer_k_cache=None, mla_kv_cache=None,
                    mla_kv_cache_dtype="auto", mla_k_scale=None, has_indexer=False, index_rope_interleave=False, kv_c_out=None,
                    k_pe_out=None, index_k_out=None):
    m = q_c.shape[0]
    if has_indexer:
        topk_indices_buffer[:m] = -1
    kv_c_out.copy_(_rms(kv_c))
    k_pe_out.copy_(k_pe.flip(-1) * 0.5)
    for t in range(m):
        s = int(slot_mapping[t])
        if s < 0:
            continue
        blk, off = divmod(s, mla_kv_cache.shape[1])
        if mla_kv_cache is not None:
            mla_kv_cache[blk, off] = torch.cat((kv_c_out[t], k_pe_out[t]))
        if indexer_k_cache is not None:
            key, scale = index_key(index_k[t])
            flat = indexer_k_cache.view(indexer_k_cache.shape[0], -1)
            bs = indexer_k_cache.shape[1]
            flat[blk, off * ID:(off + 1) * ID] = key
            flat[blk, bs * ID + off * 4: bs * ID + off * 4 + 4] = scale
    return _rms(q_c)


def fused_q(positions, q_pe, cos_sin, index_q, index_cos_sin, ql_nope, q_scale, index_weights, softmax_scale, n_head_scale, *,
            has_indexer=False, index_rope_interleave=False, quantize_mqa=False):
    mqa_q = q_pe * 2
    if not has_indexer:
        return None, None, mqa_q
    return index_q.to(torch.float8_e4m3fn), index_weights.float() * softmax_scale * n_head_scale, mqa_q


class Impl:
    """`FlashAttnMLASparseImpl` (forward_mqa) with the prefill backend's `forward_mha`, as deterministic functions of their inputs."""

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        q_nope, q_rope = q
        return (q_nope * 0.25 + q_rope.sum(-1, keepdim=True)), None

    def forward_mha(self, q, kv_c_normed, k_pe, kv_c_and_k_pe_cache, attn_metadata, k_scale, output, output_scale=None):
        v = q[..., :DV].reshape(q.shape[0], NHL * DV) + kv_c_normed[:, :1]
        output.copy_(v)


class Attention(torch.nn.Module):
    """The pinned vLLM's sparse MLA layer: its route decision, its call structure and the values it hands each call."""

    def setup(self, layer: int, topk_buffer: torch.Tensor, w_uk_t: torch.Tensor, w_uv: torch.Tensor) -> None:
        self.layer, self.topk_indices_buffer, self.W_UK_T, self.W_UV = layer, topk_buffer, w_uk_t, w_uv
        self.impl = Impl()
        self.kv_cache = torch.zeros(BLOCKS, BLOCK, KVL + DR, dtype=torch.bfloat16)
        self.has_indexer = layer in INDEXER
        if self.has_indexer:
            self.indexer.k_cache.kv_cache = torch.zeros(BLOCKS, BLOCK, ID + 4, dtype=torch.uint8)
        self.md: Any = None
        self.slots: torch.Tensor | None = None
        self.seen: dict[str, Any] = {}

    def _use_sparse_mha(self, attn_metadata: Any) -> bool:
        prefill = attn_metadata.prefill
        if prefill is None:
            return False
        return bool(prefill.use_dense_mha)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        m = hidden_states.shape[0]
        g = torch.Generator().manual_seed(1000 * self.layer + int(positions[0]))
        draw = lambda *shape: torch.randn(*shape, generator=g).to(torch.bfloat16)  # noqa: E731
        q_c, kv_c, k_pe = draw(m, QL), draw(m, KVL), draw(m, DR)
        index_k = draw(m, ID) if self.has_indexer else None
        index_weights = draw(m, IH) if self.has_indexer else None
        output = torch.empty(m, NHL * DV, dtype=torch.bfloat16)
        kv_c_out, k_pe_out = torch.empty_like(kv_c), torch.empty_like(k_pe)
        q_c = fused_norm_rope(positions, q_c, None, 1e-6, kv_c, None, 1e-6, k_pe, None, index_k, None, None, 1e-6, None,
                              self.topk_indices_buffer, slot_mapping=self.slots,
                              indexer_k_cache=self.indexer.k_cache.kv_cache if self.has_indexer else None, mla_kv_cache=self.kv_cache,
                              has_indexer=self.has_indexer, kv_c_out=kv_c_out, k_pe_out=k_pe_out)
        q = draw(m, NHL, DN + DR)
        q_nope, q_pe = q.split([DN, DR], dim=-1)
        ql_nope = torch.bmm(q_nope.transpose(0, 1), self.W_UK_T).transpose(0, 1)
        index_q = draw(m, IH, ID) if self.has_indexer else None
        index_q_fp8, index_weights_out, mqa_q = fused_q(positions, q_pe, None, index_q, None, ql_nope, None, index_weights, 0.5, 0.25,
                                                        has_indexer=self.has_indexer)
        self.seen = {"q_c": q_c, "kv_c": kv_c_out, "k_pe": k_pe_out, "ql_nope": ql_nope, "mqa_q": mqa_q, "index_q_fp8": index_q_fp8,
                     "index_weights_out": index_weights_out, "index_k": index_k, "output": output, "mqa_rows": 0}
        self._sparse_indexer_and_attn(positions, q_c, q_nope, q_pe, index_q_fp8, None, index_weights_out, kv_c_out, k_pe_out, ql_nope,
                                      mqa_q, output)
        return output

    def _sparse_indexer_and_attn(self, positions, q_c, q_nope, q_pe, index_q_fp8, index_k, index_weights_out, kv_c, k_pe, ql_nope,
                                 mqa_q, output) -> None:
        md = self.md
        if self.has_indexer and not (md.prefill is not None and md.prefill.use_dense_mha and md.num_decode_tokens == 0):
            m = q_c.shape[0]
            self.topk_indices_buffer[:m, :TOPK] = (torch.arange(TOPK, dtype=torch.int32) + self.layer) % (int(positions[-1]) + 1)
            self.seen["topk"] = self.topk_indices_buffer[:m, :TOPK].clone()
        n = md.num_actual_tokens
        if self._use_sparse_mha(md):
            self.forward_impl(torch.cat((q_nope, mqa_q), dim=-1), kv_c, k_pe.unsqueeze(1), self.kv_cache, md, output)
            return
        attn_out, _ = self.impl.forward_mqa((ql_nope[:n], mqa_q[:n]), self.kv_cache, md, self)
        self.seen.update(attn_mqa=attn_out, mqa_rows=n)
        torch.bmm(attn_out.view(n, NHL, KVL).transpose(0, 1), self.W_UV, out=output[:n].view(n, NHL, DV).transpose(0, 1))

    def forward_impl(self, q, k_c_normed, k_pe, kv_cache, attn_metadata, output) -> torch.Tensor:
        n = attn_metadata.num_actual_tokens
        n_mqa = attn_metadata.num_decode_tokens
        n_mha = n - n_mqa
        if n_mha > 0 and not self._use_sparse_mha(attn_metadata):
            n_mqa, n_mha = n, 0
        if n_mha > 0:
            self.impl.forward_mha(q[n_mqa:], k_c_normed[n_mqa:], k_pe[n_mqa:], kv_cache, attn_metadata, None, output=output[n_mqa:n],
                                  output_scale=None)
            self.seen["attn_mha"] = output[n_mqa:n].clone()
        if n_mqa > 0:
            q_nope, q_pe = q[:n_mqa].split([DN, DR], dim=-1)
            ql = torch.bmm(q_nope.transpose(0, 1), self.W_UK_T).transpose(0, 1)
            attn_out, _ = self.impl.forward_mqa((ql, q_pe), kv_cache, attn_metadata, self)
            self.seen.update(attn_mqa=attn_out, mqa_rows=n_mqa)
            torch.bmm(attn_out.transpose(0, 1), self.W_UV, out=output[:n_mqa].view(n_mqa, NHL, DV).transpose(0, 1))
        return output


def glm_model(attention: type = Attention) -> torch.nn.Module:
    """A module tree with every module of the map at its layers, the root of the map's model class, `attention` layers at
    `model.layers.{i}.self_attn`, linears that hold the tensor-parallel size, and the config the map's symbols read."""
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
                if not hasattr(parent, part):
                    parent.add_module(part, torch.nn.Module())
                parent = getattr(parent, part)
            base = attention if name.endswith(".self_attn") else torch.nn.Module
            cls = classes.setdefault(e["type"], type(e["type"], (base,), {}))
            m = cls()
            if e["type"].endswith("Linear"):
                m.tp_size = TP
            parent.add_module(name.rsplit(".", 1)[-1], m)
    g = torch.Generator().manual_seed(7)
    topk = torch.full((16, TOPK), -1, dtype=torch.int32)
    w_uk_t = (torch.randn(NHL, DN, KVL, generator=g) * 0.05).to(torch.bfloat16)
    w_uv = (torch.randn(NHL, KVL, DV, generator=g) * 0.05).to(torch.bfloat16)
    for i in range(L):
        a = root.get_submodule(f"model.layers.{i}.self_attn")
        if isinstance(a, Attention):
            a.setup(i, topk, w_uk_t, w_uv)
    return root


def metadata(*, prefill: bool, dense: bool = False, decodes: int = 0, prefill_tokens: int = 3) -> types.SimpleNamespace:
    pre = types.SimpleNamespace(use_dense_mha=dense, chunked_context=None) if prefill else None
    n_pre = prefill_tokens if prefill else 0
    return types.SimpleNamespace(prefill=pre, num_decode_tokens=decodes, num_prefill_tokens=n_pre, num_actual_tokens=decodes + n_pre)


#: (attention metadata, the routes the step takes, its phase): a short prefill, a decode, a mixed step, a prefill over TOPK keys
STEPS = [(metadata(prefill=True, dense=True), DENSE, "prefill"),
         (metadata(prefill=False, decodes=1), SPARSE, "decode"),
         (metadata(prefill=True, dense=True, decodes=1), ("dense-mha", "sparse-mqa"), "prefill"),
         (metadata(prefill=True, dense=False), SPARSE, "prefill")]


def run_step(model: torch.nn.Module, md: Any, first_slot: int, skip: tuple[int, ...] = ()) -> int:
    """Every attention layer's forward on one step of `md.num_actual_tokens` tokens at slots from `first_slot`; the next free slot."""
    n = md.num_actual_tokens
    positions = torch.arange(first_slot, first_slot + n)
    hidden = torch.zeros(n, 1)
    for i in range(L):
        if i in skip:
            continue
        a = model.get_submodule(f"model.layers.{i}.self_attn")
        a.md, a.slots = md, torch.arange(first_slot, first_slot + n)
        a(positions, hidden)
    return first_slot + n
