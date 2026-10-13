"""A CPU stand-in for DeepSeek-V4-Pro served under the capture map `deepseek-v4-eager-hopper-tp8-v0`: every module of the map at its
layers (the map's model class at the root, linears that hold the tensor-parallel size), whose decoder layers, attention layers and
indexers run the pinned vLLM's call structure on small tensors.

`DecoderLayer.forward` is `DeepseekV4DecoderLayer.forward`: layer 0's attention side calls this module's
`mhc_pre_broadcast_tilelang` with the embedding, every other one `mhc_fused_post_pre_tilelang` with `hc_attn_fn`, and every ffn
side `mhc_fused_post_pre_tilelang` with `hc_ffn_fn`, each by the call-time global as vLLM's module does; `Model.forward` runs the
layers, then `mhc_post_tilelang` and `hc_head_fused_kernel_tilelang`.  The mHC functions take vLLM's parameters and return its
shapes, and say whether they took the FMA path (`num_tokens <= 16`, kernels/mhc/tilelang.py:518) in `FMA_CALLS`; a decoder layer
keeps each side's four returns in `seen`.
`Attention.forward` runs the indexer on a CSA layer (its forward returns (None, None, None) when the step's longest sequence has at
most TOPK * CSA_R tokens) and `forward_mqa`, which calls `_forward_prefill` on the prefill rows and `_forward_decode` on the decode
rows, as FlashMLA's does.  Each module keeps what it computed (`seen`) for the tests to compare with what a source bound."""
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


class Indexer(torch.nn.Module):
    """`DeepseekV4Indexer.forward`: the short path when the step's longest sequence has at most TOPK * CSA_R tokens."""

    def forward(self, hidden_states, qr, compressed_kv_score, indexer_weights, positions, rotary_emb, qr_scale=None):
        md = self.md
        if md.max_seq_len // CSA_R <= TOPK:
            self.seen = {"short": True}
            return None, None, None
        q = hidden_states[:, :1].expand(-1, 4).to(torch.float8_e4m3fn)
        self.seen = {"short": False, "q": q}
        return q, None, indexer_weights


class Attention(torch.nn.Module):
    """`DeepseekV4FlashMLAAttention`: the indexer on a CSA layer, then `forward_mqa`'s split of the step's rows."""

    def setup(self, layer: int) -> None:
        self.layer, self.md, self.seen = layer, None, {}

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, llama_4_scaling: Any = None) -> torch.Tensor:
        m = hidden_states.shape[0]
        if self.layer in CSA:
            self.indexer(hidden_states, None, None, torch.zeros(m, 4), positions, None)
        out = torch.zeros(m, 2, H, dtype=torch.bfloat16)
        self.seen = {"rows": []}
        self.forward_mqa(hidden_states.unsqueeze(1).expand(m, 2, H), None, positions, out)
        return out[:, 0] + hidden_states

    def forward_mqa(self, q, kv, positions, output) -> None:
        md = self.md
        if md.num_prefills > 0:
            self._forward_prefill(q[md.num_decode_tokens:], positions[md.num_decode_tokens:], None, None,
                                  output[md.num_decode_tokens:], None, md)
        if md.num_decodes > 0:
            self._forward_decode(q[:md.num_decode_tokens], None, md, None, self.layer % 2 == 0, output[:md.num_decode_tokens])

    def _forward_decode(self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output) -> None:
        output.copy_(q * 2)
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


BASES = {"model": Model, "model.layers.{i}": DecoderLayer, "model.layers.{i}.attn": Attention, "model.layers.{i}.attn.indexer": Indexer}


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
    for i in range(L):
        root.model.layers[i].setup(i)
        root.model.layers[i].attn.setup(i)
    for k in ("fn", "scale", "base"):
        setattr(root.model, f"hc_head_{k}", torch.nn.Parameter(torch.ones(3) * 0.5, requires_grad=False))
    return root


def metadata(*, decodes: int = 0, prefill_tokens: int = 0, max_seq_len: int = 64) -> types.SimpleNamespace:
    """A step's attention metadata as the sources read it: `swa_metadata`'s split and the indexer metadata's longest sequence."""
    return types.SimpleNamespace(num_decodes=decodes, num_decode_tokens=decodes, num_prefills=int(prefill_tokens > 0),
                                 num_prefill_tokens=prefill_tokens, num_actual_tokens=decodes + prefill_tokens, max_seq_len=max_seq_len)


LONG = 4 * TOPK * CSA_R
#: (attention metadata, the routes the step takes, its phase), the steps of `tests/check/test_capture_map_dsv4.py`
STEPS = [(metadata(decodes=4), ("decode-tokens", "hc-fused", "indexer-short"), "decode"),
         (metadata(decodes=4, max_seq_len=LONG), ("decode-tokens", "hc-fused", "indexer-scored"), "decode"),
         (metadata(prefill_tokens=20, max_seq_len=LONG), ("hc-split", "indexer-scored", "prefill-tokens"), "prefill"),
         (metadata(prefill_tokens=10), ("hc-fused", "indexer-short", "prefill-tokens"), "prefill"),
         (metadata(decodes=2, prefill_tokens=18, max_seq_len=LONG), ("decode-tokens", "hc-split", "indexer-scored", "prefill-tokens"),
          "prefill")]


def run_step(model: torch.nn.Module, md: Any, first_position: int = 0, skip: tuple[int, ...] = ()) -> None:
    """The model's forward on one step of `md.num_actual_tokens` tokens; the layers in `skip` take a step without metadata (their
    attention runs no FlashMLA path and their indexer no route)."""
    for i in range(L):
        a = model.model.layers[i].attn
        a.md = md if i not in skip else metadata()
        if i in CSA:
            a.indexer.md = md if i not in skip else types.SimpleNamespace(max_seq_len=md.max_seq_len)
    n = md.num_actual_tokens
    model.model(torch.arange(n), torch.arange(first_position, first_position + n))
