"""The request wrappers of a request whose first C prompt rows vLLM's prefix cache serves: the cached rows are the wrapper's inputs
(`prefix_input`), handed to the op that opens every history with them (`vllm_iface.kv_prefix`, `vllm_iface.mla_prefix`), and the prefill
is the uncached suffix prompt[C:] at positions C..LP-1."""

from __future__ import annotations

import torch
from torch import nn

from verity_vllm.program.frontend import vllm_meta as vm


class CachedGreedyRequest(vm.GreedyRequest):
    """`GreedyRequest` for a request whose first `C` prompt rows vLLM's prefix cache serves: `forward(prompt, prefix)` hands `prefix`
    (bf16 [L, 2, C, KVH*D]: every attention layer's cached key and value rows, layers in forward order) to `vllm_iface.kv_prefix`, so
    each layer's history opens with them; prefills the uncached suffix prompt[C:] at positions C..LP-1 (the engine's launch rows), then
    decodes as `GreedyRequest`.  The whole prompt stays the token input (Match wires prompt rows by position); no gate reads rows 0..C-1."""

    DERIVE_SOURCE = "verity-vllm/cached-greedy-request/v1"
    n_prefix = 1

    def __init__(self, model: nn.Module, LP: int, T: int, C: int) -> None:
        super().__init__(model, LP, T)
        self.C = C

    def forward(self, prompt: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        open_prefix((prefix,))
        return self._decode(self._step(prompt[self.C:], self.C, 0))


class CachedMlaGreedyRequest(CachedGreedyRequest):
    """`CachedGreedyRequest` for a DeepSeek sparse attention model: `forward(prompt, latent, index_k, index_k_scale)` hands every MLA
    layer's cached latent rows and every indexer layer's cached index-K rows (`prefix_input`) to `vllm_iface.mla_prefix`."""

    DERIVE_SOURCE = "verity-vllm/cached-mla-greedy-request/v1"
    n_prefix = 3

    def forward(self, prompt: torch.Tensor, latent: torch.Tensor, index_k: torch.Tensor, index_k_scale: torch.Tensor) -> torch.Tensor:
        open_prefix((latent, index_k, index_k_scale))
        return self._decode(self._step(prompt[self.C:], self.C, 0))


class CachedStochasticRequest(vm.StochasticRequest):
    """`StochasticRequest` with a cached prefix of `C` rows (see `CachedGreedyRequest`): `forward(prompt, seed, [splits,] *prefix)`, the
    `n_prefix` prefix inputs (`prefix_input`) after the request's other inputs."""

    DERIVE_SOURCE = "verity-vllm/cached-stochastic-request/v1"

    def __init__(self, model: nn.Module, LP: int, T: int, temperature: float, top_p: float, C: int, S: int | None = None,
                 n_prefix: int = 1) -> None:
        super().__init__(model, LP, T, temperature, top_p, S=S)
        self.C, self.n_prefix = C, n_prefix

    def forward(self, prompt: torch.Tensor, seed: torch.Tensor, *rest: torch.Tensor) -> torch.Tensor:
        open_prefix(rest[len(rest) - self.n_prefix:])
        return self._request(prompt, self.C, seed, rest[0] if len(rest) == self.n_prefix + 1 else None)


def open_prefix(prefix: tuple) -> None:
    """Hand a cached request's prefix inputs to the op that opens the histories with them: `kv_prefix` (one dense KV input) or
    `mla_prefix` (MLA's latent, index K and its scales)."""
    from verity_vllm.program.frontend.rules.vllm_iface import iface
    if len(prefix) == 3:
        iface.mla_prefix(*prefix)
    else:
        iface.kv_prefix(*prefix)


def prefix_input(model: nn.Module, C: int) -> tuple[torch.Tensor, ...]:
    """The example prefix inputs of a cached request wrapper.  Dense attention: bf16 [L, 2, C, KVH*D] over the model's L attention layers
    (one KV geometry for every layer).  DeepSeek sparse attention (every layer MLA): the latent rows bf16 [NL, C, L+R] of its NL layers,
    and the index-K rows e4m3 [NI, C, D] and their f32 scales [NI, C] of its NI indexer layers."""
    impls = vm.attention_impls(model)
    kinds = {v.get("kind", "dense") for v in impls.values()}
    if kinds == {"mla"}:
        lat = {v["kv_lora_rank"] + v["qk_rope_head_dim"] for v in impls.values()}
        ix = [v for v in impls.values() if v.get("indexer")]
        dk = {v["index_head_dim"] for v in ix}
        if len(lat) != 1 or len(dk) != 1:
            raise ValueError(f"an MLA cached prefix holds one latent and one index-K width; the layers have L+R in {sorted(lat)}, D in {sorted(dk)}")
        (w,), (d,) = lat, dk
        return (torch.empty((len(impls), C, w), dtype=torch.bfloat16, device="meta"),
                torch.empty((len(ix), C, d), dtype=torch.float8_e4m3fn, device="meta"), torch.empty((len(ix), C), dtype=torch.float32, device="meta"))
    if kinds != {"dense"}:
        raise ValueError(f"a cached prefix is one kind of attention cache's rows; this model's attention layers are {sorted(kinds)}")
    geo = {(v["num_kv_heads"], v["head_size"]) for v in impls.values()}
    if len(geo) != 1:
        raise ValueError(f"a cached prefix holds one KV geometry for every layer; the layers have (KVH, D) in {sorted(geo)}")
    (kvh, d), = geo
    return (torch.empty((len(impls), 2, C, kvh * d), dtype=torch.bfloat16, device="meta"),)


def cached_wrapper(mod: nn.Module, inputs: tuple, model: nn.Module, C: int) -> tuple[nn.Module, tuple]:
    """The request wrapper `mod` (with its example `inputs`) for a request whose first `C` prompt rows the prefix cache serves
    (`CachedGreedyRequest`, `CachedMlaGreedyRequest`, `CachedStochasticRequest`), the prefix inputs (`prefix_input`) its last; `mod`
    itself when C = 0."""
    if not C:
        return mod, inputs
    if type(mod) not in (vm.GreedyRequest, vm.StochasticRequest) or not 0 < C < mod.LP:
        raise ValueError(f"a cached prefix of C={C} rows needs a request wrapper and 0 < C < LP, got {type(mod).__name__}")
    prefix = prefix_input(model, C)
    if isinstance(mod, vm.StochasticRequest):
        mod = CachedStochasticRequest(model, mod.LP, mod.T, mod.temperature, mod.top_p, C, S=mod.S, n_prefix=len(prefix))
    else:
        mod = (CachedMlaGreedyRequest if len(prefix) == 3 else CachedGreedyRequest)(model, mod.LP, mod.T, C)
    return mod, (*inputs, *prefix)


__all__ = ["CachedGreedyRequest", "CachedMlaGreedyRequest", "CachedStochasticRequest", "cached_wrapper", "open_prefix", "prefix_input"]
