"""The int8 attention's K-scale calibrator: max |k| per (layer, KV head) over every token vLLM's attention layers are
handed, which `benchmarks/ppl/int8_kscale_gpu.py` runs over the perplexity bench's calibration windows."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from verity_vllm.engine import int8_k_scales as K


class Calibrator:
    """amax [L, G] of K over every token an attention layer is handed, kept by wrapping its impl's forward."""

    def __init__(self, model: Any):
        self.mods = {n: m for n, m in model.named_modules() if hasattr(m, "impl") and hasattr(m.impl, "num_kv_heads")}
        if not self.mods:
            raise ValueError("no attention layers to calibrate")
        g = {m.impl.num_kv_heads for m in self.mods.values()}
        if len(g) != 1:
            raise ValueError(f"attention layers with different KV head counts {sorted(g)}")
        self.amax = torch.zeros(1 + max(K.layer_index(n) for n in self.mods), g.pop(), dtype=torch.float32)
        self.tokens = 0
        self.saved = {n: m.impl.forward for n, m in self.mods.items()}
        for n, m in self.mods.items():
            m.impl.forward = self._wrap(K.layer_index(n), self.saved[n])

    def _wrap(self, layer: int, forward):
        def wrapped(lay, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
            if attn_metadata is not None:
                n = attn_metadata.num_actual_tokens
                k = key[:n].float().abs().amax(dim=(0, 2)).cpu()
                self.amax[layer] = self.amax[layer].maximum(k)
                if layer == 0:
                    self.tokens += n
            return forward(lay, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
        return wrapped

    def close(self) -> np.ndarray:
        for n, m in self.mods.items():
            m.impl.forward = self.saved[n]
        return self.amax.numpy()
