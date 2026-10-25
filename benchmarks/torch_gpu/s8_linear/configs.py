"""int8_fused's GEMM config per large-model linear on sm_120: (N, K) -> [(m_max, config)], the first m_max >= M wins and an M
past the last takes the last (`int8_fused.Cfg` text: BLOCK_MxBLOCK_NxBLOCK_K:warps:stages:split[:tma]).  tune.py's gated
winners; a shape with no row falls back to `int8_fused.pick` (vLLM's tiles)."""
from __future__ import annotations

CONFIGS: dict[tuple[int, int], list[tuple[int, str]]] = {}
