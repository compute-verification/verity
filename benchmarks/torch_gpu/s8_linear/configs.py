"""Each exact kernel's GEMM config per large-model linear, by "kernel:arch" (run.py's `Wgmma` and `Triton`, `int8_fused.arch`):
(N, K) -> [(m_max, config)], the first m_max >= M wins and an M past the last takes the last (`wgmma.Cfg` text
TMxBN:cCTAS:gGROUP; `int8_fused.Cfg` text BLOCK_MxBLOCK_NxBLOCK_K:warps:stages:split[:tma]).  tune.py's gated winners; a
shape with no row takes the kernel's default (`wgmma.default_cfg`, `int8_fused.pick`)."""
from __future__ import annotations

CONFIGS: dict[str, dict[tuple[int, int], list[tuple[int, str]]]] = {"wgmma:sm_90": {}, "triton:sm_120": {}}
