"""vLLM worker extension (``worker_extension_cls="pu_ext.PUExt"``): importing it in the engine's worker installs the arm
named by ``VLLM_PU_ARM`` before the model is built, and its methods answer ``LLM.collective_rpc``."""

from __future__ import annotations

import os

import pu_commit

pu_commit.install(os.environ.get("VLLM_PU_ARM", "control"))


class PUExt:
    def pu_stats(self, since_step: int | None = None) -> dict:
        cc = self.vllm_config.compilation_config
        return {**pu_commit.stats(since_step), "cudagraph_mode": str(cc.cudagraph_mode), "compilation_mode": str(getattr(cc, "mode", None)),
                "cudagraph_capture_sizes": list(cc.cudagraph_capture_sizes or ()),
                "num_gpu_blocks": self.vllm_config.cache_config.num_gpu_blocks,
                "block_size": self.vllm_config.cache_config.block_size}

    def pu_verify(self, step: int) -> dict:
        return pu_commit.verify(step)
