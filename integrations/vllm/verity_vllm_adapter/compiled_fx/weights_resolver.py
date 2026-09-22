"""Resolve weight values from the checkpoint instead of from ``weights/values.bin`` (CONTRACT §16.8:
bundles reference weights by checkpoint sha, never copy them).

A bundle written with ``--weights-values hash`` has a ``weights/index.json`` (canonical id, dtype, shape,
sha256, leaf hash, ``source_sha256`` of the checkpoint's safetensors) and no values. The verifier
installs :class:`CheckpointWeightResolver` on the capture reader: it checks the local checkpoint files
against the index's ``source_sha256``, loads the model once through vLLM's own loader (eager engine, small
KV cache; the same loader that produced the committed values: fused qkv / gate_up, tied lm_head, the
rotary ``cos_sin_cache`` buffer), and answers each ``WeightEntry`` with the parameter's bytes. The reader
re-hashes every resolved value against the index (``WeightsReader.read``), so a checkpoint that does not
reproduce the committed weights fails authentication.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import torch

from verity_vllm_adapter import profile as prof
from verity_vllm_adapter.capture import normalize_module


class CheckpointWeightResolver:
    def __init__(self, cp: dict[str, Any], source_sha256: dict[str, str], *, gpu_memory_utilization: float = 0.15, log=print) -> None:
        self.cp = cp
        self.source_sha256 = dict(source_sha256)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.log = log
        self._params: dict[str, torch.Tensor] | None = None
        self.checked_files: dict[str, str] = {}
        self.build_s = 0.0
        self.resolved = 0

    def check_files(self) -> None:
        """The checkpoint on disk must be the one the index names (sha256 per safetensors shard)."""
        local = Path(self.cp["local_path"])
        for fname, sha in self.source_sha256.items():
            p = local / fname
            if not p.exists():
                raise FileNotFoundError(f"checkpoint file {p} named by the weights index is missing")
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            got = h.hexdigest()
            if got != sha:
                raise ValueError(f"{p}: sha256 {got} != index source_sha256 {sha}")
            self.checked_files[fname] = got

    def _load(self) -> dict[str, torch.Tensor]:
        if self._params is not None:
            return self._params
        self.check_files()
        from vllm import LLM

        kw = prof.engine_kwargs(self.cp, replay=False)
        kw.update({"enforce_eager": True, "gpu_memory_utilization": self.gpu_memory_utilization, "max_model_len": 512, "enable_prefix_caching": False})
        t0 = time.perf_counter()
        llm = LLM(**kw)
        model = prof.get_model_runner(llm).get_model()
        params: dict[str, torch.Tensor] = {}
        seen: set[int] = set()
        for n, p in list(model.named_parameters()) + list(model.named_buffers()):
            if p.data_ptr() in seen or p.numel() == 0:
                continue
            seen.add(p.data_ptr())
            params[normalize_module(n)] = p.detach().contiguous().cpu()
        del llm, model
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        self.build_s = time.perf_counter() - t0
        self._params = params
        self.log(f"weights resolver: {len(params)} parameters/buffers loaded from {self.cp['local_path']} in {self.build_s:.1f}s; "
                 f"checkpoint files verified {self.checked_files}")
        return params

    def __call__(self, entry: Any) -> bytes:
        params = self._load()
        name = entry.param
        if name not in params:
            raise KeyError(f"{name}: not a parameter/buffer of the loaded model (have {len(params)})")
        p = params[name]
        if str(p.dtype).replace("torch.", "") != entry.dtype or tuple(p.shape) != tuple(entry.shape):
            raise ValueError(f"{name}: loaded {p.dtype} {tuple(p.shape)} != index {entry.dtype} {tuple(entry.shape)}")
        self.resolved += 1
        return p.reshape(-1).view(torch.uint8).numpy().tobytes()

    def to_json(self) -> dict[str, Any]:
        return {"kind": "checkpoint", "repo": self.cp.get("repo"), "revision": self.cp.get("revision"), "local_path": self.cp.get("local_path"),
                "checked_files": self.checked_files, "build_s": self.build_s, "resolved": self.resolved}
