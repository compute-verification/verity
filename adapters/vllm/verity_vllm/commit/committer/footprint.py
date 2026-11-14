"""The host committer's two monitors, both of which read no value and change no root, layout or opening.

`VERITY_FOOTPRINT=<path.jsonl>`: a per-step device-memory timeline written from the serving thread (begin/end of step) and the worker
(absorb = hashing done, release = arena block back in the pool): bytes staged by value family, arena block sizes, retained copies and
tree levels, torch allocated/reserved.  Design input for the streaming absorb/release work; no effect on what is bound.

`GcPauses`: Python's collections while a run is installed.
"""

from __future__ import annotations

import gc
import json
import threading
import time
from typing import Any

from verity_vllm.commit.committer.acquired import TensorMeta
from verity_vllm.commit.committer.optional_torch import torch

FAMILY_RULES = (   # (substring of the tensor name -> value family) for the footprint profile; first match wins
    ("fa2.m1/", "fa2_hidden_m1"), ("fa3.m1/", "fa3_hidden_m1"), ("sampled_token_ids", "sampled_ids"), ("logits", "logits"), ("embed_tokens/in", "input_ids"),
    ("embed_tokens", "embedding"), ("qkv_proj", "gemm_qkv"), ("o_proj", "gemm_o"), ("gate_up_proj", "gemm_gate_up"), ("down_proj", "gemm_down"),
    ("lm_head", "gemm_lm_head"), ("norm", "norm"), ("rotary", "rope"), ("act_fn", "act"), ("attn", "attention_out"),
)


def value_family(name: str) -> str:
    for sub, fam in FAMILY_RULES:
        if sub in name:
            return fam
    return "other"


class GcPauses:
    """`gc.callbacks` while a run is installed: each collection of at least `min_s`, as (generation, start, seconds, thread), the start on
    `time.perf_counter` like the steps' `t_step_end`, so a pause lands on its step.

    `install` first collects and freezes the heap, as vLLM does once its warm-up is done (`gc_utils.freeze_gc_heap`): the committer, its
    sources and the manifests they hold are built after that freeze, and a full collection over them took 720-960 ms on every rank's
    serving thread of GLM-5.3 TP8, stalling all ranks behind the next all-reduce.  Collections never touch a value."""

    def __init__(self, min_s: float = 1e-3) -> None:
        self.min_s = min_s
        self.pauses: list[tuple[int, float, float, str]] = []
        self.freeze_s: list[float] = []
        self._t0 = 0.0

    def __call__(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._t0 = time.perf_counter()
        elif (dt := time.perf_counter() - self._t0) >= self.min_s:
            self.pauses.append((int(info["generation"]), self._t0, dt, threading.current_thread().name))

    def install(self) -> None:
        if self in gc.callbacks:
            return
        t0 = time.perf_counter()
        gc.collect()
        gc.freeze()
        self.freeze_s.append(time.perf_counter() - t0)
        gc.callbacks.append(self)

    def uninstall(self) -> None:
        if self in gc.callbacks:
            gc.callbacks.remove(self)

    def record(self) -> dict[str, Any]:
        return {"min_s": self.min_s, "pauses": list(self.pauses), "total_s": sum(p[2] for p in self.pauses), "freeze_s": list(self.freeze_s)}


class _Footprint:
    """Per-step device-memory timeline (`VERITY_FOOTPRINT`).  One JSON line per event; `t` is perf_counter seconds since the profiler was
    created.  Events: begin_step, end_step (serving thread), absorbed (leaves + levels computed and the copy taken; the arena block is released
    right after), plus `run` summaries written by the committer's finalize."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.t0 = time.perf_counter()
        with open(path, "a"):   # the file is made here, or the path refused here, before any step; each event opens it again
            pass
        self._lock = threading.Lock()

    @staticmethod
    def cuda() -> dict[str, int]:
        if torch is None or not torch.cuda.is_available():
            return {}
        free, total = torch.cuda.mem_get_info()
        return {"allocated": int(torch.cuda.memory_allocated()), "reserved": int(torch.cuda.memory_reserved()),
                "device_free": int(free), "device_total": int(total)}

    def event(self, kind: str, **kw: Any) -> None:
        rec = {"t": round(time.perf_counter() - self.t0, 6), "event": kind, "thread": threading.current_thread().name, **kw, "cuda": self.cuda()}
        line = json.dumps(rec, default=str) + "\n"
        with self._lock, open(self.path, "a") as f:
            f.write(line)

    @staticmethod
    def families(metas: list[TensorMeta]) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for m in metas:
            d = out.setdefault(value_family(m.name), {"bytes": 0, "tensors": 0})
            d["bytes"] += int(m.nbytes)
            d["tensors"] += 1
        return out
