"""What a capture is given and what it holds of one step: its configuration (`ObserverConfig`), the run header its log
opens with (`make_header`), and the step's events, timings and views (`_Step`), which the capture buffers until it writes
the step out.  `verity_vllm.observe.capture` reads them and hands `ObserverConfig` on."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.engine.build import Engine
from verity_vllm.engine.run_facts import model_config_subset, profile_manifest
from verity_vllm.observe.events import SCHEMA, Event, RunHeader, ViewDesc
from verity_vllm.observe.observer import reenter_impl_name

__all__ = ["ObserverConfig", "make_header"]


@dataclass
class ObserverConfig:
    mode: str = "record"                 # "none" (baseline timing only) | "empty" (EmptyMode) | "record"
    reenter: bool = True
    triton: bool = True
    module_hooks: bool = True
    write_log: bool = True               # False: build every event, discard the buffer at step end (the overhead ladder's rungs c-f)
    snapshot_steps: frozenset = frozenset()
    snapshot_dir: Path | None = None
    snapshot_max_bytes: int = 32 << 20
    snapshot_max_bytes_op: int | None = None   # limit at dispatcher-call events (None: snapshot_max_bytes); Triton launches keep the full limit
    arena_bytes: int = 1 << 30
    draft: bool = False                  # "record" only: the speculator's `propose` as the step's phase "draft", its model's tensors as roots


@dataclass
class _Step:
    step: int
    events: list[Event] = field(default_factory=list)
    forward_s: float = 0.0
    sample_s: float = 0.0
    draft_s: float = 0.0
    awaiting_draft: bool = False         # sampled, held open for the speculator's propose (`ObserverConfig.draft`)
    t_wall0: float = 0.0
    wall_s: float = 0.0
    facts: dict = field(default_factory=dict)
    input_ids_view: ViewDesc | None = None
    positions_view: ViewDesc | None = None
    sampled_view: ViewDesc | None = None
    logits_view: ViewDesc | None = None
    sampled_shape: tuple[int, int] = (0, 0)
    event: Any = None
    n_snap: int = 0


def make_header(engine: Engine, workload: dict, *, adapter_revision: str, wheel_sha256: str | None = None,
                vllm_commit: str | None = None, extra_notes: dict | None = None) -> RunHeader:
    manifest, pid, notes = profile_manifest(engine, adapter_revision=adapter_revision, wheel_sha256=wheel_sha256,
                                            vllm_commit=vllm_commit, sampling=workload["sampling"])
    notes["reenter_impls"] = {op: reenter_impl_name(op) for op in ("vllm.unified_attention_with_output", "vllm.unified_kv_cache_update")}
    notes["phases"] = ("forward = inside the model's root forward; other = the runner's input preparation / post-forward work inside "
                       "execute_model; sample = runner.sample (lm_head GEMM + sampler)")
    notes["observer_sync"] = ("no observer-issued synchronisation; step facts + sampled ids are copied D2H asynchronously behind a "
                              "CUDA event and read at the next step start; torch.cuda.synchronize only at the end of snapshot steps")
    if extra_notes:
        notes.update(extra_notes)
    model = {"repo": engine.cp["repo"], "revision": engine.cp["revision"], "local_path": engine.cp.get("local_path"),
             "case": engine.case, "config": model_config_subset(engine)}
    wl = {"requests": workload["requests"], "sampling": workload["sampling"], "source": workload.get("source"), "sha256": workload.get("sha256")}
    # n > 1: the request ids the runner sees are vLLM's children (`parallel_sampling.ParentRequest.get_child_info`:
    # id f"{index}_{request_id}", seed + index when seeded).  Recorded as a fact of the fork; the request record itself is unchanged.
    children = [{"request_id": f"{i}_{r['request_id']}", "parent": r["request_id"], "index": i,
                 "seed": None if r.get("seed") is None else int(r["seed"]) + i}
                for r in workload["requests"] for i in range(int(r.get("n") or 1)) if int(r.get("n") or 1) > 1]
    if children:
        wl["children"] = children
    return RunHeader(SCHEMA, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), pid, manifest, model, wl, notes)
