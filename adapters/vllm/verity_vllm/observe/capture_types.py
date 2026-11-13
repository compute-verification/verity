"""What a capture is given and what it holds of one step: its configuration (`ObserverConfig`), and the step's events,
timings and views (`_Step`), which the capture buffers until it writes the step out.  `verity_vllm.observe.capture` reads
both and hands `ObserverConfig` on."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.observe.events import Event, ViewDesc


@dataclass
class ObserverConfig:
    mode: str = "record"                 # "none" (baseline timing only) | "empty" (EmptyMode) | "record"
    reenter: bool = True
    triton: bool = True
    module_hooks: bool = True
    write_log: bool = True               # False: build every event, discard the buffer at step end (M0 rungs c-f)
    snapshot_steps: frozenset = frozenset()
    snapshot_dir: Path | None = None
    snapshot_max_bytes: int = 32 << 20
    snapshot_max_bytes_op: int | None = None   # limit at dispatcher-call events (None: snapshot_max_bytes); Triton launches keep the full limit
    arena_bytes: int = 1 << 30


@dataclass
class _Step:
    step: int
    events: list[Event] = field(default_factory=list)
    forward_s: float = 0.0
    sample_s: float = 0.0
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
