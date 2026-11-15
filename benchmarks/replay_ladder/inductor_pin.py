"""Inductor's on-device choices in a served or replayed engine, pinned and logged in each of its processes: a vLLM general
plugin (`register`, loaded by every process of a server whose site directory `site` wrote, configured by the file it wrote
there) and a module the replay calls (`apply`, configured by the ladder's settings).

Inductor times candidates on the device and keeps the fastest: a kernel's configs on its first launch in each process
(``CachingAutotuner``), where a reduction's config sets the order of its sums, and at compile time whether to fuse kernels
(combo kernels). Two processes can choose differently. Pinned, every one of those timings is the same, so each choice
falls to Inductor's first candidate in every process. Inductor's own deterministic mode (``TORCHINDUCTOR_DETERMINISTIC=1``)
can't stand in at the integration's torch: Dynamo turns it off after a process's first compile (restoring
``torch.use_deterministic_algorithms``, which sets it), and while it holds it refuses the combo-kernel timing vLLM's
compile runs.

Logging, each process appends every kernel that chose among more than one config at run time, with the one it kept, to
``tilings.PID.jsonl`` in the log's directory.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

#: the benchmarker's timings Inductor chooses by
TIMINGS = ("benchmark", "benchmark_cpu", "benchmark_gpu", "benchmark_gpu_with_cuda_graph")
#: the site directory's configuration of its servers' processes, beside the module
CONFIG = "inductor_pin.json"


def register() -> None:
    f = Path(__file__).with_name(CONFIG)
    cfg = json.loads(f.read_text()) if f.is_file() else {}
    apply(pin=bool(cfg.get("pin")), tilings=cfg.get("tilings"))


def apply(*, pin: bool, tilings: str | None) -> None:
    """Pin Inductor's timings, and log its run-time choices to the directory ``tilings``."""
    if pin:
        pin_timings()
    if tilings:
        log_tilings(tilings)


def pin_timings(benchmarker: Any = None) -> None:
    """Make every timing of Inductor's benchmarker (or ``benchmarker``) 1.0."""
    if benchmarker is None:
        from torch._inductor.runtime.benchmarking import benchmarker
    if getattr(benchmarker, "ladder_pinned", False):
        return

    def timing(*args: Any, **kwargs: Any) -> float:
        return 1.0

    for name in TIMINGS:
        if hasattr(benchmarker, name):
            setattr(benchmarker, name, timing)
    benchmarker.ladder_pinned = True


def log_tilings(d: str, autotuner: Any = None) -> None:
    """Wrap Inductor's ``CachingAutotuner.autotune_to_one_config`` (or ``autotuner``'s) to log each choice to ``d``."""
    if autotuner is None:
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner as autotuner
    real = autotuner.autotune_to_one_config
    if getattr(real, "ladder", False):
        return

    def cfg(c: Any) -> dict:
        return {**c.kwargs, "num_warps": c.num_warps, "num_stages": c.num_stages}

    def autotune_to_one_config(self: Any, *args: Any, **kwargs: Any) -> Any:
        candidates = [cfg(x.config) for x in self.launchers]
        out = real(self, *args, **kwargs)
        rec = {"kernel": self.inductor_meta.get("kernel_name") or self.fn.__name__,
               "heuristic": getattr(getattr(self, "heuristic_type", None), "name", None), "candidates": candidates,
               "kept": cfg(self.launchers[0].config)}
        Path(d).mkdir(parents=True, exist_ok=True)
        with open(Path(d) / f"tilings.{os.getpid()}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        return out

    autotune_to_one_config.ladder = True  # type: ignore[attr-defined]
    autotuner.autotune_to_one_config = autotune_to_one_config


def site(d: Path, *, pin: bool, tilings: str | None) -> Path:
    """Add this module to the site directory ``d``, which a vLLM server has on its path, as a general plugin that pins and
    logs as `apply` does."""
    info = d / "verity_replay_ladder-0.0.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: verity-replay-ladder\nVersion: 0.0\n")
    (info / "entry_points.txt").write_text("[vllm.general_plugins]\nverity_replay_ladder = inductor_pin:register\n")
    shutil.copy(Path(__file__), d / "inductor_pin.py")
    (d / CONFIG).write_text(json.dumps({"pin": pin, "tilings": tilings}))
    return d
