"""Sampled proofs' Commit as an arm of the shared ``verity-bench vllm`` driver.

``COMMIT`` serves with the Commit's collector installed on the engine: every module-output Call of the eager step, the
embedding's ids and the sampled ids are staged through a ring of device windows, hashed in place by the GPU chunk tree
(``chunk-leaf-v2``) and bound into one step root per engine step and one run root per measured run, while serving
continues (``native_collect_v2b``, the collector ``verity-vllm row``'s Commit runs).  It is gated against ``stock-fp8``,
the same engine with nothing installed: greedy tokens must be identical, and every measured run must close with a run
root, no collector error and every staged tensor committed.

What the arm does not do, against the Commit of record:

* no Build: the acquisition follows the committer's class tables, not a required-value manifest, and the root binds no
  Program digest;
* no FlashAttention hidden-value tap (the Commit's ``--hidden-so``), so attention interiors are not bound;
* retention is roots only (``VERITY_RETAIN=none``, levels dropped after each step's root): an opening regenerates its
  step by replay, which this arm does not time.

Each measured rep reports, as extras, the bytes and leaves committed, the steps, the root-ready delay (the time from
the end of the last step to the run root) and ``serve_to_root_s``, the run's wall time to its root.

Settings, all optional: ``VERITY_SP_ARM_WINDOW_MB`` (default 256), ``VERITY_SP_ARM_WINDOW_SLOTS`` (8) and
``VERITY_SP_ARM_LEVELS`` (``drop``, or ``keep`` to retain every step's tree levels on host, as ``--retain none`` does).
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ARM_ID = "sp-commit"
CONTROL = "stock-fp8"

_COMMITTER_SOURCES = (
    "native_collect.py",
    "native_host.py",
    "native_collect.cpp",
    "native_gather.cu",
    "native_tree.cu",
    "native_plans.h",
    "hidden_gpu_src/hidden_gpu_tree.cu",
)


def settings(environ: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    window_mb = int(env.get("VERITY_SP_ARM_WINDOW_MB", "256"))
    slots = int(env.get("VERITY_SP_ARM_WINDOW_SLOTS", "8"))
    levels = env.get("VERITY_SP_ARM_LEVELS", "drop")
    if window_mb < 1 or window_mb & (window_mb - 1):
        raise ValueError("VERITY_SP_ARM_WINDOW_MB must be a power of two")
    if slots < 3:
        raise ValueError("VERITY_SP_ARM_WINDOW_SLOTS must be at least 3")
    if levels not in ("drop", "keep"):
        raise ValueError("VERITY_SP_ARM_LEVELS must be 'drop' or 'keep'")
    return {"window_mb": window_mb, "window_slots": slots, "levels": levels}


def digest(paths: list[Path]) -> str:
    out = hashlib.sha256()
    for path in paths:
        name = path.name.encode()
        out.update(len(name).to_bytes(4, "big"))
        out.update(name)
        out.update(path.read_bytes())
    return out.hexdigest()


def run_passed(record: dict[str, Any]) -> bool:
    """A measured or gate run counts only with a run root, steps, no collector error and every staged tensor committed."""

    return (
        bool(record.get("run_root"))
        and record.get("steps", 0) > 0
        and record.get("errors_n", 1) == 0
        and record.get("expected_tensors") == record.get("committed_tensors")
    )


def record_of(rc: Any, t_end: float | None) -> dict[str, Any]:
    extra = rc.extra or {}
    last_end = max((s.t_step_end for s in rc.steps), default=0.0)
    out = {
        "run_root": rc.run_root.hex(),
        "steps": len(rc.steps),
        "committed_bytes": float(sum(s.n_bytes for s in rc.steps)),
        "leaves": float(sum(s.n_leaves for s in rc.steps)),
        "root_ready_delay_s": float(rc.root_ready_delay_s),
        "finalize_s": float(rc.t_finalize_s),
        "gpu_leaf_ms": float((extra.get("gpu_tree_ms") or {}).get("leaf_ms", 0.0)),
        "gpu_tree_ms": float((extra.get("gpu_tree_ms") or {}).get("tree_ms", 0.0)),
        "host_retained_bytes": float(extra.get("host_retained_bytes") or 0),
        "errors_n": int(extra.get("errors_n", 0)),
        "errors": list(extra.get("errors") or [])[:3],
        "expected_tensors": int(extra.get("expected_tensors", 0)),
        "committed_tensors": int(extra.get("committed_tensors", 0)),
    }
    if t_end is not None and last_end:
        out["root_after_serve_end_s"] = max(0.0, last_end + float(rc.root_ready_delay_s) - t_end)
    return out


class _Committing:
    """The collector on one engine: built once the engine exists, finalized and reset around every run."""

    def __init__(self) -> None:
        self.settings = settings()
        self.com: Any | None = None
        self.pending = False
        self.t_start: float | None = None
        self.last: dict[str, Any] | None = None
        self.runs: list[dict[str, Any]] = []
        self.weights: dict[str, Any] = {}
        self.window: dict[str, Any] = {}
        self.kernel = ""
        self.closed = False

    def ready(self, target: Any) -> None:
        from verity_vllm.commit.committer import native_collect
        from verity_vllm.engine.engine_profile import get_model_runner

        s = self.settings
        os.environ["VERITY_WINDOW_MB"] = str(s["window_mb"])
        os.environ["VERITY_WINDOW_SLOTS"] = str(s["window_slots"])
        os.environ["VERITY_RETAIN"] = "none"
        os.environ.setdefault("VERITY_WEIGHTS_HASH", "device")
        raw = target.engine._engine.llm
        engine = SimpleNamespace(llm=raw, runner=get_model_runner(raw), model=target.engine._engine.model)
        com = native_collect.NativeCollectCommitter(
            engine,
            run_id=f"bench-{ARM_ID}",
            arena_bytes=1 << 20,
            wrap=True,
            gather=True,
            native_hooks=True,
            auto_blocks=True,
            native_worker=True,
            layout="chunk-leaf-v2",
        )
        if not com.windowed or com.retain != "none":
            raise RuntimeError(f"sampled-proofs arm: collector not windowed with retain none ({com.name}, retain {com.retain})")
        if s["levels"] == "drop":
            _drop_levels_after_each_step(com)
        self.window = com.enable_windows(None)
        self.weights = dict(com.register_weights())
        com.install()
        self.com = com
        here = Path(native_collect.__file__).resolve().parent
        self.kernel = digest([Path(__file__).resolve(), *(here / p for p in _COMMITTER_SOURCES)])

    def _finish(self, t_end: float | None) -> dict[str, Any] | None:
        if not self.pending or self.com is None:
            return None
        self.pending = False
        stepped = self.com._step_index > 0
        rc = self.com.finalize()
        if not stepped:
            return None
        rec = record_of(rc, t_end)
        if t_end is not None and self.t_start is not None:
            rec["serve_to_root_s"] = (t_end - self.t_start) + rec.get("root_after_serve_end_s", 0.0)
        self.runs.append(rec)
        return rec

    def spans(self) -> dict[str, float]:
        """The driver calls this before every run and again after a measured one, so each call closes the run in flight
        (a run with no steps, opened by the call after a measured run, is dropped) and opens the next."""

        if self.com is None:
            raise RuntimeError("sampled-proofs arm is not ready")
        now = time.perf_counter()
        rec = self._finish(now)
        if rec is not None:
            self.last = rec
        self.com.reset()
        self.pending = True
        self.t_start = time.perf_counter()
        return {}

    def extras(self) -> dict[str, float]:
        rec = self.last or {}
        return {k: float(v) for k, v in rec.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}

    def gate(self) -> dict[str, Any]:
        self._finish(None)
        failed = [r for r in self.runs if not run_passed(r)]
        return {
            "passed": bool(self.runs) and not failed,
            "runs": len(self.runs),
            "failed_runs": failed[:3],
            "collector": getattr(self.com, "name", None),
            "modules_hooked": len(getattr(self.com, "_modules", ()) or ()),
            "weights": {k: v for k, v in self.weights.items() if k in ("root", "tensors", "bytes", "wall_s", "hash")},
            "window": {k: self.window.get(k) for k in ("window_bytes", "slots", "layout")},
            "levels": self.settings["levels"],
        }

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.com is not None:
            try:
                self._finish(None)
            finally:
                self.com.close()


def _drop_levels_after_each_step(com: Any) -> None:
    """Keep each step's root and give its host tree levels back to the pool as soon as the root is bound."""

    inner = com._commit_step_windowed
    pool = com.pool

    def commit_step(job: Any) -> None:
        taken: list[Any] = []
        take = pool.take

        def take_block(nbytes: int) -> Any:
            blk = pool.take_block(int(nbytes))
            taken.append(blk)
            return blk[: int(nbytes)]

        pool.take = take_block
        try:
            inner(job)
        finally:
            pool.take = take
        com._gpu_levels[job.step] = []
        for blk in taken:
            pool.give_back(blk)

    com._commit_step_windowed = commit_step


class _ArmModule:
    precision = "fp8"
    protocols: tuple[str, ...] = ()
    resident = True

    def __init__(self, arm_id: str, control: str) -> None:
        self.id = arm_id
        self.control = control
        self.state: _Committing | None = None

    def install(self, target: Any) -> _Committing:
        self.state = _Committing()
        return self.state

    def gate(self, target: Any) -> dict[str, Any]:
        return self._state().gate()

    def spans(self, target: Any) -> dict[str, float]:
        return self._state().spans()

    def extras(self, target: Any) -> dict[str, float]:
        return self._state().extras()

    def kernel_digest(self, target: Any) -> str:
        return self._state().kernel

    def _state(self) -> _Committing:
        if self.state is None:
            raise RuntimeError(f"sampled-proofs arm {self.id!r} is not installed")
        return self.state


COMMIT = _ArmModule(ARM_ID, CONTROL)

__all__ = ["COMMIT"]
