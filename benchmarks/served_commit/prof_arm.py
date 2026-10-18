"""served_commit's noretain arm with NVTX ranges for nsys: one per run, per engine step (exec:<tokens>), per drain and per
committer step end. Measurement only."""
from __future__ import annotations

import time
from typing import Any, ClassVar

import torch

from served_commit.arm import ServedCommitArm


def _thread_cpu() -> dict[str, float]:
    """CPU seconds (user + system) each thread of this process has used, summed by Python thread name (as pipeline/tp/trace.py)."""
    import os
    import threading
    from pathlib import Path

    names = {t.native_id: t.name for t in threading.enumerate()}
    tick = os.sysconf("SC_CLK_TCK")
    cpu: dict[str, float] = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            head, tail = Path(f"/proc/self/task/{tid}/stat").read_text().rsplit(")", 1)
        except (OSError, ValueError):
            continue
        f = tail.split()
        key = names.get(int(tid)) or f"native:{head.split('(', 1)[1]}"
        cpu[key] = cpu.get(key, 0.0) + (int(f[11]) + int(f[12])) / tick
    return cpu


def _cpu_delta(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {k: round(v - a.get(k, 0.0), 3) for k, v in sorted(b.items()) if v - a.get(k, 0.0) > 0}


def _ranged(obj: Any, name: str, label: Any) -> None:
    orig = getattr(obj, name)

    def run(*a: Any, **k: Any) -> Any:
        torch.cuda.nvtx.range_push(label(a, k) if callable(label) else label)
        try:
            return orig(*a, **k)
        finally:
            torch.cuda.nvtx.range_pop()

    setattr(obj, name, run)


class ProfArm(ServedCommitArm):
    def ready(self, target: Any) -> None:
        super().ready(target)
        _ranged(self.source, "drain", "drain")
        _ranged(self.committer, "_end_step", "end_step")

    def begin_run(self, target: Any) -> None:
        super().begin_run(target)
        runner = self.committer.runner
        _ranged(runner, "execute_model",
                lambda a, k: f"exec:{getattr(a[0] if a else k.get('scheduler_output'), 'total_num_scheduled_tokens', -1)}")
        if hasattr(runner, "sample"):
            _ranged(runner, "sample", "sample")
        torch.cuda.nvtx.range_push(f"run:{self.id}")

    def end_run(self, target: Any) -> None:
        super().end_run(target)
        torch.cuda.nvtx.range_pop()


class RootsArm(ProfArm):
    """Each finished run's root, bounded-staging record and per-step (plan key, predicted length, bytes) appended to
    $RESEARCH_RUN_DIR/out/roots.jsonl, and each distinct run layout (every step's (name, dtype, shape, bytes), once per process) to
    out/layouts.jsonl. Every RootsArm commits under one run id, so their roots compare across arms. ``retain_levels`` None is the
    committer's default (no levels under retain=none)."""

    retain_levels: bool | None = None
    dumped: ClassVar[set[str]] = set()

    def ready(self, target: Any) -> None:
        import hashlib

        super().ready(target)
        com, self.plan_log = self.committer, []
        com.run_id = "bench:roots"
        com.program_digest = hashlib.sha256(b"cmt-integ/no-program-digest/" + com.run_id.encode()).digest()
        if self.retain_levels is not None:
            com.retain_levels = self.retain_levels
        orig = com._begin_step

        def begin_step() -> None:
            orig()
            self.plan_log.append((int(com._cur.step), int(com._plan_key), int(getattr(com._cur, "plan_used", -2))))

        com._begin_step = begin_step

    def begin_run(self, target: Any) -> None:
        self.plan_log = []
        super().begin_run(target)
        self.cpu0 = _thread_cpu()

    def end_run(self, target: Any) -> None:
        cpu = _cpu_delta(self.cpu0, _thread_cpu())
        super().end_run(target)
        import hashlib
        import json
        import os

        rc = self.pending
        bs = dict((rc.extra or {}).get("bounded_staging") or {})
        w = dict((rc.extra or {}).get("windows") or {})
        st = dict(self.committer._nc.stats()) if hasattr(self.committer._nc, "stats") else {}
        lay = self.committer._layouts
        rec = {"arm": self.id, "run": len(self.runs), "name": self.run_name, "run_root": bytes(rc.run_root).hex(), "steps": len(lay),
               "learned_only_steps": len(bs.get("learned_only_steps") or ()), "plan_mismatches": bs.get("plan_mismatches"),
               "bounded": bs.get("enabled"), "reupload_bytes": w.get("reupload_bytes"),
               "plans": {k: st.get(k) for k in ("plans", "plan_entries", "plans_dropped", "learn_steps")},
               "plan_hits": sum(1 for _s, _k, u in self.plan_log if u >= 0), "thread_cpu_s": cpu,
               "steps_log": [(s, k & 0xFFFF, u, sum(m.nbytes for m in lay.get(s, ()))) for s, k, u in self.plan_log]}
        out = os.path.join(os.environ.get("RESEARCH_RUN_DIR", "."), "out", "roots.jsonl")
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        steps = {str(s): [[m.name, m.dtype, list(m.shape), int(m.nbytes)] for m in lay[s]] for s in sorted(lay)}
        sig = hashlib.sha256(json.dumps(steps).encode()).hexdigest()[:16]
        if sig not in RootsArm.dumped:
            RootsArm.dumped.add(sig)
            with open(os.path.join(os.path.dirname(out), "layouts.jsonl"), "a") as f:
                f.write(json.dumps({"arm": self.id, "run": len(self.runs), "sig": sig, "steps": steps}) + "\n")


class StockProfArm:
    """The stock engine with ProfArm's exec/sample and run ranges and nothing else: the control's own per-step times."""

    protocols: tuple[str, ...] = ()
    precision = "bfloat16"
    control = "stock"
    resident = True

    def __init__(self, id: str) -> None:
        self.id = id

    def install(self, target: Any) -> StockProfArm:
        return self

    def ready(self, target: Any) -> None:
        runner = target.engine._engine.runner
        _ranged(runner, "execute_model",
                lambda a, k: f"exec:{getattr(a[0] if a else k.get('scheduler_output'), 'total_num_scheduled_tokens', -1)}")
        if hasattr(runner, "sample"):
            _ranged(runner, "sample", "sample")

    def begin_run(self, target: Any) -> None:
        self.cpu0 = _thread_cpu()
        torch.cuda.nvtx.range_push(f"run:{self.id}")

    def end_run(self, target: Any) -> None:
        import json
        import os

        torch.cuda.nvtx.range_pop()
        out = os.path.join(os.environ.get("RESEARCH_RUN_DIR", "."), "out", "roots.jsonl")
        with open(out, "a") as f:
            f.write(json.dumps({"arm": self.id, "thread_cpu_s": _cpu_delta(self.cpu0, _thread_cpu())}) + "\n")


class SwitchArm(RootsArm):
    """The prof arm with a 0.5 ms GIL switch interval (default 5 ms) in its engine's process: if the serving thread waits for the
    worker's Python, its step time falls."""

    def ready(self, target: Any) -> None:
        import sys

        sys.setswitchinterval(0.0005)
        super().ready(target)


class NativeDrainArm(RootsArm):
    """The prof arm whose compiled source drains a replayed step's slots through the committer's slot plans (one native call)."""

    def ready(self, target: Any) -> None:
        super().ready(target)
        self.source.native_drain = True


class RowsArm(NativeDrainArm):
    """The nativedrain arm committing under rows-sha512, COMMIT-GPU's leaves and trees on the windowed worker."""

    commit_scheme = "rows-sha512"


class FinalizeProfArm(NativeDrainArm):
    """The nativedrain arm with the committer's finalize under cProfile: the top functions by cumulative time of each run's finalize
    (inside the timed window) appended to $RESEARCH_RUN_DIR/out/finalize_prof.txt."""

    def ready(self, target: Any) -> None:
        import cProfile
        import io
        import os
        import pstats

        super().ready(target)
        com, orig = self.committer, self.committer.finalize
        out = os.path.join(os.environ.get("RESEARCH_RUN_DIR", "."), "out", "finalize_prof.txt")

        def finalize():
            pr = cProfile.Profile()
            t0 = time.perf_counter()
            pr.enable()
            try:
                return orig()
            finally:
                pr.disable()
                buf = io.StringIO()
                buf.write(f"== {self.id} run {len(self.runs)} finalize wall {1e3 * (time.perf_counter() - t0):.2f} ms\n")
                pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(30)
                with open(out, "a") as f:
                    f.write(buf.getvalue())

        com.finalize = finalize


class _PassThroughBoundaries:
    """A call-boundary source that takes every name as it is: the identity check's verdict is unchanged, but it judges every step."""

    short_name = "pass_through"

    def split_step_names(self, names: Any, step: int, padding_only: bool) -> tuple[Any, dict]:  # noqa: ARG002
        return names, {}


class NoMemoArm(NativeDrainArm):
    """The nativedrain arm whose finalize checks identities step by step, as before the per-tuple verdicts."""

    def ready(self, target: Any) -> None:
        super().ready(target)
        com, orig = self.committer, self.committer.finalize

        def finalize():
            src = _PassThroughBoundaries()
            com.sources.append(src)
            try:
                return orig()
            finally:
                com.sources.remove(src)

        com.finalize = finalize


SERVED_COMMIT_PROF = RootsArm("served-commit-prof", retain="none")
SERVED_COMMIT_SWITCH = SwitchArm("served-commit-switch", retain="none")
SERVED_COMMIT_NATIVEDRAIN = NativeDrainArm("served-commit-nativedrain", retain="none")
SERVED_COMMIT_ROWS = RowsArm("served-commit-rows", retain="none")
SERVED_COMMIT_FPROF = FinalizeProfArm("served-commit-fprof", retain="none")
SERVED_COMMIT_NOMEMO = NoMemoArm("served-commit-nomemo", retain="none")
STOCK_PROF = StockProfArm("stock-prof")
