"""served_commit's arm (`arm.py`) with NVTX ranges for nsys: one per run, per engine step (exec:<tokens>), per drain and per
committer step end. Measurement only."""
from __future__ import annotations

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
    """Each finished run's root, bounded-staging record, per-step (plan key, predicted length, bytes) and per-thread CPU appended to
    $RESEARCH_RUN_DIR/out/roots.jsonl, and each distinct run layout (every step's (name, dtype, shape, bytes), once per process) to
    out/layouts.jsonl. Every RootsArm commits under one run id, so their roots compare across arms. ``retain_levels`` None is the
    committer's default (no levels under retain=none).

    The records are written after the run's timed period, where `ServedCommitArm` checks a run (the next ``begin_run``, before its
    clock, or the harness reading ``extras`` or ``gate``): building and hashing a run's layout costs tens of milliseconds at 26k
    tensors (and its garbage collections), which a run's time must not hold. A run's thread CPU runs from its ``begin_run`` to its
    record, so it includes the finalize."""

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
        self._check()
        self.plan_log = []
        super().begin_run(target)
        self.cpu0 = _thread_cpu()

    def _check(self) -> None:
        if self.pending is not None:
            self._record(self.pending)
        super()._check()

    def _record(self, rc: Any) -> None:
        import hashlib
        import json
        import os

        cpu = _cpu_delta(self.cpu0, _thread_cpu())
        bs = dict((rc.extra or {}).get("bounded_staging") or {})
        w = dict((rc.extra or {}).get("windows") or {})
        st = dict(self.committer._nc.stats()) if hasattr(self.committer._nc, "stats") else {}
        lay = self.committer._layouts
        com, src = self.committer, self.source
        rec = {"arm": self.id, "committer": self.committer.name, "run": len(self.runs), "name": self.run_name,
               "toggles": {"batch_copies": bool(src.codegen.get("batch")), "slot_fast": bool(com._nc.slot_fast), "native_end": com.native_end,
                           "rows_off_thread": com.rows_off_thread, "native_drain": src.native_drain},
               "ring": {k: self.windows.get(k) for k in ("window_bytes", "slots")},
               "source": {k: src.stats.get(k) for k in ("replayed_steps", "eager_steps", "drain_s", "capture_copies", "capture_flushes", "strided_copies")},
               "codegen": {k: src.codegen.get(k) for k in ("hooks", "flushes", "kept")},
               "spans_s": {k: round(v, 6) for k, v in com.spans.total.items()}, "spans_calls": dict(com.spans.calls),
               "run_root": bytes(rc.run_root).hex(), "steps": len(lay),
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
    """The stock engine with ProfArm's exec/sample and run ranges and nothing else: the control's own per-step times. Its
    per-thread CPU line in out/roots.jsonl is written after the run's timed period, as `RootsArm`'s records are."""

    protocols: tuple[str, ...] = ()
    precision = "bfloat16"
    control = "stock"
    resident = True

    def __init__(self, id: str) -> None:
        self.id = id
        self.cpu0: dict[str, float] | None = None

    def install(self, target: Any) -> StockProfArm:
        return self

    def ready(self, target: Any) -> None:
        from verity_vllm.engine import handle

        runner = handle.model_runner(target.engine.vllm)
        _ranged(runner, "execute_model",
                lambda a, k: f"exec:{getattr(a[0] if a else k.get('scheduler_output'), 'total_num_scheduled_tokens', -1)}")
        if hasattr(runner, "sample"):
            _ranged(runner, "sample", "sample")

    def begin_run(self, target: Any) -> None:
        self._record()
        self.cpu0 = _thread_cpu()
        torch.cuda.nvtx.range_push(f"run:{self.id}")

    def end_run(self, target: Any) -> None:
        torch.cuda.nvtx.range_pop()

    def extras(self, target: Any) -> dict[str, Any]:
        self._record()
        return {}

    def _record(self) -> None:
        import json
        import os

        if self.cpu0 is None:
            return
        cpu, self.cpu0 = _cpu_delta(self.cpu0, _thread_cpu()), None
        out = os.path.join(os.environ.get("RESEARCH_RUN_DIR", "."), "out", "roots.jsonl")
        with open(out, "a") as f:
            f.write(json.dumps({"arm": self.id, "thread_cpu_s": cpu}) + "\n")


class NativeDrainArm(RootsArm):
    """`RootsArm` whose compiled source drains a replayed step's slots through the committer's slot plans (one native call).
    ``host_fast`` False puts the committer's step back as it was before its host-cost changes (slots staged as views, the step end in
    several calls, the row order expanded on the serving thread); ``batch_copies`` False, the source's captured copies one by one.
    ``window`` (MiB, slots) sizes the staging ring in place of the collector's default (256 MiB x 8)."""

    host_fast = True
    window: tuple[int, int] | None = None

    def ready(self, target: Any) -> None:
        import os

        if self.window is None:
            super().ready(target)
        else:
            os.environ["VERITY_WINDOW_MB"], os.environ["VERITY_WINDOW_SLOTS"] = map(str, self.window)
            try:
                super().ready(target)
            finally:
                del os.environ["VERITY_WINDOW_MB"], os.environ["VERITY_WINDOW_SLOTS"]
        self.source.native_drain = True
        com = self.committer
        com._nc.slot_fast = com.native_end = com.rows_off_thread = self.host_fast


class RowsArm(NativeDrainArm):
    """`NativeDrainArm` committing under rows-sha512, COMMIT-GPU's leaves and trees on the windowed worker."""

    commit_scheme = "rows-sha512"


class RowsBaseArm(RowsArm):
    batch_copies = host_fast = False


class RowsBatchArm(RowsArm):
    host_fast = False


class RowsRing8Arm(RowsArm):
    window = (256, 32)


class RowsW64Arm(RowsArm):
    window = (64, 32)


class RowsCapArm(RowsArm):
    """The rows arm hashing each window in launches of at most PROF_ROWS_CAP (default 32) CTAs while two ring slots are free (the
    committer's rows_cap_blocks policy): the same bytes, with the other SMs left to serving."""

    def ready(self, target: Any) -> None:
        import os

        super().ready(target)
        self.committer.rows_cap_blocks = int(os.environ.get("PROF_ROWS_CAP", "32"))


class RowsRing8CapArm(RowsCapArm):
    window = (256, 32)


class Rows4kArm(RowsArm):
    """The rows arm with every chunk constant at 4 KiB: rows-sha512 commits 4 KiB rows itself (`step_rows.ROW_BYTES`), which this
    sets again, and vllm-v1's 256 B chunk constant (`GPU_CHUNK_BYTES`) with it. Set for the life of the arm's engine process (one
    process per arm). Measurement only."""

    row_bytes = 4096

    def ready(self, target: Any) -> None:
        from verity_vllm.commit import step_rows
        from verity_vllm.commit.committer import host_retention, native_host

        step_rows.ROW_BYTES = native_host.GPU_CHUNK_BYTES = host_retention.GPU_CHUNK_BYTES = self.row_bytes
        super().ready(target)


def _layer_source(**kw: Any) -> Any:
    """A served compiled source acquiring only what a per-layer proof block commits: per compiled piece its first convert_element_type (the
    fused add-norm's residual, one hidden vector per token: the layer-folded proof block's) and the embedding's first output, the keys and
    values each attention layer is given (post-RoPE K, and V), the token inputs and the logits. Measurement only: not the step's
    values, so not a Commit of the step."""
    from verity_vllm.acquire.sources.served_compiled_source import ServedCompiledSource

    class LayerSource(ServedCompiledSource):
        def __init__(self, **kw: Any) -> None:
            super().__init__(**kw)
            self.firsts: set[str] = set()

        def _enter_piece(self, backend: Any, args: tuple) -> None:
            self.firsts = set()
            super()._enter_piece(backend, args)

        def _tap(self, sub: int, name: str, val: torch.Tensor, defer: bool = True) -> None:
            op = name.split(".", 1)[1].split("/", 1)[0]
            if op in ("convert_element_type", "embedding"):
                if op in self.firsts:
                    return
                self.firsts.add(op)
            elif op not in ("input", "kv"):
                return
            super()._tap(sub, name, val, defer)

        def _attention_forward(self, orig: Any, layer: str) -> Any:
            src = self

            def forward(*a: Any, **kw: Any) -> Any:
                for part, i in (("k", 2), ("v", 3)):
                    t = a[i] if len(a) > i else kw.get("key" if part == "k" else "value")
                    if isinstance(t, torch.Tensor):
                        src._tap(1, f"c{src.piece:02d}.kv/{layer}.{part}", t)
                return orig(*a, **kw)
            return forward

    return LayerSource(**kw)


class RowsLayerArm(RowsArm):
    """The rows arm acquiring only the per-layer proof block's values (`_layer_source`)."""

    def install(self, target: Any) -> RowsLayerArm:
        self.source = _layer_source(into=target.hooks, batch_copies=self.batch_copies)
        self.source.install_hooks()
        return self


class RowsLayer4kArm(RowsLayerArm, Rows4kArm):
    """The per-layer rows arm at 4 KiB rows."""


class Rows4kCapArm(RowsCapArm, Rows4kArm):
    """The capped rows arm at 4 KiB rows."""


class RowsLayerCapArm(RowsCapArm, RowsLayerArm):
    """The capped per-layer rows arm."""


class HiPrioArm(NativeDrainArm):
    """`NativeDrainArm` with each engine step on a high-priority stream (the committer's VERITY_HI_PRIO): the worker's hashing,
    on its side stream at the default priority (CUDA's lowest), gets a CTA slot only when no serving CTA waits for one."""

    def ready(self, target: Any) -> None:
        import os

        os.environ["VERITY_HI_PRIO"] = "1"
        try:
            super().ready(target)
        finally:
            del os.environ["VERITY_HI_PRIO"]


class RowsHiPrioArm(HiPrioArm):
    """`HiPrioArm` committing under rows-sha512."""

    commit_scheme = "rows-sha512"


SERVED_COMMIT_ROWS = RowsArm("served-commit-rows", retain="none")
SERVED_COMMIT_ROWS_BASE = RowsBaseArm("served-commit-rows-base", retain="none")
SERVED_COMMIT_ROWS_BATCH = RowsBatchArm("served-commit-rows-batch", retain="none")
SERVED_COMMIT_ROWS_RING8 = RowsRing8Arm("served-commit-rows-ring8", retain="none")
SERVED_COMMIT_ROWS_W64 = RowsW64Arm("served-commit-rows-w64", retain="none")
SERVED_COMMIT_ROWS_CAP = RowsCapArm("served-commit-rows-cap", retain="none")
SERVED_COMMIT_ROWS_RING8_CAP = RowsRing8CapArm("served-commit-rows-ring8-cap", retain="none")
SERVED_COMMIT_ROWS_4K = Rows4kArm("served-commit-rows-4k", retain="none")
SERVED_COMMIT_ROWS_LAYER = RowsLayerArm("served-commit-rows-layer", retain="none")
SERVED_COMMIT_ROWS_LAYER_4K = RowsLayer4kArm("served-commit-rows-layer-4k", retain="none")
SERVED_COMMIT_ROWS_4K_CAP = Rows4kCapArm("served-commit-rows-4k-cap", retain="none")
SERVED_COMMIT_ROWS_LAYER_CAP = RowsLayerCapArm("served-commit-rows-layer-cap", retain="none")
SERVED_COMMIT_ROWS_HIPRIO = RowsHiPrioArm("served-commit-rows-hiprio", retain="none")
STOCK_PROF = StockProfArm("stock-prof")
