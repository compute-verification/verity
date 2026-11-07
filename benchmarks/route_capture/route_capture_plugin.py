"""A vLLM general plugin that records what a served engine executes: every engine step's launch-shape class, and the device
kernels of the first steps of each class (README.md).

`register` runs in every process vLLM starts (the API server, the engine core, each worker) and does nothing unless
ROUTECAP_OUT names a directory (settings.py).  It patches the model runner, V1's or V2's, when its module is imported, and
on each recording rank (ROUTECAP_RANKS) writes under ROUTECAP_OUT:

- ``steps.rank<R>.jsonl``, one row per step from its scheduler output: tokens and requests; each prefill's query length,
  computed tokens (a prefix-cache hit or an earlier chunk) and prompt length; the decodes' count and context lengths; the
  requests' sampling kinds; the CUDA-graph dispatch the runner chose (mode, padded tokens and requests); the step's class
  (`key`) and, when it was profiled, its trace's name;
- ``kineto/rank<R>/step<N>.json.gz``, a profiled step's Chrome trace (`torch.profiler`, CPU and CUDA activity with input
  shapes): its kernels with their grids and blocks, those a CUDA graph replays included, and the eager operators' shapes;
- ``capture_ops.rank<R>.json``, the operators run while CUDA graphs are captured, by name and input shapes (a CPU-only
  profiler around ``capture_model``): the custom ops and ATen calls inside the graphs, at every captured size;
- ``engine.rank<R>.json``, the runner's configuration (model, cache, scheduler, parallel and compilation) and versions.

Recording never raises into vLLM: an error goes to ``errors.rank<R>.log`` and the step runs on.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

#: the model runners' modules, V1's and V2's
RUNNERS = ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner")
#: V2 dispatches through this module function, V1 through this runner method
V2_DISPATCH = "dispatch_cg_and_sync_dp"
V1_DISPATCH = "_determine_batch_execution_and_padding"
#: a batch descriptor's fields a row keeps, when the descriptor has them
DESC_FIELDS = ("cg_mode", "num_tokens", "num_reqs", "uniform", "uniform_token_count", "max_query_len", "num_ubatches")

_INSTALLED: list[Any] = []
_RECORDER: list[Recorder | None] = []


def config(environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """The plugin's settings, or None when ROUTECAP_OUT is unset."""
    from verity_config.settings import of

    s = of(__file__)
    out = s.get("ROUTECAP_OUT", environ)
    if not out:
        return None
    names = ("RANKS", "PER_CLASS", "BUDGET", "DUTY", "WARM", "M_EDGES", "CAPTURE_OPS")
    return {"out": out, **{n.lower(): s.get(f"ROUTECAP_{n}", environ) for n in names}}


# ------------------------------------------------------------------------------------------------------ a step's class

def bucket(n: int, edges: Iterable[int]) -> int:
    """The smallest edge at least ``n``; past the last, the next power of two."""
    for e in edges:
        if n <= e:
            return e
    return 1 << max(0, n - 1).bit_length()


def sampling_kind(sp: Any) -> tuple[str, ...]:
    """A request's sampling, as the sampler routes it: greedy, random or seeded, and which filters and extras it applies."""
    if sp is None:
        return ("none",)
    t = float(getattr(sp, "temperature", 1.0) or 0.0)
    kind = ["greedy" if t < 1e-5 else ("seeded" if getattr(sp, "seed", None) is not None else "random")]
    if kind[0] != "greedy":
        if int(getattr(sp, "top_k", 0) or 0) > 0:
            kind.append("top_k")
        if float(getattr(sp, "top_p", 1.0) or 1.0) < 1.0:
            kind.append("top_p")
        if float(getattr(sp, "min_p", 0.0) or 0.0) > 0.0:
            kind.append("min_p")
    if any(float(getattr(sp, a, d) or d) != d for a, d in (("presence_penalty", 0.0), ("frequency_penalty", 0.0),
                                                           ("repetition_penalty", 1.0))):
        kind.append("penalties")
    if getattr(sp, "logprobs", None) is not None:
        kind.append("logprobs")
    return tuple(kind)


class Requests:
    """What a step's scheduler output leaves implicit about its running requests (each one's prompt length and sampling
    kind, given once when it is first scheduled), and the step's row (`row`)."""

    def __init__(self) -> None:
        self.prompt: dict[str, int] = {}
        self.kind: dict[str, tuple[str, ...]] = {}

    def row(self, so: Any, topk: int | None) -> dict[str, Any]:
        computed: dict[str, int] = {}
        hit = 0
        for r in so.scheduled_new_reqs:
            ids = r.prompt_token_ids if r.prompt_token_ids is not None else (getattr(r, "prefill_token_ids", None) or [])
            self.prompt[r.req_id] = len(ids)
            self.kind[r.req_id] = sampling_kind(r.sampling_params)
            computed[r.req_id] = int(r.num_computed_tokens)
            hit += int(r.num_computed_tokens)
        c = so.scheduled_cached_reqs
        computed.update(zip(c.req_ids, map(int, c.num_computed_tokens)))
        prefill, ctx, kinds = [], [], Counter()
        for rid, q in so.num_scheduled_tokens.items():
            done, p = computed.get(rid, 0), self.prompt.get(rid)
            if (p is not None and done < p) or (p is None and q > 1):
                prefill.append([int(q), done, p])
            else:
                ctx.append(done + int(q))
            kinds.update(self.kind.get(rid, ("unknown",)))
        for rid in so.finished_req_ids:
            self.prompt.pop(rid, None)
            self.kind.pop(rid, None)
        longest = max((q + done for q, done, _ in prefill), default=0)
        return {"tokens": int(so.total_num_scheduled_tokens), "reqs": len(so.num_scheduled_tokens),
                "new": len(so.scheduled_new_reqs), "prefix_hit_tokens": hit, "prefill": prefill,
                "decode": len(ctx), "decode_ctx_max": max(ctx, default=0), "decode_ctx_sum": sum(ctx),
                "prefill_seq_max": longest,
                "attention": None if not prefill else ("dense" if topk is not None and longest <= topk else "sparse"),
                "spec_tokens": sum(len(v) for v in (getattr(so, "scheduled_spec_decode_tokens", None) or {}).values()),
                "sampling": dict(sorted(kinds.items()))}


def key(row: Mapping[str, Any], edges: Iterable[int]) -> str:
    """A step's class: its phase mix, token and request buckets, the prefills' attention route, whether a new request hit
    the prefix cache, and the sampler's mix."""
    phase = "decode" if not row["prefill"] else ("prefill" if not row["decode"] else "mixed")
    s = row["sampling"]
    greedy, n = s.get("greedy", 0), row["reqs"]
    mix = "greedy" if greedy == n else ("sampled" if greedy == 0 else "mixed")
    extras = "+".join(k for k in ("top_k", "top_p", "min_p", "penalties", "logprobs", "seeded") if s.get(k))
    parts = [phase, f"M{bucket(row['tokens'], edges)}", f"R{bucket(n, (1, 2, 4, 8, 16, 32, 64, 128, 256))}",
             row["attention"] or "-", "hit" if row["prefix_hit_tokens"] else "cold", mix + (f"+{extras}" if extras else "")]
    return "/".join(parts)


class Chooser:
    """Which steps a rank profiles: the first ``per_class`` of each class, at most ``budget`` in all, and, past the first
    ``warm``, only while profiling has held the rank for at most ``duty`` of the time since its first step."""

    def __init__(self, per_class: int, budget: int, duty: float, warm: int) -> None:
        self.per_class, self.budget, self.duty, self.warm = per_class, budget, duty, warm
        self.seen: Counter[str] = Counter()
        self.taken = 0
        self.busy = 0.0
        self.t0: float | None = None

    def want(self, k: str, now: float) -> bool:
        if self.t0 is None:
            self.t0 = now
        if self.seen[k] >= self.per_class or self.taken >= self.budget:
            return False
        return self.taken < self.warm or self.busy <= self.duty * (now - self.t0)

    def took(self, k: str, seconds: float) -> None:
        self.seen[k] += 1
        self.taken += 1
        self.busy += seconds


# ---------------------------------------------------------------------------------------------------------- recording

def _plain(x: Any) -> Any:
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    name = getattr(x, "name", None)
    return name if isinstance(name, str) else str(x)


def describe(desc: Any, mode: Any = None) -> dict[str, Any]:
    """A batch descriptor's fields (and V1's separate runtime mode) as plain values."""
    d = {f: _plain(getattr(desc, f)) for f in DESC_FIELDS if hasattr(desc, f)}
    if mode is not None:
        d["cg_mode"] = _plain(mode)
    return d


class Recorder:
    """One recording rank's files and its open step."""

    def __init__(self, cfg: Mapping[str, Any], rank: int, topk: int | None, torch: Any = None) -> None:
        self.cfg, self.rank, self.topk, self.torch = cfg, rank, topk, torch
        self.out = Path(cfg["out"])
        self.traces = self.out / "kineto" / f"rank{rank}"
        self.traces.mkdir(parents=True, exist_ok=True)
        self.steps = (self.out / f"steps.rank{rank}.jsonl").open("a", buffering=1)
        self.requests = Requests()
        self.chooser = Chooser(cfg["per_class"], cfg["budget"], cfg["duty"], cfg["warm"])
        self.n = 0
        self.open: dict[str, Any] | None = None
        self.profiler: Any = None
        self.t0 = time.monotonic()
        self.t_prof = 0.0
        self.errors = 0

    def error(self, where: str) -> None:
        self.errors += 1
        if self.errors <= 50:
            with (self.out / f"errors.rank{self.rank}.log").open("a") as f:
                f.write(f"== {where} step {self.n}\n{traceback.format_exc()}\n")

    def begin(self, so: Any) -> None:
        if self.open is not None:
            self.end()
        if not so.total_num_scheduled_tokens:
            return
        row = self.requests.row(so, self.topk)
        now = time.monotonic()
        row["step"], row["t"] = self.n, round(now - self.t0, 4)
        row["class"] = k = key(row, self.cfg["m_edges"])
        self.open = row
        if self.torch is not None and self.chooser.want(k, now):
            self.t_prof = time.monotonic()
            self.torch.cuda.synchronize()
            P = self.torch.profiler
            self.profiler = P.profile(activities=[P.ProfilerActivity.CPU, P.ProfilerActivity.CUDA], record_shapes=True)
            self.profiler.start()

    def dispatched(self, desc: Any, mode: Any = None) -> None:
        if self.open is not None:
            self.open["cg"] = describe(desc, mode)

    def end(self, export: bool = True) -> None:
        row, self.open = self.open, None
        if row is None:
            return
        self.n += 1
        if self.profiler is not None:
            prof, self.profiler = self.profiler, None
            try:
                self.torch.cuda.synchronize()
                prof.stop()
                if export:
                    name = f"step{row['step']:06d}.json.gz"
                    prof.export_chrome_trace(str(self.traces / name))
                    row["profiled"] = name
            except Exception:  # noqa: BLE001
                self.error("profile")
            spent = time.monotonic() - self.t_prof
            row["profile_s"] = round(spent, 3)
            self.chooser.took(row["class"], spent)
        if not export:
            row["failed"] = True
        self.steps.write(json.dumps(row, separators=(",", ":")) + "\n")


def _rank(torch: Any) -> int:
    d = torch.distributed
    return d.get_rank() if d.is_available() and d.is_initialized() else 0


def _topk(runner: Any) -> int | None:
    mc = getattr(runner, "model_config", None)
    hf = getattr(mc, "hf_text_config", None) or getattr(mc, "hf_config", None)
    v = getattr(hf, "index_topk", None)
    return int(v) if v is not None else None


def engine_record(runner: Any, rank: int, topk: int | None) -> dict[str, Any]:
    """The runner's configuration and the stack's versions."""
    import torch
    import vllm

    vc = getattr(runner, "vllm_config", None)
    sections = ("model_config", "cache_config", "scheduler_config", "parallel_config", "compilation_config",
                "attention_config", "kernel_config", "speculative_config")
    return {"rank": rank, "runner": f"{type(runner).__module__}.{type(runner).__qualname__}", "vllm": vllm.__version__,
            "torch": torch.__version__, "cuda": torch.version.cuda, "index_topk": topk,
            "model_class": type(getattr(runner, "model", None)).__qualname__,
            **{s: repr(getattr(vc, s, None)) for s in sections}}


def recorder(runner: Any, cfg: Mapping[str, Any]) -> Recorder | None:
    """This process's recorder, made at its runner's first step or graph capture; None off the recording ranks."""
    if _RECORDER:
        return _RECORDER[0]
    import torch

    rank = _rank(torch)
    rec = None
    if rank in cfg["ranks"]:
        topk = _topk(runner)
        rec = Recorder(cfg, rank, topk, torch)
        try:
            (rec.out / f"engine.rank{rank}.json").write_text(json.dumps(engine_record(runner, rank, topk), indent=1))
        except Exception:  # noqa: BLE001
            rec.error("engine record")
    _RECORDER.append(rec)
    return rec


# ---------------------------------------------------------------------------------------------------------- the patches

def _execute(execute: Callable[..., Any], cfg: Mapping[str, Any]) -> Callable[..., Any]:
    def execute_model(self: Any, scheduler_output: Any, *a: Any, **kw: Any) -> Any:
        rec = None if kw.get("dummy_run") else recorder(self, cfg)
        if rec is not None:
            try:
                rec.begin(scheduler_output)
            except Exception:  # noqa: BLE001
                rec.error("begin")
                rec.open, rec.profiler = None, None
        try:
            out = execute(self, scheduler_output, *a, **kw)
        except BaseException:
            if rec is not None and rec.open is not None:
                try:
                    rec.end(export=False)
                except Exception:  # noqa: BLE001
                    rec.error("end after a failed step")
            raise
        if rec is not None and out is not None and rec.open is not None:
            try:
                rec.end()
            except Exception:  # noqa: BLE001
                rec.error("end")
        return out
    return execute_model


def _sample(sample: Callable[..., Any]) -> Callable[..., Any]:
    def sample_tokens(self: Any, *a: Any, **kw: Any) -> Any:
        out = sample(self, *a, **kw)
        rec = _RECORDER[0] if _RECORDER else None
        if rec is not None and rec.open is not None:
            try:
                rec.end()
            except Exception:  # noqa: BLE001
                rec.error("end")
        return out
    return sample_tokens


def _dispatch_v1(dispatch: Callable[..., Any]) -> Callable[..., Any]:
    def determine(self: Any, *a: Any, **kw: Any) -> Any:
        res = dispatch(self, *a, **kw)
        rec = _RECORDER[0] if _RECORDER else None
        if rec is not None and rec.open is not None:
            try:
                rec.dispatched(res[1], res[0])
            except Exception:  # noqa: BLE001
                rec.error("dispatch")
        return res
    return determine


def _dispatch_v2(dispatch: Callable[..., Any]) -> Callable[..., Any]:
    def dispatch_cg(*a: Any, **kw: Any) -> Any:
        res = dispatch(*a, **kw)
        rec = _RECORDER[0] if _RECORDER else None
        if rec is not None and rec.open is not None:
            try:
                rec.dispatched(res[0])
            except Exception:  # noqa: BLE001
                rec.error("dispatch")
        return res
    return dispatch_cg


def capture_ops(prof: Any) -> list[dict[str, Any]]:
    """A profiler's operators by name and input shapes, most called first."""
    rows = [{"op": e.key, "shapes": e.input_shapes, "count": e.count} for e in prof.key_averages(group_by_input_shape=True)]
    return sorted(rows, key=lambda r: (-r["count"], r["op"]))


def _capture(capture: Callable[..., Any], cfg: Mapping[str, Any]) -> Callable[..., Any]:
    def capture_model(self: Any, *a: Any, **kw: Any) -> Any:
        rec = recorder(self, cfg) if cfg["capture_ops"] and not kw.get("profile_only") else None
        prof = None
        if rec is not None:
            try:
                P = rec.torch.profiler
                prof = P.profile(activities=[P.ProfilerActivity.CPU], record_shapes=True)
                prof.start()
            except Exception:  # noqa: BLE001
                rec.error("capture profiler start")
                prof = None
        t = time.monotonic()
        try:
            return capture(self, *a, **kw)
        finally:
            if prof is not None:
                try:
                    prof.stop()
                    ops = {"rank": rec.rank, "seconds": round(time.monotonic() - t, 1), "ops": capture_ops(prof)}
                    (rec.out / f"capture_ops.rank{rec.rank}.json").write_text(json.dumps(ops))
                except Exception:  # noqa: BLE001
                    rec.error("capture ops")
    return capture_model


def patch_runner(module: Any, cfg: Mapping[str, Any], hooks: Any) -> None:
    """Wrap one runner module's GPUModelRunner (and V2's dispatch function)."""
    runner = getattr(module, "GPUModelRunner", None)
    if runner is None:
        return
    hooks.wrap(runner, "execute_model", lambda f: _execute(f, cfg))
    if hasattr(runner, "sample_tokens"):
        hooks.wrap(runner, "sample_tokens", _sample)
    if hasattr(runner, "capture_model"):
        hooks.wrap(runner, "capture_model", lambda f: _capture(f, cfg))
    if hasattr(runner, V1_DISPATCH):
        hooks.wrap(runner, V1_DISPATCH, _dispatch_v1)
    if hasattr(module, V2_DISPATCH):
        hooks.wrap(module, V2_DISPATCH, _dispatch_v2)


def register() -> None:
    """Patch each model runner this process imports, when ROUTECAP_OUT is set.  Idempotent."""
    cfg = config()
    if cfg is None or _INSTALLED:
        return
    from verity_vllm.engine.hooks import Hooks, OnImport

    hooks = Hooks(tag="route-capture")
    Path(cfg["out"]).mkdir(parents=True, exist_ok=True)
    for name in RUNNERS:
        if name in sys.modules:
            patch_runner(sys.modules[name], cfg, hooks)
        else:
            sys.meta_path.insert(0, OnImport(name, lambda m: patch_runner(m, cfg, hooks)))
    _INSTALLED.append(hooks)
