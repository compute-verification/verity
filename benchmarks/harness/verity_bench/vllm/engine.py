"""One arm's vLLM engine (`_VllmEngine`), in a spawned process of its own (`_worker`), which the runner drives through
`_ProcessEngine` and builds with `ProcessEngineFactory`.

torch and vLLM are imported inside the engine's methods: the harness imports without them, and the engine pins vLLM's
environment (`verity_vllm.engine.env.apply_engine`) before either is imported."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import statistics
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from verity_vllm import LLM
from verity_vllm.engine import handle

from verity_bench import settings
from verity_bench.adapter import (
    INDUCTOR_DIR_MARK,
    Hooks,
    apply_compile_fresh,
    apply_engine,
    apply_target,
    h1_run,
    int8_exact_of,
    live,
    own_inductor_dir,
    pearl_c_device_run,
    register_int8_parts,
    run_requests_arrivals,
)
from verity_bench.vllm.arms import (
    H1,
    Arm,
    Target,
    build_bound,
    h1_config,
    is_h1,
    plugin,
    protocols_config,
    window_steps,
)
from verity_bench.vllm.presets import GREEDY, Preset, SweepPoint
from verity_bench.vllm.runner import (
    BenchEngine,
    ResidentCapacityError,
    _numeric_leaves,
    _pearl_c_gate,
    _token_digest,
    h1_gate,
    h1_summary,
)
from verity_bench.vllm.timing import (
    HostProbe,
    decode_time_per_token,
    inter_token_latencies,
    percentile,
    run_trace,
)

# `TargetProfile.int8_exact` "all" as vLLM takes it (`TargetProfile.engine_args`, `INT8_EXACT_SCOPES`)
INT8_EXACT_SCOPE = "all"
INT8_EXACT_PARTS = ("linears", "attention")
INT8_EXACT_QUANTIZATION, INT8_EXACT_ATTENTION_BACKEND = "verity_int8", "CUSTOM"
# S of the H1 window `verity_vllm.LLM` opens at the build, before the engine knows its points: the first run's
# `reinstall` closes it unserved and opens that run's own
H1_OPENING_STEPS = 1


def _callback(obj: Any, name: str, target: Target, default: Any) -> Any:
    fn = obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)
    return default if not callable(fn) else fn(target)


def _delta(after: Any, before: Any) -> dict[str, float]:
    a, b = _numeric_leaves(after), _numeric_leaves(before)
    return {k: v - b.get(k, 0.0) for k, v in a.items()}


def _allocator_counts() -> dict[str, float]:
    """The caching allocator's device allocations and frees, its out-of-memory retries (each frees every cached block)
    and its synchronizations of every stream: a run that needs any of them waits on the driver."""

    import torch

    stats = torch.cuda.memory_stats()
    keys = ("num_device_alloc", "num_device_free", "num_alloc_retries", "num_sync_all_streams")
    return {k: stats.get(k, 0) for k in keys}


class _DirectLLM:
    """The subset of ``verity_vllm.LLM`` needed for a content-bound local checkpoint."""

    def __init__(self, llm: Any, engine_overrides: Mapping[str, Any]) -> None:
        runner = handle.model_runner(llm)
        self.vllm, self.model = llm, runner.get_model()
        self.engine_record = {"engine_overrides": dict(engine_overrides)}
        self.protocols = None

    def close(self) -> None:
        # The worker process exits immediately after close and releases vLLM's
        # process-owned CUDA state.
        pass


PASS_COUNTS = ("units", "tiles", "screened_rows", "failing_rows", "unsure_rows")


def _pearl_c_run(llm: Any) -> Any | None:
    """The arm's Pearl-C device run, the owner of its option's ``end_pass``; None when the arm runs none. The adapter ends
    a pass each time the model runner's ``execute_model`` returns (`pouw.install`), so the harness only reads the passes
    (`_pass_totals`) and gates the run (`_pearl_c_gate`)."""

    option = llm.protocols.get("pouw") if llm.protocols is not None else None
    run = getattr(getattr(option, "end_pass", None), "__self__", None)
    return run if isinstance(run, pearl_c_device_run()) else None


def _pass_totals(run: Any) -> dict[str, int]:
    """The running totals of the passes the run has ended (``details["passes"]``): `PASS_COUNTS` and ``passes``."""

    passes = run.details["passes"]
    return {"passes": len(passes), **{key: sum(int(p.get(key) or 0) for p in passes) for key in PASS_COUNTS}}


def _kv_cache_tokens(llm: Any) -> int | None:
    """The engine's actual KV pool in tokens (in-process V1 engine), None when the engine does not expose it."""

    engine = getattr(getattr(llm, "vllm", None), "llm_engine", None)
    cache = getattr(getattr(engine, "vllm_config", None), "cache_config", None)
    blocks, block_size = getattr(cache, "num_gpu_blocks", None), getattr(cache, "block_size", None)
    if blocks is None or block_size is None:
        return None
    return int(blocks) * int(block_size)


def engine_args(preset: Preset, precision: str, *, resident: bool, eager: bool) -> dict[str, Any]:
    """The engine options every arm of a preset shares, before an arm's own overrides.

    With ``serving_defaults`` the engine compiles as vLLM serves (torch.compile, its CUDA-graph mode and capture sizes,
    the execution label ``compiled``, or ``eager`` for an eager engine); otherwise a graph engine replays decode-only CUDA
    graphs at the preset's ``capture_sizes``. Engines of one group
    share one exact KV pool, ``resident_kv_cache_memory_bytes`` or ``block_kv_cache_memory_bytes``: vLLM schedules by
    free KV pages, and an arm whose install leaves a different amount of free memory would otherwise get a different
    pool, a different schedule, and (through batch-composition-dependent kernels) different tokens than its control.
    Without a pin the kv-pool gate catches unequal pools. ``attention_backend`` and ``attention_config`` pass to vLLM as
    set, for a preset whose Build's row declares them (the B200's FlashAttention 2)."""

    settings = preset.engine_settings
    engine: dict[str, Any] = {
        "tensor_parallel_size": preset.gpu_count,
        "max_num_seqs": int(settings.get("max_num_seqs", 128)),
        "max_model_len": int(settings["max_model_len"]),
        "gpu_memory_utilization": float(settings[
            "resident_gpu_memory_utilization" if resident else "block_gpu_memory_utilization"
        ]),
        "enforce_eager": eager,
    }
    if settings.get("max_num_batched_tokens") is not None:
        engine["max_num_batched_tokens"] = int(settings["max_num_batched_tokens"])
    if precision == "fp8":
        engine["quantization"] = "fp8"
    elif precision == "int8_exact":
        if not eager:
            raise ValueError("int8_exact runs eagerly only, on its own attention backend: an eager spec")
        engine["quantization"], engine["attention_backend"] = INT8_EXACT_QUANTIZATION, INT8_EXACT_ATTENTION_BACKEND
    elif precision != "bfloat16":
        engine["dtype"] = precision
    if settings.get("serving_defaults"):
        engine["execution"] = "eager" if eager else "compiled"
    elif not eager:
        engine["compilation_config"] = {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": list(settings["capture_sizes"]),
        }
    kv_pin = settings.get("resident_kv_cache_memory_bytes" if resident else "block_kv_cache_memory_bytes")
    if kv_pin is not None:
        engine["kv_cache_memory_bytes"] = int(kv_pin)
    if settings.get("attention_backend") is not None:
        engine["attention_backend"] = str(settings["attention_backend"])
    if settings.get("attention_config") is not None:
        engine["attention_config"] = dict(settings["attention_config"])
    return engine


class _VllmEngine:
    """One worker process's live vLLM engine."""

    def __init__(self, arm: Arm, preset: Preset, resident: bool, eager: bool) -> None:
        if preset.revision is None and preset.model_digest is None:
            raise ValueError(f"preset {preset.id} has no pinned checkpoint revision")
        if preset.engine_settings.get("serving_defaults"):
            apply_compile_fresh()
        apply_engine(None)
        import torch
        import vllm

        self.arm, self.obj, self.installed, self.pearl_c, self.h1 = arm, plugin(arm), None, None, None
        self.target = Target(arm, None, None, Hooks(tag=f"bench:{arm.id}"))
        if self.obj is not None:
            before = {id(e) for e in live()}
            self.installed = _callback(self.obj, "install", self.target, None)
            added = [e for e in live() if id(e) not in before]
            owned = {id(e) for e in self.target.hooks.entries}
            foreign = [e for e in added if id(e) not in owned or e.tag != self.target.hooks.tag]
            if foreign:
                self.target.hooks.uninstall()
                raise RuntimeError(f"experimental arm {arm.id!r} patched outside target.hooks")

        kernel_digest = arm.kernel_digest
        precision = arm.precision or preset.precision
        engine = engine_args(preset, precision, resident=resident, eager=eager)
        if precision == "int8_exact":
            # `engine.build.register_int8_parts` registers a TargetProfile's parts, and verity_vllm.LLM builds with none
            register_int8_parts(INT8_EXACT_PARTS)
        overrides = _callback(self.obj, "engine_overrides", self.target, {}) if self.obj is not None else {}
        engine.update(overrides or {})
        if preset.model_digest is not None:
            if arm.protocols:
                raise ValueError("a local checkpoint override supports stock and experimental arms, not protocol-option arms")
            self.llm = _DirectLLM(
                vllm.LLM(
                    model=preset.model,
                    enable_prefix_caching=False,
                    seed=0,
                    **{k: v for k, v in engine.items() if k != "execution"},
                ),
                {k: engine[k] for k in ("dtype", "quantization", "compilation_config", "kv_cache_memory_bytes")
                 if engine.get(k) is not None},
            )
        else:
            config: dict[str, Any] = {}
            if arm.protocols:
                config = protocols_config(arm, settings.environ())
            if is_h1(arm):
                rows = config.get("pouw", {}).get("rows")
                if rows != engine.get("max_num_batched_tokens"):
                    raise ValueError(f"{arm.id}: the pouw setting 'rows' (T, {rows}) is the engine's "
                                     f"max_num_batched_tokens ({engine.get('max_num_batched_tokens')})")
                self.h1 = {"config": config, "rows": rows, "windows": []}
                config = h1_config(config, H1_OPENING_STEPS)
            if arm.protocols:
                engine["protocols"] = ",".join(arm.protocols)
                if config:
                    engine["protocols_config"] = config
            self.llm = LLM(preset.model, revision=preset.revision, **engine)
            if arm.protocols and self.h1 is None:
                self.pearl_c = _pearl_c_run(self.llm)
                cubin = config.get("pouw", {}).get("cubin")
                if cubin:
                    kernel_digest = hashlib.sha256(Path(cubin).read_bytes()).hexdigest()
        self.target.engine, self.target.model = self.llm, self.llm.model
        if precision == "int8_exact":
            got = int8_exact_of(self.llm.model)
            if got != INT8_EXACT_SCOPE:
                raise RuntimeError(f"{arm.id}: an int8_exact engine whose layers run int8 scope {got!r}, not "
                                   f"{INT8_EXACT_SCOPE!r}")
        ready = getattr(self.installed, "ready", None)
        if callable(ready):
            ready(self.target)
        self.identity = {
            "vllm_version": vllm.__version__,
            "kv_cache_tokens": _kv_cache_tokens(self.llm),
            "kernel_digest": (
                _callback(self.obj, "kernel_digest", self.target, None)
                if self.obj is not None and callable(
                    self.obj.get("kernel_digest") if isinstance(self.obj, Mapping) else getattr(self.obj, "kernel_digest", None)
                )
                else kernel_digest
            ),
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": f"{torch.__version__} cuda {torch.version.cuda}",
            "engine_overrides": dict(self.llm.engine_record.get("engine_overrides") or {}),
        }
        self.ready_at = time.perf_counter()

    def _requests(self, point: SweepPoint) -> list[Any]:
        vocab = int(self.llm.vllm.llm_engine.vllm_config.model_config.hf_config.vocab_size)
        seed = int.from_bytes(hashlib.sha256(point.id.encode()).digest()[:8], "big")
        return [
            SimpleNamespace(
                request_id=f"{point.id}-{i}",
                prompt_token_ids=[
                    3 + (seed + i * 1_000_003 + j * 97_409) % (vocab - 4)
                    for j in range(shape.prompt_tokens)
                ],
                max_tokens=shape.output_tokens,
                arrive_step=shape.arrive_step,
                seed=i,
            )
            for i, shape in enumerate(point.requests())
        ]

    def _spans(self) -> Any:
        if self.obj is not None:
            got = _callback(self.obj, "spans", self.target, None)
            if got is not None:
                return got
        info: Mapping[str, Any] = {} if self.llm.protocols is None else self.llm.protocols.info()
        return info if self.pearl_c is None else {**info, "pearl_c": _pass_totals(self.pearl_c)}

    def _run(self, point: SweepPoint, phase: str) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
        """One run; an H1 arm's in its own window, opened before the clock starts (`verity_vllm.LLM.reinstall`, S the run's
        step bound, `window_steps`) and closed after it stops."""

        if self.h1 is None:
            return self._serve(point, phase)
        llm = cast(LLM, self.llm)  # an H1 arm's: a local checkpoint serves no protocol arm
        start = time.perf_counter()
        llm.reinstall(h1_config(self.h1["config"], window_steps(point, self.h1["rows"])))
        install_s = time.perf_counter() - start
        run = h1_run(H1)
        try:
            tokens, out = self._serve(point, phase)
        finally:
            start = time.perf_counter()
            llm.close()
            window = {"point": point.id, "phase": phase, **h1_summary(run), "install_s": round(install_s, 3),
                      "close_s": round(time.perf_counter() - start, 3)}
            self.h1["windows"].append(window)
        if phase == "measure":
            out = {**out, "extras": {**out["extras"], "h1_window": window}}
        return tokens, out

    def _serve(self, point: SweepPoint, phase: str) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
        import torch
        from vllm import SamplingParams

        measured = phase == "measure"
        requests = self._requests(point)
        params = lambda r: SamplingParams(max_tokens=r.max_tokens, ignore_eos=True,  # noqa: E731
                                          detokenize=False, seed=r.seed, **GREEDY)
        returns: list[tuple[float, dict[str, int]]] = []

        def on_step(outs: list[Any]) -> None:
            returns.append((time.perf_counter(), {o.request_id: len(o.outputs[0].token_ids) for o in outs if o.outputs}))

        before = self._spans()
        if measured:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.target.phase, self.target.point = phase, point.id
        self.target.tokens = sum(len(r.prompt_token_ids) + r.max_tokens for r in requests)
        if self.obj is not None:
            _callback(self.obj, "begin_run", self.target, None)
        probe = HostProbe(_allocator_counts)
        start = time.perf_counter()
        finished, timing = run_requests_arrivals(
            self.llm.vllm,
            requests,
            params,
            on_step=on_step,
            request_timing=True,
            step_timing=True,
        )
        returned = time.perf_counter()
        if self.obj is not None:
            _callback(self.obj, "end_run", self.target, None)
        torch.cuda.synchronize()
        end = time.perf_counter()
        host = probe.close()
        wall, device_tail = end - start, end - returned
        tokens = {rid: tuple(int(t) for t in out.outputs[0].token_ids) for rid, out in finished.items()}
        ttft = [float(r["time_to_first_token_s"]) for r in timing["requests"].values()]
        if not measured:
            return tokens, {"wall_s": round(wall, 6), "time_to_first_token_s": round(statistics.fmean(ttft), 6),
                            "since_ready_s": round(start - self.ready_at, 6), "host": host}
        generated = sum(map(len, tokens.values()))
        after = self._spans()
        extras = _callback(self.obj, "extras", self.target, {}) if self.obj is not None else {}
        metrics: dict[str, float | None] = {
            "time_per_output_token_s": wall / generated,
            "time_to_first_token_s": statistics.fmean(ttft),
            "tokens_per_s": generated / wall,
            "peak_gpu_memory_bytes": float(torch.cuda.max_memory_allocated()),
        }
        lag = int(timing.get("output_lag_steps") or 0)
        decode = decode_time_per_token(timing.get("steps") or [], output_lag=lag, device_tail_s=device_tail,
                                       generated=generated)
        if decode is not None:
            metrics["decode_time_per_token_s"] = decode
        latencies = inter_token_latencies(returns)
        if latencies:
            metrics["inter_token_latency_p50_s"] = percentile(latencies, 50)
            metrics["inter_token_latency_p99_s"] = percentile(latencies, 99)
        return tokens, {
            "metrics": metrics,
            "token_digest": _token_digest(tokens),
            "host_timing": {"output_lag_steps": lag, "device_tail_s": device_tail},
            "host": host,
            "trace": run_trace(timing, start, self.ready_at),
            "spans": _delta(after, before),
            "extras": extras or {},
            "generated_tokens": generated,
            "wall_s": wall,
        }

    def gate(self, point: SweepPoint) -> Mapping[str, tuple[int, ...]]:
        return self._run(point, "gate")[0]

    def custom_gate(self) -> Mapping[str, Any]:
        if self.h1 is not None:
            return h1_gate(self.h1["windows"])
        if self.obj is None:
            return {"passed": True, "provided": False} if self.pearl_c is None else _pearl_c_gate(self.pearl_c)
        got = _callback(self.obj, "gate", self.target, None)
        if got is None:
            return {"passed": True, "provided": False}
        if isinstance(got, bool):
            return {"passed": got, "provided": True}
        return dict(got)

    def warmup(self, point: SweepPoint) -> Mapping[str, Any]:
        return self._run(point, "warmup")[1]

    def measure(self, point: SweepPoint) -> Mapping[str, Any]:
        return self._run(point, "measure")[1]

    def close(self) -> None:
        try:
            close = getattr(self.installed, "close", None)
            if callable(close):
                close()
        finally:
            try:
                self.target.hooks.uninstall()
            finally:
                self.llm.close()


def _capacity_error(message: str) -> bool:
    text = message.lower()
    return any(s in text for s in ("out of memory", "cuda oom", "free memory", "memory profiling", "no available memory"))


def _worker(conn: Any, arm: Arm, preset: Preset, resident: bool, eager: bool) -> None:
    engine: _VllmEngine | None = None
    try:
        engine = _VllmEngine(arm, preset, resident, eager)
        conn.send({"ready": dict(engine.identity)})
        while True:
            name, arg = conn.recv()
            if name == "close":
                conn.send({"value": None})
                break
            value = getattr(engine, name)(arg) if arg is not None else getattr(engine, name)()
            conn.send({"value": value})
    except Exception as e:  # noqa: BLE001
        conn.send({"error": f"{type(e).__name__}: {e}", "capacity": _capacity_error(str(e)),
                   "trace": traceback.format_exc()[-4000:]})
    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
        conn.close()


@dataclass
class _ProcessEngine:
    process: Any
    conn: Any
    identity: Mapping[str, Any]
    closed: bool = False

    def _ask(self, name: str, arg: Any = None) -> Any:
        try:
            self.conn.send((name, arg))
            response = self.conn.recv()
        except (EOFError, ConnectionResetError, BrokenPipeError):
            raise RuntimeError(self._death_report(name)) from None
        if "error" in response:
            raise RuntimeError(response["error"] + "\n" + response.get("trace", ""))
        return response["value"]

    def _death_report(self, doing: str) -> str:
        """The engine process died under ``doing``: its own last error when it managed to send one, else its exit code,
        so a kill during a gate surfaces as what it was instead of a connection error."""
        final, unread = None, ""
        try:
            if self.conn.poll(0):
                final = self.conn.recv()
        except (EOFError, ConnectionResetError, OSError) as e:
            unread = f"; its last message could not be read ({type(e).__name__})"
        self.process.join()
        if isinstance(final, dict) and "error" in final:
            return f"benchmark engine process died during {doing}: " + final["error"] + "\n" + final.get("trace", "")
        code = self.process.exitcode
        hint = " (SIGKILL: likely out of memory; its own last error is on the job's stderr)" if code == -9 else ""
        return f"benchmark engine process died during {doing} with exit code {code}{hint}{unread}"

    def gate(self, point: SweepPoint) -> Mapping[str, tuple[int, ...]]:
        return self._ask("gate", point)

    def custom_gate(self) -> Mapping[str, Any]:
        return self._ask("custom_gate")

    def warmup(self, point: SweepPoint) -> Mapping[str, Any] | None:
        return self._ask("warmup", point)

    def measure(self, point: SweepPoint) -> Mapping[str, Any]:
        return self._ask("measure", point)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.is_alive():
                try:
                    self.conn.send(("close", None))
                    self.conn.recv()
                except (EOFError, ConnectionResetError, BrokenPipeError) as e:
                    # A worker that sent its error has left its loop and is closing its engine on its own; raising here
                    # would replace that error, which the caller is propagating, with this one.
                    print(f"verity-bench: the engine process left before the close handshake ({type(e).__name__}); "
                          "its exit code says how it ended", file=sys.stderr)
        finally:
            self.process.join()
            self.conn.close()
        if self.process.exitcode:
            raise RuntimeError(f"benchmark engine process exited with code {self.process.exitcode} while closing")


def arm_execution(arm: Arm, preset: Preset, eager: bool) -> tuple[Preset, bool]:
    """The preset and eagerness the arm's engine is built with: the spec's, or for a served variant
    (`arms.served_variant`) vLLM's serving defaults in the variant's execution (`engine_args`), on its own checkpoint
    when it names one."""

    if arm.execution is None:
        return preset, eager
    eager = arm.execution == "eager"
    settings = {**preset.engine_settings, "serving_defaults": True, "cuda_graphs": not eager, "enforce_eager": eager}
    if arm.model is not None:
        preset = replace(preset, model=arm.model, revision=arm.revision, case=None, model_digest=None)
    return replace(preset, engine_settings=settings), eager


class ProcessEngineFactory:
    """Spawn-isolated engines, allowing several arms to remain resident on one GPU."""

    def __init__(self, *, eager: bool = False) -> None:
        self.eager = eager
        self.context = mp.get_context("spawn")

    def create(self, arm: Arm, preset: Preset, *, resident: bool) -> BenchEngine:
        preset, eager = arm_execution(arm, preset, self.eager)
        if not eager:
            bound = build_bound(arm, preset.gpu_count)
            if bound:
                raise ValueError(
                    f"arm {arm.id!r} binds its served kernels to the row's Build ({', '.join(bound)}); CUDA graphs can "
                    "change the served kernels so they no longer match it: run with --eager (or eager = true in the "
                    "spec)")
        if resident and not arm.resident:
            raise ResidentCapacityError(f"arm {arm.id!r} requires an isolated engine lifetime")
        if preset.engine_settings.get("serving_defaults") and not own_inductor_dir():
            # Every arm's process compiles on its own, and the kernel configurations Inductor autotunes reach the numerics
            # (a reduction's tile size): one cache directory gives every arm the configurations the first one picked.
            apply_target({"TORCHINDUCTOR_CACHE_DIR": tempfile.mkdtemp(prefix="verity_bench_inductor_"), INDUCTOR_DIR_MARK: "1"})
        parent, child = self.context.Pipe()
        process = self.context.Process(target=_worker, args=(child, arm, preset, resident, eager))
        process.start()
        child.close()
        try:
            response = parent.recv()
        except (EOFError, ConnectionResetError):
            process.join()
            code = process.exitcode
            hint = " (SIGKILL: likely out of memory; its own last error is on the job's stderr)" if code == -9 else ""
            raise RuntimeError(f"benchmark engine {arm.id!r} exited during startup with code {code}{hint}") from None
        if "error" in response:
            process.join()
            error = response["error"] + "\n" + response.get("trace", "")
            if resident and response.get("capacity"):
                raise ResidentCapacityError(error)
            raise RuntimeError(error)
        return _ProcessEngine(process, parent, response["ready"])
