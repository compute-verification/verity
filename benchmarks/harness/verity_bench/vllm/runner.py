"""Gate, warm, interleave, measure, and aggregate vLLM benchmark arms."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import statistics
import subprocess
import tempfile
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol

from verity_vllm.engine.env import INDUCTOR_DIR_MARK

from verity_bench.vllm import result as result_format
from verity_bench.vllm.arms import Arm, Target, build_bound, plugin
from verity_bench.vllm.presets import SAMPLING, Preset, SweepPoint

METRICS = ("time_per_output_token_s", "time_to_first_token_s", "tokens_per_s", "peak_gpu_memory_bytes")
# Recorded for an arm at a point only when its every rep and the baseline's have it (the engine exposes its scheduler;
# the point's requests generate more than one token).
OPTIONAL_METRICS = ("decode_time_per_token_s", "inter_token_latency_p50_s", "inter_token_latency_p99_s")
ENGINE_IDENTITY = ("vllm_version", "torch_version", "gpu", "kv_cache_tokens", "kernel_digest", "engine_overrides")
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
       25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}
# A step's host time may carry device work its own forward didn't enqueue; past this share of the decode time the
# run's trailing device work is too large to attribute, and the decode metric is dropped.
DECODE_TAIL_SHARE = 0.01


def t95(df: int) -> float:
    """The two-sided 95% Student-t quantile, from the largest tabulated df at or below `df` (wider, never narrower)."""

    return T95[max(k for k in T95 if k <= df)]


class ResidentCapacityError(RuntimeError):
    """An engine could not join the resident set, so the runner should use blocks."""


class BenchEngine(Protocol):
    identity: Mapping[str, Any]

    def gate(self, point: SweepPoint) -> Mapping[str, tuple[int, ...]]: ...
    def custom_gate(self) -> Mapping[str, Any]: ...
    def warmup(self, point: SweepPoint) -> None: ...
    def measure(self, point: SweepPoint) -> Mapping[str, Any]: ...
    def close(self) -> None: ...


class EngineFactory(Protocol):
    def create(self, arm: Arm, preset: Preset, *, resident: bool) -> BenchEngine: ...


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "n": 0}
    mean = statistics.fmean(values)
    out: dict[str, Any] = {"mean": mean, "n": len(values)}
    if len(values) >= 2:
        half = t95(len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
        out["ci95"] = [mean - half, mean + half]
    return out


def _numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(value, Mapping):
        out: dict[str, float] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(_numeric_leaves(child, path))
        return out
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {prefix: float(value)}
    return {}


def _numeric_summaries(values: list[Any]) -> dict[str, dict[str, Any]]:
    leaves = [_numeric_leaves(v) for v in values]
    keys = sorted({k for d in leaves for k in d})
    return {key: _summary([d[key] for d in leaves if key in d]) for key in keys}


def _delta(after: Any, before: Any) -> dict[str, float]:
    a, b = _numeric_leaves(after), _numeric_leaves(before)
    return {k: v - b.get(k, 0.0) for k, v in a.items()}


def _token_digest(tokens: Mapping[str, tuple[int, ...]]) -> str:
    body = json.dumps({k: list(v) for k, v in sorted(tokens.items())}, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def _gate_passed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        return bool(value.get("passed", value.get("ok", False)))
    return False


def _first_divergence(
    got: Mapping[str, tuple[int, ...]], control: Mapping[str, tuple[int, ...]]
) -> dict[str, Any] | None:
    """The earliest request (by id) whose tokens differ from the control's, and where."""

    for rid in sorted(set(got) | set(control)):
        g, c = tuple(got.get(rid, ())), tuple(control.get(rid, ()))
        if g == c:
            continue
        at = next((i for i, (x, y) in enumerate(zip(g, c, strict=False)) if x != y), min(len(g), len(c)))
        return {
            "request": rid,
            "position": at,
            "token": int(g[at]) if at < len(g) else None,
            "control_token": int(c[at]) if at < len(c) else None,
        }
    return None


def _evaluate_gates(
    arms: tuple[Arm, ...],
    points: tuple[SweepPoint, ...],
    tokens: Mapping[tuple[str, str], Mapping[str, tuple[int, ...]]],
    custom: Mapping[str, Mapping[str, Any]],
    versions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for arm in arms:
        c = custom[arm.id]
        out.append({"arm": arm.id, "kind": "arm", "passed": _gate_passed(c), "detail": dict(c)})
        if arm.control:
            # An arm gated token-for-token against a control must see the very scheduler decisions the control saw:
            # vLLM admits and preempts by free KV blocks, and a step's batch composition reaches the numerics (attention
            # split counts, chunked-prefill GEMM shapes). Equal pools and a deterministic scheduler give equal schedules.
            mine = versions.get(arm.id, {}).get("kv_cache_tokens")
            theirs = versions.get(arm.control, {}).get("kv_cache_tokens")
            gate = {
                "arm": arm.id,
                "control": arm.control,
                "kind": "kv-pool",
                "passed": mine is not None and mine == theirs,
                "tokens": mine,
                "control_tokens": theirs,
            }
            if mine is None or theirs is None:
                gate["detail"] = "an engine did not report its KV pool, so the pools' equality is unchecked"
            elif mine != theirs:
                gate["detail"] = (
                    f"KV cache pools differ ({mine} vs {theirs} tokens), so vLLM schedules the arms differently "
                    "and their tokens can diverge; pin resident_kv_cache_memory_bytes and "
                    "block_kv_cache_memory_bytes in the preset's engine_settings so every arm gets the same pool"
                )
            out.append(gate)
        for point in points:
            got = tokens[(arm.id, point.id)]
            control = tokens.get((arm.control, point.id)) if arm.control else None
            passed = control is None or got == control
            out.append({
                "arm": arm.id,
                "control": arm.control,
                "point": point.id,
                "kind": "tokens",
                "passed": passed,
                "tokens": sum(map(len, got.values())),
                "digest": _token_digest(got),
                **({"control_digest": _token_digest(control)} if control is not None else {}),
                **(
                    {
                        "first_divergence": _first_divergence(got, control),
                        "divergent_requests": sum(
                            1 for rid in set(got) | set(control)
                            if tuple(got.get(rid, ())) != tuple(control.get(rid, ()))
                        ),
                    }
                    if not passed and control is not None
                    else {}
                ),
            })
    return out


def _timed_token_gates(
    arms: tuple[Arm, ...], points: tuple[SweepPoint, ...], samples: Mapping[tuple[str, str], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Every timed rep of an arm with a control must produce the control's tokens in that rep: the greedy gate runs
    before timing, so an arm that computed something else only when timed would pass it."""

    out = []
    for arm in arms:
        if not arm.control:
            continue
        for point in points:
            mine = {r["rep"]: r.get("token_digest") for r in samples[(arm.id, point.id)]}
            theirs = {r["rep"]: r.get("token_digest") for r in samples[(arm.control, point.id)]}
            bad = sorted(r for r in mine.keys() | theirs.keys() if mine.get(r) is None or mine.get(r) != theirs.get(r))
            out.append({"arm": arm.id, "control": arm.control, "point": point.id, "kind": "timed-tokens",
                        "passed": not bad, **({"reps": bad} if bad else {})})
    return out


def _engine_identity_gate(arm: Arm, cycle: int, gated: Mapping[str, Any], timed: Mapping[str, Any]) -> dict[str, Any]:
    """A block-mode arm is gated and timed by different engines; the timed one must be the gated one's twin."""

    differs = {k: [gated.get(k), timed.get(k)] for k in ENGINE_IDENTITY if gated.get(k) != timed.get(k)}
    return {"arm": arm.id, "cycle": cycle, "kind": "engine-identity", "passed": not differs,
            **({"differs": differs} if differs else {})}


def _collect_gates(
    engines: Mapping[str, BenchEngine], arms: tuple[Arm, ...], points: tuple[SweepPoint, ...]
) -> tuple[dict[tuple[str, str], Mapping[str, tuple[int, ...]]], dict[str, Mapping[str, Any]]]:
    tokens: dict[tuple[str, str], Mapping[str, tuple[int, ...]]] = {}
    custom: dict[str, Mapping[str, Any]] = {}
    for arm in arms:
        engine = engines[arm.id]
        for point in points:
            tokens[(arm.id, point.id)] = engine.gate(point)
        custom[arm.id] = engine.custom_gate()
    return tokens, custom


def _closing_gate(arm: Arm, engine: BenchEngine, **where: Any) -> list[dict[str, Any]]:
    """The arm's own gate read again after its warm-up and timed runs, before its engine closes, so an arm that judges
    every run it saw judges those too; nothing for an arm without a gate."""

    c = engine.custom_gate()
    if c.get("provided") is False:
        return []
    return [{"arm": arm.id, "kind": "arm-closing", **where, "passed": _gate_passed(c), "detail": dict(c)}]


def _close_all(engines: Mapping[str, BenchEngine]) -> None:
    first: BaseException | None = None
    for engine in reversed(list(engines.values())):
        try:
            engine.close()
        except BaseException as e:  # noqa: BLE001
            first = first or e
    if first is not None:
        raise first


def interleave_order(arms: tuple[Arm, ...], baseline: str, rep: int) -> tuple[str, ...]:
    """Every arm once per rep, with the baseline alternating between the two ends."""

    others = [a.id for a in arms if a.id != baseline]
    return tuple([baseline, *others] if rep % 2 == 0 else [*reversed(others), baseline])


def _record(
    samples: dict[tuple[str, str], list[dict[str, Any]]],
    schedule: list[dict[str, Any]],
    arm: str,
    point: SweepPoint,
    rep: int,
    engine: BenchEngine,
    mode: str,
) -> None:
    got = dict(engine.measure(point))
    got["rep"] = rep
    samples[(arm, point.id)].append(got)
    schedule.append({"phase": "timed", "mode": mode, "arm": arm, "point": point.id, "rep": rep})


def decode_time_per_token(
    steps: list[Mapping[str, Any]], *, output_lag: int = 0, device_tail_s: float = 0.0, generated: int | None = None
) -> float | None:
    """Host time of the engine steps that returned a decode-only forward's tokens (a forward with decode tokens and no
    prefill tokens), over the tokens it decoded: per-token decode cost with prefill left out, which time per output
    token mixes in under mixed traffic. A step's host time holds the device time of the forward it waits on to return
    sampled tokens, which is its own forward's when `output_lag` is 0 and the forward `output_lag` iterations back's
    under async scheduling, so the time of iteration i + lag is charged to the forward of iteration i. None when no
    step qualifies, the steps carry no token counts (`run_requests_arrivals` without a reachable scheduler), or the
    device work left after the last step (`device_tail_s`) exceeds DECODE_TAIL_SHARE of the decode time, which shows
    device work the host step times didn't see. Also None when the steps' decode tokens add up to more than the run
    `generated`: a decode token is one generated token, so a larger total means the counts misclassify prompt tokens
    as decode (18 Oct: `arrivals.scheduled_tokens` looked prompt lengths up by the submitted request id, which this vLLM
    pin renames, so every prompt token counted as decode)."""

    if generated is not None and sum(int(s.get("decode_tokens") or 0) for s in steps) > generated:
        return None
    lag = int(output_lag)
    decode = [
        (float(steps[i + lag]["s"]), int(s["decode_tokens"]))
        for i, s in enumerate(steps[: len(steps) - lag])
        if s.get("forward") and s.get("prefill_tokens") == 0 and s.get("decode_tokens")
    ]
    tokens = sum(n for _, n in decode)
    seconds = sum(t for t, _ in decode)
    if not tokens or device_tail_s > DECODE_TAIL_SHARE * seconds:
        return None
    return seconds / tokens


def inter_token_latencies(returns: list[tuple[float, Mapping[str, int]]]) -> list[float]:
    """One latency per generated token after each request's first: the time from the step that returned the request's
    previous tokens to the step that returned this one, shared evenly by the tokens a step returned at once. ``returns``
    holds, per engine step, the clock reading as it returned and each returned request's tokens generated so far. This
    is the per-token latency a streaming client sees; the first token's is ``time_to_first_token_s``."""

    last: dict[str, tuple[float, int]] = {}
    out: list[float] = []
    for t, counts in returns:
        for rid, n in counts.items():
            if rid in last:
                t0, n0 = last[rid]
                if n <= n0:
                    continue
                out.extend([(t - t0) / (n - n0)] * (n - n0))
            last[rid] = (t, n)
    return out


def percentile(values: list[float], q: int) -> float | None:
    """The q-th percentile (1..99), interpolated between the closest ranks; None without values."""

    if len(values) < 2:
        return values[0] if values else None
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]


def _aggregate(
    arms: tuple[Arm, ...],
    points: tuple[SweepPoint, ...],
    baseline: str,
    samples: Mapping[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    out = []
    for point in points:
        base = sorted(samples[(baseline, point.id)], key=lambda x: x["rep"])
        row: dict[str, Any] = {"point": {**point.to_json(), "headline": point.headline}, "arms": {}}
        for arm in arms:
            reps = sorted(samples[(arm.id, point.id)], key=lambda x: x["rep"])
            if [r["rep"] for r in reps] != [b["rep"] for b in base]:
                raise ValueError(f"{arm.id} at {point.id} has reps {[r['rep'] for r in reps]}, its baseline "
                                 f"{[b['rep'] for b in base]}: paired ratios need the same reps")
            metrics = {}
            present = tuple(m for m in OPTIONAL_METRICS if all(r["metrics"].get(m) is not None for r in reps + base))
            for metric in METRICS + present:
                absolute = _summary([float(r["metrics"][metric]) for r in reps])
                pairs = [(float(r["metrics"][metric]), float(b["metrics"][metric])) for r, b in zip(reps, base)]
                # A zero baseline rep is a broken measurement; dropping it would report a mean over the other reps.
                ratios = _summary([x / y for x, y in pairs]) if all(y != 0 for _, y in pairs) else {"mean": None, "n": 0}
                metrics[metric] = {**absolute, "ratio_to_stock": ratios}
            row["arms"][arm.id] = {
                "metrics": metrics,
                "span_totals": _numeric_summaries([r.get("spans", {}) for r in reps]),
                "extras": _numeric_summaries([r.get("extras", {}) for r in reps]),
                "reps": reps,
            }
        out.append(row)
    return out


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _identity(
    preset: Preset,
    arms: tuple[Arm, ...],
    sweep: str,
    git_commit: str,
    versions: Mapping[str, Mapping[str, Any]],
    config_identity: Mapping[str, Any] | None = None,
    *,
    reps: int,
    interleave: str,
) -> dict[str, Any]:
    shared = {}
    for key in ("vllm_version", "torch_version", "gpu"):
        seen = {v.get(key) for v in versions.values()}
        if len(seen) != 1:
            raise RuntimeError(f"benchmark arms loaded different {key}s: {sorted(map(str, seen))}")
        shared[key] = next(iter(seen))
    vllm_version = str(shared["vllm_version"])
    if preset.vllm_pin[:9] not in vllm_version:
        raise RuntimeError(f"benchmark preset pins vLLM {preset.vllm_pin}, but the runtime reports {vllm_version}")
    arm_docs = []
    for arm in arms:
        runtime = versions[arm.id]
        arm_docs.append({
            **arm.to_json(),
            "kernel_digest": runtime.get("kernel_digest") or arm.kernel_digest,
            "engine_overrides": dict(runtime.get("engine_overrides") or {}),
            "kv_cache_tokens": runtime.get("kv_cache_tokens"),
        })
    return {
        "preset": preset.to_json(),
        "vllm_version": vllm_version,
        "vllm_pin": preset.vllm_pin,
        "torch_version": shared["torch_version"],
        "gpu": shared["gpu"],
        "arms": arm_docs,
        "git_commit": git_commit,
        "sweep": sweep,
        "reps": reps,
        "interleave": interleave,
        **({"config": dict(config_identity)} if config_identity is not None else {}),
    }


def run_benchmark(
    preset: Preset,
    arms: tuple[Arm, ...],
    points: tuple[SweepPoint, ...],
    *,
    sweep: str,
    reps: int,
    factory: EngineFactory,
    git_commit: str | None = None,
    config_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every gate before any timing, then use resident interleaving or alternating blocks; an arm's own gate is read
    again after its timed runs (`arm-closing`), and a failure there leaves the result without points."""

    if reps < 1 or not arms or not points:
        raise ValueError("benchmark needs at least one arm, point, and repetition")
    if any(a.control == a.id for a in arms):
        raise ValueError("an arm cannot be its own control: its greedy gate would compare it with itself")
    baseline = "stock-fp8" if any(a.id == "stock-fp8" for a in arms) else "stock" if any(a.id == "stock" for a in arms) else arms[0].id
    schedule: list[dict[str, Any]] = []
    samples = {(a.id, p.id): [] for a in arms for p in points}
    versions: dict[str, Mapping[str, Any]] = {}
    engines: dict[str, BenchEngine] = {}
    resident_failure: str | None = None

    try:
        for arm in arms:
            engine = factory.create(arm, preset, resident=True)
            engines[arm.id] = engine
            versions[arm.id] = engine.identity
    except ResidentCapacityError as e:
        resident_failure = str(e)
        _close_all(engines)
        engines = {}
    except BaseException:
        # A live engine is a non-daemon process: left open, it keeps the point from exiting and holds its GPU lease.
        _close_all(engines)
        raise

    if resident_failure is None:
        try:
            tokens, custom = _collect_gates(engines, arms, points)
            gates = _evaluate_gates(arms, points, tokens, custom, versions)
            if all(g["passed"] for g in gates):
                for point in points:
                    for arm in arms:
                        engines[arm.id].warmup(point)
                        schedule.append({"phase": "warmup", "mode": "resident", "arm": arm.id, "point": point.id})
                    for rep in range(reps):
                        for arm_id in interleave_order(arms, baseline, rep):
                            _record(samples, schedule, arm_id, point, rep, engines[arm_id], "resident")
                gates += [g for arm in arms for g in _closing_gate(arm, engines[arm.id])]
        finally:
            _close_all(engines)
        mode = "resident"
    else:
        # Gate each arm in its own lifetime. Nothing is timed until all gates have passed.
        tokens, custom, gates = {}, {}, []
        for arm in arms:
            engine = factory.create(arm, preset, resident=False)
            versions[arm.id] = engine.identity
            try:
                for point in points:
                    tokens[(arm.id, point.id)] = engine.gate(point)
                custom[arm.id] = engine.custom_gate()
            finally:
                engine.close()
        gates = _evaluate_gates(arms, points, tokens, custom, versions)
        if all(g["passed"] for g in gates):
            # Two engine-lifetime cycles, the second in reverse arm order so that drift over the run cancels; each rep
            # belongs to one cycle.
            for cycle in range(min(2, reps)):
                for arm in arms if cycle == 0 else tuple(reversed(arms)):
                    indices = [r for r in range(reps) if r % min(2, reps) == cycle]
                    if not indices or not all(g["passed"] for g in gates):
                        continue
                    engine = factory.create(arm, preset, resident=False)
                    gates.append(_engine_identity_gate(arm, cycle, versions[arm.id], engine.identity))
                    if not gates[-1]["passed"]:
                        engine.close()
                        continue
                    try:
                        for point in points:
                            engine.warmup(point)
                            schedule.append({"phase": "warmup", "mode": "blocks", "cycle": cycle,
                                             "arm": arm.id, "point": point.id})
                            for rep in indices:
                                _record(samples, schedule, arm.id, point, rep, engine, "blocks")
                        gates += _closing_gate(arm, engine, cycle=cycle)
                    finally:
                        engine.close()
        mode = "blocks"

    if all(g["passed"] for g in gates):
        gates += _timed_token_gates(arms, points, samples)
    identity = _identity(preset, arms, sweep, git_commit or _git_head(), versions, config_identity,
                         reps=reps, interleave=mode)
    doc = {
        "schema": result_format.SCHEMA,
        "identity": identity,
        "baseline": baseline,
        "sweep": [p.to_json() for p in points],
        "gates": gates,
        "execution": {"interleave": mode, "resident_failure": resident_failure, "reps": reps, "schedule": schedule},
        "points": _aggregate(arms, points, baseline, samples) if all(g["passed"] for g in gates) else [],
    }
    result_format.validate(doc)
    return doc


def _callback(obj: Any, name: str, target: Target, default: Any) -> Any:
    fn = obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)
    return default if not callable(fn) else fn(target)


class _DirectLLM:
    """The subset of ``verity_vllm.LLM`` needed for a content-bound local checkpoint."""

    def __init__(self, llm: Any, engine_overrides: Mapping[str, Any]) -> None:
        from verity_vllm.engine.engine_profile import get_model_runner

        runner = get_model_runner(llm)
        self.vllm, self.model = llm, runner.get_model()
        # benchmarks/vllm_encoded/bench_arm.py still reads verity_vllm.LLM's private `_engine.llm`.
        self._engine = SimpleNamespace(llm=llm, model=self.model)
        self.engine_record = {"engine_overrides": dict(engine_overrides)}
        self.protocols = None

    def close(self) -> None:
        # The worker process exits immediately after close and releases vLLM's
        # process-owned CUDA state.
        pass


def _kv_cache_tokens(llm: Any) -> int | None:
    """The engine's actual KV pool in tokens (in-process V1 engine), None when the engine does not expose it."""

    try:
        cache = llm.vllm.llm_engine.vllm_config.cache_config
        blocks, block_size = cache.num_gpu_blocks, cache.block_size
    except AttributeError:
        return None
    if blocks is None or block_size is None:
        return None
    return int(blocks) * int(block_size)


def engine_args(preset: Preset, precision: str, *, resident: bool, eager: bool) -> dict[str, Any]:
    """The engine options every arm of a preset shares, before an arm's own overrides.

    With ``serving_defaults`` the engine compiles as vLLM serves (torch.compile, its CUDA-graph mode and capture sizes,
    the execution label ``compiled``, or ``eager`` for an eager engine); otherwise a graph engine replays decode-only CUDA
    graphs at the preset's ``capture_sizes``. Engines of one group
    share one exact KV pool, ``resident_kv_cache_memory_bytes`` or ``block_kv_cache_memory_bytes``: vLLM schedules by
    free KV blocks, and an arm whose install leaves a different amount of free memory would otherwise get a different
    pool, a different schedule, and (through batch-composition-dependent kernels) different tokens than its control.
    Without a pin the kv-pool gate catches unequal pools. ``attention_backend``, ``attention_config`` and
    ``moe_backend`` pass to vLLM as set, for a preset whose Build's row declares them (its ``engine.args``)."""

    settings = preset.engine_settings
    engine: dict[str, Any] = {
        "tensor_parallel_size": preset.gpu_count,
        "max_num_seqs": 128,
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
    for key in ("attention_backend", "attention_config", "moe_backend"):
        if settings.get(key) is not None:
            engine[key] = dict(settings[key]) if key == "attention_config" else str(settings[key])
    return engine


class _VllmEngine:
    """One worker process's live vLLM engine."""

    def __init__(self, arm: Arm, preset: Preset, resident: bool, eager: bool) -> None:
        if preset.unavailable_reason:
            raise ValueError(f"preset {preset.id} is unavailable: {preset.unavailable_reason}")
        if preset.revision is None and preset.model_digest is None:
            raise ValueError(f"preset {preset.id} has no pinned checkpoint revision")
        from verity_vllm.engine import env as engine_env

        if preset.engine_settings.get("serving_defaults"):
            engine_env.apply_compile_fresh()
        engine_env.apply_engine(None)
        import torch
        import vllm

        from verity_vllm.engine.hooks import Hooks, live
        from verity_vllm.pipeline.llm import LLM

        self.arm, self.obj, self.installed = arm, plugin(arm), None
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

        engine = engine_args(preset, arm.precision or preset.precision, resident=resident, eager=eager)
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
            if arm.protocols:
                engine["protocols"] = ",".join(arm.protocols)
            self.llm = LLM(preset.model, revision=preset.revision, **engine)
        self.target.engine, self.target.model = self.llm, self.llm.model
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
                else arm.kernel_digest
            ),
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": f"{torch.__version__} cuda {torch.version.cuda}",
            "engine_overrides": dict(self.llm.engine_record.get("engine_overrides") or {}),
        }

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
        return {} if self.llm.protocols is None else self.llm.protocols.info()

    def _run(self, point: SweepPoint, phase: str) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
        import torch
        from vllm import SamplingParams

        from verity_vllm.engine.arrivals import run_requests_arrivals

        measured = phase == "measure"
        requests = self._requests(point)
        sampling = SAMPLING[point.sampling]
        params = lambda r: SamplingParams(max_tokens=r.max_tokens, ignore_eos=True,  # noqa: E731
                                          detokenize=False, seed=r.seed, **sampling)
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
        wall, device_tail = end - start, end - returned
        tokens = {rid: tuple(int(t) for t in out.outputs[0].token_ids) for rid, out in finished.items()}
        if not measured:
            return tokens, {}
        generated = sum(map(len, tokens.values()))
        ttft = [float(r["time_to_first_token_s"]) for r in timing["requests"].values()]
        after = self._spans()
        extras = _callback(self.obj, "extras", self.target, {}) if self.obj is not None else {}
        metrics = {
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
            "spans": _delta(after, before),
            "extras": extras or {},
            "generated_tokens": generated,
            "wall_s": wall,
        }

    def gate(self, point: SweepPoint) -> Mapping[str, tuple[int, ...]]:
        return self._run(point, "gate")[0]

    def custom_gate(self) -> Mapping[str, Any]:
        if self.obj is None:
            return {"passed": True, "provided": False}
        got = _callback(self.obj, "gate", self.target, None)
        if got is None:
            return {"passed": True, "provided": False}
        if isinstance(got, bool):
            return {"passed": got, "provided": True}
        return dict(got)

    def warmup(self, point: SweepPoint) -> None:
        self._run(point, "warmup")

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
        so a kill during a gate surfaces as what it was instead of a connection error (re-baseline bug list, 14 Oct)."""
        final = None
        try:
            if self.conn.poll(0):
                final = self.conn.recv()
        except (EOFError, ConnectionResetError, OSError):
            pass
        self.process.join()
        if isinstance(final, dict) and "error" in final:
            return f"benchmark engine process died during {doing}: " + final["error"] + "\n" + final.get("trace", "")
        code = self.process.exitcode
        hint = " (SIGKILL: likely out of memory; its own last error is on the job's stderr)" if code == -9 else ""
        return f"benchmark engine process died during {doing} with exit code {code}{hint}"

    def gate(self, point: SweepPoint) -> Mapping[str, tuple[int, ...]]:
        return self._ask("gate", point)

    def custom_gate(self) -> Mapping[str, Any]:
        return self._ask("custom_gate")

    def warmup(self, point: SweepPoint) -> None:
        self._ask("warmup", point)

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
                except (EOFError, ConnectionResetError, BrokenPipeError):
                    # A worker that sent its error has left its loop and is closing its engine on its own; raising here
                    # would replace that error, which the caller is propagating, with this one.
                    pass
        finally:
            self.process.join()
            self.conn.close()
        if self.process.exitcode:
            raise RuntimeError(f"benchmark engine process exited with code {self.process.exitcode} while closing")


class ProcessEngineFactory:
    """Spawn-isolated engines, allowing several arms to remain resident on one GPU."""

    def __init__(self, *, eager: bool = False) -> None:
        self.eager = eager
        self.context = mp.get_context("spawn")

    def create(self, arm: Arm, preset: Preset, *, resident: bool) -> BenchEngine:
        if not self.eager:
            bound = build_bound(arm, preset.gpu_count)
            if bound:
                raise ValueError(
                    f"arm {arm.id!r} binds its served kernels to the row's Build ({', '.join(bound)}); CUDA graphs can "
                    "change the served kernels so they no longer match it: run with --eager (or enforce_eager = true "
                    "in the preset's engine settings)")
        if resident and not arm.resident:
            raise ResidentCapacityError(f"arm {arm.id!r} requires an isolated engine lifetime")
        if preset.engine_settings.get("serving_defaults") and os.environ.get(INDUCTOR_DIR_MARK) != "1":
            # Every arm's process compiles on its own, and the kernel configurations Inductor autotunes reach the numerics
            # (a reduction's block size): one cache directory gives every arm the configurations the first one picked.
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="verity_bench_inductor_")
            os.environ[INDUCTOR_DIR_MARK] = "1"
        parent, child = self.context.Pipe()
        process = self.context.Process(target=_worker, args=(child, arm, preset, resident, self.eager))
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
