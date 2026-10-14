"""Gate, warm, interleave, measure, and aggregate vLLM benchmark arms."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import statistics
import subprocess
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol

from verity_bench.vllm import result as result_format
from verity_bench.vllm.arms import Arm, Target, plugin
from verity_bench.vllm.presets import Preset, SweepPoint

METRICS = ("time_per_output_token_s", "time_to_first_token_s", "tokens_per_s", "peak_gpu_memory_bytes")
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


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
        half = T95.get(len(values) - 1, 1.96) * statistics.stdev(values) / math.sqrt(len(values))
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


def _evaluate_gates(
    arms: tuple[Arm, ...],
    points: tuple[SweepPoint, ...],
    tokens: Mapping[tuple[str, str], Mapping[str, tuple[int, ...]]],
    custom: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for arm in arms:
        c = custom[arm.id]
        out.append({"arm": arm.id, "kind": "arm", "passed": _gate_passed(c), "detail": dict(c)})
        for point in points:
            got = tokens[(arm.id, point.id)]
            control = tokens.get((arm.control, point.id)) if arm.control else None
            passed = control is None or got == control
            out.append({
                "arm": arm.id,
                "control": arm.control,
                "point": point.id,
                "kind": "greedy-tokens",
                "passed": passed,
                "tokens": sum(map(len, got.values())),
                "digest": _token_digest(got),
                **({"control_digest": _token_digest(control)} if control is not None else {}),
            })
    return out


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
            metrics = {}
            for metric in METRICS:
                absolute = _summary([float(r["metrics"][metric]) for r in reps])
                ratios = [
                    float(r["metrics"][metric]) / float(b["metrics"][metric])
                    for r, b in zip(reps, base, strict=True)
                    if float(b["metrics"][metric]) != 0
                ]
                metrics[metric] = {**absolute, "ratio_to_stock": _summary(ratios)}
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
) -> dict[str, Any]:
    vllm_versions = {v.get("vllm_version") for v in versions.values()}
    if len(vllm_versions) != 1:
        raise RuntimeError(f"benchmark arms loaded different vLLM versions: {sorted(map(str, vllm_versions))}")
    vllm_version = str(next(iter(vllm_versions)))
    if preset.vllm_pin[:9] not in vllm_version:
        raise RuntimeError(f"benchmark preset pins vLLM {preset.vllm_pin}, but the runtime reports {vllm_version}")
    arm_docs = []
    for arm in arms:
        runtime = versions[arm.id]
        arm_docs.append({
            **arm.to_json(),
            "kernel_digest": runtime.get("kernel_digest") or arm.kernel_digest,
            "engine_overrides": dict(runtime.get("engine_overrides") or {}),
        })
    return {
        "preset": preset.to_json(),
        "vllm_version": vllm_version,
        "vllm_pin": preset.vllm_pin,
        "arms": arm_docs,
        "git_commit": git_commit,
        "sweep": sweep,
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
    """Run every gate before any timing, then use resident interleaving or alternating blocks."""

    if reps < 1 or not arms or not points:
        raise ValueError("benchmark needs at least one arm, point, and repetition")
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

    if resident_failure is None:
        try:
            tokens, custom = _collect_gates(engines, arms, points)
            gates = _evaluate_gates(arms, points, tokens, custom)
            if all(g["passed"] for g in gates):
                for point in points:
                    for arm in arms:
                        engines[arm.id].warmup(point)
                        schedule.append({"phase": "warmup", "mode": "resident", "arm": arm.id, "point": point.id})
                    for rep in range(reps):
                        for arm_id in interleave_order(arms, baseline, rep):
                            _record(samples, schedule, arm_id, point, rep, engines[arm_id], "resident")
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
        gates = _evaluate_gates(arms, points, tokens, custom)
        if all(g["passed"] for g in gates):
            # Two alternating engine-lifetime cycles; each rep belongs to one cycle.
            for cycle in range(min(2, reps)):
                for arm in arms:
                    indices = [r for r in range(reps) if r % min(2, reps) == cycle]
                    if not indices:
                        continue
                    engine = factory.create(arm, preset, resident=False)
                    try:
                        for point in points:
                            engine.warmup(point)
                            schedule.append({"phase": "warmup", "mode": "blocks", "cycle": cycle,
                                             "arm": arm.id, "point": point.id})
                            for rep in indices:
                                _record(samples, schedule, arm.id, point, rep, engine, "blocks")
                    finally:
                        engine.close()
        mode = "blocks"

    identity = _identity(preset, arms, sweep, git_commit or _git_head(), versions, config_identity)
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
        self._engine = SimpleNamespace(llm=llm, model=runner.get_model())
        self.engine_record = {"engine_overrides": dict(engine_overrides)}
        self.protocols = None

    def close(self) -> None:
        # The worker process exits immediately after close and releases vLLM's
        # process-owned CUDA state.
        pass


class _VllmEngine:
    """One worker process's live vLLM engine."""

    def __init__(self, arm: Arm, preset: Preset, resident: bool, eager: bool) -> None:
        if preset.unavailable_reason:
            raise ValueError(f"preset {preset.id} is unavailable: {preset.unavailable_reason}")
        if preset.revision is None and preset.model_digest is None:
            raise ValueError(f"preset {preset.id} has no pinned checkpoint revision")
        from verity_vllm.engine import env as engine_env

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

        settings = preset.engine_settings
        precision = arm.precision or preset.precision
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
        if not eager:
            engine["compilation_config"] = {
                "mode": 0,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": list(settings["capture_sizes"]),
            }
        if preset.model_digest is not None:
            if arm.protocols:
                raise ValueError("a local checkpoint override supports stock and experimental arms, not protocol-option arms")
            self.llm = _DirectLLM(
                vllm.LLM(
                    model=preset.model,
                    enable_prefix_caching=False,
                    seed=0,
                    **engine,
                ),
                {k: engine[k] for k in ("dtype", "quantization", "compilation_config") if engine.get(k) is not None},
            )
        else:
            engine["protocols"] = ",".join(arm.protocols) or None
            self.llm = LLM(preset.model, revision=preset.revision, **engine)
        self.target.engine, self.target.model = self.llm, self.llm._engine.model
        ready = getattr(self.installed, "ready", None)
        if callable(ready):
            ready(self.target)
        self.identity = {
            "vllm_version": vllm.__version__,
            "kernel_digest": (
                _callback(self.obj, "kernel_digest", self.target, None)
                if self.obj is not None and callable(
                    self.obj.get("kernel_digest") if isinstance(self.obj, Mapping) else getattr(self.obj, "kernel_digest", None)
                )
                else arm.kernel_digest
            ),
            "gpu": torch.cuda.get_device_name(0),
            "engine_overrides": dict(self.llm.engine_record.get("engine_overrides") or {}),
        }

    def _requests(self, point: SweepPoint) -> list[Any]:
        vocab = int(self.llm._engine.llm.llm_engine.vllm_config.model_config.hf_config.vocab_size)
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
            )
            for i, shape in enumerate(point.requests())
        ]

    def _spans(self) -> Any:
        if self.obj is not None:
            got = _callback(self.obj, "spans", self.target, None)
            if got is not None:
                return got
        return {} if self.llm.protocols is None else self.llm.protocols.info()

    def _run(self, point: SweepPoint, measured: bool) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
        import torch
        from vllm import SamplingParams

        from verity_vllm.engine.arrivals import run_requests_arrivals

        requests = self._requests(point)
        params = lambda r: SamplingParams(temperature=0.0, max_tokens=r.max_tokens, ignore_eos=True,  # noqa: E731
                                          detokenize=False, seed=0)
        before = self._spans()
        if measured:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        finished, timing = run_requests_arrivals(
            self.llm._engine.llm,
            requests,
            params,
            request_timing=True,
        )
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
        tokens = {rid: tuple(int(t) for t in out.outputs[0].token_ids) for rid, out in finished.items()}
        if not measured:
            return tokens, {}
        generated = sum(map(len, tokens.values()))
        ttft = [float(r["time_to_first_token_s"]) for r in timing["requests"].values()]
        after = self._spans()
        extras = _callback(self.obj, "extras", self.target, {}) if self.obj is not None else {}
        return tokens, {
            "metrics": {
                "time_per_output_token_s": wall / generated,
                "time_to_first_token_s": statistics.fmean(ttft),
                "tokens_per_s": generated / wall,
                "peak_gpu_memory_bytes": float(torch.cuda.max_memory_allocated()),
            },
            "spans": _delta(after, before),
            "extras": extras or {},
            "generated_tokens": generated,
            "wall_s": wall,
        }

    def gate(self, point: SweepPoint) -> Mapping[str, tuple[int, ...]]:
        return self._run(point, False)[0]

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
        self._run(point, False)

    def measure(self, point: SweepPoint) -> Mapping[str, Any]:
        return self._run(point, True)[1]

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
        self.conn.send((name, arg))
        try:
            response = self.conn.recv()
        except EOFError:
            raise RuntimeError(f"benchmark engine process exited with code {self.process.exitcode}") from None
        if "error" in response:
            raise RuntimeError(response["error"] + "\n" + response.get("trace", ""))
        return response["value"]

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
                self._ask("close")
        finally:
            self.process.join()
            self.conn.close()


class ProcessEngineFactory:
    """Spawn-isolated engines, allowing several arms to remain resident on one GPU."""

    def __init__(self, *, eager: bool = False) -> None:
        self.eager = eager
        self.context = mp.get_context("spawn")

    def create(self, arm: Arm, preset: Preset, *, resident: bool) -> BenchEngine:
        if resident and not arm.resident:
            raise ResidentCapacityError(f"arm {arm.id!r} requires an isolated engine lifetime")
        parent, child = self.context.Pipe()
        process = self.context.Process(target=_worker, args=(child, arm, preset, resident, self.eager))
        process.start()
        child.close()
        try:
            response = parent.recv()
        except EOFError:
            process.join()
            raise RuntimeError(f"benchmark engine {arm.id!r} exited during startup with code {process.exitcode}") from None
        if "error" in response:
            process.join()
            error = response["error"] + "\n" + response.get("trace", "")
            if resident and response.get("capacity"):
                raise ResidentCapacityError(error)
            raise RuntimeError(error)
        return _ProcessEngine(process, parent, response["ready"])
