"""Gate, warm, interleave, measure, and aggregate vLLM benchmark arms.

The runner drives each arm's engine through `BenchEngine`, made by an `EngineFactory`: `engine.ProcessEngineFactory` builds
one arm's vLLM engine in a process of its own, and `timing` holds what each of its timed runs records."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from collections.abc import Mapping
from typing import Any, Protocol

from verity.protocols.accounting.pouw.verifier.salt import epoch_open_root

from verity_bench.vllm import result as result_format
from verity_bench.vllm.arms import Arm
from verity_bench.vllm.presets import Preset, SweepPoint

METRICS = ("time_per_output_token_s", "time_to_first_token_s", "tokens_per_s", "peak_gpu_memory_bytes")
# Recorded for an arm at a point only when its every rep and the baseline's have it (the engine exposes its scheduler;
# the point's requests generate more than one token).
OPTIONAL_METRICS = ("decode_time_per_token_s", "inter_token_latency_p50_s", "inter_token_latency_p99_s")
ENGINE_IDENTITY = ("vllm_version", "torch_version", "gpu", "kv_cache_tokens", "kernel_digest", "engine_overrides")
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
       25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}
# Past a spec's `warmups`, an engine warms until two consecutive runs' wall times differ by at most this share. On a host
# at load 600-900 a bandwidth engine's stalled runs were 8% to 8x slower than the next; an extra warm-up costs
# seconds against an engine's minutes of start-up, so the bar is strict and `max_warmups` bounds it.
WARMUP_AGREEMENT = 0.05


def t95(df: int) -> float:
    """The two-sided 95% Student-t quantile, from the largest tabulated df at or below `df` (wider, never narrower)."""

    return T95[max(k for k in T95 if k <= df)]


class ResidentCapacityError(RuntimeError):
    """An engine could not join the resident set, so the runner should use `blocks`."""


class BenchEngine(Protocol):
    identity: Mapping[str, Any]

    def gate(self, point: SweepPoint) -> Mapping[str, tuple[int, ...]]: ...
    def custom_gate(self) -> Mapping[str, Any]: ...
    def warmup(self, point: SweepPoint) -> Mapping[str, Any] | None: ...
    def measure(self, point: SweepPoint) -> Mapping[str, Any]: ...
    def close(self) -> None: ...


class _Named(Protocol):
    @property
    def id(self) -> str: ...


class EngineFactory(Protocol):
    def create(self, arm: Arm, preset: Preset, *, resident: bool) -> BenchEngine: ...


def summarize(values: list[float]) -> dict[str, Any]:
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
    return {key: summarize([d[key] for d in leaves if key in d]) for key in keys}


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
            # vLLM admits and preempts by free KV pages, and a step's batch composition reaches the numerics (attention
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


#: an arm gate's key, true for an arm whose tokens carry each pass's coins, so no two runs of one engine need agree:
#: Pearl-C's, whose every call's noise is seeded by the install's salt and the call's index across its epoch
PER_PASS_TOKENS = "per_pass_tokens"


def _pearl_c_gate(run: Any) -> dict[str, Any]:
    """The experimental arm's own gate, `e2e.gate`'s call check, on the protocol arm's installed run: at every distinct
    (n, k), served y against x·Wᵀ within `bench_arm.ERR`, the commitment twice, and the ``hash_leaf``-omitted negative
    control. Pearl-C's y carries the epoch's live salt, so no stock arm is its token control, and each call's index across
    the epoch, so no two of its own engine's passes need agree either (`PER_PASS_TOKENS`). Its ``coins`` are the
    installed run's: a live challenge phase's (``challenger``, its salt epoch ``pass_index`` and the root its salt coin follows,
    `salt.epoch_open_root` of the weights' root and that epoch) when the option took its salt from one, else
    ``in-process``."""

    # The experimental arm's module is the tree's (`benchmarks/`, on a run's path), not this distribution's.
    from benchmarks.pouw.pearl_c_vllm.bench_arm import PearlCArm

    check = PearlCArm()
    check.run, check.schedule = run, run.schedule.name
    c = run.commits
    check.coins = ({"source": c["coin_source"], "session": c.get("session"), "pass_index": c["pass_index"],
                    "salt_after": epoch_open_root(bytes.fromhex(c["weights_root"]), c["pass_index"]).hex()}
                   if c.get("coin_source", "in-process") != "in-process" else {"source": "in-process"})
    check.weights = {w: x.tensor for w, x in run.weights.items()}
    got = check.gate(None)
    # The check begins and ends its own passes; the next forward's calls need an open pass, as the adapter's hook leaves
    # one between steps.
    run.begin(stepped=True)
    return {**got, "provided": True}


def h1_summary(run: Any) -> dict[str, Any]:
    """One closed H1 window (`pouw_h1.H1Run`): ``ok`` when every call fell inside its S steps and T rows at its public
    position, in order (``static``), the credit gate passed every served call (its y the GEMM that wrote its z/y
    records), every admitted request has its entry and the salt opened. Its coins are in-process, so it credits nothing."""

    verdicts = list(run.verdicts.values())
    passed = sum(bool(v.get("credit")) for v in verdicts)
    gaps = run.request_gaps()
    out = {"steps": run.steps, "steps_served": run.details.get("steps_served"), "rows": run.rows, "calls": len(run.calls),
           "served_calls": len(verdicts), "refused_calls": len(run.refused), "order_ok": bool(run.order_ok),
           "static": bool(run.details.get("static")), "credit_gate_passed": passed,
           "credit_gate_refusals": sorted({str(v.get("reason")) for v in verdicts if not v.get("credit")})[:3],
           "requests": len(run.requests), "refused_requests": len(run.refused_requests), "request_gaps": len(gaps),
           "unopened": run.details.get("unopened"), "u": run.details.get("u"), "coin_source": run.coin_source}
    out["ok"] = bool(verdicts and out["static"] and not run.refused and passed == len(verdicts) and not gaps
                     and not run.refused_requests and out["unopened"] is None)
    return out


def h1_gate(windows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """An H1 arm's own gate: every window its engine has closed so far is `h1_summary`'s ``ok``."""

    bad = [w for w in windows if not w.get("ok")]
    return {"passed": bool(windows) and not bad, "provided": True, "windows": len(windows),
            "failed": [dict(w) for w in bad[:2]],
            "coins": "in-process: each window credits nothing (pouw_h1.IN_PROCESS); this arm measures cost only"}


def _timed_token_gates(
    arms: tuple[Arm, ...], points: tuple[SweepPoint, ...], samples: Mapping[tuple[str, str], list[dict[str, Any]]],
    tokens: Mapping[tuple[str, str], Mapping[str, tuple[int, ...]]], mode: str, custom: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Every timed rep of an arm with a control must produce the control's tokens in that rep: the greedy gate runs
    before timing, so an arm that computed something else only when timed would pass it. In resident mode an arm
    without a control is gated and timed by one engine, whose timed reps must produce its own gated tokens; in `blocks`
    mode they are other engines, whose tokens may differ from the gated one's, and an arm whose own gate says its tokens
    carry each pass's coins (`PER_PASS_TOKENS`) has no such gate in either mode: its closing gate holds its timed reps."""

    out = []
    for arm in arms:
        if not arm.control:
            if mode != "resident" or (custom.get(arm.id) or {}).get(PER_PASS_TOKENS):
                continue
            for point in points:
                want = _token_digest(tokens[(arm.id, point.id)])
                bad = sorted(r["rep"] for r in samples[(arm.id, point.id)] if r.get("token_digest") != want)
                out.append({"arm": arm.id, "control": None, "against": "gate", "point": point.id,
                            "kind": "timed-tokens", "passed": not bad, **({"reps": bad} if bad else {})})
            continue
        for point in points:
            mine = {r["rep"]: r.get("token_digest") for r in samples[(arm.id, point.id)]}
            theirs = {r["rep"]: r.get("token_digest") for r in samples[(arm.control, point.id)]}
            bad = sorted(r for r in mine.keys() | theirs.keys() if mine.get(r) is None or mine.get(r) != theirs.get(r))
            out.append({"arm": arm.id, "control": arm.control, "point": point.id, "kind": "timed-tokens",
                        "passed": not bad, **({"reps": bad} if bad else {})})
    return out


def _engine_identity_gate(arm: Arm, cycle: int, gated: Mapping[str, Any], timed: Mapping[str, Any]) -> dict[str, Any]:
    """A `blocks` arm is gated and timed by different engines; the timed one must be the gated one's twin."""

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


def _warm(engine: BenchEngine, point: SweepPoint, warmups: int, max_warmups: int, schedule: list[dict[str, Any]],
          entry: Mapping[str, Any]) -> None:
    """`warmups` discarded runs, then more, up to `max_warmups` in all, until the last two runs' wall times agree within
    WARMUP_AGREEMENT. An engine whose first runs stall (the bandwidth arms' first prefill step, by seconds,
    for one to three runs after the engine starts) is timed only once it runs as it will keep running. An engine that
    reaches `max_warmups` without settling says so in its last warm-up's schedule entry (`settled: false`)."""

    walls: list[float | None] = []
    for index in range(max_warmups):
        got = engine.warmup(point)
        run = dict(got) if isinstance(got, Mapping) else {}
        walls.append(run.get("wall_s"))
        schedule.append({"phase": "warmup", **entry, "point": point.id, "warmup": index, **({"run": run} if run else {})})
        if index + 1 >= warmups and (max_warmups == warmups or _agree(walls[-2:])):
            return
    schedule[-1]["settled"] = False


def _agree(walls: list[float | None]) -> bool:
    if len(walls) != 2:
        return False
    a, b = walls
    if a is None or b is None:
        return False
    return abs(a - b) <= WARMUP_AGREEMENT * min(a, b)


def _aggregate(
    arms: tuple[_Named, ...],
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
                absolute = summarize([float(r["metrics"][metric]) for r in reps])
                pairs = [(float(r["metrics"][metric]), float(b["metrics"][metric])) for r, b in zip(reps, base)]
                # A zero baseline rep is a broken measurement; dropping it would report a mean over the other reps.
                ratios = summarize([x / y for x, y in pairs]) if all(y != 0 for _, y in pairs) else {"mean": None, "n": 0}
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
    warmups: int = 1,
    max_warmups: int = 1,
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
        "warmups": warmups,
        "max_warmups": max_warmups,
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
    warmups: int = 1,
    max_warmups: int | None = None,
) -> dict[str, Any]:
    """Run every gate before any timing, then use resident interleaving or `blocks`, with at least `warmups`
    discarded runs of a point before an engine's first timed one and, up to `max_warmups`, as many more as it takes two
    consecutive runs to agree (`_warm`). An arm's own gate is read again after its timed runs (`arm-closing`), and a
    failure there leaves the result without points."""

    if reps < 1 or not arms or not points:
        raise ValueError("benchmark needs at least one arm, point, and repetition")
    max_warmups = warmups if max_warmups is None else max_warmups
    if warmups < 1 or max_warmups < warmups:
        raise ValueError("benchmark needs at least one warm-up a point, and max_warmups no fewer than warmups")
    if any(a.control == a.id for a in arms):
        raise ValueError("an arm cannot be its own control: its greedy gate would compare it with itself")
    baseline = "stock-fp8" if any(a.id == "stock-fp8" for a in arms) else "stock" if any(a.id == "stock" for a in arms) else arms[0].id
    schedule: list[dict[str, Any]] = []
    samples: dict[tuple[str, str], list[dict[str, Any]]] = {(a.id, p.id): [] for a in arms for p in points}
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
                        _warm(engines[arm.id], point, warmups, max_warmups, schedule, {"mode": "resident", "arm": arm.id})
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
                            _warm(engine, point, warmups, max_warmups, schedule, {"mode": "blocks", "cycle": cycle, "arm": arm.id})
                            for rep in indices:
                                _record(samples, schedule, arm.id, point, rep, engine, "blocks")
                        gates += _closing_gate(arm, engine, cycle=cycle)
                    finally:
                        engine.close()
        mode = "blocks"

    if all(g["passed"] for g in gates):
        gates += _timed_token_gates(arms, points, samples, tokens, mode, custom)
    identity = _identity(preset, arms, sweep, git_commit or _git_head(), versions, config_identity,
                         reps=reps, warmups=warmups, max_warmups=max_warmups, interleave=mode)
    doc = {
        "schema": result_format.SCHEMA,
        "identity": identity,
        "baseline": baseline,
        "sweep": [p.to_json() for p in points],
        "gates": gates,
        "execution": {"interleave": mode, "resident_failure": resident_failure, "reps": reps, "warmups": warmups,
                      "max_warmups": max_warmups,
                      "schedule": schedule},
        "points": _aggregate(arms, points, baseline, samples) if all(g["passed"] for g in gates) else [],
    }
    result_format.validate(doc)
    return doc
