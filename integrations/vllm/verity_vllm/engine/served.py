"""The switches that select a served engine's kernels: what a workload's config declares, what every serving process reads back, and the
verdict that holds one to the other.

A switch is an environment variable vLLM reads when it builds an engine, or an engine argument its config resolves; `DEFAULTS` has each
at vLLM d9105ea80's own value.  `declared` layers what sets them, in the order a process applies it, a later layer replacing an earlier
one and keeping its reason when the two agree:

* the integration's engine profile, `engine.env.ENGINE`, with `engine.env.REASONS`;
* the workload's execution label (`EXECUTIONS`): a ``bi-*`` workload is batch invariant, an eager one eager;
* a tensor-parallel workload's transport (`engine.profiles.transport`);
* the workload file: ``engine.env``, ``engine.args``, its ``engine.target``'s `TargetProfile.env()` and ``engine_args()``, and
  ``engine.reasons``, a reason by switch name;
* vLLM's own config, last: under batch invariance its ``ParallelConfig`` turns the custom all-reduce off (`BATCH_INVARIANT`).

`judge` refuses a switch away from vLLM's default that nothing declares, a declaration away from the default without a reason, and a
read-back that is not what was declared.  `require` judges every rank of a built engine (`read_rank`) and fails the Commit closed; the
Commit records the result (`record`) in versions.json and summary.json as ``served``, and the circuit database's config record carries
it (`pipeline.config_record`).  Torch-free: vLLM is imported only inside a serving process.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from verity_vllm.config import row_world
from verity_vllm.engine import env as engine_env
from verity_vllm.engine.profiles import transport

SCHEMA = "verity-vllm/served-switches/v1"
#: The default of a switch vLLM resolves from the rest of the config: whatever it resolves is its own.
RESOLVED = "resolved by vLLM"
#: Each switch as vLLM leaves it when nothing sets it (vllm/envs.py, vllm/config/).
DEFAULTS: dict[str, Any] = {
    "VLLM_BATCH_INVARIANT": False,
    "VLLM_USE_DEEP_GEMM": True,
    "VLLM_DISABLE_SHARED_EXPERTS_STREAM": False,
    "VLLM_USE_FLASHINFER_SAMPLER": True,
    "VLLM_USE_V2_MODEL_RUNNER": RESOLVED,
    "NCCL_P2P_DISABLE": None,
    "disable_custom_all_reduce": False,
    "enforce_eager": False,
    "compilation_mode": RESOLVED,
    "cudagraph_mode": RESOLVED,
    "attention_backend": None,
    "flash_attn_version": None,
    "moe_backend": "auto",
}
_FLAGS = ("VLLM_BATCH_INVARIANT", "VLLM_USE_DEEP_GEMM", "VLLM_DISABLE_SHARED_EXPERTS_STREAM", "VLLM_USE_FLASHINFER_SAMPLER",
          "VLLM_USE_V2_MODEL_RUNNER")
ENVS = (*_FLAGS, "NCCL_P2P_DISABLE")
#: Where engine arguments set each config switch: a key of `LLM(**kw)`, or a key of one of its config dicts.
ARGS: dict[str, tuple[tuple[str, ...], ...]] = {
    "disable_custom_all_reduce": (("disable_custom_all_reduce",),),
    "enforce_eager": (("enforce_eager",),),
    "compilation_mode": (("compilation_config", "mode"),),
    "cudagraph_mode": (("compilation_config", "cudagraph_mode"),),
    "attention_backend": (("attention_backend",), ("attention_config", "backend")),
    "flash_attn_version": (("attention_config", "flash_attn_version"),),
    "moe_backend": (("moe_backend",), ("kernel_config", "moe_backend")),
}
#: What vLLM resolves an eager engine's compilation to (VllmConfig.__post_init__).
EAGER = {"compilation_mode": "NONE", "cudagraph_mode": "NONE"}
_MODES = ("NONE", "STOCK_TORCH_COMPILE", "DYNAMO_TRACE_ONCE", "VLLM_COMPILE")

_BI = "a batch-invariant row (bi-*), not served: its Programs are of vLLM's batch-invariant kernels"
_COMPILED = "a compiled row runs TargetProfile.COMPILATIONS' piecewise CUDA graphs"
#: Each execution label's switches and their reason.  None is a workload without a row id: the harness's batch-invariant eager engine.
EXECUTIONS: dict[str | None, tuple[dict[str, Any], str]] = {
    None: ({"VLLM_BATCH_INVARIANT": True, "enforce_eager": True}, "a workload without a row id runs the harness's batch-invariant eager engine"),
    "eager": ({"enforce_eager": True}, "an eager row: its Programs are of the eager forward"),
    "compiled": ({"enforce_eager": False}, "a compiled row at vLLM's serving defaults: torch.compile with vLLM's own CUDA-graph mode, "
                                           "no batch invariance; its Program is P_compiled of that execution"),
    "bi-eager": ({"VLLM_BATCH_INVARIANT": True, "enforce_eager": True}, _BI),
    "bi-eager-fa2": ({"VLLM_BATCH_INVARIANT": True, "enforce_eager": True, "flash_attn_version": 2}, f"{_BI}, with attention_config.flash_attn_version 2"),
    "bi-compiled": ({"VLLM_BATCH_INVARIANT": True, "enforce_eager": False, "cudagraph_mode": "PIECEWISE"}, f"{_BI}; {_COMPILED}"),
}
#: The ``enforce_eager`` each execution label builds its engine with.
ENFORCE_EAGER = {k: v[0]["enforce_eager"] for k, v in EXECUTIONS.items() if k is not None}
#: What vLLM's config sets under ``VLLM_BATCH_INVARIANT``, over every argument (`ParallelConfig._verify_args`), and why.
BATCH_INVARIANT: tuple[dict[str, Any], str] = ({"disable_custom_all_reduce": True},
                                               "vLLM's batch invariance turns its custom all-reduce off (ParallelConfig._verify_args)")
_MISSING = object()


def execution(label: str | None) -> tuple[dict[str, Any], str]:
    """An execution label's switches and their reason; ValueError for a label the harness does not model."""
    if label not in EXECUTIONS:
        raise ValueError(f"execution label {label!r} is not modelled by the harness (known: {sorted(ENFORCE_EAGER)})")
    return EXECUTIONS[label]


def execution_env(label: str | None) -> dict[str, str]:
    """The environment an execution label runs under: ``VLLM_BATCH_INVARIANT`` "1" for a batch-invariant one, else "0"."""
    return {"VLLM_BATCH_INVARIANT": "1" if execution(label)[0].get("VLLM_BATCH_INVARIANT") else "0"}


def apply(label: str | None) -> dict[str, str]:
    """`execution_env`, written into this process's environment before vLLM reads it; the values in force."""
    return engine_env.apply_target(execution_env(label))


def value(switch: str, v: Any) -> Any:
    """A switch's value in one spelling: a flag a bool, NCCL's its string, an enum or a backend by name."""
    v = getattr(v, "name", v)
    if v is None:
        return None
    if switch in _FLAGS:
        return v if isinstance(v, bool) else bool(int(v))
    if switch == "NCCL_P2P_DISABLE":
        return str(v)
    if switch in ("enforce_eager", "disable_custom_all_reduce"):
        return bool(v)
    if switch == "flash_attn_version":
        return int(v)
    if switch == "compilation_mode" and isinstance(v, int):
        return _MODES[v]
    if switch == "moe_backend":
        return str(v).lower().replace("-", "_")
    return str(v).upper()


def _at(args: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(args, Mapping) or key not in args:
            return _MISSING
        args = args[key]
    return args


def settings(env: Mapping[str, str], args: Mapping[str, Any]) -> dict[str, Any]:
    """The switches an environment and a set of engine arguments set."""
    out = {k: value(k, env[k]) for k in ENVS if k in env}
    for k, paths in ARGS.items():
        for p in paths:
            if (v := _at(args, p)) is not _MISSING:
                out[k] = value(k, v)
    return out


def declared(wl: Mapping[str, Any], target: Any = None) -> dict[str, dict[str, Any]]:
    """What sets each switch of a workload (`vllm_adapter.load_workload`'s) whose engine is built for ``target`` (its `TargetProfile`,
    or None): ``{switch: {"value", "source", "reason"}}`` for every switch a layer sets."""
    out: dict[str, dict[str, Any]] = {}

    def put(source: str, values: Mapping[str, Any], reasons: Mapping[str, str]) -> None:
        if values.get("enforce_eager") is True:
            values, reasons = {**EAGER, **values}, {**dict.fromkeys(EAGER, reasons.get("enforce_eager")), **reasons}
        for k, v in values.items():
            was = out.get(k)
            out[k] = {"value": v, "source": source,
                      "reason": reasons.get(k) or (was["reason"] if was is not None and was["value"] == v else None)}

    put("engine.env.ENGINE", settings(engine_env.ENGINE, {}), engine_env.REASONS)
    label = wl.get("execution")
    values, why = execution(label)
    put(f"execution {label}", values, dict.fromkeys(values, why))
    row = wl.get("row_id")
    if row and row_world(row) > 1:
        values = settings(transport.row_p2p_env(row), {"disable_custom_all_reduce": transport.row_custom_all_reduce_off(row)})
        put("engine.profiles.transport", values, dict.fromkeys(values, transport.REASON))
    eng = wl.get("engine") or {}
    env = {**(eng.get("env") or {}), **(target.env() if target is not None else {})}
    args = {**(target.engine_args() if target is not None else {}), **(eng.get("args") or {})}
    put("row file", settings(env, args), eng.get("reasons") or {})
    if (out.get("VLLM_BATCH_INVARIANT") or {}).get("value") is True:
        values, why = BATCH_INVARIANT
        put("vLLM ParallelConfig", values, dict.fromkeys(values, why))
    return out


def judge(switches: Mapping[str, Any], decl: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """What is wrong with one serving process's ``switches`` under ``decl`` (`declared`): nothing when each is vLLM's default or
    declared with a reason, and is what was declared."""
    bad: list[str] = []
    for k, default in DEFAULTS.items():
        d = decl.get(k)
        if k not in switches:
            bad.append(f"{k}: not read back")
            continue
        got = switches[k]
        if d is None:
            if default != RESOLVED and got != default:
                bad.append(f"{k}={got!r} is not vLLM's default {default!r} and nothing declares it")
            continue
        if got != d["value"]:
            bad.append(f"{k}={got!r} where {d['source']} declares {d['value']!r}")
        if d["value"] != default and not d["reason"]:
            bad.append(f"{k}={d['value']!r} ({d['source']}) is not vLLM's default {default!r} and has no reason (the row file's "
                       f"engine.reasons)")
    return bad


def read_rank(worker: Any) -> dict[str, Any]:
    """In a serving process (`LLM.collective_rpc` with a callable): its rank and each switch as vLLM resolved it there."""
    import vllm.envs as envs
    from vllm.distributed import get_tensor_model_parallel_rank

    vc = worker.vllm_config
    cc, ac = vc.compilation_config, vc.attention_config
    got = {k: getattr(envs, k) for k in _FLAGS if k != "VLLM_USE_V2_MODEL_RUNNER"}
    got.update(VLLM_USE_V2_MODEL_RUNNER=vc.use_v2_model_runner, NCCL_P2P_DISABLE=engine_env.snapshot().get("NCCL_P2P_DISABLE"),
               disable_custom_all_reduce=vc.parallel_config.disable_custom_all_reduce, enforce_eager=vc.model_config.enforce_eager,
               compilation_mode=cc.mode, cudagraph_mode=cc.cudagraph_mode, attention_backend=ac.backend,
               flash_attn_version=ac.flash_attn_version, moe_backend=vc.kernel_config.moe_backend)
    return {"rank": int(get_tensor_model_parallel_rank()), "switches": {k: value(k, v) for k, v in got.items()}}


def record(ranks: list[Mapping[str, Any]], decl: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The served switches of record: what was declared, what each rank read back, and every problem (`judge`)."""
    problems = [f"rank {r['rank']}: {p}" for r in ranks for p in judge(r["switches"], decl)] or ([] if ranks else ["no rank answered"])
    return {"schema": SCHEMA, "ok": not problems, "vllm_defaults": DEFAULTS, "declared": dict(decl), "ranks": list(ranks),
            "problems": problems}


def require(llm: Any, wl: Mapping[str, Any], target: Any, out: str | Path, timeout: float | None = None) -> dict[str, Any]:
    """`record` of every rank of ``llm`` against what ``wl`` and ``target`` declare.  On a problem ``out/summary.json`` says so
    (``pass`` false) and SystemExit fails the Commit closed."""
    rec = record(list(llm.collective_rpc(read_rank, timeout=timeout)), declared(wl, target))
    if not rec["ok"]:
        doc = {"schema": SCHEMA, "pass": False, "why": "served switches: " + "; ".join(rec["problems"]), "served": rec}
        Path(out, "summary.json").write_text(json.dumps(doc, indent=1) + "\n")
        raise SystemExit(f"fail-closed: {doc['why']}")
    return rec


def of_summary(sm: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The served record a Commit's summary.json carries (a TP Commit's at the top, a single-GPU Commit's in ``versions``)."""
    sm = sm or {}
    return sm.get("served") or (sm.get("versions") or {}).get("served")
