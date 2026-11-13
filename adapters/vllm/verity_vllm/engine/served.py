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

The collectives' switches are among them: which of `CudaCommunicator.all_reduce`'s routes are on, FlashInfer's all-reduce backend, and
NCCL's own (`NCCL_ENVS`, those that change its algorithm, protocol, channels or chunking).  What vLLM resolved from them on a rank, which
the node decides too (FlashInfer's ``auto`` backend falls back to ``trtllm`` without NVSwitch multicast), is read back beside the
switches as ``collectives`` (`collectives`) and recorded, not judged: two engines' records compare with `collectives_differ`.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from verity_vllm.config import row_world
from verity_vllm.engine import env as engine_env
from verity_vllm.engine.profiles import transport
from verity_vllm.records.schema_ids import SERVED_SWITCHES_SCHEMA

__all__ = ["DEFAULTS", "ENFORCE_EAGER", "apply", "declared", "execution", "execution_env", "of_summary", "read_rank",
           "record", "require"]

SCHEMA = SERVED_SWITCHES_SCHEMA
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
    "VLLM_ALLREDUCE_USE_FLASHINFER": True,
    "VLLM_FLASHINFER_ALLREDUCE_BACKEND": "auto",
    "VLLM_ALLREDUCE_USE_SYMM_MEM": True,
    "VLLM_USE_NCCL_SYMM_MEM": False,
    "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": False,
    "VLLM_DISABLE_PYNCCL": False,
}
#: NCCL's switches that change which algorithm, protocol, channels, chunks or intra-node transport (whose topology search orders
#: the ranks) reduce a call, and so the order of its adds; NCCL tunes each itself when unset, vLLM's default but under batch
#: invariance.
NCCL_ENVS = ("NCCL_ALGO", "NCCL_PROTO", "NCCL_NVLS_ENABLE", "NCCL_COLLNET_ENABLE", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS",
             "NCCL_MIN_CTAS", "NCCL_MAX_CTAS", "NCCL_NTHREADS", "NCCL_BUFFSIZE", "NCCL_P2P_LEVEL", "NCCL_SHM_DISABLE",
             "NCCL_CUMEM_ENABLE", "NCCL_GRAPH_FILE", "NCCL_TOPO_FILE")
DEFAULTS.update(dict.fromkeys(NCCL_ENVS))
_FLAGS = ("VLLM_BATCH_INVARIANT", "VLLM_USE_DEEP_GEMM", "VLLM_DISABLE_SHARED_EXPERTS_STREAM", "VLLM_USE_FLASHINFER_SAMPLER",
          "VLLM_USE_V2_MODEL_RUNNER", "VLLM_ALLREDUCE_USE_FLASHINFER", "VLLM_ALLREDUCE_USE_SYMM_MEM", "VLLM_USE_NCCL_SYMM_MEM",
          "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC")
ENVS = (*_FLAGS, "VLLM_DISABLE_PYNCCL", "VLLM_FLASHINFER_ALLREDUCE_BACKEND", "NCCL_P2P_DISABLE", *NCCL_ENVS)
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
    "compiled": ({"enforce_eager": False}, ("a compiled row at vLLM's serving defaults: torch.compile with vLLM's own CUDA-graph mode, "
                                            "or, for an architecture vLLM serves with breakable CUDA graphs (VLLM_USE_BREAKABLE_CUDAGRAPH), "
                                            "those graphs over the eager kernels; no batch invariance; its Program is of that execution")),
    "bi-eager": ({"VLLM_BATCH_INVARIANT": True, "enforce_eager": True}, _BI),
    "bi-eager-fa2": ({"VLLM_BATCH_INVARIANT": True, "enforce_eager": True, "flash_attn_version": 2}, f"{_BI}, with attention_config.flash_attn_version 2"),
    "bi-compiled": ({"VLLM_BATCH_INVARIANT": True, "enforce_eager": False, "cudagraph_mode": "PIECEWISE"}, f"{_BI}; {_COMPILED}"),
}
#: The ``enforce_eager`` each execution label builds its engine with.
ENFORCE_EAGER = {k: v[0]["enforce_eager"] for k, v in EXECUTIONS.items() if k is not None}
#: What vLLM sets under ``VLLM_BATCH_INVARIANT``, over every argument and the environment (`ParallelConfig._verify_args`, and
#: `override_envs_for_invariance` in each worker), and why.  It turns FlashInfer's all-reduce off in `CudaCommunicator` itself, which
#: ``collectives`` shows and no switch does.
BATCH_INVARIANT: tuple[dict[str, Any], str] = (
    {"disable_custom_all_reduce": True, "VLLM_ALLREDUCE_USE_SYMM_MEM": False, "NCCL_COLLNET_ENABLE": "0", "NCCL_NVLS_ENABLE": "0",
     "NCCL_MIN_NCHANNELS": "1", "NCCL_MAX_NCHANNELS": "1", "NCCL_PROTO": "Simple", "NCCL_ALGO": "allreduce:tree", "NCCL_NTHREADS": "1"},
    ("vLLM's batch invariance turns its custom all-reduce and torch's symmetric memory off and pins NCCL to one channel of its tree "
     "(ParallelConfig._verify_args, override_envs_for_invariance)"))
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
    if switch == "VLLM_DISABLE_PYNCCL":
        return v if isinstance(v, bool) else str(v).lower() in ("true", "1")
    if switch.startswith("NCCL_"):
        return str(v)
    if switch == "VLLM_FLASHINFER_ALLREDUCE_BACKEND":
        return str(v).lower()
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


def _block(doc: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """The block a document holds under ``key``; an empty one where it leaves the block out or holds none (null, empty)."""
    held = doc.get(key)
    return held if held else {}


def declared(wl: Mapping[str, Any], target: Any = None) -> dict[str, dict[str, Any]]:
    """What sets each switch of a workload (`vllm_adapter.load_workload`'s) whose engine is built for ``target`` (its `TargetProfile`,
    or None): ``{switch: {"value", "source", "reason"}}`` for every switch a layer sets."""
    out: dict[str, dict[str, Any]] = {}

    def put(source: str, values: Mapping[str, Any], reasons: Mapping[str, str | None]) -> None:
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
    eng = _block(wl, "engine")
    env = {**_block(eng, "env"), **(target.env() if target is not None else {})}
    args = {**(target.engine_args() if target is not None else {}), **_block(eng, "args")}
    put("row file", settings(env, args), _block(eng, "reasons"))
    if "VLLM_BATCH_INVARIANT" in out and out["VLLM_BATCH_INVARIANT"]["value"] is True:
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
    env = engine_env.snapshot()
    got = {k: getattr(envs, k) for k in (*_FLAGS, "VLLM_DISABLE_PYNCCL", "VLLM_FLASHINFER_ALLREDUCE_BACKEND")
           if k != "VLLM_USE_V2_MODEL_RUNNER"}
    got.update({k: env.get(k) for k in ("NCCL_P2P_DISABLE", *NCCL_ENVS)})
    got.update(VLLM_USE_V2_MODEL_RUNNER=vc.use_v2_model_runner,
               disable_custom_all_reduce=vc.parallel_config.disable_custom_all_reduce, enforce_eager=vc.model_config.enforce_eager,
               compilation_mode=cc.mode, cudagraph_mode=cc.cudagraph_mode, attention_backend=ac.backend,
               flash_attn_version=ac.flash_attn_version, moe_backend=vc.kernel_config.moe_backend)
    return {"rank": int(get_tensor_model_parallel_rank()), "switches": {k: value(k, v) for k, v in got.items()},
            "collectives": collectives()}


#: The switches that pick a TP all-reduce's route or NCCL's order of adds, which `collectives_differ` compares
COLLECTIVE_SWITCHES = ("disable_custom_all_reduce", "VLLM_ALLREDUCE_USE_FLASHINFER", "VLLM_FLASHINFER_ALLREDUCE_BACKEND",
                       "VLLM_ALLREDUCE_USE_SYMM_MEM", "VLLM_USE_NCCL_SYMM_MEM", "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC",
                       "VLLM_DISABLE_PYNCCL", "NCCL_P2P_DISABLE", *NCCL_ENVS)


def collectives() -> dict[str, Any]:
    """In a serving process: the TP group's all-reduce routes as vLLM resolved them on this rank, in `CudaCommunicator.all_reduce`'s
    order: each communicator on or off, FlashInfer's workspace backend (which the fused all-reduce + RMSNorm shares; None before its
    first call, and in a process that never loaded FlashInfer's all-reduce), the custom all-reduce's and torch symmetric memory's
    byte caps, and the NCCL version pynccl loaded.  Without a TP group it raises, as `read_rank`'s rank does."""
    from vllm.distributed.parallel_state import get_tp_group

    dc = getattr(get_tp_group(), "device_communicator", None)
    ws = getattr(sys.modules.get("vllm.distributed.device_communicators.flashinfer_all_reduce"), "_fi_ar_workspace", None)

    def on(name: str) -> bool:
        c = getattr(dc, name, None)
        return c is not None and not getattr(c, "disabled", False)

    ca, sm, nc = (getattr(dc, n, None) for n in ("ca_comm", "symm_mem_comm", "pynccl_comm"))
    return {"world": getattr(dc, "world_size", None), "flashinfer": on("fi_ar_comm"),
            "flashinfer_backend": getattr(ws, "backend", None), "quick": on("qr_comm"),
            "flashinfer_pcie_ipc": on("fi_pcie_ipc_ar_comm"), "aiter": on("aiter_ar_comm"), "custom": on("ca_comm"),
            "custom_max_bytes": getattr(ca, "max_size", None) if on("ca_comm") else None, "symm_mem": on("symm_mem_comm"),
            "symm_mem_max_bytes": getattr(sm, "max_size", None) if on("symm_mem_comm") else None, "pynccl": on("pynccl_comm"),
            "nccl_version": getattr(nc, "nccl_version", None)}


def collectives_differ(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    """Where two engines' served records (`record`) differ, rank by rank, in a `COLLECTIVE_SWITCHES` switch or a resolved route
    (``collectives``).  Two engines whose all-reduces are to agree bit for bit, such as a replay on another node, need none; the
    order a node's NVSwitch reduces in is its own, and neither record shows it."""
    ra, rb = ({r["rank"]: r for r in x["ranks"]} for x in (a, b))
    out = [] if set(ra) == set(rb) else [f"ranks {sorted(ra)} against {sorted(rb)}"]
    for k in sorted(set(ra) & set(rb)):
        sa, sb = ra[k]["switches"], rb[k]["switches"]
        out += [f"rank {k}: {s} {sa.get(s)!r} against {sb.get(s)!r}" for s in COLLECTIVE_SWITCHES if sa.get(s) != sb.get(s)]
        ca, cb = ra[k].get("collectives", {}), rb[k].get("collectives", {})
        out += [f"rank {k}: {c} {ca.get(c)!r} against {cb.get(c)!r}" for c in sorted(set(ca) | set(cb)) if ca.get(c) != cb.get(c)]
    return out


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
    if not sm:
        return None
    return sm.get("served") or _block(sm, "versions").get("served")
