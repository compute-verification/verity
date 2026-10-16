"""Execution profile (CONTRACT.md §2 / §2.1) shared by production and replay.

Everything that decides *what vLLM runs* lives here so that production and replay
construct the engine from one place. The declared flags are:

* ``VLLM_BATCH_INVARIANT=1`` (contract §2),
* ``VLLM_USE_V2_MODEL_RUNNER=1`` (trace replay requires Model Runner V2, which is not
  the default for Llama/Qwen2 architectures in the pinned wheel: see
  ``vllm/config/vllm.py::DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES``), and
* ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` (engine core in-process so the recorder can read
  the scheduler's step trace and the replay adapter can intercept the model runner; it
  changes process topology, not the executed kernels).

Import this module *before* ``vllm`` so the environment is set first.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from verity_vllm.engine import env

DECLARED_ENV = env.ENGINE

CASES = {
    "B0": {"repo": "HuggingFaceTB/SmolLM2-135M"},
    # [E1 NIGHT 2026-09-12, G3; PROPOSED] the second-model cases: SMOL360 = SmolLM2-360M (B0's architecture, larger), QWEN05 = Qwen2.5-0.5B (B1's
    # architecture at head dim 64); registry configs verity_vllm.program.registry.b1.{SMOL360,QWEN05}_CONFIG.
    "SMOL360": {"repo": "HuggingFaceTB/SmolLM2-360M"},
    "QWEN05": {"repo": "Qwen/Qwen2.5-0.5B"},
    "B1": {"repo": "Qwen/Qwen2.5-1.5B"},
    # B2 = the same dense baseline as B0, on Ampere (contract: "does the same adapter work on
    # another GPU?"); same checkpoint, same workload, the GPU is recorded in the profile manifest.
    "B2": {"repo": "HuggingFaceTB/SmolLM2-135M"},
    "B5": {"repo": "microsoft/phi-2"},
    # B7 = the contract's 7B dense checkpoint (§1: `Qwen/Qwen2.5-7B` optional subrow), used by W4d for
    # the compiled-vs-eager production A/B on Hopper (§15.5 / §16.7); same B1 workload (same tokenizer).
    "B7": {"repo": "Qwen/Qwen2.5-7B"},
    # M1/M2 = the exploratory MoE lane (W9, wave 3): Qwen1.5-MoE-A2.7B-Chat (Qwen2MoeForCausalLM,
    # 60 routed experts top-4 + shared expert, 24 layers), frozen in manifests/checkpoints.json.
    "M1": {"repo": "Qwen/Qwen1.5-MoE-A2.7B-Chat"},
    "M2": {"repo": "Qwen/Qwen1.5-MoE-A2.7B-Chat"},
    # M0 = the persisted MoE bundle (§16.8, <= 100 MB): one short request (the first 16 prompt
    # tokens of M1's request 0, 8 generated tokens), same checkpoint and profile as M1.
    "M0": {"repo": "Qwen/Qwen1.5-MoE-A2.7B-Chat"},
}

MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.80


def max_model_len_for(cp: dict[str, Any]) -> int:
    """[coverage cov-ln, additive] `MAX_MODEL_LEN` capped at the checkpoint's `max_position_embeddings` (its config.json) when that
    is smaller: vLLM refuses `max_model_len` above the derived maximum (Pythia-160M: 2048) unless `VLLM_ALLOW_LONG_MAX_MODEL_LEN`
    is set, which would let positions run past the rotary table.  Every earlier case (SmolLM2 8192, Qwen2.5 32768, Gemma-2 8192,
    Phi-3 4096) keeps 4096, so their profile ids are unchanged; the value is recorded in versions.json / run.json as before."""
    try:
        cfg = json.loads((Path(cp["local_path"]) / "config.json").read_text())
    except (OSError, ValueError, KeyError):
        return MAX_MODEL_LEN
    mpe = cfg.get("max_position_embeddings")
    if isinstance(mpe, (int, float)) and 0 < int(mpe) < MAX_MODEL_LEN:
        return int(mpe)
    return MAX_MODEL_LEN


def apply_env() -> dict[str, str]:
    from verity_vllm.config import checkpoints_hf_home

    return env.apply_engine(checkpoints_hf_home())


def load_checkpoints(path: str | Path) -> dict[str, dict[str, Any]]:
    doc = json.loads(Path(path).read_text())
    return {c["repo"]: c for c in doc["checkpoints"]}


def checkpoint_for(case: str, checkpoints_path: str | Path) -> dict[str, Any]:
    if case in CASES:
        cp = load_checkpoints(checkpoints_path)[CASES[case]["repo"]]
    else:
        # generalization sweep: roles registered by verity_vllm/input_provenance/add_checkpoint.py carry `role == case`
        doc = json.loads(Path(checkpoints_path).read_text())
        matches = [c for c in doc["checkpoints"] if c.get("role") == case]
        if not matches:
            raise SystemExit(f"case {case!r}: not in CASES and no checkpoints.json entry with role {case!r}")
        cp = matches[0]
    if not cp.get("downloaded") or not cp.get("local_path"):
        raise SystemExit(f"{cp['repo']} is not downloaded on this pod (manifest says resolve-only)")
    if not Path(cp["local_path"]).exists():
        raise SystemExit(f"{cp['repo']}: local snapshot {cp['local_path']} missing")
    return cp


def verify_local_weights(cp: dict[str, Any]) -> None:
    """Fail loudly if the snapshot on disk does not match the frozen shas (contract §1)."""
    local = Path(cp["local_path"])
    for fname, sha in cp["safetensors_sha256"].items():
        h = hashlib.sha256()
        with (local / fname).open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != sha:
            raise SystemExit(f"{fname}: sha256 {h.hexdigest()} != frozen {sha}")


REPLAY_EXECS = ("eager", "compiled-nograph")


TP_WORKER_EXTENSION = "verity_vllm.tp.poc_tp_worker.VerityTPWorkerExtension"


def engine_kwargs(cp: dict[str, Any], *, replay: bool, seed: int = 0, enforce_eager: bool = False, custom_ops: str | None = None,
                  replay_exec: str = "eager", max_num_batched_tokens: int | None = None,
                  tensor_parallel_size: int | None = 1, quantization: str | None = None) -> dict[str, Any]:
    """LLM(...) kwargs. Production: stock defaults (``enforce_eager`` is a declared flag for the
    eager-production profile row, §15.5; ``custom_ops`` = ``"all"`` keeps vLLM's CUDA custom ops
    inside the compiled forward instead of inductor-generated kernels, another declared flag / new
    profile row; ``max_num_batched_tokens`` is a declared scheduler flag, used by the varied-batch
    cases (M2) to change batch composition; replay must use the production's value so the recorded
    schedule is reproducible). Replay: + trace replay; ``replay_exec`` = ``eager`` (dispatch-mode
    interception, ``enforce_eager=True``) or ``compiled-nograph`` (§16.1: the production compilation
    with CUDA graphs off in replay only, ``compilation_config.cudagraph_mode = NONE``).
    ``tensor_parallel_size`` > 1 (lane TP, coverage wave axis 2; a profile port's
    ``parallelism.tensor_parallel_size``) is a declared production flag: the model is sharded over that
    many GPUs (vLLM's multiprocess executor; the engine core stays in-process). The key is absent at 1 so
    earlier profile ids are unchanged. In replay with TP > 1 the model lives in worker processes, so the
    capture is installed there through vLLM's ``worker_extension_cls`` seam (``tp_worker.py``);
    production never loads the extension."""
    kw: dict[str, Any] = {
        "model": cp["local_path"],
        "tokenizer": cp["local_path"],
        "dtype": "bfloat16",
        "seed": seed,
        "max_model_len": max_model_len_for(cp),
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "trust_remote_code": False,
    }
    if tensor_parallel_size is not None and int(tensor_parallel_size) > 1:
        kw["tensor_parallel_size"] = int(tensor_parallel_size)
        if replay:
            kw["worker_extension_cls"] = TP_WORKER_EXTENSION
    if quantization:
        # [QZ, additive] lane QZ (coverage wave axis "quantization", port b1-fp8): `--quantization fp8` is a declared
        # production flag (LLM(quantization=...)); the key is absent when unset so earlier profile ids are unchanged.
        # Replay reads it from the manifest's `quantization` field (replay.py) so the same online-quantised weights
        # and the same FP8 kernels run under the capture.
        kw["quantization"] = str(quantization)
    if max_num_batched_tokens is not None:
        kw["max_num_batched_tokens"] = int(max_num_batched_tokens)
    if enforce_eager:
        kw["enforce_eager"] = True
    if custom_ops and not replay:
        kw["compilation_config"] = {"custom_ops": [custom_ops]}
    if replay and replay_exec not in REPLAY_EXECS:
        raise ValueError(f"replay_exec must be one of {REPLAY_EXECS}")
    if replay and replay_exec == "compiled-nograph":
        if enforce_eager:
            raise ValueError("compiled-nograph replay of an eager production profile makes no sense (the production ran no compiled kernels)")
        kw["enable_trace_replay"] = True
        kw["compilation_config"] = {"cudagraph_mode": "NONE"}
        kw["enable_prefix_caching"] = False  # same replay restriction as below
        return kw
    if replay:
        kw["enable_trace_replay"] = True
        kw["enforce_eager"] = True  # replay restriction: dispatch-mode interception is eager-only
        # Replay restriction: with prefix caching a request whose first block(s) match another
        # request's never computes those positions itself (its KV reads there are the other request's
        # values, its boundary has no step for them). Stock production keeps caching; the B0/B1 rows
        # bind every position of a request to that request's own steps. Cross-request KV identity
        # is the R3 (prefix cache) row's rule.
        kw["enable_prefix_caching"] = False
    return kw


def get_model_runner(llm: Any) -> Any:
    """Navigate LLM -> in-process engine core -> worker -> model runner (uniproc executor)."""
    eng = llm.llm_engine
    core = getattr(eng, "engine_core", None)
    inner = getattr(core, "engine_core", None) or core
    executor = getattr(inner, "model_executor", None)
    worker = getattr(executor, "driver_worker", None)
    for attr in ("worker", "model_runner"):
        if hasattr(worker, attr):
            worker = getattr(worker, attr)
            if attr == "model_runner":
                return worker
    raise RuntimeError("could not find the in-process model runner (is VLLM_ENABLE_V1_MULTIPROCESSING=0?)")


def get_scheduler(llm: Any) -> Any:
    core = llm.llm_engine.engine_core
    inner = getattr(core, "engine_core", None) or core
    return inner.scheduler


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def gpu_info() -> dict[str, Any]:
    import torch

    props = torch.cuda.get_device_properties(0)
    drv = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    return {
        "name": props.name,
        "count": torch.cuda.device_count(),
        "driver": drv,
        "compute_capability": f"{props.major}.{props.minor}",
        "pod": os.environ.get("RUNPOD_POD_ID", os.uname().nodename),
        "total_memory_gb": round(props.total_memory / 2**30, 2),
        # W1cF F7/F10: the batch-invariant GEMM specialises on the SM count (NUM_SMS); a verifier on a
        # different device reports `specialised: NUM_SMS` for its PTX rather than a mismatch.
        "multi_processor_count": int(props.multi_processor_count),
        # [R13 ampere] the tuned-GEMM arch family vLLM selects its batch-invariant GEMM table by, from vLLM's OWN function on this
        # device's capability (None = no table = default tiling, e.g. Ampere): the fold's GEMM target selection
        # (verity_vllm.program.frontend.gemm_targets) checks the record against it.
        "tuned_gemm_arch_family": _tuned_gemm_arch_family(props.major, props.minor),
    }


def _tuned_gemm_arch_family(major: int, minor: int):
    try:
        from vllm.model_executor.determinism.batch_invariant_configs import _get_tuned_matmul_arch_family
        from vllm.platforms.interface import DeviceCapability

        return _get_tuned_matmul_arch_family(DeviceCapability(int(major), int(minor)))
    except Exception:  # noqa: BLE001  (another vLLM layout: the field is absent, not guessed)
        return "<unavailable>"


def dist_identity(name: str) -> dict[str, Any]:
    """Identity of an installed distribution: version, sha256 of its RECORD (every installed file with
    its hash: an installation fingerprint), pip's ``direct_url.json`` and, when that names a local wheel
    file that still exists, the wheel's sha256."""
    import importlib.metadata as md

    try:
        dist = md.distribution(name)
    except md.PackageNotFoundError:
        return {"installed": False}
    out: dict[str, Any] = {"installed": True, "version": dist.version, "record_sha256": None, "direct_url": None, "wheel_sha256": None}
    rec = dist.read_text("RECORD")
    if rec:
        out["record_sha256"] = hashlib.sha256(rec.encode()).hexdigest()
    du = dist.read_text("direct_url.json")
    if du:
        try:
            d = json.loads(du)
        except ValueError:
            d = {}
        url = d.get("url")
        out["direct_url"] = url
        h = (d.get("archive_info") or {}).get("hash")
        if isinstance(h, str) and h.startswith("sha256="):
            out["wheel_sha256"] = h.split("=", 1)[1]
        if out["wheel_sha256"] is None and isinstance(url, str) and url.startswith("file://"):
            p = Path(url[len("file://"):])
            if p.is_file():
                hh = hashlib.sha256()
                with p.open("rb") as f:
                    for chunk in iter(lambda: f.read(1 << 22), b""):
                        hh.update(chunk)
                out["wheel_sha256"] = hh.hexdigest()
    return out


def software_info(wheel_sha256: str | None, vllm_commit: str | None) -> dict[str, Any]:
    import importlib.metadata as md

    import torch
    import vllm

    def ver(name: str) -> str | None:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    return {
        "vllm": {"version": vllm.__version__, "wheel_sha256": wheel_sha256, "commit": vllm_commit},
        "python": {
            "version": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": ver("triton"),
            "flashinfer": ver("flashinfer-python"),
            "xformers": ver("xformers"),
            "transformers": ver("transformers"),
        },
        # W1cF F7: installation identities (dist_identity):
        # version, sha256 of the dist's RECORD (every installed file with its hash), pip's direct_url and, when
        # the installed wheel file is still present, its sha256 (the caller-supplied wheel_sha256 above is a claim).
        "dists": {name: dist_identity(name) for name in ("vllm", "torch", "triton")},
    }


def effective_compilation(llm: Any) -> dict[str, Any]:
    vc = llm.llm_engine.vllm_config
    cc = vc.compilation_config
    return {
        "mode": str(getattr(cc, "mode", None)),
        "cudagraph_mode": str(getattr(cc, "cudagraph_mode", None)),
        "enforce_eager": bool(vc.model_config.enforce_eager),
        "model_runner": "v2" if vc.use_v2_model_runner else "v1",
        "custom_ops": list(getattr(cc, "custom_ops", [])),
        "optimization_level": int(getattr(vc, "optimization_level", -1)),
    }


def cache_info(llm: Any) -> dict[str, Any]:
    """Effective KV-cache flags (§15.8: 'prefix caching enabled' is recorded as the effective
    value, and whether the recorded workload hit it is in production/workload_exercised.json)."""
    cc = llm.llm_engine.vllm_config.cache_config
    return {
        "enable_prefix_caching": bool(cc.enable_prefix_caching),
        "block_size": int(cc.block_size),
        "kv_cache_dtype": str(getattr(cc, "cache_dtype", "auto")),
        "enable_chunked_prefill": bool(getattr(llm.llm_engine.vllm_config.scheduler_config, "enable_chunked_prefill", False)),
        "max_num_seqs": int(getattr(llm.llm_engine.vllm_config.scheduler_config, "max_num_seqs", 0)),
        "max_num_batched_tokens": int(getattr(llm.llm_engine.vllm_config.scheduler_config, "max_num_batched_tokens", 0)),
    }


def _backends_of_runner(runner: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"attention": None, "linear": None, "moe": None, "sampler": None}
    model = runner.get_model()
    for name, mod in model.named_modules():
        impl = getattr(mod, "impl", None)
        if impl is not None and hasattr(mod, "kv_cache_spec") or (impl is not None and name.endswith(".attn")):
            out["attention"] = f"{impl.__class__.__module__}.{impl.__class__.__name__}"
            break
    out["linear"] = (
        "vllm.model_executor.determinism.batch_invariant.linear_batch_invariant (triton matmul_persistent)"
        if os.environ.get("VLLM_BATCH_INVARIANT") == "1"
        else "UnquantizedLinearMethod._gemm_impl"
    )
    sampler = getattr(runner, "sampler", None)
    out["sampler"] = f"{sampler.__class__.__module__}.{sampler.__class__.__name__}" if sampler else None
    return out


def _backends_probe(worker: Any) -> dict[str, Any]:
    """Runs inside a TP worker process (``LLM.collective_rpc`` with a callable): read-only class names of
    the layers that ran, plus the collective implementation the TP group chose (lane TP)."""
    out = _backends_of_runner(worker.model_runner)
    try:
        from vllm.distributed import parallel_state as ps

        tp = ps.get_tp_group()
        comm = getattr(tp, "device_communicator", None)
        ca = getattr(comm, "ca_comm", None)
        out["tp_rank"] = int(tp.rank_in_group)
        out["tp_world_size"] = int(tp.world_size)
        out["all_reduce"] = {
            "device_communicator": f"{comm.__class__.__module__}.{comm.__class__.__name__}" if comm is not None else None,
            "custom_allreduce": None if ca is None else {"class": ca.__class__.__name__, "disabled": bool(getattr(ca, "disabled", False)),
                                                          "max_size_bytes": int(getattr(ca, "max_size", 0))},
            "symm_mem": getattr(comm, "symm_mem_comm", None) is not None and not getattr(getattr(comm, "symm_mem_comm", None), "disabled", True),
            "quick_allreduce": getattr(comm, "qr_comm", None) is not None and not getattr(getattr(comm, "qr_comm", None), "disabled", True),
            "flashinfer_allreduce": getattr(comm, "fi_allreduce_fusion", None) is not None,
            "pynccl": getattr(comm, "pynccl_comm", None) is not None and not getattr(getattr(comm, "pynccl_comm", None), "disabled", True),
            "nccl_env": {k: os.environ.get(k) for k in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS", "NCCL_NTHREADS",
                                                      "NCCL_BUFFSIZE", "NCCL_LAUNCH_MODE") if os.environ.get(k) is not None},
        }
    except Exception as e:  # noqa: BLE001
        out["tp_error"] = repr(e)
    return out


def backends_info(llm: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"attention": None, "linear": None, "moe": None, "sampler": None}
    try:
        tp = int(llm.llm_engine.vllm_config.parallel_config.tensor_parallel_size)
        if tp > 1:
            # the model lives in the worker processes: ask rank 0 (all ranks run the same classes)
            per_rank = llm.collective_rpc(_backends_probe)
            out = dict(per_rank[0])
            out["ranks"] = len(per_rank)
            out.pop("tp_rank", None)
            return out
        out = _backends_of_runner(get_model_runner(llm))
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
    return out


def build_manifest(
    *,
    case: str,
    cp: dict[str, Any],
    llm: Any,
    wheel_sha256: str | None,
    vllm_commit: str | None,
    adapter_revision: str,
    interception: str,
    sampling: dict[str, Any],
    env_flags: dict[str, str],
    enforce_eager: bool = False,
    custom_ops: str | None = None,
    max_num_batched_tokens: int | None = None,
    request_order: str = "workload",
    engine_overrides: dict[str, Any] | None = None,
    tensor_parallel_size: int = 1,
    quantization: str | None = None,
) -> dict[str, Any]:
    sw = software_info(wheel_sha256, vllm_commit)
    kw = {k: v for k, v in engine_kwargs(cp, replay=False, enforce_eager=enforce_eager, custom_ops=custom_ops,
                                         max_num_batched_tokens=max_num_batched_tokens,
                                         tensor_parallel_size=tensor_parallel_size,
                                         quantization=quantization).items() if k not in ("model", "tokenizer")}
    # state rows (W10, R2/R3/R4): declared scheduler / cache overrides are part of the profile
    # (they change the profile id) and are replayed under the same values (state_rows.py)
    if engine_overrides:
        kw.update(engine_overrides)
    doc = {
        "schema": "verity-vllm/profile/v1",
        "case": case,
        "model": {
            "repo": cp["repo"],
            "revision": cp["revision"],
            "weights_sha256": dict(cp["safetensors_sha256"]),
            "config_sha256": cp.get("config_sha256"),
        },
        "vllm": sw["vllm"],
        "python": sw["python"],
        "dists": sw["dists"],
        "gpu": gpu_info(),
        "dtype": "bfloat16",
        # [QZ, additive] the declared quantization scheme (None = the bf16 profiles; "fp8" = lane QZ's b1-fp8 port,
        # vLLM online per-tensor weight quantisation + dynamic per-token activation quantisation); the harness's C0
        # check compares this field with the port's `quantization`
        "quantization": (str(quantization) if quantization else None),
        "backends": backends_info(llm),
        "parallelism": {"tensor_parallel_size": int(tensor_parallel_size), "pipeline_parallel_size": 1},
        "compilation": effective_compilation(llm),
        "cache": cache_info(llm),
        "engine_flags": dict(env_flags),
        "engine_kwargs": kw,
        "sampling": sampling,
        "adapter": {"revision": adapter_revision, "interception": interception},
        **({"engine_overrides": dict(engine_overrides)} if engine_overrides else {}),
    }
    if request_order != "workload":
        # varied batch composition (M2): the workload's requests are submitted in another order;
        # key present only when non-default so earlier profile ids are unchanged
        doc["workload_order"] = request_order
    return doc


def canonical_json_bytes(doc: Any) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def profile_id_of(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()[:12]


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%MZ", time.gmtime())
