"""Lane tp2 (mechanism prototype): install the UNCHANGED observer inside every tensor-parallel rank's worker process.

With `tensor_parallel_size > 1` the pinned vLLM (V1 engine, `d9105ea80`) runs one worker process per rank under the
multiprocess executor; `VLLM_ENABLE_V1_MULTIPROCESSING=0` keeps the *engine core* in the driver process but the model,
its parameters, the KV caches and every dispatcher call live in the rank processes.  `vllm_adapter.build_engine` +
`Capture` (installed in-process on `get_model_runner(llm)`) therefore observe nothing: `get_model_runner` finds no
`driver_worker` (the executor is `MultiprocExecutor`), and a `TorchDispatchMode` entered in the driver would not see
the workers' dispatcher calls.

The seam prototyped here is vLLM's `worker_extension_cls` (`LLM(worker_extension_cls="<module>.<class>")`): the
methods of this class are mixed into the Worker class of every rank; the driver calls them with
`LLM.collective_rpc("tp2_<method>", kwargs=...)`, which runs the method in every rank process and returns the
per-rank results.  Nothing in `verity_vllm` core is touched: `tp2_install` constructs the protected `Capture` /
`ObserverConfig` / `make_header` over a per-rank `Engine` record whose `llm` is a shim (the rank has `vllm_config`
and the worker; `profile.build_manifest`'s probes run against them) and whose `runner` / `model` are the rank's own
`GPUModelRunner` and sharded model.  The raw log becomes per-rank: `log.rank<r>.jsonl.gz`, `rank<r>/values/...`,
`rank<r>/snapshots.json`, `stats.rank<r>.json`, `tokens.rank<r>.json`, `cos_sin_cache.rank<r>.npy`; the engine step
index is the shared step id (every rank executes the same `scheduler_output`).

`worker_extension_cls` is a documented engine argument with an RPC channel, so no plugin or patched Worker class.

Also here: `tp2_probe` (process topology + the TP group's communicator choice), and a per-rank kernel census
(`tp2_profiler_start` / `tp2_profiler_stop`: one `torch.profiler` session per rank over the whole run — the protected
`census.py` profiles in the driver process, which under TP launches no kernel).
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from verity_vllm.collectives.collective_sites import MOE_COLLECTIVE_CLASSES, MOE_COLLECTIVE_MODULES
from verity_vllm.engine import hooks, sparse_mla

# A rank imports this module when vLLM mixes the extension into its Worker, before it builds the model: the serving switches are
# armed here, so the rank's first forward (and every CUDA graph it captures) already runs them.
sparse_mla.arm()


def repo_modules_loaded() -> dict[str, Any]:
    """[R16 tp TP-11, rev F-r16-19] the repo-local files THIS process has imported (sys.modules, filtered to the repo root and vllm-poc) plus
    the collector's JIT C++/CUDA sources when native_collect was loaded -- what RAN, written from the running process (not a static closure).
    Paths relative to the repo root; a module outside the repo (site-packages, vllm) is not listed."""
    import sys as _sys
    from verity_vllm.config import ROOT as root
    roots = [root, root / "vllm-poc"]
    files: set[str] = set()
    for m in list(_sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        try:
            fp = Path(f).resolve()
        except OSError:
            continue
        for r in roots:
            try:
                rel = fp.relative_to(r)
            except ValueError:
                continue
            files.add(str(Path("vllm-poc") / rel) if r != root else str(rel))
            break
    nc = _sys.modules.get("verity_vllm.commit.committer.native_collect")
    jit = []
    if nc is not None and getattr(nc, "_EXT", None) is not None:
        here = Path(getattr(nc, "HERE", str(root / "verity_vllm" / "commit" / "committer")))
        jit = [str(Path(here, n).resolve().relative_to(root)) for n in ("native_collect.cpp", "native_gather.cu", "native_tree.cu", "native_plans.h") if Path(here, n).exists()]
        files.update(jit)
    return {"method": "runtime: sys.modules of this process filtered to the repo root + vllm-poc, plus native_collect's cpp_extension sources when loaded (TP-11)",
            "pid": os.getpid(), "n": len(files), "files": sorted(files), "native_collect_jit_sources": jit}

EXTENSION = "verity_vllm.engine.rank_worker.TP2CaptureWorkerExtension"


def _moe_collective_modules() -> list[Any]:
    import importlib

    mods = []
    for name in MOE_COLLECTIVE_MODULES:
        try:
            m = importlib.import_module(name)
        except Exception:  # noqa: BLE001 -- older/newer vLLM without this module
            continue
        if hasattr(m, "tensor_model_parallel_all_reduce"):
            mods.append(m)
    return mods


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


class _LLMShim:
    """What `vllm_adapter.profile_manifest` / `model_config_subset` / `throwaway_requests` read from an `LLM`: the
    `vllm_config`, and `collective_rpc(fn)` — here the probe runs on this rank only."""

    def __init__(self, worker: Any) -> None:
        self.llm_engine = SimpleNamespace(vllm_config=worker.vllm_config)
        self._worker = worker

    def collective_rpc(self, fn: Any, *a: Any, **k: Any) -> list:
        return [fn(self._worker)]


def _tp_facts(worker: Any) -> dict[str, Any]:
    import torch
    from vllm.distributed import parallel_state as ps

    out: dict[str, Any] = {"pid": os.getpid(), "ppid": os.getppid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                           "worker_class": type(worker).__mro__[1].__module__ + "." + type(worker).__mro__[1].__name__,
                           "worker_mro": [f"{c.__module__}.{c.__name__}" for c in type(worker).__mro__[:4]],
                           "runner_class": f"{type(worker.model_runner).__module__}.{type(worker.model_runner).__name__}"}
    try:
        out["device"] = {"current": int(torch.cuda.current_device()), "name": torch.cuda.get_device_name(),
                         "count_visible": int(torch.cuda.device_count())}
    except Exception as e:  # noqa: BLE001
        out["device_error"] = repr(e)
    try:
        tp = ps.get_tp_group()
        comm = getattr(tp, "device_communicator", None)
        out["tp"] = {"rank": int(tp.rank_in_group), "world_size": int(tp.world_size), "global_rank": int(tp.rank),
                     "group_name": getattr(tp, "unique_name", None), "use_custom_op_call": getattr(tp, "use_custom_op_call", None),
                     "device_communicator": f"{comm.__class__.__module__}.{comm.__class__.__name__}" if comm is not None else None}
        for attr in ("ca_comm", "qr_comm", "symm_mem_comm", "pynccl_comm", "fi_allreduce_fusion", "all2all_manager"):
            c = getattr(comm, attr, None)
            out["tp"][attr] = None if c is None else {"class": c.__class__.__name__, "disabled": bool(getattr(c, "disabled", False)),
                                                        "max_size": int(getattr(c, "max_size", 0) or 0), "available": bool(getattr(c, "available", True))}
        out["tp"]["pp_world_size"] = int(ps.get_pp_group().world_size)
    except Exception as e:  # noqa: BLE001
        out["tp_error"] = repr(e)
    out["nccl_env"] = {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}
    pc = worker.vllm_config.parallel_config
    out["parallel_config"] = {k: getattr(pc, k, None) for k in ("tensor_parallel_size", "pipeline_parallel_size", "world_size",
                                                                 "distributed_executor_backend", "disable_custom_all_reduce",
                                                                 "worker_cls", "worker_extension_cls")}
    out["parallel_config"] = _jsonable(out["parallel_config"])
    return out


def parse_swap_spec(spec: list | None) -> dict | None:
    """`[ra, rb]` = swap at EVERY step (rev-cb M-0359 T6-3); `[ra, rb, lo, hi]` = swap ONLY on engine steps lo..hi inclusive (rev M-0460: the
    decode-only swap -- prefill rows untouched, every swapped row one token, coverage/composition blind; attribution alone must catch it)."""
    if not spec or len(spec) < 2:
        return None
    ra, rb = str(spec[0]), str(spec[1])
    if len(spec) >= 4 and spec[2] is not None and spec[3] is not None:
        return {"a": ra, "b": rb, "lo": int(spec[2]), "hi": int(spec[3]), "mode": "steps"}
    return {"a": ra, "b": rb, "lo": None, "hi": None, "mode": "all"}


def swap_label(label: str, sw: dict | None, step: int | None) -> str:
    """The label a request's rows carry under the swap negative at `step` (None = no step context: the ALL-steps swap applies, a ranged one
    does not -- a ranged swap never touches a step-less consumer)."""
    if not sw:
        return label
    if sw.get("mode") == "steps":
        if step is None or not (int(sw["lo"]) <= int(step) <= int(sw["hi"])):
            return label
    if label == sw["a"]:
        return sw["b"]
    if label == sw["b"]:
        return sw["a"]
    return label


class TP2CaptureWorkerExtension:
    """Mixed into vLLM's Worker (`worker_extension_cls`); `self` is the Worker of one rank."""

    # ------------------------------------------------------------------ topology

    def tp2_probe(self) -> dict[str, Any]:
        """Per-rank facts: process, device, TP group, communicator choice, model/runner classes, sharded shapes."""
        out = _tp_facts(self)
        try:
            model = self.model_runner.get_model()
            shapes = {}
            for name, p in model.named_parameters():
                if name.startswith("model.layers.0.") or not name.startswith("model.layers."):
                    shapes[name] = list(p.shape)
            out["param_shapes"] = shapes
            out["n_params"] = sum(1 for _ in model.named_parameters())
            out["modules"] = {name: type(m).__name__ for name, m in model.named_modules()
                              if name in ("model.embed_tokens", "lm_head", "logits_processor", "model.layers.0.self_attn.qkv_proj",
                                          "model.layers.0.self_attn.o_proj", "model.layers.0.mlp.gate_up_proj", "model.layers.0.mlp.down_proj",
                                          "model.layers.0.self_attn.attn")}
            lp = getattr(model, "logits_processor", None)
            if lp is not None:
                out["logits_processor"] = {k: getattr(lp, k, None) for k in ("use_all_gather", "org_vocab_size", "vocab_size")}
        except Exception as e:  # noqa: BLE001
            out["model_error"] = repr(e)
        return _jsonable(out)

    # ------------------------------------------------------------------ observer

    def tp2_install(self, *, out_dir: str, workload: dict, case: str, cp: dict, engine_kwargs: dict, env_flags: dict,
                    snapshot_steps: list[int], snapshot_max_bytes: int, adapter_revision: str, wheel_sha256: str | None,
                    vllm_commit: str | None, reenter: bool = True, triton: bool = True, module_hooks: bool = True,
                    weights_verified: bool = False, argv: list[str] | None = None) -> dict[str, Any]:
        import torch

        from verity_vllm.engine import vllm_adapter as va
        from verity_vllm.pipeline.m1_capture import dump_cos_sin

        facts = _tp_facts(self)
        rank, world = int(facts["tp"]["rank"]), int(facts["tp"]["world_size"])
        out = Path(out_dir)
        rank_dir = out / f"rank{rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        runner = self.model_runner
        model = runner.get_model()
        engine = va.Engine(llm=_LLMShim(self), runner=runner, model=model, cp=dict(cp), kwargs=dict(engine_kwargs),
                           env_flags=dict(env_flags), build_s=0.0, case=case)
        header = va.make_header(engine, workload, adapter_revision=adapter_revision, wheel_sha256=wheel_sha256, vllm_commit=vllm_commit,
                                extra_notes={"tp2": {"rank": rank, "world_size": world, "log": f"log.rank{rank}.jsonl.gz",
                                                     "seam": "worker_extension_cls " + EXTENSION, "facts": facts},
                                             "snapshot_steps": sorted(snapshot_steps), "weights_verified": bool(weights_verified),
                                             "argv": list(argv or [])})
        # the protected adapter builds the manifest with tensor_parallel_size=1 (its signature has no TP argument): record
        # the truth in the manifest copy this rank writes and derive the profile id from it (PROPOSED: profile_manifest passes
        # engine.kwargs["tensor_parallel_size"] to build_manifest)
        man = header.profile
        pid_adapter = header.profile_id
        if isinstance(man, dict) and man.get("schema") == "verity-vllm/profile/v1":
            from verity_vllm.engine import engine_profile as prof
            man.setdefault("parallelism", {})["tensor_parallel_size"] = world
            man.setdefault("engine_kwargs", {})["tensor_parallel_size"] = world
            pid = prof.profile_id_of(man)
            header = va.RunHeader(header.schema, header.created_utc, pid, man, header.model, header.workload, dict(header.notes))
            header.notes["profile_id_method"] = (header.notes.get("profile_id_method", "") +
                                                 f"; tp2_worker: parallelism/engine_kwargs.tensor_parallel_size={world} added to the manifest "
                                                 f"before hashing (the adapter's own id was {pid_adapter})")
        cfg = va.ObserverConfig(mode="record", reenter=reenter, triton=triton, module_hooks=module_hooks, write_log=True,
                                snapshot_steps=frozenset(int(s) for s in snapshot_steps),
                                snapshot_dir=rank_dir if snapshot_steps else None, snapshot_max_bytes=int(snapshot_max_bytes))
        log_path = out / f"log.rank{rank}.jsonl.gz"

        def log(*a: Any) -> None:
            print(f"[tp2 rank {rank}] " + " ".join(str(x) for x in a), flush=True)

        cap = va.Capture(engine, header=header, log_path=log_path, config=cfg, log=log)
        cap.install()
        cs = dump_cos_sin(engine, rank_dir)
        (rank_dir / "versions.json").write_text(json.dumps(va.versions_doc(engine), indent=1) + "\n")
        torch.cuda.synchronize()
        self._tp2 = {"cap": cap, "rank": rank, "world": world, "out": out, "rank_dir": rank_dir, "header": header, "log_path": log_path,
                     "t0": time.perf_counter()}
        log(f"observer installed: {len(cap.roots)} roots, profile_id={header.profile_id} (adapter {pid_adapter}), log={log_path}, "
            f"snapshots={sorted(snapshot_steps)[:4]}{'...' if len(snapshot_steps) > 4 else ''}, cos_sin={cs}")
        return _jsonable({"rank": rank, "world_size": world, "pid": os.getpid(), "roots": len(cap.roots), "profile_id": header.profile_id,
                          "profile_id_adapter": pid_adapter, "log": str(log_path), "rank_dir": str(rank_dir),
                          "reenter_impls": header.notes.get("reenter_impls"), "cos_sin": str(cs) if cs else None,
                          "root_names_head": [r.name for r in cap.roots[:6]], "root_shapes_head": [list(r.view.shape) for r in cap.roots[:6]]})

    def tp2_close(self) -> dict[str, Any]:
        import torch

        v = self._tp2
        cap, rank = v["cap"], v["rank"]
        torch.cuda.synchronize()
        stats = cap.close()
        run_s = time.perf_counter() - v["t0"]
        (v["out"] / f"stats.rank{rank}.json").write_text(json.dumps(stats, indent=1, default=str) + "\n")
        per_step = {str(s): ids for s, ids in cap.sample_ids.items()}
        rows = {str(s): r for s, r in cap.step_rows.items()}
        (v["out"] / f"tokens.rank{rank}.json").write_text(json.dumps({"rank": rank, "per_step_sampled": per_step, "step_rows": rows,
                                                                       "profile_id": v["header"].profile_id}, indent=1) + "\n")
        try:
            size = v["log_path"].stat().st_size
        except OSError:
            size = None
        print(f"[tp2 rank {rank}] closed: {stats['events_written']} events, {stats['steps']} steps, recorder={stats['recorder']}, "
              f"snapshot_bytes={stats['snapshot_bytes']}, errors={len(stats['errors'])}, log {size} B, {run_s:.1f}s", flush=True)
        return _jsonable({"rank": rank, "events_written": stats["events_written"], "steps": stats["steps"], "recorder": stats["recorder"],
                          "snapshot_bytes": stats["snapshot_bytes"], "snapshot_files": stats["snapshot_files"], "errors": stats["errors"][:5],
                          "n_errors": len(stats["errors"]), "roots": stats["roots"], "log_bytes": size, "per_step_sampled": per_step,
                          "timing": stats["timing"], "storage": stats["storage"]})

    # ------------------------------------------------------------------ root sharding evidence

    def tp2_param_hashes(self, *, out_dir: str) -> dict[str, Any]:
        """sha256 + shape of every parameter/buffer shard this rank holds, and the sharding attributes of the parallel
        layers (`tp2_shards.py` on the pod maps every shard hash to the checkpoint slice it equals)."""
        import hashlib

        import torch

        rank = int(_tp_facts(self)["tp"]["rank"])
        model = self.model_runner.get_model()
        params: dict[str, Any] = {}
        for kind, it in (("param", model.named_parameters()), ("buffer", model.named_buffers())):
            for name, p in it:
                t = p.detach().contiguous().cpu()
                raw = t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""
                params[name] = {"kind": kind, "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
                                "sha256": hashlib.sha256(raw).hexdigest(), "nbytes": len(raw)}
        mods: dict[str, Any] = {}
        attrs = ("tp_size", "tp_rank", "input_size", "output_size", "input_size_per_partition", "output_size_per_partition",
                 "output_sizes", "total_num_heads", "total_num_kv_heads", "num_heads", "num_kv_heads", "head_size",
                 "num_embeddings", "num_embeddings_padded", "num_embeddings_per_partition", "org_vocab_size", "embedding_dim",
                 "reduce_results", "gather_output", "skip_bias_add", "use_fused_embedding")
        for name, m in model.named_modules():
            cls = type(m).__name__
            if not (cls.endswith(("ParallelLinear", "Embedding", "LMHead")) or cls == "LogitsProcessor"):
                continue
            d = {k: getattr(m, k, None) for k in attrs if hasattr(m, k)}
            si = getattr(m, "shard_indices", None)
            if si is not None:
                d["shard_indices"] = {k: getattr(si, k) for k in dir(si) if not k.startswith("_") and isinstance(getattr(si, k), int)}
            d["class"] = f"{type(m).__module__}.{cls}"
            mods[name] = d
        doc = {"schema": "verity-capture/tp2-param-hashes/v0", "rank": rank, "world_size": int(_tp_facts(self)["tp"]["world_size"]),
               "n_params": sum(1 for v in params.values() if v["kind"] == "param"), "n_buffers": sum(1 for v in params.values() if v["kind"] == "buffer"),
               "total_bytes": sum(v["nbytes"] for v in params.values()), "tensors": params, "modules": _jsonable(mods)}
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"param_hashes.rank{rank}.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
        return {"rank": rank, "n_params": doc["n_params"], "n_buffers": doc["n_buffers"], "total_bytes": doc["total_bytes"],
                "modules": len(mods)}

    # ------------------------------------------------------------------ R13 P4: TP2 numerical-semantics probe

    def tp2_sem_install(self, *, out_dir: str, layers: list[int], max_calls: int = 3) -> dict[str, Any]:
        """Record, for the first `max_calls` forward calls of the parallel layers of `layers` (+ embed_tokens, logits_processor),
        the rank-local INPUT, the rank-local partial GEMM output (the argument of the all-reduce / all-gather), the collective
        OUTPUT, and the module output, plus this rank's weight shards.  `tp2_semantics_analyze.py` recomputes the TP1 and TP2
        semantics offline and compares bit-exactly.  Collectives are seen by wrapping the module-global names
        `tensor_model_parallel_all_reduce` / `_all_gather` in vLLM's linear / vocab_parallel_embedding / logits_processor modules
        while a hooked module is executing (the wrapper is pass-through otherwise; the run is unchanged)."""
        import torch
        import vllm.model_executor.layers.linear as L
        import vllm.model_executor.layers.logits_processor as LP
        import vllm.model_executor.layers.vocab_parallel_embedding as VE

        facts = _tp_facts(self)
        rank, world = int(facts["tp"]["rank"]), int(facts["tp"]["world_size"])
        out = Path(out_dir) / f"rank{rank}"
        out.mkdir(parents=True, exist_ok=True)
        model = self.model_runner.get_model()
        names = ["model.embed_tokens", "logits_processor"]
        for li in layers:
            names += [f"model.layers.{li}.self_attn.qkv_proj", f"model.layers.{li}.self_attn.o_proj",
                      f"model.layers.{li}.mlp.gate_up_proj", f"model.layers.{li}.mlp.down_proj"]
        mods = dict(model.named_modules())
        st: dict[str, Any] = {"active": None, "calls": {n: 0 for n in names}, "handles": [], "hooks": hooks.Hooks(), "rank": rank, "out": out,
                              "max_calls": int(max_calls), "records": []}
        self._tp2_sem = st

        def cpu(t: Any) -> Any:
            if isinstance(t, torch.Tensor):
                return t.detach().to("cpu", copy=True)
            if t is None or isinstance(t, (int, float, str, bool)):
                return t
            return f"<{type(t).__module__}.{type(t).__name__}>"   # e.g. the lm_head module passed to LogitsProcessor.forward

        def coll_wrap(orig: Any, kind: str) -> Any:
            def w(x: Any, *a: Any, **k: Any) -> Any:
                y = orig(x, *a, **k)
                act = st["active"]
                if act is not None:
                    torch.cuda.synchronize()
                    act["coll"].append({"kind": kind, "in": cpu(x), "out": cpu(y), "args": [str(v) for v in a], "kwargs": {kk: str(v) for kk, v in k.items()}})
                return y
            return w

        for mod, attr, kind in ((L, "tensor_model_parallel_all_reduce", "all_reduce"), (VE, "tensor_model_parallel_all_reduce", "all_reduce"),
                                (LP, "tensor_model_parallel_all_gather", "all_gather"), (LP, "tensor_model_parallel_gather", "gather")):
            if hasattr(mod, attr):
                st["hooks"].wrap(mod, attr, lambda orig, kind=kind: coll_wrap(orig, kind))

        def pre(name: str) -> Any:
            def h(m: Any, args: tuple, kwargs: dict) -> None:
                c = st["calls"][name]
                if c >= st["max_calls"] or st["active"] is not None:
                    return
                torch.cuda.synchronize()
                st["active"] = {"name": name, "call": c, "in": [cpu(a) for a in args], "in_kw": {k: cpu(v) for k, v in kwargs.items()}, "coll": []}
            return h

        def post(name: str) -> Any:
            def h(m: Any, args: tuple, kwargs: dict, output: Any) -> None:
                act = st["active"]
                if act is None or act["name"] != name:
                    return
                torch.cuda.synchronize()
                o = output[0] if isinstance(output, tuple) else output
                act["out"] = cpu(o)
                act["out_extra"] = [cpu(x) for x in output[1:]] if isinstance(output, tuple) else []
                act["module_class"] = type(m).__name__
                act["module_attrs"] = _jsonable({k: getattr(m, k, None) for k in ("tp_size", "tp_rank", "input_size", "output_size",
                                                 "input_size_per_partition", "output_size_per_partition", "output_sizes", "reduce_results",
                                                 "gather_output", "skip_bias_add", "num_embeddings", "num_embeddings_per_partition",
                                                 "org_vocab_size", "embedding_dim", "use_all_gather", "input_is_parallel") if hasattr(m, k)})
                torch.save(act, out / f"{name}.call{act['call']}.pt")
                st["records"].append({"name": name, "call": act["call"], "n_coll": len(act["coll"]),
                                      "in_shape": [list(a.shape) for a in act["in"] if isinstance(a, torch.Tensor)],
                                      "out_shape": list(act["out"].shape) if isinstance(act["out"], torch.Tensor) else None})
                st["calls"][name] = act["call"] + 1
                st["active"] = None
            return h

        for n in names:
            m = mods.get(n)
            if m is None:
                st["records"].append({"name": n, "missing": True})
                continue
            st["handles"].append(m.register_forward_pre_hook(pre(n), with_kwargs=True))
            st["handles"].append(m.register_forward_hook(post(n), with_kwargs=True))
        # weight shards of the hooked modules (+ lm_head, which logits_processor uses without a module call)
        W: dict[str, Any] = {}
        for n in names + ["lm_head"]:
            m = mods.get(n)
            if m is None:
                continue
            for pn, p in m.named_parameters(recurse=False):
                W[f"{n}.{pn}"] = cpu(p)
            for bn, b in m.named_buffers(recurse=False):
                W[f"{n}.{bn}"] = cpu(b)
        lm = mods.get("lm_head")
        meta = {"rank": rank, "world_size": world, "facts": facts, "layers": list(layers), "names": names,
                "lm_head_is_embed": bool(lm is not None and getattr(lm, "weight", None) is not None and
                                        lm.weight.data_ptr() == mods["model.embed_tokens"].weight.data_ptr()),
                "weights": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in W.items()}}
        torch.save(W, out / "weights.pt")
        (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
        return _jsonable({"rank": rank, "world_size": world, "hooked": [n for n in names if n in mods], "weights": list(W)})

    def tp2_sem_close(self) -> dict[str, Any]:
        import torch

        st = self._tp2_sem
        torch.cuda.synchronize()
        for h in st["handles"]:
            h.remove()
        st["hooks"].uninstall()
        (st["out"] / "records.json").write_text(json.dumps(st["records"], indent=1, default=str) + "\n")
        return _jsonable({"rank": st["rank"], "records": st["records"]})

    # ------------------------------------------------------------------ R13 P4 step 3: every collective of the run, per rank, in order

    def tp2_coll_install(self, *, out_dir: str, store_steps: int = 1 << 30) -> dict[str, Any]:
        """Record EVERY TP collective this rank issues during the instrumented run, in call order: `step` (the model forward it
        belongs to — the logits all-gather runs after the forward, before the next), `k` (index within the step), kind, the calling
        module (path + class), the partial's shape/dtype and sha256, the received output's sha256; the partial and the output
        tensors of steps < `store_steps` are saved (`coll.rank<r>.step<S>.pt`) for the offline cross-rank provenance check
        (`pipeline/tp/match.py`: received == AllReduce2(part_rank0, part_rank1) / AllGather2, bit-exact).  Same wrapping seam as
        `tp2_sem_install` (module-global collective names); the run's computation is unchanged."""
        import hashlib

        import torch
        import vllm.model_executor.layers.linear as L
        import vllm.model_executor.layers.logits_processor as LP
        import vllm.model_executor.layers.vocab_parallel_embedding as VE

        facts = _tp_facts(self)
        rank, world = int(facts["tp"]["rank"]), int(facts["tp"]["world_size"])
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        model = self.model_runner.get_model()
        st: dict[str, Any] = {"rank": rank, "world": world, "out": out, "step": -1, "k": 0, "cur": None, "handles": [], "hooks": hooks.Hooks(),
                              "moe_sites": _moe_collective_modules(),
                              "records": [], "values": {"part": [], "out": []}, "store_steps": int(store_steps), "n_stored": 0}
        self._tp2_coll = st

        def sha(t: torch.Tensor) -> str:
            return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes() if t.dtype == torch.bfloat16 else t.contiguous().numpy().tobytes()).hexdigest()

        def flush_step() -> None:
            if st["step"] >= 0 and st["step"] < st["store_steps"] and st["values"]["part"]:
                torch.save({"step": st["step"], "part": st["values"]["part"], "out": st["values"]["out"]}, out / f"coll.rank{rank}.step{st['step']}.pt")
                st["n_stored"] += 1
            st["values"] = {"part": [], "out": []}

        def model_pre(m: Any, args: tuple, kwargs: dict) -> None:
            torch.cuda.synchronize()
            flush_step()
            st["step"] += 1
            st["k"] = 0

        def observer_rec() -> Any:
            """[R17 tp D90] the aten observer of `tp2_install`, when this rank runs BOTH (one engine run = one fold + one recorder): the
            recorder's own bookkeeping (device->host copies, digests) is hidden from it, the collective launch itself stays observed."""
            cap = (getattr(self, "_tp2", None) or {}).get("cap")
            return getattr(cap, "rec", None) if cap is not None else None

        def coll_wrap(orig: Any, kind: str) -> Any:
            def w(x: Any, *a: Any, **k: Any) -> Any:
                y = orig(x, *a, **k)
                torch.cuda.synchronize()
                rec = observer_rec()
                was = rec.active if rec is not None else None
                if rec is not None:
                    rec.active = False
                try:
                    xc, yc = x.detach().to("cpu", copy=True), y.detach().to("cpu", copy=True)
                    cur = st["cur"] or {}
                    st["records"].append({"rank": rank, "step": st["step"], "k": st["k"], "kind": kind, "module": cur.get("name"), "module_class": cur.get("cls"),
                                          "shape": list(xc.shape), "dtype": str(xc.dtype).replace("torch.", ""), "out_shape": list(yc.shape),
                                          "part_sha256": sha(xc), "out_sha256": sha(yc), "args": [str(v) for v in a]})
                    if st["step"] < st["store_steps"]:
                        st["values"]["part"].append(xc)
                        st["values"]["out"].append(yc)
                finally:
                    if rec is not None:
                        rec.active = was
                st["k"] += 1
                return y
            return w

        # [R15 moe TP-MOE-AR] the fused-MoE block's FINAL all-reduce (RowParallel-style reduction of the expert partials, one per MoE
        # layer per step) is issued from vLLM's fused_moe runner module through ITS OWN module-level import of
        # `tensor_model_parallel_all_reduce`, not from linear/vocab_parallel_embedding/logits_processor -- patching only those three left
        # every MoE all-reduce unrecorded (Qwen3-30B-A3B TP2 on 2x L40S: 50 recorded collectives per rank per step vs the rank Build's
        # 98-link peer-input signature = 1 + 48 o_proj + 48 mlp.experts + 1; provenance/tokens PASS, structure FAIL on every step).
        # The Build's signature names the site `layers.N.mlp.experts (MoERunner)`; the same modules are hooked below for the name.
        sites = [(L, "tensor_model_parallel_all_reduce", "all_reduce"), (VE, "tensor_model_parallel_all_reduce", "all_reduce"),
                 (LP, "tensor_model_parallel_all_gather", "all_gather"), (LP, "tensor_model_parallel_gather", "gather")]
        sites += [(mod, "tensor_model_parallel_all_reduce", "all_reduce") for mod in st["moe_sites"]]
        # [R19 moe F-r19-moe-1, row #70] MID-module sites: a model class that imports a collective name into ITS OWN module and calls it
        # from inside its forward (OlmoeAttention._apply_qk_norm: q and k all-gathered, twice per layer) is invisible to the wrappers
        # above -- R17 row #70 recorded 34 collectives per step against the Build's 66 links (structure FAIL).  The one census shared
        # with the committer's TPPartialSource names the modules (hooked below) and the python modules to patch, a ruled-out call's too.
        from verity_vllm.collectives.collective_sites import mid_module_site_census, patch_targets
        census = mid_module_site_census(model, world)
        st["mid_sites"] = {s["name"]: s for s in census["sites"]}
        sites += patch_targets(census["sites"] + census["ruled_out"])
        wrapped: set[tuple[str, str]] = set()
        for mod, attr, kind in sites:
            if hasattr(mod, attr) and (mod.__name__, attr) not in wrapped:
                wrapped.add((mod.__name__, attr))
                st["hooks"].wrap(mod, attr, lambda orig, kind=kind: coll_wrap(orig, kind))

        # [R15 rev-cb T6-1, mirrored on the Match side] `cur` is a stack: a collective issued by an outer module after a nested hooked module
        # returned is attributed to the OUTER module; a fused-MoE runner nested under a hooked fused-MoE wrapper is NOT hooked, so the
        # block's all-reduce is recorded at the outermost MoE module = the Program's module path (the same site the committer binds and
        # the site tp_links names) -- the site-identity criterion compares record.module to that path per k.
        st["cur_stack"] = []

        def pre(name: str, cls: str) -> Any:
            def h(m: Any, args: tuple, kwargs: dict) -> None:
                if st["cur"] is not None:
                    st["cur_stack"].append(st["cur"])
                st["cur"] = {"name": name, "cls": cls}
            return h

        def post(m: Any, args: tuple, kwargs: dict, output: Any) -> None:
            st["cur"] = st["cur_stack"].pop() if st["cur_stack"] else None

        cand = [(n, type(m).__name__, m) for n, m in model.named_modules()
                if type(m).__name__ in ("RowParallelLinear", "VocabParallelEmbedding", "LogitsProcessor", "ParallelLMHead") + MOE_COLLECTIVE_CLASSES
                or getattr(m, "reduce_results", False) is True
                or n in st["mid_sites"]]                                          # [R19 moe] mid-module sites: hooked for the module name
        moe_paths = [n for n, c, _m in cand if c in MOE_COLLECTIVE_CLASSES]
        nested_moe = [n for n, c, _m in cand if c in MOE_COLLECTIVE_CLASSES and any(n.startswith(p + ".") for p in moe_paths)]
        hooked = []
        for n, cls, m in cand:
            if n in nested_moe:
                continue
            st["handles"].append(m.register_forward_pre_hook(pre(n, cls), with_kwargs=True))
            st["handles"].append(m.register_forward_hook(post, with_kwargs=True))
            hooked.append(n)
        st["handles"].append(model.register_forward_pre_hook(model_pre, with_kwargs=True))
        meta = {"rank": rank, "world_size": world, "facts": facts, "hooked": hooked, "nested_moe_sites_dropped": nested_moe, "store_steps": int(store_steps),
                "mid_module_sites": {n: {"cls": s["cls"], "py_module": s["py_module"], "calls": s["calls"]} for n, s in st["mid_sites"].items()}, "mid_module_sites_ruled_out": census["ruled_out"],
                "attribution_rule": "outermost hooked module (cur is a stack; a fused-MoE runner nested under a hooked fused-MoE wrapper is not hooked)"}
        (out / f"coll_meta.rank{rank}.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
        return _jsonable({"rank": rank, "world_size": world, "n_hooked": len(hooked), "mid_module_sites_ruled_out": len(census["ruled_out"])})

    def tp2_coll_close(self) -> dict[str, Any]:
        import torch

        st = self._tp2_coll
        torch.cuda.synchronize()
        # the last step's values (no further model forward flushed them)
        if st["step"] >= 0 and st["step"] < st["store_steps"] and st["values"]["part"]:
            torch.save({"step": st["step"], "part": st["values"]["part"], "out": st["values"]["out"]}, st["out"] / f"coll.rank{st['rank']}.step{st['step']}.pt")
            st["n_stored"] += 1
        for h in st["handles"]:
            h.remove()
        st["hooks"].uninstall()
        with (st["out"] / f"coll.rank{st['rank']}.jsonl").open("w") as f:
            for r in st["records"]:
                f.write(json.dumps(r) + "\n")
        by_kind: dict[str, int] = {}
        for r in st["records"]:
            by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        return _jsonable({"rank": st["rank"], "n_steps": st["step"] + 1, "n_collectives": len(st["records"]), "by_kind": by_kind, "n_stored_steps": st["n_stored"]})

    # ------------------------------------------------------------------ R13 P4 step 4: rank-aware commitment (the collector in every rank)

    def tp2_commit_make(self, *, kind: str, run_id: str, cp: dict, case: str, engine_kwargs: dict, hidden_so: str | None, hidden_ns: str | None,
                        envelope: dict, arena_mb: int | None = None, tp_partials: bool = True, program_digest: str | None = None,
                        n_requests: int = 1, required_manifest: str | None = None, request_order: list[str] | None = None,
                        arrivals: dict[str, int] | None = None, caps: dict[str, int] | None = None, tp_relabel: dict | None = None,
                        tp_partials_omit_rank: int | None = None, tp_partials_omit_site: dict | None = None,
                        request_ids: dict[str, str] | None = None, match_tokens: dict[str, list[int]] | None = None,
                        tp_swap_requests: dict | None = None, match_capture_dir: str | None = None, replay: dict | None = None,
                        weights_pin: dict | None = None, flip_weight: dict | None = None, tp_partials_flip_site: dict | None = None,
                        serve_shard_from: dict | None = None, tap_sources: Any = None) -> dict[str, Any]:
        """Construct THIS rank's committer (`commit_delta.make_committer` on a per-rank engine shim: this worker's model runner + model,
        the rank's loaded weight SHARDS), attach the FA2 hidden source, size the arena from the workload envelope, register the rank's
        weights.  The rank's root binds its own values (its shards' roots, its partial GEMMs, its FA2 hidden stream for its heads); the
        driver (`pipeline/tp/commit.py`) combines the ranks' roots deterministically and fails closed on a missing rank."""
        from types import SimpleNamespace
        from verity_vllm.pipeline.commit import baseline_state, make_committer

        facts = _tp_facts(self)
        rank = int(facts["tp"]["rank"])
        runner = self.model_runner
        model = runner.get_model()
        eng = SimpleNamespace(llm=_LLMShim(self), runner=runner, model=model, cp=cp, kwargs=engine_kwargs, env_flags={}, build_s=0.0, case=case)
        args = SimpleNamespace(arena_mb=arena_mb, chunk=None, hash_threads=None, profile_hooks=False, fast_copy=True, gpu_tree=True)
        com = make_committer(kind, eng, args, f"{run_id}/rank{rank}")
        res: dict[str, Any] = {"rank": rank, "committer": com.name}
        if hidden_so:
            from verity_vllm.acquire.sources.hidden_source import HiddenM1Source, make_hidden_source
            man = json.loads(Path(required_manifest).read_text()) if required_manifest else None      # its ROW word 3 / MS class must be the build's
            com.attach_source(make_hidden_source(com, hidden_so, ns=(hidden_ns or Path(hidden_so).stem), manifest=man,
                                                 fa2_cls=getattr(com, "hidden_source_cls", HiddenM1Source), fa3_cls=getattr(com, "hidden_source_cls_fa3", None)))
            res["hidden_source"] = hidden_so
        # [R15 moe TP-MOE-SRC] the fused-MoE block source (R13 MOE-03: router_w/ids, up, act, down, sum per MoE layer per step, class
        # moe_block_stream), attached exactly as commit_delta's `--moe-source auto` does: when the rank's model has MoE-runner modules.
        # Without it a TP MoE row's Commit binds no `moe/L<n>` plane (Qwen3-30B-A3B TP2 B2: 37 440 of 132 708 required identities per
        # rank unbound, coverage FAIL) -- the manifest requires the six planes per rank at shard widths, and this rank's kernel sees them.
        try:
            from verity_vllm.acquire.sources.moe_source import MoeBlockSource

            msrc = MoeBlockSource(com, model)
            if msrc.layers:
                com.attach_source(msrc)
                res["moe_source"] = {"layers": len(msrc.layers), "tensors_per_layer": msrc.stats.get("tensors_per_layer"),
                                     "modules": sorted(msrc.layers)[:4] + (["..."] if len(msrc.layers) > 4 else [])}
            else:
                res["moe_source"] = None
        except Exception as e:  # noqa: BLE001 -- a model without vLLM's fused_moe modules; the rank reports it and binds no MoE plane
            res["moe_source"] = {"error": repr(e)[:200]}
        if tap_sources is not None:                   # the taps the row's manifest names: `acquire.sources.taps.for_ranks`, bound by the TP driver
            res["tap_sources"] = tap_sources(com, model, required_manifest, rank)
        if tp_partials_omit_rank is not None and int(tp_partials_omit_rank) == rank:
            tp_partials = False          # NEGATIVE (R15 TP-04): THIS rank binds collective outputs only; its manifest partials must go unbound
        if tp_partials:
            # [R14 TP-03] bind THIS rank's rank-local inputs of every TP collective (partial GEMMs before the all-reduce, masked embedding
            # partial, logits shard before the all-gather) -- the values the rank Program requires; the module hooks only see the outputs
            from verity_vllm.acquire.sources.partial_source import TPPartialSource

            relabel = None
            if isinstance(tp_relabel, dict) and str(rank) in {str(k) for k in tp_relabel}:
                relabel = int(tp_relabel.get(str(rank), tp_relabel.get(rank)))       # NEGATIVE: this rank tags its partials as another rank's
            omit_site = None
            if isinstance(tp_partials_omit_site, dict) and str(rank) in {str(k) for k in tp_partials_omit_site}:
                omit_site = str(tp_partials_omit_site.get(str(rank), tp_partials_omit_site.get(rank)))   # NEGATIVE (rev-cb M-0214): ONE site's partial dropped
            flip_site = None
            if isinstance(tp_partials_flip_site, dict) and str(rank) in {str(k) for k in tp_partials_flip_site}:
                flip_site = str(tp_partials_flip_site.get(str(rank), tp_partials_flip_site.get(rank)))   # NEGATIVE (R16 TP-12): the collective's input is not the committed partial
            tps = TPPartialSource(com, model, rank=rank, world=int(facts["tp"]["world_size"]), relabel_rank=relabel, omit_site=omit_site, flip_site=flip_site)
            com.attach_source(tps)
            res["tp_partial_source"] = {"sites": tps.sites, "extra_step_bytes": tps.extra_step_bytes(envelope, model), "relabel_rank": relabel, "omit_site": omit_site,
                                        "flip_site": flip_site}
        else:
            res["tp_partial_source"] = None      # negative: collective outputs only (the R13 state) -- must fail the required classes
        if hasattr(com, "size_arena") and getattr(com, "arena", None) is not None:
            res["arena_sizing"] = _jsonable(com.size_arena(envelope, hidden_m1=bool(hidden_so)))
            if getattr(com, "windowed", False):
                # [R13 tp / P0-01 parity] windowed staging (VERITY_WINDOW_MB > 0): the ring replaces the per-step blocks -- the same call
                # commit_delta makes after size_arena; without it every windowed step fails closed (`win_log2_chunks` unset -> 0 tensors).
                res["window_ring"] = _jsonable(com.enable_windows(com.arena_sizing))
        # NEGATIVE (rev-cb M-0610 flipweight, TP form): {"rank": r, "param": <served name>, "index": i, "value": v} -- ONE element of THIS
        # rank's shard is changed BEFORE register_weights (the committer serves a model that is not the checkpoint the Program names)
        res["flip_weight"] = None
        if isinstance(flip_weight, dict) and int(flip_weight.get("rank", -1)) == rank:
            import torch
            pmap = dict(model.named_parameters())
            pname = str(flip_weight["param"])
            if pname not in pmap:
                raise RuntimeError(f"rank {rank}: --flip-weight names no parameter {pname!r} on this rank")
            with torch.no_grad():
                flat = pmap[pname].data.view(-1)
                idx = int(flip_weight.get("index", 0)) % flat.numel()
                before = float(flat[idx].item())
                flat[idx] = float(flip_weight.get("value", 1.0))
            res["flip_weight"] = {"rank": rank, "param": pname, "index": idx, "before": before, "after": float(flat[idx].item())}
        # NEGATIVE (R16 tp TP-10, pin's acceptance set 'TP shard / reference mismatch'): THIS rank serves ANOTHER rank's shard of a parameter,
        # composed from the SAME pinned checkpoint by weights_of_record.compose(rank=<from>) -- every byte is checkpoint bytes and the set of
        # shards over the world is unchanged (a whole-tensor / whole-checkpoint hash is blind to it); only the rank<->slice assignment the rank
        # Program's shard map prescribes is wrong.  Applied BEFORE register_weights (the rank commits what it serves); the per-rank pin,
        # composed for THIS rank, must FAIL naming the parameter.  A replicated parameter cannot be mis-served this way (refused by name).
        res["serve_shard_from"] = None
        ssf = list((serve_shard_from or {}).get(str(rank)) or []) if isinstance(serve_shard_from, dict) else []
        if ssf:
            if not (isinstance(weights_pin, dict) and weights_pin.get("checkpoint_dir")):
                raise RuntimeError(f"rank {rank}: --serve-shard-from needs --checkpoint-dir (the other rank's shard is composed from the pinned checkpoint)")
            import hashlib
            import torch
            import numpy as _np
            from verity_vllm.check import weights_of_record as WR
            pmap = dict(model.named_parameters())
            ck = WR.CheckpointIndex(str(weights_pin["checkpoint_dir"]), checkpoints_manifest=weights_pin.get("checkpoints_manifest"),
                                    model=weights_pin.get("model"), verify_shards=False)
            world_ = int(facts["tp"]["world_size"])
            applied = []
            for s in ssf:
                pname, frm = str(s["param"]), int(s["from_rank"])
                if pname not in pmap:
                    raise RuntimeError(f"rank {rank}: --serve-shard-from names no parameter {pname!r} on this rank")
                if not (0 <= frm < world_) or frm == rank:
                    raise RuntimeError(f"rank {rank}: --serve-shard-from from_rank {frm} is not another rank of world {world_}")
                p = pmap[pname]
                bits = int(p.element_size()) * 8
                arr, note = WR.compose(pname, list(p.shape), bits, ck, world=world_, rank=frm)
                if "replicated" in str(note.get("shard") or ""):
                    raise RuntimeError(f"rank {rank}: {pname} is replicated on every rank ({note.get('shard')}) -- no other rank's shard to serve")
                own, _ = WR.compose(pname, list(p.shape), bits, ck, world=world_, rank=rank)
                with torch.no_grad():
                    before = hashlib.sha256(p.data.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                    t = torch.from_numpy(_np.ascontiguousarray(arr)).view(p.dtype).reshape(p.shape)
                    p.data.copy_(t.to(p.device))
                    after = hashlib.sha256(p.data.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                own_sha = hashlib.sha256(_np.ascontiguousarray(own).tobytes()).hexdigest()
                applied.append({"rank": rank, "param": pname, "from_rank": frm, "shard_rule": note.get("shard"), "sources": note.get("sources"),
                                "served_before_sha256": before, "served_after_sha256": after, "own_shard_of_record_sha256": own_sha,
                                "own_equals_before": own_sha == before, "after_equals_before": after == before})
            res["serve_shard_from"] = applied
        res["weight_registration"] = _jsonable(com.register_weights())
        # [manbind MAN-09, COORD M-0614 (1)] WEIGHTS OF RECORD per rank: sha256 of every live parameter of THIS rank (the shards the root
        # covers) == the digest composed from the PINNED checkpoint shards by the rank Program's shard map (weights_of_record.build_record,
        # engine-free, from result.json tp {world, rank}); self-consistency with the run's own root is not this check.  Absent -> unpinned
        # (named); the driver fails closed on it with --required-manifest unless told not to.
        res["weights_pin"] = {"rank": rank, "result": None, "why": "weights unpinned (MAN-09): no checkpoint of record given to this rank"}
        if isinstance(weights_pin, dict) and weights_pin.get("checkpoint_dir"):
            try:
                from verity_vllm.check import weights_of_record as WR
                pdir = (weights_pin.get("program_dirs") or {}).get(str(rank))
                if not pdir:
                    raise RuntimeError(f"rank {rank}: no rank Program dir for the weights of record")
                if not os.path.exists(os.path.join(str(pdir), "instances.json.gz")):
                    # a B>=2 rank Build dir holds one request Program per (LP, T): every one declares the same rank weights struct -- take the first
                    import glob as _g
                    subs = sorted(d for d in _g.glob(os.path.join(str(pdir), "build_request*")) if os.path.exists(os.path.join(d, "instances.json.gz")))
                    if not subs:
                        raise RuntimeError(f"rank {rank}: no rank Program (instances.json.gz) under {pdir}")
                    pdir = subs[0]
                rec = WR.build_record(str(pdir), str(weights_pin["checkpoint_dir"]), checkpoints_manifest=weights_pin.get("checkpoints_manifest"),
                                      model=weights_pin.get("model"))
                rtp = (rec.get("tp") or {})
                world_ = int(facts["tp"]["world_size"])
                if int(rtp.get("world") or 1) != world_ or int(rtp.get("rank") or 0) != int(rank):
                    raise RuntimeError(f"rank {rank}: weights of record built from a Program of tp={rtp}, not this rank's (world {world_}, rank {rank})")
                live = WR.live_param_digests(model)
                chk = WR.check(rec, live)
                chk["rank"] = rank
                chk["program_dir"] = str(pdir)
                # [rev-cb M-0659 W1/W2 on MAN-09, applied on the TP record] check() alone says 'served == the files at --checkpoint-dir'; the
                # checkpoint OF RECORD needs (W1) every shard's sha256 == manifests/checkpoints.json's pin and (W2) the record built from THIS
                # rank's Program of record.  W2 holds by construction here (built in-process from the rank Program dir, never a stored file);
                # W1 is gated: unpinned / unverified shards -> result False, NAMED.
                chk["w2_program_of_record"] = {"built_in_process_from": str(pdir), "record_program_digest": rec.get("program_digest"),
                                               "rank_program_digest_of_record": (program_digest.get(str(rank)) if isinstance(program_digest, dict) else program_digest)}
                if chk.get("all_shards_pinned") is not True:
                    chk["result_before_w1"] = chk.get("result")
                    chk["result"] = False
                    chk["why"] = (f"rank {rank}: checkpoint shards NOT pinned by manifests/checkpoints.json (all_shards_pinned={chk.get('all_shards_pinned')!r}): "
                                  f"the record is over the files at the checkpoint dir, not the checkpoint of record (rev-cb M-0659 W1); " + str(chk.get("why") or ""))[:400]
                elif isinstance(chk["w2_program_of_record"]["rank_program_digest_of_record"], str) and os.path.exists(os.path.join(str((weights_pin.get("program_dirs") or {}).get(str(rank))), "instances.json.gz")) \
                        and str(rec.get("program_digest") or "") not in ("", chk["w2_program_of_record"]["rank_program_digest_of_record"]):
                    chk["result_before_w2"] = chk.get("result")
                    chk["result"] = False
                    chk["why"] = (f"rank {rank}: weights of record built from Program {str(rec.get('program_digest'))[:16]} != this rank's Program of record "
                                  f"{chk['w2_program_of_record']['rank_program_digest_of_record'][:16]} (rev-cb M-0659 W2)")
                chk["live_params"] = live.get("n")
                chk["committed_weights_root"] = (res["weight_registration"] or {}).get("root") if isinstance(res["weight_registration"], dict) else None
                res["weights_pin"] = chk
            except Exception as e:  # noqa: BLE001  -- fail closed: an unbuilt record is 'unpinned', never PASS
                res["weights_pin"] = {"rank": rank, "result": None, "why": f"rank {rank}: weights of record not evaluated: {e!r}"[:400], "error": repr(e)[:300]}
        st_pin = res["weights_pin"]
        if isinstance(program_digest, dict):
            program_digest = program_digest.get(str(rank)) or program_digest.get(rank)
        rid_map = dict(request_ids) if request_ids else {str(rid): f"r{i}" for i, rid in enumerate(request_order or [])}
        swapped = None
        swap_rule = None
        if isinstance(tp_swap_requests, dict) and str(rank) in {str(k) for k in tp_swap_requests}:
            # NEGATIVE (rev-cb M-0359 T6-3 "equal-count request-label swap"): THIS rank's request->rows map labels request a's rows as b and
            # b's as a.  With equal (prompt, max_tokens) the per-request identity COUNTS and the composition are unchanged, so coverage
            # cannot see it -- the per-(request, engine step) attribution against the Match record (sampled ids) must FAIL naming this rank.
            # [rev M-0460] `[ra, rb, lo, hi]` = the swap applies ONLY on engine steps lo..hi (decode-only: prefill rows untouched, each swapped
            # row one token) -- the discriminating case for ruling 26: coverage cannot see it, the per-(step, request) attribution must.
            swap_rule = parse_swap_spec(list(tp_swap_requests.get(str(rank), tp_swap_requests.get(rank))))
            if swap_rule is not None:
                ka = [k for k, v in rid_map.items() if v == swap_rule["a"]]
                kb = [k for k, v in rid_map.items() if v == swap_rule["b"]]
                if len(ka) == 1 and len(kb) == 1:
                    swapped = [swap_rule["a"], swap_rule["b"]] + ([swap_rule["lo"], swap_rule["hi"]] if swap_rule["mode"] == "steps" else [])
                else:
                    swap_rule = None
            res["swap_requests_applied"] = swapped
        # [lane vllm-tp-n; x03 F-r19-int-22 on the TP record] the per-layer attention plane class is DERIVED from THIS RANK's Program record
        # (`attention_impls_observed` of the rank Build: the rank-local NH/world, KVH/world, D and window each rank Call's Definition was
        # selected from) x the step's active sequences -- exactly commit_delta's wiring, on the rank's own record: a GQA layer that folds
        # on decode at world 1 (NH > KVH) may be MHA-shaped at a world where NH/world == KVH/world, so the class is per rank, never copied.
        # Without a rank record the committer keeps the R13 uniformity rule and the result says so (named, not assumed).
        res["attention_plane_classes"] = {"rank": rank, "declarations": 0, "source": None, "rule": "R13 uniformity (no rank Build record with attention_impls_observed)"}
        if hasattr(com, "identity_check"):
            try:
                from verity_vllm.acquire import fa2_plane_classes as FP
                _pdir = ((replay or {}).get("programs") or {}).get(str(rank)) or ((weights_pin or {}).get("program_dirs") or {}).get(str(rank))
                _impls, _impls_path = FP.attention_impls_of_record(program_dir=_pdir, programs_root=_pdir) if _pdir else (None, None)
                _decls = FP.declarations_from_attention_impls(_impls) if _impls else []
                if _decls:
                    com.attention_layer_decls = _decls
                    _classes = sorted(set(("windowed" if d.window is not None else "global") + ("-gqa" if d.NH > d.KVH else "-mha") for d in _decls))
                    res["attention_plane_classes"] = {"rank": rank, "declarations": len(_decls), "source": _impls_path, "classes": _classes,
                                                      "per_rank_heads": sorted({(d.NH, d.KVH, d.D) for d in _decls}),
                                                      "rule": "per-layer plane class derived per step from the rank Program's attention statics (fa2_plane_classes)"}
            except Exception as e:  # noqa: BLE001
                res["attention_plane_classes"]["error"] = repr(e)[:300]
        self._tp2_commit = {"com": com, "eng": eng, "base": baseline_state(eng), "rank": rank, "rc": None, "program_digest": program_digest,
                            "tps": (tps if tp_partials else None), "world": int(facts["tp"]["world_size"]), "n_requests": int(n_requests),
                            "required_manifest": required_manifest, "request_order": list(request_order or []), "arrivals": dict(arrivals or {}),
                            "caps": dict(caps or {}), "match_tokens": dict(match_tokens) if match_tokens else None, "swap_requests": swapped, "swap_rule": swap_rule,
                            "match_capture_dir": (str(match_capture_dir) if match_capture_dir else None), "weights_pin": st_pin,
                            # [TP-08] per-rank sampled replay config: {"programs": {"<rank>": <build dir>}, "of_record": {"<rank>": [digests]},
                            # "requests": [{request_id r<i>, arrive_step, prompt_len, max_tokens, prompt_token_ids, seed (declared)}], per_stratum, max_evaluate, seed, workers}
                            "replay": (dict(replay) if replay else None),
                            # workload request_id -> r<declared index> (COORD R15 ruling 2); the driver's map when given, else declaration order
                            "request_ids": rid_map}
        if required_manifest:
            # [R15 tp TP-04] the population of record is the Program x query manifest (never the structural table, never the collector):
            # this rank's Program digest must be the manifest's digest for this rank (a manifest of another Program = wrong Program)
            try:
                man = json.loads(Path(required_manifest).read_text())
                pdr = (man.get("program_digest_per_rank") or {}).get(str(rank))
                if pdr is None and isinstance(man.get("program_digest"), str):
                    pdr = man["program_digest"]
                if not program_digest:
                    program_digest = pdr
                    self._tp2_commit["program_digest"] = pdr
                res["required_manifest"] = {"path": required_manifest, "manifest_digest": man.get("manifest_digest"), "complete": man.get("complete"),
                                            "program_digest_for_rank": pdr, "program_digest_matches": (pdr == program_digest),
                                            "identities": (man.get("populations") or {}).get("identities"), "ranks": man.get("ranks")}
                self._tp2_commit["manifest_doc"] = man
            except Exception as e:  # noqa: BLE001
                res["required_manifest"] = {"path": required_manifest, "error": repr(e)[:300]}
        return _jsonable(res)

    def tp2_commit_install(self) -> dict[str, Any]:
        st = self._tp2_commit
        st["com"].reset()
        st["com"].install()
        st["rc"] = None
        return {"rank": st["rank"], "installed": True}

    def tp2_commit_finalize(self, checks: bool = True, defer: dict[str, Any] | None = None) -> dict[str, Any]:
        """After the run: the rank's run root (fixed BEFORE any challenge), leaf counts, coverage declaration; a warm-up (`pipeline.tp.warmup`)
        is never bound, attributed, compared or replayed.  ``checks=False``: the root only (a leased Commit's last run: `pipeline.tp.lease`).
        ``defer`` ({"out", "pair"}, `--replay-deferred`): the rank's replay bundle in place of its sampled replay (`_deferred_replay`)."""
        from verity_vllm.pipeline.commit import assert_pristine, class_coverage

        st = self._tp2_commit
        com = st["com"]
        if st.get("rc") is None:
            st["rc"] = com.finalize()
            com.uninstall()
            assert_pristine(st["eng"], st["base"])
        rc, record = st["rc"], not st.get("warmup")
        if not checks:
            return {"rank": st["rank"], "run_root": rc.run_root.hex()}
        d = st["decl"] = com.coverage_declaration(rc)
        binding = self._tp2_binding(rc) if record else None
        attribution = self._tp2_attribution(rc) if record else None
        # [coord M-0481 C2 form (B), TP family] committed partials / MoE received outputs vs the Match capture's stored collectives (same rank)
        match_oracle, match_dir = None, record and st.get("match_capture_dir")
        if match_dir and st.get("tps") is not None:
            try:
                from verity_vllm.check import oracle_compare as OC

                match_oracle = st["tps"].compare_match_oracle(st["match_capture_dir"], OC.OpenedReader(com, rc))
            except Exception as e:  # noqa: BLE001
                match_oracle = {"rank": st["rank"], "verdict": "FAIL", "error": repr(e)[:300]}
        elif match_dir:
            match_oracle = {"rank": st["rank"], "verdict": "FAIL", "why": "no tp partial source on this rank (partials not bound)"}
        # [manbind MAN-06 / coord M-0451 form (A), per rank -- tp TP-08] the C2 OF RECORD on THIS rank's map, Programs and shards
        sampled_replay = None
        if record and st.get("replay"):
            sampled_replay = self._deferred_replay(rc, defer) if defer else self._tp2_sampled_replay(rc)
        return _jsonable({"rank": st["rank"], "run_root": rc.run_root.hex(), "binding": binding, "attribution": attribution, "match_oracle": match_oracle,
                          "sampled_replay": sampled_replay, "weights_pin": st.get("weights_pin"),
                          "steps": len(rc.steps), "leaves_per_step": [s.n_leaves for s in rc.steps],
                          "leaves": sum(s.n_leaves for s in rc.steps), "bytes_bound": sum(s.n_bytes for s in rc.steps), "tensors": sum(s.n_tensors for s in rc.steps),
                          "root_ready_delay_s": rc.root_ready_delay_s, "finalize_s": rc.t_finalize_s, "spans_total_s": rc.spans["total_s"],
                          "expected_tensors": rc.extra.get("expected_tensors"), "committed_tensors": rc.extra.get("committed_tensors"),
                          "errors_n": rc.extra.get("errors_n", len(rc.extra.get("errors") or [])), "errors": (rc.extra.get("errors") or [])[:10],
                          "identity_check": rc.extra.get("identity_check"), "expected_manifest": com.expected_manifest() if hasattr(com, "expected_manifest") else None,
                          "classes": class_coverage(d), "declaration": d, "weight_registration": rc.weight_registration,
                          "persistent_bytes": rc.persistent_bytes, "temporary_bytes_peak": rc.temporary_bytes_peak, "bounded_staging": rc.extra.get("bounded_staging"),
                          "plans_cached": (rc.extra.get("native_collect") or {}).get("plans_cached"), "pinned_pool_bytes": rc.extra.get("pinned_pool_bytes")})

    def _tp2_logical_request(self, rid: str, step: int | None = None) -> str:
        """Engine row id -> workload `r<i>` (commit_delta's rule, mirrored per rank): vLLM v1 suffixes the external request id with
        `-<8 hex>` internally -- match exact, then with the suffix stripped, then by unique prefix.  An unmapped id stays raw, so every
        identity of that request goes UNBOUND and coverage FAILS for it (fail-closed; this was the r15-tp06 B8 first-attempt FAIL:
        96330/96493 unbound per rank with the raw `<uuid>-<hex>` ids in the map)."""
        st = self._tp2_commit
        order = dict(st.get("request_ids") or {}) or {str(r): f"r{i}" for i, r in enumerate(st.get("request_order") or [])}
        rid = str(rid)
        if rid in order:
            lg = order[rid]
        else:
            base = rid.rsplit("-", 1)[0]
            if base in order:
                lg = order[base]
            else:
                hits = [v for k, v in order.items() if rid.startswith(k)]
                lg = hits[0] if len(hits) == 1 else rid
        return swap_label(lg, st.get("swap_rule"), step)     # the swap NEGATIVE (all steps, or steps lo..hi only) -- identity on the honest path

    def _tp2_attribution(self, rc: Any) -> dict[str, Any] | None:
        """[R15 rev-cb M-0359 T6-3] per-(request, engine step) attribution of THIS rank's committed values against the MATCH RECORD: the
        committed `runner.sampler/sampled_token_ids` row j of step s belongs (by the rank's request->rows map) to request r; its value must
        be the Match record's k-th generated token of r (k = how many steps r has been sampled in so far).  The Match record is
        tp2_match's capture arm tokens.json (per request generated ids), the same greedy workload -- an independent run, not this one.
        A request-label swap between two requests of equal (prompt, max_tokens) keeps every per-request identity COUNT and the
        composition intact; only this check sees it (the two requests' sampled sequences differ).  None when no Match record was given."""
        st = self._tp2_commit
        mt = st.get("match_tokens")
        if not mt:
            return None
        import numpy as np

        from verity_vllm.check import oracle_compare as OC
        from verity_vllm.acquire.sources.partial_source import _committed

        com, rank = st["com"], int(st["rank"])
        rd = OC.OpenedReader(com, rc)
        step_rows = getattr(com, "step_rows", None) or {}
        layouts = getattr(com, "_layouts", None) or {}
        seen: dict[str, int] = {}
        checked, steps_checked = 0, 0
        mismatches: list[dict] = []
        missing: list[dict] = []
        for s in sorted(int(k) for k in layouts):
            metas = layouts[s] if s in layouts else layouts[str(s)]
            m = next((x for x in metas if getattr(x, "name", None) == "runner.sampler/sampled_token_ids"), None)
            rows = step_rows.get(int(s))
            if m is None or not rows:
                missing.append({"step": s, "why": ("no committed sampled ids" if m is None else "no request rows recorded")})
                continue
            try:
                b = _committed(rd, s, m)
            except OC.OpeningNotVerified as e:
                mismatches.append({"step": s, "why": f"opened value not verified: {e}"[:200]})
                continue
            if b is None:
                missing.append({"step": s, "why": "committed bytes not retained (VERITY_RETAIN)"})
                continue
            dt = {"int64": np.int64, "int32": np.int32, "long": np.int64, "int": np.int32}.get(str(m.dtype).replace("torch.", ""))
            if dt is None:
                missing.append({"step": s, "why": f"sampled ids dtype {m.dtype} not an integer type"})
                continue
            ids = np.frombuffer(b, dtype=dt)
            if len(m.shape) >= 1 and int(np.prod(m.shape)) == ids.size and len(m.shape) == 2:
                ids = ids.reshape(m.shape)[:, 0]     # (n_reqs, 1) sampled ids per request row
            if len(ids) != len(rows):
                mismatches.append({"step": s, "why": f"{len(ids)} committed sampled rows vs {len(rows)} request rows in the map"})
                continue
            for j, (rid, _lo, _hi) in enumerate(rows):
                lg = self._tp2_logical_request(rid, step=int(s))
                k = seen.get(lg, 0)
                seen[lg] = k + 1
                exp = mt.get(lg)
                got = int(ids[j])
                want = int(exp[k]) if (exp is not None and k < len(exp)) else None
                if want is None or want != got:
                    if len(mismatches) < 24:
                        mismatches.append({"step": s, "row": j, "request": lg, "k": k, "committed": got, "match_record": want})
                    else:
                        mismatches.append(None)
                checked += 1
            steps_checked += 1
        n_mis = len(mismatches)
        mismatches = [x for x in mismatches if x is not None]
        lens = {lg: {"sampled_rows_committed": seen.get(lg, 0), "match_record_generated": len(mt[lg])} for lg in sorted(mt)}
        len_ok = all(v["sampled_rows_committed"] == v["match_record_generated"] for v in lens.values())
        ok = n_mis == 0 and not missing and len_ok and checked > 0 and steps_checked > 0
        why = None
        if not ok:
            if n_mis:
                f = mismatches[0]
                why = (f"rank {rank}: step {f.get('step')}: {f['why']}" if "why" in f else
                       f"rank {rank}: {n_mis} of {checked} committed sampled rows disagree with the Match record (first: step {f.get('step')} row {f.get('row')} "
                       f"labelled {f.get('request')} k={f.get('k')}: committed {f.get('committed')} != match {f.get('match_record')})")
            elif missing:
                why = f"rank {rank}: {len(missing)} steps without attributable sampled ids (first: {missing[0]})"
            elif not len_ok:
                why = f"rank {rank}: per-request sampled row counts != Match record lengths: {lens}"
            else:
                why = f"rank {rank}: nothing checked"
        return {"ok": ok, "rank": rank, "reference": "tp2_match capture arm tokens.json (per request generated ids; independent instrumented run)",
                "identity": "runner.sampler/sampled_token_ids row j of step s -> request of row j in this rank's request->rows map (step_rows) -> k-th sampled token of that request",
                "steps_checked": steps_checked, "rows_checked": checked, "requests": lens, "mismatches_n": n_mis, "mismatches": mismatches[:12],
                "missing_steps": missing[:8], "swap_requests_negative": st.get("swap_requests"), "value_source": rd.record(), "why": why}

    def _tp2_binding(self, rc: Any) -> dict[str, Any] | None:
        """[R14 TP-03 x BIND-01] THIS rank's binding map (bind's `verity_vllm.commit.binding`: identity -> commitment location, fixed at
        finalize before any challenge) and its coverage against the rank's EXPECTED population = model structure x workload steps x this
        rank, where every collective-issuing module additionally requires its rank-local member `part_rank<r>` / `shard_rank<r>` (the TP
        partial source's sites).  The population is structural (install-time module set + site set), never read from what was emitted."""
        st = self._tp2_commit
        pd = st.get("program_digest")
        if not pd:
            return {"skipped": "no program_digest for this rank (pass --build-rank<r> or --required-manifest to the driver)"}
        n_req = int(st.get("n_requests") or 1)
        try:
            from verity_vllm.commit import binding as bd
        except Exception as e:  # noqa: BLE001
            return {"skipped": f"binding module unavailable: {e!r}"}
        com, rank = st["com"], int(st["rank"])
        # B>1 [R15 TP-04 x BIND-01]: the per-step request->row map recorded by the rank's collector at sample time (`InputBatch.req_ids` x
        # `scheduler_output.num_scheduled_tokens`, every rank executes the same scheduler_output) names the batched rows' requests
        step_rows = getattr(com, "step_rows", None) or {}
        _logical = self._tp2_logical_request
        request_rows = {int(s): [(_logical(rid, step=int(s)), lo, hi) for rid, lo, hi in rows] for s, rows in step_rows.items()} if n_req > 1 else None
        if n_req > 1 and not step_rows:
            return {"skipped": f"B={n_req} > 1 but the rank's collector recorded no step_rows ({getattr(com, 'step_rows_why', None)}) -- no coverage verdict"}
        try:
            bmap = bd.build_binding_map(com._layouts, rc.steps, rc.run_root, pd, rank=rank, weight_registration=rc.weight_registration, request_rows=request_rows,
                                        collector={"committer": com.name, "layout": "chunk-leaf-v1", "retain": getattr(com, "retain", None), "windowed": bool(getattr(com, "windowed", False))})
            struct = bd.structure_of(com)
            tps = st.get("tps")
            if tps is None and int(st.get("world") or 1) > 1:
                # the population is STRUCTURAL: the rank-local collective members are required by the model's TP structure whether or
                # not a source binds them (a collective-outputs-only collector must fail coverage, not just the required-class check)
                from verity_vllm.acquire.sources.partial_source import TPPartialSource
                tps = TPPartialSource(com, st["eng"].model, rank=rank, world=int(st["world"]))     # site set only; never installed
            tp_members = 0
            if tps is not None:
                by_path = {d["op_path"]: d for d in struct["modules"]}
                for site, cls in tps.sites:
                    mem = f"shard_rank{rank}" if cls == "LogitsProcessor" else f"part_rank{rank}"
                    if site in by_path:
                        by_path[site]["members"] = list(by_path[site]["members"]) + [mem]
                    else:
                        struct["modules"].append({"op_path": site, "class": cls, "invocations": 1, "members": [mem]})
                    tp_members += 1
                struct["tp_partial_sites"] = len(tps.sites)
            requests = [f"r{i}" for i in range(n_req)]
            arrivals = {f"r{i}": int((st.get("arrivals") or {}).get(f"r{i}", 0)) for i in range(n_req)}
            caps = {f"r{i}": int((st.get("caps") or {}).get(f"r{i}", 0)) for i in range(n_req)}
            horizon = max(arrivals[k] + caps[k] for k in caps) if any(caps.values()) else len(rc.steps)
            wl = {"program_digest": pd, "requests": requests, "steps": horizon, "ranks": [rank],
                  "request_steps": {k: list(range(arrivals[k], arrivals[k] + caps[k])) for k in caps} if any(caps.values()) else None}
            if wl["request_steps"] is None:
                wl.pop("request_steps")
            pop_struct = list(bd.expected_population(struct, wl))
            man_hdr = None
            source = "structure(model modules x MEMBER_RULES + TP partial sites) x workload(requests, steps, rank)"
            if st.get("required_manifest"):
                # [R15 TP-04] population of record = manbind's Program x canonical query manifest for THIS rank (rows with rank=r; the
                # manifest's per-rank Program digest already checked at make); the structural population is recorded beside it
                pop_all, man_hdr = bd.load_required_manifest(st["required_manifest"], pd, requests=requests, steps=horizon, ranks=[rank], arrivals=arrivals)
                pop = [i for i in pop_all if int(i[7]) == rank]          # identity = (pd, rid, step, inv, op, member, (lo, hi), rank)
                man_hdr["rows_for_rank"], man_hdr["rows_all_ranks"] = len(pop), len(pop_all)
                man_hdr["structural_population_digest"], man_hdr["structural_population_n"] = bd.population_digest(pop_struct)
                man_hdr["structural_coverage_ok"] = bd.coverage_check(pop_struct, bmap)["ok"]
                man_hdr["program_digest_for_rank"] = ((st.get("manifest_doc") or {}).get("program_digest_per_rank") or {}).get(str(rank))
                source = "required-manifest (manbind: Program(rank) x canonical query, rows with rank=r)"
                if not pop:
                    return {"error": f"required manifest has no rows for rank {rank} ({man_hdr.get('path')})", "rank": rank}
            else:
                pop = pop_struct
            pdig, pn = bd.population_digest(pop)
            cov = bd.coverage_check(pop, bmap)
            # [R15 manbind B8, mirrored per rank] B>1: the map's per-engine-step request->rows partition of THIS rank's packed members must be
            # exactly the manifest's declared composition (Program x workload, never the collector) -- request-local identity ranges are blind
            # to WHICH block a request was bound to; this is what makes an omitted / relabelled / duplicated request fail named on a rank
            comp = bd.composition_check(man_hdr, bmap) if (man_hdr is not None and hasattr(bd, "composition_check")) else None
            if comp is not None:
                cov["composition"] = {k: v for k, v in comp.items() if k != "problems"} | {"problems": (comp.get("problems") or [])[:3]}
            if man_hdr is not None and man_hdr.get("complete") is False:            # [MAN-02] unmodelled Program families -> not a population
                cov = dict(cov, ok=False, error="manifest incomplete: unmodelled Program families " + "; ".join(f"{k} x{v}" for k, v in sorted((man_hdr.get("unmodelled") or {}).items())))
            elif man_hdr is not None and man_hdr.get("program_digest_for_rank") not in (None, pd):
                cov = dict(cov, ok=False, error=f"manifest Program digest for rank {rank} {str(man_hdr.get('program_digest_for_rank'))[:16]} != this rank's Program {pd[:16]}")
            elif comp is not None and not comp.get("ok"):
                cov = dict(cov, ok=False, missing=list(cov.get("missing") or []), error=f"rank {rank}: " + str(comp.get("error") or "composition check failed"))
            ctx_ok, ctx_why = bd.map_context_check(bmap, rc) if hasattr(bd, "map_context_check") else (None, "n/a")
            st["bmap"] = bmap
            st["pop"] = pop
            return {"rules_version": bd.RULES_VERSION, "rank": rank, "program_digest": pd, "map_digest": bmap["digest"], "entries_n": bmap["entries_n"],
                    "population": {"source": source, "manifest": man_hdr, "digest": pdig, "n": pn, "steps": horizon, "requests": requests, "ranks": [rank],
                                   "tp_partial_members": tp_members, "step_rows_recorded": len(step_rows), "step_rows_why": getattr(com, "step_rows_why", None),
                                   "structural_n": len(pop_struct)},
                    "coverage": cov, "context": {"ok": ctx_ok, "why": ctx_why}, "structure_modules": len(struct["modules"]),
                    "note": ("population = required manifest rows with rank=r" if man_hdr else "population = structure x workload (no manifest given)")
                            + (f"; B={n_req}: request rows from the rank's collector step_rows" if n_req > 1 else "; B=1: request r0 = every step")}
        except Exception as e:  # noqa: BLE001
            import traceback
            return {"error": repr(e)[:400], "trace": traceback.format_exc()[-1200:]}

    def _tp2_sampled_replay(self, rc: Any) -> dict[str, Any]:
        """[TP-08; manbind MAN-06 (`verity_vllm.check.replay.sampled_replay`, taken verbatim), coord M-0451 form (A), interface note M-0523]
        The C2 OF RECORD on THIS rank (`pipeline.tp.rank_replay`): a seeded stratified sample of the rank Program's rows is re-evaluated
        from the rank's COMMITTED input members + the rank's registered weight SHARDS (the rank Program's weights params ARE the shards
        this worker holds) and compared bit-for-bit with the rank's committed output members; plus the exhaustive boundary linkage
        (committed input ids at arrival == the declared prompt; later steps == the previous step's committed sampled id, per request).
        Programs = THIS rank's request Programs (`derive_step --tp 2 --rank r`; B>=2: one component per (LP, T) under the rank's Build
        root), REFUSED by name unless every loaded digest is a digest of record for this rank (the GP-01 workload_program's component
        list / the manifest's per-rank Program, both already checked against the Build by the driver).  Rows that read a peer rank's
        partial (`peers_*` inputs of AllReduce2 / AllGather2) have no evaluator and are counted by name -- never PASS by omission.  The
        stratum key's rank label comes from manbind's population(rank=0) at v1 (M-0523 (1) pending); this record carries the true rank
        beside it."""
        from verity_vllm.pipeline.tp import rank_replay as RR

        return RR.in_process(self._tp2_commit, rc)

    def _deferred_replay(self, rc: Any, defer: dict[str, Any]) -> dict[str, Any]:
        """`--replay-deferred`: this rank's replay bundle in place of `_tp2_sampled_replay` (`pipeline.tp.rank_replay.defer`)."""
        from verity_vllm.pipeline.tp import rank_replay as RR

        return RR.defer(self._tp2_commit, rc, defer)

    def tp2_repo_modules(self) -> dict[str, Any]:
        """[TP-11] the repo-local modules THIS rank process imported so far (call after finalize: replay / pin / linkage imports included)."""
        return dict(repo_modules_loaded(), rank=int(self._tp2_commit["rank"]) if getattr(self, "_tp2_commit", None) else None)

    def tp2_commit_xrank_dump(self, *, out_dir: str, seed_hex: str, pair: int, per_stratum: int = 2, max_rows: int = 16) -> dict[str, Any]:
        """[R16 tp TP-12, rev F-r16-18] THIS rank's picks for the cross-rank collective-output pass, from the COMMITTED bytes (every member opened
        and verified against this rank's run root, the same reader as the value check and the sampled replay; a member that does not verify
        is a pick without bytes, named -> the driver's check FAILs it): per pick the rank's partial rows and the
        rows of the received collective output as this rank committed it (`<site>/0` dense module output, `<site>/out` MoE block).  Picks are
        drawn identically on every rank from the pair's combined root (fixed after commitment) -> xrank_pair<k>_rank<r>.pt; the driver pairs
        the ranks and evaluates the definition (xrank_collectives.check_picks)."""
        import time as _t

        import torch

        from verity_vllm.check import oracle_compare as OC
        from verity_vllm.check.match import xrank_collectives as XC
        from verity_vllm.acquire.sources.partial_source import _committed

        st = self._tp2_commit
        com, rank = st["com"], int(st["rank"])
        t0 = _t.perf_counter()
        layouts = {int(s): [(m.name, tuple(m.shape)) for m in ms] for s, ms in getattr(com, "_layouts", {}).items()}
        plan = XC.pick_strata(layouts, rank, bytes.fromhex(seed_hex), per_stratum=int(per_stratum), max_rows=int(max_rows), world=int(st["world"]))
        picks, missing, nbytes = [], 0, 0
        rd = OC.OpenedReader(com, st["rc"]) if st.get("rc") is not None else None
        unverified: dict[tuple[int, str], str] = {}

        def tensor_of(step: int, meta) -> Any:
            if rd is None:
                unverified[(step, meta.name)] = "no run commitment on this rank (not finalized)"
                return None
            try:
                b = _committed(rd, step, meta)
            except OC.OpeningNotVerified as e:
                unverified[(step, meta.name)] = f"opened value not verified: {e}"[:200]
                return None
            if b is None:
                return None
            dt = getattr(torch, str(meta.dtype).replace("torch.", ""))
            return torch.frombuffer(bytearray(b), dtype=dt).reshape(tuple(meta.shape)).clone()

        for pk in plan["picks"]:
            step = int(pk["step"])
            metas = {m.name: m for m in com._layouts.get(step, [])}
            pm = metas.get(pk["member"])
            rname = next((n for n in pk["recv_candidates"] if n in metas), None)
            rm = metas.get(rname) if rname else None
            rows = torch.tensor(pk["rows"], dtype=torch.long)
            part = tensor_of(step, pm) if pm is not None else None
            recv = tensor_of(step, rm) if rm is not None else None
            why = None
            if part is None or recv is None:
                def _gone(meta) -> str:
                    return unverified.get((step, meta.name), "bytes not retained")
                why = ("partial " + ("not in this step's layout" if pm is None else _gone(pm)) if part is None else "") + \
                      ("; " if part is None and recv is None else "") + \
                      (("received output " + ("not committed under any of " + "/".join(pk["recv_candidates"]) if rm is None else _gone(rm))) if recv is None else "")
                missing += 1
            else:
                if part.dim() >= 2 and recv.dim() >= 2 and part.shape[0] == recv.shape[0] == pk["rows_total"]:
                    part, recv = part[rows], recv[rows]
                elif part.dim() >= 2 and recv.dim() >= 2 and part.shape[0] != recv.shape[0]:
                    why = f"row counts differ: partial {tuple(part.shape)} vs received {tuple(recv.shape)}"
                nbytes += part.numel() * part.element_size() + recv.numel() * recv.element_size()
            emb = None
            if pk["kind"] == "all_reduce" and XC.is_embedding_site(pk["site"]):
                # EmbeddingShard_v1 leg (ruling 27): this rank's committed partial rows == masked gather of the committed ids from ITS vocab shard
                # (the live parameter, which the per-rank weights pin ties to the checkpoint of record); rows outside the rank's range are zero
                try:
                    im = metas.get(f"{pk['site']}/in")
                    ids_t = tensor_of(step, im) if im is not None else None
                    mod = dict(st["eng"].model.named_modules()).get(pk["site"])
                    si = getattr(mod, "shard_indices", None)
                    if ids_t is None or mod is None or si is None or part is None:
                        emb = {"ok": None, "why": (((unverified.get((step, im.name)) if im is not None else None) or "no committed ids member <site>/in") if ids_t is None
                                                   else "no VocabParallelEmbedding shard_indices on the module" if si is None else "partial not committed")}
                    else:
                        lo, hi = int(si.org_vocab_start_index), int(si.org_vocab_end_index)
                        idx = ids_t.reshape(-1)
                        idx = idx[rows] if (idx.numel() == pk["rows_total"] and part.shape[0] == len(pk["rows"])) else idx
                        mask = (idx >= lo) & (idx < hi)
                        local = torch.where(mask, idx - lo, torch.zeros_like(idx)).long()
                        W = mod.weight
                        exp = W[local.to(W.device)].masked_fill((~mask).unsqueeze(-1).to(W.device), 0).to("cpu")
                        ok = bool(exp.shape == part.shape and exp.dtype == part.dtype and torch.equal(exp, part))
                        emb = {"ok": ok, "n_diff": (int((exp.view(torch.int16) != part.view(torch.int16)).sum()) if (exp.shape == part.shape and exp.dtype == torch.bfloat16) else None),
                               "vocab_range": [lo, hi], "masked_rows": int((~mask).sum()), "rows": int(idx.numel()), "why": None if ok else "committed partial != masked gather from this rank's shard"}
                except Exception as e:  # noqa: BLE001
                    emb = {"ok": None, "why": f"embedding-shard leg error: {e!r}"[:200]}
            picks.append(dict(pk, part=part, recv=recv, recv_member=rname, why=why, emb=emb))
        rec = {"schema": "verity/tp2/xrank-picks/v1", "rank": rank, "pair": int(pair), "seed": seed_hex, "strata": plan["strata"], "picks": picks,
               "per_stratum": int(per_stratum), "max_rows": int(max_rows), "steps_in_layout": len(layouts)}
        fn = Path(out_dir) / f"xrank_pair{int(pair)}_rank{rank}.pt"
        torch.save(rec, fn)
        return {"rank": rank, "file": str(fn), "picks": len(picks), "strata": len(plan["strata"]), "missing": missing, "bytes": nbytes,
                "value_source": rd.record() if rd is not None else {"source": "opened", "reads": 0, "why": "no run commitment on this rank"},
                "unreplayable_sites": st.get("xrank_unreplayable_sites"), "strata_sites": sorted({v["site"] for v in plan["strata"].values()}),
                "families": sorted({p["family"] for p in picks}), "flipped_collective_inputs": int(getattr(st.get("tps"), "flipped", 0) or 0),
                "wall_s": round(_t.perf_counter() - t0, 3)}

    def tp2_commit_binding_map_dump(self, *, out_dir: str) -> dict[str, Any]:
        """[R15 TP-04] write THIS rank's binding map of the last finalized run (identity -> commitment location, fixed at finalize before any
        challenge) beside the driver's evidence (`binding_map_rank<r>.json`), so a reviewer can attack the map itself and not only its digest."""
        st = self._tp2_commit
        bmap = st.get("bmap")
        if not bmap:
            return {"rank": st.get("rank"), "skipped": "no binding map for this run"}
        p = Path(out_dir) / f"binding_map_rank{int(st['rank'])}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(bmap, default=str) + "\n")
        return {"rank": int(st["rank"]), "path": str(p), "digest": bmap.get("digest"), "entries_n": bmap.get("entries_n")}

    def _claimed(self, rc: Any, claimed_root_hex: str | None) -> Any:
        """The commitment the verifier holds = the driver's commitment vector entry for this rank (`claimed_root_hex`), not the worker's
        private copy: a driver-side wrong root for one rank must fail that rank's openings (Commit C3 negative).  None = the worker's own."""
        if not claimed_root_hex or claimed_root_hex == rc.run_root.hex():
            return rc
        import dataclasses
        try:
            return dataclasses.replace(rc, run_root=bytes.fromhex(claimed_root_hex))
        except Exception:  # noqa: BLE001
            import copy
            rc2 = copy.copy(rc)
            rc2.run_root = bytes.fromhex(claimed_root_hex)
            return rc2

    def tp2_commit_open_population(self, *, k: int, salt: str, combined_root_hex: str, claimed_root_hex: str | None = None) -> dict[str, Any]:
        """[BIND-01 x TP] openings drawn from THIS rank's EXPECTED population (bind's `challenge_identities`: Fiat-Shamir from the combined
        run root + this rank's map digest + salt; forced late step + one per family), opened against the rank's retained tree and verified
        with the production core against the CLAIMED root (the driver's commitment-vector entry for this rank).  An identity without a
        binding counts as a failed opening (fail-closed)."""
        import time as _t

        from verity_vllm.commit import binding as bd

        st = self._tp2_commit
        com, rc = st["com"], st["rc"]
        pop, bmap = st.get("pop"), st.get("bmap")
        if rc is None or not pop or not bmap:
            return {"rank": st["rank"], "n": 0, "verified": 0, "error": "no population / binding map on this rank"}
        rcv = self._claimed(rc, claimed_root_hex)
        draws = bd.challenge_identities(pop, bmap, bytes.fromhex(combined_root_hex), int(k), salt=salt.encode())
        unbound = [bd.identity_dict(i) for i, e, _l in draws if e is None]
        positions = [(int(i[2]), int(leaf)) for i, e, leaf in draws if e is not None]
        t0 = _t.perf_counter()
        ops = com.open(positions) if positions else []
        ok = [bool(com.verify(rcv, o)) for o in ops]
        ctx_ok, ctx_why = bd.map_context_check(bmap, rcv) if hasattr(bd, "map_context_check") else (None, "n/a")
        late = max(p[2] for p in pop)
        fams = sorted({i[4].split(".")[-1].split("/")[0] if not i[4].startswith(bd.HIDDEN_FAMILIES) else i[4].split("/")[0] for i, e, _l in draws})
        tp_draws = sum(1 for i, e, _l in draws if str(i[5]).startswith(("part_rank", "shard_rank")))
        return _jsonable({"rank": st["rank"], "n": len(draws), "opened": len(ops), "verified": sum(ok), "unbound_n": len(unbound), "unbound": unbound[:5],
                          "late_step_in_sample": any(i[2] == late for i, e, _l in draws), "families": fams, "tp_partial_identities_drawn": tp_draws,
                          "open_verify_s": _t.perf_counter() - t0, "map_digest": bmap["digest"],
                          "claimed_root": (claimed_root_hex or rc.run_root.hex()), "claimed_root_is_own": rcv is rc, "context": {"ok": ctx_ok, "why": ctx_why}})

    def tp2_commit_open(self, *, positions_by_rank: dict, claimed_roots: dict | None = None) -> dict[str, Any]:
        """Open THIS rank's challenged positions (`positions_by_rank[str(rank)]`, sampled by the driver from the COMBINED run root) against
        the rank's retained tree; verify each opening with the production core against the CLAIMED root for this rank
        (`claimed_roots[str(rank)]`: the driver's commitment-vector entry; absent = the worker's own).  A rank without a commitment reports, never raises."""
        import statistics
        import time as _t

        st = self._tp2_commit
        com, rc = st["com"], st["rc"]
        if rc is None:
            return {"rank": st["rank"], "n": 0, "verified": 0, "error": "no run commitment on this rank"}
        sample = [tuple(p) for p in positions_by_rank.get(str(st["rank"]), [])]
        if not sample:
            return {"rank": st["rank"], "n": 0, "verified": 0, "error": "no positions for this rank"}
        rcv = self._claimed(rc, (claimed_roots or {}).get(str(st["rank"])))
        t0 = _t.perf_counter()
        ops = com.open(sample)
        t_open = _t.perf_counter() - t0
        t1 = _t.perf_counter()
        ok = [bool(com.verify(rcv, o)) for o in ops]
        t_ver = _t.perf_counter() - t1
        return _jsonable({"rank": st["rank"], "n": len(ops), "verified": sum(ok), "generate_s": t_open, "verify_s": t_ver,
                          "claimed_root": rcv.run_root.hex(), "claimed_root_is_own": rcv is rc,
                          "bytes_mean": statistics.fmean(len(o.value) + 32 * sum(1 for p in o.path if p is not None) for o in ops) if ops else None})

    # ------------------------------------------------------------------ [R15 TP-04 C2] value correspondence, per rank

    def tp2_commit_value_check_install(self) -> dict[str, Any]:
        """One extra instrumented run: the committer + bind's ValueChecker (hook-time reference clones + re-execution of the registered
        relation for every module output) + the TP partial source's reference tap (clone of every collective INPUT + re-execution of the
        rank-local GEMM on the shard).  Compared after finalize by `tp2_commit_value_check`."""
        from verity_vllm.check.value_check import ValueChecker

        st = self._tp2_commit
        com = st["com"]
        com.reset()
        com.install()
        st["rc"] = None
        vc = ValueChecker(com)
        vc.install()
        st["vc"] = vc
        tps = st.get("tps")
        if tps is not None:
            tps.reference = True
            tps.ref, tps.ref_in = {}, {}
        return {"rank": st["rank"], "installed": True, "value_checker": True, "tp_reference_tap": tps is not None}

    def tp2_commit_value_check(self) -> dict[str, Any]:
        """After the value-check run: finalize, then committed bytes (opened and verified against this run's root) vs (T) reference clones /
        (R) re-execution, for the module outputs (ValueChecker) and for THIS rank's partials (TPPartialSource.compare).  PASS only if both
        pass on this rank."""
        from verity_vllm.check import oracle_compare as OC
        from verity_vllm.pipeline.commit import assert_pristine

        st = self._tp2_commit
        com, vc, tps = st["com"], st.pop("vc", None), st.get("tps")
        rc = com.finalize()
        res: dict[str, Any] = {"rank": st["rank"], "run_root": rc.run_root.hex()}
        try:
            res["modules"] = vc.compare(OC.OpenedReader(com, rc)) if vc is not None else {"verdict": "NOT_RUN"}
        except Exception as e:  # noqa: BLE001
            res["modules"] = {"verdict": "FAIL", "error": repr(e)[:300]}
        try:
            res["tp_partials"] = tps.compare(OC.OpenedReader(com, rc)) if tps is not None else {"verdict": "NOT_RUN", "why": "no tp partial source"}
        except Exception as e:  # noqa: BLE001
            res["tp_partials"] = {"verdict": "FAIL", "error": repr(e)[:300]}
        try:                                   # [R15 moe, rev-cb M-0424 T6-4 / report M-0435] while the retained bytes still exist (before reset)
            res["t6_4"] = self._t6_4_check(com, rc)
            # [report M-0478] the bind-time-clone count for the SAME /out members: committed /out != TPPartialSource's _bytes(y) at bind
            # (compare()'s by_member['out']: tensors - tap_equal); any nonzero = late read whatever the next-norm/0 comparison says
            tpp = res.get("tp_partials") or {}
            bm = (tpp.get("by_member") or {}).get("out") or {}
            res["t6_4"]["out_tensors_vs_clone"] = int(bm.get("tensors") or 0)
            res["t6_4"]["clone_mismatches"] = int(bm.get("tensors") or 0) - int(bm.get("tap_equal") or 0) if bm else None
            res["t6_4"]["clone_rule"] = "committed <L>.mlp.experts/out == TPPartialSource bind-time clone _bytes(y) (tp_partials.by_member.out); None = no /out compared"
            if res["t6_4"].get("result") is True and (res["t6_4"]["clone_mismatches"] is None or res["t6_4"]["clone_mismatches"] > 0):
                res["t6_4"]["result"] = False
            elif res["t6_4"].get("result") is None and (res["t6_4"]["clone_mismatches"] or 0) > 0:
                res["t6_4"]["result"] = False           # not-applicable by the pair rule, but a committed /out differs from its bind-time clone
        except Exception as e:  # noqa: BLE001
            res["t6_4"] = {"schema": "verity/commit-integ/t6_4-check/v1", "result": False, "error": repr(e)[:300]}
        if vc is not None:
            vc.uninstall()
        if tps is not None:
            tps.reference = False
        com.uninstall()
        com.reset()
        st["rc"] = None
        assert_pristine(st["eng"], st["base"])
        res["verdict"] = "PASS" if (res["modules"].get("verdict") == "PASS" and res["tp_partials"].get("verdict") == "PASS") else "FAIL"
        # COORD M-0111 (1) vocabulary: this is a per-identity ORACLE compare (committed bytes vs values computed independently of the
        # collector -- the hook-time clone and a re-executed relation), not a root compare against another run; the oracle is named so
        # the reader knows it is NOT the Match fold's values (M-0111 (2) `--oracle` is manbind's shared check, not yet released)
        res["method"] = ("oracle-compare: per identity, committed bytes == hook-time reference clone (T) AND == re-execution of the registered relation on the "
                         "clone's inputs (R) per module output [value_check.ValueChecker]; TP partials: == collective INPUT reference clone (T) AND == rank-local "
                         "shard-GEMM re-execution (R) [TPPartialSource.compare]; per rank; FAIL names the identity (tap_mismatch/reexec_mismatch/missing_in_commit)")
        res["oracle"] = "hook-time reference clone + re-executed relation (same attempt, same rank, independent of the collector's bytes); not the Match fold"
        return _jsonable(res)

    def _t6_4_check(self, com: Any, rc: Any) -> dict[str, Any]:
        """The fused-MoE late-read detector over the last finalized run on this rank (`partial_source.late_read_check`)."""
        from verity_vllm.acquire.sources.partial_source import late_read_check

        return late_read_check(com, rc, int(self._tp2_commit["rank"]))

    def tp2_commit_reset(self) -> dict[str, Any]:
        self._tp2_commit["com"].reset()
        return {"rank": self._tp2_commit["rank"], "reset": True}

    # ------------------------------------------------------------------ per-rank kernel census

    def tp2_profiler_start(self) -> dict[str, Any]:
        import torch
        from torch.profiler import ProfilerActivity, profile

        torch.cuda.synchronize()
        p = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False)
        p.__enter__()
        self._tp2_prof = p
        return {"rank": int(_tp_facts(self)["tp"]["rank"]), "started": True, "pid": os.getpid()}

    def tp2_profiler_stop(self, *, out_dir: str, keep_trace: bool = False) -> dict[str, Any]:
        import torch

        from verity_vllm.properties import census

        p = self._tp2_prof
        torch.cuda.synchronize()
        p.__exit__(None, None, None)
        rank = int(_tp_facts(self)["tp"]["rank"])
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tr = out / f"trace.rank{rank}.json"
        p.export_chrome_trace(str(tr))
        evs = census._load_trace(tr)
        ks = census.kernels_of_trace_events(evs)
        summary = census.summarize_kernels(ks)
        cpu_ops = {}
        for e in evs:
            if e.get("ph") == "X" and e.get("cat") == "cpu_op":
                cpu_ops[e.get("name", "")] = cpu_ops.get(e.get("name", ""), 0) + 1
        if keep_trace:
            with open(tr, "rb") as fi, gzip.open(str(tr) + ".gz", "wb") as fo:
                shutil.copyfileobj(fi, fo)
        tr.unlink()
        doc = {"schema": "verity-capture/census-rank/v0", "rank": rank, "pid": os.getpid(), "n_kernels": len(ks),
               "device_us": round(sum(float(e.get("dur", 0.0) or 0.0) for e in ks), 1), "kernels": summary,
               "cpu_ops": dict(sorted(cpu_ops.items(), key=lambda kv: -kv[1])[:200]),
               "method": "one torch.profiler.profile(activities=[CPU, CUDA]) session in this rank's process over the whole observed-free "
                         "run (warm-up excluded); device kernels = Kineto events with cat in {kernel, gpu_memcpy, gpu_memset}"}
        (out / f"census.rank{rank}.json").write_text(json.dumps(doc, indent=1) + "\n")
        return {"rank": rank, "n_kernels": len(ks), "distinct": len(summary), "device_us": doc["device_us"],
                "collective_kernels": {k: v["count"] for k, v in summary.items()
                                       if any(t in k.lower() for t in ("nccl", "allreduce", "all_reduce", "allgather", "cross_device", "reduce_kernel_"))}}
