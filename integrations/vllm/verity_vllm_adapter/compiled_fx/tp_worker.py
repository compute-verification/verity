"""Per-rank compiled-fx capture seam for tensor-parallel replay (lane COMP, CX1: compiled x TP).

TP's ``tp_worker.py`` shape over W1d's compiled-fx capture. With ``tensor_parallel_size > 1`` the model lives in
vLLM's worker processes, and the record pass must be installed *in each of them before the piecewise compile*.
Both happen here:

* **import time** (this module is imported in every worker when vLLM resolves ``worker_extension_cls``, before
  ``load_model`` and the compile): if ``VERITY_COMPILED_FX_TP`` is set (a JSON of the pass options the driver chose),
  the module creates the worker's ``FxIdTable`` + ``RecordInsertionPass``, installs W1d's hooks on vLLM's pass manager
  and piecewise interpreter, installs the ``verity::record`` op and its Inductor lowering, and gives the process its
  own Inductor cache directory (``<TORCHINDUCTOR_CACHE_DIR>/pid<pid>``): vLLM keeps Inductor's FX-graph cache on and
  the ranks' graphs are identical for every piece but the first, so a shared directory could serve rank 1 a wrapper
  compiled (record ops included) by rank 0 without running the pass in rank 1 - an empty id table on that rank.
* **RPCs** from the driver (``LLM.collective_rpc("verity_cfx_<method>", kwargs=...)``): ``install`` builds the
  ``StepHooks`` + ``CompiledCapture(shard=r)`` for ``<out>/rank<r>/``; ``arm`` installs the step hooks after the
  warm-up; ``finish`` writes the rank's boundaries, weight shards (``w=<param>/s=<r>``), description and kernels.

Nothing here changes what the workers compute beyond what W1d's single-process replay changes (the record op's copies).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

ENV_KEY = "VERITY_COMPILED_FX_TP"
STATE: dict[str, Any] = {}  # per worker process: table, pass, hooks_installed, lowering, cache_dir


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


def _install_at_import() -> None:
    cfg_s = os.environ.get(ENV_KEY)
    if not cfg_s or "table" in STATE:
        return
    cfg = json.loads(cfg_s)
    # 1. a private Inductor cache dir for this process (see the module docstring); torch may have cached the resolved
    #    dir already (functools.lru_cache on cache_dir()): clear it
    base = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or str(Path(cfg.get("cache_base", "/tmp/verity-inductor")))
    mine = str(Path(base) / f"pid{os.getpid()}")
    Path(mine).mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = mine
    try:
        from torch._inductor.runtime import cache_dir_utils

        for name in ("cache_dir", "default_cache_dir"):
            f = getattr(cache_dir_utils, name, None)
            if f is not None and hasattr(f, "cache_clear"):
                f.cache_clear()
    except Exception:  # noqa: BLE001
        pass
    # 2. W1d's pass + hooks + record op, exactly as compiled_fx/replay.py does in the single-process replay
    from verity_vllm_adapter.compiled_fx import fx_pass, record_op

    table = fx_pass.FxIdTable()
    roles = tuple(cfg.get("roles", ["extern_out", "extern_in", "graph_out", "input"]))
    rpass = fx_pass.RecordInsertionPass(table, dry_run=bool(cfg.get("dry_run", False)), dump_graphs=bool(cfg.get("dump_graphs", False)),
                                        roles=roles, placement=cfg.get("placement", "late"))
    hooks_installed = fx_pass.install_hooks(rpass)
    record_op.install()
    lowering = record_op.install_inductor_lowering() if cfg.get("lowering", "explicit") == "explicit" else "implicit"
    if cfg.get("record_impl") == "device":
        record_op.RING = record_op.DeviceRing(int(cfg.get("ring_rows", 1024)))
    STATE.update({"table": table, "pass": rpass, "hooks_installed": hooks_installed, "lowering": lowering, "cache_dir": mine, "cfg": cfg,
                  "pid": os.getpid()})
    print(f"[verity cfx pid {os.getpid()}] record pass installed at import: uuid {rpass.uuid()}; hooks {hooks_installed}; lowering {lowering}; "
          f"inductor cache {mine}", flush=True)


_install_at_import()


def _find_cache_dir(candidates: list[str]) -> str | None:
    """The directory whose Inductor wrappers carry the record ops (this process's compile)."""
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if not p.exists():
            continue
        for f in p.rglob("*.py"):
            try:
                txt = f.read_text(errors="ignore")
            except Exception:  # noqa: BLE001
                continue
            if "verity.record" in txt and "def call(" in txt:
                return str(p)
    return None


class VerityCompiledFxTPWorkerExtension:
    """Mixed into vLLM's Worker (``worker_extension_cls``); ``self`` is the worker."""

    def verity_cfx_state(self) -> dict[str, Any]:
        """What the import-time installation did in this worker (called right after the engine is built)."""
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

        from verity_vllm_adapter.compiled_fx import record_op

        table = STATE.get("table")
        rpass = STATE.get("pass")
        cands = [STATE.get("cache_dir") or "", os.environ.get("TORCHINDUCTOR_CACHE_DIR") or ""]
        try:
            from torch._inductor.runtime.cache_dir_utils import cache_dir as _cd

            cands.append(_cd())
        except Exception:  # noqa: BLE001
            pass
        found = _find_cache_dir(cands)
        STATE["cache_dir_effective"] = found
        return _jsonable({
            "rank": int(get_tensor_model_parallel_rank()), "world_size": int(get_tensor_model_parallel_world_size()), "pid": os.getpid(),
            "installed": "table" in STATE, "env_seen_at_import": ENV_KEY in os.environ, "pass_calls": rpass.calls if rpass else 0, "pieces": len(table.pieces) if table else 0, "ids": len(table.entries) if table else 0,
            "pass_uuid": rpass.uuid() if rpass else None, "hooks": STATE.get("hooks_installed"), "lowering": STATE.get("lowering"),
            "record_lowering_stats": dict(record_op.LOWERING_STATS), "inductor_cache_dir": STATE.get("cache_dir"), "inductor_cache_dir_effective": found,
            "cache_dir_candidates": cands, "env_inductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "ring": record_op.RING.stats() if record_op.RING is not None else None,
        })

    def verity_cfx_install(self, *, run_dir: str, profile_id: str, manifest: dict[str, Any], mode: str, prompt_lens: dict[str, int],
                           arena_bytes: int, hash_records: bool) -> dict[str, Any]:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

        from verity_vllm.capture import RunWriter
        from verity_vllm_adapter.compiled_fx.capture import CompiledCapture
        from verity_vllm_adapter.compiled_fx.replay import StepHooks

        if "table" not in STATE:
            raise RuntimeError(f"{ENV_KEY} was not set when this worker imported the extension (or the module was already imported in the "
                               f"forked parent): no record pass in this process")
        rank = int(get_tensor_model_parallel_rank())
        world = int(get_tensor_model_parallel_world_size())
        out = Path(run_dir) / f"rank{rank}"
        out.mkdir(parents=True, exist_ok=True)
        log_path = out / "replay.log"

        def log(*a: Any) -> None:
            msg = " ".join(str(x) for x in a)
            print(f"[verity cfx rank {rank}] {msg}", flush=True)
            with log_path.open("a") as f:
                f.write(msg + "\n")

        runner = self.model_runner
        model = runner.get_model()
        table = STATE["table"]
        rpass = STATE["pass"]
        if rpass.calls == 0 or not table.entries:
            raise RuntimeError(f"rank {rank}: the record pass did not run in this worker (pass calls {rpass.calls}, ids {len(table.entries)}): "
                               f"compile-cache hit or hook not installed")
        table.write(out / "fx_ids.json")
        cache_dir = STATE.get("cache_dir_effective") or _find_cache_dir([STATE.get("cache_dir") or "", os.environ.get("TORCHINDUCTOR_CACHE_DIR") or ""])
        if cache_dir is None:
            raise RuntimeError(f"rank {rank}: no Inductor wrapper with record ops found under {STATE.get('cache_dir')} / {os.environ.get('TORCHINDUCTOR_CACHE_DIR')}")
        hooks = StepHooks(runner, int(arena_bytes), dict(prompt_lens), hash_records=bool(hash_records))
        writer = None
        cap = None
        if mode in ("commit", "reference"):
            if mode == "commit":
                writer = RunWriter(out, manifest)
                if writer.profile_id != profile_id:
                    raise RuntimeError(f"rank {rank}: profile_id mismatch: writer {writer.profile_id} vs production {profile_id}")
            cap = CompiledCapture(runner=runner, model=model, profile_id=profile_id, run_dir=out, writer=writer, table=table, prompt_lens=dict(prompt_lens),
                                  log=log, cache_dir=Path(cache_dir), shard=rank, world_size=world)
            hooks.on_step = cap.on_step
        STATE.update({"hooks": hooks, "cap": cap, "writer": writer, "out": out, "log": log, "rank": rank, "world": world, "mode": mode})
        torch.cuda.synchronize()
        log(f"compiled-fx capture prepared: rank {rank}/{world}, mode {mode}, pass calls {rpass.calls}, pieces {len(table.pieces)}, ids {len(table.entries)}, "
            f"inductor cache {cache_dir}, arena {arena_bytes / 2**30:.1f} GiB")
        return _jsonable({"rank": rank, "world_size": world, "out": str(out), "pass_calls": rpass.calls, "pieces": len(table.pieces), "ids": len(table.entries),
                          "inductor_cache_dir": cache_dir, "pass_uuid": rpass.uuid(), "hooks": STATE.get("hooks_installed"), "lowering": STATE.get("lowering")})

    def verity_cfx_arm(self) -> dict[str, Any]:
        """After the driver's warm-up: install the step hooks and start the capture (record ops ran disabled so far)."""
        from verity_vllm_adapter.compiled_fx import record_op

        hooks = STATE["hooks"]
        cap = STATE.get("cap")
        calls_before = record_op.SINK.calls_total
        hooks.install()
        if cap is not None:
            cap.begin()
        STATE["calls_before"] = calls_before
        torch.cuda.synchronize()
        STATE["log"](f"armed: record calls while disabled {record_op.SINK.calls_while_disabled}, templates {sorted(cap.programs) if cap else None}")
        return _jsonable({"rank": STATE["rank"], "calls_while_disabled": record_op.SINK.calls_while_disabled,
                          "templates": sorted(cap.programs) if cap is not None else None,
                          "nodes_per_template": {str(p): len(n) for p, n in cap.templates.items()} if cap is not None else None,
                          "ports_mapped": len(cap.port_of) if cap is not None else None, "errors": (cap.errors[:10] if cap is not None else [])})

    def verity_cfx_sync(self) -> dict[str, Any]:
        torch.cuda.synchronize()
        hooks = STATE["hooks"]
        cap = STATE.get("cap")
        return _jsonable({"rank": STATE["rank"], "steps": len(hooks.steps), "leaves": cap.leaf_count if cap else None, "replay_units": cap.ru_count if cap else None,
                          "bytes_captured": cap.bytes_captured if cap else None, "errors": hooks.errors[:10]})

    def verity_cfx_finish(self, *, requests_info: dict[str, dict[str, Any]], checkpoint: dict[str, Any], weights_values: str,
                          production_inductor_cache: str | None) -> dict[str, Any]:
        """Digests, record statistics, kernel identity, and (commit/reference) the rank's bundle."""
        import hashlib

        from verity_vllm_adapter.compiled_fx import fx_pass, kernels, record_op
        from verity_vllm_adapter.replay import Req

        hooks = STATE["hooks"]
        cap = STATE.get("cap")
        out = STATE["out"]
        log = STATE["log"]
        rank = STATE["rank"]
        table = STATE["table"]
        rpass = STATE["pass"]
        with (out / "record_digests.jsonl").open("w") as f:
            for row in hooks.digest_rows:
                f.write(json.dumps(row) + "\n")
        ids_seen: dict[int, int] = {}
        piece_to_template: dict[int, set[int]] = {}
        for row in hooks.digest_rows:
            ids_seen[row["id"]] = ids_seen.get(row["id"], 0) + 1
            piece_to_template.setdefault(row["piece"], set()).add(row["id"] // fx_pass.ID_STRIDE)
        calls_per_step = [s["record_calls"] for s in hooks.steps]
        record_stats = {
            "steps": len(hooks.steps), "record_calls_total": record_op.SINK.calls_total - STATE.get("calls_before", 0),
            "calls_per_step_min": min(calls_per_step) if calls_per_step else 0, "calls_per_step_max": max(calls_per_step) if calls_per_step else 0,
            "table_ids": len(table.entries), "distinct_ids_seen": len(ids_seen),
            "ids_in_table_never_seen": sorted(set(table.entries) - set(ids_seen))[:50], "ids_seen_not_in_table": sorted(set(ids_seen) - set(table.entries))[:50],
            "duplicate_id_steps": sum(1 for s in hooks.steps if s["duplicate_ids"]), "arena_overflow_allocs": hooks.arena.overflow_allocs,
            "bytes_total": int(sum(s["bytes"] for s in hooks.steps)), "pieces_compiled_with_pass": len(table.pieces), "pass_calls": rpass.calls,
            "runtime_piece_to_template": {str(k): sorted(v) for k, v in sorted(piece_to_template.items())},
            "pieces_with_ambiguous_template": [k for k, v in piece_to_template.items() if len(v) != 1],
            "ring_final": None if record_op.RING is None else record_op.RING.stats(),
        }
        kernel_identity = None
        cache_dir = STATE.get("cache_dir_effective") or STATE.get("cache_dir")
        if production_inductor_cache:
            prod_dir = Path(production_inductor_cache)
            # production's per-rank Inductor cache when it holds the kernels (vLLM's rank_<r>_0 layout), else the flat directory:
            # torch resolves TORCHINDUCTOR_CACHE_DIR once (lru_cache), before vLLM redirects it per rank, so a production run
            # with the variable set writes both ranks' kernels into the flat directory (observed on cx1)
            def _has_wrappers(p: Path) -> bool:
                return any("async_compile" in f.read_text(errors="ignore") for f in p.rglob("*.py"))

            per_rank = [p for p in sorted(prod_dir.rglob(f"rank_{rank}_*")) if p.is_dir() and _has_wrappers(p)]
            src = per_rank[0] if per_rank else prod_dir
            try:
                kernel_identity = kernels.compare(src, Path(cache_dir))
                kernel_identity["record_attribution"] = {str(k): v for k, v in sorted(kernels.record_attribution(Path(cache_dir)).items())}
                kernel_identity["production_dir_used"] = str(src)
                (out / "kernel_identity.json").write_text(json.dumps(kernel_identity, indent=1) + "\n")
                log(f"kernel identity: identical={kernel_identity['kernels_identical']} prod {kernel_identity['production']} replay {kernel_identity['replay']} "
                    f"only_prod {len(kernel_identity['only_in_production'])} only_replay {len(kernel_identity['only_in_replay'])}")
            except Exception as e:  # noqa: BLE001
                log(f"kernel identity failed: {e!r}")
                kernel_identity = {"error": repr(e)}
        commit_doc: dict[str, Any] = {}
        if cap is not None:
            reqs = [Req(dict(v, request_id=k)) for k, v in requests_info.items()]
            commit_doc = cap.finish(reqs, checkpoint, weights_values, kernel_identity if kernel_identity and "error" not in kernel_identity else None)
        (out / "capture_summary.json").write_text(json.dumps(commit_doc or {"mode": STATE["mode"]}, indent=1, default=str) + "\n")
        (out / "sampler_unforced.json").write_text(json.dumps({rid: {str(t): tok for t, tok in d.items()} for rid, d in hooks.unforced.items()}, indent=0) + "\n")
        (out / "steps.json").write_text(json.dumps({"steps": hooks.steps, "timing": hooks.timing, "errors": hooks.errors}, indent=1, default=str) + "\n")
        log(f"finished: steps {len(hooks.steps)}, leaves {cap.leaf_count if cap else None}, RUs {cap.ru_count if cap else None}, "
            f"{(cap.bytes_captured / 1e9) if cap else 0:.2f} GB; record stats {json.dumps({k: record_stats[k] for k in ('record_calls_total', 'calls_per_step_min', 'calls_per_step_max', 'distinct_ids_seen', 'table_ids')})}")
        digest_sha = hashlib.sha256((out / "record_digests.jsonl").read_bytes()).hexdigest()
        return _jsonable({
            "rank": rank, "out": str(out), "mode": STATE["mode"], "record_stats": record_stats, "steps": hooks.steps, "timing": hooks.timing, "errors": hooks.errors,
            "unforced": {rid: {str(t): tok for t, tok in d.items()} for rid, d in hooks.unforced.items()},
            "commit": commit_doc, "boundary_problems": commit_doc.get("boundary_problems", []) if commit_doc else [],
            "kernel_identity_summary": None if not kernel_identity or "error" in kernel_identity else
            {k: kernel_identity[k] for k in ("kernels_identical", "production", "replay", "common_distinct_kernels", "replay_record_ops_in_wrappers") if k in kernel_identity},
            "kernel_identity_error": kernel_identity.get("error") if kernel_identity else None,
            "record_digests_sha256": digest_sha, "pass": {"uuid": rpass.uuid(), "calls": rpass.calls, "hooks": STATE.get("hooks_installed"), "lowering": STATE.get("lowering"),
                                                          "record_lowering_stats": dict(record_op.LOWERING_STATS)},
            "inductor_cache_dir": cache_dir,
        })
