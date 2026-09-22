"""Per-rank capture seam for tensor-parallel replay (lane TP, coverage wave axis 2).

With ``tensor_parallel_size > 1`` the model does not live in the driver process: vLLM's multiprocess
executor runs one worker process per rank and the driver only sees scheduler state and outputs. The
dispatch-mode capture (``capture.Capture``) is therefore installed *inside each worker* through vLLM's
``worker_extension_cls`` seam: the methods of this class are mixed into the Worker class of every
rank and are called from the driver with ``LLM.collective_rpc("verity_tp_<method>", kwargs=...)``.

Each rank writes its own run directory ``<out>/rank<r>/`` in the ordinary capture format; every value
and weight it materialises carries ``/s=<r>`` in its canonical id (capture-format field ``s``, the
tensor-parallel rank). ``tp_merge.merge_ranks`` joins the per-rank directories into the run's bundle.

Nothing here changes what the workers compute: the wrappers are the same dispatcher-visible custom
ops the unsharded replay installs (``capture.install_kernel_wrappers``), and the collectives are
recorded as the custom ops vLLM already exposes (``vllm.all_reduce``, ``vllm.all_gather``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


class VerityTPWorkerExtension:
    """Mixed into vLLM's Worker (``worker_extension_cls``); ``self`` is the worker."""

    def verity_tp_install(
        self,
        *,
        run_dir: str,
        profile_id: str,
        manifest: dict[str, Any],
        store_values: bool,
        prompt_lens: dict[str, int],
        arena_bytes: int,
        no_record: bool = False,
    ) -> dict[str, Any]:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

        from verity_vllm.capture import RunWriter
        from verity_vllm_adapter.capture import Capture, install_kernel_wrappers

        rank = int(get_tensor_model_parallel_rank())
        world = int(get_tensor_model_parallel_world_size())
        patches = install_kernel_wrappers()
        out = Path(run_dir) / f"rank{rank}"
        out.mkdir(parents=True, exist_ok=True)
        log_path = out / "replay.log"

        def log(*a: Any) -> None:
            msg = " ".join(str(x) for x in a)
            print(f"[verity rank {rank}] {msg}", flush=True)
            with log_path.open("a") as f:
                f.write(msg + "\n")

        runner = self.model_runner
        model = runner.get_model()
        writer = RunWriter(out, manifest) if store_values else None
        if writer is not None and writer.profile_id != profile_id:
            raise RuntimeError(f"rank {rank}: profile_id mismatch: writer {writer.profile_id} vs production {profile_id}")
        cap = Capture(
            runner=runner, model=model, profile_id=profile_id, run_dir=out, writer=writer, hash_only=not store_values,
            prompt_lens=dict(prompt_lens), arena_bytes=int(arena_bytes), log=log, shard=rank,
        )
        cap.no_record = bool(no_record)
        # unforced sampler result before trace replay overwrites it (§15.3; same hook as replay.py)
        unforced: dict[str, dict[int, int]] = {}
        trs = getattr(getattr(runner, "sampler", None), "trace_replay_state", None)
        if trs is not None:
            orig_apply = trs.apply_trace

            def apply_trace(sampled, idx_mapping):
                cap.cur_unforced = sampled.detach().clone()
                return orig_apply(sampled, idx_mapping)

            trs.apply_trace = apply_trace
        orig_after_sample = cap.after_sample

        def after_sample(res, input_batch):
            orig_after_sample(res, input_batch)
            u = getattr(cap, "cur_unforced", None)
            if u is not None and cap.cur_logits_map is not None:
                vals = u.cpu().numpy().reshape(-1)
                for i, (rid, t) in enumerate(cap.cur_logits_map):
                    if i < len(vals) and cap._is_sampled_position(rid, t):
                        unforced.setdefault(rid, {})[t] = int(vals[i])
                cap.cur_unforced = None

        cap.after_sample = after_sample
        self._verity = {"cap": cap, "writer": writer, "out": out, "unforced": unforced, "patches": patches, "rank": rank,
                        "world": world, "log": log, "trace_replay_hooked": trs is not None}
        torch.cuda.synchronize()
        log(f"capture installed: rank {rank}/{world}, store_values={store_values}, arena {arena_bytes / 2**30:.1f} GiB, patches {patches}")
        return {"rank": rank, "world_size": world, "out": str(out), "patches": patches, "trace_replay_hooked": trs is not None}

    def verity_tp_sync(self) -> dict[str, Any]:
        """Device barrier + the capture's running counters (called after the run)."""
        torch.cuda.synchronize()
        v = self._verity
        cap = v["cap"]
        return {"rank": v["rank"], "leaves": cap.leaf_count, "replay_units": cap.ru_count, "bytes_captured": cap.bytes_captured,
                "steps": cap.step_index}

    def verity_tp_finish(
        self,
        *,
        requests_info: dict[str, dict[str, Any]],
        model_repo: str,
        weights_values: str,
        checkpoint: dict[str, Any],
    ) -> dict[str, Any]:
        """Boundaries, weights (``w=<param>/s=<rank>``), description, capture.json of this rank's run dir."""
        from verity_vllm_adapter import weight_sources

        v = self._verity
        cap, writer, out, log, rank = v["cap"], v["writer"], v["out"], v["log"], v["rank"]
        model = cap.model
        symbols = {"H": int(model.config.hidden_size), "V": int(getattr(model.config, "vocab_size", 0)),
                   "TP": int(v["world"])}
        boundary_problems: list[str] = []
        weights_info: dict[str, Any] = {}
        description = cap.description(model_repo, symbols)
        if writer is not None:
            for b in cap.boundaries(requests_info):
                probs = b.validate()
                if probs:
                    boundary_problems.append(f"{b.request_id}: {probs}")
                else:
                    writer.write_boundary(b)
            t_w0 = time.perf_counter()
            ww = writer.weights(store_values=(weights_values in ("store", "checkpoint")), source_sha256=checkpoint.get("safetensors_sha256"))
            n_params = 0
            n_referenced = 0
            for _ptr, (name, p) in sorted(cap.params.items(), key=lambda kv: kv[1][0]):
                data = p.detach().contiguous().cpu().reshape(-1)
                raw = data.view(torch.uint8).numpy().tobytes()
                src = weight_sources.source_for(name, checkpoint) if weights_values == "checkpoint" else None
                # a rank's parameter is its shard of the checkpoint tensor (or a replica of an unsharded one):
                # identified per rank; how it derives from the checkpoint is the profile's binding
                ww.add(name, str(p.dtype).replace("torch.", ""), list(p.shape), raw, shard=rank, source=src, store=(None if src is None else False))
                n_params += 1
                n_referenced += src is not None
            ww.finish()
            weights_info = {"parameters": n_params, "seconds": time.perf_counter() - t_w0, "values_stored": weights_values == "store",
                            "mode": weights_values, "referenced_to_checkpoint": n_referenced, "shard": rank}
            writer.write_description(description)
            writer.finish()
        else:
            (out / "description.json").write_text(json.dumps(description, indent=1, sort_keys=True))
        summary = cap.summary()
        (out / "capture_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
        (out / "kv_audit.json").write_text(json.dumps(cap.kv_audit, indent=1, default=str) + "\n")
        (out / "sampler_unforced.json").write_text(json.dumps(v["unforced"], indent=0) + "\n")
        log(f"finished: {summary['leaves']} leaves, {cap.ru_count} RUs, {cap.bytes_captured / 1e9:.2f} GB, weights {weights_info}")
        return _jsonable({
            "rank": rank,
            "out": str(out),
            "summary": summary,
            "kv_audit_unresolved": sum(a.get("unresolved", 0) for a in cap.kv_audit),
            "unforced": {rid: {str(t): tok for t, tok in d.items()} for rid, d in v["unforced"].items()},
            "boundary_problems": boundary_problems,
            "weights": weights_info,
            "description_nodes": len(description["nodes"]),
            "patches": v["patches"],
        })
