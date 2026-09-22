"""Minimal REQUIRED-recording path (lane MINREC, 2026-09-09; GUIDANCE §57.9/§57.10 priority 2).

What the protocol needs ONLINE, before the replay draw ``r_q``, is the first boundary of the selected partition
(paper §5.2; instance field ``partition.first_boundary``):

  * rule ``designated outputs only`` (whole-request RUs): the served tokens (+ params / logprobs when bound) — what the
    production recorder (``production.py``) already writes; NO activation row crosses the boundary.
  * rule ``outgoing boundary runs of every CU`` (multi-RU partitions): every value produced in one CU and read in another.
    For the B0 program (Serve_v1, SmolLM2-135M) this module derives the value set per plan from the plan's CU structure
    (``first-stage-boundary.md`` §3.1; DERIVED, not measured):

        layer   RU = (request, decoder layer): the layer's output tuple (hidden, residual) + the embedding output
                = 2 rows / layer / position + 1
        block   P3/cu=block, CUs Embedding | LayerPre | Attention | LayerPost per (layer, position) | Final:
                qkv (v -> Attention), RoPE q, RoPE k (-> Attention, k also -> later positions), attention out
                (-> LayerPost), fused add+norm residual port of input_layernorm (LayerPre -> LayerPost),
                mlp out and post_attention_layernorm residual port (LayerPost -> next LayerPre) = 7 rows / layer /
                position, + the embedding output (Embedding -> LayerPre_0 / LayerPost_0)

This recorder hashes EXACTLY that set, online, per engine step (sha256 leaf per row; Merkle root at close), keeps
everything else off, and never interposes on kernels (module-level forward hooks + pinned D2H copies only).  The
values themselves are not needed after the commitment: post-challenge acquisition (``acquire.replay``) re-executes the
selected RUs under the profile and checks the re-derived boundary rows against the committed leaves; ``--sink file``
retains the raw tape as an alternative to replay.

Security-relevant: a mode that commits LESS than the default must commit exactly the declared first boundary.  The
commitment written here (``boundary_commitment.json``: rule, hook points, rows, bytes, leaves, root, per-step
(request -> scheduled tokens) map) is what the chronology's ``boundary`` event must digest before ``r_q``.  Default
behaviour of every entry point is unchanged (opt-in flag).
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import torch

FIRST_BOUNDARY_RULE = {
    "whole": "designated outputs only",
    "layer": "outgoing boundary runs of every CU",
    "block": "outgoing boundary runs of every CU",
}

# hook point -> (output indices to record, row names).  ``{l}`` = decoder layer index.  Module paths are relative to
# the vLLM model's ``named_modules()`` (LlamaForCausalLM: model.embed_tokens, model.layers.{l}...).
PLAN_BOUNDARY: dict[str, list[dict[str, Any]]] = {
    "whole": [],
    "layer": [
        {"module": "embed_tokens", "outputs": [0], "rows": ["embed"], "note": "Embedding -> layer 0 (residual stream in)"},
        {"module": "layers.{l}", "outputs": [0, 1], "rows": ["mlp_out", "residual"], "note": "decoder layer output tuple -> next layer"},
    ],
    "block": [
        {"module": "embed_tokens", "outputs": [0], "rows": ["embed"], "note": "Embedding -> LayerPre_0 (norm) and LayerPost_0 (residual add)"},
        {"module": "layers.{l}.input_layernorm", "outputs": [1], "rows": ["residual_in"], "min_layer": 1,
         "note": "fused add+norm port 1 (residual stream after layer l-1) LayerPre -> LayerPost; layer 0 has no residual arg (the embed row is it)"},
        {"module": "layers.{l}.self_attn.qkv_proj", "outputs": [0], "rows": ["qkv"], "note": "v read by Attention (q, k by RoPE inside LayerPre)"},
        {"module": "layers.{l}.self_attn.rotary_emb", "outputs": [0, 1], "rows": ["q_rope", "k_rope"], "note": "-> Attention; k also read by later positions"},
        {"module": "layers.{l}.self_attn.attn", "outputs": [0], "rows": ["attn_out"], "note": "Attention -> LayerPost (o_proj)"},
        {"module": "layers.{l}", "outputs": [0, 1], "rows": ["mlp_out", "residual_post"], "note": "LayerPost -> next LayerPre (fused add)"},
    ],
}


def _model_prefix(named: dict[str, Any]) -> tuple[str, int]:
    prefix = "model." if any(n.startswith("model.layers.") for n in named) else ""
    n_layers = sum(1 for n in named if n.startswith(prefix + "layers.") and n[len(prefix) + len("layers."):].isdigit())
    return prefix, n_layers


def merkle_root(leaves: list[bytes]) -> str:
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0].hex()


class _Slab:
    """One pinned host buffer used as a bump allocator for one engine step."""

    def __init__(self, nbytes: int):
        self.buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        self.off = 0
        self.items: list[tuple[str, int, int, int, int]] = []  # (row_name, offset, nbytes, rows, row_bytes)
        self.overflow = 0

    def reset(self):
        self.off = 0
        self.items = []

    def take(self, t: torch.Tensor, row_name: str) -> torch.Tensor | None:
        n = t.numel() * t.element_size()
        start = (self.off + 255) // 256 * 256
        if start + n > self.buf.numel():
            self.overflow += 1
            return None
        self.off = start + n
        rows = t.shape[0] if t.dim() > 1 else 1
        self.items.append((row_name, start, n, rows, n // max(rows, 1)))
        return self.buf[start:start + n].view(t.dtype).view(t.shape)


class BoundaryRecorder:
    """Records ONLY the plan's first-boundary rows: pinned D2H copy on a side stream inside module forward hooks, per-step
    synchronisation, sha256 leaf per row on a background thread (double-buffered slabs), Merkle root at ``close``.

    ``sink``: ``hash`` (copy + leaf hashes; the required online work) | ``file`` (hash + retain the raw tape) | ``none``
    (copy only; the mechanism floor, no commitment — diagnostic rung)."""

    def __init__(self, *, runner: Any, model: torch.nn.Module, plan: str, out_dir: Path, sink: str = "hash",
                 slab_bytes: int = 256 << 20, log: Any = print):
        if plan not in PLAN_BOUNDARY:
            raise ValueError(f"unknown plan {plan!r}")
        self.plan, self.sink, self.log = plan, sink, log
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.runner, self.model = runner, model
        self.slabs = [_Slab(slab_bytes), _Slab(slab_bytes)]
        self.cur = 0
        self.handles: list[Any] = []
        self.hook_points: list[dict[str, Any]] = []
        self.rows = 0
        self.bytes_copied = 0
        self.bytes_hashed = 0
        self.bytes_written = 0
        self.leaves: list[bytes] = []
        self.steps: list[dict[str, Any]] = []
        self.overflow = 0
        self.t_hook = 0.0
        self.t_sync = 0.0
        self.t_hash = 0.0
        self._q: list[tuple[int, list[tuple[str, int, int, int, int]], dict[str, Any]]] = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._stop = False
        self._busy = {0: False, 1: False}
        self.file = open(self.out_dir / f"boundary-{plan}.tape", "ab") if sink == "file" else None
        self._install_hooks()
        self._wrap_runner()
        self._hasher = threading.Thread(target=self._hash_loop, daemon=True) if sink in ("hash", "file") else None
        if self._hasher:
            self._hasher.start()
        self.log(f"BoundaryRecorder(plan={plan}, rule={FIRST_BOUNDARY_RULE[plan]!r}, sink={sink}): {len(self.handles)} hooks; "
                 f"{len(self.hook_points)} hook points")

    # -- installation -------------------------------------------------------------------------------------------------
    def _install_hooks(self) -> None:
        named = dict(self.model.named_modules())
        prefix, n_layers = _model_prefix(named)
        self.n_layers = n_layers
        seen: dict[int, list[str]] = {}
        for spec in PLAN_BOUNDARY[self.plan]:
            layers = range(n_layers) if "{l}" in spec["module"] else [None]
            for l in layers:
                if l is not None and l < spec.get("min_layer", 0):
                    continue
                name = prefix + spec["module"].format(l=l)
                mod = named.get(name)
                if mod is None:
                    if l and any(prefix + spec["module"].format(l=j) in named for j in range(l)):
                        # vLLM shares one RotaryEmbedding instance across layers (named_modules lists it under the first layer
                        # only); the single hook fires once per layer
                        shared_key = id(named[next(prefix + spec["module"].format(l=j) for j in range(l) if prefix + spec["module"].format(l=j) in named)])
                        seen[shared_key].append(name)
                        continue
                    raise SystemExit(f"first-boundary hook point {name} not in the model; adjust PLAN_BOUNDARY[{self.plan}]")
                key = id(mod)
                if key in seen:  # vLLM shares one RotaryEmbedding module across layers: hook it once, it fires once per layer
                    seen[key].append(name)
                    continue
                seen[key] = [name]
                self.handles.append(mod.register_forward_hook(self._hook(spec["outputs"], spec["rows"])))
                self.hook_points.append({"module": name, "outputs": spec["outputs"], "rows": spec["rows"], "note": spec["note"]})
        for key, names in seen.items():
            if len(names) > 1:
                self.hook_points.append({"shared_module": names[0], "also_called_as": names[1:], "note": "one module instance, one hook, fires once per layer"})

    def _hook(self, outputs: list[int], rows: list[str]):
        def fn(module, inputs, output):
            t0 = time.perf_counter()
            outs = output if isinstance(output, (tuple, list)) else (output,)
            slab = self.slabs[self.cur]
            # D2H on the CURRENT stream (as capture.PinnedArena.snapshot): later kernels that reuse the source's memory
            # are ordered behind the copy; the host reads only after the per-step synchronize.
            for idx, rname in zip(outputs, rows):
                if idx >= len(outs) or not torch.is_tensor(outs[idx]):
                    continue
                t = outs[idx].detach()
                src = t if t.is_contiguous() else t.contiguous()
                host = slab.take(src, rname)
                if host is None:
                    continue
                host.copy_(src, non_blocking=True)
                self.rows += src.shape[0] if src.dim() > 1 else 1
                self.bytes_copied += src.numel() * src.element_size()
            self.t_hook += time.perf_counter() - t0
        return fn

    def _wrap_runner(self) -> None:
        rec = self
        self._orig_execute = self.runner.execute_model

        def execute_model(scheduler_output, *a, **kw):
            dummy = kw.get("dummy_run", False) or (len(a) >= 2 and a[1])
            if dummy or getattr(scheduler_output, "total_num_scheduled_tokens", 0) == 0:
                return rec._orig_execute(scheduler_output, *a, **kw)
            rec._begin_step()
            out = rec._orig_execute(scheduler_output, *a, **kw)
            rec._end_step(scheduler_output)
            return out

        self.runner.execute_model = execute_model

    # -- per step ------------------------------------------------------------------------------------------------------
    def _begin_step(self) -> None:
        # take the free slab (wait until the hasher released it)
        with self._cv:
            while self._busy[self.cur]:
                self._cv.wait()
        self.slabs[self.cur].reset()

    def _end_step(self, scheduler_output) -> None:
        t0 = time.perf_counter()
        torch.cuda.current_stream().synchronize()  # every D2H copy of this step has landed
        self.t_sync += time.perf_counter() - t0
        slab = self.slabs[self.cur]
        self.overflow += slab.overflow
        slab.overflow = 0
        sched = {}
        try:
            sched = {str(k): int(v) for k, v in dict(scheduler_output.num_scheduled_tokens).items()}
        except Exception:  # noqa: BLE001
            pass
        batch = None
        try:  # the runner's input batch = the row order of every hooked tensor (as capture.Capture.after_forward reads it)
            ib = self.runner.execute_model_state.input_batch
            n = int(ib.num_reqs)
            batch = {"req_ids": [str(r) for r in list(ib.req_ids)[:n]],
                     "query_start_loc": [int(x) for x in ib.query_start_loc_np[: n + 1]],
                     "num_computed_tokens": [int(x) for x in ib.num_computed_tokens_np[:n]]}
        except Exception:  # noqa: BLE001
            pass
        meta = {"step": len(self.steps), "scheduled_tokens": sched, "batch": batch,
                "rows": sum(i[3] for i in slab.items), "bytes": sum(i[2] for i in slab.items)}
        self.steps.append(meta)
        if self._hasher is None:
            return
        with self._cv:
            self._busy[self.cur] = True
            self._q.append((self.cur, list(slab.items), meta))
            self._cv.notify_all()
        self.cur ^= 1

    def _hash_loop(self) -> None:
        while True:
            with self._cv:
                while not self._q and not self._stop:
                    self._cv.wait()
                if not self._q and self._stop:
                    return
                idx, items, meta = self._q.pop(0)
            t0 = time.perf_counter()
            buf = self.slabs[idx].buf
            step_leaves = 0
            for rname, off, n, rows, row_bytes in items:
                mv = memoryview(buf.numpy())[off:off + n]
                for r in range(rows):
                    chunk = mv[r * row_bytes:(r + 1) * row_bytes]
                    self.leaves.append(hashlib.sha256(chunk).digest())
                    step_leaves += 1
                self.bytes_hashed += n
                if self.file is not None:
                    self.file.write(mv)
                    self.bytes_written += n
            meta["leaves"] = step_leaves
            self.t_hash += time.perf_counter() - t0
            with self._cv:
                self._busy[idx] = False
                self._cv.notify_all()

    # -- teardown ------------------------------------------------------------------------------------------------------
    def close(self) -> dict[str, Any]:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._hasher:
            self._hasher.join()
        for h in self.handles:
            h.remove()
        self.runner.execute_model = self._orig_execute
        if self.file:
            self.file.flush()
            self.file.close()
        root = merkle_root(self.leaves) if self._hasher else None
        summary = {
            "plan": self.plan, "first_boundary_rule": FIRST_BOUNDARY_RULE[self.plan], "sink": self.sink,
            "n_layers": self.n_layers, "hook_points": self.hook_points,
            "rows": self.rows, "bytes_copied": self.bytes_copied, "bytes_hashed": self.bytes_hashed, "bytes_written": self.bytes_written,
            "leaves": len(self.leaves), "root_sha256_merkle": root, "steps": len(self.steps), "slab_overflows": self.overflow,
            "timing_s": {"hooks": self.t_hook, "step_sync": self.t_sync, "hash_thread": self.t_hash},
            "per_step": self.steps,
            "note": "leaf = sha256 of one boundary row (one token position of one hooked output); root = binary Merkle over leaves in "
                    "(step, hook, row) order; per_step.scheduled_tokens + batch_req_order map rows to (request, step) slots",
        }
        (self.out_dir / f"boundary_commitment-{self.plan}.json").write_text(json.dumps(summary, indent=1) + "\n")
        return summary
