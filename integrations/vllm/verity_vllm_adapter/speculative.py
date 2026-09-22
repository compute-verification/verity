"""Lane SD: capture and driver for the two-model draft/target speculative configuration (port
``b1-spec``; docs/vllm-poc/speculative.md).

Production and replay are the SAME configuration (stock vLLM with ``speculative_config`` declared as
engine kwargs, the legacy V1 GPUModelRunner because the pinned build's V2 runner does not implement
the ``draft_model`` speculator, batch-invariant eager bf16, TP=1). vLLM's trace replay rejects
speculative decoding, so replay is a deterministic re-execution of the same batch with the capture
installed, and the trajectory check is a post-hoc comparison of the replayed token tape and the
engine's own per-verify-step acceptance record (``per_request_spec_decode_metrics=detailed``) against
production's.

``SpecCapture`` extends the shared ``Capture`` (verity_vllm_adapter/capture.py) ADDITIVELY: the base
class's V2-runner hooks are replaced by V1 ones in this file only; both models are intercepted by one
TorchDispatchMode; the draft model's modules and parameters are named under the ``draft.`` prefix
(capture.py ``DRAFT_PREFIX``, the only shared-file edit besides ``normalize_module`` / ``layer_of`` /
``node_sort_key`` learning that prefix); values of a speculative step at positions the step did not
commit (the target's rejected verification slots and the draft's proposal rows) are captured under
the ``x=<step>`` qualifier of the canonical-id grammar (verity_vllm/identity.py) and the replay
unit's ``speculative`` record (verity_vllm/capture.py).

Run ON THE POD (``/workspace/venv-cu129``)::

    python -m verity_vllm_adapter.speculative production --out <dir> [--requests r1,r3]
    python -m verity_vllm_adapter.speculative replay --production <dir> --out <dir> [--store-values]
    python -m verity_vllm_adapter.speculative compare --a <dir> --b <dir>      # C0 / C1 tapes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# declared environment of the port (vllm_config.json environment_flags) — before vllm is imported
SD_ENV = {
    "VLLM_BATCH_INVARIANT": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_USE_V2_MODEL_RUNNER": "0",
}
for _k, _v in SD_ENV.items():
    os.environ.setdefault(_k, _v)
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402

from verity_vllm import hashing  # noqa: E402
from verity_vllm.capture import Boundary, RunWriter, StepRecord  # noqa: E402
from verity_vllm.identity import value_id  # noqa: E402
from verity_vllm_adapter import profile as prof  # noqa: E402
from verity_vllm_adapter.capture import (  # noqa: E402
    DRAFT_PREFIX,
    Capture,
    NodeTemplate,
    Record,
    install_kernel_wrappers,
    layer_of,
    normalize_module,
    op_name,
)
from verity_vllm_adapter.driver import external_id, run_requests  # noqa: E402

PORT = "b1-spec"
TARGET_REPO = "Qwen/Qwen2.5-1.5B"
DRAFT_REPO = "Qwen/Qwen2.5-0.5B"
NUM_SPECULATIVE_TOKENS = 3
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.8

# The SD workload (speculative.md §2): short greedy prompts chosen to exercise the five cases; the
# cases are OUTCOMES of the run (which request hits which case is read off the acceptance record).
SD_PROMPTS: list[tuple[str, str, int]] = [
    ("sd-count", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,", 24),
    ("sd-alpha", "a b c d e f g h i j k l m n o p", 24),
    ("sd-capital", "The capital of France is", 12),
    ("sd-code", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return", 20),
    ("sd-story", "Once upon a time, in a small village by the sea, there lived", 40),
    ("sd-lipsum", "The quick brown fox", 10),
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------
# capture
# ----------------------------------------------------------------------------------------------
class SpecCapture(Capture):
    """Capture over vLLM's legacy V1 ``GPUModelRunner`` running the two-model speculative loop.

    Per engine step: ``execute_model`` runs the target over every scheduled row (a request's k+1
    verification slots, or its prefill chunk) and its lm_head over the k+1 slots; ``sample_tokens``
    runs the rejection sampler (greedy: accepted iff draft == target argmax), then the drafter's
    first pass (the draft model over the request's committed rows it has not seen — the accepted
    slots — plus the bonus row, padded to k+2 rows per request) and k-1 single-row passes.
    """

    def __init__(self, *, drafter: Any, num_speculative_tokens: int, **kw: Any) -> None:
        self.drafter = drafter
        self.draft_model = drafter.model
        self.k = int(num_speculative_tokens)
        self.draft_pass = -1
        self.draft_positions: dict[int, np.ndarray] = {}
        self.draft_input_ids: dict[int, np.ndarray] = {}
        self.draft_rows: dict[int, int] = {}
        self.spec_problems: list[dict[str, Any]] = []
        self.acceptance: dict[str, list[dict[str, Any]]] = {}  # request -> per-verify-step record
        self.padding_rows_skipped = 0
        self.draft_leaves = 0
        self.dead_leaves = 0
        self.rejected_slot_leaves = 0
        self.spec_steps = 0
        self.discarded_steps: list[dict[str, Any]] = []
        self.cur_sampled_full: torch.Tensor | None = None
        self.cur_draft_out: torch.Tensor | None = None
        self.cur_spec_md: Any = None
        self.cur_n_logits = 0
        self.cur_logits_rows: list[tuple[int, int, int]] = []  # (request index, slot j, position)
        self.cur_target_argmax: np.ndarray | None = None
        self.cur_k: list[int] = []
        self.cur_cands: list[list[int]] = []
        self.prev_draft_out: dict[str, list[int]] = {}
        self._active = False
        super().__init__(**kw)
        self._install_draft()

    # ---------------- installation ----------------
    def _install_draft(self) -> None:
        dm = self.draft_model
        for name, p in list(dm.named_parameters()) + list(dm.named_buffers()):
            if not p.is_cuda:
                continue
            ptr = p.data_ptr()
            nm = normalize_module("draft_model." + name)
            if ptr not in self.params:
                self.params[ptr] = (nm, p)
                self.param_ranges.append((ptr, ptr + p.numel() * p.element_size(), nm))
        self.param_ranges.sort()
        cap = self

        def root_pre(mod, args, kwargs):
            cap.draft_pass += 1
            cap.phase = "forward"
            pos = kwargs.get("positions") if kwargs else None
            ids = kwargs.get("input_ids") if kwargs else None
            if pos is None and len(args) >= 2:
                ids, pos = args[0], args[1]
            try:
                if isinstance(pos, torch.Tensor):
                    cap.draft_positions[cap.draft_pass] = pos.detach().cpu().numpy().reshape(-1).copy()
                    cap.draft_rows[cap.draft_pass] = int(pos.shape[-1])
                if isinstance(ids, torch.Tensor):
                    cap.draft_input_ids[cap.draft_pass] = ids.detach().cpu().numpy().reshape(-1).copy()
            except Exception as e:  # noqa: BLE001
                cap.spec_problems.append({"where": "draft_root_pre", "error": repr(e), "step": cap.step_index})

        def root_post(mod, args, output):
            cap.phase = "sample"

        def pre(mod, args):
            name = mod._verity_name
            if cap.module_stack:
                parent = cap.module_stack[-1]
                # a module instance shared across the draft's layers (get_rope cache) is registered
                # under the first layer's path; name it after the calling layer
                pp = parent[len(DRAFT_PREFIX):] if parent.startswith(DRAFT_PREFIX) else None
                own = name[len(DRAFT_PREFIX):]
                if pp is not None and pp.startswith("layers.") and own.startswith("layers."):
                    lp, lo = pp.split(".")[1], own.split(".")[1]
                    if lp != lo:
                        name = DRAFT_PREFIX + "layers." + lp + own[len("layers." + lo):]
            cap.module_stack.append(name)

        def post(mod, args, output):
            name = cap.module_stack.pop() if cap.module_stack else None
            if not cap.mode.enabled:
                return
            outs = output if isinstance(output, (tuple, list)) else (output,)
            for o in outs:
                if isinstance(o, torch.Tensor) and o.is_cuda and o.numel() > 0:
                    rec, _ = cap.registry.lookup(o)
                    if rec is None and cap._param_of(o) is None:
                        cap.module_audit.append({"module": name, "shape": list(o.shape), "dtype": str(o.dtype), "step": cap.step_index})

        for name, mod in dm.named_modules():
            if name == "":
                mod.register_forward_pre_hook(root_pre, with_kwargs=True)
                mod.register_forward_hook(root_post)
                continue
            mod._verity_name = normalize_module("draft_model." + name)
            mod.register_forward_pre_hook(pre)
            mod.register_forward_hook(post)

    def _wrap_runner(self) -> None:
        runner = self.runner
        cap = self
        orig_execute = runner.execute_model
        orig_sample_tokens = runner.sample_tokens
        orig__sample = runner._sample
        orig_propose = runner.propose_draft_token_ids
        orig_compute_logits = self.model.compute_logits

        def execute_model(scheduler_output, *a, **kw):
            if scheduler_output.total_num_scheduled_tokens == 0:
                return orig_execute(scheduler_output, *a, **kw)
            cap.begin_step(scheduler_output)
            cap._active = True
            t0 = time.perf_counter()
            if cap.no_record:
                out = orig_execute(scheduler_output, *a, **kw)
            else:
                with cap.mode:
                    cap.mode.enabled = True
                    try:
                        out = orig_execute(scheduler_output, *a, **kw)
                    finally:
                        cap.mode.enabled = False
            cap.timing["forward_s"] += time.perf_counter() - t0
            cap.after_forward()
            return out

        def compute_logits(*a, **kw):
            prev = cap.phase
            cap.phase = "sample"
            try:
                return orig_compute_logits(*a, **kw)
            finally:
                cap.phase = prev

        def _sample(logits, spec_md):
            res = orig__sample(logits, spec_md)
            try:
                cap.cur_sampled_full = res.sampled_token_ids.detach().clone()
            except Exception as e:  # noqa: BLE001
                cap.errors.append({"where": "_sample", "error": repr(e)})
            return res

        def propose_draft_token_ids(*a, **kw):
            out = orig_propose(*a, **kw)
            cap.cur_draft_out = out.detach().clone() if isinstance(out, torch.Tensor) else out
            return out

        def sample_tokens(grammar_output):
            if not cap._active:
                return orig_sample_tokens(grammar_output)
            cap.phase = "sample"
            t0 = time.perf_counter()
            if cap.no_record:
                res = orig_sample_tokens(grammar_output)
            else:
                with cap.mode:
                    cap.mode.enabled = True
                    try:
                        res = orig_sample_tokens(grammar_output)
                    finally:
                        cap.mode.enabled = False
            cap.timing["forward_s"] += time.perf_counter() - t0
            cap.phase = "forward"
            cap._active = False
            cap.after_sample_v1()
            return res

        runner.execute_model = execute_model
        runner.sample_tokens = sample_tokens
        runner._sample = _sample
        runner.propose_draft_token_ids = propose_draft_token_ids
        self.model.compute_logits = compute_logits

    # ---------------- step lifecycle ----------------
    def begin_step(self, scheduler_output) -> None:
        super().begin_step(scheduler_output)
        self.draft_pass = -1
        self.draft_positions = {}
        self.draft_input_ids = {}
        self.draft_rows = {}
        self.cur_sampled_full = None
        self.cur_draft_out = None
        self.cur_target_argmax = None

    def after_forward(self) -> None:
        runner = self.runner
        st = runner.execute_model_state
        ib = runner.input_batch
        n = int(ib.num_reqs)
        qsl = np.asarray(runner.query_start_loc.np[: n + 1]).astype(np.int64).copy()
        nct = np.asarray(ib.num_computed_tokens_cpu[:n]).astype(np.int64).copy()
        self.cur_qsl = qsl
        self.cur_nct = nct
        self.cur_req_ids = [external_id(r) for r in ib.req_ids[:n]]
        self.cur_num_tokens = int(qsl[-1])
        rows_req = np.repeat(np.arange(n), np.diff(qsl))
        rows_pos = nct[rows_req] + (np.arange(qsl[-1]) - qsl[rows_req])
        self.cur_rows_req = rows_req
        self.cur_rows_pos = rows_pos
        md = st.spec_decode_metadata
        self.cur_spec_md = md
        if md is None:
            self.cur_k = [0] * n
            self.cur_cands = [[] for _ in range(n)]
        else:
            self.cur_k = [int(x) for x in md.num_draft_tokens]
            flat = md.draft_token_ids.detach().cpu().numpy().reshape(-1).tolist()
            cands, off = [], 0
            for k in self.cur_k:
                cands.append([int(x) for x in flat[off: off + k]])
                off += k
            self.cur_cands = cands
        rows: list[tuple[int, int, int]] = []
        for i in range(n):
            nsched = int(qsl[i + 1] - qsl[i])
            k = self.cur_k[i]
            for j in range(k + 1):
                rows.append((i, j, int(nct[i] + nsched - 1 - k + j)))
        self.cur_logits_rows = rows
        self.cur_n_logits = len(rows)
        self.cur_logits_map = [(self.cur_req_ids[i], t) for i, _, t in rows]
        try:
            logits = st.logits
            if logits is not None and int(logits.shape[0]) >= self.cur_n_logits:
                self.cur_target_argmax = logits[: self.cur_n_logits].argmax(dim=-1).detach().cpu().numpy().reshape(-1)
            else:
                self.spec_problems.append({"where": "after_forward", "step": self.step_index,
                                           "error": f"logits rows {None if logits is None else list(logits.shape)} != {self.cur_n_logits}"})
        except Exception as e:  # noqa: BLE001
            self.spec_problems.append({"where": "after_forward.argmax", "step": self.step_index, "error": repr(e)})
        self._record_kv_owners_v1(st)

    def _record_kv_owners_v1(self, st) -> None:
        sm = st.slot_mappings
        first = None
        if isinstance(sm, dict) and sm:
            first = next(iter(sm.values()))
        elif isinstance(sm, list) and sm and isinstance(sm[0], dict) and sm[0]:
            first = next(iter(sm[0].values()))
        if first is None:
            return
        slots = first[: self.cur_num_tokens].detach().cpu().numpy()
        self.kv_groups = 1
        for i in range(self.cur_num_tokens):
            self.owner[int(slots[i])] = (self.cur_req_ids[self.cur_rows_req[i]], int(self.cur_rows_pos[i]))
        # audit the target's reads: block-table walk over positions 0..t_max per request
        try:
            bt_obj = self.runner.input_batch.block_table.block_tables[0]
            bt0 = np.asarray(bt_obj.block_table.np[: len(self.cur_req_ids)])
            bs = int(self.runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size)
        except Exception as e:  # noqa: BLE001
            self.kv_audit.append({"step": self.step_index, "error": repr(e)})
            return
        for i, rid in enumerate(self.cur_req_ids):
            t_max = int(self.cur_nct[i] + (self.cur_qsl[i + 1] - self.cur_qsl[i]) - 1)
            bad = []
            nbad = 0
            for t in range(t_max + 1):
                slot = int(bt0[i][t // bs]) * bs + t % bs
                own = self.owner.get(slot)
                if own == (rid, t):
                    continue
                nbad += 1
                if len(bad) <= 8:
                    bad.append({"t": t, "slot": slot, "owner": own})
            self.kv_audit.append({"step": self.step_index, "request_id": rid, "positions_read": t_max + 1, "unresolved": nbad,
                                  "examples": bad, "block_size": bs, "cross_request_resolved": 0})

    def after_sample_v1(self) -> None:
        try:
            torch.cuda.synchronize()
            self.cur_sampled = None
            if self.cur_sampled_full is not None:
                self.cur_sampled_full_np = self.cur_sampled_full.detach().cpu().numpy().reshape(len(self.cur_req_ids), -1)
            else:
                self.cur_sampled_full_np = None
            if isinstance(self.cur_draft_out, torch.Tensor):
                self.cur_draft_out_np = self.cur_draft_out.detach().cpu().numpy().reshape(len(self.cur_req_ids), -1)
            else:
                self.cur_draft_out_np = None
        except Exception as e:  # noqa: BLE001
            self.errors.append({"where": "after_sample_v1", "error": repr(e)})
            self.cur_sampled_full_np = None
            self.cur_draft_out_np = None
        if not self.no_record:
            self.emit_step()
        self.step_index += 1

    # ---------------- op recording ----------------
    def _input_kind(self, t: torch.Tensor) -> str | None:
        ptr = t.data_ptr()
        r = self.runner
        try:
            if ptr == r.input_ids.gpu.data_ptr() or ptr == self.drafter.input_ids.data_ptr():
                return "input_ids"
            if ptr == r.positions.data_ptr() or ptr == self.drafter.positions.data_ptr():
                return "positions"
        except Exception:  # noqa: BLE001
            pass
        return self.input_alias.get(ptr)

    def on_op(self, func, args, kwargs, out) -> None:
        name = op_name(func)
        if not self.module_stack and self.phase == "forward":
            # V1 runner: the sampled-row gather (hidden_states[logits_indices]; the drafter's
            # last_hidden_states[token_indices_to_sample]) is a top-level op of the forward phase, an
            # alias with a row remap; every other top-level op of the runner's own bookkeeping
            # (slot mappings, index arithmetic) is not a value of the graph
            if name in ("aten.index", "aten.index_select"):
                src = args[0] if args else None
                if isinstance(src, torch.Tensor) and self.registry.lookup(src)[0] is not None:
                    self.op_histogram[name] = self.op_histogram.get(name, 0) + 1
                    self._on_gather_alias(name, args, out)
                else:
                    self.op_histogram["forward-top-skipped:" + name] = self.op_histogram.get("forward-top-skipped:" + name, 0) + 1
                return
            if name not in ("aten.copy_", "aten._to_copy"):
                self.op_histogram["forward-top-skipped:" + name] = self.op_histogram.get("forward-top-skipped:" + name, 0) + 1
                return
        n_before = len(self.registry.records)
        super().on_op(func, args, kwargs, out)
        if self.draft_pass >= 0 and len(self.registry.records) > n_before:
            for r in self.registry.records[n_before:]:
                if r.node.startswith(DRAFT_PREFIX):
                    r.extra["draft_pass"] = self.draft_pass

    def _on_gather_alias(self, name: str, args, out) -> None:
        # the drafter's gathers of its own hidden states are aliases of draft nodes; row kind
        # "logits" so the base's classification by row count does not apply (emit_step decides)
        src = args[0] if args else None
        if self.phase == "sample" and isinstance(src, torch.Tensor) and self.registry.lookup(src)[0] is None \
                and self._input_kind(src) is None:
            # the rejection sampler's own row gathers: its float32 copy of the verification logits split
            # into bonus / target rows, and its index tensors. Sampler internals, not values of the graph:
            # the target's logits at every slot are recorded at lm_head.gemm, the per-slot argmax at
            # sampler.select, and the verdicts are recomputed from those (Accept). The base rule for
            # sampler kernels (sample-phase-skipped) applies to these gathers too.
            key = f"sample-phase-skipped-gather:{name}:{list(src.shape)}"
            self.op_histogram[key] = self.op_histogram.get(key, 0) + 1
            return
        super()._on_gather_alias(name, args, out)

    def _attn_consts(self, module: str) -> dict[str, Any]:
        if module.startswith(DRAFT_PREFIX):
            out: dict[str, Any] = {}
            try:
                for name, mod in self.draft_model.named_modules():
                    if normalize_module("draft_model." + name) == module:
                        impl = mod.impl
                        out = {
                            "num_heads": int(impl.num_heads), "num_kv_heads": int(impl.num_kv_heads), "head_size": int(impl.head_size),
                            "scale": float(impl.scale), "sliding_window": getattr(impl, "sliding_window", None),
                            "logits_soft_cap": getattr(impl, "logits_soft_cap", None), "kv_cache_dtype": str(getattr(impl, "kv_cache_dtype", "auto")),
                            "fa_version": getattr(impl, "vllm_flash_attn_version", None), "num_splits": 1,
                            "block_size": int(self.runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size),
                            "backend": getattr(getattr(mod, "attn_backend", None), "__name__", None), "model": "draft",
                        }
                        break
            except Exception as e:  # noqa: BLE001
                out["error"] = repr(e)
            return out
        return super()._attn_consts(module)

    def _attn_impl(self, module: str) -> str:
        if module.startswith(DRAFT_PREFIX):
            try:
                for name, mod in self.draft_model.named_modules():
                    if normalize_module("draft_model." + name) == module:
                        impl = getattr(mod, "impl", None)
                        be = getattr(mod, "attn_backend", None)
                        return f"{impl.__class__.__module__}.{impl.__class__.__name__} backend={getattr(be, '__name__', be)}"
            except Exception:  # noqa: BLE001
                pass
            return "unresolved"
        return super()._attn_impl(module)

    # ---------------- emission ----------------
    def _sampler_template(self, op: str, *, layer: str, runtime_impl: str, args: list[dict[str, Any]], dtype: str, row_shape: list[Any]) -> NodeTemplate:
        tpl = self.templates.get(op)
        if tpl is None:
            tpl = NodeTemplate(op=op, layer=layer, family="sampler", op_type="sampler", runtime_op="vllm.v1.worker.gpu_model_runner.sample_tokens",
                               runtime_impl=runtime_impl, args=args, row_kind="logits")
            self.templates[op] = tpl
        tpl.ports.setdefault(0, {"dtype": dtype, "row_shape": row_shape, "row_kind": "logits"})
        return tpl

    def emit_step(self) -> None:
        t0 = time.perf_counter()
        torch.cuda.synchronize()
        self.timing["sync_s"] += time.perf_counter() - t0
        t1 = time.perf_counter()
        self.cur_sampled_positions = set()
        n = len(self.cur_req_ids)
        n_tokens = self.cur_num_tokens
        n_logits = self.cur_n_logits
        recs = [r for r in self.registry.records if r.host is not None]
        target_token: list[Record] = []
        target_logits: list[Record] = []
        draft_token: dict[int, list[Record]] = {}
        draft_logits: dict[int, list[Record]] = {}
        for r in recs:
            if r.node.startswith(DRAFT_PREFIX):
                p = r.extra.get("draft_pass")
                if p is None:
                    self.non_token_outputs[r.node] = self.non_token_outputs.get(r.node, 0) + 1
                    continue
                if r.node.startswith(DRAFT_PREFIX + "lm_head"):
                    if r.shape[0] == n:
                        draft_logits.setdefault(p, []).append(r)
                    else:
                        self.non_token_outputs[r.node] = self.non_token_outputs.get(r.node, 0) + 1
                else:
                    rows = self.draft_rows.get(p)
                    if rows is not None and r.shape[0] == rows:
                        draft_token.setdefault(p, []).append(r)
                    else:
                        r.row_kind = "none"
                        self.non_token_outputs[r.node] = self.non_token_outputs.get(r.node, 0) + 1
            elif r.node.startswith("lm_head") or (r.row_kind == "logits" and r.shape[0] == n_logits and r.shape[0] != n_tokens):
                if r.shape[0] == n_logits:
                    target_logits.append(r)
                else:
                    self.non_token_outputs[r.node] = self.non_token_outputs.get(r.node, 0) + 1
            elif r.shape[0] == n_tokens:
                target_token.append(r)
            else:
                r.row_kind = "none"
                self.non_token_outputs[r.node] = self.non_token_outputs.get(r.node, 0) + 1

        sampled = getattr(self, "cur_sampled_full_np", None)
        draft_out = getattr(self, "cur_draft_out_np", None)
        argmax = self.cur_target_argmax
        pos0 = self.draft_positions.get(0)
        n_passes = self.draft_pass + 1
        rows_by_req: dict[int, list[int]] = {i: [] for i in range(n)}
        for row in range(n_tokens):
            rows_by_req[int(self.cur_rows_req[row])].append(row)
        logit_rows_by_req: dict[int, list[tuple[int, int, int]]] = {i: [] for i in range(n)}
        for lrow, (i, j, t) in enumerate(self.cur_logits_rows):
            logit_rows_by_req[i].append((lrow, j, t))

        for i, rid in enumerate(self.cur_req_ids):
            step = self.req_steps.get(rid, 0)
            nsched = int(self.cur_qsl[i + 1] - self.cur_qsl[i])
            k = self.cur_k[i]
            cand = self.cur_cands[i]
            scheduled = [int(self.cur_nct[i] + q) for q in range(nsched)]
            slots = logit_rows_by_req[i]  # k+1 verification slots (or the single sampled row of a chunk)
            t0_slot = slots[0][2]
            # --- Accept: the rejection sampler's row against the candidates ---
            row = sampled[i] if sampled is not None else None
            a = 0
            committed: list[int] = []
            verdicts: list[bool] = []
            problems: list[str] = []
            if row is not None:
                while a < k and int(row[a]) == cand[a]:
                    a += 1
                verdicts = [bool(j < a) for j in range(k)]
                committed = [int(x) for x in row[: a + 1]]
                if any(x < 0 for x in committed):
                    problems.append(f"committed slot negative: {committed}")
                tail = [int(x) for x in row[a + 1:]]
                if any(x != -1 for x in tail):
                    problems.append(f"slots after the first rejection not -1: {tail}")
                if k and argmax is not None:
                    for lrow, j, t in slots:
                        if j <= a and int(argmax[lrow]) != int(row[j]):
                            problems.append(f"slot {j}: target argmax {int(argmax[lrow])} != emitted {int(row[j])}")
            else:
                problems.append("no sampled row")
            plen = self.prompt_lens.get(rid)
            is_prefill_chunk = plen is not None and scheduled[-1] < plen - 1
            # previous step's proposal must be this step's candidates
            prev = self.prev_draft_out.get(rid)
            if k and prev is not None and prev[:k] != cand:
                problems.append(f"candidates {cand} != previous proposal {prev}")
            # --- Advance: live positions of this step (termination truncates the block) ---
            total = self.tokens.get(rid)
            limit = (len(total) - 2) if total else None  # last position the request ever processes
            live = list(scheduled)
            if k or not is_prefill_chunk:
                block = [t for _, _, t in slots]
                live = [t for t in scheduled if t not in block[a + 1:]]
            if limit is not None:
                live = [t for t in live if t <= limit]
            live_set = set(live)
            dead_target = [t for t in scheduled if t not in live_set]
            t_bonus = t0_slot + a + 1 if not is_prefill_chunk else None
            draft_props = [t_bonus + p for p in range(self.k)] if t_bonus is not None and n_passes > 0 else []
            dead = sorted(set(dead_target) | set(draft_props))
            dead = [t for t in dead if t not in live_set]
            if not live:
                # async scheduling (vLLM's default with draft_model): a request that reached its end at
                # the previous step is scheduled once more before the stop check runs; the engine
                # discards this step's tokens (they are not in the tape, the scheduler's own acceptance
                # record does not count the step). Nothing reads the KV rows it wrote. No replay unit:
                # the step is recorded as discarded, its leaves are not captured.
                self.discarded_steps.append({"request_id": rid, "engine_step": self.step_index, "k": k, "accepted": a,
                                             "positions": scheduled, "candidates": cand, "would_commit": committed})
                self.acceptance.setdefault(rid, []).append({"step": None, "engine_step": self.step_index, "k": k, "accepted": a,
                                                            "candidates": cand, "committed_tokens": committed, "live": [], "dead": scheduled,
                                                            "discarded": True})
                if draft_out is not None:
                    self.prev_draft_out[rid] = [int(x) for x in draft_out[i]]
                continue
            if plen is None or live[-1] < plen:
                phase = "prefill"
            elif live[0] >= plen:
                phase = "decode"
            else:
                phase = "mixed"
            spec_rec: dict[str, Any] = {
                "k": k, "accepted": a, "candidates": cand, "verdicts": verdicts, "committed_tokens": committed,
                "slots": [t for _, _, t in slots], "dead": dead, "num_speculative_tokens": self.k,
                "kv": {"committed": [t for t in scheduled if t in live_set], "discarded": dead_target},
                "advance": {"n_next": (live[-1] + 1), "live": [bool(t in live_set) for _, _, t in slots]},
                "draft": {"bonus_position": t_bonus, "proposal_positions": draft_props,
                          "proposed": [int(x) for x in draft_out[i]] if draft_out is not None else None},
                "engine_step": self.step_index,
            }
            if problems:
                spec_rec["problems"] = problems
                self.spec_problems.append({"request_id": rid, "step": step, "engine_step": self.step_index, "problems": problems})
            if k:
                self.spec_steps += 1
            self.acceptance.setdefault(rid, []).append({"step": step, "engine_step": self.step_index, "k": k, "accepted": a,
                                                        "candidates": cand, "committed_tokens": committed, "live": live, "dead": dead})
            if draft_out is not None:
                self.prev_draft_out[rid] = [int(x) for x in draft_out[i]]
            emitted = self.req_emitted_positions.setdefault(rid, set())
            again = [t for t in live if t in emitted]
            recompute = bool(again)
            emitted.update(live)
            self.req_records.setdefault(rid, []).append(StepRecord(step, phase, tuple(live), recompute, spec_rec))
            self.req_steps[rid] = step + 1

            # --- entries: (record, row, position, qualified) ---
            entries: list[tuple[Record, int, int, bool]] = []
            for r in target_token:
                for row_i in rows_by_req[i]:
                    t = int(self.cur_rows_pos[row_i])
                    entries.append((r, row_i, t, t not in live_set))
            for r in target_logits:
                for lrow, j, t in slots:
                    if is_prefill_chunk and not k:
                        continue  # a partial chunk's logits row is discarded by the scheduler
                    q = t not in live_set
                    entries.append((r, lrow, t, q))
                    if q:
                        self.rejected_slot_leaves += 1
            # draft rows: first pass = valid (catch-up) rows + bonus row + padding; gen passes = one row per request
            if pos0 is not None and n_passes > 0:
                rej = k - a
                base = int(self.cur_qsl[i]) + i
                n_valid = nsched - rej
                exp_rows = [(base + q, scheduled[q]) for q in range(n_valid)]
                if t_bonus is not None:
                    exp_rows.append((base + n_valid, t_bonus))
                ok = all(r_i < len(pos0) and int(pos0[r_i]) == t for r_i, t in exp_rows)
                if not ok:
                    got = [int(pos0[r_i]) if r_i < len(pos0) else None for r_i, _ in exp_rows]
                    self.spec_problems.append({"request_id": rid, "step": step, "engine_step": self.step_index,
                                               "problems": [f"draft first-pass rows: expected positions {[t for _, t in exp_rows]} got {got}"]})
                else:
                    self.padding_rows_skipped += (nsched + 1) - len(exp_rows)
                    for r in draft_token.get(0, []):
                        for r_i, t in exp_rows:
                            entries.append((r, r_i, t, t not in live_set))
                    for r in draft_logits.get(0, []):
                        if t_bonus is not None:
                            entries.append((r, i, t_bonus, True))
                for p in range(1, n_passes):
                    posp = self.draft_positions.get(p)
                    t_p = t_bonus + p if t_bonus is not None else None
                    if posp is None or t_p is None or i >= len(posp) or int(posp[i]) != t_p:
                        self.spec_problems.append({"request_id": rid, "step": step, "engine_step": self.step_index,
                                                   "problems": [f"draft pass {p}: expected position {t_p} got {None if posp is None or i >= len(posp) else int(posp[i])}"]})
                        continue
                    for r in draft_token.get(p, []):
                        entries.append((r, i, t_p, True))
                    for r in draft_logits.get(p, []):
                        entries.append((r, i, t_p, True))
            # scalar leaves: (layer, op, t, qualified, dtype, shape, bytes)
            scalars: list[tuple[str, str, int, bool, str, list[int], bytes]] = []
            if row is not None and not (is_prefill_chunk and not k):
                for lrow, j, t in slots:
                    if j <= a:
                        if t in live_set:
                            tok = int(row[j])
                            scalars.append(("sampler", "sampler.select", t, False, "int32", [], np.array(tok, dtype=np.int32).tobytes()))
                            self.req_outputs.setdefault(rid, []).append(tok)
                            self.req_sampled[rid] = self.req_sampled.get(rid, 0) + 1
                            self.cur_sampled_positions.add((rid, t))
                        elif argmax is not None:
                            scalars.append(("sampler", "sampler.select", t, True, "int32", [], np.array(int(argmax[lrow]), dtype=np.int32).tobytes()))
                    elif argmax is not None:
                        scalars.append(("sampler", "sampler.select", t, True, "int32", [], np.array(int(argmax[lrow]), dtype=np.int32).tobytes()))
                if k:
                    scalars.append(("sampler", "sampler.candidates", t0_slot, False, "int32", [k], np.asarray(cand, dtype=np.int32).tobytes()))
                    scalars.append(("sampler", "sampler.accept", t0_slot, False, "int32", [k + 1], np.asarray([int(x) for x in row[: k + 1]], dtype=np.int32).tobytes()))
            if draft_out is not None and t_bonus is not None:
                for p in range(min(self.k, draft_out.shape[1])):
                    scalars.append(("sampler", DRAFT_PREFIX + "sampler.select", t_bonus + p, True, "int32", [], np.array(int(draft_out[i][p]), dtype=np.int32).tobytes()))
            self._write_ru_spec(rid, step, phase, live, entries, scalars, spec_rec, recompute)
        self.timing["emit_s"] += time.perf_counter() - t1

    def _write_ru_spec(self, rid: str, step: int, phase: str, positions: list[int], entries, scalars, spec_rec: dict[str, Any], recompute: bool) -> None:
        leaves: dict[str, dict[str, Any]] = {}
        ru = None
        if self.writer is not None:
            ru = self.writer.ru(rid, step, phase=phase, positions=positions, on_duplicate="flag", speculative=spec_rec)
        for r, row, t, qualified in entries:
            tpl = self.templates.get(r.node)
            layer = tpl.layer if tpl is not None else layer_of(r.node)
            dtype = str(r.dtype).replace("torch.", "")
            cid = str(value_id(self.profile_id, rid, t=t, phase=self._phase_at(rid, t), layer=layer, op=r.node, port=r.port, shard=self.shard,
                               speculative=(step if qualified else None)))
            data = r.host[row].reshape(-1).view(torch.uint8).numpy().tobytes()
            shape = list(r.shape[1:])
            if ru is not None:
                ru.add(cid, dtype, shape, data)
            else:
                leaves[cid] = {"dtype": dtype, "shape": shape, "nbytes": len(data), "leaf": hashing.leaf_hash(cid, dtype, shape, data).hex()}
            self.leaf_count += 1
            self.bytes_captured += len(data)
            if qualified:
                self.dead_leaves += 1
            if r.node.startswith(DRAFT_PREFIX):
                self.draft_leaves += 1
        for layer, op, t, qualified, dtype, shape, data in scalars:
            cid = str(value_id(self.profile_id, rid, t=t, phase=self._phase_at(rid, t), layer=layer, op=op, port=0, shard=self.shard,
                               speculative=(step if qualified else None)))
            if ru is not None:
                ru.add(cid, dtype, shape, data)
            else:
                leaves[cid] = {"dtype": dtype, "shape": shape, "nbytes": len(data), "leaf": hashing.leaf_hash(cid, dtype, shape, data).hex()}
            self.leaf_count += 1
            if qualified:
                self.dead_leaves += 1
            if op == "sampler.select":
                self._sampler_template("sampler.select", layer="sampler",
                                       runtime_impl="vllm.v1.sample.rejection_sampler (greedy branch: target argmax per verification slot; emitted iff the slot is live)",
                                       args=[{"name": "logits", "source": "node", "node": "lm_head.gemm", "port": 0},
                                             {"name": "sampling_params", "source": "boundary", "field": "sampling_params"}], dtype="int32", row_shape=[]).instances += 1
            elif op == "sampler.candidates":
                self._sampler_template("sampler.candidates", layer="sampler",
                                       runtime_impl="SpecDecodeMetadata.draft_token_ids: the k draft tokens verified at this step (the previous step's draft.sampler.select values)",
                                       args=[{"name": "draft", "source": "node", "node": DRAFT_PREFIX + "sampler.select", "port": 0}], dtype="int32", row_shape=[self.k]).instances += 1
            elif op == "sampler.accept":
                self._sampler_template("sampler.accept", layer="sampler",
                                       runtime_impl="vllm.v1.sample.rejection_sampler.RejectionSampler output row: candidate where accepted, the target argmax at the first rejection (or the bonus slot), -1 after (verity_vllm.registry.spec Accept)",
                                       args=[{"name": "candidates", "source": "node", "node": "sampler.candidates", "port": 0},
                                             {"name": "target_argmax", "source": "node", "node": "sampler.select", "port": 0}], dtype="int32", row_shape=[self.k + 1]).instances += 1
            elif op == DRAFT_PREFIX + "sampler.select":
                self._sampler_template(DRAFT_PREFIX + "sampler.select", layer="sampler",
                                       runtime_impl="vllm.v1.spec_decode.llm_base_proposer.SpecDecodeBaseProposer._greedy_sample: argmax over the draft's logits",
                                       args=[{"name": "logits", "source": "node", "node": DRAFT_PREFIX + "lm_head.gemm", "port": 0}], dtype="int32", row_shape=[]).instances += 1
        for tpl in self.templates.values():
            tpl.phases.update({self._phase_at(rid, positions[0]), self._phase_at(rid, positions[-1])})
        if ru is not None:
            ru.finish()
        else:
            d = self.run_dir / "ru" / rid / f"{step:06d}"
            d.mkdir(parents=True, exist_ok=True)
            doc = {"request_id": rid, "step": step, "phase": phase, "positions": positions, "leaves": leaves, "speculative": spec_rec}
            if recompute:
                doc["recompute"] = True
            (d / "leaves.json").write_text(json.dumps(doc))
        self.ru_count += 1

    def summary(self) -> dict[str, Any]:
        s = super().summary()
        hist: dict[str, int] = {}
        for recs in self.acceptance.values():
            for r in recs:
                if r["k"] and not r.get("discarded"):
                    hist[str(r["accepted"])] = hist.get(str(r["accepted"]), 0) + 1
        s["speculative"] = {
            "method": "draft_model", "num_speculative_tokens": self.k, "verify_steps": self.spec_steps,
            "acceptance_histogram": hist, "draft_leaves": self.draft_leaves, "dead_leaves": self.dead_leaves,
            "rejected_target_slot_leaves": self.rejected_slot_leaves, "padding_rows_skipped": self.padding_rows_skipped,
            "problems": self.spec_problems[:50], "problem_count": len(self.spec_problems),
            "discarded_steps": self.discarded_steps, "async_scheduling_note": "a request's final step under vLLM's async scheduling: computed, tokens discarded by the engine, no replay unit",
        }
        return s


# ----------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------
def load_cp(checkpoints: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cps = prof.load_checkpoints(checkpoints)
    tgt, drf = cps[TARGET_REPO], cps[DRAFT_REPO]
    for cp in (tgt, drf):
        if not cp.get("local_path") or not Path(cp["local_path"]).exists():
            raise SystemExit(f"{cp['repo']}: local snapshot missing")
    return tgt, drf


def engine_kwargs(tgt: dict[str, Any], drf: dict[str, Any], k: int) -> dict[str, Any]:
    return {
        "model": tgt["local_path"], "tokenizer": tgt["local_path"], "dtype": "bfloat16", "seed": 0,
        "max_model_len": MAX_MODEL_LEN, "gpu_memory_utilization": GPU_MEMORY_UTILIZATION, "trust_remote_code": False,
        "enforce_eager": True, "enable_prefix_caching": False,
        "speculative_config": {"method": "draft_model", "model": drf["local_path"], "num_speculative_tokens": int(k)},
        "per_request_spec_decode_metrics": "detailed",
    }


class Req:
    def __init__(self, request_id: str, prompt_token_ids: list[int], max_tokens: int) -> None:
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.max_tokens = max_tokens


def build_workload(llm, subset: list[str] | None) -> list[Req]:
    tok = llm.get_tokenizer()
    reqs = []
    for rid, text, mt in SD_PROMPTS:
        if subset and rid not in subset:
            continue
        reqs.append(Req(rid, list(tok.encode(text)), mt))
    if subset:
        order = {r: i for i, r in enumerate(subset)}
        reqs.sort(key=lambda r: order[r.request_id])
    return reqs


def sd_params(r: Req):
    from vllm import SamplingParams

    return SamplingParams(temperature=0.0, max_tokens=r.max_tokens, logprobs=1, seed=0)


def outputs_rows(reqs: list[Req], outs: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for r in reqs:
        o = outs[r.request_id]
        c = o.outputs[0]
        lps = []
        if c.logprobs:
            for tok, lp in zip(c.token_ids, c.logprobs, strict=False):
                e = lp.get(tok)
                lps.append(e.logprob if e is not None else None)
        # vLLM keeps the per-sequence acceptance record on the CompletionOutput (outputs.py)
        sdm = getattr(c, "spec_decode_metrics", None) or getattr(o, "spec_decode_metrics", None)
        rows.append({
            "request_id": r.request_id, "prompt_token_ids": list(r.prompt_token_ids), "output_token_ids": list(c.token_ids),
            "finish_reason": c.finish_reason, "chosen_logprobs": lps,
            "sampling_params": {"temperature": 0.0, "max_tokens": r.max_tokens, "logprobs": 1, "seed": 0, "n": 1, "top_p": 1.0, "top_k": 0},
            "seed": 0, "num_cached_tokens": getattr(o, "num_cached_tokens", None),
            "spec_decode_metrics": sdm.to_dict() if sdm is not None and hasattr(sdm, "to_dict") else (None if sdm is None else str(sdm)),
        })
    return rows


def manifest_doc(tgt: dict[str, Any], drf: dict[str, Any], llm, k: int, adapter_revision: str,
                 wheel_sha256: str | None = None, vllm_commit: str | None = None) -> dict[str, Any]:
    sw = prof.software_info(wheel_sha256, vllm_commit)
    kw = {kk: v for kk, v in engine_kwargs(tgt, drf, k).items() if kk not in ("model", "tokenizer")}
    kw["speculative_config"] = {"method": "draft_model", "model": DRAFT_REPO, "num_speculative_tokens": int(k)}
    return {
        "schema": "verity-vllm/profile/v1", "case": "B1", "port": PORT,  # case = the checkpoint role (the port's checkpoint.role)
        "model": {"repo": tgt["repo"], "revision": tgt["revision"], "weights_sha256": dict(tgt["safetensors_sha256"]), "config_sha256": tgt.get("config_sha256")},
        "draft_model": {"repo": drf["repo"], "revision": drf["revision"], "weights_sha256": dict(drf["safetensors_sha256"]), "config_sha256": drf.get("config_sha256")},
        "vllm": sw["vllm"], "python": sw["python"], "dists": sw["dists"], "gpu": prof.gpu_info(), "dtype": "bfloat16", "quantization": None,
        "backends": prof.backends_info(llm), "parallelism": {"tensor_parallel_size": 1, "pipeline_parallel_size": 1},
        "compilation": prof.effective_compilation(llm), "cache": prof.cache_info(llm),
        "engine_flags": dict(SD_ENV), "engine_kwargs": kw,
        "sampling": {"temperature": 0.0, "logprobs": 1, "seed": 0, "target": "greedy", "draft": "greedy argmax"},
        "decoding": {"mode": "speculative", "method": "draft_model", "num_speculative_tokens": int(k), "model_runner": "v1"},
        "adapter": {"revision": adapter_revision, "interception": "none (stock production) / TorchDispatchMode over both models (replay)"},
    }


def cmd_production(args: argparse.Namespace) -> None:
    from vllm import LLM

    from verity_vllm_adapter.steptrace import StepTrace

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tgt, drf = load_cp(Path(args.checkpoints))
    kw = engine_kwargs(tgt, drf, args.k)
    t0 = time.perf_counter()
    llm = LLM(**kw)
    build_s = time.perf_counter() - t0
    subset = args.requests.split(",") if args.requests else None
    reqs = build_workload(llm, subset)
    trace = StepTrace()
    trace.install(prof.get_scheduler(llm))
    outs, timing = run_requests(llm, reqs, sd_params)
    trace.uninstall()
    rows = outputs_rows(reqs, outs)
    man = manifest_doc(tgt, drf, llm, args.k, args.adapter_revision, args.wheel_sha256, args.vllm_commit)
    (out / "manifest.json").write_bytes(prof.canonical_json_bytes(man))
    with (out / "production.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "steptrace.json").write_text(json.dumps({"summary": trace.summary(), "steps": trace.steps}, indent=1, default=str) + "\n")
    doc = {"port": PORT, "mode": "production", "profile_id": prof.profile_id_of(man), "requests": len(reqs), "engine_kwargs": kw,
           "env": {kk: os.environ.get(kk) for kk in SD_ENV}, "timing": {"engine_build_s": build_s, **timing},
           "generated_tokens": sum(len(r["output_token_ids"]) for r in rows),
           "acceptance": {r["request_id"]: r["spec_decode_metrics"] for r in rows}, "utc": prof.utc_stamp(), "argv": sys.argv}
    (out / "production.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"DONE production requests={len(reqs)} tokens={doc['generated_tokens']} profile={doc['profile_id']}", flush=True)


def load_production(d: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    man = json.loads((d / "manifest.json").read_text())
    rows = [json.loads(l) for l in (d / "production.jsonl").read_text().splitlines() if l.strip()]
    return man, rows


def cmd_replay(args: argparse.Namespace) -> None:
    from vllm import LLM

    from verity_vllm import commit as vcommit
    from verity_vllm import required as vrequired
    from verity_vllm.description import Description
    from verity_vllm_adapter.steptrace import StepTrace

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_f = (out / "replay.log").open("a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    tgt, drf = load_cp(Path(args.checkpoints))
    prod_man, prod_rows = load_production(Path(args.production))
    profile_id = prof.profile_id_of(prod_man)
    k = int(prod_man["engine_kwargs"]["speculative_config"]["num_speculative_tokens"])
    kw = engine_kwargs(tgt, drf, k)
    patches = install_kernel_wrappers()
    log(f"kernel wrappers: {patches}")
    t0 = time.perf_counter()
    llm = LLM(**kw)
    build_s = time.perf_counter() - t0
    runner = prof.get_model_runner(llm)
    model = runner.model
    drafter = runner.drafter
    log(f"runner={type(runner).__name__} drafter={type(drafter).__name__} draft_model={type(drafter.model).__name__}")
    subset = [r["request_id"] for r in prod_rows]
    reqs = build_workload(llm, subset)
    by_id = {r["request_id"]: r for r in prod_rows}
    for r in reqs:
        pr = by_id[r.request_id]
        if list(pr["prompt_token_ids"]) != r.prompt_token_ids:
            raise SystemExit(f"{r.request_id}: prompt token ids differ from production")
        r.max_tokens = int(pr["sampling_params"]["max_tokens"])
    prompt_lens = {r.request_id: len(r.prompt_token_ids) for r in reqs}
    tokens = {r.request_id: list(by_id[r.request_id]["prompt_token_ids"]) + list(by_id[r.request_id]["output_token_ids"]) for r in reqs}
    store_values = bool(args.store_values)
    writer = RunWriter(out, prod_man) if store_values else None
    if writer is not None and writer.profile_id != profile_id:
        raise SystemExit(f"profile_id mismatch: writer {writer.profile_id} vs production {profile_id}")
    cap = SpecCapture(
        drafter=drafter, num_speculative_tokens=k, runner=runner, model=model, profile_id=profile_id, run_dir=out, writer=writer,
        hash_only=not store_values, prompt_lens=prompt_lens, arena_bytes=int(args.arena_gb * (1 << 30)), log=log, tokens=tokens,
    )
    cap.no_record = bool(args.no_captures)
    trace = StepTrace()
    trace.install(prof.get_scheduler(llm))
    torch.cuda.synchronize()
    t_run0 = time.perf_counter()
    outs, timing = run_requests(llm, reqs, sd_params)
    torch.cuda.synchronize()
    run_s = time.perf_counter() - t_run0
    trace.uninstall()
    rows = outputs_rows(reqs, outs)
    log(f"run {run_s:.1f}s; capture: {cap.leaf_count} leaves, {cap.ru_count} RUs, errors={len(cap.errors)} spec_problems={len(cap.spec_problems)}")

    # --- trajectory + acceptance record vs production ---
    per_request = []
    followed = 0
    acc_equal = 0
    for r in reqs:
        pr = by_id[r.request_id]
        rr = next(x for x in rows if x["request_id"] == r.request_id)
        ok = rr["output_token_ids"] == pr["output_token_ids"]
        followed += ok
        pa, ra = pr.get("spec_decode_metrics") or {}, rr.get("spec_decode_metrics") or {}
        keys = ("per_step_accepted", "per_step_drafted", "acceptance_histogram", "num_draft_tokens")
        a_eq = all(pa.get(kk) == ra.get(kk) for kk in keys) if isinstance(pa, dict) and isinstance(ra, dict) else None
        acc_equal += bool(a_eq)
        cap_rec = [(x["k"], x["accepted"]) for x in cap.acceptance.get(r.request_id, []) if x["k"] and not x.get("discarded")]
        eng_rec = list(zip(ra.get("per_step_drafted") or [], ra.get("per_step_accepted") or [], strict=False)) if isinstance(ra, dict) else []
        per_request.append({
            "request_id": r.request_id, "trajectory_followed": ok, "emitted_len": len(rr["output_token_ids"]), "output_len": len(pr["output_token_ids"]),
            "finish_reason": rr["finish_reason"], "production_finish_reason": pr["finish_reason"],
            "logprobs_equal": rr["chosen_logprobs"] == pr["chosen_logprobs"],
            "acceptance_record_equal": a_eq, "production_acceptance": pa, "replay_acceptance": ra,
            "capture_acceptance": cap.acceptance.get(r.request_id, []),
            "capture_vs_engine_record_equal": cap_rec == [(int(d), int(a)) for d, a in eng_rec],
        })
    with (out / "replay_outputs.jsonl").open("w") as f:
        for rr in rows:
            f.write(json.dumps(rr) + "\n")

    boundary_problems: list[str] = []
    requests_info = {r.request_id: {"prompt_token_ids": by_id[r.request_id]["prompt_token_ids"], "output_token_ids": by_id[r.request_id]["output_token_ids"],
                                    "sampling_params": by_id[r.request_id]["sampling_params"], "seed": 0,
                                    "finish_reason": by_id[r.request_id]["finish_reason"], "chosen_logprobs": by_id[r.request_id]["chosen_logprobs"]} for r in reqs}
    cfg = getattr(model, "config", None)
    symbols = {"V": int(getattr(cfg, "vocab_size", 0) or 0), "H": int(getattr(cfg, "hidden_size", 0) or 0)}
    description = cap.description(f"{tgt['repo']} (target) + {drf['repo']} (draft)", symbols)
    description["speculative"] = {"method": "draft_model", "num_speculative_tokens": k, "draft_repo": drf["repo"], "draft_prefix": DRAFT_PREFIX,
                                  "composites": ["verity_vllm.registry.spec.Accept", "verity_vllm.registry.spec.Advance"]}
    weights_info: dict[str, Any] = {}
    for b in cap.boundaries(requests_info):
        probs = b.validate()
        if probs:
            boundary_problems.append(f"{b.request_id}: {probs}")
        elif writer is not None:
            writer.write_boundary(b)
    if writer is not None:
        t_w0 = time.perf_counter()
        ww = writer.weights(store_values=(args.weights_values == "store"), source_sha256=None)
        n_params = 0
        for ptr, (name, p) in sorted(cap.params.items(), key=lambda kv: kv[1][0]):
            data = p.detach().contiguous().cpu().reshape(-1)
            ww.add(name, str(p.dtype).replace("torch.", ""), list(p.shape), data.view(torch.uint8).numpy().tobytes())
            n_params += 1
        ww.finish()
        weights_info = {"parameters": n_params, "seconds": time.perf_counter() - t_w0, "mode": args.weights_values}
        writer.write_description(description)
        writer.finish()
        try:
            desc_obj = Description(json.loads((out / "description.json").read_text()))
            required = vrequired.freeze_required(out / "required.json", desc_obj, {"partition_id": "row-default", "decomposition": {"default": "row"}})
            log(f"description: {len(desc_obj.nodes)} nodes, required interfaces: {len(required['interfaces'])}")
        except Exception as e:  # noqa: BLE001
            weights_info["required_json_error"] = repr(e)
            log(f"required.json not frozen: {e!r}")
    else:
        (out / "description.json").write_text(json.dumps(description, indent=1, sort_keys=True))
        (out / "manifest.json").write_bytes(prof.canonical_json_bytes(prod_man))
    summary = cap.summary()
    (out / "capture_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    (out / "kv_audit.json").write_text(json.dumps(cap.kv_audit, indent=1, default=str) + "\n")
    (out / "acceptance.json").write_text(json.dumps(cap.acceptance, indent=1) + "\n")
    (out / "steptrace.json").write_text(json.dumps({"summary": trace.summary(), "steps": trace.steps}, indent=1, default=str) + "\n")
    doc = {
        "port": PORT, "mode": "replay" + ("-commit" if store_values else ""), "profile_id": profile_id, "argv": sys.argv, "engine_kwargs": kw,
        "env": {kk: os.environ.get(kk) for kk in SD_ENV}, "patches": patches, "requests": len(reqs),
        "trajectory_followed": {"requests": len(reqs), "followed": followed},
        "acceptance_record_equal": {"requests": len(reqs), "equal": acc_equal},
        "capture_vs_engine_record_equal": sum(1 for p in per_request if p["capture_vs_engine_record_equal"]),
        "logprobs_vs_production": {"requests_compared": len(reqs), "equal": sum(1 for p in per_request if p["logprobs_equal"])},
        "per_request": per_request,
        "timing": {"engine_build_s": build_s, "run_s": run_s, "driver": timing, "capture": summary["timing"], "weights": weights_info},
        "capture": {kk: summary.get(kk) for kk in ("steps", "replay_units", "leaves", "bytes_captured", "templates", "unbound_inputs",
                                                   "module_outputs_not_recorded", "kv_reads_unresolved", "kv_reads_checked", "arena_overflow_allocs")},
        "speculative": summary["speculative"], "boundary_problems": boundary_problems, "errors": cap.errors[:20], "utc": prof.utc_stamp(),
    }
    (out / "replay.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    if boundary_problems:
        log(f"BOUNDARY_PROBLEMS {len(boundary_problems)}: {boundary_problems[:3]}")
        raise SystemExit(3)
    if writer is not None:
        cap.arena.buf = None
        t_c0 = time.perf_counter()
        commitments = vcommit.build_commitments(out, recompute_leaves=True, write=True)
        doc["commitments"] = {"seconds": time.perf_counter() - t_c0, "trees": len(commitments.get("trees", {})), "sha256": sha256_file(out / "commitments.json")}
        (out / "replay.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
        log(f"commitments: {doc['commitments']}")
    log(f"DONE replay followed={followed}/{len(reqs)} acceptance_equal={acc_equal}/{len(reqs)} leaves={summary['leaves']} "
        f"dead={summary['speculative']['dead_leaves']} unbound={summary['unbound_inputs']} spec_problems={summary['speculative']['problem_count']}")


def cmd_compare(args: argparse.Namespace) -> None:
    """C0: two productions (or a production and a replay's outputs) — token tapes, logprobs, acceptance records."""
    a_rows = {r["request_id"]: r for r in load_production(Path(args.a))[1]}
    bp = Path(args.b)
    b_file = bp / ("production.jsonl" if (bp / "production.jsonl").exists() else "replay_outputs.jsonl")
    b_rows = {r["request_id"]: r for r in (json.loads(l) for l in b_file.read_text().splitlines() if l.strip())}
    res = []
    for rid, ra in a_rows.items():
        rb = b_rows.get(rid)
        if rb is None:
            res.append({"request_id": rid, "missing_in_b": True})
            continue
        ka = ("per_step_accepted", "per_step_drafted", "acceptance_histogram", "num_draft_tokens")
        sa, sb = ra.get("spec_decode_metrics") or {}, rb.get("spec_decode_metrics") or {}
        res.append({"request_id": rid, "tokens_equal": ra["output_token_ids"] == rb["output_token_ids"],
                    "logprobs_equal": ra["chosen_logprobs"] == rb["chosen_logprobs"],
                    "acceptance_equal": all(sa.get(kk) == sb.get(kk) for kk in ka), "output_len": len(ra["output_token_ids"]),
                    "per_step_accepted": sa.get("per_step_accepted"), "per_step_drafted": sa.get("per_step_drafted")})
    doc = {"a": str(args.a), "b": str(args.b), "requests": len(res), "tokens_equal": sum(1 for r in res if r.get("tokens_equal")),
           "logprobs_equal": sum(1 for r in res if r.get("logprobs_equal")), "acceptance_equal": sum(1 for r in res if r.get("acceptance_equal")),
           "per_request": res}
    txt = json.dumps(doc, indent=1)
    if args.out:
        Path(args.out).write_text(txt + "\n")
    print(txt)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("production")
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoints", default="manifests/checkpoints.json")
    p.add_argument("--requests", default=None, help="comma-separated subset of the SD workload, in submission order")
    p.add_argument("--k", type=int, default=NUM_SPECULATIVE_TOKENS)
    p.add_argument("--adapter-revision", default="unknown")
    p.add_argument("--wheel-sha256", default=None, help="sha256 of the installed vLLM wheel (manifest.vllm.wheel_sha256)")
    p.add_argument("--vllm-commit", default=None, help="the pinned vLLM commit (manifest.vllm.commit)")
    p.set_defaults(fn=cmd_production)
    p = sub.add_parser("replay")
    p.add_argument("--production", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoints", default="manifests/checkpoints.json")
    p.add_argument("--store-values", action="store_true")
    p.add_argument("--no-captures", action="store_true")
    p.add_argument("--weights-values", default="hash", choices=("hash", "store"))
    p.add_argument("--arena-gb", type=float, default=8.0)
    p.set_defaults(fn=cmd_replay)
    p = sub.add_parser("compare")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("compare-refs", help="commit store vs hash-mode reference replays, leaf hash by leaf hash (the port's compare.json)")
    p.add_argument("--commit", required=True)
    p.add_argument("--refs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_compare_refs)
    args = ap.parse_args()
    args.fn(args)


def _store_leaf_hashes(root: Path) -> dict[str, str]:
    """canonical id -> leaf hash for every RU of a store (commit: ru/*/*/index.json entries; hash: leaves.json)."""
    out: dict[str, str] = {}
    for req in sorted(p for p in (root / "ru").iterdir() if p.is_dir()):
        for d in sorted(p for p in req.iterdir() if p.is_dir()):
            if (d / "index.json").exists():
                for e in json.loads((d / "index.json").read_text())["entries"]:
                    out[e["canonical_id"]] = e["leaf"]
            elif (d / "leaves.json").exists():
                for cid, e in json.loads((d / "leaves.json").read_text())["leaves"].items():
                    out[cid] = e["leaf"]
    return out


def _pair(a_name: str, a: dict[str, str], b_name: str, b: dict[str, str]) -> dict[str, Any]:
    shared = set(a) & set(b)
    mism = sorted(c for c in shared if a[c] != b[c])
    return {"a": a_name, "b": b_name, "leaves_a": len(a), "leaves_b": len(b), "compared_scalars": len(shared),
            "mismatched_scalars": len(mism), "only_in_a": len(set(a) - shared), "only_in_b": len(set(b) - shared),
            "identical": len(a) == len(b) == len(shared) and not mism, "first_mismatches": mism[:8]}


def cmd_compare_refs(args) -> None:
    """The commit-mode capture (values stored) against fresh-process hash-mode replays of the same production
    (references): the same leaf hashes at the same canonical ids shows the commit run did not perturb the values."""
    commit = _store_leaf_hashes(Path(args.commit))
    refs = [(r, _store_leaf_hashes(Path(r))) for r in args.refs]
    pairs = [_pair(args.commit, commit, r, h) for r, h in refs]
    rvr = [_pair(refs[i][0], refs[i][1], refs[j][0], refs[j][1]) for i in range(len(refs)) for j in range(i + 1, len(refs))]
    doc = {"schema": "verity-vllm/sd-compare-refs/v0", "commit": args.commit, "references": [r for r, _ in refs],
           "method": "leaf hash (sha256 of the canonical value's bytes) at every canonical id of every replay unit; "
                     "a reference is a hash-mode replay of the same production in a fresh process",
           "identical": all(p["identical"] for p in pairs), "compared_scalars": sum(p["compared_scalars"] for p in pairs),
           "mismatched_scalars": sum(p["mismatched_scalars"] for p in pairs), "pairs": pairs,
           "reference_vs_reference": {"identical": all(p["identical"] for p in rvr), "pairs": rvr}}
    Path(args.out).write_text(json.dumps(doc, indent=1))
    print(f"[sd] compare-refs identical={doc['identical']} compared={doc['compared_scalars']} mismatched={doc['mismatched_scalars']} "
          f"ref_vs_ref={doc['reference_vs_reference']['identical']} -> {args.out}")


if __name__ == "__main__":
    main()
