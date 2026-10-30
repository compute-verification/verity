"""S3 acquisition of R(P_compiled, Q) under torch.compile + PIECEWISE CUDA graphs.

The Program of an `Execution=compiled` row is P_compiled at Inductor-kernel granularity (compiled_relations.py); its VU boundaries are
the buffers the production implementation already materializes (every kernel argument).  This source acquires exactly those buffers
without touching the kernels: Inductor's wrapper-level debug-printer seam (`aot_inductor.debug_intermediate_value_printer = SAVE_ONLY`)
is routed to `torch._inductor.hooks.run_intermediate_hooks`, so the generated HOST wrapper hands every buffer argument to `_hook`
before ("b") and after ("a") each launch.  Kernel sources, fusion and autotuning are unchanged (kernel_set_sha256 == production;
compiled_merge compares the two processes).  Post-optimization instrumentation: the copies are added after Inductor fixed the kernels.

Two execution paths, both graph-safe:
  * captured pieces (CUDAGraphMode PIECEWISE, batch size in the capture sizes): the hook lines execute ONCE at graph capture; there
    they issue a device `copy_` of the buffer into a preallocated slot keyed by (piece, launch, argument, shape).  The graph replays the
    copies every step; nothing Python runs during replay.  At the committer's step end the slots of the replayed batch size are staged
    under their P_compiled names (`c<piece>.<kernel>/<role>`) -> gathered into the step's leaves -> root (C3: before any opening).
  * uncaptured steps (the prefill, larger than the largest capture size): the wrapper runs eagerly; inputs are staged before the launch
    and outputs after it, each gathered at once (Inductor reuses dead buffers within a piece).
Registered weights (WEIGHT_ARGS) are never committed (they are the model's registered inputs).

Faults (negatives): `compiled_omit:<kernel substring>` — that kernel's buffers are not acquired (coverage FAIL, C1);
`compiled_late_drain` — a replayed step is drained from the PREVIOUS step's slot contents (post-overwrite read: the linkage check
ids(step s) == token(s-1) FAILS, C2).
"""
from __future__ import annotations

import time
from typing import Any

import torch

from verity_vllm.engine import hooks
from verity_vllm.program.compiled_relations import OUTPUT_ROLES, roles_of

_INDUCTOR_DEBUG_LEVEL = "1"                                                     # torch._inductor IntermediateValueDebuggingLevel.SAVE_ONLY


def install_wrapper_hooks(hook, into: hooks.Hooks) -> dict[str, Any]:
    """Patch Inductor so the generated wrappers call `hook(name, tensor)` around every kernel launch.  Must run BEFORE the engine compiles."""
    from torch._inductor import config as icfg
    from torch._inductor import hooks as ih
    from torch._inductor.codegen import debug_utils as du

    into.patch(icfg.aot_inductor, "debug_intermediate_value_printer", _INDUCTOR_DEBUG_LEVEL)
    into.patch(du.DebugPrinterManager, "codegen_intermediate_tensor_value_save", _patched_save)
    if hook not in ih.INTERMEDIATE_HOOKS:
        into.append(ih.INTERMEDIATE_HOOKS, hook)
    return {"debug_printer_level": _INDUCTOR_DEBUG_LEVEL, "intermediate_hooks": len(ih.INTERMEDIATE_HOOKS)}


def _patched_save(self, args_to_save, kernel_name, before_launch=True, arg_signatures=None):   # DebugPrinterManager (torch 2.13)
    from torch._inductor.virtualized import V

    tag = "b" if before_launch else "a"
    for i, arg in enumerate(args_to_save):
        if not isinstance(arg, str):
            continue
        if arg_signatures is not None and i < len(arg_signatures) and not isinstance(arg_signatures[i], torch.dtype):
            continue
        V.graph.wrapper_code.writeline(f"run_intermediate_hooks({kernel_name + '|' + str(i) + '|' + arg + '|' + tag!r}, {arg})")


class CompiledKernelSource:
    short_name = "compiled_kernels"
    step_rows_at_step_end = True                    # `on_step_end` reads the committer's `step_rows[step]`

    def __init__(self, fault: str | None = None) -> None:
        self.fault = fault
        self.omit = fault.split(":", 1)[1] if fault and fault.startswith("compiled_omit:") else None
        self.c: Any = None
        self.slots: dict[tuple, torch.Tensor] = {}
        self.slot_meta: dict[tuple, tuple[int, str, str]] = {}                 # key -> (launch, name, role)
        self.piece = -1
        self.cap_launch = -1
        self.cap_last_b = -1
        self.in_cgw = False
        self.capture_sizes: set[int] = set()
        self.eager_launch: dict[int, int] = {}
        self.eager_last_b = -1
        self.eager_this_step = False
        self.step_ntok = 0
        self.stale: dict[tuple, torch.Tensor] = {}
        self.errors: list[str] = []
        self.hooks_info: dict[str, Any] = {}
        self._hooks = hooks.Hooks()
        self.stats: dict[str, Any] = {"capture_copies": 0, "capture_skipped_weights": 0, "eager_acquisitions": 0, "replay_drains": 0,
                                      "steps": 0, "eager_steps": 0, "replayed_steps": 0, "hook_calls_outside_step": 0, "omitted": 0,
                                      "slot_bytes": 0, "unknown_kernel_args": 0, "role_map_misses": 0, "drain_s": 0.0}
        self.role_maps: dict[str, dict[str, tuple[dict[int, str], set[int]]]] | None = None   # (piece kind -> kernel -> (roles, weight args)) from the manifest
        self.pieces_n = 0

    # ------------------------------------------------------------------ argument roles: the manifest's role maps FIRST (they are what the identities were derived from)

    def set_role_maps(self, manifest: dict[str, Any] | None) -> None:
        """Take the argument roles from the manifest's program.role_maps (resolved per launch site from the kernel's
        SIGNATURE, e.g. Qwen's 6-pointer fused RoPE+QKV-bias kernel vs Llama's 5-pointer RoPE): the hook binds exactly the roles the
        identities were derived from.  Without them the name-keyed table is ambiguous for such kernels and every argument is 'unknown'
        (on Qwen B8, 67,760 of 353,320 identities unbound, all of them the RoPE kernel's).  Must run BEFORE the engine compiles so
        capture-time copies use the same roles."""
        prog = (manifest or {}).get("program") if isinstance(manifest, dict) else None
        rm = (prog or {}).get("role_maps")
        if not rm:
            self.role_maps = None
            return
        self.pieces_n = int((prog or {}).get("pieces") or 0)
        self.role_maps = {kind: {str(e.get("kernel")): ({int(k): str(v) for k, v in (e.get("roles") or {}).items()}, {int(i) for i in (e.get("weight_args") or [])})
                                 for e in lst if isinstance(e, dict)} for kind, lst in rm.items() if isinstance(lst, list)}
        self.hooks_info["role_maps"] = {k: len(v) for k, v in self.role_maps.items()}

    def _roles_for(self, piece: int, kernel: str) -> tuple[dict[int, str], set[int]]:
        if self.role_maps:
            kind = "first" if piece == 0 else ("last" if self.pieces_n and piece == self.pieces_n - 1 else "middle")
            e = self.role_maps.get(kind, {}).get(kernel)
            if e is not None:
                return e
            self.stats["role_map_misses"] += 1
        return roles_of(kernel)

    # ------------------------------------------------------------------ installation (before the engine compiles)

    def install_hooks(self) -> None:
        self.hooks_info = install_wrapper_hooks(self._hook, self._hooks)
        from vllm.compilation import piecewise_backend as pb

        src = self
        orig_call = pb.PiecewiseBackend.__call__

        def call(self_pb, *args):
            if src.in_cgw:
                return orig_call(self_pb, *args)
            src._enter_piece(self_pb.piecewise_compile_index)
            return orig_call(self_pb, *args)

        self._hooks.patch(pb.PiecewiseBackend, "__call__", call)
        try:
            from vllm.compilation import cuda_graph as cg

            orig_cgw = cg.CUDAGraphWrapper.__call__

            def cgw_call(self_w, *args, **kwargs):
                pbk = getattr(self_w, "runnable", None)
                idx = getattr(pbk, "piecewise_compile_index", None)
                if idx is None:
                    return orig_cgw(self_w, *args, **kwargs)
                src._enter_piece(idx)
                src.in_cgw = True
                try:
                    return orig_cgw(self_w, *args, **kwargs)
                finally:
                    src.in_cgw = False

            self._hooks.patch(cg.CUDAGraphWrapper, "__call__", cgw_call)
            self.hooks_info["graph_hook"] = "CUDAGraphWrapper.__call__"
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"CUDAGraphWrapper patch: {e!r}")

    def _enter_piece(self, idx: int) -> None:
        self.piece = idx
        self.cap_launch = -1
        self.cap_last_b = -1
        self.eager_launch.setdefault(idx, -1)
        self.eager_last_b = -1

    def note_engine(self, llm: Any) -> None:
        try:
            cc = llm.llm_engine.vllm_config.compilation_config
            self.capture_sizes = {int(x) for x in (getattr(cc, "cudagraph_capture_sizes", None) or [])}
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"capture sizes: {e!r}")

    # ------------------------------------------------------------------ the hook (wrapper host code): capture -> slot copy; eager step -> acquire now

    @staticmethod
    def _name(piece: int, kernel: str, role: str) -> str:
        return f"c{piece:02d}.{kernel}/{role}"

    def _hook(self, name: str, val: Any) -> None:
        if not isinstance(val, torch.Tensor):
            return
        kernel, idx_s, _arg, tag = name.rsplit("|", 3)
        idx = int(idx_s)
        roles, weights = self._roles_for(self.piece, kernel)
        if torch.cuda.is_current_stream_capturing():
            if tag == "b":
                if self.cap_last_b == -1 or idx <= self.cap_last_b:
                    self.cap_launch += 1
                self.cap_last_b = idx
            else:
                self.cap_last_b = -1
            if idx in weights:
                self.stats["capture_skipped_weights"] += 1
                return
            role = roles.get(idx)
            if role is None:
                self.stats["unknown_kernel_args"] += 1
                return
            if tag == "b" and role in OUTPUT_ROLES:
                return                                                         # the output's bytes exist only after the launch
            if tag == "a" and role not in OUTPUT_ROLES:
                return                                                         # inputs were copied before the launch
            key = (self.piece, self.cap_launch, kernel, role, tuple(val.shape))
            slot = self.slots.get(key)
            if slot is None:
                slot = torch.empty_like(val)
                self.slots[key] = slot
                self.slot_meta[key] = (self.cap_launch, self._name(self.piece, kernel, role), role)
                self.stats["slot_bytes"] += slot.numel() * slot.element_size()
            slot.copy_(val)
            self.stats["capture_copies"] += 1
            return
        c = self.c
        job = getattr(c, "_cur", None) if c is not None else None
        if job is None:
            self.stats["hook_calls_outside_step"] += 1                        # engine warm-up / profiling / graph warm-up runs
            return
        if idx in weights:
            return
        role = roles.get(idx)
        if role is None:
            self.stats["unknown_kernel_args"] += 1
            return
        if (tag == "b") == (role in OUTPUT_ROLES):
            return
        self.eager_this_step = True
        if self.omit and self.omit in kernel:
            self.stats["omitted"] += 1
            return
        c._acquire(self._name(self.piece, kernel, role), val, job)
        self.stats["eager_acquisitions"] += 1
        if getattr(c, "gather", False):
            c._nc.flush()                                                      # the buffer may be reused by the next kernel of the piece

    # ------------------------------------------------------------------ committer lifecycle

    def attach(self, committer: Any, manifest: dict[str, Any] | None = None) -> None:
        self.c = committer
        committer.attach_source(self)
        setattr(committer, "external_acquisition", True)
        if manifest is not None:
            self.expected = sorted({r["op_path"] for r in manifest["identities"]})     # every launch of P_compiled, every step (identity check)
            if self.role_maps is None:
                self.set_role_maps(manifest)                                          # late (post-capture) attach: still better than the name-keyed table

    def expected_modules(self) -> list[str]:
        """The per-step producer set the committer's identity check expects from this source: P_compiled's kernel launches (from the
        required manifest); empty until attached with a manifest (the check then skips)."""
        return list(getattr(self, "expected", []) or [])

    def begin_run(self) -> None:
        """Per-run counters (the slots and hooks persist across runs of the process; the stats reported per pair are this run's)."""
        for k in ("steps", "eager_steps", "replayed_steps", "hook_calls_outside_step", "omitted", "replay_drains", "eager_acquisitions"):
            if k in self.stats:
                self.stats[k] = 0

    def install(self) -> None:
        pass

    def uninstall(self) -> None:
        pass

    def remove_hooks(self) -> None:
        """Undo `install_hooks`.  Code Inductor generated while they were installed keeps its hook lines."""
        self._hooks.uninstall()

    def declaration_classes(self) -> list[dict]:
        # the committer's coverage_declaration tallies this source's tensors itself (class `compiled_kernel_buffers`, bound-as-leaf; the
        # embedding kernel's ids argument = `input_token_ids`): nothing to add here
        return []

    def on_step_begin(self) -> None:
        self.eager_this_step = False
        for k in self.eager_launch:
            self.eager_launch[k] = -1
        c = self.c
        self.step_ntok = int(getattr(c, "_plan_key", 0) >> 16) if c is not None else 0

    def on_step_end(self, job: Any) -> None:
        c = self.c
        if c is None:
            return
        t0 = time.perf_counter()
        self.stats["steps"] += 1
        if self.eager_this_step:
            self.stats["eager_steps"] += 1
            self.stats["drain_s"] += time.perf_counter() - t0
            return
        # the step's ACTUAL token rows (the engine's row order recorded at step begin, B>=2) and the CUDA graph that ran them: vLLM pads a
        # decode batch to the smallest captured size >= num_tokens, so the replayed slots carry `n_pad` rows of which the first `n` are the
        # step's tokens (the rest = padding, never a Program value) -> acquire slot[:n]
        rows = (getattr(c, "step_rows", None) or {}).get(int(job.step))
        n = max((hi for _r, _lo, hi in rows), default=0) if rows else self.step_ntok
        fit = [s for s in self.capture_sizes if s >= n]
        if n <= 0 or not fit:
            self.stats.setdefault("uncaptured_without_hooks", 0)
            self.stats["uncaptured_without_hooks"] += 1
            return
        n_pad = min(fit)
        if n_pad != n:
            self.stats["padded_steps"] = self.stats.get("padded_steps", 0) + 1
        self.stats["replayed_steps"] += 1
        late = self.fault == "compiled_late_drain"
        for key in sorted(self.slots, key=lambda k: (k[0], k[1], k[2], k[3])):
            shape = key[4]
            if not shape or shape[0] != n_pad:
                continue
            _launch, name, _role = self.slot_meta[key]
            if self.omit and self.omit in key[2]:
                self.stats["omitted"] += 1
                continue
            slot = self.slots[key]
            src = slot
            if late:
                prev = self.stale.get(key)
                src = prev if prev is not None else slot
            if n_pad != n:
                src = src[:n]
            c._acquire(name, src, job)
            self.stats["replay_drains"] += 1
            if late:
                self.stale[key] = slot.clone()
        if getattr(c, "gather", False):
            c._nc.flush()
        self.stats["drain_s"] += time.perf_counter() - t0

    # ------------------------------------------------------------------ evidence

    def summary(self) -> dict[str, Any]:
        return {"short_name": self.short_name, "fault": self.fault, "slots": len(self.slots), "capture_sizes": sorted(self.capture_sizes),
                "hooks": self.hooks_info, "stats": dict(self.stats), "errors": list(self.errors)[:20]}

    def stats_reset_step_counters(self) -> None:
        for k in ("eager_acquisitions", "replay_drains", "steps", "eager_steps", "replayed_steps", "omitted"):
            self.stats[k] = 0
        self.stats["drain_s"] = 0.0
