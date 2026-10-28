"""Graph-replay-safe acquisition for `Execution=compiled` workloads (vLLM torch.compile + CUDA graphs, enforce_eager=False).

Why the eager collector cannot see anything here: vLLM compiles `LlamaModel.forward` with Dynamo `fullgraph=True`
(`support_torch_compile`) and replays the Inductor-compiled pieces as CUDA graphs.  The committer's module-forward wrappers are installed AFTER
the model was compiled, so Dynamo's guards were already skipped (`TorchCompileWrapperWithCustomDispatcher`) and the wrappers are never traced;
during graph replay no Python runs at all.  The eager collector binds 1 of 189 required tensors per step (the sampled ids): Commit FAIL.

The boundary chosen here (graph-replay-safe taps):

  * `torch.ops.verity.collect_c(x, id)` -- a side-effecting custom op with NO outputs whose body is a device-to-device copy of `x` into a
    preallocated per-(module, invocation) staging slot ("ring").  The op is inserted into the FX graph by wrapping `module.forward` of every
    acquired module BEFORE vLLM compiles the model (a class-level patch of `GPUModelRunner.load_model`), so Dynamo traces the wrapper and the
    op becomes a node of the compiled graph.  Inductor lowers it as an opaque fallback kernel at the module boundary (its input is realized
    there, so the recorded value is the module's output BEFORE any later in-place op such as RoPE).  During CUDA graph capture the copy is
    recorded into the graph; on replay it runs as part of the graph -- no host involvement, nothing is skipped.
  * Identities: one static id per acquired module (+ member); a shared module that is called several times per step (Llama's rotary_emb) gets
    an invocation-numbered id per call site by renumbering the record nodes of the WHOLE Dynamo graph in execution order before vLLM splits
    it into pieces (`VllmBackend.__call__` patch).  Names are the committer's (`model.layers.N.mlp.down_proj`, `<name>/k`, `<name>/in`),
    so the Program/manifest identity rules, the binding map and the value checks are unchanged.
  * The host drains the ring once per step (`on_step_end`, inside the committer's step; before its final gather flush): every slot written
    this step is staged into the collector under its name in id order -> the step stream is identical in structure to the eager one.
    Row count per slot = the step's scheduled tokens (a replayed graph is padded to its capture size; the padding rows are never staged).
  * Steps too large for the ring (m > rows; only uncaptured prefill steps can be that large) fall back to the DIRECT path: the op's body
    runs eagerly (piece not captured) and acquires straight into the collector at the module boundary, with the eager design's flush points.

What is NOT substituted: the treatment runs the same CompilationMode / CUDAGraphMode / capture sizes as the control (recorded per attempt);
the control process has no record ops in its graphs (separate process, `--compiled-taps off`).  A slot is never allocated inside capture
(refused + counted); a slot written by a replay is only drained if that slot was written by the captured graph (`captured_keys`).

Negatives (VERITY_FAULT):
  compiled_omit:<leaf>   -- the record op of one module class is not inserted (omitted acquisition under replay) -> coverage FAIL (C1)
  compiled_late_drain    -- the drain reads each slot's value from the PREVIOUS step (late read under replay)   -> value check FAIL (C2)
"""
from __future__ import annotations

import importlib
import os
import time
from typing import Any

import torch

from verity_vllm import settings
from verity_vllm.engine import env as engine_env, hooks

OP_NS = "verity"
OP_NAME = "collect_c"
INV_SHIFT = 20                    # id layout after renumbering: FLAG | (invocation << INV_SHIFT) | base_id
INV_MASK = (1 << 20) - 1
BASE_MASK = (1 << INV_SHIFT) - 1
FLAG = 1 << 40

_SRC: "CompiledGraphSource | None" = None      # the op body dispatches to the live source (one per process)


def _op():
    return getattr(getattr(torch.ops, OP_NS), OP_NAME)


def compile_settings(llm: Any = None, runner: Any = None) -> dict[str, Any]:
    """What is actually compiled / graphed in this engine: CompilationMode, CUDAGraphMode, capture sizes, splitting ops, inductor graph
    partition, enforce_eager.  Recorded for every attempt (control and treatment must agree)."""
    out: dict[str, Any] = {}
    try:
        vc = None
        eng = getattr(llm, "llm_engine", None)
        vc = getattr(eng, "vllm_config", None)
        if vc is None and runner is not None:
            vc = getattr(runner, "vllm_config", None)
        if vc is None and llm is not None:
            try:
                from verity_vllm.engine import handle

                vc = getattr(handle.model_runner(llm), "vllm_config", None)
            except Exception:  # noqa: BLE001
                vc = None
        if vc is not None:
            cfg = vc.compilation_config
            out["compilation_mode"] = str(getattr(cfg, "mode", getattr(cfg, "level", None)))
            out["cudagraph_mode"] = str(getattr(cfg, "cudagraph_mode", None))
            sizes = getattr(cfg, "cudagraph_capture_sizes", None) or []
            out["cudagraph_capture_sizes"] = sorted(int(s) for s in list(sizes)[:256])
            out["max_cudagraph_capture_size"] = getattr(cfg, "max_cudagraph_capture_size", None)
            out["splitting_ops"] = list(getattr(cfg, "splitting_ops", None) or [])
            out["use_inductor_graph_partition"] = bool(getattr(cfg, "use_inductor_graph_partition", False))
            out["backend"] = str(getattr(cfg, "backend", None))
            out["custom_ops"] = list(getattr(cfg, "custom_ops", None) or [])
            # the Inductor config of the pieces: emulate_precision_casts = the compiled kernels perform the Program's bf16
            # roundings at every op boundary (default Inductor carries fp32 through fused regions, so a materialised boundary value
            # changes the computation); recorded and compared across arms
            out["inductor_compile_config"] = dict(getattr(cfg, "inductor_compile_config", None) or {})
            out["emulate_precision_casts"] = bool(out["inductor_compile_config"].get("emulate_precision_casts", False))
            out["enforce_eager"] = bool(getattr(vc.model_config, "enforce_eager", False))
            out["torch"] = torch.__version__
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)[:200]
    out["compile_cache_disabled"] = settings.raw("VLLM_DISABLE_COMPILE_CACHE") == "1"
    return out


def _register_op() -> None:
    if hasattr(torch.ops, OP_NS) and hasattr(getattr(torch.ops, OP_NS), OP_NAME):
        return

    @torch.library.custom_op(f"{OP_NS}::{OP_NAME}", mutates_args=())
    def collect_c(x: torch.Tensor, id: int) -> None:
        src = _SRC
        if src is not None:
            src._record(x, int(id))

    @collect_c.register_fake
    def _(x: torch.Tensor, id: int) -> None:
        return None

    op = _op().default
    try:
        from torch._higher_order_ops.effects import _register_effectful_op, _EffectType

        _register_effectful_op(op, _EffectType.ORDERED)      # never DCE'd, never reordered across other effectful ops
    except Exception:  # noqa: BLE001  (older torch: has_side_effect only)
        pass
    try:
        torch.fx.node.has_side_effect(op)
    except Exception:  # noqa: BLE001
        pass
    _install_inductor_lowering(op)


def _install_inductor_lowering(op) -> None:
    """Explicit Inductor fallback for the record op.  The implicit `make_fallback` path would add the op to `needs_realized_inputs`, which
    changes realize hints for OTHER nodes that share the recorded buffer (W1d audit); an explicit fallback handler realizes only the recorded
    input at the op's position and leaves the rest of the schedule alone."""
    from torch._inductor import lowering as L

    if op in L.lowerings:
        return
    fallback = L.fallback_handler(op, add_to_fallback_set=False)

    def handler(x, id):
        return fallback(x, id)

    handler._is_fallback_handler = True
    L.register_lowering(op, type_promotion_kind=None)(handler)


def _make_fwd(orig, nid_out: int, nids_k: tuple[int, ...], nid_in: int, op):
    """The traced wrapper: record the acquired input (embedding ids, a consumer-input site) and every tensor output of the module under its
    static id; `nid_out < 0` = an input-only module (a plan's consumer-input site on a module whose output is not acquired)."""
    Tensor = torch.Tensor

    if nid_out < 0:
        def fwd(*a, **kw):
            if a and isinstance(a[0], Tensor):
                op(a[0], nid_in)
            return orig(*a, **kw)
    elif nid_in >= 0:
        def fwd(*a, **kw):
            if a and isinstance(a[0], Tensor):
                op(a[0], nid_in)
            out = orig(*a, **kw)
            if isinstance(out, Tensor):
                op(out, nid_out)
            elif isinstance(out, (tuple, list)):
                for k, o in enumerate(out):
                    if isinstance(o, Tensor):
                        op(o, nids_k[k])
            return out
    else:
        def fwd(*a, **kw):
            out = orig(*a, **kw)
            if isinstance(out, Tensor):
                op(out, nid_out)
            elif isinstance(out, (tuple, list)):
                for k, o in enumerate(out):
                    if isinstance(o, Tensor):
                        op(o, nids_k[k])
            return out
    return fwd


class CompiledGraphSource:
    """Acquisition source for the compiled boundary (see module docstring).  Lifecycle:
    prepare() -> [vLLM builds + compiles the model; load_model patch wraps the acquired modules first] -> attach(committer) -> per step
    on_step_begin()/on_step_end(job) driven by the committer."""

    short_name = "compiled_graph"

    def __init__(self, rows: int = 512, fault: str | None = None) -> None:
        self.rows = int(rows)
        self.fault = (settings.VERITY_FAULT if fault is None else fault).strip()
        self.c: Any = None
        self.names: list[str] = []
        self.id_of: dict[str, int] = {}
        self.slots: dict[tuple[int, int], torch.Tensor] = {}
        self.captured_keys: set[tuple[int, int]] = set()
        self.eager_rows: dict[tuple[int, int], int] = {}      # slot -> rows written by an eager (Python-body) call THIS step
        self.eager_inv: dict[int, int] = {}                   # unrenumbered id -> eager invocation counter this step
        self.direct_step = False
        self.step_ntok = 0
        self.stale: dict[tuple[int, int], torch.Tensor] = {}  # compiled_late_drain: previous step's slot values
        self.wrapped: list[str] = []
        self.omitted: list[str] = []
        # [v2 acquire] the row's required-value manifest (`use_plan`) -> the acquisition plan derived at wrap time; `selection` = the
        # module boundaries wrapped (before a compiled_omit fault), held to the committer's at install
        self.plan_manifest: Any = None
        self.plan_dir: str | None = None
        self.plan: Any = None
        self.selection: dict[str, Any] | None = None
        self.renumber: dict[str, Any] = {}
        self.model = None
        self.runner = None
        self.prepared = False
        self._hooks = hooks.Hooks()
        self.stats: dict[str, Any] = {"capture_calls": 0, "eager_calls": 0, "direct_calls": 0, "direct_calls_outside_step": 0,
                                      "replay_drains": 0, "eager_drains": 0, "direct_steps": 0, "refused_capture_alloc": 0,
                                      "shape_mismatch": 0, "unwritten_skipped": 0, "oversize_in_capture": 0, "slot_bytes": 0,
                                      "slots": 0, "steps": 0, "drain_s": 0.0}
        self.errors: list[str] = []

    # ------------------------------------------------------------------ before the engine exists

    def prepare(self) -> None:
        """Register the op, patch vLLM so the acquired modules are wrapped before compilation and record nodes are renumbered per call site.
        Must run before `LLM()` is constructed; disables vLLM's compile cache (a cached graph would not contain the record nodes)."""
        global _SRC
        if self.prepared:
            return
        _SRC = self
        engine_env.disable_compile_cache()
        _register_op()
        me = self

        import vllm.compilation.backends as be

        if not getattr(be.VllmBackend.__call__, "_verity_compiled", False):
            orig_call = be.VllmBackend.__call__

            def call(self_, graph, example_inputs):
                try:
                    me._renumber(graph, getattr(self_, "prefix", ""))
                except Exception as e:  # noqa: BLE001
                    me.errors.append(f"renumber: {e!r}"[:300])
                return orig_call(self_, graph, example_inputs)

            call._verity_compiled = True      # type: ignore[attr-defined]
            self._hooks.patch(be.VllmBackend, "__call__", call)

        patched = []
        for modname in ("vllm.v1.worker.gpu.model_runner", "vllm.v1.worker.gpu_model_runner"):
            try:
                mod = importlib.import_module(modname)
            except Exception:  # noqa: BLE001
                continue
            cls = getattr(mod, "GPUModelRunner", None)
            if cls is None or getattr(cls.load_model, "_verity_compiled", False):
                continue
            orig_load = cls.load_model

            def load_model(self_, *a, __orig=orig_load, **kw):
                __orig(self_, *a, **kw)
                try:
                    me.wrap_model(self_.get_model(), self_)
                except Exception as e:
                    me.errors.append(f"wrap_model: {e!r}"[:300])
                    raise

            load_model._verity_compiled = True      # type: ignore[attr-defined]
            self._hooks.patch(cls, "load_model", load_model)
            patched.append(modname)
        self.stats["patched_runners"] = patched
        self.prepared = True

    def _id(self, name: str) -> int:
        nid = self.id_of.get(name)
        if nid is None:
            nid = len(self.names)
            self.names.append(name)
            self.id_of[name] = nid
        return nid

    def use_plan(self, manifest: Any, out_dir: str | os.PathLike) -> None:
        """The workload's required-value manifest: `wrap_model` (inside the engine build) derives the acquisition plan from it and
        the built model exactly as the committer's stage entry does (`acquire.stage.plan_at_stage`; documents under `out_dir`)."""
        self.plan_manifest, self.plan_dir = manifest, str(out_dir)

    def _plan_selection(self, model: torch.nn.Module, runner: Any, plan: Any) -> tuple[list[str], list[str]]:
        from verity_vllm.acquire.install import plan_selection

        if plan is None:
            if self.plan_manifest is None:
                raise RuntimeError("compiled rows wrap the acquisition plan's module boundaries and no plan or required-value manifest was "
                                   "passed in (commit_delta --required-manifest -> CompiledGraphSource.use_plan, before the engine build)")
            from verity_vllm.acquire.stage import plan_at_stage

            plan = plan_at_stage(model, runner, self.plan_manifest, self.plan_dir or ".", source="compiled_source wrap_model").plan
        self.plan = plan
        return plan_selection(plan, model)

    def wrap_model(self, model: torch.nn.Module, runner: Any, plan: Any = None) -> None:
        """Wrap `forward` of every acquired module with the record op, with the committer's selection: the acquisition plan's
        (`install.plan_selection`: producer_output modules record their outputs, consumer_input modules their first argument; `plan`,
        else derived from the `use_plan` manifest, else refused by name).  A VERITY_SKIP class's outputs are not recorded (the
        committer's `apply_skip`); the sampler is acquired by the committer's runner.sample wrapper, outside the compiled region."""
        from verity_vllm.commit.committer.native_host import _mro_names, skip_classes_of_env

        self.model, self.runner = model, runner
        if self.rows <= 0:
            # auto: the ring only needs to hold a CUDA-graph-sized step (larger steps are uncaptured -> direct path)
            cfg = getattr(getattr(runner, "vllm_config", None), "compilation_config", None)
            sizes = list(getattr(cfg, "cudagraph_capture_sizes", None) or [])
            mx = getattr(cfg, "max_cudagraph_capture_size", None) or (max(sizes) if sizes else None)
            self.rows = int(mx) if mx else 512
        self.stats["rows"] = self.rows
        op = _op()
        omit_leaf = self.fault.split(":", 1)[1] if self.fault.startswith("compiled_omit:") else None
        skip = skip_classes_of_env()
        targets: list[tuple[str, Any, bool, bool]] = []          # (name, module, record outputs, record first argument)
        outs, ins = self._plan_selection(model, runner, plan)
        outs_set, ins_set = set(outs), set(ins)
        for name, m in model.named_modules():
            out = name in outs_set and not (_mro_names(m) & skip)
            if out or name in ins_set:
                targets.append((name, m, out, name in ins_set))
        self.selection = {"engine": "v2", "plan_digest": getattr(self.plan, "digest", None),
                          "outputs": [n for n, _m, out, _pre in targets if out], "inputs": [n for n, _m, _out, pre in targets if pre]}
        for name, m, out, pre in targets:
            leaf = name.rsplit(".", 1)[-1]
            if omit_leaf is not None and leaf == omit_leaf:
                self.omitted.append(name)
                continue
            if "forward" in m.__dict__:
                raise RuntimeError(f"{name}: forward already an instance attribute")
            nid_out = self._id(name) if out else -1
            nids_k = tuple(self._id(f"{name}/{k}") for k in range(8)) if out else ()
            nid_in = self._id(f"{name}/in") if pre else -1
            self._hooks.patch(m, "forward", _make_fwd(m.forward, nid_out, nids_k, nid_in, op))
            self.wrapped.append(name)
        self.stats["wrapped"] = len(self.wrapped)
        self.stats["omitted"] = list(self.omitted)

    def _renumber(self, graph: torch.fx.GraphModule, prefix: str) -> None:
        """Give every record node of the whole (pre-split) Dynamo graph an invocation-numbered id: the k-th record of base id b in graph order
        becomes FLAG | k << INV_SHIFT | b.  Modules called once keep k = 0; the shared rotary module gets k = 0..L-1 (one per call site)."""
        packet = _op()
        targets = (packet, packet.default)          # Dynamo emits the OpOverloadPacket as target; later passes may resolve the overload
        count: dict[int, int] = {}
        n = 0
        for node in graph.graph.nodes:
            if node.op != "call_function" or not any(node.target is t for t in targets):
                continue
            if len(node.args) < 2 or not isinstance(node.args[1], int):
                continue
            base = int(node.args[1]) & BASE_MASK
            k = count.get(base, 0)
            count[base] = k + 1
            node.args = (node.args[0], FLAG | (k << INV_SHIFT) | base)
            n += 1
        graph.graph.lint()
        graph.recompile()
        self.renumber[prefix or "<root>"] = {"record_nodes": n, "bases": len(count), "max_invocations": max(count.values(), default=0),
                                             "multi_call": {self.names[b]: c for b, c in count.items() if c > 1}}

    # ------------------------------------------------------------------ attached to a committer

    def attach(self, committer: Any) -> None:
        self.c = committer
        committer.attach_source(self)
        setattr(committer, "external_acquisition", True)     # the committer must not wrap the (compiled) module forwards itself

    def check_selection(self, committer: Any = None) -> None:
        """[v2 acquire] the record ops compiled into the graphs must be the committer's selection -- its `_modules` outputs and acquired
        inputs, after its plan (`apply_plan_to_committer`) and skip: a compiled row never acquires a set other than the committer's plan.
        Refused by name on a difference."""
        c = committer if committer is not None else self.c
        if c is None or not hasattr(c, "_modules") or self.selection is None or self.model is None:
            return
        mods = dict(self.model.named_modules())
        outs = {n for n, _m in c._modules if n in mods}
        ins = {n for n, m in c._modules if c._acquires_input(n, m)} | {n for n, _m in c._input_only_modules()}
        mine_out, mine_in = set(self.selection["outputs"]), set(self.selection["inputs"])
        if outs != mine_out or ins != mine_in:
            diff = {"outputs_only_compiled": sorted(mine_out - outs)[:4], "outputs_only_committer": sorted(outs - mine_out)[:4],
                    "inputs_only_compiled": sorted(mine_in - ins)[:4], "inputs_only_committer": sorted(ins - mine_in)[:4]}
            diff = {k: v for k, v in diff.items() if v}
            raise RuntimeError(f"compiled source: the record ops compiled into the graphs are not the committer's plan selection: {diff}")

    def install(self) -> None:      # committer lifecycle: the taps are in the graphs; nothing to install per run
        self.check_selection()
        self.eager_inv = {}
        self.eager_rows = {}
        self.direct_step = False

    def uninstall(self) -> None:
        pass

    def remove_hooks(self) -> None:
        """Undo `prepare` and `wrap_model`.  Graphs vLLM compiled meanwhile keep their record nodes."""
        self._hooks.uninstall()
        self.prepared = False

    def declaration_classes(self) -> list[dict]:
        # the drained tensors carry the committer's own module names -> they are declared by the committer's layout-derived classes
        # (instance_outputs_native, input_token_ids, ...); this source adds no class of its own, only the mechanism (versions.compilation,
        # commit.compiled_source in every instrumented row)
        return []

    def on_step_begin(self) -> None:
        self.eager_inv = {}
        self.eager_rows = {}
        self.direct_step = False
        c = self.c
        self.step_ntok = int((getattr(c, "_plan_shape", None) or (0, 0))[0]) if c is not None else 0

    # the op body: runs during eager execution, piece capture (recorded into the graph) -- never during replay
    def _record(self, x: torch.Tensor, id: int) -> None:
        c = self.c
        in_step = c is not None and getattr(c, "_cur", None) is not None
        if id & FLAG:                                          # a record node of a compiled graph (renumbered per call site)
            base, inv = id & BASE_MASK, (id >> INV_SHIFT) & INV_MASK
        else:                                                  # uncompiled call path (logits_processor, eager modules): per-step numbering
            if not in_step:
                self.stats["direct_calls_outside_step"] += 1   # engine warm-up / profile runs: nothing to bind, no slot
                return
            base = id
            inv = self.eager_inv.get(id, 0)
            self.eager_inv[id] = inv + 1
        key = (base, inv)
        m = int(x.shape[0]) if x.dim() > 0 else 1
        capturing = torch.cuda.is_current_stream_capturing()
        if m > self.rows:
            if capturing:
                self.stats["oversize_in_capture"] += 1
                return
            self._direct(base, x)
            return
        slot = self.slots.get(key)
        if slot is None:
            if capturing:
                self.stats["refused_capture_alloc"] += 1
                return
            if not (id & FLAG) and not in_step:
                return
            slot = torch.empty((self.rows, *x.shape[1:]), dtype=x.dtype, device=x.device)
            self.slots[key] = slot
            self.stats["slot_bytes"] += slot.numel() * slot.element_size()
            self.stats["slots"] = len(self.slots)
        if slot.dtype != x.dtype or tuple(slot.shape[1:]) != tuple(x.shape[1:]):
            self.stats["shape_mismatch"] += 1
            return
        slot[:m].copy_(x)
        if capturing:
            self.captured_keys.add(key)
            self.stats["capture_calls"] += 1
        else:
            self.stats["eager_calls"] += 1
            self.eager_rows[key] = m

    def _direct(self, base: int, x: torch.Tensor) -> None:
        """Uncaptured step too large for the ring (prefill): acquire at the module boundary now, in execution order, and gather at once.
        Inside an Inductor-compiled piece a dead buffer may be reused by the next kernel (not only the eager design's known in-place
        overwriters), so every staged pointer is gathered before any later kernel is launched: one gather launch per acquisition, on
        uncaptured steps only (measured in the prefill step's wall time)."""
        c = self.c
        job = getattr(c, "_cur", None) if c is not None else None
        if job is None:
            self.stats["direct_calls_outside_step"] += 1
            return
        self.direct_step = True
        c._acquire(self.names[base], x, job)
        self.stats["direct_calls"] += 1
        if getattr(c, "gather", False):
            c._nc.flush()

    def on_step_end(self, job: Any) -> None:
        """Drain: stage every slot written this step (by the replayed graph or by an eager body call) under its name, in id order."""
        c = self.c
        if c is None:
            return
        t0 = time.perf_counter()
        m_step = self.step_ntok
        late = self.fault == "compiled_late_drain"
        self.stats["steps"] += 1
        if self.direct_step:
            self.stats["direct_steps"] += 1
        for key in sorted(self.slots):
            slot = self.slots[key]
            if key in self.eager_rows:
                rows = self.eager_rows[key]
                self.stats["eager_drains"] += 1
            elif self.direct_step:
                continue                                       # in-graph modules were acquired directly this step
            elif key in self.captured_keys and m_step > 0:
                rows = min(m_step, self.rows) if self._per_token(key) else self._rows_request(key)
                self.stats["replay_drains"] += 1
            else:
                self.stats["unwritten_skipped"] += 1
                continue
            src = slot
            if late:
                prev = self.stale.get(key)
                src = prev if prev is not None else slot
            c._acquire(self.names[key[0]], src[:rows], job)
            if late:
                self.stale[key] = slot.clone()
        self.stats["drain_s"] += time.perf_counter() - t0

    def _per_token(self, key: tuple[int, int]) -> bool:
        name = self.names[key[0]]
        return "logits_processor" not in name and "sampler" not in name

    def _rows_request(self, key: tuple[int, int]) -> int:
        c = self.c
        nreq = int((getattr(c, "_plan_shape", None) or (0, 0))[1]) if c is not None else 1
        return max(1, min(nreq, self.rows))

    # ------------------------------------------------------------------ evidence

    def compile_settings(self, llm: Any = None) -> dict[str, Any]:
        return compile_settings(llm, self.runner)

    def summary(self) -> dict[str, Any]:
        return {"short_name": self.short_name, "rows": self.rows, "fault": self.fault, "wrapped_modules": len(self.wrapped),
                "omitted_modules": list(self.omitted), "selection": self.selection, "renumber": self.renumber, "captured_keys": len(self.captured_keys),
                "stats": dict(self.stats), "errors": list(self.errors)[:20]}

    def stats_reset_step_counters(self) -> None:
        for k in ("capture_calls", "eager_calls", "direct_calls", "replay_drains", "eager_drains", "direct_steps", "steps", "drain_s",
                  "unwritten_skipped"):
            self.stats[k] = 0 if k != "drain_s" else 0.0
