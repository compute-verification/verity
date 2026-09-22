"""Compiled-preserving replay (CONTRACT §16.1): interception inside Inductor-compiled pieces.

Production runs stock compiled vLLM (torch.compile/Inductor, CUDA graphs FULL_AND_PIECEWISE). The
``compiled-nograph`` replay runs the *same compiled computation* — same Inductor config, same
generated kernels (proved by content hash, :mod:`kernel_identity`) — with CUDA graphs disabled so the
Python wrapper of every piece executes and can be observed between kernel launches.

What is observed, and how
-------------------------
* ``torch._inductor.runtime.triton_heuristics.CachingAutotuner.run`` is wrapped (class attribute,
  the launch itself is untouched): after every Triton kernel launched by a compiled piece, the
  kernel's *output buffers* are snapshotted to pinned host memory on the same stream (the mechanism
  of :class:`verity_vllm_adapter.capture.PinnedArena`, §15.4: a later in-place kernel or Inductor's
  buffer reuse cannot change what was captured). Output buffers are identified by Inductor's own
  argument naming (``out_ptr*``, ``in_out_ptr*``; ``c_ptr`` for vLLM's user-defined
  ``matmul_kernel_persistent``). Every other tensor argument is an input whose identity is resolved
  by address against the step registry (a value produced earlier in the step, a weight, the
  ``input_ids``/``positions`` buffers), exactly as the eager dispatch-mode capture does.
* ``vllm.compilation.piecewise_backend.PiecewiseBackend.__call__`` is wrapped only to know the
  index of the piece being executed (naming, expected-count audit).
* Ops *between* the pieces (attention, KV-cache write) and after the compiled forward (lm_head GEMM,
  sampler) go through the dispatcher and are recorded by the unchanged eager mechanism
  (``TorchDispatchMode``); the ``verity::bi_mm`` wrapper for the eager lm_head GEMM is installed
  *after* the engine compiled, so the traced pieces contain the original ``matmul_persistent``.

Why not ``torch._inductor.config.generate_intermediate_hooks``: verified on the pinned torch
(2.13.0): with the flag on (globally and via ``inductor_compile_config``) the generated wrappers
contain only the ``run_intermediate_hooks`` import and **zero** hook calls — ``origin_node``
assignment is best effort (``torch/_inductor/ir.py::assign_origin_node``) and none of the fused
buffers of vLLM's post-grad graphs carry one. The launch-level observation above needs no codegen
change at all (the cache key of the wrappers is even identical), which is stronger for §16.1.

Identity of a fused region
--------------------------
A fused Inductor kernel is one node: ``runtime_op = inductor.<kernel fn name>``, ``runtime_impl``
names the generated source by sha256 (the kernel module file Inductor wrote is byte-identical to the
``async_compile.triton('<name>', '''<source>''')`` string embedded in the wrapper), and the node's
``launch`` spec records, per launch argument, its role (in/out/weight/token/positions) and, for
integer arguments, their value as a linear function of the row count (fitted over the steps
observed). The relation ``inductor.fused.same_runtime.v1`` (:mod:`relations_inductor`) launches the
bundled source standalone on the authenticated operands for one row.

Node names follow the eager capture where the semantics coincide (``layers.3.self_attn.qkv_proj.gemm``,
``layers.3.post_attention_layernorm.rmsnorm``, ``layers.3.mlp.act_fn.act_mul``,
``layers.3.self_attn.rotary_emb.rope``, ``norm.rmsnorm``); ports are in output-argument order.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from verity_vllm_adapter.capture import Capture, NodeTemplate, Record, _extent, layer_of, node_name
from verity_vllm_adapter.launchspec import GEMM_FN, OUT_ARG_RE, USER_KERNEL_OUTPUTS, IntArg, classify_name

PATCHES_COMPILED = [
    "torch._inductor.runtime.triton_heuristics.CachingAutotuner.run wrapped on the class: after each launch inside a compiled piece the kernel's output buffers are snapshotted; the launch (args, grid, stream) is untouched",
    "vllm.compilation.piecewise_backend.PiecewiseBackend.__call__ wrapped on the class: current piece index only",
    "GPUModelRunner instance: execute_model/sample wrapped (dispatch mode for the ops outside the pieces: attention, kv-cache write, lm_head GEMM, sampler)",
    "batch_invariant.mm_batch_invariant / layernorm.rms_norm_batch_invariant rebound to torch.ops.verity.* AFTER the engine compiled (eager lm_head GEMM only; the compiled pieces traced the originals)",
]

@dataclass
class KernelInfo:
    fn_name: str
    kernel_name: str
    filename: str | None
    source: str
    sha256: str
    arg_names: list[str]
    out_args: list[str]
    constexprs: list[int]
    signature: dict[str, Any]


class CompiledCapture:
    """Launch-level capture inside the compiled pieces; attaches to an existing :class:`Capture`."""

    def __init__(self, cap: Capture, *, log: Any = print) -> None:
        self.cap = cap
        self.log = log
        self.kernels: dict[int, KernelInfo] = {}  # id(autotuner) -> info
        self.sources: dict[str, str] = {}  # sha256 -> source
        self.piece: int | None = None
        self.piece_launch = 0
        self.launches_per_step: list[int] = []
        self.launches_this_step = 0
        self.launches_per_piece: dict[int, int] = {}
        self.int_args: dict[tuple[str, str], IntArg] = {}  # (node, arg name) -> model
        self.launch_configs: dict[str, dict[str, Any]] = {}  # node -> autotuner config actually used
        self.reuse_events = 0  # output written into memory that held an earlier record of this step
        self.errors: list[dict[str, Any]] = []
        self.unaligned_args = 0
        self.installed = False
        self._orig_run = None
        self._orig_call = None
        cap.interception = "compiled-launch"  # description/summary label
        cap.patches_doc = PATCHES_COMPILED

    # ---------------- installation ----------------
    def install(self) -> None:
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        from vllm.compilation.piecewise_backend import PiecewiseBackend

        cc = self
        orig_run = CachingAutotuner.run
        orig_call = PiecewiseBackend.__call__

        def run(self_k, *args, stream, **kw):
            out = orig_run(self_k, *args, stream=stream, **kw)
            if cc.cap.mode.enabled:
                try:
                    cc.on_launch(self_k, args)
                except Exception as e:  # noqa: BLE001
                    cc.errors.append({"kernel": getattr(getattr(self_k, "fn", None), "__name__", "?"), "error": repr(e)})
                    if len(cc.errors) > 50:
                        raise
            return out

        def call(self_p, *args):
            prev, prev_n = cc.piece, cc.piece_launch
            cc.piece, cc.piece_launch = self_p.piecewise_compile_index, 0
            try:
                return orig_call(self_p, *args)
            finally:
                cc.piece, cc.piece_launch = prev, prev_n

        CachingAutotuner.run = run
        PiecewiseBackend.__call__ = call
        self._orig_run, self._orig_call = orig_run, orig_call
        self.installed = True
        # per-step counters ride on the Capture's step lifecycle
        orig_begin = self.cap.begin_step
        orig_after = self.cap.after_forward

        def begin_step(scheduler_output):
            orig_begin(scheduler_output)
            cc.launches_this_step = 0

        def after_forward():
            orig_after()
            cc.launches_per_step.append(cc.launches_this_step)

        self.cap.begin_step = begin_step
        self.cap.after_forward = after_forward

    def uninstall(self) -> None:
        if not self.installed:
            return
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        from vllm.compilation.piecewise_backend import PiecewiseBackend

        CachingAutotuner.run = self._orig_run
        PiecewiseBackend.__call__ = self._orig_call
        self.installed = False

    # ---------------- kernel identity ----------------
    def kernel_info(self, autotuner: Any) -> KernelInfo:
        info = self.kernels.get(id(autotuner))
        if info is not None:
            return info
        fn = autotuner.fn
        fn_name = str(getattr(fn, "__name__", "?"))
        meta = getattr(autotuner, "inductor_meta", {}) or {}
        kernel_name = str(meta.get("kernel_name") or fn_name)
        filename = getattr(autotuner, "filename", None)
        source = ""
        if filename and Path(filename).exists():
            source = Path(filename).read_text()
        elif hasattr(fn, "src"):
            source = str(fn.src)
        sha = hashlib.sha256(source.encode()).hexdigest()
        self.sources.setdefault(sha, source)
        arg_names = list(getattr(fn, "arg_names", []))
        constexprs = list(getattr(fn, "constexprs", []) or [])
        outs = [n for n in arg_names if OUT_ARG_RE.match(n)]
        if not outs:
            outs = USER_KERNEL_OUTPUTS.get(fn_name, [])
        tm = getattr(autotuner, "triton_meta", {}) or {}
        info = KernelInfo(fn_name, kernel_name, filename, source, sha, arg_names, outs, constexprs, dict(tm.get("signature", {})) if isinstance(tm, dict) else {})
        self.kernels[id(autotuner)] = info
        return info

    @staticmethod
    def _config_of(autotuner: Any) -> dict[str, Any] | None:
        try:
            launchers = getattr(autotuner, "launchers", None)
            if launchers and len(launchers) == 1:
                c = launchers[0].config
                return {"kwargs": dict(c.kwargs), "num_warps": int(c.num_warps), "num_stages": int(c.num_stages)}
        except Exception:  # noqa: BLE001
            return None
        return None

    # ---------------- per-launch recording ----------------
    def on_launch(self, autotuner: Any, args: tuple) -> None:
        cap = self.cap
        info = self.kernel_info(autotuner)
        self.launches_this_step += 1
        self.piece_launch += 1
        if self.piece is not None:
            self.launches_per_piece[self.piece] = self.launches_per_piece.get(self.piece, 0) + 1
        names = info.arg_names
        if len(args) > len(names):
            self.unaligned_args += 1
        # classify arguments (inputs first: an in_out_ptr's input identity must be read before the
        # output record is added; weights by pointer range; boundary buffers by pointer)
        slots: list[dict[str, Any]] = []  # one entry per launch argument, in order (the relation rebuilds the launch from it)
        outs: list[tuple[int, torch.Tensor, str]] = []
        arg_refs: list[dict[str, Any]] = []
        consts: dict[str, Any] = {}
        weight_modules: list[str] = []
        producer_layers: list[str] = []
        int_slots: list[tuple[int, str, int]] = []
        rows = None
        for i, a in enumerate(args):
            nm = names[i] if i < len(names) else f"arg{i}"
            if isinstance(a, torch.Tensor):
                if nm in info.out_args:
                    outs.append((i, a, nm))
                    if rows is None and a.dim() >= 1:
                        rows = int(a.shape[0])
                    slots.append({"index": i, "name": nm, "kind": "out", "port": len(outs) - 1, "dtype": str(a.dtype).replace("torch.", ""), "row_shape": list(a.shape[1:])})
                    continue
                if not a.is_cuda or a.numel() == 0:
                    slots.append({"index": i, "name": nm, "kind": "tensor", "role": "const", "detail": f"shape={list(a.shape)} dtype={a.dtype}"})
                    continue
                ref = cap._classify_arg(a)
                if ref.source == "unbound":
                    cap.unbound.append({"node": f"<{info.kernel_name}>", "arg": nm, "detail": ref.detail, "step": cap.step_index, "piece": self.piece})
                d = ref.to_json()
                d["name"] = nm
                d["shape"] = list(a.shape)
                d["dtype"] = str(a.dtype).replace("torch.", "")
                arg_refs.append(d)
                role = {"node": "in", "weight": "weight", "token": "token", "positions": "positions"}.get(ref.source, ref.source)
                slots.append({"index": i, "name": nm, "kind": "tensor", "role": role, "dtype": d["dtype"],
                              "row_shape": list(a.shape[1:]) if role in ("in", "token", "positions") else list(a.shape)})
                if ref.source == "weight" and ref.param:
                    weight_modules.append(ref.param)
                if ref.source == "node" and ref.node:
                    producer_layers.append(layer_of(ref.node))
            elif isinstance(a, bool) or a is None:
                consts[nm] = a
                slots.append({"index": i, "name": nm, "kind": "const", "value": a})
            elif isinstance(a, int):
                int_slots.append((i, nm, int(a)))
                slots.append({"index": i, "name": nm, "kind": "int"})
            elif isinstance(a, float):
                consts[nm] = a
                slots.append({"index": i, "name": nm, "kind": "const", "value": a})
            else:
                slots.append({"index": i, "name": nm, "kind": "const", "value": repr(a)})
        if not outs:
            self.errors.append({"kernel": info.kernel_name, "error": "no output argument identified", "arg_names": names[:12]})
            return
        # --- naming ---
        family, op_type, node_base, module = self._name(info, weight_modules, producer_layers)
        node = node_name(module, family, cap.op_counter, f"inductor.{info.fn_name}") if node_base is None else node_base
        if node_base is not None:  # generic name still needs the per-step counter for repeats
            k = cap.op_counter.get((module, node_base), 0)
            cap.op_counter[(module, node_base)] = k + 1
            if k:
                node = f"{node_base}{k}"
        layer = layer_of(module)
        # int args: fit the row-count model per node
        if rows is not None:
            for _, nm, v in int_slots:
                self.int_args.setdefault((node, nm), IntArg(nm)).add(rows, v)
        # GEMM: eager-compatible arg names so the existing gemm relation applies (x = a_ptr, w = b_ptr as weight.t())
        if info.fn_name in GEMM_FN:
            for d in arg_refs:
                d["name"] = {"a_ptr": "x", "b_ptr": "w"}.get(d["name"], d["name"])
        tpl = cap.templates.get(node)
        if tpl is None:
            tpl = NodeTemplate(
                op=node, layer=layer, family=family, op_type=op_type, runtime_op=f"inductor.{info.fn_name}",
                runtime_impl=f"inductor-triton:{info.kernel_name} src=sha256:{info.sha256}", args=arg_refs, consts=consts, module_path=module,
                relation="gemm.same_runtime.v1" if info.fn_name in GEMM_FN else "inductor.fused.same_runtime.v1",
                extra={"kernel_name": info.kernel_name, "kernel_fn": info.fn_name, "kernel_source_sha256": info.sha256, "piece": self.piece,
                       "piece_launch_index": self.piece_launch - 1,
                       "launch": {"n_args": len(args), "slots": slots, "arg_names": info.arg_names, "constexprs": info.constexprs}},
            )
            cap.templates[node] = tpl
        else:
            same = [cap._arg_key(a) for a in arg_refs] == [cap._arg_key(a) for a in tpl.args]
            if not same and len(tpl.arg_violations) < 20:
                tpl.arg_violations.append({"step": cap.step_index, "got": arg_refs})
        tpl.instances += 1
        cfg = self._config_of(autotuner)
        if cfg is not None:
            self.launch_configs[node] = cfg
        # --- outputs: snapshot now, on the launch stream ---
        t0 = time.perf_counter()
        for port, (i, o, nm) in enumerate(outs):
            if o.dim() == 0 or o.numel() == 0:
                continue
            prev, _ = cap.registry.lookup(o)
            if prev is not None and prev.host is not None:
                self.reuse_events += 1
            host = cap.arena.snapshot(o)
            rec = Record(
                node=node, port=port, seq=len(cap.registry.records), storage=o.untyped_storage().data_ptr(),
                start=o.data_ptr(), end=_extent(o), shape=tuple(o.shape), stride=tuple(o.stride()), dtype=o.dtype,
                host=host, row_kind="token", op=f"inductor.{info.fn_name}", family=family,
            )
            cap.registry.add(rec)
            tpl.ports.setdefault(port, {"dtype": str(o.dtype).replace("torch.", ""), "row_shape": list(o.shape[1:]), "row_kind": "token"})
        cap.timing["snapshot_s"] += time.perf_counter() - t0

    def _name(self, info: KernelInfo, weight_modules: list[str], producer_layers: list[str]) -> tuple[str, str, str | None, str]:
        """(family, op_type, explicit node name or None, module) for a launch (rules: launchspec.classify_name)."""
        return classify_name(info.fn_name, info.kernel_name, weight_modules, producer_layers)

    # ---------------- finish ----------------
    def finalize_templates(self) -> None:
        """Attach the fitted integer-argument models and launch configs to the node templates."""
        for node, tpl in self.cap.templates.items():
            if not tpl.runtime_op.startswith("inductor."):
                continue
            for s in tpl.extra["launch"]["slots"]:
                if s["kind"] == "int":
                    ia = self.int_args.get((node, s["name"]))
                    s["model"] = (ia.model() if ia else None) or {"model": "unobserved"}
                    s["violations"] = ia.violations if ia else None
            if node in self.launch_configs:
                tpl.extra["launch_config"] = self.launch_configs[node]

    def attach_production_launch_configs(self, production_kernels: dict[str, Any] | None) -> dict[str, Any]:
        """W1cF F8: carry production's recorded autotune choice (``kernels.json.autotune_configs``, the
        ``.best_config`` Inductor wrote in the production process) into every fused node as
        ``launch_config_production``; the verifier's standalone launch pins to it. Returns a summary
        (nodes with / without a production choice, and where the replay's own choice differs)."""
        from verity_vllm_adapter.launchspec import production_launch_config

        entries = (production_kernels or {}).get("autotune_configs") or []
        summary: dict[str, Any] = {"entries": len(entries), "with_production_config": 0, "without": [], "replay_differs": []}
        for node, tpl in self.cap.templates.items():
            if not tpl.runtime_op.startswith("inductor."):
                continue
            sha = str(tpl.extra.get("kernel_source_sha256", ""))
            cfg = production_launch_config(entries, sha)
            if cfg is None:
                summary["without"].append(node)
                continue
            tpl.extra["launch_config_production"] = cfg
            summary["with_production_config"] += 1
            rep = tpl.extra.get("launch_config")
            if rep is not None and rep != cfg:
                summary["replay_differs"].append({"node": node, "production": cfg, "replay": rep})
        return summary

    def write_sources(self, out_dir: Path) -> dict[str, Any]:
        """``kernels/<sha256>.py`` for every generated source launched during the capture (bundle
        revision: capture-format §11) plus an index."""
        d = Path(out_dir) / "kernels"
        d.mkdir(parents=True, exist_ok=True)
        index = {}
        for sha, src in sorted(self.sources.items()):
            (d / f"{sha}.py").write_text(src)
            names = sorted({k.kernel_name for k in self.kernels.values() if k.sha256 == sha})
            fns = sorted({k.fn_name for k in self.kernels.values() if k.sha256 == sha})
            index[sha] = {"kernel_names": names, "fn_names": fns, "bytes": len(src)}
        (d / "index.json").write_text(__import__("json").dumps({"schema": "verity-vllm/kernels/v1", "kernels": index}, indent=1) + "\n")
        return index

    def summary(self) -> dict[str, Any]:
        return {
            "kernels_seen": len(self.kernels),
            "distinct_sources": len(self.sources),
            "launches_per_step": self.launches_per_step[:64],
            "launches_per_step_distinct": sorted(set(self.launches_per_step)),
            "launches_per_piece": dict(sorted(self.launches_per_piece.items())),
            "buffer_reuse_events": self.reuse_events,
            "unaligned_arg_lists": self.unaligned_args,
            "errors": self.errors[:20],
            "launch_configs": self.launch_configs,
            "patches": PATCHES_COMPILED,
        }


