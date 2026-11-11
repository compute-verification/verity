"""The Inductor wrapper hooks of the served-default compiled source (`served_compiled_source`): after each scheduled node's codegen
the generated wrapper calls `run_intermediate_hooks('verity|<op>|<buf>', <buf>)` for each of the node's outputs that is allocated and
still live, and under `batch` also `run_intermediate_hooks(FLUSH, None)` wherever a hooked buffer's bytes could next change.
"""
from __future__ import annotations

import re
from typing import Any

from verity_vllm.engine import hooks

PREFIX = "verity|"
FLUSH = "verity_flush"
#: The combined CUDA scheduler's per-node codegen entry points (`Scheduler._codegen` dispatches each node to one of them; extern
#: nodes go to `Scheduler.codegen_extern_call`).  The hooks are emitted after the call, for `scheduler.current_node`.
CODEGEN_METHODS = ("codegen_node", "codegen_template", "codegen_combo_kernel", "codegen_nested_reduction", "codegen_mix_order_reduction")


def _label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", text).strip("_") or "op"


def op_label(node: Any) -> str:
    """A kernel label for a scheduler node: an extern kernel's name (`mm`, `moe_forward`), else the origin op of its first output."""
    from torch._inductor import ir

    inner = getattr(node, "node", None)
    if isinstance(inner, ir.ExternKernel):
        name = getattr(inner, "python_kernel_name", None) or ""
        parts = [p for p in name.split(".") if p and p != "default"]
        return _label(parts[-1]) if parts else "extern"
    for buf in node.get_outputs():
        origin = buf.node.get_origin_node() if hasattr(buf.node, "get_origin_node") else None
        target = getattr(origin, "target", None)
        if target is not None:
            return _label(str(getattr(target, "_opname", None) or getattr(target, "__name__", target)))
    return "fused"


def hookable_outputs(node: Any, wrapper: Any, graph: Any) -> tuple[list[str], dict[str, int]]:
    """The output buffers of `node` that exist as CUDA tensors in the generated wrapper right after the node's code, and why the
    others don't."""
    from torch._inductor import ir

    mutation = getattr(ir, "MutationOutput", ())
    keep: list[str] = []
    skipped: dict[str, int] = {}
    for buf in node.get_outputs():
        name = buf.get_name()
        layout = getattr(buf.node, "layout", None)
        device = buf.node.get_device() if hasattr(buf.node, "get_device") else None
        why = ("removed" if name in graph.removed_buffers or name in graph.inplaced_to_remove else
               "freed" if name in wrapper.freed else
               "mutation" if isinstance(buf.node, mutation) else
               "no_tensor" if isinstance(layout, (ir.NoneLayout, ir.MultiOutputLayout)) else
               "not_cuda" if device is None or device.type != "cuda" else None)
        if why is None:
            keep.append(name)
        else:
            skipped[why] = skipped.get(why, 0) + 1
    return keep, skipped


def storage_roots(node: Any) -> set[str]:
    """The names of `node`'s buffer and of every buffer whose storage it aliases (a NonOwningLayout view, an aliasing extern output)."""
    from torch._inductor.virtualized import V

    out: set[str] = set()
    todo = [node]
    while todo:
        n = todo.pop()
        name = n.get_name()
        if name in out:
            continue
        out.add(name)
        try:
            aliases = n.get_inputs_that_alias_output()
        except (AttributeError, NotImplementedError):
            aliases = ()
        todo.extend(b for b in map(V.graph.try_get_buffer, aliases) if b is not None)
    return out


def _writes_elsewhere(node: Any) -> bool:
    """Whether a scheduler node writes a buffer other than its own outputs (a mutation)."""
    own = set(node.get_buffer_names())
    return any(dep.name not in own for dep in node.read_writes.writes) or any(buf.get_mutations() for buf in node.get_outputs())


def install_codegen_hooks(hook: Any, into: hooks.Hooks, batch: bool = False) -> dict[str, Any]:
    """Patch Inductor so every generated wrapper calls `hook(name, tensor)` on each materialized buffer right after the node that
    writes it.  Must run before the engine compiles; returns the live codegen counters.

    `batch`: the wrapper also calls `hook(FLUSH, None)` where a hooked buffer's bytes could next change -- before a node that writes a
    buffer other than its own outputs, before a kernel that reuses a hooked buffer in place, and at the wrapper's end -- and a hooked
    buffer's storage is never handed to a later allocation (its free is kept out of Inductor's reuse planning), so a hook may record
    its copy and issue it at the next flush.  Kernels and their in-place decisions are unchanged; only allocation lines differ."""
    from torch._functorch import config as fcfg
    from torch._inductor import config as icfg
    from torch._inductor import hooks as ih
    from torch._inductor import scheduler as isched
    from torch._inductor.codegen import wrapper as iw
    from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
    from torch._inductor.virtualized import V

    into.patch(icfg, "fx_graph_cache", False)
    into.patch(fcfg, "enable_autograd_cache", False)
    batch = batch and not icfg.memory_planning                         # the static planner reuses by type, not through `plan`
    stats: dict[str, Any] = {"nodes": 0, "hooks": 0, "skipped": {}, "batch": batch, "flushes": 0, "kept": 0}

    class KeptFree(iw.FreeIfNotReusedLine):
        def plan(self, state: Any) -> Any:
            return iw.NullLine(self.wrapper) if self.node.get_name() in V.graph.removed_buffers else self

    def flush(wrapper: Any) -> None:
        wrapper.writeline(f"run_intermediate_hooks({FLUSH!r}, None)")
        wrapper.__dict__["_verity_pending"] = set()
        stats["flushes"] += 1

    def emit() -> None:
        sched, wrapper = V.graph.scheduler, V.graph.wrapper_code
        node = getattr(sched, "current_node", None)
        if node is None or not getattr(wrapper, "supports_intermediate_hooks", False):
            return
        seen = wrapper.__dict__.setdefault("_verity_hooked", set())
        if node.get_name() in seen:
            return
        seen.add(node.get_name())
        names, skipped = hookable_outputs(node, wrapper, V.graph)
        op = op_label(node)
        stats["nodes"] += 1
        for why, k in skipped.items():
            stats["skipped"][why] = stats["skipped"].get(why, 0) + k
        for name in names:
            wrapper.writeline(f"run_intermediate_hooks({PREFIX + op + '|' + name!r}, {name})")
            stats["hooks"] += 1
            if batch:
                roots = storage_roots(V.graph.get_buffer(name))
                wrapper.__dict__.setdefault("_verity_pending", set()).update(roots)
                wrapper.__dict__.setdefault("_verity_roots", set()).update(roots)

    def around(orig: Any) -> Any:
        def run(self: Any, *a: Any, **kw: Any) -> Any:
            wrapper, node = V.graph.wrapper_code, getattr(V.graph.scheduler, "current_node", None)
            if batch and node is not None and wrapper.__dict__.get("_verity_pending") and _writes_elsewhere(node):
                flush(wrapper)
            out = orig(self, *a, **kw)
            emit()
            return out
        return run

    def inplace_reuse(orig: Any) -> Any:
        def run(self_w: Any, input_buffer: Any, output_buffer: Any) -> Any:
            if self_w.__dict__.get("_verity_pending", set()) & storage_roots(input_buffer):
                flush(self_w)
            return orig(self_w, input_buffer, output_buffer)
        return run

    def ir_passes(orig: Any) -> Any:
        def run(self_w: Any, *a: Any, **kw: Any) -> Any:
            roots = self_w.__dict__.get("_verity_roots", set())
            for line in self_w.lines if roots else ():
                if type(line) is iw.FreeIfNotReusedLine and not line.comm_buffer and storage_roots(line.node) & roots:
                    line.__class__ = KeptFree
                    stats["kept"] += 1
            out = orig(self_w, *a, **kw)
            if getattr(self_w, "supports_intermediate_hooks", False):
                self_w.lines.append(f"run_intermediate_hooks({FLUSH!r}, None)")
                stats["flushes"] += 1
            return out
        return run

    for name in CODEGEN_METHODS:
        if hasattr(CUDACombinedScheduling, name):
            into.wrap(CUDACombinedScheduling, name, around)
    into.wrap(isched.Scheduler, "codegen_extern_call", around)
    if batch:
        into.wrap(iw.PythonWrapperCodegen, "codegen_inplace_reuse", inplace_reuse)
        into.wrap(iw.PythonWrapperCodegen, "run_wrapper_ir_passes", ir_passes)
    if hook not in ih.INTERMEDIATE_HOOKS:
        into.append(ih.INTERMEDIATE_HOOKS, hook)
    return stats
