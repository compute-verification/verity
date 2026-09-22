"""The post-grad FX pass that inserts ``verity::record`` after materialised buffers, and the two
replay-process hooks that (1) run it as the *last* post-grad pass of vLLM's ``PostGradPassManager``
and (2) tell it which piecewise subgraph is being compiled.

Insertion policy (``policy="materialised"``; the safety of every point is *measured* afterwards by
:mod:`.kernels`, this is only the candidate generator):

======================  ======================================================================
role                    buffer
======================  ======================================================================
``extern_out``          output of a user-defined Triton kernel (``triton_kernel_wrapper_*``, the
                        batch-invariant GEMM), of an extern aten kernel (mm/addmm/bmm) or of a
                        custom op (``vllm::``/``_C::``); a mutated output argument counts
``extern_in``           base buffer of every tensor input of such a kernel (through views)
``graph_out``          base buffer of every tensor the piecewise graph returns
``input``               activation placeholder: floating tensor input with a symbolic dim
======================  ======================================================================

Everything else (values that live only inside an Inductor fusion group, e.g. the residual stream
between the attention add and the MLP add of one layer, or fp32 rmsnorm intermediates) is *not* a
buffer and cannot be observed without changing codegen; the pass never touches it.

Each record gets ``id = piece_index * ID_STRIDE + k``; encoding the piece makes every piece's graph
distinct (no Inductor FX-graph-cache sharing between layers), so the pass runs for every piece and the
id table is exact. The table (``FxIdTable``) is written to ``fx_ids.json`` in the run directory.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import fx

from verity_vllm_adapter.compiled_fx import record_op

ID_STRIDE = 10_000

VIEW_TARGETS = {
    "aten.view", "aten._unsafe_view", "aten.reshape", "aten.t", "aten.transpose", "aten.permute",
    "aten.slice", "aten.select", "aten.unsqueeze", "aten.squeeze", "aten.expand", "aten.alias",
    "aten.detach", "aten.narrow", "aten.as_strided", "aten.view_as_real", "aten.flatten",
    "aten.split", "aten.split_with_sizes", "aten.unbind",
}
ALLOC_TARGETS = {"aten.empty", "aten.empty_strided", "aten.new_empty", "aten.empty_like", "aten.zeros", "aten.full", "aten.empty_permuted"}
EXTERN_ATEN = {"aten.mm", "aten.addmm", "aten.bmm", "aten.baddbmm", "aten._scaled_mm", "aten.convolution", "aten.embedding_bag", "aten._embedding_bag"}
HOP_NAMES = {"triton_kernel_wrapper_functional", "triton_kernel_wrapper_mutation", "auto_functionalized", "auto_functionalized_v2"}


def target_name(node: fx.Node) -> str:
    t = node.target
    if isinstance(t, torch._ops.OpOverload):
        return f"{t.namespace}.{t._schema.name.split('::')[-1]}"
    if isinstance(t, torch._ops.OpOverloadPacket):
        return f"{t._qualified_op_name.replace('::', '.')}"
    if isinstance(t, torch._ops.HigherOrderOperator):
        return t.__name__
    return getattr(t, "__name__", str(t))


def is_tensor_val(node: Any) -> bool:
    return isinstance(node, fx.Node) and isinstance(node.meta.get("val"), torch.Tensor)


def _is_symbolic(d: Any) -> bool:
    """True for a SymInt with free symbols (a SymInt that is a constant, as fake modes hand out for static
    dims, counts as static)."""
    if isinstance(d, int):
        return False
    try:
        from torch.fx.experimental.symbolic_shapes import free_symbols

        return bool(free_symbols(d))
    except Exception:  # noqa: BLE001
        return True


def shape_of(node: fx.Node) -> list[Any]:
    v = node.meta.get("val")
    if not isinstance(v, torch.Tensor):
        return []
    return [str(d) if _is_symbolic(d) else int(d) for d in v.shape]


def has_symbolic_dim(node: fx.Node) -> bool:
    v = node.meta.get("val")
    return isinstance(v, torch.Tensor) and any(_is_symbolic(d) for d in v.shape)


def module_path(node: fx.Node) -> str:
    """Innermost module path from ``nn_module_stack`` (``model.layers.3.self_attn.qkv_proj`` style)."""
    st = node.meta.get("nn_module_stack") or {}
    last = None
    for _k, v in st.items():
        last = v
    if last is None:
        return ""
    path = last[0] if isinstance(last, (tuple, list)) else str(last)
    return clean_module_path(str(path))


def clean_module_path(path: str) -> str:
    """``L['self'].layers.3.self_attn.qkv_proj`` / ``L__self___layers_3_self_attn_qkv_proj`` ->
    ``layers.3.self_attn.qkv_proj``."""
    if path.startswith("L['self']."):
        return path[len("L['self']."):]
    if path == "L['self']":
        return ""
    if path.startswith("L__self___"):
        path = path[len("L__self___"):]
        path = re.sub(r"_(\d+)_", r".\1.", path)
        path = path.replace("__", ".").replace("_modules_", ".").replace("_", ".")
    return path


def split_placeholder_identity(name: str) -> dict[str, str]:
    """Dynamo's flat-arg names of the split submodule carry the parameter identity:
    ``l_self_modules_layers_modules_0_modules_self_attn_modules_o_proj_parameters_weight_`` ->
    ``{"kind": "param", "name": "layers.0.self_attn.o_proj.weight"}``; ``..._buffers_cos_sin_cache_`` ->
    buffer; ``l_positions_`` -> positions; ``hidden_states`` / ``output_N`` -> activations; the rest are
    SymInts computed by the parent graph."""
    if name.startswith("l_self_modules_") or name.startswith("l_self_"):
        body = name[len("l_self_"):].rstrip("_")
        if body.startswith("modules_"):
            body = body[len("modules_"):]
        parts = re.split(r"_(?:modules|parameters|buffers)_", body)
        kind = "buffer" if "_buffers_" in name else "param"
        return {"kind": kind, "name": ".".join(parts)}
    if name.startswith("l_") and name.endswith("_"):
        return {"kind": "input", "name": name[2:-1]}
    if (name in ("hidden_states", "residual", "positions", "input_ids", "inputs_embeds") or name.startswith("output")
            or re.match(r"^(hidden_states|residual|inputs_embeds)_\d+$", name)):
        return {"kind": "activation", "name": name}
    return {"kind": "sym", "name": name}


def original_aten(node: fx.Node) -> str:
    oa = node.meta.get("original_aten")
    return str(oa) if oa is not None else ""


def base_of(node: fx.Node) -> fx.Node:
    """Walk back through view-like ops to the node owning the storage."""
    seen = 0
    while node.op == "call_function" and seen < 32:
        tn = target_name(node)
        if tn in VIEW_TARGETS and node.args and isinstance(node.args[0], fx.Node):
            node = node.args[0]
        elif tn == "<built-in function getitem>" or (node.target is __import__("operator").getitem):
            src = node.args[0]
            if isinstance(src, fx.Node) and target_name(src) in ("aten.split", "aten.split_with_sizes", "aten.unbind", "aten.chunk"):
                node = src.args[0]
            else:
                break
        else:
            break
        seen += 1
    return node


@dataclass
class IdEntry:
    id: int
    piece: int
    k: int
    node: str
    base_node: str
    target: str
    role: str
    shape: list[Any]
    dtype: str
    module_path: str
    module_stack: list[str]
    original_aten: str
    producer_target: str
    producer_module_path: str
    consumers: list[dict[str, str]]
    inserted_after: str
    stack_trace_tail: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PieceInfo:
    index: int
    submod_name: str
    placeholders: list[str]
    placeholder_shapes: list[Any]
    outputs: list[str]
    postgrad_placeholders: list[str] = field(default_factory=list)
    postgrad_outputs: list[str] = field(default_factory=list)
    placeholder_binding: list[dict[str, Any]] = field(default_factory=list)  # post-grad arg -> split-graph name -> param/activation/sym
    record_count: int = 0
    node_count: int = 0
    extern_nodes: list[str] = field(default_factory=list)
    pass_ran: bool = False


class FxIdTable:
    """id -> identity information, filled by the pass; serialisable to ``fx_ids.json``."""

    def __init__(self) -> None:
        self.entries: dict[int, IdEntry] = {}
        self.pieces: dict[int, PieceInfo] = {}
        self.graphs: dict[int, str] = {}  # piece -> readable final graph (optional, dump mode)
        self.skipped: list[dict[str, Any]] = []

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "verity-vllm/fx-ids/v1",
            "id_stride": ID_STRIDE,
            "entries": [asdict(e) for e in sorted(self.entries.values(), key=lambda e: e.id)],
            "pieces": [asdict(p) for p in sorted(self.pieces.values(), key=lambda p: p.index)],
            "skipped": self.skipped[:200],
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=1, default=str) + "\n")
        if self.graphs:
            d = path.parent / "fx_graphs"
            d.mkdir(exist_ok=True)
            for i, g in self.graphs.items():
                (d / f"piece_{i:03d}.py").write_text(g)


# --------------------------------------------------------------------------------------------------
# the pass
# --------------------------------------------------------------------------------------------------

CURRENT_PIECE: dict[str, Any] = {"index": -1, "submod_name": None, "placeholders": [], "placeholder_shapes": [], "outputs": []}


class RecordInsertionPass:
    """Callable ``(fx.Graph) -> None``; ``uuid`` for Inductor's cache key (vLLM ``InductorPass`` protocol)."""

    VERSION = "verity-record-pass/v1"

    def __init__(self, table: FxIdTable, *, policy: str = "materialised", dry_run: bool = False, dump_graphs: bool = False,
                 roles: tuple[str, ...] = ("extern_out", "extern_in", "graph_out", "input"), placement: str = "late") -> None:
        self.table = table
        self.policy = policy
        if placement not in ("producer", "late"):
            raise ValueError(f"placement must be 'producer' or 'late', got {placement!r}")
        # producer: the record call goes right after the node that materialises the buffer (or the last
        #   in-place extern writing it). Inductor orders every later reader of the buffer behind the
        #   side-effecting call, so a reader that production fused with the producer (a pointwise
        #   consumer of a materialised pointwise output, e.g. Qwen2's qkv bias add feeding the rope) is
        #   forced into a separate kernel: codegen changes and the §3 comparison fails (B1).
        # late: the record call goes after the buffer's last in-graph reader (through views), i.e. the
        #   last point where Inductor keeps the buffer alive anyway; no reader is reordered.
        self.placement = placement
        self.dry_run = dry_run
        self.dump_graphs = dump_graphs
        self.roles = set(roles)
        self.overload = record_op.install()
        self.calls = 0

    def uuid(self) -> str:
        # the record op's Inductor lowering kind is part of the codegen, hence of the compile-cache key
        return (f"{self.VERSION}/policy={self.policy}/placement={self.placement}/lowering={record_op.lowering_kind() or 'implicit'}"
                f"/dry_run={self.dry_run}/roles={','.join(sorted(self.roles))}")

    def is_applicable_for_range(self, compile_range: Any) -> bool:  # vLLM InductorPass protocol
        return True

    # -- classification ---------------------------------------------------------------------------
    @staticmethod
    def _is_extern(node: fx.Node) -> bool:
        if node.op != "call_function":
            return False
        t = node.target
        if isinstance(t, torch._ops.HigherOrderOperator):
            return t.__name__ in HOP_NAMES
        if isinstance(t, torch._ops.OpOverload):
            if t.namespace not in ("aten", "prims"):
                return t is not record_op.overload()
            return target_name(node) in EXTERN_ATEN
        return False

    @staticmethod
    def _tensor_args(node: fx.Node) -> list[fx.Node]:
        out: list[fx.Node] = []

        def visit(a: Any) -> None:
            if isinstance(a, fx.Node):
                if is_tensor_val(a):
                    out.append(a)
            elif isinstance(a, (list, tuple)):
                for x in a:
                    visit(x)
            elif isinstance(a, dict):
                for x in a.values():
                    visit(x)

        visit(node.args)
        visit(node.kwargs)
        return out

    @staticmethod
    def _mutated_args(node: fx.Node) -> list[fx.Node]:
        """Tensor args the extern node writes: schema ``(a!)`` args of an OpOverload, or, for the
        mutation-form Triton HOP, the allocation nodes passed as kwargs."""
        t = node.target
        outs: list[fx.Node] = []
        if isinstance(t, torch._ops.OpOverload):
            for i, a in enumerate(t._schema.arguments):
                if a.alias_info is not None and a.alias_info.is_write:
                    v = node.args[i] if i < len(node.args) else node.kwargs.get(a.name)
                    if isinstance(v, fx.Node) and is_tensor_val(v):
                        outs.append(v)
        elif isinstance(t, torch._ops.HigherOrderOperator) and t.__name__ == "triton_kernel_wrapper_mutation":
            kw = node.kwargs.get("kwargs", {})
            for v in (kw.values() if isinstance(kw, dict) else []):
                if isinstance(v, fx.Node) and is_tensor_val(v) and target_name(base_of(v)) in ALLOC_TARGETS:
                    outs.append(v)
        return outs

    # -- the pass ---------------------------------------------------------------------------------
    def __call__(self, graph: fx.Graph) -> None:
        self.calls += 1
        piece = int(CURRENT_PIECE.get("index", -1))
        if piece < 0:
            piece = 9000 + self.calls  # not compiled through the piecewise interpreter (tests)
        info = self.table.pieces.get(piece)
        if info is None:
            info = PieceInfo(index=piece, submod_name=str(CURRENT_PIECE.get("submod_name")), placeholders=list(CURRENT_PIECE.get("placeholders", [])),
                             placeholder_shapes=list(CURRENT_PIECE.get("placeholder_shapes", [])), outputs=list(CURRENT_PIECE.get("outputs", [])))
            self.table.pieces[piece] = info
        info.pass_ran = True
        nodes = list(graph.nodes)
        info.node_count = len(nodes)
        phs = [n for n in nodes if n.op == "placeholder"]
        info.postgrad_placeholders = [n.name for n in phs]
        out_node = graph.output_node()
        info.postgrad_outputs = [a.name if isinstance(a, fx.Node) else repr(a) for a in _flatten(out_node.args)]
        # positional binding post-grad placeholder <-> split-graph placeholder (AOT keeps flat-arg order)
        binding = []
        for i, n in enumerate(phs):
            split_name = info.placeholders[i] if i < len(info.placeholders) else None
            v = n.meta.get("val")
            b: dict[str, Any] = {"postgrad": n.name, "split": split_name, "shape": shape_of(n) if isinstance(v, torch.Tensor) else None,
                                 "dtype": str(v.dtype).replace("torch.", "") if isinstance(v, torch.Tensor) else None,
                                 "sym": None if isinstance(v, torch.Tensor) or v is None else str(v)}
            if split_name is not None:
                b.update(split_placeholder_identity(split_name))
                if isinstance(v, torch.Tensor) and b["kind"] == "sym":
                    b["kind"] = "activation"  # a tensor placeholder with a graph-derived name (e.g. residual_1)
                elif not isinstance(v, torch.Tensor) and v is not None and b["kind"] not in ("sym",):
                    b["kind"] = "sym"
            binding.append(b)
        info.placeholder_binding = binding

        # candidate: base node -> (role, insert_after, extra)
        cands: dict[fx.Node, tuple[str, fx.Node, dict[str, Any]]] = {}
        order = {n: i for i, n in enumerate(nodes)}

        def propose(base: fx.Node, role: str, after: fx.Node, extra: dict[str, Any]) -> None:
            if not is_tensor_val(base):
                return
            if base.op == "placeholder" and role != "input":
                return  # inputs are handled by the placeholder rule (weights are never recorded)
            if base.meta["val"].numel() == 0 if all(isinstance(d, int) for d in base.meta["val"].shape) else False:
                return
            if target_name(base) in ALLOC_TARGETS and after is base:
                # a bare allocation: the functional Triton HOP's ``c_ptr`` slot (its content is the
                # HOP's getitem output, recorded there) or an output buffer filled by an op outside
                # this piece (the attention output). Uninitialised here: never a record point.
                self.table.skipped.append({"piece": piece, "node": base.name, "role": role, "why": "bare allocation"})
                return
            prev = cands.get(base)
            if prev is None or order[after] > order[prev[1]]:
                # keep the latest insertion point: after the last extern kernel that mutates the buffer
                merged = dict(prev[2]) if prev else {}
                merged.update(extra)
                cands[base] = (prev[0] if prev and prev[0] != "extern_in" else role, after, merged)

        last_placeholder = None
        for n in nodes:
            if n.op == "placeholder":
                last_placeholder = n
        for n in nodes:
            if n.op == "placeholder" and "input" in self.roles:
                if is_tensor_val(n) and has_symbolic_dim(n) and n.meta["val"].is_floating_point():
                    consumers = [{"node": u.name, "target": target_name(u), "module_path": module_path(u)} for u in n.users]
                    propose(n, "input", last_placeholder, {"consumers": consumers})
            if n.op == "call_function" and self._is_extern(n):
                info.extern_nodes.append(f"{n.name}:{target_name(n)}")
                mutated = self._mutated_args(n)
                for a in mutated:
                    b = base_of(a)
                    if "extern_out" in self.roles:
                        propose(b, "extern_out", n, {"producer": n.name, "producer_target": target_name(n), "producer_module_path": module_path(n)})
                for a in self._tensor_args(n):
                    if a in mutated:
                        continue
                    b = base_of(a)
                    if b.op == "placeholder":
                        continue
                    if "extern_in" in self.roles:
                        propose(b, "extern_in", b, {"consumer": n.name, "consumer_target": target_name(n), "consumer_module_path": module_path(n)})
                # functional outputs (the HOP's returned tensors / a returning custom op)
                if is_tensor_val(n) and "extern_out" in self.roles:
                    propose(n, "extern_out", n, {"producer": n.name, "producer_target": target_name(n), "producer_module_path": module_path(n)})
                for u in list(n.users):
                    if u.target is __import__("operator").getitem and is_tensor_val(u) and "extern_out" in self.roles:
                        propose(u, "extern_out", u, {"producer": n.name, "producer_target": target_name(n), "producer_module_path": module_path(n), "getitem": str(u.args[1])})
        if "graph_out" in self.roles:
            for a in _flatten(out_node.args):
                if isinstance(a, fx.Node) and is_tensor_val(a):
                    b = base_of(a)
                    if b.op == "placeholder":
                        continue
                    after = b
                    # a graph output produced by an in-place extern kernel: insert after that kernel
                    for c, (_r, aft, _x) in cands.items():
                        if c is b and order[aft] > order[after]:
                            after = aft
                    propose(b, "graph_out", after, {"returned_as": a.name})

        alias_base: dict[fx.Node, fx.Node] = {n: base_of(n) for n in nodes if n.op == "call_function"}
        mutators: dict[fx.Node, list[fx.Node]] = {}  # base -> extern nodes writing it in place
        for n in nodes:
            if n.op == "call_function" and self._is_extern(n):
                for a in self._mutated_args(n):
                    mutators.setdefault(base_of(a), []).append(n)

        def late_point(base: fx.Node, after: fx.Node) -> fx.Node:
            """After the last in-graph reader of ``base`` (through views), never earlier than ``after``;
            not moved when an in-place extern writes the buffer later (the value would change)."""
            if any(order[m] > order[after] for m in mutators.get(base, [])):
                return after
            aliases = {n for n, b in alias_base.items() if b is base} | {base}
            last = after
            for n in aliases:
                for u in n.users:
                    if u is out_node or u in aliases or u.target is self.overload:
                        continue
                    if order[u] > order[last]:
                        last = u
            return last

        # insert, in graph order, deterministic ids
        k = 0
        for base, (role, after, extra) in sorted(cands.items(), key=lambda kv: order[kv[0]]):
            rid = piece * ID_STRIDE + k
            if self.placement == "late":
                moved = late_point(base, after)
                if moved is not after:
                    extra = {**extra, "producer_point": after.name}
                    after = moved
            if not self.dry_run:
                with graph.inserting_after(after):
                    graph.call_function(self.overload, (base, rid))
            prod = extra.get("producer_target", target_name(base) if base.op == "call_function" else "placeholder")
            entry = IdEntry(
                id=rid, piece=piece, k=k, node=base.name, base_node=base.name, target=target_name(base) if base.op == "call_function" else base.op,
                role=role, shape=shape_of(base), dtype=str(base.meta["val"].dtype).replace("torch.", ""),
                module_path=module_path(base) or extra.get("producer_module_path", ""), module_stack=_stack_paths(base),
                original_aten=original_aten(base), producer_target=prod, producer_module_path=extra.get("producer_module_path", module_path(base)),
                consumers=extra.get("consumers", [{"node": u.name, "target": target_name(u), "module_path": module_path(u)} for u in base.users if u.op == "call_function"][:6]),
                inserted_after=after.name, stack_trace_tail=(base.meta.get("stack_trace") or "").strip().splitlines()[-1:][0] if base.meta.get("stack_trace") else "",
                extra={kk: v for kk, v in extra.items() if kk not in ("consumers",)},
            )
            self.table.entries[rid] = entry
            k += 1
        info.record_count = k
        if self.dump_graphs:
            try:
                self.table.graphs[piece] = graph.owning_module.print_readable(print_output=False) if graph.owning_module is not None else str(graph)
            except Exception as e:  # noqa: BLE001
                self.table.graphs[piece] = f"# print_readable failed: {e!r}\n{graph}"
        if not self.dry_run and k:
            graph.lint()


def _flatten(a: Any) -> list[Any]:
    if isinstance(a, (list, tuple)):
        out: list[Any] = []
        for x in a:
            out.extend(_flatten(x))
        return out
    return [a]


def _stack_paths(node: fx.Node) -> list[str]:
    st = node.meta.get("nn_module_stack") or {}
    out = []
    for _k, v in st.items():
        out.append(str(v[0] if isinstance(v, (tuple, list)) else v))
    return out


# --------------------------------------------------------------------------------------------------
# replay-process hooks (never loaded by production)
# --------------------------------------------------------------------------------------------------

_HOOKS: dict[str, Any] = {}


def install_hooks(pass_: RecordInsertionPass) -> list[str]:
    """(1) Subclass vLLM's ``PostGradPassManager`` so ``pass_`` runs after ``fix_functionalization``
    (i.e. on the final graph Inductor lowers) and is part of the Inductor cache uuid; (2) wrap
    ``PiecewiseCompileInterpreter.call_module`` to publish the piece index and the split submodule's
    placeholder/output names while its pieces are being compiled. Idempotent."""
    if _HOOKS:
        return list(_HOOKS)
    import vllm.compilation.backends as be
    from vllm.compilation.passes.inductor_pass import InductorPass
    from vllm.compilation.passes.pass_manager import PostGradPassManager

    class RecordingPassManager(PostGradPassManager):
        def __call__(self, graph: fx.Graph) -> None:
            super().__call__(graph)
            pass_(graph)

        def uuid(self) -> str:
            return InductorPass.hash_dict({"base": super().uuid(), "verity_record": pass_.uuid()})

    # VllmBackend resolves the class by qualname (``current_platform.get_pass_manager_cls()`` ->
    # ``vllm.compilation.passes.pass_manager.PostGradPassManager``) at engine build: patch the module
    # attribute it resolves, and the name imported into backends.py (isinstance check).
    import vllm.compilation.passes.pass_manager as pm

    pm.PostGradPassManager = RecordingPassManager
    be.PostGradPassManager = RecordingPassManager
    _HOOKS["vllm.compilation.passes.pass_manager.PostGradPassManager"] = "RecordingPassManager (runs RecordInsertionPass last; uuid extended)"

    orig_call_module = be.PiecewiseCompileInterpreter.call_module

    def call_module(self, target, args, kwargs):
        try:
            if target in self.compile_submod_names:
                submod = self.fetch_attr(target)
                ph = [n for n in submod.graph.nodes if n.op == "placeholder"]
                CURRENT_PIECE.update({
                    "index": self.compile_submod_names.index(target), "submod_name": target,
                    "placeholders": [n.name for n in ph],
                    "placeholder_shapes": [shape_of(n) if is_tensor_val(n) else repr(n.meta.get("example_value", n.meta.get("val"))) for n in ph],
                    "outputs": [a.name if isinstance(a, fx.Node) else repr(a) for a in _flatten(submod.graph.output_node().args)],
                })
            else:
                CURRENT_PIECE.update({"index": -1, "submod_name": None, "placeholders": [], "placeholder_shapes": [], "outputs": []})
        except Exception as e:  # noqa: BLE001
            CURRENT_PIECE.update({"index": -1, "submod_name": f"error:{e!r}"})
        try:
            return orig_call_module(self, target, args, kwargs)
        finally:
            CURRENT_PIECE.update({"index": -1, "submod_name": None})

    be.PiecewiseCompileInterpreter.call_module = call_module
    _HOOKS["vllm.compilation.backends.PiecewiseCompileInterpreter.call_module"] = "publishes piece index + split submodule placeholders/outputs"

    # runtime: which piece is executing (Inductor's FX-graph cache serves identical middle pieces from
    # one compiled artifact, so the record ids alone identify the template, not the layer)
    from vllm.compilation.piecewise_backend import PiecewiseBackend

    orig_pb_call = PiecewiseBackend.__call__

    def pb_call(self, *args):
        prev = record_op.RUNTIME["piece"]
        record_op.RUNTIME["piece"] = int(self.piecewise_compile_index)
        try:
            return orig_pb_call(self, *args)
        finally:
            record_op.RUNTIME["piece"] = prev

    PiecewiseBackend.__call__ = pb_call
    _HOOKS["vllm.compilation.piecewise_backend.PiecewiseBackend.__call__"] = "publishes the executing piece index to the record sink"
    return list(_HOOKS)
