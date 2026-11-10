"""An exported graph, guarded and translated: every accepted node has an explicit Verity translation (a minted op, a static
view or `getitem` plumbing) and everything else is a `FrontendError`; the guarded graph becomes a composite body."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from verity.ml.circuits.defs import Builder
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array, Type
from verity_vllm.program.frontend.errors import FrontendError
from verity_vllm.program.frontend.minted_ops import _OPINFO
from verity_vllm.program.frontend.strided_views import _dim_len, _expand, _norm_dim, _permute, _select, _slice, _stack
from verity_vllm.program.frontend.value_types import flat_types, join_coll, shape_of, split_coll, type_of_tensor, width_of


@dataclass
class _Spec:
    param_types: tuple[Type, ...]
    ret_type: Type
    ep: Any  # torch.export.ExportedProgram
    erased: tuple  # key of the authoring-level (non-static) bindings that shaped this body
    accepted: set = field(default_factory=set)  # other erased keys verified to produce this same body


aten = torch.ops.aten
_IDENTITY = {aten.alias.default, aten.detach.default, aten.clone.default, aten.contiguous.default, aten.lift_fresh.default}
_RESHAPE = {aten.view.default, aten.reshape.default, aten._unsafe_view.default, aten.flatten.using_ints, aten.squeeze.dim,
            aten.squeeze.default, aten.squeeze.dims, aten.unsqueeze.default}
_VIEWS = _IDENTITY | _RESHAPE | {aten.slice.Tensor, aten.select.int, aten.cat.default, aten.stack.default, aten.expand.default,
                                  aten.permute.default, aten.t.default, aten.transpose.int, aten.unbind.int, aten.view.dtype}


def _guard(ep, where: str) -> None:
    sig = ep.graph_signature
    hidden = dict(sig.inputs_to_parameters) | dict(sig.inputs_to_buffers) | dict(sig.inputs_to_lifted_tensor_constants)
    if hidden:
        raise FrontendError("E_HIDDEN_STATE", f"lifted parameters/buffers/constants {sorted(hidden.values())}: pass them as "
                                              f"explicit inputs (they are Input gates) or author constants with ops.const", where)
    if sig.user_inputs_to_mutate or sig.buffers_to_mutate:
        raise FrontendError("E_MUTATION", f"export recorded mutations {dict(sig.user_inputs_to_mutate)} {dict(sig.buffers_to_mutate)}", where)
    for node in ep.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        if node.op == "get_attr":
            raise FrontendError("E_DYNAMIC_CONTROL_FLOW", f"subgraph attribute {node.target} (higher-order op body)", where)
        if node.op != "call_function":
            raise FrontendError("E_UNREGISTERED_OP", f"{node.op} {node.target}", where)
        t = node.target
        if t is operator.getitem:
            continue
        if isinstance(t, torch._ops.HigherOrderOperator):
            raise FrontendError("E_DYNAMIC_CONTROL_FLOW", f"higher-order op {t}: data-dependent control flow has no Verity meaning", where)
        if isinstance(t, torch._ops.OpOverload):
            if t.namespace == "verity":
                if t.__name__.split(".")[0] not in _OPINFO:
                    raise FrontendError("E_UNREGISTERED_OP", f"{t} was not minted by this frontend", where)
                continue
            if t._schema.is_mutable:
                raise FrontendError("E_MUTATION", f"{t} mutates its operand; Verity bodies are pure dataflow", where)
            if t in _VIEWS:
                continue
            if t.namespace == "aten":
                raise FrontendError("E_UNREGISTERED_OP", f"ATen op {t} has no Verity meaning; call a registered function through ops.*", where)
            raise FrontendError("E_UNREGISTERED_OP", f"custom op {t} has no Verity binding", where)
        raise FrontendError("E_UNREGISTERED_OP", f"Python callable {t!r} in the graph", where)


def _translate(spec: _Spec, B: Builder, params: Sequence[Coll], where: str) -> Coll:
    ep = spec.ep
    env: dict[Any, Any] = {}
    flat_params = [c for p in params for c in split_coll(p)]
    user_inputs = list(ep.graph_signature.user_inputs)
    if len(flat_params) != len(user_inputs):
        raise FrontendError("E_SHAPE", f"{len(flat_params)} parameter tensors vs {len(user_inputs)} graph inputs", where)
    pos = {name: i for i, name in enumerate(user_inputs)}

    def val(a):
        if isinstance(a, torch.fx.Node):
            return env[a]
        if isinstance(a, (list, tuple)):
            return [val(x) for x in a]
        return a

    def meta_type(node) -> Type:
        return type_of_tensor(node.meta["val"])

    out_node = None
    for node in ep.graph.nodes:
        if node.op == "placeholder":
            env[node] = flat_params[pos[node.name]]
        elif node.op == "output":
            out_node = node
        else:
            t = node.target
            if t is operator.getitem:
                env[node] = val(node.args[0])[node.args[1]]
            elif isinstance(t, torch._ops.OpOverload) and t.namespace == "verity":
                info = _OPINFO[t.__name__.split(".")[0]]
                flats = val(node.args[0])
                name = node.args[1] or None
                colls, i = [], 0
                for at in info.arg_types:
                    k = len(flat_types(at))
                    colls.append(join_coll(at, flats[i:i + k]))
                    i += k
                if info.kind == "call":
                    out = B.call(info.fn, *colls, name=name)
                elif info.kind == "batch":
                    out = B.batch(info.fn, *colls, axes=info.axes, n=info.n, name=name)
                else:
                    out = B.scan(info.fn, colls[0], xs=colls[1:1 + info.n_xs], shared=colls[1 + info.n_xs:], name=name)
                env[node] = split_coll(out)
            elif t in _IDENTITY:
                env[node] = val(node.args[0])
            elif t in _RESHAPE:
                x = val(node.args[0])
                try:
                    env[node] = x.reshape(meta_type(node))
                except ValueError as e:
                    raise FrontendError("E_SHAPE", str(e), where) from None
            elif t is aten.view.dtype:
                x = val(node.args[0])
                if width_of(node.args[1], where) != shape_of(x.type)[1]:
                    raise FrontendError("E_WIDTH", f"view.dtype changes the width ({x.type!r} -> {node.args[1]})", where)
                env[node] = x
            elif t is aten.slice.Tensor:
                x = val(node.args[0])
                dim = _norm_dim(node.args[1] if len(node.args) > 1 else 0, x.type)
                lo = node.args[2] if len(node.args) > 2 else None
                hi = node.args[3] if len(node.args) > 3 else None
                step = node.args[4] if len(node.args) > 4 else 1
                env[node] = _slice(x, dim, lo, hi, step, where)
            elif t is aten.select.int:
                x = val(node.args[0])
                dim = _norm_dim(node.args[1], x.type)
                i = node.args[2]
                i = i + _dim_len(x.type, dim) if i < 0 else i
                env[node] = _select(x, dim, i, where)
            elif t in (aten.cat.default, aten.stack.default):
                parts = val(node.args[0])
                dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
                if dim < 0:
                    dim += len(shape_of(meta_type(node))[0])
                if dim != 0:
                    raise FrontendError("E_UNSUPPORTED_VIEW", f"{t} along dim {dim}: only the leading axis (dim 0) is a view", where)
                if t is aten.stack.default:
                    env[node] = _stack(parts, where)
                else:
                    elem = parts[0].type.elem
                    if any(p.type.elem != elem for p in parts):
                        raise FrontendError("E_SHAPE", "cat of arrays with different element types", where)
                    env[node] = Coll(Array(sum(p.type.n for p in parts), elem), concat([p.refs for p in parts]))
            elif t in (aten.permute.default, aten.t.default, aten.transpose.int):
                x = val(node.args[0])
                nd = len(shape_of(x.type)[0])
                if t is aten.permute.default:
                    dims = list(node.args[1])
                elif t is aten.t.default:
                    dims = [1, 0] if nd == 2 else list(range(nd))
                else:
                    d0, d1 = (_norm_dim(node.args[1], x.type), _norm_dim(node.args[2], x.type))
                    dims = list(range(nd))
                    dims[d0], dims[d1] = dims[d1], dims[d0]
                env[node] = _permute(x, dims, where)
            elif t is aten.unbind.int:
                x = val(node.args[0])
                dim = _norm_dim(node.args[1] if len(node.args) > 1 else 0, x.type)
                env[node] = [_select(x, dim, i, where) for i in range(_dim_len(x.type, dim))]
            elif t is aten.expand.default:
                x = val(node.args[0])
                env[node] = _expand(x, list(node.args[1]), where)
            else:  # unreachable after _guard
                raise FrontendError("E_UNREGISTERED_OP", f"{t}", where)
    flats = [val(a) for a in out_node.args[0]]
    for f in flats:
        if not isinstance(f, Coll):
            raise FrontendError("E_RETURN_TYPE", "a returned object is not a value collection", where)
    return join_coll(spec.ret_type, flats)
