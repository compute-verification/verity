"""The structure a training-efficiency score reads, from one captured step graph, in multiply-accumulates (MACs).

Every matmul-class node (``mm``, ``addmm``, ``bmm``, ``baddbmm``, ``addbmm``, ``mv``, ``addmv``, ``dot``, ``addr``,
``linear``, ``matmul``, ``convolution`` and ``convolution_backward``'s two GEMMs, ``_scaled_mm``, ``_int_mm``) is one
`Matmul` per GEMM it runs, with dims (batch, m, k, n), MACs = batch·m·k·n, and which token axes index each dim (from
the varied captures, `trace`). A matmul-like op outside that list is refused. `macs` is the GEMMs' MACs. The rest of
the step's share of |C| is counted by te_core.flops's rule, so `c_macs` matches the harness's count: one MAC per output
element of each elementwise multiplication (`elementwise_macs`: ``mul``, ``div``, ``addcmul``, ``addcdiv``, ``lerp``,
``add`` and ``sub`` with a scale, their in-place and ``_foreach_`` forms, fused optimizer steps at their unfused forms'
multiplications), and dense linear algebra at n³ per matrix (`approx_macs`). Any other forward op that combines an
activation with an updated operand (a norm with a trained gain, an add then ``exp``) is an updated node as in
te_core.flops, at one MAC per element of its largest operand or result (only its indices and result, for a gather):
a using `Matmul` of part ``elementwise`` (`using_elementwise`), with k its MACs per token, which counts toward the
upstream MACs of the GEMMs it reaches but not toward their `upstream` counts.

Terms are the training ontology's:

- A GEMM's result is **token-indexed** when one of its batch, m or n dims is indexed by a token axis, else
  **parameter-indexed** (a weight-shaped sum over tokens, as dW).
- Each operand's class is its provenance. Through views, casts and scalings by a constant, an operand that is a
  parameter is **updated** if its name is in `updated` (it differs from θ when read: the harness passes the set) and
  **registered** otherwise, as is every buffer. Any other operand is a **gradient** when it is computed from the backward,
  an **activation** when it is token-indexed (an embedding's rows gathered by token ids are, as in te_core.flops), and
  when it is weight-shaped, **updated** or
  **registered** by the parameters it is computed from (a merged LoRA weight, MeZO's perturbed copy). The GEMM's
  **weight** is its highest operand class in the order updated, registered, gradient, activation, constant.
- **Using**: a token-indexed result with an updated operand (the forward's and the backward's dX GEMMs against an updated
  weight). **Producing**: a parameter-indexed result that reaches the final value of a parameter the step writes through
  weight-shaped values only (dW into the optimizer's state and the parameter; a matrix optimizer's own products).
  Every other GEMM is **neither**.
- **Upstream**: for each GEMM, the using GEMMs whose results reach its operands (`upstream`, of them those in the forward
  `upstream_fwd`), and their MACs per token (`upstream_macs`), the depth D reads. The count is at node granularity: a
  GEMM counts every MAC of an upstream using GEMM, which is exact where a token's row is mixed between them (a norm, a
  softmax, a contraction over the row) and an upper bound otherwise (an up projection feeding its own layer's down
  projection through an elementwise activation).
- **φ(D)**: the share of the step's MACs in using GEMMs with at least D upstream (`phi`).

Dataflow follows storage: an in-place op (``add_``, a foreach optimizer kernel) makes a new version of the storage it
writes, every view of it reads that version, and each element of a foreach list is its own value.
"""
from __future__ import annotations

import hashlib
import json
import operator
from dataclasses import dataclass, field
from fractions import Fraction
from math import prod
from typing import Any

import torch

from te_core.capture.trace import BACKWARD, PHASE, CaptureRefused, Trace
from te_core.flops import (ELEMENTWISE_NAMES, FUSED_OPTIMIZER_MACS, GATHERS, GEMM_LIKE, LINALG_NAMES, NOT_GEMM, SCALED_NAMES,
                           TOKEN_INDEX, _is_view, gather_macs)

__all__ = ["CLASSES", "Matmul", "StepGraph"]

aten = torch.ops.aten
#: operand classes, highest first: a GEMM's weight is its highest
CLASSES = ("updated", "registered", "gradient", "activation", "constant")
_RANK = {c: i for i, c in enumerate(CLASSES)}
#: ops an operand's provenance walks through to its first argument: views and casts
_WALK = {"view", "_unsafe_view", "reshape", "t", "transpose", "permute", "expand", "alias", "detach", "clone", "contiguous",
         "_to_copy", "to", "lift_fresh_copy", "lift_fresh", "unsqueeze", "squeeze", "select", "slice", "as_strided",
         "unflatten", "flatten", "split", "split_with_sizes", "chunk", "unbind", "narrow", "numpy_T", "mT", "adjoint", "conj", "resolve_conj", "resolve_neg", "type_as", "view_as", "expand_as",
         "repeat", "squeeze_copy", "unsqueeze_copy", "view_copy", "t_copy", "transpose_copy", "permute_copy",
         "expand_copy", "_reshape_alias"}
#: elementwise ops walked through when their only tensor argument is the first (a scaling by a Python constant)
_SCALE = {"mul", "div", "neg", "add", "sub"}
#: ops that read only their argument's shape
_SHAPE_ONLY = {"ones_like", "zeros_like", "empty_like", "full_like", "new_zeros", "new_ones", "new_empty", "new_full",
               "sym_size", "sym_numel", "sym_stride", "empty_strided"}
#: matmul-like names that `_gemms` must account for (te_core.flops's); any other op with one of these in its name is refused
_GEMM_LIKE, _NOT_GEMM = GEMM_LIKE, NOT_GEMM
_ELEMENTWISE, _SCALED, _FUSED, _LINALG = ELEMENTWISE_NAMES, SCALED_NAMES, FUSED_OPTIMIZER_MACS, LINALG_NAMES
_TOKEN_INDEX = {p.__name__ for p in TOKEN_INDEX}
_GATHERS = {p.__name__ for p in GATHERS}


def _pk(t) -> str:
    return t.overloadpacket.__name__ if isinstance(t, torch._ops.OpOverload) else ""


def _shape(v) -> tuple[int, ...] | None:
    return tuple(int(s) for s in v.shape) if isinstance(v, torch.Tensor) else None


def _sh(a) -> tuple[int, ...]:
    return _shape(a.meta["val"])


@dataclass(frozen=True)
class _G:
    """One GEMM of a node: its part name, dims, operand nodes (A, B, then the accumulator if any) and which of the
    node's outputs is its result (None: the node's one output)."""
    part: str
    dims: dict
    operands: tuple
    out: int | None


def _gemms(node) -> list[_G]:
    t = node.target
    if node.op != "call_function" or not isinstance(t, torch._ops.OpOverload):
        return []
    name, a = _pk(t), node.args
    if name in ("mm", "_int_mm", "_scaled_mm"):
        (M, K), (_, N) = _sh(a[0]), _sh(a[1])
        return [_G("", {"batch": 1, "m": M, "k": K, "n": N}, (a[0], a[1]), None)]
    if name in ("addmm", "addmm_"):
        (M, K), (_, N) = _sh(a[1]), _sh(a[2])
        return [_G("", {"batch": 1, "m": M, "k": K, "n": N}, (a[1], a[2], a[0]), None)]
    if name == "bmm":
        (B, M, K), (_, _, N) = _sh(a[0]), _sh(a[1])
        return [_G("", {"batch": B, "m": M, "k": K, "n": N}, (a[0], a[1]), None)]
    if name in ("baddbmm", "baddbmm_"):
        (B, M, K), (_, _, N) = _sh(a[1]), _sh(a[2])
        return [_G("", {"batch": B, "m": M, "k": K, "n": N}, (a[1], a[2], a[0]), None)]
    if name in ("addbmm", "addbmm_"):
        (B, M, K), (_, _, N) = _sh(a[1]), _sh(a[2])
        return [_G("", {"batch": 1, "m": M, "k": B * K, "n": N}, (a[1], a[2], a[0]), None)]
    if name == "mv":
        (M, K) = _sh(a[0])
        return [_G("", {"batch": 1, "m": M, "k": K, "n": 1}, (a[0], a[1]), None)]
    if name in ("addmv", "addmv_"):
        (M, K) = _sh(a[1])
        return [_G("", {"batch": 1, "m": M, "k": K, "n": 1}, (a[1], a[2], a[0]), None)]
    if name in ("dot", "vdot"):
        return [_G("", {"batch": 1, "m": 1, "k": _sh(a[0])[0], "n": 1}, (a[0], a[1]), None)]
    if name in ("addr", "addr_"):
        return [_G("", {"batch": 1, "m": _sh(a[1])[0], "k": 1, "n": _sh(a[2])[0]}, (a[1], a[2], a[0]), None)]
    if name == "linear":
        x, w = _sh(a[0]), _sh(a[1])
        ops = (a[0], a[1]) + ((a[2],) if len(a) > 2 and a[2] is not None else ())
        return [_G("", {"batch": 1, "m": prod(x[:-1]), "k": x[-1], "n": w[0]}, ops, None)]
    if name == "matmul":
        x, w, out = _sh(a[0]), _sh(a[1]), _shape(node.meta["val"])
        K = x[-1]
        N = w[-1] if len(w) >= 2 else 1
        return [_G("", {"batch": 1, "m": prod(out) // N, "k": K, "n": N}, (a[0], a[1]), None)]
    if name in ("convolution", "_convolution"):
        return [_conv(_sh(a[0]), _sh(a[1]), _shape(node.meta["val"]), bool(a[6]), int(a[8]), (a[0], a[1]), "", None)]
    if name == "convolution_backward":
        go, x, w = _sh(a[0]), _sh(a[1]), _sh(a[2])
        transposed, groups, mask = bool(a[7]), int(a[9]), list(a[10])
        out = []
        if mask[0]:
            out.append(_conv_bwd_input(go, x, w, transposed, groups, (a[0], a[2]), 0))
        if mask[1]:
            out.append(_conv_bwd_weight(go, x, w, transposed, groups, (a[0], a[1]), 1))
        return out
    if any(s in name for s in _GEMM_LIKE) and name not in _NOT_GEMM and t.namespace == "aten":
        raise CaptureRefused("uncounted_op", str(t), f"{t} at node {node.name} computes products this package doesn't count; "
                                                     f"capture with a decomposition of it")
    return []


def _conv(x, w, out, transposed, groups, operands, part, idx) -> _G:
    kern = prod(w[2:])
    if not transposed:
        return _G(part, {"batch": 1, "m": out[0] * prod(out[2:]), "k": w[1] * kern, "n": w[0]}, operands, idx)
    return _G(part, {"batch": 1, "m": x[0] * prod(x[2:]), "k": x[1], "n": w[1] * kern}, operands, idx)


def _conv_bwd_input(go, x, w, transposed, groups, operands, idx) -> _G:
    kern = prod(w[2:])
    if not transposed:
        return _G("grad_input", {"batch": 1, "m": go[0] * prod(go[2:]), "k": w[0], "n": w[1] * kern}, operands, idx)
    return _G("grad_input", {"batch": 1, "m": x[0] * prod(x[2:]), "k": w[1] * kern, "n": x[1]}, operands, idx)


def _conv_bwd_weight(go, x, w, transposed, groups, operands, idx) -> _G:
    kern = prod(w[2:])
    if not transposed:
        return _G("grad_weight", {"batch": 1, "m": w[0], "k": go[0] * prod(go[2:]), "n": w[1] * kern}, operands, idx)
    return _G("grad_weight", {"batch": 1, "m": w[0], "k": x[0] * prod(x[2:]), "n": w[1] * kern}, operands, idx)


def _macs(d: dict) -> int:
    return d["batch"] * d["m"] * d["k"] * d["n"]


def _numel(x) -> int:
    xs = x if isinstance(x, (list, tuple)) else [x]
    return sum(v.numel() for v in (a.meta.get("val") if isinstance(a, torch.fx.Node) else a for a in xs)
               if isinstance(v, torch.Tensor))


def _other_macs(node) -> tuple[int, str]:
    """A non-GEMM node's MACs by te_core.flops's rule (module docstring): (MACs, "elementwise" | "approx" | "")."""
    if node.op != "call_function" or not isinstance(node.target, torch._ops.OpOverload):
        return 0, ""
    name, a = _pk(node.target), node.args
    if name in _ELEMENTWISE or name in _SCALED and node.kwargs.get("alpha", 1) != 1:
        return _numel(node.meta.get("val")) or _numel(a[0]), "elementwise"
    if name in _FUSED:
        return _FUSED[name] * _numel(a[0]), "elementwise"
    if name in _LINALG:
        s = _sh(a[0])
        n = s[-1] if s else 1
        return (prod(s) // max(n, 1)) * n * n, "approx"
    return 0, ""


def _work(node) -> int:
    """A non-GEMM node's work by te_core.flops's rule (`flops._work`): one per element of its largest operand or result,
    of its index operands and result for a gather."""
    name = _pk(node.target)
    args = list(node.args[1:]) + list(node.kwargs.values()) if name in _TOKEN_INDEX else list(node.args) + list(node.kwargs.values())
    vals = [a.meta.get("val") for a in _arg_nodes_of(args)]
    if name not in _TOKEN_INDEX or name in _GATHERS:
        out = node.meta.get("val")
        vals += list(out) if isinstance(out, (list, tuple)) else [out]
    return max((v.numel() for v in vals if isinstance(v, torch.Tensor)), default=0)


def _arg_nodes_of(args) -> list:
    out = []
    for a in args:
        if isinstance(a, torch.fx.Node):
            out.append(a)
        elif isinstance(a, (list, tuple)):
            out.extend(x for x in a if isinstance(x, torch.fx.Node))
    return out


def _width(dtype: str) -> int:
    return getattr(torch, dtype).itemsize * 8


@dataclass(frozen=True)
class Matmul:
    """One GEMM of the step (module docstring)."""
    node: str
    op: str
    part: str
    phase: str
    macs: int
    dims: dict
    token_dims: dict
    operands: tuple[str, ...]
    params: tuple[str | None, ...]
    dtypes: tuple[str, ...]
    weight: str
    result: str
    role: str
    per_token: Fraction
    upstream: int
    upstream_fwd: int
    upstream_macs: Fraction

    def to_json(self) -> dict:
        d = {k: getattr(self, k) for k in ("node", "op", "part", "phase", "macs", "dims", "operands", "weight", "result", "role",
                                           "upstream", "upstream_fwd")}
        d["token_dims"] = {k: list(v) for k, v in self.token_dims.items()}
        d["params"] = list(self.params)
        d["dtypes"] = list(self.dtypes)
        d["per_token"] = str(self.per_token)
        d["upstream_macs"] = str(self.upstream_macs)
        return d


def _phases(gm) -> dict:
    """Each node's phase from the `record_function` markers around it (module docstring of `trace`)."""
    out, stack = {}, []
    for n in gm.graph.nodes:
        name = _pk(n.target) if n.op == "call_function" else ""
        if name == "_record_function_enter_new":
            stack.append((n, str(n.args[0])))
        elif name == "_record_function_exit":
            while stack and stack[-1][0] is not n.args[0]:
                stack.pop()
            if stack:
                stack.pop()
        ph = "forward"
        for _, m in reversed(stack):
            if m == BACKWARD:
                ph = "backward"
                break
            if m.startswith("Optimizer."):
                ph = "optimizer"
                break
            if m.startswith(PHASE):
                ph = m[len(PHASE):]
                break
        out[n] = ph
    return out


class _Flow:
    """Values of the graph with their direct dependences, storage-versioned (module docstring)."""

    def __init__(self, gm, phases: dict, tok) -> None:
        self.preds: list[tuple[int, ...]] = []
        self.phase: list[str] = []
        self.tok: list[bool] = []
        self.source: dict[int, tuple[str, str]] = {}   # value -> ("attr", target) | ("input", name): a graph source
        self.val: dict[Any, Any] = {}
        self.base: dict[Any, Any] = {}
        self.cur: dict[Any, int] = {}
        self.written: set = set()
        self.reads: dict[tuple[Any, Any], frozenset[int]] = {}   # (node, argument node) -> the values it read when it ran
        for n in gm.graph.nodes:
            if n.op == "call_function":
                for a in _arg_nodes(n):
                    self.reads[(n, a)] = frozenset(self._rd(a))
            self._node(n, phases.get(n, "forward"), tok)

    def _new(self, preds, phase: str, tok: bool) -> int:
        self.preds.append(tuple(sorted(set(preds))))
        self.phase.append(phase)
        self.tok.append(tok)
        return len(self.preds) - 1

    def _rd(self, a) -> set[int]:
        out = set()
        if isinstance(a, torch.fx.Node):
            v = self.val.get(a)
            if isinstance(v, int):
                out.add(v)
            elif isinstance(v, list):
                out.update(x for x in v if isinstance(x, int))
            b = self.base.get(a)
            for k in (b if isinstance(b, list) else [b]):
                if k is not None and k in self.cur:
                    out.add(self.cur[k])
        return out

    def _node(self, n, ph: str, tok) -> None:
        if n.op == "placeholder":
            v = self._new((), ph, tok(n, None))
            self.val[n], self.base[n] = v, ("input", n.name)
            self.cur[("input", n.name)] = v
            self.source[v] = ("input", n.name)
            return
        if n.op == "get_attr":
            key = ("attr", n.target)
            if key not in self.cur:
                v = self._new((), ph, tok(n, None))
                self.cur[key] = v
                self.source[v] = key
            self.val[n], self.base[n] = self.cur[key], key
            return
        if n.op != "call_function":
            return
        t = n.target
        if t is operator.getitem:
            src, i = n.args
            sv = self.val.get(src)
            if isinstance(sv, list):
                self.val[n] = sv[i]
                bases = self.base.get(src)
                self.base[n] = bases[i] if isinstance(bases, list) else bases
            else:
                self.val[n], self.base[n] = sv, self.base.get(src)
            return
        name = _pk(t)
        if name in ("_record_function_enter_new", "_record_function_exit"):
            return
        schema = getattr(t, "_schema", None)
        flat = [(i, a) for i, a in enumerate(n.args)] + [(k, a) for k, a in n.kwargs.items()]
        written_pos = set()
        if schema is not None:
            for i, arg in enumerate(schema.arguments):
                if arg.alias_info is not None and arg.alias_info.is_write:
                    written_pos.add(i if not arg.kwarg_only else arg.name)
        lists = [(p, a) for p, a in flat if isinstance(a, (list, tuple)) and any(isinstance(x, torch.fx.Node) for x in a)]
        singles = [(p, a) for p, a in flat if isinstance(a, torch.fx.Node)]
        shape_only = name in _SHAPE_ONLY
        if lists and any(len(a) > 1 for _, a in lists) and name.startswith("_foreach"):
            k = max(len(a) for _, a in lists)
            common = set()
            for _, a in singles:
                common |= self._rd(a)
            reads = [set(common) for _ in range(k)]
            for _, a in lists:
                if len(a) == k:
                    for i, x in enumerate(a):
                        reads[i] |= self._rd(x)
                else:
                    for x in a:
                        for r in reads:
                            r |= self._rd(x)
            mutated = [a for p, a in lists if p in written_pos]
            if mutated:
                for m in mutated:
                    for i, x in enumerate(m):
                        v = self._new(reads[i], ph, tok(x, None))
                        b = self.base.get(x, ("node", x.name))
                        self.cur[b] = v
                        self.written.add(b)
                self.val[n] = None
                return
            vals = [self._new(reads[i], ph, tok(n, i)) for i in range(k)]
            self.val[n] = vals
            self.base[n] = [("node", n.name, i) for i in range(k)]
            for i, v in enumerate(vals):
                self.cur[("node", n.name, i)] = v
            return
        reads = set()
        if not shape_only:
            for _, a in singles:
                reads |= self._rd(a)
            for _, a in lists:
                for x in a:
                    reads |= self._rd(x)
        mutated = [a for p, a in singles if p in written_pos]
        if mutated:
            first = None
            for a in mutated:
                v = self._new(reads, ph, tok(a, None))
                b = self.base.get(a, ("node", a.name))
                self.cur[b] = v
                self.written.add(b)
                first = (v, b) if first is None else first
            self.val[n], self.base[n] = first
            return
        ret = schema.returns if schema is not None else ()
        is_view = bool(ret) and ret[0].alias_info is not None and not ret[0].alias_info.is_write and singles
        out = n.meta.get("val")
        if isinstance(out, (list, tuple)):
            vals = [self._new(reads, ph, tok(n, i)) if isinstance(o, torch.Tensor) else None for i, o in enumerate(out)]
            self.val[n] = vals
            if is_view:
                self.base[n] = [self.base.get(singles[0][1], ("node", singles[0][1].name))] * len(vals)
            else:
                self.base[n] = [("node", n.name, i) for i in range(len(vals))]
                for i, v in enumerate(vals):
                    if v is not None:
                        self.cur[("node", n.name, i)] = v
            return
        v = self._new(reads, ph, tok(n, None))
        self.val[n] = v
        if is_view:
            self.base[n] = self.base.get(singles[0][1], ("node", singles[0][1].name))
        else:
            self.base[n] = ("node", n.name)
            self.cur[("node", n.name)] = v


def _arg_nodes(n) -> list:
    out = []
    for a in list(n.args) + list(n.kwargs.values()):
        if isinstance(a, torch.fx.Node):
            out.append(a)
        elif isinstance(a, (list, tuple)):
            out.extend(x for x in a if isinstance(x, torch.fx.Node))
    return out


def _closure(flow: _Flow, seed) -> list[bool]:
    out = []
    for v, ps in enumerate(flow.preds):
        out.append(bool(seed(v)) or any(out[p] for p in ps))
    return out


@dataclass
class StepGraph:
    """One captured training step's structure (module docstring); `graph` is the base capture's GraphModule."""
    name: str
    tokens: int
    axes: dict
    matmuls: list[Matmul]
    params: list[dict]
    step_crossing: list[dict]
    updated: frozenset
    trained: frozenset
    graph: Any = field(repr=False)
    digest: str = ""
    elementwise_macs: int = 0
    approx_macs: int = 0
    #: the forward's non-GEMM using nodes (an activation combined with an updated operand), as te_core.flops counts them
    using_elementwise: list[Matmul] = field(default_factory=list)
    _inputs: Any = field(default=None, repr=False)

    # -- building ---------------------------------------------------------------------------------------------------
    @classmethod
    def build(cls, base: Trace, varied: list[Trace], axis_names: tuple[str, ...], *, updated, name: str) -> "StepGraph":
        nodes = list(base.gm.graph.nodes)
        vnodes = [list(t.gm.graph.nodes) for t in varied]
        for t, vn in zip(varied, vnodes):
            if len(vn) != len(nodes) or any((a.op, str(a.target)) != (b.op, str(b.target)) for a, b in zip(nodes, vn)):
                i = next((i for i, (a, b) in enumerate(zip(nodes, vn)) if (a.op, str(a.target)) != (b.op, str(b.target))),
                         min(len(nodes), len(vn)))
                what = f"node {i} is {nodes[i].target if i < len(nodes) else 'absent'} at {base.axis_sizes} and " \
                       f"{vn[i].target if i < len(vn) else 'absent'} at {t.axis_sizes}"
                raise CaptureRefused("batch_dependent", str(nodes[i].target) if i < len(nodes) else None,
                                     f"the step's graph changes with the token axes ({what}); pass token_dims naming "
                                     f"the axes it doesn't branch on")
        return cls._analyze(base, varied, axis_names, updated, name)

    @classmethod
    def _analyze(cls, base: Trace, varied: list[Trace], axis_names, updated, name) -> "StepGraph":
        gm = base.gm
        nodes = list(gm.graph.nodes)
        pos = {n: i for i, n in enumerate(nodes)}
        vnodes = [list(t.gm.graph.nodes) for t in varied]

        def vshape(n, idx, vs):
            v = (vs[pos[n]] if vs is not None else n).meta.get("val")
            if idx is not None:
                v = v[idx] if isinstance(v, (list, tuple)) and idx < len(v) else None
            return _shape(v)

        def tok(n, idx) -> bool:
            s = vshape(n, idx, None)
            return s is not None and any(vshape(n, idx, vs) != s for vs in vnodes)

        phases = _phases(gm)
        flow = _Flow(gm, phases, tok)
        attrs = base.attrs
        params = {t: v for t, v in attrs.items() if v[0] == "parameter"}
        trained = frozenset(params[k[1]][1] for k in flow.written if k[0] == "attr" and k[1] in params)
        upd = trained if updated is None else frozenset(updated)
        seed = _closure(flow, lambda v: flow.phase[v] == "backward")
        dep_upd = _closure(flow, lambda v: flow.source.get(v, ("", ""))[0] == "attr" and
                           flow.source[v][1] in params and params[flow.source[v][1]][1] in upd)
        dep_par = _closure(flow, lambda v: flow.source.get(v, ("", ""))[0] == "attr" and flow.source[v][1] in attrs)
        tokens = prod(base.axis_sizes) if base.axis_sizes else 1

        def walk(a):
            seen = 0
            while isinstance(a, torch.fx.Node) and a.op == "call_function" and seen < 10_000:
                seen += 1
                if a.target is operator.getitem:
                    a = a.args[0]
                    continue
                nm = _pk(a.target)
                if nm in _WALK:
                    a = a.args[0]
                    continue
                if nm in _SCALE and sum(isinstance(x, torch.fx.Node) for x in list(a.args) + list(a.kwargs.values())) == 1 \
                        and isinstance(a.args[0], torch.fx.Node):
                    a = a.args[0]
                    continue
                break
            return a

        def klass(n, a) -> tuple[str, str | None]:
            src = walk(a)
            if isinstance(src, torch.fx.Node) and src.op == "get_attr":
                hit = attrs.get(src.target)
                if hit is None:
                    return "constant", None
                if hit[0] == "parameter":
                    return ("updated" if hit[1] in upd else "registered"), hit[1]
                return "registered", hit[1]
            vs = flow.reads.get((n, a), frozenset())
            if not vs:
                return "constant", None
            if any(seed[x] for x in vs):
                return "gradient", None
            if any(flow.tok[x] for x in vs):
                return "activation", None
            if any(dep_upd[x] for x in vs):
                return "updated", None
            if any(dep_par[x] for x in vs):
                return "registered", None
            return "constant", None

        # GEMMs and their dims in every capture
        raw = []
        for n in nodes:
            gs = _gemms(n)
            if not gs:
                continue
            vgs = [_gemms(vs[pos[n]]) for vs in vnodes]
            for j, g in enumerate(gs):
                tdims = {}
                for d, size in g.dims.items():
                    axes = tuple(axis_names[a] for a, vg in enumerate(vgs) if vg[j].dims[d] != size)
                    if axes:
                        tdims[d] = axes
                raw.append((n, g, tdims))

        def result_value(n, g) -> int | None:
            v = flow.val.get(n)
            if isinstance(v, list):
                return v[g.out] if g.out is not None and g.out < len(v) else None
            return v

        # producing: parameter-indexed results reaching a written parameter's final value through weight-shaped values
        finals = [flow.cur[k] for k in flow.written if k[0] == "attr" and k[1] in params]
        reach: set[int] = set()
        stack = list(finals)
        while stack:
            v = stack.pop()
            if v in reach:
                continue
            reach.add(v)
            stack.extend(p for p in flow.preds[v] if not flow.tok[p] and p not in reach)

        recs = []
        for n, g, tdims in raw:
            cls_ = [klass(n, a) for a in g.operands if isinstance(a, torch.fx.Node)]
            classes = tuple(c for c, _ in cls_)
            names = tuple(p for _, p in cls_)
            result = "token" if any(d in tdims for d in ("batch", "m", "n")) else "parameter"
            rv = result_value(n, g)
            if result == "token" and "updated" in classes:
                role = "using"
            elif result == "parameter" and rv is not None and rv in reach:
                role = "producing"
            else:
                role = "neither"
            weight = min(classes, key=lambda c: _RANK[c]) if classes else "constant"
            recs.append([n, g, tdims, classes, names, result, role, weight, rv])

        # a forward node that combines an activation with an updated operand is a using node whatever op it is
        # (te_core.flops): at its elementwise multiplications, else its work (`_work`); a gather's source isn't combined,
        # but an updated source gathered by token data is the circuit's gather (`gather_macs`)
        extra = 0
        for n in nodes:
            if (n.op != "call_function" or not isinstance(n.target, torch._ops.OpOverload) or _gemms(n)
                    or phases.get(n, "forward") != "forward" or _is_view(n.target)
                    or _pk(n.target) in ("_record_function_enter_new", "_record_function_exit")):
                continue
            ops = _arg_nodes(n)
            src = set(_arg_nodes_of(n.args[:1])) if _pk(n.target) in _GATHERS else set()
            cls_ = [klass(n, a) for a in ops]
            mixed = [c for a, (c, _) in zip(ops, cls_) if a not in src]
            source = [c for a, (c, _) in zip(ops, cls_) if a in src]
            if "updated" in source and "activation" in mixed:
                # an updated table read by token data: the circuit's gather of its rows (te_core.flops, gather_macs)
                k = gather_macs(_pk(n.target), n.args[0].meta["val"].shape, n.args[1] if len(n.args) > 1 else None,
                                n.meta["val"].numel())
                extra += k
            elif "activation" in mixed and "updated" in mixed:
                k, kind = _other_macs(n)
                if not kind:
                    k = _work(n)
                    extra += k
            else:
                continue
            if not k:
                continue
            out = n.meta.get("val")
            g = _G("elementwise", {"batch": 1, "m": k, "k": 1, "n": 1}, tuple(ops), 0 if isinstance(out, (list, tuple)) else None)
            classes = tuple(c for c, _ in cls_)
            recs.append([n, g, {}, classes, tuple(p for _, p in cls_), "token", "using",
                         min(classes, key=lambda c: _RANK[c]), result_value(n, g)])

        # upstream using GEMMs, as bitsets over the using records
        using_idx = [i for i, r in enumerate(recs) if r[6] == "using"]
        own: dict[int, int] = {}
        for bit, i in enumerate(using_idx):
            rv = recs[i][8]
            if rv is not None:
                own[rv] = own.get(rv, 0) | (1 << bit)
        anc = [0] * len(flow.preds)
        for v, ps in enumerate(flow.preds):
            m = 0
            for p in ps:
                m |= anc[p] | own.get(p, 0)
            anc[v] = m
        per_tok = [Fraction(_macs(recs[i][1].dims), tokens) for i in using_idx]
        fwd_mask = gemm_mask = 0
        for bit, i in enumerate(using_idx):
            if phases.get(recs[i][0]) == "forward":
                fwd_mask |= 1 << bit
            if recs[i][1].part != "elementwise":
                gemm_mask |= 1 << bit

        matmuls = []
        for n, g, tdims, classes, names, result, role, weight, rv in recs:
            m = 0
            for a in g.operands:
                if isinstance(a, torch.fx.Node):
                    for x in flow.reads.get((n, a), ()):
                        m |= anc[x] | own.get(x, 0)
            up_macs = sum((per_tok[b] for b in _bits(m)), Fraction(0))
            dtypes = tuple(str(a.meta["val"].dtype).removeprefix("torch.") for a in g.operands if isinstance(a, torch.fx.Node))
            matmuls.append(Matmul(n.name, str(n.target), g.part, phases.get(n, "forward"), _macs(g.dims), dict(g.dims), tdims,
                                  classes, names, dtypes, weight, result, role, Fraction(_macs(g.dims), tokens),
                                  _popcount(m & gemm_mask), _popcount(m & fwd_mask & gemm_mask), up_macs))
        using_elementwise = [m for m in matmuls if m.part == "elementwise"]
        matmuls = [m for m in matmuls if m.part != "elementwise"]
        other = {"elementwise": extra, "approx": 0}
        for n in nodes:
            k, kind = _other_macs(n)
            if kind:
                other[kind] += k
        plist = []
        seen_names = set()
        for t, (kind, pname, aliases) in attrs.items():
            if kind != "parameter" or pname in seen_names:
                continue
            seen_names.add(pname)
            v = getattr(gm, t)
            plist.append({"name": pname, "aliases": list(aliases), "shape": list(v.shape), "dtype": str(v.dtype).removeprefix("torch."),
                          "trained": pname in trained, "updated": pname in upd})
        crossing = []
        for k in sorted(flow.written, key=str):
            if k[0] == "attr" and k[1] in attrs:
                kind, pname, _ = attrs[k[1]]
                v = getattr(gm, k[1])
                crossing.append({"kind": kind, "name": pname, "shape": list(v.shape), "dtype": str(v.dtype).removeprefix("torch.")})
        crossing += [{"kind": "optimizer_state", "name": f"{s['param']}.{s['key']}", "shape": s["shape"], "dtype": s["dtype"]}
                     for s in base.state]
        return cls(name, tokens, dict(zip(axis_names, base.axis_sizes)), matmuls, plist, crossing, upd, trained, gm,
                   digest=_digest(gm), elementwise_macs=other["elementwise"], approx_macs=other["approx"],
                   using_elementwise=using_elementwise, _inputs=(base, varied, axis_names))

    def with_updated(self, updated) -> "StepGraph":
        """The same graph classified with another `updated` set (no recapture)."""
        base, varied, names = self._inputs
        return StepGraph._analyze(base, varied, names, frozenset(updated), self.name)

    # -- totals -----------------------------------------------------------------------------------------------------
    @property
    def macs(self) -> int:
        """The GEMMs' MACs."""
        return sum(m.macs for m in self.matmuls)

    @property
    def c_macs(self) -> int:
        """The step's share of |C| as te_core.flops counts it: GEMMs, elementwise multiplications, linear algebra."""
        return self.macs + self.elementwise_macs + self.approx_macs

    def structure(self) -> dict:
        """The step in te_core.flops's record form (`Counter.record()["structures"][sid]`), which te_core.annotate reads:
        `updated` holds one row [per-token MACs, updated MACs per token upstream, MACs, tokens, weight bits] per using GEMM
        outside the backward, sorted by (upstream, per-token MACs); `log` counts every GEMM by (phase, role, op, dims)."""
        rows = []
        for m in self.matmuls + self.using_elementwise:
            if m.role != "using" or m.phase == "backward":
                continue
            bits = min(_width(d) for c, d in zip(m.operands, m.dtypes) if c == "updated")
            rows.append([_num(m.per_token), _num(m.upstream_macs), m.macs, self.tokens, bits])
        log: dict[tuple, list[int]] = {}
        for m in self.matmuls:
            e = log.setdefault((m.phase, m.role, m.op, json.dumps(m.dims, sort_keys=True)), [0, 0])
            e[0] += 1
            e[1] += m.macs
        return {"updated": sorted(rows, key=lambda r: (r[1], r[0])), "log": sorted([list(k) + v for k, v in log.items()])}

    def role_macs(self, role: str) -> int:
        return sum(m.macs for m in self.matmuls if m.role == role)

    def phase_macs(self) -> dict:
        out: dict[str, int] = {}
        for m in self.matmuls:
            out[m.phase] = out.get(m.phase, 0) + m.macs
        return out

    def depth_profile(self, unit: str = "macs") -> list[tuple[Fraction | int, int]]:
        """(upstream, MACs) of every using GEMM, upstream in per-token MACs (``unit="macs"``) or GEMM nodes
        (``"matmuls"``, ``"matmuls_fwd"``)."""
        key = {"macs": lambda m: m.upstream_macs, "matmuls": lambda m: m.upstream,
               "matmuls_fwd": lambda m: m.upstream_fwd}[unit]
        return sorted((key(m), m.macs) for m in self.matmuls if m.role == "using")

    def deep_macs(self, D, unit: str = "macs") -> int:
        return sum(macs for up, macs in self.depth_profile(unit) if up >= D)

    def phi(self, D, unit: str = "macs") -> Fraction:
        """The share of the step's MACs in using GEMMs with at least D upstream (`depth_profile`'s unit)."""
        return Fraction(self.deep_macs(D, unit), self.macs) if self.macs else Fraction(0)

    def summary(self) -> dict:
        roles = {r: self.role_macs(r) for r in ("using", "producing", "neither")}
        return {"name": self.name, "digest": self.digest, "tokens": self.tokens, "axes": self.axes, "macs": self.macs,
                "elementwise_macs": self.elementwise_macs, "approx_macs": self.approx_macs, "c_macs": self.c_macs,
                "macs_per_token": str(Fraction(self.macs, self.tokens)), "roles": roles, "phases": self.phase_macs(),
                "depth_max": str(max((m.upstream_macs for m in self.matmuls if m.role == "using"), default=0)),
                "updated": sorted(self.updated), "trained": sorted(self.trained), "params": self.params,
                "step_crossing": self.step_crossing, "matmuls": [m.to_json() for m in self.matmuls]}


def _bits(m: int):
    i = 0
    while m:
        if m & 1:
            yield i
        m >>= 1
        i += 1


def _popcount(m: int) -> int:
    return bin(m).count("1")


def _num(x: Fraction) -> int | float:
    return int(x) if x.denominator == 1 else float(x)


def _digest(gm) -> str:
    """The graph's structure: every node's op, target and output shapes and dtypes, and its argument nodes."""
    h = hashlib.sha256()
    idx = {n: i for i, n in enumerate(gm.graph.nodes)}
    for n in gm.graph.nodes:
        v = n.meta.get("val")
        vs = v if isinstance(v, (list, tuple)) else [v]
        shapes = [(tuple(x.shape), str(x.dtype)) if isinstance(x, torch.Tensor) else None for x in vs]
        args = [idx.get(a, repr(a)) if isinstance(a, torch.fx.Node) else
                [idx.get(x, repr(x)) for x in a] if isinstance(a, (list, tuple)) else repr(a) for a in n.args]
        h.update(repr((n.op, str(n.target) if n.op != "get_attr" else "attr", shapes, args)).encode())
    return h.hexdigest()[:16]
