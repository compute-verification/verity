"""The structure a training-efficiency score reads, from one captured step graph, in multiply-accumulates (MACs).

Every matmul-class node (``mm``, ``addmm``, ``bmm``, ``baddbmm``, ``addbmm``, ``mv``, ``addmv``, ``dot``, ``addr``,
``linear``, ``matmul``, ``convolution`` and ``convolution_backward``'s two GEMMs, ``_scaled_mm``, ``_int_mm``) is one
`Matmul` per GEMM it runs, with dims (batch, m, k, n), MACs = batch·m·k·n, and which token axes index each dim (from
the varied captures, `trace`). A matmul-like op outside that list is refused. `macs` is the GEMMs' MACs. The rest of
the step's share of |C| is counted by training_slowdown.flops's rule, so `c_macs` matches the harness's count: one MAC per output
element of each elementwise multiplication (`elementwise_macs`: ``mul``, ``div``, ``addcmul``, ``addcdiv``, ``lerp``,
``add`` and ``sub`` with a scale, their in-place and ``_foreach_`` forms, fused optimizer steps at a constant per
parameter, never below their unfused forms' multiplications), dense linear algebra at n³ per matrix (`approx_macs`),
and any other forward op that combines an activation with a parameter that differs from θ (a norm with a trained gain,
an add then ``exp``) at one MAC per element of its largest operand or result (only its indices and result, for a
gather), as training_slowdown.flops counts it.

Terms are the training ontology's, with T2 read as deep use of activations that depend on the run's own persistent
values, however the circuit is written.

- A value is **token-indexed** when one of its dims changes with a token axis (a GEMM's result when its batch, m or n
  dim does), else **parameter-indexed** (a weight-shaped sum over tokens, as dW).
- A **slot** is a storage the step reads that outlives it: a parameter, a buffer, an optimizer's state, an error-feedback
  buffer (a graph attribute, `Capture.attrs`, or one the step's earlier steps wrote, `Capture.written_before`). A slot is
  **dependent** when its value is one the run produced: a parameter named in `updated` (when the caller passes it), a
  slot an earlier step wrote that this one doesn't, and every slot the step writes from a dependent slot (its own old
  value included: W ← W − u). Of the slots the step writes, the largest set closed under that rule is dependent, so a
  cache the step rewrites from θ, constants and token data alone (``copy_``, ``fill_`` and in-place samplers write a
  whole storage without reading it) is not, and a parameter the step updates in place is. A storage copied onto itself
  is read: the copy's source is the storage's own value (``b.copy_(b.float().mul_(μ).add_(g))`` on an fp32 buffer,
  whose ``float()`` is the buffer), so a momentum buffer written that way keeps what it held.
- A value is **update-dependent** when it is computed, through any path, from a dependent slot's value at the step's
  start. A gather of a dependent table is (its rows are), with no MAC; a value computed only from θ, constants and token
  data is not.
- **Updated**: a MAC of the forward (outside autograd's backward and an optimizer's step) with a token-indexed result and
  an update-dependent operand: a GEMM (role ``updated``), an elementwise multiplication or a counted op of
  training_slowdown.flops's rule (`updated_elementwise`, one `Matmul` of part ``elementwise`` per node). Backward products against
  the same values are left out (counting them halves β per MAC and doubles φ, leaving φβ unchanged and D ≥ k/β harder).
  **Producing**: a parameter-indexed result that reaches the final value of a parameter the step writes through
  weight-shaped values only (dW into the optimizer's state and the parameter; a matrix optimizer's own products).
  Every other GEMM is **neither**.
- **Depth** (`upstream_macs`): for each MAC node, the updated MACs per token of its step upstream of it, through the
  forward values of its own step (`upstream` counts the updated GEMMs). The count is at node granularity: exact where a
  token's row is mixed between nodes (a norm, a softmax, a contraction over the row), an upper bound otherwise.
- **φ(D)**: the share of the step's MACs in updated MACs with at least D upstream (`phi`).
- **Cut** (`cut_bits`, T1's witness): the narrowest vertex cut, in bits, between the step's gradient-side computation
  and the new values of the dependent slots its forward reads (the sinks). Paths start at the token data and at the old
  values of the dependent slots the forward doesn't read (optimizer moments, error feedback, gradient accumulators).
  The forward's own values (its token-indexed values and everything they read) can't be cut, nor can a value computed
  only from θ, constants and token data; every other update-dependent value can (the loss, the backward, an
  optimizer's step or an update written by hand, those old values), at its dtype's width times its elements (a bool is
  one bit, a sparse tensor its stored elements), and a sink at its elements times the width the forward reads it at
  (through casts). A circuit that sends its update through a sign bit with error feedback, a top-k list or low-rank
  factors has that cut; a dense update's is its elements at that width. `beta` is the cut over the step's updated MACs
  per token.
- A program recorded from its start (a capture of several steps whose `Capture.region` names one) is measured at that
  step of its history, and its slots are whatever it carries: a value is update-dependent when it depends on an earlier step's token data
  (the run's weights, optimizer state and error feedback, kept in place, rebound or written by hand; a cache made from
  θ and this step's tokens is not), and the cut's sinks are the update-dependent values the step makes that the next
  step's forward reads, at the width it reads them (none in a program's last step, whose cut is 0).

Dataflow follows storage: an in-place op (``add_``, a foreach optimizer kernel) makes a new version of the storage it
writes, every view of it reads that version, and each element of a foreach list is its own value. A Python scalar read
from a tensor (`Capture` from a recorded program, ``.item()``) carries its tensor's dependence into the ops it reaches.
"""
from __future__ import annotations

import hashlib
import json
import operator
from dataclasses import dataclass, field
from fractions import Fraction
from math import prod
from typing import Any

import numpy as np
import torch

from training_slowdown.capture.circuit import ATTR, CARRIED, COMPUTED, OTHER_INPUT, RANDOM, TOKEN_INPUT, Net, StepCircuit, measure
from training_slowdown.capture.trace import BACKWARD, Capture, CaptureRefused
from training_slowdown.flops import (ELEMENTWISE_NAMES, FUSED_OPTIMIZER_MACS, GATHERS, GEMM_LIKE, LINALG_NAMES, NOT_GEMM, SCALED_NAMES,
                           TOKEN_INDEX, _is_view, gather_macs)
from training_slowdown.gates import kind_of, per_mac

__all__ = ["CLASSES", "Matmul", "StepGraph"]

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
               "sym_size", "sym_numel", "sym_stride", "empty_strided", "randn_like", "rand_like", "randint_like"}
#: in-place ops that write every element of what they write without reading it. `_Flow` skips the written argument by
#: its position, so a source that is the storage written (``b.copy_(b)``) is still read
_OVERWRITE = {"copy_", "fill_", "zero_", "set_", "normal_", "uniform_", "bernoulli_", "random_", "exponential_",
              "geometric_", "log_normal_", "cauchy_", "_foreach_copy_", "_foreach_zero_"}
#: matmul-like names that `_gemms` must account for (training_slowdown.flops's); any other op with one of these in its name is refused
_GEMM_LIKE, _NOT_GEMM = GEMM_LIKE, NOT_GEMM
_ELEMENTWISE, _SCALED, _FUSED, _LINALG = ELEMENTWISE_NAMES, SCALED_NAMES, FUSED_OPTIMIZER_MACS, LINALG_NAMES
_TOKEN_INDEX = {p.__name__ for p in TOKEN_INDEX}
_GATHERS = {p.__name__ for p in GATHERS}
#: ops whose values are random: a circuit's are a PRNG's of a seed it reads
_RANDOM = {"randn", "rand", "randint", "randperm", "normal", "normal_", "uniform_", "bernoulli", "bernoulli_",
           "native_dropout", "_fused_dropout", "multinomial", "random_", "exponential_", "geometric_", "log_normal_",
           "cauchy_", "rand_like", "randn_like", "randint_like", "poisson", "_standard_gamma", "rrelu_with_noise"}
_PHASE_CODE = {"forward": 0, "backward": 1}


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
    """A non-GEMM node's MACs by training_slowdown.flops's rule (module docstring): (MACs, "elementwise" | "approx" | "")."""
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
    """A non-GEMM node's work by training_slowdown.flops's rule (`flops._work`): one per element of its largest operand or result,
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


def _dtype_bits(dtype: torch.dtype) -> int:
    return 1 if dtype == torch.bool else dtype.itemsize * 8


_SPARSE = (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc)


def _value_bits(v) -> int:
    """A value's bits: its elements (a sparse tensor's stored ones) at its dtype's width; 64 for a Python scalar."""
    if isinstance(v, torch.Tensor):
        n = v._nnz() * (1 + v.sparse_dim()) if v.layout in _SPARSE else v.numel()
        return n * _dtype_bits(v.dtype)
    return 64 if isinstance(v, (int, float, bool, torch.SymInt, torch.SymFloat)) else 0


@dataclass(frozen=True)
class Matmul:
    """One MAC node of the step (module docstring): a GEMM, or (part ``elementwise``) an updated elementwise or counted op."""
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
    step: int = 0
    tokens: int = 1
    #: the narrowest width of its update-dependent operands (0: none)
    width: int = 0


def _phases(gm) -> dict:
    """Each node's phase, from the `record_function` markers around it (module docstring of `trace`)."""
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
        out[n] = ph
    return out


class _Flow:
    """Values of the graph with their direct dependences, storage-versioned (module docstring)."""

    def __init__(self, gm, phases: dict, tok) -> None:
        self.gm = gm
        self.preds: list[tuple[int, ...]] = []
        self.phase: list[str] = []
        self.tok: list[bool] = []
        self.step: list[int] = []
        self.bits: list[int] = []
        self.width: list[int] = []                     # value -> its dtype's bits (64: a Python scalar)
        self.node: list[Any] = []                      # value -> the node that made it
        self.source: dict[int, tuple[str, str]] = {}   # value -> ("attr", target) | ("input", name): a graph source
        self.val: dict[Any, Any] = {}
        self.base: dict[Any, Any] = {}
        self.cur: dict[Any, int] = {}
        self.size: dict[Any, int] = {}                 # storage -> its elements
        self.written: set = set()
        self.writes: list[tuple[Any, int, Any]] = []   # (storage, its new value, the node that wrote it)
        self.reads: dict[tuple[Any, Any], frozenset[int]] = {}   # (node, argument node) -> the values it read when it ran
        self._at, self._step = None, 0
        for n in gm.graph.nodes:
            if n.op == "call_function":
                for a in _arg_nodes(n) + list(n.meta.get("te_scalar_deps", ())):
                    self.reads[(n, a)] = frozenset(self._rd(a))
            self._at, self._step = n, int(n.meta.get("te_step", 0))
            self._node(n, phases.get(n, "forward"), tok)

    def _new(self, preds, phase: str, tok: bool, val=None) -> int:
        """A value; `tok`: whether it is token-indexed."""
        self.preds.append(tuple(sorted(set(preds))))
        self.phase.append(phase)
        self.tok.append(bool(tok))
        self.step.append(self._step)
        self.bits.append(_value_bits(val))
        self.width.append(_dtype_bits(val.dtype) if isinstance(val, torch.Tensor) else 64)
        self.node.append(self._at)
        return len(self.preds) - 1

    def _write(self, a, reads, ph: str, tok: bool) -> tuple[int, Any]:
        """A new value of the storage `a` (a node's argument) writes."""
        b = self.base.get(a, ("node", a.name))
        val = a.meta.get("val")
        bits = self.size.get(b, val.numel() if isinstance(val, torch.Tensor) else 1)
        v = self._new(reads, ph, tok)
        if isinstance(val, torch.Tensor):
            self.bits[v] = bits * _dtype_bits(val.dtype)
            self.width[v] = _dtype_bits(val.dtype)
        self.cur[b] = v
        self.written.add(b)
        self.writes.append((b, v, self._at))
        return v, b

    def _whole(self, a) -> bool:
        """Whether the argument `a` covers all of its storage."""
        val = a.meta.get("val")
        b = self.base.get(a, ("node", a.name))
        return isinstance(val, torch.Tensor) and val.numel() == self.size.get(b, val.numel())

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
            val = n.meta.get("val")
            v = self._new((), ph, tok(n, None), val)
            self.val[n], self.base[n] = v, ("input", n.name)
            self.cur[("input", n.name)] = v
            self.size[("input", n.name)] = val.numel() if isinstance(val, torch.Tensor) else 1
            self.source[v] = ("input", n.name)
            return
        if n.op == "get_attr":
            key = ("attr", n.target)
            if key not in self.cur:
                val = getattr(self.gm, n.target, None)
                v = self._new((), ph, tok(n, None), val)
                self.cur[key] = v
                self.size[key] = val.numel() if isinstance(val, torch.Tensor) else 1
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
        scalars = set()
        for d in n.meta.get("te_scalar_deps", ()):
            scalars |= self._rd(d)
        shape_only = name in _SHAPE_ONLY
        overwrite = name in _OVERWRITE
        if lists and (any(p in written_pos for p, _ in lists)
                      or (name.startswith("_foreach") and any(len(a) > 1 for _, a in lists))):
            k = max(len(a) for _, a in lists)
            common = set(scalars)
            for _, a in singles:
                common |= self._rd(a)
            reads = [set(common) for _ in range(k)]
            for p, a in lists:
                if overwrite and p in written_pos:
                    continue
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
                        own = set() if not overwrite or self._whole(x) else self._rd(x)
                        self._write(x, reads[i] | own, ph, tok(x, None))
                self.val[n] = None
                return
            outs = n.meta.get("val")
            vals = [self._new(reads[i], ph, tok(n, i), outs[i] if isinstance(outs, (list, tuple)) and i < len(outs) else None)
                    for i in range(k)]
            self.val[n] = vals
            self.base[n] = [("node", n.name, i) for i in range(k)]
            for i, v in enumerate(vals):
                self.cur[("node", n.name, i)] = v
                o = outs[i] if isinstance(outs, (list, tuple)) and i < len(outs) else None
                self.size[("node", n.name, i)] = o.numel() if isinstance(o, torch.Tensor) else 1
            return
        mutated = [a for p, a in singles if p in written_pos]
        reads = set(scalars)
        if not shape_only:
            for p, a in singles:
                if not (overwrite and p in written_pos and self._whole(a)):
                    reads |= self._rd(a)
            for _, a in lists:
                for x in a:
                    reads |= self._rd(x)
        if mutated:
            first = None
            for a in mutated:
                v, b = self._write(a, reads, ph, tok(a, None))
                first = (v, b) if first is None else first
            self.val[n], self.base[n] = first
            return
        ret = schema.returns if schema is not None else ()
        is_view = bool(ret) and ret[0].alias_info is not None and not ret[0].alias_info.is_write and singles
        out = n.meta.get("val")
        if isinstance(out, (list, tuple)):
            vals = [self._new(reads, ph, tok(n, i), o) if isinstance(o, torch.Tensor) else None for i, o in enumerate(out)]
            self.val[n] = vals
            if is_view:
                self.base[n] = [self.base.get(singles[0][1], ("node", singles[0][1].name))] * len(vals)
            else:
                self.base[n] = [("node", n.name, i) for i in range(len(vals))]
                for i, v in enumerate(vals):
                    if v is not None:
                        self.cur[("node", n.name, i)] = v
                        self.size[("node", n.name, i)] = out[i].numel()
            return
        v = self._new(reads, ph, tok(n, None), out)
        self.val[n] = v
        if is_view:
            self.base[n] = self.base.get(singles[0][1], ("node", singles[0][1].name))
        else:
            self.base[n] = ("node", n.name)
            self.cur[("node", n.name)] = v
            self.size[("node", n.name)] = out.numel() if isinstance(out, torch.Tensor) else 1


def _arg_nodes(n) -> list:
    out = []
    for a in list(n.args) + list(n.kwargs.values()):
        if isinstance(a, torch.fx.Node):
            out.append(a)
        elif isinstance(a, (list, tuple)):
            out.extend(x for x in a if isinstance(x, torch.fx.Node))
    return out


def _slots(base, flow) -> tuple[dict, dict, set]:
    """The graph attributes' values where the capture first reads them, by slot; the values the capture writes to each
    slot; and the slots that hold the run's own value there, not θ's: the ones steps before the capture wrote
    (`Capture.written_before`), and in a capture that starts after its run's first step (`Capture.mid_run`) every slot
    the capture writes."""
    entry = {k: v for v, k in flow.source.items() if k[0] == "attr"}
    writes: dict = {}
    for k, v, _ in flow.writes:
        if k[0] == "attr":
            writes.setdefault(k, []).append(v)
    before = {("attr", t) for t in base.written_before if ("attr", t) in entry}
    return entry, writes, before | (set(writes) & set(entry) if base.mid_run else set())


def _past(flow, region: int, entry: dict, before: set) -> set:
    """What the run produced before step `region` of a capture that names it: the token inputs of the steps before it,
    and the slots that hold the run's own value where the capture first reads them (`_slots`)."""
    return ({v for v, k in flow.source.items() if k[0] == "input" and flow.step[v] < region}
            | {entry[k] for k in before})


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
    #: the forward's updated MAC nodes that aren't GEMMs (module docstring)
    updated_elementwise: list[Matmul] = field(default_factory=list)
    #: the dependent slots, by parameter or attribute name
    dependent: list = field(default_factory=list)
    #: the cut (module docstring) in bits, and its bits per updated MAC per token (None: no updated MAC)
    cut_bits: int = 0
    beta: float | None = None
    #: (value, bits) on one narrowest cut, widest first; the dependent slots whose new values are the cut's sinks
    cut: list = field(default_factory=list)
    sinks: list = field(default_factory=list)
    #: a recorded program's measured step as a circuit (`training_slowdown.capture.circuit`), and the formalism's quantities on it
    circuit: StepCircuit | None = field(default=None, repr=False)
    formalism: dict | None = None
    _inputs: Any = field(default=None, repr=False)

    # -- building ---------------------------------------------------------------------------------------------------
    @classmethod
    def build(cls, base: Capture, varied: list[Capture], axis_names: tuple[str, ...], *, updated, name: str) -> "StepGraph":
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
    def _analyze(cls, base: Capture, varied: list[Capture], axis_names, updated, name) -> "StepGraph":
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
        tokens = prod(base.axis_sizes) if base.axis_sizes else 1
        step_tokens = dict(base.step_tokens or {})
        region_tokens = sum(step_tokens.values()) if step_tokens else tokens

        def tokens_of(n) -> int:
            return step_tokens.get(int(n.meta.get("te_step", 0)), tokens) if step_tokens else tokens

        region = base.region

        def in_region(n) -> bool:
            return region is None or int(n.meta.get("te_step", 0)) == region

        # dependent slots: the largest set of written slots closed under the rule (module docstring); a program recorded
        # from its start reads it off its history: what a step reads is the run's own when it depends on an earlier
        # step's token data, through whatever the program carried (in place, rebound, by hand)
        entry, writes, before = _slots(base, flow)
        carried: list[int] = []
        if region is not None:
            dep = _closure(flow, _past(flow, region, entry, before).__contains__)
            carried = sorted({p for v, ps in enumerate(flow.preds) if flow.step[v] == region for p in ps
                              if flow.step[p] < region and dep[p]})
            dependent: set = set()
            upd = frozenset()
        else:
            if updated is None:
                seeded, slots = set(), set(writes) | before
            else:
                seeded = {k for k in entry if k[1] in params and params[k[1]][1] in updated}
                slots = {k for k in set(writes) | before if k[1] not in params}
            while True:
                seeds = {entry[k] for k in slots | seeded if k in entry}
                dep = _closure(flow, seeds.__contains__)
                kept = {k for k in slots if k not in writes or any(dep[v] for v in writes[k])}
                if kept == slots:
                    break
                slots = kept
            dependent = slots | seeded
            upd = (frozenset(params[k[1]][1] for k in dependent if k[1] in params) if updated is None
                   else frozenset(updated))
        seed = _closure(flow, lambda v: flow.phase[v] == "backward" and (region is None or flow.step[v] == region))
        dep_par = _closure(flow, lambda v: flow.source.get(v, ("",))[0] == "attr")

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
            if region is None and isinstance(src, torch.fx.Node) and src.op == "get_attr":
                hit = attrs.get(src.target)
                if hit is None:
                    return "constant", None
                if hit[0] == "parameter":
                    return ("updated" if hit[1] in upd else "registered"), hit[1]
                return ("updated" if ("attr", src.target) in dependent else "registered"), hit[1]
            vs = flow.reads.get((n, a), frozenset())
            if not vs:
                return "constant", None
            if any(seed[x] for x in vs):
                return "gradient", None
            if any(flow.tok[x] for x in vs):
                return "activation", None
            if any(dep[x] for x in vs):
                return "updated", None
            if any(dep_par[x] for x in vs):
                return "registered", None
            return "constant", None

        def depends(n, ops) -> bool:
            return any(dep[x] for a in ops if isinstance(a, torch.fx.Node) for x in flow.reads.get((n, a), ()))

        def width(n, ops) -> int:
            ws = [_dtype_bits(a.meta["val"].dtype) for a in ops if isinstance(a, torch.fx.Node)
                  and isinstance(a.meta.get("val"), torch.Tensor) and any(dep[x] for x in flow.reads.get((n, a), ()))]
            return min(ws, default=0)

        # GEMMs and their dims in every capture; each counted op's gates at its Definition's price (training_slowdown.gates)
        node_gates: dict = {}

        def priced(n, macs, dtype, kind=None, width=16) -> None:
            if macs:
                k = kind or kind_of(phases.get(n, "forward"), "", _pk(n.target), dtype)[0]
                node_gates[n] = node_gates.get(n, 0.0) + macs * per_mac(k, width)

        raw = []
        for n in nodes:
            gs = _gemms(n) if in_region(n) else []
            if not gs:
                continue
            vgs = [_gemms(vs[pos[n]]) for vs in vnodes]
            for j, g in enumerate(gs):
                priced(n, _macs(g.dims), _dtype_name(g.operands[0]))
                tdims = {}
                for d, size in g.dims.items():
                    axes = tuple(axis_names[a] for a, vg in enumerate(vgs) if vg[j].dims[d] != size)
                    if axes:
                        tdims[d] = axes
                # a decode step's product over its context (one position a sequence, its sequences batched): the
                # position is the size-1 dim, which the sequences index as they index the batch
                if "sequences" in tdims.get("batch", ()) and any(d in tdims for d in ("m", "k", "n")):
                    for d in ("m", "k", "n"):
                        if d not in tdims and g.dims[d] == 1:
                            tdims[d] = ("sequences",)
                raw.append((n, g, tdims))

        def result_value(n, g) -> int | None:
            v = flow.val.get(n)
            if isinstance(v, list):
                return v[g.out] if g.out is not None and g.out < len(v) else None
            return v

        region_sinks = _region_sinks(flow, nodes, region, dep) if region is not None else {}
        # producing: parameter-indexed results reaching a written parameter's final value through weight-shaped values
        finals = (list(region_sinks) if region is not None
                  else [flow.cur[k] for k in flow.written if k[0] == "attr" and k[1] in params])
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
            if result == "token" and phases.get(n, "forward") == "forward" and depends(n, g.operands):
                role = "updated"
            elif result == "parameter" and rv is not None and rv in reach:
                role = "producing"
            else:
                role = "neither"
            weight = min(classes, key=lambda c: _RANK[c]) if classes else "constant"
            recs.append([n, g, tdims, classes, names, result, role, weight, rv])

        # |C|'s other forward work (training_slowdown.flops): an activation combined with an operand that differs from θ, at its
        # elementwise multiplications, else its work (`_work`); an updated source gathered by token data is the
        # circuit's gather (`gather_macs`). The updated ones: an elementwise or counted op whose result is
        # token-indexed with an update-dependent operand; a gather moves a dependent table's rows with no MAC.
        extra = 0
        for n in nodes:
            if (n.op != "call_function" or not isinstance(n.target, torch._ops.OpOverload) or not in_region(n) or _gemms(n)
                    or phases.get(n, "forward") != "forward" or _is_view(n.target)
                    or _pk(n.target) in ("_record_function_enter_new", "_record_function_exit")):
                continue
            ops = _arg_nodes(n)
            gather = _pk(n.target) in _GATHERS
            src = set(_arg_nodes_of(n.args[:1])) if gather else set()
            cls_ = [klass(n, a) for a in ops]
            mixed = [c for a, (c, _) in zip(ops, cls_) if a not in src]
            source = [c for a, (c, _) in zip(ops, cls_) if a in src]
            k, kind = _other_macs(n)
            work = 0
            if "updated" in source and "activation" in mixed:
                gm_ = gather_macs(_pk(n.target), n.args[0].meta["val"].shape, n.args[1] if len(n.args) > 1 else None,
                                  n.meta["val"].numel())
                extra += gm_
                priced(n, gm_, None, "gather", _dtype_bits(n.args[0].meta["val"].dtype))
            elif "activation" in mixed and "updated" in mixed and not kind:
                work = _work(n)
                extra += work
                priced(n, work, _dtype_name(n))
            macs = k if kind else work
            out = n.meta.get("val")
            g = _G("elementwise", {"batch": 1, "m": macs, "k": 1, "n": 1}, tuple(ops),
                   0 if isinstance(out, (list, tuple)) else None)
            rv = result_value(n, g)
            if gather or not macs or rv is None or not flow.tok[rv] or not depends(n, ops):
                continue
            classes = tuple(c for c, _ in cls_)
            recs.append([n, g, {}, classes, tuple(p for _, p in cls_), "token", "updated",
                         min(classes, key=lambda c: _RANK[c]) if classes else "constant", rv])

        # depth: the forward's updated MACs upstream of each node within its step, as bitsets over the updated records
        fwd = [i for i, r in enumerate(recs) if r[6] == "updated" and phases.get(r[0], "forward") == "forward"]
        own: dict[int, int] = {}
        for bit, i in enumerate(fwd):
            rv = recs[i][8]
            if rv is not None:
                own[rv] = own.get(rv, 0) | (1 << bit)
        anc = [0] * len(flow.preds)
        for v, ps in enumerate(flow.preds):
            if flow.phase[v] != "forward":
                continue
            m = 0
            for p in ps:
                if flow.phase[p] == "forward" and flow.step[p] == flow.step[v]:
                    m |= anc[p] | own.get(p, 0)
            anc[v] = m
        upstream = _bit_sums([_macs(recs[i][1].dims) for i in fwd], [tokens_of(recs[i][0]) for i in fwd])
        gemm_mask = 0
        for bit, i in enumerate(fwd):
            if recs[i][1].part != "elementwise":
                gemm_mask |= 1 << bit

        matmuls = []
        for n, g, tdims, classes, names, result, role, weight, rv in recs:
            m = 0
            if phases.get(n, "forward") == "forward":
                st = int(n.meta.get("te_step", 0))
                for a in g.operands:
                    if isinstance(a, torch.fx.Node):
                        for x in flow.reads.get((n, a), ()):
                            if flow.phase[x] == "forward" and flow.step[x] == st:
                                m |= anc[x] | own.get(x, 0)
            up_macs = upstream(m)
            dtypes = tuple(str(a.meta["val"].dtype).removeprefix("torch.") for a in g.operands
                           if isinstance(a, torch.fx.Node) and isinstance(a.meta.get("val"), torch.Tensor))
            matmuls.append(Matmul(n.name, str(n.target), g.part, phases.get(n, "forward"), _macs(g.dims), dict(g.dims), tdims,
                                  classes, names, dtypes, weight, result, role, Fraction(_macs(g.dims), tokens_of(n)),
                                  _popcount(m & gemm_mask), _popcount(m & gemm_mask), up_macs,
                                  int(n.meta.get("te_step", 0)), tokens_of(n), width(n, g.operands) if role == "updated" else 0))
        updated_elementwise = [m for m in matmuls if m.part == "elementwise"]
        matmuls = [m for m in matmuls if m.part != "elementwise"]
        other = {"elementwise": extra, "approx": 0}
        for n in nodes:
            if not in_region(n):
                continue
            k, kind = _other_macs(n)
            if kind:
                other[kind] += k
                priced(n, k, _dtype_name(n))

        # the cut (module docstring)
        fr = [False] * len(flow.preds)
        for v in range(len(flow.preds) - 1, -1, -1):
            if flow.phase[v] == "forward" and flow.tok[v] and (region is None or flow.step[v] == region):
                fr[v] = True
            if fr[v] and (region is None or flow.phase[v] == "forward"):
                for p in flow.preds[v]:
                    if region is not None or flow.phase[p] == "forward":
                        fr[p] = True
        if region is None:
            read_width: dict[str, int] = {}
            for n in nodes:
                out = flow.val.get(n)
                if n.op != "call_function" or phases.get(n, "forward") != "forward" or not isinstance(out, int) \
                        or not flow.tok[out]:
                    continue
                for a in _arg_nodes(n):
                    s = walk(a)
                    if isinstance(s, torch.fx.Node) and s.op == "get_attr" and isinstance(a.meta.get("val"), torch.Tensor):
                        read_width[s.target] = max(read_width.get(s.target, 0), _dtype_bits(a.meta["val"].dtype))
            sources = [v for v, k in flow.source.items() if k[0] == "input"]
            sources += [entry[k] for k in dependent if k in entry and not fr[entry[k]]]
            sinks: dict[int, int] = {}
            for k in dependent:
                if k in writes and k in entry and fr[entry[k]]:
                    t = getattr(gm, k[1], None)
                    own_w = _dtype_bits(t.dtype) if isinstance(t, torch.Tensor) else 64
                    sinks[flow.cur[k]] = flow.size.get(k, 1) * min(own_w, read_width.get(k[1], own_w))
            cuttable = [d and not f for d, f in zip(dep, fr)]
        else:
            sources = [v for v, k in flow.source.items() if k[0] == "input" and flow.step[v] == region]
            sources += [v for v in carried if not fr[v]]
            sinks = region_sinks
            cuttable = [d and not f and not (s == region + 1 and ph == "forward")
                        for d, f, s, ph in zip(dep, fr, flow.step, flow.phase)]
        cut_bits, on_cut = _min_cut(flow, cuttable, sources, sinks)
        u_macs = sum(_macs(recs[i][1].dims) for i in fwd)
        beta = cut_bits / (u_macs / region_tokens) if u_macs and region_tokens else None

        def label(v) -> str:
            n = flow.node[v]
            return n.target if n is not None and n.op == "get_attr" else (n.name if n is not None else "?")

        cut = sorted(((label(v), flow.bits[v] if v not in sinks else sinks[v]) for v in on_cut), key=lambda x: -x[1])
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
        names_of = {k: (attrs[k[1]][1] if k[1] in attrs else k[1]) for k in dependent}
        if region is None:
            dep_names = sorted(names_of.values())
            sink_names = sorted(names_of[k] for k in dependent if k in writes and k in entry and fr[entry[k]])
        else:
            dep_names, sink_names = sorted({label(v) for v in carried}), sorted({label(v) for v in sinks})
        circuit = (_step_circuit(flow, nodes, region, dep, carried, node_gates, step_tokens.get(region, tokens))
                   if region is not None else None)
        return cls(name, region_tokens, dict(zip(axis_names, base.axis_sizes)), matmuls, plist, crossing, upd, trained, gm,
                   digest=_digest(gm), elementwise_macs=other["elementwise"], approx_macs=other["approx"],
                   updated_elementwise=updated_elementwise, dependent=dep_names, cut_bits=cut_bits,
                   beta=beta, cut=cut, sinks=sink_names, circuit=circuit,
                   formalism=measure(circuit) if circuit is not None else None, _inputs=(base, varied, axis_names))

    # -- totals -----------------------------------------------------------------------------------------------------
    @property
    def macs(self) -> int:
        """The GEMMs' MACs."""
        return sum(m.macs for m in self.matmuls)

    @property
    def c_macs(self) -> int:
        """The step's share of |C| as training_slowdown.flops counts it: GEMMs, elementwise multiplications, linear algebra."""
        return self.macs + self.elementwise_macs + self.approx_macs

    def _updated(self) -> list[Matmul]:
        return [m for m in self.matmuls + self.updated_elementwise if m.role == "updated" and m.phase == "forward"]

    def structure(self) -> dict:
        """The step in training_slowdown.flops's record form (`Counter.record()["structures"][sid]`), which training_slowdown.annotate reads:
        `updated` holds one row [MACs per token, updated MACs per token upstream, MACs, tokens, width] per updated MAC node,
        sorted by (upstream, MACs per token), the width its update-dependent operands' narrowest; `beta` the cut's bits
        per updated MAC per token (None: no updated MAC), `cut_bits` the cut; `log` counts every GEMM by (phase, role,
        op, dims)."""
        rows = [[_num(Fraction(m.macs, self.tokens)), _num(m.upstream_macs), m.macs, self.tokens, m.width]
                for m in self._updated()]
        log: dict[tuple, list[int]] = {}
        for m in self.matmuls:
            e = log.setdefault((m.phase, m.role, m.op, json.dumps(m.dims, sort_keys=True)), [0, 0])
            e[0] += 1
            e[1] += m.macs
        out = {"updated": sorted(rows, key=lambda r: (r[1], r[0])), "log": sorted([list(k) + v for k, v in log.items()]),
               "cut_bits": self.cut_bits, "beta": self.beta}
        return out | ({"formalism": self.formalism} if self.formalism is not None else {})

    def updated_macs(self) -> int:
        """The forward's updated MACs, GEMMs and elementwise."""
        return sum(m.macs for m in self._updated())


def _walks(a) -> bool:
    """Whether a value read through node `a` is read as its argument's (a view, a cast, a scaling by a constant)."""
    if not (isinstance(a, torch.fx.Node) and a.op == "call_function" and a.args and isinstance(a.args[0], torch.fx.Node)):
        return False
    nm = _pk(a.target)
    return nm in _WALK or nm in _SCALE and sum(isinstance(x, torch.fx.Node) for x in list(a.args) + list(a.kwargs.values())) == 1


def _region_sinks(flow: _Flow, nodes: list, region: int, dep: list[bool]) -> dict[int, int]:
    """A recorded program's new values (`Capture.region`): each update-dependent value the measured step made, or the next
    step made before its forward, that the next step's forward reads; its capacity its elements at the narrowest of its
    own width and the widest the forward reads it at (through views and casts into a token-indexed op). The next step's
    forward is its token-indexed values and what they read through its own values outside backward and an optimizer's
    step, so its gradient clipping, which reads the step's gradients outside both, isn't."""
    nxt = region + 1
    ahead = [flow.step[v] == nxt and flow.phase[v] == "forward" for v in range(len(flow.preds))]
    model = [False] * len(flow.preds)
    for v in range(len(flow.preds) - 1, -1, -1):
        if ahead[v] and (model[v] or flow.tok[v]):
            model[v] = True
            for p in flow.preds[v]:
                if ahead[p]:
                    model[p] = True
    out: dict[int, int] = {}
    for v, ps in enumerate(flow.preds):
        if model[v]:
            for p in ps:
                if dep[p] and flow.step[p] >= region and not ahead[p]:
                    out[p] = 0
    read: dict[int, int] = {}
    for n in nodes:
        o = flow.val.get(n)
        if (n.op != "call_function" or int(n.meta.get("te_step", 0)) != nxt or not isinstance(o, int)
                or flow.phase[o] != "forward" or not flow.tok[o]):
            continue
        for a in _arg_nodes(n):
            val = a.meta.get("val")
            if not isinstance(val, torch.Tensor):
                continue
            prev, cur = n, a
            while _walks(cur) and int(cur.meta.get("te_step", 0)) == nxt:
                prev, cur = cur, cur.args[0]
            for x in flow.reads.get((prev, cur), ()):
                if x in out:
                    read[x] = max(read.get(x, 0), _dtype_bits(val.dtype))
    for x in out:
        w = max(flow.width[x], 1)
        out[x] = flow.bits[x] // w * min(w, read.get(x, w))
    return out


def _min_cut(flow: _Flow, cuttable: list[bool], sources: list[int], sinks: dict[int, int]) -> tuple[int, list[int]]:
    """The least bits of a vertex set separating `sources` from `sinks` (value -> its capacity) through `cuttable`
    values and sinks only, and the values on one such cut."""
    if not sinks or not sources:
        return 0, []
    n = len(flow.preds)
    succ: list[list[int]] = [[] for _ in range(n)]
    for v, ps in enumerate(flow.preds):
        for p in ps:
            succ[p].append(v)
    fwd = [False] * n
    stack = list(sources)
    for s in sources:
        fwd[s] = True
    while stack:
        for w in succ[stack.pop()]:
            if not fwd[w]:
                fwd[w] = True
                stack.append(w)
    back = [False] * n
    stack = [s for s in sinks if fwd[s]]
    for s in stack:
        back[s] = True
    while stack:
        for p in flow.preds[stack.pop()]:
            if fwd[p] and not back[p]:
                back[p] = True
                stack.append(p)
    keep = [v for v in range(n) if fwd[v] and back[v]]
    if not keep:
        return 0, []
    idx = {v: i for i, v in enumerate(keep)}
    finite = [sinks[v] if v in sinks else flow.bits[v] for v in keep]
    inf = sum(finite) + 1
    token = {v for v, k in flow.source.items() if k[0] == "input"}
    net = Net(2 * len(keep) + 2)
    S, T = 2 * len(keep), 2 * len(keep) + 1
    for i, v in enumerate(keep):
        net.add(2 * i, 2 * i + 1, finite[i] if v in sinks or (cuttable[v] and v not in token) else inf)
        for p in flow.preds[v]:
            if p in idx:
                net.add(2 * idx[p] + 1, 2 * i, inf)
    for s in sources:
        if s in idx:
            net.add(S, 2 * idx[s], inf)
    for t in sinks:
        if t in idx:
            net.add(2 * idx[t] + 1, T, inf)
    total, seen = net.max_flow(S, T)
    return min(total, inf), [v for i, v in enumerate(keep) if seen[2 * i] and not seen[2 * i + 1]]


def _dtype_name(x) -> str | None:
    v = x.meta.get("val") if isinstance(x, torch.fx.Node) else None
    if isinstance(v, (list, tuple)):
        v = next((y for y in v if isinstance(y, torch.Tensor)), None)
    return str(v.dtype).removeprefix("torch.") if isinstance(v, torch.Tensor) else None


def _storage_positions(v: torch.Tensor) -> np.ndarray:
    """The storage positions a (fake) tensor's elements sit at, as training_slowdown.flops's `read_positions`."""
    pos = np.array(int(v.storage_offset()), dtype=np.int64)
    for n, s in zip(v.shape, v.stride()):
        pos = pos[..., None] + np.arange(int(n), dtype=np.int64) * int(s)
    return np.unique(pos)


def _step_circuit(flow: _Flow, nodes: list, region: int, dep: list[bool], carried: list[int], node_gates: dict,
                  step_tokens: int) -> StepCircuit:
    """The measured step `region` as a `StepCircuit` (training_slowdown.capture.circuit): its values and the carried ones it reads,
    each op's gates split over its values by elements, every read of token data through a storage a step-R token
    input owns or by a token-index op (a sampled id's lookup's ids, ``meta["te_sampled"]``, token data as an input's
    are)."""
    ph = [n for n in nodes if n.op == "placeholder" and int(n.meta.get("te_step", -1)) == region]
    token_names = {n.name for n in ph if n.name.startswith("tokens_")}
    token_in = {v for v, k in flow.source.items() if k[0] == "input" and k[1] in token_names}
    tokdep = _closure(flow, token_in.__contains__)
    carried_set = set(carried)
    keep = sorted(carried_set | {v for v in range(len(flow.preds)) if flow.step[v] == region})
    ids = {v: i for i, v in enumerate(keep)}
    op_id = {n: i for i, n in enumerate(nodes)}
    read_next = np.zeros(len(keep), bool)
    for v, ps in enumerate(flow.preds):
        if flow.step[v] == region + 1:
            for p in ps:
                if p in ids:
                    read_next[ids[p]] = True
    made: dict = {}
    for v in keep:
        if v not in carried_set:
            made.setdefault(flow.node[v], []).append(v)
    gates = np.zeros(len(keep))
    kind = np.full(len(keep), COMPUTED, dtype=np.int8)
    for n, vs in made.items():
        elems = [max(flow.bits[v] // max(flow.width[v], 1), 1) for v in vs]
        for v, e in zip(vs, elems):
            gates[ids[v]] = node_gates.get(n, 0.0) * e / sum(elems)
        if n is not None and n.op == "call_function" and _pk(n.target) in _RANDOM:
            kind[[ids[v] for v in vs]] = RANDOM
    for v in keep:
        src = flow.source.get(v)
        if v in carried_set:
            kind[ids[v]] = CARRIED
        elif src is not None:
            kind[ids[v]] = ATTR if src[0] == "attr" else TOKEN_INPUT if v in token_in else OTHER_INPUT
    indptr, indices = [0], []
    outside = np.zeros(len(keep), bool)
    for v in keep:
        indices.extend(ids[p] for p in flow.preds[v] if p in ids)
        indptr.append(len(indices))
        outside[ids[v]] = any(p not in ids for p in flow.preds[v])
    keys: dict[str, int] = {}
    token_key: list[bool] = []
    tr_op, tr_key, tr_index, tr_pos, tr_indptr = [], [], [], [], [0]
    for n in nodes:
        if (n.op != "call_function" or int(n.meta.get("te_step", 0)) != region
                or not isinstance(n.target, torch._ops.OpOverload) or _is_view(n.target)):
            continue
        name = _pk(n.target)
        for a in _arg_nodes(n):
            val, key = a.meta.get("val"), flow.base.get(a)
            if not isinstance(val, torch.Tensor) or not isinstance(key, tuple):
                continue
            owned = key[0] == "input" and key[1] in token_names
            sampled = a.name in n.meta.get("te_sampled", ())
            index = name in _TOKEN_INDEX and not val.dtype.is_floating_point and val.dtype != torch.bool \
                and any(tokdep[x] for x in flow.reads.get((n, a), ()))
            if not owned and not index:
                continue
            k = keys.setdefault(repr(key), len(keys))
            if k == len(token_key):
                token_key.append(owned or sampled)
            elif sampled:
                token_key[k] = True
            pos = _storage_positions(val)
            tr_op.append(op_id[n])
            tr_key.append(k)
            tr_index.append(index)
            tr_pos.append(pos)
            tr_indptr.append(tr_indptr[-1] + len(pos))
    phase = np.array([_PHASE_CODE.get(flow.phase[v], 2) for v in keep], dtype=np.int8)
    return StepCircuit(
        indptr=np.array(indptr, dtype=np.int64), indices=np.array(indices, dtype=np.int64), phase=phase,
        tok=np.array([flow.tok[v] for v in keep], bool), dep=np.array([dep[v] for v in keep], bool),
        tokdep=np.array([tokdep[v] for v in keep], bool), kind=kind, outside=outside, read_next=read_next,
        bits=np.array([flow.bits[v] for v in keep], dtype=np.int64),
        width=np.array([flow.width[v] for v in keep], dtype=np.int32), gates=gates,
        node=np.array([op_id.get(flow.node[v], -1) if v not in carried_set else -1 for v in keep], dtype=np.int64),
        tr_op=np.array(tr_op, dtype=np.int64), tr_key=np.array(tr_key, dtype=np.int64), tr_index=np.array(tr_index, bool),
        tr_indptr=np.array(tr_indptr, dtype=np.int64),
        tr_pos=np.concatenate(tr_pos) if tr_pos else np.zeros(0, dtype=np.int64),
        keys=np.array(list(keys), dtype=str), token_key=np.array(token_key, bool), region=region,
        step_tokens=int(step_tokens))



def _bit_sums(macs: list[int], tokens: list[int]):
    """The bitset m -> the sum of macs[b] / tokens[b] over its set bits b, exactly, in one pass over m's bytes: walking
    m bit by bit is quadratic in its width, which a generation round's decode makes wide."""
    width, k = (len(macs) + 7) // 8, len(macs)
    vals = np.array(macs, dtype=np.int64 if sum(macs) < 2 ** 63 else object)
    groups = [(t, np.array(tokens) == t) for t in sorted(set(tokens))]

    def total(m: int) -> Fraction:
        if not m:
            return Fraction(0)
        on = np.unpackbits(np.frombuffer(m.to_bytes(width, "little"), dtype=np.uint8), bitorder="little")[:k].astype(bool)
        return sum((Fraction(int(vals[on & g].sum()), t) for t, g in groups), Fraction(0))
    return total


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
