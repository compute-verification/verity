"""A covert_witness program recorded from its start to one step, on the CPU with fake tensors: te_core 4.0's re-trace of
a run the runtime counter measured.

    python -m te_core.capture.script --script islands/CLASS/train.py --run RUN.json --step A --starts DIR \\
        [--out FILE] -- TRAIN_ARGS...

The program runs as te_core.run runs it (cw.py: prepare from the nearest directory above the script, the script as
``__main__`` with its arguments, CW_TIER at the tier the record was scored at, `tier_of`), with every tensor a fake (a
shape and a dtype, no values), until step A + 1 ends, or to its end, where its last step ends as te_core.run's `close`
ends it (so A may be its last step, which no step follows:
its Δ_grad ends where its update meets the carried values, and te_core.capture.structure's cut, to the values a next
step's forward reads, is 0). Each ATen op it runs is a node of one graph (`Trace`, its ``region`` A) with its phase
(``backward`` inside `torch.autograd.backward` or `torch.autograd.grad`, ``optimizer`` inside any `Optimizer.step`,
else ``forward``) and its step, cut as the harness cuts a standalone program's: at each `TrainData.next` once the step
in progress has read tokens (`te_core.flops.TOKEN_INDEX`), or at `hooks.step()`'s boundaries. prepare's start, data
and output calls are stubbed:

- `start_record`, `load_theta` and `load_start`: the start's record (``--starts``), θ as fakes of its tensors, the model
  built from its config as prepare builds it;
- `TrainData.next(n)` and `hooks.tokens(t)`: a graph input of token data, n windows of ctx + 1 tokens, as is a tensor
  made (``torch.from_numpy``, ``as_tensor``) from the stream's own windows (`TrainData.tokens` or its ``parts``, read
  through numpy); the stream is zeros in prepare's parts, starts and order: for data v3's tokenizer, the parts each
  start read names (its provenance's ``data.parts``), listed with one unread part in a manifest at ``data_v3.TRAIN``
  as on a node, else the unread part alone, as long as the run record's steps read with headroom (`stream_tokens`);
- the encoders run on the fakes unrecorded (E is the run's output, which its steps don't read back), so a program that
  sizes E by encoding it (a rank or a width to fit its class's cap) sees the length it saw for a format that stores its
  tensors raw; a compressed format's zeros compress further, and an encoder that can't run on fakes returns a stub;
  `prepare._quant_array` (CWQUAN1's arrays, which a program calls to size E without an encode) runs unrecorded too,
  with no stub (`_E_PARTS`);
- `decode`, `same_bits`, `submit` and `declare_step`: no tensor work besides decode's copy of θ;
- devices: prepare.DEVICE stays ``cuda``, so the program takes its GPU branches, and every tensor, generator and
  autocast region is the CPU's; `torch.compile` returns its function; attention runs on its math backend;
- time: a clock that moves by each recorded step's wall seconds (``--run``), so a program on a time budget plans as it
  did.

A Python number read from a tensor (`item`, `float`, `int`, `bool`, `tolist`, `numpy`) has no value: it is a fixed
stand-in, and a float carries the nodes it was read from into the ops it reaches (``meta["te_scalar_deps"]``) through
arithmetic, `math`'s functions and ``float(...)`` (whose calls in the program and prepare are rewritten to keep it).
Anything else a stand-in decides is the stand-in's, not the run's, so the recording fails closed: each read-back whose
value the graph can't carry is listed in the measured step's quantities (`formalism["data_dependent"]`, its kind, the
program's line and the steps it ran in; te_core.formalism then refuses the record's measured quantities):

- a comparison, truth test or integer conversion of a read-back float (``min``/``max``, ``if``, ``round``, ``int``),
  and `math` or numpy reading one into anything but a float;
- ``bool()``, ``int()`` or ``__index__`` of a tensor, an integer or boolean tensor read back (``item``, ``tolist``), any
  ``numpy()`` of a tensor the program made, and a read-back inside an op (C++ code reading a value).

A Python float folded over steps (an EMA of read-back estimates) carries every earlier step's read-back into the ops
it reaches as values of their own, so its Δ_grad grows with the step though the program carries one number: an op
reached by read-backs from two or more earlier steps is listed too (`formalism["carried_scalars"]`), and a program
keeps such state as a device tensor (a 0-dim fp32 tensor updated in place), which the graph carries as one value. An op
whose output size is set by values (``nonzero``, ``unique``, ``masked_select``, ``x[mask]``) is taken at its largest. The run is
recorded again with ctx halved (prepare.EVAL_CTX and every window), for its token axes, or, when that changes its steps (a
plan read off EVAL_CTX), with only the stream's windows halved (axis ``window``). A measured step that feeds sampled ids
back (a generation round's decode: in its forward, a lookup by an index computed from what a lookup by the stream's ids
made, of a table such a lookup reads) counts their positions among its tokens (``meta["te_sampled"]``,
te_core.capture.structure), and is recorded once more with twice the windows it reads from the stream (axis
``sequences``), which puts each decode position on a token axis; it is refused if a sampled id keeps its shape there.
The measured step's quantities
keep its matrix products as te_core.io_bound reads them (`formalism["products"]`, te_core.capture.products' rows).
`record(...)` returns step A's `StepGraph`; the CLI writes its structure (te_core.flops's record form) and its forward's GEMM MACs beside the record's
forward MACs (its backward's differ by the recompute the GPU's fused attention runs and the math backend doesn't).
"""
from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import json
import math
import operator
import os
import shutil
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils._pytree as pytree
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakTensorKeyDictionary

from te_core import VERSION
from te_core.capture import products
from te_core.capture.structure import StepGraph, _shape
from te_core.capture.trace import CaptureRefused, Trace
from te_core.flops import GATHERS, TOKEN_INDEX, _is_view
from te_core.formalism import READ_BACKS
from te_core.opnames import gemm_like

__all__ = ["Scalar", "record"]

aten = torch.ops.aten
_ENCODERS = ("encode_full", "encode_lora", "encode_lora2", "encode_seedlog", "encode_sparse", "encode_quant",
             "encode_model")
#: prepare's serialization of E's parts, which a program calls to size E without encoding it (an encode in the steps
#: is a checkpoint under te_core.run): its checks and bytes are E's, unrecorded as the encoders' are
_E_PARTS = ("_quant_array",)
#: the stubbed data stream's unread tokens at least, and over the run record's (`stream_tokens`): a step's record can
#: count fewer tokens than it reads (one micro-batch's of four, for a batch of 128 windows read 32 at a time)
STREAM_TOKENS = 3 * 10 ** 9
STREAM_HEADROOM = 8
_NP = {torch.float32: np.float32, torch.float64: np.float64, torch.float16: np.float16, torch.int64: np.int64,
       torch.int32: np.int32, torch.int16: np.int16, torch.int8: np.int8, torch.uint8: np.uint8, torch.bool: np.bool_}


class _Stop(BaseException):
    """The recorded steps are done."""


#: the recording in progress (`_patched`), which a read-back's events go to
_ACTIVE = None
#: te_core's modules' directories, and the trees under which a frame is never the program's line (`_where`)
_TE_CORE = (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1])
_LIBRARIES = tuple(str(Path(m.__file__).resolve().parent) for m in (torch, np, os))


def _where() -> str | None:
    """The program's innermost line on the stack: the first frame outside te_core's modules, torch, numpy and the
    standard library."""
    f = sys._getframe(1)
    while f is not None:
        name = f.f_code.co_filename
        path = Path(name).resolve()
        if not name.startswith("<") and path.parent not in _TE_CORE and not str(path).startswith(_LIBRARIES):
            return f"{path.name}:{f.f_lineno}"
        f = f.f_back
    return None


def _read_back(kind: str, *values) -> None:
    """A value read back from a traced tensor decides something the graph can't carry (module docstring): listed on the
    recording in progress when any of `values` carries graph nodes (none given: the read-back itself)."""
    if _ACTIVE is not None and (not values or any(getattr(v, "deps", None) for v in values)):
        _ACTIVE.data_dependent(kind)


class Scalar(float):
    """A float read from a tensor: a stand-in value, carrying the graph nodes it was read from (`deps`)."""
    deps: frozenset = frozenset()
    __hash__ = float.__hash__

    def __new__(cls, v: float, deps=()):
        o = super().__new__(cls, v)
        o.deps = frozenset(deps)
        return o

    def __round__(self, ndigits=None):
        if ndigits is None:
            _read_back("round() of a read-back float", self)
            return round(float(self))
        return Scalar(round(float(self), ndigits), self.deps)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _numpy_on(getattr(ufunc, method), inputs, kwargs)

    def __array_function__(self, func, types, args, kwargs):
        return _numpy_on(func, args, kwargs)


def _plain(x):
    return float(x) if isinstance(x, Scalar) else x


def _deps_of(tree) -> frozenset:
    return frozenset().union(*(x.deps for x in pytree.tree_leaves(tree) if isinstance(x, Scalar)))


def _numpy_on(f, args, kwargs):
    """numpy reading read-back floats: a float result keeps their nodes, anything else is listed (module docstring)."""
    d = _deps_of((args, kwargs))
    r = f(*pytree.tree_map(_plain, args), **pytree.tree_map(_plain, kwargs))
    if isinstance(r, (float, np.floating)) and not isinstance(r, bool):
        return Scalar(float(r), d)
    if d:
        _read_back(f"numpy {getattr(f, '__name__', f)} of a read-back float")
    return r


def _lift(name: str):
    f = getattr(float, name)

    def op(self, *other):
        r = f(self, *other)
        if r is NotImplemented or not isinstance(r, float):
            return r
        d = set(self.deps)
        for o in other:
            d |= getattr(o, "deps", frozenset())
        return Scalar(r, d)
    return op


def _decides(name: str, what: str):
    """A float method whose result the graph can't carry: listed when an operand is a read-back."""
    f = getattr(float, name)

    def op(self, *other):
        r = f(self, *other)
        if r is not NotImplemented:
            _read_back(what, self, *other)
        return r
    return op


for _n in ("__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__truediv__", "__rtruediv__",
           "__pow__", "__rpow__", "__floordiv__", "__rfloordiv__", "__mod__", "__rmod__", "__neg__", "__pos__", "__abs__"):
    setattr(Scalar, _n, _lift(_n))
for _n in ("__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"):
    setattr(Scalar, _n, _decides(_n, "a comparison of a read-back float"))
for _n, _what in (("__bool__", "a truth test of a read-back float"), ("__int__", "int() of a read-back float"),
                  ("__trunc__", "an integer of a read-back float"), ("__floor__", "an integer of a read-back float"),
                  ("__ceil__", "an integer of a read-back float"), ("__divmod__", "divmod() of a read-back float"),
                  ("__rdivmod__", "divmod() of a read-back float"), ("is_integer", "is_integer() of a read-back float"),
                  ("as_integer_ratio", "as_integer_ratio() of a read-back float")):
    setattr(Scalar, _n, _decides(_n, _what))


def _lift_math(f):
    """A `math` function on read-back floats: a float result keeps their nodes, anything else is listed."""
    def op(*a, **k):
        r = f(*a, **k)
        d = _deps_of((a, k))
        if d and isinstance(r, float):
            return Scalar(r, d)
        if d:
            _read_back(f"math.{f.__name__} of a read-back float")
        return r
    return op


def _te_float(x=0.0):
    """``float(x)`` in the recorded program and prepare (`_exec_lifted`): a Scalar, or a fake tensor's stand-in, kept."""
    if isinstance(x, Scalar):
        return x
    if isinstance(x, torch.Tensor):
        v = x.__float__()
        return v if isinstance(v, float) else float(v)
    return float(x)


class _Floats(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "float":
            node.func = ast.copy_location(ast.Name("_te_float", ast.Load()), node.func)
        return node


def _exec_lifted(path: Path, ns: dict) -> None:
    """Run the source at `path` in `ns` with each ``float(...)`` call `_te_float`'s: Python's float() returns a float
    subclass's plain value, which drops a Scalar's deps (``float`` as a type, ``isinstance(x, float)``, is unchanged)."""
    tree = ast.fix_missing_locations(_Floats().visit(ast.parse(path.read_text(), str(path))))
    ns["_te_float"] = _te_float
    exec(compile(tree, str(path), "exec"), ns)  # noqa: S102 - the recorder runs the program it records


def _skey(t: torch.Tensor):
    try:
        return t.untyped_storage()._cdata
    except (RuntimeError, NotImplementedError):
        return ("id", id(t))


def _of_stream(x, step: int) -> bool:
    """Whether node `x` is a view of a token input of step `step` (its base, as te_core.capture.structure's dataflow
    reads it)."""
    while isinstance(x, torch.fx.Node) and x.op == "call_function" and x.args and (
            x.target is operator.getitem or (isinstance(x.target, torch._ops.OpOverload) and _is_view(x.target))):
        x = x.args[0]
    return isinstance(x, torch.fx.Node) and x.op == "placeholder" and x.name.startswith("tokens_") \
        and x.meta.get("te_step") == step


def _cpu(x):
    if isinstance(x, torch.device) and x.type == "cuda":
        return torch.device("cpu")
    if isinstance(x, str) and (x == "cuda" or x.startswith("cuda:")):
        return "cpu"
    return x


class _Gen(torch.Generator):
    """`torch.Generator` on the CPU whatever device it is asked for."""

    def __new__(cls, device="cpu"):
        return super().__new__(cls, device="cpu")

    def __init__(self, device="cpu") -> None:
        pass


class _Autocast(torch.autocast):
    def __init__(self, device_type: str = "cuda", *args, **kwargs) -> None:
        super().__init__("cpu" if str(device_type).startswith("cuda") else device_type, *args, **kwargs)


class _Recorder:
    """The graph being recorded and the program's state the stubs read (module docstring)."""

    def __init__(self, stop: int, wall: list[float]) -> None:
        self.g = torch.fx.Graph()
        self.root = torch.nn.Module()
        self.tensors = WeakTensorKeyDictionary()   # tensor -> (node, its storage when mapped)
        self.owner: dict = {}                      # storage -> the node that made it
        self.storages: dict = {}                   # storage -> itself, held while `owner` names it (`_hold`)
        self.named = WeakTensorKeyDictionary()     # tensor -> (kind, name): θ, the model's parameters and buffers
        self.attrs: dict = {}
        self.keep: list = []
        self.phases = ["forward"]
        self.step, self.stop, self.wall = -1, stop, wall
        self.explicit = False
        self.read_tokens = False
        self.tok: set = set()
        self.scalars: list[frozenset] = []
        self.off = 0
        self.n_read = 0
        self.now = 1.0e6
        self.inputs = 0
        self.ctx = 0
        self.n_steps = 0
        self.stream_tokens = STREAM_TOKENS
        self.events: dict[tuple, dict] = {}
        # the measured step's forward lookups (`_lookup`): values a lookup by its stream's ids produced, the tables
        # those lookups read, and the lookups whose index was computed from such a value (node, index node, table)
        self.looked: set = set()
        self.tables: set = set()
        self.sampled: list[tuple] = []
        self.sequences, self.windows = 1, 0

    # -- read-backs the graph can't carry (module docstring)
    def _event(self, family: str, kind: str, **extra) -> None:
        if self.off:
            return
        where = _where()
        e = self.events.setdefault((family, kind, where), {"kind": kind, "where": where, "steps": [self.step, self.step],
                                                           "count": 0} | extra)
        e["steps"][1], e["count"] = self.step, e["count"] + 1
        for k, v in extra.items():
            e[k] = max(e[k], v)

    def data_dependent(self, kind: str) -> None:
        self._event("data_dependent", kind)

    def listed(self, family: str) -> list[dict]:
        return [e for (f, _, _), e in self.events.items() if f == family]

    # -- steps and time
    def clock(self) -> float:
        self.now += 1e-6
        return self.now

    def _end(self) -> None:
        if self.step >= self.stop:
            raise _Stop
        if 0 <= self.step < len(self.wall):
            self.now += float(self.wall[self.step] or 0.0)
        self.step += 1
        self.read_tokens = False

    def boundary(self) -> None:
        """`TrainData.next`: end the step in progress if it read tokens (te_core.run's `boundary`)."""
        if self.explicit:
            return
        if self.step < 0:
            self.step = 0
        elif self.read_tokens:
            self._end()

    @contextlib.contextmanager
    def explicit_step(self):
        """`hooks.step()` (te_core.run's `explicit_step`)."""
        if self.step >= 0 and self.read_tokens and not self.explicit:
            self._end()
        self.explicit = True
        if self.step < 0:
            self.step = 0
        yield self.step
        self._end()

    def close(self) -> None:
        """The program returned: its last step ends (te_core.run's `close`)."""
        if not self.explicit and self.step >= 0 and self.read_tokens:
            self._end()

    @contextlib.contextmanager
    def phase(self, name: str):
        self.phases.append(name)
        try:
            yield
        finally:
            self.phases.pop()

    @contextlib.contextmanager
    def suspended(self):
        self.off += 1
        try:
            yield
        finally:
            self.off -= 1

    def dummy(self) -> float:
        self.n_read += 1
        return 1.0 + 1e-3 * self.n_read

    # -- nodes
    def _meta(self, n, val) -> None:
        n.meta.update(val=val, te_phase=self.phases[-1], te_step=self.step)

    def _hold(self, t: torch.Tensor, sk) -> None:
        """Keep the storage `t` is mapped at until the recording ends. `owner` names a storage by its address (`_skey`)
        and `keep` holds the tensor, not its storage: a program that replaces a mapped tensor's storage
        (``p.data = ...``) would free it while `owner` still names it, and a tensor the recorder has not seen, allocated
        at that address later, would be recorded as a view of the old owner, in one recording and not in the next."""
        if not isinstance(sk, tuple) and sk not in self.storages:
            self.storages[sk] = t.untyped_storage()

    def node_of(self, t: torch.Tensor):
        sk = _skey(t)
        e = self.tensors.get(t)
        if e is not None and e[1] == sk:
            return e[0]
        self._hold(t, sk)
        own = self.owner.get(sk)
        if own is not None:
            n = self.g.call_function(aten.as_strided.default, (own, list(t.shape), list(t.stride()), t.storage_offset()))
            self._meta(n, t)
            if own in self.tok:
                self.tok.add(n)
            if own in self.looked:
                self.looked.add(n)
        else:
            name = f"_t{len(self.keep)}"
            setattr(self.root, name, t)
            n = self.g.get_attr(name)
            self._meta(n, t)
            info = self.named.get(t)
            if info is not None:
                self.attrs[name] = (info[0], info[1], ())
            self.owner[sk] = n
        self.tensors[t] = (n, sk)
        self.keep.append(t)
        return n

    def token_input(self, t: torch.Tensor) -> torch.Tensor:
        """`t` is token data: a graph input of the step in progress."""
        n = self.g.placeholder(f"tokens_{self.inputs}")
        self.inputs += 1
        self._meta(n, t)
        self.tok.add(n)
        sk = _skey(t)
        self._hold(t, sk)
        self.tensors[t] = (n, sk)
        self.owner.setdefault(sk, n)
        self.keep.append(t)
        return t

    def _bind(self, o: torch.Tensor, n) -> None:
        sk = _skey(o)
        self._hold(o, sk)
        self.tensors[o] = (n, sk)
        self.owner.setdefault(sk, n)
        self.keep.append(o)

    def add(self, func, args, kwargs, out) -> None:
        schema = func._schema
        writes = any(a.alias_info is not None and a.alias_info.is_write for a in schema.arguments)
        if not writes and not any(isinstance(x, torch.Tensor) for x in pytree.tree_leaves(out)):
            return
        def to_node(x):
            return self.node_of(x) if isinstance(x, torch.Tensor) else float(x) if isinstance(x, Scalar) else x

        a, k = pytree.tree_map(to_node, args), pytree.tree_map(to_node, kwargs)
        n = self.g.call_function(func, tuple(a), dict(k))
        self._meta(n, out)
        deps = set().union(*self.scalars) if self.scalars else set()
        if deps:
            n.meta["te_scalar_deps"] = sorted(deps, key=lambda d: d.name)
            earlier = {s for d in deps if 0 <= (s := int(d.meta.get("te_step", -1))) < self.step}
            if len(earlier) >= 2:
                self._event("carried_scalars", "a Python number read back in several earlier steps",
                            earlier_steps=len(earlier))
        ins = [x for x in pytree.tree_leaves((a, k)) if isinstance(x, torch.fx.Node)]
        tok = any(x in self.tok for x in ins)
        if tok:
            self.tok.add(n)
            if func.overloadpacket in TOKEN_INDEX:
                idx = pytree.tree_leaves(a[:1] if func.overloadpacket is aten.bincount else a[1:])
                if any(isinstance(x, torch.fx.Node) and x in self.tok for x in idx):
                    self.read_tokens = True
        looked = self._lookup(func, args, a, ins, n)
        if looked:
            self.looked.add(n)
        if isinstance(out, (tuple, list)):
            for i, o in enumerate(out):
                if isinstance(o, torch.Tensor):
                    gi = self.g.call_function(operator.getitem, (n, i))
                    self._meta(gi, o)
                    if tok:
                        self.tok.add(gi)
                    if looked:
                        self.looked.add(gi)
                    self._bind(o, gi)
        elif isinstance(out, torch.Tensor):
            self._bind(out, n)

    def _lookup(self, func, args, a, ins, n) -> bool:
        """Whether node `n` (`func` on `args`, `a` their nodes) makes a looked value: in the measured step's forward, a
        gather by the stream's ids (an index that is a view of a token input of the step), or an op reading a looked
        value. A gather of a table such a lookup reads, by a token-derived index computed from a looked value, is a
        sampled id's lookup (`sampled`): the model's own output fed back as its next token, as a generation round's
        decode reads it."""
        if self.step != self.stop - 1 or self.phases[-1] != "forward":
            return False
        looked = any(x in self.looked for x in ins)
        if func.overloadpacket in GATHERS and isinstance(args[0], torch.Tensor):
            table = _skey(args[0])
            for t, x in zip(pytree.tree_leaves(args[1:]), pytree.tree_leaves(a[1:])):
                if not isinstance(t, torch.Tensor) or t.dtype.is_floating_point or t.dtype == torch.bool \
                        or x not in self.tok:
                    continue
                if _of_stream(x, self.step):
                    self.tables.add(table)
                    looked = True
                elif x in self.looked:
                    self.sampled.append((n, x, table))
        return looked

    def mark_sampled(self) -> int:
        """Name each sampled id's lookup's index nodes in its ``meta["te_sampled"]`` (te_core.capture.structure counts
        their positions as the step's tokens), for the lookups of a table a lookup by the stream's ids read; how many."""
        marked = 0
        for n, x, table in self.sampled:
            if table in self.tables and x.name not in n.meta.setdefault("te_sampled", []):
                n.meta["te_sampled"].append(x.name)
                marked += 1
        return marked

    def stand_in(self, t: torch.Tensor, how: str | None = None):
        """A Python number read from `t` (module docstring); an integer or boolean one read by `how` is listed."""
        if t.dtype.is_floating_point or t.dtype.is_complex:
            return Scalar(self.dummy(), (self.node_of(t),) if not self.off else ())
        if how:
            self.data_dependent(f"{how} of a {str(t.dtype).removeprefix('torch.')} tensor")
        return True if t.dtype == torch.bool else 1


def _dynamic(func, args, kwargs):
    """An op whose output size is set by values, at its largest (module docstring); None if it isn't one."""
    p = func.overloadpacket
    x = args[0]
    if p in (aten.nonzero, aten.argwhere):
        return torch.empty((x.numel(), x.dim()), dtype=torch.int64, device=x.device)
    if p is aten.masked_select:
        n = torch.broadcast_shapes(x.shape, args[1].shape).numel()
        return torch.empty((n,), dtype=x.dtype, device=x.device)
    if p is aten._unique2:
        inv = kwargs.get("return_inverse", args[2] if len(args) > 2 else False)
        cnt = kwargs.get("return_counts", args[3] if len(args) > 3 else False)
        return (torch.empty((x.numel(),), dtype=x.dtype, device=x.device),
                torch.empty(x.shape if inv else (0,), dtype=torch.int64, device=x.device),
                torch.empty((x.numel(),) if cnt else (0,), dtype=torch.int64, device=x.device))
    if p in (aten.unique_dim, aten.unique_consecutive, aten.unique_dim_consecutive):
        return (torch.empty_like(x), torch.empty((x.numel(),), dtype=torch.int64, device=x.device),
                torch.empty((x.numel(),), dtype=torch.int64, device=x.device))
    if p is aten.repeat_interleave and isinstance(x, torch.Tensor) and not x.dtype.is_floating_point and len(args) == 1:
        return torch.empty((x.numel(),), dtype=torch.int64, device=x.device)
    masks = [i for i in (args[1] if p is aten.index else ()) if isinstance(i, torch.Tensor) and i.dtype in (torch.bool,
                                                                                                           torch.uint8)]
    if masks and all(m.dim() for m in masks):
        idx = []
        for i in args[1]:
            if isinstance(i, torch.Tensor) and i.dtype in (torch.bool, torch.uint8):
                idx += [torch.zeros((i.numel(),), dtype=torch.int64, device=x.device)] * i.dim()
            else:
                idx.append(i)
        return func(x, idx)
    return None


class _Dispatch(TorchDispatchMode):
    def __init__(self, rec: _Recorder) -> None:
        super().__init__()
        self.rec = rec

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        rec = self.rec
        if func is aten._local_scalar_dense.default:
            rec.data_dependent("a value read back inside an op")
            return rec.stand_in(args[0])
        if func.namespace == "prim" or rec.off:
            return func(*args, **kwargs)
        out = _dynamic(func, args, kwargs)
        if out is None:
            out = func(*args, **kwargs)
        rec.add(func, args, kwargs, out)
        return out


class _Function(TorchFunctionMode):
    """Every device the CPU's, and a float read from a tensor carried into the ops it reaches (module docstring)."""

    def __init__(self, rec: _Recorder) -> None:
        super().__init__()
        self.rec = rec

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (torch.Tensor.cuda, torch.Tensor.pin_memory):
            return args[0]
        args, kwargs = pytree.tree_map(_cpu, (args, kwargs))
        deps = frozenset().union(*(x.deps for x in pytree.tree_leaves((args, kwargs)) if isinstance(x, Scalar)))
        if deps:
            self.rec.scalars.append(deps)
        try:
            return func(*args, **kwargs)
        finally:
            if deps:
                self.rec.scalars.pop()


class _Tok(np.ndarray):
    """Windows read from the stubbed stream, kept through numpy's functions (`np.stack`, ``astype``), so a tensor made
    from them is token data."""

    def __array_function__(self, func, types, args, kwargs):
        out = super().__array_function__(func, types, args, kwargs)
        return out.view(_Tok) if type(out) is np.ndarray else out


class _Zeros:
    """A token stream of `n` zeros, read by slices."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, s):
        if isinstance(s, slice):
            return np.zeros(len(range(*s.indices(self.n))), dtype=np.uint16).view(_Tok)
        return np.uint16(0)


@contextlib.contextmanager
def _patched(rec: _Recorder, prepare, hooks, starts: Path, scale: int, windows_only: bool = False):
    """The stubs and patches of the module docstring, for one recording; `windows_only` divides the stream's windows by
    `scale` and leaves prepare.EVAL_CTX."""
    T = torch.Tensor
    saved: list[tuple[Any, str, Any]] = []

    def put(obj, name, value):
        saved.append((obj, name, getattr(obj, name, None)))
        setattr(obj, name, value)

    def _item(self):
        return rec.stand_in(self, "item()")

    def _float(self):
        v = rec.stand_in(self, "float()")
        return v if isinstance(v, float) else float(v)

    def _int(self):
        rec.data_dependent("int() of a tensor")
        return int(float(rec.stand_in(self)))

    def _bool(self):
        rec.data_dependent("bool() of a tensor (a branch on its value)")
        return True

    def _numpy(self):
        if self.dtype not in _NP:
            raise TypeError(f"Got unsupported ScalarType {self.dtype}")
        rec.data_dependent("numpy() of a tensor")
        return np.zeros(tuple(self.shape), dtype=_NP[self.dtype])

    def _tolist(self):
        flat = [rec.stand_in(self, "tolist()" if i == 0 else None) for i in range(max(self.numel(), 0))]
        if self.dim() == 0:
            return flat[0]
        arr = np.empty(tuple(self.shape), dtype=object)
        arr.reshape(-1)[:] = flat
        return arr.tolist()

    def wrapped(f, o):
        def wrap(self, *a, **k):
            return f(self) if isinstance(self, FakeTensor) else o(self, *a, **k)
        return wrap

    for name, f in (("item", _item), ("__float__", _float), ("__int__", _int), ("__bool__", _bool),
                    ("__index__", _int), ("tolist", _tolist), ("numpy", _numpy), ("__array__", _numpy)):
        put(T, name, wrapped(f, getattr(T, name)))

    ab, ag = torch.autograd.backward, torch.autograd.grad

    def backward(*a, **k):
        with rec.phase("backward"):
            return ab(*a, **k)

    def grad(*a, **k):
        with rec.phase("backward"):
            return ag(*a, **k)

    put(torch.autograd, "backward", backward)
    put(torch.autograd, "grad", grad)
    pre = register_optimizer_step_pre_hook(lambda *_: rec.phases.append("optimizer"))
    post = register_optimizer_step_post_hook(lambda *_: rec.phases.pop())
    put(torch, "compile", lambda f=None, *a, **k: f if f is not None else (lambda g: g))
    put(torch, "Generator", _Gen)
    put(torch, "autocast", _Autocast)
    put(torch.amp, "autocast", _Autocast)
    for name, value in (("synchronize", lambda *a, **k: None), ("empty_cache", lambda *a, **k: None),
                        ("max_memory_allocated", lambda *a, **k: 0), ("memory_allocated", lambda *a, **k: 0),
                        ("max_memory_reserved", lambda *a, **k: 0), ("reset_peak_memory_stats", lambda *a, **k: None),
                        ("mem_get_info", lambda *a, **k: (96 << 30, 96 << 30)),
                        ("get_device_name", lambda *a, **k: "NVIDIA RTX PRO 6000 Blackwell (recorded on CPU)")):
        put(torch.cuda, name, value)
    for name in ("time", "perf_counter", "monotonic"):
        put(time, name, rec.clock)
    put(time, "sleep", lambda s: None)
    for name in [n for n in dir(math) if not n.startswith("_") and callable(getattr(math, n))]:
        put(math, name, _lift_math(getattr(math, name)))
    put(sys.modules[__name__], "_ACTIVE", rec)

    def from_stream(f):
        def make(data, *a, **k):
            t = f(data, *a, **k)
            return rec.token_input(t) if isinstance(data, _Tok) and isinstance(t, torch.Tensor) else t
        return make

    for name in ("from_numpy", "as_tensor", "asarray", "tensor"):
        put(torch, name, from_stream(getattr(torch, name)))

    # prepare's start, data and outputs
    ctx = rec.ctx = int(prepare.EVAL_CTX) // scale
    if not windows_only:
        put(prepare, "EVAL_CTX", ctx)
    put(prepare, "DEVICE", "cuda")

    # data v3's stream: a manifest of the parts each start read names (its provenance's data.parts), then one unread part
    v3 = getattr(prepare, "data_v3", None)
    v3 = v3 if hasattr(v3, "TRAIN") and hasattr(v3, "TOKENIZER") else None
    read: dict[str, dict] = {}

    def manifest():
        parts = sorted(read.values(), key=lambda q: q["index"])
        unread = {"index": max((q["index"] for q in parts), default=-1) + 1, "sha256": "unread",
                  "tokens": rec.stream_tokens}
        (train / "manifest.json").write_text(json.dumps({"parts": [*parts, unread]}) + "\n")

    if v3 is not None:
        train = Path(tempfile.mkdtemp(prefix="te-capture-v3-"))
        put(v3, "TRAIN", train)
        manifest()

    def start_record(name):
        r = json.loads((starts / f"{name}.start.json").read_text())
        if v3 is not None and r.get("tokenizer") == v3.TOKENIZER:
            for q in ((r.get("provenance") or {}).get("data") or {}).get("parts") or []:
                if {"index", "sha256", "tokens"} <= set(q):
                    read.setdefault(q["sha256"], {k: q[k] for k in ("index", "sha256", "tokens")})
            manifest()
        return r

    def theta_of(name):
        rec_ = start_record(name)
        with rec.suspended():
            return {n: torch.empty(tuple(shape), dtype=getattr(torch, dt.removeprefix("torch.")))
                    for n, (dt, shape) in rec_["tensors"].items()}

    def load_theta(name, check=True):
        th = theta_of(name)
        for n, t in th.items():
            rec.named[t] = ("theta", n)
        return th

    def load_start(name, device=None):
        r = start_record(name)
        th = load_theta(name)
        with rec.suspended():
            net = prepare.Net(r["config"])
            with torch.no_grad():
                for n, p in net.params().items():
                    p.data = th[n].clone()
        for n, p in net.params().items():
            rec.named[p] = ("parameter", n)
        for n, b in net.named_buffers():
            rec.named[b] = ("buffer", n)
        return net, r, th

    base_td = prepare.TrainData

    class TrainData:
        def __init__(self, tokenizer, seed=1, ctx=None, parts=None):
            self.ctx = int(ctx) // scale if ctx is not None else rec.ctx
            sizes = [rec.stream_tokens]
            if v3 is not None and tokenizer == v3.TOKENIZER:
                sizes = [q["tokens"] for q in json.loads((train / "manifest.json").read_text())["parts"]][:parts]
            self.parts = [_Zeros(n) for n in sizes]
            self.tokens = _Zeros(sum(sizes))
            self.starts = np.cumsum([0] + [(n - 1) // self.ctx for n in sizes])
            self.windows, self.seed, self._order = int(self.starts[-1]), seed, None
            self.at = 0

        @property
        def order(self):
            """prepare's seeded permutation of the windows, made when a program first reads it: the stream's own
            reads need only its length, and a stream sized from a run record can hold more windows than fit in memory."""
            if self._order is None:
                self._order = np.random.default_rng(self.seed).permutation(self.windows)
            return self._order

        @order.setter
        def order(self, value):
            self._order = value

        def next(self, n, device=None):
            rec.boundary()
            if rec.step == rec.stop - 1:
                n = int(n) * rec.sequences
                rec.windows += n
            if self.at + n > (self.windows if self._order is None else len(self._order)):
                self.at = 0
            self.at += n
            with rec.suspended():
                t = torch.empty((int(n), self.ctx + 1), dtype=torch.int64)
            return rec.token_input(t)

    TrainData.__doc__ = base_td.__doc__
    put(prepare, "start_record", start_record)
    put(prepare, "load_theta", load_theta)
    put(prepare, "load_start", load_start)
    put(prepare, "TrainData", TrainData)
    def encoder(f, name):
        def encode(*a, **k):
            with rec.suspended():
                try:
                    return f(*a, **k)
                except Exception:  # noqa: BLE001 - an encoder that needs real values gets stand-in bytes
                    return b"CW" + name.encode() + b"\n"
        return encode

    for name in _ENCODERS:
        if hasattr(prepare, name):
            put(prepare, name, encoder(getattr(prepare, name), name))

    def unrecorded(f):
        def run(*a, **k):
            with rec.suspended():
                return f(*a, **k)
        return run

    for name in _E_PARTS:
        if hasattr(prepare, name):
            put(prepare, name, unrecorded(getattr(prepare, name)))
    put(prepare, "decode", lambda theta, blob, device=None: {n: v.detach().clone() for n, v in theta.items()})
    put(prepare, "same_bits", lambda a, b: True)
    put(prepare, "submit", lambda *a, **k: None)
    put(prepare, "declare_step", lambda *a, **k: None)

    class _Session:
        def plan(self, n):
            rec.n_steps = int(n)

        def output(self, fn):
            pass

        def current_step(self) -> int:
            return rec.step

        def progress_bucket(self) -> int:
            """te_core.run's: which 5% of the planned steps the step in progress is in, else its index (a capture
            counts no MACs, so a run planned by its MAC budget reads its index here)."""
            from te_core.run import CHECKPOINTS

            if rec.n_steps:
                return math.ceil((rec.step + 1) * CHECKPOINTS / rec.n_steps)
            return rec.step

        def explicit_step(self):
            return rec.explicit_step()

        def tokens(self, batch):
            return rec.token_input(batch)

        def family(self, name, plan, uniform=True):
            return hooks._NoFamily()

    put(hooks, "ACTIVE", _Session())
    try:
        yield
    finally:
        pre.remove()
        post.remove()
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)
        if v3 is not None:
            shutil.rmtree(train, ignore_errors=True)


def _record_once(script: Path, args: list[str], starts: Path, stop: int, wall: list[float], scale: int, log,
                 windows_only: bool = False, stream_tokens: int | None = None, sequences: int = 1) -> _Recorder:
    from te_core import cw, hooks

    island = script.resolve().parent
    for m in [m for m, mod in sys.modules.items() if str(getattr(mod, "__file__", "") or "").startswith(str(island))]:
        del sys.modules[m]
    argv0, path0 = sys.argv, list(sys.path)
    prepare = cw.import_prepare(script)
    _exec_lifted(Path(prepare.__file__), prepare.__dict__)
    rec = _Recorder(stop, wall)
    rec.sequences = sequences
    if stream_tokens is not None:
        rec.stream_tokens = int(stream_tokens)
    sys.argv = [str(script), *args]
    sys.path.insert(0, str(island))
    main, main0 = types.ModuleType("__main__"), sys.modules.get("__main__")
    main.__dict__.update(__file__=str(script), __builtins__=builtins)
    fake = FakeTensorMode(allow_non_fake_inputs=True)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    try:
        with _patched(rec, prepare, hooks, starts, scale, windows_only), contextlib.redirect_stdout(log), \
                sdpa_kernel([SDPBackend.MATH]), fake, _Function(rec), _Dispatch(rec):
            sys.modules["__main__"] = main
            try:
                try:
                    _exec_lifted(script, main.__dict__)
                except SystemExit as e:
                    if e.code not in (None, 0):
                        raise CaptureRefused("trace", None, f"the program exited {e.code} at step {rec.step}") from e
                rec.close()
            except _Stop:
                pass
    finally:
        sys.argv, sys.path[:] = argv0, path0
        sys.modules["__main__"] = main0
    if rec.step < stop:
        raise CaptureRefused("trace", None, f"the program ended after {max(rec.step, 0)} steps, before step {stop - 1} "
                                            f"ended")
    return rec


class _Graph(torch.nn.Module):
    """A recorded graph's module: its attributes, and `graph`."""

    def __init__(self, rec: _Recorder) -> None:
        super().__init__()
        self.__dict__["graph"] = rec.g
        for name in [n.target for n in rec.g.nodes if n.op == "get_attr"]:
            self.__dict__[name] = getattr(rec.root, name)


def stream_tokens(run: dict) -> int:
    """The stubbed stream's unread tokens for a run record: its steps' tokens, STREAM_HEADROOM times over, and at least
    STREAM_TOKENS. The sum, not the step count times the largest step: one scoring step can count thousands of training
    steps' tokens, and a program may make an array of every window in its stream (ref-data's ids)."""
    return max(STREAM_TOKENS, STREAM_HEADROOM * sum(int(s.get("tokens") or 0) for s in run.get("steps") or []))


def tier_of(run: dict) -> str | None:
    """The budget tier a run's record was scored at (evaluate.py's `evaluation.budget.tier`; te_core.run's top-level
    `tier` is the promotion tier, not this), None for a record no evaluation scored."""
    return ((run.get("evaluation") or {}).get("budget") or {}).get("tier")


@contextlib.contextmanager
def _tier(tier: str | None):
    """CW_TIER at `tier` while the program and the prepare it re-executes run (prepare reads it at import), restored
    after; None leaves the environment's."""
    if tier is None:
        yield
        return
    old = os.environ.get("CW_TIER")
    os.environ["CW_TIER"] = tier
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("CW_TIER", None)
        else:
            os.environ["CW_TIER"] = old


def record(script: Path, args: list[str], *, step: int, starts: Path, wall: list[float], step_tokens: int,
           log=None, name: str | None = None, tier: str | None = None, stream_tokens: int | None = None) -> StepGraph:
    """Step `step` of the program `script` run with `args` (module docstring), as a `StepGraph` whose tokens are
    `step_tokens` (the record's count for that step), at the budget tier `tier` (`tier_of` the record: a program that
    plans from prepare.budget plans the run's own steps only at the run's tier), its stubbed stream's unread part
    `stream_tokens` long (the module's `stream_tokens` of the record; None: STREAM_TOKENS). A step that feeds sampled
    ids back (module docstring) is recorded once more with twice its stream's windows (axis ``sequences``), and is
    refused if a sampled id keeps its shape there."""
    log = log if log is not None else open(os.devnull, "w")
    name = name or Path(script).parent.name
    listed: dict[str, list[dict]] = {k: [] for k in READ_BACKS}
    sampled = 0

    def trace(scale, windows_only=False, sequences=1):
        with _tier(tier):
            rec = _record_once(Path(script), list(args), Path(starts), step + 1, wall, scale, log, windows_only,
                               stream_tokens, sequences)
        for k, have in listed.items():
            seen = {(e["kind"], e["where"]) for e in have}
            have += [e for e in rec.listed(k) if (e["kind"], e["where"]) not in seen]
        nonlocal sampled
        if scale == sequences == 1 and not windows_only:
            sampled = rec.mark_sampled()
        return Trace(_Graph(rec), rec.attrs, [], axis_sizes=(rec.ctx, rec.windows) if sampled else (rec.ctx,),
                     step_tokens={step: step_tokens}, region=step)

    def fed_back(g: StepGraph) -> StepGraph:
        """`g`, once every sampled id the base trace reads changes its shape with the sequences."""
        if not seq:
            return g
        nodes, other = list(base.gm.graph.nodes), list(seq[0].gm.graph.nodes)
        ids = {x for n in nodes for x in n.meta.get("te_sampled", ())}
        fixed = [a.name for a, b in zip(nodes, other) if a.name in ids
                 and _shape(a.meta.get("val")) == _shape(b.meta.get("val"))]
        if fixed:
            raise CaptureRefused("batch_dependent", None, f"the step feeds sampled ids back, and with twice its stream's "
                                 f"windows its sampled ids {fixed[:4]} keep their shape, so their positions are on no "
                                 f"token axis")
        return g

    def flagged(g: StepGraph) -> StepGraph:
        rows = products.rows_on_dial(g) if g.formalism is not None else None
        if rows is not None:
            rows = rows | {"capture_code": VERSION}
        if g.circuit is not None:
            g.circuit.read_backs = {k: list(v) for k, v in listed.items()}
            g.circuit.products = rows
        if g.formalism is not None:
            g.formalism = g.formalism | listed | {"products": rows}
        return g

    base = trace(1)
    seq = [trace(1, sequences=2)] if sampled else []
    names = ("sequences",) if seq else ()
    try:
        g = StepGraph.build(base, [trace(2), *seq], ("ctx", *names), updated=None, name=name)
    except CaptureRefused as e:
        if e.reason != "batch_dependent":
            raise
        refused = e.with_traceback(None)
    else:
        return flagged(fed_back(g))
    # a program that plans its steps from prepare.EVAL_CTX (a FLOP budget over batch x ctx tokens) runs other steps when
    # it is halved; halving only the stream's windows keeps its plan, and holds only if every window steps A and A + 1
    # read is one of the stream's (a program that cuts its own windows at EVAL_CTX + 1 would read them unhalved)
    try:
        varied = trace(2, windows_only=True)
        g = StepGraph.build(base, [varied, *seq], ("window", *names), updated=None, name=name)
    except Exception:  # noqa: BLE001 - the fallback failing for any reason leaves the first refusal standing
        raise refused from None
    fixed = [a.name for a, b in zip(base.gm.graph.nodes, varied.gm.graph.nodes)
             if a.op == "placeholder" and a.name.startswith("tokens_") and int(a.meta.get("te_step", -1)) in (step, step + 1)
             and _shape(a.meta.get("val")) == _shape(b.meta.get("val"))]
    if fixed:
        raise CaptureRefused("batch_dependent", None, f"{refused}; with only the stream's windows halved, its token "
                             f"inputs {fixed[:4]} keep their shape") from None
    return flagged(fed_back(g))


def gemm_macs(g: StepGraph) -> int:
    return sum(m.macs for m in g.matmuls)


def recorded_forward_gemm_macs(run: dict, step: int) -> int:
    """The counter's forward GEMMs at `step`, from its structure's op log (te_core.flops), which `forward_gemm_macs`
    counts on the trace. The counter's whole forward (`recorded_forward_macs`) holds more that |C| counts and no
    product is: its forward elementwise work, and a trained table's lookup by token data at the gather Definition's
    multiplexers (12% of the forward of an 18-layer model of width 768 with a trained 32,000-row embedding)."""
    s = run["steps"][step]
    log = ((run.get("structures") or {}).get(s.get("struct")) or {}).get("log") or []
    return sum(r[5] for r in log if r[0] == "fwd" and gemm_like(r[2]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--script", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True, help="the run's run.json (te_core.flops's record)")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--starts", type=Path, required=True, help="a directory of NAME.start.json")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--log", type=Path, default=None, help="the program's standard output")
    ap.add_argument("--circuit", type=Path, default=None,
                    help="save the measured step's circuit (te_core.capture.circuit) to this .npz")
    argv = sys.argv[1:] if argv is None else argv
    split = argv.index("--") if "--" in argv else len(argv)
    a = ap.parse_args(argv[:split])
    if torch.__version__.split("+")[0] != products.TORCH:
        raise SystemExit(f"te_core.capture.script: torch {torch.__version__} here, but the capture is frozen at "
                         f"{products.CAPTURE} under torch {products.TORCH}, the trace job's (products.TORCH): trace "
                         f"under torch=={products.TORCH}+cpu")
    run = json.loads(a.run.read_text())
    steps = run["steps"]
    wall = [float(s.get("wall_s") or 0.0) for s in steps]
    tier = tier_of(run)
    t0 = time.perf_counter()
    with (open(a.log, "w") if a.log else open(os.devnull, "w")) as log:
        g = record(a.script, argv[split + 1:], step=a.step, starts=a.starts, wall=wall,
                   step_tokens=int(steps[a.step]["tokens"]), log=log, tier=tier, stream_tokens=stream_tokens(run))
    by = steps[a.step].get("by") or {}
    out = {"step": a.step, "tier": tier, "structure": g.structure(), "gemm_macs": gemm_macs(g),
           "recorded_gemm_macs": int(steps[a.step]["matmul_macs"]),
           "forward_gemm_macs": g.phase_macs().get("forward", 0),
           "recorded_forward_gemm_macs": recorded_forward_gemm_macs(run, a.step),
           "recorded_forward_macs": sum(v for k, v in by.items() if k.startswith("fwd/")), "tokens": g.tokens, "axes": g.axes,
           "updated_macs": g.updated_macs(), "dependent": g.dependent[:40], "sinks": g.sinks[:40],
           "cut_bits": g.cut_bits, "beta": g.beta, "cut": g.cut[:12], "formalism": g.formalism,
           "seconds": round(time.perf_counter() - t0, 1)}
    if a.circuit and g.circuit is not None:
        g.circuit.save(a.circuit)
    text = json.dumps(out, indent=1, default=str)
    if a.out:
        a.out.write_text(text + "\n")
    f = g.formalism or {}
    print(json.dumps({k: out[k] for k in ("step", "forward_gemm_macs", "recorded_forward_gemm_macs", "recorded_forward_macs",
                                          "updated_macs", "cut_bits", "seconds")}
                     | {k: f.get(k) for k in ("affected_share", "token_factor", "a_bits", "u1_bits", "tail_share", "delta_grad_bits")}
                     | {k: [f"{e['kind']} at {e['where']}" for e in f.get(k, [])][:8] for k in READ_BACKS}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
