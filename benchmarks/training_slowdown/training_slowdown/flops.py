"""The harness's own count of a run's circuit C: every multiplication the process computes through torch, by its shape,
with the structure the annotation reads. Nothing a program reports about its compute is read.

`Counter` is a TorchDispatchMode installed around the whole program (run.py), so it sees every aten op on every device,
forward, backward (autograd's threads inherit the mode; a program's own threads don't, and run.py refuses them) and
optimizer alike. It counts, in multiply-accumulates (MACs):

- matmuls (`mm`, `addmm`, `bmm`, `baddbmm`, `dot`, `mv`, `addmv`, `addbmm`, `addr`, `_int_mm`, `_scaled_mm`, their
  in-place forms) by shape, and convolutions by torch's FLOP formulas (`torch.utils.flop_counter`) halved; distances
  (`cdist`, `pdist`) at one MAC per coordinate pair, and `embedding_bag` at one per index and column;
- a fused attention kernel as the ops the math kernel computes attention by, which are the ones a step's trace holds
  (`Counter._attention`): the forward's two scalings and two products, and in the backward the products and scalings
  of the gradients the step asks for. A job's record has the same ops, shapes and MACs under either kernel. The
  dtype the kernel computes them in is not the same: a fused kernel's is the query's, and the math kernel computes
  a 16-bit call in float32;
- elementwise multiplications (`mul`, `div`, `addcmul`, `addcdiv`, `lerp`, `add` with a scale, their in-place and
  `_foreach_` forms), one MAC per output element, so a matmul written as a broadcast product or quotient is counted
  too, and a fused optimizer step at a constant per parameter (`opnames.FUSED_OPTIMIZER_MACS`);
- dense linear algebra (`eigh`, `svd`, `qr`, `cholesky`, `inv`, solves), at n**3 per matrix, marked approximate;
- any other forward op that combines token-indexed data with an updated weight (an add, a norm with a trained gain,
  `exp` of a sum), at one MAC per element of its largest operand or result, since a product can be written through
  any of them (`x + w` then `exp`, `log`): it is an updated node. A gather (`embedding`, `index`, ...) moves its
  source's rows without computing on them, so only its indices combine.

An op outside torch's own namespaces (a `torch.library` custom op, whose body runs where the mode can't see it,
`foreign_ops`) or an aten op whose name says it computes products but that has no formula here (an RNN kernel,
`_trilinear`, a packed matmul, `unknown_ops`) would count as zero; run.py makes such a run invalid. So does token data
read out of torch (`numpy`, `tolist`, DLPack: `guard_exits`) at one value or more per token of its step (`exited`), since
what is computed on it outside torch isn't counted.

|C| is their sum (`macs`). Each multiplication is a node classified by its operands' provenance, which the counter tracks
per tensor through every op within a step:

- *token-indexed*: computed from this step's token data (a batch from the start's data stream, `Counter.batch`, through
  any op; an integer tensor indexed into an embedding; or a batch the program passed to `Counter.tokens`);
- a weight's *roots*: the persistent tensors a parameter-indexed value is computed from that differ from θ. A registered
  weight (`register`) differs from θ when its value, checked bit for bit against θ at its first use after any write, isn't
  θ's. A persistent tensor the counter hasn't seen (made before this step, or outside torch's dispatch, as `p.data =
  p.data.float()` makes one) is θ's weight of its shape when it covers its storage and equals that weight exactly in both
  dtypes, and is then checked as a registered one; any other is a weight θ doesn't hold, so it differs. A tensor made
  inside the step by a factory op (`randn`, `zeros`, ...) is a fast value with no roots. The counter holds every tensor
  it has named a θ weight, so no other tensor takes its memory.
- a *constant*: a buffer of the registered model that θ doesn't hold (a position table, in the state dict or not),
  which `register` holds under its own name at the value it has then. It is checked and counted as a θ weight is: a
  product with it is θ's while it has that value, a buffer a step writes is a weight with roots from that write on
  (an updated weight, by its name), and a write the dispatcher doesn't see is caught like one into a weight
  (`unseen_writes`). It is not θ: no unseen tensor is taken for one, since a buffer's value (zeros) is not one only
  the model's buffer has.

A node is `updated` when it is in a forward pass (outside autograd's backward), one operand is token-indexed and another
(the weight, or an initial accumulator) has roots: the counter's updated multiplications, which were U before training_slowdown
4.0. 4.0 reads U by update dependence from the step's traced circuit (training_slowdown.capture, `grade.with_circuit`), and
this count stays |C|. Others are `theta` (the weight equals
θ's, or a fast value made in the step), `act` (two token-indexed operands: attention), `param` (no token-indexed
operand: optimizer work), and every product in autograd's backward is `bwd/grad`. Each updated node records its MACs per token row and the updated MACs per token
row upstream of it in its token's pass (the cone of updated forward nodes its operands were computed from). Every value
autograd's backward produces is a weight from then on (`derived`): a gradient used as a weight, directly or through an
out-of-place update, is updated. So is a token-derived operand shaped like a weight in a two-token-operand matmul (a
gradient computed by hand); a forward activation that happens to have a weight's 2-D shape is counted as one too. A
node's token rows are its token operand's rows (its elements, for integer token data), at most the step's tokens. A
forward gather (embedding, index, index_select, take, gather) of a weight with roots by token data is an updated node too,
at its position (so a tied embedding's update reaches every layer from the lookup on): its MACs are the multiplexers the
circuit's gather Definition spends on its result (`gather_macs`), one multiplication block (b·(x − y) + y) per word
selected per tree level. A gather of θ's table, or by indices that aren't token data, computes nothing on the token.

Tags are cleared at every step boundary, so a value carried from one step to the next is a weight from then on. Writes
the dispatcher doesn't see (a custom kernel into a weight's memory) are caught at each step's end, when every registered
weight is checked against θ again (`unseen_writes`). Work it can't see at all (custom GPU kernels) shows only as GPU time
the counted MACs don't explain, which `Flags` reports beside CPU time large relative to GPU time.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils import flop_counter
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.weak import WeakTensorKeyDictionary

from .opnames import (ELEMENTWISE_NAMES, FUSED_OPTIMIZER_MACS, GATHER_NAMES, GEMM_LIKE, LINALG_NAMES, NOT_GEMM,  # noqa: F401
                      SCALED_NAMES, gemm_like)

aten = torch.ops.aten


def _shape_macs(func, args) -> int:
    p = func.overloadpacket
    if p in (aten.mm, aten._int_mm):
        (m, k), (_, n) = args[0].shape, args[1].shape
        return m * k * n
    if p in (aten.addmm, aten.addmm_, aten._addmm_activation):
        (m, k), (_, n) = args[1].shape, args[2].shape
        return m * k * n
    if p is aten.bmm:
        (b, m, k), (_, _, n) = args[0].shape, args[1].shape
        return b * m * k * n
    if p in (aten.baddbmm, aten.baddbmm_, aten.addbmm, aten.addbmm_):
        (b, m, k), (_, _, n) = args[1].shape, args[2].shape
        return b * m * k * n
    if p in (aten.dot, aten.vdot):
        return args[0].numel()
    if p is aten.mv:
        return args[0].numel()
    if p in (aten.addmv, aten.addmv_):
        return args[1].numel()
    if p in ADDR:
        return args[1].numel() * args[2].numel()
    if p in (aten._cdist_forward, aten._euclidean_dist):
        (*batch, n, d), m = args[0].shape, args[1].shape[-2]
        return math.prod(batch) * n * m * d
    if p is aten._cdist_backward:
        (*batch, n, d), m = args[1].shape, args[2].shape[-2]
        return math.prod(batch) * n * m * d
    if p in (aten._pdist_forward, aten._pdist_backward):
        n, d = args[0 if p is aten._pdist_forward else 1].shape
        return n * (n - 1) // 2 * d
    if p in EMBEDDING_BAG:
        idx = next(t for t in _tensors(args) if not (t.is_floating_point() or t.is_complex()))
        return idx.numel() * args[0].shape[-1]
    raise KeyError(p)


def _ops(*names) -> set:
    return {getattr(aten, n) for n in names if hasattr(aten, n)}


EMBEDDING_BAG = _ops("_embedding_bag", "_embedding_bag_forward_only", "_embedding_bag_backward",
                     "_embedding_bag_dense_backward", "_embedding_bag_per_sample_weights_backward")
ADDR = _ops("addr", "addr_")
#: matmuls whose first operand is an accumulator
ACCUMULATING = _ops("addmm", "addmm_", "_addmm_activation", "baddbmm", "baddbmm_", "addbmm", "addbmm_", "addmv", "addmv_") | ADDR
SHAPE_OPS = _ops("mm", "_int_mm", "addmm", "addmm_", "_addmm_activation", "bmm", "baddbmm", "baddbmm_", "addbmm", "addbmm_",
                 "dot", "vdot", "mv", "addmv", "addmv_", "_cdist_forward", "_cdist_backward", "_euclidean_dist",
                 "_pdist_forward", "_pdist_backward") | ADDR | EMBEDDING_BAG
#: the fused attention kernels, forward and backward: each is counted as the math kernel's ops (`Counter._attention`),
#: by no formula of its own
FUSED_ATTENTION = _ops(*(f"_scaled_dot_product_{kernel}{part}" for part in ("", "_backward")
                         for kernel in ("flash_attention", "flash_attention_for_cpu", "efficient_attention",
                                        "cudnn_attention")))
#: torch's FLOP formulas, for the ops priced neither by shape nor as a fused attention kernel: convolutions
FLOP_FORMULAS = {p: f for p, f in flop_counter.flop_registry.items() if p not in FUSED_ATTENTION}
#: the op namespaces whose kernels the counter knows; any other (a `torch.library` custom op, whose body it can't see)
#: makes a run invalid
NAMESPACES = frozenset({"aten", "prims", "profiler"})
ELEMENTWISE = {getattr(aten, n) for n in ELEMENTWISE_NAMES}
SCALED = {getattr(aten, n) for n in SCALED_NAMES}
FUSED_OPTIMIZERS = {getattr(aten, n): k for n, k in FUSED_OPTIMIZER_MACS.items() if hasattr(aten, n)}
LINALG = {getattr(aten, n) for n in LINALG_NAMES if hasattr(aten, n)}
FACTORY = {getattr(aten, n) for n in ("randn", "rand", "randint", "randperm", "zeros", "ones", "full", "empty",
                                      "empty_strided", "arange", "linspace", "eye", "scalar_tensor", "normal",
                                      "zeros_like", "ones_like", "empty_like", "full_like", "randn_like", "rand_like")
           if hasattr(aten, n)}
EMBEDDING = {aten.embedding}
#: ops that read integer token data as indices: where a batch's tokens are counted
TOKEN_INDEX = {getattr(aten, n) for n in ("embedding", "_embedding_bag", "index", "index_select", "take", "gather",
                                          "scatter", "scatter_", "scatter_add", "scatter_add_", "scatter_reduce",
                                          "scatter_reduce_", "index_put", "index_put_", "index_add", "index_add_",
                                          "index_copy", "index_copy_", "bincount") if hasattr(aten, n)}
#: ops whose result is gathered from their first argument: their work is the result, not the whole source
GATHERS = _ops(*GATHER_NAMES)


#: a gather's index width and chunk as the circuit's gather Definition has them (verity's
#: `ml/definitions/registry/boolean_gather.py`: a 32-bit token, chunks of 1024 words by its low 10 bits)
GATHER_INDEX_BITS, GATHER_CHUNK_BITS = 32, 10


def mux_tree(n: int, k: int) -> int:
    """The multiplexers per word bit of `verity.circuits.boolean.gather`'s tree over n words by a k-bit index: at each
    level the words meet in pairs, one without a partner meeting the fill word."""
    out = 0
    for _ in range(k):
        n = (n + n % 2) // 2
        out += n
    return out


def gather_selects(n: int) -> int:
    """The multiplexers per selected word of the circuit's gather Definition over n words: one tree over the token's 32
    bits, or for n over a chunk, the whole chunks by the low 10 bits, the partial chunk likewise, then the chunks' words
    by the high 22."""
    chunk = 1 << GATHER_CHUNK_BITS
    if n <= chunk:
        return mux_tree(n, GATHER_INDEX_BITS)
    full, r = divmod(n, chunk)
    return (full * mux_tree(chunk, GATHER_CHUNK_BITS) + (mux_tree(r, GATHER_CHUNK_BITS) if r else 0)
            + mux_tree(full + (1 if r else 0), GATHER_INDEX_BITS - GATHER_CHUNK_BITS))


def gather_candidates(name: str, shape, arg1) -> int:
    """How many words each element of a gather's result is selected from: the table's rows (embedding), the indexed dims
    (index), the dim (index_select, gather), or every element (take)."""
    shape = list(shape)
    if name == "embedding":
        return shape[0]
    if name == "index":
        return math.prod(n for n, ix in zip(shape, arg1) if ix is not None)
    if name in ("index_select", "gather"):
        return shape[arg1]
    return math.prod(shape)


def gather_macs(name: str, shape, arg1, out_numel: int) -> int:
    return gather_selects(max(gather_candidates(name, shape, arg1), 1)) * out_numel


def _is_view(func) -> bool:
    ret = func._schema.returns
    return bool(ret) and ret[0].alias_info is not None and not ret[0].alias_info.is_write


def _work(func, args, kwargs, out) -> int:
    """An op's work with no multiplication formula: one per element of its largest operand or result (of its index
    operands and result, for a gather)."""
    p = func.overloadpacket
    if p in TOKEN_INDEX:
        sizes = [t.numel() for t in _tensors((args[1:], kwargs))]
        sizes += [t.numel() for t in _tensors(out)] if p in GATHERS else []
    else:
        sizes = [t.numel() for t in _tensors((args, kwargs, out))]
    return max(sizes, default=0)


def _tensors(x) -> list[torch.Tensor]:
    return [t for t in tree_flatten(x)[0] if isinstance(t, torch.Tensor)]


def _ptr(t: torch.Tensor) -> int:
    """`t`'s storage's address; 0 for a tensor with no storage (a sparse or a wrapper tensor)."""
    return t.untyped_storage().data_ptr() if t.layout == torch.strided and torch._C._has_storage(t) else 0


def read_positions(t: torch.Tensor) -> np.ndarray:
    """The storage positions `t`'s elements occupy (its offset and strides; no data read), sorted, each once."""
    pos = np.array(t.storage_offset(), dtype=np.int64)
    for n, s in zip(t.shape, t.stride()):
        pos = pos[..., None] + np.arange(n, dtype=np.int64) * s
    return np.unique(pos)


def _written(func, args, kwargs) -> list[torch.Tensor]:
    out = []
    for i, a in enumerate(func._schema.arguments):
        if a.alias_info is not None and a.alias_info.is_write:
            v = args[i] if i < len(args) else kwargs.get(a.name)
            out.extend(_tensors(v))
    return out


PLAIN = (type(None), bool, int, float, complex, str, bytes, torch.dtype, torch.device)


def plain(x):
    """`x` rebuilt from plain values (dicts, lists, tuples, detached `torch.Tensor`s, numbers, strings, bytes, paths). The
    harness converts what the program hands it while the count still runs, so no method of the program's objects runs
    in the harness's uncounted work."""
    t = type(x)
    if t in PLAIN:
        return x
    if isinstance(x, torch.Tensor):
        y = x.detach()
        if type(y) is not torch.Tensor:
            raise TypeError(f"training_slowdown: a {t.__name__} is not a plain tensor; hand the harness torch.Tensor values")
        return y
    if isinstance(x, Mapping):
        return {plain(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return (list if isinstance(x, list) else tuple)(plain(v) for v in x)
    if isinstance(x, np.generic):
        return plain(x.item())
    if isinstance(x, np.ndarray) and x.dtype != object:
        return np.array(x)
    if isinstance(x, (bytearray, memoryview)):
        return bytes(x)
    if isinstance(x, os.PathLike):
        return Path(plain(os.fspath(x)))
    for base in (int, float, complex, str, bytes):
        if isinstance(x, base) and type(y := base(x)) is base:
            return y
    raise TypeError(f"training_slowdown: the harness takes plain values, not a {t.__name__}")


def digest(t: torch.Tensor) -> str:
    """sha256 of a tensor's dtype, shape and bytes: θ's and an output model's identity."""
    t = t.detach().contiguous()
    h = hashlib.sha256(f"{str(t.dtype)}\0{tuple(t.shape)}\0".encode())
    h.update(t.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def _signed(x: int) -> int:
    x &= (1 << 64) - 1
    return x - (1 << 64) if x >> 63 else x


_MIX1, _MIX2, _GOLDEN = _signed(0xBF58476D1CE4E5B9), _signed(0x94D049BB133111EB), _signed(0x9E3779B97F4A7C15)


def _mix(x: torch.Tensor) -> torch.Tensor:
    """splitmix64's finalizer on int64 (wrapping multiplies, logical shifts)."""
    for s, m in ((30, _MIX1), (27, _MIX2)):
        x = (x ^ ((x >> s) & ((1 << (64 - s)) - 1))) * m
    return x ^ ((x >> 31) & ((1 << 33) - 1))


class KeyedDigest:
    """The value digest a counterexample's rebuild is checked against: Σ_i w_i·k_i mod 2^64 over the tensor's 32-bit (else
    16- or 8-bit) words w_i, k_i = splitmix64(i·φ + key) for a key sampled from the OS when the run starts, and its dtype and
    shape mixed in. It runs on the tensor's device with one sync per step (`Counter`), so a run can keep it on. For a
    fixed nonzero difference the chance it is missed is at most 2^-33 over the key."""

    def __init__(self, key: bytes):
        self.key = key
        self._k = _signed(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))
        self._keys: dict[str, torch.Tensor] = {}

    def _words(self, t: torch.Tensor) -> torch.Tensor:
        b = t.detach().contiguous().reshape(-1)
        size = b.numel() * b.element_size()
        b = b.view(torch.uint8) if b.dtype != torch.uint8 else b
        word = torch.int32 if size % 4 == 0 else torch.int16 if size % 2 == 0 else torch.uint8
        return b.view(word).to(torch.int64)

    def _key_stream(self, n: int, device) -> torch.Tensor:
        dev = str(device)
        ks = self._keys.get(dev)
        if ks is None or ks.numel() < n:
            idx = torch.arange(max(n, 1 << 16), dtype=torch.int64, device=device)
            ks = self._keys[dev] = _mix(idx * _GOLDEN + self._k)
        return ks[:n]

    def tensor(self, t: torch.Tensor) -> torch.Tensor:
        """The digest as an int64 scalar on t's device (no sync)."""
        with torch.no_grad():
            w = self._words(t)
            h = (w * self._key_stream(w.numel(), w.device)).sum()
            meta = int.from_bytes(hashlib.sha256(f"{t.dtype}\0{tuple(t.shape)}".encode()).digest()[:8], "little")
            return h ^ _signed(meta)

    @staticmethod
    def hex(h: int) -> str:
        return f"{h & ((1 << 64) - 1):016x}"

    def __call__(self, t: torch.Tensor) -> str:
        return self.hex(int(self.tensor(t)))


@dataclass(frozen=True)
class Tag:
    tok: bool = False
    roots: frozenset = frozenset()
    cone: frozenset = frozenset()
    src: frozenset = frozenset()  # the token sources a token value is computed from (`Counter._token_source`)

    def merge(self, other: Tag) -> Tag:
        if other is NONE:
            return self
        if self is NONE:
            return other
        return Tag(self.tok or other.tok, self.roots | other.roots, self.cone | other.cone, self.src | other.src)


NONE = Tag()
#: the root of a value computed from gradients or token data and used as a weight: it differs from θ, with no tensor of
#: its own to name
DERIVED = "derived"


@dataclass
class Step:
    """One step's count: MACs by node class, its token count, its updated nodes and the updated weights it used."""
    index: int
    tokens: int = 0
    macs: int = 0
    matmul_macs: int = 0
    elementwise_macs: int = 0
    approx_macs: int = 0
    by: dict = field(default_factory=dict)
    updated: list = field(default_factory=list)  # [per_row_macs, upstream_per_row, macs, rows, weight_bits]
    log: dict = field(default_factory=dict)  # (phase, kind, op, shapes) -> [count, macs]
    weights: dict = field(default_factory=dict)  # weight key -> [digest of each value used, in order]
    shapes: dict = field(default_factory=dict)  # weight key -> [dtype, shape]
    cpu_s: float = 0.0
    wall_s: float = 0.0
    gpu_s: float = 0.0
    device_macs: dict = field(default_factory=dict)
    exited: int = 0  # elements of token data read out of torch (`guard_exits`)


#: torch.Tensor's methods that hand its values to Python or another library past the dispatcher (`__array__` calls
#: `numpy`)
EXITS = ("numpy", "tolist", "__dlpack__")
_RUNNING: list = []


def guard_exits() -> None:
    """Wrap `EXITS`, once per process, so the innermost running counter sees token data leave torch (`Step.exited`).
    Install it before anything snapshots torch.Tensor's attributes."""
    for name in EXITS:
        fn = getattr(torch.Tensor, name)
        if getattr(fn, "_te_exit", False):
            continue

        def exit_(self, *args, _fn=fn, **kwargs):
            if _RUNNING:
                _RUNNING[-1]._exit(self)
            return _fn(self, *args, **kwargs)

        exit_._te_exit = True
        exit_.__name__ = exit_.__qualname__ = name
        setattr(torch.Tensor, name, exit_)


def structure_id(struct: dict) -> str:
    """The name a record's `structures` holds a step's structure (its updated nodes and its log) under."""
    return hashlib.sha256(json.dumps(struct, sort_keys=True).encode()).hexdigest()[:16]


def leaked(step: dict) -> bool:
    """Whether a step's record shows token data read out of torch at one value or more per token."""
    return step.get("exited", 0) >= max(step["tokens"], 2)


class Counter(TorchDispatchMode):
    """See the module docstring. `register(model)` names θ's weights; `begin_step`/`end_step` bound a step; work outside
    any step is counted in step -1 (setup). A counter that continues a job counts its own steps from `first` on, and
    digests under the job's `key`."""

    def __init__(self, theta: dict[str, torch.Tensor] | None = None, *, digests: bool = True,
                 clock=time.perf_counter, cpu_clock=time.process_time, first: int = 0, key: bytes | None = None):
        super().__init__()
        self.theta_cpu = theta or {}
        self.constants: set[str] = set()
        self.digests = digests
        self.wdigest = KeyedDigest(key or os.urandom(16)) if digests else None
        self.first = first
        self.clock, self.cpu_clock = clock, cpu_clock
        self.theta_ptr: dict[int, str] = {}
        self.held: dict[int, torch.Tensor] = {}
        self.weights: dict[str, torch.Tensor] = {}
        self.refs: dict[tuple, torch.Tensor] = {}
        self.probes: dict[str, torch.Tensor] = {}
        self.epoch: dict[int, int] = {}
        self.verdict: dict[int, tuple[int, bool]] = {}
        self.new_keys: dict[int, str] = {}
        self.new_tensors: dict[str, torch.Tensor] = {}
        self.steps: list[Step] = []
        self.setup = Step(-1)
        self.cur = self.setup
        self.paused = 0
        self.unseen_writes: list[tuple[int, str]] = []
        self.unknown_ops: dict[str, int] = {}
        self.foreign_ops: dict[str, int] = {}
        self.batches = WeakTensorKeyDictionary()
        self.n_batches = 0
        self.blank: dict[tuple, torch.Tensor] = {}
        self._reset_tags()
        self._t0 = (self.clock(), self.cpu_clock())
        self._events = None

    # -- registration and steps

    @property
    def done(self) -> int:
        """The index of the next step to end: the steps a job ran before this counter began (`first`, a job that
        continues from a kept state) and this counter's own."""
        return self.first + len(self.steps)

    def register(self, named: dict[str, torch.Tensor], buffers: dict[str, torch.Tensor] | None = None) -> None:
        """Name θ's weights by the live tensors that hold them (a model's `state_dict(keep_vars=True)`), each of which must
        equal θ's value (cast to its dtype) bit for bit. `buffers` are the model's (`named_buffers()`): each that θ
        doesn't hold is a constant from here on (the module docstring), held under its name at the value it has
        now."""
        with torch.utils._python_dispatch._disable_current_modes():
            for name, t in named.items():
                if name not in self.theta_cpu:
                    continue
                if not self._same(t, name):
                    raise ValueError(f"registered weight {name!r} is not θ's value")
                self._hold(t, name)
            missing = sorted(set(self.theta_cpu) - set(self.weights))
            if missing:
                raise ValueError(f"θ weights not registered: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            for name, t in (buffers or {}).items():
                # a buffer in memory the counter already holds is checked as what holds it
                if name in self.theta_cpu or not t.numel() or _ptr(t) in self.theta_ptr:
                    continue
                # a new dict: the one the counter was given is the harness's θ
                self.theta_cpu = self.theta_cpu | {name: t.detach().cpu().clone()}
                self.constants.add(name)
                self._hold(t, name)

    def _hold(self, t: torch.Tensor, name: str) -> None:
        ptr = _ptr(t)
        self.theta_ptr[ptr] = name
        self.held[ptr] = t
        self.weights[name] = t

    def _ref(self, name: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        key = (name, dtype, str(device))
        ref = self.refs.get(key)
        if ref is None:
            ref = self.refs[key] = self.theta_cpu[name].to(device=device, dtype=dtype)
        return ref

    def _same(self, t: torch.Tensor, name: str) -> bool:
        """Whether `t` holds θ's `name` exactly: equal to θ cast to t's dtype, and to θ when cast back."""
        th = self.theta_cpu[name]
        t = t.detach()
        if tuple(t.shape) != tuple(th.shape) or not torch.equal(t, self._ref(name, t.dtype, t.device)):
            return False
        return t.dtype == th.dtype or torch.equal(t.to(th.dtype), self._ref(name, th.dtype, t.device))

    def _adopt(self, t: torch.Tensor, ptr: int) -> str | None:
        """θ's weight an unseen persistent tensor holds, if any (the module docstring). Never a constant: a tensor
        that equals a buffer of the model (an optimizer's zero state, an average's first value) is not that buffer."""
        n = t.numel()
        if (not self.theta_cpu or not t.is_contiguous() or t.storage_offset()
                or t.untyped_storage().nbytes() != n * t.element_size()):
            return None
        names = [k for k, v in self.theta_cpu.items() if k not in self.constants and tuple(v.shape) == tuple(t.shape)]
        if not names:
            return None
        at = sorted({0, n // 3, 2 * n // 3, n - 1})
        mine = t.detach().reshape(-1)[torch.tensor(at, device=t.device)].double().cpu()
        for name in names:
            probe = self.probes.get(name)
            if probe is None:
                probe = self.probes[name] = self.theta_cpu[name].reshape(-1)[at].double()
            if torch.equal(mine, probe) and self._same(t, name):
                self._hold(t, name)
                return name
        return None

    def _reset_tags(self) -> None:
        self.tags = WeakTensorKeyDictionary()
        self.token_seen: dict = {}
        self.sources: set[int] = set()  # the token sources `_token_source` made (not the data stream's batches)
        self.node_rows: dict[int, int] = {}
        self.used: set[tuple[str, int]] = set()
        for t in list(self.batches.keys()):
            self._tag_batch(t)

    def begin_step(self) -> None:
        """Start the next step; work since the last step ended is the new step's."""
        self._close(self.cur)
        if self.cur is self.setup:
            self.cur = Step(self.done)
        self._reset_tags()
        self._open(self.cur)

    def end_step(self) -> None:
        self._close(self.cur)
        with torch.utils._python_dispatch._disable_current_modes():
            for ptr, t in self.held.items():
                cached = self.verdict.get(ptr)
                if cached is not None and cached[0] == self.epoch.get(ptr, 0):
                    actual = not self._same(t, self.theta_ptr[ptr])
                    if actual != cached[1]:
                        self.unseen_writes.append((self.cur.index, self.theta_ptr[ptr]))
                        self.epoch[ptr] = self.epoch.get(ptr, 0) + 1
        self._resolve(self.cur)
        self.steps.append(self.cur)
        self.cur = Step(self.done)
        self._reset_tags()
        self._open(self.cur)

    def _open(self, s: Step) -> None:
        s._t = (self.clock(), self.cpu_clock())
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            s._ev = torch.cuda.Event(enable_timing=True)
            s._ev.record()

    def _close(self, s: Step) -> None:
        t = getattr(s, "_t", None)
        if t is not None:
            s.wall_s += self.clock() - t[0]
            s.cpu_s += self.cpu_clock() - t[1]
            s._t = None
        ev = getattr(s, "_ev", None)
        if ev is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            end.synchronize()
            s.gpu_s += ev.elapsed_time(end) / 1000.0
            s._ev = None

    def pause(self):
        """Work in this context (the fixed evaluator, the checker) isn't the program's and isn't counted."""
        counter = self

        class _Paused:
            def __enter__(self):
                if not counter.paused:
                    self.timed = getattr(counter.cur, "_t", None) is not None
                    counter._close(counter.cur)
                counter.paused += 1

            def __exit__(self, *exc):
                counter.paused -= 1
                if not counter.paused and getattr(self, "timed", False):
                    counter._open(counter.cur)

        return _Paused()

    def __enter__(self):
        out = super().__enter__()
        _RUNNING.append(self)
        if self.cur is self.setup:
            self._open(self.setup)
        return out

    def __exit__(self, *exc):
        if _RUNNING and _RUNNING[-1] is self:
            _RUNNING.pop()
        return super().__exit__(*exc)

    def _exit(self, t) -> None:
        if not self.paused and isinstance(t, torch.Tensor):
            tag = self.tags.get(t)
            if tag is not None and tag.tok:
                self.cur.exited += t.numel()

    def tokens(self, batch: torch.Tensor) -> torch.Tensor:
        """Mark `batch` as this step's token data (an index tensor the program passes through some route other than an
        embedding lookup)."""
        self.tags[batch] = self._token_source(batch, Tag(tok=True))
        return batch

    def batch(self, t: torch.Tensor) -> torch.Tensor:
        """Mark `t`, a batch from the start's data stream, as token data in every step from now on while it lives, so
        whatever reads it is token-indexed (not only an embedding lookup). Its tokens are counted where an op reads it as
        indices (`TOKEN_INDEX`), each position once per step (`_token_source`): an input slice read twice counts once,
        and micro-batches of it add up."""
        self.batches[t] = True
        self._tag_batch(t)
        return t

    def _tag_batch(self, t: torch.Tensor) -> None:
        self.n_batches += 1
        self.tags[t] = Tag(tok=True, src=frozenset((self.n_batches,)))

    def _token_source(self, idx: torch.Tensor, tag: Tag) -> Tag:
        """Count `idx`'s elements as this step's tokens, once each: a source's tokens are the union of the positions
        its reads cover in each tensor they view (`read_positions`), so disjoint reads of one batch (micro-batches)
        add up and a read of positions already read adds nothing; and return its tag. A batch of the data stream is
        its own source; any other token tensor becomes one at its first read, held in its tag, so a view of it or a
        value computed from it is the same source, a new tensor at a freed one's address is another, and so is the
        tensor once an op writes into it (`sources`). A value computed from a source is its own tensor, whose reads
        count beside the source's. A θ tensor keeps θ's tag and is keyed by its memory, which the counter holds."""
        if not tag.src:
            if _ptr(idx) in self.theta_ptr:
                key = (_ptr(idx), idx.storage_offset(), tuple(idx.shape), idx._version)
                if key not in self.token_seen:
                    self.token_seen[key] = idx.numel()
                    self.cur.tokens += idx.numel()
                return tag
            self.n_batches += 1
            self.sources.add(self.n_batches)
            tag = tag.merge(Tag(tok=True, src=frozenset((self.n_batches,))))
            self.tags[idx] = tag
        by_base = self.token_seen.setdefault(tag.src, WeakTensorKeyDictionary())
        base = idx if idx._base is None else idx._base
        seen = by_base.get(base)
        now = read_positions(idx) if seen is None else np.union1d(seen, read_positions(idx))
        self.cur.tokens += len(now) - (0 if seen is None else len(seen))
        by_base[base] = now
        return tag

    # -- provenance

    def _tag(self, t: torch.Tensor) -> Tag:
        tag = self.tags.get(t)
        if tag is not None:
            return tag
        ptr = _ptr(t)
        name = self.theta_ptr.get(ptr)
        if name is None:
            if not (t.is_floating_point() or t.is_complex()) or t.numel() <= 1 or ptr == 0:
                return NONE
            if ptr not in self.new_keys:
                name = self._adopt(t, ptr)
        if name is not None:
            self.weights[name] = self.held[ptr]
            return Tag(roots=frozenset((name,))) if self._differs(ptr) else NONE
        key = self.new_keys.get(ptr)
        if key is None:
            key = self.new_keys[ptr] = f"new/{len(self.new_keys)}"
        self.new_tensors[key] = t
        return Tag(roots=frozenset((key,)))

    def _differs(self, ptr: int) -> bool:
        ep = self.epoch.get(ptr, 0)
        cached = self.verdict.get(ptr)
        if cached is not None and cached[0] == ep:
            return cached[1]
        d = not self._same(self.held[ptr], self.theta_ptr[ptr])
        self.verdict[ptr] = (ep, d)
        return d

    def _shaped(self, like: torch.Tensor, shape) -> torch.Tensor:
        """A stand-in for an operand a fused kernel computes inside itself: `shape` at `like`'s dtype and device, one
        zero repeated, with no memory and no value of its own."""
        key = (like.dtype, like.device)
        if key not in self.blank:
            self.blank[key] = like.new_zeros(())
        return self.blank[key].expand(*shape)

    def _record_use(self, roots: frozenset, operand: torch.Tensor) -> None:
        """Digest each updated weight a node reads, once per value: a root at its value now, and a derived operand (computed
        from gradients) as the operand itself. A derived operand a fused kernel computes inside itself (`_shaped`) has
        no value to digest, and is listed without one."""
        for key in roots:
            if key == DERIVED:
                key, t = f"derived/{sum(k.startswith('derived/') for k in self.cur.shapes)}", operand
                ep = (key, 0)
            else:
                t = self.weights.get(key)
                if t is None:
                    t = self.new_tensors[key]
                ep = (key, self.epoch.get(_ptr(t), 0))
            if ep in self.used:
                continue
            self.used.add(ep)
            self.cur.shapes[key] = [str(t.dtype).removeprefix("torch."), list(t.shape)]
            blank = self.blank.get((t.dtype, t.device))
            valued = self.digests and (blank is None or _ptr(t) != _ptr(blank))
            self.cur.weights.setdefault(key, []).append(self.wdigest.tensor(t) if valued else "")

    def _resolve(self, s: Step) -> None:
        """A step's value digests, from device scalars to hex: one transfer per device."""
        pending: dict = {}
        for key, hs in s.weights.items():
            for i, h in enumerate(hs):
                if isinstance(h, torch.Tensor):
                    pending.setdefault(h.device, []).append((key, i, h))
        with torch.utils._python_dispatch._disable_current_modes():
            for items in pending.values():
                for (key, i, _), v in zip(items, torch.stack([h for _, _, h in items]).tolist()):
                    s.weights[key][i] = KeyedDigest.hex(v)

    # -- counting

    def _macs(self, func, args, kwargs, out) -> tuple[int, str]:
        p = func.overloadpacket
        if p in SHAPE_OPS:
            return _shape_macs(func, args), "matmul"
        if p in FLOP_FORMULAS:
            return int(FLOP_FORMULAS[p](*args, **kwargs, out_val=out)) // 2, "matmul"
        if p in ELEMENTWISE or p in SCALED and kwargs.get("alpha", 1) != 1:
            return sum(t.numel() for t in _tensors(out) or _tensors(args[0])), "elementwise"
        if p in FUSED_OPTIMIZERS:
            return FUSED_OPTIMIZERS[p] * sum(t.numel() for t in _tensors(args[0])), "elementwise"
        if p in LINALG:
            a = args[0]
            n = a.shape[-1] if a.dim() else 1
            return (a.numel() // max(n, 1)) * n * n, "approx"
        return 0, ""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if self.paused:
            for t in _written(func, args, kwargs):
                self.epoch[_ptr(t)] = self.epoch.get(_ptr(t), 0) + 1
            return out
        p = func.overloadpacket
        ins = _tensors((args, kwargs))
        if p in FACTORY and not ins:
            for t in _tensors(out):
                self.tags[t] = NONE
            return out
        bwd = torch._C._current_graph_task_id() != -1
        written = _written(func, args, kwargs)
        persistent = {id(t) for t in written if t not in self.tags}
        tags = [self._tag(t) for t in ins]
        if p in EMBEDDING and not bwd:
            tags = [Tag(tok=True).merge(g) if t is args[1] else g for t, g in zip(ins, tags)]
        if p in TOKEN_INDEX and not bwd:
            tags = [self._token_source(t, g) if g.tok and not (t.is_floating_point() or t.is_complex()
                                                               or t.dtype == torch.bool) else g
                    for t, g in zip(ins, tags)]
        if p in FUSED_ATTENTION:
            tag = self._attention(func, args, kwargs, dict(zip(map(id, ins), tags)), bwd)
        else:
            tag = self._op(func, args, kwargs, out, tags, bwd)
        if bwd:
            tag = Tag(False, tag.roots | {DERIVED})
        for t in written:
            ptr = _ptr(t)
            self.epoch[ptr] = self.epoch.get(ptr, 0) + 1
        wrote = {id(t) for t in written}
        rewritten = Tag(tag.tok, tag.roots, tag.cone, tag.src - self.sources)
        for t in _tensors(out) + written:
            if _ptr(t) not in self.theta_ptr and id(t) not in persistent:
                self.tags[t] = rewritten if id(t) in wrote else tag
        return out

    def _op(self, func, args, kwargs, out, tags: list[Tag], bwd: bool) -> Tag:
        """Count one op, whose tensor operands (`_tensors((args, kwargs))`) carry `tags`, and return its result's tag."""
        p = func.overloadpacket
        ins = _tensors((args, kwargs))
        tag = NONE
        for g in tags:
            tag = tag.merge(g)
        macs, cls = self._macs(func, args, kwargs, out)
        if func.namespace not in NAMESPACES:
            self.foreign_ops[str(func)] = self.foreign_ops.get(str(func), 0) + 1
        elif not macs and func.namespace == "aten" and gemm_like(p.__name__):
            self.unknown_ops[str(func)] = self.unknown_ops.get(str(func), 0) + 1
        if (not macs and not bwd and p in GATHERS and ins and ins[0] is args[0] and tags[0].roots and not tags[0].tok
                and any(g.tok for g in tags[1:])):
            # an updated weight read by token data: the circuit's gather of its rows (module docstring)
            macs, cls = gather_macs(p.__name__, args[0].shape, args[1] if len(args) > 1 else None,
                                    sum(t.numel() for t in _tensors(out))), "elementwise"
        if not macs and not bwd and not _is_view(func):
            # token data and an updated weight combined by any op are an updated node; a gather moves its source's
            # rows without computing on them, so only its indices combine
            src = {id(t) for t in _tensors(args[:1])} if p in GATHERS else set()
            mixed = [g for t, g in zip(ins, tags) if id(t) not in src]
            if any(g.tok for g in mixed) and any(g.roots and not g.tok for g in mixed):
                macs, cls = _work(func, args, kwargs, out), "elementwise"
        return self._count(func, ins, tags, macs, cls, tag, bwd) if macs else tag

    def _attention(self, func, args, kwargs, tags: dict[int, Tag], bwd: bool) -> Tag:
        """A fused attention kernel, counted as the ops the math kernel computes attention by, each on its own operands'
        provenance: they are the ops a step's trace holds (`capture` traces under the math kernel), so a job's record
        has the same ops, shapes and MACs under either kernel. Their dtype is the query's (`_shaped`), the one a fused
        kernel computes the call in; the math kernel computes a 16-bit call from operands it casts up to float32, so
        under it the same ops run in float32, and a trace, which is taken under it, prices them at the query's
        (`capture.products.priced_dtype`). With q [*b, L, E], k [.., S, E], v [.., S, Ev] and n the product of q's
        batch dims (k and v counted at q's heads, as grouped-query attention repeats them):

        - forward: the scalings of q and of kᵀ (`mul`, each by the root of the scale); the scores (`bmm`, [n, L, E] by
          [n, E, S]); a mask added to them, where that combines token data with an updated weight (`add`, as any
          such op counts); dropout's scaling of its keep mask and of the weights by it (`div_` and `mul`, at
          [*b, L, S]); and the weights by v (`bmm`, [n, L, S] by [n, S, Ev]);
        - backward, for the gradients the step asks for: v's product ([n, S, L] by [n, L, Ev]); the weights' ([n, L, Ev]
          by [n, Ev, S]) where q, k or the mask asks, and with it dropout's scaling; and for each of q and k its
          product ([n, L, S] by [n, S, E]; [n, E, L] by [n, L, S]) and its scaling.

        The card's fused backward computes the scores again, a fifth product, which this count leaves out because a
        circuit reads the forward's.

        What the step asks for is the kernel's own word where it takes one (`grad_input_mask`), else which of q, k and v
        require a gradient. A kernel that takes no such word computes all three gradients whatever `autograd.grad` was
        asked, so a step asking for fewer than require one is counted at those that require one: the larger count.
        An operand of these ops that the kernel never holds is a stand-in of its shape (`_shaped`)."""
        a = dict(zip((x.name for x in func._schema.arguments), args)) | kwargs
        q, k, v, bias = a["query"], a["key"], a["value"], a.get("attn_mask", a.get("attn_bias"))
        *b, L, E = q.shape
        S, Ev = k.shape[-2], v.shape[-1]
        n, weights = math.prod(b), [*b, L, S]
        dropout = a.get("dropout_p", 0) > 0
        mul, bmm = aten.mul.Tensor, aten.bmm.default

        def op(f, shapes, of=(), held=()) -> Tag:
            ins = (*(self._shaped(q, s) for s in shapes), *held)
            return self._op(f, ins, {}, None, list(of) or [NONE] * len(ins), bwd)

        if not bwd:
            tq = op(mul, [[*b, L, E]], [tags[id(q)]])
            tk = op(mul, [[*b, E, S]], [tags[id(k)]])
            tw = op(bmm, [[n, L, E], [n, E, S]], [tq, tk])
            if bias is not None:
                tw = op(aten.add.Tensor, [weights], [tw, tags[id(bias)]], [bias])
            if dropout:
                op(aten.div_.Tensor, [weights])
                tw = op(mul, [weights, weights], [tw, NONE])
            return op(bmm, [[n, L, S], [n, S, Ev]], [tw, tags[id(v)]])
        gq, gk, gv, gb = [*(a.get("grad_input_mask") or [t.requires_grad for t in (q, k, v)]), False][:4]
        if gv:
            op(bmm, [[n, S, L], [n, L, Ev]])
        if gq or gk or gb:
            op(bmm, [[n, L, Ev], [n, Ev, S]])
            if dropout:
                op(mul, [weights, weights])
        if gq:
            op(bmm, [[n, L, S], [n, S, E]])
            op(mul, [[*b, L, E]])
        if gk:
            op(bmm, [[n, E, L], [n, L, S]])
            op(mul, [[*b, E, S]])
        tag = NONE
        for g in tags.values():
            tag = tag.merge(g)
        return tag

    def _weight_shaped(self, t: torch.Tensor) -> bool:
        return t.dim() == 2 and tuple(t.shape) in self.weight_shapes

    @property
    def weight_shapes(self) -> set:
        shapes = {tuple(t.shape) for t in [*self.held.values(), *self.new_tensors.values()] if t.dim() == 2}
        return shapes | {s[::-1] for s in shapes}

    def _count(self, func, ins, tags, macs, cls, tag: Tag, bwd: bool) -> Tag:
        s = self.cur
        s.macs += macs
        setattr(s, f"{cls}_macs", getattr(s, f"{cls}_macs") + macs)
        dev = ins[0].device.type if ins else "cpu"
        s.device_macs[dev] = s.device_macs.get(dev, 0) + macs
        if not bwd and sum(g.tok for g in tags) >= 2 and func.overloadpacket in SHAPE_OPS:
            # a token-derived operand shaped like a weight (a gradient computed by hand) is a weight
            tags = [Tag(False, g.roots | {DERIVED}) if g.tok and self._weight_shaped(t) else g for t, g in zip(ins, tags)]
        tok = [t for t, g in zip(ins, tags) if g.tok]
        weights = [(t, g) for t, g in zip(ins, tags) if not g.tok and g.roots]
        if bwd:
            kind = "grad"
        elif not tok:
            kind = "param"
        elif weights:
            kind = "updated"
        elif len(tok) >= 2:
            kind = "act"
        else:
            kind = "theta"
        phase = "bwd" if bwd else ("fwd" if tok else "other")
        key = f"{phase}/{kind}"
        s.by[key] = s.by.get(key, 0) + macs
        sig = (phase, kind, str(func.overloadpacket).removeprefix("aten."), json.dumps([list(t.shape) for t in ins]))
        entry = s.log.setdefault(sig, [0, 0])
        entry[0] += 1
        entry[1] += macs
        if phase != "fwd" or not tok:
            return tag
        a = tok[0]
        first = next(i for i, t in enumerate(ins) if t is a) == (1 if func.overloadpacket in ACCUMULATING else 0)
        if not (a.is_floating_point() or a.is_complex()):
            rows = a.numel()
        elif a.dim() >= 2 and not first and cls == "matmul":
            rows = (a.numel() // (a.shape[-2] * a.shape[-1])) * a.shape[-1]
        else:
            rows = a.numel() // a.shape[-1] if a.dim() else 1
        # a token row's work is per token: an operand reshaped so its rows outnumber the step's tokens has no more
        rows = max(min(rows, s.tokens) if s.tokens else rows, 1)
        cone = frozenset().union(*(g.cone for g in tags))
        upstream = sum(self.node_rows[n] for n in cone)
        if kind != "updated":
            return tag
        for t, g in weights:
            self._record_use(g.roots, t)
        per_row = macs // rows
        node = len(self.node_rows)
        self.node_rows[node] = per_row
        bits = min(t.element_size() * 8 for t, _ in weights)
        s.updated.append([per_row, upstream, macs, rows, bits])
        return Tag(True, tag.roots, cone | {node}, tag.src)

    # -- the record

    def finish(self) -> None:
        self._close(self.cur)

    def record(self) -> dict:
        """The run's count: totals, per-step records with their structures deduplicated, timings and flags."""
        structures, steps = {}, []
        tail = self.cur if self.cur is not self.setup else Step(self.done)
        for s in [self.setup, *self.steps, tail]:
            self._resolve(s)
            struct = {"updated": sorted(s.updated, key=lambda u: (u[1], u[0])),
                      "log": sorted([list(k) + v for k, v in s.log.items()])}
            sid = structure_id(struct)
            structures[sid] = struct
            steps.append({"step": s.index, "tokens": s.tokens, "macs": s.macs, "matmul_macs": s.matmul_macs,
                          "elementwise_macs": s.elementwise_macs, "approx_macs": s.approx_macs, "by": s.by,
                          "struct": sid, "weights": s.weights, "shapes": s.shapes, "cpu_s": s.cpu_s, "wall_s": s.wall_s,
                          "gpu_s": s.gpu_s, "device_macs": s.device_macs, "exited": s.exited})
        ru = resource.getrusage(resource.RUSAGE_CHILDREN)
        totals = {
            "macs": sum(s["macs"] for s in steps),
            "cpu_s": sum(s["cpu_s"] for s in steps),
            "wall_s": sum(s["wall_s"] for s in steps),
            "gpu_s": sum(s["gpu_s"] for s in steps),
            "process_cpu_s": self.cpu_clock() - self._t0[1],
            "process_wall_s": self.clock() - self._t0[0],
            "children_cpu_s": ru.ru_utime + ru.ru_stime,
            "approx_macs": sum(s["approx_macs"] for s in steps),
        }
        devs: dict[str, int] = {}
        for s in steps:
            for d, m in s["device_macs"].items():
                devs[d] = devs.get(d, 0) + m
        totals["device_macs"] = devs
        return {"totals": totals, "setup": steps[0], "steps": steps[1:-1], "tail": steps[-1], "structures": structures,
                "unseen_writes": self.unseen_writes, "new_weights": sorted(self.new_tensors),
                "unknown_ops": dict(sorted(self.unknown_ops.items())), "foreign_ops": dict(sorted(self.foreign_ops.items())),
                "digest_key": self.wdigest.key.hex() if self.wdigest else None}


#: CPU time over GPU time above which a GPU run is flagged as doing uncounted host work
CPU_OVER_GPU = 2.0
#: CPU time over wall time above which any run is flagged (more cores busy than one Python thread drives)
CPU_OVER_WALL = 1.5
#: counted MACs per GPU-second over the device's peak below which GPU time is flagged as unexplained
MIN_UTILIZATION = 0.01


def flags(totals: dict, peak_macs_per_s: float = 0.0) -> list[str]:
    """What the count can't vouch for: host compute beside the GPU's, child processes, GPU time the counted MACs don't
    explain, CPU-side MACs on a GPU run, and approximate linear algebra."""
    out = []
    cpu, wall, gpu = totals["cpu_s"], totals["wall_s"], totals["gpu_s"]
    if gpu > 0 and cpu > CPU_OVER_GPU * gpu:
        out.append(f"cpu_heavy: {cpu:.1f} CPU-s against {gpu:.1f} GPU-s")
    if wall > 0 and cpu > CPU_OVER_WALL * wall:
        out.append(f"cpu_parallel: {cpu:.1f} CPU-s in {wall:.1f} s")
    if totals.get("children_cpu_s", 0) > 0.1 * max(wall, 1e-9):
        out.append(f"children: {totals['children_cpu_s']:.1f} CPU-s in child processes")
    if gpu > 0 and peak_macs_per_s > 0 and totals["macs"] < MIN_UTILIZATION * gpu * peak_macs_per_s:
        out.append(f"gpu_unexplained: {totals['macs'] / (gpu * peak_macs_per_s):.4f} of peak over the GPU time")
    if gpu > 0 and totals["device_macs"].get("cpu", 0) > 0:
        out.append(f"cpu_macs: {totals['device_macs']['cpu']} MACs on host tensors")
    if totals.get("approx_macs"):
        out.append(f"approx: {totals['approx_macs']} MACs of linear algebra at n**3")
    return out


def forward_macs_per_token(model: torch.nn.Module, ctx: int, vocab: int, device=None) -> float:
    """The model's MACs per token for one forward pass over a `ctx`-token sequence, counted by `Counter` with its weights
    as θ (so a lookup of its own table computes nothing) and its other buffers as constants, as a job counts them: the
    start model's figure, from which the inference block's c is set."""
    device = device or next(model.parameters()).device
    idx = torch.zeros(1, ctx, dtype=torch.long, device=device) % max(vocab, 1)
    state = model.state_dict(keep_vars=True)
    c = Counter({k: v.detach().cpu() for k, v in state.items()}, digests=False)
    c.register(state, dict(model.named_buffers()))
    with torch.no_grad(), c:
        model(idx)
    return c.setup.macs / ctx
