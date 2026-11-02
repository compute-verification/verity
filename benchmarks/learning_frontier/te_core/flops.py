"""The harness's own count of a run's circuit C: every multiplication the process computes through torch, by its shape,
with the structure the annotation reads. Nothing a program reports about its compute is read.

`Counter` is a TorchDispatchMode installed around the whole program (run.py), so it sees every aten op on every device,
forward, backward (autograd's threads inherit the mode) and optimizer alike. It counts, in multiply-accumulates (MACs):

- matmuls (`mm`, `addmm`, `bmm`, `baddbmm`, `dot`, `mv`, `addmv`, `addbmm`, `_int_mm`, `_scaled_mm`), attention and
  convolutions, by torch's FLOP formulas (`torch.utils.flop_counter`) halved, or by shape;
- elementwise multiplications (`mul`, `addcmul`, `addcdiv`, `lerp`, their in-place and `_foreach_` forms), one MAC per output
  element, so a matmul written as a broadcast product is counted too;
- dense linear algebra (`eigh`, `svd`, `qr`, `cholesky`, `inv`, solves), at n**3 per matrix, marked approximate.

|C| is their sum (`macs`). Each multiplication is a node classified by its operands' provenance, which the counter tracks
per tensor through every op within a step:

- *token-indexed*: computed from this step's token data (an integer tensor indexed into an embedding, or a batch the
  program passed to `Counter.tokens`);
- a weight's *roots*: the persistent tensors a parameter-indexed value is computed from that differ from θ. A registered
  weight (`register`) differs from θ when its value, checked bit for bit against θ at its first use after any write, isn't
  θ's; any other persistent tensor (made before this step, or outside torch's dispatch) is a weight θ doesn't hold, so it
  differs. A tensor made inside the step by a factory op (`randn`, `zeros`, ...) is a fast value with no roots.

A node is `updated` when it is in a forward pass (outside autograd's backward), one operand is token-indexed and another
(the weight, or an initial accumulator) has roots: the updated multiplications U. Others are `theta` (the weight equals
θ's), `act` (two token-indexed operands: attention), `param` (no token-indexed operand: optimizer work) or, in the
backward pass, the same kinds under `bwd`. Each updated node records its MACs per token row and the updated MACs per token
row upstream of it in its token's pass (the cone of updated forward nodes its operands were computed from).

Tags are cleared at every step boundary, so a value carried from one step to the next is a weight from then on. Writes
the dispatcher doesn't see (a custom kernel into a weight's memory) are caught at each step's end, when every registered
weight is checked against θ again (`unseen_writes`). Work it can't see at all (custom GPU kernels) shows only as GPU time
the counted MACs don't explain, which `Flags` reports beside CPU time large relative to GPU time.
"""
from __future__ import annotations

import hashlib
import json
import resource
import time
from dataclasses import dataclass, field

import torch
from torch.utils import flop_counter
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.weak import WeakTensorKeyDictionary

aten = torch.ops.aten


def _shape_macs(func, args) -> int:
    p = func.overloadpacket
    if p in (aten.mm, aten._int_mm):
        (m, k), (_, n) = args[0].shape, args[1].shape
        return m * k * n
    if p is aten.addmm:
        (m, k), (_, n) = args[1].shape, args[2].shape
        return m * k * n
    if p is aten.bmm:
        (b, m, k), (_, _, n) = args[0].shape, args[1].shape
        return b * m * k * n
    if p in (aten.baddbmm, aten.addbmm):
        (b, m, k), (_, _, n) = args[1].shape, args[2].shape
        return b * m * k * n
    if p in (aten.dot, aten.vdot):
        return args[0].numel()
    if p is aten.mv:
        return args[0].numel()
    if p is aten.addmv:
        return args[1].numel()
    raise KeyError(p)


SHAPE_OPS = {aten.mm, aten._int_mm, aten.addmm, aten.bmm, aten.baddbmm, aten.addbmm, aten.dot, aten.vdot, aten.mv, aten.addmv}
ELEMENTWISE = {aten.mul, aten.mul_, aten.addcmul, aten.addcmul_, aten.addcdiv, aten.addcdiv_, aten.lerp, aten.lerp_,
               aten._foreach_mul, aten._foreach_mul_, aten._foreach_addcmul, aten._foreach_addcmul_,
               aten._foreach_addcdiv, aten._foreach_addcdiv_, aten._foreach_lerp, aten._foreach_lerp_}
SCALED = {aten.add, aten.add_, aten.sub, aten.sub_, aten._foreach_add, aten._foreach_add_, aten._foreach_sub,
          aten._foreach_sub_}
LINALG = {getattr(aten, n) for n in ("_linalg_eigh", "linalg_eigh", "_linalg_svd", "linalg_svd", "linalg_qr",
                                     "linalg_cholesky_ex", "linalg_inv_ex", "_linalg_solve_ex", "linalg_solve_triangular",
                                     "linalg_lu_factor_ex", "cholesky_solve", "triangular_solve") if hasattr(aten, n)}
FACTORY = {getattr(aten, n) for n in ("randn", "rand", "randint", "randperm", "zeros", "ones", "full", "empty",
                                      "empty_strided", "arange", "linspace", "eye", "scalar_tensor", "normal",
                                      "zeros_like", "ones_like", "empty_like", "full_like", "randn_like", "rand_like")
           if hasattr(aten, n)}
EMBEDDING = {aten.embedding}


def _tensors(x) -> list[torch.Tensor]:
    return [t for t in tree_flatten(x)[0] if isinstance(t, torch.Tensor)]


def _ptr(t: torch.Tensor) -> int:
    try:
        return t.untyped_storage().data_ptr()
    except (RuntimeError, NotImplementedError):
        return 0


def _written(func, args, kwargs) -> list[torch.Tensor]:
    out = []
    for i, a in enumerate(func._schema.arguments):
        if a.alias_info is not None and a.alias_info.is_write:
            v = args[i] if i < len(args) else kwargs.get(a.name)
            out.extend(_tensors(v))
    return out


def digest(t: torch.Tensor) -> str:
    """sha256 of a tensor's dtype, shape and bytes: the bit-for-bit identity a rebuild is checked against."""
    t = t.detach().contiguous()
    h = hashlib.sha256(f"{str(t.dtype)}\0{tuple(t.shape)}\0".encode())
    h.update(t.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class Tag:
    tok: bool = False
    roots: frozenset = frozenset()
    cone: frozenset = frozenset()

    def merge(self, other: Tag) -> Tag:
        if other is NONE:
            return self
        if self is NONE:
            return other
        return Tag(self.tok or other.tok, self.roots | other.roots, self.cone | other.cone)


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


class Counter(TorchDispatchMode):
    """See the module docstring. `register(model)` names θ's weights; `begin_step`/`end_step` bound a step; work outside
    any step is counted in step -1 (setup)."""

    def __init__(self, theta: dict[str, torch.Tensor] | None = None, *, digests: bool = True,
                 clock=time.perf_counter, cpu_clock=time.process_time):
        super().__init__()
        self.theta_cpu = theta or {}
        self.digests = digests
        self.clock, self.cpu_clock = clock, cpu_clock
        self.theta_ptr: dict[int, str] = {}
        self.weights: dict[str, torch.Tensor] = {}
        self.theta_dev: dict[str, torch.Tensor] = {}
        self.epoch: dict[int, int] = {}
        self.verdict: dict[str, tuple[int, bool]] = {}
        self.new_keys: dict[int, str] = {}
        self.new_tensors: dict[str, torch.Tensor] = {}
        self.steps: list[Step] = []
        self.setup = Step(-1)
        self.cur = self.setup
        self.paused = 0
        self.unseen_writes: list[tuple[int, str]] = []
        self.unknown_ops: dict[str, int] = {}
        self._reset_tags()
        self._t0 = (self.clock(), self.cpu_clock())
        self._events = None

    # -- registration and steps

    def register(self, named: dict[str, torch.Tensor]) -> None:
        """Name θ's weights by the live tensors that hold them (a model's `state_dict(keep_vars=True)`), each of which must
        equal θ's value (cast to its dtype) bit for bit."""
        with torch.utils._python_dispatch._disable_current_modes():
            for name, t in named.items():
                if name not in self.theta_cpu:
                    continue
                ref = self.theta_cpu[name].to(device=t.device, dtype=t.dtype)
                if ref.shape != t.shape or not torch.equal(ref, t.detach()):
                    raise ValueError(f"registered weight {name!r} is not θ's value")
                self.theta_ptr[_ptr(t)] = name
                self.weights[name] = t
                self.theta_dev[name] = ref
            missing = sorted(set(self.theta_cpu) - set(self.weights))
        if missing:
            raise ValueError(f"θ weights not registered: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    def _reset_tags(self) -> None:
        self.tags = WeakTensorKeyDictionary()
        self.token_seen: set[tuple] = set()
        self.node_rows: dict[int, int] = {}
        self.used: set[tuple[str, int]] = set()

    def begin_step(self) -> None:
        """Start the next step; work since the last step ended is the new step's."""
        self._close(self.cur)
        if self.cur is self.setup:
            self.cur = Step(len(self.steps))
        self._reset_tags()
        self._open(self.cur)

    def end_step(self) -> None:
        self._close(self.cur)
        with torch.utils._python_dispatch._disable_current_modes():
            for name, t in self.weights.items():
                ptr = _ptr(t)
                cached = self.verdict.get(name)
                if cached is not None and cached[0] == self.epoch.get(ptr, 0):
                    actual = not torch.equal(t.detach(), self.theta_dev[name])
                    if actual != cached[1]:
                        self.unseen_writes.append((self.cur.index, name))
                        self.epoch[ptr] = self.epoch.get(ptr, 0) + 1
        self.steps.append(self.cur)
        self.cur = Step(len(self.steps))
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
                counter.paused += 1

            def __exit__(self, *exc):
                counter.paused -= 1

        return _Paused()

    def tokens(self, batch: torch.Tensor) -> torch.Tensor:
        """Mark `batch` as this step's token data (an index tensor the program passes through some route other than an
        embedding lookup)."""
        self._token_source(batch)
        self.tags[batch] = Tag(tok=True)
        return batch

    def _token_source(self, idx: torch.Tensor) -> None:
        key = (_ptr(idx), idx.storage_offset(), tuple(idx.shape), idx._version)
        if key not in self.token_seen:
            self.token_seen.add(key)
            self.cur.tokens += idx.numel()

    # -- provenance

    def _tag(self, t: torch.Tensor) -> Tag:
        tag = self.tags.get(t)
        if tag is not None:
            return tag
        ptr = _ptr(t)
        name = self.theta_ptr.get(ptr)
        if name is not None:
            return Tag(roots=frozenset((name,))) if self._differs(name, ptr) else NONE
        if not (t.is_floating_point() or t.is_complex()) or t.numel() <= 1 or ptr == 0:
            return NONE
        key = self.new_keys.get(ptr)
        if key is None:
            key = self.new_keys[ptr] = f"new/{len(self.new_keys)}"
        self.new_tensors[key] = t
        return Tag(roots=frozenset((key,)))

    def _differs(self, name: str, ptr: int) -> bool:
        ep = self.epoch.get(ptr, 0)
        cached = self.verdict.get(name)
        if cached is not None and cached[0] == ep:
            return cached[1]
        d = not torch.equal(self.weights[name].detach(), self.theta_dev[name])
        self.verdict[name] = (ep, d)
        return d

    def _record_use(self, roots: frozenset, operand: torch.Tensor) -> None:
        """Digest each updated weight a node reads, once per value: a root at its value now, and a derived operand (computed
        from gradients) as the operand itself."""
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
            self.cur.weights.setdefault(key, []).append(digest(t) if self.digests else "")

    # -- counting

    def _macs(self, func, args, kwargs, out) -> tuple[int, str]:
        p = func.overloadpacket
        if p in SHAPE_OPS:
            return _shape_macs(func, args), "matmul"
        if p in flop_counter.flop_registry:
            return int(flop_counter.flop_registry[p](*args, **kwargs, out_val=out)) // 2, "matmul"
        if p in ELEMENTWISE or p in SCALED and kwargs.get("alpha", 1) != 1:
            return sum(t.numel() for t in _tensors(out) or _tensors(args[0])), "elementwise"
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
        if p in EMBEDDING and not bwd:
            self._token_source(args[1])
            tag = Tag(tok=True).merge(self._tag(args[0]))
        else:
            tag = NONE
            for t in ins:
                tag = tag.merge(self._tag(t))
        macs, cls = self._macs(func, args, kwargs, out)
        if macs:
            tag = self._count(func, ins, macs, cls, tag, bwd)
        if bwd:
            tag = Tag(False, tag.roots | {DERIVED})
        for t in written:
            ptr = _ptr(t)
            self.epoch[ptr] = self.epoch.get(ptr, 0) + 1
        for t in _tensors(out) + written:
            if _ptr(t) not in self.theta_ptr and id(t) not in persistent:
                self.tags[t] = tag
        return out

    def _weight_shaped(self, t: torch.Tensor) -> bool:
        return t.dim() == 2 and tuple(t.shape) in self.weight_shapes

    @property
    def weight_shapes(self) -> set:
        shapes = {tuple(t.shape) for t in [*self.weights.values(), *self.new_tensors.values()] if t.dim() == 2}
        return shapes | {s[::-1] for s in shapes}

    def _count(self, func, ins, macs, cls, tag: Tag, bwd: bool) -> Tag:
        s = self.cur
        s.macs += macs
        setattr(s, f"{cls}_macs", getattr(s, f"{cls}_macs") + macs)
        dev = ins[0].device.type if ins else "cpu"
        s.device_macs[dev] = s.device_macs.get(dev, 0) + macs
        tags = [self._tag(t) for t in ins]
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
        first = ins.index(a) == (1 if func.overloadpacket in (aten.addmm, aten.baddbmm, aten.addbmm, aten.addmv) else 0)
        if a.dim() >= 2 and not first and cls == "matmul":
            rows = (a.numel() // (a.shape[-2] * a.shape[-1])) * a.shape[-1]
        else:
            rows = a.numel() // a.shape[-1] if a.dim() else 1
        rows = max(rows, 1)
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
        return Tag(True, tag.roots, cone | {node})

    # -- the record

    def finish(self) -> None:
        self._close(self.cur)

    def record(self) -> dict:
        """The run's count: totals, per-step records with their structures deduplicated, timings and flags."""
        structures, steps = {}, []
        tail = self.cur if self.cur is not self.setup else Step(len(self.steps))
        for s in [self.setup, *self.steps, tail]:
            struct = {"updated": sorted(s.updated, key=lambda u: (u[1], u[0])),
                      "log": sorted([list(k) + v for k, v in s.log.items()])}
            sid = hashlib.sha256(json.dumps(struct, sort_keys=True).encode()).hexdigest()[:16]
            structures[sid] = struct
            steps.append({"step": s.index, "tokens": s.tokens, "macs": s.macs, "matmul_macs": s.matmul_macs,
                          "elementwise_macs": s.elementwise_macs, "approx_macs": s.approx_macs, "by": s.by,
                          "struct": sid, "weights": s.weights, "shapes": s.shapes, "cpu_s": s.cpu_s, "wall_s": s.wall_s,
                          "gpu_s": s.gpu_s, "device_macs": s.device_macs})
        ru = resource.getrusage(resource.RUSAGE_CHILDREN)
        totals = {
            "macs": sum(s["macs"] for s in steps),
            "cpu_s": self.cpu_clock() - self._t0[1],
            "wall_s": self.clock() - self._t0[0],
            "gpu_s": sum(s["gpu_s"] for s in steps),
            "children_cpu_s": ru.ru_utime + ru.ru_stime,
            "approx_macs": sum(s["approx_macs"] for s in steps),
        }
        devs: dict[str, int] = {}
        for s in steps:
            for d, m in s["device_macs"].items():
                devs[d] = devs.get(d, 0) + m
        totals["device_macs"] = devs
        return {"totals": totals, "setup": steps[0], "steps": steps[1:-1], "tail": steps[-1], "structures": structures,
                "unseen_writes": self.unseen_writes, "new_weights": sorted(self.new_tensors)}


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
    """The model's MACs per token for one forward pass over a `ctx`-token sequence, counted by `Counter`: the start model's
    figure, from which the inference block's c is set."""
    device = device or next(model.parameters()).device
    idx = torch.zeros(1, ctx, dtype=torch.long, device=device) % max(vocab, 1)
    c = Counter(digests=False)
    with torch.no_grad(), c:
        model(idx)
    return c.setup.macs / ctx
