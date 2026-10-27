"""The step driver: one decode step of a batch of sequence slots through every layer, every unit's output committed by the
per-step commit call as each group of the step (the embedding, a layer, the head) is written.

The weights (`Params`) are the plain INT8 codes, loaded once into the device's memory through the store's decode call and
resident across steps (Demo A, M1-M3); ``resident=False`` keeps M4's path, which stages each layer's six linears into staging
buffers every step (`store.Store.stage`). The small tensors (scales, gains, embeddings) are loaded once either way.

A step (`Driver.step_slots`) runs rows b = 0 .. B - 1, row b the KV pool's slot ``slots[b]`` at its own position
``positions[b]`` on byte ``tokens[b]``; `Driver.step` is the lockstep step (every slot at one position) that `Driver.run`
drives: position t reads byte t of each prompt while t < P, else the previous step's argmax. `Driver.serve` runs requests
through `slots.SlotKV`'s sequence slots by continuous batching: each step admits queued requests into the lowest free slots,
runs every active slot at its own position through `step_slots` (K and V written to and read from the SlotKV), and frees the
slots whose request finished, for a later step to reuse. Per layer, in micro-batches of
rows written into the layer's full-batch tables: RMSNorm, quantize, Q, K and V; the slot's key and value into its KV cache;
scores, softmax and PV over the slot's first position + 1 keys (`attention.Attention`); quantize, O, the residual add;
RMSNorm, quantize, up; ReLU and quantize, down, the residual add. Then the final RMSNorm, quantize, the head and argmax.

Each table of the step (`commit.StepTables`, laid out by `commit.tables` from circuits' FAMILIES map, `plan.Map`) is one
op's outputs, one row a slot: ``embed``; per layer ``l<i>.`` ``attn_sums`` (RMSNorm's f32 chunk sums), ``attn_norm``,
``attn_amax`` (the quantizer's amax words: its parts', then the row's), ``attn_quant`` (codes, and the scale as the row's
word), ``wq``, ``wk``, ``wv``, ``softmax`` (heads x 32, the slot's first position + 1 of each head's its probabilities and the
rest +0), ``pv``, ``o_amax``, ``o_quant``, ``wo``, ``attn_residual``, ``mlp_sums``, ``mlp_norm``, ``up_amax``, ``up_quant``,
``wup``, ``down_amax``, ``down_quant`` (RELU), ``wdown``, ``mlp_residual``; then ``final_sums``, ``final_norm``,
``head_amax``, ``head_quant``, ``head`` and ``argmax``; under the committer's ``residual`` fusion (`plan.Map`) there is no
``wo`` or ``wdown``, their coordinates' words being added straight into the residual, under its ``norm`` fusion no
``attn_norm``, ``mlp_norm`` or ``final_norm``, and those quantizers' amax words are every part's, then the row's, and
under its ``pv`` fusion no ``softmax``. The map's units
(`plan.Units`) are slices of these rows. RMSNorm sums in chunks and groups (`shape.Forms`), which changes its words from one
sequential sum wherever a row is more than one chunk. With ``keep=False`` a group's tables are dropped once committed (a
run of record keeps no step's tables: a drawn slot is opened by replaying its request, `opening`).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
import torch

from . import attention as A
from . import ops as O
from .commit import Committer, StepTables, tables
from .shape import EPS, PV_K, Shape, check, forms
from .slots import Finished, Request, SlotKV
from .store import Store
from .weights import LINEARS


def words(t: torch.Tensor) -> np.ndarray:
    """A bf16 tensor's u16 words on the host."""
    return t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def from_words(w: np.ndarray, device: Any) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(w, dtype=np.uint16).view(np.int16)).to(device).view(torch.bfloat16)


class Params:
    """The weights on ``device``: every linear's codes resident (``resident``, loaded once), or staged a layer at a time."""

    def __init__(self, shape: Shape, store: Store, device: Any = "cpu", resident: bool = True,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        self.shape, self.store, self.device, self.resident = check(shape), store, torch.device(device), resident
        dev, w = self.device, store.weights
        self.names = [f"l{i}.{k}" for i in range(shape.layers) for k in LINEARS] + ["head"]
        self.sw = {n: torch.from_numpy(np.ascontiguousarray(store.load(f"{n}.scale"))).to(dev) for n in self.names}
        self.gain = {n: from_words(store.load(n), dev)
                     for n in [f"l{i}.{k}" for i in range(shape.layers) for k in ("attn_norm", "mlp_norm")] + ["final_norm"]}
        self.tok = from_words(np.ascontiguousarray(store.load("tok").T), dev)    # (V, H): row v is token v's embedding
        self.pos = from_words(np.ascontiguousarray(store.load("pos").T), dev)    # (C, H)
        self.eps = torch.tensor([EPS], dtype=torch.int32).view(torch.float32)[0].to(dev)
        t0 = clock()
        if resident:
            self.lin = {n: store.stage(n, torch.empty(w.dims[n], dtype=torch.int8, device=dev)) for n in self.names}
            self.staging: dict[str, torch.Tensor] = {}
        else:
            self.lin = {}
            self.staging = {k: torch.empty(w.dims[f"l0.{k}"], dtype=torch.int8, device=dev) for k in LINEARS}
            self.staging["head"] = torch.empty(w.dims["head"], dtype=torch.int8, device=dev)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        self.load_seconds = clock() - t0

    @property
    def resident_bytes(self) -> int:
        return sum(t.numel() for t in self.lin.values())

    def layer(self, layer: int) -> dict[str, torch.Tensor]:
        """Layer ``layer``'s linears by kind (staged into the staging buffers unless resident)."""
        if self.resident:
            return {k: self.lin[f"l{layer}.{k}"] for k in LINEARS}
        for k in LINEARS:
            self.store.stage(f"l{layer}.{k}", self.staging[k])
        return {k: self.staging[k] for k in LINEARS}

    def head(self) -> torch.Tensor:
        return self.lin["head"] if self.resident else self.store.stage("head", self.staging["head"])

    def describe(self) -> dict:
        return {"weights": "resident (loaded once, plain INT8)" if self.resident else "staged a layer a step (M4's path)",
                "resident_bytes": self.resident_bytes, "load_seconds": self.load_seconds}


@dataclass
class StepResult:
    step: int
    position: int
    tokens_in: np.ndarray
    tokens_out: np.ndarray
    root: bytes
    table_roots: tuple[tuple[str, bytes], ...]
    value_bytes: int
    seconds: dict[str, float] = field(default_factory=dict)
    slots: np.ndarray | None = None
    positions: np.ndarray | None = None
    slot_roots: tuple[bytes, ...] = ()
    salt_key: bytes | None = field(default=None, repr=False)
    seqs: tuple[int, ...] = ()                                  # served: row b's request
    finished: list[Finished] = field(default_factory=list)      # served: the requests whose last position this was


class Driver:
    def __init__(self, shape: Shape, store: Store | Params, batch: int, committer: Committer, device: str = "cpu",
                 clock: Callable[[], float] = time.perf_counter, tamper: str | None = None, keep: bool = True,
                 micro_batch: int | None = None, resident: bool = True, attention: str | None = None,
                 subcircuits: Sequence[Any] = ()) -> None:
        """``batch``: the KV pool's slots. ``tamper``: a table whose committed bytes have bit 0 flipped at every step (a loop
        whose committed outputs are not its model's), which replay or opening must find. ``micro_batch``: the rows a
        layer's ops run on at once (all by default). ``attention``: `attention.Attention`'s kind (by default its pick).
        ``subcircuits``: `units.Subcircuit`s, each served after every step, and one with a ``call`` method also called on
        every served linear call (`_linear`)."""
        self.shape, self.batch, self.committer, self.tamper = check(shape), batch, committer, tamper
        self.device, self.clock, self.keep = torch.device(device), clock, keep
        self.params = store if isinstance(store, Params) else Params(shape, store, device, resident, clock)
        self.store = self.params.store
        self.mb = micro_batch or batch
        self.subcircuits = list(subcircuits)
        self.calls = [s for s in self.subcircuits if hasattr(s, "call")]
        self.attn = A.Attention(self.device, attention)
        self.groups = tables(shape, committer.fused)
        self.forms = forms(shape)
        self.k: torch.Tensor | None = None      # the driver's own KV pool, (layers, batch, context, head_dim) int16
        self.v: torch.Tensor | None = None
        self.kv: SlotKV | None = None           # serving: the slots' KV cache instead
        self.steps: list[StepTables] = []
        self.results: list[StepResult] = []
        self.finished: list[Finished] = []

    @property
    def tok(self) -> torch.Tensor:
        return self.params.tok

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _tick(self, sec: dict[str, float], key: str, t0: float) -> float:
        self._sync()
        t1 = self.clock()
        sec[key] = sec.get(key, 0.0) + t1 - t0
        return t1

    def _pool(self) -> None:
        if self.k is None:
            kv = (self.shape.layers, self.batch, self.shape.context, self.shape.head_dim)
            self.k = torch.zeros(kv, dtype=torch.int16, device=self.device)
            self.v = torch.zeros(kv, dtype=torch.int16, device=self.device)

    def _alloc(self, st: StepTables, group: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {t.op: st.alloc(t, self.device) for t in self.groups[group]}

    def _close(self, st: StepTables, group: int) -> None:
        ops = [t.op for t in self.groups[group]]
        if self.tamper in ops:
            st.tables[self.tamper][0].bitwise_xor_(1)
        self.committer.group(st, group)
        if not self.keep and self.committer.streams:
            st.drop(ops)

    def _linear(self, st: StepTables, op: str, lo: int, q: O.Quant, w: torch.Tensor) -> torch.Tensor:
        """Linear ``op``'s served call on rows ``lo ..`` of the step: its s32 sums, each subcircuit's ``call`` on them (a PoUW
        unit's z is this accumulator), then the epilogue."""
        acc = O.s32_mm(q.codes, w)
        for sub in self.calls:
            sub.call(self, st, op, lo, q.codes, w, acc)
        return O.scale(acc, q.scale, self.params.sw[op])

    @staticmethod
    def _bf(t: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return t[0].view(torch.bfloat16)

    def _norm(self, T: dict, op: str, gain: str, lo: int, hi: int, x: torch.Tensor) -> torch.Tensor:
        """RMSNorm ``op`` (``attn``, ``mlp``, ``final``) of rows lo .. hi: its chunk sums into ``op_sums``, its output into
        ``op_norm`` (none under the ``norm`` fusion)."""
        f, n = self.forms, x.shape[1]
        h, sums = O.rmsnorm(x, self.params.gain[gain], self.params.eps, f.chunk, f.groups(n))
        T[op + "_sums"][0].view(torch.float32)[lo:hi] = sums
        if op + "_norm" in T:
            self._bf(T[op + "_norm"])[lo:hi] = h
        return h

    def _quantize(self, T: dict, op: str, lo: int, hi: int, x: torch.Tensor, relu: bool = False,
                  normed: bool = False) -> O.Quant:
        """The quantizer ``op`` of rows lo .. hi: its amax words into ``op_amax``, its codes and scale into ``op_quant``.
        ``normed``: it quantizes an RMSNorm's output, whose amax words under the ``norm`` fusion are every part's (the fused
        parts of `shape.Forms.norm_parts`), then the row's."""
        n = x.shape[1]
        if normed and "norm" in self.committer.fused:
            parts, pw = self.forms.norm_parts(n)
            q = O.quantize(x, relu, pw, every=parts > 1)
        else:
            q = O.quantize(x, relu, self.forms.parts(n)[1])
        T[op + "_amax"][0].view(torch.int16)[lo:hi] = q.amax
        t = T[op + "_quant"]
        t[0][lo:hi] = q.codes
        t[1][lo:hi] = q.scale
        return q

    def step(self, t: int, tokens: np.ndarray, **kw: Any) -> StepResult:
        """Position ``t`` for every slot on ``tokens`` (B,): commits the step and returns its argmax bytes."""
        b = len(tokens)
        return self.step_slots(np.arange(b), np.full(b, t), tokens, **kw)

    def step_slots(self, slots: np.ndarray, positions: np.ndarray, tokens: np.ndarray, salt_key: bytes | None = None,
                   nonce0: int = 0) -> StepResult:
        """Rows b of slots ``slots[b]`` (of the KV pool, distinct) at positions ``positions[b]`` on bytes ``tokens[b]``: commits
        the step (row b its slot ``nonce0 + b`` of the commit, under ``salt_key`` if given) and returns its argmax bytes."""
        sh, dev = self.shape, self.device
        slots, positions = np.asarray(slots, np.int64), np.asarray(positions, np.int64)
        n = len(slots)
        if len(set(slots.tolist())) != n or n != len(positions) or n != len(tokens) or not n:
            raise ValueError("a step's rows are distinct slots, each with a position and a byte")
        if slots.min() < 0 or slots.max() >= self.batch or positions.min() < 0 or positions.max() >= sh.context:
            raise ValueError(f"slots are below {self.batch} and positions below {sh.context}")
        kv = self.kv
        if kv is None:
            self._pool()
        st = StepTables(len(self.results), positions, n, slots=slots)
        self.committer.begin(st, salt_key, nonce0)
        sec: dict[str, float] = {}
        c0 = self.clock()
        sl, pos = torch.from_numpy(slots).to(dev), torch.from_numpy(positions).to(dev)
        whole = n == self.batch and bool((slots == np.arange(n)).all())
        tok = torch.from_numpy(np.asarray(tokens, np.int64)).to(dev)
        T = self._alloc(st, 0)
        x = O.add(self.params.tok[tok], self.params.pos[pos])
        self._bf(T["embed"])[:] = x
        self._close(st, 0)
        ch = [(lo, min(lo + self.mb, n)) for lo in range(0, n, self.mb)]
        for layer in range(sh.layers):
            p = f"l{layer}."
            if not self.params.resident:
                c0 = self._tick(sec, "forward", c0)
            w = self.params.layer(layer)
            if not self.params.resident:
                c0 = self._tick(sec, "stage", c0)
            T = {op[len(p):]: v for op, v in self._alloc(st, layer + 1).items()}
            for lo, hi in ch:
                m = hi - lo
                xs = x[lo:hi]
                h = self._norm(T, "attn", p + "attn_norm", lo, hi, xs)
                q = self._quantize(T, "attn", lo, hi, h, normed=True)
                qv = self._linear(st, p + "wq", lo, q, w["wq"])
                self._bf(T["wq"])[lo:hi] = qv
                kk = self._linear(st, p + "wk", lo, q, w["wk"])
                vv = self._linear(st, p + "wv", lo, q, w["wv"])
                self._bf(T["wk"])[lo:hi], self._bf(T["wv"])[lo:hi] = kk, vv
                s_c, p_c = sl[lo:hi], pos[lo:hi]
                if kv is None:
                    self.k[layer, s_c, p_c] = kk.view(torch.int16)
                    self.v[layer, s_c, p_c] = vv.view(torch.int16)
                    keys = (self.k[layer, lo:hi] if whole else self.k[layer, s_c]).view(torch.bfloat16)
                    vals = (self.v[layer, lo:hi] if whole else self.v[layer, s_c]).view(torch.bfloat16)
                else:
                    kv.write(layer, kk, vv, slice(lo, hi))
                    keys, vals, _ = kv.read(layer, slice(lo, hi))
                a0 = self._tick(sec, "forward", c0)
                sc, pr, att = self.attn(qv.view(m, sh.heads, sh.head_dim), keys, vals, p_c + 1)
                c0 = self._tick(sec, "attention", a0)
                del sc
                if "softmax" in T:
                    soft = self._bf(T["softmax"])[lo:hi].view(m, sh.heads, PV_K)
                    soft.zero_()
                    soft[:, :, :pr.shape[-1]] = pr
                att = att.reshape(m, sh.width)
                self._bf(T["pv"])[lo:hi] = att
                q = self._quantize(T, "o", lo, hi, att)
                o = self._linear(st, p + "wo", lo, q, w["wo"])
                if "wo" in T:
                    self._bf(T["wo"])[lo:hi] = o
                xs = O.add(xs, o)
                self._bf(T["attn_residual"])[lo:hi] = xs
                h = self._norm(T, "mlp", p + "mlp_norm", lo, hi, xs)
                q = self._quantize(T, "up", lo, hi, h, normed=True)
                u = self._linear(st, p + "wup", lo, q, w["wup"])
                self._bf(T["wup"])[lo:hi] = u
                q = self._quantize(T, "down", lo, hi, u, relu=True)
                del u
                dn = self._linear(st, p + "wdown", lo, q, w["wdown"])
                if "wdown" in T:
                    self._bf(T["wdown"])[lo:hi] = dn
                xs = O.add(xs, dn)
                self._bf(T["mlp_residual"])[lo:hi] = xs
                x[lo:hi] = xs
            self._close(st, layer + 1)
        c0 = self._tick(sec, "forward", c0)
        head = self.params.head()
        if not self.params.resident:
            c0 = self._tick(sec, "stage", c0)
        g = len(self.groups) - 1
        T = self._alloc(st, g)
        nxt = torch.empty(n, dtype=torch.int64, device=dev)
        for lo, hi in ch:
            h = self._norm(T, "final", "final_norm", lo, hi, x[lo:hi])
            q = self._quantize(T, "head", lo, hi, h, normed=True)
            logits = self._linear(st, "head", lo, q, head)
            self._bf(T["head"])[lo:hi] = logits
            nxt[lo:hi] = O.argmax(logits)
        T["argmax"][0].view(torch.int16)[:, 0] = nxt.to(torch.int16)
        self._close(st, g)
        c0 = self._tick(sec, "forward", c0)
        roots = self.committer.finish(st)
        self._tick(sec, "commit_wait" if self.committer.streams else "commit", c0)
        if roots.seconds is not None:
            sec["commit_gpu"] = roots.seconds
        out = nxt.cpu().numpy().astype(np.int64)
        res = StepResult(st.step, int(positions[0]), np.asarray(tokens).copy(), out, roots.root, roots.tables,
                         self.value_bytes(n), sec, slots, positions, roots.slot_roots, st.salt_key)
        if self.keep:
            self.steps.append(StepTables(st.step, positions, n, st.host(), slots=slots, salt_key=st.salt_key))
        self.results.append(res)
        for sub in self.subcircuits:
            sub.serve(self, res)
        return res

    def value_bytes(self, n: int) -> int:
        return n * sum(t.nbytes + 4 * t.quant for g in self.groups for t in g)

    def serve(self, kv: SlotKV, requests: Sequence[Request], steps: int | None = None,
              salt_keys: list[bytes] | None = None) -> list[StepResult]:
        """Continuous batching: ``requests`` through ``kv``'s slots (one a driver slot), step after step until every one has
        finished (or ``steps`` steps). Finished requests are in ``self.finished``."""
        sh = self.shape
        if (kv.shape.slots, kv.shape.layers, kv.shape.ctx, kv.shape.head_dim) != \
                (self.batch, sh.layers, sh.context, sh.kv_heads * sh.head_dim):
            raise ValueError(f"the slots' {kv.shape} aren't {self.batch} slots of {sh.name}'s KV (layers, context, head_dim)")
        self.kv = kv
        kv.submit(requests)
        while not kv.idle and (steps is None or len(self.results) < steps):
            plan = kv.begin()
            res = self.step_slots(plan.slots.cpu().numpy(), plan.positions.cpu().numpy(), plan.tokens.cpu().numpy(),
                                  salt_key=salt_keys[plan.step] if salt_keys else None)
            res.seqs, res.finished = plan.seqs, kv.end(res.tokens_out)
            self.finished += res.finished
        return self.results

    def run(self, prompts: np.ndarray, steps: int, salt_keys: list[bytes] | None = None) -> list[StepResult]:
        """``prompts`` (B, P) bytes; positions 0 .. steps - 1 (prefill one byte a step, then decode)."""
        if prompts.shape[0] != self.batch or steps > self.shape.context:
            raise ValueError(f"{self.batch} prompts and at most {self.shape.context} steps")
        toks = prompts[:, 0].astype(np.int64)
        for t in range(steps):
            res = self.step(t, toks, salt_key=salt_keys[t] if salt_keys else None)
            toks = prompts[:, t + 1].astype(np.int64) if t + 1 < prompts.shape[1] else res.tokens_out
        return self.results
