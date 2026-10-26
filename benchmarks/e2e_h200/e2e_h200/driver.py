"""The step driver: one decode step of every active sequence slot through every layer, weights staged from the store by its
decode call, every unit's output committed by the per-step commit call, the step's tables kept on the host for replay.

The slots are `slots.SlotKV`'s: each step admits queued requests into the lowest free slots, runs every active slot at its own
position (position p reads byte p of its prompt while p < P, else the slot's previous argmax), and frees the slots whose
request finished, for the next step to reuse. `serve` runs requests through the slots by continuous batching; `run` is the
lockstep case, every slot admitted at step 0.

Per layer: stage the six linears into their staging buffers (`store.Store.stage`); RMSNorm, quantize, Q, K and V; the slot's
key and value into its KV cache (`SlotKV.write`, K's and V's bf16 rows); scores, softmax and PV over the slot's positions
0 .. p (`attention.attend`, on the CPU, a group of equal positions at a time); quantize, O, the residual add; RMSNorm,
quantize, up; ReLU and quantize, down, the residual add. Then the final RMSNorm, quantize, the head and argmax. The small
tensors (scales, gains, embeddings) are read through the same decode call once (`store.Store.load`).

The units are circuits' plan's, one unit a row (CMP-92 af6de1f1, revised 23:30Z, section 1b), and each table of the step
(`commit.StepTables`) is one op's outputs, one row a slot at its position, named as `cpu_step` labels them: ``embed``; per
layer ``l<i>.`` ``attn_norm``, ``attn_quant`` (codes, and the scale as the row's word), ``wq``, ``wk``, ``wv``, ``scores``
and ``softmax`` (heads x context, +0 past the slot's position, so a row's layout depends only on the shape), ``pv``,
``o_quant``, ``wo``, ``attn_residual``, ``mlp_norm``, ``up_quant``, ``wup``, ``down_quant`` (RELU), ``wdown``,
``mlp_residual``; then ``final_norm``, ``head_quant``, ``head`` and ``argmax``. A row unit's output is its table's row; a
coordinate unit's (a linear's, QK^T's, PV's) is one 16-bit word of it. The plan's gathers are wiring: the gathered row is the
coordinates' table row.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from . import attention as A
from . import ops as O
from .commit import Committer, StepTables
from .shape import EPS, Shape, check
from .slots import Finished, Request, SlotKV, SlotShape
from .store import Store
from .weights import LINEARS


def words(t: torch.Tensor) -> np.ndarray:
    """A bf16 tensor's u16 words on the host."""
    return t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def from_words(w: np.ndarray, device: Any) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(w, dtype=np.uint16).view(np.int16)).to(device).view(torch.bfloat16)


@dataclass
class StepResult:
    """One step: row b is slot ``slots[b]`` (ascending) at ``positions[b]``, with request ``seqs[b]``, input byte
    ``tokens_in[b]`` and argmax ``tokens_out[b]``; ``finished`` the requests whose last position this was."""
    step: int
    slots: np.ndarray
    positions: np.ndarray
    seqs: tuple[int, ...]
    tokens_in: np.ndarray
    tokens_out: np.ndarray
    root: bytes
    table_roots: tuple[tuple[str, bytes], ...]
    value_bytes: int
    finished: list[Finished] = field(default_factory=list)
    seconds: dict[str, float] = field(default_factory=dict)


class Driver:
    def __init__(self, shape: Shape, store: Store, batch: int, committer: Committer, device: str = "cpu",
                 clock: Callable[[], float] = time.perf_counter, tamper: str | None = None) -> None:
        """``tamper``: a table whose committed bytes have bit 0 flipped at every step (a loop whose committed outputs are not
        its model's), which replay must find."""
        self.shape, self.store, self.batch, self.committer, self.tamper = check(shape), store, batch, committer, tamper
        self.device, self.clock = torch.device(device), clock
        dev, w = self.device, store.weights
        names = [f"l{i}.{k}" for i in range(shape.layers) for k in LINEARS] + ["head"]
        self.sw = {n: torch.from_numpy(np.ascontiguousarray(store.load(f"{n}.scale"))).to(dev) for n in names}
        self.gain = {n: from_words(store.load(n), dev)
                     for n in [f"l{i}.{k}" for i in range(shape.layers) for k in ("attn_norm", "mlp_norm")] + ["final_norm"]}
        self.tok = from_words(store.load("tok"), dev)    # (V, H): row v is token v's embedding
        self.pos = from_words(np.ascontiguousarray(store.load("pos").T), dev)    # (C, H)
        self.eps = torch.tensor([EPS], dtype=torch.int32).view(torch.float32)[0].to(dev)
        self.staging = {k: torch.empty(w.dims[f"l0.{k}"], dtype=torch.int8, device=dev) for k in LINEARS}
        self.head = torch.empty(w.dims["head"], dtype=torch.int8, device=dev)
        self.kv: SlotKV | None = None
        self.steps: list[StepTables] = []
        self.results: list[StepResult] = []
        self.finished: list[Finished] = []

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _timed(self, sec: dict[str, float], key: str, fn: Callable[[], Any]) -> Any:
        self._sync()
        t0 = self.clock()
        out = fn()
        self._sync()
        sec[key] = sec.get(key, 0.0) + self.clock() - t0
        return out

    @staticmethod
    def _put(st: StepTables, op: str, y: Any) -> Any:
        st.put(op, y)
        return y

    def _quant(self, st: StepTables, op: str, x: torch.Tensor, relu: bool = False) -> O.Quant:
        q = O.quantize(x, relu)
        st.put(op, q.codes, q.scale)
        return q

    def _linear(self, st: StepTables, op: str, q: O.Quant, w: torch.Tensor, name: str) -> torch.Tensor:
        return self._put(st, op, O.s8_linear(q, w, self.sw[name]))

    def step(self) -> StepResult:
        """One step of the slots: admit, run every active slot at its position, commit, and free the finished slots."""
        sh, dev, kv = self.shape, self.device, self.kv
        if kv is None:
            raise RuntimeError("no slots: call serve() or run()")
        plan = kv.begin()
        b = plan.n
        slots, positions = plan.slots.cpu().numpy(), plan.positions.cpu().numpy()
        st = StepTables(plan.step, slots, positions)
        sec: dict[str, float] = {}
        x = self._put(st, "embed", O.add(self.tok[plan.tokens.to(dev)], self.pos[plan.positions.to(dev)]))
        for layer in range(sh.layers):
            p, sg = f"l{layer}.", self.staging
            for k in LINEARS:
                self._timed(sec, "stage", lambda k=k: self.store.stage(p + k, sg[k]))
            c0 = self.clock()
            h = self._put(st, p + "attn_norm", O.rmsnorm(x, self.gain[p + "attn_norm"], self.eps))
            q = self._quant(st, p + "attn_quant", h)
            qw = words(self._linear(st, p + "wq", q, sg["wq"], p + "wq"))
            kv.write(layer, self._linear(st, p + "wk", q, sg["wk"], p + "wk"), self._linear(st, p + "wv", q, sg["wv"], p + "wv"))
            keys, values, _ = kv.read(layer)
            kw, vw = words(keys), words(values)
            self._sync()
            sec["linear+row"] = sec.get("linear+row", 0.0) + self.clock() - c0
            c0 = self.clock()
            sc, sm, att = A.attend(qw.reshape(b, sh.heads, sh.head_dim), kw, vw, positions)
            sec["attention"] = sec.get("attention", 0.0) + self.clock() - c0
            st.put(p + "scores", sc)
            st.put(p + "softmax", sm.probs)
            st.put(p + "pv", att)
            c0 = self.clock()
            q = self._quant(st, p + "o_quant", from_words(att.reshape(b, sh.width), dev))
            x = self._put(st, p + "attn_residual", O.add(x, self._linear(st, p + "wo", q, sg["wo"], p + "wo")))
            h = self._put(st, p + "mlp_norm", O.rmsnorm(x, self.gain[p + "mlp_norm"], self.eps))
            q = self._quant(st, p + "up_quant", h)
            u = self._linear(st, p + "wup", q, sg["wup"], p + "wup")
            q = self._quant(st, p + "down_quant", u, relu=True)
            x = self._put(st, p + "mlp_residual", O.add(x, self._linear(st, p + "wdown", q, sg["wdown"], p + "wdown")))
            self._sync()
            sec["linear+row"] += self.clock() - c0
        self._timed(sec, "stage", lambda: self.store.stage("head", self.head))
        h = self._put(st, "final_norm", O.rmsnorm(x, self.gain["final_norm"], self.eps))
        q = self._quant(st, "head_quant", h)
        nxt = O.argmax(self._linear(st, "head", q, self.head, "head"))
        st.put("argmax", nxt.to(torch.int16)[:, None])
        if self.tamper:
            codes, word = st.tables[self.tamper]
            st.tables[self.tamper] = (codes ^ 1, word)
        roots = self._timed(sec, "commit", lambda: self.committer.commit(st))
        out = nxt.cpu().numpy().astype(np.int64)
        done = kv.end(out)
        res = StepResult(st.step, slots, positions, plan.seqs, plan.tokens.cpu().numpy(), out, roots.root, roots.tables,
                         st.value_bytes(), done, sec)
        self.steps.append(StepTables(st.step, slots, positions, st.host()))
        self.results.append(res)
        self.finished += done
        return res

    def serve(self, kv: SlotKV, requests: Sequence[Request], steps: int | None = None) -> list[StepResult]:
        """Continuous batching: ``requests`` through ``kv``'s slots, step after step until every one has finished (or
        ``steps`` steps). Finished requests are in ``self.finished``."""
        sh = self.shape
        if (kv.shape.layers, kv.shape.ctx, kv.shape.head_dim) != (sh.layers, sh.context, sh.head_dim):
            raise ValueError(f"the slots' {kv.shape} don't hold {sh.name}'s KV (layers, context, head_dim)")
        self.kv = kv
        kv.submit(requests)
        while not kv.idle and (steps is None or len(self.results) < steps):
            self.step()
        return self.results

    def run(self, prompts: np.ndarray, steps: int) -> list[StepResult]:
        """Lockstep: ``prompts`` (B, P) bytes, all B admitted at step 0 into the driver's B slots; positions 0 .. steps - 1
        (prefill one byte a step, then decode)."""
        sh, p = self.shape, prompts.shape[1]
        if prompts.shape[0] != self.batch or steps > sh.context or p > sh.context:
            raise ValueError(f"{self.batch} prompts of at most {sh.context} bytes and at most {sh.context} steps")
        kv = SlotKV(SlotShape(self.batch, sh.layers, sh.context, sh.head_dim, prompt=p, generate=max(1, steps - p + 1)),
                    self.device, admit=self.batch)
        return self.serve(kv, [Request(b, bytes(prompts[b].astype(np.uint8))) for b in range(self.batch)], steps)
