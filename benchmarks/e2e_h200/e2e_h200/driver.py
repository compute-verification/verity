"""The step driver: one decode step of every sequence slot through every layer, weights staged from the store by its decode
call, every unit's output committed by the per-step commit call, the step's tables kept on the host for replay.

Per layer: stage the six linears into their staging buffers (`store.Store.stage`); RMSNorm, quantize, Q, K and V; the slot's
key and value into its KV cache; scores, softmax and PV (`attention`, on the CPU); quantize, O, the residual add; RMSNorm,
quantize, up; ReLU and quantize, down, the residual add. Then the final RMSNorm, quantize, the head and argmax. Every slot
runs in lockstep at one position per step: position t reads byte t of its prompt while t < P, else the previous step's
argmax. The small tensors (scales, gains, embeddings) are read through the same decode call once (`store.Store.load`).

The units are circuits' plan's, one unit a row (CMP-92 af6de1f1, revised 23:30Z, section 1b), and each table of the step
(`commit.StepTables`) is one op's outputs, one row a slot, named as `cpu_step` labels them: ``embed``; per layer ``l<i>.``
``attn_norm``, ``attn_quant`` (codes, and the scale as the row's word), ``wq``, ``wk``, ``wv``, ``scores`` and ``softmax``
(heads x (t + 1)), ``pv``, ``o_quant``, ``wo``, ``attn_residual``, ``mlp_norm``, ``up_quant``, ``wup``, ``down_quant``
(RELU), ``wdown``, ``mlp_residual``; then ``final_norm``, ``head_quant``, ``head`` and ``argmax``. A row unit's output is its
table's row; a coordinate unit's (a linear's, QK^T's, PV's) is one 16-bit word of it. The plan's gathers are wiring: the
gathered row is the coordinates' table row.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from . import attention as A
from . import ops as O
from .commit import Committer, StepTables
from .shape import EPS, Shape, check
from .store import Store
from .weights import LINEARS


def words(t: torch.Tensor) -> np.ndarray:
    """A bf16 tensor's u16 words on the host."""
    return t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def from_words(w: np.ndarray, device: Any) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(w, dtype=np.uint16).view(np.int16)).to(device).view(torch.bfloat16)


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
        self.k = np.zeros((shape.layers, batch, shape.context, shape.head_dim), np.uint16)
        self.v = np.zeros_like(self.k)
        self.steps: list[StepTables] = []
        self.results: list[StepResult] = []

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

    def step(self, t: int, tokens: np.ndarray) -> StepResult:
        """Position ``t`` for every slot on ``tokens`` (B,): commits the step and returns its argmax bytes."""
        sh, dev, b = self.shape, self.device, self.batch
        st = StepTables(len(self.steps), t, b)
        sec: dict[str, float] = {}
        tok = torch.from_numpy(tokens.astype(np.int64)).to(dev)
        x = self._put(st, "embed", O.add(self.tok[tok], self.pos[t][None, :].expand(b, -1)))
        for layer in range(sh.layers):
            p, sg = f"l{layer}.", self.staging
            for k in LINEARS:
                self._timed(sec, "stage", lambda k=k: self.store.stage(p + k, sg[k]))
            c0 = self.clock()
            h = self._put(st, p + "attn_norm", O.rmsnorm(x, self.gain[p + "attn_norm"], self.eps))
            q = self._quant(st, p + "attn_quant", h)
            qw = words(self._linear(st, p + "wq", q, sg["wq"], p + "wq"))
            self.k[layer, :, t] = words(self._linear(st, p + "wk", q, sg["wk"], p + "wk"))
            self.v[layer, :, t] = words(self._linear(st, p + "wv", q, sg["wv"], p + "wv"))
            self._sync()
            sec["linear+row"] = sec.get("linear+row", 0.0) + self.clock() - c0
            c0 = self.clock()
            sc = A.scores(qw.reshape(b, sh.heads, sh.head_dim), self.k[layer, :, :t + 1])
            sm = A.softmax(sc)
            att = A.pv(sm.probs, self.v[layer, :, :t + 1])
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
        res = StepResult(st.step, t, tokens.copy(), out, roots.root, roots.tables, st.value_bytes(), sec)
        self.steps.append(StepTables(st.step, t, b, st.host()))
        self.results.append(res)
        return res

    def run(self, prompts: np.ndarray, steps: int) -> list[StepResult]:
        """``prompts`` (B, P) bytes; positions 0 .. steps - 1 (prefill one byte a step, then decode)."""
        if prompts.shape[0] != self.batch or steps > self.shape.context:
            raise ValueError(f"{self.batch} prompts and at most {self.shape.context} steps")
        toks = prompts[:, 0].astype(np.int64)
        for t in range(steps):
            res = self.step(t, toks)
            toks = prompts[:, t + 1].astype(np.int64) if t + 1 < prompts.shape[1] else res.tokens_out
        return self.results
