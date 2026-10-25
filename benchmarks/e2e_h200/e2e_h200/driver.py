"""The step driver: one decode step of every sequence slot through every layer, weights staged from the store by its decode
call, every op's output committed by the per-step commit call, the step's tables kept on the host for replay.

Per layer: stage QKV, O, up and down into the staging buffers (`store.Store.stage`); RMSNorm, quantize, QKV; the slot's key and
value into its KV cache; scores, softmax and PV (`attention`, on the CPU); quantize, O, the residual add; RMSNorm, quantize,
up; ReLU and quantize, down, the residual add. Then the final RMSNorm, quantize, the head and argmax. Every slot runs in
lockstep at one position per step: position t reads byte t of its prompt while t < P, else the previous step's argmax.

Each op's output is one table of the step (`commit.StepTables`), named by its op path (`model.layers.{l}.…`, `embed`,
`final_norm…`, `lm_head`, `argmax`): every unit's 16-bit outputs are words of those rows (an f32 is its two halves).
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
from .shape import Shape
from .store import Store

LINEARS = ("qkv", "o", "up", "down")


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
        """``tamper``: an op path whose committed words have bit 0 flipped at every step (a loop whose committed outputs
        are not its model's), which replay must find."""
        self.shape, self.store, self.batch, self.committer, self.tamper = shape, store, batch, committer, tamper
        self.device, self.clock = torch.device(device), clock
        w = store.weights
        dev = self.device
        self.sw = {name: torch.from_numpy(w.scales(name)).to(dev) for name, _, _ in shape.tensors()}
        self.gain = {(layer, which): from_words(w.gain(layer, which), dev)
                     for layer in range(shape.layers) for which in ("attn", "mlp")}
        self.gain[None, "final"] = from_words(w.gain(None, "final"), dev)
        self.tok, self.pos = from_words(w.tok(), dev), from_words(w.pos(), dev)
        dims = {kind: w.dims[f"l0.{kind}"] for kind in LINEARS}
        self.staging = {kind: torch.empty(dims[kind], dtype=torch.int8, device=dev) for kind in LINEARS}
        self.head = torch.empty(w.dims["head"], dtype=torch.int8, device=dev)
        self.k = np.zeros((shape.layers, batch, shape.ctx, shape.head_dim), np.uint16)
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

    def _rms(self, st: StepTables, op: str, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        r = O.rmsnorm(x, g, self.shape.eps)
        st.put(f"{op}.sumsq", r.chunks)
        st.put(f"{op}.groups", r.groups)
        st.put(f"{op}.rs", r.rs[:, None])
        st.put(op, r.out)
        return r.out

    def _quant(self, st: StepTables, op: str, x: torch.Tensor, relu: bool = False) -> O.Quant:
        q = O.quantize(x, relu)
        st.put(f"{op}.amax_parts", q.parts)
        st.put(f"{op}.amax", q.amax[:, None])
        st.put(op, q.codes, q.scale)
        return q

    def step(self, t: int, tokens: np.ndarray) -> StepResult:
        """Position ``t`` for every slot on ``tokens`` (B,): commits the step and returns its argmax bytes."""
        sh, dev, b = self.shape, self.device, self.batch
        st = StepTables(len(self.steps), t, b)
        sec: dict[str, float] = {}
        tok = torch.from_numpy(tokens.astype(np.int64)).to(dev)
        x = O.add(self.tok[tok], self.pos[t][None, :].expand(b, -1))
        st.put("embed", x)
        for layer in range(sh.layers):
            p = f"model.layers.{layer}"
            for kind in LINEARS:
                self._timed(sec, "stage", lambda kind=kind: self.store.stage(f"l{layer}.{kind}", self.staging[kind]))
            c0 = self.clock()
            h = self._rms(st, f"{p}.input_layernorm", x, self.gain[layer, "attn"])
            q = self._quant(st, f"{p}.self_attn.qkv_quant", h)
            qkv = O.s8_linear(q, self.staging["qkv"], self.sw[f"l{layer}.qkv"])
            st.put(f"{p}.self_attn.qkv_proj", qkv)
            qw = words(qkv)
            self.k[layer, :, t] = qw[:, sh.d:sh.d + sh.head_dim]
            self.v[layer, :, t] = qw[:, sh.d + sh.head_dim:]
            self._sync()
            sec["linear+row"] = sec.get("linear+row", 0.0) + self.clock() - c0
            c0 = self.clock()
            sc = A.scores(qw[:, :sh.d].reshape(b, sh.heads, sh.head_dim), self.k[layer, :, :t + 1])
            sm = A.softmax(sc)
            att = A.pv(sm.probs, self.v[layer, :, :t + 1])
            sec["attention"] = sec.get("attention", 0.0) + self.clock() - c0
            st.put(f"{p}.self_attn.scores", sc)
            st.put(f"{p}.self_attn.softmax_max", sm.m)
            st.put(f"{p}.self_attn.softmax_sum", sm.s)
            st.put(f"{p}.self_attn.softmax_recip", sm.r)
            st.put(f"{p}.self_attn.probs", sm.probs)
            st.put(f"{p}.self_attn.pv", att)
            c0 = self.clock()
            q = self._quant(st, f"{p}.self_attn.o_quant", from_words(att.reshape(b, sh.d), dev))
            x = O.add(x, self._put(st, f"{p}.self_attn.o_proj", O.s8_linear(q, self.staging["o"], self.sw[f"l{layer}.o"])))
            st.put(f"{p}.resid_attn", x)
            h = self._rms(st, f"{p}.post_attention_layernorm", x, self.gain[layer, "mlp"])
            q = self._quant(st, f"{p}.mlp.up_quant", h)
            u = self._put(st, f"{p}.mlp.up_proj", O.s8_linear(q, self.staging["up"], self.sw[f"l{layer}.up"]))
            q = self._quant(st, f"{p}.mlp.down_quant", u, relu=True)
            x = O.add(x, self._put(st, f"{p}.mlp.down_proj", O.s8_linear(q, self.staging["down"], self.sw[f"l{layer}.down"])))
            st.put(f"{p}.resid_mlp", x)
            self._sync()
            sec["linear+row"] += self.clock() - c0
        self._timed(sec, "stage", lambda: self.store.stage("head", self.head))
        h = self._rms(st, "final_norm", x, self.gain[None, "final"])
        q = self._quant(st, "final_quant", h)
        logits = self._put(st, "lm_head", O.s8_linear(q, self.head, self.sw["head"]))
        nxt = O.argmax(logits)
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

    @staticmethod
    def _put(st: StepTables, op: str, y: torch.Tensor) -> torch.Tensor:
        st.put(op, y)
        return y

    def run(self, prompts: np.ndarray, steps: int) -> list[StepResult]:
        """``prompts`` (B, P) bytes; positions 0 .. steps - 1 (prefill one byte a step, then decode)."""
        if prompts.shape[0] != self.batch or steps > self.shape.ctx:
            raise ValueError(f"{self.batch} prompts and at most {self.shape.ctx} steps")
        toks = prompts[:, 0].astype(np.int64)
        for t in range(steps):
            res = self.step(t, toks)
            toks = prompts[:, t + 1].astype(np.int64) if t + 1 < prompts.shape[1] else res.tokens_out
        return self.results
