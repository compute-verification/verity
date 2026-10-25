"""The step driver: one decode step of every active sequence slot through every layer, weights staged from the store by its
decode call, every op's output committed by the per-step commit call, the step's tables kept on the host for replay.

The slots are `slots.SlotKV`'s: each step admits queued requests into the lowest free slots, runs every active slot at its own
position (position p reads byte p of its prompt while p < P, else the slot's previous argmax), and frees the slots whose
request finished, for the next step to reuse. `serve` runs requests through the slots by continuous batching; `run` is the
lockstep case, every slot admitted at step 0.

Per layer: stage QKV, O, up and down into the staging buffers (`store.Store.stage`); RMSNorm, quantize, QKV; the slot's key and
value into its KV cache (`SlotKV.write`, the bf16 words of QKV's output); scores, softmax and PV over the slot's positions
0 .. p (`attention.attend`, on the CPU); quantize, O, the residual add; RMSNorm, quantize, up; ReLU and quantize, down, the
residual add. Then the final RMSNorm, quantize, the head and argmax.

Each op's output is one table of the step (`commit.StepTables`), named by its op path (`model.layers.{l}.…`, `embed`,
`final_norm…`, `lm_head`, `argmax`): every unit's 16-bit outputs are words of those rows (an f32 is its two halves).
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
from .commit import Committer, StepTables
from .shape import Shape
from .slots import Finished, Request, SlotKV, SlotShape
from .store import Store

LINEARS = ("qkv", "o", "up", "down")


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
        """``batch``: the sequence slots `run` uses (`serve` takes its own `SlotKV`). ``tamper``: an op path whose committed
        words have bit 0 flipped at every step (a loop whose committed outputs are not its model's), which replay must find."""
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
        x = O.add(self.tok[plan.tokens.to(dev)], self.pos[plan.positions.to(dev)])
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
            kv.write(layer, qkv[:, sh.d:sh.d + sh.head_dim], qkv[:, sh.d + sh.head_dim:])
            keys, values, _ = kv.read(layer)
            qw, kw, vw = words(qkv[:, :sh.d]), words(keys), words(values)
            self._sync()
            sec["linear+row"] = sec.get("linear+row", 0.0) + self.clock() - c0
            c0 = self.clock()
            sc, sm, att = A.attend(qw.reshape(b, sh.heads, sh.head_dim), kw, vw, positions)
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
        done = kv.end(out)
        res = StepResult(st.step, slots, positions, plan.seqs, plan.tokens.cpu().numpy(), out, roots.root, roots.tables,
                         st.value_bytes(), done, sec)
        self.steps.append(StepTables(st.step, slots, positions, st.host()))
        self.results.append(res)
        self.finished += done
        return res

    @staticmethod
    def _put(st: StepTables, op: str, y: torch.Tensor) -> torch.Tensor:
        st.put(op, y)
        return y

    def serve(self, kv: SlotKV, requests: Sequence[Request], steps: int | None = None) -> list[StepResult]:
        """Continuous batching: ``requests`` through ``kv``'s slots, step after step until every one has finished (or
        ``steps`` steps). Finished requests are in ``self.finished``."""
        sh = self.shape
        if (kv.shape.layers, kv.shape.ctx, kv.shape.head_dim) != (sh.layers, sh.ctx, sh.head_dim):
            raise ValueError(f"the slots' {kv.shape} don't hold {sh.name}'s KV (layers, ctx, head_dim)")
        self.kv = kv
        kv.submit(requests)
        while not kv.idle and (steps is None or len(self.results) < steps):
            self.step()
        return self.results

    def run(self, prompts: np.ndarray, steps: int) -> list[StepResult]:
        """Lockstep: ``prompts`` (B, P) bytes, all B admitted at step 0 into the driver's B slots; positions 0 .. steps - 1
        (prefill one byte a step, then decode)."""
        p = prompts.shape[1]
        if prompts.shape[0] != self.batch or steps > self.shape.ctx or p > self.shape.ctx:
            raise ValueError(f"{self.batch} prompts of at most {self.shape.ctx} bytes and at most {self.shape.ctx} steps")
        sh = self.shape
        kv = SlotKV(SlotShape(self.batch, sh.layers, sh.ctx, sh.head_dim, prompt=p, generate=max(1, steps - p + 1)),
                    self.device, admit=self.batch)
        return self.serve(kv, [Request(b, bytes(prompts[b].astype(np.uint8))) for b in range(self.batch)], steps)
