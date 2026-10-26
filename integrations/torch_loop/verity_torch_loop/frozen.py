"""The frozen H200 shape's decode step on CPU (CMP-92 Phase 2, frozen by compute-accounting 25 Oct 23:00Z), an integration of
Verity's (`verity.ml.windows.interface`) beside `loop`'s MLP: PRF-173's fallback, so that drawn units of the demo's shape are
proved and checked by Lean whether or not the H200's replay bundle arrives.

* **The shape** (:data:`FROZEN`): an MQA decoder of width 16,384, 256 query heads of 64 with one KV head, MLP 65,536 with
  ReLU, byte vocabulary 256, context 32, pre-norm (RMSNorm), learned absolute positions, greedy decoding; 52 layers, of which
  the loop runs the first ``layers``. Seeded random weights (the text needn't be readable), drawn directly as s8 codes with
  an f32 scale per output channel, so a layer's 2.6 GB of weights take seconds.
* **One decode step** of one sequence at position ``t``: positions 0 .. t - 1 run first, unrecorded, to fill the KV cache
  (their K and V rows are the step's inputs), then position t's units are recorded, through ``layers`` layers and the head.
* **Its units: every matmul's output coordinates.** The W8A8 linears (Q, K, V, O, up and the head over K = 16,384; down
  over K = 65,536) are `ScaledMmS8Coordinate_v1{K, GEMM_SM90_S8}`, the H200's s8 binding: ``loop.linear``'s function
  (an exact s32 sum, then the token's and the channel's f32 scales and the bf16 cast). Attention's QK^T (each head's query
  against each cached key, K = 64) and PV (each head's probabilities against one dimension of V over the context) are
  `GemmCoordinate_v3{K=64, GEMM_HOPPER}`, computed through the Definition's own kernel (`evaluate_batch`). PV's 32 positions
  are padded with +0 to K = 64, because a 32-word bf16 row is 64 bytes and C-Flock's unit statement needs a row longer
  than one SHA-512 block (111 bytes), so QK^T and PV share one template.
* **The other ops** (the embedding add, RMSNorm, softmax, ReLU, the per-row quantizer, the residual adds, argmax) run in
  torch and are not units until circuits registers their Definitions (CIR-100). Their outputs are committed as the next
  unit's inputs.
* **One window per template** (:data:`TAGS`), as the sampling planner hands a plan off (PRF-166 part 3): each its own circuit,
  run root (the run id and the tag) and law, so each is drawn on its own (``subset:k`` per template).
* ``tamper``: every recorded output word has bit 0 flipped, which the proof of any drawn unit must refuse. ``keep``: only
  the first ``keep`` units of each op are recorded (the step is still computed whole), for a smoke run at the shape's K.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as Fn

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops import gemm as G
from verity.ml.windows import interface as IF

from .loop import bits, quantize

#: RMSNorm's epsilon, added in f32
EPS = 1e-5
NAME = "torch-loop/frozen-h200"


@dataclass(frozen=True)
class Shape:
    width: int = 16384
    heads: int = 256
    head_dim: int = 64
    mlp: int = 65536
    vocab: int = 256
    context: int = 32
    layers: int = 52

    @property
    def kv(self) -> int:
        """The one KV head's width."""
        return self.head_dim

    @property
    def dot_k(self) -> int:
        """The BF16 coordinates' K: the head dimension, and the context padded to it."""
        return max(self.head_dim, self.context)


#: compute-accounting's frozen shape for one H200 (CMP-92, 25 Oct 23:00Z)
FROZEN = Shape()


def static(k: int):
    """A W8A8 linear's output coordinate over ``k`` inputs, the H200's s8 binding."""
    return bind(G.ScaledMmS8Coordinate, K=k, **G.GEMM_SM90_S8)


def dynamic(k: int):
    """A BF16 output coordinate of two run values over ``k`` inputs, the H200's BF16 step."""
    return bind(G.GemmCoordinate, K=k, **G.GEMM_HOPPER)


#: the window tags: the K = width and K = MLP linears, and attention's BF16 coordinates
TAGS = ("s8-width", "s8-mlp", "bf16-dot")


def templates(shape: Shape) -> dict[str, object]:
    return {"s8-width": static(shape.width), "s8-mlp": static(shape.mlp), "bf16-dot": dynamic(shape.dot_k)}


def units(shape: Shape, t: int, layers: int, keep: int | None = None) -> dict[str, int]:
    """Each window's unit count for one step at position ``t`` through ``layers`` layers and the head; ``keep``: of each op, at
    most that many."""
    k = (lambda n: n) if keep is None else (lambda n: min(n, keep))
    s, h, d = shape, shape.heads, shape.head_dim
    return {"s8-width": layers * (k(s.width) + k(s.kv) + k(s.kv) + k(s.width) + k(s.mlp)) + k(s.vocab),
            "s8-mlp": layers * k(s.width),
            "bf16-dot": layers * (k(h * (t + 1)) + k(h * d))}


@dataclass(frozen=True)
class Linear:
    wq: torch.Tensor     # (N, K) s8
    sw: torch.Tensor     # (N,) f32


@dataclass(frozen=True)
class Layer:
    g_attn: torch.Tensor  # (width,) bf16
    q: Linear             # (width, width)
    k: Linear             # (kv, width)
    v: Linear             # (kv, width)
    o: Linear             # (width, width)
    g_mlp: torch.Tensor
    up: Linear            # (mlp, width)
    down: Linear          # (width, mlp)


class Model:
    """The decoder's weights from ``seed``, its first ``layers`` layers: s8 codes uniform on [-127, 127] and channel scales
    that give each linear unit-variance outputs on unit-variance inputs."""

    def __init__(self, shape: Shape = FROZEN, layers: int = 1, seed: int = 0) -> None:
        if not 1 <= layers <= shape.layers:
            raise ValueError(f"layers in 1 .. {shape.layers}, not {layers}")
        g = torch.Generator().manual_seed(seed)
        self.shape, self.seed = shape, seed

        def lin(n: int, k: int, scale: float = 1.0) -> Linear:
            wq = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8)
            sw = (1 + 0.1 * torch.randn(n, generator=g)) * (scale / (73.3 * math.sqrt(k)))
            return Linear(wq, sw.to(torch.float32))

        def gain() -> torch.Tensor:
            return (1 + 0.1 * torch.randn(shape.width, generator=g)).to(torch.bfloat16)

        w = shape.width
        self.tok = torch.randn(shape.vocab, w, generator=g).to(torch.bfloat16)
        self.pos = (0.5 * torch.randn(shape.context, w, generator=g)).to(torch.bfloat16)
        # the score scale 1/sqrt(head_dim) is folded into the query channels' scales
        self.layers = [Layer(gain(), lin(w, w, shape.head_dim ** -0.5), lin(shape.kv, w), lin(shape.kv, w), lin(w, w),
                             gain(), lin(shape.mlp, w), lin(w, shape.mlp)) for _ in range(layers)]
        self.g_final = gain()
        self.head = lin(shape.vocab, w)


def rmsnorm(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return Fn.rms_norm(x.to(torch.float32), (x.shape[-1],), g.to(torch.float32), EPS).to(torch.bfloat16)


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 + bf16: the f32 sum rounded to nearest even."""
    return (a.to(torch.float32) + b.to(torch.float32)).to(torch.bfloat16)


def bf16(a: np.ndarray) -> torch.Tensor:
    """u16 words as a bf16 tensor."""
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.uint16).view(np.int16)).view(torch.bfloat16)


def linear(xq: torch.Tensor, sx: torch.Tensor, lin: Linear) -> torch.Tensor:
    """`loop.linear` on one token row ``xq`` (1, K), in chunks of output channels of 1 GiB in f64: the s32 sum is exact in f64
    (|sum| <= K * 2^14 < 2^31 < 2^53), so BLAS computes it; then ``bf16((f32(acc) * sx) * sw)`` as `loop.linear` does.
    -> (N,) bf16."""
    x = xq.to(torch.float64)[0]
    chunk, out = max(1, (1 << 27) // x.numel()), []
    for lo in range(0, lin.wq.shape[0], chunk):
        acc = (lin.wq[lo:lo + chunk].to(torch.float64) @ x).to(torch.int32)
        out.append(((acc.to(torch.float32) * sx) * lin.sw[lo:lo + chunk]).to(torch.bfloat16))
    return torch.cat(out)


class _Cache:
    def __init__(self, shape: Shape, layers: int) -> None:
        self.k = torch.zeros(layers, shape.context, shape.kv, dtype=torch.bfloat16)
        self.v = torch.zeros(layers, shape.context, shape.kv, dtype=torch.bfloat16)


@dataclass
class FrozenStep:
    """One decode step of ``model`` at position ``t`` of the sequence ``prompt`` (its bytes, then greedy bytes once it runs
    out), every unit of position t recorded into its template's window (module docstring); ``run_id`` names the run, which
    every window's root binds, beside its tag. ``check``: the units of each recorded op held to their Definition, at most
    that many (`mismatched`)."""
    model: Model
    prompt: bytes
    t: int
    tags: tuple[str, ...] = TAGS
    tamper: bool = False
    keep: int | None = None
    run_id: str = ""
    check: int = 4
    mismatched: dict[str, int] = field(default_factory=dict)
    checked: dict[str, int] = field(default_factory=dict)
    generated: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0 <= self.t < self.model.shape.context or not self.prompt or set(self.tags) - set(TAGS) or not self.tags:
            raise ValueError(f"a position in 0 .. {self.model.shape.context - 1}, a prompt and tags among {TAGS}")
        self.templates = templates(self.model.shape)
        self._windows: dict[str, IF.Window] = {}

    @property
    def units(self) -> dict[str, int]:
        n = units(self.model.shape, self.t, len(self.model.layers), self.keep)
        return {tag: n[tag] for tag in self.tags}

    def run_root(self, tag: str) -> str:
        """Window ``tag``'s run root: SHA-512 of what the loop served (its shape, seed, layers, prompt and position), the tag and
        the run's id."""
        s = self.model.shape
        return hashlib.sha512(json.dumps({"loop": NAME, "shape": [s.width, s.heads, s.head_dim, s.mlp, s.vocab, s.context],
                                          "layers": len(self.model.layers), "seed": self.model.seed, "prompt": self.prompt.hex(),
                                          "t": self.t, "keep": self.keep, "tag": tag, "id": self.run_id}).encode()).hexdigest()

    def circuit(self, tag: str) -> IF.Circuit:
        return IF.Circuit(f"{NAME}/{tag}", ((self.templates[tag], self.units[tag]),))

    def windows(self, law: dict, cap: int, caps: dict[str, int] | None = None) -> dict[str, IF.Window]:
        """Every window under ``law`` in members of at most ``cap`` units (``caps``: a tag's own), every unit recorded."""
        self._windows = {tag: IF.Window(self.circuit(tag), self.run_root(tag), law, (caps or {}).get(tag, cap))
                         for tag in self.tags}
        self.run()
        return self._windows

    def _record(self, tag: str, inputs: list[np.ndarray], out: np.ndarray, label: dict) -> None:
        win = self._windows.get(tag)
        if win is None:
            return
        n = len(out) if self.keep is None else min(len(out), self.keep)
        inputs, out = [a[:n] for a in inputs], out[:n]
        if self.tamper:
            out = out ^ np.uint16(1)
        fn = self.templates[tag]
        c = min(n, self.check)
        if c:
            pick = np.unique(np.linspace(0, n - 1, c).astype(int))
            want = np.asarray(evaluate_batch(fn, *[a[pick] for a in inputs]), dtype=np.uint16).reshape(-1)
            self.checked[tag] = self.checked.get(tag, 0) + len(pick)
            self.mismatched[tag] = self.mismatched.get(tag, 0) + int((want != out[pick]).sum())
        win.record(fn, inputs, [out], label)

    def _static(self, rec: bool, x: torch.Tensor, lin: Linear, label: dict) -> torch.Tensor:
        """The W8A8 linear on one bf16 row ``x``: its quantization and its N coordinates, recorded as units when ``rec``."""
        xq, sx = quantize(x[None, :])
        y = linear(xq, sx, lin)
        if rec:
            n, k = lin.wq.shape
            tag = "s8-width" if k == self.model.shape.width else "s8-mlp"
            m = n if self.keep is None else min(n, self.keep)
            self._record(tag, [np.repeat(bits(xq), m, 0), np.repeat(bits(sx), m), bits(lin.wq[:m]), bits(lin.sw[:m])],
                         bits(y), label)
        return y

    def _dynamic(self, rec: bool, x: np.ndarray, w: np.ndarray, label: dict) -> torch.Tensor:
        """``len(w)`` BF16 coordinates ``x[i] . w[i]`` (u16 words), recorded as units when ``rec``."""
        y = np.asarray(evaluate_batch(self.templates["bf16-dot"], x, w), dtype=np.uint16).reshape(-1)
        if rec:
            self._record("bf16-dot", [x, w], y, label)
        return bf16(y)

    def _position(self, p: int, tok: int, cache: _Cache) -> int:
        """Position ``p`` on byte ``tok`` through the layers and the head; its units recorded when it is the step's."""
        s, rec = self.model.shape, p == self.t
        hd, kd = s.head_dim, s.dot_k
        x = add(self.model.tok[tok], self.model.pos[p])
        for li, ly in enumerate(self.model.layers):
            at = {"position": p, "layer": li}
            h = rmsnorm(x, ly.g_attn)
            q = self._static(rec, h, ly.q, {**at, "op": f"model.layers.{li}.attn.q_proj"})
            cache.k[li, p] = self._static(rec, h, ly.k, {**at, "op": f"model.layers.{li}.attn.k_proj"})
            cache.v[li, p] = self._static(rec, h, ly.v, {**at, "op": f"model.layers.{li}.attn.v_proj"})
            qw = bits(q).reshape(s.heads, hd)
            kw = bits(cache.k[li, :p + 1])
            scores = self._dynamic(rec, np.repeat(qw, p + 1, 0), np.tile(kw, (s.heads, 1)),
                                   {**at, "op": f"model.layers.{li}.attn.qk"}).view(s.heads, p + 1)
            probs = torch.softmax(scores.to(torch.float32), -1).to(torch.bfloat16)
            pw = np.zeros((s.heads, kd), np.uint16)
            pw[:, :p + 1] = bits(probs)
            vt = np.zeros((hd, kd), np.uint16)
            vt[:, :s.context] = bits(cache.v[li].T.contiguous())
            a = self._dynamic(rec, np.repeat(pw, hd, 0), np.tile(vt, (s.heads, 1)), {**at, "op": f"model.layers.{li}.attn.pv"})
            x = add(x, self._static(rec, a, ly.o, {**at, "op": f"model.layers.{li}.attn.o_proj"}))
            h = rmsnorm(x, ly.g_mlp)
            u = self._static(rec, h, ly.up, {**at, "op": f"model.layers.{li}.mlp.up_proj"})
            r = torch.relu(u.to(torch.float32)).to(torch.bfloat16)
            x = add(x, self._static(rec, r, ly.down, {**at, "op": f"model.layers.{li}.mlp.down_proj"}))
        logits = self._static(rec, rmsnorm(x, self.model.g_final), self.model.head, {"position": p, "op": "lm_head"})
        return int(torch.argmax(logits.to(torch.float32)))

    def run(self) -> None:
        """Positions 0 .. t, each on the prompt's byte or the previous position's argmax."""
        cache = _Cache(self.model.shape, len(self.model.layers))
        nxt = self.prompt[0]
        for p in range(self.t + 1):
            tok = self.prompt[p] if p < len(self.prompt) else nxt
            nxt = self._position(p, tok, cache)
            self.generated.append(nxt)
