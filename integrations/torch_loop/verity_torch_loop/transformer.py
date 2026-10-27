"""The PyTorch demo's standard workload (CMP-92, Daniel 25 Oct): a 2-layer decoder-only transformer in plain PyTorch on CPU,
an integration of Verity's (`verity.ml.windows.interface`) beside the two-layer MLP of `loop`.

* **The model** (:class:`Model`, seeded random weights): width ``H`` = 128, ``HEADS`` = 2 heads of ``D`` = 64, MLP width
  ``F`` = 512, byte vocabulary ``VOCAB`` = 256, context ``CONTEXT`` = 32, learned absolute positions, pre-norm blocks
  (RMSNorm), ReLU MLP, greedy decoding. Every static matmul (QKV, output projection, MLP up and down, the head) is W8A8:
  s8 weight codes with an f32 scale per output channel, each token row quantized per row (`loop.quantize`), and the linear
  `loop.linear`, which is `ScaledMmS8Coordinate_v1{K, DOT=BlackwellS8ImmaDot32_v1}`'s function: one unit per output
  coordinate. The score scale 1/sqrt(D) = 2^-3 is folded into the query channels' scales, which is exact.
* **Attention** is causal over the KV cache. Its two matmuls take both operands from the run: score (h, s) is
  ``q_h . k_h[s]`` for s <= t, and output coordinate (h, d) is ``p_h . V_h[:, d]`` over all ``CONTEXT`` positions and
  ``PV_K - CONTEXT`` more of +0. Past t, p is exactly +0 (softmax of -inf) and the cache's rows are +0. Each is one
  `GemmCoordinate_v2{K=64, DOT}` unit, BF16 with fp32 accumulation. ``DOT`` is the BF16 step of ``dot`` (`BF16_DOTS`):
  ``hopper``, `GEMM_HOPPER`'s (the H200 target, circuits' ruling and the default), or ``ampere``, `GEMM_AMPERE`'s, since
  sm_120 registers no BF16 step. A BF16 ``torch.matmul`` on CPU is another function (oneDNN's blocked fp32 sum), so the loop computes these coordinates
  through the Definition's own numpy kernel (`evaluation.evaluate_batch`).
* **The other ops** (RMSNorm, softmax, ReLU, the per-row quantizer, the residual adds, the embedding add, argmax) run in
  torch. Until circuits registers their Definitions (CIR-100), they are not units, and their outputs are committed as
  the next unit's inputs. The quantizer, ReLU, adds and argmax are already the functions
  ``note:compute-accounting/20261025T2154Z-report-allp-xfmr`` asks for. RMSNorm and softmax are torch's f32 forms until
  then.
* **Serving** (:class:`DecoderLoop`, an `interface.Integration`): each request is ``P`` prompt bytes, decoded one position
  at a time (prefill too) to ``P + G`` bytes. Positions 0 .. P + G - 2 run, and the head and argmax run at P - 1 onwards.
  Every unit is recorded into the window as it is computed, labelled with the request, position, layer and op path.
  ``tamper``: every recorded output word has bit 0 flipped, which the proof of any drawn unit must refuse.
* **The reference** (`reference_logits`): the same network in float32 PyTorch (``torch.nn.functional``), with
  dequantized weights and no activation quantization, teacher-forced on the loop's bytes.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as Fn

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops import gemm as G
from verity.ml.windows import interface as IF

from .loop import bits, linear, quantize

H, HEADS, D, F, VOCAB, CONTEXT = 128, 2, 64, 512, 256, 32
#: RMSNorm's epsilon, added in f32
EPS = 1e-5
#: the BF16 GEMM's tensor-core step by name (module docstring), and the default
BF16_DOTS = {"hopper": G.GEMM_HOPPER, "ampere": G.GEMM_AMPERE}
BF16_DOT = "hopper"
NAME = f"torch-loop/decoder-l2-h{H}"


def static(k: int):
    """A static matmul's template: one output coordinate of a W8A8 linear over ``k`` inputs, sm_120's s8 binding."""
    return bind(G.ScaledMmS8Coordinate, K=k, **G.GEMM_SM120_S8)


def dynamic(k: int, dot: str = BF16_DOT):
    """A dynamic matmul's template: one output coordinate of a BF16 product of two run values over ``k`` inputs, under the
    BF16 step ``dot``."""
    return bind(G.GemmCoordinate, K=k, **BF16_DOTS[dot])


#: attention's output reduction: the ``CONTEXT`` positions, then +0 products to 64. C-Flock pins a unit through a row of
#: more than one SHA-512 block, and 32 BF16 words are one, so PV takes the scores' template.
PV_K = 2 * CONTEXT
assert PV_K == D
def templates(dot: str = BF16_DOT) -> tuple:
    """The circuit's templates in order: the K = H and K = F static coordinates, and attention's scores and outputs (K = D)
    under the BF16 step ``dot``."""
    return static(H), static(F), dynamic(D, dot)


TEMPLATES = templates()


@dataclass(frozen=True)
class Linear:
    wq: torch.Tensor     # (N, K) s8
    sw: torch.Tensor     # (N,) f32

    def weight(self) -> torch.Tensor:
        """The dequantized weight the reference multiplies by, (N, K) f32."""
        return self.wq.to(torch.float32) * self.sw[:, None]


@dataclass(frozen=True)
class Block:
    g_attn: torch.Tensor  # (H,) bf16
    qkv: Linear           # (3H, H): q, then k, then v
    o: Linear             # (H, H)
    g_mlp: torch.Tensor   # (H,) bf16
    up: Linear            # (F, H)
    down: Linear          # (H, F)


class Model:
    """The decoder's weights, drawn from ``seed``: bf16 embeddings and norm gains, W8A8 linears."""

    def __init__(self, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self.seed = seed

        def lin(n: int, k: int, scale: float = 1.0) -> Linear:
            wq, sw = quantize(torch.randn(n, k, generator=g) / math.sqrt(k))
            return Linear(wq, sw * scale)

        def gain() -> torch.Tensor:
            return (1 + 0.1 * torch.randn(H, generator=g)).to(torch.bfloat16)

        self.tok = torch.randn(VOCAB, H, generator=g).to(torch.bfloat16)
        self.pos = (0.5 * torch.randn(CONTEXT, H, generator=g)).to(torch.bfloat16)
        self.blocks: list[Block] = []
        for _ in range(2):
            ga = gain()
            qkv = lin(3 * H, H)
            qkv = Linear(qkv.wq, torch.cat([qkv.sw[:H] * 2.0 ** -3, qkv.sw[H:]]))
            self.blocks.append(Block(ga, qkv, lin(H, H), gain(), lin(F, H), lin(H, F)))
        self.g_final = gain()
        self.head = lin(VOCAB, H)

    def op(self, layer: int | None, name: str) -> str:
        return name if layer is None else f"model.layers.{layer}.{name}"


def rmsnorm(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return Fn.rms_norm(x.to(torch.float32), (x.shape[-1],), g.to(torch.float32), EPS).to(torch.bfloat16)


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """bf16 + bf16: the f32 sum rounded to nearest even (`Bf16Add_v1` on every non-NaN word)."""
    return (a.to(torch.float32) + b.to(torch.float32)).to(torch.bfloat16)


def bf16(a: np.ndarray) -> torch.Tensor:
    """u16 words as a bf16 tensor."""
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.uint16).view(np.int16)).view(torch.bfloat16)


def units(prompts: int, p: int, g: int, dot: str = BF16_DOT) -> dict[str, int]:
    """Each template's unit count (by descriptor id) for ``prompts`` requests of ``p`` prompt bytes decoded to ``p + g``."""
    sh, sf, bf = (descriptor_id(t) for t in templates(dot))
    n = {sh: 0, sf: 0, bf: 0}
    for t in range(p + g - 1):
        n[sh] += 2 * (3 * H + H + F) + (VOCAB if t >= p - 1 else 0)
        n[sf] += 2 * H
        n[bf] += 2 * HEADS * (t + 1 + D)
    return {k: prompts * v for k, v in n.items()}


class _Cache:
    """One request's KV cache: per layer, every position's k and v rows (bf16), +0 where no position has written."""

    def __init__(self) -> None:
        self.k = torch.zeros(2, CONTEXT, H, dtype=torch.bfloat16)
        self.v = torch.zeros(2, CONTEXT, H, dtype=torch.bfloat16)


class DecoderLoop:
    """The decoder serving ``prompts`` (byte strings of one length P), each decoded greedily to P + ``generate`` bytes
    (module docstring); ``run_id`` names the run, which its root binds, and ``dot`` the BF16 step (`BF16_DOTS`)."""

    def __init__(self, model: Model, prompts: list[bytes], generate: int, tamper: bool = False, run_id: str = "",
                 dot: str = BF16_DOT) -> None:
        if dot not in BF16_DOTS:
            raise ValueError(f"a BF16 step of {sorted(BF16_DOTS)}, not {dot!r}")
        lengths = {len(p) for p in prompts}
        if len(lengths) != 1 or min(lengths) < 1 or generate < 1 or min(lengths) + generate > CONTEXT:
            raise ValueError(f"prompts of one length P >= 1 and generate >= 1 with P + generate <= {CONTEXT}, not "
                             f"{sorted(lengths)} and {generate}")
        self.model, self.prompts, self.generate, self.tamper, self.run_id = model, list(prompts), generate, tamper, run_id
        self.dot = dot
        self.p = lengths.pop()
        self.outputs: list[bytes] = []
        self.logits: list[torch.Tensor] = []

    @property
    def units(self) -> dict[str, int]:
        return units(len(self.prompts), self.p, self.generate, self.dot)

    @property
    def run_root(self) -> str:
        """The run's root: SHA-512 of what the loop served (its model's seed, its prompts and length) and its id."""
        return hashlib.sha512(json.dumps({"loop": NAME, "seed": self.model.seed, "prompts": [p.hex() for p in self.prompts],
                                          "generate": self.generate, "id": self.run_id}).encode()).hexdigest()

    def circuit(self) -> IF.Circuit:
        n = self.units
        return IF.Circuit(NAME, tuple((t, n[descriptor_id(t)]) for t in templates(self.dot)))

    def _record(self, window: IF.Window, template, inputs: list[np.ndarray], out: np.ndarray, label: dict) -> None:
        if self.tamper:
            out = out ^ np.uint16(1)
        window.record(template, inputs, [out], label)

    def _static(self, window: IF.Window, x: torch.Tensor, lin: Linear, label: dict) -> torch.Tensor:
        """The W8A8 linear on one bf16 row ``x``: its quantization and its N coordinates, each recorded as a unit."""
        xq, sx = quantize(x[None, :])
        y = linear(xq, sx, lin.wq, lin.sw)[0]
        n, k = lin.wq.shape
        self._record(window, static(k), [np.repeat(bits(xq), n, 0), np.repeat(bits(sx), n), bits(lin.wq), bits(lin.sw)],
                     bits(y), label)
        return y

    def _dynamic(self, window: IF.Window, x: np.ndarray, w: np.ndarray, label: dict) -> torch.Tensor:
        """``len(w)`` BF16 coordinates ``x[i] . w[i]`` (u16 words), each recorded as a unit."""
        fn = dynamic(x.shape[1], self.dot)
        y = np.asarray(evaluate_batch(fn, x, w), dtype=np.uint16)
        self._record(window, fn, [x, w], y, label)
        return bf16(y)

    def _position(self, window: IF.Window, r: int, t: int, tok: int, cache: _Cache) -> torch.Tensor:
        """Position ``t`` of request ``r`` on byte ``tok``: the residual stream after both blocks."""
        m = self.model

        def lab(layer: int, op: str) -> dict:
            return {"request": f"r{r}", "pos": t, "layer": layer, "op": m.op(layer, op)}

        x = add(m.tok[tok], m.pos[t])
        for layer, b in enumerate(m.blocks):
            qkv = self._static(window, rmsnorm(x, b.g_attn), b.qkv, lab(layer, "self_attn.qkv_proj"))
            q, cache.k[layer, t], cache.v[layer, t] = qkv[:H], qkv[H:2 * H], qkv[2 * H:]
            heads = []
            for h in range(HEADS):
                cols = slice(h * D, (h + 1) * D)
                ks = bits(cache.k[layer, :t + 1, cols])
                score = self._dynamic(window, np.repeat(bits(q[cols])[None], t + 1, 0), ks, lab(layer, f"self_attn.qk.h{h}"))
                row = torch.full((CONTEXT,), float("-inf"))
                row[:t + 1] = score.to(torch.float32)
                p = np.zeros(PV_K, np.uint16)
                p[:CONTEXT] = bits(torch.softmax(row, 0).to(torch.bfloat16))
                vt = np.zeros((D, PV_K), np.uint16)
                vt[:, :CONTEXT] = bits(cache.v[layer, :, cols].T.contiguous())
                heads.append(self._dynamic(window, np.repeat(p[None], D, 0), vt, lab(layer, f"self_attn.pv.h{h}")))
            x = add(x, self._static(window, torch.cat(heads), b.o, lab(layer, "self_attn.o_proj")))
            u = self._static(window, rmsnorm(x, b.g_mlp), b.up, lab(layer, "mlp.up_proj"))
            x = add(x, self._static(window, torch.relu(u), b.down, lab(layer, "mlp.down_proj")))
        return x

    def run(self, window: IF.Window) -> None:
        """Serve every request, recording each unit into ``window`` as it is computed."""
        m = self.model
        for r, prompt in enumerate(self.prompts):
            seq, cache, logits = list(prompt), _Cache(), []
            for t in range(self.p + self.generate - 1):
                x = self._position(window, r, t, seq[t], cache)
                if t >= self.p - 1:
                    lab = {"request": f"r{r}", "pos": t, "layer": None, "op": "lm_head"}
                    y = self._static(window, rmsnorm(x, m.g_final), m.head, lab)
                    logits.append(y)
                    seq.append(int(torch.argmax(y.to(torch.float32))))
            self.outputs.append(bytes(seq))
            self.logits.append(torch.stack(logits))

    def window(self, law: dict, cap: int = 16) -> IF.Window:
        """The loop's window under ``law`` in members of at most ``cap`` units, every unit recorded."""
        win = IF.Window(self.circuit(), self.run_root, law, cap)
        self.run(win)
        return win


def reference_logits(model: Model, seq: bytes) -> torch.Tensor:
    """The float32 network on ``seq`` (module docstring): every position's logits, (len(seq), VOCAB)."""
    n = len(seq)
    x = model.tok.to(torch.float32)[list(seq)] + model.pos.to(torch.float32)[:n]
    mask = torch.ones(n, n, dtype=torch.bool).tril()
    for b in model.blocks:
        h = Fn.rms_norm(x, (H,), b.g_attn.to(torch.float32), EPS)
        q, k, v = Fn.linear(h, b.qkv.weight()).split(H, dim=-1)
        q, k, v = (a.view(n, HEADS, D).transpose(0, 1) for a in (q, k, v))
        a = torch.softmax((q @ k.transpose(1, 2)).masked_fill(~mask, float("-inf")), -1) @ v
        x = x + Fn.linear(a.transpose(0, 1).reshape(n, H), b.o.weight())
        h = Fn.rms_norm(x, (H,), b.g_mlp.to(torch.float32), EPS)
        x = x + Fn.linear(torch.relu(Fn.linear(h, b.up.weight())), b.down.weight())
    return Fn.linear(Fn.rms_norm(x, (H,), model.g_final.to(torch.float32), EPS), model.head.weight())
