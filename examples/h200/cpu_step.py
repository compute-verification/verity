"""The frozen architecture's decode step on the CPU: an MQA decoder of any `weights.Shape` (the H200's frozen shape, or a
tiny instance of it), every operation a proof block in circuits' forms (their plan's row semantics), and every unit's inputs and
outputs recorded, so `replay` has a step to draw from before the GPU driver commits one.

It computes in torch, the way a kernel computes: f32 tensors with one IEEE rounding an operation, every reduction a loop in
ascending index from +0, ``torch.round`` (half to even) for the quantizer, and torch's bf16 cast (RNE). Two functions torch
has no form of come from main's Definitions' kernels: the BF16 tensor-core coordinate (`gemm.GemmCoordinate_v2` on
``GEMM_HOPPER``, QK^T and PV) and ``ex2.approx.ftz`` (`mufu.MufuEx2Ftz`, softmax's exponent). None of `replay`'s functions
is used, so a replay that matches is two implementations of circuits' semantics agreeing.

**The forms** (`Forms`; `FROZEN` is circuits' choice at the frozen shape, a tiny instance takes smaller ones to exercise
every level):

* RMSNorm of a row of N: ``RmsSumSqPart{C}`` per chunk of C, ``F32SumPart{G}`` per group of G chunk sums, ``RmsNormRecip{N,
  G=N/(C G)}``, then ``RmsNormOut`` per coordinate;
* the quantizer of a row of N (``RELU`` on the MLP's hidden row): ``RowAbsMax{A, RELU}`` per chunk of A, then
  ``RowAbsMax{.., false}`` over the partials until one is left, ``QuantS8Scale`` twice, ``QuantS8Pair`` per pair;
* a W8A8 linear over K: per output coordinate, ``S8AccumulatorPart{K=row_codes}`` per part (the weights' registered rows),
  ``S32SumPart{P=fanin}`` levels while more than ``fanin`` parts are left, then ``ScaledMmS8Epilogue{P}``;
* attention at position t, per query head: ``GemmCoordinate_v2{K=d}`` per key s <= t (the one KV head), ``SoftmaxMax{t+1}``,
  ``SoftmaxRecip{t+1}`` twice, ``SoftmaxOut`` per key, ``GemmCoordinatePad{T=t+1, K=pv_k}`` per head dimension;
* ``Bf16Add_v2`` per coordinate of each residual add, ``EmbeddingCoordinate{V}`` per coordinate of the input row, and argmax
  as ``ArgmaxChunkBest{C}`` and ``ArgmaxChunkIdx{C}`` per chunk of the logits and ``ArgmaxCombine{V, C}``.

Every unit outputs 16 bits: a 32-bit value (a part's s32, a scale, rs, a softmax reciprocal) is two units, one a half, which
its readers read as one word. `Recorder` keeps each template's units in the order they are computed, the step's unit order
template-major (`Recorder.population`), each with its label (position, layer, op).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import mufu as MU

HOPPER = G.GEMM_HOPPER["DOT"]
LOG2E = torch.tensor([0x3FB8AA3B], dtype=torch.int32).view(torch.float32)
EPS_1E5 = 0x3727C5AC


@dataclass(frozen=True)
class Forms:
    rms_chunk: int = 128
    rms_group: int = 64
    amax_chunk: int = 128
    part_fanin: int = 64
    argmax_chunk: int = 128
    pv_k: int = 32
    eps: int = EPS_1E5


FROZEN = Forms()


def key(name: str, **statics: Any) -> str:
    if not statics:
        return name
    return name + "{" + ",".join(f"{k}={str(v).lower() if isinstance(v, bool) else getattr(v, 'id', v)}"
                                 for k, v in statics.items()) + "}"


@dataclass
class _Units:
    inputs: dict[str, list[np.ndarray]] = field(default_factory=dict)
    outputs: list[np.ndarray] = field(default_factory=list)
    labels: list[tuple[int, Any]] = field(default_factory=list)
    n: int = 0


class Recorder:
    """Every unit of a run, per template key in the order recorded."""

    def __init__(self) -> None:
        self.t: dict[str, _Units] = {}

    def record(self, k: str, inputs: Mapping[str, np.ndarray], outputs: np.ndarray, label: Any) -> None:
        n = len(outputs)
        u = self.t.setdefault(k, _Units())
        for name, a in inputs.items():
            a = np.asarray(a)
            if len(a) != n:
                raise ValueError(f"{k}: input {name} has {len(a)} rows for {n} units")
            u.inputs.setdefault(name, []).append(a.reshape(n, -1))
        u.outputs.append(np.asarray(outputs).reshape(n, -1))
        u.labels.append((n, label))
        u.n += n

    def population(self) -> list[tuple[str, int]]:
        return [(k, u.n) for k, u in self.t.items()]

    def _arrays(self, k: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
        u = self.t[k]
        ins = {name: np.concatenate(v) for name, v in u.inputs.items()}
        u.inputs, u.outputs = {name: [a] for name, a in ins.items()}, [np.concatenate(u.outputs)]
        return ins, u.outputs[0]

    def label(self, k: str, i: int) -> Any:
        for n, lab in self.t[k].labels:
            if i < n:
                return lab
            i -= n
        raise IndexError(i)

    def units(self, indices: list[int] | None = None) -> list[dict[str, Any]]:
        """The step's units ``indices`` (global, template-major; all by default) in `replay`'s form."""
        out, base = [], 0
        want = None if indices is None else sorted(indices)
        for k, n in self.population():
            local = range(n) if want is None else [i - base for i in want if base <= i < base + n]
            if local:
                ins, outs = self._arrays(k)
                for i in local:
                    out.append({"unit": base + i, "template": k, "label": self.label(k, i),
                                "inputs": {name: _uwords(a[i]) for name, a in ins.items()}, "outputs": _uwords(outs[i])})
            base += n
        return out


def _uwords(a: np.ndarray) -> list[int]:
    a = np.asarray(a).reshape(-1)
    if a.dtype.kind == "i":
        a = a.view(f"u{a.dtype.itemsize}")
    return a.astype(np.uint64).tolist()


# ---- words and torch -----------------------------------------------------------------------------------------------------

def f32(w: np.ndarray) -> torch.Tensor:
    """bf16 (u16) or f32 (u32) words as an f32 tensor."""
    w = np.asarray(w)
    if w.dtype == np.uint16:
        w = w.astype(np.uint32) << np.uint32(16)
    return torch.from_numpy(np.ascontiguousarray(w.astype(np.uint32)).view(np.int32)).view(torch.float32)


def words32(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int32).numpy().view(np.uint32).copy()


def words16(t: torch.Tensor) -> np.ndarray:
    """An f32 tensor's bf16 words (torch's RNE cast)."""
    return t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16).copy()


def halves(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = np.asarray(w).astype(np.uint32)
    return (w & 0xFFFF).astype(np.uint16), (w >> 16).astype(np.uint16)


def seq(t: torch.Tensor) -> torch.Tensor:
    """(..., m) f32 -> (...): sums in ascending index from +0."""
    acc = torch.zeros(t.shape[:-1], dtype=torch.float32)
    for j in range(t.shape[-1]):
        acc = acc + t[..., j]
    return acc


def sqrt_rn(t: torch.Tensor) -> torch.Tensor:
    """sqrt.rn.f32. torch's f32 sqrt on the CPU is not correctly rounded on its vectorized path; f64's, rounded once to
    f32, is (53 >= 2 * 24 + 2 bits)."""
    return torch.sqrt(t.to(torch.float64)).to(torch.float32)


def order16(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).astype(np.int64)
    return np.where(x & 0x8000, -(x & 0x7FFF), x & 0x7FFF)


# ---- the step ------------------------------------------------------------------------------------------------------------

class Step:
    """The decoder of ``shape`` over ``weights`` (name -> `weights.generate`'s array), ``batch`` sequences, recording into
    ``rec``. `position` runs one decode position for every sequence, with the KV cache of the earlier ones."""

    def __init__(self, shape: Any, weights: Callable[[str], np.ndarray], batch: int, forms: Forms = FROZEN,
                 rec: Recorder | None = None) -> None:
        self.s, self.f, self.b = shape, forms, batch
        self.rec = rec if rec is not None else Recorder()
        self.w = {}
        for name in ("tok", "pos", "final_norm", "head", "head.scale"):
            self.w[name] = weights(name)
        for i in range(shape.layers):
            for n in ("attn_norm", "wq", "wk", "wv", "wo", "mlp_norm", "wup", "wdown"):
                self.w[f"l{i}.{n}"] = weights(f"l{i}.{n}")
                if n.startswith("w"):
                    self.w[f"l{i}.{n}.scale"] = weights(f"l{i}.{n}.scale")
        if shape.kv_heads != 1:
            raise ValueError("the frozen architecture is MQA: one KV head")
        self.kc = [np.zeros((batch, 0, shape.head_dim), np.uint16) for _ in range(shape.layers)]
        self.vc = [np.zeros((batch, 0, shape.head_dim), np.uint16) for _ in range(shape.layers)]
        self.t = 0
        self._label: dict[str, Any] = {}

    def _r(self, k: str, inputs: Mapping[str, np.ndarray], outputs: np.ndarray, op: str) -> None:
        self.rec.record(k, inputs, outputs, {**self._label, "op": op})

    # -- row ops

    def rmsnorm(self, x: np.ndarray, gain: np.ndarray, op: str) -> np.ndarray:
        n, f = x.shape[1], self.f
        c = min(f.rms_chunk, n)
        nc = n // c
        g = min(f.rms_group, nc)
        if n % c or nc % g:
            raise ValueError(f"RMSNorm of {n}: chunks of {c}, groups of {g}")
        xf = f32(x)
        sq = xf * xf
        cs = torch.stack([seq(sq[:, j * c:(j + 1) * c]) for j in range(nc)], 1)
        csw = words32(cs)
        for j in range(nc):
            for h, part in enumerate(halves(csw[:, j])):
                self._r(key("RmsSumSqPart_v1", C=c, HALF=h), {"x": x[:, j * c:(j + 1) * c]}, part, f"{op}.sumsq")
        ng = nc // g
        gs = torch.stack([seq(cs[:, j * g:(j + 1) * g]) for j in range(ng)], 1)
        gsw = words32(gs)
        for j in range(ng):
            for h, part in enumerate(halves(gsw[:, j])):
                self._r(key("F32SumPart_v1", G=g, HALF=h), {"s": csw[:, j * g:(j + 1) * g]}, part, f"{op}.group")
        mean = seq(gs) / torch.tensor(float(n), dtype=torch.float32)
        rs = torch.tensor(1.0, dtype=torch.float32) / sqrt_rn(mean + f32(np.array([f.eps], np.uint32)))
        rsw = words32(rs)
        for h, part in enumerate(halves(rsw)):
            self._r(key("RmsNormRecip_v1", N=n, G=ng, EPS=f.eps, HALF=h), {"s": gsw}, part, f"{op}.recip")
        hh = f32(words16(xf * rs[:, None]))
        out = words16(f32(gain.reshape(1, n)) * hh)
        self._r(key("RmsNormOut_v1"), {"x": x.reshape(-1), "rs": np.repeat(rsw, n), "w": np.tile(gain.reshape(-1), len(x))},
                out.reshape(-1), f"{op}.out")
        return out

    def quantize(self, x: np.ndarray, relu: bool, op: str) -> tuple[np.ndarray, np.ndarray]:
        """x (B, N) bf16 -> codes (B, N) s8 and sx (B,) f32 words."""
        b, n = x.shape
        xf = f32(x)
        if relu:
            xf = torch.where(xf < 0, torch.zeros_like(xf), xf)
        level, first, cur = x, True, None
        while True:
            m = level.shape[1]
            a = min(self.f.amax_chunk, m)
            if m % a:
                raise ValueError(f"an absmax tree over {m}: chunks of {a}")
            vals = f32(level).abs() if not first else xf.abs()
            parts = torch.stack([vals[:, j * a:(j + 1) * a].amax(1) for j in range(m // a)], 1)
            pw = words16(parts)
            for j in range(m // a):
                self._r(key("RowAbsMax_v1", N=a, RELU=relu and first), {"x": level[:, j * a:(j + 1) * a]}, pw[:, j], f"{op}.amax")
            level, first = pw, False
            if pw.shape[1] == 1:
                cur = pw[:, 0]
                break
        amax = f32(cur)
        scale = torch.where(amax > 0, amax / torch.tensor(127.0, dtype=torch.float32), torch.ones_like(amax))
        sw = words32(scale)
        for h, part in enumerate(halves(sw)):
            self._r(key("QuantS8Scale_v1", HALF=h), {"amax": cur}, part, f"{op}.scale")
        codes = torch.round(xf / scale[:, None]).clamp(-128, 127).to(torch.int8).numpy()
        self._r(key("QuantS8Pair_v1", RELU=relu), {"x": x.reshape(b * n // 2, 2), "amax": np.repeat(cur, n // 2)},
                codes.reshape(b * n // 2, 2), f"{op}.pair")
        return codes, sw

    def linear(self, xq: np.ndarray, sx: np.ndarray, name: str, op: str) -> np.ndarray:
        """codes (B, K) s8, sx (B,) f32 words, the weights ``name`` (N, K) s8 and scales -> (B, N) bf16."""
        w, sw = self.w[name], self.w[f"{name}.scale"].reshape(-1).view(np.uint32)
        b, k = xq.shape
        n, rc = w.shape[0], self.s.row_codes
        p = k // rc
        xp = torch.from_numpy(xq.astype(np.int64)).view(b, p, rc)
        wp = torch.from_numpy(w.astype(np.int64)).view(n, p, rc)
        acc = torch.einsum("bpk,npk->bnp", xp, wp).contiguous()
        accw = (acc.numpy() & 0xFFFFFFFF).astype(np.uint32)
        for j in range(p):
            xr, wr = np.repeat(xq[:, j * rc:(j + 1) * rc], n, 0), np.tile(w[:, j * rc:(j + 1) * rc], (b, 1))
            for h, part in enumerate(halves(accw[:, :, j].reshape(-1))):
                self._r(key("S8AccumulatorPart_v1", K=rc, HALF=h), {"x": xr, "wrow": wr}, part, f"{op}.part")
        fan = self.f.part_fanin
        while p > fan:
            if p % fan:
                raise ValueError(f"an s32 tree over {p} parts: groups of {fan}")
            nxt = acc.view(b, n, p // fan, fan).sum(-1)
            nw = (nxt.numpy() & 0xFFFFFFFF).astype(np.uint32)
            for j in range(p // fan):
                for h, part in enumerate(halves(nw[:, :, j].reshape(-1))):
                    self._r(key("S32SumPart_v1", P=fan, HALF=h), {"parts": accw[:, :, j * fan:(j + 1) * fan].reshape(b * n, fan)},
                            part, f"{op}.sum")
            acc, accw, p = nxt, nw, p // fan
        total = acc.sum(-1).to(torch.int32).to(torch.float32)
        y = words16((total * f32(sx)[:, None]) * f32(sw)[None, :])
        self._r(key("ScaledMmS8Epilogue_v1", P=p), {"parts": accw.reshape(b * n, p), "sx": np.repeat(sx, n), "sw": np.tile(sw, b)},
                y.reshape(-1), f"{op}.epilogue")
        return y

    def softmax(self, x: np.ndarray, op: str) -> np.ndarray:
        """x (R, N) bf16 scores -> (R, N) bf16 probabilities."""
        r, n = x.shape
        o = order16(x)
        m = x[np.arange(r), np.argmax(o, 1)]
        self._r(key("SoftmaxMax_v1", N=n), {"x": x}, m, f"{op}.max")
        d = f32(x) - f32(m)[:, None]
        t = d * LOG2E
        e = torch.from_numpy(np.asarray(evaluate_batch(MU.MufuEx2Ftz, words32(t).reshape(-1)), np.uint32).view(np.int32)
                             ).view(torch.float32).view(r, n)
        rr = torch.tensor(1.0, dtype=torch.float32) / seq(e)
        rw = words32(rr)
        for h, part in enumerate(halves(rw)):
            self._r(key("SoftmaxRecip_v1", N=n, HALF=h), {"x": x, "m": m}, part, f"{op}.recip")
        p = words16(e * rr[:, None])
        self._r(key("SoftmaxOut_v1"), {"x": x.reshape(-1), "m": np.repeat(m, n), "r": np.repeat(rw, n)}, p.reshape(-1), f"{op}.out")
        return p

    def attention(self, q: np.ndarray, layer: int, op: str) -> np.ndarray:
        s, b = self.s, self.b
        d, hq, t1 = s.head_dim, s.heads, self.t + 1
        kc, vc = self.kc[layer], self.vc[layer]
        qh = q.reshape(b, hq, d)
        xs = np.repeat(qh[:, :, None, :], t1, 2).reshape(-1, d)
        ks = np.repeat(kc[:, None, :, :], hq, 1).reshape(-1, d)
        qk = bind(G.GemmCoordinate, K=d, DOT=HOPPER)
        scores = np.asarray(evaluate_batch(qk, xs, ks), np.uint16)
        self._r(qk.id, {"x": xs, "wrow": ks}, scores, f"{op}.qk")
        p = self.softmax(scores.reshape(b * hq, t1), f"{op}.softmax").reshape(b, hq, t1)
        pv = bind(G.GemmCoordinate, K=self.f.pv_k, DOT=HOPPER)
        px = np.repeat(p[:, :, None, :], d, 2).reshape(-1, t1)
        vw = np.repeat(np.transpose(vc, (0, 2, 1))[:, None, :, :], hq, 1).reshape(-1, t1)
        pad = np.zeros((len(px), self.f.pv_k - t1), np.uint16)
        out = np.asarray(evaluate_batch(pv, np.concatenate([px, pad], 1), np.concatenate([vw, pad], 1)), np.uint16)
        self._r(key("GemmCoordinatePad_v1", T=t1, K=self.f.pv_k, DOT=HOPPER), {"x": px, "wrow": vw}, out, f"{op}.pv")
        return out.reshape(b, hq * d)

    def add(self, a: np.ndarray, c: np.ndarray, op: str) -> np.ndarray:
        y = words16(f32(a) + f32(c))
        self._r(key("Bf16Add_v2"), {"a": a.reshape(-1), "b": c.reshape(-1)}, y.reshape(-1), op)
        return y

    def argmax(self, logits: np.ndarray) -> np.ndarray:
        b, v = logits.shape
        c = min(self.f.argmax_chunk, v)
        o = np.where(((logits & 0x7F80) == 0x7F80) & ((logits & 0x7F) != 0), np.iinfo(np.int64).min, order16(logits))
        bests, idxs = [], []
        for j in range(-(-v // c)):
            oc = o[:, j * c:(j + 1) * c]
            i = np.argmax(oc, 1)
            some = oc[np.arange(b), i] > order16(np.array(0xFF80))
            best = np.where(some, logits[np.arange(b), j * c + i], 0xFF80).astype(np.uint16)
            idx = np.where(some, i, 0).astype(np.uint16)
            chunk = logits[:, j * c:(j + 1) * c]
            self._r(key("ArgmaxChunkBest_v1", C=chunk.shape[1]), {"x": chunk}, best, "argmax.best")
            self._r(key("ArgmaxChunkIdx_v1", C=chunk.shape[1]), {"x": chunk}, idx, "argmax.idx")
            bests.append(best)
            idxs.append(idx)
        best, idx = np.stack(bests, 1), np.stack(idxs, 1)
        run, run_i = logits[:, 0].copy(), np.zeros(b, np.int64)
        for j in range(best.shape[1]):
            gt = order16(best[:, j]) > order16(run)
            run_i = np.where(gt, j * c + idx[:, j].astype(np.int64), run_i)
            run = np.where(gt, best[:, j], run)
        out = run_i.astype(np.uint16)
        self._r(key("ArgmaxCombine_v1", V=v, C=c), {"x0": logits[:, 0], "best": best, "idx": idx}, out, "argmax.combine")
        return out

    # -- the position

    def position(self, toks: np.ndarray) -> np.ndarray:
        """Decode position ``self.t`` on each sequence's byte ``toks`` (B,): every unit recorded; -> the next bytes (B,)."""
        s, b, t = self.s, self.b, self.t
        if t >= s.context:
            raise ValueError(f"position {t} is past the context {s.context}")
        self._label = {"pos": t, "layer": None}
        tok_w, pos_w = self.w["tok"], self.w["pos"]
        toks = np.asarray(toks, np.uint16)
        col = np.tile(tok_w, (b, 1))
        pos = np.tile(pos_w[:, t], b)
        x = words16(f32(tok_w[:, toks].T.reshape(-1)) + f32(pos))
        self._r(key("EmbeddingCoordinate_v1", V=s.vocab), {"tok": np.repeat(toks, s.width), "col": col, "pos": pos}, x, "embed")
        x = x.reshape(b, s.width)
        for i in range(s.layers):
            self._label = {"pos": t, "layer": i}
            p = f"l{i}."
            n1 = self.rmsnorm(x, self.w[p + "attn_norm"], "attn_norm")
            c1, s1 = self.quantize(n1, False, "attn_quant")
            q = self.linear(c1, s1, p + "wq", "wq")
            k = self.linear(c1, s1, p + "wk", "wk")
            v = self.linear(c1, s1, p + "wv", "wv")
            self.kc[i] = np.concatenate([self.kc[i], k[:, None, :]], 1)
            self.vc[i] = np.concatenate([self.vc[i], v[:, None, :]], 1)
            a = self.attention(q, i, "attn")
            c2, s2 = self.quantize(a, False, "o_quant")
            x = self.add(x, self.linear(c2, s2, p + "wo", "wo"), "attn_residual")
            n2 = self.rmsnorm(x, self.w[p + "mlp_norm"], "mlp_norm")
            c3, s3 = self.quantize(n2, False, "up_quant")
            u = self.linear(c3, s3, p + "wup", "wup")
            c4, s4 = self.quantize(u, True, "down_quant")
            x = self.add(x, self.linear(c4, s4, p + "wdown", "wdown"), "mlp_residual")
        self._label = {"pos": t, "layer": None}
        nf = self.rmsnorm(x, self.w["final_norm"], "final_norm")
        cf, sf = self.quantize(nf, False, "head_quant")
        nxt = self.argmax(self.linear(cf, sf, "head", "head"))
        self.t += 1
        return nxt


def decode(step: Step, prompts: list[bytes], generate: int) -> list[bytes]:
    """Run ``prompts`` (one length P each, one a sequence) for P + ``generate`` - 1 positions: the bytes each produced."""
    p = {len(x) for x in prompts}
    if len(p) != 1 or len(prompts) != step.b:
        raise ValueError(f"{step.b} prompts of one length")
    p = p.pop()
    seqs = [list(x) for x in prompts]
    for t in range(p + generate - 1):
        nxt = step.position(np.array([sq[t] for sq in seqs]))
        for sq, n in zip(seqs, nxt):
            if t >= p - 1:
                sq.append(int(n))
    return [bytes(sq) for sq in seqs]
