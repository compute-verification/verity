"""Draw units of a committed step and replay each from the step's committed values, bit for bit.

Every unit's inputs are read from the committed tables (its own step's, and the earlier steps' for the KV cache) and from
the registered weights (`weights.Weights`, never the store: so a replayed linear also checks the decode to staging), and its
output word is recomputed:

* by the Definition on main through `evaluation.evaluate_batch`: the W8A8 coordinates (`ScaledMmS8Coordinate_v1{K,
  GEMM_SM90_S8}`; the down projection's K-split parts and epilogue give the same word, an exact s32 sum) and attention's
  QK^T and PV (`GemmCoordinate_v2{64 | 32, GEMM_HOPPER}`, PV padded with +0);
* by a numpy twin of circuits' plan where the Definition is not on main yet (CIR-100 and circuits' batches 1 and 2), each
  op as section 1 states it, named ``twin:<Definition>`` in the bundle: RMSNorm's parts, the quantizer's absmax, scale and
  pairs, softmax's max, sum, reciprocal and outputs, the residual and embedding adds, argmax.

The draw is per family (one template at one place in the step): ``k`` units uniformly with replacement from a FIXTURE coin
(SHA-512 of the run's seed), until PRF-170's plan and live coins draw them.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import mufu as MU

from . import attention as A
from .commit import StepTables
from .shape import AMAX_CHUNK, RMS_CHUNK, RMS_GROUP, Shape
from .weights import Weights


@lru_cache(maxsize=None)
def w8a8(k: int):
    return bind(G.ScaledMmS8Coordinate, K=k, **G.GEMM_SM90_S8)


def u16(row: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(row).view(np.uint16)


def f32w(row: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(row).view(np.float32)


def bf(w) -> np.ndarray:
    return A.f32_of_bf16(np.asarray(w, dtype=np.uint16))


def rbf(x) -> np.ndarray:
    return A.bf16_of_f32(np.atleast_1d(np.asarray(x, dtype=np.float32))).reshape(np.shape(x))


@dataclass
class Unit:
    family: str
    template: str
    step: int
    position: int
    slot: int
    index: int
    inputs: list[np.ndarray]
    got: int
    want: int

    @property
    def ok(self) -> bool:
        return self.got == self.want

    def doc(self) -> dict:
        return {"family": self.family, "template": self.template, "step": self.step, "position": self.position,
                "slot": self.slot, "index": self.index, "got": f"{self.got:#x}", "replayed": f"{self.want:#x}", "ok": self.ok,
                "inputs": [{"dtype": str(a.dtype), "shape": list(a.shape),
                            "sha512": hashlib.sha512(np.ascontiguousarray(a).tobytes()).hexdigest()} for a in self.inputs]}


@dataclass
class Family:
    name: str
    template: str
    per_slot: int
    replay: Callable[[int, int], tuple[list[np.ndarray], int, int]]   # (slot, index) -> (inputs, got word, replayed word)


class Replayer:
    def __init__(self, shape: Shape, weights: Weights, steps: list[StepTables], tokens: list[np.ndarray]) -> None:
        """``steps[s]`` the committed tables of step s (position s), ``tokens[s]`` the bytes it embedded."""
        self.shape, self.weights, self.steps, self.tokens = shape, weights, steps, tokens
        self._codes = lru_cache(maxsize=4)(weights.codes)
        self._scales = lru_cache(maxsize=None)(weights.scales)

    def row(self, step: int, op: str, slot: int) -> np.ndarray:
        return self.steps[step].tables[op][0][slot]

    def word(self, step: int, op: str, slot: int) -> np.ndarray:
        return self.steps[step].tables[op][1][slot:slot + 1]

    # -- families ---------------------------------------------------------------------------------------------------

    def families(self, step: int) -> list[Family]:
        sh, st = self.shape, self.steps[step]
        t = st.position
        out = [self._add(step, "embed", None, None)]
        prev = "embed"
        for layer in range(sh.layers):
            p = f"model.layers.{layer}"
            out += self._rms(step, f"{p}.input_layernorm", prev, self.weights.gain(layer, "attn"))
            out += self._quant(step, f"{p}.self_attn.qkv_quant", f"{p}.input_layernorm", False)
            out.append(self._linear(step, f"{p}.self_attn.qkv_proj", f"{p}.self_attn.qkv_quant", f"l{layer}.qkv"))
            out += self._attention(step, p, t)
            out += self._quant(step, f"{p}.self_attn.o_quant", f"{p}.self_attn.pv", False)
            out.append(self._linear(step, f"{p}.self_attn.o_proj", f"{p}.self_attn.o_quant", f"l{layer}.o"))
            out.append(self._add(step, f"{p}.resid_attn", prev, f"{p}.self_attn.o_proj"))
            out += self._rms(step, f"{p}.post_attention_layernorm", f"{p}.resid_attn", self.weights.gain(layer, "mlp"))
            out += self._quant(step, f"{p}.mlp.up_quant", f"{p}.post_attention_layernorm", False)
            out.append(self._linear(step, f"{p}.mlp.up_proj", f"{p}.mlp.up_quant", f"l{layer}.up"))
            out += self._quant(step, f"{p}.mlp.down_quant", f"{p}.mlp.up_proj", True)
            out.append(self._linear(step, f"{p}.mlp.down_proj", f"{p}.mlp.down_quant", f"l{layer}.down"))
            out.append(self._add(step, f"{p}.resid_mlp", f"{p}.resid_attn", f"{p}.mlp.down_proj"))
            prev = f"{p}.resid_mlp"
        out += self._rms(step, "final_norm", prev, self.weights.gain(None, "final"))
        out += self._quant(step, "final_quant", "final_norm", False)
        out.append(self._linear(step, "lm_head", "final_quant", "head"))
        out.append(self._argmax(step))
        return out

    def _linear(self, step: int, op: str, qop: str, tensor: str) -> Family:
        n, k = self.weights.dims[tensor]
        tpl = w8a8(k)

        def replay(b: int, i: int):
            x, sx = self.row(step, qop, b), self.word(step, qop, b).view(np.uint32)
            wrow, sw = self._codes(tensor)[i], self._scales(tensor)[i:i + 1].view(np.uint32)
            want = int(np.asarray(evaluate_batch(tpl, x.view(np.uint8)[None], sx, wrow.view(np.uint8)[None], sw))[0])
            return [x, sx, wrow, sw], int(u16(self.row(step, op, b))[i]), want
        return Family(op, descriptor_id(tpl), n, replay)

    def _attention(self, step: int, p: str, t: int) -> list[Family]:
        sh = self.shape
        qkv, d, hd = f"{p}.self_attn.qkv_proj", sh.d, sh.head_dim
        n = t + 1

        def keys(b: int) -> np.ndarray:
            return np.stack([u16(self.row(s, qkv, b))[d:d + hd] for s in range(step - t, step + 1)])

        def values(b: int) -> np.ndarray:
            return np.stack([u16(self.row(s, qkv, b))[d + hd:] for s in range(step - t, step + 1)])

        def scores(b: int) -> np.ndarray:
            return u16(self.row(step, f"{p}.self_attn.scores", b)).reshape(sh.heads, n)

        def qk(b: int, i: int):
            h, s = divmod(i, n)
            q, k = u16(self.row(step, qkv, b))[h * hd:(h + 1) * hd], keys(b)[s]
            want = int(np.asarray(evaluate_batch(A.qk_template(), q[None], k[None]))[0])
            return [q, k], int(scores(b)[h, s]), want

        def soft(b: int, h: int):
            return A.softmax(scores(b)[h][None, None])

        def smax(b: int, h: int):
            return [scores(b)[h]], int(u16(self.row(step, f"{p}.self_attn.softmax_max", b))[h]), int(soft(b, h).m[0, 0])

        def ssum(b: int, h: int):
            return ([scores(b)[h]], int(np.ascontiguousarray(self.row(step, f"{p}.self_attn.softmax_sum", b)).view(np.uint32)[h]),
                    int(soft(b, h).s[0, 0]))

        def srecip(b: int, h: int):
            got = int(np.ascontiguousarray(self.row(step, f"{p}.self_attn.softmax_recip", b)).view(np.uint32)[h])
            return [scores(b)[h]], got, int(soft(b, h).r[0, 0])

        def sout(b: int, i: int):
            h, j = divmod(i, n)
            got = int(u16(self.row(step, f"{p}.self_attn.probs", b)).reshape(sh.heads, n)[h, j])
            return [scores(b)[h]], got, int(soft(b, h).probs[0, 0, j])

        def pv(b: int, i: int):
            h, c = divmod(i, hd)
            pr = u16(self.row(step, f"{p}.self_attn.probs", b)).reshape(sh.heads, n)[h]
            x, w = A.pad(pr), A.pad(values(b)[:, c])
            want = int(np.asarray(evaluate_batch(A.pv_template(), x[None], w[None]))[0])
            return [x, w], int(u16(self.row(step, f"{p}.self_attn.pv", b))[i]), want

        return [Family(f"{p}.self_attn.scores", descriptor_id(A.qk_template()), sh.heads * n, qk),
                Family(f"{p}.self_attn.softmax_max", "twin:SoftmaxMax_v1", sh.heads, smax),
                Family(f"{p}.self_attn.softmax_sum", "twin:SoftmaxRecip_v1(sum)", sh.heads, ssum),
                Family(f"{p}.self_attn.softmax_recip", "twin:SoftmaxRecip_v1", sh.heads, srecip),
                Family(f"{p}.self_attn.probs", "twin:SoftmaxOut_v1", sh.heads * n, sout),
                Family(f"{p}.self_attn.pv", descriptor_id(A.pv_template()), sh.heads * hd, pv)]

    def _rms(self, step: int, op: str, src: str, gain: np.ndarray) -> list[Family]:
        sh = self.shape
        nc = sh.d // RMS_CHUNK
        ng = -(-nc // RMS_GROUP)

        def x(b: int) -> np.ndarray:
            return bf(u16(self.row(step, src, b)))

        def chunk_sum(b: int, c: int) -> np.float32:
            acc = np.float32(0)
            for v in x(b)[c * RMS_CHUNK:(c + 1) * RMS_CHUNK]:
                acc = np.float32(acc + np.float32(v * v))
            return acc

        def chunks(b: int) -> np.ndarray:
            return f32w(self.row(step, f"{op}.sumsq", b))

        def groups(b: int) -> np.ndarray:
            return f32w(self.row(step, f"{op}.groups", b))

        def sumsq(b: int, c: int):
            return [u16(self.row(step, src, b))], int(chunks(b).view(np.uint32)[c]), int(np.float32(chunk_sum(b, c)).view(np.uint32))

        def group(b: int, g: int):
            acc = np.float32(0)
            for v in chunks(b)[g * RMS_GROUP:(g + 1) * RMS_GROUP]:
                acc = np.float32(acc + v)
            return [chunks(b)], int(groups(b).view(np.uint32)[g]), int(acc.view(np.uint32))

        def recip(b: int, _: int):
            acc = np.float32(0)
            for v in groups(b):
                acc = np.float32(acc + v)
            mean = np.float32(acc / np.float32(sh.d))
            rs = np.float32(np.float32(1) / np.float32(np.sqrt(np.float32(mean + np.float32(sh.eps)))))
            return [groups(b)], int(f32w(self.row(step, f"{op}.rs", b)).view(np.uint32)[0]), int(rs.view(np.uint32))

        def out(b: int, i: int):
            rs = f32w(self.row(step, f"{op}.rs", b))[0]
            h = rbf(np.float32(x(b)[i] * rs))
            want = int(rbf(np.float32(bf(gain[i]) * bf(h))))
            return [u16(self.row(step, src, b))[i:i + 1], np.array([rs]), gain[i:i + 1]], int(u16(self.row(step, op, b))[i]), want

        return [Family(f"{op}.sumsq", "twin:RmsSumSqPart_v1{C=128}", nc, sumsq),
                Family(f"{op}.groups", "twin:F32SumPart_v1{G=64}", ng, group),
                Family(f"{op}.rs", "twin:RmsNormRecip_v1", 1, recip),
                Family(op, "twin:RmsNormOut_v1", sh.d, out)]

    def _quant(self, step: int, op: str, src: str, relu: bool) -> list[Family]:
        k = self.steps[step].tables[src][0].shape[1] // 2
        nc = k // AMAX_CHUNK

        def x(b: int) -> np.ndarray:
            v = bf(u16(self.row(step, src, b)))
            return np.maximum(v, np.float32(0)) if relu else v

        def amax(b: int) -> np.float32:
            return bf(u16(self.row(step, f"{op}.amax", b)))[0]

        def scale(a: np.float32) -> np.float32:
            return np.float32(a / np.float32(127)) if a > 0 else np.float32(1)

        def parts(b: int, c: int):
            want = int(rbf(np.abs(x(b)[c * AMAX_CHUNK:(c + 1) * AMAX_CHUNK]).max()))
            return [u16(self.row(step, src, b))], int(u16(self.row(step, f"{op}.amax_parts", b))[c]), want

        def total(b: int, _: int):
            ps = u16(self.row(step, f"{op}.amax_parts", b))
            return [ps], int(u16(self.row(step, f"{op}.amax", b))[0]), int(rbf(bf(ps).max()))

        def sc(b: int, half: int):
            got = int(self.word(step, op, b).view(np.uint32)[0])
            want = int(scale(amax(b)).view(np.uint32))
            sh = 16 * half
            return [u16(self.row(step, f"{op}.amax", b))], (got >> sh) & 0xFFFF, (want >> sh) & 0xFFFF

        def pair(b: int, j: int):
            s = scale(amax(b))
            v = x(b)[2 * j:2 * j + 2]
            codes = np.clip(np.rint((v / s).astype(np.float32)), -128, 127).astype(np.int8)
            got = self.row(step, op, b)[2 * j:2 * j + 2]
            return [v, np.array([amax(b)])], int(got.view(np.uint16)[0]), int(codes.view(np.uint16)[0])

        tag = "{RELU}" if relu else ""
        return [Family(f"{op}.amax_parts", f"twin:RowAbsMax_v1{{128}}{tag}", nc, parts),
                Family(f"{op}.amax", "twin:RowAbsMax_v1{chunks}", 1, total),
                Family(f"{op}.scale", "twin:QuantS8Scale_v1{HALF}", 2, sc),
                Family(op, f"twin:QuantS8Pair_v1{tag}", k // 2, pair)]

    def _add(self, step: int, op: str, a: str | None, b_op: str | None) -> Family:
        sh = self.shape
        tok, pos = (self.weights.tok(), self.weights.pos()) if a is None else (None, None)

        def replay(b: int, i: int):
            if a is None:
                st = self.steps[step]
                byte = int(self._token(step, b))
                x, y = tok[byte, i], pos[st.position, i]
            else:
                x, y = u16(self.row(step, a, b))[i], u16(self.row(step, b_op, b))[i]
            want = int(rbf(np.float32(bf(x) + bf(y))))
            return [np.array([x, y], np.uint16)], int(u16(self.row(step, op, b))[i]), want
        return Family(op, "twin:EmbeddingCoordinate_v1" if a is None else "twin:Bf16Add_v2", sh.d, replay)

    def _token(self, step: int, b: int) -> int:
        return int(self.tokens[step][b])

    def _argmax(self, step: int) -> Family:
        def replay(b: int, _: int):
            logits = u16(self.row(step, "lm_head", b))
            return [logits], int(u16(self.row(step, "argmax", b))[0]), int(np.argmax(bf(logits)))
        return Family("argmax", "twin:ArgmaxCombine_v1", 1, replay)

    # -- the draw ---------------------------------------------------------------------------------------------------

    def draw(self, step: int, k: int, seed: bytes, only: Callable[[Family], bool] = lambda f: True) -> list[Unit]:
        """``k`` units of every family of ``step`` (or of those ``only`` keeps), replayed."""
        st = self.steps[step]
        units = []
        for fam in self.families(step):
            if not only(fam):
                continue
            total = fam.per_slot * st.batch
            h = hashlib.sha512(b"e2e-h200/draw-fixture\0" + seed + fam.name.encode() + step.to_bytes(4, "big")).digest()
            rng = np.random.Generator(np.random.PCG64(int.from_bytes(h[:16], "big")))
            for u in rng.integers(0, total, size=k):
                slot, idx = divmod(int(u), fam.per_slot)
                inputs, got, want = fam.replay(slot, idx)
                units.append(Unit(fam.name, fam.template, step, st.position, slot, idx, inputs, got, want))
        return units
