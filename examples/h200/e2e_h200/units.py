"""A step's units, read from its committed tables, and their replay on the CPU against circuits' semantics.

The units are circuits' FAMILIES map's (`plan.Map`): each `Family` is one `plan.Units`, one template key
over one table, its unit i's output that table's bytes at ``offset + i * stride``. `Family.replay` gives a unit's opened
inputs, its committed outputs (its output rows' bytes, as 16- or 32-bit words where the rows are) and what replay computes.

Every unit's words are compositions of the replay lane's templates (`tpl`, ``replay.template(key).evaluate``: main's
Definition where it exists, else the lane's reference), until the replay lane adds the map's templates:

* ``EmbeddingCoordinate_v1{V}``: that template;
* ``RmsSumSqChunks_v1{C, K}``: ``RmsSumSqPart_v1{C = K}`` a chunk; ``RmsNormOutSums_v1{C, S, G, N, EPS}``:
  ``F32SumPart_v1{G = S / G}`` a group, ``RmsNormRecip_v1{N, G, EPS}`` and ``RmsNormOut_v1`` a word;
* ``RowAbsMax_v1{N, RELU}``: that template; ``QuantChunkS8_v1{N, RELU}``: ``QuantS8Pair_v1{RELU}`` a pair and
  ``QuantS8Scale_v1``; ``RmsNormAbsMax_v1{C, S, G, N, EPS}`` and ``RmsNormQuantCodes_v1{C, S, G, N, EPS}`` (``fused``
  ``norm``): ``RmsNormOutSums_v1``'s, then ``RowAbsMax_v1{C, false}``'s or ``QuantChunkS8_v1{C, false}``'s codes, and
  ``QuantScaleS8_v1``: ``QuantS8Scale_v1``;
* ``ScaledMmS8CoordinateRows_v1{K, R, DOT}``: ``ScaledMmS8Coordinate_v1{K}`` on the R rows joined (main's test holds the two
  equal); ``ScaledMmS8ResidualCoordinateRows_v1{K, R, DOT}``: that, then ``Bf16Add_v2`` with the residual stream's word;
* ``QkSoftmaxRowPad_v1{T, D}``: ``GemmCoordinate_v2{K = D}`` a key and `softmax_row` over the slot's n = position + 1 keys,
  +0 past them; ``PvRow_v1{T, K, D, DOT}``: ``GemmCoordinatePad_v1{T, K}`` a dimension, absent keys' V rows all-zero;
  ``QkSoftmaxPvPad_v1{T, K, D, DOT}`` (``fused`` ``pv``): the first's row, then the second's of it;
* ``Bf16AddRow_v2{N}``: ``Bf16Add_v2`` a word; ``Argmax_v2{V}``: `argmax_row`;
* ``AllGather_v1{WORLD, N}``: the copy of its parts' rows. Its parts and its output are the same table bytes here, so its
  replay only restates them; the commit holds both, and a drawn gather opens its own rows against the step's root.

A unit's inputs are read from the committed tables (an earlier op's row, an earlier step's K or V row) and the registered
weights (`weights.Weights`, from the generator, not the store, so a wrong decode shows as a mismatch). Row b of a step is
slot ``slots[b]`` at its own position p; its attention reads the slot's K and V rows committed at positions 0 .. p, each at
the step and row the slot ran it (`Replayer.history`), never the live cache.

A `Subcircuit` joins the step's integrity registration as one more execution phase, drawn with the step's families: PoUW's
per-call-noise tiles (`percall`) plug in as ``MODULE:FACTORY`` (`load_subcircuit`, ``run.py --subcircuit``).
"""
from __future__ import annotations

import hashlib
import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import numpy as np

from ._replay import RP
from .commit import StepTables
from .plan import Map, Units
from .shape import EPS, PV_K, Shape
from .weights import Weights

Replay = Callable[[int, int], tuple[list[np.ndarray], np.ndarray, np.ndarray]]


def tpl(name: str, **statics: Any) -> RP.Template:
    """The replay lane's template ``name`` at ``statics`` (in its stated order): `replay.template`, so every unit here is
    evaluated by the one reference of each template (main's Definition where it exists)."""
    return RP.template(RP.key_of(name, statics))


def _ev(t: RP.Template, *arrays: np.ndarray) -> np.ndarray:
    return t.evaluate(*arrays).reshape(len(arrays[0]), -1)


def _word(name: str, *arrays: np.ndarray, **statics: Any) -> np.ndarray:
    """A 32-bit value two units output as halves (``HALF`` 0 and 1), joined: (n,) words."""
    return RP.join16(*(_ev(tpl(name, **statics, HALF=h), *arrays)[:, 0] for h in (0, 1)))


def sums_chunks(x: np.ndarray, k: int) -> np.ndarray:
    """``RmsSumSqChunks_v1{C, K}``: x (C,) bf16 words -> each K-chunk's sum of RN(x_i^2) from +0, (C / K,) f32 words."""
    return _word("RmsSumSqPart_v1", x.reshape(-1, k), C=k)


def norm_out_sums(x: np.ndarray, w: np.ndarray, sums: np.ndarray, n: int, groups: int, eps: int = EPS) -> np.ndarray:
    """``RmsNormOutSums_v1{C, S, G, N, EPS}``: a chunk's x and w (C,) bf16 words and the row's S chunk sums -> (C,): each
    group's sum of its S / G chunk sums from +0, rs from the G group sums, then ``RmsNormOut_v1`` a word."""
    g = _word("F32SumPart_v1", sums.reshape(groups, -1), G=len(sums) // groups)
    rs = _word("RmsNormRecip_v1", g[None], N=n, G=groups, EPS=eps)
    return _ev(tpl("RmsNormOut_v1"), x, np.repeat(rs, len(x)), w)[:, 0].astype(np.uint16)


def row_abs_max(x: np.ndarray, relu: bool) -> np.uint16:
    """``RowAbsMax_v1{N, RELU}``: x (N,) bf16 words -> max |x| (relu'd first)."""
    return np.uint16(_ev(tpl("RowAbsMax_v1", N=len(x), RELU=relu), x[None])[0, 0])


def quant_chunk(x: np.ndarray, amax: np.uint16, relu: bool) -> tuple[np.ndarray, np.uint32]:
    """``QuantChunkS8_v1{N, RELU}``: a chunk's x (N,) bf16 words and the row's amax -> codes (N,) s8 words and the scale's f32
    word."""
    codes = _ev(tpl("QuantS8Pair_v1", RELU=relu), x.reshape(-1, 2), np.full(len(x) // 2, amax, np.uint16)).reshape(-1)
    return codes.astype(np.uint8), np.uint32(_word("QuantS8Scale_v1", np.array([amax], np.uint16))[0])


def quant_row(x: np.ndarray, relu: bool) -> tuple[np.ndarray, np.ndarray]:
    """``QuantRowS8_v1{N, RELU}``: x (n, N) bf16 words -> codes (n, N) s8 words, scale (n,) f32 words: one part and one
    chunk."""
    out = [quant_chunk(r, row_abs_max(r, relu), relu) for r in x]
    return np.stack([c for c, _ in out]), np.array([s for _, s in out], np.uint32)


def softmax_row(x: np.ndarray) -> np.ndarray:
    """``SoftmaxRow_v1{N}``: scores (n, N) bf16 words -> probabilities (n, N), as ``SoftmaxMax_v1{N}``,
    ``SoftmaxRecip_v1{N}`` and ``SoftmaxOut_v1`` a word."""
    n, N = x.shape
    m = _ev(tpl("SoftmaxMax_v1", N=N), x)[:, 0]
    r = _word("SoftmaxRecip_v1", x, m, N=N)
    return _ev(tpl("SoftmaxOut_v1"), x.reshape(-1), np.repeat(m, N), np.repeat(r, N)).reshape(n, N).astype(np.uint16)


def qk_softmax_pad(q: np.ndarray, keys: np.ndarray, t: int = PV_K) -> np.ndarray:
    """``QkSoftmaxRowPad_v1{T, D}``: a head's q (D,) and the slot's n key rows (n, D) -> (T,): ``GemmCoordinate_v2{K = D}``
    a key, `softmax_row` over the n scores, +0 past them."""
    n, d = keys.shape
    s = _qk(d).evaluate(np.repeat(q[None], n, 0), keys)[:, 0].astype(np.uint16)
    return np.concatenate([softmax_row(s[None])[0], np.zeros(t - n, np.uint16)])


def pv_row(p: np.ndarray, values: np.ndarray, t: int = PV_K) -> np.ndarray:
    """``PvRow_v1{T, K, D}``: a head's T probabilities and its T value rows (T, D), absent keys' rows zero -> (D,):
    ``GemmCoordinatePad_v1{T, K}`` a dimension."""
    return _pv(t).evaluate(np.repeat(p[None], values.shape[1], 0), values.T)[:, 0].astype(np.uint16)


def argmax_row(x: np.ndarray) -> np.ndarray:
    """``Argmax_v2{V}``: logits (n, V) bf16 words -> (n,), `TokenSelect` over one chunk (``ArgmaxChunkBest_v1{C = V}``,
    ``ArgmaxChunkIdx_v1{C = V}``, ``ArgmaxCombine_v1{V, C = V}``): the first index of the greatest, a NaN never taken."""
    V = x.shape[1]
    best, idx = _ev(tpl("ArgmaxChunkBest_v1", C=V), x), _ev(tpl("ArgmaxChunkIdx_v1", C=V), x)
    return _ev(tpl("ArgmaxCombine_v1", V=V, C=V), x[:, 0], best, idx)[:, 0].astype(np.uint16)


def bf16_add_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``Bf16AddRow_v2{N}``: ``Bf16Add_v2`` a word."""
    return _ev(tpl("Bf16Add_v2"), a.reshape(-1), b.reshape(-1))[:, 0].astype(np.uint16).reshape(a.shape)


@dataclass
class Unit:
    family: str
    template: str
    step: int
    position: int
    slot: int
    index: int
    inputs: list[np.ndarray]
    got: np.ndarray
    want: np.ndarray
    opened: dict | None = None

    @property
    def ok(self) -> bool:
        return self.got.shape == self.want.shape and np.array_equal(self.got, self.want)

    def doc(self) -> dict:
        diff = np.flatnonzero(self.got != self.want) if self.got.shape == self.want.shape else np.array([0])
        return {"family": self.family, "template": self.template, "step": self.step, "position": self.position,
                "slot": self.slot, "index": self.index, "ok": self.ok, "words": int(self.got.size),
                "first_difference": None if self.ok else {"word": int(diff[0]), "got": int(self.got.reshape(-1)[diff[0]]),
                                                          "replayed": int(self.want.reshape(-1)[diff[0]])},
                "inputs": [{"dtype": str(a.dtype), "shape": list(a.shape),
                            "sha512": hashlib.sha512(np.ascontiguousarray(a).tobytes()).hexdigest()} for a in self.inputs],
                **({"opened": self.opened} if self.opened is not None else {})}


@dataclass
class Family:
    name: str
    template: str
    status: str
    per_slot: int
    replay: Replay      # (row, index) -> (inputs, got, want)


def _s8mm(k: int) -> RP.Template:
    return tpl("ScaledMmS8Coordinate_v1", K=k, DOT=RP.G.GEMM_SM90_S8["DOT"])


def _qk(d: int) -> RP.Template:
    return tpl("GemmCoordinate_v2", K=d, DOT=RP.G.GEMM_HOPPER["DOT"])


def _pv(t: int) -> RP.Template:
    return tpl("GemmCoordinatePad_v1", T=t, K=PV_K, DOT=RP.G.GEMM_HOPPER["DOT"])


class Subcircuit(Protocol):
    """A subcircuit of the step's integrity registration besides the model's units: ``serve`` runs after each served step
    (with the driver's resident weights, ``driver.params``), ``families`` gives its units at a step for the sample, and
    ``manifest`` what the registration records of it. Optionally ``call(driver, st, op, lo, codes, w, acc)``, which the driver
    calls on each served linear call: rows ``lo ..`` of linear ``op``'s A codes, its weight and its s32 sums."""
    name: str

    def serve(self, driver: Any, result: Any) -> None: ...

    def families(self, replayer: Replayer, step: int) -> list[Family]: ...

    def manifest(self) -> dict: ...


def load_subcircuit(spec: str, **kw: Any) -> Subcircuit:
    """``MODULE:FACTORY`` called with ``kw`` (``shape``, ``weights``, ``run_id``)."""
    module, sep, factory = spec.partition(":")
    if not sep:
        raise ValueError(f"a subcircuit is MODULE:FACTORY, not {spec!r}")
    return getattr(importlib.import_module(module), factory)(**kw)


def sources(shape: Shape) -> dict[str, Any]:
    """What each of the map's ops reads: an RMSNorm's residual row, a quantizer's row, a linear's quantizer, a residual
    add's two rows."""
    src: dict[str, Any] = {}
    prev = "embed"
    for i in range(shape.layers):
        p = f"l{i}."
        src |= {p + "attn": prev, p + "mlp": p + "attn_residual",
                p + "attn_quant": p + "attn_norm", p + "o_quant": p + "pv", p + "up_quant": p + "mlp_norm",
                p + "down_quant": p + "wup",
                p + "wq": p + "attn", p + "wk": p + "attn", p + "wv": p + "attn", p + "wo": p + "o", p + "wup": p + "up",
                p + "wdown": p + "down",
                p + "attn_residual": (prev, p + "wo"), p + "mlp_residual": (p + "attn_residual", p + "wdown")}
        prev = p + "mlp_residual"
    return src | {"final": prev, "head_quant": "final_norm", "head": "head"}


class Replayer:
    def __init__(self, shape: Shape, weights: Weights, steps: list[StepTables], tokens: list[np.ndarray],
                 subcircuits: Sequence[Subcircuit] = (), fused: frozenset[str] = frozenset()) -> None:
        """``steps[s]`` the committed tables of step s (row b: slot ``slots[b]`` at ``positions[b]``), ``tokens[s]`` the bytes
        its rows embedded; ``fused`` the map's fusions."""
        self.shape, self.weights, self.steps, self.tokens = shape, weights, steps, tokens
        self.subcircuits = list(subcircuits)
        self.map = Map(shape, fused)
        self.src = sources(shape)
        self._array = lru_cache(maxsize=8)(weights.array)

    # -- committed words

    def raw(self, step: int, op: str, row: int) -> np.ndarray:
        return np.ascontiguousarray(self.steps[step].tables[op][0][row])

    def u16(self, step: int, op: str, row: int) -> np.ndarray:
        return self.raw(step, op, row).view(np.uint16)

    def word(self, step: int, op: str, row: int) -> np.uint32:
        return np.asarray(self.steps[step].tables[op][1][row], np.float32).view(np.uint32)

    def out(self, step: int, u: Units, row: int, i: int) -> np.ndarray:
        """Unit i's committed output bytes in its table's row."""
        lo = u.offset + i * u.stride
        return self.raw(step, u.table, row)[lo:lo + u.nbytes]

    def history(self, step: int, b: int) -> list[tuple[int, int]]:
        """The (step, row) at which row ``b`` of ``step``'s slot ran each of its positions 0 .. p."""
        st = self.steps[step]
        slot, p = int(st.slots[b]), int(st.positions[b])
        out = []
        for j in range(p + 1):
            s = step - p + j
            r = self.steps[s].row_of(slot)
            if int(self.steps[s].positions[r]) != j:
                raise ValueError(f"slot {slot} is at position {p} at step {step} but at {self.steps[s].positions[r]}, "
                                 f"not {j}, at step {s}")
            out.append((s, r))
        return out

    def kv(self, step: int, layer: int, b: int) -> tuple[np.ndarray, np.ndarray]:
        """Row ``b``'s keys and values at ``layer``, (p + 1, head_dim) u16 each: K's and V's committed rows at its positions."""
        hist = self.history(step, b)
        return (np.stack([self.u16(s, f"l{layer}.wk", r) for s, r in hist]),
                np.stack([self.u16(s, f"l{layer}.wv", r) for s, r in hist]))

    def _gain(self, name: str) -> np.ndarray:
        return np.asarray(self._array(name), np.uint16)

    # -- families

    def families(self, step: int) -> list[Family]:
        """The step's model families, then each subcircuit's."""
        return self.model_families(step) + [f for sub in self.subcircuits for f in sub.families(self, step)]

    def model_families(self, step: int) -> list[Family]:
        """One family a `plan.Units` of the map, in the order the step writes them."""
        build = {"EmbeddingCoordinate_v1": self._embed, "AllGather_v1": self._gather, "RmsSumSqChunks_v1": self._sums,
                 "RmsNormOutSums_v1": self._norm, "RowAbsMax_v1": self._amax, "QuantChunkS8_v1": self._quant,
                 "RmsNormAbsMax_v1": self._normed, "RmsNormQuantCodes_v1": self._normed, "QuantScaleS8_v1": self._scale,
                 "ScaledMmS8CoordinateRows_v1": self._linear, "ScaledMmS8ResidualCoordinateRows_v1": self._linear,
                 "QkSoftmaxRowPad_v1": self._softmax, "PvRow_v1": self._pv, "QkSoftmaxPvPad_v1": self._head,
                 "Bf16AddRow_v2": self._add, "Argmax_v2": self._argmax}
        return [Family(u.name, u.key, u.status, u.count, build[u.template](step, u))
                for g in self.map.groups() for _, us in g for u in us]

    def registration(self, step: int) -> dict:
        """The step's execution phases: the model's families by template, then each subcircuit's."""
        fams = self.model_families(step)
        return {"blocks": [{"name": "model", "families": len(fams), "templates": sorted({f.template for f in fams})}]
                + [{"name": sub.name, **sub.manifest()} for sub in self.subcircuits]}

    def _embed(self, step: int, u: Units) -> Replay:
        def replay(b: int, i: int):
            tok, t = int(self.tokens[step][b]), int(self.steps[step].positions[b])
            col, pos = self._gain("tok")[i], self._gain("pos")[i, t]
            want = _ev(tpl("EmbeddingCoordinate_v1", V=len(col)), np.array([tok], np.uint16), col[None], np.array([pos]))[0]
            return [np.array([tok], np.uint16), col, np.array([pos], np.uint16)], self.out(step, u, b, i).view(np.uint16), \
                want.astype(np.uint16)
        return replay

    def _gather(self, step: int, u: Units) -> Replay:
        def replay(b: int, i: int):
            got = self.out(step, u, b, i).view(np.uint16)
            return [got.copy()], got, got.copy()
        return replay

    def _sums(self, step: int, u: Units) -> Replay:
        op, c = u.table[:-len("_sums")], self.map.f.chunk
        n = c * (u.nbytes // 4)

        def replay(b: int, i: int):
            x = self.u16(step, self.src[op], b)[i * n:(i + 1) * n]
            return [x], self.out(step, u, b, i).view(np.uint32), sums_chunks(x, c)
        return replay

    def _norm(self, step: int, u: Units) -> Replay:
        op, c, n = u.table[:-len("_norm")], self.map.f.chunk, self.shape.width
        groups = self.map.f.groups(n)

        def replay(b: int, i: int):
            x = self.u16(step, self.src[op], b)[i * c:(i + 1) * c]
            w = self._gain(op + "_norm")[i * c:(i + 1) * c]
            sums = self.raw(step, op + "_sums", b).view(np.uint32)
            return [x, w, sums], self.out(step, u, b, i).view(np.uint16), norm_out_sums(x, w, sums, n, groups)
        return replay

    def _amax(self, step: int, u: Units) -> Replay:
        op = u.table[:-len("_amax")]
        relu = op.endswith("down")
        if "norm" in self.map.fused and self.src[op + "_quant"].endswith("_norm"):
            def combine(b: int, i: int):
                x = self.u16(step, u.table, b)[:u.offset // 2]
                return [x], self.out(step, u, b, i).view(np.uint16), np.array([row_abs_max(x, False)], np.uint16)
            return combine

        def replay(b: int, i: int):
            x = self.u16(step, self.src[op + "_quant"], b)
            parts, w = self.map.f.parts(len(x))
            if u.name.endswith(".part"):
                x = x[i * w:(i + 1) * w]
            elif parts > 1:
                x = np.concatenate([x[(parts - 1) * w:], self.u16(step, u.table, b)[:parts - 1]])
            return [x], self.out(step, u, b, i).view(np.uint16), np.array([row_abs_max(x, relu)], np.uint16)
        return replay

    def _quant(self, step: int, u: Units) -> Replay:
        op = u.table
        relu, q = op.endswith("down_quant"), u.nbytes

        def replay(b: int, i: int):
            x = self.u16(step, self.src[op], b)[i * q:(i + 1) * q]
            amax = self.u16(step, op[:-len("_quant")] + "_amax", b)[-1]
            codes, scale = quant_chunk(x, amax, relu)
            got = np.concatenate([self.out(step, u, b, i).view(np.uint8), np.atleast_1d(self.word(step, op, b)).view(np.uint8)])
            return [x, np.array([amax], np.uint16)], got, np.concatenate([codes, np.array([scale], np.uint32).view(np.uint8)])
        return replay

    def _normed(self, step: int, u: Units) -> Replay:
        """The ``norm`` fusion's amax part i or code block i: RMSNorm's output there from the stream, the gain and the sums,
        then `row_abs_max` of it, or its codes (`quant_chunk`'s) under the row's amax."""
        op = u.table.rsplit("_", 1)[0]
        norm, n = self.src[op + "_quant"][:-len("_norm")], self.shape.width
        codes = u.template == "RmsNormQuantCodes_v1"
        groups, c = self.map.f.groups(n), u.nbytes if codes else self.map.f.norm_parts(n)[1]

        def replay(b: int, i: int):
            x = self.u16(step, self.src[norm], b)[i * c:(i + 1) * c]
            w = self._gain(norm + "_norm")[i * c:(i + 1) * c]
            sums = self.raw(step, norm + "_sums", b).view(np.uint32)
            h = norm_out_sums(x, w, sums, n, groups)
            if not codes:
                return [x, w, sums], self.out(step, u, b, i).view(np.uint16), np.array([row_abs_max(h, False)], np.uint16)
            amax = self.u16(step, op + "_amax", b)[-1]
            return [x, w, sums, np.array([amax], np.uint16)], self.out(step, u, b, i).view(np.uint8), \
                quant_chunk(h, amax, False)[0]
        return replay

    def _scale(self, step: int, u: Units) -> Replay:
        """``QuantScaleS8_v1``: the scale of quantizer ``u.table``'s chunk row i, from the row's amax."""
        op = u.table[:-len("_quant")]

        def replay(b: int, i: int):
            amax = np.array([self.u16(step, op + "_amax", b)[-1]], np.uint16)
            return [amax], np.atleast_1d(self.word(step, u.table, b)), _word("QuantS8Scale_v1", amax)
        return replay

    def _linear(self, step: int, u: Units) -> Replay:
        """A coordinate of linear ``u.name`` (its weight's name); fused with a residual add, its table's stream word is r."""
        tensor, qop = u.name, self.src[u.name] + "_quant"
        k = self.weights.dims[tensor][1]
        t = _s8mm(k)
        stream = self.src[u.table][0] if u.template == "ScaledMmS8ResidualCoordinateRows_v1" else None

        def replay(b: int, i: int):
            x = self.raw(step, qop, b).view(np.uint8)
            sx = self.word(step, qop, b)
            wrow = self.weights.row(tensor, i).view(np.uint8)
            sw = np.asarray(self._array(f"{tensor}.scale")[i], np.float32).view(np.uint32)
            want = t.evaluate(x[None], np.array([sx]), wrow[None], np.array([sw]))[0].astype(np.uint16)
            inputs = [x, np.array([sx]), wrow, np.array([sw])]
            if stream is not None:
                r = self.u16(step, stream, b)[i:i + 1]
                inputs, want = inputs + [r], bf16_add_row(r, want)
            return inputs, self.out(step, u, b, i).view(np.uint16), want
        return replay

    def _softmax(self, step: int, u: Units) -> Replay:
        layer, d = int(u.table[1:].split(".")[0]), self.shape.head_dim

        def replay(b: int, head: int):
            q = self.u16(step, f"l{layer}.wq", b)[head * d:(head + 1) * d]
            keys = self.kv(step, layer, b)[0]
            padded = np.zeros((PV_K, d), np.uint16)
            padded[:len(keys)] = keys
            return [q, padded, np.array([len(keys)], np.uint16)], self.out(step, u, b, head).view(np.uint16), \
                qk_softmax_pad(q, keys)
        return replay

    def _pv(self, step: int, u: Units) -> Replay:
        layer = int(u.table[1:].split(".")[0])

        def replay(b: int, head: int):
            p = self.u16(step, f"l{layer}.softmax", b)[head * PV_K:(head + 1) * PV_K]
            values = self.kv(step, layer, b)[1]
            padded = np.zeros((PV_K, values.shape[1]), np.uint16)
            padded[:len(values)] = values
            return [p, padded], self.out(step, u, b, head).view(np.uint16), pv_row(p, padded)
        return replay

    def _head(self, step: int, u: Units) -> Replay:
        """The ``pv`` fusion's head: `qk_softmax_pad` of q and the slot's keys, then `pv_row` of that row and its values."""
        layer, d = int(u.table[1:].split(".")[0]), self.shape.head_dim

        def replay(b: int, head: int):
            q = self.u16(step, f"l{layer}.wq", b)[head * d:(head + 1) * d]
            keys, values = self.kv(step, layer, b)
            k, v = np.zeros((PV_K, d), np.uint16), np.zeros((PV_K, d), np.uint16)
            k[:len(keys)], v[:len(values)] = keys, values
            want = pv_row(qk_softmax_pad(q, keys), v)
            return [q, k, v, np.array([len(keys)], np.uint16)], self.out(step, u, b, head).view(np.uint16), want
        return replay

    def _add(self, step: int, u: Units) -> Replay:
        a_op, b_op = self.src[u.table]
        c = u.nbytes // 2

        def replay(b: int, i: int):
            x, y = self.u16(step, a_op, b)[i * c:(i + 1) * c], self.u16(step, b_op, b)[i * c:(i + 1) * c]
            return [x, y], self.out(step, u, b, i).view(np.uint16), bf16_add_row(x, y)
        return replay

    def _argmax(self, step: int, u: Units) -> Replay:
        def replay(b: int, _: int):
            logits = self.u16(step, "head", b)
            return [logits], self.out(step, u, b, 0).view(np.uint16), argmax_row(logits[None])
        return replay

    # -- the sample

    def unit(self, fam: Family, step: int, row: int, index: int) -> Unit:
        inputs, got, want = fam.replay(row, index)
        st = self.steps[step]
        return Unit(fam.name, fam.template, step, int(st.positions[row]), int(st.slots[row]), index, inputs,
                    np.asarray(got).reshape(-1), np.asarray(want).reshape(-1).astype(np.asarray(got).dtype))

    def draw(self, step: int, k: int, seed: bytes, only: Callable[[Family], bool] = lambda f: True) -> list[Unit]:
        """``k`` units of every family of ``step`` (or of those ``only`` keeps), replayed: a FIXTURE coin per family, until
        the sampling plan draws from live coins (`replay.draw`)."""
        batch = self.steps[step].batch
        units = []
        for fam in self.families(step):
            if not only(fam):
                continue
            h = hashlib.sha512(b"e2e-h200/draw-fixture\0" + seed + fam.name.encode() + step.to_bytes(4, "big")).digest()
            rng = np.random.Generator(np.random.PCG64(int.from_bytes(h[:16], "big")))
            for u in rng.integers(0, fam.per_slot * batch, size=k):
                r, idx = divmod(int(u), fam.per_slot)
                units.append(self.unit(fam, step, r, idx))
        return units

    def every(self, step: int, only: Callable[[Family], bool] = lambda f: True) -> list[Unit]:
        """Every unit of ``step``'s families that ``only`` keeps."""
        batch = self.steps[step].batch
        return [self.unit(f, step, b, i) for f in self.families(step) if only(f) for b in range(batch)
                for i in range(f.per_slot)]
