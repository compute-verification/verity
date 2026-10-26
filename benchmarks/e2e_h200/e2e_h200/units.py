"""A step's units, read from its committed tables, and their replay on the CPU against circuits' semantics.

The units are circuits' plan's, one unit a row (CMP-92 af6de1f1, revised 23:30Z, section 1b; `driver` names the tables).
Each `Family` is one template over one table: its units are the table's rows (a row unit) or the rows' words (a coordinate
unit), and `Family.replay` gives a unit's opened inputs, its committed output words and the words replay computes:

* coordinate units by `replay`'s templates (the replay lane's `benchmarks/e2e_h200/replay.py`), which evaluate main's
  Definitions: ``ScaledMmS8Coordinate_v1{K, GEMM_SM90_S8}`` (Q, K, V, O, up, down, head), ``GemmCoordinate_v2{K=d,
  GEMM_HOPPER}`` (QK^T) and ``GemmCoordinatePad_v1{T, K=32}`` (PV);
* row units by the compositions below of `replay`'s word functions (F32AddV2 / MulV2 / DivV2, PouwF32SqrtRn, MufuEx2Ftz,
  F2fpBf16, Bf16Add_v1, TokenSelect, the quantizer of #4433), the plan's row semantics, until circuits' row Definitions
  (``RmsNormRow_v1``, ``QuantRowS8_v1``, ``SoftmaxRow_v1``, ``Bf16AddRow_v1``, the embedding and argmax rows) are on main:
  their status is ``pending-definition``.

A unit's inputs are read from the committed tables (an earlier op's row, an earlier step's K or V row) and the registered
weights (`weights.Weights`, from the generator, not the store, so a wrong decode shows as a mismatch).

Row b of a step is slot ``slots[b]`` at its own position p. Its attention reads the slot's K and V rows committed at positions
0 .. p, each at the step and row the slot ran it (`Replayer.history`), never the live cache. The attention families come one
per group of rows at one position t, so each is one template (``SoftmaxRow_v1{N = t + 1}``, ``GemmCoordinatePad_v1{T = t +
1}``). QK^T's unit (head, j), j <= t, is index head * (t + 1) + j of its family and word head * context + j of the
``scores`` row.

A `Subcircuit` joins the step's integrity registration as one more sampling block, drawn with the step's families: PoUW's
per-call-noise tiles (H200-PERCALL, CMP-126) plug in as ``MODULE:FACTORY`` (`load_subcircuit`, ``run.py --subcircuit``).
"""
from __future__ import annotations

import hashlib
import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import numpy as np

import replay as RP

from .commit import StepTables
from .shape import EPS, PV_K, Shape
from .weights import Weights

ROW, COORD = "pending-definition", "definition"


def rms_norm_row(x: np.ndarray, w: np.ndarray, eps: int = EPS) -> np.ndarray:
    """``RmsNormRow_v1{N, EPS}``: x, w (n, N) bf16 words -> (n, N)."""
    xw = RP.widen(x)
    rs = RP.rms_recip(RP.seq_sum(RP.mul(xw, xw))[:, None], x.shape[1], eps)
    return RP.rms_out(x, rs[:, None], w)


def quant_row(x: np.ndarray, relu: bool) -> tuple[np.ndarray, np.ndarray]:
    """``QuantRowS8_v1{N, RELU}``: x (n, N) bf16 words -> codes (n, N) s8 words, scale (n,) f32 words."""
    scale = RP.quant_scale(RP.row_abs_max(x, relu))
    return RP.to_s8_rni_sat(RP.div(RP.widen(RP.relu16(x, relu)), scale[:, None])), scale


def softmax_row(x: np.ndarray) -> np.ndarray:
    """``SoftmaxRow_v1{N}``: scores (n, N) bf16 words -> probabilities (n, N)."""
    m = RP.softmax_max(x)
    e = RP.softmax_exp(x, m)
    r = RP.div(np.uint32(RP.F32_ONE), RP.seq_sum(e))
    return RP.bf16(RP.mul(e, r[:, None]))


def argmax_row(x: np.ndarray) -> np.ndarray:
    """The argmax row unit: logits (n, V) bf16 words -> (n,) the first index of the greatest (`TokenSelect` over one chunk)."""
    best, idx = RP.chunk_best(x)
    return RP.argmax_combine(x[:, 0], best[:, None], idx[:, None], x.shape[1])


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
    span: tuple[int, int, bool] | None = None
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
    replay: Callable[[int, int], tuple[list[np.ndarray], np.ndarray, np.ndarray]]   # (row, index) -> (inputs, got, want)
    #: unit index -> (first byte, end byte, whether the row's scale too) of its outputs in the table's row (None: the row)
    span: Callable[[int], tuple[int, int, bool]] | None = None
    rows: tuple[int, ...] | None = None     # the rows of one position group (the attention families), else every row
    position: int | None = None

    def rows_of(self, batch: int) -> Sequence[int]:
        return range(batch) if self.rows is None else self.rows


@lru_cache(maxsize=None)
def _s8mm(k: int) -> RP.Template:
    return RP.scaled_mm_s8_coordinate(k)


@lru_cache(maxsize=None)
def _qk(d: int) -> RP.Template:
    return RP.gemm_coordinate(d)


@lru_cache(maxsize=None)
def _pv(t: int) -> RP.Template:
    return RP.gemm_coordinate_pad(t, PV_K)


class Subcircuit(Protocol):
    """A subcircuit of the step's integrity registration besides the model's units: ``serve`` runs after each served step
    (with the driver's resident weights, ``driver.params``), ``families`` gives its units at a step for the draw, and
    ``manifest`` what the registration records of it."""
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


class Replayer:
    def __init__(self, shape: Shape, weights: Weights, steps: list[StepTables], tokens: list[np.ndarray],
                 subcircuits: Sequence[Subcircuit] = ()) -> None:
        """``steps[s]`` the committed tables of step s (row b: slot ``slots[b]`` at ``positions[b]``), ``tokens[s]`` the bytes
        its rows embedded."""
        self.shape, self.weights, self.steps, self.tokens = shape, weights, steps, tokens
        self.subcircuits = list(subcircuits)
        self._array = lru_cache(maxsize=8)(weights.array)

    # -- committed words

    def raw(self, step: int, op: str, row: int) -> np.ndarray:
        return np.ascontiguousarray(self.steps[step].tables[op][0][row])

    def u16(self, step: int, op: str, row: int) -> np.ndarray:
        return self.raw(step, op, row).view(np.uint16)

    def word(self, step: int, op: str, row: int) -> np.uint32:
        return np.asarray(self.steps[step].tables[op][1][row], np.float32).view(np.uint32)

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
        sh = self.shape
        out = [self._embed(step)]
        prev = "embed"
        for layer in range(sh.layers):
            p = f"l{layer}."
            out += [self._rms(step, p + "attn_norm", prev), self._quant(step, p + "attn_quant", p + "attn_norm", False)]
            out += [self._linear(step, p + k, p + "attn_quant") for k in ("wq", "wk", "wv")]
            out += self._attention(step, layer)
            out += [self._quant(step, p + "o_quant", p + "pv", False), self._linear(step, p + "wo", p + "o_quant"),
                    self._add(step, p + "attn_residual", prev, p + "wo"),
                    self._rms(step, p + "mlp_norm", p + "attn_residual"),
                    self._quant(step, p + "up_quant", p + "mlp_norm", False), self._linear(step, p + "wup", p + "up_quant"),
                    self._quant(step, p + "down_quant", p + "wup", True), self._linear(step, p + "wdown", p + "down_quant"),
                    self._add(step, p + "mlp_residual", p + "attn_residual", p + "wdown")]
            prev = p + "mlp_residual"
        out += [self._rms(step, "final_norm", prev), self._quant(step, "head_quant", "final_norm", False),
                self._linear(step, "head", "head_quant"), self._argmax(step)]
        return out

    def registration(self, step: int) -> dict:
        """The step's sampling blocks: the model's families by template, then each subcircuit's."""
        fams = self.model_families(step)
        return {"blocks": [{"name": "model", "families": len(fams), "templates": sorted({f.template for f in fams})}]
                + [{"name": sub.name, **sub.manifest()} for sub in self.subcircuits]}

    def _embed(self, step: int) -> Family:
        sh = self.shape

        def replay(b: int, _: int):
            tok, t = int(self.tokens[step][b]), int(self.steps[step].positions[b])
            col, pos = self._gain("tok")[:, tok], self._gain("pos")[:, t]
            return [np.array([tok], np.uint16), col, pos], self.u16(step, "embed", b), RP.bf16_add(col, pos)
        return Family("embed", RP.key_of("EmbeddingRow_v1", {"V": sh.vocab}), ROW, 1, replay)

    def _rms(self, step: int, op: str, src: str) -> Family:
        n = self.shape.width

        def replay(b: int, _: int):
            x, w = self.u16(step, src, b), self._gain(op)
            return [x, w], self.u16(step, op, b), rms_norm_row(x[None], w[None])[0]
        return Family(op, RP.key_of("RmsNormRow_v1", {"N": n, "EPS": EPS}), ROW, 1, replay)

    def _quant(self, step: int, op: str, src: str, relu: bool) -> Family:
        n = self.steps[step].tables[src][0].shape[1] // 2

        def replay(b: int, _: int):
            x = self.u16(step, src, b)
            codes, scale = quant_row(x[None], relu)
            got = np.concatenate([self.raw(step, op, b).view(np.uint8).astype(np.uint32), [self.word(step, op, b)]])
            return [x], got, np.concatenate([codes[0].astype(np.uint32), scale])
        return Family(op, RP.key_of("QuantRowS8_v1", {"N": n, "RELU": relu}), ROW, 1, replay, lambda _: (0, n, True))

    def _linear(self, step: int, op: str, qop: str) -> Family:
        tensor = op
        n, k = self.weights.dims[tensor]
        tpl = _s8mm(k)

        def replay(b: int, i: int):
            x = self.raw(step, qop, b).view(np.uint8)
            sx = self.word(step, qop, b)
            wrow = self.weights.row(tensor, i).view(np.uint8)
            sw = np.asarray(self._array(f"{tensor}.scale")[i], np.float32).view(np.uint32)
            want = tpl.evaluate(x[None], np.array([sx]), wrow[None], np.array([sw]))[0]
            return [x, np.array([sx]), wrow, np.array([sw])], self.u16(step, op, b)[i:i + 1], want.astype(np.uint16)
        return Family(op, tpl.key, COORD, n, replay, lambda i: (2 * i, 2 * i + 2, False))

    def _attention(self, step: int, layer: int) -> list[Family]:
        positions = self.steps[step].positions
        out = []
        for t in np.unique(positions).tolist():
            out += self._attention_at(step, layer, int(t), tuple(np.flatnonzero(positions == t).tolist()))
        return out

    def _attention_at(self, step: int, layer: int, t: int, rows: tuple[int, ...]) -> list[Family]:
        """QK^T, softmax and PV of the rows at position ``t``: a row's scores and probabilities are words head * context + j,
        j <= t, of its (heads, context) row."""
        sh, p = self.shape, f"l{layer}."
        d, h, t1, c = sh.head_dim, sh.heads, t + 1, sh.context
        qk, pv = _qk(d), _pv(t1)

        def scores(b: int, i: int):
            head, j = divmod(i, t1)
            q = self.u16(step, p + "wq", b)[head * d:(head + 1) * d]
            k = self.kv(step, layer, b)[0][j]
            want = qk.evaluate(q[None], k[None])[0]
            return [q, k], self.u16(step, p + "scores", b)[head * c + j:head * c + j + 1], want.astype(np.uint16)

        def soft(b: int, head: int):
            x = self.u16(step, p + "scores", b)[head * c:head * c + t1]
            return [x], self.u16(step, p + "softmax", b)[head * c:head * c + t1], softmax_row(x[None])[0]

        def pvu(b: int, i: int):
            head, c_ = divmod(i, d)
            pr = self.u16(step, p + "softmax", b)[head * c:head * c + t1]
            v = self.kv(step, layer, b)[1][:, c_]
            want = pv.evaluate(pr[None], v[None])[0]
            return [pr, v], self.u16(step, p + "pv", b)[i:i + 1], want.astype(np.uint16)

        def at(i: int) -> tuple[int, int, bool]:
            head, j = divmod(i, t1)
            return 2 * (head * c + j), 2 * (head * c + j) + 2, False

        return [Family(p + "scores", qk.key, COORD, h * t1, scores, at, rows, t),
                Family(p + "softmax", RP.key_of("SoftmaxRow_v1", {"N": t1}), ROW, h, soft,
                       lambda head: (2 * head * c, 2 * (head * c + t1), False), rows, t),
                Family(p + "pv", pv.key, pv.status, h * d, pvu, lambda i: (2 * i, 2 * i + 2, False), rows, t)]

    def _add(self, step: int, op: str, a: str, b_op: str) -> Family:
        def replay(b: int, _: int):
            x, y = self.u16(step, a, b), self.u16(step, b_op, b)
            return [x, y], self.u16(step, op, b), RP.bf16_add(x, y)
        return Family(op, RP.key_of("Bf16AddRow_v1", {"N": self.shape.width}), ROW, 1, replay)

    def _argmax(self, step: int) -> Family:
        def replay(b: int, _: int):
            logits = self.u16(step, "head", b)
            return [logits], self.u16(step, "argmax", b), argmax_row(logits[None])
        return Family("argmax", RP.key_of("ArgmaxRow_v1", {"V": self.shape.vocab}), ROW, 1, replay)

    # -- the draw

    def unit(self, fam: Family, step: int, row: int, index: int) -> Unit:
        inputs, got, want = fam.replay(row, index)
        st = self.steps[step]
        return Unit(fam.name, fam.template, step, int(st.positions[row]), int(st.slots[row]), index, inputs,
                    np.asarray(got).reshape(-1), np.asarray(want).reshape(-1).astype(np.asarray(got).dtype),
                    fam.span(index) if fam.span else None)

    def draw(self, step: int, k: int, seed: bytes, only: Callable[[Family], bool] = lambda f: True) -> list[Unit]:
        """``k`` units of every family of ``step`` (or of those ``only`` keeps), replayed: a FIXTURE coin per family, until
        PRF-170's plan draws from live coins (`replay.draw`)."""
        st = self.steps[step]
        units = []
        for fam in self.families(step):
            if not only(fam):
                continue
            at = b"" if fam.position is None else fam.position.to_bytes(4, "big")
            h = hashlib.sha512(b"e2e-h200/draw-fixture\0" + seed + fam.name.encode() + step.to_bytes(4, "big") + at).digest()
            rng = np.random.Generator(np.random.PCG64(int.from_bytes(h[:16], "big")))
            rows = fam.rows_of(st.batch)
            for u in rng.integers(0, fam.per_slot * len(rows), size=k):
                r, idx = divmod(int(u), fam.per_slot)
                units.append(self.unit(fam, step, rows[r], idx))
        return units

    def every(self, step: int, only: Callable[[Family], bool] = lambda f: True) -> list[Unit]:
        """Every unit of ``step``'s families that ``only`` keeps."""
        st = self.steps[step]
        return [self.unit(f, step, b, i) for f in self.families(step) if only(f)
                for b in f.rows_of(st.batch) for i in range(f.per_slot)]
