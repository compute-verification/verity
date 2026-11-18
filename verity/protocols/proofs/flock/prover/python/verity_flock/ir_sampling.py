"""Flock lowering of ``gumbel-top-p-token-select`` (``GumbelTopPTokenSelect_v1{V}``, one vocabulary row): the lane unit, its
pin, and the native cut words of the statement ``verity/flock-ir-sampling/v1`` (``live/src/ir_sampling.rs``).

The Definition is ``TemperatureScale -> TopPMask -> GumbelSelectF32``: three scans over the V lanes of the row around three
row-level values (the temperature's reciprocal, the top-p keep word, the Gumbel stream key). The lowering keeps the scans'
lane bodies in the circuit and the row-level values out of it:

- **unit** (one per lane v, all identical, lowered gate by gate from the IR composites): ``TemperatureLane(x_v, temp, rcp,
  skip) -> scaled_v``; ``TopPMaskStep``'s select ``SelectF32(kbit_v, scaled_v, -inf) -> masked_v``, its ``BitAtx{V}(keep, v)``
  resolved by the arrangement (lane v reads bit v); ``GumbelSelectStep((best, best_i, i), masked_v, key, noisy)``, its
  ``GumbelNoiseLane(key, i)`` resolved the same way (lane v reads ``g_v``), gives the carry lane v hands on. Lane 0 (``i ==
  0``) takes the scan's init ``(y_0, 0, 1)`` instead of the step, as the Definition does (``y_0`` is the step's own ``y``).
- **cut words** (public, per instance, :data:`SCALARS`, :data:`INIT`, then :data:`LANE` per lane): the row scalars ``temp``,
  ``rcp = DivFullRcp(temp)``, ``skip = temp in {0, 1}``, ``noisy = temp != 0``; the initial carry ``(0, 0, 0)``; per lane
  ``kbit_v`` (bit v of ``TopPMaskWordx{V}(scaled, top_p, splits)``), ``g_v = GumbelNoiseLane(GumbelStreamKey(seed, pos), v)``,
  ``scaled_v`` and the carry ``(best_v, best_i_v, i_v + 1)``. A unit computes ``scaled_v`` and its carry; every other cut
  word is **native**: the verifier computes it (:func:`native_words`) from the public words. The top-p keep word is native
  because its split pipeline runs the measured MUFU EX2 / RCP tables over whole-row reductions, which no circuit here
  lowers; the noise needs only the public key.
- **ports:** the logits row is a private frame-v3 row (``blake3-keyed/row/v2``, hashed in the proof like every frame-v3 row);
  the scalar operands (top_p, seed, pos, temp, splits) are public, their frame-v3 row digests computed natively; the token
  (lane V-1's ``best_i``, a u64 as the row kernel returns it) is public.

The unit's circuit description does not depend on V (only the arrangement does) and is pinned by :data:`PIN`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import ir_lower as IL
from .gf2 import NOT, Circuit, mux, or_reduce

TEMPLATE = "gumbel-top-p-token-select"
UNIT = "gumbel-top-p-token-select/fp32/lane"
PIN = "c9aeca364cb08b68a174fa9811ff505306c6a8f9af7458689ff23e66409cc1b6"
FORMAT = "flock-ir-sampling/v1"
STATEMENT = "verity/flock-ir-sampling/v1"

SCALARS = ("temp", "rcp", "skip", "noisy")
INIT = ("best", "best_i", "i")
LANE = ("kbit", "g", "scaled", "best", "best_i", "i_next")
#: unit ports: the logit (the leaf), then the cut inputs; outputs are all cut words
IN_PORTS = ("x", "temp", "rcp", "skip", "noisy", "kbit", "g", "best", "best_i", "i")
OUT_PORTS = ("scaled", "best", "best_i", "i_next")
PUBLIC_PORTS = ("in1", "in2", "in3", "in4", "in5")       # top_p, seed, pos, temp, splits
LOGITS, OUTPUT = "in0", "out.sampled_token_ids"
PT_LOCAL = 16
NEG_INF = 0xFF800000
ONE_F32 = 0x3F800000


def words(V: int) -> int:
    return len(SCALARS) + len(INIT) + len(LANE) * V


def lane_word(v: int, name: str) -> int:
    return len(SCALARS) + len(INIT) + len(LANE) * v + LANE.index(name)


def lane_ports(v: int) -> tuple[list[int], list[int]]:
    """Lane v's cut input words (``IN_PORTS[1:]``) and cut output words (``OUT_PORTS``)."""
    carry = [len(SCALARS) + k for k in range(3)] if v == 0 else [lane_word(v - 1, n) for n in ("best", "best_i", "i_next")]
    return ([0, 1, 2, 3, lane_word(v, "kbit"), lane_word(v, "g"), *carry],
            [lane_word(v, n) for n in OUT_PORTS[:1]] + [lane_word(v, n) for n in ("best", "best_i", "i_next")])


# --- pieces: ir_lower's (the lane's F32 ops, compare, selects and integer add are tail_pieces', NaN payloads included) --------

PIECES = IL.PIECES


# --- the lane unit, lowered from the IR composites ------------------------------------------------------------------------------

def _composites(V: int):
    from verity.circuits.defs import bind
    from verity_vllm.definitions.registry import sampling as SM

    return SM.TemperatureLane, bind(SM.TopPMaskStep, V=V), SM.GumbelSelectStep


def _walk(C, target, args, resolved: dict):
    """Lower ``target``'s gates on ``args`` (one word per argument, ``None`` for an argument only resolved gates read); a gate
    whose primitive id is in ``resolved`` takes that word instead of a piece. Returns ``(call, wires)``."""
    _prog, call = IL.standalone(target)
    wires = {a: w for a, w in zip(call.args, args) if w is not None}
    todo = []
    for g in call:
        if g.prim.id in resolved:
            wires[g.index] = list(resolved[g.prim.id])
        else:
            todo.append(g)
    IL.lower_gates(C, todo, wires, PIECES)
    return call, wires


def lane_circuit(V: int) -> Circuit:
    """The lane unit: inputs ``in0`` (the bf16 logit) and ``in1 .. in9`` (``IN_PORTS[1:]``, 32 bits each; skip, noisy and kbit
    use bit 0), outputs ``out0 .. out3`` (``OUT_PORTS``)."""
    temp_lane, mask_step, select_step = _composites(V)
    C = Circuit(cse=True)
    x = C.input("in0", 16)
    temp, rcp, skip, noisy, kbit, g, best, best_i, i = (C.input(f"in{j}", 32) for j in range(1, 10))
    call, w = _walk(C, temp_lane, [x, temp, rcp, skip[:1]], {})
    scaled = w[call.returns[0]]
    call, w = _walk(C, mask_step, [i, scaled, None], {f"BitAtx{V}_v1": kbit[:1]})
    masked = w[call.returns[1]]
    call, w = _walk(C, select_step, [best, best_i, i, masked, None, noisy[:1]], {"GumbelNoiseLane_v1": g})
    nb, ni, nxt = (w[r] for r in call.returns[:3])
    (gt,) = [gg for gg in call if gg.prim.id == "F32GtStrict_v1"]
    y = w[gt.operands[0]]
    first = NOT(or_reduce(C, i))
    C.output("out0", scaled)
    C.output("out1", mux(C, first, y, nb))
    C.output("out2", mux(C, first, i, ni))
    C.output("out3", nxt)
    return C


@dataclass
class Lane:
    """The lane unit's circuit and its ``flock-ir-unit/v2`` layout (the leaf group, one cut input group, one cut output group)."""

    V: int
    circuit: Circuit
    layout: IL.LoweredUnit

    @property
    def name(self) -> str:
        return UNIT

    @property
    def units(self):
        """The lowering's units: the count per instance, and cut words (so the MUFU tables, for the native ``DivFullRcp``)."""
        from types import SimpleNamespace

        return SimpleNamespace(n=self.V, cut=True)

    @property
    def cut(self) -> dict:
        """The cut structure the circuit description pins: the word count, no tail program (every unit-less cut word is native),
        the native rule, and per lane its cut ports as ``[column, cut word]``."""
        lay = self.layout
        cin = lay.in_port_cols[1:]
        cout = [c - lay.out_col * IL.WORD for c in lay.out_port_cols]
        ins, outs = [], []
        for v in range(self.V):
            a, b = lane_ports(v)
            ins.append([[c, k] for c, k in zip(cin, a)])
            outs.append([[c, k] for c, k in zip(cout, b)])
        return {"words": words(self.V), "tail": [], "native": STATEMENT, "in": ins, "out": outs}

    @property
    def text(self) -> str:
        return IL.circuit_description(self.layout, self.name, self.cut)

    @property
    def unit_text(self) -> str:
        """The circuit description without its cut line: what does not depend on V (:data:`PIN`)."""
        return IL.circuit_description(self.layout, self.name)

    @property
    def sha256(self) -> str:
        return IL.digest(self.text)

    def evaluate(self, ports: list[np.ndarray], flip: tuple[int, int] | None = None):
        """``(output words per port, lane mask of satisfied lanes)`` of the laid-out rows over lane-major port words."""
        return IL.evaluate_layout(self.layout, ports, len(ports[0]), flip)


@lru_cache(maxsize=None)
def lane(V: int) -> Lane:
    C = lane_circuit(V)
    lay = IL.layout(C, [f"in{j}" for j in range(len(IN_PORTS))], [f"out{j}" for j in range(len(OUT_PORTS))], [1], None)
    return Lane(V, C, lay)


# --- the words of one row, as the verifier computes them ------------------------------------------------------------------------

def _prim(name: str):
    from verity_vllm.definitions.registry import prims as P
    from verity.circuits.ops import scalar as S

    return getattr(P, name, None) or getattr(S, name)


def row_scalars(temp: int) -> list[int]:
    """``[temp, rcp, skip, noisy]`` by the IR primitives (``TemperatureScale``'s and ``GumbelSelectF32``'s row values)."""
    eq = _prim("F32Eq").evaluate
    rcp = _prim("DivFullRcp").evaluate(temp)
    return [temp & 0xFFFFFFFF, rcp, eq(temp, 0) | eq(temp, ONE_F32), 1 - eq(temp, 0)]


def _add_ftz_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``F32AddFtz_v1`` lane for lane: the vectorised twin, and the primitive itself on every lane with a NaN operand."""
    from verity_vllm.definitions.registry import sampling_rows as SR

    out = np.asarray(SR.add_ftz(a, b), dtype=np.uint32)
    nan = ((a & 0x7F800000) == 0x7F800000) & ((a & 0x7FFFFF) != 0) | ((b & 0x7F800000) == 0x7F800000) & ((b & 0x7FFFFF) != 0)
    prim = _prim("F32AddFtz").evaluate
    for v in np.flatnonzero(nan):
        out[v] = prim(int(a[v]), int(b[v]))
    return out


def _scan(y: np.ndarray):
    """``GumbelSelectF32``'s scan over y: every lane's carry out ``(best, best_i, i + 1)`` (strict ``>``, NaN never wins, a
    NaN ``y_0`` pins the best)."""
    V = len(y)
    idx = np.arange(V, dtype=np.int64)
    yf = y.view(np.float32)
    if np.isnan(yf[0]):
        return np.full(V, y[0], np.uint32), np.zeros(V, np.uint32), (idx + 1).astype(np.uint32)
    cand = np.where(np.isnan(yf), np.float32(-np.inf), yf)
    run = np.maximum.accumulate(cand)
    new = np.zeros(V, bool)
    new[0] = True
    new[1:] = cand[1:] > run[:-1]
    best_i = np.maximum.accumulate(np.where(new, idx, 0))
    return y[best_i].astype(np.uint32), best_i.astype(np.uint32), (idx + 1).astype(np.uint32)


def row_words(logits: np.ndarray, top_p: int, seed: int, pos: int, temp: int, splits: int) -> tuple[np.ndarray, int]:
    """Every cut word of one instance (:func:`words` of them) and its token: the units' words by the vectorised twins of the
    circuits' primitives (``verity_vllm.definitions.registry.sampling_rows``, held word-equal to them), the native ones by the
    primitives and the top-p split pipeline (``TopPMaskWordx{V}``'s own reference)."""
    from verity_vllm.definitions.registry import sampling as SM
    from verity_vllm.definitions.registry import sampling_rows as SR

    logits = np.asarray(logits, dtype=np.uint16).reshape(-1)
    V = len(logits)
    temp_, rcp, skip, noisy = row_scalars(temp)
    scaled = np.asarray(SR.temperature_scale_row(logits, temp_), dtype=np.uint32)
    keep = SM.topp_keep(scaled.view(np.float32), SM._f(int(top_p)), int(splits))
    masked = np.where(keep, scaled, np.uint32(NEG_INF)).astype(np.uint32)
    g = np.asarray(SR.noise_row(SM.stream_key(int(seed), int(pos)), V), dtype=np.uint32)
    y = _add_ftz_rows(masked, g) if noisy else masked
    best, best_i, i_next = _scan(y)
    out = np.zeros(words(V), dtype=np.uint32)
    out[:4] = [temp_, rcp, skip, noisy]
    lane = out[len(SCALARS) + len(INIT):].reshape(V, len(LANE))
    for k, col in enumerate((keep.astype(np.uint32), g, scaled, best, best_i, i_next)):
        lane[:, k] = col
    return out, int(best_i[-1])


def native_words(cuts: np.ndarray, top_p: int, seed: int, pos: int, temp: int, splits: int) -> np.ndarray:
    """The native cut words of one instance recomputed from its public words (the scalar ports and the units' ``scaled``
    words): what :func:`check_native` holds a file's cut words to."""
    from verity_vllm.definitions.registry import sampling as SM
    from verity_vllm.definitions.registry import sampling_rows as SR

    V = (len(cuts) - len(SCALARS) - len(INIT)) // len(LANE)
    lane = np.asarray(cuts[len(SCALARS) + len(INIT):], dtype=np.uint32).reshape(V, len(LANE))
    scaled = lane[:, LANE.index("scaled")].copy()
    out = np.asarray(cuts, dtype=np.uint32).copy()
    out[:4] = row_scalars(temp)
    out[4:7] = 0
    nl = out[len(SCALARS) + len(INIT):].reshape(V, len(LANE))
    nl[:, LANE.index("kbit")] = SM.topp_keep(scaled.view(np.float32), SM._f(int(top_p)), int(splits))
    nl[:, LANE.index("g")] = SR.noise_row(SM.stream_key(int(seed), int(pos)), V)
    return out


# --- the input set's rows ----------------------------------------------------------------------------------------------------------

def _scalar(input_set, name: str, lo: int, hi: int) -> np.ndarray:
    return np.asarray(input_set.port(name, lo, hi), dtype=np.uint64).reshape(hi - lo, -1)[:, 0]


def instances(input_set, lo: int, hi: int):
    """``(logits (n, V) u16, scalars {port: (n,) u64}, outputs (n,) u64)`` of ids [lo, hi)."""
    n = hi - lo
    logits = np.asarray(input_set.port(LOGITS, lo, hi), dtype=np.uint16).reshape(n, -1)
    scalars = {p: _scalar(input_set, p, lo, hi) for p in PUBLIC_PORTS}
    return logits, scalars, _scalar(input_set, OUTPUT, lo, hi)


def unit_ports(cuts: np.ndarray, logits: np.ndarray) -> list[np.ndarray]:
    """Per lane-unit input port, the lane-major words of every (instance, lane) (instance-major): what the units read."""
    n, V = logits.shape
    ports = [np.asarray(logits, dtype=np.uint64).reshape(-1)]
    idx = np.array([lane_ports(v)[0] for v in range(V)], dtype=np.int64)            # (V, 9)
    for j in range(idx.shape[1]):
        ports.append(np.asarray(cuts, dtype=np.uint64)[:, idx[:, j]].reshape(-1))
    return ports


def unit_outputs(cuts: np.ndarray, V: int) -> list[np.ndarray]:
    idx = np.array([lane_ports(v)[1] for v in range(V)], dtype=np.int64)
    return [np.asarray(cuts, dtype=np.uint64)[:, idx[:, j]].reshape(-1) for j in range(idx.shape[1])]


def pin_digest(low: Lane) -> str:
    return hashlib.sha256(low.unit_text.encode()).hexdigest()
