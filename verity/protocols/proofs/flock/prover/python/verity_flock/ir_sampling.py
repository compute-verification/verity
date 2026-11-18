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
  word is **native**: the verifier computes it from the public words. The top-p keep word is native
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

def _composites(d):
    """``d``'s (``GumbelTopPTokenSelect_v1{V}``) three scans' lane bodies: ``TemperatureLane``, ``TopPMaskStep{V}``,
    ``GumbelSelectStep``; refuses a Definition without them."""
    want = ("TemperatureLane_v1", f"TopPMaskStep_v1{{V={int(d.bindings['V'])}}}", "GumbelSelectStep_v1")
    found, todo = {}, [d]
    while todo:
        for n in todo.pop().body.nodes:
            if n.fn.id in want:
                found.setdefault(n.fn.id, n.fn)
            elif hasattr(n.fn, "body"):
                todo.append(n.fn)
    if missing := [w for w in want if w not in found]:
        raise ValueError(f"{d.id}: its scans have no {', '.join(missing)}")
    return tuple(found[w] for w in want)


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


def lane_circuit(d) -> Circuit:
    """``d``'s lane unit: inputs ``in0`` (the bf16 logit) and ``in1 .. in9`` (``IN_PORTS[1:]``, 32 bits each; skip, noisy and kbit
    use bit 0), outputs ``out0 .. out3`` (``OUT_PORTS``)."""
    V = int(d.bindings["V"])
    temp_lane, mask_step, select_step = _composites(d)
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
def lane(d) -> Lane:
    C = lane_circuit(d)
    lay = IL.layout(C, [f"in{j}" for j in range(len(IN_PORTS))], [f"out{j}" for j in range(len(OUT_PORTS))], [1], None)
    return Lane(int(d.bindings["V"]), C, lay)


def pin_digest(low: Lane) -> str:
    return hashlib.sha256(low.unit_text.encode()).hexdigest()
