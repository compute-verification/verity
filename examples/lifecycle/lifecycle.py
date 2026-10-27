"""The lifecycle both composite protocols step through, on a tiny model, with the parts they share: the phases as functions
with typed inputs and outputs, and the developer and verifier as parties of the deployment layer (`verity.core.deploy`).
`onsite.py` and `remote.py` add what each composite has of its own; `run.py` runs both; README.md walks through it.

The phases, each a function here:

1. **Registration** (`register`, `check_registration`): the hardware records, theta's commitment, the circuit builder B by
   digest with its advice cap, and the initial state's commitment, all before any coin.
2. **Operation** (`operate`): the developer serves one segment's requests through the model (`forward`) on simulated
   devices, realizing its run-time randomness: placement on a registered device and batching.
3. **Close of segment k** (`close`, `check_close`): the developer posts the advice a_k, B builds the segment's circuit from
   a_1 || ... || a_k (`Builder.grow`: B is prefix-extending, so the earlier segments' circuits are unchanged), the
   segment's values are committed, and its record binds the previous record's digest.
4. **Audit of segment k** (`draw`, `prove`, `verify`, `pouw_window`): live coins after the commitment, sampled proofs'
   draw at PoUW's size K', each drawn unit proved by the firewall and checked by the verifier.
5. **Accounting** (`compose`, `account`): each segment's composite verdict by the composition's rule, and the period's.

`prove` and `verify` are a stand-in for C-Flock behind the firewall: the proof is the drawn unit's opening, which `verify`
checks against the commitments and re-evaluates in the clear. It is sound for the example and not zero-knowledge.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from verity.core.deploy.party import Finish, Identity, Received, Record, Send, Timer
from verity.core.protocols.pouw import window as pouw_window_rules
from verity.core.protocols.sampled_proofs.coins import OwnRandomness, draw_first
from verity.core.protocols.sampled_proofs.one_stage import registered as R
from verity.core.protocols.sampled_proofs.parties import IDS
from verity.core.wire import Body, Int, Message, Schema
from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits.boolean import trace
from verity.ml.circuits.boolean.forms import NOT, ZERO, add, compress_columns, or_reduce
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation import evaluate_words
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array, Tuple
from verity.ml.commitments import hm96

ROOT = Path(__file__).resolve().parents[2]

# --- the tiny model and its circuit ------------------------------------------------------------------------------------

D = 4
"""The model's width: two D x D int8 matmuls a request."""
ACC = 20
"""An accumulator's bits: four int8 products sum to less than 2^17 in magnitude."""
STATE = 16
"""The state word's bits: a running sum of every output, carried from unit to unit and from segment to segment."""
BIT = BL.BIT
BYTE = Array(8, BIT)
UNIT_PARAMS = (("state", Array(STATE, BIT)), ("w1", Array(D * D, BYTE)), ("w2", Array(D * D, BYTE)), ("x", Array(D, BYTE)))
UNIT_RET = Tuple(Array(D, BYTE), Array(STATE, BIT))
MACS = 2 * D * D
"""A unit's work: its multiply-accumulates, PoUW's work per tile."""


def _sext(x: Sequence[Any], n: int) -> list[Any]:
    return list(x) + [x[-1]] * (n - len(x))


def _smul(C: Any, a: Sequence[Any], b: Sequence[Any], n: int = 16) -> list[Any]:
    """a * b mod 2^n for two's complement a and b, sign-extended to n bits."""
    a, b = _sext(a, n), _sext(b, n)
    cols: list[list[Any]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n - i):
            cols[i + j].append(C.AND(a[i], b[j]))
    return compress_columns(C, cols)[:n]


def _dot(C: Any, w: Sequence[Sequence[Any]], x: Sequence[Sequence[Any]]) -> list[Any]:
    acc = [ZERO] * ACC
    for wj, xj in zip(w, x):
        acc = add(C, acc, _sext(_smul(C, wj, xj), ACC), width=ACC)
    return acc


def _or(C: Any, a: Any, b: Any) -> Any:
    return a ^ b ^ C.AND(a, b)


def _relu_shift(C: Any, acc: Sequence[Any]) -> list[Any]:
    """clamp(acc >> 7, 0, 127) as an int8."""
    s, over = acc[-1], or_reduce(C, acc[14:ACC - 1])
    keep = NOT(s)
    return [_or(C, C.AND(keep, b), C.AND(keep, over)) for b in acc[7:14]] + [ZERO]


def _sat_shift(C: Any, acc: Sequence[Any]) -> list[Any]:
    """clamp(acc >> 7, -128, 127) as an int8."""
    s = acc[-1]
    wide = or_reduce(C, [h ^ s for h in acc[15:ACC - 1]] + [acc[14] ^ s])
    sat = [NOT(s)] * 7 + [s]
    return [v ^ C.AND(wide, v ^ t) for v, t in zip(acc[7:15], sat)]


def _request(C: Any, state: Sequence[Any], *rest: Sequence[Any]) -> list[list[Any]]:
    w1, w2, x = rest[:D * D], rest[D * D:2 * D * D], rest[2 * D * D:]
    h = [_relu_shift(C, _dot(C, w1[D * i:D * i + D], x)) for i in range(D)]
    y = [_sat_shift(C, _dot(C, w2[D * i:D * i + D], h)) for i in range(D)]
    s = list(state)
    for yi in y:
        s = add(C, s, _sext(yi, STATE), width=STATE)
    return [*y, s]


UNIT = trace.definition("LifecycleRequest", 1, _request, UNIT_PARAMS, UNIT_RET, register=False,
                        doc="one request through the tiny model: y = sat(W2 relu(W1 x) >> 7) >> 7, state' = state + sum(y)")
"""The proof unit: one request, as a Boolean Definition (And/Xor/Not gates)."""


def _segment_signature(S: Any) -> tuple[tuple[tuple[str, Any], ...], Any]:
    return (UNIT_PARAMS[:3] + (("x", Array(S.N, Array(D, BYTE))),), Tuple(Array(S.N, Array(D, BYTE)), Array(STATE, BIT)))


def _segment_body(B: Any, S: Any, state: Coll, w1: Coll, w2: Coll, x: Coll) -> Coll:
    ys = []
    for i in range(S.N):
        out = B.call(UNIT, state, w1, w2, x[i])
        ys.append(out[0])
        state = out[1]
    return Coll(_segment_signature(S)[1], concat([y.refs for y in ys] + [state.refs]))


SEGMENT = CompositeDefinition("LifecycleSegment", 1, ("N",), _segment_signature, _segment_body, register=False,
                              doc="a segment of the growing circuit: N units chained through the state wire")


@cache
def unit_ands() -> int:
    return UNIT.traced().counts["and"]


@cache
def segment_digest(n: int) -> str:
    """The segment circuit's Program digest (`verity.ml.circuits.codec`): its input state wires are the previous segment's
    output state wires, and theta is a registered input."""
    return codec.program_digest(codec.encode_program(Program(bind(SEGMENT, N=n), registered=("w1", "w2"), theta=("w1", "w2"))))


def s8(v: int) -> int:
    return ((int(v) + 128) & 0xFF) - 128


def forward(theta: Theta, state: int, x: Sequence[int]) -> tuple[list[int], int]:
    """The served model, in integers: what the developer's serving stack computes."""
    h = []
    for i in range(D):
        a = sum(theta.w1[D * i + j] * x[j] for j in range(D))
        h.append(0 if a < 0 else min(127, a >> 7))
    y = []
    for i in range(D):
        a = sum(theta.w2[D * i + j] * h[j] for j in range(D))
        y.append(max(-128, min(127, a >> 7)))
    return y, (state + sum(y)) % (1 << STATE)


def evaluate_unit(theta: Theta, state: int, x: Sequence[int]) -> tuple[list[int], int]:
    """The unit's circuit evaluated in the clear (`verity.ml.circuits.evaluation`, through the Boolean Definition's word
    view)."""
    u8 = lambda v: [c & 0xFF for c in v]  # noqa: E731
    out = evaluate_words(UNIT, np.array([state]), np.array([u8(theta.w1)]), np.array([u8(theta.w2)]), np.array([u8(x)]))[0]
    return [s8(c) for c in out[:D]], int(out[D])


# --- values and their commitments --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Theta:
    """The developer's information: the model's weights, row-major."""

    w1: tuple[int, ...]
    w2: tuple[int, ...]

    def rows(self) -> list[list[int]]:
        """Theta as registered rows: each matrix row one row of D int8 codes."""
        return [[c & 0xFF for c in m[D * i:D * i + D]] for m in (self.w1, self.w2) for i in range(D)]

    @classmethod
    def of_rows(cls, rows: Sequence[Sequence[int]]) -> Theta:
        flat = [s8(c) for r in rows for c in r]
        return cls(tuple(flat[:D * D]), tuple(flat[D * D:]))


THETA_ROWS = 2 * D
ROW_WORDS = 2 * D + 1
"""A unit's committed row: its D input codes, its D output codes and its output state word."""
HM = hm96.SHA512


class Tape:
    """The developer's private randomness: one seed from its coins source, stretched by SHA-512 under labels."""

    def __init__(self, seed: bytes) -> None:
        self.seed = bytes(seed)

    def bytes(self, label: str, n: int) -> bytes:
        out, i = b"", 0
        while len(out) < n:
            out += hashlib.sha512(b"lifecycle/tape/v1\0" + self.seed + label.encode() + i.to_bytes(8, "big")).digest()
            i += 1
        return out[:n]

    def salts(self, label: str, n: int) -> list[bytes]:
        return [self.bytes(f"{label}/salt/{i}", HM.salt_bytes) for i in range(n)]


def commit(program: str, value: str, rows: Sequence[Sequence[int]], tape: Tape) -> R.Registered:
    """Rows committed as hm96-sha512 leaves in a frame-v3-sha512 tree (`one_stage.registered`)."""
    return R.Registered.commit(program, value, rows, tape.salts(f"{program}/{value}", len(rows)))


def opening(reg: R.Registered, position: int) -> dict[str, Any]:
    return {"position": position, "row": list(reg.rows[position]), "salt": reg.salts[position].hex(),
            "path": reg.path(position).hex()}


def opens(where: Mapping[str, Any], o: Mapping[str, Any]) -> bool:
    """Whether opening `o` opens the committed value `where` ({program, value, words, leaves, root}) under the verifier's
    own domain."""
    dom = R.domain(where["program"], where["value"], where["words"], where["leaves"])
    bc = HM.commit_string(HM.default_key, R.row_digest(o["row"]), bytes.fromhex(o["salt"]))
    return R.verify_read(dom, bytes.fromhex(where["root"]), o["position"], bc, bytes.fromhex(o["path"]))


def committed(reg: R.Registered) -> dict[str, Any]:
    """What the verifier holds of a committed value: where it is, never its rows."""
    return {"program": reg.program, "value": reg.value, "words": reg.words, "leaves": reg.leaves, "root": reg.root.hex()}


def digest(tag: str, obj: Any) -> str:
    return hashlib.sha512(tag.encode() + b"\0" + codec.canonical_json(obj)).hexdigest()


# --- 1. registration ---------------------------------------------------------------------------------------------------

SKUS = {"sim-int8-a": {"int8_macs_per_ns": 64, "mem_bytes_per_ns": 400},
        "sim-int8-b": {"int8_macs_per_ns": 128, "mem_bytes_per_ns": 800}}
"""Simulated SKUs and their data-sheet constants."""


@dataclass(frozen=True)
class Device:
    name: str
    sku: str
    link: str = ""
    """Onsite: the uplink it sits behind."""


@dataclass(frozen=True)
class Hardware:
    devices: tuple[Device, ...]
    certifiers: tuple[tuple[str, str], ...] = ()
    """Onsite: each uplink and the network certifier on it."""

    def to_json(self) -> dict[str, Any]:
        return {"devices": [{"name": d.name, "sku": d.sku, **SKUS[d.sku], **({"link": d.link} if d.link else {})}
                            for d in self.devices],
                "certifiers": [{"link": link, "certifier": c} for link, c in self.certifiers]}


@dataclass(frozen=True)
class Builder:
    """B, the registered circuit builder: advice to the growing circuit. A segment's advice is self-delimiting (its unit
    count, then each unit's device and batch), so B(a_1 || ... || a_k) parses segment by segment and extends
    B(a_1 || ... || a_{k-1}): B is prefix-extending. `cap` is the public cap on one segment's advice, in bits."""

    cap: int = 40
    COUNT_BITS = 8
    DEVICE_BITS = 4
    BATCH_BITS = 4

    def descriptor(self) -> dict[str, Any]:
        return {"format": "lifecycle/builder/v1", "unit": segment_digest(1), "segment": SEGMENT.name, "cap": self.cap,
                "advice": {"count": self.COUNT_BITS, "device": self.DEVICE_BITS, "batch": self.BATCH_BITS},
                "grid": pouw_window_rules.GRID_BITS}

    def digest(self) -> str:
        return digest("lifecycle/builder/v1", self.descriptor())

    def encode(self, placements: Sequence[Placement]) -> str:
        bits = format(len(placements), f"0{self.COUNT_BITS}b")
        for p in placements:
            bits += format(p.device, f"0{self.DEVICE_BITS}b") + format(p.batch, f"0{self.BATCH_BITS}b")
        return bits

    def segments(self, advice: str) -> list[str]:
        """The advice cut into its segments' advice, or ValueError."""
        out, i = [], 0
        if not set(advice) <= {"0", "1"}:
            raise ValueError("advice is a bit string")
        while i < len(advice):
            if i + self.COUNT_BITS > len(advice):
                raise ValueError("advice ends inside a segment's count")
            n = int(advice[i:i + self.COUNT_BITS], 2)
            end = i + self.COUNT_BITS + n * (self.DEVICE_BITS + self.BATCH_BITS)
            if end > len(advice):
                raise ValueError("advice ends inside a segment")
            out.append(advice[i:end])
            i = end
        return out

    def build(self, advice: str) -> list[SegmentCircuit]:
        """B(advice): every segment's circuit, from the concatenated advice."""
        out = []
        for k, a in enumerate(self.segments(advice), 1):
            step = self.DEVICE_BITS + self.BATCH_BITS
            units = tuple(Placement(int(a[j:j + self.DEVICE_BITS], 2), int(a[j + self.DEVICE_BITS:j + step], 2))
                          for j in range(self.COUNT_BITS, len(a), step))
            if not units:
                raise ValueError(f"segment {k} serves nothing")
            work = pouw_window_rules.bucket(len(units))
            out.append(SegmentCircuit(k, units, work.n, segment_digest(work.n), len(a)))
        return out

    def grow(self, circuit: Sequence[SegmentCircuit], advice: str, a_k: str) -> SegmentCircuit:
        """Segment k's circuit: B(advice || a_k), which must extend the circuit so far."""
        if len(a_k) > self.cap:
            raise Refusal("advice-cap", f"segment {len(circuit) + 1}'s advice is {len(a_k)} bits, over the cap of {self.cap}")
        try:
            grown = self.build(advice + a_k)
        except ValueError as e:
            raise Refusal("advice", str(e)) from None
        if len(grown) != len(circuit) + 1 or grown[:-1] != list(circuit):
            raise Refusal("prefix", "B(a_1 || ... || a_k) does not extend the circuit so far")
        return grown[-1]


@dataclass(frozen=True)
class Placement:
    device: int
    batch: int


@dataclass(frozen=True)
class SegmentCircuit:
    """Segment k of the growing circuit: its real units' placements, its units padded to PoUW's grid (`n`, fillers after
    the real units), its Program's digest and its advice's length."""

    index: int
    units: tuple[Placement, ...]
    n: int
    program: str
    advice_bits: int

    @property
    def real(self) -> int:
        return len(self.units)

    def ands(self) -> int:
        return self.n * unit_ands()

    def credited_ands(self) -> int:
        """The AND gates PoUW credits: the grid point at or below the real units, fillers credited nothing."""
        return pouw_window_rules.bucket(self.real).credited * unit_ands()


class Refusal(Exception):
    """A phase's refusal, by name."""

    def __init__(self, name: str, detail: str) -> None:
        super().__init__(f"{name}: {detail}")
        self.name, self.detail = name, detail


@dataclass(frozen=True)
class Registration:
    """Everything registered before any coin: public digests and hiding roots, no value."""

    product: str
    hardware: Hardware
    theta: dict[str, Any]
    builder: str
    cap: int
    state0: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"product": self.product, "hardware": self.hardware.to_json(), "theta": self.theta,
                "builder": {"digest": self.builder, "cap": self.cap}, "state0": self.state0}

    def digest(self) -> str:
        return digest("lifecycle/registration/v1", self.to_json())


@dataclass(frozen=True)
class Private:
    """The developer's side of the registration: theta's and the initial state's rows and salts."""

    theta: R.Registered
    state0: R.Registered


def register(product: str, hardware: Hardware, theta: Theta, builder: Builder, state0: int, tape: Tape
             ) -> tuple[Registration, Private]:
    """Phase 1, the developer's side."""
    b = builder.digest()
    t = commit(b, "theta", theta.rows(), tape)
    s = commit(b, "state0", [[state0]], tape)
    return Registration(product, hardware, committed(t), b, builder.cap, committed(s)), Private(t, s)


def check_registration(j: Mapping[str, Any], builder: Builder, certifiers: Sequence[str]) -> Registration:
    """Phase 1, the verifier's side: B is the verifier's own copy, every SKU is one it has a data sheet for, every unit
    of work has a registered device, and onsite every uplink's certifier is one of the auditor's own devices."""
    if j["builder"] != {"digest": builder.digest(), "cap": builder.cap}:
        raise Refusal("builder", "the registered builder is not the verifier's copy of B")
    devices = tuple(Device(d["name"], d["sku"], d.get("link", "")) for d in j["hardware"]["devices"])
    if not devices or any(d.sku not in SKUS or {k: j_d[k] for k in SKUS[d.sku]} != SKUS[d.sku]
                          for d, j_d in zip(devices, j["hardware"]["devices"])):
        raise Refusal("hardware", "a device's SKU or data-sheet constants are not the auditor's")
    links = tuple((c["link"], c["certifier"]) for c in j["hardware"]["certifiers"])
    if any(c not in certifiers for _, c in links) or any(d.link and d.link not in dict(links) for d in devices):
        raise Refusal("hardware", "an uplink's network certifier is not one of the auditor's devices")
    for name in ("theta", "state0"):
        w = j[name]
        if (w["program"], w["value"]) != (j["builder"]["digest"], name):
            raise Refusal("registration", f"{name} is not committed under the registered builder")
    return Registration(j["product"], Hardware(devices, links), dict(j["theta"]), builder.digest(), builder.cap,
                        dict(j["state0"]))


# --- 2. operation ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Served:
    """One segment as served: each request's placement, input, output and the state after it. `wire` is the output
    each request's egress frame carried."""

    placements: tuple[Placement, ...]
    xs: tuple[tuple[int, ...], ...]
    ys: tuple[tuple[int, ...], ...]
    states: tuple[int, ...]
    wire: tuple[tuple[int, ...], ...]


BATCH = 2
"""Requests a batch holds at most."""


def operate(requests: Sequence[Sequence[int]], theta: Theta, state: int, hardware: Hardware, tape: Tape, k: int
            ) -> Served:
    """Phase 2: serve the segment's requests in arrival order, each on a device the developer's tape picks among the
    registered ones, batched by arrival."""
    picks = tape.bytes(f"placement/{k}", len(requests))
    placements, ys, states = [], [], []
    for i, x in enumerate(requests):
        placements.append(Placement(picks[i] % len(hardware.devices), i // BATCH))
        y, state = forward(theta, state, x)
        ys.append(tuple(y))
        states.append(state)
    return Served(tuple(placements), tuple(tuple(x) for x in requests), tuple(ys), tuple(states), tuple(ys))


# --- 3. close of segment k ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Close:
    """What closing segment k posts: its advice and its record, whose digest the audit commits to."""

    k: int
    advice: str
    record: dict[str, Any]

    @property
    def digest(self) -> str:
        return digest("lifecycle/segment/v1", self.record)


def rows_of(served: Served, n: int, state: int) -> list[list[int]]:
    """The segment's rows: each real unit's, then each filler's (input zero, so output zero and the state unchanged)."""
    rows = [[*(c & 0xFF for c in x), *(c & 0xFF for c in y), s] for x, y, s in zip(served.xs, served.ys, served.states)]
    last = served.states[-1] if served.states else state
    return rows + [[0] * (2 * D) + [last]] * (n - len(rows))


def close(k: int, served: Served, prev: str, builder: Builder, circuit: Sequence[SegmentCircuit], advice: str, state: int,
          tape: Tape) -> tuple[Close, SegmentCircuit, R.Registered]:
    """Phase 3, the developer's side: the advice, B's segment circuit, the values committed and the record."""
    a_k = builder.encode(served.placements)
    seg = builder.grow(circuit, advice, a_k)
    values = commit(seg.program, f"segment/{k}", rows_of(served, seg.n, state), tape)
    record = {"segment": k, "prev": prev, "advice_bits": len(a_k), "circuit": seg.program, "units": seg.n,
              "values": committed(values)}
    return Close(k, a_k, record), seg, values


def check_close(c: Close, reg: Registration, builder: Builder, circuit: Sequence[SegmentCircuit], advice: str
                ) -> SegmentCircuit:
    """Phase 3, the verifier's side: the advice within the cap, B's circuit grown from it, every unit placed on a
    registered device, and the record naming that circuit and its values."""
    seg = builder.grow(circuit, advice, c.advice)
    if any(p.device >= len(reg.hardware.devices) for p in seg.units):
        raise Refusal("placement", f"segment {c.k} places a unit on a device the hardware registration doesn't hold")
    v = c.record.get("values", {})
    want = {"segment": c.k, "advice_bits": len(c.advice), "circuit": seg.program, "units": seg.n}
    if ({k: c.record.get(k) for k in want} != want or set(c.record) != {*want, "prev", "values"}
            or (v.get("program"), v.get("value"), v.get("words"), v.get("leaves")) != (seg.program, f"segment/{c.k}",
                                                                                   ROW_WORDS, seg.n)):
        raise Refusal("record", f"segment {c.k}'s record is not B's circuit and its values")
    return seg


# --- 4. audit of segment k ---------------------------------------------------------------------------------------------

EPS, DELTA = Fraction(1, 2), Fraction(1, 4)
"""What one segment's audit establishes: except with probability δ, fewer than a share ε of its credited work is wrong."""


def draw_size(seg: SegmentCircuit) -> int:
    """PoUW's K' for the segment (`pouw.window.k_prime`): subset:K' over its n units makes a wrong share ε of the credited
    work missed with probability at most δ."""
    work = pouw_window_rules.bucket(seg.real)
    return pouw_window_rules.k_prime(EPS, DELTA, work.credited * MACS, work.n, MACS)


def coin_bytes(k: int, n: int) -> int:
    """Enough coins for `draw_first`'s subset:k of n, four times the stream it reads first."""
    return 4 * (2 * k * ((n.bit_length() + 64 + 7) // 8) + 64)


def draw(coins: bytes, seg: SegmentCircuit) -> list[int]:
    """Sampled proofs' draw (`sampled_proofs.coins.draw_first`) on the verifier's live coins."""
    buf = bytearray(coins)

    def read(n: int) -> bytes:
        out = bytes(buf[:n])
        del buf[:n]
        return out

    return draw_first(OwnRandomness(read), {"law": "subset", "k": draw_size(seg)}, seg.n)["units"]


@dataclass(frozen=True)
class Statement:
    """A drawn unit's public statement: the segment it is in, its index, whether it is a filler, where its values, its
    input state and theta are committed, and onsite the certifier's commitments to its frames."""

    segment: str
    unit: int
    filler: bool
    values: dict[str, Any]
    state_in: dict[str, Any]
    theta: dict[str, Any]
    frames: dict[str, str] | None = None


def prove(statement: Statement, witness: Mapping[str, Any]) -> dict[str, Any]:
    """STAND-IN for C-Flock behind the firewall: the proof is the unit's opening."""
    return dict(witness)


def verify(statement: Statement, proof: Mapping[str, Any]) -> dict[str, str | None]:
    """STAND-IN for the verifier of record: each opening against its commitment, then the unit's circuit evaluated in the
    clear on the opened inputs and held to the opened outputs; onsite, the opened I/O held to the frames the certifier
    committed to. The reason for each failure, by check (`values`, `io`), None where it holds."""
    out: dict[str, str | None] = {"values": None, "io": None}
    try:
        unit, state_in, theta = proof["unit"], proof["state_in"], proof["theta"]
        if unit["position"] != statement.unit or state_in["position"] != statement.state_in["position"]:
            out["values"] = "an opening is at another position"
        elif not opens(statement.values, unit) or not opens(statement.state_in, state_in):
            out["values"] = "an opening doesn't open its commitment"
        elif [o["position"] for o in theta] != list(range(THETA_ROWS)) or not all(opens(statement.theta, o) for o in theta):
            out["values"] = "theta's openings don't open its commitment"
        else:
            row, s_in = unit["row"], state_in["row"][-1]
            x, y = [s8(c) for c in row[:D]], [s8(c) for c in row[D:2 * D]]
            want_y, want_s = evaluate_unit(Theta.of_rows([o["row"] for o in theta]), s_in, x)
            if (y, row[-1]) != (want_y, want_s):
                out["values"] = f"unit {statement.unit}'s committed outputs are not its circuit's on its inputs"
            elif statement.filler and any(x):
                out["values"] = f"filler unit {statement.unit} has an input"
            if statement.frames is not None and not statement.filler:
                frames = proof["frames"]
                if (frame_commitment(bytes.fromhex(frames["in"]), bytes.fromhex(frames["in_salt"])) != statement.frames["in"]
                        or frame_commitment(bytes.fromhex(frames["out"]), bytes.fromhex(frames["out_salt"]))
                        != statement.frames["out"]):
                    out["io"] = "the frames don't open the certifier's commitments"
                elif (frame(x), frame(y)) != (bytes.fromhex(frames["in"]), bytes.fromhex(frames["out"])):
                    out["io"] = f"unit {statement.unit}'s I/O gates disagree with the frames the certifier saw"
    except (KeyError, TypeError, ValueError) as e:
        out["values"] = f"malformed proof: {type(e).__name__}: {e}"
    return out


def frame(codes: Sequence[int]) -> bytes:
    """A request's or a response's frame on the wire: its D int8 codes."""
    return bytes(c & 0xFF for c in codes)


def frame_commitment(data: bytes, salt: bytes) -> str:
    """The certifier's hiding commitment to a frame it saw."""
    return hashlib.sha512(b"lifecycle/frame/v1\0" + salt + data).hexdigest()


def answer(protocol: str, verdict: str, commitment: str | None, reason: str | None = None,
           public: Any = None, profile: Any = None) -> dict[str, Any]:
    """`Core.Protocol`'s outcome (`verity/core/protocols/interface`): public outputs and a profile on acceptance only."""
    ok = verdict == "accept"
    return {"protocol": protocol, "verdict": verdict, "commitment": commitment, "reason": None if ok else reason,
            "public": public if ok else None, "profile": profile if ok else None}


def sampled_proofs(close_: Close, seg: SegmentCircuit, units: Sequence[int], checks: Mapping[int, Mapping[str, Any]]
                   ) -> dict[str, Any]:
    """Sampled proofs' answer for the segment: every drawn unit's values check held."""
    bad = [f"unit {u}: {checks[u]['values']}" for u in units if u not in checks or checks[u]["values"]]
    law = {"law": "subset", "k": len(units)}
    return answer(IDS["sampled-proofs"], "reject" if bad else "accept", close_.digest, "; ".join(bad) or None,
                  {"draw": {**law, "population": seg.n, "units": list(units)}},
                  {"law": law, "population": seg.n, "delta": str(DELTA)})


def pouw_window(sp: Mapping[str, Any], seg: SegmentCircuit) -> dict[str, Any]:
    """PoUW's window check over sampled proofs' answer: sampled proofs accepted the segment's registration, at subset:K'
    over the grid's n units; it publishes the credited work at ε and δ."""
    work = pouw_window_rules.bucket(seg.real)
    if sp["verdict"] != "accept":
        return answer(IDS["pouw"], sp["verdict"], sp["commitment"], "sampled proofs did not accept the window")
    d = sp["public"]["draw"]
    if (d["law"], d["k"], d["population"]) != ("subset", draw_size(seg), work.n):
        return answer(IDS["pouw"], "reject", sp["commitment"], "the draw is not subset:K' over the window's grid")
    return answer(IDS["pouw"], "accept", sp["commitment"], public={"work": work.credited * MACS, "tiles": work.credited,
                                                                   "eps": str(EPS), "delta": str(DELTA)})


# --- 5. accounting -----------------------------------------------------------------------------------------------------

def _reference() -> Any:
    """The composition's Python mirror of its Lean verdict (`verity/core/protocols/composition/tests/reference.py`)."""
    name = "lifecycle_composition_reference"
    if name not in sys.modules:
        path = ROOT / "verity/core/protocols/composition/tests/reference.py"
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules[name]


def products() -> dict[str, tuple[str, tuple[str, ...]]]:
    """Each composite's product id and its components' ids, in the order of their ids (`Core.Onsite`, `Core.Remote`)."""
    ref = _reference()
    return {"onsite": (ref.ONSITE.id, tuple(c.id for c in ref.ONSITE.components)),
            "remote": (ref.REMOTE.id, tuple(c.id for c in ref.REMOTE.components))}


def compose(protocol: str, answers: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """A segment's composite verdict: the composition's rule over exactly the product's components (any reject rejects,
    then any undecided leaves it undecided), and its commitment the binding of their digests (`Core.Composition.bind`)."""
    ref = _reference()
    pid, ids = products()[protocol]
    missing = [i for i in ids if i not in answers]
    verdicts = [answers[i]["verdict"] if i in answers else "undecided" for i in ids]
    rule = ref.verdict(verdicts)
    digests = [(i, bytes.fromhex(answers[i]["commitment"])) for i in ids if i in answers and answers[i]["commitment"]]
    bound = ref.bind(pid, digests).hex() if len(digests) == len(ids) else None
    reason = next((f"{i}: {answers[i]['reason']}" for i in ids if i in answers and answers[i]["verdict"] == "reject"),
                  f"no answer from {missing}" if missing else None)
    return {"product": pid, "verdict": rule, "commitment": bound, "reason": None if rule == "accept" else reason,
            "components": {i: v for i, v in zip(ids, verdicts)}}


SEGMENT_SECONDS = 60
"""A stated parameter: how long a segment lasts in a deployment, which scales a period's segment count."""
PERIODS = {"week": 7 * 24 * 3600, "month": 30 * 24 * 3600, "year": 365 * 24 * 3600}


def pooled(segments: Sequence[Mapping[str, Any]]) -> tuple[Fraction, int, int]:
    """The period's pooled bound, from its segments' draws: (q, N, W). A wrong unit escapes a subset:K' draw of n with
    probability at most q = 1 - K'/n, and wrong units in one segment escape together with at most q^w, so with segments
    drawn on independent coins W wrong units escape every draw with probability at most q_max^W. N is the period's
    credited units and W = ceil(eps N): a share eps of the credited work wrong escapes with probability at most q^W."""
    q = max((Fraction(s["draw"]["population"] - s["draw"]["k"], s["draw"]["population"]) for s in segments),
            default=Fraction(1))
    n = sum(s["credited_ands"] for s in segments) // unit_ands()
    return q, n, -(-EPS.numerator * n // EPS.denominator)


def account(reg: str, closes: Sequence[Mapping[str, Any]], segments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Phase 5 over the period's segments: the chain from the registration through every segment's record, each
    segment's composite verdict, the period's (the same rule over the segments), the credited AND gates, the pooled
    bound, and the same scaled to monitoring periods of a week, a month and a year of segments like these."""
    ref = _reference()
    prev, broken = reg, None
    for c, s in zip(closes, segments):
        if c["record"]["prev"] != prev:
            broken = f"segment {c['k']}'s record names {c['record']['prev'][:16]}..., not the previous record"
            break
        prev = s["record"]
    verdict = "reject" if broken else ref.verdict([s["composite"]["verdict"] for s in segments]) if segments else "undecided"
    credited = sum(s["credited_ands"] for s in segments) if verdict == "accept" else 0
    appended = sum(s["ands"] for s in segments)
    q, n, w = pooled(segments)
    delta = q ** w
    scaled = {}
    for name, seconds in PERIODS.items():
        m = seconds // SEGMENT_SECONDS
        n_m = n * m // max(1, len(segments))
        w_m = -(-EPS.numerator * n_m // EPS.denominator)
        scaled[name] = {"segments": m, "credited_ands": n_m * unit_ands(),
                        "log10_delta": f"{w_m * float(np.log10(float(q))):.0f}" if q else None}
    return {"verdict": verdict, "reason": broken or next((f"segment {s['k']}: {s['composite']['reason']}" for s in segments
                                                          if s["composite"]["verdict"] != "accept"), None),
            "head": prev if not broken else None, "segments": len(segments), "appended_ands": appended,
            "credited_ands": credited, "advice_bits_charged": sum(c["record"]["advice_bits"] for c in closes),
            "bound": {"eps": str(EPS), "delta": str(delta), "per_segment_delta": str(DELTA), "q": str(q),
                      "credited_units": n, "wrong_units": w,
                      "statement": f"except with probability {delta} = ({q})^{w}, fewer than {EPS} of the period's "
                                   f"{credited} credited AND gates ({n} units) are in wrong units"},
            "scaled": scaled}


# --- the parties ---------------------------------------------------------------------------------------------------------

DEVELOPER, VERIFIER, CERTIFIER, CHALLENGER = "developer", "verifier", "certifier", "challenger"


def schema(name: str, kind: str) -> Schema:
    """A message: a segment number in its head and canonical JSON in its body."""
    return Schema(name, 1, kind, {"k": Int(0, 1 << 16)}, Body(max=1 << 22))


def msg(name: str, k: int, body: Any = None) -> Message:
    return Message(name, {"k": k}, codec.canonical_json(body if body is not None else {}))


def body(m: Message) -> Any:
    return json.loads(m.body or b"{}")


SHARED = (schema("Register", "commitment"), schema("Close", "commitment"), schema("Draw", "output"),
          schema("Proofs", "structure"), schema("Done", "structure"))


@dataclass
class Segment:
    """The developer's side of one closed segment."""

    close: Close
    circuit: SegmentCircuit
    values: R.Registered
    served: Served
    frames: dict[int, dict[str, str]] = field(default_factory=dict)


class Developer:
    """The developer: registers, serves each segment's requests, closes it, and proves the drawn units behind the
    firewall. Its capabilities: `theta`, `state0`, `hardware`, `product`, `load` (remote: its segments' requests; onsite
    they arrive through the certifier) and `cheat`, the one way it deviates, if any (the tests' cases)."""

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        c = identity.capabilities
        self.theta, self.hardware, self.cheat = c["theta"], c["hardware"], c.get("cheat")
        self.load, self.state = list(c.get("load", ())), c["state0"]
        self.product, self.builder = c["product"], Builder()
        self.onsite = params["protocol"] == "onsite"
        self.segments: list[Segment] = []
        self.pending: list[tuple[list[int], tuple[int, ...]]] = []
        self.held: list[Send] = []
        return [Send("coins", Message("CoinsDraw", {"n": 64}))]

    def on(self, r: Received) -> list[Any]:
        m = r.message
        if r.channel == "coins":
            self.tape = Tape(m.body)
            reg, self.private = register(self.product, self.hardware, self.theta, self.builder, self.state, self.tape)
            j = reg.to_json()
            if self.cheat == "builder":
                j["builder"] = {"digest": digest("lifecycle/builder/v1", {"another": "builder"}), "cap": self.builder.cap}
            self.registration, self.prev = reg, reg.digest()
            out = [Send(VERIFIER, msg("Register", 0, j))]
            if not self.onsite:
                out += self._serve_and_close(self.load[0])
            return out
        if r.channel == "clock":
            return [self.held.pop(0)]
        if m.type in ("Done", "Closed", "Refused"):
            return [Finish({"segments": len(self.segments)})]
        if m.type == "Draw":
            return self._prove(m.fields["k"], body(m)["units"])
        if m.type == "Ingress":
            x = list(body(m)["x"])
            y, _ = forward(self.theta, self._state_after(), x)
            self.pending.append((x, tuple(y)))
            if self.cheat == "io-mismatch" and len(self.segments) == 1:
                y = [s8(y[0] + 1), *y[1:]]
            return [Send(CERTIFIER, msg("Egress", m.fields["k"], {"request": body(m)["request"], "y": y}))]
        if m.type == "Mark":
            xs = [x for x, _ in self.pending]
            self.pending = []
            out = self._serve_and_close(xs)
            self.segments[-1].frames = {int(i): f for i, f in body(m)["frames"].items()}
            return out
        if m.type == "Challenge":
            rows = [opening(self.private.theta, i) for i in body(m)["rows"]]
            if self.cheat == "lost-theta":
                rows[0] = {**rows[0], "row": [(c + 1) & 0xFF for c in rows[0]["row"]]}
            answer_ = Send(CHALLENGER, msg("Answer", m.fields["k"], {"rows": rows}))
            if self.cheat == "off-node":
                self.held.append(answer_)
                return [Timer(r.at + OFF_NODE_NS)]
            return [answer_]
        return []

    def _state_after(self) -> int:
        s = self.state
        for x, _ in self.pending:
            _, s = forward(self.theta, s, x)
        return s

    def _serve_and_close(self, requests: Sequence[Sequence[int]]) -> list[Any]:
        k = len(self.segments) + 1
        if self.cheat == "unreported" and k == 2:
            requests = requests[:-1]
        served = operate(requests, self.theta, self.state, self.hardware, self.tape, k)
        if self.cheat == "placement" and k == 2:
            served = Served((Placement(len(self.hardware.devices), 0), *served.placements[1:]), served.xs, served.ys,
                            served.states, served.wire)
        if self.cheat == "wrong-value" and k == 2:
            ys, states, s = [], [], self.state
            for y in served.ys:
                ys.append((s8(y[0] + 1), *y[1:]))
                s = (s + sum(ys[-1])) % (1 << STATE)
                states.append(s)
            served = Served(served.placements, served.xs, tuple(ys), tuple(states), served.wire)
        circuit = [s.circuit for s in self.segments]
        advice = "".join(s.close.advice for s in self.segments)
        prev = self.registration.digest() if self.cheat == "fork" and k == 2 else self.prev
        c, seg, values = close(k, served, prev, self.builder, circuit, advice, self.state, self.tape)
        if self.cheat == "advice-cap" and k == 2:
            c = Close(k, c.advice + "0" * (self.builder.cap + 1 - len(c.advice)), c.record)
        self.segments.append(Segment(c, seg, values, served))
        self.state, self.prev = (served.states[-1] if served.states else self.state), c.digest
        return [Send(VERIFIER, msg("Close", k, {"advice": c.advice, "record": c.record}))]

    def _prove(self, k: int, units: Sequence[int]) -> list[Any]:
        s = self.segments[k - 1]
        proofs = {}
        for u in units:
            st, witness = self.statement(k, u)
            proofs[str(u)] = prove(st, witness)
        out = [Send(VERIFIER, msg("Proofs", k, proofs))]
        if self.cheat == "recommit" and k == 1:
            rows = [list(r) for r in s.values.rows]
            rows[units[0]][-1] = (rows[units[0]][-1] + 1) % (1 << STATE)
            values = commit(s.circuit.program, f"segment/{k}", rows, Tape(self.tape.seed + b"again"))
            out.append(Send(VERIFIER, msg("Close", k, {"advice": s.close.advice,
                                                       "record": {**s.close.record, "values": committed(values)}})))
        if k < len(self.load) and not self.onsite:
            out += self._serve_and_close(self.load[k])
        return out

    def statement(self, k: int, u: int) -> tuple[Statement, dict[str, Any]]:
        """Unit u of segment k: its statement and the witness the firewall proves it from."""
        s = self.segments[k - 1]
        if u > 0:
            src, pos = s.values, u - 1
        elif k > 1:
            src = self.segments[k - 2].values
            pos = src.leaves - 1
        else:
            src, pos = self.private.state0, 0
        frames = None
        witness = {"unit": opening(s.values, u), "state_in": opening(src, pos),
                   "theta": [opening(self.private.theta, i) for i in range(THETA_ROWS)]}
        if self.onsite and u < s.circuit.real:
            f = s.frames.get(u, {})
            frames = {d: frame_commitment(bytes.fromhex(f[d]), bytes.fromhex(f[f"{d}_salt"])) for d in ("in", "out") if d in f}
            witness["frames"] = f
        return Statement(s.close.digest, u, u >= s.circuit.real, committed(s.values), {**committed(src), "position": pos},
                         committed(self.private.theta), frames), witness


class Verifier:
    """The verifier: checks the registration, each segment's close, draws on its live coins after the commitment, checks
    the drawn units' proofs, runs PoUW's window check, composes each segment's verdict with the components the composite
    lists (onsite: the certifier's view and the memory challenger's report) and accounts for the period. Its outcome is
    every phase's public output."""

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        self.protocol, self.total = params["protocol"], params["segments"]
        self.certifiers = list(params.get("certifiers", ()))
        self.builder = Builder()
        self.registration: Registration | None = None
        self.circuit: list[SegmentCircuit] = []
        self.closes: list[dict[str, Any]] = []
        self.awaiting: list[int] = []
        self.draws: dict[int, list[int]] = {}
        self.parts: dict[int, dict[str, Any]] = {}
        self.lines: list[dict[str, Any]] = []
        self.public: dict[str, Any] = {}
        return []

    def peers(self) -> list[str]:
        return [DEVELOPER] + ([CERTIFIER, CHALLENGER] if self.protocol == "onsite" else [])

    def _finish(self, refused: str | None = None, detail: str | None = None) -> list[Any]:
        if refused is not None:
            self.public["refused"] = {"name": refused, "detail": detail}
        self.public["lines"] = self.lines
        out = [Send(p, msg("Done", 0)) for p in self.peers()]
        return [*out, Record(self.public, "output"), Finish(self.public, refused)]

    def on(self, r: Received) -> list[Any]:
        m = r.message
        try:
            return self._on(r, m)
        except Refusal as e:
            return self._finish(e.name, e.detail)

    def _on(self, r: Received, m: Message) -> list[Any]:
        if m.type in ("Closed", "Refused"):
            return [] if len(self.lines) == self.total else self._finish("closed", f"{r.channel} left early")
        if r.channel == "coins":
            k = self.awaiting.pop(0)
            seg = self.circuit[k - 1]
            self.draws[k] = draw(m.body, seg)
            self.parts[k]["draw"] = {"law": "subset", "k": draw_size(seg), "population": seg.n, "units": self.draws[k]}
            return [Send(DEVELOPER, msg("Draw", k, {"units": self.draws[k]}))]
        k, j = m.fields["k"], body(m)
        if m.type == "Register":
            if self.registration is not None:
                raise Refusal("out-of-phase", "a second registration")
            self.registration = check_registration(j, self.builder, self.certifiers)
            self.public["registration"] = {"digest": self.registration.digest(), **self.registration.to_json()}
            return self.registered()
        if self.registration is None:
            raise Refusal("out-of-phase", f"{m.type} before the registration")
        if m.type == "Close":
            if k in self.parts:
                raise Refusal("recommit", f"segment {k} was committed already, and its coins are drawn after that commitment")
            if k != len(self.circuit) + 1:
                raise Refusal("out-of-phase", f"segment {k} closed out of order")
            c = Close(k, j["advice"], j["record"])
            seg = check_close(c, self.registration, self.builder, self.circuit, "".join(x["advice"] for x in self.closes))
            self.circuit.append(seg)
            self.closes.append({"k": k, "advice": c.advice, "record": c.record, "digest": c.digest})
            self.parts[k] = {}
            self.awaiting.append(k)
            out = [Send("coins", Message("CoinsDraw", {"n": coin_bytes(draw_size(seg), seg.n)}))]
            return out + self.closed(k)
        if m.type == "Proofs":
            if k not in self.draws or "proofs" in self.parts[k]:
                raise Refusal("out-of-phase", f"proofs of segment {k} before its draw, or twice")
            self.parts[k]["proofs"] = j
            return self._advance()
        self.component(m, k, j)
        return self._advance()

    def registered(self) -> list[Any]:
        """What the composite does once the registration is in (onsite: tell its devices)."""
        return []

    def closed(self, k: int) -> list[Any]:
        """What the composite does once segment k is committed (onsite: the memory challenger's round)."""
        return []

    def component(self, m: Message, k: int, j: Any) -> None:
        raise Refusal("out-of-phase", f"{m.type} is no message of the {self.protocol} protocol")

    def ready(self, k: int) -> bool:
        return "proofs" in self.parts[k]

    def answers(self, k: int, checks: Mapping[int, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        """The components' answers for segment k beyond sampled proofs and PoUW (onsite: the certifier, PoUS and the
        bandwidth audit)."""
        return {}

    def _advance(self) -> list[Any]:
        out: list[Any] = []
        while len(self.lines) < len(self.closes) and self.ready(len(self.lines) + 1):
            out += self._decide(len(self.lines) + 1)
        if len(self.lines) == self.total:
            self.public["period"] = account(self.registration.digest(), self.closes, self.lines)
            return out + self._finish()
        return out

    def _decide(self, k: int) -> list[Any]:
        c, seg, proofs = self.closes[k - 1], self.circuit[k - 1], self.parts[k]["proofs"]
        checks = {}
        for u in self.draws[k]:
            checks[u] = verify(self.statement(k, u), proofs.get(str(u), {}))
        sp = sampled_proofs(Close(k, c["advice"], c["record"]), seg, self.draws[k], checks)
        answers = {sp["protocol"]: sp, IDS["pouw"]: pouw_window(sp, seg), **self.answers(k, checks)}
        line = {"k": k, "record": c["digest"], "advice": c["advice"], "advice_bits": len(c["advice"]),
                "circuit": seg.program, "units": seg.n, "real": seg.real, "ands": seg.ands(),
                "credited_ands": seg.credited_ands(), "draw": self.parts[k]["draw"],
                "answers": {i: {"verdict": a["verdict"], "reason": a["reason"]} for i, a in answers.items()},
                "composite": compose(self.protocol, answers)}
        self.lines.append(line)
        return [Record(line, "output")]

    def statement(self, k: int, u: int) -> Statement:
        """Unit u of segment k's statement, from what the verifier holds: the record, the previous record and the
        registration, never the developer's word."""
        rec = self.closes[k - 1]["record"]
        seg = self.circuit[k - 1]
        if u > 0:
            src, pos = rec["values"], u - 1
        elif k > 1:
            src = self.closes[k - 2]["record"]["values"]
            pos = src["leaves"] - 1
        else:
            src, pos = self.registration.state0, 0
        return Statement(self.closes[k - 1]["digest"], u, u >= seg.real, rec["values"], {**src, "position": pos},
                         self.registration.theta, self.frames(k, u))

    def frames(self, k: int, u: int) -> dict[str, str] | None:
        return None


def theta_of(seed: int) -> Theta:
    """A model's weights for the example, from a seed."""
    rng = np.random.default_rng(seed)
    return Theta(tuple(int(v) for v in rng.integers(-128, 128, D * D)), tuple(int(v) for v in rng.integers(-128, 128, D * D)))


def load_of(seed: int, sizes: Sequence[int]) -> list[list[list[int]]]:
    """The example's requests: one list a segment, each request D int8 codes."""
    rng = np.random.default_rng(seed)
    return [[[int(v) for v in rng.integers(-128, 128, D)] for _ in range(n)] for n in sizes]


CHEATS = ("builder", "placement", "unreported", "advice-cap", "recommit", "wrong-value", "io-mismatch", "lost-theta",
          "off-node", "fork")
"""Each way the developer deviates in the tests, by the phase that refuses it: registration (`builder`), operation
(`placement`, `unreported`), close (`advice-cap`, `recommit`), audit (`wrong-value`, and onsite `io-mismatch`,
`lost-theta`, `off-node`) and accounting (`fork`)."""
OFF_NODE_NS = 3_000_000
"""`off-node`: the round trip to where a developer that doesn't hold theta on the node keeps it."""


def summary(outcome: Mapping[str, Any]) -> list[str]:
    """The verifier's public outcome as lines, phase by phase."""
    out = []
    reg = outcome.get("registration")
    if reg:
        devices = ", ".join(f"{d['name']} ({d['sku']}{', ' + d['link'] if 'link' in d else ''})"
                            for d in reg["hardware"]["devices"])
        out += ["1 registration", f"  digest       {reg['digest'][:32]}...", f"  hardware     {devices}"]
        if reg["hardware"]["certifiers"]:
            out.append("  certifiers   " + ", ".join(f"{c['certifier']} on {c['link']}" for c in reg["hardware"]["certifiers"]))
        out += [f"  theta        root {reg['theta']['root'][:32]}... ({reg['theta']['leaves']} hidden rows)",
                f"  builder B    {reg['builder']['digest'][:32]}..., advice cap {reg['builder']['cap']} bits a segment",
                f"  state0       root {reg['state0']['root'][:32]}..."]
    for line in outcome.get("lines", ()):
        d = line["draw"]
        out += [f"segment {line['k']}",
                f"  2 operation  {line['real']} requests served",
                f"  3 close      advice {line['advice']} ({line['advice_bits']} bits charged), {line['units']} units "
                f"({line['units'] - line['real']} filler), circuit {line['circuit'][:16]}..., record {line['record'][:16]}...",
                f"  4 audit      drew subset:{d['k']} of {d['population']}: units {d['units']}"]
        for i, a in line["answers"].items():
            out.append(f"               {i:40s} {a['verdict']}" + (f"  ({a['reason']})" if a["reason"] else ""))
        cm = line["composite"]
        out.append(f"               composite {cm['product']:30s} {cm['verdict']}")
    p = outcome.get("period")
    if p:
        b = p["bound"]
        out += ["5 accounting", f"  period       {p['segments']} segments: {p['verdict']}"
                + (f"  ({p['reason']})" if p["reason"] else ""),
                f"  credited     {p['credited_ands']} of {p['appended_ands']} appended AND gates",
                f"  advice       {p['advice_bits_charged']} bits charged as leakage",
                f"  bound        {b['statement']}; a segment alone: delta {b['per_segment_delta']} at eps {b['eps']}"]
        for name, s in p["scaled"].items():
            d = "0" if s["log10_delta"] is None else f"10^{s['log10_delta']}"
            out.append(f"  scaled {name:6s}{s['segments']} segments of {SEGMENT_SECONDS} s like these: "
                       f"{s['credited_ands']:.2e} credited AND gates, delta {d} at eps {b['eps']}")
    if outcome.get("refused"):
        out.append(f"refused {outcome['refused']['name']}: {outcome['refused']['detail']}")
    return out
