"""The lifecycle both composite protocols step through, on a tiny model, with the parts they share: the phases as functions
with typed inputs and outputs, and the developer and verifier as parties of the deployment layer (`verity.core.deploy`).
`onsite.py` and `remote.py` add what each composite has of its own, through the verifier's hooks; `run.py` runs both;
README.md walks through it.

The phases, each a function here:

1. **Hardware registration** (`check_registration`): the site's `Layout`, one tree of locations (site, hosts, devices by
   card digest, memory leaves by tier and bytes, and the auditor's certifiers and challengers) with its links, folded
   from an append-only log of `Registration` entries. The first entry comes before anything else; each later one is
   checked against the layout its prefix folds to before anything is placed on what it adds.
2. **Registration** (`register`, `check_baseline`): the builder B by its descriptor (its advice cap and its static
   placement plan) and the registered information, theta's and the initial state's hiding commitments, before any coin.
3. **Operation** (`operate`): the developer serves a declaration period's requests through the model (`forward`),
   realizing what only run time decides: the batching.
4. **Declaration k** (`declare`, `check_declaration`): the developer reports the declared advice a_k; B appends the
   declaration block to the declared circuit C = B(a_1 || ... || a_k) (`Builder.grow`: B is prefix-extending, so the
   blocks before are unchanged) and places it (`placement`: its units on device leaves, theta on a memory leaf, naming
   the layout's head); the block's values are committed; and its record binds the previous record and that placement.
5. **Audit of declaration k** (`draw`, `prove`, `verify`, `pouw_window`): live coins after the commitment, sampled
   proofs' draw at PoUW's size K', each drawn unit proved behind the firewall and checked by the verifier.
6. **Accounting** (`compose`, `account`): each declaration's composite verdict by the composition's rule, the monitoring
   period's, its pooled bound, and its credit against the compute the layout holds: each subprotocol's accounting is a
   projection of the layout (`Layout.accounting`).

`prove` and `verify` are a stand-in for Flock behind the firewall: the proof is the drawn unit's opening, which `verify`
checks against the commitments and re-evaluates in the clear. It is sound for the example and not zero-knowledge.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
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
"""The state word's bits: a running sum of every output, carried from unit to unit and from block to block."""
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
"""The proof block: one request, as a Boolean Definition (And/Xor/Not gates)."""


def _block_signature(S: Any) -> tuple[tuple[tuple[str, Any], ...], Any]:
    return (UNIT_PARAMS[:3] + (("x", Array(S.N, Array(D, BYTE))),), Tuple(Array(S.N, Array(D, BYTE)), Array(STATE, BIT)))


def _block_body(B: Any, S: Any, state: Coll, w1: Coll, w2: Coll, x: Coll) -> Coll:
    ys = []
    for i in range(S.N):
        out = B.call(UNIT, state, w1, w2, x[i])
        ys.append(out[0])
        state = out[1]
    return Coll(_block_signature(S)[1], concat([y.refs for y in ys] + [state.refs]))


BLOCK = CompositeDefinition("LifecycleBlock", 1, ("N",), _block_signature, _block_body, register=False,
                            doc="a declaration block of the declared circuit: N units chained through the state wire")


@cache
def unit_ands() -> int:
    return UNIT.traced().counts["and"]


@cache
def block_digest(n: int) -> str:
    """A declaration block's Program digest (`verity.ml.circuits.codec`): its input state wires are the previous block's
    output state wires, and theta is a registered input."""
    return codec.program_digest(codec.encode_program(Program(bind(BLOCK, N=n), registered=("w1", "w2"), theta=("w1", "w2"))))


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
    """A party's private randomness: one seed from its coins source, stretched by SHA-512 under labels."""

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


class Refusal(Exception):
    """A phase's refusal, by name."""

    def __init__(self, name: str, detail: str) -> None:
        super().__init__(f"{name}: {detail}")
        self.name, self.detail = name, detail


# --- 1. hardware registration ------------------------------------------------------------------------------------------

OUTSIDE = "outside"
"""The far end of a boundary link: the rest of the world, a location every layout has."""
TIERS = ("hbm", "dram", "disk")
"""A memory leaf's tier."""
PARENTS = {"site": (None,), "host": ("site",), "device": ("host",), "memory": ("device", "host"), "certifier": ("site",),
           "challenger": ("site",)}
"""Each kind of location and the kinds its parent may be: memory under a device or a host, a device under a host, a host
under the site, and the auditor's devices at the site, placed by their links."""
LINKS = (("host", "host"), ("host", "outside"), ("certifier", "host"), ("certifier", "outside"), ("challenger", "host"))
"""The kinds a link may join, sorted. A certifier sits on an uplink by splitting it: host to certifier, certifier to
`OUTSIDE`."""
AUDITOR_KINDS = ("certifier", "challenger")


@dataclass(frozen=True)
class Card:
    """An SKU's data-sheet card, in `Accounting.Device`'s terms: int8 tensor-core multiply-adds a tick (`rate .tc`), bits
    from device memory to the chip a tick (`bw`), bits of on-chip state (`chip`) and bytes of device memory (`mem`). A
    tick is a nanosecond. A device names its card by digest."""

    sku: str
    tc: int
    bw: int
    chip: int
    mem: int

    def digest(self) -> str:
        return digest("lifecycle/card/v1", asdict(self))


SHEETS = {c.sku: c for c in (Card("sim-int8-a", 64, 3200, 1 << 25, 64 << 30),
                             Card("sim-int8-b", 128, 6400, 1 << 26, 128 << 30))}
"""The auditor's data sheets for the simulated SKUs. A registered card must be one of them: a device's limits are the
auditor's, never the developer's word."""


@dataclass(frozen=True)
class Loc:
    """A location in the site's tree, and who owns it. A device names its card's digest, a memory leaf its tier and its
    bytes, a certifier its signing key."""

    id: str
    kind: str
    parent: str | None = None
    owner: str = "prover"
    card: str = ""
    tier: str = ""
    bytes: int = 0
    key: str = ""


@dataclass(frozen=True)
class Link:
    """A link between two locations (`OUTSIDE` among them), with its bits a tick each way (`β`) and its latency in
    ticks."""

    id: str
    ends: tuple[str, str]
    beta: int
    lat: int


@dataclass(frozen=True)
class Registration:
    """One entry of the hardware registration log: the cards, locations and links it adds, chained to the log's head
    (`previous`). It is atomic: checked as a whole against the layout its prefix folds to."""

    previous: str
    cards: tuple[Card, ...] = ()
    locs: tuple[Loc, ...] = ()
    links: tuple[Link, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {"previous": self.previous, "cards": [asdict(c) for c in self.cards], "locs": [asdict(x) for x in self.locs],
                "links": [{**asdict(x), "ends": list(x.ends)} for x in self.links]}

    @classmethod
    def of(cls, j: Mapping[str, Any]) -> Registration:
        return cls(str(j["previous"]),
                   tuple(Card(str(c["sku"]), int(c["tc"]), int(c["bw"]), int(c["chip"]), int(c["mem"])) for c in j["cards"]),
                   tuple(Loc(str(x["id"]), str(x["kind"]), None if x["parent"] is None else str(x["parent"]),
                             str(x["owner"]), str(x["card"]), str(x["tier"]), int(x["bytes"]), str(x["key"]))
                         for x in j["locs"]),
                   tuple(Link(str(x["id"]), (str(x["ends"][0]), str(x["ends"][1])), int(x["beta"]), int(x["lat"]))
                         for x in j["links"] if len(x["ends"]) == 2))

    def digest(self) -> str:
        return digest("lifecycle/hardware/v1", self.to_json())


@dataclass(frozen=True)
class Layout:
    """The site, as one object: its cards, its tree of locations with their owners, and its links, folded from the
    hardware registration log. `head` is the log's head, the digest a placement names. Every subprotocol's accounting is
    a projection of it (`accounting`)."""

    protocol: str
    head: str
    cards: tuple[Card, ...] = ()
    locs: tuple[Loc, ...] = ()
    links: tuple[Link, ...] = ()

    @classmethod
    def genesis(cls, protocol: str) -> Layout:
        """The empty layout a deployment's log starts from: its head names the composite."""
        return cls(protocol, digest("lifecycle/hardware/genesis", protocol))

    def digest(self) -> str:
        return self.head

    def add(self, entry: Registration) -> Layout:
        """The layout `entry` folds this one to, unchecked (`check_registration` checks it)."""
        return Layout(self.protocol, entry.digest(), self.cards + entry.cards, self.locs + entry.locs,
                      self.links + entry.links)

    def loc(self, i: str) -> Loc | None:
        return next((x for x in self.locs if x.id == i), None)

    def kind(self, i: str) -> str:
        return "outside" if i == OUTSIDE else self.loc(i).kind

    def of_kind(self, kind: str) -> list[Loc]:
        return [x for x in self.locs if x.kind == kind]

    def card(self, device: str) -> Card:
        return next(c for c in self.cards if c.digest() == self.loc(device).card)

    def host(self, i: str) -> str:
        """The host a location is in."""
        x = self.loc(i)
        while x.kind != "host":
            x = self.loc(x.parent)
        return x.id

    def boundary(self) -> list[Link]:
        """The prover's boundary links: a host's link to `OUTSIDE`, or to the certifier that splits its uplink."""
        return [x for x in self.links if tuple(sorted(map(self.kind, x.ends))) in (("host", "outside"), ("certifier", "host"))]

    def certified(self, link: Link) -> bool:
        return "certifier" in map(self.kind, link.ends)

    def device(self, i: str) -> dict[str, int]:
        """`Accounting.Device`'s view of a device leaf: its card's rate, bandwidth and on-chip state, and as its memory the
        bytes of the memory leaves under it."""
        c = self.card(i)
        return {"tc": c.tc, "bw": c.bw, "chip": c.chip,
                "mem": sum(x.bytes for x in self.of_kind("memory") if x.parent == i)}

    def accounting(self, ticks: int) -> dict[str, dict[str, Any]]:
        """Each subprotocol's accounting over `ticks`, projected from the layout: PoUW's compute (the devices' cards'
        multiply-adds), PoUS's memory (the prover's memory leaves, by tier), the bandwidth audit's rates (device memory
        to chip, and the boundary links') and the network certifier's boundary (the links it splits, and the bits the
        rest could carry unseen)."""
        devs = [self.device(x.id) for x in self.of_kind("device")]
        boundary = self.boundary()
        return {"pouw": {"devices": len(devs), "tc": sum(d["tc"] for d in devs) * ticks},
                "pous": {t: sum(x.bytes for x in self.of_kind("memory") if x.tier == t) for t in TIERS},
                "bandwidth": {"device_bits": sum(d["bw"] for d in devs) * ticks,
                              "boundary_bits": sum(x.beta for x in boundary) * ticks},
                "certifier": {"boundary": [x.id for x in boundary],
                              "certified": [x.id for x in boundary if self.certified(x)],
                              "uncertified_bits": sum(x.beta for x in boundary if not self.certified(x)) * ticks}}


def check_registration(layout: Layout, registration: Registration, auditor: Mapping[str, str] | None = None) -> Layout:
    """Phase 1, the verifier's side: one entry of the hardware registration log checked against the layout its prefix
    folds to, and the layout it folds to, or a refusal by name, the first hit winning. `auditor` maps the auditor's own
    devices to their keys.

    - `previous`: the entry isn't chained to the log's head.
    - `duplicate-id`: an id, or a card, is registered already.
    - `unknown-ref`: a parent or a link's end isn't registered.
    - `kind`: the tree is ill-formed (`PARENTS`, `LINKS`), or a certifier doesn't split exactly one uplink.
    - `card-digest`: a card isn't the auditor's data sheet for its SKU, or a device's card isn't registered.
    - `capacity`: a memory leaf holds nothing, or a device's memory leaves exceed its card's memory.
    - `owner`: a certifier or challenger isn't one of the auditor's own devices with its key, or a prover location is
      registered as the auditor's.
    - onsite, `uncertified`: some boundary link has no certifier splitting it.
    - onsite, `challenger-reach`: a host holding the prover's memory has no link to a memory challenger."""
    auditor = auditor or {}
    if registration.previous != layout.head:
        raise Refusal("previous", f"the entry follows {registration.previous[:16]}..., not the log's head "
                                  f"{layout.head[:16]}...")
    m = layout.add(registration)
    ids = [x.id for x in m.locs] + [x.id for x in m.links] + [OUTSIDE]
    cards = [c.digest() for c in m.cards]
    if dup := sorted({i for i in ids if ids.count(i) > 1} | {c.sku for c in m.cards if cards.count(c.digest()) > 1}):
        raise Refusal("duplicate-id", f"{dup} registered twice")
    known = {x.id for x in m.locs} | {OUTSIDE}
    for x in registration.locs:
        if x.parent is not None and x.parent not in known:
            raise Refusal("unknown-ref", f"{x.id}'s parent {x.parent!r} isn't registered")
    for x in registration.links:
        if bad := [e for e in x.ends if e not in known]:
            raise Refusal("unknown-ref", f"link {x.id} names {bad}, which aren't registered")
    for x in registration.locs:
        if (x.kind not in PARENTS or (None if x.parent is None else m.kind(x.parent)) not in PARENTS[x.kind]
                or (x.kind == "memory") != (x.tier in TIERS)):
            raise Refusal("kind", f"{x.kind} {x.id} can't sit under {x.parent!r}")
    for x in registration.links:
        if x.ends[0] == x.ends[1] or tuple(sorted(map(m.kind, x.ends))) not in LINKS or x.beta <= 0 or x.lat < 0:
            raise Refusal("kind", f"link {x.id} can't join {list(x.ends)}")
    for c in m.of_kind("certifier"):
        far = sorted(m.kind(e) for x in m.links if c.id in x.ends for e in x.ends if e != c.id)
        if far != ["host", "outside"]:
            raise Refusal("kind", f"certifier {c.id} doesn't split one uplink: it links to {far}")
    for c in registration.cards:
        if SHEETS.get(c.sku) != c:
            raise Refusal("card-digest", f"card {c.sku} ({c.digest()[:16]}...) isn't the auditor's data sheet for it")
    for x in registration.locs:
        if x.kind == "device" and x.card not in cards:
            raise Refusal("card-digest", f"{x.id}'s card {x.card[:16]}... isn't registered")
    for x in registration.locs:
        if x.kind == "memory" and x.bytes <= 0:
            raise Refusal("capacity", f"memory {x.id} holds nothing")
    for d in m.of_kind("device"):
        if (held := m.device(d.id)["mem"]) > m.card(d.id).mem:
            raise Refusal("capacity", f"{d.id}'s memory leaves hold {held} bytes, over its card's {m.card(d.id).mem}")
    for x in registration.locs:
        if x.kind in AUDITOR_KINDS and (x.owner != "auditor" or auditor.get(x.id) != x.key):
            raise Refusal("owner", f"{x.kind} {x.id} isn't one of the auditor's own devices")
        if x.kind not in AUDITOR_KINDS and x.owner != "prover":
            raise Refusal("owner", f"{x.kind} {x.id} is a prover's location, registered as the {x.owner}'s")
    if m.protocol == "onsite":
        if bare := [x.id for x in m.boundary() if not m.certified(x)]:
            raise Refusal("uncertified", f"boundary links {bare} have no certifier splitting them")
        reached = {e for x in m.links if "challenger" in map(m.kind, x.ends) for e in x.ends}
        if far := sorted({m.host(x.id) for x in m.of_kind("memory")} - reached):
            raise Refusal("challenger-reach", f"hosts {far} hold the prover's memory, and no memory challenger links "
                                              "to them")
    return m


CARD_A, CARD_B = SHEETS["sim-int8-a"], SHEETS["sim-int8-b"]
NODE = Registration("", (CARD_A, CARD_B), (
    Loc("site0", "site"), Loc("node0", "host", "site0"),
    Loc("gpu0", "device", "node0", card=CARD_A.digest()), Loc("hbm0", "memory", "gpu0", tier="hbm", bytes=64 << 30),
    Loc("gpu1", "device", "node0", card=CARD_B.digest()), Loc("hbm1", "memory", "gpu1", tier="hbm", bytes=128 << 30),
    Loc("dram0", "memory", "node0", tier="dram", bytes=512 << 30),
    Loc("nvme0", "memory", "node0", tier="disk", bytes=1 << 40)))
"""What both composites' sites register first: one node with two devices, each with its HBM, and the host's DRAM and
disk. Each composite adds its links and the auditor's devices (`remote.LAYOUT`, `onsite.LAYOUT`)."""
LATER = Registration("", locs=(
    Loc("gpu2", "device", "node0", card=CARD_B.digest()), Loc("hbm2", "memory", "gpu2", tier="hbm", bytes=128 << 30),
    Loc("nvme1", "memory", "node0", tier="disk", bytes=1 << 40)))
"""The hardware a site adds after its first declaration: a third device and a disk."""

# --- 2. registration ---------------------------------------------------------------------------------------------------

BATCH = 2
"""Requests a batch holds at most."""


@dataclass(frozen=True)
class Placement:
    device: str
    batch: int


@dataclass(frozen=True)
class Block:
    """Declaration k's block of the declared circuit: its real units' placements, its units padded to PoUW's grid (`n`,
    fillers after the real units), its Program's digest and its declared advice's length."""

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


@dataclass(frozen=True)
class Builder:
    """B, the registered builder: declared advice to the declared circuit. A declaration's advice is self-delimiting (its
    unit count, then each unit's batch), so B(a_1 || ... || a_k) parses declaration by declaration and extends
    B(a_1 || ... || a_{k-1}): B is prefix-extending. Placement is B's and static, not the advice's: from declaration k on,
    `plan` names the device leaves B places units on, unit j on the (j mod their number)-th, and `store` the memory
    leaf theta is held in. `cap` is the public cap on one declaration's advice, in bits."""

    cap: int = 32
    plan: tuple[tuple[int, tuple[str, ...]], ...] = ((1, ("gpu0", "gpu1")),)
    store: str = "hbm0"
    COUNT_BITS = 8
    BATCH_BITS = 4
    FORMAT = "lifecycle/builder/v1"

    def descriptor(self) -> dict[str, Any]:
        return {"format": self.FORMAT, "unit": UNIT.name, "block": BLOCK.name, "cap": self.cap,
                "advice": {"count": self.COUNT_BITS, "batch": self.BATCH_BITS, "batch_max": BATCH},
                "plan": [[k, list(ds)] for k, ds in self.plan], "store": self.store,
                "grid": pouw_window_rules.GRID_BITS}

    @classmethod
    def of(cls, j: Any) -> Builder:
        """The verifier's B from a registered descriptor: one of this builder's format, or a refusal (`builder`)."""
        try:
            b = cls(int(j["cap"]), tuple((int(k), tuple(str(d) for d in ds)) for k, ds in j["plan"]), str(j["store"]))
        except (KeyError, TypeError, ValueError) as e:
            raise Refusal("builder", f"the registered builder's descriptor doesn't parse: {type(e).__name__}") from None
        starts = [k for k, _ in b.plan]
        if b.descriptor() != j or starts[:1] != [1] or starts != sorted(set(starts)) or not all(ds for _, ds in b.plan):
            raise Refusal("builder", "the registered builder is not one the verifier runs")
        return b

    def digest(self) -> str:
        return digest("lifecycle/builder/v1", self.descriptor())

    def devices(self, k: int) -> tuple[str, ...]:
        return next(ds for start, ds in reversed(self.plan) if start <= k)

    def encode(self, batches: Sequence[int]) -> str:
        return format(len(batches), f"0{self.COUNT_BITS}b") + "".join(format(b, f"0{self.BATCH_BITS}b") for b in batches)

    def declarations(self, advice: str) -> list[str]:
        """The advice cut into its declarations' advice, or ValueError."""
        out, i = [], 0
        if not set(advice) <= {"0", "1"}:
            raise ValueError("advice is a bit string")
        while i < len(advice):
            if i + self.COUNT_BITS > len(advice):
                raise ValueError("advice ends inside a declaration's count")
            end = i + self.COUNT_BITS + int(advice[i:i + self.COUNT_BITS], 2) * self.BATCH_BITS
            if end > len(advice):
                raise ValueError("advice ends inside a declaration")
            out.append(advice[i:end])
            i = end
        return out

    def build(self, advice: str) -> list[Block]:
        """B(advice): every declaration block, from the concatenated declared advice."""
        out = []
        for k, a in enumerate(self.declarations(advice), 1):
            batches = [int(a[j:j + self.BATCH_BITS], 2) for j in range(self.COUNT_BITS, len(a), self.BATCH_BITS)]
            if not batches:
                raise ValueError(f"declaration {k} declares nothing")
            if (batches[0] != 0 or any(b - p not in (0, 1) for p, b in zip(batches, batches[1:]))
                    or max(batches.count(b) for b in batches) > BATCH):
                raise ValueError(f"declaration {k}'s batches are not consecutive batches of at most {BATCH}")
            devs = self.devices(k)
            units = tuple(Placement(devs[j % len(devs)], b) for j, b in enumerate(batches))
            work = pouw_window_rules.bucket(len(units))
            out.append(Block(k, units, work.n, block_digest(work.n), len(a)))
        return out

    def grow(self, circuit: Sequence[Block], advice: str, a_k: str) -> Block:
        """Declaration k's block: B(advice || a_k)'s last, which must extend the declared circuit so far."""
        if len(a_k) > self.cap:
            raise Refusal("advice-cap", f"declaration {len(circuit) + 1}'s advice is {len(a_k)} bits, over the cap of "
                                        f"{self.cap}")
        try:
            grown = self.build(advice + a_k)
        except ValueError as e:
            raise Refusal("advice", str(e)) from None
        if len(grown) != len(circuit) + 1 or grown[:-1] != list(circuit):
            raise Refusal("prefix", "B(a_1 || ... || a_k) does not extend the declared circuit so far")
        return grown[-1]


@dataclass(frozen=True)
class Baseline:
    """The registration: the product, B's descriptor and the registered information, theta's and the initial state's hiding
    commitments: public digests and roots, no value. With the declaration blocks it is Lean's `Declaration` (d.D, d.θ)
    so far: B's blocks are d.D, and the gates theta and the initial state are committed to are d.θ."""

    product: str
    builder: dict[str, Any]
    theta: dict[str, Any]
    state0: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"product": self.product, "builder": self.builder, "theta": self.theta, "state0": self.state0}

    def digest(self) -> str:
        return digest("lifecycle/registration/v1", self.to_json())


@dataclass(frozen=True)
class Private:
    """The developer's side of the registration: theta's and the initial state's rows and salts."""

    theta: R.Registered
    state0: R.Registered


def register(product: str, theta: Theta, builder: Builder, state0: int, tape: Tape) -> tuple[Baseline, Private]:
    """Phase 2, the developer's side."""
    b = builder.digest()
    t = commit(b, "theta", theta.rows(), tape)
    s = commit(b, "state0", [[state0]], tape)
    return Baseline(product, builder.descriptor(), committed(t), committed(s)), Private(t, s)


def check_baseline(j: Mapping[str, Any]) -> tuple[Baseline, Builder]:
    """Phase 2, the verifier's side: B is a builder the verifier runs (`builder`), and theta and the initial state are
    committed under it, in their shapes (`registration`)."""
    builder = Builder.of(j.get("builder"))
    for name, shape in (("theta", (D, THETA_ROWS)), ("state0", (1, 1))):
        w = j[name]
        if (w["program"], w["value"], w["words"], w["leaves"]) != (builder.digest(), name, *shape):
            raise Refusal("registration", f"{name} is not committed under the registered builder")
    return Baseline(str(j["product"]), dict(j["builder"]), dict(j["theta"]), dict(j["state0"])), builder


# --- 3. operation ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Served:
    """One declaration period as served: each request's input, output, the state after it and its batch."""

    xs: tuple[tuple[int, ...], ...]
    ys: tuple[tuple[int, ...], ...]
    states: tuple[int, ...]
    batches: tuple[int, ...]


def operate(requests: Sequence[Sequence[int]], theta: Theta, state: int, tape: Tape, k: int) -> Served:
    """Phase 3: serve the declaration period's requests in arrival order, batched as they come: a batch closes when it is
    full or when the developer's tape says the next request came late."""
    late = tape.bytes(f"batching/{k}", len(requests))
    ys, states, batches, b, size = [], [], [], 0, 0
    for i, x in enumerate(requests):
        if i and (size == BATCH or late[i] & 1):
            b, size = b + 1, 0
        y, state = forward(theta, state, x)
        ys.append(tuple(y))
        states.append(state)
        batches.append(b)
        size += 1
    return Served(tuple(tuple(x) for x in requests), tuple(ys), tuple(states), tuple(batches))


# --- 4. declaration k --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Declaration:
    """What declaration k posts: its declared advice and its record, whose digest the audit commits to."""

    k: int
    advice: str
    record: dict[str, Any]

    @property
    def digest(self) -> str:
        return digest("lifecycle/declaration/v1", self.record)


def rows_of(served: Served, n: int, state: int) -> list[list[int]]:
    """The block's rows: each real unit's, then each filler's (input zero, so output zero and the state unchanged)."""
    rows = [[*(c & 0xFF for c in x), *(c & 0xFF for c in y), s] for x, y, s in zip(served.xs, served.ys, served.states)]
    last = served.states[-1] if served.states else state
    return rows + [[0] * (2 * D) + [last]] * (n - len(rows))


THETA_BYTES = THETA_ROWS * D
"""Theta's bytes in the memory leaf B holds it in."""


def placement(builder: Builder, blk: Block, head: str) -> dict[str, Any]:
    """B's placement of a declaration block (`Requirement.Placement`): each unit's device leaf and theta's memory leaf,
    naming the layout they are placed on by its head."""
    return {"layout": head, "units": [p.device for p in blk.units], "theta": builder.store}


def check_placement(layout: Layout, p: Mapping[str, Any]) -> None:
    """A placement against the layout it names, or a refusal by name: `placement-off-layout` (it names another layout than
    the log's head, or a leaf the layout doesn't hold), `placement-auditor` (a leaf is the auditor's), `placement-kind`
    (a unit off a device leaf, theta off a memory leaf) and `placement-capacity` (theta over its memory leaf's bytes)."""
    if p["layout"] != layout.head:
        raise Refusal("placement-off-layout", f"the placement names layout {p['layout'][:16]}..., not the log's head")
    leaves = [*p["units"], p["theta"]]
    if off := sorted({i for i in leaves if layout.loc(i) is None}):
        raise Refusal("placement-off-layout", f"B places on {off}, which no hardware registration holds")
    if aud := sorted({i for i in leaves if layout.loc(i).owner != "prover"}):
        raise Refusal("placement-auditor", f"B places on {aud}, the auditor's")
    if bad := sorted({i for i in p["units"] if layout.kind(i) != "device"} | {p["theta"]} - {
            x.id for x in layout.of_kind("memory")}):
        raise Refusal("placement-kind", f"B places units off a device, or theta off a memory leaf: {bad}")
    if THETA_BYTES > layout.loc(p["theta"]).bytes:
        raise Refusal("placement-capacity", f"theta's {THETA_BYTES} bytes don't fit in {p['theta']}")


def declare(k: int, served: Served, prev: str, layout: str, builder: Builder, circuit: Sequence[Block], advice: str,
            state: int, tape: Tape) -> tuple[Declaration, Block, R.Registered]:
    """Phase 4, the developer's side: the declared advice, B's block and its placement on the layout so far (`layout`, its
    head), its values committed and the record."""
    a_k = builder.encode(served.batches)
    blk = builder.grow(circuit, advice, a_k)
    values = commit(blk.program, f"block/{k}", rows_of(served, blk.n, state), tape)
    record = {"declaration": k, "prev": prev, "placement": placement(builder, blk, layout), "advice_bits": len(a_k),
              "circuit": blk.program, "units": blk.n, "values": committed(values)}
    return Declaration(k, a_k, record), blk, values


def check_declaration(d: Declaration, layout: Layout, builder: Builder, circuit: Sequence[Block], advice: str) -> Block:
    """Phase 4, the verifier's side: the advice within the cap, B's block grown from it, its placement on the layout as
    registered now (`check_placement`), and the record naming that block, its placement and the values (`record`)."""
    blk = builder.grow(circuit, advice, d.advice)
    p = placement(builder, blk, layout.head)
    check_placement(layout, p)
    v = d.record.get("values", {})
    want = {"declaration": d.k, "placement": p, "advice_bits": len(d.advice), "circuit": blk.program, "units": blk.n}
    if ({k: d.record.get(k) for k in want} != want or set(d.record) != {*want, "prev", "values"}
            or (v.get("program"), v.get("value"), v.get("words"), v.get("leaves"))
            != (blk.program, f"block/{d.k}", ROW_WORDS, blk.n)):
        raise Refusal("record", f"declaration {d.k}'s record is not B's block, its placement and its values")
    return blk


# --- 5. audit of declaration k -----------------------------------------------------------------------------------------

EPS, DELTA = Fraction(1, 2), Fraction(1, 4)
"""What one declaration's audit establishes: except with probability δ, fewer than a share ε of its credited work is
wrong."""


def draw_size(blk: Block) -> int:
    """PoUW's K' for the block (`pouw.window.k_prime`): subset:K' over its n units makes a wrong share ε of the credited
    work missed with probability at most δ."""
    work = pouw_window_rules.bucket(blk.real)
    return pouw_window_rules.k_prime(EPS, DELTA, work.credited * MACS, work.n, MACS)


def coin_bytes(k: int, n: int) -> int:
    """Enough coins for `draw_first`'s subset:k of n, four times the stream it reads first."""
    return 4 * (2 * k * ((n.bit_length() + 64 + 7) // 8) + 64)


def draw(coins: bytes, blk: Block) -> list[int]:
    """Sampled proofs' draw (`sampled_proofs.coins.draw_first`) on the verifier's live coins."""
    buf = bytearray(coins)

    def read(n: int) -> bytes:
        out = bytes(buf[:n])
        del buf[:n]
        return out

    return draw_first(OwnRandomness(read), {"law": "subset", "k": draw_size(blk)}, blk.n)["units"]


@dataclass(frozen=True)
class Statement:
    """A drawn unit's public statement: the declaration it is in, its index, whether it is a filler, where its values,
    its input state and theta are committed, and onsite the certifier's commitments to its frames."""

    declaration: str
    unit: int
    filler: bool
    values: dict[str, Any]
    state_in: dict[str, Any]
    theta: dict[str, Any]
    frames: dict[str, str] | None = None


def prove(statement: Statement, witness: Mapping[str, Any]) -> dict[str, Any]:
    """STAND-IN for Flock behind the firewall: the proof is the unit's opening."""
    return dict(witness)


def verify(statement: Statement, proof: Mapping[str, Any]) -> dict[str, str | None]:
    """STAND-IN for Flock's verifier: each opening against its commitment, then the unit's circuit evaluated in the
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


def sampled_proofs(d: Declaration, blk: Block, units: Sequence[int], checks: Mapping[int, Mapping[str, Any]]
                   ) -> dict[str, Any]:
    """Sampled proofs' answer for the declaration: every drawn unit's values check held."""
    bad = [f"unit {u}: {checks[u]['values']}" for u in units if u not in checks or checks[u]["values"]]
    law = {"law": "subset", "k": len(units)}
    return answer(IDS["sampled-proofs"], "reject" if bad else "accept", d.digest, "; ".join(bad) or None,
                  {"draw": {**law, "population": blk.n, "units": list(units)}},
                  {"law": law, "population": blk.n, "delta": str(DELTA)})


def pouw_window(sp: Mapping[str, Any], blk: Block) -> dict[str, Any]:
    """PoUW's window check over sampled proofs' answer: sampled proofs accepted the declaration's registration, at
    subset:K' over the grid's n units; it publishes the credited work at ε and δ."""
    work = pouw_window_rules.bucket(blk.real)
    if sp["verdict"] != "accept":
        return answer(IDS["pouw"], sp["verdict"], sp["commitment"], "sampled proofs did not accept the window")
    d = sp["public"]["draw"]
    if (d["law"], d["k"], d["population"]) != ("subset", draw_size(blk), work.n):
        return answer(IDS["pouw"], "reject", sp["commitment"], "the draw is not subset:K' over the window's grid")
    return answer(IDS["pouw"], "accept", sp["commitment"], public={"work": work.credited * MACS, "tiles": work.credited,
                                                                   "eps": str(EPS), "delta": str(DELTA)})


# --- 6. accounting -----------------------------------------------------------------------------------------------------

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
    """Each composite's product id and its subprotocols' ids, in the order of their ids (`Core.Onsite`, `Core.Remote`)."""
    ref = _reference()
    return {"onsite": (ref.ONSITE.id, tuple(c.id for c in ref.ONSITE.components)),
            "remote": (ref.REMOTE.id, tuple(c.id for c in ref.REMOTE.components))}


def compose(protocol: str, answers: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """A declaration's composite verdict: the composition's rule over exactly the product's subprotocols (any reject
    rejects, then any undecided leaves it undecided), and its commitment the binding of their digests
    (`Core.Composition.bind`)."""
    ref = _reference()
    pid, ids = products()[protocol]
    missing = [i for i in ids if i not in answers]
    verdicts = [answers[i]["verdict"] if i in answers else "undecided" for i in ids]
    rule = ref.verdict(verdicts)
    digests = [(i, bytes.fromhex(answers[i]["commitment"])) for i in ids if i in answers and answers[i]["commitment"]]
    bound = ref.bind(pid, digests).hex() if len(digests) == len(ids) else None
    reason = " | ".join(f"{i}: {answers[i]['reason']}" for i in ids if i in answers and answers[i]["verdict"] == "reject"
                        ) or (f"no answer from {missing}" if missing else None)
    return {"product": pid, "verdict": rule, "commitment": bound, "reason": None if rule == "accept" else reason,
            "components": {i: v for i, v in zip(ids, verdicts)}}


DECLARATION_SECONDS = 60
"""A stated parameter: a declaration period's length in a deployment, which scales a monitoring period's declarations and
the layout's capacity in each."""
TICKS_PER_SECOND = 10 ** 9
PERIODS = {"week": 7 * 24 * 3600, "month": 30 * 24 * 3600, "year": 365 * 24 * 3600}


def pooled(lines: Sequence[Mapping[str, Any]]) -> tuple[Fraction, int, int]:
    """The period's pooled bound, from its declarations' draws: (q, N, W). A wrong unit escapes a subset:K' draw of n with
    probability at most q = 1 - K'/n, and w wrong units in one block escape together with at most q^w, so on independent
    coins W wrong units escape every draw with probability at most q_max^W. N is the period's credited units and
    W = ceil(eps N): a share eps of the credited work wrong escapes with probability at most q_max^W."""
    q = max((Fraction(x["draw"]["population"] - x["draw"]["k"], x["draw"]["population"]) for x in lines),
            default=Fraction(1))
    n = sum(x["credited_ands"] for x in lines) // unit_ands()
    return q, n, -(-EPS.numerator * n // EPS.denominator)


def _share(num: int, den: int) -> str:
    return f"{num / den:.2e}" if den else "-"


def account(baseline: str, declarations: Sequence[Mapping[str, Any]], lines: Sequence[Mapping[str, Any]], layout: Layout
            ) -> dict[str, Any]:
    """Phase 6 over the monitoring period's declarations: the chain from the registration through every declaration's
    record, each declaration's composite verdict, the period's (the same rule over the declarations), the credited AND
    gates and multiply-adds against the compute the layout registered in each declaration period held (its PoUW
    projection), the pooled bound, each subprotocol's projection of `layout` over a declaration period, and the same
    scaled to monitoring periods of a week, a month and a year of declarations like these on `layout`."""
    ref = _reference()
    prev, broken = baseline, None
    for d, x in zip(declarations, lines):
        if d["record"]["prev"] != prev:
            broken = f"declaration {d['k']}'s record names {d['record']['prev'][:16]}..., not the previous record"
            break
        prev = x["record"]
    verdict = "reject" if broken else ref.verdict([x["composite"]["verdict"] for x in lines]) if lines else "undecided"
    credited = sum(x["credited_ands"] for x in lines) if verdict == "accept" else 0
    macs = credited // unit_ands() * MACS
    capacity = sum(x["capacity_tc"] for x in lines)
    q, n, w = pooled(lines)
    delta = q ** w
    scaled = {}
    for name, seconds in PERIODS.items():
        m = seconds // DECLARATION_SECONDS
        n_m = n * m // max(1, len(lines))
        w_m = -(-EPS.numerator * n_m // EPS.denominator)
        cap_m = layout.accounting(seconds * TICKS_PER_SECOND)
        scaled[name] = {"declarations": m, "credited_ands": n_m * unit_ands(), "capacity_tc": cap_m["pouw"]["tc"],
                        "share": _share(n_m * MACS, cap_m["pouw"]["tc"]),
                        "boundary_bits": cap_m["bandwidth"]["boundary_bits"],
                        "log10_delta": f"{w_m * float(np.log10(float(q))):.0f}" if q else None}
    return {"verdict": verdict, "reason": broken or next((f"declaration {x['k']}: {x['composite']['reason']}" for x in lines
                                                          if x["composite"]["verdict"] != "accept"), None),
            "head": prev if not broken else None, "declarations": len(lines),
            "appended_ands": sum(x["ands"] for x in lines), "credited_ands": credited, "credited_macs": macs,
            "capacity_tc": capacity, "share": _share(macs, capacity),
            "advice_bits_charged": sum(d["record"]["advice_bits"] for d in declarations),
            "bound": {"eps": str(EPS), "delta": str(delta), "per_declaration_delta": str(DELTA), "q": str(q),
                      "credited_units": n, "wrong_units": w,
                      "statement": f"except with probability {delta} = ({q})^{w}, fewer than {EPS} of the period's "
                                   f"{credited} credited AND gates ({n} units) are in wrong units"},
            "accounting": layout.accounting(DECLARATION_SECONDS * TICKS_PER_SECOND), "scaled": scaled}


# --- the parties ---------------------------------------------------------------------------------------------------------

DEVELOPER, VERIFIER, CERTIFIER, CHALLENGER = "developer", "verifier", "certifier", "challenger"


def schema(name: str, kind: str) -> Schema:
    """A message: a declaration's number in its head and canonical JSON in its body."""
    return Schema(name, 1, kind, {"k": Int(0, 1 << 16)}, Body(max=1 << 22))


def msg(name: str, k: int, body: Any = None) -> Message:
    return Message(name, {"k": k}, codec.canonical_json(body if body is not None else {}))


def body(m: Message) -> Any:
    return json.loads(m.body or b"{}")


SHARED = (schema("Hardware", "structure"), schema("Register", "commitment"), schema("Declare", "commitment"),
          schema("Draw", "output"), schema("Proofs", "private"), schema("Done", "structure"))
"""The messages both composites use. `Proofs` is private only because the stand-in's proof is the opening."""


@dataclass
class Declared:
    """The developer's side of one declaration."""

    declaration: Declaration
    block: Block
    values: R.Registered
    served: Served
    frames: dict[int, dict[str, str]] = field(default_factory=dict)


class Developer:
    """The developer: registers its hardware and its registered information, serves each declaration period's
    requests, declares, and proves the drawn proof blocks behind the firewall. Its capabilities: `theta`, `state0`, `layout` (its first hardware
    registration entry), `later` (the one it adds after its first declaration), `plan` (B's placement), `product`, `load`
    (remote: its declaration periods' requests; onsite they arrive through the certifier) and `cheat`, the one way it
    deviates, if any (`CHEATS`)."""

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        c = identity.capabilities
        self.theta, self.cheat, self.product = c["theta"], c.get("cheat"), c["product"]
        self.load, self.state = list(c.get("load", ())), c["state0"]
        first, later, plan = c["layout"], c["later"], c["plan"]
        new = next(x.id for x in later.locs if x.kind == "device")
        if self.cheat == "placement":
            plan = ((1, (*plan[0][1][:-1], new)), *plan[1:])
        if self.cheat == "hardware":
            old = next(x.id for x in first.locs if x.kind == "device")
            later = replace(later, locs=tuple(replace(x, id=old) if x.id == new else x for x in later.locs))
        self.builder = Builder(plan=plan)
        self.onsite = params["protocol"] == "onsite"
        genesis = Layout.genesis(params["protocol"])
        self.first = replace(first, previous=genesis.head)
        self.layout = genesis.add(self.first)
        self.later = replace(later, previous=self.layout.head)
        self.blocks: list[Declared] = []
        self.pending: list[list[int]] = []
        self.held: list[Send] = []
        return [Send("coins", Message("CoinsDraw", {"n": 64}))]

    def on(self, r: Received) -> list[Any]:
        m = r.message
        if r.channel == "coins":
            self.tape = Tape(m.body)
            base, self.private = register(self.product, self.theta, self.builder, self.state, self.tape)
            j = base.to_json()
            if self.cheat == "builder":
                j["builder"] = {**j["builder"], "format": "another/builder/v1"}
            self.baseline, self.prev = base, base.digest()
            out = [Send(VERIFIER, msg("Hardware", 0, self.first.to_json())), Send(VERIFIER, msg("Register", 0, j))]
            if not self.onsite:
                out += self._serve_and_declare(self.load[0])
            return out
        if r.channel == "clock":
            return [self.held.pop(0)]
        if m.type in ("Done", "Closed", "Refused"):
            return [Finish({"declarations": len(self.blocks)})]
        if m.type == "Draw":
            return self._prove(m.fields["k"], body(m)["units"])
        if m.type == "Ingress":
            x = list(body(m)["x"])
            y, _ = forward(self.theta, self._state_after(), x)
            self.pending.append(x)
            if self.cheat == "io-mismatch" and len(self.blocks) == 1:
                y = [s8(y[0] + 1), *y[1:]]
            return [Send(CERTIFIER, msg("Egress", m.fields["k"], {"request": body(m)["request"], "y": y}))]
        if m.type == "Mark":
            xs, self.pending = self.pending, []
            out = self._serve_and_declare(xs)
            self.blocks[-1].frames = {int(i): f for i, f in body(m)["frames"].items()}
            return out
        if m.type == "Challenge":
            rows = [opening(self.private.theta, i) for i in body(m)["rows"]]
            if self.cheat == "lost-theta":
                rows[0] = {**rows[0], "row": [(c + 1) & 0xFF for c in rows[0]["row"]]}
            reply = Send(CHALLENGER, msg("Answer", m.fields["k"], {"rows": rows}))
            if self.cheat == "off-node":
                self.held.append(reply)
                return [Timer(r.at + OFF_NODE_TICKS)]
            return [reply]
        return []

    def _state_after(self) -> int:
        s = self.state
        for x in self.pending:
            _, s = forward(self.theta, s, x)
        return s

    def current(self) -> Layout:
        """The layout as registered so far: the first entry, and the later one once it is sent."""
        return self.layout.add(self.later) if self.blocks else self.layout

    def _serve_and_declare(self, requests: Sequence[Sequence[int]]) -> list[Any]:
        k = len(self.blocks) + 1
        if self.cheat == "unreported" and k == 2:
            requests = requests[:-1]
        served = operate(requests, self.theta, self.state, self.tape, k)
        if self.cheat == "wrong-value" and k == 2:
            ys, states, s = [], [], self.state
            for y in served.ys:
                ys.append((s8(y[0] + 1), *y[1:]))
                s = (s + sum(ys[-1])) % (1 << STATE)
                states.append(s)
            served = Served(served.xs, tuple(ys), tuple(states), served.batches)
        circuit = [b.block for b in self.blocks]
        advice = "".join(b.declaration.advice for b in self.blocks)
        prev = self.baseline.digest() if self.cheat == "fork" and k == 2 else self.prev
        d, blk, values = declare(k, served, prev, self.current().digest(), self.builder, circuit, advice, self.state,
                                 self.tape)
        if self.cheat == "advice-cap" and k == 2:
            d = Declaration(k, d.advice + "0" * (self.builder.cap + 1 - len(d.advice)), d.record)
        self.blocks.append(Declared(d, blk, values, served))
        self.state, self.prev = (served.states[-1] if served.states else self.state), d.digest
        out = [Send(VERIFIER, msg("Declare", k, {"advice": d.advice, "record": d.record}))]
        if k == 1:
            out.append(Send(VERIFIER, msg("Hardware", 1, self.later.to_json())))
        return out

    def _prove(self, k: int, units: Sequence[int]) -> list[Any]:
        b = self.blocks[k - 1]
        proofs = {}
        for u in units:
            st, witness = self.statement(k, u)
            proofs[str(u)] = prove(st, witness)
        out = [Send(VERIFIER, msg("Proofs", k, proofs))]
        if self.cheat == "recommit" and k == 1:
            rows = [list(r) for r in b.values.rows]
            rows[units[0]][-1] = (rows[units[0]][-1] + 1) % (1 << STATE)
            values = commit(b.block.program, f"block/{k}", rows, Tape(self.tape.seed + b"again"))
            out.append(Send(VERIFIER, msg("Declare", k, {"advice": b.declaration.advice,
                                                         "record": {**b.declaration.record, "values": committed(values)}})))
        if k < len(self.load) and not self.onsite:
            out += self._serve_and_declare(self.load[k])
        return out

    def statement(self, k: int, u: int) -> tuple[Statement, dict[str, Any]]:
        """Unit u of declaration k: its statement and the witness the firewall proves it from."""
        b = self.blocks[k - 1]
        if u > 0:
            src, pos = b.values, u - 1
        elif k > 1:
            src = self.blocks[k - 2].values
            pos = src.leaves - 1
        else:
            src, pos = self.private.state0, 0
        frames = None
        witness = {"unit": opening(b.values, u), "state_in": opening(src, pos),
                   "theta": [opening(self.private.theta, i) for i in range(THETA_ROWS)]}
        if self.onsite and u < b.block.real:
            f = b.frames.get(u, {})
            frames = {w: frame_commitment(bytes.fromhex(f[w]), bytes.fromhex(f[f"{w}_salt"])) for w in ("in", "out") if w in f}
            witness["frames"] = f
        return Statement(b.declaration.digest, u, u >= b.block.real, committed(b.values),
                         {**committed(src), "position": pos}, committed(self.private.theta), frames), witness


class Verifier:
    """The verifier: checks each hardware registration, the registration, and each declaration; draws on its live coins
    after the commitment; checks the drawn proof blocks' proofs; runs PoUW's window check; composes each declaration's verdict
    with the subprotocols the composite lists (onsite: network certification and the memory challenger's report); and
    accounts for the monitoring period. Its outcome is every phase's public output."""

    def start(self, params: Mapping[str, Any], identity: Identity) -> list[Any]:
        self.protocol, self.total = params["protocol"], params["declarations"]
        self.auditor = dict(params.get("auditor", {}))
        self.layout = Layout.genesis(self.protocol)
        self.baseline: Baseline | None = None
        self.builder: Builder | None = None
        self.circuit: list[Block] = []
        self.declarations: list[dict[str, Any]] = []
        self.awaiting: list[int] = []
        self.draws: dict[int, list[int]] = {}
        self.parts: dict[int, dict[str, Any]] = {}
        self.lines: list[dict[str, Any]] = []
        self.public: dict[str, Any] = {"hardware": []}
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
        try:
            return self._on(r, r.message)
        except Refusal as e:
            return self._finish(e.name, e.detail)

    def _on(self, r: Received, m: Message) -> list[Any]:
        if m.type in ("Closed", "Refused"):
            return [] if len(self.lines) == self.total else self._finish("closed", f"{r.channel} left early")
        if r.channel == "coins":
            k = self.awaiting.pop(0)
            blk = self.circuit[k - 1]
            self.draws[k] = draw(m.body, blk)
            self.parts[k]["draw"] = {"law": "subset", "k": draw_size(blk), "population": blk.n, "units": self.draws[k]}
            return [Send(DEVELOPER, msg("Draw", k, {"units": self.draws[k]}))]
        k, j = m.fields["k"], body(m)
        if m.type == "Hardware":
            try:
                reg = Registration.of(j)
            except (KeyError, TypeError, ValueError, IndexError) as e:
                raise Refusal("hardware", f"the hardware registration doesn't parse: {type(e).__name__}") from None
            self.layout = check_registration(self.layout, reg, self.auditor)
            self.public["hardware"].append({"after": len(self.declarations), "head": self.layout.head, **j})
            return []
        if not self.public["hardware"]:
            raise Refusal("out-of-phase", f"{m.type} before the hardware registration")
        if m.type == "Register":
            if self.baseline is not None:
                raise Refusal("out-of-phase", "a second registration")
            self.baseline, self.builder = check_baseline(j)
            self.public["registration"] = {"digest": self.baseline.digest(), **self.baseline.to_json()}
            return self.registered()
        if self.baseline is None:
            raise Refusal("out-of-phase", f"{m.type} before the registration")
        if m.type == "Declare":
            if k in self.parts:
                raise Refusal("recommit", f"declaration {k}'s values were committed already, and its coins are drawn "
                                          "after that commitment")
            if k != len(self.circuit) + 1:
                raise Refusal("out-of-phase", f"declaration {k} out of order")
            d = Declaration(k, j["advice"], j["record"])
            blk = check_declaration(d, self.layout, self.builder, self.circuit,
                                    "".join(x["advice"] for x in self.declarations))
            self.circuit.append(blk)
            self.declarations.append({"k": k, "advice": d.advice, "record": d.record, "digest": d.digest,
                                      "accounting": self.layout.accounting(DECLARATION_SECONDS * TICKS_PER_SECOND)})
            self.parts[k] = {}
            self.awaiting.append(k)
            return [Send("coins", Message("CoinsDraw", {"n": coin_bytes(draw_size(blk), blk.n)})), *self.declared(k)]
        if m.type == "Proofs":
            if k not in self.draws or "proofs" in self.parts[k]:
                raise Refusal("out-of-phase", f"proofs of declaration {k} before its draw, or twice")
            self.parts[k]["proofs"] = j
            return self._advance()
        self.component(m, k, j)
        return self._advance()

    def registered(self) -> list[Any]:
        """What the composite does once the registration is in (onsite: start the certifier)."""
        return []

    def declared(self, k: int) -> list[Any]:
        """What the composite does once declaration k is committed (onsite: the memory challenger's round)."""
        return []

    def component(self, m: Message, k: int, j: Any) -> None:
        raise Refusal("out-of-phase", f"{m.type} is no message of the {self.protocol} protocol")

    def ready(self, k: int) -> bool:
        return "proofs" in self.parts[k]

    def answers(self, k: int, checks: Mapping[int, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        """The subprotocols' answers for declaration k beyond sampled proofs and PoUW (onsite: the certifier, PoUS and the
        bandwidth audit)."""
        return {}

    def _advance(self) -> list[Any]:
        out: list[Any] = []
        while len(self.lines) < len(self.declarations) and self.ready(len(self.lines) + 1):
            out += self._decide(len(self.lines) + 1)
        if len(self.lines) == self.total:
            self.public["period"] = account(self.baseline.digest(), self.declarations, self.lines, self.layout)
            return out + self._finish()
        return out

    def _decide(self, k: int) -> list[Any]:
        d, blk, proofs = self.declarations[k - 1], self.circuit[k - 1], self.parts[k]["proofs"]
        checks = {u: verify(self.statement(k, u), proofs.get(str(u), {})) for u in self.draws[k]}
        sp = sampled_proofs(Declaration(k, d["advice"], d["record"]), blk, self.draws[k], checks)
        answers = {sp["protocol"]: sp, IDS["pouw"]: pouw_window(sp, blk), **self.answers(k, checks)}
        line = {"k": k, "record": d["digest"], "advice": d["advice"], "advice_bits": len(d["advice"]),
                "layout": d["record"]["placement"]["layout"], "placement": d["record"]["placement"]["units"],
                "theta_leaf": d["record"]["placement"]["theta"],
                "batches": [p.batch for p in blk.units], "circuit": blk.program, "units": blk.n, "real": blk.real,
                "ands": blk.ands(), "credited_ands": blk.credited_ands(), "capacity_tc": d["accounting"]["pouw"]["tc"],
                "draw": self.parts[k]["draw"],
                "answers": {i: {"verdict": a["verdict"], "reason": a["reason"]} for i, a in answers.items()},
                "composite": compose(self.protocol, answers)}
        self.lines.append(line)
        return [Record(line, "output")]

    def statement(self, k: int, u: int) -> Statement:
        """Unit u of declaration k's statement, from what the verifier holds: the record, the previous record and the
        registration, never the developer's word."""
        rec, blk = self.declarations[k - 1]["record"], self.circuit[k - 1]
        if u > 0:
            src, pos = rec["values"], u - 1
        elif k > 1:
            src = self.declarations[k - 2]["record"]["values"]
            pos = src["leaves"] - 1
        else:
            src, pos = self.baseline.state0, 0
        return Statement(self.declarations[k - 1]["digest"], u, u >= blk.real, rec["values"], {**src, "position": pos},
                         self.baseline.theta, self.frames(k, u))

    def frames(self, k: int, u: int) -> dict[str, str] | None:
        return None


# --- the example's inputs ----------------------------------------------------------------------------------------------

def theta_of(seed: int) -> Theta:
    """A model's weights for the example, from a seed."""
    rng = np.random.default_rng(seed)
    return Theta(tuple(int(v) for v in rng.integers(-128, 128, D * D)), tuple(int(v) for v in rng.integers(-128, 128, D * D)))


def load_of(seed: int, sizes: Sequence[int]) -> list[list[list[int]]]:
    """The example's requests: one list a declaration period, each request D int8 codes."""
    rng = np.random.default_rng(seed)
    return [[[int(v) for v in rng.integers(-128, 128, D)] for _ in range(n)] for n in sizes]


SIZES = (3, 4, 2)
"""Requests in each declaration period of the example's monitoring period."""
PLAN = ((1, ("gpu0", "gpu1")), (2, ("gpu0", "gpu1", "gpu2")))
"""B's placement: gpu2 from declaration 2 on, once the later hardware registration holds it."""
CHEATS = ("hardware", "builder", "placement", "unreported", "advice-cap", "recommit", "wrong-value", "fork")
"""Each way the developer deviates in the tests, by the phase that refuses it: hardware registration (`hardware`: a later
entry names a device again), registration (`builder`), operation (`placement`: B places a unit on a device before its
registration; `unreported`), declaration (`advice-cap`, `recommit`), audit (`wrong-value`) and accounting (`fork`)."""
ONSITE_CHEATS = ("io-mismatch", "lost-theta", "off-node")
"""The deviations only an onsite composite's own devices see: the wire carrying other outputs than the committed ones,
answering the memory challenger with a row that isn't theta's, and fetching theta from off the node."""
OFF_NODE_TICKS = 3_000_000
"""`off-node`: the round trip to where a developer that doesn't hold theta on the node keeps it."""


def summary(outcome: Mapping[str, Any]) -> list[str]:
    """The verifier's public outcome as lines, phase by phase."""
    out: list[str] = []
    hw = list(outcome.get("hardware", ()))
    skus = {digest("lifecycle/card/v1", c): c["sku"] for h in hw for c in h["cards"]}

    def where(x: Mapping[str, Any]) -> str:
        detail = (skus.get(x["card"], "?") if x["kind"] == "device" else f"{x['tier']} {x['bytes'] >> 30} GiB"
                  if x["kind"] == "memory" else x["owner"] if x["owner"] != "prover" else "")
        return f"{x['id']} ({x['kind']}{' ' + detail if detail else ''}{' in ' + x['parent'] if x['parent'] else ''})"

    def wrap(head: str, items: Sequence[str]) -> None:
        if items:
            out.extend(textwrap.wrap(", ".join(items), 120, initial_indent=head, subsequent_indent=" " * len(head)))

    def hardware(after: int) -> None:
        for h in [h for h in hw if h["after"] == after]:
            out.append(f"1 hardware     registration{' after declaration ' + str(after) if after else ''}: entry "
                       f"{h['head'][:16]}..., after {h['previous'][:16]}...")
            wrap(f"{'':15s}cards      ", [c["sku"] for c in h["cards"]])
            wrap(f"{'':15s}locations  ", [where(x) for x in h["locs"]])
            wrap(f"{'':15s}links      ", [f"{x['id']} {x['ends'][0]}-{x['ends'][1]}" for x in h["links"]])

    hardware(0)
    reg = outcome.get("registration")
    if reg:
        plan = "; ".join(f"from declaration {k}: {', '.join(ds)}" for k, ds in reg["builder"]["plan"])
        out += ["2 registration digest " + reg["digest"][:32] + "...",
                f"               builder B {reg['builder']['format']}, advice cap {reg['builder']['cap']} bits a "
                f"declaration, placement {plan}",
                f"               theta root {reg['theta']['root'][:32]}... ({reg['theta']['leaves']} hidden rows), "
                f"state0 root {reg['state0']['root'][:32]}..."]
    for line in outcome.get("lines", ()):
        d = line["draw"]
        out += [f"declaration {line['k']}",
                f"  3 operation  {line['real']} requests served, batches {line['batches']}",
                f"  4 declare    advice {line['advice']} ({line['advice_bits']} bits charged); block of {line['units']} "
                f"units ({line['units'] - line['real']} filler) on {', '.join(line['placement'])}",
                f"               circuit {line['circuit'][:16]}..., layout {line['layout'][:16]}..., record "
                f"{line['record'][:16]}...",
                f"  5 audit      drew subset:{d['k']} of {d['population']}: units {d['units']}"]
        for i, a in line["answers"].items():
            out.append(f"               {i:40s} {a['verdict']}" + (f"  ({a['reason']})" if a["reason"] else ""))
        cm = line["composite"]
        out.append(f"               composite {cm['product']:30s} {cm['verdict']}")
        hardware(line["k"])
    p = outcome.get("period")
    if p:
        b = p["bound"]
        out += ["6 accounting", f"  period       {p['declarations']} declarations: {p['verdict']}"
                + (f"  ({p['reason']})" if p["reason"] else ""),
                f"  credited     {p['credited_ands']} of {p['appended_ands']} appended AND gates; {p['credited_macs']} "
                f"multiply-adds of the {p['capacity_tc']:.2e} the registered devices could do: share {p['share']}",
                f"  advice       {p['advice_bits_charged']} bits charged as leakage",
                f"  bound        {b['statement']}; a declaration alone: delta {b['per_declaration_delta']} at eps "
                f"{b['eps']}"]
        a = p["accounting"]
        wrap("  projections  ", [f"a declaration period on the final layout: PoUW {a['pouw']['devices']} devices, "
                          f"{a['pouw']['tc']:.2e} multiply-adds",
                          "PoUS " + ", ".join(f"{t} {n >> 30} GiB" for t, n in a["pous"].items()),
                          f"bandwidth {a['bandwidth']['device_bits']:.2e} device-memory bits, "
                          f"{a['bandwidth']['boundary_bits']:.2e} boundary bits",
                          f"certifier: boundary {a['certifier']['boundary']}, certified {a['certifier']['certified']}, "
                          f"{a['certifier']['uncertified_bits']:.2e} bits unseen"])
        for name, s in p["scaled"].items():
            d = "0" if s["log10_delta"] is None else f"10^{s['log10_delta']}"
            out.append(f"  {name:6s}       {s['declarations']} declarations of {DECLARATION_SECONDS} s like these: "
                       f"{s['credited_ands']:.2e} credited AND gates, share {s['share']} of the layout's "
                       f"{s['capacity_tc']:.2e} multiply-adds; delta {d} at eps {b['eps']}")
    if outcome.get("refused"):
        out.append(f"refused {outcome['refused']['name']}: {outcome['refused']['detail']}")
    return out
