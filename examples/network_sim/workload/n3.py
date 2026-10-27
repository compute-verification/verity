"""The in-the-clear N3 check: the frames recorded on every certified unit boundary are exactly φ of the crossing values the
units committed, at the placement's buckets.

The check reads three things, all JSON-shaped as `run` writes them:

- `public`: the Program's digest, its workload and spec (`model.build`), the hardware, the frame size B, the placement's
  root, each unit's value root and count, and the occupancy profile.
- `witness`: the placement's rows and salts, and each unit's value rows and salts. In the clear the checker reads them; in
  the protocol a zero-knowledge proof shows the same relation over the commitments.
- `records`: per certified link direction (`Hardware.key`), the (bucket, frame) pairs logged on it (`merge` joins the units'
  logs; under certifiers each direction's certifier writes it).

**Steps.** (1) The witness's placement rows register to the public root, and the placement they decode to is paced and
causal on the hardware, whose units are all covered. From the placement, the Program and the hardware come the crossing
edges and φ, and φ's occupancy profile must be the public one. (2) Each unit's value rows register to its public root, and
every crossing value is opened at its source unit (`values.verify`). (3) The schedule, which needs no witness: on every
certified direction, the frames recorded in each bucket number what the occupancy profile says. (4) Every frame is explained
(direction 2): the direction's recorded frames, in grid order (by bucket, then as recorded), are φ's on it in its grid
order (`Phi.slots` by bucket, then send order), each equal byte for byte to φ's frame of the committed values (a
connection's `HELLO`, or frame j of a message) and in φ's bucket. (5) Every crossing value appears (direction 1): on every
direction a message crosses, the recorded frames at its positions hold each crossing value's committed bits at φ's
position, and every frame φ puts there was recorded.

The canonical framing names no message on the wire (`framing`), so a frame is matched to φ by its position alone: φ is
public, and the order in which a link's connections interleave in a bucket is the grid order N3's layout declares.

**Violation codes**, each with the link direction (or the unit) it concerns:

- `program`, `placement`, `hardware`, `occupancy`, `values`: step 1 or 2 failed (the digest isn't the spec's Program's; the
  placement doesn't register, decode, pace or respect causality; a unit isn't covered; the public profile isn't the
  placement's; a unit's rows don't register to its root or an opening fails).
- `records`: two logs of one direction disagree.
- `schedule`: a direction's frames per bucket differ from the occupancy profile.
- `malformed`: not a B-byte frame with the canonical header.
- `extra`: a frame past the frames φ puts on the direction.
- `content`: a frame whose payload differs from φ's at its position.
- `fill`: a frame whose data are φ's but whose fill past them isn't zero, a covert channel in the padding.
- `bucket`: a frame of φ's recorded in another bucket.
- `missing`: a frame φ puts on a direction that wasn't recorded there.
- `value`: a crossing value whose bits at φ's position in the recorded payload aren't its committed value.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from verity.core.protocols.certifier.active import HEADER
from verity.ml.circuits.codec import program_digest
from verity.ml.commitments import advice

from . import framing as F
from . import model as M
from . import placement as P
from . import values as V
from .hardware import Hardware

Records = dict[str, list[tuple[int, bytes]]]


@dataclass(frozen=True)
class Violation:
    code: str
    where: str
    detail: str


@dataclass
class Report:
    violations: list[Violation] = field(default_factory=list)
    stats: Counter = field(default_factory=Counter)

    @property
    def ok(self) -> bool:
        return not self.violations

    def add(self, code: str, where: str, detail: str) -> None:
        self.violations.append(Violation(code, where, detail))

    def codes(self) -> Counter:
        return Counter(v.code for v in self.violations)

    def to_json(self) -> dict:
        return {"ok": self.ok, "stats": dict(sorted(self.stats.items())), "codes": dict(sorted(self.codes().items())),
                "violations": [{"code": v.code, "where": v.where, "detail": v.detail} for v in self.violations]}


def merge(results: dict[str, dict]) -> tuple[Records, list[Violation]]:
    """One log per link direction from the units' results; a direction two units log must be logged the same by both."""
    out: Records = {}
    bad = []
    for unit in sorted(results):
        for key, entries in results[unit]["records"].items():
            log = [(b, bytes.fromhex(f)) for b, f in entries]
            if key in out and out[key] != log:
                bad.append(Violation("records", key, f"{unit}'s log differs from another unit's"))
            out.setdefault(key, log)
    return out, bad


def _registration(rows: list[str], salts: list[str], row_bits: int, binding: bytes) -> advice.Registration:
    return advice.register([bytes.fromhex(r) for r in rows], row_bits, binding, [bytes.fromhex(s) for s in salts])


def placement_of(public: dict, witness: dict, model: M.Model, hw: Hardware, report: Report) -> P.Placement | None:
    """The placement whose rows register to the public root, checked to be paced and causal on the hardware; None when
    it doesn't decode or register."""
    w = witness["placement"]
    try:
        reg = _registration(w["rows"], w["salts"], P.ROW_BITS, P.BINDING)
        placement = P.Placement.from_rows(reg.rows)
    except (ValueError, KeyError) as e:
        report.add("placement", "", f"the placement's rows don't decode: {e}")
        return None
    if reg.root.hex() != public["placement"]:
        report.add("placement", "", "the placement's rows don't register to the public root")
        return None
    if placement.n != model.n or len(placement.ticks) != len(hw.leaves) or \
            any(not 0 <= x < len(hw.leaves) for x in placement.leaf) or \
            any(not 0 <= s < len(placement.ticks[0]) for s in placement.step):
        report.add("placement", "", "the placement doesn't place this Program's values on this hardware's leaves")
        return None
    if not placement.paced(range(len(hw.leaves))):
        report.add("placement", "", "a device's ticks don't increase step by step")
    bad = P.causal(model, placement, hw)
    if bad:
        report.add("placement", "", f"{len(bad)} reads break causality, the first gate {bad[0][0]} reading {bad[0][1]}")
    for u in sorted(hw.units):
        if not hw.covered(u):
            report.add("hardware", u, "a boundary link isn't certified")
    return placement


def open_values(public: dict, witness: dict, placement: P.Placement, hw: Hardware, edges: list[P.Edge], width: int,
          report: Report) -> dict[int, int]:
    """Each crossing value, opened at its source unit's commitment and checked against the unit's public root."""
    by_unit: dict[str, list[P.Edge]] = defaultdict(list)
    for e in edges:
        by_unit[e.src].append(e)
    opened: dict[int, int] = {}
    for unit, es in sorted(by_unit.items()):
        pub, wit = public["values"].get(unit), witness["values"].get(unit)
        if pub is None or wit is None:
            report.add("values", unit, "the unit committed no values")
            continue
        gates = tuple(placement.at(hw.leaves_of(unit)))
        root, count = bytes.fromhex(pub["root"]), pub["count"]
        if count != len(gates):
            report.add("values", unit, f"the unit committed {count} values, the placement puts {len(gates)} there")
            continue
        reg = _registration(wit["rows"], wit["salts"], V.ROW_BITS, V.binding(unit))
        if reg.root != root:
            report.add("values", unit, "the unit's rows don't register to its public root")
            continue
        uv = V.UnitValues(unit, gates, width, reg)
        openings: dict[int, advice.Opening] = {}
        for e in es:
            row, offset = uv.locate(e.value)
            if row not in openings:
                openings[row] = uv.open(e.value)
            v = V.verify(root, unit, count, width, openings[row], row, offset)
            if v is None:
                report.add("values", unit, f"value {e.value} doesn't open")
                continue
            opened[e.value] = v
        report.stats["openings"] += len(openings)
    report.stats["values"] = len(opened)
    return opened


def _well_formed(frame: bytes, B: int) -> bool:
    return len(frame) == B and frame[:HEADER.size] == F.head(B)


def check(public: dict, witness: dict, records: Records) -> Report:
    report = Report()
    model = M.build(public.get("workload", "mlp"), public["spec"])
    if program_digest(model.program) != public["program"]:
        report.add("program", "", "the public digest isn't the spec's Program's")
    hw = Hardware.from_json(public["hardware"])
    B = public["frame_bytes"]
    placement = placement_of(public, witness, model, hw, report)
    if placement is None:
        return report
    edges = P.crossings(model, placement, hw)
    phi = F.phi(hw, edges, B)
    occupancy = phi.occupancy()
    if occupancy != public["occupancy"]:
        report.add("occupancy", "", "the public occupancy profile isn't the committed placement's")
    opened = open_values(public, witness, placement, hw, edges, model.value_bits, report)
    report.stats.update(edges=len(edges), messages=len(phi.messages), directions=len(occupancy))

    for key in sorted(set(public["occupancy"]) | set(records)):
        counts = Counter(b for b, _ in records.get(key, []))
        want = public["occupancy"].get(key, [])
        got = [counts.get(b, 0) for b in range(max([len(want)] + [b + 1 for b in counts]))]
        if got != want + [0] * (len(got) - len(want)):
            report.add("schedule", key, f"frames per bucket {got}, the profile says {want}")

    streams = phi.by_stream()
    slots = phi.slots()
    cut: dict[int, list[bytes] | None] = {}

    def expected(s: F.Slot) -> tuple[bytes | None, int]:
        """φ's frame at the slot, of the committed values (None when one didn't open), and the data bytes it holds."""
        if s.kind == "hello":
            return F.hello(hw, s.src, s.dst, B), F.HELLO.size
        m = streams[s.stream]
        if s.stream not in cut:
            cut[s.stream] = phi.frames(m, opened) if all(e.value in opened for e in m.edges) else None
        room = F.room(B)
        frames = cut[s.stream]
        return (frames[s.index] if frames else None), min(room, m.nbytes - s.index * room)

    for key in sorted(set(slots) | set(records)):
        want = sorted(slots.get(key, []), key=lambda s: s.bucket)
        got = sorted(records.get(key, []), key=lambda x: x[0])
        for i, (b, f) in enumerate(got):
            report.stats["frames"] += 1
            if i >= len(want):
                report.add("extra", key, f"frame {i} in bucket {b}, past the {len(want)} φ puts on the direction")
                continue
            s = want[i]
            what = "the HELLO" if s.kind == "hello" else f"frame {s.index} of message {s.stream}"
            if not _well_formed(f, B):
                report.add("malformed", key, f"a {len(f)}-byte frame in bucket {b}, at {what}'s position {i}")
                continue
            wf, n = expected(s)
            if wf is not None and f != wf:
                fill = f[:HEADER.size + n] == wf[:HEADER.size + n]
                report.add("fill" if fill else "content", key, f"{what} (position {i}) " +
                           ("has non-zero fill" if fill else "isn't φ of the committed values"))
                continue
            if b != s.bucket:
                report.add("bucket", key, f"{what} (position {i}) in bucket {b}, φ's is {s.bucket}")
                continue
            report.stats["declared" if s.kind == "hello" else "explained"] += 1
        if len(got) < len(want):
            report.add("missing", key, f"frames {len(got)}..{len(want) - 1} of the {len(want)} φ puts on the direction")
        at: dict[int, list[int]] = defaultdict(list)
        for i, s in enumerate(want):
            if s.kind == "data":
                at[s.stream].append(i)
        for stream, positions in at.items():
            m = streams[stream]
            payload = b""
            for i in positions:
                if i >= len(got) or not _well_formed(got[i][1], B):
                    break
                payload += got[i][1][HEADER.size:]
            have, acc = min(len(payload), m.nbytes) * 8, int.from_bytes(payload, "little")
            offsets = m.positions()
            for e in m.edges:
                o = offsets[e.value]
                if e.value not in opened:
                    continue
                if o + e.width > have:
                    report.add("missing", key, f"value {e.value} of message {stream}")
                elif (acc >> o) & ((1 << e.width) - 1) != opened[e.value]:
                    report.add("value", key, f"value {e.value} at bit {o} of message {stream} isn't its committed value")
                else:
                    report.stats["appearances"] += 1
    return report
