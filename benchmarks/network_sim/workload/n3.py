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
(direction 2): a recorded frame is a declared `HELLO` of a connection whose route crosses the direction, or frame j of a
stream φ puts on the direction, equal byte for byte to frame j of φ applied to the committed values and in φ's bucket.
(5) Every crossing value appears (direction 1): on every direction a message crosses, its stream's recorded payload holds
each crossing value's committed bits at φ's position, and every frame φ puts there was recorded.

A frame is matched to φ by its stream and its rank among that stream's frames on the direction, so the order in which
different streams interleave on a link doesn't matter.

**Violation codes**, each with the link direction (or the unit) it concerns:

- `program`, `placement`, `hardware`, `occupancy`, `values`: step 1 or 2 failed (the digest isn't the spec's Program's; the
  placement doesn't register, decode, pace or respect causality; a unit isn't covered; the public profile isn't the
  placement's; a unit's rows don't register to its root or an opening fails).
- `records`: two logs of one direction disagree.
- `schedule`: a direction's frames per bucket differ from the occupancy profile.
- `malformed`: not a B-byte frame with a `DATA` or `LAST` header whose length fits.
- `unexplained`: a frame of a stream φ doesn't put on the direction, or a stream-0 frame that is no declared `HELLO`.
- `extra`: a frame of a φ stream past that stream's frame count.
- `content`: a frame whose header or payload differs from φ's.
- `fill`: a frame whose header and payload are φ's but whose fill isn't zero, a covert channel in the padding.
- `bucket`: a frame of φ's recorded in another bucket.
- `missing`: a frame φ puts on a direction that wasn't recorded there.
- `value`: a crossing value whose bits at φ's position in the recorded payload aren't its committed value.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from verity.core.protocols.certifier.active import DATA, HEADER, LAST, decode_frame
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
    if len(frame) != B:
        return False
    kind, _, n = HEADER.unpack_from(frame)
    return kind in (DATA, LAST) and HEADER.size + n <= B


def _canonical_prefix(frame: bytes) -> bytes:
    return frame[:HEADER.size + HEADER.unpack_from(frame)[2]]


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
    for key in sorted(set(slots) | set(records)):
        expected: dict[int, list[tuple[int, bytes | None]]] = defaultdict(list)
        hellos: list[tuple[int, bytes]] = []
        for s in slots.get(key, []):
            if s.kind == "hello":
                hellos.append((s.bucket, F.hello(hw, s.src, s.dst, B)))
            else:
                m = streams[s.stream]
                ok = all(e.value in opened for e in m.edges)
                expected[s.stream].append((s.bucket, phi.frames(m, opened)[s.index] if ok else None))
        seen: dict[int, list[tuple[int, bytes]]] = defaultdict(list)
        for b, f in records.get(key, []):
            report.stats["frames"] += 1
            if not _well_formed(f, B):
                report.add("malformed", key, f"a {len(f)}-byte frame in bucket {b}")
                continue
            kind, stream, _ = decode_frame(f)
            if stream == F.HELLO_STREAM:
                if (b, f) in hellos:
                    hellos.remove((b, f))
                    report.stats["declared"] += 1
                else:
                    report.add("unexplained", key, f"a stream-0 frame in bucket {b} that is no declared HELLO")
                continue
            if stream not in expected:
                report.add("unexplained", key, f"a frame of stream {stream:#x} in bucket {b}, which φ doesn't put here")
                continue
            j = len(seen[stream])
            seen[stream].append((b, f))
            if j >= len(expected[stream]):
                report.add("extra", key, f"frame {j} of stream {stream}, which has {len(expected[stream])}")
                continue
            want_b, want_f = expected[stream][j]
            if want_f is not None and f != want_f:
                if _canonical_prefix(f) == _canonical_prefix(want_f):
                    report.add("fill", key, f"frame {j} of stream {stream} has non-zero fill")
                else:
                    report.add("content", key, f"frame {j} of stream {stream} isn't φ of the committed values")
                continue
            if b != want_b:
                report.add("bucket", key, f"frame {j} of stream {stream} in bucket {b}, φ's is {want_b}")
                continue
            report.stats["explained"] += 1
        for b, _ in hellos:
            report.add("missing", key, f"the HELLO of bucket {b}")
        for stream, want in expected.items():
            got = seen.get(stream, [])
            if len(got) < len(want):
                report.add("missing", key, f"frames {len(got)}..{len(want) - 1} of stream {stream}")
            m = streams[stream]
            payload = b""
            for _, f in got:
                kind, _, chunk = decode_frame(f)
                payload += chunk
                if kind == LAST:
                    break
            have, acc = len(payload) * 8, int.from_bytes(payload, "little")
            positions = m.positions()
            for e in m.edges:
                o = positions[e.value]
                if e.value not in opened:
                    continue
                if o + e.width > have:
                    report.add("missing", key, f"value {e.value} of stream {stream}")
                elif (acc >> o) & ((1 << e.width) - 1) != opened[e.value]:
                    report.add("value", key, f"value {e.value} at bit {o} of stream {stream} isn't its committed value")
                else:
                    report.stats["appearances"] += 1
    return report
