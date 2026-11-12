"""The count over a span between two marks (`NetworkCheck.Span`, ``verity/certifier/span/v0``), in Python: the mirror
its Lean check is held to (``network/tests/test_network_span_protocol.py``).

A span runs from the execution phase of a start mark to the execution phase of an end mark, on one link of the network
certifier. The count answers, from certificates alone: is every execution phase of the span and of its tail certified
under the schedule, and how many bits can have left on the link over them. Its formats are the Lean module's docstring;
this module decodes them (`params`, `commitment`, `evidence`), checks them (`check`) and names what a composition ties
(`named`). The framing of an answer (canonical JSON, hex, the outcome) is the core interface's, which a difftest's
reference adds around these.

**The marks.** Each mark is a `certificate.CertifiedMark`: the 64-byte message it was asked for, the 64-byte salt the
network certifier sampled, and the certificate that signs the message under the salt (`mark_digest`) on the link's
marks link name (`marks_link`). The count opens each (`CertifiedMark.refusal`) and reads its execution phase from its
certificate. A mark's bytes in the commitment are `mark_bytes`: its certificate's bytes (`Certificate.encode`) behind
their length, its salt, then its message. A link's name is what the period's check takes: at most 65,535 UTF-8 bytes.

**The span.** From the start mark's execution phase w_s to the end mark's w_e, then a tail (`tail`): the network
certifier holds a connection's response for a bounded time before it releases it, so bytes it held at the end mark
leave in the tail's execution phases. The count is `phases = w_e + tail − w_s + 1` execution phases, each at the
schedule's capacity (`capacity_bits`): every cell of both egress records, the responses' and the fails', which both
leave on the link. A certificate hides its record, so the count takes each certified record as full; it is certified
only when complete, so nothing left outside its cells. What of the count is public is `public`'s to say: the two
marks' execution phases are not, since they say when the span ran, which a mark's salt hides.

**Each execution phase** of the span and its tail shows its two egress certificates (the responses' record's on the
link, the fails record's on the link's fails link name) and the signed head of its egress cells, the one signed thing
that names the schedule's digest (`WR.check_registration`). The network certifier certifies no record of an
execution phase it faulted or refused a connection in, so a span that holds one is rejected.

**A reason** holds no number: the two marks' execution phases are not public, and a reason is part of an answer."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

from verity.commitments import domains
from verity.commitments.crypto import lms
from verity.commitments.identity import tagged_sha512
from verity.protocols.certifier.verifier import warden_rows as WR
from verity.protocols.certifier.verifier.certificate import (DIGEST_BYTES, SALT_BYTES, Certificate, CertifiedMark,
                                                            Direction, LmsKey)
from verity.protocols.certifier.verifier.warden import INTEGER_FIELDS, WardenParams as Schedule

ID = domains.CERTIFIER_SPAN_PROTOCOL
COMMITMENT_TAG = domains.CERTIFIER_SPAN
#: what a link's name gains for the link its fails record is certified under
FAILS = "fails"
PARAMS_KEYS = ["key", "link", "lms", "schedule", "session"]
SCHEDULE_KEYS = sorted(INTEGER_FIELDS)
#: with the bound on a response's length: every field but the egress form and the lengths rows' form
BOUNDED_SCHEDULE_KEYS = sorted(f.name for f in fields(Schedule) if f.name not in ("egress", "lengths"))
CERTIFICATE_KEYS = ["digest", "direction", "key_id", "link", "signature", "window"]
PHASE_KEYS = ["certificates", "phase", "registration"]
ROWS_KEYS = ["cell_bytes", "certificate", "direction", "domain", "format", "layout", "link", "parameters", "root",
             "row_schema", "rows", "value", "window", "wiring", "word_bits", "words"]
LAYOUT_KEYS = ["buckets", "connection_slots", "first_bucket", "slots"]
HEX = frozenset("0123456789abcdef")
U32 = 1 << 32
#: the two marks, in the commitment's order
MARKS = ("start", "end")


class Refused(ValueError):
    """A decoder's refusal: parameters, a commitment or evidence that isn't in the count's format."""


@dataclass(frozen=True)
class Params:
    link: str
    key: LmsKey
    session: bytes
    schedule: Schedule


@dataclass(frozen=True)
class Commitment:
    """The two marks, the start's then the end's, and the bytes they were read from."""

    marks: tuple[CertifiedMark, CertifiedMark]
    certificates: tuple[bytes, bytes]
    bytes: bytes


@dataclass(frozen=True)
class Phase:
    phase: int
    certificates: tuple[Certificate, ...]
    registration: Mapping[str, Any]


def _exactly(j: Any, keys: list[str], ctx: str) -> None:
    if not isinstance(j, dict):
        raise Refused(f"{ctx}: not an object")
    if sorted(j) != keys:
        raise Refused(f"{ctx}: the keys must be {keys}, not {sorted(j)}")


def _nat(v: Any, ctx: str) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
        raise Refused(f"{ctx}: not a natural number")
    return v


def _of(v: Any, kind: type, ctx: str) -> Any:
    if not isinstance(v, kind) or isinstance(v, bool) != (kind is bool):
        raise Refused(f"{ctx}: not a {kind.__name__}")
    return v


def _unhex(v: Any, ctx: str) -> bytes:
    s = _of(v, str, ctx)
    if len(s) % 2 or not HEX.issuperset(s):
        raise Refused(f"{ctx}: not even-length lowercase hex")
    return bytes.fromhex(s)


def _schedule(j: Any) -> Schedule:
    """`scheduleOf`: the schedule's integer fields, each a positive integer, a cell past its header, and B, S and T
    u32s, in the id egress form, with its bound on a response's length when it has one."""
    keys = BOUNDED_SCHEDULE_KEYS if isinstance(j, dict) and sorted(j) == BOUNDED_SCHEDULE_KEYS else SCHEDULE_KEYS
    _exactly(j, keys, "schedule")
    values = {k: _nat(j[k], "schedule") for k in keys}
    try:
        s = Schedule(**values)
    except ValueError as e:
        raise Refused(f"schedule: {e}") from None
    if not (s.cell_bytes < U32 and s.lanes * s.slots < U32 and s.window < U32):
        raise Refused("schedule: B, S = lanes · slots and T are u32s")
    return s


def params(j: Any) -> Params:
    """The count's parameters, all public: the network certifier's LMS key and its parameters, the certifier session,
    the link, its name held to what the period's check holds it to, and the schedule."""
    _exactly(j, PARAMS_KEYS, "the object")
    link = _of(j["link"], str, "link")
    if len(link.encode()) >= 1 << 16:
        raise Refused("link: a name is at most 65,535 UTF-8 bytes")
    lj = j["lms"]
    _exactly(lj, ["H", "N", "W", "toy"], "lms")
    sizes, toy = [_nat(lj[k], "lms") for k in ("N", "W", "H")], _of(lj["toy"], bool, "lms")
    try:
        lp = lms.Params(*sizes, toy=toy)
    except ValueError as e:
        raise Refused(f"lms: {e}") from None
    key = _unhex(j["key"], "key")
    if len(key) != 16 + lp.N:
        raise Refused(f"key: an LMS public key is {16 + lp.N} bytes, not {len(key)}")
    session = _unhex(j["session"], "session")
    if len(session) != lms.SESSION_BYTES:
        raise Refused(f"session: a certifier session id is {lms.SESSION_BYTES} bytes, not {len(session)}")
    return Params(link, LmsKey(lp, key), session, _schedule(j["schedule"]))


def mark_bytes(m: CertifiedMark) -> bytes:
    """A mark's bytes in the commitment: u32be len(certificate) ‖ certificate (`Certificate.encode`) ‖ salt ‖
    message."""
    certificate = m.certificate.encode()
    return struct.pack(">I", len(certificate)) + certificate + m.salt + m.mark


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data, self.at = data, 0

    def take(self, n: int) -> bytes:
        if self.at + n > len(self.data):
            raise Refused(f"the bytes end {self.at + n - len(self.data)} short at {self.at}")
        self.at += n
        return self.data[self.at - n:self.at]

    def be(self, n: int) -> int:
        return int.from_bytes(self.take(n), "big")


def commitment(b: bytes) -> Commitment:
    """The commitment's two marks, the start's then the end's, each `mark_bytes`, with nothing left over. Each
    certificate is read from exactly the bytes its length gives it (`Certificate.decode`)."""
    r, marks, certificates = _Reader(b), [], []
    for _ in MARKS:
        certificates.append(r.take(r.be(4)))
        try:
            c = Certificate.decode(certificates[-1])
        except ValueError as e:
            raise Refused(str(e)) from None
        salt = r.take(SALT_BYTES)
        marks.append(CertifiedMark(r.take(DIGEST_BYTES), salt, c))
    if r.at != len(b):
        raise Refused("bytes left after the two marks")
    return Commitment((marks[0], marks[1]), (certificates[0], certificates[1]), b)


def digest(c: Commitment) -> bytes:
    """The digest that binds the commitment: the tagged SHA-512 of its bytes."""
    return tagged_sha512(COMMITMENT_TAG, c.bytes)


def _json_certificate(j: Any, ctx: str) -> Certificate:
    _exactly(j, CERTIFICATE_KEYS, ctx)
    try:
        direction = Direction(_of(j["direction"], str, ctx))
    except ValueError:
        raise Refused(f"{ctx}: not a direction") from None
    return Certificate(_unhex(j["key_id"], ctx), _of(j["link"], str, ctx), direction, _nat(j["window"], ctx),
                       _unhex(j["digest"], ctx), _unhex(j["signature"], ctx))


def _rows(j: Any, ctx: str) -> dict:
    """`rowsOf`: a row registration with exactly its keys (and its layout's and certificate's), its numbers naturals,
    its strings strings and its byte fields lowercase hex."""
    _exactly(j, ROWS_KEYS, ctx)
    _exactly(j["layout"], LAYOUT_KEYS, f"{ctx}.layout")
    _exactly(j["certificate"], ["digest", "key", "signature"], f"{ctx}.certificate")
    for k in ("format", "value", "link", "direction", "parameters", "row_schema", "domain"):
        _of(j[k], str, f"{ctx}.{k}")
    for k in ("window", "cell_bytes", "rows", "words", "word_bits"):
        _nat(j[k], f"{ctx}.{k}")
    for k in LAYOUT_KEYS:
        _nat(j["layout"][k], f"{ctx}.layout")
    for o, k in ((j, "wiring"), (j, "root"), *((j["certificate"], k) for k in ("digest", "key", "signature"))):
        _unhex(o[k], f"{ctx}.{k}")
    return j


def evidence(j: Any) -> list[Phase]:
    """The evidence, already read as canonical JSON: ``{"phases": [...]}``, one entry per execution phase the prover
    shows, each ``{"certificates": [...], "phase": w, "registration": rows}``."""
    _exactly(j, ["phases"], "the object")
    out = []
    for x in _of(j["phases"], list, "phases"):
        ctx = "an execution phase"
        _exactly(x, PHASE_KEYS, ctx)
        certificates = tuple(_json_certificate(c, f"{ctx}.certificates")
                             for c in _of(x["certificates"], list, f"{ctx}.certificates"))
        out.append(Phase(_nat(x["phase"], f"{ctx}.phase"), certificates, _rows(x["registration"], f"{ctx}.registration")))
    return out


def tail(s: Schedule) -> int:
    """The span's tail in execution phases, ⌈(D + ⌊(egress_cells − 1) / r⌋) / T⌉. A connection's egress cell i leaves in
    bucket s + D + ⌊i / r⌋, s its admission's bucket, and it has at most egress_cells cells, so its last cell leaves at
    most D + ⌊(egress_cells − 1) / r⌋ buckets after s. A connection the network certifier holds bytes of at the end mark
    was admitted in the end mark's execution phase at the latest, so its cells have left that many buckets after that
    execution phase's last, T buckets an execution phase."""
    buckets = s.delay + (s.egress_cells - 1) // s.slots
    return -(-buckets // s.window)


def public(start: int, end: int, t: int, capacity: int) -> dict[str, int]:
    """What of the count is public, the one place that says so: the bits that can have left over the span and its
    tail, and the tail. The two marks' execution phases, `start` and `end`, are not: they say when the span ran, which
    a mark's salt hides, and they stay in the commitment."""
    return {"egress_bits": (end + t - start + 1) * capacity, "tail": t}


def capacity_bits(s: Schedule) -> int:
    """The bits that can leave on the link in one execution phase: every cell of its two egress records, the
    responses' (T buckets of lanes · r cells) and the fails' (T buckets of lanes cells), B bytes a cell."""
    return sum(g.buckets * g.allocation * g.frame_bytes * 8 for g in (s.grid, s.fails_grid))


def fails_link(link: str) -> str:
    """The link name `link`'s fails record is certified under, `<link>-fails`."""
    return f"{link}-{FAILS}"


def records(link: str) -> tuple[tuple[str, str], ...]:
    """The two egress records the count reads at each execution phase, by what the reasons call them and the link
    name each is certified under."""
    return (("egress", link), ("fails", fails_link(link)))


REFUSALS = {
    "link": "it is on another link than the link's marks link name",
    "egress": "it is not an egress certificate",
    "key": "it is under another key than the network certifier's",
    "opening": "its salt and message do not open the digest it signs",
    "signature": "its signature does not verify under the key in the certifier session",
}
"""Why a mark is refused, by `CertifiedMark.refusal`'s code."""

NONE_CERTIFIED = ("the network certifier certifies no record of an execution phase it faulted or refused a connection "
                  "in, and a span that holds one is rejected")
MISSING = f"an execution phase of the span or its tail is missing: {NONE_CERTIFIED}"


def _certified(p: Params, w: int, x: Phase) -> str | None:
    """Why execution phase `w`'s entry doesn't certify it under the schedule, or None when it does."""
    if len(x.certificates) != len(records(p.link)):
        return f"not two certificates, its egress record's and its fails record's: {NONE_CERTIFIED}"
    for (what, link), c in zip(records(p.link), x.certificates):
        if c.direction is not Direction.EGRESS:
            return f"the {what} certificate is not an egress certificate"
        if c.link != link:
            return f"the {what} certificate is on another link than {link}"
        if c.window != w:
            return f"the {what} certificate is at another execution phase"
        if not c.verify(p.key, p.session):
            return (f"the {what} certificate does not verify under the key in the certifier session: "
                    f"{NONE_CERTIFIED}")
    doc = x.registration
    if doc["parameters"] != p.schedule.digest.hex():
        return "the egress cells' head names another schedule's digest than the parameters'"
    try:
        WR.check_registration(doc, p.key, p.session, p.schedule, p.link, Direction.EGRESS, w)
    except ValueError:
        # the registration check's own message names the execution phase, which no reason of the count does
        return ("the egress cells' head is not the link's at its execution phase under the schedule, signed under the "
                "key in the certifier session")
    return None


def check(p: Params, c: Commitment, e: list[Phase]) -> tuple:
    """The count: ``("accept", public, profile)`` or ``("reject", reason)``. One pass over the two marks and the
    evidence's entries, never over the span's length."""
    for name, m in zip(MARKS, c.marks):
        code = m.refusal(p.key, p.session, p.link)
        if code is not None:
            return "reject", f"the {name} mark: {REFUSALS[code]}"
    start, end = (m.window for m in c.marks)
    if end < start:
        return "reject", "the marks are out of order: the end mark's execution phase is before the start mark's"
    t = tail(p.schedule)
    last = end + t
    for i, x in enumerate(e):
        want = start + i
        if not start <= x.phase <= last:
            return "reject", "an execution phase shown is outside the span and its tail"
        if x.phase < want:
            return "reject", "an execution phase is shown twice"
        if x.phase > want:
            return "reject", MISSING
        why = _certified(p, want, x)
        if why is not None:
            return "reject", f"an execution phase shown: {why}"
    n = last - start + 1
    if len(e) < n:
        return "reject", MISSING
    return "accept", public(start, end, t, capacity_bits(p.schedule)), {"link": p.link, "phases": n}


def named(p: Params, c: Commitment) -> dict[str, Any]:
    """What a composition's tie reads off a case's parameters and commitment, meaningful once the count accepts
    them: the SHA-512 of the start certificate's bytes (`Certificate.encode`), which a wipe's signed message names as
    its start mark; the messages the two marks open to; and the bits that can have left over the span and its tail.
    The first three are witness, read off the commitment. No rounding is applied here."""
    start, end = (m.window for m in c.marks)
    return {"start_mark": hashlib.sha512(c.certificates[0]).digest(), "start_message": c.marks[0].mark,
            "end_message": c.marks[1].mark,
            "egress_bits": (end + tail(p.schedule) - start + 1) * capacity_bits(p.schedule) if start <= end else None}
