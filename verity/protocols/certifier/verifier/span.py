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
certificate. A mark's bytes in the commitment are `mark_bytes`: its certificate's bytes (`certificate_bytes`, as a
certified record's bytes hold a certificate, `service.encode`) behind their length, its salt, then its message.

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
execution phase it faulted or refused a connection in, so a span that holds one is rejected."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from verity.commitments import domains
from verity.commitments.crypto import lms
from verity.commitments.identity import tagged_sha512
from verity.protocols.certifier.verifier import warden_rows as WR
from verity.protocols.certifier.verifier.certificate import (DIGEST_BYTES, SALT_BYTES, Certificate, CertifiedMark,
                                                            Direction, LmsKey, marks_link)
from verity.protocols.certifier.verifier.warden import INTEGER_FIELDS, WardenParams as Schedule

ID = domains.CERTIFIER_SPAN_PROTOCOL
COMMITMENT_TAG = domains.CERTIFIER_SPAN
#: what a link's name gains for the link its fails record is certified under
FAILS = "fails"
PARAMS_KEYS = ["key", "link", "lms", "schedule", "session"]
SCHEDULE_KEYS = sorted(INTEGER_FIELDS)
BOUNDED_SCHEDULE_KEYS = sorted([*INTEGER_FIELDS, "max_tokens_cap"])
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


def printable(b: bytes) -> bool:
    """Whether every byte is printable ASCII, 0x20 to 0x7e: what a link's name is here."""
    return all(0x20 <= c < 0x7f for c in b)


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
    the link and the schedule."""
    _exactly(j, PARAMS_KEYS, "the object")
    link = _of(j["link"], str, "link")
    if not all(0x20 <= ord(c) < 0x7f for c in link):
        raise Refused("link: a name is printable ASCII")
    if len(marks_link(link)) >= 1 << 16:
        raise Refused("link: its marks link name is at most 65,535 bytes")
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


def certificate_bytes(c: Certificate) -> bytes:
    """A certificate's bytes, every field fixed-width or prefixed with its length, the certificate's part of
    `service.encode`: u16be len(I) ‖ I ‖ u16be len(link) ‖ link ‖ u8 direction ‖ u64be window ‖ signed digest (64) ‖
    u32be len(signature) ‖ signature. A composition ties a wipe's signed start mark to their SHA-512 (`named`)."""
    # STAND-IN for the issuer's `Certificate.encode`, deleted when this branch is recut on it
    link = c.link.encode()
    return b"".join([struct.pack(">H", len(c.key_id)), c.key_id, struct.pack(">H", len(link)), link,
                     struct.pack(">BQ", c.direction.code, c.window), c.digest, struct.pack(">I", len(c.signature)),
                     c.signature])


def mark_bytes(m: CertifiedMark) -> bytes:
    """A mark's bytes in the commitment: u32be len(certificate) ‖ certificate (`certificate_bytes`) ‖ salt ‖ message."""
    certificate = certificate_bytes(m.certificate)
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


def _certificate(b: bytes) -> Certificate:
    """The certificate `b` is the bytes of (`certificate_bytes`), read exactly: every length fits and nothing is left
    over. A link's name is printable ASCII here."""
    # STAND-IN for the issuer's `Certificate.decode`, deleted when this branch is recut on it
    r = _Reader(b)
    key_id = r.take(r.be(2))
    link = r.take(r.be(2))
    code = r.be(1)
    if code not in (0, 1):
        raise Refused(f"a direction code is 0 or 1, not {code}")
    window, digest = r.be(8), r.take(DIGEST_BYTES)
    signature = r.take(r.be(4))
    if r.at != len(b):
        raise Refused("bytes left after a certificate")
    if not printable(link):
        raise Refused("a mark certificate's link name is not printable ASCII")
    return Certificate(key_id, link.decode(), Direction.EGRESS if code == 0 else Direction.INGRESS, window, digest,
                       signature)


def commitment(b: bytes) -> Commitment:
    """The commitment's two marks, the start's then the end's, each `mark_bytes`, with nothing left over."""
    r, marks, certificates = _Reader(b), [], []
    for _ in MARKS:
        certificates.append(r.take(r.be(4)))
        c = _certificate(certificates[-1])
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
    """The span's tail in execution phases, ⌈(D + ⌈egress_cells / r⌉) / T⌉: a connection's response starts D buckets
    after its admission and is released r cells a bucket, at most egress_cells of them, so what the network certifier
    holds at the end mark has left within D + ⌈egress_cells / r⌉ buckets, T buckets an execution phase."""
    buckets = s.delay + -(-s.egress_cells // s.slots)
    return -(-buckets // s.window)


def public(start: int, end: int, t: int, capacity: int) -> dict[str, int]:
    """What of the count is public, the one place that says so: the bits that can have left over the span and its
    tail, the count of execution phases and the tail. The two marks' execution phases, `start` and `end`, are not:
    they say when the span ran, which a mark's salt hides, and they stay in the commitment."""
    phases = end + t - start + 1
    return {"egress_bits": phases * capacity, "phases": phases, "tail": t}


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


def _certified(p: Params, w: int, x: Phase) -> str | None:
    """Why execution phase `w`'s entry doesn't certify it under the schedule, or None when it does."""
    if len(x.certificates) != len(records(p.link)):
        return (f"{len(x.certificates)} certificates, not its egress record's and its fails record's: "
                f"{NONE_CERTIFIED}")
    for (what, link), c in zip(records(p.link), x.certificates):
        if c.direction is not Direction.EGRESS:
            return f"the {what} certificate is not an egress certificate"
        if c.link != link:
            return f"the {what} certificate is on another link than {link}"
        if c.window != w:
            return f"the {what} certificate is at execution phase {c.window}"
        if not c.verify(p.key, p.session):
            return (f"the {what} certificate does not verify under the key in the certifier session: "
                    f"{NONE_CERTIFIED}")
    doc = x.registration
    if doc["parameters"] != p.schedule.digest.hex():
        return "the egress cells' head names another schedule's digest than the parameters'"
    try:
        WR.check_registration(doc, p.key, p.session, p.schedule, p.link, Direction.EGRESS, w)
    except ValueError as e:
        return f"the egress cells' head: {e}"
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
        return "reject", (f"the marks are out of order: the end mark's execution phase {end} is before the start's "
                          f"{start}")
    t = tail(p.schedule)
    last = end + t
    for i, x in enumerate(e):
        want = start + i
        if not start <= x.phase <= last:
            return "reject", f"execution phase {x.phase} is outside the span and its tail, {start} to {last}"
        if x.phase < want:
            return "reject", f"execution phase {x.phase} is shown twice"
        if x.phase > want:
            return "reject", f"execution phase {want} is missing: {NONE_CERTIFIED}"
        why = _certified(p, want, x)
        if why is not None:
            return "reject", f"execution phase {want}: {why}"
    n = last - start + 1
    if len(e) < n:
        return "reject", f"execution phase {start + len(e)} is missing: {NONE_CERTIFIED}"
    return "accept", public(start, end, t, capacity_bits(p.schedule)), {"link": p.link, "phases": n}


def named(p: Params, c: Commitment) -> dict[str, Any]:
    """What a composition's tie reads off a case's parameters and commitment, meaningful once the count accepts
    them: the SHA-512 of the start certificate's bytes (`certificate_bytes`), which a wipe's signed message names as
    its start mark; the messages the two marks open to; and the bits that can have left over the span and its tail.
    The first three are witness, read off the commitment. No rounding is applied here."""
    start, end = (m.window for m in c.marks)
    return {"start_mark": hashlib.sha512(c.certificates[0]).digest(), "start_message": c.marks[0].mark,
            "end_message": c.marks[1].mark,
            "egress_bits": (end + tail(p.schedule) - start + 1) * capacity_bits(p.schedule) if start <= end else None}
