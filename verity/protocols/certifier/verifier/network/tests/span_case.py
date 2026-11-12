"""The count over a span between two marks (`NetworkCheck.Span`, ``verity/certifier/span/v0``) for every difftest that
runs it: its reference, `reference`, and its cases, `build`.

`reference(case)` is the outcome of one `serve` line in the core interface's answer format
(`verity_difftest.protocol.answer`), from the mirror's decoders and check
(`verity.protocols.certifier.verifier.span`). Its verdict, public, profile and digest are Lean's exactly. So are its
reasons, but for the egress cells' head's own checks, which Lean tells apart and the mirror, whose check of a head is
its row registration's (`WR.check_registration`), gives one reason (`REASONS` has both).

`build(m, seed)` makes a span with one change `m` (`EXPECT`, each aimed at one rejection of the check, or malformed
parameters, commitments, evidence and cases). The span is on one link, its name any UTF-8 (`LINKS`), under a small
sampled schedule
(`period_case._small_schedule`, one in four with a bound on a response's length, its tail at most `TAIL` execution
phases) and a small LMS key, in a sampled certifier session. An `Issuer` on that key certifies the two marks
(`Issuer.mark`, each a sampled 64-byte message under a salt of the issuer's coins), the start's at an execution phase
and the end's at most two later, and signs each execution phase of the span and its tail: its two egress certificates,
the responses' record's and the fails record's, each over a sampled digest (the count opens no record), and its egress
cells' head (`WR.Cells.commit` and `WR.registration`). `far` puts the span at the last execution phases a
certificate can name, and `edge` puts it under the schedule `EDGE`, whose tail is one execution phase and no bucket
less.

This is test-side Python (it imports `verity_difftest`, which the certifier's package may not), so it lives beside
the count's difftest (`test_network_span_protocol.py`)."""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Callable
from typing import Any

import verity_difftest as D
from period_case import KEYS, _cert_json, _flip, _key, _small_schedule, schedule_json
from verity_difftest import protocol as P
from verity.commitments.crypto import lms
from verity.protocols.certifier.verifier import span
from verity.protocols.certifier.verifier import warden_rows as WR
from verity.protocols.certifier.verifier.certificate import (DIGEST_BYTES, SALT_BYTES, Certificate, CertifiedMark,
                                                            Direction, Issuer, certificate_digest, mark_digest,
                                                            marks_link)

ACCEPT, REJECT = "accept", "reject"
Schedule = span.Schedule
EGRESS, INGRESS = Direction.EGRESS, Direction.INGRESS
#: the longest tail of a generated schedule, in execution phases
TAIL = 3
#: A schedule at the edge of its tail: D = 8, r = 2, T = 16 and 17 egress cells a response, as the Lean's `#guard`s'.
#: A response's last cell leaves 8 + ⌊16 / 2⌋ = 16 buckets after its admission, so in the execution phase after the
#: end mark's and never in the one after that: a tail of one execution phase, where a cell more a response makes two.
EDGE = Schedule(cell_bytes=11, slots=2, bucket_ms=100, delay=8, lanes=1, queue=4, window=16, ingress_cells=4,
                ingress_buckets=4, egress_cells=17, every=1)
#: the last execution phase a certificate names: its message holds a u64
LAST = (1 << 64) - 1
#: link names: ASCII, with a space, one byte, long, and outside ASCII in two, three and four UTF-8 bytes a character,
#: with a control character and DEL, which canonical JSON escapes
LINKS = ("eth0", "rack/out", "a b", "l", "x" * 40, "r\u00e9seau/0", "\u7ebf\u8def-1", "\U0001f517", "a\tb\x7f")

EXPECT = {
    "honest": ACCEPT, "far": ACCEPT, "edge": ACCEPT,
    # a mark: on another link name, an ingress certificate, under another key, under the key's identifier with other
    # one-time keys, in another session, not opened by its salt and message, its signature or its execution phase
    # changed; and the two marks out of order
    "mark-link": REJECT, "mark-ingress": REJECT, "mark-key": REJECT, "mark-leaves": REJECT, "mark-session": REJECT,
    "mark-opening": REJECT, "mark-forged": REJECT, "mark-moved": REJECT, "order": REJECT,
    # the span's execution phases: one missing, the tail cut short or left out, one twice, one outside, two out of
    # order, a tail past the last execution phase a certificate can name, and under `EDGE` a tail of two
    "missing": REJECT, "tail-short": REJECT, "no-tail": REJECT, "twice": REJECT, "outside": REJECT,
    "disorder": REJECT, "past-u64": REJECT, "edge-tail": REJECT,
    # an execution phase's certificates: none (as the network certifier leaves one it faulted or refused a connection
    # in), one or three, one on another link name, at another execution phase, an ingress one, in another session,
    # under another key, forged
    "faulted": REJECT, "cert-count": REJECT, "cert-link": REJECT, "cert-phase": REJECT, "cert-ingress": REJECT,
    "cert-session": REJECT, "cert-key": REJECT, "cert-forged": REJECT,
    # an execution phase's egress cells' head: under another schedule (the head's, or the parameters'), another
    # execution phase's, another link's, the ingress cells', in another session, under another key, forged, a field
    # changed
    "head-schedule": REJECT, "schedule": REJECT, "head-phase": REJECT, "head-link": REJECT, "head-ingress": REJECT,
    "head-session": REJECT, "head-key": REJECT, "head-forged": REJECT, "head-field": REJECT,
    "params": REJECT, "commitment": REJECT, "evidence": REJECT, "case": REJECT,
}
MUTATIONS = list(EXPECT)

_MARK = {m: f"mark: {span.REFUSALS[code]}" for m, code in {
    "mark-link": "link", "mark-ingress": "egress", "mark-key": "key", "mark-leaves": "signature",
    "mark-session": "signature", "mark-opening": "opening", "mark-forged": "signature",
    "mark-moved": "signature"}.items()}
_UNSIGNED = "certificate does not verify under the key in the certifier session"
_HEAD = "the egress cells' head: "
_REFUSED = "the egress cells' head is not the link's at its execution phase under the schedule"
REASONS: dict[str, tuple[str, str]] = {m: (why, why) for m, why in ({
    "order": "the marks are out of order: the end mark's execution phase is before the start mark's",
    "missing": span.MISSING, "tail-short": span.MISSING, "no-tail": span.MISSING,
    "twice": "an execution phase is shown twice", "outside": "an execution phase shown is outside the span and its tail",
    "edge-tail": "an execution phase shown is outside the span and its tail",
    "past-u64": f"{_UNSIGNED}: {span.NONE_CERTIFIED}",
    "faulted": f"not two certificates, its egress record's and its fails record's: {span.NONE_CERTIFIED}",
    "cert-count": "not two certificates, its egress record's and its fails record's",
    "cert-link": "certificate is on another link than", "cert-phase": "certificate is at another execution phase",
    "cert-ingress": "certificate is not an egress certificate",
    "cert-session": _UNSIGNED, "cert-key": _UNSIGNED, "cert-forged": _UNSIGNED,
    "head-schedule": "the egress cells' head names another schedule's digest than the parameters'",
    "schedule": "the egress cells' head names another schedule's digest than the parameters'",
} | _MARK).items()} | {
    "head-phase": (f"{_HEAD}not the link's egress rows at its execution phase", _REFUSED),
    "head-link": (f"{_HEAD}not the link's egress rows at its execution phase", _REFUSED),
    "head-ingress": (f"{_HEAD}not the link's egress rows at its execution phase", _REFUSED),
    "head-session": (f"{_HEAD}not the execution phase's domain", _REFUSED),
    "head-key": (f"{_HEAD}its certificate does not verify under the key in the certifier session", _REFUSED),
}
"""Each rejection of the check a change is aimed at, by what Lean's reason and the mirror's say of it. Neither holds a
number: a reason names no execution phase."""


def _refusing(f: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def decode(x: Any) -> Any:
        try:
            return f(x)
        except span.Refused as e:
            raise D.Reject(str(e)) from None
    return decode


def reference(case: Any) -> dict:
    """The mirror's outcome of one `serve` case."""
    return P.answer(span.ID, case, params=_refusing(span.params), commitment=_refusing(span.commitment),
                    evidence=_refusing(lambda b: span.evidence(P.canonical_json(b))), check=span.check,
                    digest=span.digest)


class _Leaves:
    """An `Issuer`'s signer without leaf state: it starts any session, and its n-th signature is at leaf n mod 2^H."""

    def __init__(self, key: lms.PrivateKey) -> None:
        self.key, self.n = key, 0

    @property
    def key_id(self) -> bytes:
        return self.key.I

    def start(self, session: bytes) -> None:
        pass

    def sign(self, message: bytes) -> bytes:
        self.n += 1
        return lms.sign_at(self.key, (self.n - 1) % 2 ** self.key.params.H, message).encode()


def _schedule(r: random.Random) -> Schedule:
    """A small schedule, one in four with a bound on a response's length, whose tail is at most `TAIL` execution
    phases."""
    while True:
        s = _small_schedule(r, False, r.random() < 0.25)
        if span.tail(s) <= TAIL:
            return s


def _other_schedule(r: random.Random, s: Schedule) -> Schedule:
    """`s` with one field a step up: a schedule of the same rows and another digest."""
    f = r.choice(["delay", "queue", "bucket_ms", "egress_cells", "ingress_cells", "ingress_buckets"])
    return dataclasses.replace(s, **{f: getattr(s, f) + 1})


def build(m: str, seed: int) -> dict:
    """A span with the one change `m` (`EXPECT`), sampled from `seed`."""
    if m not in EXPECT:
        raise ValueError(f"a change is one of {MUTATIONS}, not {m!r}")
    r = random.Random(f"network-span {m} {seed}")
    i = r.randrange(len(KEYS))
    N, W, H, toy = KEYS[i]
    key = _key(i)
    session, link = r.randbytes(lms.SESSION_BYTES), r.choice(LINKS)
    elsewhere = r.choice([x for x in LINKS if x != link])
    sched = EDGE if m in ("edge", "edge-tail") else _schedule(r)
    t, length = span.tail(sched), r.randint(0, 2)
    if m == "far":
        start = LAST - t - length - r.randint(0, 2)
    elif m == "past-u64":
        start = LAST - length - r.randrange(t)
    else:
        start = r.randint(0, 6)
    end = start + length
    last = end + t

    def issuer(k: lms.PrivateKey = key, ses: bytes = session) -> Issuer:
        return Issuer(_Leaves(k), ses, coins=r.randbytes)

    own, stranger, twin = issuer(), issuer(_key(i, 1)), issuer(_key(i, 1, same_id=True))
    other_session = _flip(session, r.randrange(len(session)))
    moved = issuer(ses=other_session)

    def marked(w: int, name: str = marks_link(link), d: Direction = EGRESS, by: Issuer = own,
               digest: Callable[[bytes, bytes], bytes] = mark_digest) -> CertifiedMark:
        """A mark at `w` whose certificate is on the link name `name` in direction `d` and signs `digest` of its salt
        and message."""
        salt, message = r.randbytes(SALT_BYTES), r.randbytes(DIGEST_BYTES)
        return CertifiedMark(message, salt, by.sign(name, d, w, digest(salt, message)))

    marks = [own.mark(link, w, r.randbytes(DIGEST_BYTES)) for w in (start, end)]
    which = r.randrange(2)
    at, old = (start, end)[which], marks[which]
    if m == "mark-link":
        marks[which] = marked(at, name=r.choice([link, span.fails_link(link), marks_link(elsewhere)]))
    elif m == "mark-ingress":
        marks[which] = marked(at, d=INGRESS)
    elif m == "mark-key":
        marks[which] = marked(at, by=stranger)
    elif m == "mark-leaves":
        marks[which] = marked(at, by=twin)
    elif m == "mark-session":
        marks[which] = marked(at, by=moved)
    elif m == "mark-opening":
        n = r.randrange(DIGEST_BYTES)
        marks[which] = r.choice([
            dataclasses.replace(old, salt=_flip(old.salt, n)), dataclasses.replace(old, mark=_flip(old.mark, n)),
            dataclasses.replace(old, salt=old.mark, mark=old.salt), marked(at, digest=certificate_digest)])
    elif m == "mark-forged":
        c = old.certificate
        marks[which] = dataclasses.replace(old, certificate=dataclasses.replace(
            c, signature=_flip(c.signature, r.randrange(len(c.signature)))))
    elif m == "mark-moved":
        c = old.certificate
        marks[which] = dataclasses.replace(old, certificate=dataclasses.replace(
            c, window=c.window + r.choice([1, -1] if c.window else [1])))
    elif m == "order":
        marks = [own.mark(link, w, r.randbytes(DIGEST_BYTES)) for w in (start + r.randint(1, 3), start)]

    def certificates(w: int, on: str = link, by: Issuer = own, d: Direction = EGRESS) -> list[Certificate]:
        return [by.sign(name, d, w, r.randbytes(DIGEST_BYTES)) for _, name in span.records(on)]

    def head(w: int, s: Schedule = sched, on: str = link, d: Direction = EGRESS, by: Issuer = own,
             ses: bytes = session) -> dict:
        cells = WR.Cells.commit(on, d, w, [r.randbytes(WR.row_bytes(s)) for _ in range(s.window)],
                                WR.Wiring((), r.randbytes(WR.WIRING_SALT_BYTES)), s, ses,
                                [r.randbytes(WR.SALT_BYTES) for _ in range(s.window)])
        return WR.registration(cells.head, by.sign(on, d, w, cells.head.digest()), s)

    def phase(w: int, certs: list[Certificate] | None = None, registration: dict | None = None) -> dict:
        return {"certificates": [_cert_json(c) for c in (certificates(w) if certs is None else certs)], "phase": w,
                "registration": head(w) if registration is None else registration}

    phases = [phase(w) for w in range(start, min(last, LAST) + 1)]
    # the execution phases past the last a certificate can name: no certificate of one verifies
    phases += [phase(w, [dataclasses.replace(c, window=w) for c in certificates(LAST)], head(LAST))
               for w in range(LAST + 1, last + 1)]
    k, j = r.randrange(len(phases)), r.randrange(2)
    w, name = start + k, span.records(link)[j][1]
    if m == "missing":
        del phases[k]
    elif m == "tail-short":
        del phases[len(phases) - r.randint(1, t):]
    elif m == "no-tail":
        del phases[len(phases) - t:]
    elif m == "twice":
        phases.insert(k, phases[k])
    elif m == "edge-tail":
        phases.append(phase(last + 1))
    elif m == "outside":
        if start and r.random() < 0.5:
            phases.insert(0, phase(start - 1))
        else:
            phases.append(phase(last + 1))
    elif m == "disorder":
        a = r.randrange(len(phases) - 1)
        phases[a], phases[a + 1] = phases[a + 1], phases[a]
    elif m == "faulted":
        phases[k]["certificates"] = []
    elif m == "cert-count":
        cs = phases[k]["certificates"]
        requests = _cert_json(own.sign(link, INGRESS, w, r.randbytes(DIGEST_BYTES)))
        phases[k]["certificates"] = r.choice([cs[:j] + cs[j + 1:], [*cs, cs[j]], [requests, *cs]])
    elif m in ("cert-link", "cert-phase", "cert-ingress", "cert-session", "cert-key", "cert-forged"):
        cs = certificates(w)
        if m == "cert-link":
            cs = r.choice([
                cs[::-1], cs[:j] + [certificates(w, on=elsewhere)[j]] + cs[j + 1:],
                cs[:j] + [own.sign(marks_link(link), EGRESS, w, r.randbytes(DIGEST_BYTES))] + cs[j + 1:]])
        elif m == "cert-phase":
            cs[j] = certificates(w + 1)[j]
        elif m == "cert-ingress":
            cs[j] = own.sign(name, INGRESS, w, r.randbytes(DIGEST_BYTES))
        elif m == "cert-session":
            cs[j] = certificates(w, by=moved)[j]
        elif m == "cert-key":
            cs[j] = certificates(w, by=r.choice([stranger, twin]))[j]
        else:
            f = r.choice(["digest", "signature"])
            cs[j] = dataclasses.replace(cs[j], **{f: _flip(getattr(cs[j], f), r.randrange(len(getattr(cs[j], f))))})
        phases[k] = phase(w, cs, phases[k]["registration"])
    elif m == "head-schedule":
        phases[k]["registration"] = head(w, s=_other_schedule(r, sched))
    elif m == "head-phase":
        phases[k]["registration"] = head(w + 1)
    elif m == "head-link":
        phases[k]["registration"] = head(w, on=elsewhere)
    elif m == "head-ingress":
        phases[k]["registration"] = head(w, d=INGRESS)
    elif m == "head-session":
        phases[k]["registration"] = head(w, by=moved, ses=other_session)
    elif m == "head-key":
        phases[k]["registration"] = head(w, by=r.choice([stranger, twin]))
    elif m == "head-forged":
        reg = phases[k]["registration"]
        f = r.choice(["digest", "signature"])
        b = bytes.fromhex(reg["certificate"][f])
        reg["certificate"] = reg["certificate"] | {f: _flip(b, r.randrange(len(b))).hex()}
    elif m == "head-field":
        reg = phases[k]["registration"]
        f = r.choice(["root", "wiring", "domain", "rows", "words", "word_bits", "cell_bytes", "window", "value",
                      "format", "row_schema", "link", "direction", "layout"])
        if f == "layout":
            x = r.choice(sorted(reg[f]))
            reg[f] = reg[f] | {x: reg[f][x] + 1}
        elif isinstance(reg[f], int):
            reg[f] += 1
        elif f in ("root", "wiring", "domain"):
            b = bytes.fromhex(reg[f])
            reg[f] = _flip(b, r.randrange(len(b))).hex()
        else:
            reg[f] = {"link": elsewhere, "direction": INGRESS.value}.get(f, reg[f] + "x")

    params = {"key": lms.public_key(key).hex(), "link": link, "lms": {"H": H, "N": N, "W": W, "toy": toy},
              "schedule": schedule_json(_other_schedule(r, sched) if m == "schedule" else sched),
              "session": session.hex()}
    e_json = {"phases": phases}
    c_bytes, e_bytes = b"".join(span.mark_bytes(x) for x in marks), D.canonical(e_json).encode()
    if m == "params":
        params = r.choice(_bad_params(params, seed))
    elif m == "commitment":
        c_bytes = r.choice(_bad_commitments(c_bytes, marks))
    elif m == "evidence":
        e_bytes = r.choice(_bad_evidence(e_json, e_bytes, seed))
    case = {"commitment": c_bytes.hex(), "evidence": e_bytes.hex(), "params": params}
    if m == "case":
        case = r.choice([case | {"commitment": case["commitment"].upper()}, case | {"evidence": case["evidence"] + "0"},
                         case | {"commitment": 0}, {x: v for x, v in case.items() if x != "params"}, [case]])
    return case


def _without(j: dict, n: int) -> dict:
    return {k: v for k, v in j.items() if k != sorted(j)[n % len(j)]}


def _bad_params(p: dict, n: int) -> list:
    """Parameters `span.params` refuses, each one change to `p`: among them a link's name a byte past the most a name
    has, in ASCII and in two-byte characters. The last two it reads, and the check rejects them, their marks being on
    another link's marks link name: a name of the most bytes a name has, and the link's name with a character more."""
    s, most = p["schedule"], (1 << 16) - 1
    return [_without(p, n), p | {"x": 0}, [p], None,
            p | {"link": 1}, p | {"link": None}, p | {"link": "x" * (most + 1)}, p | {"link": "\u00e9" * (most // 2 + 1)},
            p | {"key": p["key"][:-2]}, p | {"key": p["key"] + "00"}, p | {"key": p["key"].upper()},
            p | {"key": p["key"][:-1]}, p | {"session": p["session"][:-2]}, p | {"session": ""}, p | {"session": 0},
            p | {"lms": p["lms"] | {"W": 3}}, p | {"lms": p["lms"] | {"H": 0}}, p | {"lms": p["lms"] | {"toy": 0}},
            p | {"lms": p["lms"] | {"x": 0}}, p | {"lms": _without(p["lms"], n)},
            p | {"schedule": {k: v for k, v in s.items() if k != span.SCHEDULE_KEYS[n % len(span.SCHEDULE_KEYS)]}},
            p | {"schedule": s | {"lengths": 1}},
            p | {"schedule": s | {sorted(s)[n % len(s)]: 0}}, p | {"schedule": s | {"cell_bytes": 10}},
            p | {"schedule": s | {"cell_bytes": 1 << 32}}, p | {"schedule": s | {"window": 2, "every": 3}},
            p | {"schedule": s | {"delay": True}}, p | {"schedule": s | {"delay": "1"}}, p | {"schedule": [s]},
            p | {"link": "x" * most}, p | {"link": p["link"] + "\u00e9"}]


def _bad_commitments(b: bytes, marks: list[CertifiedMark]) -> list[bytes]:
    """Bytes `span.commitment` refuses, each one change to `b`, the two marks' bytes: bytes missing or left over, a
    mark missing or one too many; in the first certificate, a link's name that is not UTF-8 (a byte no UTF-8 holds, a
    continuation byte first, an overlong form, a surrogate, a character past the last, a character cut short at the
    name's end), a direction that is none, a length past the certificate's end (its key id's, its link name's, its
    signature's, its own), and a byte left after it (its signature a byte shorter, or its own length a byte longer);
    and its own length a byte short."""
    one, c = span.mark_bytes(marks[0]), marks[0].certificate
    size = len(c.encode())
    link = 4 + 2 + len(c.key_id) + 2
    direction = link + len(c.link.encode())
    signature = direction + 1 + 8 + DIGEST_BYTES

    def put(at: int, x: bytes) -> bytes:
        return b[:at] + x + b[at + len(x):]

    def u32(n: int) -> bytes:
        return n.to_bytes(4, "big")
    return [b"", b[:1], b[:-1], b + b"\0", one, b + one,
            put(link, b"\xff"), put(link, b"\x80"), put(link, b"\xc0\xaf"), put(link, b"\xed\xa0\x80"),
            put(link, b"\xf4\x90\x80\x80"), put(direction - 1, b"\xc3"), put(direction, b"\x02"),
            put(4, b"\xff\xff"), put(link - 2, b"\xff\xff"), put(signature, u32((1 << 32) - 1)),
            put(0, u32((1 << 32) - 1)), put(signature, u32(len(c.signature) - 1)),
            u32(size + 1) + b[4:4 + size] + b"\0" + b[4 + size:], put(0, u32(size - 1))]


def _bad_evidence(j: dict, b: bytes, n: int) -> list[bytes]:
    """Bytes `span.evidence` refuses, each one change to `b`, the canonical JSON of `j`, whose first execution phase
    holds its certificates."""
    x = j["phases"][0]
    c, reg = x["certificates"][0], x["registration"]

    def first(**changes: Any) -> dict:
        return {"phases": [x | changes, *j["phases"][1:]]}

    def certificate(**changes: Any) -> dict:
        return first(certificates=[c | changes, *x["certificates"][1:]])

    def registration(**changes: Any) -> dict:
        return first(registration=reg | changes)
    docs = [j | {"x": 0}, {}, {"phases": {}}, [j], {"phases": [[]]},
            {"phases": [_without(x, n), *j["phases"][1:]]}, first(x=0), first(phase=-1), first(phase=True),
            first(phase="0"), first(certificates={}), first(certificates=[[]]),
            first(certificates=[_without(c, n)]), certificate(x=0), certificate(direction="out"),
            certificate(direction=0), certificate(digest=c["digest"].upper()), certificate(digest=c["digest"][:-1]),
            certificate(window=-1), certificate(link=0), certificate(key_id=None),
            first(registration=_without(reg, n)), registration(x=0), registration(rows="1"), registration(value=0),
            registration(root=reg["root"].upper()), registration(layout=_without(reg["layout"], n)),
            registration(layout=reg["layout"] | {"slots": "1"}),
            registration(certificate=_without(reg["certificate"], n)), first(registration=[reg])]
    return [b"", b"null", b[:-1], b + b" ", b.replace(b":", b": ", 1), b[:1] + b'"\x80":0,' + b[1:],
            *(D.canonical(d).encode() for d in docs)]
