"""The network certifier's period check (`NetworkCheck.Period`, `verity/certifier/v0`) in Python, for every difftest
that runs it: its mirror, `answer` (as `reference`), and its cases, `build` and `case`.

`answer(case)` is the outcome of one `serve` line in the core interface's answer format
(`verity_difftest.protocol.answer`): it decodes the protocol's formats (`NetworkCheck/Period.lean`'s module docstring)
and checks a period with the Python the period's pieces already have: one-stage's answers and integrity profiles as PoUW
reads them (`pouw.core_check.answer_of` and `integrity`), the network certifier's certificates and clock certificates
under its LMS key (`certifier.certificate`), Σ_sync (`certifier.schedule`), `registration.digest`, and in the rows form
the schedule (`warden.WardenParams`) and the verifier's read of each row registration
(`warden_rows.check_registration`). Its verdict, public, profile and digest are Lean's exactly; its reason is its own.
Public is `K` and its terms, each audit's (and in the rows form each scan's) with its own `delta`, and `eps`, the union
of those deltas as an exact ratio string.

`build(m, seed, small)` makes a period with one change `m` (`EXPECT`, each aimed at one path of the check, or malformed
parameters, commitments, evidence and cases). In the clock form: the link-windows of one link from the session's window
`first`, each registered under the link's Program with its provenance, certified at both ends by a small LMS key in a
drawn network certifier session, its clock certificate signing a sync of Σ_sync salted (`clock_commitment`, the sync and
the salt in no case), and answered by one-stage with its one unit proved. In the rows form (a change named `rows-…`):
each link-window's ingress and egress rows committed and registered under a small drawn schedule
(`warden_rows.Cells.commit` and `registration`, signed by the same key), its three records certified, and the scan's
registration holding both rows' roots and domains, answered by one-stage with a stratified or subset draw. Beside them,
the compute audits, each answered with a subset or stratified draw (a work law's) and its bound on wrong units.
`case(kind, seed, form)` is a small period of the form aimed at a verdict, or at a decoding failure (`KINDS`), and
`cases(seed, form)` one of each with its answer, as a composition's difftest takes a real component's cases.

This is test-side Python (it imports `verity_difftest` and PoUW's `core_check`, which the certifier's package may not),
so it lives beside the period's difftest (`test_network_period_protocol.py`), and other suites load it by path."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import re
from fractions import Fraction
from functools import lru_cache

import verity_difftest as D
from verity_difftest import protocol as P
from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.certifier.certificate import (SALT_BYTES, Certificate, Direction, LmsKey,
                                                            certificate_digest, clock_digest)
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet
from verity.core.protocols.certifier.warden import WardenParams
from verity.core.protocols.pouw import core_check as CC
from verity.ml.commitments.identity import tagged_sha512
from verity.ml.crypto import lms

ID = "verity/certifier/v0"
COMMITMENT_TAG = "verity-network-certifier/period/v0"
ONE_STAGE = "verity/sampled-proofs/one-stage/v1"
FORMATS = ("verity/registration/v1", "verity/registration/v2")
#: a proof unit's outputs are at most 16 bits
OUT_BITS = 16
UNIT = "unit"
EGRESS, INGRESS = Direction.EGRESS, Direction.INGRESS
ACCEPT, REJECT, UNDECIDED = "accept", "reject", "undecided"
PARAMS_KEYS = ["audits", "cells", "first", "jitter", "key", "link", "lms", "program", "session", "sync_set", "windows"]
ROWS_PARAMS_KEYS = ["audits", "cells", "first", "jitter", "key", "link", "lms", "program", "scan", "schedule",
                    "session", "windows"]
SCHEDULE_KEYS = sorted(f.name for f in dataclasses.fields(WardenParams))
CERTIFICATE_KEYS = ["digest", "direction", "key_id", "link", "signature", "window"]
CLOCK_WINDOW_KEYS = ["answer", "certificates", "clock"]
ROWS_WINDOW_KEYS = ["answer", "certificates", "registrations"]
ROWS_KEYS = ["cell_bytes", "certificate", "direction", "domain", "format", "layout", "link", "parameters", "root",
             "row_schema", "rows", "value", "window", "wiring", "word_bits", "words"]
LAYOUT_KEYS = ["buckets", "connection_slots", "first_bucket", "slots"]
DIGEST = re.compile("[0-9a-f]{128}")
U32 = 1 << 32
FORMS = ("clock", "rows")

# --- the mirror --------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Params:
    audits: tuple[tuple[int, int], ...]
    cells: tuple[int, ...]
    first: int
    jitter: int
    link: str
    key: LmsKey
    session: bytes
    program: str
    #: the clock form's Σ_sync; None in the rows form
    sync_set: SyncSet | None
    windows: int
    #: the rows form's schedule and the scan's layout terms (free, steps); None in the clock form
    schedule: WardenParams | None = None
    scan: tuple[int, int] | None = None


@dataclasses.dataclass(frozen=True)
class Registration:
    record: dict
    digest: bytes


@dataclasses.dataclass(frozen=True)
class Commitment:
    audits: list[Registration]
    windows: list[Registration]
    bytes: bytes


@dataclasses.dataclass(frozen=True)
class Window:
    answer: dict
    certificates: list[Certificate]
    #: the clock form's clock certificate, or the rows form's registrations
    clock: Certificate | None
    registrations: list[dict] | None = None


@dataclasses.dataclass(frozen=True)
class Facts:
    units: int
    drawn: int
    delta: Fraction
    wrong: int


def _exactly(j, keys: list[str], ctx: str) -> None:
    if not isinstance(j, dict):
        raise D.Reject(f"{ctx}: not an object")
    if sorted(j) != keys:
        raise D.Reject(f"{ctx}: the keys must be {keys}, not {sorted(j)}")


def _nat(v, ctx: str) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
        raise D.Reject(f"{ctx}: not a natural number")
    return v


def _of(v, kind, ctx: str):
    if not isinstance(v, kind) or isinstance(v, bool) != (kind is bool):
        raise D.Reject(f"{ctx}: not a {kind.__name__}")
    return v


def _rho(j, ctx: str) -> Fraction:
    if not isinstance(j, list) or len(j) != 2:
        raise D.Reject(f"{ctx}: not [numerator, denominator]")
    n, d = (_nat(x, ctx) for x in j)
    if n < 1 or d < 1:
        raise D.Reject(f"{ctx}: a numerator and a denominator of at least 1")
    return Fraction(n, d)


def _sync_set(j) -> SyncSet:
    _exactly(j, ["deltas", "rhos"], "sync_set")
    deltas = [_nat(x, "sync_set.deltas") for x in _of(j["deltas"], list, "sync_set.deltas")]
    rhos = [_rho(x, "sync_set.rhos") for x in _of(j["rhos"], list, "sync_set.rhos")]
    if not deltas or not rhos or len(set(deltas)) < len(deltas) or len(set(rhos)) < len(rhos):
        raise D.Reject("sync_set: non-empty, each delta and each rho listed once")
    return SyncSet(tuple(deltas), tuple(rhos))


def _audit(a, ctx: str) -> tuple[int, int]:
    _exactly(a, ["free", "steps"], ctx)
    return _nat(a["free"], ctx), _nat(a["steps"], ctx)


def _schedule(j) -> WardenParams:
    """`scheduleOf`: `WardenParams`' fields, each a positive integer, B past a cell's header, and B, S and T u32s."""
    _exactly(j, SCHEDULE_KEYS, "schedule")
    values = {k: _nat(j[k], "schedule") for k in SCHEDULE_KEYS}
    try:
        s = WardenParams(**values)
    except ValueError as e:
        raise D.Reject(f"schedule: {e}") from None
    if not (s.cell_bytes < U32 and s.lanes * s.slots < U32 and s.window < U32):
        raise D.Reject("schedule: B, S = lanes · slots and T = window are u32s")
    return s


def params(j) -> Params:
    if not isinstance(j, dict):
        raise D.Reject("the object: not an object")
    if sorted(j) not in (PARAMS_KEYS, ROWS_PARAMS_KEYS):
        raise D.Reject(f"the object: the keys must be {PARAMS_KEYS} or {ROWS_PARAMS_KEYS}, not {sorted(j)}")
    audits = [_audit(a, "an audit") for a in _of(j["audits"], list, "audits")]
    cells = tuple(_nat(x, "cells") for x in _of(j["cells"], list, "cells"))
    first = _nat(j["first"], "first")
    jitter = _nat(j["jitter"], "jitter")
    if jitter < 1:
        raise D.Reject("jitter: the observations' jitter bound is at least 1")
    link = _of(j["link"], str, "link")
    if len(link.encode()) >= 1 << 16:
        raise D.Reject("link: a name is at most 65,535 UTF-8 bytes")
    lj = j["lms"]
    _exactly(lj, ["H", "N", "W", "toy"], "lms")
    try:
        lp = lms.Params(*(_nat(lj[k], "lms") for k in ("N", "W", "H")), toy=_of(lj["toy"], bool, "lms"))
    except ValueError as x:
        raise D.Reject(str(x)) from None
    key = P.unhex(_of(j["key"], str, "key"))
    if len(key) != 16 + lp.N:
        raise D.Reject(f"an LMS public key is {16 + lp.N} bytes")
    session = P.unhex(_of(j["session"], str, "session"))
    if len(session) != lms.SESSION_BYTES:
        raise D.Reject(f"a certifier session id is {lms.SESSION_BYTES} bytes")
    program = _of(j["program"], str, "program")
    if not DIGEST.fullmatch(program):
        raise D.Reject("program: not a SHA-512 digest")
    if "sync_set" in j:
        sync_set, schedule, scan = _sync_set(j["sync_set"]), None, None
    else:
        sync_set, schedule, scan = None, _schedule(j["schedule"]), _audit(j["scan"], "scan")
    windows = _nat(j["windows"], "windows")
    if windows < 1:
        raise D.Reject("windows: a period has at least one link-window")
    return Params(tuple(audits), cells, first, jitter, link, LmsKey(lp, key), session, program, sync_set, windows,
                  schedule, scan)


def registration_digest(record: dict) -> bytes:
    """`registration.digest`: SHA-512(format ‖ 0 ‖ the canonical record)."""
    return hashlib.sha512(record["format"].encode() + b"\0" + D.canonical(record).encode()).digest()


def _registration(j, ctx: str) -> Registration:
    if not isinstance(j, dict) or not isinstance(j.get("format"), str):
        raise D.Reject(f"{ctx}: no format")
    if j["format"] not in FORMATS:
        raise D.Reject(f"{ctx}: format is not {' or '.join(FORMATS)}")
    return Registration(j, registration_digest(j))


def commitment(b: bytes) -> Commitment:
    j = P.canonical_json(b)
    _exactly(j, ["audits", "windows"], "the object")
    return Commitment([_registration(x, "an audit's registration") for x in _of(j["audits"], list, "audits")],
                      [_registration(x, "a link-window's registration") for x in _of(j["windows"], list, "windows")], b)


def _answer(j, ctx: str) -> dict:
    try:
        return CC.answer_of(j, ctx)
    except CC.Refused as e:
        raise D.Reject(str(e)) from None


def _certificate(j, ctx: str) -> Certificate:
    _exactly(j, CERTIFICATE_KEYS, ctx)
    try:
        direction = Direction(_of(j["direction"], str, ctx))
    except ValueError:
        raise D.Reject(f"{ctx}: not a direction") from None
    return Certificate(P.unhex(_of(j["key_id"], str, ctx)), _of(j["link"], str, ctx), direction,
                       _nat(j["window"], ctx), P.unhex(_of(j["digest"], str, ctx)),
                       P.unhex(_of(j["signature"], str, ctx)))


def _rows(j, ctx: str) -> dict:
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
        P.unhex(_of(o[k], str, f"{ctx}.{k}"))
    return j


def _window(j, ctx: str) -> Window:
    clock = isinstance(j, dict) and sorted(j) == CLOCK_WINDOW_KEYS
    if not clock:
        _exactly(j, ROWS_WINDOW_KEYS, ctx)
    answer = _answer(j["answer"], f"{ctx}.answer")
    certificates = [_certificate(c, f"{ctx}.certificates") for c in _of(j["certificates"], list, ctx)]
    if clock:
        return Window(answer, certificates, _certificate(j["clock"], f"{ctx}.clock"))
    return Window(answer, certificates, None,
                  [_rows(x, f"{ctx}.registrations") for x in _of(j["registrations"], list, ctx)])


def evidence(b: bytes) -> tuple[list[dict], list[Window]]:
    j = P.canonical_json(b)
    _exactly(j, ["audits", "windows"], "the object")
    return ([_answer(a, "an audit's answer") for a in _of(j["audits"], list, "audits")],
            [_window(x, "a link-window") for x in _of(j["windows"], list, "windows")])


class Finding(Exception):
    """A finding against the period: a rejection, or what leaves it undecided."""

    def __init__(self, verdict: str, reason: str):
        super().__init__(reason)
        self.verdict, self.reason = verdict, reason


def _certifies_on(p: Params, link: str, d: Direction, w: int, c: Certificate) -> bool:
    return c.direction is d and c.link == link and c.window == w and c.verify(p.key, p.session)


def _certifies(p: Params, d: Direction, w: int, c: Certificate) -> bool:
    return _certifies_on(p, p.link, d, w, c)


def _clock_digest(s: ClockSync) -> bytes | None:
    try:
        return clock_digest(s)
    except ValueError:
        return None


def clear_clocks(ss: SyncSet) -> set[bytes]:
    """Each sync of Σ_sync's `clock_digest` itself: the unsalted form, whose digest shows the sync to anyone holding
    Σ_sync."""
    return {d for d in (_clock_digest(ClockSync(x, rho)) for x in ss.deltas for rho in ss.rhos) if d is not None}


def _answered(what: str, reg: Registration, a: dict) -> tuple[dict, object]:
    if a["protocol"] != ONE_STAGE:
        raise Finding(REJECT, f"{what}: protocol {a['protocol']}")
    if a["commitment"] != reg.digest.hex():
        raise Finding(REJECT, f"{what}: the answer is for another registration")
    if a["verdict"] == ACCEPT:
        try:
            return CC.integrity(a["profile"]), a["public"]
        except CC.Refused as e:
            raise Finding(REJECT, f"{what}: {e}") from None
    raise Finding(a["verdict"], f"{what}: service: {a['reason']}")


def served(w: int) -> dict:
    return {"kind": "served", "window": w}


def _window_check(p: Params, clear: set[bytes], earlier: list[bytes], w: int, reg: Registration, x: Window) -> None:
    what = f"window {w}"
    if x.clock is None:
        raise Finding(REJECT, f"{what}: registrations, not a clock certificate")
    if not (reg.record.get("program") == p.program
            and D.canonical(reg.record.get("window")) == D.canonical(served(w))):
        raise Finding(REJECT, f"{what}: registration")
    if not (len(x.certificates) == 2
            and all(_certifies(p, d, w, c) for c, d in zip(x.certificates, (EGRESS, INGRESS)))):
        raise Finding(REJECT, f"{what}: certificates")
    if not _certifies(p, EGRESS, w, x.clock):
        raise Finding(REJECT, f"{what}: clock")
    if x.clock.digest in clear:
        raise Finding(REJECT, f"{what}: clock: unsalted")
    if x.clock.digest in earlier:
        raise Finding(REJECT, f"{what}: clock: reused")
    prof, _ = _answered(what, reg, x.answer)
    if ([(lv["name"], lv["size"], lv["draw"]) for lv in prof["levels"]] != [(UNIT, 1, {"subset": 1})]
            or prof["drawn"] != [(UNIT, 1)]):
        raise Finding(REJECT, f"{what}: profile")


def _rows_check(p: Params, w: int, d: Direction, doc: dict) -> None:
    """`rowsCheck`: the verifier's own read of the registration (`warden_rows.check_registration`)."""
    try:
        WR.check_registration(doc, p.key, p.session, p.schedule, p.link, d, w)
    except ValueError as e:
        raise Finding(REJECT, f"window {w}: the {d.value} rows: {e}") from None


def _holds(reg: Registration, doc: dict) -> bool:
    """The scan's registration holds the rows: an entry of its `roots` or `registered` names their root and domain."""
    entries = [e for k in ("roots", "registered") if isinstance(reg.record.get(k), list) for e in reg.record[k]]
    return any(isinstance(e, dict) and e.get("root") == doc["root"] and e.get("domain") == doc["domain"]
               for e in entries)


def records(link: str) -> tuple[tuple[str, Direction], ...]:
    """The rows form's certified records of a link-window (`warden_serve.CERTIFIED`): requests, responses, fails."""
    return (link, INGRESS), (link, EGRESS), (f"{link}-fails", EGRESS)


def _rows_window_check(p: Params, w: int, reg: Registration, x: Window) -> Facts:
    what = f"window {w}"
    if x.registrations is None:
        raise Finding(REJECT, f"{what}: a clock certificate, not registrations")
    if reg.record.get("program") != p.program:
        raise Finding(REJECT, f"{what}: registration")
    if not (len(x.certificates) == 3
            and all(_certifies_on(p, link, d, w, c) for c, (link, d) in zip(x.certificates, records(p.link)))):
        raise Finding(REJECT, f"{what}: certificates")
    if len(x.registrations) != 2:
        raise Finding(REJECT, f"{what}: registrations")
    for doc, d in zip(x.registrations, (INGRESS, EGRESS)):
        _rows_check(p, w, d, doc)
    if not all(_holds(reg, doc) for doc in x.registrations):
        raise Finding(REJECT, f"{what}: another window's answer")
    prof, pub = _answered(what, reg, x.answer)
    return _facts(what, prof, pub)


def _facts(what: str, prof: dict, pub) -> Facts:
    """`factsOf`: one level drawn `subset:k` or stratified, and the public's bound."""
    if len(prof["levels"]) != 1:
        raise Finding(REJECT, f"{what}: profile: not one level")
    (lv,) = prof["levels"]
    draw = lv["draw"]
    if isinstance(draw, dict) and "subset" in draw:
        drawn = draw["subset"]
    elif isinstance(draw, dict) and draw.get("law") == "stratified":
        drawn = sum(s["k"] for s in draw["strata"])
    else:
        raise Finding(REJECT, f"{what}: profile: not a subset or stratified draw")
    bound = pub.get("bound") if isinstance(pub, dict) else None
    if not isinstance(bound, int) or isinstance(bound, bool) or bound < 0:
        raise Finding(REJECT, f"{what}: no bound")
    return Facts(lv["size"], drawn, prof["delta"], bound)


def _audit_check(i: int, reg: Registration, a: dict) -> Facts:
    what = f"audit {i}"
    prof, pub = _answered(what, reg, a)
    return _facts(what, prof, pub)


def clog2(x: int) -> int:
    """`⌈log2 x⌉`, 0 for x ≤ 1."""
    return 0 if x <= 1 else (x - 1).bit_length()


def at_most(n: int, b: int) -> int:
    """`Σ_{j ≤ b} C(n, j)`: the sets of at most `b` of `n` units."""
    return sum(math.comb(n, j) for j in range(min(b, n) + 1))


def _terms(free: int, steps: int, f: Facts, jitter: int) -> dict:
    return {"delta": str(f.delta), "drawn": f.drawn, "faulty_bits": OUT_BITS * f.wrong,
            "faulty_sets": clog2(at_most(f.units, f.wrong)), "free": free, "jitter": clog2(jitter ** steps),
            "units": f.units, "wrong_units": f.wrong}


def public(p: Params, facts: list[Facts], scans: list[Facts] | None = None) -> dict:
    """K and its terms; `scans`, the rows form's link-windows' scans' facts (None in the clock form)."""
    audits = [_terms(free, steps, f, p.jitter) for (free, steps), f in zip(p.audits, facts)]
    scan_terms = None if scans is None else [_terms(*p.scan, f, p.jitter) for f in scans]
    cells = sum(clog2(cap + 1) for cap in p.cells)
    syncs = 1 if p.sync_set is None else len(p.sync_set.deltas) * len(p.sync_set.rhos)
    schedules = clog2(syncs ** p.windows)
    units = audits + (scan_terms or [])
    K = schedules + cells + sum(t["free"] + t["faulty_sets"] + t["faulty_bits"] + t["jitter"] for t in units)
    eps = sum((f.delta for f in [f for _, f in zip(p.audits, facts)] + (scans or [])), Fraction(0))
    out = {"K": K, "audits": audits, "cells": cells, "eps": str(eps), "schedules": schedules, "windows": p.windows}
    return out if scan_terms is None else out | {"scans": scan_terms}


def check(p: Params, c: Commitment, e) -> tuple:
    audits, windows = e
    if len(c.windows) != p.windows:
        return REJECT, "the commitment registers another number of link-windows"
    if len(c.audits) != len(p.audits):
        return REJECT, "the commitment registers another number of audits"
    if len(windows) != len(c.windows):
        return REJECT, "evidence: another number of link-windows"
    if len(audits) != len(c.audits):
        return REJECT, "evidence: another number of audits' answers"
    findings, facts, scans = [], [], []
    clear = clear_clocks(p.sync_set) if p.schedule is None else set()
    for i, (reg, x) in enumerate(zip(c.windows, windows)):
        try:
            if p.schedule is None:
                _window_check(p, clear, [y.clock.digest for y in windows[:i] if y.clock is not None], p.first + i,
                              reg, x)
            else:
                scans.append(_rows_window_check(p, p.first + i, reg, x))
        except Finding as f:
            findings.append(f)
    for i, (reg, a) in enumerate(zip(c.audits, audits)):
        try:
            facts.append(_audit_check(i, reg, a))
        except Finding as f:
            findings.append(f)
    if rejected := next((f for f in findings if f.verdict == REJECT), None):
        return REJECT, rejected.reason
    if findings:
        return UNDECIDED, findings[0].reason
    return ACCEPT, public(p, facts, None if p.schedule is None else scans), {"link": p.link, "windows": p.windows}


def reference(case) -> dict:
    return P.answer(ID, case, params=params, commitment=commitment, evidence=evidence, check=check,
                    digest=lambda c: tagged_sha512(COMMITMENT_TAG, c.bytes))


# --- the generator --------------------------------------------------------------------------------------------------------

#: Small LMS parameter sets (N, W, H, toy): the toy N = 8, and the deployed N = 64 at height 1.
KEYS = [(8, 8, 1, True), (8, 2, 2, True), (64, 4, 1, False)]


@lru_cache(maxsize=None)
def _key(i: int, seed: int = 0, same_id: bool = False) -> lms.PrivateKey:
    """KEYS[i]'s key from `seed`; `same_id`: the identifier of seed 0's key with another seed's one-time keys."""
    N, W, H, toy = KEYS[i]
    p = lms.Params(N, W, H, toy=toy)
    s = hashlib.sha512(b"network-period %d %d" % (i, seed)).digest()[:N]
    return lms.key(p, _key(i).I, s) if same_id else lms.keygen(p, s)


EXPECT = {
    "honest": ACCEPT,
    "program": REJECT, "provenance": REJECT,
    "cert-swap": REJECT, "cert-count": REJECT, "cert-forged": REJECT, "cert-window": REJECT, "cert-link": REJECT,
    "cert-session": REJECT, "cert-key": REJECT,
    "clock-ingress": REJECT, "clock-forged": REJECT, "clock-window": REJECT, "clock-session": REJECT,
    "clock-reused": REJECT, "clock-unsalted": REJECT,
    "answer-protocol": REJECT, "answer-other": REJECT, "answer-reject": REJECT, "answer-undecided": UNDECIDED,
    "answer-profile": REJECT,
    "audit-other": REJECT, "audit-reject": REJECT, "audit-undecided": UNDECIDED, "audit-profile": REJECT,
    "audit-bound": REJECT,
    "undecided-and-reject": REJECT, "form": REJECT,
    "first": REJECT, "count-windows": REJECT, "count-audits": REJECT, "evidence-count": REJECT,
    "params": REJECT, "commitment": REJECT, "evidence": REJECT, "case": REJECT,
    "rows-honest": ACCEPT,
    "rows-program": REJECT,
    "rows-cert-missing": REJECT, "rows-cert-swap": REJECT, "rows-cert-fails-link": REJECT,
    "rows-cert-direction": REJECT, "rows-cert-forged": REJECT, "rows-cert-window": REJECT, "rows-cert-session": REJECT,
    "rows-cert-key": REJECT,
    "rows-reg-count": REJECT, "rows-reg-swap": REJECT, "rows-reg-forged": REJECT, "rows-reg-window": REJECT,
    "rows-reg-link": REJECT, "rows-reg-session": REJECT, "rows-reg-key": REJECT, "rows-reg-field": REJECT,
    "rows-schedule": REJECT,
    "rows-answer-protocol": REJECT, "rows-answer-other": REJECT, "rows-answer-window": REJECT,
    "rows-answer-reject": REJECT, "rows-answer-undecided": UNDECIDED, "rows-answer-profile": REJECT,
    "rows-answer-bound": REJECT,
    "rows-audit-other": REJECT, "rows-audit-reject": REJECT, "rows-audit-undecided": UNDECIDED,
    "rows-audit-profile": REJECT, "rows-audit-bound": REJECT,
    "rows-undecided-and-reject": REJECT, "rows-form": REJECT,
    "rows-first": REJECT, "rows-count-windows": REJECT, "rows-count-audits": REJECT, "rows-evidence-count": REJECT,
    "rows-params": REJECT, "rows-commitment": REJECT, "rows-evidence": REJECT,
}
MUTATIONS = list(EXPECT)


def form_of(m: str) -> str:
    """The form a change's period is in: the rows form's changes are named `rows-…`."""
    return "rows" if m.startswith("rows-") else "clock"


def _flip(b: bytes, i: int) -> bytes:
    i %= len(b)
    return b[:i] + bytes([b[i] ^ 1]) + b[i + 1:]


def _cert_json(c: Certificate) -> dict:
    return {"digest": c.digest.hex(), "direction": c.direction.value, "key_id": c.key_id.hex(), "link": c.link,
            "signature": c.signature.hex(), "window": c.window}


def clock_commitment(salt: bytes, sync: ClockSync) -> bytes:
    """The digest a salted clock certificate signs (`certificate.CertifiedClock`): the sync's clock digest under the
    certifier's salt, SHA-512("verity/net-cert/v1" ‖ salt_c ‖ clock digest)."""
    return certificate_digest(salt, clock_digest(sync))


def _accepted(digest: bytes, program: str, query: str, size: int, k: int, delta: str, bound: int,
              strata: list[dict] | None = None) -> dict:
    """One-stage's answer accepting `k` of `size` units drawn `subset:k`, or drawn stratified over `strata` (each
    `{k, name, size}`, k and size their sums), as `OneStage.check`'s public and profile have it."""
    units = list(range(k))
    draw = {"subset": k} if strata is None else {"law": "stratified", "strata": strata}
    return {"commitment": digest.hex(),
            "profile": {"covered": [], "delta": delta, "drawn": {UNIT: k},
                        "levels": [{"draw": draw, "name": UNIT, "query": query, "size": size}],
                        "linked": False, "prescribed": [], "programs": [program]},
            "protocol": ONE_STAGE, "public": {"bound": bound, "drawn": units, "per_template": None, "proved": units},
            "reason": None, "verdict": ACCEPT}


def _refused(answer: dict, verdict: str) -> dict:
    return answer | {"profile": None, "public": None, "reason": f"the audit is {verdict}", "verdict": verdict}


#: The schedule's fields that leave its rows' shape (B, S, T) as it is: a change to one changes only its digest.
SAME_SHAPE = ("bucket_ms", "delay", "egress_cells", "ingress_buckets", "ingress_cells", "queue")
#: The orders of a link-window's three records' certificates other than their own.
SWAPS = ((0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


def schedule_json(s: WardenParams) -> dict:
    return {k: getattr(s, k) for k in SCHEDULE_KEYS}


def _small_schedule(r: random.Random, small: bool) -> WardenParams:
    """A schedule of small rows: B just past a cell's header, S = lanes · slots at most 4, T at most 3 (2 if
    `small`)."""
    return WardenParams(cell_bytes=r.choice([11, 12, 16]), slots=r.randint(1, 2), bucket_ms=r.randint(1, 200),
                        delay=r.randint(1, 4), lanes=r.randint(1, 2), queue=r.randint(1, 8),
                        window=r.randint(1, 2 if small else 3), ingress_cells=r.randint(1, 8),
                        ingress_buckets=r.randint(1, 4), egress_cells=r.randint(1, 16))


def build(m: str, seed: int, small: bool = False) -> dict:
    """A period's case with the change `m`, in the form `m` names (`form_of`), every choice drawn from `seed`; `small`:
    the toy key, as few link-windows and audits as `m` needs."""
    r = random.Random(f"network-period {m} {seed}")
    rows = form_of(m) == "rows"
    m = m.removeprefix("rows-")
    ki = 0 if small else r.randrange(len(KEYS))
    key = _key(ki)
    N, W, H, toy = KEYS[ki]
    session = r.randbytes(lms.SESSION_BYTES)
    link = r.choice(["eth0", "a", "dé"])
    program = hashlib.sha512(b"link program %d" % r.randrange(3)).hexdigest()
    least_windows = 2 if m in ("undecided-and-reject", "clock-reused") else 1
    least_audits = 1 if m.startswith("audit") or m in ("undecided-and-reject", "count-audits") else 0
    n_windows = r.randint(least_windows, least_windows if small else 3)
    n_audits = r.randint(least_audits, max(least_audits, 1) if small else 2)
    first = r.choice([0, 0, r.randrange(1, 6)])
    if rows:
        sched = bound = _small_schedule(r, small)
    else:
        deltas = r.sample(range(4), r.randint(1, 3))
        rhos = r.sample([Fraction(1), Fraction(2), Fraction(1, 2), Fraction(3, 2)], r.randint(1, 3))
    t = r.randrange(1, n_windows) if m == "clock-reused" else r.randrange(n_windows)
    a = r.randrange(max(n_audits, 1))
    signed = iter(range(1 << 30))

    def sign(link: str, d: Direction, w: int, digest: bytes, k: lms.PrivateKey = key, s: bytes = session) -> Certificate:
        c = Certificate(k.I, link, d, w, digest, b"")
        return dataclasses.replace(c, signature=lms.sign_at(k, next(signed) % 2 ** k.params.H, c.message(s)).encode())

    def registrations(w: int, s: WardenParams | None = None, on: str = link, ses: bytes = session) -> list[dict]:
        """Link `on`'s ingress and egress rows at window `w`, committed under schedule `s` (the period's) in session
        `ses` and registered, each head signed there."""
        s = s or sched
        out = []
        for d in (INGRESS, EGRESS):
            cells = WR.Cells.commit(on, d, w, [r.randbytes(WR.row_bytes(s)) for _ in range(s.window)],
                                    WR.Wiring((), r.randbytes(WR.WIRING_SALT_BYTES)), s, ses,
                                    [r.randbytes(WR.SALT_BYTES) for _ in range(s.window)])
            out.append(WR.registration(cells.head, sign(on, d, w, cells.head.digest(), s=ses), s))
        return out

    def resigned(doc: dict, **kw) -> dict:
        c = sign(doc["link"], Direction(doc["direction"]), doc["window"], bytes.fromhex(doc["certificate"]["digest"]),
                 **kw)
        return doc | {"certificate": {"digest": c.digest.hex(), "key": c.key_id.hex(), "signature": c.signature.hex()}}

    window_regs, windows, clocks = [], [], []
    for w in range(first, first + n_windows) if rows else ():
        aimed = w == first + t
        i, j = r.randrange(2), r.randrange(3)
        regs = registrations(w)
        named = regs
        if aimed and m == "answer-window":
            there = registrations(w + r.choice([1, -1] if w else [1]))
            named = r.choice([there, [regs[0], there[1]], [there[0], regs[1]],
                              [x | {"domain": y["domain"]} for x, y in zip(regs, there)]])
        elif aimed and m == "schedule":
            other = dataclasses.replace(sched, **{f: getattr(sched, f) + 1 for f in [r.choice(SAME_SHAPE)]})
            if r.random() < 0.5:
                bound = other
            else:
                under = registrations(w, other)
                regs = r.choice([under, [under[0], regs[1]], [regs[0], under[1]]])
        elif aimed and m == "reg-count":
            regs = r.choice([regs[:1], regs[1:], [], regs + regs[1:]])
        elif aimed and m == "reg-swap":
            regs = regs[::-1]
        elif aimed and m in ("reg-forged", "reg-window", "reg-link", "reg-session", "reg-key", "reg-field"):
            doc, c = regs[i], regs[i]["certificate"]
            sig, dig = bytes.fromhex(c["signature"]), bytes.fromhex(c["digest"])
            if m == "reg-forged":
                doc = r.choice([doc | {"certificate": c | {"signature": s.hex()}}
                                for s in (_flip(sig, seed), sig[:-1], b"")]
                               + [doc | {"certificate": c | {"digest": _flip(dig, seed).hex()}}]
                               + [doc | {k: r.randbytes(n).hex()} for k in ("root", "wiring") for n in (64, 63)])
            elif m == "reg-window":
                doc = registrations(w + r.choice([1, -1] if w else [1]))[i]
            elif m == "reg-link":
                doc = registrations(w, on=link + "x")[i]
            elif m == "reg-session":
                ses = _flip(session, seed)
                doc = resigned(doc, s=ses) if r.random() < 0.5 else registrations(w, ses=ses)[i]
            elif m == "reg-key":
                doc = resigned(doc, k=_key(ki, 1, r.random() < 0.5))
            else:
                lay = doc["layout"]
                f = r.choice(LAYOUT_KEYS)
                doc = doc | r.choice([
                    {"cell_bytes": doc["cell_bytes"] + 1}, {"rows": doc["rows"] + 1}, {"words": doc["words"] + 1},
                    {"word_bits": 8}, {"row_schema": "hm96-sha512/row/v1"},
                    {"format": "verity-network-warden/registration/v1"}, {"value": doc["value"] + "x"},
                    {"domain": r.randbytes(64).hex()}, {"parameters": r.randbytes(64).hex()},
                    {"direction": (EGRESS if i == 0 else INGRESS).value}, {"layout": lay | {f: lay[f] + 1}}])
            regs = regs[:i] + [doc] + regs[i + 1:]
        recs = [sign(on, d, w, r.randbytes(64)) for on, d in records(link)]
        if aimed and m.startswith("cert-"):
            on, d = records(link)[j]
            if m == "cert-missing":
                recs = r.choice([recs[:j] + recs[j + 1:], []])
            elif m == "cert-swap":
                recs = [recs[x] for x in r.choice(SWAPS)]
            elif m == "cert-fails-link":
                recs[2] = sign(r.choice([link, f"{link}-fail", f"{link}x-fails"]), EGRESS, w, recs[2].digest)
            elif m == "cert-direction":
                recs[j] = sign(on, INGRESS if d is EGRESS else EGRESS, w, recs[j].digest)
            elif m == "cert-forged":
                sig = recs[j].signature
                recs[j] = dataclasses.replace(recs[j], signature=r.choice([_flip(sig, seed), sig[:-1], b""]))
            elif m == "cert-window":
                recs[j] = sign(on, d, w + 1, recs[j].digest)
            elif m == "cert-session":
                recs[j] = sign(on, d, w, recs[j].digest, s=_flip(session, seed))
            elif m == "cert-key":
                recs[j] = sign(on, d, w, recs[j].digest, k=_key(ki, 1, r.random() < 0.5))
        entries = [{"domain": x["domain"], "root": x["root"], "value": x["value"]} for x in named]
        entries.insert(r.randrange(len(entries) + 1),
                       {"domain": r.randbytes(64).hex(), "root": r.randbytes(64).hex(), "value": "decoy"})
        T = sched.window
        strata = [{"k": r.randint(1, T), "name": d.value, "size": T} for d in (INGRESS, EGRESS)] \
            if r.random() < 0.5 else None
        k = sum(s["k"] for s in strata) if strata else r.randint(1, 2 * T)
        rec = {"format": r.choice(FORMATS), "law": {"k": k, "law": "subset" if strata is None else "stratified"},
               "partition": f"{link}/scan/{w}", "population": 2 * T, "program": program, "protocol": ONE_STAGE,
               r.choice(["roots", "registered"]): entries, "window": {"kind": "stand-in", "window": w}}
        if aimed and m == "program":
            rec["program"] = r.choice([hashlib.sha512(b"another program").hexdigest(), program.upper(), None])
        digest = registration_digest(rec)
        answer = _accepted(digest, program, rec["partition"], 2 * T, k, r.choice(["1/100", "1/1000000", "1/7"]),
                           r.randint(0, 2 * T), strata)
        if aimed:
            if m == "answer-protocol":
                answer["protocol"] = "verity/sampled-proofs/two-stage/v1"
            elif m == "answer-other":
                answer["commitment"] = r.choice([r.randbytes(64), registration_digest(rec | {"population": 1})]).hex()
            elif m == "answer-reject":
                answer = _refused(answer, REJECT)
            elif m in ("answer-undecided", "undecided-and-reject"):
                answer = _refused(answer, UNDECIDED)
            elif m == "answer-profile":
                lv = answer["profile"]["levels"][0]
                answer["profile"]["levels"] = r.choice([
                    [lv, lv | {"name": "token", "size": 3}], [lv | {"draw": {"bernoulli": "1/2"}}],
                    [lv | {"draw": None}], []])
            elif m == "answer-bound":
                answer["public"] = r.choice([{"drawn": []}, answer["public"] | {"bound": -1},
                                             answer["public"] | {"bound": "3"}, []])
        window_regs.append(rec)
        shown = {"clock": _cert_json(recs[1])} if aimed and m == "form" else {"registrations": regs}
        windows.append({"answer": answer, "certificates": [_cert_json(c) for c in recs]} | shown)
    for w in range(first, first + n_windows) if not rows else ():
        rec = {"format": r.choice(FORMATS), "law": {"k": 1, "law": "subset"}, "partition": f"{link}/{w}",
               "population": 1, "program": program, "protocol": ONE_STAGE, "window": served(w)}
        sync = ClockSync(r.choice(deltas), r.choice(rhos))
        aimed = w == first + t
        if aimed and m == "program":
            rec["program"] = r.choice([hashlib.sha512(b"another program").hexdigest(), program.upper(), None])
        elif aimed and m == "provenance":
            rec["window"] = r.choice([served(w + 1), {"kind": "batch", "window": w}, served(w) | {"run": 0}, None])
        digest = registration_digest(rec)
        egress = sign(link, EGRESS, w, r.randbytes(64))
        ingress = sign(link, INGRESS, w, r.randbytes(64))
        clock = sign(link, EGRESS, w, clock_commitment(r.randbytes(SALT_BYTES), sync))
        answer = _accepted(digest, program, rec["partition"], 1, 1, r.choice(["1/100", "1/1000000"]), 0)
        if aimed:
            if m == "cert-forged":
                i = r.randrange(2)
                c = (egress, ingress)[i]
                sig = r.choice([_flip(c.signature, seed), c.signature[:-1], b""])
                forged = dataclasses.replace(c, signature=sig)
                egress, ingress = (forged, ingress) if i == 0 else (egress, forged)
            elif m == "cert-window":
                egress = sign(link, EGRESS, w + 1, egress.digest)
            elif m == "cert-link":
                ingress = sign(link + "x", INGRESS, w, ingress.digest)
            elif m == "cert-session":
                egress = sign(link, EGRESS, w, egress.digest, s=_flip(session, seed))
            elif m == "cert-key":
                ingress = sign(link, INGRESS, w, ingress.digest, k=_key(ki, 1, r.random() < 0.5))
            elif m == "clock-ingress":
                clock = sign(link, INGRESS, w, clock.digest)
            elif m == "clock-forged":
                sig, dig = clock.signature, clock.digest
                clock = r.choice([dataclasses.replace(clock, signature=s) for s in (_flip(sig, seed), sig[:-1], b"")]
                                 + [dataclasses.replace(clock, digest=d) for d in (_flip(dig, seed), dig[:-1])])
            elif m == "clock-window":
                clock = sign(link, EGRESS, w + 1, clock.digest)
            elif m == "clock-session":
                clock = sign(link, EGRESS, w, clock.digest, s=_flip(session, seed))
            elif m == "clock-reused":
                clock = sign(link, EGRESS, w, r.choice(clocks).digest)
            elif m == "clock-unsalted":
                clock = sign(link, EGRESS, w, clock_digest(sync))
            elif m == "answer-protocol":
                answer["protocol"] = "verity/sampled-proofs/two-stage/v1"
            elif m == "answer-other":
                answer["commitment"] = r.choice([r.randbytes(64), registration_digest(rec | {"population": 2})]).hex()
            elif m == "answer-reject":
                answer = _refused(answer, REJECT)
            elif m in ("answer-undecided", "undecided-and-reject"):
                answer = _refused(answer, UNDECIDED)
            elif m == "answer-profile":
                answer = _accepted(digest, program, rec["partition"], *r.choice([(2, 1), (2, 2), (1, 0)]), "1/100", 0)
                if r.random() < 0.3:
                    answer["profile"]["delta"] = "2/200"
        certificates = [egress, ingress]
        if aimed and m == "cert-swap":
            certificates = [ingress, egress]
        elif aimed and m == "cert-count":
            certificates = r.choice([[egress], [egress, ingress, ingress], []])
        window_regs.append(rec)
        clocks.append(clock)
        shown = {"registrations": []} if aimed and m == "form" else {"clock": _cert_json(clock)}
        windows.append({"answer": answer, "certificates": [_cert_json(c) for c in certificates]} | shown)

    audit_params, audit_regs, audits = [], [], []
    for i in range(n_audits):
        units = r.randint(1, 40)
        cut = r.randint(1, units - 1) if units > 1 and r.random() < 0.5 else units
        strata = [{"k": r.randint(1, n), "name": f"s{j}", "size": n} for j, n in enumerate([cut, units - cut]) if n] \
            if r.random() < 0.5 else None
        k = sum(s["k"] for s in strata) if strata else r.randint(1, units)
        rec = {"format": r.choice(FORMATS), "law": {"k": k, "law": "subset" if strata is None else "stratified"},
               "partition": f"loop/{i}", "population": units, "program": hashlib.sha512(b"loop %d" % i).hexdigest(),
               "protocol": ONE_STAGE, "window": {"kind": "served", "window": i}}
        digest = registration_digest(rec)
        answer = _accepted(digest, rec["program"], rec["partition"], units, k, r.choice(["1/100", "1/1000000", "1/10"]),
                           r.randint(0, units), strata)
        if i == a:
            if m == "audit-other":
                answer["commitment"] = r.randbytes(64).hex()
            elif m in ("audit-reject", "undecided-and-reject"):
                answer = _refused(answer, REJECT)
            elif m == "audit-undecided":
                answer = _refused(answer, UNDECIDED)
            elif m == "audit-profile":
                lv = answer["profile"]["levels"][0]
                answer["profile"]["levels"] = r.choice([
                    [lv, lv | {"name": "token", "size": 3}], [lv | {"draw": {"bernoulli": "1/2"}}], [lv | {"draw": None}],
                    []])
            elif m == "audit-bound":
                answer["public"] = r.choice([{"drawn": []}, answer["public"] | {"bound": -1},
                                             answer["public"] | {"bound": "3"}, []])
        audit_params.append({"free": r.randint(0, 5), "steps": r.randint(0, 4)})
        audit_regs.append(rec)
        audits.append(answer)

    case_params = {"audits": audit_params, "cells": [r.randint(0, 100) for _ in range(r.randint(0, 3))], "first": first,
                   "jitter": r.randint(1, 5), "key": lms.public_key(key).hex(), "link": link,
                   "lms": {"H": H, "N": N, "W": W, "toy": toy}, "program": program, "session": session.hex(),
                   "windows": n_windows}
    if rows:
        case_params |= {"scan": {"free": r.randint(0, 5), "steps": r.randint(0, 4)}, "schedule": schedule_json(bound)}
    else:
        case_params["sync_set"] = {"deltas": deltas, "rhos": [[x.numerator, x.denominator] for x in rhos]}
    c_json = {"audits": audit_regs, "windows": window_regs}
    e_json = {"audits": audits, "windows": windows}
    if m == "first":
        case_params["first"] = first + r.choice([1, -1] if first else [1])
    elif m == "count-windows":
        case_params["windows"] = n_windows + r.choice([1, -1] if n_windows > 1 else [1])
    elif m == "count-audits":
        case_params["audits"] = r.choice([audit_params[1:], [*audit_params, {"free": 0, "steps": 0}]])
    elif m == "evidence-count":
        e_json = r.choice([{"audits": audits, "windows": windows[1:]}, {"audits": audits, "windows": windows + windows[:1]},
                           {"audits": [*audits, audits[0] if audits else windows[0]["answer"]], "windows": windows}])
    c_bytes, e_bytes = D.canonical(c_json).encode(), D.canonical(e_json).encode()
    if m == "params":
        case_params = r.choice(_bad_params(case_params, seed))
    elif m == "commitment":
        c_bytes = r.choice(_bad_commitments(c_json, c_bytes))
    elif m == "evidence":
        e_bytes = r.choice(_bad_evidence(e_json, e_bytes))
    case = {"commitment": c_bytes.hex(), "evidence": e_bytes.hex(), "params": case_params}
    if m == "case":
        case = r.choice([case | {"commitment": case["commitment"].upper()}, case | {"evidence": case["evidence"] + "0"},
                         {k: v for k, v in case.items() if k != "params"}, [case]])
    return case


def _bad_params(p: dict, n: int) -> list:
    """Parameters `params` refuses, each one change to `p`."""
    out = [{k: v for k, v in p.items() if k != sorted(p)[n % len(p)]}, p | {"x": 0}, [p], p | {"windows": 0},
           p | {"jitter": 0}, p | {"jitter": True}, p | {"link": "x" * (1 << 16)}, p | {"link": 1},
           p | {"key": p["key"][:-2]}, p | {"key": p["key"] + "00"}, p | {"key": p["key"].upper()},
           p | {"session": p["session"][:-2]}, p | {"session": ""}, p | {"program": p["program"][:-2]},
           p | {"program": p["program"].upper()}, p | {"program": 0},
           p | {"lms": p["lms"] | {"W": 3}}, p | {"lms": p["lms"] | {"H": 0}}, p | {"lms": p["lms"] | {"toy": 0}},
           p | {"lms": p["lms"] | {"x": 0}}, p | {"lms": {k: v for k, v in p["lms"].items() if k != "toy"}},
           p | {"audits": [{"free": 0}]}, p | {"audits": [{"free": -1, "steps": 0}]}, p | {"audits": "x"},
           p | {"cells": [-1]}, p | {"cells": [True]}, p | {"first": -1}, p | {"first": True}, p | {"first": "0"}]
    if "sync_set" in p:
        ss = p["sync_set"]
        return out + [p | {"sync_set": ss | {"deltas": []}}, p | {"sync_set": ss | {"rhos": [[0, 1]]}},
                      p | {"sync_set": ss | {"rhos": [[1, 0]]}}, p | {"sync_set": ss | {"deltas": [1, 1]}},
                      p | {"sync_set": ss | {"rhos": [[1, 2], [2, 4]]}}, p | {"sync_set": ss | {"x": 0}}]
    s = p["schedule"]
    return out + [p | {"schedule": {k: v for k, v in s.items() if k != SCHEDULE_KEYS[n % len(s)]}},
                  p | {"schedule": s | {"x": 1}}, p | {"schedule": [s]}, p | {"schedule": s | {"delay": 0}},
                  p | {"schedule": s | {"lanes": True}}, p | {"schedule": s | {"window": -1}},
                  p | {"schedule": s | {"queue": "1"}}, p | {"schedule": s | {"cell_bytes": 10}},
                  p | {"schedule": s | {"cell_bytes": U32}}, p | {"schedule": s | {"lanes": U32}},
                  p | {"schedule": s | {"window": U32}}, p | {"scan": {"free": 0}},
                  p | {"scan": {"free": 0, "steps": -1}}, p | {"scan": "x"},
                  p | {"sync_set": {"deltas": [0], "rhos": [[1, 1]]}}]


def _bad_commitments(c: dict, b: bytes) -> list[bytes]:
    """Commitments that don't decode: bytes that aren't canonical JSON, or a registration without a known format."""
    enc = lambda j: D.canonical(j).encode()  # noqa: E731
    return [b + b" ", b[:-1], b"", json.dumps(c).encode(), enc(c | {"x": 1}), enc({"audits": c["audits"]}),
            enc(c | {"windows": [1]}), enc(c | {"windows": [{"format": "verity/registration/v3"}]}),
            enc(c | {"audits": [{"format": 1}]}), enc(c | {"audits": "x"}), enc([c])]


def _bad_evidence(e: dict, b: bytes) -> list[bytes]:
    """Evidence that doesn't decode, each one change to `e`."""
    enc = lambda j: D.canonical(j).encode()  # noqa: E731
    x, rest = e["windows"][0], e["windows"][1:]
    cert = x["certificates"][0]
    bad_windows = [x | {"certificates": [cert | {"direction": "up"}]}, x | {"certificates": [cert | {"key_id": "abc"}]},
                   x | {"certificates": [cert | {"window": -1}]}, x | {"certificates": [cert | {"x": 0}]},
                   x | {"certificates": "x"}, x | {"answer": {k: v for k, v in x["answer"].items() if k != "reason"}},
                   x | {"answer": x["answer"] | {"verdict": "maybe"}}, x | {"answer": x["answer"] | {"reason": "r"}}]
    if "clock" in x:
        clock = x["clock"]
        bad_windows += [{k: v for k, v in x.items() if k != "clock"},
                        x | {"clock": {"certificate": clock, "sync": [0, 1, 1]}}, x | {"clock": clock | {"x": 0}},
                        x | {"clock": {k: v for k, v in clock.items() if k != "signature"}},
                        x | {"clock": clock | {"direction": "up"}}, x | {"clock": [clock]}]
    else:
        doc = x["registrations"][0]
        lay, c = doc["layout"], doc["certificate"]
        bad_rows = [{k: v for k, v in doc.items() if k != "root"}, doc | {"x": 0}, doc | {"layout": lay | {"x": 0}},
                    doc | {"layout": {k: v for k, v in lay.items() if k != "slots"}}, doc | {"layout": [lay]},
                    doc | {"layout": lay | {"buckets": -1}},
                    doc | {"certificate": {k: v for k, v in c.items() if k != "key"}},
                    doc | {"certificate": c | {"signature": c["signature"] + "0"}}, doc | {"certificate": "x"},
                    doc | {"root": doc["root"][:-1]}, doc | {"root": "g" + doc["root"][1:]}, doc | {"wiring": 0},
                    doc | {"window": "0"}, doc | {"window": -1}, doc | {"cell_bytes": True}, doc | {"format": 1},
                    doc | {"domain": 0}, "x"]
        bad_windows += [{k: v for k, v in x.items() if k != "registrations"}, x | {"clock": cert},
                        x | {"registrations": "x"}, x | {"registrations": [1]}] + \
            [x | {"registrations": [d, *x["registrations"][1:]]} for d in bad_rows]
    return [b[:-1], b"\xff" + b, json.dumps(e).encode(), enc(e | {"x": 1}), enc({"windows": e["windows"]}),
            enc(e | {"audits": [1]})] + [enc(e | {"windows": [w, *rest]}) for w in bad_windows]


answer = reference
"""The period check's outcome on one case: `reference` under the name a composition's components answer by."""

#: What a case can be aimed at: each verdict, and a case that fails to decode (its params, commitment or evidence).
KINDS = (ACCEPT, REJECT, UNDECIDED, "broken-params", "broken-commitment", "broken-evidence")
#: Each form's changes (`build`'s `m`) by the kind they aim at: a rejection by the check itself, not by decoding, for
#: `reject`; `case` breaks the line `serve` reads rather than the period, so no kind uses it.
AIMS: dict[str, dict[str, tuple[str, ...]]] = {
    form: {
        ACCEPT: tuple(m for m in ms if m.removeprefix("rows-") == "honest"),
        REJECT: tuple(m for m in ms if EXPECT[m] == REJECT
                      and m.removeprefix("rows-") not in ("params", "commitment", "evidence", "case")),
        UNDECIDED: tuple(m for m in ms if EXPECT[m] == UNDECIDED),
        **{f"broken-{x}": tuple(m for m in ms if m.removeprefix("rows-") == x)
           for x in ("params", "commitment", "evidence")},
    } for form in FORMS for ms in [[m for m in EXPECT if form_of(m) == form]]
}


def verdict_of(kind: str) -> str:
    """The verdict a case aimed at `kind` gets: a case that fails to decode is rejected."""
    return kind if kind in (ACCEPT, REJECT, UNDECIDED) else REJECT


def case(kind: str, seed: int = 0, form: str = "clock") -> dict:
    """A small period of the form (`build(..., small=True)`: the toy key, as few link-windows and audits as its change
    needs) aimed at `kind`, its change and every choice drawn from `seed`."""
    ms = AIMS[form][kind]
    return build(ms[random.Random(f"network-period {kind} {seed}").randrange(len(ms))], seed, small=True)


def cases(seed: int = 0, form: str = "clock") -> dict[str, tuple[dict, dict]]:
    """One small period of the form of each kind with its answer, `{kind: (case, answer)}`."""
    return {k: (c, answer(c)) for k in KINDS for c in [case(k, seed, form)]}
