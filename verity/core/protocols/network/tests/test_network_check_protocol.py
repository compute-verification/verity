"""The network certificate check (`NetworkCheck.Check`, `network-check serve`) against `audit.py`, through the difftest
harness's `ProtocolPort`.

The reference decodes the protocol's formats (`NetworkCheck/Check.lean`'s module docstring) and runs the real `Audit`:
`check_link` on each egress link, then `check_ingress_link` on each ingress link, under the certifier's `LmsKey`. The cases
are generated spatial units: an honest network certifier (`grid.Certifier` driven by `schedule.run_shaper` on the egress
links, fed directly on the ingress links) whose records `certificate.Issuer` signs with a small LMS key, then one change
(`MUTATIONS`), each aimed at one path of the audit, and malformed parameters, commitments and evidence."""

from __future__ import annotations

import dataclasses
import hashlib
import shutil
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity_difftest import protocol as P
from verity.ml.commitments.identity import tagged_sha512
from verity.ml.crypto import lms
from verity.core.protocols.certifier.audit import Audit
from verity.core.protocols.certifier.certificate import (STATUS_CODES, Certificate, CertifiedRecord, Direction,
                                                                Issuer, LmsKey, RecordHead, SignerRefused,
                                                                certificate_message, parameters_digest)
from verity.core.protocols.certifier.grid import COMPLETE, Certifier, GridParams, Reason, Record, Status
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet, expected_grids, run_shaper
from verity.core.protocols.certifier.service import encode

LEAN = Path(__file__).resolve().parents[1] / "lean"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

ID = "verity/network-certificate/v2"
COMMITMENT_TAG = "verity-network-certifier/certificates/v2"
EGRESS, INGRESS = Direction.EGRESS, Direction.INGRESS
REASONS = {code: reason for reason, code in STATUS_CODES.items()}

# --- the reference --------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Unit:
    egress: tuple[str, ...]
    ingress: tuple[str, ...]
    windows: int
    grid: GridParams
    ingress_grid: GridParams
    sync_set: SyncSet
    key: LmsKey


def _get(j, k, ctx="the object"):
    if not isinstance(j, dict) or k not in j:
        raise D.Reject(f"{ctx}: {k} missing")
    return j[k]


def _nat(v, ctx):
    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
        raise D.Reject(f"{ctx}: not a natural number")
    return v


def _of(v, kind, ctx):
    if not isinstance(v, kind) or isinstance(v, bool) != (kind is bool):
        raise D.Reject(f"{ctx}: not a {kind.__name__}")
    return v


def _grid(j, ctx):
    u32 = {k: _nat(_get(j, k, ctx), k) for k in ("buckets", "allocation", "frame_bytes", "bucket_ms")}
    if any(not 1 <= v < 1 << 32 for v in u32.values()):
        raise D.Reject(f"{ctx}: T, r, B and τ_b are u32s of at least 1")
    q = _nat(_get(j, "queue_capacity", ctx), "queue_capacity")
    if q < 1:
        raise D.Reject(f"{ctx}: queue_capacity is at least 1")
    return GridParams(u32["buckets"], u32["allocation"], u32["frame_bytes"], u32["bucket_ms"], q)


def _links(j, ctx):
    ls = [_of(x, str, f"{ctx}.links") for x in _of(_get(j, "links", ctx), list, ctx)]
    if any(len(x.encode()) >= 1 << 16 for x in ls) or len(set(ls)) < len(ls):
        raise D.Reject(f"{ctx}.links: each listed once, at most 65,535 UTF-8 bytes")
    return tuple(ls)


def _rho(j, ctx):
    if not isinstance(j, list) or len(j) != 2:
        raise D.Reject(f"{ctx}: not [numerator, denominator]")
    n, d = (_nat(x, ctx) for x in j)
    if n < 1 or d < 1:
        raise D.Reject(f"{ctx}: a numerator and a denominator of at least 1")
    return Fraction(n, d)


def params(j) -> Unit:
    e, i = _get(j, "egress"), _get(j, "ingress")
    egress, ingress = _links(e, "egress"), _links(i, "ingress")
    if not egress + ingress:
        raise D.Reject("a spatial unit has at least one link")
    windows = _nat(_get(j, "windows"), "windows")
    if windows < 1:
        raise D.Reject("at least one window")
    lj = _get(j, "lms")
    try:
        lp = lms.Params(*(_nat(_get(lj, k, "lms"), k) for k in ("N", "W", "H")), toy=_of(_get(lj, "toy", "lms"), bool, "toy"))
    except ValueError as x:
        raise D.Reject(str(x)) from None
    key = P.unhex(_of(_get(j, "key"), str, "key"))
    if len(key) != 16 + lp.N:
        raise D.Reject(f"an LMS public key is {16 + lp.N} bytes")
    ss = _get(j, "sync_set")
    deltas = [_nat(x, "sync_set.deltas") for x in _of(_get(ss, "deltas", "sync_set"), list, "deltas")]
    rhos = [_rho(x, "sync_set.rhos") for x in _of(_get(ss, "rhos", "sync_set"), list, "rhos")]
    if not deltas or not rhos or len(set(deltas)) < len(deltas) or len(set(rhos)) < len(rhos):
        raise D.Reject("sync_set: non-empty, each delta and each rho listed once")
    return Unit(egress, ingress, windows, _grid(_get(e, "grid", "egress"), "egress.grid"),
                _grid(_get(i, "grid", "ingress"), "ingress.grid"), SyncSet(tuple(deltas), tuple(rhos)), LmsKey(lp, key))


def commitment(b: bytes) -> tuple[list[CertifiedRecord], bytes]:
    """The records, `service.encode`'s bytes one after another, decoded exactly."""
    pos = 0

    def take(n):
        nonlocal pos
        if pos + n > len(b):
            raise D.Reject("truncated")
        pos += n
        return b[pos - n:pos]

    def be(n):
        return int.from_bytes(take(n), "big")

    def text():
        try:
            return take(be(2)).decode("utf-8")
        except UnicodeDecodeError:
            raise D.Reject("a link is not UTF-8") from None

    def direction():
        c = be(1)
        if c not in (0, 1):
            raise D.Reject(f"direction code {c}")
        return (EGRESS, INGRESS)[c]

    out = []
    while pos < len(b):
        link, d, window, T, r, B, tau = text(), direction(), be(8), be(4), be(4), be(4), be(4)
        code = be(1)
        if code not in REASONS:
            raise D.Reject(f"status code {code}")
        deadline = be(1)
        if deadline not in (0, 1):
            raise D.Reject(f"deadline byte {deadline}")
        parameters = take(64)
        if pos + 4 * T + 64 > len(b):
            raise D.Reject("truncated")
        counts = tuple(be(4) for _ in range(T))
        if any(n > r for n in counts):
            raise D.Reject("a count past the allocation")
        head = RecordHead(link, d, window, T, r, B, tau, Status(REASONS[code]), bool(deadline), parameters, counts, take(64))
        key_id = take(be(2))
        c_link, c_d, c_window, digest = text(), direction(), be(8), take(64)
        out.append(CertifiedRecord(head, Certificate(key_id, c_link, c_d, c_window, digest, take(be(4)))))
    return out, b


def _frame(f):
    if not isinstance(f, list) or len(f) != 3:
        raise D.Reject("a frame is [payload, anchor, index]")
    return LogicalFrame(P.unhex(_of(f[0], str, "payload")), _nat(f[1], "anchor"), _nat(f[2], "index"))


def _sync(s):
    if not isinstance(s, list) or len(s) != 3:
        raise D.Reject("a sync is [delta, num, den]")
    return ClockSync(_nat(s[0], "delta"), _rho(s[1:], "rho"))


def evidence(b: bytes):
    j = P.canonical_json(b)
    return [([_frame(f) for f in _of(_get(x, "frames", "a link"), list, "frames")],
             [_sync(s) for s in _of(_get(x, "syncs", "a link"), list, "syncs")])
            for x in _of(_get(j, "links"), list, "links")]


def check(u: Unit, c, e) -> tuple:
    records, _ = c
    if len(e) != len(u.egress):
        return "reject", "evidence: another number of links"
    if any(len(syncs) != u.windows for _, syncs in e):
        return "reject", "evidence: another number of syncs"
    if any(len(f.payload) != u.grid.frame_bytes for frames, _ in e for f in frames):
        return "reject", "evidence: a frame of another size"
    audit = Audit()
    verdicts = []
    for link, (frames, syncs) in zip(u.egress, e):
        verdicts += [("egress", v) for v in audit.check_link(link, records, frames, syncs, u.sync_set, u.grid, u.key)]
    for link in u.ingress:
        verdicts += [("ingress", v) for v in audit.check_ingress_link(link, records, u.windows, u.key, u.ingress_grid)]
    for d, v in verdicts:
        if v.violation not in (None, "no-verdict"):
            return "reject", f"{d} link {v.link} window {v.window}: {v.violation}"
    for d, v in verdicts:
        if v.violation == "no-verdict":
            return "undecided", f"{d} link {v.link} window {v.window}: no-verdict"
    assert audit.accepted
    return "accept", {}, {
        "link_windows": [{"direction": d, "link": v.link, "verdict": "accept", "window": v.window} for d, v in verdicts],
        "syncs": [[s.delta, s.rho.numerator, s.rho.denominator] for _, s in sorted(audit.syncs.items())],
        "windows": list(range(u.windows))}


def reference(case, check=check) -> dict:
    return P.answer(ID, case, params=params, commitment=commitment, evidence=evidence, check=check,
                    digest=lambda c: tagged_sha512(COMMITMENT_TAG, c[1]))


# --- the generator --------------------------------------------------------------------------------------------------------

#: Small LMS parameter sets (N, W, H, toy): the deployed N = 64 at low heights, and the toy N = 8.
KEYS = [(64, 1, 2, False), (64, 1, 3, False), (64, 2, 2, False), (64, 4, 1, False), (8, 1, 3, True), (8, 2, 2, True),
        (8, 8, 1, True)]


@lru_cache(maxsize=None)
def _key(i: int, seed: int = 0, same_id: bool = False) -> lms.PrivateKey:
    """KEYS[i]'s key from `seed`; `same_id`: the identifier of seed 0's key with another seed's one-time keys."""
    N, W, H, toy = KEYS[i]
    p = lms.Params(N, W, H, toy=toy)
    s = hashlib.sha512(b"network-check %d %d" % (i, seed)).digest()[:N]
    return lms.key(p, _key(i).I, s) if same_id else lms.keygen(p, s)


class _Signer:
    """A test signer without leaf state: its n-th signature is at leaf n mod 2^H. It refuses while `refuse` is set."""

    def __init__(self, key: lms.PrivateKey):
        self.key, self.n, self.refuse = key, 0, False

    @property
    def key_id(self) -> bytes:
        return self.key.I

    def sign(self, message: bytes) -> bytes:
        if self.refuse:
            raise SignerRefused("refused")
        self.n += 1
        return lms.sign_at(self.key, (self.n - 1) % 2 ** self.key.params.H, message).encode()


class _Faulty(Certifier):
    """A certifier that faults with `faults[t]` before releasing global bucket t."""

    def __init__(self, link, params, faults):
        super().__init__(link, params)
        self.t, self.faults = 0, faults

    def tick(self):
        if self.t in self.faults:
            self.fault(self.faults[self.t])
        self.t += 1
        return super().tick()


EXPECT = {
    "honest": "accept", "shuffled": "accept", "extra": "accept",
    "missing": "no-verdict", "refused": "no-verdict", "deadline": "no-verdict",
    "status": "status", "duplicate": "duplicate",
    "replay-window": "signature", "replay-link": "signature", "replay-direction": "signature", "other-key": "signature",
    "forged": "signature", "tampered": "signature",
    "advice-outside": "advice", "advice-unit": "advice",
    "late": "mismatch", "frames": "mismatch", "parameters": "mismatch", "ingress-grid": "mismatch",
    "params": "params", "commitment": "commitment", "evidence": "evidence", "case": "case",
}
MUTATIONS = list(EXPECT)
#: The generator's tag for each case it made, by its canonical text, so the reference can count outcomes per path.
TAGS: dict[str, str] = {}
SEEN: Counter = Counter()


def _flip(b: bytes, i: int) -> bytes:
    i %= len(b)
    return b[:i] + bytes([b[i] ^ 1]) + b[i + 1:]


def _sync_json(s: ClockSync) -> list[int]:
    return [s.delta, s.rho.numerator, s.rho.denominator]


def _grid_json(g: GridParams) -> dict:
    return {"allocation": g.allocation, "bucket_ms": g.bucket_ms, "buckets": g.buckets, "frame_bytes": g.frame_bytes,
            "queue_capacity": g.queue_capacity}


@dataclasses.dataclass
class Drawn:
    """A generated spatial unit before it is encoded: the change `m` (drawn with `n`), its parameters as JSON, the
    certified records, each link-window's raw record and grid by `(direction, link, window)`, the egress links' frames and
    syncs, and the link-window the change aims at (`target`, and an egress one, `egress_target`, when there is one)."""

    m: str
    n: int
    params: dict
    records: list[CertifiedRecord]
    raws: dict[tuple[str, str, int], tuple[Record, GridParams]]
    egress: list[str]
    frames: dict[str, list[LogicalFrame]]
    syncs: dict[str, list[ClockSync]]
    target: tuple[str, str, int]
    egress_target: tuple[str, int] | None


@st.composite
def units(draw, only: tuple[str, ...] = tuple(MUTATIONS), need_egress: bool = False) -> Drawn:
    m = draw(st.sampled_from(only))
    n = draw(st.integers(0, 1 << 16))
    carries = m in ("late", "frames")
    W = draw(st.integers(2 if m == "replay-window" or carries else 1, 3))
    T, S, B = draw(st.integers(2 if carries else 1, 3)), draw(st.integers(1, 3)), draw(st.integers(1, 3))
    grid = GridParams(T, S, B, draw(st.sampled_from([100, 50])), draw(st.sampled_from([1 << 16, 7])))
    ingress_grid = draw(st.sampled_from([grid, GridParams(T, max(1, S - 1), B + 1, 100)]))
    deltas = draw(st.lists(st.integers(0, 3), min_size=2 if m == "advice-unit" else 1, max_size=3, unique=True))
    rhos = draw(st.lists(st.sampled_from([Fraction(1), Fraction(2), Fraction(1, 2), Fraction(3, 2)]), min_size=1,
                         max_size=3, unique=True))
    ss = SyncSet(tuple(deltas), tuple(rhos))
    names = draw(st.permutations(["a", "b", "c", "dé"]))
    ne = draw(st.integers(2 if m in ("advice-unit", "replay-link") else
                          1 if need_egress or m in ("late", "frames", "advice-outside", "replay-direction") else 0, 2))
    ni = draw(st.integers(1 if m in ("ingress-grid", "replay-direction") else 0, 2))
    if ne + ni == 0:
        ne = 1
    egress = list(names[:ne])
    ingress = list(names[:ni] if m == "replay-direction" or draw(st.booleans()) else names[ne:ne + ni])
    ki = draw(st.integers(0, len(KEYS) - 1))
    key = _key(ki)
    unit = [ClockSync(draw(st.sampled_from(deltas)), draw(st.sampled_from(rhos))) for _ in range(W)]
    syncs = {l: list(unit) for l in egress}
    frames = {}
    for l in egress:
        n_frames = draw(st.integers(1 if carries else 0, 6))
        frames[l] = [LogicalFrame(draw(st.binary(min_size=B, max_size=B)), draw(st.integers(0, W * T + 1)),
                                  draw(st.integers(0, 3))) for _ in range(n_frames)]
    shipped = {l: list(fs) for l, fs in frames.items()}
    ready = {l: [0] * len(fs) for l, fs in frames.items()}
    faults: dict[tuple[str, str], dict[int, Reason]] = {}
    ingress_frames = {l: [[draw(st.binary(min_size=ingress_grid.frame_bytes, max_size=ingress_grid.frame_bytes))
                           for _ in range(draw(st.integers(0, ingress_grid.allocation)))]
                          for _ in range(W * ingress_grid.buckets)] for l in ingress}
    egress_parameters, ingress_parameters = parameters_digest(grid, ss), parameters_digest(ingress_grid)
    certified_grid = ingress_grid
    refuse, unsettled = set(), set()
    targets = [("egress", l, w) for l in egress for w in range(W)] + [("ingress", l, w) for l in ingress for w in range(W)]
    target = targets[n % len(targets)]
    if m == "replay-link":
        target = ("egress", egress[n % ne], n % W)
    elif m == "replay-direction":
        target = (draw(st.sampled_from(["egress", "ingress"])), names[0], n % W)
    d, tl, tw = target
    egress_target = (tl, tw) if d == "egress" else (egress[n % len(egress)], n % W) if egress else None

    if m == "refused":
        refuse.add(target)
    elif m == "deadline":
        unsettled.add(target)
    elif m == "status":
        T_d = grid.buckets if d == "egress" else ingress_grid.buckets
        bucket = tw * T_d + n % T_d
        faults[(d, tl)] = {bucket: draw(st.sampled_from(list(Reason)))}
    elif m == "advice-outside":
        l, w = egress_target
        s = syncs[l][w]
        syncs[l][w] = draw(st.sampled_from([ClockSync(max(deltas) + 1, s.rho), ClockSync(s.delta, Fraction(5, 7))]))
    elif m == "advice-unit":
        l, w = egress[1 + n % (len(egress) - 1)], n % W
        syncs[l][w] = ClockSync(next(x for x in deltas if x != unit[w].delta), unit[w].rho)
    elif m in ("late", "frames"):
        # a frame the Program's grids carry, so the change reaches a checked window
        carried = {l: sum(f is not None for g in expected_grids(frames[l], syncs[l], grid) for row in g for f in row)
                   for l in egress}
        l = next((x for x in egress[n % ne:] + egress[:n % ne] if carried[x]), None)
        if l is not None:
            carried = carried[l]
            k = n % carried
            if m == "late":
                ready[l][k] = frames[l][k].anchor + draw(st.sampled_from([1, T, W * T]))
            else:
                shipped[l][k] = dataclasses.replace(shipped[l][k], payload=_flip(shipped[l][k].payload, n))
    elif m == "parameters":
        if d == "egress":
            egress_parameters = draw(st.sampled_from([parameters_digest(grid), parameters_digest(grid, SyncSet((9,), (1,)))]))
        else:
            ingress_parameters = parameters_digest(ingress_grid, ss)
    elif m == "ingress-grid":
        certified_grid = draw(st.sampled_from([dataclasses.replace(ingress_grid, bucket_ms=ingress_grid.bucket_ms + 1),
                                               dataclasses.replace(ingress_grid, queue_capacity=ingress_grid.queue_capacity + 1)]))
        ingress_parameters = parameters_digest(certified_grid)

    signer = _Signer(key)
    issuer = Issuer(signer)
    records: list[CertifiedRecord] = []
    raws: dict[tuple[str, str, int], tuple[Record, GridParams]] = {}

    def certify(record, direction, g, parameters):
        lw = (direction.value, record.link, record.window)
        raws[lw] = (record, g)
        signer.refuse = lw in refuse
        r = issuer.certify(record, direction, g, parameters, deadline=lw not in unsettled)
        if r is not None:
            records.append(r)

    for l in egress:
        cert = _Faulty(l, grid, faults.get(("egress", l), {}))
        for record in run_shaper(shipped[l], ready[l], syncs[l], cert):
            certify(record, EGRESS, grid, egress_parameters)
    for l in ingress:
        cert = _Faulty(l, certified_grid, faults.get(("ingress", l), {}))
        for t, fs in enumerate(ingress_frames[l]):
            for f in fs:
                cert.enqueue(f)
            cert.tick()
            if (t + 1) % certified_grid.buckets == 0:
                certify(cert.close_window(), INGRESS, certified_grid, ingress_parameters)

    signer.refuse = False

    def find(direction, link, window):
        return next((i for i, r in enumerate(records)
                     if (r.direction.value, r.link, r.window) == (direction, link, window)), None)

    at = find(*target)
    if m in ("replay-window", "replay-link", "replay-direction") and at is not None:
        other = {"replay-window": (d, tl, (tw + 1) % W),
                 "replay-link": (d, next((l for l in (egress if d == "egress" else ingress) if l != tl), tl), tw),
                 "replay-direction": ("ingress" if d == "egress" else "egress", tl, tw)}[m]
        o = find(*other)
        if o is not None:
            head, c = records[at].head, records[o].certificate
            records[at] = CertifiedRecord(head, draw(st.sampled_from([
                c, dataclasses.replace(c, link=head.link, direction=head.direction, window=head.window),
                dataclasses.replace(c, link=head.link, direction=head.direction, window=head.window, digest=head.digest())])))
    elif m == "other-key" and at is not None:
        r = records[at]
        same_id = draw(st.booleans())
        k2 = _key(ki, 1, same_id)
        c = dataclasses.replace(r.certificate, key_id=k2.I)
        records[at] = CertifiedRecord(r.head, dataclasses.replace(c, signature=lms.sign_at(k2, 0, c.message).encode()))
    elif m == "forged" and at is not None:
        r = records[at]
        sig = r.certificate.signature
        sig = draw(st.sampled_from([_flip(sig, n), sig[:-1], sig + b"\0", b"", bytes(len(sig))]))
        records[at] = CertifiedRecord(r.head, dataclasses.replace(r.certificate, signature=sig))
    elif m == "tampered" and at is not None:
        r = records[at]
        h = r.head
        counts = list(h.counts)
        counts[n % len(counts)] = (counts[n % len(counts)] + 1) % (h.allocation + 1)
        head = draw(st.sampled_from([
            dataclasses.replace(h, counts=tuple(counts)), dataclasses.replace(h, root=_flip(h.root, n)),
            dataclasses.replace(h, parameters=_flip(h.parameters, n)), dataclasses.replace(h, deadline=not h.deadline),
            dataclasses.replace(h, status=COMPLETE if h.status.reason else Status(Reason.DROPPED)),
            dataclasses.replace(h, bucket_ms=h.bucket_ms + 1)]))
        c = r.certificate
        records[at] = CertifiedRecord(head, draw(st.sampled_from([c, dataclasses.replace(c, digest=head.digest())])))
    elif m == "missing" and at is not None:
        del records[at]
    elif m == "duplicate" and at is not None:
        r = records[at]
        records.append(draw(st.sampled_from([r, CertifiedRecord(r.head, dataclasses.replace(
            r.certificate, signature=lms.sign_at(key, 1, r.certificate.message).encode()))])))
    elif m == "extra":
        empty = tuple((None,) * grid.allocation for _ in range(grid.buckets))
        extra = draw(st.sampled_from([Record("zz", 0, empty, COMPLETE), Record(tl, W, empty, COMPLETE)]))
        r = issuer.certify(extra, draw(st.sampled_from([EGRESS, INGRESS])), grid, parameters_digest(grid, ss))
        raws[(r.direction.value, extra.link, extra.window)] = (extra, grid)
        records.insert(n % (len(records) + 1), r)
    if m == "shuffled" or draw(st.booleans()):
        records = draw(st.permutations(records))

    case_params = {"egress": {"grid": _grid_json(grid), "links": egress},
                   "ingress": {"grid": _grid_json(ingress_grid), "links": ingress},
                   "key": lms.public_key(key).hex(), "lms": {"H": KEYS[ki][2], "N": KEYS[ki][0], "W": KEYS[ki][1],
                                                              "toy": KEYS[ki][3]},
                   "sync_set": {"deltas": deltas, "rhos": [[r.numerator, r.denominator] for r in rhos]}, "windows": W}
    return Drawn(m, n, case_params, list(records), raws, egress, frames, syncs, target, egress_target)


def evidence_bytes(d: Drawn) -> bytes:
    return D.canonical({"links": [{"frames": [[f.payload.hex(), f.anchor, f.index] for f in d.frames[l]],
                                   "syncs": [_sync_json(s) for s in d.syncs[l]]} for l in d.egress]}).encode()


def malformed(draw, m: str, n: int, case_params: dict, case_commitment: bytes, case_evidence: bytes, bad_commitments) -> dict:
    """The case, with the change `m` makes to its framing: bad parameters, commitment (one of `bad_commitments()`),
    evidence or case."""
    if m == "params":
        case_params = draw(st.sampled_from(_bad_params(case_params, n)))
    elif m == "commitment":
        case_commitment = draw(st.sampled_from(bad_commitments()))
    elif m == "evidence":
        case_evidence = draw(st.sampled_from(_bad_evidence(case_evidence, n)))
    case = {"commitment": case_commitment.hex(), "evidence": case_evidence.hex(), "params": case_params}
    if m == "case":
        case = draw(st.sampled_from([case | {"commitment": case["commitment"].upper() or "0"},
                                     case | {"evidence": case["evidence"] + "0"},
                                     {k: v for k, v in case.items() if k != "params"}]))
    return case


@st.composite
def cases(draw, only: tuple[str, ...] = tuple(MUTATIONS)) -> dict:
    d = draw(units(only))
    c = b"".join(encode(r) for r in d.records)
    case = malformed(draw, d.m, d.n, d.params, c, evidence_bytes(d), lambda: _bad_commitments(c, d.records, d.n))
    TAGS[D.canonical(case)] = d.m
    return case


def _bad_params(p: dict, n: int) -> list:
    """Parameters `params` refuses, each one change to `p`."""
    out = [{k: v for k, v in p.items() if k != sorted(p)[n % len(p)]}, p | {"windows": 0}, [p],
           p | {"egress": p["egress"] | {"links": []}, "ingress": p["ingress"] | {"links": []}},
           p | {"egress": p["egress"] | {"links": ["a", "a"]}}, p | {"ingress": p["ingress"] | {"links": [1]}},
           p | {"egress": p["egress"] | {"links": ["x" * (1 << 16)]}}, p | {"egress": "a"},
           p | {"key": p["key"][:-2]}, p | {"key": p["key"] + "00"},
           p | {"lms": p["lms"] | {"W": 3}}, p | {"lms": p["lms"] | {"H": 0}}, p | {"lms": p["lms"] | {"N": 8, "toy": False}},
           p | {"lms": p["lms"] | {"toy": 0}},
           p | {"sync_set": p["sync_set"] | {"deltas": []}}, p | {"sync_set": p["sync_set"] | {"rhos": []}},
           p | {"sync_set": p["sync_set"] | {"rhos": [*p["sync_set"]["rhos"], [2 * x for x in p["sync_set"]["rhos"][0]]]}},
           p | {"sync_set": p["sync_set"] | {"rhos": [[0, 1]]}}, p | {"sync_set": p["sync_set"] | {"rhos": [[1, 0]]}},
           p | {"sync_set": p["sync_set"] | {"deltas": [1, 1]}}, p | {"sync_set": p["sync_set"] | {"deltas": [-1]}}]
    for side in ("egress", "ingress"):
        g = p[side]["grid"]
        out += [p | {side: p[side] | {"grid": g | {"buckets": 0}}}, p | {side: p[side] | {"grid": g | {"bucket_ms": 1 << 32}}},
                p | {side: p[side] | {"grid": g | {"queue_capacity": 0}}},
                p | {side: p[side] | {"grid": {k: v for k, v in g.items() if k != sorted(g)[n % len(g)]}}}]
    return out


def _bad_commitments(c: bytes, records: list[CertifiedRecord], n: int) -> list[bytes]:
    """Commitments that don't decode, or don't decode as these records: truncated, extended, a field out of its range."""
    out = [c + bytes([n % 256]), c + b"\0\0"]
    if c:
        out.append(c[:n % len(c)])
    if records:
        L = len(records[0].head.link.encode())
        at = {"link": 2, "direction": 2 + L, "status": L + 27, "deadline": L + 28, "count": L + 93}
        for field, value in (("direction", 2), ("status", 7), ("deadline", 2), ("link", 0xFF)):
            if field != "link" or L:
                out.append(c[:at[field]] + bytes([value]) + c[at[field] + 1:])
        if records[0].head.buckets:
            out.append(c[:at["count"]] + (records[0].head.allocation + 1).to_bytes(4, "big") + c[at["count"] + 4:])
    return out


def _bad_evidence(e: bytes, n: int) -> list[bytes]:
    j = P.canonical_json(e)
    links = j["links"]
    out = [e[:-1] + b" }", b"\xff" + e, D.canonical({}).encode(), D.canonical({"links": [*links, links[-1]] if links
                                                                               else [{"frames": [], "syncs": []}]}).encode()]
    for i, l in enumerate(links):
        bad = [l | {"syncs": l["syncs"][:-1]}, l | {"syncs": [*l["syncs"], [0, 1, 1]]}, l | {"syncs": [[0, 0, 1]] * len(l["syncs"])},
               l | {"syncs": [[0, 1]] * len(l["syncs"])}, {"frames": l["frames"]}]
        if l["frames"]:
            f = l["frames"][0]
            bad += [l | {"frames": [[f[0] + "00", *f[1:]], *l["frames"][1:]]}, l | {"frames": [f[:2], *l["frames"][1:]]},
                    l | {"frames": [[f[0].upper(), *f[1:]], *l["frames"][1:]]}, l | {"frames": [[f[0], -1, f[2]], *l["frames"][1:]]}]
        out += [D.canonical({"links": links[:i] + [b] + links[i + 1:]}).encode() for b in bad]
    return out


def _kind(out: dict) -> str:
    """The path an outcome took: `accept`, the decoder that refused, or the first violation's kind."""
    if out["verdict"] == "accept":
        return "accept"
    reason = out["reason"]
    for prefix in ("params", "commitment", "evidence", "case"):
        if reason.startswith(prefix + ": "):
            return prefix
    return reason.rsplit(": ", 1)[1].split(":")[0]


def counted(case) -> dict:
    out = reference(case)
    SEEN[(TAGS.get(D.canonical(case), "?"), out["verdict"], _kind(out))] += 1
    return out


PORT = D.ProtocolPort("network-check", counted, cases(), package=LEAN)
server = D.server_fixture(PORT)


@pytest.mark.slow
def test_the_network_check_agrees_with_the_audit(server):
    """Lean and the audit agree on every case, and every path is reached: each change gives its own verdict at least once
    (a change can miss, as a late frame that still fits its bucket)."""
    SEEN.clear()
    stats = D.run(PORT, examples=1200, server=server)
    by_path = Counter()
    for (tag, _, kind), k in SEEN.items():
        by_path[(tag, kind)] += k
    print("\n".join(f"{tag:17} {kind:12} {k}" for (tag, kind), k in sorted(by_path.items())))
    missed = [tag for tag, kind in EXPECT.items() if not by_path[(tag, kind)]]
    assert not missed, (missed, by_path)
    assert min(stats.counts.get(v, 0) for v in ("accept", "reject", "undecided")) > 50, stats


def test_each_verdict_is_the_audits(server):
    """A short run over three paths, an honest unit, a missing record and a forged signature: all three verdicts."""
    port = dataclasses.replace(PORT, inputs=cases(("honest", "missing", "forged")))
    stats = D.run(port, examples=30, server=server)
    assert all(stats.counts.get(v) for v in ("accept", "reject", "undecided")), stats


@pytest.mark.slow
def test_a_reference_wrong_on_one_path_is_caught(server):
    """A reference that charges each link its own clock (no `advice` for a sync other than the unit's) differs."""
    def own_clock(u, c, e):
        out = check(u, c, e)
        if out[0] == "reject" and out[1].endswith(": advice") and all(
                s in u.sync_set for _, syncs in e for s in syncs):
            return "accept", {}, {}
        return out

    port = D.ProtocolPort("network-check", lambda case: reference(case, own_clock), cases(), package=LEAN)
    with pytest.raises(D.Mismatch) as x:
        D.run(port, examples=1200, server=server)
    assert (x.value.lean[0], x.value.python[0]) == ("reject", "accept")
