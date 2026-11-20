"""The network attestation check (`NetworkCheck.Check`, `network-check serve`) on generated spatial units: an honest
network attester (`grid.Certifier` driven by `schedule.run_shaper` on the egress links, fed directly on the ingress links)
whose records `certificate.Issuer` signs with a small LMS key in a sampled attester session, then one change
(`MUTATIONS`), each aimed at one path of the check, and malformed parameters, commitments and evidence. The Lean check
gives each change its own verdict (`EXPECT`). The issuer certifies no incomplete record, so the `status` change either
leaves its faulted window unattested or signs its head anyway (`_signed_anyway`), as only a forger could, which is what
keeps the check's status path reachable."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import shutil
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import verity_difftest as D
from verity_difftest import protocol as P
from verity.commitments.crypto import lms
from verity.protocols.accounting.network.verifier.certificate import (SALT_BYTES, CertifiedRecord,
                                                                Direction, Issuer, SignerRefused,
                                                                certificate_digest, parameters_digest, record_head)
from verity.protocols.accounting.network.verifier.grid import COMPLETE, Certifier, GridParams, Reason, BlockGrid, Status
from verity.protocols.accounting.network.verifier.schedule import ClockSync, LogicalFrame, SyncSet, expected_grids, run_shaper
from verity.protocols.accounting.network.verifier.service import encode

LEAN = Path(__file__).resolve().parents[7] / "verity" / "protocols" / "accounting" / "network" / "verifier" / "lean"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

EGRESS, INGRESS = Direction.EGRESS, Direction.INGRESS

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
    """A test signer without leaf state: it starts any session, and its n-th signature is at leaf n mod 2^H. It refuses
    to sign while `refuse` is set."""

    def __init__(self, key: lms.PrivateKey):
        self.key, self.n, self.refuse = key, 0, False

    @property
    def key_id(self) -> bytes:
        return self.key.I

    def start(self, session: bytes) -> None:
        pass

    def sign(self, message: bytes) -> bytes:
        if self.refuse:
            raise SignerRefused("refused")
        self.n += 1
        return lms.sign_at(self.key, (self.n - 1) % 2 ** self.key.params.H, message).encode()


class _Faulty(Certifier):
    """A network attester that faults with `faults[t]` before releasing global bucket t."""

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
    "other-session": "signature", "forged": "signature", "tampered": "signature",
    "advice-outside": "advice", "advice-unit": "advice",
    "late": "mismatch", "frames": "mismatch", "parameters": "mismatch", "ingress-grid": "mismatch",
    "params": "params", "commitment": "commitment", "evidence": "evidence", "case": "case",
}
MUTATIONS = list(EXPECT)


def _signed_anyway(issuer: Issuer, record: BlockGrid, direction: Direction, g: GridParams, parameters: bytes,
                   deadline: bool) -> CertifiedRecord | None:
    """An attestation of a faulted record, which `Issuer.certify` refuses: its head signed under the issuer's key."""
    head = record_head(record.link, direction, record.window, record.grid, record.status, g, parameters,
                       deadline=deadline)
    salt = issuer.coins(SALT_BYTES)
    c = issuer.sign(record.link, direction, record.window, certificate_digest(salt, head.digest()))
    return None if c is None else CertifiedRecord(head, salt, c)


def _flip(b: bytes, i: int) -> bytes:
    i %= len(b)
    return b[:i] + bytes([b[i] ^ 1]) + b[i + 1:]


def _sync_json(s: ClockSync) -> list[int]:
    return [s.delta, s.rho.numerator, s.rho.denominator]


def _grid_json(g: GridParams) -> dict:
    return {"allocation": g.allocation, "bucket_ms": g.bucket_ms, "buckets": g.buckets, "frame_bytes": g.frame_bytes,
            "queue_capacity": g.queue_capacity}


@dataclasses.dataclass
class Sampled:
    """A generated spatial unit before it is encoded: the change `m` (sampled with `n`), its parameters as JSON, the
    attested records, each execution phase's raw record and grid by `(direction, link, window)`, the egress links'
    frames and syncs, and the execution phase the change aims at (`target`, and an egress one, `egress_target`, when
    there is one)."""

    m: str
    n: int
    params: dict
    records: list[CertifiedRecord]
    raws: dict[tuple[str, str, int], tuple[BlockGrid, GridParams]]
    egress: list[str]
    frames: dict[str, list[LogicalFrame]]
    syncs: dict[str, list[ClockSync]]
    target: tuple[str, str, int]
    egress_target: tuple[str, int] | None


@st.composite
def units(draw, only: tuple[str, ...] = tuple(MUTATIONS), need_egress: bool = False) -> Sampled:
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
    session = draw(st.binary(min_size=lms.SESSION_BYTES, max_size=lms.SESSION_BYTES))
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
    attested_grid = ingress_grid
    refuse, unsettled = set(), set()
    forge_status = False
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
        forge_status = draw(st.sampled_from([True, False]))
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
        attested_grid = draw(st.sampled_from([dataclasses.replace(ingress_grid, bucket_ms=ingress_grid.bucket_ms + 1),
                                               dataclasses.replace(ingress_grid, queue_capacity=ingress_grid.queue_capacity + 1)]))
        ingress_parameters = parameters_digest(attested_grid)

    signer = _Signer(key)
    sampled_salts = itertools.count()
    issuer = Issuer(signer, session,
                    coins=lambda k: hashlib.shake_256(session + next(sampled_salts).to_bytes(8, "big")).digest(k))
    records: list[CertifiedRecord] = []
    raws: dict[tuple[str, str, int], tuple[BlockGrid, GridParams]] = {}

    def certify(record, direction, g, parameters):
        lw = (direction.value, record.link, record.window)
        raws[lw] = (record, g)
        signer.refuse = lw in refuse
        r = issuer.certify(record, direction, g, parameters, deadline=lw not in unsettled)
        if r is None and forge_status and not record.status.complete:
            r = _signed_anyway(issuer, record, direction, g, parameters, lw not in unsettled)
        if r is not None:
            records.append(r)

    for l in egress:
        cert = _Faulty(l, grid, faults.get(("egress", l), {}))
        for record in run_shaper(shipped[l], ready[l], syncs[l], cert):
            certify(record, EGRESS, grid, egress_parameters)
    for l in ingress:
        cert = _Faulty(l, attested_grid, faults.get(("ingress", l), {}))
        for t, fs in enumerate(ingress_frames[l]):
            for f in fs:
                cert.enqueue(f)
            cert.tick()
            if (t + 1) % attested_grid.buckets == 0:
                certify(cert.close_window(), INGRESS, attested_grid, ingress_parameters)

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
            head, salt, c = records[at].head, records[at].salt, records[o].certificate
            records[at] = CertifiedRecord(head, salt, draw(st.sampled_from([
                c, dataclasses.replace(c, link=head.link, direction=head.direction, window=head.window),
                dataclasses.replace(c, link=head.link, direction=head.direction, window=head.window,
                                    digest=certificate_digest(salt, head.digest()))])))
    elif m == "other-key" and at is not None:
        r = records[at]
        same_id = draw(st.booleans())
        k2 = _key(ki, 1, same_id)
        c = dataclasses.replace(r.certificate, key_id=k2.I)
        records[at] = CertifiedRecord(r.head, r.salt,
                                      dataclasses.replace(c, signature=lms.sign_at(k2, 0, c.message(session)).encode()))
    elif m == "other-session" and at is not None:
        # the same key's attestation of the same record in another of the attester's sessions
        r = records[at]
        c = r.certificate
        other = draw(st.sampled_from([_flip(session, n), bytes(lms.SESSION_BYTES), bytes(range(lms.SESSION_BYTES))])
                     .filter(lambda s: s != session))
        records[at] = CertifiedRecord(r.head, r.salt, dataclasses.replace(
            c, signature=lms.sign_at(key, n % 2 ** key.params.H, c.message(other)).encode()))
    elif m == "forged" and at is not None:
        r = records[at]
        sig = r.certificate.signature
        sig = draw(st.sampled_from([_flip(sig, n), sig[:-1], sig + b"\0", b"", bytes(len(sig))]))
        records[at] = CertifiedRecord(r.head, r.salt, dataclasses.replace(r.certificate, signature=sig))
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
        c, salt = r.certificate, r.salt
        records[at] = draw(st.sampled_from([
            CertifiedRecord(head, salt, c),
            CertifiedRecord(head, salt, dataclasses.replace(c, digest=certificate_digest(salt, head.digest()))),
            CertifiedRecord(head, salt, dataclasses.replace(c, digest=head.digest())),
            CertifiedRecord(h, _flip(salt, n), c)]))
    elif m == "missing" and at is not None:
        del records[at]
    elif m == "duplicate" and at is not None:
        r = records[at]
        records.append(draw(st.sampled_from([r, CertifiedRecord(r.head, r.salt, dataclasses.replace(
            r.certificate, signature=lms.sign_at(key, 1, r.certificate.message(session)).encode()))])))
    elif m == "extra":
        empty = tuple((None,) * grid.allocation for _ in range(grid.buckets))
        extra = draw(st.sampled_from([BlockGrid("zz", 0, empty, COMPLETE), BlockGrid(tl, W, empty, COMPLETE)]))
        r = issuer.certify(extra, draw(st.sampled_from([EGRESS, INGRESS])), grid, parameters_digest(grid, ss))
        raws[(r.direction.value, extra.link, extra.window)] = (extra, grid)
        records.insert(n % (len(records) + 1), r)
    if m == "shuffled" or draw(st.booleans()):
        records = draw(st.permutations(records))

    case_params = {"egress": {"grid": _grid_json(grid), "links": egress},
                   "ingress": {"grid": _grid_json(ingress_grid), "links": ingress},
                   "key": lms.public_key(key).hex(), "lms": {"H": KEYS[ki][2], "N": KEYS[ki][0], "W": KEYS[ki][1],
                                                              "toy": KEYS[ki][3]},
                   "session": session.hex(),
                   "sync_set": {"deltas": deltas, "rhos": [[r.numerator, r.denominator] for r in rhos]}, "windows": W}
    return Sampled(m, n, case_params, list(records), raws, egress, frames, syncs, target, egress_target)


def evidence_bytes(d: Sampled) -> bytes:
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
    return case


def _bad_params(p: dict, n: int) -> list:
    """Parameters `params` refuses, each one change to `p`."""
    out = [{k: v for k, v in p.items() if k != sorted(p)[n % len(p)]}, p | {"windows": 0}, [p],
           p | {"egress": p["egress"] | {"links": []}, "ingress": p["ingress"] | {"links": []}},
           p | {"egress": p["egress"] | {"links": ["a", "a"]}}, p | {"ingress": p["ingress"] | {"links": [1]}},
           p | {"egress": p["egress"] | {"links": ["x" * (1 << 16)]}}, p | {"egress": "a"},
           p | {"key": p["key"][:-2]}, p | {"key": p["key"] + "00"},
           p | {"session": p["session"][:-2]}, p | {"session": p["session"] + "00"}, p | {"session": ""},
           p | {"session": 0},
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


@pytest.fixture(scope="module")
def server():
    with D.Server([str(D.build(LEAN, "network-verifier")), "serve"]) as s:
        yield s


def lean_paths(server: D.Server, inputs, examples: int) -> Counter:
    """How often the Lean check's outcome on `examples` derandomized draws of `inputs` took each path (`_kind`)."""
    seen = Counter()

    @settings(derandomize=True, database=None, deadline=None, max_examples=examples)
    @given(inputs)
    def one(case):
        seen[_kind(server.exchange(case))] += 1

    one()
    return seen


@pytest.mark.parametrize("m", MUTATIONS)
def test_each_change_gets_its_verdict(server, m):
    """The Lean check gives each change its own verdict on some generated unit (a change can miss, as a late frame that
    still fits its bucket)."""
    seen = lean_paths(server, cases((m,)), 80 if m in ("late", "frames") else 30)
    assert seen[EXPECT[m]], seen
