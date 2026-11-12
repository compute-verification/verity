"""The count over a span between two marks (`NetworkCheck.Span`, `verity/certifier/span/v0`, `network-span serve`)
against its Python mirror (`verity.protocols.certifier.verifier.span`), through the difftest harness's `ProtocolPort`,
and the mirror's own arithmetic and formats.

The reference and the generator are `span_case.py`'s: `reference` answers a case in the core interface's answer
format, and `build` makes a span with one change (`EXPECT`, each aimed at one rejection of the check, or malformed
parameters, commitments, evidence and cases). The tests that start the Lean executable skip without `lake`; the
mirror's own run everywhere."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from period_case import _key, schedule_json
from span_case import ACCEPT, EXPECT, MUTATIONS, REASONS, REJECT, _Leaves, build, reference
from verity.protocols.certifier.verifier import service, span
from verity.protocols.certifier.verifier.certificate import Certificate, CertifiedMark, Direction, Issuer
from verity.protocols.certifier.verifier.grid import COMPLETE, GridParams, Record

LEAN = Path(__file__).resolve().parents[2] / "lean"
lean = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

TAGS: dict[str, str] = {}
SEEN: Counter = Counter()

#: The changes sampled more often than the rest: the honest spans, the only accepted ones.
WEIGHTS = {"honest": 6, "far": 2}

#: The schedule of the Lean's `#guard`s (`guardSchedule`): D = 8, r = 2, T = 16 and 17 egress cells a response.
GUARD = span.Schedule(cell_bytes=256, slots=2, bucket_ms=100, delay=8, lanes=16, queue=64, window=16,
                      ingress_cells=64, ingress_buckets=50, egress_cells=17, every=1)
#: The parameters of the Lean's `#guard`s (`guardParams`): the link `eth0` under `GUARD`, a zero key and session.
GUARD_PARAMS = {"key": "00" * 24, "link": "eth0", "lms": {"H": 1, "N": 8, "W": 8, "toy": True},
                "schedule": schedule_json(GUARD), "session": "00" * 32}
#: The SHA-512 of `guard_mark(5, 1)`'s certificate's bytes, which the Lean's `#guard` holds too.
GUARD_START = ("37ccc33e0f5191d268bc91c122be57358555be87a54a8b0fa8384ca157a82be17d31e4d6bae5b7c54bc387c9e7d6f8d0"
               "00111046dd3372fafaef4af0455cfe69")


def guard_mark(phase: int, fill: int) -> CertifiedMark:
    """The Lean's `guardMark`: an egress certificate on `eth0-marks` at `phase`, its key id and its signed digest
    bytes of `fill` and its signature empty, with a salt of `fill + 1` and a mark of `fill + 2`."""
    c = Certificate(bytes([fill]) * 16, "eth0-marks", Direction.EGRESS, phase, bytes([fill]) * 64, b"")
    return CertifiedMark(bytes([fill + 2]) * 64, bytes([fill + 1]) * 64, c)


@st.composite
def cases(sample, only: tuple[str, ...] = tuple(MUTATIONS)) -> dict:
    m = sample(st.sampled_from([h for h, k in WEIGHTS.items() if h in only for _ in range(k)] + list(only)))
    case = build(m, sample(st.integers(0, 1 << 16)))
    TAGS[D.canonical(case)] = m
    return case


def counted(case) -> dict:
    out = reference(case)
    SEEN[(TAGS.get(D.canonical(case), "?"), out["verdict"])] += 1
    return out


PORT = D.ProtocolPort("network-span", counted, cases(), package=LEAN)
server = D.server_fixture(PORT)


@lean
@pytest.mark.slow
def test_the_count_agrees_with_its_mirror(server):
    """Lean and the mirror agree on every case, and each change reaches its verdict at least once."""
    SEEN.clear()
    examples = 600
    stats = D.run(PORT, examples=examples, server=server)
    print("\n".join(f"{tag:14} {verdict:8} {k}" for (tag, verdict), k in sorted(SEEN.items())))
    missed = [tag for tag, verdict in EXPECT.items() if not SEEN[(tag, verdict)]]
    assert not missed, (missed, SEEN)
    assert min(stats.counts.get(v, 0) for v in (ACCEPT, REJECT)) > examples // 10, stats


@lean
def test_each_rejection_has_its_own_reason(server):
    """Lean and the mirror each reject a span for the reason its change is aimed at: a mark on another link name, an
    ingress one, one under another key or in another session, one its salt and message do not open; marks out of
    order; an execution phase missing, in the span or in its tail, shown twice or outside; an execution phase with no
    certificates, as the network certifier leaves one it faulted or refused a connection in; an egress or a fails
    certificate on another link name, at another execution phase, an ingress one or one that does not verify; and an
    egress cells' head that names another schedule, or is another execution phase's, link's, session's or key's."""
    whats = Counter()
    for m, (lean_why, mirror_why) in REASONS.items():
        for n in range(3):
            case = build(m, n)
            assert mirror_why in reference(case)["reason"], (m, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            reason = json.loads(out)["reason"]
            assert verdict == REJECT and lean_why in reason, (m, n, out)
            whats[(m, next((w for w, _ in span.records("") if f"the {w} certificate" in reason), None))] += 1
    for m in ("cert-link", "cert-phase"):
        assert all(whats[(m, w)] for w, _ in span.records("")), whats


@lean
def test_a_mirror_without_the_tail_is_caught(server, monkeypatch):
    """A mirror that counts no tail accepts a span whose evidence stops at the end mark's execution phase, and Lean
    rejects it for the first execution phase of the tail."""
    built = [build("no-tail", n) for n in range(2)]
    monkeypatch.setattr(span, "tail", lambda s: 0)
    for case in built:
        with pytest.raises(D.Mismatch) as x:
            D.replay(PORT, case, server)
        assert (x.value.lean[0], x.value.python[0]) == (REJECT, ACCEPT), x.value
        assert "is missing" in json.loads(x.value.lean[1])["reason"], x.value


def test_every_change_reaches_its_verdict_in_the_mirror():
    """The mirror accepts the honest spans, rejects every other change, and rejects each for its reason."""
    for m, verdict in EXPECT.items():
        for n in range(2):
            out = reference(build(m, n))
            assert out["verdict"] == verdict, (m, n, out)
            assert m not in REASONS or REASONS[m][1] in out["reason"], (m, n, out)


def test_the_tail_is_the_execution_phases_a_held_response_leaves_in():
    """⌈(D + ⌈egress_cells / r⌉) / T⌉ at the `#guard`s' schedules, and on sampled schedules the least n with
    n · T ≥ D + b, b the least with b · r ≥ egress_cells."""
    def held(**fields: int) -> int:
        return span.tail(span.Schedule(**{k: getattr(GUARD, k) for k in schedule_json(GUARD)} | fields))
    assert held() == 2 and held(egress_cells=16) == 1
    assert held(delay=1, egress_cells=1, slots=1, window=1) == 2
    assert held(delay=10, egress_cells=4096, slots=4, window=600) == 2
    assert held(delay=1, egress_cells=1, slots=1, window=2) == 1 and held(delay=16, egress_cells=1, slots=1) == 2
    r = random.Random(0)
    for _ in range(300):
        s = dict(delay=r.randint(1, 40), slots=r.randint(1, 5), window=r.randint(1, 12), egress_cells=r.randint(1, 60))
        buckets = s["delay"] + next(b for b in range(s["egress_cells"] + 1) if b * s["slots"] >= s["egress_cells"])
        assert held(**s) == next(n for n in range(buckets + 1) if n * s["window"] >= buckets), s


def test_an_execution_phase_counts_every_cell_of_both_egress_records():
    """An execution phase's capacity is T buckets of lanes · r cells of the responses' record and of lanes cells of
    the fails record, B bytes a cell: the fails record leaves on the link too."""
    assert span.capacity_bits(GUARD) == 16 * (16 * 2 + 16) * 256 * 8 == 1572864
    responses = GUARD.grid.buckets * GUARD.grid.allocation * GUARD.grid.frame_bytes * 8
    assert span.capacity_bits(GUARD) - responses == 16 * 16 * 256 * 8
    assert span.capacity_bits(span.Schedule(**{k: getattr(GUARD, k) for k in schedule_json(GUARD)}
                                            | {"slots": 4, "window": 600})) == 98304000


def test_a_marks_certificate_is_in_a_certified_records_bytes():
    """`certificate_bytes` is the certificate's part of `service.encode`, and `commitment` reads back the two marks
    and each certificate's bytes."""
    r = random.Random(0)
    issuer = Issuer(_Leaves(_key(0)), r.randbytes(32), coins=r.randbytes)
    x = issuer.certify(Record("eth0", 3, ((None,), (b"abcd",)), COMPLETE), Direction.EGRESS, GridParams(2, 1, 4),
                       r.randbytes(64))
    assert service.encode(x)[len(x.head.encode()) + len(x.salt):] == span.certificate_bytes(x.certificate)
    marks = (issuer.mark("eth0", 5, r.randbytes(64)), issuer.mark("eth0", 7, r.randbytes(64)))
    c = span.commitment(b"".join(span.mark_bytes(m) for m in marks))
    assert c.marks == marks
    assert c.certificates == tuple(span.certificate_bytes(m.certificate) for m in marks)


def test_what_a_tie_reads_off_a_case():
    """`named`: the SHA-512 of the start certificate's bytes, the two marks' messages and the bits over the span and
    its tail, at the vector the Lean's `#guard` holds; no bits when the marks are out of order; and on a generated
    span the count's own `egress_bits`, public beside the count of execution phases and the tail and nothing else:
    the two marks' execution phases stay in the commitment."""
    p = span.params(GUARD_PARAMS)
    start, end = guard_mark(5, 1), guard_mark(7, 4)
    n = span.named(p, span.commitment(span.mark_bytes(start) + span.mark_bytes(end)))
    assert n == {"start_mark": bytes.fromhex(GUARD_START), "start_message": bytes([3]) * 64,
                 "end_message": bytes([6]) * 64, "egress_bits": 5 * 1572864}
    assert n["start_mark"] == hashlib.sha512(span.certificate_bytes(start.certificate)).digest()
    late = span.commitment(span.mark_bytes(guard_mark(7, 1)) + span.mark_bytes(guard_mark(5, 4)))
    assert span.named(p, late)["egress_bits"] is None
    for seed in range(4):
        case = build("honest", seed)
        got = span.named(span.params(case["params"]), span.commitment(bytes.fromhex(case["commitment"])))
        public = reference(case)["public"]
        assert sorted(public) == ["egress_bits", "phases", "tail"] and got["egress_bits"] == public["egress_bits"]
