"""The count over a span between two marks (`NetworkCheck.Span`, `verity/certifier/span/v0`, `network-span serve`)
against its Python mirror (`verity.protocols.certifier.verifier.span`), through the difftest harness's `ProtocolPort`,
and the mirror's own arithmetic and formats.

The reference and the generator are `span_case.py`'s: `reference` answers a case in the core interface's answer
format, and `build` makes a span with one change (`EXPECT`, each aimed at one rejection of the check, or malformed
parameters, commitments, evidence and cases). The tests that start the Lean executable skip without `lake`; the
mirror's own run everywhere."""

from __future__ import annotations

import dataclasses
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
from span_case import (ACCEPT, EDGE, EXPECT, MUTATIONS, REASONS, REGISTERED_REASONS, REJECT, SITE_FORMS, _Leaves, build,
                       reference)
from verity.protocols.certifier.verifier import registered, span
from verity.protocols.certifier.verifier.certificate import Certificate, CertifiedMark, Direction, Issuer

LEAN = Path(__file__).resolve().parents[2] / "lean"
lean = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

TAGS: dict[str, str] = {}
SEEN: Counter = Counter()

#: The changes sampled more often than the rest: the honest spans, the only accepted ones.
WEIGHTS = {"honest": 6, "far": 2, "edge": 2, "registered": 3}

#: The schedule of the Lean's `#guard`s (`guardSchedule`): D = 8, r = 2, T = 16 and 17 egress cells a response.
GUARD = span.Schedule(cell_bytes=256, slots=2, bucket_ms=100, delay=8, lanes=16, queue=64, window=16,
                      ingress_cells=64, ingress_buckets=50, egress_cells=17, every=1)
#: The parameters of the Lean's `#guard`s (`guardParams`): the link `eth0` under `GUARD`, a zero key and session.
GUARD_PARAMS = {"key": "00" * 24, "link": "eth0", "lms": {"H": 1, "N": 8, "W": 8, "toy": True},
                "schedule": schedule_json(GUARD), "session": "00" * 32}
#: The registered hardware of the Lean's `#guard`s (`guardSite`): `eth0` joins the host `node0` and the network
#: certifier `cert0`, which registers the parameters' key, and `out0` is the certifier's other half, to the outside.
GUARD_SITE = {
    "links": [{"beta": 100, "ends": ["node0", "cert0"], "id": "eth0", "latency": 1},
              {"beta": 100, "ends": ["cert0", "outside"], "id": "out0", "latency": 1}],
    "locs": [{"bytes": 0, "id": "node0", "key": "", "kind": "host", "owner": "prover", "parent": None, "sheet": "",
              "tier": None},
             {"bytes": 0, "id": "cert0", "key": "00" * 24, "kind": "certifier", "owner": "auditor", "parent": None,
              "sheet": "", "tier": None}],
    "sheets": []}
#: The SHA-512 of `guard_mark(5, 1)`'s certificate's bytes, which the Lean's `#guard` holds too.
GUARD_START = ("37ccc33e0f5191d268bc91c122be57358555be87a54a8b0fa8384ca157a82be17d31e4d6bae5b7c54bc387c9e7d6f8d0"
               "00111046dd3372fafaef4af0455cfe69")


def guard_mark(phase: int, fill: int) -> CertifiedMark:
    """The Lean's `guardMark`: an egress certificate on `eth0-marks` at `phase`, its key id and its signed digest
    bytes of `fill` and its signature empty, with a salt of `fill + 1` and a mark of `fill + 2`."""
    c = Certificate(bytes([fill]) * 16, "eth0-marks", Direction.EGRESS, phase, bytes([fill]) * 64, b"")
    return CertifiedMark(bytes([fill + 2]) * 64, bytes([fill + 1]) * 64, c)


def numbered(reason: str, case: dict) -> bool:
    """Whether a reason holds a digit outside the link's name, which is public."""
    return any(c.isdigit() for c in reason.replace(case["params"]["link"], ""))


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
    order; an execution phase missing, in the span or in its tail, shown twice or outside, a second execution phase
    of tail under the edge schedule among them; an execution phase with no certificates, as the network certifier
    leaves one it faulted or refused a connection in; an egress or a fails certificate on another link name, at
    another execution phase, an ingress one or one that does not verify; and an egress cells' head that names another
    schedule, or is another execution phase's, link's, session's or key's. Neither reason holds a number: none names
    an execution phase."""
    whats, both = Counter(), ("cert-link", "cert-phase")
    for m, (lean_why, mirror_why) in REASONS.items():
        for n in range(3 if m in both else 2):
            case = build(m, n)
            assert mirror_why in reference(case)["reason"], (m, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            reason = json.loads(out)["reason"]
            assert verdict == REJECT and lean_why in reason, (m, n, out)
            assert not numbered(reason, case), (m, n, out)
            whats[(m, next((w for w, _ in span.records("") if f"the {w} certificate" in reason), None))] += 1
    for m in both:
        assert all(whats[(m, w)] for w, _ in span.records("")), whats


@lean
def test_the_registered_hardware_refuses_the_parameters(server):
    """Lean and the mirror refuse the parameters on every form of each change to the registered hardware: a link that
    is none of the site's, or that joins something else than a network certifier and a host; a certifier that
    registers another key, or one that is not lower-case hex; a site without a head and a head without a site; a site
    that is not one a registration log's view writes. A refusal by the registered hardware names what is public
    there, in the same words on both sides: the link, the site's network certifiers' links, the certifier and its
    registered key. With the site, an honest span's answer is the one without it."""
    for m, forms in SITE_FORMS.items():
        for n in range(forms):
            case = build(m, n)
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT, (m, n, out)
            if m in REGISTERED_REASONS:
                mirror, lean_why = reference(case)["reason"], json.loads(out)["reason"]
                assert REGISTERED_REASONS[m] in mirror and mirror == lean_why, (m, n, mirror, lean_why)
    for n in range(2):
        case = build("registered", n)
        bare = case | {"params": {k: v for k, v in case["params"].items() if k not in ("head", "site")}}
        (verdict, out), (bare_verdict, bare_out) = D.replay(PORT, case, server), D.replay(PORT, bare, server)
        assert verdict == bare_verdict == ACCEPT and json.loads(out) == json.loads(bare_out), (n, out, bare_out)


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


@lean
def test_the_tail_at_its_edge_is_one_execution_phase(server, monkeypatch):
    """Under the edge schedule, where a response's last cell leaves exactly one execution phase after its admission,
    Lean's tail is 1 and it counts the span's execution phases and one more; and a mirror whose tail is
    ⌈(D + ⌈egress_cells / r⌉) / T⌉, a bucket longer, accepts a span that shows a second execution phase of tail, which
    Lean rejects."""
    for n in range(2):
        case = build("edge", n)
        verdict, out = D.replay(PORT, case, server)
        answer, c = json.loads(out), span.commitment(bytes.fromhex(case["commitment"]))
        phases = c.marks[1].window - c.marks[0].window + 2
        assert verdict == ACCEPT and answer["public"] == {"egress_bits": phases * span.capacity_bits(EDGE), "tail": 1}
        assert answer["profile"] == {"link": case["params"]["link"], "phases": phases}, answer
    built = [build("edge-tail", n) for n in range(2)]
    monkeypatch.setattr(span, "tail", lambda s: -(-(s.delay + -(-s.egress_cells // s.slots)) // s.window))
    for case in built:
        with pytest.raises(D.Mismatch) as x:
            D.replay(PORT, case, server)
        assert (x.value.lean[0], x.value.python[0]) == (REJECT, ACCEPT), x.value
        assert "outside the span and its tail" in json.loads(x.value.lean[1])["reason"], x.value


def test_every_change_reaches_its_verdict_in_the_mirror():
    """The mirror accepts the honest spans, rejects every other change, and rejects each for its reason, which holds
    no number: a reason names no execution phase."""
    for m, verdict in EXPECT.items():
        for n in range(SITE_FORMS.get(m, 2)):
            case = build(m, n)
            out = reference(case)
            assert out["verdict"] == verdict, (m, n, out)
            assert m not in REASONS or REASONS[m][1] in out["reason"] and not numbered(out["reason"], case), (m, n, out)
            assert m not in REGISTERED_REASONS or REGISTERED_REASONS[m] in out["reason"], (m, n, out)


def test_the_registered_hardware_in_the_parameters():
    """With the site and the head, the parameters hold what the registered hardware states of the link and the key,
    as the Lean's `#guard`s have it, and the count is the one without them; a site without a head, a head without a
    site, a certifier that registers another key, the certifier's other half and a site that is none are refused."""
    with_site = GUARD_PARAMS | {"head": "h", "site": GUARD_SITE}
    assert span.params(with_site).registered == registered.Registered(
        head="h", link="eth0", host="node0", certifier="cert0", key="00" * 24, beta=100)
    assert span.params(GUARD_PARAMS).registered is None
    assert dataclasses.replace(span.params(with_site), registered=None) == span.params(GUARD_PARAMS)
    other_key = GUARD_SITE | {"locs": [GUARD_SITE["locs"][0], GUARD_SITE["locs"][1] | {"key": "aa" * 24}]}
    for bad, why in [({"site": GUARD_SITE}, "or with the registered hardware"), ({"head": "h"}, "the keys must be"),
                     ({"head": "h", "site": other_key}, "is not network certifier cert0's registered key"),
                     ({"head": "h", "site": GUARD_SITE, "link": "out0"}, "it joins certifier cert0 and the outside"),
                     ({"head": "h", "site": GUARD_SITE, "link": "eth1"}, "no link of the registered hardware"),
                     ({"head": "h", "site": []}, "the site"), ({"head": 0, "site": GUARD_SITE}, "head")]:
        with pytest.raises(span.Refused, match=why):
            span.params(GUARD_PARAMS | bad)
    for n in range(4):
        case = build("registered", n)
        bare = case | {"params": {k: v for k, v in case["params"].items() if k not in ("head", "site")}}
        out = reference(case)
        assert out["verdict"] == ACCEPT and out == reference(bare), (n, out)


def test_the_tail_is_the_execution_phases_a_held_response_leaves_in():
    """⌈(D + ⌊(egress_cells − 1) / r⌋) / T⌉ at the `#guard`s' schedules and at the edge schedule; and on sampled
    schedules the last execution phase, counted from the end mark's, that a cell leaves in: cell i of a connection
    admitted in bucket a leaves in bucket a + D + ⌊i / r⌋, over every bucket a of the end mark's execution phase and
    every cell of a response."""
    def held(**fields: int) -> int:
        return span.tail(span.Schedule(**{k: getattr(GUARD, k) for k in schedule_json(GUARD)} | fields))
    assert held() == span.tail(EDGE) == 1 and held(egress_cells=18) == 1 and held(egress_cells=19) == 2
    assert held(delay=1, egress_cells=1, slots=1, window=1) == 1
    assert held(delay=16, egress_cells=1) == 1 and held(delay=17, egress_cells=1) == 2
    assert held(delay=10, egress_cells=4096, slots=4, window=600) == 2
    r = random.Random(0)
    for _ in range(300):
        s = dict(delay=r.randint(1, 40), slots=r.randint(1, 5), window=r.randint(1, 12), egress_cells=r.randint(1, 60))
        D_, r_, T, E = s["delay"], s["slots"], s["window"], s["egress_cells"]
        assert held(**s) == max((a + D_ + i // r_) // T for a in range(T) for i in range(E)), s


def test_an_execution_phase_counts_every_cell_of_both_egress_records():
    """An execution phase's capacity is T buckets of lanes · r cells of the responses' record and of lanes cells of
    the fails record, B bytes a cell: the fails record leaves on the link too."""
    assert span.capacity_bits(GUARD) == 16 * (16 * 2 + 16) * 256 * 8 == 1572864
    responses = GUARD.grid.buckets * GUARD.grid.allocation * GUARD.grid.frame_bytes * 8
    assert span.capacity_bits(GUARD) - responses == 16 * 16 * 256 * 8
    assert span.capacity_bits(span.Schedule(**{k: getattr(GUARD, k) for k in schedule_json(GUARD)}
                                            | {"slots": 4, "window": 600})) == 98304000


def test_a_commitment_reads_back_its_two_marks():
    """`commitment` reads back the two marks and each certificate's bytes, the issuer's own (`Certificate.encode`), on
    a link whose name is any UTF-8; and the parameters take a link's name as the period's check does, to 65,535 UTF-8
    bytes and no further."""
    r = random.Random(0)
    issuer = Issuer(_Leaves(_key(0)), r.randbytes(32), coins=r.randbytes)
    for link in ("eth0", "r\u00e9seau/0", "\U0001f517"):
        marks = (issuer.mark(link, 5, r.randbytes(64)), issuer.mark(link, 7, r.randbytes(64)))
        c = span.commitment(b"".join(span.mark_bytes(m) for m in marks))
        assert c.marks == marks
        assert c.certificates == tuple(m.certificate.encode() for m in marks)
    for link, taken in (("x" * 65535, True), ("\u00e9" * 32767 + "x", True), ("x" * 65536, False),
                        ("\u00e9" * 32768, False)):
        try:
            assert span.params(GUARD_PARAMS | {"link": link}).link == link and taken
        except span.Refused as e:
            assert not taken and str(e) == "link: a name is at most 65,535 UTF-8 bytes"


def test_what_a_tie_reads_off_a_case():
    """`named`: the SHA-512 of the start certificate's bytes, the two marks' messages and the bits over the span and
    its tail, at the vector the Lean's `#guard` holds; no bits when the marks are out of order; and on a generated
    span the count's own `egress_bits`, public beside the tail and nothing else: the two marks' execution phases stay
    in the commitment."""
    p = span.params(GUARD_PARAMS)
    start, end = guard_mark(5, 1), guard_mark(7, 4)
    n = span.named(p, span.commitment(span.mark_bytes(start) + span.mark_bytes(end)))
    assert n == {"start_mark": bytes.fromhex(GUARD_START), "start_message": bytes([3]) * 64,
                 "end_message": bytes([6]) * 64, "egress_bits": 4 * 1572864}
    assert n["start_mark"] == hashlib.sha512(start.certificate.encode()).digest()
    late = span.commitment(span.mark_bytes(guard_mark(7, 1)) + span.mark_bytes(guard_mark(5, 4)))
    assert span.named(p, late)["egress_bits"] is None
    for seed in range(4):
        case = build("honest", seed)
        got = span.named(span.params(case["params"]), span.commitment(bytes.fromhex(case["commitment"])))
        public = reference(case)["public"]
        assert sorted(public) == ["egress_bits", "tail"] and got["egress_bits"] == public["egress_bits"]
