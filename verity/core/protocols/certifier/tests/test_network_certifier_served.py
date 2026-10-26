"""N2's served profile (`served`): the circuit held to its reference, the derivation held to the Program's grid on an
idle run, and served link-windows through the proof service: honest traffic accepted in every window, salted or not,
and a late block, an extra frame, a dropped frame and a request whose response straddles the window's end each
refused."""

import json
import random
from dataclasses import replace
from fractions import Fraction

import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.core.protocols.certifier import head_check as HC, served as SV
from verity.core.protocols.certifier.active import Bucketer
from verity.core.protocols.certifier.certificate import Direction, Issuer, parameters_digest
from verity.core.protocols.certifier.grid import Certifier, GridParams, occupancy
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet, expected_grids, run_shaper
from verity.core.protocols.certifier.service import PUBLIC_SERVED, LinkWindows
from verity.core.protocols.sampled_proofs import service as S

import signers
import standin

LINK = "responses"
P = GridParams(buckets=8, allocation=2, frame_bytes=1)
F, FIN, WINDOWS = 3, 1, 3
PROFILE = SV.ServedProfile(F, FIN, WINDOWS)
SYNC_SET = SyncSet((1, 2), (1, 2))
SYNCS = [ClockSync(1, 1), ClockSync(2, 1), ClockSync(1, 2)]
#: requests per bucket at the ingress certifier, each window idle by its end under its sync
ARRIVALS = ([1, 0, 1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0])
SALTS = [tuple(bytes([4 * w + k + 1]) * 192 for k in range(4)) for w in range(WINDOWS)]


@pytest.fixture(autouse=True)
def lean(monkeypatch):
    monkeypatch.setattr(standin.SVC.PF, "verify", standin.lean)


def frames_of(arrivals, request_frames=FIN, response_frames=F):
    """Each request's response, request-major: its F frames anchored at the bucket that completes the request."""
    out = []
    for w, counts in enumerate(arrivals):
        for a in SV.anchors(counts, request_frames):
            out += [LogicalFrame(bytes([len(out) % 256]), w * P.buckets + a, k) for k in range(response_frames)]
    return out


def egress_records(frames, ready=None, syncs=SYNCS):
    return run_shaper(frames, ready or [f.anchor for f in frames], syncs, Certifier(LINK, P))


def ingress_records(arrivals, own=0):
    """The ingress certifier's records, with `own` frames of its own (its salt's) first in each window."""
    certifier, out = Certifier(LINK, P), []
    for counts in arrivals:
        for t, n in enumerate(counts):
            for _ in range(n + (own if t == 0 else 0)):
                certifier.enqueue(b"q")
            certifier.tick()
        out.append(certifier.close_window())
    return out


class Run:
    """A served link with its certifier's signer and the challenger's stream, on injected coins and clock."""

    def __init__(self, tmp_path, profile=PROFILE) -> None:
        self.tmp = tmp_path
        self.signer = signers.signer(tmp_path)
        self.issuer = Issuer(self.signer, signers.SESSION)
        self.lw = LinkWindows(LINK, P, SYNC_SET, self.signer.public_key(), "r-served-test", signers.SESSION,
                              proof_deadline=60.0, served=profile)
        self.stream = self.lw.stream(budget=8, delta=2.0 ** -40, challenger="challenger-test",
                                     coins=lambda n: b"\x07" * n, clock=lambda: 0.0)
        self.previous = None

    def round(self, w, egress, ingress) -> S.Outcome:
        opening = self.lw.commit(w, self.issuer.certify(egress, Direction.EGRESS, P, parameters_digest(P, SYNC_SET)),
                                 self.issuer.certify(ingress, Direction.INGRESS, P, parameters_digest(P)), SALTS[w])
        reg = self.lw.register(self.stream, self.lw.registration(opening, self.previous))
        self.previous = reg.digest
        draw = self.stream.select(reg)
        return standin.decide(self.lw, self.stream, draw, opening, SYNCS, self.tmp / f"proof-{reg.order}")


def test_honest_served_traffic_is_accepted_in_every_window(tmp_path):
    """No egress profile: each window's egress counts are derived from its ingress counts under its sync, and the
    link's network verdict, over the served profile's windows, accepts."""
    run = Run(tmp_path)
    egress, ingress = egress_records(frames_of(ARRIVALS)), ingress_records(ARRIVALS)
    for w in range(WINDOWS):
        assert SV.derive(occupancy(ingress[w].grid), SYNCS[w], PROFILE, P.allocation) == list(occupancy(egress[w].grid))
    assert [run.round(w, egress[w], ingress[w]).accepted for w in range(WINDOWS)] == [True] * WINDOWS
    verdict = run.lw.network_verdict(run.stream)
    assert verdict["accepted"] and verdict["windows"] == WINDOWS and verdict["refused"] == []


def test_a_salted_link_counts_the_certifiers_own_frame(tmp_path):
    """With the salt on, every window's first ingress frame is the certifier's own, which answers no request: a
    profile that counts it (`own` 1) derives each window's counts and accepts, and an unsalted run under it, or a
    salted run under a profile without it, is refused in every window."""
    egress = egress_records(frames_of(ARRIVALS))
    salted, plain = ingress_records(ARRIVALS, own=1), ingress_records(ARRIVALS)
    profile = replace(PROFILE, own=1)
    assert [occupancy(r.grid)[0] for r in salted] == [a[0] + 1 for a in ARRIVALS]
    for w in range(WINDOWS):
        counts = list(occupancy(egress[w].grid))
        assert SV.derive(occupancy(salted[w].grid), SYNCS[w], profile, P.allocation) == counts
        assert SV.derive(occupancy(plain[w].grid), SYNCS[w], profile, P.allocation) != counts
        assert SV.derive(occupancy(salted[w].grid), SYNCS[w], PROFILE, P.allocation) != counts
    for k, (ingress, prof, want) in enumerate(((salted, profile, True), (plain, profile, False),
                                               (salted, PROFILE, False))):
        (tmp_path / str(k)).mkdir()
        run = Run(tmp_path / str(k), prof)
        assert [run.round(w, egress[w], ingress[w]).accepted for w in range(WINDOWS)] == [want] * WINDOWS
        assert run.lw.network_verdict(run.stream)["accepted"] is want


def _tampered(edit):
    """Window 1's egress record with the frames edited, beside the honest records."""
    frames = frames_of(ARRIVALS)
    first = next(k for k, f in enumerate(frames) if f.anchor >= P.buckets)
    return egress_records(*edit(frames, first))[1]


@pytest.mark.parametrize("kind", ["late", "extra", "dropped"])
def test_a_late_block_an_extra_frame_or_a_dropped_frame_is_refused(tmp_path, kind):
    """Window 1 with its first response's frames handed two buckets late, one frame more, or one frame fewer: the
    certificate verifies and the audit, comparing no grid, accepts, but the counts aren't the derived ones."""
    edits = {
        "late": lambda fr, k: (fr, [f.anchor + (4 if k <= i < k + F else 0) for i, f in enumerate(fr)]),
        "extra": lambda fr, k: (fr[:k + F] + [LogicalFrame(b"x", fr[k].anchor, F)] + fr[k + F:],),
        "dropped": lambda fr, k: (fr[:k + F - 1] + fr[k + F:],),
    }
    run = Run(tmp_path)
    egress, ingress = egress_records(frames_of(ARRIVALS)), ingress_records(ARRIVALS)
    bad = _tampered(edits[kind])
    assert occupancy(bad.grid) != occupancy(egress[1].grid)
    outs = [run.round(0, egress[0], ingress[0]), run.round(1, bad, ingress[1]), run.round(2, egress[2], ingress[2])]
    assert [o.accepted for o in outs] == [True, False, True]
    verdict = run.lw.network_verdict(run.stream)
    assert not verdict["accepted"] and verdict["refused"] == [1]


def test_a_request_that_straddles_the_windows_end_is_refused_and_a_held_one_accepted(tmp_path):
    """A request in window 0's bucket 6 is answered in buckets 7 to 9 under δ = 1, ρ = 1: window 0 doesn't end idle
    and window 1's counts carry frames its ingress counts don't derive, so both are refused. Behind the proxy's
    admission rule (`Admission`, `active.Bucketer`) the same arrivals are admitted at window 1's start instead, and
    every window is accepted."""
    arrivals = ([1, 0, 0, 0, 0, 0, 1, 0], [0] * 8, [0] * 8)
    bucketer = Bucketer(Certifier(LINK, P), served=SV.Admission(PROFILE, P.allocation, lambda w: SYNCS[w]))
    for t, n in enumerate(c for counts in arrivals for c in counts):
        for _ in range(n):
            bucketer.arrive(b"q", t)
    held = [r for r in (bucketer.tick()[1] for _ in range(WINDOWS * P.buckets)) if r is not None]
    assert [occupancy(r.grid) for r in held[:2]] == [(1,) + (0,) * 7] * 2
    for ingress, want in ((ingress_records(arrivals), [False, False, True]), (held, [True] * WINDOWS)):
        (tmp_path / str(all(want))).mkdir()
        run = Run(tmp_path / str(all(want)))
        egress = egress_records(frames_of([occupancy(r.grid) for r in ingress]))
        assert [run.round(w, egress[w], ingress[w]).accepted for w in range(WINDOWS)] == want
        assert run.lw.network_verdict(run.stream)["accepted"] is all(want)


@pytest.mark.parametrize("own", [0, 1])
def test_behind_the_admission_rule_every_window_derives_the_programs_grid_under_load(own):
    """Random dense arrivals, sizes and syncs, several requests a window, salted or not: behind the admission rule
    every request is admitted, whole, in a window that still ends idle with it, so each window's derived counts are the
    Program's grid under its sync (`schedule.expected_grids` of each response anchored where the bucketer released its
    request), the counts N2's served check accepts."""
    rng = random.Random(3051 + own)
    syncs = list(SyncSet((0, 1, 2, 3), (Fraction(1), Fraction(2), Fraction(1, 2), Fraction(2, 3))))
    checked = 0
    for _ in range(150):
        request_frames, response_frames, windows = rng.choice((1, 2)), rng.randint(1, 4), 8
        profile = SV.ServedProfile(response_frames, request_frames, windows, own)
        run = [rng.choice(syncs) for _ in range(windows)]
        try:
            bucketer = Bucketer(Certifier(LINK, P), (lambda w: b"s") if own else None,
                                served=SV.Admission(profile, P.allocation, run.__getitem__))
        except ValueError:
            continue
        n = request_frames * rng.randint(2, 10)
        for t in sorted(rng.randrange(2 * P.buckets) for _ in range(n)):
            bucketer.arrive(b"q", t)
        ingress = [r for r in (bucketer.tick()[1] for _ in range(windows * P.buckets)) if r is not None]
        arrivals = [list(occupancy(r.grid)) for r in ingress]
        assert all(a[0] >= own for a in arrivals)
        requests = [[a[0] - own, *a[1:]] for a in arrivals]
        if sum(map(sum, requests)) < n:
            continue
        grids = expected_grids(frames_of(requests, request_frames, response_frames), run, P)
        derived = [SV.derive(a, s, profile, P.allocation) for a, s in zip(arrivals, run)]
        assert derived == [list(occupancy(g)) for g in grids]
        checked += 1
    assert checked > 50


def test_derive_is_the_programs_grid_on_an_idle_run():
    """On random runs whose every window ends idle, each window's derived counts are the Program's grid's (`schedule.
    expected_grids`) under its own sync, from that window's ingress counts alone."""
    rng = random.Random(54)
    syncs = list(SyncSet((0, 1, 2, 3), (Fraction(1), Fraction(2), Fraction(1, 2), Fraction(2, 3))))
    checked = 0
    for _ in range(300):
        request_frames, response_frames = rng.choice((1, 2)), rng.randint(1, 4)
        profile = SV.ServedProfile(response_frames, request_frames, 3)
        arrivals = [[rng.choice((0, 0, 1, 2)) for _ in range(3)] + [0] * (P.buckets - 3) for _ in range(3)]
        run = [rng.choice(syncs) for _ in range(3)]
        derived = [SV.derive(a, s, profile, P.allocation) for a, s in zip(arrivals, run)]
        if None in derived:
            continue
        grids = expected_grids(frames_of(arrivals, request_frames, response_frames), run, P)
        assert derived == [list(occupancy(g)) for g in grids]
        checked += 1
    assert checked > 50


def test_a_window_that_doesnt_end_idle_has_no_derivation():
    """A request cut by the window's end, a frame handed after its last bucket, and frames handed in time but still
    queued at its end: each leaves the window not idle, so nothing derives its counts."""
    sync, one = ClockSync(1, Fraction(1)), [1, 0, 0, 0, 0, 0, 0, 0]
    assert SV.derive([0] * 8, sync, PROFILE, 2) == [0] * 8
    assert SV.derive(one, sync, SV.ServedProfile(3, 2, 1), 2) is None
    assert SV.derive([0, 0, 0, 0, 0, 0, 1, 0], sync, PROFILE, 2) is None
    four = SV.ServedProfile(4, 1, 1)
    assert SV.derive(one, ClockSync(4, Fraction(4)), four, 1) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert SV.derive(one, ClockSync(5, Fraction(4)), four, 1) is None


KEYS = [(8, 3, 1, 2, 2, 0), (12, 3, 2, 2, 2, 1), (10, 5, 1, 3, 1, 0), (9, 2, 2, 1, 3, 0), (10, 3, 1, 2, 2, 1),
        (9, 2, 2, 1, 3, 1)]
EDGE = [(0, 1, 1), (1, 1, 1), (2, 2, 1), (1, 1, 2), (0, 5, 3), (25, 1, 1), (1, 1, 40), (2, 7, 5), (0, 0, 1), (5, 0, 0),
        (65535, 1, 1), (1, 65535, 65535), (0, 3, 65535)]


def _cases(key, rng, n):
    T, F_, FIN_, R, RIN, OWN = key
    cases = []
    for _ in range(n):
        sync = rng.choice(EDGE)
        if rng.random() < 0.6:
            ingress = [0] * T
            ingress[0] = OWN if rng.random() < 0.9 else 1 - OWN
            for _ in range(rng.randint(0, 3)):
                t = rng.randrange(T)
                ingress[t] = min(RIN, ingress[t] + FIN_)
        else:
            ingress = [rng.choice([0, 0, 1, FIN_, RIN, RIN + 1]) for _ in range(T)]
        profile = SV.ServedProfile(F_, FIN_, 1, OWN)
        d = SV.derive(ingress, ClockSync(sync[0], Fraction(sync[1], sync[2])), profile, R) \
            if sync[1] and sync[2] else None
        if d is not None and rng.random() < 0.8:
            egress = list(d)
            if rng.random() < 0.3:
                k = rng.randrange(T)
                egress[k] = max(0, egress[k] + rng.choice([-1, 1, 2, 1 << 20]))
        else:
            egress = [rng.randint(0, R) for _ in range(T)]
        cases.append((ingress, egress, sync))
    return cases


@pytest.mark.parametrize("key", KEYS)
def test_the_circuit_is_its_reference(key):
    """Random windows of each size class, salted and not, idle and not, with derived counts, counts one off and counts
    past the allocation, under syncs that include p or q = 0 and δ, p and q at 2^16 − 1, and windows whose first
    bucket lacks the own frame or has one more."""
    cases = _cases(key, random.Random(sum(key)), 300)
    want = [int(SV.reference(*key, *c)) for c in cases]
    assert SV.evaluate(key, cases) == want and 0 < sum(want) < len(want)


def test_the_served_unit_is_n2s_beside_the_served_counts(tmp_path):
    """The served unit holds N2's checks and the served counts as one gate, the spec declares the window's sync and the
    served sizes public, and a served link has no egress occupancy profile and one T and τ_b on both ends."""
    lw = Run(tmp_path).lw
    assert lw.unit.id == "NetworkServedLinkWindow_v4{EGRESS=(9,8,0),INGRESS=(9,8,0),SERVED=(8,3,1,2,2,0)}"
    assert lw.spec.public == PUBLIC_SERVED and {"sync", "served"} <= {p.item for p in lw.spec.public}
    assert lw.spec.digest() != replace(lw, served=SV.ServedProfile(4, 1, WINDOWS)).spec.digest()
    assert replace(lw, served=replace(PROFILE, own=1)).unit.id.endswith("SERVED=(8,3,1,2,2,1)}")
    with pytest.raises(ValueError, match="no egress occupancy"):
        replace(lw, occupancy=(((0,) * 8,), None))
    with pytest.raises(ValueError, match="share T"):
        replace(lw, ingress=GridParams(4, 2, 1))
    with pytest.raises(ValueError, match="below 2"):
        replace(lw, sync_set=SyncSet((1 << 16,), (1,)))
    with pytest.raises(ValueError, match="power of two"):
        SV.ServedProfile(3, 3, 1)
    with pytest.raises(ValueError, match="0 or 1 own"):
        SV.ServedProfile(3, 1, 1, 2)


def _derives_any_counts(B, S_, *args):
    return B.call(BL.constant(1))


@pytest.mark.parametrize("key", [(8, 3, 1, 2, 2, 0), (12, 3, 2, 2, 2, 1)])
def test_the_served_gate_names_its_pinned_circuit_digest(key):
    """The served counts' gate names its Definition's circuit digest as `served.circuit-check.json` pins it, which is the
    Definition's own (`head_check.expand`, cheap at the tests' sizes, where circuit-check holds the deployed ones), and
    served counts with the same id and signature that accept any counts expand no gate of it."""
    d, gate = SV.unit(*key), SV.gate(*key)
    pinned = json.loads(SV.PINS.read_text())["digests"][d.id]["sha512"]
    assert gate.id == f"Gate[{d.id}@{pinned}]_v2" and HC.pinned(gate) and HC.expand(gate, d) is d
    forged = bind(replace(SV.NetworkServedCounts, body=_derives_any_counts, register=False),
                  **dict(zip(("T", "FOUT", "FIN", "R", "RIN", "OWN"), key)))
    assert forged.id == d.id and forged.signature() == d.signature()
    with pytest.raises(ValueError, match="another circuit"):
        HC.expand(gate, forged)
