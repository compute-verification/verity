"""The ingress tie through the proof service (`tie_service`): each ingress link-window's IoTieRequest proved (a
stand-in proof, `standin.prove_tie`, decided by the verifier of record on its own statement), and `advised.IoTieVerdict`
built from the stream's outcomes of record alone. An honest link is tied, `advised.ends` reads the tie, and the advised
link takes it only from `Ties`, the tie and its stream as its source of record; REDTEAM
phase 9's carried field count, its forged advice and phase 6's split are refused, as are an outcome on another advice,
in the clear or late, a stream short of the advice's windows, out of order or past them, a profile that frames
requests otherwise than the tie cuts them, and an unpinned size class. Nothing in `verity` but the tie service builds
the verdict."""

import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.core.protocols.certifier import advice as ADV, advised as AD, head_check as HC, io_request as Q, \
    io_tie as IT, service as SVC, tie_service as TS
from verity.core.protocols.certifier.active import id_request
from verity.core.protocols.certifier.grid import GridParams
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import registration as R

import signers
import standin
from advised_traffic import StandInClocks

B = 512
SHAPE = Q.Shape(B, 4, 2, 1)
LINK, RUN = "requests", "r-tie-test"
#: REDTEAM phase 9's span: request 1 (40 tokens) takes 2 frames, one in each window, behind each window's own frame
SPAN, COUNTS = (5, 40, 5), (1, 0, 0, 1, 1, 1, 0, 0)
FRAMING = TS.framing(B, 64)
PROFILE = AD.AdvisedProfile(2, FRAMING, 1)
#: the prover proves on the verifier's own view of the advice
OWN_VIEW = object()


@pytest.fixture(autouse=True)
def lean(monkeypatch):
    monkeypatch.setattr(SVC.PF, "verify", standin.lean)


def _requests(prompt_lens):
    return [id_request(model="glm", user=f"u{i}", prompt=[(7919 * i + 104729 * k) % 151329 for k in range(n)],
                       max_tokens=16, temperature=0.7, top_p=0.95, seed=i, frame_bytes=B)
            for i, n in enumerate(prompt_lens)]


def _received(counts, shape=SHAPE):
    return tuple(sum(counts[w * shape.T:(w + 1) * shape.T]) for w in range(len(counts) // shape.T))


def _advice(prompt_lens, received, release=None):
    """An advice v0 of one prefill step, each request one kept output, with the link's `received`."""
    requests = tuple(ADV.RequestAdvice(f"u{j}", n, 1, "eos") for j, n in enumerate(prompt_lens))
    steps = (ADV.StepAdvice(tuple(ADV.Row(j, "prefill", 0, n) for j, n in enumerate(prompt_lens))),)
    release = tuple(range(len(prompt_lens))) if release is None else tuple(release)
    return AD.LinkAdvice(ADV.Advice(requests, steps, release), tuple(received))


HONEST = Q.windows_of(SHAPE, _requests(SPAN), COUNTS)
RECEIVED = _received(COUNTS)


class Run:
    """An ingress link's tie: the warden's commitments, the challenger's stream on injected coins and clock, and each
    window decided by the verifier of record on a stand-in proof, the in-the-clear check beside it."""

    def __init__(self, tmp_path, shape=SHAPE) -> None:
        self.tmp, self.now, self.previous = tmp_path, [0.0], None
        self.tie = TS.IngressTie(LINK, shape, RUN, proof_deadline=60.0)
        self.stream = self.tie.stream(budget=8, delta=2.0 ** -40, challenger="challenger-test",
                                      coins=lambda n: b"\x07" * n, clock=lambda: self.now[0])

    def draw(self, w, cells):
        c = self.tie.commit(w, cells, [IT.fake_salt(b"warden" + bytes([w]), t) for t in range(self.tie.shape.T)])
        reg = self.tie.register(self.stream, self.tie.registration(c, self.previous))
        self.previous = reg.digest
        return c, self.stream.select(reg)

    def prove(self, w, cells, advice, view=OWN_VIEW, **kw):
        c, draw = self.draw(w, cells)
        view = self.tie.view(advice, w) if view is OWN_VIEW else view
        return c, draw, standin.prove_tie(self.tie, c, draw, view, self.tmp / f"tie-{draw.registration.order}", **kw)

    def round(self, w, cells, advice, view=OWN_VIEW, **kw) -> S.Outcome:
        c, draw, delivered = self.prove(w, cells, advice, view, **kw)
        late = self.stream.deliver(draw, delivered.sessions())
        if late is not None:
            return late
        outcome = self.tie.outcome(self.stream, draw, delivered, circuit_sha512=standin.circuit_sha512(delivered),
                                   advice=advice, opening=c)
        verdict = self.stream.records[-1]["audit"]["verdicts"][TS.VERIFIER]
        assert verdict["decided_by"] == "proof" and verdict["accepted"] == outcome.accepted
        if view is OWN_VIEW and not kw:
            assert verdict["clear"]["accepted"] == outcome.accepted
        return outcome

    def link(self, windows, advice) -> list[bool]:
        return [self.round(w, cells, advice).accepted for w, cells in enumerate(windows)]

    def held(self, verdict):
        """`advised.ends`' keyword arguments under `verdict`: it, the link and the source's registrations."""
        return {"tie": verdict, "link_name": LINK, "registrations": TS.Ties(self.tie, self.stream).registrations(LINK)}


def test_an_honest_link_is_tied_and_advised_ends_reads_the_tie(tmp_path):
    run, advice = Run(tmp_path), _advice(SPAN, RECEIVED)
    assert run.tie.layout["wit"] == ("cells", "salts", "switches", "pad") and run.tie.pad == 12
    assert run.link(HONEST, advice) == [True, True]
    verdict = run.tie.tie(run.stream, advice, PROFILE)
    regs = tuple(R.digest(r) for r in run.stream.log)
    assert verdict == AD.IoTieVerdict(LINK, regs, AD.digest(advice), (True, True))
    assert AD.ends(advice, PROFILE.request_frames, **run.held(verdict)) == Q.ends(SPAN, B) == (1, 3, 4)
    public = run.stream.records[-1]["public"]
    assert public == {"verdict": 1, "advice": AD.digest(advice)}


def test_the_advised_link_reads_the_tie_only_from_its_source_of_record(tmp_path):
    """`Ties`, the tie and its stream as `LinkWindows.ties`: the advised link of the tie's name takes each window's view
    under the verdict it gives; a link of another name takes none (the source refuses another link), as with no
    source, and so does the link under the source of a stream whose outcomes are on a forged advice (REDTEAM phase 9)
    or of another link's tie."""
    advice = _advice(SPAN, RECEIVED, release=(1, 3, 5))
    run = Run(tmp_path / "honest")
    assert run.link(HONEST, advice) == [True, True]
    ties = TS.Ties(run.tie, run.stream)
    assert ties.registrations(LINK) == tuple(R.digest(r) for r in run.stream.log)
    with pytest.raises(ValueError, match="not 'another'"):
        ties.verdict("another", advice, PROFILE)
    forged = Run(tmp_path / "forged")
    assert forged.link(HONEST, _advice((5, 45, 5), RECEIVED, release=(1, 3, 5))) == [False, False]
    elsewhere = Run(tmp_path / "elsewhere")
    elsewhere.tie = TS.IngressTie("another", SHAPE, RUN, proof_deadline=60.0)
    elsewhere.stream = elsewhere.tie.stream(budget=8, delta=2.0 ** -40, challenger="challenger-test",
                                            coins=lambda n: b"\x07" * n, clock=lambda: 0.0)
    assert elsewhere.link(HONEST, advice) == [True, True]
    with pytest.raises(ValueError, match="its own stream"):
        TS.Ties(run.tie, elsewhere.stream)
    key = signers.signer(tmp_path).public_key()
    egress, ingress = GridParams(SHAPE.T, 3, 1), GridParams(SHAPE.T, SHAPE.R, 1)
    sync_set, syncs = SyncSet((1, 2), (2, 4)), [ClockSync(1, 2)] * 2

    def views(name, source):
        lw = SVC.LinkWindows(name, egress, sync_set, key, RUN, signers.SESSION, ingress=ingress, proof_deadline=60.0,
                             advised=PROFILE, ties=source, clocks=StandInClocks(tuple(syncs)))
        return [lw._view(w, syncs, advice) for w in range(2)]

    assert None not in views(LINK, ties)
    assert views("another", ties) == views(LINK, None) == [None, None]
    assert views(LINK, TS.Ties(forged.tie, forged.stream)) == [None, None]
    assert views(LINK, TS.Ties(elsewhere.tie, elsewhere.stream)) == [None, None]


@pytest.mark.parametrize("delta", [1, 5, 100])
def test_a_proof_on_an_added_carried_field_count_is_refused(tmp_path, delta):
    """REDTEAM phase 9 (b) through the verifier: IoTieRequest accepts window 1's cells on its carry-in's fields + Δ
    with its first ended request's + Δ, but the verifier's statement is its own, from the advice, so a proof on that
    view opens to no public row of the verifier's and window 1 is refused."""
    run, advice = Run(tmp_path), _advice(SPAN, RECEIVED)
    v1 = run.tie.view(advice, 1)
    added = v1._replace(carry_in=v1.carry_in._replace(fields=v1.carry_in.fields + delta),
                        ended=(v1.ended[0] + delta,) + v1.ended[1:])
    assert run.round(0, HONEST[0], advice).accepted
    c, draw, delivered = run.prove(1, HONEST[1], advice, added)
    assert delivered.output == 1, "IoTieRequest accepts the cells on the added view"
    assert run.stream.deliver(draw, delivered.sessions()) is None
    outcome = run.tie.outcome(run.stream, draw, delivered, circuit_sha512=standin.circuit_sha512(delivered),
                              advice=advice, opening=c)
    assert not outcome.accepted
    assert run.stream.records[-1]["audit"]["verdicts"][TS.VERIFIER]["clear"]["accepted"]
    assert run.tie.tie(run.stream, advice, PROFILE).accepted == (True, False)


def test_a_forged_advice_is_refused_from_window_0(tmp_path):
    """REDTEAM phase 9's forged advice, 40 → 45 prompt fields, keeps every request's frames, so `advised.ends` can't
    see it: the cells' carry-out after window 0 isn't the forged request's, so both windows are refused, and
    `advised.ends` refuses the link."""
    forged = _advice((5, 45, 5), RECEIVED)
    assert Q.ends((5, 45, 5), B) == Q.ends(SPAN, B)
    run = Run(tmp_path)
    assert run.link(HONEST, forged) == [False, False]
    verdict = run.tie.tie(run.stream, forged, PROFILE)
    assert verdict.accepted == (False, False)
    with pytest.raises(ValueError, match="doesn't accept each"):
        AD.ends(forged, PROFILE.request_frames, **run.held(verdict))


def test_redteam_phase6_split_is_refused(tmp_path):
    """REDTEAM phase 6's split at the tie's size class: prompts of 40 and 5 tokens take 2 frames and 1, and the forged
    advice swaps them, keeping the total, which lowers request 0's end from 2 to 1 and so the advised floor. The tie
    reads 40 prompt fields in request 0's cells and refuses window 0, so `advised.ends` refuses the link."""
    pl, counts = (40, 5), (1, 1, 1, 0, 0, 0, 0, 0)
    windows, received = Q.windows_of(SHAPE, _requests(pl), counts), _received(counts)
    assert Q.ends(pl, B) == (2, 3) and Q.ends(pl[::-1], B) == (1, 3)
    honest, split = _advice(pl, received), _advice(pl[::-1], received)
    run = Run(tmp_path / "honest")
    assert run.link(windows, honest) == [True, True]
    assert run.tie.tie(run.stream, honest, PROFILE).accepted == (True, True)
    run = Run(tmp_path / "split")
    assert run.link(windows, split) == [False, True]
    verdict = run.tie.tie(run.stream, split, PROFILE)
    assert verdict.accepted == (False, True)
    with pytest.raises(ValueError):
        AD.ends(split, PROFILE.request_frames, **run.held(verdict))


def test_an_outcome_on_another_advice_in_the_clear_or_late_is_refused(tmp_path):
    """Each window's outcome of record must be decided on the proof and name the advice tied: window 1 decided on
    another advice of the same view (another release), decided in the clear, or delivered past the proof deadline
    refuses window 1 alone."""
    advice = _advice(SPAN, RECEIVED)
    other = _advice(SPAN, RECEIVED, release=(5, 6, 7))
    assert AD.digest(other) != AD.digest(advice) and TS.prompt_lens(other) == TS.prompt_lens(advice)

    run = Run(tmp_path / "other")
    assert run.round(0, HONEST[0], advice).accepted and run.round(1, HONEST[1], other).accepted
    assert run.tie.tie(run.stream, advice, PROFILE).accepted == (True, False)
    assert run.tie.tie(run.stream, other, PROFILE).accepted == (False, True)

    run = Run(tmp_path / "clear")
    assert run.round(0, HONEST[0], advice).accepted
    c, draw, delivered = run.prove(1, HONEST[1], advice)
    assert run.stream.deliver(draw, delivered.sessions()) is None
    verdict = run.tie.verify(draw, delivered, circuit_sha512=standin.circuit_sha512(delivered), advice=advice)
    assert verdict["accepted"]
    outcome = run.stream.outcome(draw, {TS.VERIFIER: {**verdict, "decided_by": "clear"}}, run.tie.statement(draw),
                                 session_roots={TS.VALUE: draw.registration.record["roots"][0]["root"]})
    assert outcome.accepted
    assert run.tie.tie(run.stream, advice, PROFILE).accepted == (True, False)

    run = Run(tmp_path / "late")
    assert run.round(0, HONEST[0], advice).accepted
    c, draw, delivered = run.prove(1, HONEST[1], advice)
    run.now[0] = 61.0
    late = run.stream.deliver(draw, delivered.sessions())
    assert late is not None and not late.accepted
    assert run.tie.tie(run.stream, advice, PROFILE).accepted == (True, False)


def test_a_short_stream_one_out_of_order_or_one_past_the_advice_refuses_every_window(tmp_path):
    """The verdict is one bit a registration of the stream, each refused, and `advised.ends` refuses it."""
    advice = _advice(SPAN, RECEIVED)
    run = Run(tmp_path / "short")
    assert run.round(0, HONEST[0], advice).accepted
    assert run.tie.tie(run.stream, advice, PROFILE).accepted == (False,)
    with pytest.raises(ValueError, match="doesn't accept each"):
        AD.ends(advice, PROFILE.request_frames, **run.held(run.tie.tie(run.stream, advice, PROFILE)))

    run = Run(tmp_path / "order")
    c = run.tie.commit(1, HONEST[1])
    reg = run.stream.register(run.tie.registration(c), run.tie.ports(1))
    assert reg.record["window"] == run.tie.provenance(1)
    with pytest.raises(S.Refused):
        run.tie.register(run.stream, run.tie.registration(run.tie.commit(0, HONEST[0]), reg.digest))
    assert run.tie.tie(run.stream, advice, PROFILE).accepted == (False,)

    run = Run(tmp_path / "past")
    assert run.link(HONEST, advice) == [True, True]
    own = [Q.own_frame(2, B)] + [IT.pad_cell(B)] * (SHAPE.N - 1)
    assert not run.round(2, own, advice).accepted
    assert run.tie.tie(run.stream, advice, PROFILE).accepted == (False, False, False)


@pytest.mark.parametrize("profile", [AD.AdvisedProfile(2, tuple(n + 1 for n in FRAMING), 1),
                                     AD.AdvisedProfile(2, FRAMING[:39], 1), AD.AdvisedProfile(2, FRAMING, 0)],
                         ids=["framing", "short-framing", "own"])
def test_a_profile_that_frames_requests_otherwise_than_the_tie_refuses_every_window(tmp_path, profile):
    """`advised.ends` reads the profile's framing, so a verdict under one that isn't IoTieRequest's cut at B for the
    advice's prompt lengths, or under another OWN, would hold another link's ends: every window is refused."""
    run, advice = Run(tmp_path), _advice(SPAN, RECEIVED)
    assert run.link(HONEST, advice) == [True, True]
    assert run.tie.tie(run.stream, advice, profile).accepted == (False, False)


def _accepts_any_cells(B, S_, *args):
    return B.call(BL.constant(1))


def test_the_tie_gate_names_its_pinned_circuit_digest():
    """The tie's gate names IoTieRequest's circuit digest as `io_request.circuit-check.json` pins it (circuit-check
    holds the pin to the Definition), and an IoTieRequest with the same id and signature that accepts any cells expands
    no gate of it."""
    d, gate = bind(Q.IoTieRequest, B=B, T=4, R=2, OWN=1), TS.gate(B, 4, 2, 1)
    pinned = json.loads(Q.PINS.read_text())["digests"][d.id]["sha512"]
    assert gate.id == f"Gate[{d.id}@{pinned}]_v2" and HC.pinned(gate)
    forged = bind(replace(Q.IoTieRequest, body=_accepts_any_cells, register=False), B=B, T=4, R=2, OWN=1)
    assert forged.id == d.id and forged.signature() == d.signature()
    with pytest.raises(ValueError, match="another circuit"):
        HC.expand(gate, forged)


def test_an_unpinned_size_class_is_refused(tmp_path):
    """IoTieRequest at OWN = 0 has no pinned circuit digest, so its gate names none and no proof of its unit is
    accepted, even one Lean accepts."""
    shape = Q.Shape(B, 4, 2, 0)
    run, advice = Run(tmp_path, shape), _advice(SPAN, RECEIVED)
    assert not HC.pinned(run.tie.gates()[0])
    windows = Q.windows_of(shape, _requests(SPAN), COUNTS)
    c, draw, delivered = run.prove(0, windows[0], advice)
    assert delivered.output == 1
    assert run.stream.deliver(draw, delivered.sessions()) is None
    assert not run.tie.outcome(run.stream, draw, delivered, circuit_sha512=standin.circuit_sha512(delivered),
                               advice=advice).accepted


def test_only_the_tie_service_builds_an_ingress_tie_verdict():
    """`advised.ends` takes the tie only as `advised.IoTieVerdict`, so nothing in `verity` but the tie service, which
    builds it from the stream's outcomes of record, constructs one (the tests' stand-ins aside)."""
    root = Path(TS.__file__).resolve().parents[3]
    builders = set()
    for path in root.rglob("*.py"):
        if "tests" in path.relative_to(root).parts or "IoTieVerdict" not in path.read_text():
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            f = node.func if isinstance(node, ast.Call) else None
            if (isinstance(f, ast.Name) and f.id == "IoTieVerdict") or \
                    (isinstance(f, ast.Attribute) and f.attr == "IoTieVerdict"):
                builders.add(path.relative_to(root).as_posix())
    assert builders == {"core/protocols/certifier/tie_service.py"}
