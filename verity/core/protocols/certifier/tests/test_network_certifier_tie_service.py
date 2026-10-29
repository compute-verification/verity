"""The ingress tie through the proof system (`tie_service`): each bucket row of an ingress link-window of the warden's
cells proved as IoTieRequest (a stand-in proof, `standin.prove_tie`, its rows read off the warden's root and decided by
the verifier of record on its own lanes), and `advised.IoTieVerdict` built from the stream's outcomes of record alone.
An honest link is tied, `advised.ends` reads the tie, and the advised link takes it only from `Ties`, the tie, its
stream and the release as its source of record; the verifier's registered reads name every port of the statement, as
C-Flock's Lean verifier requires (`Registered.checkComplete`, which the stand-in mirrors); a carried field count, a
forged advice and the swapped split are refused, as are an output 0 or rows under fresh salts, an outcome on another
advice, in the clear or late, a stream short of the advice's windows, out of order or past them, a profile that frames
requests otherwise than the warden cuts them, a row that isn't the warden's, a head of another window, an advice whose
`received` isn't the release's, and an unpinned size class. Nothing in `verity` but the tie service builds the
verdict."""

import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.core.protocols.certifier import advice as ADV, advised as AD, head_check as HC, io_request as Q, \
    service as SVC, tie_service as TS, warden as W, warden_rows as WR
from verity.core.protocols.certifier.active import id_request
from verity.core.protocols.certifier.certificate import Direction
from verity.core.protocols.certifier.grid import GridParams
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import registered as G, registration as R

import signers
import standin
from advised_traffic import StandInClocks

B = 512
#: two lanes of r = 2 slots, 4-bucket windows
PARAMS = W.WardenParams(cell_bytes=B, slots=2, lanes=2, window=4, delay=1)
LINK, RUN = "requests", "r-tie-test"
SESSION = hashlib.sha512(b"tie-test").digest()
#: a span: request 1 (40 tokens) takes 2 cells, its DATA in bucket 3 and its END in bucket 4, so it
#: spans the two windows; requests 0 and 2 take one cell each, one in each window
SPAN, ARRIVALS, RATES = (5, 40, 5), (0, 2, 4), (10 ** 6, 600, 10 ** 6)
RECEIVED = (2, 2)
FRAMING = TS.framing(B, 64)
PROFILE = AD.AdvisedProfile(2, FRAMING, 0)
#: the prover proves on the verifier's own lanes of the advice
OWN_LANES = object()


@pytest.fixture(autouse=True)
def lean(monkeypatch):
    monkeypatch.setattr(SVC.PF, "verify", standin.lean)


def _requests(prompt_lens):
    return [id_request(model="glm", user=f"u{i}", prompt=[(7919 * i + 104729 * k) % 151329 for k in range(n)],
                       max_tokens=16, temperature=0.7, top_p=0.95, seed=i, frame_bytes=B, offset=0)
            for i, n in enumerate(prompt_lens)]


def _advice(prompt_lens, received=RECEIVED, release=None):
    """An advice v0 of one prefill step, each request one kept output, with the link's `received`."""
    requests = tuple(ADV.RequestAdvice(f"u{j}", n, 1, "eos") for j, n in enumerate(prompt_lens))
    steps = (ADV.StepAdvice(tuple(ADV.Row(j, "prefill", 0, n) for j, n in enumerate(prompt_lens))),)
    release = tuple(range(len(prompt_lens))) if release is None else tuple(release)
    return AD.LinkAdvice(ADV.Advice(requests, steps, release), tuple(received))


def _serve(prompt_lens=SPAN, params=PARAMS, windows=3):
    """The link served through the warden's own ingress, a third window past the advice's empty."""
    return Q.serve(params, _requests(prompt_lens), ARRIVALS, RATES, buckets=windows * params.window)


SERVED = _serve()


class Run:
    """An ingress link's tie: the warden's cells of each window (`warden_rows.Cells`, under fixed salts), the
    challenger's stream on injected coins and clock, and each window decided by the verifier of record on a stand-in
    proof, the in-the-clear check beside it."""

    def __init__(self, tmp_path, link=LINK, params=PARAMS, served=SERVED) -> None:
        self.tmp, self.now, self.previous, self.served = tmp_path, [0.0], None, served
        self.tie = TS.IngressTie(link, params, SESSION, proof_deadline=60.0)
        self.stream = self.tie.stream(budget=8, delta=2.0 ** -40, challenger="challenger-test",
                                      coins=lambda n: b"\x07" * n, clock=lambda: self.now[0])
        self.cells = [WR.Cells.of(link, Direction.INGRESS, rec, params, SESSION,
                                  [hashlib.sha512(b"salt %d %d" % (w, t)).digest() * 3 for t in range(params.window)],
                                  hashlib.sha512(b"wiring %d" % w).digest())
                      for w, rec in enumerate(served.records)]

    def draw(self, w):
        reg = self.tie.register(self.stream, self.tie.registration(self.cells[w].head, self.previous))
        self.previous = reg.digest
        return self.stream.select(reg)

    def prove(self, w, advice, lanes=OWN_LANES, releases=None, **kw):
        draw = self.draw(w)
        releases = self.served.releases if releases is None else releases
        lanes = self.tie.lanes(advice, releases, w) if lanes is OWN_LANES else lanes
        return draw, standin.prove_tie(self.tie, self.cells[w], draw, lanes,
                                       self.tmp / f"tie-{draw.registration.order}", **kw)

    def decide(self, w, draw, delivered, advice, head=None):
        return self.tie.outcome(self.stream, draw, delivered, circuit_sha512=standin.circuit_sha512(delivered),
                                advice=advice, releases=self.served.releases,
                                head=self.cells[w].head if head is None else head, opening=self.cells[w])

    def round(self, w, advice, lanes=OWN_LANES, **kw) -> S.Outcome:
        draw, delivered = self.prove(w, advice, lanes, **kw)
        late = self.stream.deliver(draw, delivered.sessions())
        if late is not None:
            return late
        outcome = self.decide(w, draw, delivered, advice)
        verdict = self.stream.records[-1]["audit"]["verdicts"][TS.VERIFIER]
        assert verdict["decided_by"] == "proof" and verdict["accepted"] == outcome.accepted
        if lanes is OWN_LANES and not kw:
            assert verdict["clear"]["accepted"] == outcome.accepted
        return outcome

    def link(self, advice, windows=2) -> list[bool]:
        return [self.round(w, advice).accepted for w in range(windows)]

    def verdict(self, advice, profile=PROFILE):
        return self.tie.tie(self.stream, advice, profile, self.served.releases)

    def held(self, verdict):
        """`advised.ends`' keyword arguments under `verdict`: it, the link and the source's registrations."""
        source = TS.Ties(self.tie, self.stream, self.served.releases)
        return {"tie": verdict, "link_name": self.tie.link, "registrations": source.registrations(self.tie.link)}


def test_an_honest_link_is_tied_and_advised_ends_reads_the_tie(tmp_path):
    run, advice = Run(tmp_path), _advice(SPAN)
    assert run.tie.layout == {"pub": ("lanes",), "wit": ("row",)} and run.tie.T == 4
    assert [r.cells for r in SERVED.releases] == [(0, 1), (0, 1, 1), (0, 1)]
    assert run.link(advice) == [True, True]
    verdict = run.verdict(advice)
    regs = tuple(R.digest(r) for r in run.stream.log)
    assert verdict == AD.IoTieVerdict(LINK, regs, AD.digest(advice), (True, True))
    assert AD.ends(advice, PROFILE.request_frames, **run.held(verdict)) == (1, 3, 4)
    assert run.stream.records[-1]["public"] == {"verdict": 1, "advice": AD.digest(advice)}
    assert run.stream.log[0]["roots"][0]["root"] == run.cells[0].root.hex()


def test_the_registered_reads_name_every_port_so_an_honest_window_is_proved(tmp_path):
    """C-Flock's Lean verifier rejects every session of a statement with a port its `--registered` file doesn't name
    (`Registered.checkComplete`), which the stand-in mirrors: the verifier's own file names ``wit`` at the warden's
    root and ``pub`` and ``o0`` at known content, its own lanes of each bucket and the output word 1 under public
    salts, so an honest window's proof is proved and accepted, where a file naming ``wit`` alone rejects it at
    ``pub``."""
    run, advice = Run(tmp_path), _advice(SPAN)
    draw, delivered = run.prove(0, advice)
    head, lanes = run.cells[0].head, run.tie.lanes(advice, SERVED.releases, 0)
    reads = run.tie.registered(head, lanes)
    assert sorted(reads["ports"]) == sorted(TS.PORTS) and TS.KNOWN_KEY not in reads["ports"][TS.REGISTERED]
    assert reads["ports"]["pub"][TS.KNOWN_KEY]["rows"] == [
        WR.words(run.tie.row(run.tie.public_arguments(ln), "pub")) for ln in lanes]
    assert reads["ports"]["o0"][TS.KNOWN_KEY]["rows"] == [[1]] * run.tie.T
    assert {p: reads["ports"][p]["reads"] for p in TS.PORTS} == {p: list(range(run.tie.T)) for p in TS.PORTS}
    assert run.tie.registered(head, lanes[1:]) is None and run.tie.registered(head, None) is None
    kw = {"circuit_sha512": standin.circuit_sha512(delivered), "program": run.tie.program,
          "partition": run.tie.partition}
    findings, refused = standin.lean(delivered.proof, registered=reads, **kw)
    assert refused is None and run.tie.proved(findings)
    wit_only = WR.reads_file({TS.REGISTERED: reads["ports"][TS.REGISTERED]})
    findings, refused = standin.lean(delivered.proof, registered=wit_only, **kw)
    assert refused is None and findings and not any(f.accepted for f in findings)
    assert "port pub is no port of the verifier's registered reads" in findings[0].why
    assert run.stream.deliver(draw, delivered.sessions()) is None
    assert run.decide(0, draw, delivered, advice).accepted


def test_an_output_0_or_rows_under_fresh_salts_fail_the_registered_reads(tmp_path, monkeypatch):
    """The ``o0`` row is read off the verifier's known root of output words 1, and the ``pub`` row off its known root
    of its own lanes under their public salts: the forged advice's proof of window 0, on the warden's rows and the
    verifier's own lanes, fails ``o0``'s read at the bucket IoTieRequest refuses, whatever word it delivers, and a
    proof whose ``pub`` and ``o0`` rows are committed under fresh salts fails ``pub``'s."""
    forged, advice = _advice((5, 45, 5)), _advice(SPAN)
    run = Run(tmp_path / "zero")
    lanes = run.tie.lanes(forged, SERVED.releases, 0)
    assert 0 in [run.tie.outputs(r, ln) for r, ln in zip(run.cells[0].rows, lanes)]
    _, delivered = run.prove(0, forged, words={u: 1 for u in range(4)})
    kw = {"circuit_sha512": standin.circuit_sha512(delivered), "program": run.tie.program,
          "partition": run.tie.partition}
    findings, refused = standin.lean(delivered.proof, registered=run.tie.registered(run.cells[0].head, lanes), **kw)
    assert refused is None and not findings[0].accepted and "port o0: table row" in findings[0].why

    run = Run(tmp_path / "fresh")
    lanes, known_tree = run.tie.lanes(advice, SERVED.releases, 0), TS.IngressTie.known_tree
    monkeypatch.setattr(TS.IngressTie, "known_tree", lambda self, head, port, rows: G.Registered.commit(
        (t := known_tree(self, head, port, rows)).program, t.value, t.rows, None, t.word_bits, t.schema,
        window=t.window))
    draw, delivered = run.prove(0, advice)
    monkeypatch.setattr(TS.IngressTie, "known_tree", known_tree)
    findings, refused = standin.lean(delivered.proof, registered=run.tie.registered(run.cells[0].head, lanes), **kw)
    assert refused is None and not findings[0].accepted and "port pub: table row" in findings[0].why
    assert run.stream.deliver(draw, delivered.sessions()) is None
    assert not run.decide(0, draw, delivered, advice).accepted


def test_the_advised_link_reads_the_tie_only_from_its_source_of_record(tmp_path):
    """`Ties`, the tie, its stream and the release as `LinkWindows.ties`: the advised link of the tie's name takes each
    window's view under the verdict it gives; a link of another name takes none (the source refuses another link), as
    with no source, and so does the link under the source of a stream whose outcomes are on a forged advice
    or of another link's tie."""
    advice = _advice(SPAN, release=(1, 3, 5))
    run = Run(tmp_path / "honest")
    assert run.link(advice) == [True, True]
    ties = TS.Ties(run.tie, run.stream, SERVED.releases)
    assert ties.registrations(LINK) == tuple(R.digest(r) for r in run.stream.log)
    with pytest.raises(ValueError, match="not 'another'"):
        ties.verdict("another", advice, PROFILE)
    forged = Run(tmp_path / "forged")
    assert forged.link(_advice((5, 45, 5), release=(1, 3, 5))) == [False, False]
    elsewhere = Run(tmp_path / "elsewhere", link="another")
    assert elsewhere.link(advice) == [True, True]
    with pytest.raises(ValueError, match="its own stream"):
        TS.Ties(run.tie, elsewhere.stream, SERVED.releases)
    key = signers.signer(tmp_path).public_key()
    egress, ingress = GridParams(PARAMS.window, 3, 1), GridParams(PARAMS.window, PARAMS.grid.allocation, 1)
    sync_set, syncs = SyncSet((1, 2), (2, 4)), [ClockSync(1, 2)] * 2

    def views(name, source):
        lw = SVC.LinkWindows(name, egress, sync_set, key, RUN, signers.SESSION, ingress=ingress, proof_deadline=60.0,
                             advised=PROFILE, ties=source, clocks=StandInClocks(tuple(syncs)))
        return [lw._view(w, syncs, advice) for w in range(2)]

    assert None not in views(LINK, ties)
    assert views("another", ties) == views(LINK, None) == [None, None]
    assert views(LINK, TS.Ties(forged.tie, forged.stream, SERVED.releases)) == [None, None]
    assert views(LINK, TS.Ties(elsewhere.tie, elsewhere.stream, SERVED.releases)) == [None, None]


@pytest.mark.parametrize("delta", [1, 5, 100])
def test_a_proof_on_an_added_carried_field_count_is_refused(tmp_path, delta):
    """The carried field count through the verifier: IoTieRequest accepts bucket 4's row on its lane's carry-in
    fields + Δ with its carry-out's + Δ, but the verifier's lanes are its own, from the advice and the release, so a
    proof on the added lanes opens to no public row of the verifier's and window 1 is refused."""
    run, advice = Run(tmp_path), _advice(SPAN)
    lanes = run.tie.lanes(advice, SERVED.releases, 1)
    ln = lanes[0][0]
    assert (ln.q, ln.left, ln.fields, ln.left_out, ln.fields_out) == (2, 310, 12, 0, 40)
    added = ((ln._replace(fields=ln.fields + delta, fields_out=ln.fields_out + delta),) + lanes[0][1:],) + lanes[1:]
    assert run.round(0, advice).accepted
    draw, delivered = run.prove(1, advice, added)
    assert delivered.outputs == {u: 1 for u in range(4)}, "IoTieRequest accepts each row on the added lanes"
    assert run.stream.deliver(draw, delivered.sessions()) is None
    assert not run.decide(1, draw, delivered, advice).accepted
    assert run.stream.records[-1]["audit"]["verdicts"][TS.VERIFIER]["clear"]["accepted"]
    assert run.verdict(advice).accepted == (True, False)


def test_a_forged_advice_is_refused_from_window_0(tmp_path):
    """A forged advice, 40 → 45 prompt fields, keeps every request's cells, so `advised.ends` and the
    release can't see it: bucket 3's carry-out isn't the forged request's, nor bucket 4's carry-in, so both windows are
    refused, and `advised.ends` refuses the link."""
    forged = _advice((5, 45, 5))
    assert [TS.request_cells(n, B) for n in (5, 45, 5)] == [TS.request_cells(n, B) for n in SPAN]
    run = Run(tmp_path)
    assert run.link(forged) == [False, False]
    verdict = run.verdict(forged)
    assert verdict.accepted == (False, False)
    with pytest.raises(ValueError, match="doesn't accept each"):
        AD.ends(forged, PROFILE.request_frames, **run.held(verdict))


def test_redteam_phase6_split_is_refused(tmp_path):
    """The swapped split at the tie's size class: prompts of 40 and 5 tokens take 2 cells and 1, and the forged
    advice swaps them, keeping the total, which lowers request 0's end from 2 to 1 and so the advised floor. The
    release puts 2 cells on connection 1, which the swapped advice's 5-token request doesn't take, so the verifier's
    lanes refuse every window and `advised.ends` refuses the link."""
    pl = (40, 5)
    served = Q.serve(PARAMS, _requests(pl), (0, 0), (600, 10 ** 6), buckets=2 * PARAMS.window)
    received = tuple(TS.window_cells(served.releases, PARAMS.window).get(w, 0) for w in range(2))
    assert [r.cells for r in served.releases] == [(0, 1, 1), (0, 1)] and received == (3, 0)
    honest, split = _advice(pl, received), _advice(pl[::-1], received)
    run = Run(tmp_path / "honest", served=served)
    assert run.link(honest) == [True, True] and run.verdict(honest).accepted == (True, True)
    run = Run(tmp_path / "split", served=served)
    assert run.tie.lanes(split, served.releases, 0) is None
    assert run.link(split) == [False, False]
    verdict = run.verdict(split)
    assert verdict.accepted == (False, False)
    with pytest.raises(ValueError):
        AD.ends(split, PROFILE.request_frames, **run.held(verdict))


def test_an_outcome_on_another_advice_in_the_clear_or_late_is_refused(tmp_path):
    """Each window's outcome of record must be decided on the proof and name the advice tied: window 1 decided on
    another advice of the same lanes (another release order), decided in the clear, or delivered past the proof
    deadline refuses window 1 alone."""
    advice, other = _advice(SPAN), _advice(SPAN, release=(5, 6, 7))
    assert AD.digest(other) != AD.digest(advice) and TS.prompt_lens(other) == TS.prompt_lens(advice)

    run = Run(tmp_path / "other")
    assert run.round(0, advice).accepted and run.round(1, other).accepted
    assert run.verdict(advice).accepted == (True, False)
    assert run.verdict(other).accepted == (False, True)

    run = Run(tmp_path / "clear")
    assert run.round(0, advice).accepted
    draw, delivered = run.prove(1, advice)
    assert run.stream.deliver(draw, delivered.sessions()) is None
    verdict = run.tie.verify(draw, delivered, circuit_sha512=standin.circuit_sha512(delivered), advice=advice,
                             releases=SERVED.releases, head=run.cells[1].head)
    assert verdict["accepted"]
    outcome = run.stream.outcome(draw, {TS.VERIFIER: {**verdict, "decided_by": "clear"}}, run.tie.statement(draw),
                                 session_roots={TS.VALUE: draw.registration.record["roots"][0]["root"]})
    assert outcome.accepted
    assert run.verdict(advice).accepted == (True, False)

    run = Run(tmp_path / "late")
    assert run.round(0, advice).accepted
    draw, delivered = run.prove(1, advice)
    run.now[0] = 61.0
    late = run.stream.deliver(draw, delivered.sessions())
    assert late is not None and not late.accepted
    assert run.verdict(advice).accepted == (True, False)


def test_a_short_stream_one_out_of_order_or_one_past_the_advice_refuses_every_window(tmp_path):
    """The verdict is one bit a registration of the stream, each refused, and `advised.ends` refuses it."""
    advice = _advice(SPAN)
    run = Run(tmp_path / "short")
    assert run.round(0, advice).accepted
    assert run.verdict(advice).accepted == (False,)
    with pytest.raises(ValueError, match="doesn't accept each"):
        AD.ends(advice, PROFILE.request_frames, **run.held(run.verdict(advice)))

    run = Run(tmp_path / "order")
    reg = run.stream.register(run.tie.registration(run.cells[1].head), run.tie.ports(1))
    assert reg.record["window"] == run.tie.provenance(1)
    with pytest.raises(S.Refused):
        run.tie.register(run.stream, run.tie.registration(run.cells[0].head, reg.digest))
    assert run.verdict(advice).accepted == (False,)

    run = Run(tmp_path / "past")
    assert run.link(advice) == [True, True]
    assert not run.round(2, advice).accepted
    assert run.verdict(advice).accepted == (False, False, False)


@pytest.mark.parametrize("profile", [AD.AdvisedProfile(2, tuple(n + 1 for n in FRAMING), 0),
                                     AD.AdvisedProfile(2, FRAMING[:39], 0), AD.AdvisedProfile(2, FRAMING, 1)],
                         ids=["framing", "short-framing", "own"])
def test_a_profile_that_frames_requests_otherwise_than_the_warden_refuses_every_window(tmp_path, profile):
    """`advised.ends` reads the profile's framing, so a verdict under one that isn't the warden's cut at B for the
    advice's prompt lengths, or that names an own frame, which the warden's ingress has none of, would hold another
    link's ends: every window is refused."""
    run, advice = Run(tmp_path), _advice(SPAN)
    assert run.link(advice) == [True, True]
    assert run.verdict(advice, profile).accepted == (False, False)


def test_an_advice_whose_received_isnt_the_release_refuses_every_window(tmp_path):
    """Each window's proof is on its buckets' lanes alone, which `received` doesn't enter, but `advised.ends` reads it
    as each window's request cells: an advice that moves one of window 1's cells to window 0, keeping the total, is
    accepted window by window and refused by the tie."""
    run, moved = Run(tmp_path), _advice(SPAN, (3, 1))
    assert run.link(moved) == [True, True]
    assert run.verdict(moved).accepted == (False, False)


def test_a_row_that_isnt_the_wardens_is_refused(tmp_path):
    """Each unit's witness row is read off the warden's root at its own bucket (`warden_rows.bucket_port`): a proof on
    another row (window 1's bucket 1, which IoTieRequest accepts on its own lanes) or on the window's rows read by
    swapped units fails the registered read, and so do its sessions."""
    advice = _advice(SPAN)
    run = Run(tmp_path / "row")
    lanes = run.tie.lanes(advice, SERVED.releases, 0)
    other = run.cells[1].rows[1]
    assert run.tie.outputs(other, run.tie.lanes(advice, SERVED.releases, 1)[1]) == 1
    rows = list(run.cells[0].rows)
    rows[1] = other
    assert not run.round(0, advice, lanes[:1] + run.tie.lanes(advice, SERVED.releases, 1)[1:2] + lanes[2:],
                         rows=rows).accepted

    run = Run(tmp_path / "swapped")
    draw, delivered = run.prove(0, advice, units=[1, 0, 2, 3])
    assert run.stream.deliver(draw, delivered.sessions()) is None
    assert not run.decide(0, draw, delivered, advice).accepted


def test_a_head_of_another_window_is_refused(tmp_path):
    """The verifier reads the registered root's head: a head of window 1 on window 0's draw, or a head of another
    session's domain, refuses it."""
    run, advice = Run(tmp_path), _advice(SPAN)
    draw, delivered = run.prove(0, advice)
    assert run.stream.deliver(draw, delivered.sessions()) is None
    assert not run.decide(0, draw, delivered, advice, head=run.cells[1].head).accepted
    other = WR.Cells.commit(LINK, Direction.INGRESS, 0, run.cells[0].rows, run.cells[0].wiring, PARAMS,
                            hashlib.sha512(b"another session").digest(), run.cells[0].salts)
    assert not run.tie.verify(draw, delivered, circuit_sha512=standin.circuit_sha512(delivered), advice=advice,
                              releases=SERVED.releases, head=other.head)["accepted"]


def _accepts_any_cells(B, S_, *args):
    return B.call(BL.constant(1))


def test_the_tie_gate_names_its_pinned_circuit_digest():
    """The tie's gate names IoTieRequest's circuit digest as `io_request.circuit-check.json` pins it (circuit-check
    holds the pin to the Definition), and an IoTieRequest with the same id and signature that accepts any cells expands
    no gate of it."""
    d, gate = bind(Q.IoTieRequest, B=B, L=2, R=2), TS.gate(B, 2, 2)
    pinned = json.loads(Q.PINS.read_text())["digests"][d.id]["sha512"]
    assert gate.id == f"Gate[{d.id}@{pinned}]_v2" and HC.pinned(gate)
    forged = bind(replace(Q.IoTieRequest, body=_accepts_any_cells, register=False), B=B, L=2, R=2)
    assert forged.id == d.id and forged.signature() == d.signature()
    with pytest.raises(ValueError, match="another circuit"):
        HC.expand(gate, forged)


def test_an_unpinned_size_class_is_refused(tmp_path):
    """IoTieRequest at L = 3 has no pinned circuit digest, so its gate names none and no proof of its unit is accepted,
    even one Lean accepts."""
    params = W.WardenParams(cell_bytes=B, slots=2, lanes=3, window=4, delay=1)
    served = _serve(params=params, windows=2)
    run, advice = Run(tmp_path, params=params, served=served), _advice(SPAN)
    assert not HC.pinned(run.tie.gates()[0])
    draw, delivered = run.prove(0, advice)
    assert delivered.outputs == {u: 1 for u in range(4)}
    assert run.stream.deliver(draw, delivered.sessions()) is None
    assert not run.decide(0, draw, delivered, advice).accepted


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
