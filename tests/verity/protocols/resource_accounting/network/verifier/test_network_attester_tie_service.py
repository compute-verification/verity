"""The ingress tie through the proof system (`tie_service`): each bucket row of an ingress execution phase of the
network attester's
cells proved as IoTieRequest (a stand-in proof, `standin.prove_tie`, its rows read off the network attester's root and decided by
the verifier on its own lanes), and `tie_service.IoTieVerdict` built from the stream's outcomes of record
alone. An honest link is tied, and `Ties`, the tie, its stream and the release, gives the verdict only on its own link
and stream; the verifier's registered reads name every port of the statement, as Flock's Lean verifier requires
(`Registered.checkComplete`, which the stand-in mirrors); a carried field count, forged inputs and the swapped split
are refused, as are an output 0 or rows under fresh salts, an outcome on another
inputs, in the clear or late, a stream short of the inputs' windows, out of order or past them, a profile that frames
requests otherwise than the network attester cuts them, a row that isn't the network attester's, a head of another window, inputs whose
`received` isn't the release's, and an unpinned size class. Nothing in `verity` but the tie service builds the
verdict."""

import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import verity
from verity.circuits import boolean as BL
from verity.circuits.defs import bind
from verity.protocols.resource_accounting.network.verifier import head_check as HC, io_request as Q, service as SVC, \
    tie_service as TS, network_attester as W, network_attester_rows as WR
from verity.protocols.resource_accounting.network.verifier.id_form import id_request
from verity.protocols.resource_accounting.network.verifier.certificate import Direction
from verity.protocols.proofs.spot.verifier import service as S
from verity.protocols.proofs.spot.verifier.one_stage import docket as R
from verity.commitments import registered_value as RV

import standin

B = 512
#: two lanes of r = 2 slots, 4-bucket windows
PARAMS = W.NetworkAttesterParams(cell_bytes=B, slots=2, lanes=2, window=4, delay=1)
LINK, RUN = "requests", "r-tie-test"
SESSION = hashlib.sha512(b"tie-test").digest()
#: a span: request 1 (40 tokens) takes 2 cells, its DATA in bucket 3 and its END in bucket 4, so it
#: spans the two windows; requests 0 and 2 take one cell each, one in each window
SPAN, ARRIVALS, RATES = (5, 40, 5), (0, 2, 4), (10 ** 6, 600, 10 ** 6)
RECEIVED = (2, 2)
FRAMING = TS.framing(B, 64)
PROFILE = TS.AdvisedProfile(2, FRAMING, 0)
#: the prover proves on the verifier's own lanes of the inputs
OWN_LANES = object()


@pytest.fixture(autouse=True)
def lean(monkeypatch):
    monkeypatch.setattr(SVC.PF, "verify", standin.lean)


def _requests(prompt_lens):
    return [id_request(model="glm", user=f"u{i}", prompt=[(7919 * i + 104729 * k) % 151329 for k in range(n)],
                       max_tokens=16, temperature=0.7, top_p=0.95, seed=i, frame_bytes=B, offset=0)
            for i, n in enumerate(prompt_lens)]


def _inputs(lengths, received=RECEIVED, places=None):
    """The requests in order (or in `places`) with their prompt lengths, and the link's `received`."""
    places = tuple(range(len(lengths))) if places is None else tuple(places)
    return TS.IngressInputs(places, tuple(lengths), tuple(received))


def _serve(prompt_lens=SPAN, params=PARAMS, windows=3):
    """The link served through the network attester's own ingress, a third window past the inputs' empty."""
    return Q.serve(params, _requests(prompt_lens), ARRIVALS, RATES, buckets=windows * params.window)


SERVED = _serve()


class Run:
    """An ingress link's tie: the network attester's cells of each window (`network_attester_rows.Cells`, under fixed salts), the
    challenger's stream on injected coins and clock, and each window decided by the verifier on a stand-in
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

    def sample(self, w):
        reg = self.tie.register(self.stream, self.tie.registration(self.cells[w].head, self.previous))
        self.previous = reg.digest
        return self.stream.select(reg)

    def prove(self, w, inputs, lanes=OWN_LANES, releases=None, **kw):
        sample = self.sample(w)
        releases = self.served.releases if releases is None else releases
        lanes = self.tie.lanes(inputs, releases, w) if lanes is OWN_LANES else lanes
        return sample, standin.prove_tie(self.tie, self.cells[w], sample, lanes,
                                       self.tmp / f"tie-{sample.docket.order}", **kw)

    def decide(self, w, sample, delivered, inputs, head=None):
        return self.tie.outcome(self.stream, sample, delivered, circuit_sha512=standin.circuit_sha512(delivered),
                                inputs=inputs, releases=self.served.releases,
                                head=self.cells[w].head if head is None else head, opening=self.cells[w])

    def round(self, w, inputs, lanes=OWN_LANES, **kw) -> S.Outcome:
        sample, delivered = self.prove(w, inputs, lanes, **kw)
        late = self.stream.deliver(sample, delivered.sessions())
        if late is not None:
            return late
        outcome = self.decide(w, sample, delivered, inputs)
        verdict = self.stream.records[-1]["audit"]["verdicts"][TS.VERIFIER]
        assert verdict["decided_by"] == "proof" and verdict["accepted"] == outcome.accepted
        if lanes is OWN_LANES and not kw:
            assert verdict["clear"]["accepted"] == outcome.accepted
        return outcome

    def link(self, inputs, windows=2) -> list[bool]:
        return [self.round(w, inputs).accepted for w in range(windows)]

    def verdict(self, inputs, profile=PROFILE):
        return self.tie.tie(self.stream, inputs, profile, self.served.releases)


def test_an_honest_link_is_tied(tmp_path):
    run, inputs = Run(tmp_path), _inputs(SPAN)
    assert run.tie.layout == {"pub": ("lanes",), "wit": ("row",)} and run.tie.T == 4
    assert [r.cells for r in SERVED.releases] == [(0, 1), (0, 1, 1), (0, 1)]
    assert run.link(inputs) == [True, True]
    verdict = run.verdict(inputs)
    regs = tuple(R.digest(r) for r in run.stream.log)
    assert verdict == TS.IoTieVerdict(LINK, regs, TS.digest(inputs), (True, True))
    assert run.stream.records[-1]["public"] == {"verdict": 1, "inputs": TS.digest(inputs)}
    assert run.stream.log[0]["roots"][0]["root"] == run.cells[0].root.hex()


def test_the_registered_reads_name_every_port_so_an_honest_window_is_proved(tmp_path):
    """Flock's Lean verifier rejects every session of a statement with a port its `--registered` file doesn't name
    (`Registered.checkComplete`), which the stand-in mirrors: the verifier's own file names ``wit`` at the network attester's
    root and ``pub`` and ``o0`` at known content, its own lanes of each bucket and the output word 1 under public
    salts, so an honest window's proof is proved and accepted, where a file naming ``wit`` alone rejects it at
    ``pub``."""
    run, inputs = Run(tmp_path), _inputs(SPAN)
    sample, delivered = run.prove(0, inputs)
    head, lanes = run.cells[0].head, run.tie.lanes(inputs, SERVED.releases, 0)
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
    assert run.stream.deliver(sample, delivered.sessions()) is None
    assert run.decide(0, sample, delivered, inputs).accepted


def test_an_output_0_or_rows_under_fresh_salts_fail_the_registered_reads(tmp_path, monkeypatch):
    """The ``o0`` row is read off the verifier's known root of output words 1, and the ``pub`` row off its known root
    of its own lanes under their public salts: the forged inputs' proof of window 0, on the network attester's rows and the
    verifier's own lanes, fails ``o0``'s read at the bucket IoTieRequest refuses, whatever word it delivers, and a
    proof whose ``pub`` and ``o0`` rows are committed under fresh salts fails ``pub``'s."""
    forged, inputs = _inputs((5, 45, 5)), _inputs(SPAN)
    run = Run(tmp_path / "zero")
    lanes = run.tie.lanes(forged, SERVED.releases, 0)
    assert 0 in [run.tie.outputs(r, ln) for r, ln in zip(run.cells[0].rows, lanes)]
    _, delivered = run.prove(0, forged, words={u: 1 for u in range(4)})
    kw = {"circuit_sha512": standin.circuit_sha512(delivered), "program": run.tie.program,
          "partition": run.tie.partition}
    findings, refused = standin.lean(delivered.proof, registered=run.tie.registered(run.cells[0].head, lanes), **kw)
    assert refused is None and not findings[0].accepted and "port o0: table row" in findings[0].why

    run = Run(tmp_path / "fresh")
    lanes, known_tree = run.tie.lanes(inputs, SERVED.releases, 0), TS.IngressTie.known_tree
    monkeypatch.setattr(TS.IngressTie, "known_tree", lambda self, head, port, rows: RV.Registered.commit(
        (t := known_tree(self, head, port, rows)).program, t.value, t.rows, None, t.word_bits, t.schema,
        window=t.window))
    sample, delivered = run.prove(0, inputs)
    monkeypatch.setattr(TS.IngressTie, "known_tree", known_tree)
    findings, refused = standin.lean(delivered.proof, registered=run.tie.registered(run.cells[0].head, lanes), **kw)
    assert refused is None and not findings[0].accepted and "port pub: table row" in findings[0].why
    assert run.stream.deliver(sample, delivered.sessions()) is None
    assert not run.decide(0, sample, delivered, inputs).accepted


def test_the_tie_source_gives_the_verdict_only_on_its_own_link_and_stream(tmp_path):
    """`Ties`, the tie, its stream and the release: the verdict it gives is the tie's on its stream, and it refuses a
    link of another name and a stream of another link's tie."""
    inputs = _inputs(SPAN)
    run = Run(tmp_path / "honest")
    assert run.link(inputs) == [True, True]
    ties = TS.Ties(run.tie, run.stream, SERVED.releases)
    assert ties.registrations(LINK) == tuple(R.digest(r) for r in run.stream.log)
    assert ties.verdict(LINK, inputs, PROFILE) == run.verdict(inputs) and all(run.verdict(inputs).accepted)
    with pytest.raises(ValueError, match="not 'another'"):
        ties.verdict("another", inputs, PROFILE)
    elsewhere = Run(tmp_path / "elsewhere", link="another")
    assert elsewhere.link(inputs) == [True, True]
    with pytest.raises(ValueError, match="its own stream"):
        TS.Ties(run.tie, elsewhere.stream, SERVED.releases)


@pytest.mark.parametrize("delta", [1, 5, 100])
def test_a_proof_on_an_added_carried_field_count_is_refused(tmp_path, delta):
    """The carried field count through the verifier: IoTieRequest accepts bucket 4's row on its lane's carry-in
    fields + Δ with its carry-out's + Δ, but the verifier's lanes are its own, from the inputs and the release, so a
    proof on the added lanes opens to no public row of the verifier's and window 1 is refused."""
    run, inputs = Run(tmp_path), _inputs(SPAN)
    lanes = run.tie.lanes(inputs, SERVED.releases, 1)
    ln = lanes[0][0]
    assert (ln.q, ln.left, ln.fields, ln.left_out, ln.fields_out) == (2, 310, 12, 0, 40)
    added = ((ln._replace(fields=ln.fields + delta, fields_out=ln.fields_out + delta),) + lanes[0][1:],) + lanes[1:]
    assert run.round(0, inputs).accepted
    sample, delivered = run.prove(1, inputs, added)
    assert delivered.outputs == {u: 1 for u in range(4)}, "IoTieRequest accepts each row on the added lanes"
    assert run.stream.deliver(sample, delivered.sessions()) is None
    assert not run.decide(1, sample, delivered, inputs).accepted
    assert run.stream.records[-1]["audit"]["verdicts"][TS.VERIFIER]["clear"]["accepted"]
    assert run.verdict(inputs).accepted == (True, False)


def test_forged_inputs_are_refused_from_window_0(tmp_path):
    """Forged inputs, 40 → 45 prompt fields, keeps every request's cells, so the release can't see it: bucket 3's
    carry-out isn't the forged request's, nor bucket 4's carry-in, so both windows are refused."""
    forged = _inputs((5, 45, 5))
    assert [TS.request_cells(n, B) for n in (5, 45, 5)] == [TS.request_cells(n, B) for n in SPAN]
    run = Run(tmp_path)
    assert run.link(forged) == [False, False]
    assert run.verdict(forged).accepted == (False, False)


def test_redteam_phase6_split_is_refused(tmp_path):
    """The swapped split at the tie's size class: prompts of 40 and 5 tokens take 2 cells and 1, and the forged
    inputs swap them, keeping the total, which lowers request 0's end from 2 to 1. The release puts 2 cells on
    connection 1, which the swapped inputs' 5-token request doesn't take, so the verifier's lanes refuse every
    window."""
    pl = (40, 5)
    served = Q.serve(PARAMS, _requests(pl), (0, 0), (600, 10 ** 6), buckets=2 * PARAMS.window)
    received = tuple(TS.window_cells(served.releases, PARAMS.window).get(w, 0) for w in range(2))
    assert [r.cells for r in served.releases] == [(0, 1, 1), (0, 1)] and received == (3, 0)
    honest, split = _inputs(pl, received), _inputs(pl[::-1], received)
    run = Run(tmp_path / "honest", served=served)
    assert run.link(honest) == [True, True] and run.verdict(honest).accepted == (True, True)
    run = Run(tmp_path / "split", served=served)
    assert run.tie.lanes(split, served.releases, 0) is None
    assert run.link(split) == [False, False]
    assert run.verdict(split).accepted == (False, False)


def test_an_outcome_on_other_inputs_in_the_clear_or_late_is_refused(tmp_path):
    """Each window's outcome of record must be decided on the proof and name the inputs tied: window 1 decided on
    other inputs of the same lanes (two requests of one length in each other's places), decided in the clear, or delivered past the proof
    deadline refuses window 1 alone."""
    inputs, other = _inputs(SPAN), _inputs(SPAN, places=(2, 1, 0))
    tie = TS.IngressTie(LINK, PARAMS, SESSION)
    assert TS.digest(other) != TS.digest(inputs) and \
        all(tie.lanes(other, SERVED.releases, w) == tie.lanes(inputs, SERVED.releases, w) for w in range(2))

    run = Run(tmp_path / "other")
    assert run.round(0, inputs).accepted and run.round(1, other).accepted
    assert run.verdict(inputs).accepted == (True, False)
    assert run.verdict(other).accepted == (False, True)

    run = Run(tmp_path / "clear")
    assert run.round(0, inputs).accepted
    sample, delivered = run.prove(1, inputs)
    assert run.stream.deliver(sample, delivered.sessions()) is None
    verdict = run.tie.verify(sample, delivered, circuit_sha512=standin.circuit_sha512(delivered), inputs=inputs,
                             releases=SERVED.releases, head=run.cells[1].head)
    assert verdict["accepted"]
    outcome = run.stream.outcome(sample, {TS.VERIFIER: {**verdict, "decided_by": "clear"}}, run.tie.statement(sample),
                                 session_roots={TS.VALUE: sample.docket.record["roots"][0]["root"]})
    assert outcome.accepted
    assert run.verdict(inputs).accepted == (True, False)

    run = Run(tmp_path / "late")
    assert run.round(0, inputs).accepted
    sample, delivered = run.prove(1, inputs)
    run.now[0] = 61.0
    late = run.stream.deliver(sample, delivered.sessions())
    assert late is not None and not late.accepted
    assert run.verdict(inputs).accepted == (True, False)


def test_a_short_stream_one_out_of_order_or_one_past_the_inputs_refuses_every_window(tmp_path):
    """The verdict is one bit a registration of the stream, each refused."""
    inputs = _inputs(SPAN)
    run = Run(tmp_path / "short")
    assert run.round(0, inputs).accepted
    assert run.verdict(inputs).accepted == (False,)

    run = Run(tmp_path / "order")
    reg = run.stream.register(run.tie.registration(run.cells[1].head), run.tie.ports(1))
    assert reg.record["window"] == run.tie.provenance(1)
    with pytest.raises(S.Refused):
        run.tie.register(run.stream, run.tie.registration(run.cells[0].head, reg.digest))
    assert run.verdict(inputs).accepted == (False,)

    run = Run(tmp_path / "past")
    assert run.link(inputs) == [True, True]
    assert not run.round(2, inputs).accepted
    assert run.verdict(inputs).accepted == (False, False, False)


@pytest.mark.parametrize("profile", [TS.AdvisedProfile(2, tuple(n + 1 for n in FRAMING), 0),
                                     TS.AdvisedProfile(2, FRAMING[:39], 0), TS.AdvisedProfile(2, FRAMING, 1)],
                         ids=["framing", "short-framing", "own"])
def test_a_profile_that_frames_requests_otherwise_than_the_network_attester_refuses_every_window(tmp_path, profile):
    """The tie reads the profile's framing: under one that isn't the network attester's cut at B for the inputs' prompt
    lengths, or that names an own frame, which the network attester's ingress has none of, every window is refused."""
    run, inputs = Run(tmp_path), _inputs(SPAN)
    assert run.link(inputs) == [True, True]
    assert run.verdict(inputs, profile).accepted == (False, False)


def test_inputs_whose_received_isnt_the_release_refuse_every_window(tmp_path):
    """Each window's proof is on its buckets' lanes alone, which `received` doesn't enter, but the tie holds it to the
    release's request cells a window: inputs that move one of window 1's cells to window 0, keeping the total, is
    accepted window by window and refused by the tie."""
    run, moved = Run(tmp_path), _inputs(SPAN, (3, 1))
    assert run.link(moved) == [True, True]
    assert run.verdict(moved).accepted == (False, False)


def test_a_row_that_isnt_the_network_attesters_is_refused(tmp_path):
    """Each proof block's witness row is read off the network attester's root at its own bucket
    (`network_attester_rows.bucket_port`): a proof on another row (window 1's bucket 1, which IoTieRequest accepts on its own
    lanes) or on the window's rows read by swapped proof blocks fails the registered read, and so do its sessions."""
    inputs = _inputs(SPAN)
    run = Run(tmp_path / "row")
    lanes = run.tie.lanes(inputs, SERVED.releases, 0)
    other = run.cells[1].rows[1]
    assert run.tie.outputs(other, run.tie.lanes(inputs, SERVED.releases, 1)[1]) == 1
    rows = list(run.cells[0].rows)
    rows[1] = other
    assert not run.round(0, inputs, lanes[:1] + run.tie.lanes(inputs, SERVED.releases, 1)[1:2] + lanes[2:],
                         rows=rows).accepted

    run = Run(tmp_path / "swapped")
    sample, delivered = run.prove(0, inputs, units=[1, 0, 2, 3])
    assert run.stream.deliver(sample, delivered.sessions()) is None
    assert not run.decide(0, sample, delivered, inputs).accepted


def test_a_head_of_another_window_is_refused(tmp_path):
    """The verifier reads the registered root's head: a head of window 1 on window 0's sample, or a head of another
    session's domain, refuses it."""
    run, inputs = Run(tmp_path), _inputs(SPAN)
    sample, delivered = run.prove(0, inputs)
    assert run.stream.deliver(sample, delivered.sessions()) is None
    assert not run.decide(0, sample, delivered, inputs, head=run.cells[1].head).accepted
    other = WR.Cells.commit(LINK, Direction.INGRESS, 0, run.cells[0].rows, run.cells[0].wiring, PARAMS,
                            hashlib.sha512(b"another session").digest(), run.cells[0].salts)
    assert not run.tie.verify(sample, delivered, circuit_sha512=standin.circuit_sha512(delivered), inputs=inputs,
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
    """IoTieRequest at L = 3 has no pinned circuit digest, so its gate names none and no proof of its proof block is
    accepted, even one Lean accepts."""
    params = W.NetworkAttesterParams(cell_bytes=B, slots=2, lanes=3, window=4, delay=1)
    served = _serve(params=params, windows=2)
    run, inputs = Run(tmp_path, params=params, served=served), _inputs(SPAN)
    assert not HC.pinned(run.tie.gates()[0])
    sample, delivered = run.prove(0, inputs)
    assert delivered.outputs == {u: 1 for u in range(4)}
    assert run.stream.deliver(sample, delivered.sessions()) is None
    assert not run.decide(0, sample, delivered, inputs).accepted


def test_inputs_refuse_what_isnt_a_place_and_a_length_an_input():
    with pytest.raises(ValueError, match="permutation"):
        _inputs(SPAN, places=(0, 0, 1))
    with pytest.raises(ValueError, match="positive u32"):
        _inputs((5, 0, 5))
    with pytest.raises(ValueError, match="u32s"):
        _inputs(SPAN, received=(2, -1))


def test_the_inputs_are_the_registered_compilers_outputs():
    """The tie's inputs come from the registered compiler's circuit's ``lengths`` output, its arrival slots in certified
    order with an empty slot 0, not from the advice's bytes."""
    assert TS.IngressInputs.compiled({"lengths": list(SPAN)}, RECEIVED) == _inputs(SPAN)
    assert TS.IngressInputs.compiled({"lengths": [SPAN[0], 0, 0, *SPAN[1:], 0]}, RECEIVED) == _inputs(SPAN)


def test_only_the_tie_service_builds_an_ingress_tie_verdict():
    """Nothing in `verity` but the tie service, which builds `IoTieVerdict` from the stream's outcomes of record,
    constructs one (the tests' stand-ins aside)."""
    root = Path(verity.__file__).resolve().parent
    builders = set()
    for path in root.rglob("*.py"):
        if "tests" in path.relative_to(root).parts or "IoTieVerdict" not in path.read_text():
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            f = node.func if isinstance(node, ast.Call) else None
            if (isinstance(f, ast.Name) and f.id == "IoTieVerdict") or \
                    (isinstance(f, ast.Attribute) and f.attr == "IoTieVerdict"):
                builders.add(path.relative_to(root).as_posix())
    assert builders == {Path(TS.__file__).resolve().relative_to(root).as_posix()}
