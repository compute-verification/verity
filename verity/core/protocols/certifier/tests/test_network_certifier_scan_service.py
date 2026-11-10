"""The egress tie through the proof system (`scan_service`): an execution phase's step-indexed scan as one registration of
the proof system, its 2T proof blocks in four statements (the first block, T − 2 ingress blocks, the last ingress block
and T egress blocks) every one sampled, each member a stand-in proof (`standin.prove_scan`: its outputs by the
templates' references, every port a registered read the verifier names, as Flock's Lean verifier requires) decided
by the verifier on its own registered reads, and `EgressTieVerdict` built from the stream's outcomes alone. Three id
requests on two lanes, the third queued until a lane frees, are served through the network certifier, and the committed
trace is the scan's state and the slot entries of the compute's outputs. An honest phase is tied. The network
certifier's egress with one token other than the compute's is refused at the egress member, and so is a trace whose one
output differs from what the network certifier saw; a view that renders the network certifier's changed egress is
refused at the ingress block that computes the view; a trace whose state doesn't follow from the bucket before is
refused at the block whose S_t the next state leaf isn't; a registration whose network certifier or trace roots aren't
the phase's is refused, as is a member's proof naming another Program, a missing member and a sample of other proof
blocks. The network verdict on a network certifier link (`LinkWindows.network_verdict(egress=…)`) counts an execution
phase only where its egress tie accepted it: it accepts the honest run, and refuses the run whose egress differs from
the committed trace by one token, whose N2 counts accept, and refuses a link with no egress tie by name. Nothing in
`verity` but the scan service builds the verdict.

A phase's registration, sample and outcome are decided by the stream's own Lean verifier on the phase's generator-form
population program (`service.Stream(program=…)`), about half a minute a phase, which no cheaper test runs on this Spec:
the honest and one-token-off phases go through the stream once each (module fixtures), and every other refusal is the
verifier's on a sample of its own or the honest one."""

import ast
import hashlib
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
import scan_traffic as ST
import signers
import standin

from verity.core.protocols.certifier import scan_service as SC, service as SVC, warden as W
from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.certifier.active import id_request
from verity.core.protocols.certifier.blocks import SCAN_STATE
from verity.core.protocols.certifier.certificate import Direction, Issuer
from verity.core.protocols.certifier.schedule import SyncSet
from verity.core.protocols.spot import service as S
from verity.core.protocols.spot.one_stage import registered as G, registration as R
from verity.ml.protocols.certifier import ingress_scan as IS

B = 512
PARAMS = W.WardenParams(cell_bytes=B, slots=2, bucket_ms=100, delay=5, lanes=2, queue=2, window=16, ingress_cells=4,
                        ingress_buckets=3, egress_cells=9, every=2)
CLS = IS.Class(B=B, r=2, lanes=2, C=4, EVERY=2, WITHIN=3, D=5, KO=8, KI=100)
LINK, SESSION = "responses", signers.SESSION
TIE = SC.EgressTie(LINK, PARAMS, SESSION, IS.verifier_scan(CLS, PARAMS), proof_deadline=60.0)
PROMPTS = ([1000, 1007, 1014], [2000 + 3 * k for k in range(60)], [131071 - k for k in range(100)])
RESPONSES = ((11, 22, 33), (131071, 0, 5, 6, 7, 8, 9, 10), (40, 41, 42, 43))
#: the server sends request 1's second token one off the compute's
ONE_OFF = (RESPONSES[0], (131071, 1, 5, 6, 7, 8, 9, 10), RESPONSES[2])
#: the compute audit's Program, whose registration names the trace's trees
COMPUTE = hashlib.sha512(b"the compute audit's program").hexdigest()
UNREAD_SYNCS = SyncSet((0,), (Fraction(1),))
EGRESS, INGRESS = Direction.EGRESS, Direction.INGRESS


@pytest.fixture(scope="module", autouse=True)
def lean():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(SVC.PF, "verify", standin.lean)
        yield


def _salts(tag: str, n: int) -> list[bytes]:
    return [hashlib.sha512(f"{tag} {i}".encode()).digest() * 3 for i in range(n)]


def _serve(responses=RESPONSES) -> ST.Served:
    reqs = [id_request(model="unsloth/Llama-3.2-1B", user=f"user-{i}", prompt=p, max_tokens=5 + i, seed=11 * i,
                       temperature=0.7, top_p=0.95, frame_bytes=B, offset=0) for i, p in enumerate(PROMPTS)]
    served = ST.serve(PARAMS, reqs, [0, 1, 1], [1 << 20] * 3, responses, 1)
    assert not served.fails
    return served


SERVED, SERVED_ONE_OFF = _serve(), _serve(ONE_OFF)


class Run:
    """An execution phase's egress tie: the network certifier's cells of both links (`warden_rows.Cells`, under fixed salts)
    of `served`, the committed trace of the compute's `outputs` (its `SCAN_STATE` and slot entries, the compute audit's
    trees), the prover's views, the challenger's stream on injected coins and clock, and the phase decided by the
    verifier on stand-in proofs."""

    def __init__(self, tmp_path: Path, served: ST.Served = SERVED, outputs=RESPONSES, states=None) -> None:
        self.tmp, self.now, self.previous, self.served, self.tie = tmp_path, [0.0], None, served, TIE
        self.stream = self.tie.stream(budget=8, delta=2.0 ** -40, challenger="challenger-test",
                                      coins=lambda n: b"\x07" * n, clock=lambda: self.now[0])
        T, nb = self.tie.T, self.tie.nb
        self.cells = SC.Heads(*(WR.Cells.of(LINK, d, rec, PARAMS, SESSION, _salts(f"cells {d.value}", T),
                                            hashlib.sha512(f"wiring {d.value}".encode()).digest())
                                for d, rec in ((INGRESS, served.ingress[0]), (EGRESS, served.egress[0]))))
        ingress, egress = served.rows()
        self.chain = ST.chain(CLS, ingress, egress, {q: xs for q, xs in enumerate(outputs, 1)})
        states = self.chain.states[:T] if states is None else states
        self.state = G.Registered.commit(COMPUTE, SCAN_STATE, [SC.words(s) for s in states],
                                         _salts("state", T), G.WORD_BITS, SC.SCHEMA)
        self.progs = [G.Registered.commit(COMPUTE, SC.value(f"prog{j}"),
                                          [SC.words(IS.prog_rows(CLS, self.chain.progs[b * CLS.EVERY])[j])
                                           for b in range(nb)], _salts(f"prog{j}", nb), G.WORD_BITS, SC.SCHEMA)
                      for j in self.tie.lanes]
        self.trace = SC.Trace(SC.Tree.of(self.state), tuple(SC.Tree.of(p) for p in self.progs))
        self.views = list(self.chain.views)

    @property
    def heads(self) -> SC.Heads:
        return SC.Heads(self.cells.ingress.head, self.cells.egress.head)

    def computed(self) -> list[tuple[int, list[int], list[int]]]:
        """Each bucket's (ok, S_t, view) by `ingress_scan.reference` on the committed rows, as the blocks compute them."""
        bits, progs = standin._bits, self.progs
        return [IS.reference(CLS, self.cells.ingress.rows[t], bits(self.state.rows[t]) if t else None,
                             [x for p in progs for x in bits(p.rows[t // CLS.EVERY])]) for t in range(self.tie.T)]

    def trees(self) -> tuple[G.Registered, G.Registered]:
        """The prover's ``scan/view`` and ``scan/last`` trees, of `views` and S_(T−1)."""
        return (self.tie.view_tree(self.heads, self.views, _salts("view", self.tie.T)),
                self.tie.last_tree(self.heads, self.chain.states[self.tie.T], _salts("last", 1)))

    def draw(self) -> S.Draw:
        """The execution phase registered on the stream, the prover's trees under it, and sampled."""
        self.view, self.last = self.trees()
        record = self.tie.registration(self.heads, self.trace, self.view.root, self.last.root, self.previous)
        reg = self.tie.register(self.stream, record, self.heads, self.trace)
        self.previous = reg.digest
        self.drawn = self.stream.select(reg)
        return self.drawn

    def prove(self, name: str, members=None, **kw) -> SC.Delivered:
        return SC.Delivered({m.name: standin.prove_scan(self.tie, CLS, self.cells, self.state, self.progs, self.view,
                                                        self.last, m, self.tmp / name / m.name, **kw)
                             for m in members or self.tie.members})

    def verify(self, delivered: SC.Delivered, draw: S.Draw | None = None, trace: SC.Trace | None = None) -> dict:
        """The verifier's verdict on `delivered` against the sampled phase (by default this run's sample)."""
        return self.tie.verify(draw or self.drawn, delivered, circuits=standin.scan_circuits(self.tie),
                               heads=self.heads, trace=trace or self.trace)

    def round(self) -> tuple[S.Outcome, dict]:
        """The execution phase registered, sampled, proved by every member, delivered and decided on the stream."""
        draw = self.draw()
        self.delivered = self.prove("round")
        assert self.stream.deliver(draw, self.delivered.sessions(self.tie.members)) is None
        outcome = self.tie.outcome(self.stream, draw, self.delivered, circuits=standin.scan_circuits(self.tie),
                                   heads=self.heads, trace=self.trace)
        verdict = self.stream.records[-1]["audit"]["verdicts"][SC.VERIFIER]
        assert verdict["decided_by"] == "proof" and verdict["accepted"] == outcome.accepted
        return outcome, verdict


@pytest.fixture(scope="module")
def honest(tmp_path_factory) -> tuple[Run, S.Outcome, dict]:
    run = Run(tmp_path_factory.mktemp("honest"))
    return (run, *run.round())


@pytest.fixture(scope="module")
def one_off(tmp_path_factory) -> tuple[Run, S.Outcome, dict]:
    """The server sent request 1's second token one off the compute's, which the network certifier certified."""
    run = Run(tmp_path_factory.mktemp("one-off"), served=SERVED_ONE_OFF)
    return (run, *run.round())


@pytest.mark.slow
def test_an_honest_execution_phase_is_tied_with_every_bucket_proved(honest):
    run, outcome, verdict = honest
    tie = run.tie
    assert (tie.T, tie.N, tie.nb) == (16, 32, 8)
    assert [(m.name, m.template, m.units[0], len(m.units)) for m in tie.members] == [
        ("first", "first", 0, 1), ("ingress", "ingress", 1, 14), ("last", "ingress", 15, 1),
        ("egress", "egress", 16, 16)]
    assert outcome.accepted and verdict["refused"] == {} and outcome.public == {"verdict": 1}
    assert run.stream.records[-1]["audit"]["draw"]["units"] == list(range(tie.N))
    regs = tuple(R.digest(r) for r in run.stream.log)
    assert tie.tie(run.stream) == SC.EgressTieVerdict(LINK, regs, (True,))
    roots = {r["port"]: r["root"] for r in run.stream.log[0]["roots"]}
    assert roots[SC.ROWS[EGRESS]] == run.cells.egress.root.hex() and roots[SC.STATE] == run.state.root.hex()
    assert run.stream.program is not None and set(roots) == set(tie.values)


@pytest.mark.slow
def test_the_registered_reads_name_every_port_and_chain_the_blocks(honest):
    """Each member's `--registered` file names every port of its template (`Registered.checkComplete`): an ingress
    block's ``row`` reads the network certifier's ingress row t, ``sin`` state leaf t, ``prog<j>`` lane j's boundary
    leaf, ``o1`` state leaf t + 1 (the last block's the ``scan/last`` leaf) and ``o2`` view leaf t; an egress block's
    ``state`` view leaf t and ``o1`` the network certifier's egress row t; every ``o0`` the known words 1."""
    run = honest[0]
    view, last = run.view, run.last
    m = {x.name: x for x in run.tie.members}
    reads = {k: run.tie.registered(m[k], run.heads, run.trace, view.root, last.root)["ports"] for k in m}
    assert set(reads["first"]) == {"row", "prog0", "prog1", "o0", "o1", "o2"}
    assert set(reads["ingress"]) == set(reads["last"]) == {"row", "sin", "prog0", "prog1", "o0", "o1", "o2"}
    assert set(reads["egress"]) == {"state", "o0", "o1"}
    ing = reads["ingress"]
    assert ing["row"]["root"] == run.cells.ingress.root.hex() and ing["row"]["reads"] == list(range(1, 15))
    assert ing["sin"]["reads"] == list(range(1, 15)) and ing["o1"]["reads"] == list(range(2, 16))
    assert ing["prog1"]["reads"] == [b // 2 for b in range(1, 15)] and ing["prog1"][G.READS_FROM] == 1
    assert reads["last"]["o1"]["root"] == last.root.hex() and reads["last"]["o1"]["reads"] == [0]
    eg = reads["egress"]
    assert eg["state"]["root"] == view.root.hex() and eg["o1"]["root"] == run.cells.egress.root.hex()
    assert eg["o0"][SC.KNOWN_KEY]["rows"] == [[1]] * run.tie.N and eg["o0"]["reads"] == list(range(16, 32))


@pytest.mark.slow
def test_one_token_on_egress_other_than_the_computes_is_refused(one_off):
    """The server sends one token other than the compute's: the network certifier's egress row of that bucket isn't
    the row the trace renders, so the egress member's ``o1`` opens no leaf of the network certifier's root, and every
    ingress member is proved."""
    run, outcome, verdict = one_off
    assert not outcome.accepted and set(verdict["refused"]) == {"egress"}
    assert "port o1: table row" in verdict["refused"]["egress"] and "does not open" in verdict["refused"]["egress"]
    assert run.tie.tie(run.stream).accepted == (False,)


@pytest.mark.slow
def test_a_trace_whose_one_output_isnt_the_network_certifiers_is_refused(tmp_path):
    """The compute's committed trace holds one output other than what the network certifier carried: the scan is
    consistent with the trace, and the egress it renders isn't the network certifier's."""
    run = Run(tmp_path, outputs=ONE_OFF)
    run.draw()
    verdict = run.verify(run.prove("trace"))
    assert not verdict["accepted"] and set(verdict["refused"]) == {"egress"}


@pytest.mark.slow
def test_a_view_that_renders_the_network_certifiers_changed_egress_is_refused_where_it_is_computed(tmp_path):
    """Against the one-off egress, the prover registers views of the one-off outputs, which render it, beside the
    compute's honest trace: the egress member opens the network certifier's rows, and the ingress block that computes
    the changed view from the trace fails its ``o2`` read."""
    run = Run(tmp_path, served=SERVED_ONE_OFF)
    off = ST.chain(CLS, *SERVED_ONE_OFF.rows(), {q: xs for q, xs in enumerate(ONE_OFF, 1)})
    changed = [t for t in range(run.tie.T) if off.views[t] != run.chain.views[t]]
    assert changed
    run.views = list(off.views)
    run.draw()
    verdict = run.verify(run.prove("view"))
    assert not verdict["accepted"] and "egress" not in verdict["refused"]
    at = {m.name for m in run.tie.members if m.template != "egress" and set(m.buckets) & set(changed)}
    assert at and set(verdict["refused"]) == at
    assert all("port o2: table row" in verdict["refused"][k] for k in at)


@pytest.mark.slow
def test_a_trace_whose_state_doesnt_follow_from_the_bucket_before_is_refused_where_it_is_computed(tmp_path):
    """The trace's state leaf k isn't the S_(k−1) that block k − 1 computes, and the prover's views are what the blocks
    compute on that trace, so every view read opens: the member holding block k − 1 is refused at its ``o1`` read of
    state leaf k, and the first block's member, whose S_0 leaf 1 holds, is proved."""
    T, chain = TIE.T, Run(tmp_path / "honest").chain
    k = next(t for t in range(2, T) if chain.states[t - 1] != chain.states[t])
    run = Run(tmp_path, states=[*chain.states[:k], chain.states[k - 1], *chain.states[k + 1:T]])
    got = run.computed()
    assert all(ok for ok, _, _ in got)
    run.views = [v for _, _, v in got]
    run.draw()
    verdict = run.verify(run.prove("state"))
    at = next(m for m in run.tie.members if m.template != "egress" and k - 1 in m.buckets)
    assert not verdict["accepted"] and "first" not in verdict["refused"]
    assert verdict["refused"][at.name] == (f"registered reads: port o1: table row {at.buckets.index(k - 1)} does not "
                                           f"open the registered root at position {k}")


@pytest.mark.slow
def test_a_registration_off_the_heads_or_the_trace_is_refused(honest):
    """The challenger refuses a registration whose trace roots aren't the trace's, or whose `window` provenance isn't
    its order in the stream; and the verifier refuses a sampled phase against another trace than the one registered."""
    run = honest[0]
    other = SC.Trace(replace(run.trace.state, root=bytes(64)), run.trace.progs)
    stream = run.tie.stream(budget=8, delta=2.0 ** -40, challenger="c", coins=lambda n: b"\x07" * n, clock=lambda: 0.0)
    record = run.tie.registration(run.heads, run.trace, run.view.root, run.last.root)
    with pytest.raises(S.Refused) as refused:
        run.tie.register(stream, record, run.heads, other)
    assert refused.value.verdict.codes == ("roots",)
    with pytest.raises(S.Refused) as refused:
        run.tie.register(stream, {**record, "window": run.tie.provenance(1)}, run.heads, run.trace)
    assert refused.value.verdict.codes == ("window",)
    verdict = run.verify(run.delivered, trace=other)
    assert not verdict["accepted"] and set(verdict["refused"]) == {"registration"}


@pytest.mark.slow
def test_a_missing_member_or_one_naming_another_program_is_refused(honest):
    run = honest[0]
    verdict = run.verify(SC.Delivered({k: p for k, p in run.delivered.proofs.items() if k != "egress"}))
    assert not verdict["accepted"] and set(verdict["refused"]) == {"egress"}
    verdict = run.verify(run.prove("program", program="0" * 128))
    assert not verdict["accepted"] and set(verdict["refused"]) == {m.name for m in run.tie.members}


class Link:
    """The network certifier link's N2 side on the same served run: its two records certified under the schedule's
    digest, committed and decided by N2's verifier on a stand-in proof."""

    def __init__(self, tmp_path: Path, served: ST.Served) -> None:
        tmp_path.mkdir(parents=True)
        signer = signers.signer(tmp_path)
        drawn = iter(range(0xc0, 0x100))
        issuer = Issuer(signer, SESSION, coins=lambda n: bytes([next(drawn)]) * n)
        self.lw = SVC.LinkWindows(LINK, PARAMS.grid, UNREAD_SYNCS, signer.public_key(), "r-scan-test", SESSION,
                                  ingress=PARAMS.grid, proof_deadline=60.0, warden=SVC.WardenProfile(1, PARAMS.digest))
        self.stream = self.lw.stream(budget=8, delta=2.0 ** -40, challenger="challenger-test",
                                     coins=lambda n: b"\x07" * n, clock=lambda: 0.0)
        certified = [issuer.certify(replace(rec, link=LINK), d, PARAMS.grid, PARAMS.digest)
                     for d, rec in ((EGRESS, served.egress[0]), (INGRESS, served.ingress[0]))]
        opening = self.lw.commit(0, *certified, tuple(bytes([k + 1]) * 192 for k in range(4)))
        draw = self.stream.select(self.lw.register(self.stream, self.lw.registration(opening, None)))
        self.outcome = standin.decide(self.lw, self.stream, draw, opening, (), tmp_path / "n2")


@pytest.mark.slow
def test_the_network_verdict_of_record_refuses_a_run_whose_egress_is_one_token_off_the_trace(tmp_path, honest,
                                                                                            one_off):
    """On a network certifier link the network verdict counts an execution phase only where its egress tie accepted
    it: the honest run is accepted; the run whose server sent one token other than the compute's has N2 counts that
    accept and is refused, by its phase; a link with no egress tie is refused by name; and an egress tie of another link, session
    or schedule is no source of this link's."""
    scan, link = honest[0], Link(tmp_path / "honest", SERVED)
    assert link.outcome.accepted
    tied = SC.Scans(scan.tie, scan.stream)
    verdict = link.lw.network_verdict(link.stream, egress=tied)
    assert verdict["accepted"] and verdict["refused"] == [] and verdict["egress"]["accepted"] == [True]
    assert verdict["egress"]["registrations"] == [R.digest(r) for r in scan.stream.log]
    none = link.lw.network_verdict(link.stream)
    assert not none["accepted"] and none["refused"] == [] and set(none["why"]) == {SVC.EGRESS_TIED}
    assert none["egress"] is None and none["digest"] != verdict["digest"]

    off_scan, off = one_off[0], Link(tmp_path / "one-off", SERVED_ONE_OFF)
    assert off.outcome.accepted and off.lw.network_verdict(off.stream)["refused"] == []
    verdict = off.lw.network_verdict(off.stream, egress=SC.Scans(off_scan.tie, off_scan.stream))
    assert not verdict["accepted"] and verdict["refused"] == [0] and verdict["egress"]["accepted"] == [False]

    for lw, match in ((replace(link.lw, link="requests"), "link"),
                      (replace(link.lw, certifier_session=bytes(32)), "session"),
                      (replace(link.lw, warden=SVC.WardenProfile(1, bytes(64))), "schedule"),
                      (replace(link.lw, params=replace(PARAMS.grid, bucket_ms=200),
                               ingress=replace(PARAMS.grid, bucket_ms=200)), "schedule")):
        with pytest.raises(ValueError, match=match):
            lw.network_verdict(link.stream, egress=tied)
    plain = replace(link.lw, warden=None, occupancy=((tuple(0 for _ in range(16)),), None))
    with pytest.raises(ValueError, match="network certifier link"):
        plain.network_verdict(link.stream, egress=tied)


def test_only_the_scan_service_builds_an_egress_tie_verdict():
    """Nothing in `verity` but the scan service, which builds `EgressTieVerdict` from the stream's outcomes, constructs
    one."""
    root = Path(SC.__file__).resolve().parents[3]
    builders = set()
    for path in root.rglob("*.py"):
        if "tests" in path.relative_to(root).parts or "EgressTieVerdict" not in path.read_text():
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            f = node.func if isinstance(node, ast.Call) else None
            if (isinstance(f, ast.Name) and f.id == "EgressTieVerdict") or \
                    (isinstance(f, ast.Attribute) and f.attr == "EgressTieVerdict"):
                builders.add(path.relative_to(root).as_posix())
    assert builders == {"core/protocols/certifier/scan_service.py"}
