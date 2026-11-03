"""The egress tie through the proof system (`scan_service`): a link-window's step-indexed scan as one registration of the
proof service, its first unit, T − 1 ingress units (one statement) and T egress units every one drawn, each member a
stand-in proof (`standin.prove_scan`: its outputs by the templates' references, every port a registered read the
verifier names, as C-Flock's Lean verifier requires) decided by the verifier of record on its own registered reads and
the ingress statement's commitment chain, and `EgressTieVerdict` built from the stream's outcomes of record alone. Three
id requests on two lanes, the third queued until a lane frees, are served through the network certifier, and the
committed trace is the scan's state and the slot entries of the compute's outputs. An honest window is tied. The
network certifier's egress with one token other than the compute's is refused at the egress member, and so is a trace
whose one output differs from what the network certifier saw; a view that renders the network certifier's changed
egress is refused at the ingress unit that computes the view; a trace whose state doesn't follow from the bucket before
is refused by the chain; a registration whose network certifier or trace roots aren't the window's is refused, as is a
member's proof naming another
Program, a missing member and a draw of other units. The network verdict of record on a network certifier link
(`LinkWindows.network_verdict(egress=…)`) counts a window only where its egress tie accepted it: it accepts the honest
run, and refuses the run whose egress differs from the committed trace by one token, whose N2 counts accept, and refuses
a link with no egress tie by name. Nothing in `verity` but the scan service builds the verdict.

A window's registration, draw and outcome are decided by the stream's own Lean verifier on the window's generator-form
population program (`service.Stream(program=…)`), about half a minute a window, which no cheaper test runs on this
Spec: the honest and one-token-off windows go through the stream once each (module fixtures), and every other refusal is
the verifier of record's on a draw of its own or the honest one."""

import ast
import hashlib
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
import scan_traffic as ST
import signers
import standin

from verity.core.protocols.certifier import ingress_scan as IS, scan_service as SC, service as SVC, warden as W
from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.certifier.active import id_request
from verity.core.protocols.certifier.certificate import Direction, Issuer
from verity.core.protocols.certifier.schedule import SyncSet
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import registered as G, registration as R

B = 512
PARAMS = W.WardenParams(cell_bytes=B, slots=2, bucket_ms=100, delay=5, lanes=2, queue=2, window=16, ingress_cells=4,
                        ingress_buckets=3, egress_cells=9, every=2)
CLS = IS.Class(B=B, r=2, lanes=2, C=4, EVERY=2, WITHIN=3, D=5, KO=8, KI=100)
LINK, SESSION = "responses", signers.SESSION
TIE = SC.EgressTie(LINK, PARAMS, SESSION, CLS, proof_deadline=60.0)
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
    """A link-window's egress tie: the network certifier's cells of both links (`warden_rows.Cells`, under fixed salts)
    of `served`, the committed trace of the compute's `outputs` (its `SCAN_STATE` and slot entries, the compute audit's
    trees), the prover's views, the challenger's stream on injected coins and clock, and the window decided by the
    verifier of record on stand-in proofs."""

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
        self.state = G.Registered.commit(COMPUTE, IS.SCAN_STATE, [SC.words(s) for s in states],
                                         _salts("state", T), G.WORD_BITS, SC.SCHEMA)
        self.progs = [G.Registered.commit(COMPUTE, SC.value(f"prog{j}"),
                                          [SC.words(IS.prog_rows(CLS, self.chain.progs[b * CLS.EVERY])[j])
                                           for b in range(nb)], _salts(f"prog{j}", nb), G.WORD_BITS, SC.SCHEMA)
                      for j in self.tie.lanes]
        self.trace = SC.Trace(SC.Tree.of(self.state), tuple(SC.Tree.of(p) for p in self.progs))
        self.views, self.nexts = list(self.chain.views), list(self.chain.states[2:T + 1])

    @property
    def heads(self) -> SC.Heads:
        return SC.Heads(self.cells.ingress.head, self.cells.egress.head)

    def computed(self) -> list[tuple[int, list[int], list[int]]]:
        """Each bucket's (ok, S_t, view) by `ingress_scan.reference` on the committed rows, as the units compute them."""
        bits, progs = standin._bits, self.progs
        return [IS.reference(CLS, self.cells.ingress.rows[t], bits(self.state.rows[t]) if t else None,
                             [x for p in progs for x in bits(p.rows[t // CLS.EVERY])]) for t in range(self.tie.T)]

    def trees(self) -> tuple[G.Registered, G.Registered]:
        """The prover's ``scan/view`` and ``scan/next`` trees, of `views` and `nexts` (S_1 … S_(T−1)), row t − 1 of
        ``scan/next`` under the trace's state leaf t + 1's salt and the last under one of its own."""
        salts = list(self.state.salts[2:]) + _salts("next", 1)
        return (self.tie.view_tree(self.heads, self.views, _salts("view", self.tie.T)),
                self.tie.next_tree(self.heads, self.nexts, salts))

    def draw(self) -> S.Draw:
        """The window registered on the stream, the prover's trees under it, and drawn."""
        self.view, self.next = self.trees()
        record = self.tie.registration(self.heads, self.trace, self.view.root, self.next.root, self.previous)
        reg = self.tie.register(self.stream, record, self.heads, self.trace)
        self.previous = reg.digest
        self.drawn = self.stream.select(reg)
        return self.drawn

    def prove(self, name: str, members=None, **kw) -> SC.Delivered:
        return SC.Delivered({m.name: standin.prove_scan(self.tie, self.cells, self.state, self.progs, self.view,
                                                        self.next, m, self.tmp / name / m.name, **kw)
                             for m in members or self.tie.members})

    def verify(self, delivered: SC.Delivered, draw: S.Draw | None = None, trace: SC.Trace | None = None) -> dict:
        """The verifier of record's verdict on `delivered` against the drawn window (by default this run's draw)."""
        return self.tie.verify(draw or self.drawn, delivered, circuits=standin.scan_circuits(self.tie),
                               heads=self.heads, trace=trace or self.trace)

    def round(self) -> tuple[S.Outcome, dict]:
        """The window registered, drawn, proved by every member, delivered and decided on the stream."""
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
def test_an_honest_window_is_tied_with_every_bucket_proved(honest):
    run, outcome, verdict = honest
    tie = run.tie
    assert (tie.T, tie.N, tie.nb) == (16, 32, 8)
    assert [(m.name, m.template, m.units[0], len(m.units)) for m in tie.members] == [
        ("first", "first", 0, 1), ("ingress", "ingress", 1, 15), ("egress", "egress", 16, 16)]
    assert outcome.accepted and verdict["refused"] == {} and outcome.public == {"verdict": 1}
    assert run.stream.records[-1]["audit"]["draw"]["units"] == list(range(tie.N))
    regs = tuple(R.digest(r) for r in run.stream.log)
    assert tie.tie(run.stream) == SC.EgressTieVerdict(LINK, regs, (True,))
    roots = {r["port"]: r["root"] for r in run.stream.log[0]["roots"]}
    assert roots[SC.ROWS[EGRESS]] == run.cells.egress.root.hex() and roots[SC.STATE] == run.state.root.hex()
    assert run.stream.population is not None and set(roots) == set(tie.values)


@pytest.mark.slow
def test_the_registered_reads_name_every_port_and_chain_the_units(honest):
    """Each member's `--registered` file names every port of its template (`Registered.checkComplete`): an ingress
    unit's ``row`` reads the network certifier's ingress row t, ``sin`` state leaf t, ``prog<j>`` lane j's boundary
    leaf, ``o1`` ``scan/next`` leaf t − 1 and ``o2`` view leaf t; the first unit's ``o1`` state leaf 1; an egress
    unit's ``state`` view leaf t and ``o1`` the network certifier's egress row t; every ``o0`` the known words 1. The
    ingress statement's S_t chain: each unit's ``o1`` commitment is the next unit's ``sin`` commitment."""
    run = honest[0]
    view, next_ = run.view, run.next
    m = {x.name: x for x in run.tie.members}
    reads = {k: run.tie.registered(m[k], run.heads, run.trace, view.root, next_.root)["ports"] for k in m}
    assert set(reads["first"]) == {"row", "prog0", "prog1", "o0", "o1", "o2"}
    assert set(reads["ingress"]) == {"row", "sin", "prog0", "prog1", "o0", "o1", "o2"}
    assert set(reads["egress"]) == {"state", "o0", "o1"}
    assert reads["first"]["o1"]["root"] == run.state.root.hex() and reads["first"]["o1"]["reads"] == [1]
    ing = reads["ingress"]
    assert ing["row"]["root"] == run.cells.ingress.root.hex() and ing["row"]["reads"] == list(range(1, 16))
    assert ing["sin"]["reads"] == list(range(1, 16)) and ing["prog1"]["reads"] == [b // 2 for b in range(1, 16)]
    assert ing["o1"]["root"] == next_.root.hex() and ing["o1"]["reads"] == list(range(15))
    assert ing["prog1"][G.READS_FROM] == 1 and ing["o1"][G.READS_FROM] == 1
    got = SC.commitments(run.delivered.proofs["ingress"].circuit, run.delivered.proofs["ingress"].public)
    assert got["o1"][:-1] == got["sin"][1:] and SC.chained(run.delivered.proofs["ingress"]) is None
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
    compute's honest trace: the egress member opens the network certifier's rows, and the ingress unit that computes
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
def test_a_trace_whose_state_doesnt_follow_from_the_bucket_before_is_refused_at_the_chain(tmp_path):
    """The trace's state leaf k isn't the S_(k−1) that unit k − 1 computes, and the prover's ``scan/next`` and views
    are what its units compute on that trace, so every registered read opens: the ingress statement is refused only
    by its commitment chain, at the instance whose ``o1`` the next one's ``sin`` doesn't commit to."""
    T, chain = TIE.T, Run(tmp_path / "honest").chain
    k = next(t for t in range(2, T) if chain.states[t - 1] != chain.states[t])
    run = Run(tmp_path, states=[*chain.states[:k], chain.states[k - 1], *chain.states[k + 1:T]])
    got = run.computed()
    assert all(ok for ok, _, _ in got)
    run.views, run.nexts = [v for _, _, v in got], [s for _, s, _ in got[1:]]
    run.draw()
    verdict = run.verify(run.prove("chain"))
    assert not verdict["accepted"] and "first" not in verdict["refused"]
    assert verdict["refused"]["ingress"] == (f"chain: instance {k - 2}'s o1 commitment isn't instance {k - 1}'s sin "
                                             "commitment")


@pytest.mark.slow
def test_the_chain_reads_each_instances_commitments_through_its_refs(honest):
    """`chained` on the honest ingress statement's public file, read through the circuit's META ports: one ``sin``
    table row changed breaks the chain at the instance before it, and a circuit file without META ports is refused."""
    run, tmp = honest[0], honest[0].tmp / "chain-read"
    proof = run.delivered.proofs["ingress"]
    tmp.mkdir()
    head, body = proof.public.read_bytes().split(b"\n", 1)
    rows = json.loads(head)["shared_rows"]
    at = (rows["row"] + 3) * SC.BC_BYTES
    public = tmp / "public.bin"
    public.write_bytes(head + b"\n" + body[:at] + bytes(SC.BC_BYTES) + body[at + SC.BC_BYTES:])
    assert SC.chained(replace(proof, public=public)) == \
        "chain: instance 2's o1 commitment isn't instance 3's sin commitment"
    bare = tmp / "circuit.txt"
    bare.write_text("stand-in circuit\n")
    assert SC.chained(replace(proof, circuit=bare)).startswith("chain: the statement's commitments can't be read")


@pytest.mark.slow
def test_a_registration_off_the_heads_or_the_trace_is_refused(honest):
    """The challenger refuses a registration whose trace roots aren't the trace's, or whose window isn't its order in
    the stream; and the verifier of record refuses a drawn window against another trace than the one registered."""
    run = honest[0]
    other = SC.Trace(replace(run.trace.state, root=bytes(64)), run.trace.progs)
    stream = run.tie.stream(budget=8, delta=2.0 ** -40, challenger="c", coins=lambda n: b"\x07" * n, clock=lambda: 0.0)
    record = run.tie.registration(run.heads, run.trace, run.view.root, run.next.root)
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
    digest, committed and decided by N2's verifier of record on a stand-in proof."""

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
    """On a network certifier link the network verdict counts a window only where its egress tie accepted it: the
    honest run is accepted; the run whose server sent one token other than the compute's has N2 counts that accept and
    is refused, by its window; a link with no egress tie is refused by name; and an egress tie of another link, session
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
    """Nothing in `verity` but the scan service, which builds `EgressTieVerdict` from the stream's outcomes of record,
    constructs one."""
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
