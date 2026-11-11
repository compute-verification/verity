"""SPOT[Flock]'s own deployment on the deployment layer (`parties.SPOT_PROTOCOL`), every party on stand-in capabilities: an
honest run draws each block's sample on the challenger's coins once its registration is in, the proof server proves it from
the inference server's replay inputs and that draw, and the verifier audits each block in order on its proof and accepts,
with no ZK server, no pin and no composite; the inference server only sends, and the proof server hears only the
inference server and the verifier and answers only the verifier; an accepting audit of another registration rejects; an
audit that is `Unavailable` is its block's error, a draw that is ends the verifier refused ``draw``, and a proof server or
inference server that can't do its step leaves the verifier refused ``closed``; parameters out of range are refused by
every party; the verifier replays from its own transcript; and the placement holds each party to its principal's machine
and the draws to a link no developer's machine relays.

In two stages (`parties.SPOT_TWO_PROTOCOL`) each block's interior is registered once its draw is in, its second stage
drawn only once that interior is in, and the block proved on both draws and audited on both, its line naming the interior;
an accepting audit of another interior rejects; a second-stage draw that is `Unavailable` ends the verifier refused
``second``, and an interior the proof server can't register leaves the verifier refused ``closed``; and a message out of
its phase (an interior of a block not drawn or registered twice, a proof before its second-stage draw, a second-stage draw
before its interior) is refused ``out-of-phase``."""

from __future__ import annotations

import json

import pytest

from verity.configuration.deploy.manifest import place, site_of
from verity.configuration.deploy.party import Identity, Received
from verity.configuration.deploy.runtime import run
from verity.protocols.spot.verifier import parties as P
from verity.protocols.spot.verifier.one_stage import registration as R
from verity.configuration.wire import Message, Refused

ROLES = [P.CHALLENGER, P.INFERENCE_SERVER, P.PROOF_SERVER, P.VERIFIER]
ADDRESS = "127.0.0.1:4242"


def registration(i: int) -> bytes:
    return json.dumps({"block": i, "root": "ab" * 64}, sort_keys=True).encode()


def replay(i: int) -> bytes:
    return f"replay {i}".encode()


def sample(i: int) -> bytes:
    return bytes([i, 7])


def proof(i: int) -> bytes:
    return f"proof {i}".encode()


def interior(i: int) -> bytes:
    return json.dumps({"block": i, "previous": R.digest(json.loads(registration(i))), "root": "cd" * 64},
                      sort_keys=True).encode()


def second(i: int) -> bytes:
    return bytes([i, 9])


def digest(raw: bytes) -> str:
    return R.digest(json.loads(raw))


class Host:
    def __init__(self) -> None:
        self.events: list[str] = []

    def start(self) -> dict[str, str]:
        self.events.append("start")
        return {"address": ADDRESS, "pin": "f" * 40}

    def stop(self) -> None:
        self.events.append("stop")


class Machine:
    """Stand-in capabilities: what each was called with, in order. ``fails`` names a capability that is `Unavailable`."""

    def __init__(self, *, accept: bool = True, commitment: str | None = None, fails: str | None = None,
                 audit_raises: Exception | None = None, audited_interior: str | None = None) -> None:
        self.host, self.calls = Host(), []
        self.accept, self.commitment, self.fails, self.audit_raises = accept, commitment, fails, audit_raises
        self.audited_interior = audited_interior

    def _step(self, name: str, *args) -> None:
        self.calls.append((name, *args))
        if self.fails == name:
            raise P.Unavailable(f"no {name}")

    def serve(self, i: int) -> bytes:
        self._step("serve", i)
        return registration(i)

    def replay(self, i: int) -> bytes:
        self._step("replay", i)
        return replay(i)

    def draw(self, i, raw, challenger):
        self._step("draw", i, raw, challenger["address"])
        return sample(i)

    def interior(self, i, replayed, drawn):
        self._step("interior", i, replayed, drawn)
        return interior(i)

    def second(self, i, raw, challenger, drawn, inner):
        self._step("second", i, raw, challenger["address"], drawn, inner)
        return second(i)

    def prove(self, i, replayed, drawn, *rest):
        self._step("prove", i, replayed, drawn, *rest)
        return proof(i)

    def audit(self, i, raw, challenger, drawn, proved, *two):
        self._step("audit", i, raw, challenger["address"], drawn, proved, *two)
        if self.audit_raises:
            raise self.audit_raises
        answer = {"verdict": "accept" if self.accept else "reject", "commitment": self.commitment or digest(raw),
                  "reason": None if self.accept else "a sampled proof block's proof rejects"}
        if two:
            answer["interior"] = self.audited_interior or digest(two[0])
        return answer

    def caps(self) -> dict:
        return {P.CHALLENGER: {"host": self.host},
                P.INFERENCE_SERVER: {"serve": self.serve, "replay": self.replay},
                P.PROOF_SERVER: {"interior": self.interior, "prove": self.prove},
                P.VERIFIER: {"draw": self.draw, "second": self.second, "audit": self.audit}}


def go(m: Machine, *, blocks: int = 1, coins: str = "simulated", params=None, state=None, protocol: str = P.SPOT_ID):
    params = params if params is not None else P.spot_params_of(blocks, coins)
    return run({"protocol": protocol, "params": params}, ROLES, "simulated", state, protocols=P.PROTOCOLS,
               capabilities=m.caps(), seed=b"SPOT[Flock]'s parties", authenticate=False)


def lines_of(r) -> list[dict]:
    return r.outcomes[P.VERIFIER]["outcome"]["verdicts"]


def test_an_honest_run_draws_proves_and_audits_each_block_in_order_and_accepts():
    m = Machine()
    r = go(m, blocks=2)
    assert r.status == "completed", r.outcomes
    lines = lines_of(r)
    assert lines == [{"block": i, "coins": "simulated", "reason": None,
                      "registration": R.digest(json.loads(registration(i))), "verdict": "accept"} for i in (0, 1)]
    assert [c for c in m.calls if c[0] == "draw"] == [("draw", i, registration(i), ADDRESS) for i in (0, 1)]
    assert [c for c in m.calls if c[0] == "prove"] == [("prove", i, replay(i), sample(i), ADDRESS) for i in (0, 1)]
    assert [c for c in m.calls if c[0] == "audit"] == [("audit", i, registration(i), ADDRESS, sample(i), proof(i))
                                                       for i in (0, 1)]
    assert m.host.events == ["start", "stop"]
    outputs = [json.loads(e.data) for e in r.transcripts[P.VERIFIER] if e.dir == "record" and e.kind == "output"]
    assert outputs == lines
    assert r.outcomes[P.PROOF_SERVER] == {"outcome": {"proved": 2}, "refused": None}


def test_its_parties_and_channels_are_spots_three_roles_with_no_zk_server():
    assert {(r.name, r.principal) for r in P.SPOT_MANIFEST.parties} == {
        (P.CHALLENGER, "auditor"), (P.INFERENCE_SERVER, "developer"), (P.PROOF_SERVER, "developer"), (P.VERIFIER, "auditor")}
    assert {(c.sender, c.receiver, c.messages) for c in P.SPOT_MANIFEST.channels} == {
        (P.CHALLENGER, P.VERIFIER, ("Challenger",)), (P.VERIFIER, P.CHALLENGER, ("Done",)),
        (P.INFERENCE_SERVER, P.VERIFIER, ("Registration",)), (P.INFERENCE_SERVER, P.PROOF_SERVER, ("Replay",)),
        (P.VERIFIER, P.PROOF_SERVER, ("Draw",)), (P.PROOF_SERVER, P.VERIFIER, ("Proof",))}
    assert not any("pin" in r.needs for r in P.SPOT_MANIFEST.parties)
    assert P.SPOT_ID not in {i for subs in P.SUBPROTOCOLS.values() for i in subs.values()}
    r = go(Machine(), blocks=2)
    heard = {p: {(e.dir, e.channel) for e in r.transcripts[p] if e.dir in ("in", "out")} for p in ROLES}
    assert heard[P.INFERENCE_SERVER] == {("out", P.VERIFIER), ("out", P.PROOF_SERVER)}
    assert heard[P.PROOF_SERVER] == {("in", P.INFERENCE_SERVER), ("in", P.VERIFIER), ("out", P.VERIFIER)}


def test_an_accepting_audit_of_another_registration_rejects():
    (line,) = lines_of(go(Machine(commitment="00" * 64)))
    assert line["verdict"] == "reject"
    assert line["reason"] == "the audit's commitment is not the registration the inference server sent"


def test_a_rejecting_audit_is_the_blocks_verdict():
    (line,) = lines_of(go(Machine(accept=False)))
    assert (line["verdict"], line["reason"]) == ("reject", "a sampled proof block's proof rejects")


def test_an_unavailable_audit_is_its_blocks_error():
    r = go(Machine(fails="audit"))
    assert r.status == "completed"
    (line,) = lines_of(r)
    assert line["verdict"] == "error"
    (why,) = [json.loads(e.data) for e in r.transcripts[P.VERIFIER] if e.dir == "record" and e.kind == "private"]
    assert why == {"block": 0, "audit": "unavailable", "error": "no audit"}


def test_an_audit_that_raises_anything_else_fails_the_run_by_name():
    r = go(Machine(audit_raises=KeyError("verdict")))
    assert r.status == "failed" and r.outcomes[P.VERIFIER]["refused"] == "party-died"


def test_an_unavailable_draw_ends_the_verifier_refused_draw_with_nothing_proved():
    m = Machine(fails="draw")
    r = go(m)
    assert r.outcomes[P.VERIFIER] == {"outcome": {"verdicts": []}, "refused": "draw"}
    assert r.outcomes[P.PROOF_SERVER]["refused"] == "closed"
    assert not [c for c in m.calls if c[0] in ("prove", "audit")] and m.host.events == ["start", "stop"]


@pytest.mark.parametrize("step, party, protocol", [
    ("serve", P.INFERENCE_SERVER, P.SPOT_ID), ("replay", P.INFERENCE_SERVER, P.SPOT_ID), ("prove", P.PROOF_SERVER, P.SPOT_ID),
    ("replay", P.INFERENCE_SERVER, P.SPOT_TWO_ID), ("interior", P.PROOF_SERVER, P.SPOT_TWO_ID),
    ("prove", P.PROOF_SERVER, P.SPOT_TWO_ID)])
def test_a_server_that_cant_do_its_step_leaves_the_verifier_refused_closed(step, party, protocol):
    m = Machine(fails=step)
    r = go(m, protocol=protocol)
    assert r.outcomes[party]["refused"] == step
    assert r.outcomes[P.VERIFIER] == {"outcome": {"verdicts": []}, "refused": "closed"}
    assert not [c for c in m.calls if c[0] == "audit"] and m.host.events == ["start", "stop"]
    if step == "interior":
        assert not [c for c in m.calls if c[0] in ("second", "prove")]


@pytest.mark.parametrize("bad", [{"blocks": 0, "coins": "simulated"}, {"blocks": 1, "coins": "gate"},
                                 {"blocks": 1, "coins": "simulated", "components": ["sampled-proofs"]}])
def test_parameters_out_of_range_are_refused(bad):
    if set(bad) == {"blocks", "coins"}:
        with pytest.raises(ValueError):
            P.spot_params_of(bad["blocks"], bad["coins"])
    m = Machine()
    r = go(m, params=bad)
    assert all(r.outcomes[p]["refused"] == "schema" for p in (P.INFERENCE_SERVER, P.PROOF_SERVER, P.VERIFIER))
    assert not [c for c in m.calls if c[0] in ("serve", "draw")] and m.host.events == ["start", "stop"]


def test_the_verifier_replays_from_its_own_transcript(tmp_path):
    r = go(Machine(), state=tmp_path)
    assert r.status == "completed"
    again = run({"protocol": P.SPOT_ID, "params": P.spot_params_of(1, "simulated")}, [P.VERIFIER], "replay", None,
                protocols=P.PROTOCOLS, capabilities={P.VERIFIER: Machine().caps()[P.VERIFIER]},
                replay_dir=tmp_path / "runs" / "1")
    assert (again.status, again.divergence) == ("completed", None)
    assert again.outcomes[P.VERIFIER] == r.outcomes[P.VERIFIER]


def test_the_placement_holds_each_party_to_its_principals_machine_and_the_draws_off_the_developers():
    every = sorted({n for r in P.SPOT_MANIFEST.parties for n in r.needs})
    machines = [{"name": "inference-1", "owner": "developer", "provides": every},
                {"name": "prover-1", "owner": "developer", "provides": every},
                {"name": "auditor-1", "owner": "auditor", "provides": every}]
    direct = [{"ends": [a, b], "latency_ms": 1} for a, b in (("inference-1", "prover-1"), ("inference-1", "auditor-1"),
                                                            ("prover-1", "auditor-1"))]
    bound = {P.INFERENCE_SERVER: "inference-1", P.PROOF_SERVER: "prover-1", P.CHALLENGER: "auditor-1",
             P.VERIFIER: "auditor-1"}
    assert place(P.SPOT_MANIFEST, site_of(machines, direct), bound).mode == "placed"
    with pytest.raises(Refused) as e:
        place(P.SPOT_MANIFEST, site_of(machines, direct), {**bound, P.VERIFIER: "prover-1"})
    assert e.value.name == "principal"
    relayed = [*direct[:2], {"ends": ["prover-1", "auditor-1"], "latency_ms": 1, "via": ["inference-1"]}]
    with pytest.raises(Refused) as e:
        place(P.SPOT_MANIFEST, site_of(machines, relayed), bound)
    assert e.value.name == "coins-hop"


def test_two_stages_draw_each_blocks_second_stage_only_once_its_interior_is_in_and_audit_both():
    m = Machine()
    r = go(m, blocks=2, protocol=P.SPOT_TWO_ID)
    assert r.status == "completed", r.outcomes
    lines = lines_of(r)
    assert lines == [{"block": i, "coins": "simulated", "interior": digest(interior(i)), "reason": None,
                      "registration": digest(registration(i)), "verdict": "accept"} for i in (0, 1)]
    for i in (0, 1):
        assert [c[0] for c in m.calls if c[0] not in ("serve", "replay") and c[1] == i] == [
            "draw", "interior", "second", "prove", "audit"]
    assert [c for c in m.calls if c[0] == "interior"] == [("interior", i, replay(i), sample(i)) for i in (0, 1)]
    assert [c for c in m.calls if c[0] == "second"] == [("second", i, registration(i), ADDRESS, sample(i), interior(i))
                                                        for i in (0, 1)]
    assert [c for c in m.calls if c[0] == "prove"] == [("prove", i, replay(i), sample(i), interior(i), second(i), ADDRESS)
                                                       for i in (0, 1)]
    assert [c for c in m.calls if c[0] == "audit"] == [
        ("audit", i, registration(i), ADDRESS, sample(i), proof(i), interior(i), second(i)) for i in (0, 1)]
    assert m.host.events == ["start", "stop"]
    assert [json.loads(e.data) for e in r.transcripts[P.VERIFIER] if e.dir == "record" and e.kind == "output"] == lines
    assert r.outcomes[P.PROOF_SERVER] == {"outcome": {"proved": 2}, "refused": None}


def test_two_stages_add_the_interior_and_the_second_draw_to_spots_channels():
    spot = {(c.sender, c.receiver, c.messages) for c in P.SPOT_MANIFEST.channels}
    two = {(c.sender, c.receiver, c.messages) for c in P.SPOT_TWO_MANIFEST.channels}
    assert two - spot == {(P.VERIFIER, P.PROOF_SERVER, ("Draw", "Second")),
                          (P.PROOF_SERVER, P.VERIFIER, ("Interior", "Proof"))}
    assert spot - two == {(P.VERIFIER, P.PROOF_SERVER, ("Draw",)), (P.PROOF_SERVER, P.VERIFIER, ("Proof",))}
    assert {r.name: (r.principal, r.needs) for r in P.SPOT_TWO_MANIFEST.parties} == {
        P.CHALLENGER: ("auditor", ("host",)), P.INFERENCE_SERVER: ("developer", ("serve", "replay")),
        P.PROOF_SERVER: ("developer", ("interior", "prove")), P.VERIFIER: ("auditor", ("draw", "second", "audit"))}
    assert P.SPOT_TWO_MANIFEST.channel(P.VERIFIER, P.PROOF_SERVER).carries_coins
    assert {s.type: s.kind for s in P.SPOT_TWO_MANIFEST.messages if s.type in ("Interior", "Second")} == {
        "Interior": "commitment", "Second": "coins"}
    assert P.SPOT_TWO_ID not in {i for subs in P.SUBPROTOCOLS.values() for i in subs.values()}


@pytest.mark.parametrize("machine, reason", [
    (Machine(audited_interior="00" * 64), "the audit's interior is not the interior registration the proof server sent"),
    (Machine(commitment="00" * 64), "the audit's commitment is not the registration the inference server sent")])
def test_in_two_stages_an_accepting_audit_of_another_record_rejects(machine, reason):
    (line,) = lines_of(go(machine, protocol=P.SPOT_TWO_ID))
    assert (line["verdict"], line["reason"]) == ("reject", reason)


def test_in_two_stages_an_unavailable_second_draw_ends_the_verifier_refused_second_with_nothing_proved():
    m = Machine(fails="second")
    r = go(m, protocol=P.SPOT_TWO_ID)
    assert r.outcomes[P.VERIFIER] == {"outcome": {"verdicts": []}, "refused": "second"}
    assert r.outcomes[P.PROOF_SERVER]["refused"] == "closed"
    (why,) = [json.loads(e.data) for e in r.transcripts[P.VERIFIER] if e.dir == "record" and e.kind == "private"]
    assert why == {"block": 0, "second": "unavailable", "error": "no second"}
    assert not [c for c in m.calls if c[0] in ("prove", "audit")] and m.host.events == ["start", "stop"]


def _started(party, blocks: int = 2):
    assert party.start(P.spot_params_of(blocks, "simulated"), Identity("p", capabilities=Machine().caps()[
        P.VERIFIER if isinstance(party, P.SpotVerifier) else P.PROOF_SERVER])) == []
    return party


def _said(party, channel: str, kind: str, i: int, body: bytes = b"", **fields):
    return party.on(Received(channel, Message(kind, {"block": i, **fields}, body)))


def _refused(out) -> str | None:
    return next((x.refused for x in out if hasattr(x, "refused")), None)


def _drawn():
    """A two-stage verifier that has drawn block 0's first stage."""
    v = _started(P.TwoStageVerifier())
    v.on(Received(P.CHALLENGER, Message("Challenger", {"address": ADDRESS, "pin": "f" * 40})))
    assert [s.message.type for s in _said(v, P.INFERENCE_SERVER, "Registration", 0, registration(0))] == ["Draw"]
    return v


def test_in_two_stages_the_verifier_refuses_an_interior_or_a_proof_out_of_its_phase():
    assert _refused(_said(_started(P.TwoStageVerifier()), P.PROOF_SERVER, "Interior", 0, interior(0))) == "out-of-phase"
    assert _refused(_said(_drawn(), P.PROOF_SERVER, "Proof", 0, proof(0))) == "out-of-phase"
    assert _refused(_said(_drawn(), P.PROOF_SERVER, "Interior", 0, b"[1]")) == "schema"
    assert _refused(_said(_drawn(), P.PROOF_SERVER, "Interior", 0, b"{")) == "schema"
    v = _drawn()
    sent = _said(v, P.PROOF_SERVER, "Interior", 0, interior(0))
    assert [(s.message.type, s.message.body) for s in sent] == [("Second", second(0))]
    assert _refused(_said(v, P.PROOF_SERVER, "Interior", 0, interior(0))) == "out-of-phase"


def test_in_two_stages_the_proof_server_refuses_a_second_draw_before_its_interior_or_twice():
    s = _started(P.TwoStageProofServer())
    assert _refused(_said(s, P.VERIFIER, "Second", 0, second(0), address=ADDRESS)) == "out-of-phase"
    s = _started(P.TwoStageProofServer())
    _said(s, P.INFERENCE_SERVER, "Replay", 0, replay(0))
    sent = _said(s, P.VERIFIER, "Draw", 0, sample(0), address=ADDRESS)
    assert [(x.message.type, x.message.body) for x in sent] == [("Interior", interior(0))]
    sent = _said(s, P.VERIFIER, "Second", 0, second(0), address=ADDRESS)
    assert [(x.message.type, x.message.body) for x in sent] == [("Proof", proof(0))]
    assert _refused(_said(s, P.VERIFIER, "Second", 0, second(0), address=ADDRESS)) == "out-of-phase"
    assert _refused(_said(_started(P.TwoStageProofServer()), P.VERIFIER, "Second", 2, second(2), address=ADDRESS)) == "schema"


def test_two_stages_replay_from_the_verifiers_own_transcript_and_place_as_spot_does(tmp_path):
    r = go(Machine(), state=tmp_path, protocol=P.SPOT_TWO_ID)
    assert r.status == "completed"
    again = run({"protocol": P.SPOT_TWO_ID, "params": P.spot_params_of(1, "simulated")}, [P.VERIFIER], "replay", None,
                protocols=P.PROTOCOLS, capabilities={P.VERIFIER: Machine().caps()[P.VERIFIER]},
                replay_dir=tmp_path / "runs" / "1")
    assert (again.status, again.divergence) == ("completed", None)
    assert again.outcomes[P.VERIFIER] == r.outcomes[P.VERIFIER]
    every = sorted({n for role in P.SPOT_TWO_MANIFEST.parties for n in role.needs})
    machines = [{"name": n, "owner": o, "provides": every}
                for n, o in (("inference-1", "developer"), ("prover-1", "developer"), ("auditor-1", "auditor"))]
    direct = [{"ends": [a, b], "latency_ms": 1} for a, b in (("inference-1", "prover-1"), ("inference-1", "auditor-1"),
                                                            ("prover-1", "auditor-1"))]
    bound = {P.INFERENCE_SERVER: "inference-1", P.PROOF_SERVER: "prover-1", P.CHALLENGER: "auditor-1",
             P.VERIFIER: "auditor-1"}
    assert place(P.SPOT_TWO_MANIFEST, site_of(machines, direct), bound).mode == "placed"
    relayed = [*direct[:2], {"ends": ["prover-1", "auditor-1"], "latency_ms": 1, "via": ["inference-1"]}]
    with pytest.raises(Refused) as e:
        place(P.SPOT_TWO_MANIFEST, site_of(machines, relayed), bound)
    assert e.value.name == "coins-hop"
