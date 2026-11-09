"""SPOT[Flock]'s own deployment on the deployment layer (`parties.SPOT_PROTOCOL`), every party on stand-in capabilities: an
honest run draws each block's sample on the challenger's coins once its registration is in, the proof server proves it from
the inference server's replay inputs and that draw, and the verifier audits each block in order on its proof and accepts,
with no ZK server, no pin and no composite; the inference server only sends, and the proof server hears only the
inference server and the verifier and answers only the verifier; an accepting audit of another registration rejects; an
audit that is `Unavailable` is its block's error, a draw that is ends the verifier refused ``draw``, and a proof server or
inference server that can't do its step leaves the verifier refused ``closed``; parameters out of range are refused by
every party; the verifier replays from its own transcript; and the placement holds each party to its principal's machine
and the draws to a link no developer's machine relays."""

from __future__ import annotations

import json

import pytest

from verity.core.deploy.manifest import place, site_of
from verity.core.deploy.runtime import run
from verity.core.protocols.spot import parties as P
from verity.core.protocols.spot.one_stage import registration as R
from verity.core.wire import Refused

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
                 audit_raises: Exception | None = None) -> None:
        self.host, self.calls = Host(), []
        self.accept, self.commitment, self.fails, self.audit_raises = accept, commitment, fails, audit_raises

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

    def prove(self, i, replayed, drawn, address):
        self._step("prove", i, replayed, drawn, address)
        return proof(i)

    def audit(self, i, raw, challenger, drawn, proved):
        self._step("audit", i, raw, challenger["address"], drawn, proved)
        if self.audit_raises:
            raise self.audit_raises
        digest = self.commitment or R.digest(json.loads(raw))
        return {"verdict": "accept" if self.accept else "reject", "commitment": digest,
                "reason": None if self.accept else "a sampled proof block's proof rejects"}

    def caps(self) -> dict:
        return {P.CHALLENGER: {"host": self.host},
                P.INFERENCE_SERVER: {"serve": self.serve, "replay": self.replay},
                P.PROOF_SERVER: {"prove": self.prove},
                P.VERIFIER: {"draw": self.draw, "audit": self.audit}}


def go(m: Machine, *, blocks: int = 1, coins: str = "simulated", params=None, state=None):
    params = params if params is not None else P.spot_params_of(blocks, coins)
    return run({"protocol": P.SPOT_ID, "params": params}, ROLES, "simulated", state, protocols=P.PROTOCOLS,
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


@pytest.mark.parametrize("step, party", [("serve", P.INFERENCE_SERVER), ("replay", P.INFERENCE_SERVER),
                                         ("prove", P.PROOF_SERVER)])
def test_a_server_that_cant_do_its_step_leaves_the_verifier_refused_closed(step, party):
    m = Machine(fails=step)
    r = go(m)
    assert r.outcomes[party]["refused"] == step
    assert r.outcomes[P.VERIFIER] == {"outcome": {"verdicts": []}, "refused": "closed"}
    assert not [c for c in m.calls if c[0] == "audit"] and m.host.events == ["start", "stop"]


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
