"""Sampled proofs' parties on the deployment layer (`parties`), every party here on stand-in capabilities: an honest
run's verifier audits each block once its registration, the firewall's pin and the challenger's address are in, in
order, and its composite (remote, over sampled proofs and PoUW) accepts; a listed subprotocol with no answer is absent
and the composite never accepts over it; an accepting audit of another registration rejects; an audit that is
`Unavailable` is its block's error, and one that raises anything else fails the run (``party-died``, its error
recorded); workers that can't serve leave the verifier refused ``closed`` with nothing audited; parameters out of their
range, a list of subprotocols that is no composite's among them, are refused by every party; the challenger's host
stops once the verifier is done, however it finishes; the verifier replays from its own transcript; and the placement
holds each party's principal to its machine's owner."""

from __future__ import annotations

import json

import pytest

from verity.core.deploy.manifest import place, site_of
from verity.core.deploy.runtime import run
from verity.core.protocols.sampled_proofs import parties as P
from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.core.wire import Refused

SP, CERT, POUW = P.ID, P.IDS["certifier"], P.IDS["pouw"]
REMOTE = ("sampled-proofs", "pouw")
ROLES = [P.CHALLENGER, P.FIREWALL, P.VERIFIER, P.WORKERS]
SHA = "5a" * 64
CASE = {"commitment": "01", "evidence": "02", "params": {"k": 8}}
#: another listed subprotocol's accepting answer and its case
OTHER = ({"verdict": "accept", "commitment": "cd" * 64}, {"commitment": "03", "evidence": "04", "params": {}})


def registration(i: int) -> bytes:
    return json.dumps({"block": i, "root": "ab" * 64}, sort_keys=True).encode()


class Host:
    def __init__(self) -> None:
        self.events: list[str] = []

    def start(self) -> dict[str, str]:
        self.events.append("start")
        return {"address": "127.0.0.1:4242", "pin": "f" * 40}

    def stop(self) -> None:
        self.events.append("stop")


class Machine:
    """Stand-in capabilities: what each was called with, in order."""

    def __init__(self, *, accept: bool = True, other: tuple | None = OTHER, commitment: str | None = None,
                 audit_raises: Exception | None = None, serve_raises: bool = False) -> None:
        self.host, self.calls = Host(), []
        self.accept, self.other, self.commitment = accept, other, commitment
        self.audit_raises, self.serve_raises = audit_raises, serve_raises

    def serve(self, i: int) -> bytes:
        self.calls.append(("serve", i))
        if self.serve_raises:
            raise P.Unavailable("no GPU")
        return registration(i)

    def pin(self) -> dict:
        self.calls.append(("pin",))
        return {"sha512": SHA, "sessions": 1}

    def audit(self, i, raw, challenger, firewall):
        self.calls.append(("audit", i, raw, challenger["address"], firewall["sha512"]))
        if self.audit_raises:
            raise self.audit_raises
        digest = self.commitment or R.digest(json.loads(raw))
        return {"verdict": "accept" if self.accept else "reject", "commitment": digest}, CASE

    def component(self, i, name):
        self.calls.append(("component", i, name))
        return self.other

    def product(self, ids, cases, digests):
        self.calls.append(("product", list(ids), dict(cases), dict(digests)))
        empty = [i for i in ids if cases[i] == P.NO_CASE]
        rejected = [i for i in ids if i == SP and not self.accept]
        bad = empty + rejected
        return {"product": "verity/remote/v0" if len(ids) == 2 else "verity/onsite/v0",
                "verdict": "reject" if bad else "accept",
                "reason": "; ".join(f"{i} reject" for i in bad) or None}

    def caps(self) -> dict:
        return {P.CHALLENGER: {"host": self.host}, P.FIREWALL: {"pin": self.pin}, P.WORKERS: {"serve": self.serve},
                P.VERIFIER: {"audit": self.audit, "component": self.component, "product": self.product}}


def go(m: Machine, *, blocks: int = 1, components=REMOTE, coins: str = "simulated", params=None,
       state=None, here=ROLES):
    params = params if params is not None else P.params_of(blocks, list(components), coins)
    return run({"protocol": P.ID, "params": params}, here, "simulated", state, protocols=P.PROTOCOLS,
               capabilities=m.caps(), seed=b"sampled proofs' parties", authenticate=False)


def test_an_honest_run_audits_each_block_in_order_and_accepts():
    m = Machine()
    r = go(m, blocks=2)
    assert r.status == "completed", r.outcomes
    lines = r.outcomes[P.VERIFIER]["outcome"]["verdicts"]
    assert [x["block"] for x in lines] == [0, 1]
    assert lines[0] == {"block": 0, "coins": "simulated", "components": {"sampled-proofs": "accept", "pouw": "accept"},
                        "product": "verity/remote/v0", "reason": None,
                        "registration": R.digest(json.loads(registration(0))), "verdict": "accept"}
    audits = [c for c in m.calls if c[0] == "audit"]
    assert audits == [("audit", i, registration(i), "127.0.0.1:4242", SHA) for i in (0, 1)]
    assert [c for c in m.calls if c[0] == "product"][0][1:] == (
        [SP, POUW], {SP: CASE, POUW: OTHER[1]}, {SP: lines[0]["registration"], POUW: OTHER[0]["commitment"]})
    assert m.host.events == ["start", "stop"]
    outputs = [json.loads(e.data) for e in r.transcripts[P.VERIFIER] if e.dir == "record" and e.kind == "output"]
    assert outputs == lines


def test_a_listed_subprotocol_with_no_answer_is_absent_and_the_composite_rejects():
    m = Machine(other=None)
    r = go(m)
    (line,) = r.outcomes[P.VERIFIER]["outcome"]["verdicts"]
    assert line["components"] == {"sampled-proofs": "accept", "pouw": P.ABSENT}
    assert line["verdict"] == "reject" and line["reason"] == f"{POUW} reject"
    _, ids, cases, digests = next(c for c in m.calls if c[0] == "product")
    assert ids == [SP, POUW] and cases[POUW] == P.NO_CASE and digests[POUW] == ""


def test_onsites_subprotocols_answers_and_cases_reach_the_composite():
    onsite = ("sampled-proofs", "certifier", "pouw", "pous", "bandwidth")
    m = Machine()
    (line,) = go(m, components=onsite).outcomes[P.VERIFIER]["outcome"]["verdicts"]
    assert line["components"] == dict.fromkeys(onsite, "accept") and line["verdict"] == "accept"
    assert line["product"] == "verity/onsite/v0"
    _, ids, _, digests = next(c for c in m.calls if c[0] == "product")
    assert ids == [P.SUBPROTOCOLS["verity/onsite/v0"][c] for c in onsite] and digests[CERT] == "cd" * 64
    assert "verity/pouw/period/v1" in ids and POUW not in ids


def test_an_accepting_audit_of_another_registration_rejects():
    m = Machine(commitment="00" * 64)
    (line,) = go(m).outcomes[P.VERIFIER]["outcome"]["verdicts"]
    assert line["components"] == {"sampled-proofs": "reject", "pouw": "accept"} and line["verdict"] == "reject"
    assert next(c for c in m.calls if c[0] == "product")[2][SP] == P.NO_CASE


def test_an_unavailable_audit_is_its_blocks_error():
    m = Machine(audit_raises=P.Unavailable("the records audit left no summary"))
    r = go(m)
    (line,) = r.outcomes[P.VERIFIER]["outcome"]["verdicts"]
    assert line["components"] == {"sampled-proofs": "error", "pouw": "accept"} and line["verdict"] == "reject"
    assert next(c for c in m.calls if c[0] == "product")[2][SP] == P.NO_CASE
    assert r.status == "completed"
    (why,) = [json.loads(e.data) for e in r.transcripts[P.VERIFIER] if e.dir == "record" and e.kind == "private"]
    assert why == {"block": 0, "audit": "unavailable", "error": "the records audit left no summary"}


def test_an_audit_that_raises_anything_else_fails_the_run_by_name():
    m = Machine(audit_raises=KeyError("verdict"))
    r = go(m)
    assert r.status == "failed"
    assert r.outcomes[P.VERIFIER]["refused"] == "party-died"
    assert "KeyError" in json.dumps(r.outcomes[P.VERIFIER]["outcome"])
    assert not [c for c in m.calls if c[0] == "product"]


def test_workers_that_cant_serve_leave_the_verifier_refused_closed_with_nothing_audited():
    m = Machine(serve_raises=True)
    r = go(m)
    assert r.outcomes[P.WORKERS]["refused"] == "serve"
    assert r.outcomes[P.VERIFIER] == {"outcome": {"verdicts": []}, "refused": "closed"}
    assert not [c for c in m.calls if c[0] == "audit"] and m.host.events == ["start", "stop"]


@pytest.mark.parametrize("bad", [{"blocks": 0, "components": list(REMOTE), "coins": "simulated"},
                                 {"blocks": 1, "components": ["certifier"], "coins": "simulated"},
                                 {"blocks": 1, "components": ["sampled-proofs"], "coins": "simulated"},
                                 {"blocks": 1, "components": ["sampled-proofs", "certifier"], "coins": "simulated"},
                                 {"blocks": 1, "components": ["sampled-proofs", "pouw", "pouw"], "coins": "simulated"},
                                 {"blocks": 1, "components": list(REMOTE), "coins": "gate"},
                                 {"blocks": 1, "components": list(REMOTE), "coins": "simulated", "law": "x"}])
def test_parameters_out_of_range_are_refused(bad):
    if set(bad) == {"blocks", "components", "coins"}:
        with pytest.raises(ValueError):
            P.params_of(bad["blocks"], bad["components"], bad["coins"])
    m = Machine()
    r = go(m, params=bad)
    assert r.outcomes[P.VERIFIER]["refused"] == "schema" and r.outcomes[P.WORKERS]["refused"] == "schema"
    assert not [c for c in m.calls if c[0] in ("serve", "audit")] and m.host.events == ["start", "stop"]


def test_the_verifier_replays_from_its_own_transcript(tmp_path):
    m = Machine()
    r = go(m, state=tmp_path)
    assert r.status == "completed"
    again = run({"protocol": P.ID, "params": P.params_of(1, list(REMOTE), "simulated")}, [P.VERIFIER], "replay",
                None, protocols=P.PROTOCOLS, capabilities={P.VERIFIER: Machine().caps()[P.VERIFIER]},
                replay_dir=tmp_path / "runs" / "1")
    assert (again.status, again.divergence) == ("completed", None)
    assert again.outcomes[P.VERIFIER] == r.outcomes[P.VERIFIER]


def test_the_placement_holds_each_party_to_its_principals_machine():
    every = sorted({n for r in P.MANIFEST.parties for n in r.needs})
    site = site_of([{"name": "prover-1", "owner": "developer", "provides": every},
                    {"name": "auditor-1", "owner": "auditor", "provides": every}],
                   [{"ends": ["prover-1", "auditor-1"], "latency_ms": 1}])
    bound = {P.WORKERS: "prover-1", P.FIREWALL: "prover-1", P.CHALLENGER: "auditor-1", P.VERIFIER: "auditor-1"}
    assert place(P.MANIFEST, site, bound).mode == "placed"
    with pytest.raises(Refused) as e:
        place(P.MANIFEST, site, {**bound, P.CHALLENGER: "prover-1"})
    assert e.value.name == "principal"
    got = place(P.MANIFEST, site, {**bound, P.CHALLENGER: "prover-1"}, simulated=True)
    assert [n for n, _ in got.unmet] == ["principal"]
