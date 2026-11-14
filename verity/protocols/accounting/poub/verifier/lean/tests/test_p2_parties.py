"""The P2 session's two parties (`p2_parties.Verifier`, `p2_party_prover.Prover`) run by the deployment layer in
simulated time, the verifier's sources seeded: an honest prover is accepted, its outcome the verifier's answer, and the
verifier's case is signed by the memory challenger; a prover whose receipts skip a unit's reads is rejected; and a replay
of the verifier from its transcript makes the recorded entries until the prover's openings, a frame past the transcript's
bound that it keeps as its digest and a replay can't feed."""

from __future__ import annotations

import json

import pytest
from test_p2_party import TOY, Clock, Prover, closer, exe, request, store  # noqa: F401

from verity.channel import Refused
from verity.protocol import ProtocolDef
from verity.protocols.accounting.poub.prover import p2_party_prover
from verity.protocols.accounting.poub.verifier import p2_lean, p2_parties
from verity.runtime.runner import run

SEED = b"p2 parties"
BOTH = [p2_parties.VERIFIER, p2_parties.PROVER]
PROTOCOLS = {p2_parties.ID: ProtocolDef(p2_parties.MANIFEST, {p2_parties.VERIFIER: p2_parties.Verifier,
                                                              p2_parties.PROVER: p2_party_prover.Prover})}


def simulated(exe, store, state, **kw):
    data = json.dumps(p2_parties.registration_json(store.registration)).encode()
    with p2_lean.Lean(exe) as lean:
        reg, plan = p2_parties.hold(lean, data, auditor=TOY)
        verifier = {"lean": lean, "registration": reg, "plan": plan, "auditor": TOY, "closer": closer(),
                    "host": "auditor-under-test"}
        prover = {"store": Prover(store, Clock(), **kw)}
        return run({"protocol": p2_parties.ID, "params": request(store)}, BOTH, "simulated", state,
                   protocols=PROTOCOLS, seed=SEED, clock_step_ns=100, authenticate=False,
                   capabilities={p2_parties.VERIFIER: verifier, p2_parties.PROVER: prover})


def test_the_runtime_runs_an_honest_session_to_its_signed_case(exe, store, tmp_path) -> None:
    r = simulated(exe, store, tmp_path)
    assert r.status == "completed", r.outcomes
    out = r.outcomes[p2_parties.VERIFIER]
    answer, case = out["outcome"]["answer"], out["outcome"]["case"]
    assert out["refused"] is None and answer["verdict"] == "accept", answer
    assert r.outcomes[p2_parties.PROVER] == {"outcome": answer, "refused": None}
    assert case is not None and json.loads(bytes.fromhex(case["evidence"]))["close"]["lines"] > 0


def test_the_runtime_rejects_a_prover_that_skips_a_units_reads(exe, store, tmp_path) -> None:
    answer = simulated(exe, store, tmp_path, skip=1).outcomes[p2_parties.VERIFIER]["outcome"]["answer"]
    assert answer["verdict"] == "reject" and answer["failure"] == "receipt", answer


@pytest.mark.parametrize("skip", [None, 1], ids=["honest", "skipping"])
def test_a_replay_of_the_verifier_matches_its_transcript_until_the_openings_kept_as_a_digest(exe, store, tmp_path,
                                                                                              skip) -> None:
    simulated(exe, store, tmp_path, skip=skip)
    data = json.dumps(p2_parties.registration_json(store.registration)).encode()
    with p2_lean.Lean(exe) as lean:
        reg, plan = p2_parties.hold(lean, data, auditor=TOY)
        verifier = {"lean": lean, "registration": reg, "plan": plan, "auditor": TOY, "closer": closer(),
                    "host": "auditor-under-test"}
        with pytest.raises(Refused) as err:
            run({"protocol": p2_parties.ID, "params": request(store)}, [p2_parties.VERIFIER], "replay", None,
                protocols=PROTOCOLS, replay_dir=tmp_path / "runs" / "1", capabilities={p2_parties.VERIFIER: verifier})
    assert err.value.name == "elided" and "of verifier's transcript" in str(err.value), err.value
