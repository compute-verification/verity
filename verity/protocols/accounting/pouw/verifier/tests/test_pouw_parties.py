"""PoUW's execution phases as three parties of the deployment layer (`parties`), run in simulated time: the challenger
commits each phase's challenges before the phase is begun, opens the salt challenge after the prover's root and the
sampling challenge after the registration's digest, and the verifier decides each phase on its own."""

from __future__ import annotations

import hashlib

from verity.channel import Message
from verity.party import Received
from verity.protocol import ProtocolDef
from verity.protocols.accounting.pouw.verifier import coins as Co
from verity.protocols.accounting.pouw.verifier import parties as P
from verity.runtime import sources as S
from verity.runtime.runner import run

SEED = b"pouw parties"
ALL = [P.CHALLENGER, P.PROVER, P.VERIFIER]


def h(*parts: bytes) -> bytes:
    return hashlib.sha512(b"".join(parts)).digest()


def root(i: int) -> bytes:
    return h(b"weights", bytes([i]))


def registration(i: int, salt: Co.CoinOpening) -> bytes:
    return b"registration %d " % i + salt.coin


def sample(i: int, reg: bytes, opening: Co.CoinOpening) -> bytes:
    return h(reg, opening.coin)[:8]


def honest_proof(salt: Co.CoinOpening, opening: Co.CoinOpening, smp: bytes) -> bytes:
    return h(b"proof", salt.coin, opening.coin, smp)


def prover_machine(serve=registration, open_=root) -> dict:
    """The prover's machine: its root, its registration on the salt and a proof that only an honest serving of that
    salt and sample makes."""
    salts: dict[int, Co.CoinOpening] = {}

    def serving(i, salt):
        salts[i] = salt
        return serve(i, salt)

    return {"open": open_, "serve": serving, "prove": lambda i, o, smp: honest_proof(salts[i], o, smp)}


def check(i, reg, salt, opening, smp, proof):
    ok = proof == honest_proof(salt, opening, smp)
    return {"verdict": "accept" if ok else "reject", "reason": None if ok else "proof"}


def registered(i: int, reg: bytes) -> bytes:
    if not reg.startswith(b"registration "):
        raise ValueError("not PoUW's")
    return h(reg)


VERIFIER_MACHINE = {"registered": registered, "sample": sample, "check": check}


def simulated(tmp_path, phases=2, prover=None, challenger=P.ChallengerParty, verifier=P.Verifier, prover_party=P.Prover):
    proto = ProtocolDef(P.MANIFEST, {P.CHALLENGER: challenger, P.PROVER: prover_party, P.VERIFIER: verifier})
    return run({"protocol": P.ID, "params": P.params_of(phases)}, ALL, "simulated", tmp_path, protocols={P.ID: proto},
               seed=SEED, authenticate=False,
               capabilities={P.PROVER: prover or prover_machine(), P.VERIFIER: VERIFIER_MACHINE})


def challengers(phases: int) -> list[Co.Challenger]:
    """The challenger's challenges as its seeded `coins` need gives them, one `coins.Challenger` a phase."""
    src, ask = S.seeded(SEED, P.CHALLENGER, "coins"), Received("coins", Message(P.ASK, {"n": P.PHASE_COINS}))
    return [Co.Challenger(i, P._Stream(src.on(ask)[0].message.body)) for i in range(phases)]


def test_each_phase_is_decided_on_coins_opened_after_what_they_follow(tmp_path) -> None:
    """Every phase accepts. The challenger's commitments are `coins.Challenger`'s on its `coins` need, and it opened
    each phase's salt challenge after the prover's root and its sampling challenge after the registration's digest, once
    each."""
    r = simulated(tmp_path, phases=3)
    assert r.status == "completed", r.outcomes
    lines = r.outcomes[P.VERIFIER]["outcome"]["phases"]
    assert [(x["phase"], x["verdict"]) for x in lines] == [(0, "accept"), (1, "accept"), (2, "accept")]
    assert r.outcomes[P.PROVER] == {"outcome": {"proved": 3}, "refused": None}
    want = []
    for c in challengers(3):
        salt = c.open(P.SALT, root(c.epoch))
        digest = h(registration(c.epoch, salt))
        assert lines[c.epoch]["registration"] == digest.hex()
        want += [{"phase": c.epoch, "label": P.SALT, "after": root(c.epoch).hex()},
                 {"phase": c.epoch, "label": P.SAMPLING, "after": digest.hex()}]
    assert r.outcomes[P.CHALLENGER] == {"outcome": {"opened": want}, "refused": None}


def test_a_wrong_proof_rejects_its_phase_and_the_next_is_still_decided(tmp_path) -> None:
    machine = prover_machine()
    honest = machine["prove"]
    machine["prove"] = lambda i, o, smp: b"wrong" if i == 0 else honest(i, o, smp)
    lines = simulated(tmp_path, prover=machine).outcomes[P.VERIFIER]["outcome"]["phases"]
    assert [(x["verdict"], x["reason"]) for x in lines] == [("reject", "proof"), ("accept", None)]


def test_a_registration_that_is_not_pouws_rejects_its_phase_with_no_sampling_coin_opened(tmp_path) -> None:
    machine = prover_machine(serve=lambda i, salt: b"other" if i == 0 else registration(i, salt))
    r = simulated(tmp_path, prover=machine)
    lines = r.outcomes[P.VERIFIER]["outcome"]["phases"]
    assert [(x["verdict"], x["registration"]) for x in lines][0] == ("reject", None)
    assert lines[1]["verdict"] == "accept"
    opened = r.outcomes[P.CHALLENGER]["outcome"]["opened"]
    assert [(o["phase"], o["label"]) for o in opened] == [(0, P.SALT), (1, P.SALT), (1, P.SAMPLING)]
    assert r.outcomes[P.PROVER] == {"outcome": {"proved": 1}, "refused": None}


class OtherCoins(P.ChallengerParty):
    """Opens phase 0's salt challenge from randomness it never committed to."""

    def on(self, r):
        if r.message.type == "Open" and (r.message.fields["phase"], r.message.fields["label"]) == (0, P.SALT):
            self.c[0] = Co.Challenger(0, P._Stream(bytes(P.PHASE_COINS)))
        return super().on(r)


def test_a_coin_that_does_not_open_against_its_commitment_refuses_the_run(tmp_path) -> None:
    r = simulated(tmp_path, challenger=OtherCoins)
    out = r.outcomes[P.VERIFIER]
    assert out["refused"] == "coin" and out["outcome"]["phases"] == []


class TwoRoots(P.Prover):
    """Sends its root twice: the second is one the verifier didn't ask for."""

    def on(self, r):
        out = super().on(r)
        return out + out if r.message.type == "Begin" else out


def test_a_frame_the_verifier_did_not_ask_for_ends_the_run_out_of_phase(tmp_path) -> None:
    out = simulated(tmp_path, prover_party=TwoRoots).outcomes[P.VERIFIER]
    assert out["refused"] == "out-of-phase"


def test_a_coin_is_opened_once(tmp_path) -> None:
    """A second `Open` of one challenge is refused: a retry never resamples."""

    class Again(P.Verifier):
        def _open(self, label, after):
            out = super()._open(label, after)
            return out + out

    r = simulated(tmp_path, verifier=Again)
    assert r.outcomes[P.CHALLENGER]["refused"] == "out-of-phase"
