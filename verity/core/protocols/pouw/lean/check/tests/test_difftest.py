"""`verity.core.protocols.pouw.core_check` held to `pouw-check serve`, PoUW's verifier of record (`pouw.verifier`), on
the shared harness (`verity_difftest`, `of_record="lean"`): one `ProtocolPort`, over the cases `pouw_check_cases`
generates from a Hypothesis-controlled `random.Random` (honest windows, and one mutation each aimed at every refusal, the
case's framing included), so a run is deterministic and shrinks. Every port here holds the Python to the Lean.

The harness compares an outcome's verdict, commitment, public outputs and profile. The reference writes the port's
reasons too, `Core.Protocol.case`'s framing included, so this port compares the whole outcome: the reason names the rule
that refused, and agreeing on it holds which rule refused, not only that one did.

`pouw-check-difftest`'s `opening` against `core_check.opening`: one `Port`, over the openings `pouw_opening_cases`
generates (honest openings of a call's tile or the filler's, unsalted or a salted unit's in a certified window, and one
mutation each aimed at every refusal or at a reading both sides share), compared output for output: null, or the same
reason.

Its `pass` and `draw` against `core_check.pass_epoch` and `core_check.draw`: one `Port` each, over the cases
`pouw_pass_cases` generates (honest passes of each coin version and a recorded beacon's, and honest draws, then one
mutation each aimed at every refusal), compared output for output: how the salt was had, or the same refusal.

Its `credit`, `outcome` and `a-link` against `credit.credited`, `credit.outcome` and `credit.a_link`: one `Port` each,
over the cases `pouw_credit_cases` generates (a served window whose draw coin opened after its draw root over the
manifest's credit bits, a draw's units each proved, a window's A trees under its round roots, then one mutation each aimed
at every refusal), compared output for output: the credited calls or null, or the same refusal.

Its `rounds` against `core_check.rounds`: one `Port`, over the live windows `pouw_rounds_cases` generates (a live epoch's
round coins, each opened after its recorded round root, read through the challenger's signed session the auditor's signed
ledger opens and ends for the window, then one mutation each aimed at every refusal), compared output for output: null, or the same refusal.
"""
from __future__ import annotations

import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import pytest
import verity_difftest as D
from hypothesis import strategies as st

import pouw_check_cases as cases
import pouw_credit_cases as credit_cases
import pouw_opening_cases as openings
import pouw_pass_cases as passes
import pouw_rounds_cases as rounds_cases
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import credit as Cr
from verity.core.protocols.pouw import verifier as V

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

LEAN = Path(__file__).resolve().parents[1]
EXAMPLES = 4000

#: one honest case in about ten
NAMES = ["honest"] * (len(cases.NAMES) // 9) + cases.NAMES
INPUTS = st.builds(cases.instance, st.randoms(use_true_random=False), st.sampled_from(NAMES))


class ExactPort(D.ProtocolPort):
    def key(self, outcome: D.Outcome):
        """The whole outcome, its reason included."""
        return outcome if outcome[0] == "error" else json.loads(outcome[1])


RULES: Counter[str] = Counter()


def reference(case) -> dict:
    a = CC.case(case)
    RULES[f"{a['verdict']} {a['reason'].split(':')[0]}" if a["reason"] else a["verdict"]] += 1
    return a


PORT = ExactPort("pouw-check", reference, inputs=INPUTS, package=LEAN, of_record=V.RECORD_VERDICT)
server = D.server_fixture(PORT)


@pytest.mark.slow
def test_pouw_check(server):
    """Generated cases, then each mutation over a few seeds (`replay`), so every verdict is reached whatever the draw: the
    seeded draw moves whenever the cases' modules change, and a run of it has drawn no `answer-undecided`."""
    RULES.clear()
    stats = D.run(PORT, examples=EXAMPLES, server=server)
    swept = Counter(D.replay(PORT, cases.instance(random.Random(f"{name}/{seed}"), name), server)[0]
                    for name in ["honest", *cases.NAMES] for seed in range(3))
    print(f"pouw-check difftest: {stats}, and {sum(swept.values())} swept ({dict(swept)}); by rule: "
          f"{dict(sorted(RULES.items()))}")
    assert {"accept", "reject", "undecided"} <= set(swept), dict(swept)


def _template(why: str) -> str:
    """A reason with each number written #."""
    return re.sub(r"-?\d+", "#", why.replace("SHA-512", "SHA512"))


#: `core_check.opening`'s reasons
REASONS = {_template(r) for r in (
    "unit # is not drawn", "the registered window domain is not the window's served domain",
    "the registered window root is not a SHA-512 digest", "the opening names no entry",
    "the opening's salt is not # bytes, or a path or a side's index is not one (paths of SHA-512 digests)",
    "entry # is not the registered window's entry", "entry #: index # is not in the entry's range",
    "the proof's public file has no JSON header", "the proof's header has no instance count",
    "the proof has # instances, not one tile",
    "the proof's circuit is not the one the verifier pins for the check's template",
    "the proof's first rows are not an A row and a B row of the registered schemas",
    "the proof's body holds no A row and B row",
    *(f"the proof's header does not list its rows {', '.join(p)} first, in its circuit's port order"
      for p in (["a0", "b0"], [*(f"a{k}" for k in range(4)), *(f"b{k}" for k in range(8))])),
    *(f"side {s}: the opening is of row #, not the unit's row #" for s in "ab"),
    *(f"side {s}: the proof's committed row is not entry #'s row #" for s in "ab"),
    "the window is certified, and its template's unit reads no salt: no drawn proof of it is held to the window's "
    "certified salt", "the window is certified, and the verifier holds no 64-byte certified salt for it",
    "the window is not certified, so it has no certified salt for a unit to read",
    "the window's template's unit reads a salt, which only a certified window has",
    f"the proof's rows {', '.join([*(f'a{k}' for k in range(4)), *(f'b{k}' for k in range(8)), 'pub'])} are not "
    f"{CC.ROW_V2} rows", "the proof's body holds no A rows, B rows and public row",
    "the proof's public row is not the window's certified salt, unit #'s position and entry #'s side roots",
)}
OPENED: Counter[str] = Counter()


def opening(case: dict) -> str | None:
    why = CC.opening(CC.params(case["params"]), CC.commitment(bytes.fromhex(case["commitment"])), case["units"],
                     case["unit"], bytes.fromhex(case["pub"]), case["opening"],
                     None if case.get("salt") is None else bytes.fromhex(case["salt"]))
    OPENED[_template(why) if why else "accept"] += 1
    return why


OPENING_NAMES = ["honest"] * (len(openings.NAMES) // 6) + openings.NAMES
OPENING = D.Port("pouw-check-difftest", "opening", opening, package=LEAN, rejects=(CC.Refused,),
                 of_record=V.RECORD_VERDICT,
                 inputs=st.builds(openings.instance, st.randoms(use_true_random=False), st.sampled_from(OPENING_NAMES)))
opening_server = D.server_fixture(OPENING)


def test_opening(opening_server):
    """Generated openings, then each mutation over a few seeds (`replay`), so every reason is reached whatever the draw."""
    OPENED.clear()
    stats = D.run(OPENING, examples=2000, server=opening_server)
    swept = Counter(D.replay(OPENING, openings.instance(random.Random(f"{name}/{seed}"), name), opening_server)[0]
                    for name in ["honest", *openings.NAMES] for seed in range(3))
    print(f"pouw-check-difftest opening: {stats}, and {sum(swept.values())} swept ({dict(swept)}); by reason: "
          f"{dict(sorted(OPENED.items()))}")
    assert set(OPENED) == REASONS | {"accept"}, set(OPENED) ^ (REASONS | {"accept"})


RULED: Counter[str] = Counter()


def registered_input(case) -> dict:
    """A `serve` case's params and registration, as `registered` reads them."""
    return {k: case[k] for k in ("params", "commitment") if k in case} if isinstance(case, dict) else {}


def registered(x: dict) -> dict:
    if not ("params" in x and isinstance(x.get("commitment"), str)):
        raise CC.Refused("the input's params or commitment is malformed")
    why = CC.registered(CC.params(x["params"]), CC.commitment(bytes.fromhex(x["commitment"])))
    RULED[why.split(":")[0] if why else "holds"] += 1
    return {"reason": why}


REGISTERED = D.Port("pouw-check-difftest", "registered", registered, package=LEAN, rejects=(ValueError,),
                    of_record=V.RECORD_VERDICT,
                    inputs=st.builds(lambda r, n: registered_input(cases.instance(r, n)), st.randoms(use_true_random=False),
                                     st.sampled_from([n for n in NAMES if not n.startswith("answer")])))
registered_server = D.server_fixture(REGISTERED)


def test_registered(registered_server):
    """PoUW's rule over generated registrations, each rule's mutation swept over a few seeds: the same reason."""
    RULED.clear()
    stats = D.run(REGISTERED, examples=1500, server=registered_server)
    swept = Counter(D.replay(REGISTERED, registered_input(cases.instance(random.Random(f"{name}/{seed}"), name)),
                             registered_server)[0]
                    for name in NAMES if not name.startswith("answer") for seed in range(3))
    print(f"pouw-check-difftest registered: {stats}, and {sum(swept.values())} swept ({dict(swept)}); by rule: "
          f"{dict(sorted(RULED.items()))}")
    assert set(RULED) >= {"holds", "window", "public-items", "level", "public-value", "population", "law", "query",
                          "scheme"}, set(RULED)


PASSED: Counter[str] = Counter()


def pass_epoch(case: dict) -> dict:
    how, refused = CC.pass_epoch(*passes.pass_args(case))
    PASSED[how or f"{refused[0]}: {_template(refused[1])}"] += 1
    return {"how": how} if refused is None else {"refused": list(refused)}


def draw(case: dict) -> dict | None:
    refused = CC.draw(*passes.draw_args(case))
    PASSED["draw " + (f"{refused[0]}: {_template(refused[1])}" if refused else "accept")] += 1
    return None if refused is None else {"refused": list(refused)}


PASS = D.Port("pouw-check-difftest", "pass", pass_epoch, package=LEAN, need_rejects=False, of_record=V.RECORD_VERDICT,
              inputs=st.builds(passes.pass_instance, st.randoms(use_true_random=False),
                               st.sampled_from(["honest"] * (len(passes.PASS_NAMES) // 6) + passes.PASS_NAMES)))
DRAW = D.Port("pouw-check-difftest", "draw", draw, package=LEAN, need_rejects=False, of_record=V.RECORD_VERDICT,
              inputs=st.builds(passes.draw_instance, st.randoms(use_true_random=False),
                               st.sampled_from(["honest"] * (len(passes.DRAW_NAMES) // 4) + passes.DRAW_NAMES)))
pass_server = D.server_fixture(PASS)


def test_pass_and_draw(pass_server):
    """Generated passes (a certified window's among them) and draws, then each mutation over a few seeds (`replay`), so
    every refusal is reached whatever the draw."""
    PASSED.clear()
    stats = {"pass": D.run(PASS, examples=1500, server=pass_server), "draw": D.run(DRAW, examples=1000, server=pass_server)}
    swept = Counter(D.replay(PASS, passes.pass_instance(random.Random(f"{name}/{seed}"), name), pass_server)[0]
                    for name in ["honest", *passes.PASS_NAMES] for seed in range(3))
    swept += Counter(D.replay(DRAW, passes.draw_instance(random.Random(f"{name}/{seed}"), name), pass_server)[0]
                     for name in ["honest", *passes.DRAW_NAMES] for seed in range(3))
    print(f"pouw-check-difftest pass: {stats['pass']}; draw: {stats['draw']}; and {sum(swept.values())} swept "
          f"({dict(swept)}); by outcome: {dict(sorted(PASSED.items()))}")
    assert {k for k in PASSED if not k.startswith("draw ")} >= {"coin", "recorded beacon", "certifier"}
    assert "draw accept" in PASSED and len(PASSED) > 40
    for seed in range(4):   # `COIN_EARLY`'s draws: accepted only as a replay of a recorded run, by both sides
        early, _ = passes.honest_draw(random.Random(f"early/{seed}"), CC.COIN_EARLY, early=True)
        live = {**early, "recorded": False}
        assert (draw(early), draw(live)["refused"][0]) == (None, "coin")
        D.replay(DRAW, early, pass_server), D.replay(DRAW, live, pass_server)


CREDITED: Counter[str] = Counter()


def _counted(refused: tuple[str, str] | None, what: str) -> None:
    CREDITED[f"{what} " + (f"{refused[0]}: {_template(refused[1])}" if refused else "accept")] += 1


def credit(case: dict) -> dict:
    got, refused = Cr.credited(*credit_cases.credit_args(case))
    _counted(refused, "credit")
    return {"credited": got} if refused is None else {"refused": list(refused)}


def outcome(case: dict) -> dict | None:
    refused = Cr.outcome(case["drawn"], case["proofs"])
    _counted(refused, "outcome")
    return None if refused is None else {"refused": list(refused)}


def a_link(case: dict) -> dict | None:
    refused = Cr.a_link(case["rounds"], case["entries"])
    _counted(refused, "a-link")
    return None if refused is None else {"refused": list(refused)}


def _port(name: str, fn, instance, names: list[str]) -> D.Port:
    return D.Port("pouw-check-difftest", name, fn, package=LEAN, need_rejects=False, of_record=V.RECORD_VERDICT,
                  inputs=st.builds(instance, st.randoms(use_true_random=False),
                                   st.sampled_from(["honest"] * (len(names) // 4) + names)))


CREDIT = _port("credit", credit, credit_cases.credit_instance, credit_cases.CREDIT_NAMES)
OUTCOME = _port("outcome", outcome, credit_cases.outcome_instance, credit_cases.OUTCOME_NAMES)
A_LINK = _port("a-link", a_link, credit_cases.a_link_instance, credit_cases.A_LINK_NAMES)


def test_credit(pass_server):
    """Generated served windows' credited sets, drawn units' outcomes and A rows, then each mutation over a few seeds
    (`replay`): a credit bit flipped after the draw coin, a drawn unit failed or missing, an A tree the round roots
    didn't commit, and every other refusal of `credit.credited`, `credit.outcome` and `credit.a_link`."""
    CREDITED.clear()
    stats = {p.fn: D.run(p, examples=n, server=pass_server) for p, n in ((CREDIT, 300), (OUTCOME, 400), (A_LINK, 400))}
    swept = Counter()
    for port, instance, names in ((CREDIT, credit_cases.credit_instance, credit_cases.CREDIT_NAMES),
                                  (OUTCOME, credit_cases.outcome_instance, credit_cases.OUTCOME_NAMES),
                                  (A_LINK, credit_cases.a_link_instance, credit_cases.A_LINK_NAMES)):
        swept += Counter(D.replay(port, instance(random.Random(f"{name}/{seed}"), name), pass_server)[0]
                         for name in ["honest", *names] for seed in range(3))
    print(f"pouw-check-difftest credit: {stats}; and {sum(swept.values())} swept ({dict(swept)}); by outcome: "
          f"{dict(sorted(CREDITED.items()))}")
    assert {"credit accept", "outcome accept", "a-link accept"} <= set(CREDITED)
    for part in ("credit draw: the draw root over the manifest", "credit draw: the window's draw root is not",
                 "credit draw: the challenger's session closed with no draw coin", "credit credit: manifest row # has no",
                 "credit credit: call # is credited with no z", "credit coin: the draw coin: ",
                 "credit coin: the challenger's session is not trusted", "outcome missing: ", "outcome failed: ",
                 "outcome outcome: unit # has two", "outcome outcome: unit # has a proof outcome and was not drawn",
                 "a-link a-link: registered A value h#/w#/a/#'s root is not the tree",
                 "a-link a-link: round #'s root is not over", "don't tile its rows",
                 "a-link a-link: the window registers no A value", "which is no registered A value"):
        assert any(part in k for k in CREDITED), part


ROUNDED: Counter[str] = Counter()


def rounds(case: dict) -> dict | None:
    refused = CC.rounds(*rounds_cases.rounds_args(case))
    ROUNDED[f"{refused[0]}: {_template(refused[1])}" if refused else "accept"] += 1
    return None if refused is None else {"refused": list(refused)}


ROUNDS = D.Port("pouw-check-difftest", "rounds", rounds, package=LEAN, need_rejects=False, of_record=V.RECORD_VERDICT,
                inputs=st.builds(rounds_cases.rounds_instance, st.randoms(use_true_random=False),
                                 st.sampled_from(["honest"] * (len(rounds_cases.ROUNDS_NAMES) // 4)
                                                 + rounds_cases.ROUNDS_NAMES)))


def test_rounds(pass_server):
    """Generated live windows' round coins, then each mutation over a few seeds (`replay`): a coin opened twice, one opened
    before its round's root was recorded, a foreign epoch's, and every other refusal of `core_check.rounds`."""
    ROUNDED.clear()
    stats = D.run(ROUNDS, examples=600, server=pass_server)
    swept = Counter(D.replay(ROUNDS, rounds_cases.rounds_instance(random.Random(f"{name}/{seed}"), name), pass_server)[0]
                    for name in ["honest", *rounds_cases.ROUNDS_NAMES] for seed in range(3))
    print(f"pouw-check-difftest rounds: {stats}; and {sum(swept.values())} swept ({dict(swept)}); by outcome: "
          f"{dict(sorted(ROUNDED.items()))}")
    assert "accept" in ROUNDED
    for part in ("is opened twice", "was opened before its root was recorded", "was never opened", ": coin is epoch #",
                 ": coin does not open", ": coin was not opened after", "the verifier's challenger key is malformed",
                 "the round roots are not", "the round coins are not a list", ": a live coin opening is",
                 "the coin commitment's ", "not trusted: no challenger close record",
                 "not trusted: the challenger record's signature does not verify",
                 "not trusted: the challenger's release log is not the one its close record signs",
                 ": a second open for root ", ", not the window's pinned key",
                 "not trusted: the auditor's ledger has no end for session",
                 "not trusted: the challenger's release log is not the one the auditor's ledger sealed",
                 "under verity/challenger-auditor/v#, which names no prover", "'s end of session s# is malformed: ",
                 "not trusted: the challenger's record is signed at leaf ",
                 "not trusted: the challenger's refused hellos are not the ",
                 "the challenger's log shows it received"):
        assert any(part in k for k in ROUNDED), part
