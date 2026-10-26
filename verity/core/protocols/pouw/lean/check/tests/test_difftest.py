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
`pouw_pass_cases` generates (honest passes of each coin version, a recorded beacon's, a certified window's and a live
Pearl-C pass's, and honest draws, then one mutation each aimed at every refusal), compared output for output: how the
salt was had, or the same refusal.

Its `window` and `period` against `credit.window` and `credit.period_credit`, the only entries, and its `credit`,
`drawn`, `window-drawn`, `outcome`, `a-link` and `weights` against the steps they run or will, `credit._credited`,
`credit.drawn`, `credit._window_drawn` (at the strata of the descriptor's layout, `Credit.descriptorDrawn`, the draw the
J10-R period's statement reads), `credit._outcome`, `credit._a_link` and `credit._weights` (`PouwCheck.Credit.Internal`'s):
one `Port` each, over the cases
`pouw_credit_cases` generates (a served window whose draw coin opened after its draw root over the manifest's credit bits,
the units a draw coin draws from a window's layout and each of them proved, a served window whole with its layout read
from its circuit's descriptor (`credit.layout_of`) and its round log held to the challenger's (`core_check.rounds`) and
to its A trees, a period of the
auditor's ledger and its windows, a window's A trees under its round roots, a window's registered weights under its
open root, then one mutation each aimed at every refusal), compared output for output: the credited calls, the drawn
units, the weights' root or null, or the same refusal.

Its `rounds` against `core_check.rounds`: one `Port`, over the live windows `pouw_rounds_cases` generates (a live epoch's
round coins, each opened after its recorded round root, read through the challenger's signed session the auditor's signed
ledger opens and ends for the window, then one mutation each aimed at every refusal), compared output for output: null, or the same refusal.
Its `salt` against `core_check.salt` likewise, over the live windows' salt coins the same module generates, read through
the same session, reserved before it opened (opened after the session's open root, `salt.window_open_root`). Its
`live-salt` against `core_check.live_salt` likewise, over the live Pearl-C passes `pouw_pass_cases` generates (a
salt-only session the auditor's ledger reserves, opens on the pass's salt epoch's root and ends complete, its commitment
read from the session's own files, a drawn `pass_index` and call indices offset within the epoch), compared output for
output: the salt, or the same refusal.

Its `chunks` against `weights.chunks` (an H1 lane's chunks along K, the chunk rows a table at K commits a row as): every
multiple of 32 below 2^17, then drawn integers and non-integers, compared output for output: the count, or the same refusal.
"""
from __future__ import annotations

import hashlib
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
from verity.core.protocols.pouw import weights as W

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
    assert {k for k in PASSED if not k.startswith("draw ")} >= {"coin", "recorded beacon", "certifier", "live coin"}
    for part in (CC.LIVE_UNHELD, CC.LIVE_FORM, CC.LIVE_NOT_SIGNED, CC.LIVE_SESSION, CC.INCOMPLETE, CC.PASS_FORM,
                 "the pass's salt coin opens against the epoch's published coin commitment",
                 "the coin commitment's coins are not a Pearl-C salt session's",
                 "the auditor's ledger opens no challenger session for this window"):
        assert any(_template(part) in k for k in PASSED), part
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
    got, refused = Cr._credited(*credit_cases.credit_args(case))
    _counted(refused, "credit")
    return {"credited": got} if refused is None else {"refused": list(refused)}


def drawn(case: dict) -> dict:
    got, refused = Cr.drawn(*(case[k] for k in credit_cases.DRAWN_KEYS))
    _counted(refused, "drawn")
    return {"drawn": got} if refused is None else {"refused": list(refused)}


def window_drawn(case: dict) -> dict:
    """`credit._window_drawn` over the strata of the descriptor's layout, as `Credit.descriptorDrawn` reads them."""
    got, refused = None, ("layout", Cr.DESCRIPTOR_HEX)
    if type(case["descriptor"]) is str and Cr._HEX.fullmatch(case["descriptor"]):
        lay, refused = Cr.layout_of(bytes.fromhex(case["descriptor"]))
        ss = None
        if refused is None:
            ss, refused = Cr.strata(lay)
        if refused is None:
            got, refused = Cr._window_drawn(case["window"], case["draw"], ss, case["k"])
    _counted(refused, "window-drawn")
    return {"drawn": got} if refused is None else {"refused": list(refused)}


def outcome(case: dict) -> dict | None:
    refused = Cr._outcome(*(case[k] for k in credit_cases.OUTCOME_KEYS))
    _counted(refused, "outcome")
    return None if refused is None else {"refused": list(refused)}


def window(case: dict) -> dict:
    got, refused = Cr._window_outcomes(*credit_cases.window_args(case))
    _counted(refused, "window")
    return {"credited": got} if refused is None else {"refused": list(refused)}


def period(case: dict) -> dict:
    got, refused = Cr._period_outcomes(*(case[k] for k in credit_cases.PERIOD_KEYS), rounds_cases.lms_verify)
    _counted(refused, "period")
    return {"credited": got} if refused is None else {"refused": list(refused)}


def a_link(case: dict) -> dict | None:
    refused = Cr._a_link(case["rounds"], case["entries"], case["calls"], case.get("rows"))
    _counted(refused, "a-link")
    return None if refused is None else {"refused": list(refused)}


def weights(case: dict) -> dict:
    got, refused = Cr._weights(*(case[k] for k in credit_cases.WEIGHTS_KEYS))
    _counted(refused, "weights")
    return {"weights": got.hex()} if refused is None else {"refused": list(refused)}


def registered_file(case: dict) -> dict:
    raw = bytes.fromhex(case["descriptor"])
    lay, refused = Cr.layout_of(raw)
    if refused is None:
        program, shape = hashlib.sha512(raw).hexdigest(), Cr.shape_of(raw)
        got, refused = Cr.registered_file(program, lay, shape, case["call"], case["roots"], case["window"],
                                          case["steps"], case["salt"])
        if refused is None and all(t["j10r"] for t in shape["tiles"].values()):
            CREDITED["registered-file accept j10r"] += 1
            if any(t["n"] > 1 for t in shape["tiles"].values()):
                CREDITED["registered-file accept j10r chunked"] += 1
    _counted(refused, "registered-file")
    return {"refused": list(refused)} if refused is not None else {"file": got,
                                                                   "partition": Cr.partition_of(program, shape)}


def _port(name: str, fn, instance, names: list[str]) -> D.Port:
    return D.Port("pouw-check-difftest", name, fn, package=LEAN, need_rejects=False, of_record=V.RECORD_VERDICT,
                  inputs=st.builds(instance, st.randoms(use_true_random=False),
                                   st.sampled_from(["honest"] * (len(names) // 4) + names)))


CREDIT = _port("credit", credit, credit_cases.credit_instance, credit_cases.CREDIT_NAMES)
DRAWN = _port("drawn", drawn, credit_cases.drawn_instance, credit_cases.DRAWN_NAMES)
WINDOW_DRAWN = _port("window-drawn", window_drawn, credit_cases.window_drawn_instance, credit_cases.WINDOW_DRAWN_NAMES)
OUTCOME = _port("outcome", outcome, credit_cases.outcome_instance, credit_cases.OUTCOME_NAMES)
WINDOW = _port("window", window, credit_cases.window_instance, credit_cases.WINDOW_NAMES)
PERIOD = _port("period", period, credit_cases.period_instance, credit_cases.PERIOD_NAMES)
A_LINK = _port("a-link", a_link, credit_cases.a_link_instance, credit_cases.A_LINK_NAMES)
WEIGHTS = _port("weights", weights, credit_cases.weights_instance, credit_cases.WEIGHTS_NAMES)
REGISTERED_FILE = _port("registered-file", registered_file, credit_cases.registered_file_instance,
                        credit_cases.REGISTERED_FILE_NAMES)


def test_credit(pass_server):
    """Generated served windows' credited sets, drawn units and their outcomes, whole windows and periods, and A rows,
    then each mutation over a few seeds (`replay`): a window's tile at a K its proof doesn't cover (2^17 or past, or no
    multiple of 32), a credit bit flipped after the draw coin, units the prover chose, a drawn unit failed or missing,
    a period unclosed or a window of it missing or out of order, an A tree the round roots
    didn't commit, a window's round log the challenger never received, entries or a drawn unit's proof of another
    registration than the window's or an entry whose root is not its sessions', B rows other than the ones a window's
    open root binds or registered under another domain, name or rows, a window whose B rows (one row's s_w, or every
    code zeroed) are not the ones its period's open signed, a period whose open signs no weights' root or names no
    model `models.json` pins (none, malformed, another checkpoint, another role), a round's A table of a K (a row off,
    its calls' rows offset by one call, a call missing) or a round's input-ids table other than the served Commit's or a
    Commit record the draw root doesn't bind, round 0's input-ids table outside round 0's request table or bound to
    another session, round 0's request table at another root than request_table, twice, under another PREFIX or bound
    to another session or to the open root, a call's served rows m_s past the period's T table rows, other than its step's or malformed (an
    honest window's empty step credits none of its calls), a T whose slot is not the window's M or that is no positive
    integer, a period whose open commits no T or a malformed one, a window's live evidence whose live Program is not
    at its T, S and PREFIX or whose round's live@<s> outcome is missing, rejected, twice, of another circuit or opened
    another m@<s> than the manifest's m_s, another round@<s> than its round, other slot or request tables or other
    ok rows, and every other
    refusal of `credit.window`, `credit.period_credit` and the steps they run."""
    CREDITED.clear()
    stats = {p.fn: D.run(p, examples=n, server=pass_server)
             for p, n in ((CREDIT, 300), (DRAWN, 300), (WINDOW_DRAWN, 100), (OUTCOME, 300), (WINDOW, 150), (PERIOD, 100), (A_LINK, 400),
                             (WEIGHTS, 300), (REGISTERED_FILE, 150))}
    swept = Counter()
    for port, instance, names in ((CREDIT, credit_cases.credit_instance, credit_cases.CREDIT_NAMES),
                                  (REGISTERED_FILE, credit_cases.registered_file_instance,
                                   credit_cases.REGISTERED_FILE_NAMES),
                                  (DRAWN, credit_cases.drawn_instance, credit_cases.DRAWN_NAMES),
                                  (WINDOW_DRAWN, credit_cases.window_drawn_instance, credit_cases.WINDOW_DRAWN_NAMES),
                                  (OUTCOME, credit_cases.outcome_instance, credit_cases.OUTCOME_NAMES),
                                  (WINDOW, credit_cases.window_instance, credit_cases.WINDOW_NAMES),
                                  (PERIOD, credit_cases.period_instance, credit_cases.PERIOD_NAMES),
                                  (A_LINK, credit_cases.a_link_instance, credit_cases.A_LINK_NAMES),
                                  (WEIGHTS, credit_cases.weights_instance, credit_cases.WEIGHTS_NAMES)):
        swept += Counter(D.replay(port, instance(random.Random(f"{name}/{seed}"), name), pass_server)[0]
                         for name in ["honest", *names] for seed in range(3))
    print(f"pouw-check-difftest credit: {stats}; and {sum(swept.values())} swept ({dict(swept)}); by outcome: "
          f"{dict(sorted(CREDITED.items()))}")
    assert {"credit accept", "drawn accept", "window-drawn accept", "outcome accept", "window accept", "period accept",
            "a-link accept", "weights accept", "registered-file accept", "registered-file accept j10r",
            "registered-file accept j10r chunked"} <= set(CREDITED)
    for part in ("credit draw: the draw root over the manifest", "credit draw: the window's draw root is not",
                 "credit draw: the challenger's session closed with no draw coin", "credit credit: manifest row # has no",
                 "credit credit: call # is credited with no z", "credit coin: the draw coin: ",
                 "credit coin: the challenger's session is not trusted", "outcome missing: ", "outcome failed: ",
                 "outcome outcome: unit # has two", "outcome outcome: unit # has a proof outcome and was not drawn",
                 "drawn layout: linear ", "drawn layout: the A slot #", "drawn layout: the A blocks of K = #",
                 "drawn layout: a call of lm_head", "drawn draw: the draw coin is not", "drawn draw: the draw's stream",
                 "window tile-k: the window's tile at K = # is outside the K its proof covers, multiples of # below #",
                 "so the period earns nothing: tile-k: the window's tile at K = #",
                 "pad its linears' A and B rows with zero codes to K = #",
                 "split its linears' k range into linears of at most K = # each",
                 "window manifest: the manifest's # rows", "window outcome: unit # has a proof outcome and was not",
                 "window manifest: manifest row # states no served rows m_s at most the period's T = # table rows",
                 "window layout: the period's T = # table rows give an A slot of # rows, not the window's M = #",
                 "window layout: the period's table rows T are not a positive integer",
                 "period layout: period ", "'s open commits no table rows T (rows, a positive integer",
                 "so the period earns nothing: layout: the period's T = # table rows give an A slot",
                 "window manifest: call # states # served rows and call # #, both of step #",
                 "period period: period ", "period missing: window #", "period order: window #",
                 "period period: the windows presented run past", "period period: window ", "period window: window #",
                 "period coin: the auditor's ledger is not trusted", "period circuit: window # of period ",
                 "drawn budget: the draw's K is not",
                 "drawn budget: the window's # units are fewer than K", "period budget: period ",
                 "the one period", "commits no cap or circuit", "the window's descriptor is not the bytes",
                 "window layout: the window's circuit's units are not its record's tiles",
                 "window layout: the window's circuit is not an H# window",
                 "window layout: the window's template at K = # is not h#window's",
                 "window layout: the window's descriptor is not a verity-ir/descriptor/v#",
                 "window layout: the window's TILES are at K = ", "window layout: the window's layout is not",
                 "window layout: a call of lm_head", "so the period earns nothing: layout: ",
                 "a-link a-link: registered A value h#/w#/a/#'s root is not the tree",
                 "a-link a-link: round #'s root is not over", "don't tile its rows",
                 "a-link a-link: the window registers no A value", "which is no registered A value",
                 "weights weights: the window's registered weights are not the ones its open root binds",
                 "weights weights: the window registers no weights", "weights weights: the window registers K = #",
                 "'s entry is not the registration of K = #'s B rows", "b/<K> of a K of its circuit",
                 "weights weights: the window's circuit is not its descriptor's SHA#",
                 "window rounds: the round roots are not the roots the challenger's log shows it received",
                 "window coin: pouw/round coin #: coin was not opened after", "window a-link: registered A value ",
                 "window a-link: the window registers no A value",
                 "so the period earns nothing: rounds: the round roots are not",
                 "the drawn units' proof outcomes are not a list of {program, registered, unit, verdict}",
                 "domain is not the one the window's registration gives it (the registered domain of h#/w#/a/#",
                 "window registration: drawn unit #'s proof opened port ",
                 "against another root than the one the window registered for",
                 "window registration: drawn unit #'s proof was verified against another circuit",
                 "opened no A value …/a/# or …/a/#@# (its call's round's) of the window", "'s proof opened no weights ",
                 "which the window registers not",
                 "which the window registers twice", "'s entry is not one-stage's {value, root, leaves, words, domain}",
                 "so the period earns nothing: registration: drawn unit #",
                 "window weights: the window's registered weights' root ",
                 "open signed: a B row's codes or s_w differ from the ones the period registered",
                 "'s open signs no weights' root (weights_root, a SHA# digest",
                 "'s open signs a weights' root but no model (model, {name, checkpoint_sha#})",
                 "with another checkpoint than the one credit pins for it: open the period with the gate's --model",
                 "which is no checkpoint role credit pins: open the period with the gate's --model",
                 "window weights: period ",
                 "window weights: the window is in no period of the auditor's ledger",
                 "window weights: the window's round# request table's root is not",
                 "window weights: the window's registered weights are not the ones its open root binds",
                 "window weights: registered weights ", "period weights: period ",
                 "so the period earns nothing: weights: the window's registered weights' root",
                 "registered-file registration: call #'s statement reads ", "which the window registers not: register",
                 "which the window registers twice: register",
                 "credit commit: the window's Commit record is not the bytes in lowercase hex",
                 "window commit: the window's Commit record is not the bytes in lowercase hex",
                 "window draw: the window's draw root is not over its manifest (each call's credit bit among it), its "
                 "Commit record", "window arow: round #'s A table of K = # ", "is not the Commit's row-table/#/k#",
                 "the window's layout gives it (credit.table_rows", "(the verifier hashes its pads)",
                 "window arow: round #'s input-ids table ", "is not the Commit's input-ids/#",
                 "window arow: the window registers no round #'s input-ids table",
                 "window arow: the Commit record registers row-table/#/k#",
                 "which only round # commits",
                 "window arow: round #'s request root (requests, the one the window's open root binds) is not",
                 "window registration: registered value ", "ids@#'s domain is not the one the window's registration",
                 "requests@#'s domain is not the one the window's registration",
                 "window arow: the window's request table's root is not round #'s request_table",
                 "window arow: the window registers request table x/",
                 "or receipt table of the window", "'s receipt table, its round #'s, which only round # commits",
                 "a-link a-link: registered receipt table ", "registered-file registration: the window registers ",
                 "'s template returns no receipt of whole #-bit words", "'s seed reads the roots of ",
                 "its units' ok rows, which are the verifier's own",
                 "registered-file layout: the window's template at K = # is not h#window's",
                 "window live: the window registers h#/w#/slots@# not", "window live: the window registers h#/w#/requests@# not",
                 "rows of # #-bit words its live unit reads",
                 "window live: the window's live evidence is not {descriptor, proofs}",
                 "window live: the window's live descriptor is not a verity-ir/descriptor/v#",
                 "window live: the window's live circuit is not a live Program",
                 "window live: the window's live circuit's units are not its rounds",
                 "window live: the window's live Program is at T = #", "rounds, not the window's #: every round's",
                 "window live: the window's live Program's PREFIX is not its circuit's",
                 "has no proof outcome: every round's served rows m_s is proved",
                 "has two proof outcomes: each round's is proved and verified once",
                 "window live: live unit # has a proof outcome and the window has # rounds",
                 "proof was rejected: its served rows m_s = # are not proved",
                 "proof was verified against another circuit than the window's live Program",
                 "proof opened other ports, values or roots than its statement reads",
                 "window live: the window's live proof outcomes are not a list",
                 "so the period earns nothing: live: round #"):
        assert any(part in k for k in CREDITED), part
    assert any(k.startswith("window a-link: ") and "the window's layout gives it" in k for k in CREDITED)


def test_honest_j10r_windows_are_credited(pass_server):
    """The compiled `Credit.window` credits honest J10-R windows (`window_instance`'s ``j10r``: J10-R units, two row tiles
    a call), as the Python verifier does: the code fact `Proofs.Pouw.PeriodJ10R.Honest`'s `j10r_meets` takes as `hacc`.
    `test_credit` replays these cases but holds only the two sides to each other."""
    for seed in range(3):
        kind, out = D.replay(WINDOW, credit_cases.window_instance(random.Random(f"j10r/{seed}"), "j10r"), pass_server)
        assert kind == "output" and json.loads(out).get("credited"), out


def leaf_words(case: dict) -> dict:
    try:
        return {"leaf_words": W.leaf_words(case["k"])}
    except ValueError:
        return {"leaf_words": None}


LEAF_WORDS = D.Port("pouw-check-difftest", "leaf-words", leaf_words, package=LEAN, need_rejects=False,
                    of_record=V.RECORD_VERDICT, inputs=st.builds(lambda k: {"k": k}, st.integers(-2, 1 << 20)))


def test_leaf_words(pass_server):
    """`weights.leaf_words` (a leaf's words at K: a chunk's codes, then the row's scale) against `Credit.leafWords` at every
    K to 4096 + 64, every multiple of 32 below 2^17 (the models' K, the ones with no chunks among them) and drawn K to
    2^20."""
    stats = D.run(LEAF_WORDS, examples=500, server=pass_server)
    got = Counter(D.replay(LEAF_WORDS, {"k": k}, pass_server)[1] == '{"leaf_words":null}'
                  for k in sorted({*range(-1, 4096 + 65), *range(32, 1 << 17, 32)}))
    print(f"pouw-check-difftest leaf-words: {stats}; swept {dict(got)} (True: no chunks)")
    assert got[True] and got[False]


def chunks(k) -> dict:
    if type(k) is not int or k < 0:
        raise D.Reject("the input is not a natural number K")
    try:
        return {"chunks": W.chunks(k)}
    except ValueError as e:
        return {"refused": str(e)}


CHUNKS = D.Port("pouw-check-difftest", "chunks", chunks, package=LEAN, of_record=V.RECORD_VERDICT,
                inputs=st.one_of(st.integers(-64, 1 << 18), st.booleans(), st.text(max_size=2), st.none()))


def test_chunks(pass_server):
    """`PouwCheck.Chunks.chunks` is `weights.chunks` at every multiple of 32 below 2^17 (every served K, the whole lane up to
    4096 and up to 256 chunks past it, each refusal's message included), then on drawn integers, negative ones a reject,
    and non-integers."""
    outs = [json.loads(D.replay(CHUNKS, k, pass_server)[1]) for k in range(0, 1 << 17, 32)]
    stats = D.run(CHUNKS, examples=500, server=pass_server)
    print(f"pouw-check-difftest chunks: 4096 Ks swept, then {stats}")
    assert Counter(next(iter(o)) for o in outs) == {"chunks": 375, "refused": 3721}
    assert [outs[k // 32] for k in (4096, 6144, 12288, 14336, 16384)] == [{"chunks": n} for n in (1, 2, 3, 4, 4)]


ROUNDED: Counter[str] = Counter()
#: the refusals of the session's pin, `rounds`' and `salt`'s alike
PIN_PARTS = ("which the host's pins log doesn't record as a commit of main's history",
             "which the host's pins log records but which is not on main's first-parent line",
             "seals no pins log (pins: {lines, sha#})", "not trusted: no pins log: ",
             "not trusted: the verifier holds no copy of main's history",
             "not trusted: the verifier's copy of main's history is malformed")


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
                 "the challenger's log shows it received", *PIN_PARTS):
        assert any(part in k for k in ROUNDED), part


SALTED: Counter[str] = Counter()


def salt(case: dict) -> dict | None:
    refused = CC.salt(*rounds_cases.salt_args(case))
    SALTED[f"{refused[0]}: {_template(refused[1])}" if refused else "accept"] += 1
    return None if refused is None else {"refused": list(refused)}


SALT = D.Port("pouw-check-difftest", "salt", salt, package=LEAN, need_rejects=False, of_record=V.RECORD_VERDICT,
              inputs=st.builds(rounds_cases.salt_instance, st.randoms(use_true_random=False),
                               st.sampled_from(["honest"] * (len(rounds_cases.SALT_NAMES) // 4) + rounds_cases.SALT_NAMES)))


def test_salt(pass_server):
    """Generated live windows' salt coins, read through the challenger's signed session the auditor's signed ledger
    reserves, opens and ends for the window, then each mutation over a few seeds (`replay`): a session opened without a
    reserve, the open root under another session's id or of another window's weights or requests, H1's salt opened
    after the weights' root, a log showing another root, and every other refusal of `core_check.salt`."""
    SALTED.clear()
    stats = D.run(SALT, examples=400, server=pass_server)
    swept = Counter(D.replay(SALT, rounds_cases.salt_instance(random.Random(f"{name}/{seed}"), name), pass_server)[0]
                    for name in ["honest", *rounds_cases.SALT_NAMES] for seed in range(3))
    print(f"pouw-check-difftest salt: {stats}; and {sum(swept.values())} swept ({dict(swept)}); by outcome: "
          f"{dict(sorted(SALTED.items()))}")
    assert "accept" in SALTED
    for part in (CC.NO_RESERVE, CC.SALT_FORM, CC.OTHER_WINDOW, CC.NOT_RECEIVED,
                 "the verifier's challenger key is malformed",
                 "not trusted: no auditor's ledger", "not trusted: the challenger record's signature does not verify",
                 "the coin commitment's ", "the salt's a live coin opening is", "the salt's coin was not opened after",
                 "not its pouw/salt coin #", "the salt's coin does not open", "the salt's coin is epoch #",
                 "the salt's coin opening is malformed", *PIN_PARTS):
        assert any(_template(part) in k for k in SALTED), part


LIVE_SALTED: Counter[str] = Counter()


def live_salt(case: dict) -> dict:
    salt, refused = CC.live_salt(*passes.live_salt_args(case))
    LIVE_SALTED[f"{refused[0]}: {_template(refused[1])}" if refused else "accept"] += 1
    return {"salt": salt.hex()} if refused is None else {"refused": list(refused)}


LIVE_SALT = D.Port("pouw-check-difftest", "live-salt", live_salt, package=LEAN, need_rejects=False,
                   of_record=V.RECORD_VERDICT,
                   inputs=st.builds(passes.live_salt_instance, st.randoms(use_true_random=False),
                                    st.sampled_from(["honest"] * (len(passes.LIVE_SALT_NAMES) // 4)
                                                    + passes.LIVE_SALT_NAMES)))


def test_live_salt(pass_server):
    """Generated live Pearl-C passes, salted by a salt-only session the auditor's signed ledger reserves, opens on the
    pass's salt epoch's root and ends complete, read through the session's own files, then each mutation over a few
    seeds (`replay`): a `pass_index` that isn't an integer in [0, 2^64), a call index of 2^32 or more, a session opened
    on another root (W itself, another epoch's, or one the manifest doesn't name), #3868's H1 layout, a session reserved for another layout or ended
    incomplete, a log showing another root, coins other than the session's, a session the manifest misnames, and every
    other refusal of `core_check.live_salt`."""
    LIVE_SALTED.clear()
    stats = D.run(LIVE_SALT, examples=300, server=pass_server)
    swept = Counter(D.replay(LIVE_SALT, passes.live_salt_instance(random.Random(f"{name}/{seed}"), name),
                             pass_server)[0]
                    for name in ["honest", *passes.LIVE_SALT_NAMES] for seed in range(3))
    print(f"pouw-check-difftest live-salt: {stats}; and {sum(swept.values())} swept ({dict(swept)}); by outcome: "
          f"{dict(sorted(LIVE_SALTED.items()))}")
    assert "accept" in LIVE_SALTED
    for part in (CC.NO_RESERVE, CC.RESERVE_LAYOUT, CC.INCOMPLETE, CC.LIVE_NOT_RECEIVED, CC.LIVE_NOT_SIGNED, CC.LIVE_SALT,
                 CC.LIVE_SESSION, CC.PASS_FORM, CC.CALL_RANGE, "the pass's manifest is not a JSON object", "the pass's manifest names no schema",
                 "the pass's matmuls are not", "every weight of the workload is committed, and only those",
                 "not trusted: the auditor's ledger opens no challenger session for this window",
                 "the verifier's challenger key is malformed", "not trusted: no auditor's ledger",
                 "not trusted: the challenger record's signature does not verify",
                 "the coin commitment's coins are not a Pearl-C salt session's", "the pass's coins are malformed: ",
                 "the salt's a live coin opening is", "the salt's coin was not opened after", "not its pouw/salt coin #",
                 "the salt's coin does not open", "the salt's coin is epoch #", "the salt's coin opening is malformed",
                 *PIN_PARTS):
        assert any(_template(part) in k for k in LIVE_SALTED), part
