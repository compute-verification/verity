"""`pouw-check serve` against `verity.core.protocols.pouw.core_check`, on the shared harness (`verity_difftest`):
one `ProtocolPort`, over the cases `pouw_check_cases` generates from a Hypothesis-controlled `random.Random` (honest
windows, and one mutation each aimed at every refusal, the case's framing included), so a run is deterministic and
shrinks.

The harness compares an outcome's verdict, commitment, public outputs and profile. The reference writes the port's
reasons too, `Core.Protocol.case`'s framing included, so this port compares the whole outcome: the reason names the rule
that refused, and agreeing on it holds which rule refused, not only that one did.

`pouw-check-difftest`'s `opening` against `core_check.opening`: one `Port`, over the openings `pouw_opening_cases`
generates (honest openings of a call's tile or the filler's, and one mutation each aimed at every refusal or at a reading
both sides share), compared output for output: null, or the same reason.

Its `pass` and `draw` against `core_check.pass_epoch` and `core_check.draw`: one `Port` each, over the cases
`pouw_pass_cases` generates (honest passes of each coin version and a recorded beacon's, and honest draws, then one
mutation each aimed at every refusal), compared output for output: how the salt was had, or the same refusal.
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
import pouw_opening_cases as openings
import pouw_pass_cases as passes
from verity.core.protocols.pouw import core_check as CC

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


PORT = ExactPort("pouw-check", reference, inputs=INPUTS, package=LEAN)
server = D.server_fixture(PORT)


@pytest.mark.slow
def test_pouw_check(server):
    RULES.clear()
    stats = D.run(PORT, examples=EXAMPLES, server=server)
    print(f"pouw-check difftest: {stats}; by rule: {dict(sorted(RULES.items()))}")
    assert stats.counts.get("accept") and stats.counts.get("reject") and stats.counts.get("undecided")


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
    "the proof's first rows are not an A row and a B row of the registered schemas",
    "the proof's body holds no A row and B row",
    *(f"side {s}: the opening is of row #, not the unit's row #" for s in "ab"),
    *(f"side {s}: the proof's committed row is not entry #'s row #" for s in "ab"),
)}
OPENED: Counter[str] = Counter()


def opening(case: dict) -> str | None:
    why = CC.opening(CC.params(case["params"]), CC.commitment(bytes.fromhex(case["commitment"])), case["units"],
                     case["unit"], bytes.fromhex(case["pub"]), case["opening"])
    OPENED[_template(why) if why else "accept"] += 1
    return why


OPENING_NAMES = ["honest"] * (len(openings.NAMES) // 6) + openings.NAMES
OPENING = D.Port("pouw-check-difftest", "opening", opening, package=LEAN, rejects=(CC.Refused,),
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


PASS = D.Port("pouw-check-difftest", "pass", pass_epoch, package=LEAN, need_rejects=False,
              inputs=st.builds(passes.pass_instance, st.randoms(use_true_random=False),
                               st.sampled_from(["honest"] * (len(passes.PASS_NAMES) // 6) + passes.PASS_NAMES)))
DRAW = D.Port("pouw-check-difftest", "draw", draw, package=LEAN, need_rejects=False,
              inputs=st.builds(passes.draw_instance, st.randoms(use_true_random=False),
                               st.sampled_from(["honest"] * (len(passes.DRAW_NAMES) // 4) + passes.DRAW_NAMES)))
pass_server = D.server_fixture(PASS)


def test_pass_and_draw(pass_server):
    """Generated passes and draws, then each mutation over a few seeds (`replay`), so every refusal is reached whatever
    the draw."""
    PASSED.clear()
    stats = {"pass": D.run(PASS, examples=1500, server=pass_server), "draw": D.run(DRAW, examples=1000, server=pass_server)}
    swept = Counter(D.replay(PASS, passes.pass_instance(random.Random(f"{name}/{seed}"), name), pass_server)[0]
                    for name in ["honest", *passes.PASS_NAMES] for seed in range(3))
    swept += Counter(D.replay(DRAW, passes.draw_instance(random.Random(f"{name}/{seed}"), name), pass_server)[0]
                     for name in ["honest", *passes.DRAW_NAMES] for seed in range(3))
    print(f"pouw-check-difftest pass: {stats['pass']}; draw: {stats['draw']}; and {sum(swept.values())} swept "
          f"({dict(swept)}); by outcome: {dict(sorted(PASSED.items()))}")
    assert {k for k in PASSED if not k.startswith("draw ")} >= {"coin", "recorded beacon"}
    assert "draw accept" in PASSED and len(PASSED) > 40
    for seed in range(4):   # `COIN_EARLY`'s draws: accepted only as a replay of a recorded run, by both sides
        early, _ = passes.honest_draw(random.Random(f"early/{seed}"), CC.COIN_EARLY, early=True)
        live = {**early, "recorded": False}
        assert (draw(early), draw(live)["refused"][0]) == (None, "coin")
        D.replay(DRAW, early, pass_server), D.replay(DRAW, live, pass_server)
