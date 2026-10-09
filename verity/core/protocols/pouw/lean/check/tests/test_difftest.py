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
