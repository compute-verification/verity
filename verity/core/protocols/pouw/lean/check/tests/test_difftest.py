"""`pouw-check serve` against `verity.core.protocols.pouw.core_check`, on the shared harness (`verity_difftest`):
one `ProtocolPort`, over the cases `pouw_check_cases` generates from a Hypothesis-controlled `random.Random` (honest
windows, and one mutation each aimed at every refusal, the case's framing included), so a run is deterministic and
shrinks.

The harness compares an outcome's verdict, commitment, public outputs and profile. The reference writes the port's
reasons too, `Core.Protocol.case`'s framing included, so this port compares the whole outcome: the reason names the rule
that refused, and agreeing on it holds which rule refused, not only that one did.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pytest
import verity_difftest as D
from hypothesis import strategies as st

import pouw_check_cases as cases
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
