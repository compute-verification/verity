"""`verity.core.protocols.pouw.period_check` held to `pouw-period serve`, PoUW's verifier of record for a period
(`PouwCheck.Period`, ``verity/pouw/period/v1``, onsite's work component), on the shared harness (`verity_difftest`,
`of_record="lean"`): one `ProtocolPort` over the cases `pouw_period_cases` generates (a period of the auditor's ledger
and its windows, honest or with one mutation aimed at every refusal of `credit.period_credit`, and the protocol's own
framing broken), compared outcome for outcome, its reason included: the reason is the refusal `credit.period_credit`
names, so agreeing on it holds which rule refused the period."""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path

import pytest
import verity_difftest as D
from hypothesis import strategies as st

import pouw_period_cases as cases
from verity.core.protocols.pouw import period_check as PC
from verity.core.protocols.pouw import verifier as V

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

LEAN = Path(__file__).resolve().parents[1]


class ExactPort(D.ProtocolPort):
    def key(self, outcome: D.Outcome):
        """The whole outcome, its reason included."""
        return outcome if outcome[0] == "error" else json.loads(outcome[1])


RULES: Counter[str] = Counter()


def reference(case) -> dict:
    a = PC.case(case)
    RULES[f"{a['verdict']} {a['reason'].split(':')[0]}" if a["reason"] else a["verdict"]] += 1
    return a


INPUTS = st.builds(cases.instance, st.randoms(use_true_random=False),
                   st.sampled_from(["honest"] * (len(cases.NAMES) // 4) + cases.NAMES))
PORT = ExactPort("pouw-period", reference, inputs=INPUTS, package=LEAN, of_record=V.RECORD_VERDICT)
server = D.server_fixture(PORT)


@pytest.mark.slow
def test_pouw_period(server):
    """Generated periods, then each mutation over two seeds (`replay`), so every refusal is reached whatever the draw: an
    honest period is accepted with its credit and profile, every refusal of `credit.period_credit` rejects it with that
    refusal, and each framing break is refused at the input it breaks."""
    RULES.clear()
    stats = D.run(PORT, examples=60, server=server)
    swept = Counter(D.replay(PORT, cases.instance(random.Random(f"{name}/{seed}"), name), server)[0]
                    for name in ["honest", *cases.NAMES] for seed in range(2))
    print(f"pouw-period difftest: {stats}, and {sum(swept.values())} swept ({dict(swept)}); by rule: "
          f"{dict(sorted(RULES.items()))}")
    assert set(swept) == {"accept", "reject"}, dict(swept)
    assert {"accept", "reject params", "reject commitment", "reject evidence", "reject period", "reject window",
            "reject circuit", "reject missing", "reject order", "reject coin", "reject budget"} <= set(RULES), RULES


def test_an_honest_period_is_its_credit():
    """An honest period's answer is `credit._period_outcomes`' credit, its digest the ledger's under the protocol's tag,
    and a framing break is refused naming its input."""
    case = cases.instance(random.Random("honest/credit"))
    a = PC.case(case)
    assert a["verdict"] == "accept", a["reason"]
    assert a["public"]["delta"] == "1/1099511627776" and a["profile"]["windows"] == len(a["public"]["windows"])
    assert a["commitment"] == PC.digest({"bytes": bytes.fromhex(case["commitment"])}).hex()
    for name, what in (("params-missing", "params"), ("ledger-object", "commitment"), ("evidence-list", "evidence")):
        got = PC.case(cases.instance(random.Random(f"{name}/credit"), name))
        assert got["verdict"] == "reject" and got["reason"].startswith(f"{what}: "), (name, got)
