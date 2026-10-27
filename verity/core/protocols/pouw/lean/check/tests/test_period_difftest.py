"""`verity.core.protocols.pouw.period_check` held to `pouw-period serve`, PoUW's verifier of record for a period
(`PouwCheck.Period`, ``verity/pouw/period/v1``, onsite's work component), on the shared harness (`verity_difftest`,
`of_record="lean"`): one `ProtocolPort` over the cases `pouw_period_cases` generates (a period of the auditor's ledger
and its windows, honest or with one mutation aimed at every refusal of `credit.period_credit`, and the protocol's own
framing broken, and a period in each form the period's guarantee doesn't cover), compared outcome for outcome, its reason included: the reason is the refusal `credit.period_credit`
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
    assert {"accept", "reject params", "reject commitment", "reject evidence", "reject form", "reject period",
            "reject window", "reject circuit", "reject missing", "reject order", "reject coin",
            "reject budget"} <= set(RULES), RULES


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


@pytest.mark.parametrize("name,why", [("form-pass", "is a Pearl-C pass"),
                                      ("form-not-h1", "proves PouwH1Tile_v1, not an H1 window"),
                                      ("form-per-unit", "takes per-unit noise"),
                                      ("form-v6", "is not over PouwH1Tile_v5"), ("form-strata", "units at 2 K's")])
def test_an_uncovered_form_is_refused_by_name(name, why):
    """Every form `H1J10RCheckedCreditPeriod` doesn't cover is refused before the credit, naming the form and the
    guarantee, and the same period in the covered form (J10-R's per-call unit over PouwH1Tile_v5, at one K) is credited."""
    for seed in range(3):
        got = PC.case(cases.instance(random.Random(f"{name}/{seed}"), name))
        assert got["verdict"] == "reject" and got["reason"].startswith("form: window "), (name, got["reason"])
        assert why in got["reason"] and PC.GUARANTEE in got["reason"], (name, got["reason"])
        assert PC.case(cases.instance(random.Random(f"{name}/{seed}"), "honest"))["verdict"] == "accept"


@pytest.mark.parametrize("raw", [b"\xff", b"\xed\xa0\x80", b'{"program_fn":"\xc0\xaf"}'])
def test_a_descriptor_not_utf8_is_left_to_the_credit(raw):
    """A window whose descriptor isn't UTF-8 (a stray byte, a surrogate, an overlong) is no form of its own, as Lean's
    `String.fromUTF8?` reads it, and the period is rejected by `credit.period_credit`, not credited."""
    assert PC._window_form("p", 0, {"descriptor": raw.hex()}) is None
    d = cases.credit_cases.period_instance(random.Random("utf8"), "honest", **cases.COVERED)
    d["windows"][0]["descriptor"] = raw.hex()
    got = PC.case(cases.serve(d))
    assert got["verdict"] == "reject" and not got["reason"].startswith("form: "), got["reason"]
