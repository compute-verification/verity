"""`verity.core.protocols.pouw.period_check` held to `pouw-period serve`, PoUW's verifier of record for a period
(`PouwCheck.Period`, ``verity/pouw/period/v1``, onsite's work component), on the shared harness (`verity_difftest`,
`of_record="lean"`): one `ProtocolPort` over the cases `pouw_period_cases` generates (a period of the auditor's ledger
and its windows, honest or with one mutation aimed at every refusal of `credit.period_credit`, and the protocol's own
framing broken, a period in each form the period's guarantee doesn't cover, a period with its filler windows, honest
or with one mutation aimed at every refusal of `fill.period_fill`, which the period reports and no verdict reads, and a
period read against a capacity), compared outcome for outcome, its reason and its public output (a refused fill's
refusal, ``gaps`` and ``utilization``) included: the reason is the refusal `credit.period_credit` names, so agreeing on
it holds which rule refused the period."""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest
import verity_difftest as D
from hypothesis import strategies as st

import pouw_fill_cases as fill_cases
import pouw_period_cases as cases
from verity.core.protocols.pouw import fill as F
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


def test_a_filler_shape_spans_at_most_t_max_noise_tiles():
    """A filler descriptor whose block spans t = min(L, K)/L′ = `fill.T_MAX` noise tiles is a shape, one more is not."""
    sh = fill_cases.D.Shape(128, 128, 32 * F.T_MAX, 32, 32 * F.T_MAX, 16384, o=False)
    assert F.descriptor_of(F.descriptor(sh, 1, bytes(64)))[0] == (sh, 1)
    j = json.loads(F.descriptor(sh, 1, bytes(64)))
    j.update(L=32 * (F.T_MAX + 1), k=32 * (F.T_MAX + 1))
    assert F.descriptor_of(fill_cases.Cr._canonical(j).encode()) == (None, ("descriptor", F.SHAPE))


def test_a_fill_is_credited_beside_work():
    """A period with honest filler windows is accepted with its fill (``{work, k, delta, eps}``) beside the served credit,
    which is the same period's credit without them, and with ``gaps`` saying no guarantee covers the fill: the fill
    changes nothing `work`'s rule reads."""
    for seed in range(3):
        a = PC.case(cases.instance(random.Random(f"fill/{seed}"), "fill"))
        assert a["verdict"] == "accept", a["reason"]
        assert a["public"].pop("gaps") == PC.GAPS
        fill = a["public"].pop("fill")
        assert sorted(fill) == ["delta", "eps", "k", "work"] and a["profile"]["fills"] == len(fill["work"]) > 0
        assert all(w > 0 for w in fill["work"]) and fill["delta"] == a["public"]["delta"]
        served = PC.case(cases.instance(random.Random(f"fill/{seed}"), "honest"))
        assert served["verdict"] == "accept" and served["public"] == a["public"], (served, a)


@pytest.mark.parametrize("name", fill_cases.NAMES)
def test_a_fill_mutation_is_reported_not_decided_on(name):
    """Each of `pouw_fill_cases`' mutations leaves the period accepted with its served credit and reports the fill
    refused, ``{refused: "code: why"}``, with the refusal it aims at (``fill`` naming the refused filler window's own
    code), and no filler windows in the profile."""
    for seed in range(2):
        got = PC.case(cases.instance(random.Random(f"{name}/{seed}"), name))
        assert got["verdict"] == "accept" and "fills" not in got["profile"], (name, got)
        assert got["public"]["gaps"] == PC.GAPS, (name, got["public"])
        code, _, why = got["public"]["fill"]["refused"].partition(": ")
        assert code == fill_cases.AIMS[name], (name, got["public"]["fill"])
        if name in fill_cases.INNER:
            assert f" the fill earns nothing: {fill_cases.INNER[name]}: " in why, (name, got["public"]["fill"])


def _without_fills(case: dict) -> dict:
    e = json.loads(bytes.fromhex(case["evidence"]))
    return {**case, "evidence": cases._hex({k: v for k, v in e.items() if k != "fills"})}


def test_fill_never_changes_the_verdict():
    """A period's verdict and reason, and its served credit, are the same with its filler windows (honest or with any of
    `pouw_fill_cases`' mutations) as without them, under no capacity or a capacity, and so is its utilization: no
    guarantee covers the fill yet, so no decision and nothing certified reads it."""
    rng = random.Random("fill-verdict")
    for name in ["fill", *fill_cases.NAMES]:
        d = fill_cases.with_fills(rng, cases.credit_cases.period_instance(rng, "honest", **cases.COVERED), name)
        case = cases.serve(d)
        served = sum(PC.case(_without_fills(case))["public"]["work"])
        for params in ({}, {"capacity": rng.randint(1, 2 * served)}):
            with_, without = (PC.case({**c, "params": {**c["params"], **params}}) for c in (case, _without_fills(case)))
            assert (with_["verdict"], with_["reason"]) == (without["verdict"], without["reason"]), (name, params, with_)
            if with_["verdict"] == "accept":
                assert {k: v for k, v in with_["public"].items() if k not in ("fill", "gaps")} == without["public"], \
                    (name, params)


def test_utilization_certifies_served_work_alone():
    """With a capacity the credit gains ``utilization``: C, the certified work, the served work alone however much fill
    the period presents (under a capacity of its served work and its fill together, it certifies the served work and
    its share is below 1), and its share of C; no parameter decides idle, so a floor beside the capacity, or a
    malformed capacity, is refused as a parameter."""
    for seed in range(4):
        a = PC.case(cases.instance(random.Random(f"utilization/{seed}"), "utilization"))
        assert a["verdict"] == "accept", a["reason"]
        pub, u = a["public"], a["public"]["utilization"]
        assert u["certified"] == sum(pub["work"]) and Fraction(u["share"]) == Fraction(u["certified"], u["capacity"])
        rng = random.Random(f"fill-utilization/{seed}")
        d = fill_cases.with_fills(rng, cases.credit_cases.period_instance(rng, "honest", **cases.COVERED), "fill")
        b = PC.case(cases.serve(d))
        served, fill = sum(b["public"]["work"]), sum(b["public"]["fill"]["work"])
        got = PC.case({**cases.serve(d), "params": {**cases.serve(d)["params"], "capacity": served + fill}})
        assert got["verdict"] == "accept" and fill > 0, got["reason"]
        assert got["public"]["utilization"] == {"capacity": served + fill, "certified": served,
                                                "share": str(Fraction(served, served + fill))}
        got = PC.case(cases.instance(random.Random(f"params-capacity/{seed}"), "params-capacity"))
        assert got["verdict"] == "reject" and got["reason"].startswith("params: "), got["reason"]


@pytest.mark.parametrize("raw", [b"\xff", b"\xed\xa0\x80", b'{"program_fn":"\xc0\xaf"}'])
def test_a_descriptor_not_utf8_is_left_to_the_credit(raw):
    """A window whose descriptor isn't UTF-8 (a stray byte, a surrogate, an overlong) is no form of its own, as Lean's
    `String.fromUTF8?` reads it, and the period is rejected by `credit.period_credit`, not credited."""
    assert PC._window_form("p", 0, {"descriptor": raw.hex()}) is None
    d = cases.credit_cases.period_instance(random.Random("utf8"), "honest", **cases.COVERED)
    d["windows"][0]["descriptor"] = raw.hex()
    got = PC.case(cases.serve(d))
    assert got["verdict"] == "reject" and not got["reason"].startswith("form: "), got["reason"]
