"""cover_count.py on a small hand-written request cover record: the count with and without the Boolean versions, an
extra whose controls did not refuse never counting, `--check`, the refusals of a stored total that disagrees with the
recount, an unknown schema and an outcome its verdicts contradict, `--card` on a hand-written card (a matching one
passes, one with a type more or fewer is refused naming it), and an `art:` id fetched with `research data fetch`."""
from __future__ import annotations

import copy
import json

import pytest

from conftest import load

ACCEPT = {"B.verify.held": {"exit": 0, "accepted": True},
          "B.verify.no-program": {"exit": 1, "accepted": False},
          "B.verify.other-def": {"exit": 1, "accepted": False}}
#: B.verify.held accepted, and so did the verifier handed no Program: the controls did not refuse
CONTROL_OFF = {**ACCEPT, "B.verify.no-program": {"exit": 0, "accepted": True}}
NOT_RUN: dict = {}

GEMM = "FooGemm_v1{K=1}"
NORM = "BarNorm_v1{D=8}"
SPLIT1, SPLIT2 = "SplitAttn_v1{T=1}", "SplitAttn_v1{T=2}"
LOGITS1, LOGITS2 = "Logits_v3{N=1}", "Logits_v3{N=2}"


def block(name, blocks, outcome, tag=None, on_card=True, statements=()):
    return {"block": name, "on_card": on_card, "blocks": blocks, "bound_outcome": outcome,
            "cir272": {"tag": tag} if tag else None, "statements": list(statements)}


def run(outcome, verdicts, why=None):
    return {"outcome": outcome, "why": why, "verdicts": verdicts}


def extra(listed_tag, name, blocks, outcome, verdicts, why=None):
    return {**run(outcome, verdicts, why), "extra": "beside",
            "earlier_form": {"tag": listed_tag, "block": name, "blocks": blocks}}


RECORD = {
    "schema": "circuits/request-cover/v7",
    "tree": "0" * 40,
    "blocks": [
        block(GEMM, 1000, "accepted-own-definition", "t001"),
        block(NORM, 500, "accepted-earlier", statements=[{"verdict": "accepted"}]),
        block(SPLIT1, 40, "not-run-estimate-past-64GB", "t002"),
        block(SPLIT2, 30, "not-run-stage-past-62GB", "t003"),
        block(LOGITS1, 20, "not-run-not-reached", "t004"),
        block(LOGITS2, 10, "not-run-not-reached", "t005"),
        block("OffCard_v1{X=1}", 7, "not-run-not-reached", "t006", on_card=False),
    ],
    "bound_runs": {"types": {
        "t001": run("accepted", ACCEPT),
        "t002": run("not-run", NOT_RUN, "estimate-past-64GB"),
        "t003": run("not-run", NOT_RUN, "stage-past-62GB"),
        "t004": run("not-run", NOT_RUN, "not-reached"),
        "t005": run("not-run", NOT_RUN, "not-reached"),
        "t006": run("not-run", NOT_RUN, "not-reached"),
    }},
    "bound_totals": {"accepted-earlier": [1, 500], "accepted-own-definition": [1, 1000],
                     "not-run-estimate-past-64GB": [1, 40], "not-run-stage-past-62GB": [1, 30],
                     "not-run-not-reached": [2, 30]},
    "bound_shares": {"accepted": {"types": 2, "blocks": 1500, "share": 0.9375},
                     "refused": {"types": 0, "blocks": 0, "share": 0.0},
                     "not-run": {"types": 4, "blocks": 100, "share": 0.0625}},
    "beside_extras": {
        "v2t002": extra("t002", SPLIT1, 40, "accepted", ACCEPT),
        "v2t003": extra("t003", SPLIT2, 30, "held-control-off", CONTROL_OFF),
        "v2t004": extra("t004", LOGITS1, 20, "not-run", NOT_RUN, "estimate-past-160GB"),
    },
    "fifth_pass": {"by_family": {
        "SplitAttn": {"types": 2, "blocks": 70, "outcomes": {"accepted": 1, "held-control-off": 1}},
        "Logits": {"types": 1, "blocks": 20, "outcomes": {"not-run/estimate-past-160GB": 1}},
    }},
}


@pytest.fixture
def cc(monkeypatch):
    return load(monkeypatch, "cover_count.py")


def write(tmp_path, record):
    p = tmp_path / "cover.json"
    p.write_text(json.dumps(record))
    return str(p)


def test_the_default_counts_a_boolean_version_only_where_its_controls_refused(cc, tmp_path, capsys):
    """t002's Boolean version was accepted, so it counts; t003's was accepted with a control that also accepted, so it
    does not; t004's was not run and t005 has none: three types with none, by family with each one's reason."""
    assert cc.main([write(tmp_path, RECORD)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[1:5] == ["on the card: 6 types, 1,600 blocks",
                        "accepted in the listed form: 2 types, 1,500 blocks",
                        "accepted on the Boolean version only: 1 types, 40 blocks",
                        "with none (the Boolean version counted): 3 types, 60 blocks"]
    assert out[5:] == [
        "  Logits: 2 types, 30 blocks",
        f"    t004  20 blocks  not-run/estimate-past-160GB (v2t004; listed form not-run-not-reached)  {LOGITS1}",
        f"    t005  10 blocks  not-run-not-reached; no Boolean version in the record  {LOGITS2}",
        "  SplitAttn: 1 types, 30 blocks",
        f"    t003  30 blocks  held-control-off (v2t003; listed form not-run-stage-past-62GB)  {SPLIT2}",
        "3 of 6 block types have no statement the Lean verifier accepted (--form boolean)"]


def test_form_listed_counts_only_the_listed_form(cc, tmp_path, capsys):
    assert cc.main([write(tmp_path, RECORD), "--form", "listed"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[3] == "accepted on the Boolean version only: 1 types, 40 blocks (not counted: --form listed)"
    assert out[4] == "with none (listed form only): 4 types, 100 blocks"
    assert f"    t002  40 blocks  not-run-estimate-past-64GB  {SPLIT1}" in out
    assert out[-1] == "4 of 6 block types have no statement the Lean verifier accepted (--form listed)"


def test_check_exits_1_only_when_the_count_is_not_0(cc, tmp_path, capsys):
    assert cc.main([write(tmp_path, RECORD), "--check"]) == 1
    covered = copy.deepcopy(RECORD)
    covered["blocks"] = covered["blocks"][:2]
    covered["bound_totals"] = {"accepted-earlier": [1, 500], "accepted-own-definition": [1, 1000]}
    covered["bound_shares"] = {"accepted": {"types": 2, "blocks": 1500, "share": 1.0},
                               "not-run": {"types": 0, "blocks": 0, "share": 0.0}}
    covered["beside_extras"] = {}
    covered["fifth_pass"]["by_family"] = {}
    assert cc.main([write(tmp_path, covered), "--check"]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith("0 of 2 block types")


def _totals(r):
    r["bound_totals"]["not-run-not-reached"] = [2, 31]


def _shares(r):
    r["bound_shares"]["accepted"]["share"] = 0.9


def _family(r):
    r["fifth_pass"]["by_family"]["SplitAttn"]["outcomes"] = {"accepted": 2}


def _schema(r):
    r["schema"] = "circuits/request-cover/v9"


def _v8_without_its_card(r):
    r["schema"] = "circuits/request-cover/v8"


def _accepted_but_control_off(r):
    r["beside_extras"]["v2t003"]["outcome"] = "accepted"
    r["fifth_pass"]["by_family"]["SplitAttn"]["outcomes"] = {"accepted": 2}


def _earlier_without_statement(r):
    r["blocks"][1]["statements"] = []


def _extra_off_the_card(r):
    r["beside_extras"]["v2t004"]["earlier_form"]["tag"] = "t006"


@pytest.mark.parametrize(("change", "says"), [
    (_totals, "bound_totals"),
    (_shares, "bound_shares['accepted']"),
    (_family, "fifth_pass.by_family"),
    (_schema, "schema 'circuits/request-cover/v9' is not one this command knows"),
    (_v8_without_its_card, "a circuits/request-cover/v8 record names its card's source under card.source"),
    (_accepted_but_control_off, "beside extra v2t003: the record says 'accepted' but a control"),
    (_earlier_without_statement, "with no accepted statement"),
    (_extra_off_the_card, "beside extra v2t004: its earlier form t006 is not that block type on the card"),
])
def test_a_record_that_disagrees_with_its_recount_is_refused_by_name(cc, tmp_path, capsys, change, says):
    r = copy.deepcopy(RECORD)
    change(r)
    assert cc.main([write(tmp_path, r)]) == 2
    got = capsys.readouterr()
    assert got.out == ""
    assert says in got.err


ON_CARD = [GEMM, NORM, SPLIT1, SPLIT2, LOGITS1, LOGITS2]
CARD_SOURCE = "art:" + "ef" * 32


def card(tmp_path, types):
    p = tmp_path / "card.json"
    p.write_text(json.dumps({"source": CARD_SOURCE, "definitions": {t: {"prefill": 1, "decode": 0} for t in types}}))
    return str(p)


def test_card_a_record_of_exactly_the_cards_types_passes(cc, tmp_path, capsys):
    """The off-card entry is not one of the card's; the v8 record names the card it is of in its first line."""
    v8 = {**copy.deepcopy(RECORD), "schema": "circuits/request-cover/v8",
          "card": {"source": CARD_SOURCE, "path": "card.json", "tree": "0" * 40}}
    assert cc.main([write(tmp_path, v8), "--card", card(tmp_path, ON_CARD)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"record: circuits/request-cover/v8, tree {'0' * 40}, of the card {CARD_SOURCE}"
    assert out[-1].startswith("3 of 6 block types")


@pytest.mark.parametrize(("types", "says"), [
    ([*ON_CARD, "Extra_v1{X=1}"], "1 on the card and not in the record: Extra_v1{X=1}"),
    (ON_CARD[:-1], f"1 in the record and not on the card: {LOGITS2}"),
    ([*ON_CARD, *(f"Extra_v1{{X={i}}}" for i in range(7))],
     "7 on the card and not in the record: " + ", ".join(f"Extra_v1{{X={i}}}" for i in range(5)) + " and 2 more"),
])
def test_card_a_type_more_or_fewer_is_refused_naming_it(cc, tmp_path, capsys, types, says):
    assert cc.main([write(tmp_path, RECORD), "--card", card(tmp_path, types)]) == 2
    got = capsys.readouterr()
    assert got.out == ""
    assert says in got.err


def test_an_art_id_is_fetched_with_research_data_fetch(cc, monkeypatch, capsys):
    calls = []

    class Done:
        returncode = 0
        stderr = ""

    class Failed:
        returncode = 1
        stderr = "no such artifact"

    def fake_run(argv, **kw):
        calls.append(argv)
        if "art:" + "cd" * 32 in argv:
            return Failed()
        to = argv[argv.index("--to") + 1]
        (cc.Path(to) / "cover.json").write_text(json.dumps(RECORD))
        return Done()

    monkeypatch.setattr(cc.shutil, "which", lambda name: "/bin/research" if name == "research" else None)
    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    assert cc.main(["art:" + "ab" * 32]) == 0
    assert len(calls) == 1 and len(calls[0]) == 6
    assert calls[0][:5] == ["/bin/research", "data", "fetch", "art:" + "ab" * 32, "--to"]
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("record: ") and out[-1].startswith("3 of 6 block types")
    assert cc.main(["art:" + "cd" * 32]) == 2
    assert "exited 1: no such artifact" in capsys.readouterr().err
