"""The `serving_rate` card and the two readings of serving's rate: the default, rho's fleet utilization, reads no card
and prices every line as before the card; `serving="measured"` converts the card's tokens a GPU-second with a
request's own tokens and leaves the scope's GPUs alone."""
import json
import shutil

import cards
import figure1 as f
import figure3
import figure4
import pytest
import total_cost as t

SCOPE = ("world year",)
DESIGNS = ("two-stage", "request")


@pytest.fixture
def card_dir(tmp_path, monkeypatch):
    """The tree's cards, copied where a test may change one."""
    shutil.copytree(cards.CARDS, tmp_path / "cards")
    monkeypatch.setattr(cards, "CARDS", tmp_path / "cards")
    return tmp_path / "cards"


def kept() -> dict:
    return json.loads(cards.path("serving_rate", "GLM-5.3").read_text())


def parts(c: dict) -> tuple:
    strata = c["partition"].value["strata"]
    return (f.TokenBlocks.of(strata, c["partition"].value["served_macs_per_token"]),
            f.Fill.of(strata, c["proof_blocks"].value))


def test_the_kept_card_is_its_owners_and_a_floor_is_provisional():
    card = cards.check(kept())
    assert card["owner"] == "@vllm" and card["status"] == "measured" and card["runs"]
    assert card["value"]["saturated"] or card["provisional"]


@pytest.mark.parametrize("change,message", [
    ({"provisional": False}, "a lower bound is provisional"),
    ({"status": "modeled"}, "a serving rate is measured"),
    ({"value": {"bound": "memory"}}, "one of 'compute', 'kv-pool', 'unknown'"),
    ({"value": {"plan": "the 128k plan"}}, "an art id .* or the id of the run that holds it"),
    ({"value": {"tokens_per_gpu_second": 0}}, "a rate above 0"),
    ({"value": {"tokens_per_gpu_second": "fast"}}, "it is a number"),
    ({"value": {"mfu": 0.4}}, "unknown field 'mfu'"),
])
def test_the_checker_names_what_breaks_a_serving_rate_card(change, message):
    card = kept() | {"provisional": True}
    card["value"] = card["value"] | {"saturated": False} | change.get("value", {})
    with pytest.raises(ValueError, match=message):
        cards.check(card | {k: v for k, v in change.items() if k != "value"})


def test_a_saturated_rate_need_not_be_provisional_and_its_plan_may_be_a_run():
    card = kept() | {"provisional": False}
    card["value"] = card["value"] | {"saturated": True, "bound": "kv-pool", "plan": card["runs"][0]}
    assert cards.check(card)["value"]["saturated"]


def test_the_default_reads_no_card_and_prices_each_line_as_before(card_dir):
    """The four lines that price serving, each held to the expression it had before the card, to the last bit, with
    the card in the tree and with none."""
    with_card = f.cards()
    (card_dir / "serving_rate.glm-5.3.json").unlink()
    c = f.cards()
    assert c == with_card == f.cards(serving="assumed") and "serving_rate" not in c
    assert "serving_rate" not in f.slot_cards("GLM-5.3")
    part, fill = parts(c)
    mfu, peak = c["mfu_serve"].value, c["peak_flops"].value
    for s in f.scopes(c, part):
        assert s.requests == s.gpu_seconds * mfu * peak / s.flop_request
        assert f.model(c, part, s, 1.0, fill).replay.cost == c["replay_factor"].value * s.flop_request / (mfu * peak)
    token_s = 2 * c["partition"].value["served_macs_per_token"] / (mfu * peak)
    assert f.token_seconds(c, c["partition"].value["served_macs_per_token"]) == token_s
    assert figure3.run("GLM-5.3")["token_seconds"] == token_s
    assert t.priced_card() == t.priced_card(serving="assumed")


def test_the_defaults_json_is_the_same_with_the_card_and_without(card_dir):
    with_card = f.run("GLM-5.3", 3, SCOPE, designs=DESIGNS, lean=False)
    (card_dir / "serving_rate.glm-5.3.json").unlink()
    without = f.run("GLM-5.3", 3, SCOPE, designs=DESIGNS, lean=False)
    assert json.dumps(with_card) == json.dumps(without)
    assert "serving" not in with_card and "serving_rate" not in with_card["inputs"]
    assert not any("tokens_per_gpu_second" in row for row in f.at_overhead(with_card))


@pytest.mark.parametrize("request_", f.REQUESTS)
def test_the_measured_reading_converts_with_the_requests_own_tokens(request_):
    c, m = f.cards(request=request_), f.cards(request=request_, serving="measured")
    rate, tokens = kept()["value"]["tokens_per_gpu_second"], m["tokens_per_request"].value
    assert m["serving_rate"].value == rate and m["serving_rate"].provisional == kept()["provisional"]
    assert {k: v for k, v in m.items() if k != "serving_rate"} == c
    part, fill = parts(m)
    for s, assumed in zip(f.scopes(m, part), f.scopes(c, part)):
        assert (s.gpu_seconds, s.flop_request) == (assumed.gpu_seconds, assumed.flop_request)
        assert s.requests == s.gpu_seconds * rate / tokens
        assert f.model(m, part, s, 1.0, fill).replay.cost == m["replay_factor"].value * tokens / rate
    assert f.token_seconds(m, part.served_macs) == 1 / rate
    assert figure3.run("GLM-5.3", "measured")["token_seconds"] == 1 / rate


def test_a_priced_tax_hashes_against_the_measured_tokens_seconds():
    c = f.cards()
    slower = (1 / kept()["value"]["tokens_per_gpu_second"]) / f.token_seconds(c, c["partition"].value["served_macs_per_token"])
    assumed, measured = (t.priced_card(serving=s)["value"]["configurations"] for s in f.SERVING)
    hashed = [(a, b) for x, y in zip(assumed, measured) for a, b in zip(x["parts"], y["parts"]) if "hashed" in a["source"]]
    assert hashed
    for a, b in hashed:
        assert b["share"] == pytest.approx(a["share"] / slower)


def test_a_reading_is_named_and_the_measured_one_needs_its_card(card_dir):
    with pytest.raises(ValueError, match="serving's rate is one of"):
        f.cards(serving="saturated")
    (card_dir / "serving_rate.glm-5.3.json").unlink()
    with pytest.raises(FileNotFoundError, match="no serving_rate card for GLM-5.3"):
        f.cards(serving="measured")


def test_a_row_at_the_measured_reading_names_the_rate_and_the_rate_at_the_goal():
    out = f.run("GLM-5.3", 3, SCOPE, designs=DESIGNS, lean=False, serving="measured")
    card = kept()
    rate = card["value"]["tokens_per_gpu_second"]
    assert out["serving"] == {"reading": "measured", **card["value"], "provisional": card["provisional"],
                              "assumed_tokens_per_gpu_second": out["serving"]["assumed_tokens_per_gpu_second"]}
    assert out["cards"]["serving_rate"] == card and out["inputs"]["serving_rate"]["value"] == rate
    assumed = f.run("GLM-5.3", 3, SCOPE, designs=DESIGNS, lean=False)
    assert out["scopes"]["world year"]["requests"] / assumed["scopes"]["world year"]["requests"] == pytest.approx(
        rate / out["serving"]["assumed_tokens_per_gpu_second"])
    rows = [r for r in f.at_overhead(out) if r["panel"] == "exfiltration"]
    assert rows
    for r in rows:
        assert (r["tokens_per_gpu_second"], r["saturated"]) == (rate, card["value"]["saturated"])
        assert r["tokens_per_gpu_second_at_goal"] == rate * f.SERVING_GOAL / r["bound"]
    assert not any("tokens_per_gpu_second" in r for r in f.at_overhead(out) if r["panel"] != "exfiltration")


def test_figure_1s_rows_at_its_budgets_name_the_rate_only_under_the_measured_reading():
    series = {"scope": "world year", "config": "tokens", "beta": [0.01], "x": [0.03], "y": [1.2e12]}
    out = {"budgets": [0.01], "scopes": {"world year": {"served_bits": 1e18, "flop": 1e30}},
           "panels": {"exfiltration": {"series": [series]}, "skipped_work": {"series": [series | {"y": [1e25]}]}}}
    assert all("tokens_per_gpu_second" not in r for r in t.at_budgets(out))
    exfil, work = t.at_budgets(out | {"serving": {"tokens_per_gpu_second": 180.0, "saturated": False}})
    assert exfil["tokens_per_gpu_second"] == 180.0 and exfil["saturated"] is False
    assert exfil["tokens_per_gpu_second_at_goal"] == 180.0 * f.SERVING_GOAL / 1.2e12 == 360.0
    assert "tokens_per_gpu_second" not in work


def test_figure_4_counts_tokens_at_the_reading_figure_1_was_planned_at(monkeypatch):
    seen = []
    monkeypatch.setattr(f, "cards", lambda model, **kw: seen.append(kw) or (_ for _ in ()).throw(LookupError))
    for one in ({}, {"serving": {"reading": "measured"}}):
        with pytest.raises(LookupError):
            figure4.run("GLM-5.3", one)
    assert seen == [{"serving": "assumed"}, {"serving": "measured"}]
