"""The card schema: every card in the tree passes it under its own name, and the checker names each way a card breaks
it."""
import json

import cards
import pytest


def good() -> dict:
    return {"schema": cards.SCHEMA, "slot": "replay", "model": "GLM-5.3", "status": "measured", "provisional": False,
            "source": "a run", "runs": ["r20261106-061510-62be"], "owner": "@proofs",
            "value": {"factor": 32.0, "batch_layout": "served", "deterministic": True}}


def test_every_card_in_the_tree_passes_under_its_name():
    found = sorted(cards.CARDS.glob("*.json"))
    assert found
    for p in found:
        c = cards.check(json.loads(p.read_text()))
        assert p == cards.path(c["slot"], c["model"]), p.name


@pytest.mark.parametrize("change,message", [
    (lambda c: c["value"].update(extra=1), "unknown field 'extra'"),
    (lambda c: c["value"].pop("factor"), "replay.value.factor: missing"),
    (lambda c: c["value"].update(factor="32"), "factor: str, but it is a number"),
    (lambda c: c["value"].update(factor=True), "factor: bool, but it is a number"),
    (lambda c: c["value"].update(batch_layout="any"), "batch_layout: 'any', but it is one of"),
    (lambda c: c.update(model=None), "a replay card names its model"),
    (lambda c: c.update(status="assumed"), "card.status"),
    (lambda c: c.update(runs=["run 7"]), "not a run id"),
    (lambda c: c.update(owner="REPLAYER"), "the maintaining lane's handle"),
    (lambda c: c.update(source=" "), "card.source: empty"),
    (lambda c: c.update(slot="figure"), "card.slot: 'figure'"),
])
def test_the_checker_names_what_breaks(change, message):
    c = good()
    cards.check(c)
    change(c)
    with pytest.raises(ValueError, match=message):
        cards.check(c)


def test_a_reference_line_has_one_value_or_one_per_scope():
    line = {"panel": "exfiltration", "label": "x", "unit": "bit", "source": "s"}
    c = good() | {"slot": "references", "model": None}
    for bad in ({}, {"value": 1.0, "by_scope": {"world year": 1.0}}):
        with pytest.raises(ValueError, match="exactly one of value and by_scope"):
            cards.check(c | {"value": {"lines": [line | bad]}})
    cards.check(c | {"value": {"lines": [line | {"by_scope": {"world year": 1.0}}]}})


def test_load_falls_back_to_every_model_and_names_a_missing_card(tmp_path, monkeypatch):
    monkeypatch.setattr(cards, "CARDS", tmp_path)
    cards.write(good() | {"slot": "design", "model": None, "value": {"delta": .01, "budgets": [.01],
                                                                    "overheads": [1, 10], "token_bytes": 4}})
    assert cards.load("design", "GLM-5.3")["model"] is None
    with pytest.raises(FileNotFoundError, match="no replay card for GLM-5.3"):
        cards.load("replay", "GLM-5.3")
    assert cards.load("replay", "GLM-5.3", required=False) is None
    p = cards.write(good() | {"art": "art:1"})
    p.write_text(p.read_text().replace('"factor": 32.0', '"factor": 33.0'))
    assert cards.write(good() | {"art": "art:1"}) == p and cards.load("replay", "GLM-5.3")["value"]["factor"] == 33.0
    cards.write(good() | {"art": "art:2"})
    assert cards.load("replay", "GLM-5.3")["art"] == "art:2"
