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
        assert p == cards.path(c["slot"], c["model"], c.get("art")), p.name


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


def test_a_workload_serves_at_least_its_proof_blocks_macs():
    stratum = {"block": "Dot_v1{K=8}", "per_token": 2.0, "macs": 8, "out_bits": 16, "modeled": True}
    value = {"ranks": 1, "device": "H200", "request": {"prefill": 1, "decode": 1}, "tokens": 2, "weights_bytes": 1.0,
             "strata": [stratum]}
    c = good() | {"slot": "workload"}
    for served in ({}, {"served_macs_per_token": None}, {"served_macs_per_token": 32.0}):
        cards.check(c | {"value": value | served})
    with pytest.raises(ValueError, match="below the strata's 16 MACs a token"):
        cards.check(c | {"value": value | {"served_macs_per_token": 8.0}})


def headline() -> dict:
    cells = {"trace S": {"without m = 37": {"world year": 0.18}, "with m = 37": {"world year": 0.135}}}
    row = {"row": "(ii)", "label": "(ii) a fused cut", "pricing": "on its own premises", "basis": "design",
           "basis_from": ["a cut no serving path commits"], "rests_on": ["λ = 1.05"], "cells": cells,
           "runs": ["r20261109-192926-0dd9"]}
    return good() | {"slot": "headline", "status": "modeled", "value": {"bases": list(cards.BASES), "rows": [row],
                                                                       "quote": "no row is served and measured"}}


def test_a_headline_card_is_measured_only_when_every_row_is_served_and_measured():
    c = headline()
    cards.check(c)
    with pytest.raises(ValueError, match="a measured headline card holds only served-measured rows"):
        cards.check(c | {"status": "measured"})
    served = c["value"]["rows"][0] | {"basis": "served-measured", "basis_from": []}
    cards.check(c | {"status": "measured", "value": c["value"] | {"rows": [served]}})


@pytest.mark.parametrize("change,message", [
    (lambda r: r.update(basis="served"), "basis: 'served', but it is one of"),
    (lambda r: r.update(basis_from=[]), "empty basis_from; name what sets it"),
    (lambda r: r["cells"]["trace S"]["with m = 37"].update({"world year": -0.1}), "a number at least 0"),
    (lambda r: r["cells"]["trace S"].update({"with m = 37": 0.1}), "an object of scopes"),
    (lambda r: r["cells"].update({"census S": [0.2]}), "an object of verifier steps"),
])
def test_a_headline_row_names_what_breaks(change, message):
    c = headline()
    change(c["value"]["rows"][0])
    with pytest.raises(ValueError, match=message):
        cards.check(c)


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


def test_references_cards_add_up_the_models_first(tmp_path, monkeypatch):
    monkeypatch.setattr(cards, "CARDS", tmp_path)
    line = {"panel": "exfiltration", "label": "x", "unit": "bit", "value": 1.0, "source": "s"}
    ref = good() | {"slot": "references", "value": {"lines": [line]}}
    for model, art in (("GLM-5.3", "art:" + "a" * 64), ("GLM-5.3", "art:" + "b" * 64), (None, "art:" + "c" * 64),
                       ("DeepSeek-V4-Pro", "art:" + "d" * 64), ("GLM-5.3", "art:" + "a" * 64)):
        cards.write(ref | {"model": model, "art": art})
    assert cards.path("references", "GLM-5.3", "art:" + "a" * 64).name == "references.glm-5.3.aaaaaaaaaaaa.json"
    assert [c["art"][4] for c in cards.load_all("references", "GLM-5.3")] == ["a", "b", "c"]
    assert [c["art"][4] for c in cards.load_all("references", "DeepSeek-V4-Pro")] == ["d", "c"]
    cards.write(ref | {"model": "GLM-5.3", "art": "art:" + "e" * 64}, replaces=("art:" + "b" * 64,))
    assert [c["art"][4] for c in cards.load_all("references", "GLM-5.3")] == ["a", "e", "c"]
    with pytest.raises(FileNotFoundError, match="replaces"):
        cards.write(ref | {"model": "GLM-5.3", "art": "art:" + "f" * 64}, replaces=("art:" + "b" * 64,))
