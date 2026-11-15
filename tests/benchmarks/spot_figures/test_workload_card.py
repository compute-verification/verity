"""The `workload` card's generator (`workload_card.py`): a block census becomes a card the schema takes."""
import json

import cards
import workload_card


def census() -> dict:
    block = {"out_bits": 16, "in_bits": 64, "ands": 100, "macs": 8, "k_log": 20}
    return {"ranks": [0, 1], "prompt_len": 3, "steps": 2, "tokens": {"prefill": 3, "decode": 1},
            "per_request": {"circuit_macs": 400},
            "definitions": {"Dot_v1{K=8}": block | {"of": {"Linear_v1": "split"}, "prefill": 30, "decode": 10},
                            "Add_v1{N=4}": block | {"macs": 0, "of": {"B_v1": "row", "A_v1": "row"}, "prefill": 6,
                                                    "decode": 2},
                            "Idle_v1{}": block | {"of": {"A_v1": "row"}, "prefill": 0, "decode": 0}},
            "gaps": {"x": {"family": "TokenSelect_v1"}}}


def test_a_census_becomes_a_card_the_schema_takes(tmp_path):
    """One stratum a Definition the census counts a block of, its blocks a token over the request's tokens, the
    MAC-heaviest first; a Definition with no block has none; the gaps are named in the source and the whole
    circuit's MACs a token kept; the command writes the same card."""
    c = workload_card.card(census(), model="GLM-5.3", source="art:" + "ab" * 32, runs=["r20261109-031946-2a90"],
                           params=10, weights_bytes=80.0)
    assert cards.check(c) is c
    v = c["value"]
    assert [x["block"] for x in v["strata"]] == ["Dot_v1{K=8}", "Add_v1{N=4}"]
    assert [x["per_token"] for x in v["strata"]] == [10.0, 2.0] and v["strata"][1]["kind"] == "A_v1, B_v1"
    assert (v["tokens"], v["ranks"], v["request"]) == (4, 2, {"prefill": 3, "decode": 2})
    assert v["served_macs_per_token"] == 100.0 and v["boundaries"] == {"every_block": 12 * 16 / 8}
    assert "TokenSelect_v1" in c["source"] and c["source"].startswith("art:abab")
    path, out = tmp_path / "census.json", tmp_path / "card.json"
    path.write_text(json.dumps(census()))
    workload_card.main([str(path), "--census", "art:" + "ab" * 32, "--run", "r20261109-031946-2a90", "--model",
                        "GLM-5.3", "--params", "10", "--weights-bytes", "80", "--out", str(out)])
    assert json.loads(out.read_text()) == c


def test_the_tree_s_card_is_a_census_s_and_nothing_else():
    """GLM-5.3's card in the tree is as the generator writes one: its source the generator's sentence over a census's
    art id, its strata in the generator's order and its boundary bytes their sum, with no stratum edited after."""
    c = cards.load("workload", "GLM-5.3")
    v = c["value"]
    assert cards.ART.match(c["source"]) and "ANDs the estimator's upper bound; rows no cut fits" in c["source"]
    key = [(-x["per_token"] * x["macs"], -x["per_token"], x["block"]) for x in v["strata"]]
    assert key == sorted(key)
    assert v["boundaries"]["every_block"] == sum(x["per_token"] * x["out_bits"] for x in v["strata"]) / 8
