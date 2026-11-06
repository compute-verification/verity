"""GLM-5.3's workload card's generator on a synthetic census, and the card in the tree against the generator's rules."""
import cards
import glm53_partition as g
import pytest


def census(**extra) -> dict:
    kinds = {
        "EmbeddingShard_v1": {"calls": 4, "gates": 100},
        "Linear_v1": {"calls": 8, "gates": 2 * 8 * 50, "path": [
            {"id": "Linear_v1{K=4096,N=64}", "per_call": 1, "gates": 50},
            {"id": "ScaledMmFp8BlockCoordinateGivenScale_v1{K=4096,G=128}", "per_call": 64, "gates": 1}]},
        "SplitK_v1": {"calls": 4, "gates": 40, "call_gates": 10, "fold_chain": [
            {"id": "SplitK_v1{K=8192}", "per_call": 1, "gates": 10},
            {"id": "DotBf16_v2{K=512}", "per_call": 16, "gates": 1}]},
        "ResidualAdd_v1": {"calls": 4, "gates": 300},
        "Const32[7]": {"calls": 3, "gates": 5},
        "AllGather_v1": {"calls": 32, "gates": 0},
    } | extra
    return {"gates": 2000, "descriptor": "synthetic", "kinds": kinds}


def test_a_matmul_is_its_coordinates_and_the_rest_its_words():
    tokens, rows = g.strata(census())
    assert tokens == 4
    by = {r["kind"]: r for r in rows}
    assert set(by) == {"EmbeddingShard_v1", "Linear_v1", "SplitK_v1", "ResidualAdd_v1"}
    pick = ("per_token", "macs", "out_bits")
    assert tuple(by["Linear_v1"][k] for k in pick) == (g.RANKS * 8 * 64 * 2 / 4, 4096, 16)
    assert tuple(by["SplitK_v1"][k] for k in pick) == (g.RANKS * 4 * 16 / 4, 512, 32)
    assert tuple(by["ResidualAdd_v1"][k] for k in pick) == (g.RANKS * 4 * 6144 / 4, 0, 16)
    assert by["Linear_v1"]["block"] == "ScaledMmFp8BlockCoordinateGivenScale_v1{K=4096,G=128}"
    cards.check(g.card(census(), "art:synthetic"))


def test_a_kind_with_no_rule_is_refused():
    with pytest.raises(KeyError, match="add its output words"):
        g.strata(census(Mystery_v1={"calls": 1, "gates": 1}))


def test_the_card_in_the_tree_follows_the_rules():
    card = cards.load("workload", "GLM-5.3")
    v = card["value"]
    assert card["status"] == "modeled" and card["source"].startswith("art:") and v["ranks"] == g.RANKS
    assert (v["weights_bytes"], v["params"]) == (g.WEIGHTS_BYTES, g.PARAMS)
    assert v["tokens"] > 0 and v["strata"]
    for s in v["strata"]:
        unit = s["block"].split("{")[0]
        if unit in g.COORDINATES:
            assert s["macs"] == g._k(s["block"]) and s["out_bits"] == g.WORD_BITS * g.COORDINATE_WORDS.get(unit, 1)
        else:
            assert s["kind"] in g.WORDS and (s["macs"], s["out_bits"]) == (0, g.WORD_BITS), s["kind"]
        assert s["per_token"] > 0
    assert v["boundaries"]["every_block"] == pytest.approx(sum(s["per_token"] * s["out_bits"] for s in v["strata"]) / 8)
