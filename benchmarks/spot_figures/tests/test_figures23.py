"""Figures 2 and 3 from the cards, the hash loop's card, and `figures.py` end to end."""
import json
import shutil

import cards
import figure1
import figure2
import figure3
import figures
import pytest


@pytest.fixture
def card_dir(tmp_path, monkeypatch):
    shutil.copytree(cards.CARDS, tmp_path / "cards")
    monkeypatch.setattr(cards, "CARDS", tmp_path / "cards")
    return tmp_path / "cards"


def test_figure2_joins_the_workload_to_the_measured_blocks():
    out = figure2.run("GLM-5.3")
    strata = cards.load("workload", "GLM-5.3")["value"]["strata"]
    assert len(out["types"]) == len(strata) and len({t["label"] for t in out["types"]}) == len(strata)
    assert sum(t["block_share"] for t in out["types"]) == pytest.approx(1)
    assert sum(t["mac_share"] for t in out["types"]) == pytest.approx(1)
    assert out["pending_mac_share"] == pytest.approx(1) and out["derived_model_overhead"] is None
    assert [s["stands_in"] for s in out["stand_ins"]] == [True]


def test_figure2_derives_the_model_overhead_once_every_type_is_measured(card_dir, tmp_path):
    po = cards.load("proof_overhead", "GLM-5.3")
    strata = cards.load("workload", "GLM-5.3")["value"]["strata"]
    peak = po["value"]["peak_mac_s"]
    blocks = [{"block": s["block"], "macs": s["macs"], "seconds_per_block": 1e-4 * (1 + s["macs"] / 1e3),
               "fixed_seconds": 0.2, "statement_blocks": 2048, "overhead": None} for s in strata]
    po["value"]["blocks"] = blocks
    cards.write(po, force=True)
    out = figure2.run("GLM-5.3")
    assert out["pending_mac_share"] == 0 and out["stand_ins"] == []
    token_s = sum(s["per_token"] * (b["seconds_per_block"] + 0.2 / 2048) for s, b in zip(strata, blocks))
    macs = sum(s["per_token"] * s["macs"] for s in strata)
    assert out["derived_model_overhead"] == pytest.approx(token_s * peak / macs)


def test_the_hash_loop_card_keeps_its_sha512_arms_with_their_runs():
    loop = {"results": [{"row": 0, "hardware": "H200", "run": "r20261018-031955-d2b3", "valid": True},
                        {"row": 1, "hardware": "H200", "run": "r20261018-171513-ed13", "valid": True},
                        {"row": 1, "hardware": "H200", "run": "r20261018-171513-ffff", "valid": False},
                        {"row": "what-if", "hardware": "H200", "run": "r20261018-231625-ec7e", "valid": True}],
            "series": {"tpot_overhead_pct": {"H200/qwen3-8b:sha512": [[0, 190.0], [1, 68.0]],
                                             "H200/qwen3-8b:blake3": [[1, 40.0]],
                                             "8x H200/glm-5.3:sha512-layer": [[0, 12.8]],
                                             "RTX PRO 6000/pouw:sha512": [[0, 156.8]]}}}
    card = cards.check(figure3.loop_card(loop, "art:loop"))
    by = {s["label"]: s for s in card["value"]["series"]}
    assert set(by) == {"Qwen3-8B on H200, every captured value", "GLM-5.3 on 8x H200, the layer boundary"}
    qwen = by["Qwen3-8B on H200, every captured value"]
    assert qwen["model"] is None and [(p["x"], p["y"], p["runs"]) for p in qwen["points"]] == [
        (0, 1.9, ["r20261018-031955-d2b3"]), (1, .68, ["r20261018-171513-ed13"])]
    assert by["GLM-5.3 on 8x H200, the layer boundary"]["model"] == "GLM-5.3"
    assert figure3.best([{"y": 3}, {"y": 1}, {"y": 2}]) == [3, 1, 1]


def test_figure3_prices_the_cuts_as_figure1_does():
    out = figure3.run("GLM-5.3")
    f1 = figure1.run("GLM-5.3", 3, ("world year",))
    share = {m["cut"]: m["share"] for m in out["modeled"]}
    assert share["every_block"] == pytest.approx(f1["scopes"]["world year"]["serve_hash_share"])
    assert share["tokens"] < 1e-6
    assert out["status"] == "card" and out["climb"] and all(s["model"] is None for s in out["climb"])
    for s in out["climb"]:
        assert all(a >= b for a, b in zip(s["best"], s["best"][1:]))
    assert {m["label"] for m in out["measured"]} == {"GLM-5.3 on 8x H200, every captured value",
                                                     "GLM-5.3 on 8x H200, the layer boundary"}


def test_figure3_without_the_hillclimb_card_is_pending(card_dir, tmp_path):
    (card_dir / "commit_hillclimb.all.json").unlink()
    out = figure3.run("GLM-5.3")
    assert out["status"] == "pending" and out["climb"] == [] and out["measured"] == []
    assert all(p.stat().st_size > 0 for p in figure3.render(out, tmp_path))


@pytest.mark.slow
def test_figures_writes_every_figure(tmp_path):
    figures.main(["--model", "GLM-5.3", "--out", str(tmp_path), "--points", "2"])
    for n, schema in ((1, figure1.SCHEMA), (2, figure2.SCHEMA), (3, figure3.SCHEMA)):
        assert json.loads((tmp_path / f"figure{n}.json").read_text())["schema"] == schema
        assert (tmp_path / f"figure{n}.png").stat().st_size > 0 and (tmp_path / f"figure{n}.svg").exists()
    assert len(list(tmp_path.glob("figure1*.png"))) == 6
