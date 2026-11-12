"""Figures 2 and 3 from the cards, the hash loop's card, and `figures.py` end to end."""
import json
import shutil

import cards
import figure1
import figure2
import figure3
import figures
import pytest
import total_cost


@pytest.fixture
def card_dir(tmp_path, monkeypatch):
    shutil.copytree(cards.CARDS, tmp_path / "cards")
    monkeypatch.setattr(cards, "CARDS", tmp_path / "cards")
    return tmp_path / "cards"


def test_figure2_joins_the_workload_to_the_measured_blocks():
    out = figure2.run("GLM-5.3")
    work = cards.load("workload", "GLM-5.3")["value"]
    strata, macs = work["strata"], sum(s["per_token"] * s["macs"] for s in work["strata"])
    assert len(out["types"]) == len(strata) and len({t["label"] for t in out["types"]}) == len(strata)
    assert out["uncovered_mac_share"] == pytest.approx(1 - macs / (work.get("served_macs_per_token") or macs))
    assert sum(t["block_share"] for t in out["types"]) == pytest.approx(1)
    assert sum(t["mac_share"] for t in out["types"]) == pytest.approx(1 - out["uncovered_mac_share"])
    blocks = cards.load("proof_overhead", "GLM-5.3")["value"]["blocks"]
    assert [t["measured"] for t in out["types"]] == [t["block"] in {b["block"] for b in blocks} for t in out["types"]]
    assert out["pending_mac_share"] + sum(t["mac_share"] for t in out["types"] if t["measured"]) == pytest.approx(
        1 - out["uncovered_mac_share"])
    assert (out["derived_model_overhead"] is None) == (not all(t["measured"] for t in out["types"]))
    assert [s["block"] for s in out["stand_ins"]] == [b["block"] for b in blocks if b.get("stands_in")]


def test_figure2_tells_apart_the_blocks_sharing_a_name():
    """Bars sharing a Definition's name and K add the parameters that differ among them, a nested function by its
    name; blocks alike in those add their kind, and then an ordinal."""
    rows = [{"block": 'A_v1{N=1,H=2,DOT={"fn":"Dot_v1"}}', "kind": "x"},
            {"block": 'A_v1{N=2,H=2,DOT={"fn":"Dot_v2"}}', "kind": "x"},
            {"block": "B_v1{K=4,H=1}", "kind": "y"}, {"block": "B_v1{K=4,H=1}", "kind": "z"},
            {"block": "C_v1", "kind": "w"}, {"block": "C_v1", "kind": "w"}]
    assert figure2.labels(rows) == ["A_v1 N=1 DOT=Dot_v1", "A_v1 N=2 DOT=Dot_v2", "B_v1 K=4 (y)", "B_v1 K=4 (z)",
                                    "C_v1 (w) #1", "C_v1 (w) #2"]
    assert figure2.params('M{K=8,SCALE={"f64":"0x1p-4"},T=[1,2]}') == {"K": "8", "SCALE": "0x1p-4", "T": "[1,2]"}


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
    token_s = sum(s["per_token"] * b["seconds_per_block"] for s, b in zip(strata, blocks))
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
    f1 = figure1.run("GLM-5.3", 3, ("world year",), lean=False)
    share = {m["cut"]: m["share"] for m in out["modeled"]}
    assert share["every_block"] == pytest.approx(f1["scopes"]["world year"]["serve_hash_share"])
    assert share["tokens"] < 1e-6
    assert out["status"] == "card" and out["climb"] and all(s["model"] is None for s in out["climb"])
    for s in out["climb"]:
        assert all(a >= b for a, b in zip(s["best"], s["best"][1:]))
    assert {m["label"] for m in out["measured"]} == {"GLM-5.3 on 8x H200, every captured value",
                                                     "GLM-5.3 on 8x H200, the layer boundary"}


def test_macs_in_no_proof_block_are_served_and_shown(card_dir):
    """A workload card whose proof blocks hold half the served MACs, one stratum modeled: Figure 1 serves half the
    requests and its captions say what no curve bounds and what is modeled, Figure 2's shares are of the served MACs,
    and Figure 3 prices a token's serving at them, against the card with every served MAC in a proof block."""
    work = cards.load("workload", "GLM-5.3")
    work["value"]["served_macs_per_token"] = None
    cards.write(work, force=True)
    base, base3 = figure1.run("GLM-5.3", 2, ("world year",), lean=False), figure3.run("GLM-5.3")
    strata = work["value"]["strata"]
    macs = sum(s["per_token"] * s["macs"] for s in strata)
    first = next(s for s in strata if s["macs"])
    first["modeled"] = True
    work["value"]["served_macs_per_token"] = 2 * macs
    cards.write(work, force=True)
    out = figure1.run("GLM-5.3", 2, ("world year",), lean=False)
    modeled = first["per_token"] * first["macs"] / (2 * macs)
    assert out["partition"]["uncovered_mac_share"] == pytest.approx(.5)
    assert out["partition"]["modeled_mac_share"] == pytest.approx(modeled)
    assert out["scopes"]["world year"]["requests"] == pytest.approx(base["scopes"]["world year"]["requests"] / 2)
    for panel in ("exfiltration", "skipped_work"):
        caption = out["scopes"]["world year"]["captions"][panel]
        assert "50% of the served MACs are in no proof block" in caption and "no cut yet" in caption
        assert "no proof block" not in base["scopes"]["world year"]["captions"][panel]
    f2 = figure2.run("GLM-5.3")
    assert f2["uncovered_mac_share"] == pytest.approx(.5)
    assert sum(t["mac_share"] for t in f2["types"]) == pytest.approx(.5)
    assert [t["block"] for t in f2["types"] if t["modeled"]] == [first["block"]]
    assert figure3.run("GLM-5.3")["token_seconds"] == pytest.approx(2 * base3["token_seconds"])


def test_figure3_without_the_hillclimb_card_is_pending(card_dir, tmp_path):
    (card_dir / "commit_hillclimb.all.json").unlink()
    out = figure3.run("GLM-5.3")
    assert out["status"] == "pending" and out["climb"] == [] and out["measured"] == []
    assert all(p.stat().st_size > 0 for p in figure3.render(out, tmp_path))


@pytest.mark.slow
def test_figures_writes_every_figure(tmp_path, monkeypatch):
    """Every figure, its JSON and the break-even's table; the break-even's search is `test_total_cost`'s, stubbed here
    because it plans each line thousands of times."""
    monkeypatch.setattr(total_cost, "BETAS", (1e-2, 1.0))
    be = {"beta": 1e-2, "targets": {"FP8": 1e3}, "rel": 1e-2, "spans": total_cost.SPANS,
          "rows": [{"config": "tokens", "registration": "machine", "scope": "world week", "overhead": 2.0,
                    "floor": 8e6, "h": {"FP8": None}, "beta": {"FP8": 3.0}}]}
    seen = []
    monkeypatch.setattr(total_cost, "breakeven", lambda *a, **k: seen.append(k) or be)
    figures.main(["--model", "GLM-5.3", "--out", str(tmp_path), "--points", "2", "--no-lean"])
    assert seen[0]["lean"] is False
    assert json.loads((tmp_path / "figure1.json").read_text())["breakeven"] == json.loads(json.dumps(be))
    assert (tmp_path / "breakeven.txt").read_text() == total_cost.breakeven_table(be)
    for name, schema in (("figure1", total_cost.SCHEMA), ("figure1_overhead", figure1.SCHEMA),
                         ("figure2", figure2.SCHEMA), ("figure3", figure3.SCHEMA)):
        assert json.loads((tmp_path / f"{name}.json").read_text())["schema"] == schema
        assert (tmp_path / f"{name}.png").stat().st_size > 0 and (tmp_path / f"{name}.svg").exists()
    for pattern in ("figure1_overhead_appendix*", "figure1_skipped_work*"):
        assert len(list(tmp_path.glob(f"{pattern}.png"))) == 6, pattern
    assert len(list(tmp_path.glob("figure1*.png"))) == 24
