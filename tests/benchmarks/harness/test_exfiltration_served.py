"""`verity bench exfiltration`'s share of compute a workload measures."""

from __future__ import annotations

import re

import pytest
from exfiltration_helpers import REPO, SERVED, YEAR_GB, recorded
from verity_bench import exfiltration as E

#: what the prove part's source says of its seconds, read from the proof server's busy seconds
BUSY = "`prove`, the proof server's busy seconds a card (`prove_seconds_source`: `busy`)"


def test_with_no_workload_the_share_is_not_measured_and_the_given_share_is_assumed(tmp_path):
    doc = E.build(store=recorded(tmp_path).store, root=REPO)
    cs = doc["compute_share"]
    assert (cs["run"], cs["value"], cs["mark"], cs["why"], cs["GB"]) == (None, None, None, E.NO_RUN, None)
    assert cs["given"] == {"share": 0.1, "mark": "assumed", "GB": doc["value"]} and doc["value"] == pytest.approx(YEAR_GB)
    by = {i["input"]: i for i in doc["inputs"]}
    assert by["compute_share"]["mark"] == "assumed" and E.NO_RUN in by["compute_share"]["source"]
    assert not any(i["input"].startswith("served_") or i["mark"] == "missing" for i in doc["inputs"])
    text = E.render(doc)
    assert ("\n\n**The audit's share of compute: not measured: no served run's record.** Every figure below is read at the "
            "given 10% of total compute (assumed).\n\n") in text
    assert "- **the compute audit's part: 14,240 GB** (modeled)" in text and "## The audit's share of compute on" not in text


def test_a_workloads_record_gives_the_share_measured_and_the_part_at_it(tmp_path):
    s = recorded(tmp_path)
    doc = E.build(served=SERVED, store=s.store, root=REPO)
    cs = doc["compute_share"]
    assert (cs["run"], cs["mark"], cs["why"], cs["missing"], cs["serving_gpu_s"]) == (SERVED["run"], "measured", None, [], 8000.0)
    assert cs["value"] == pytest.approx(0.12, rel=1e-12)
    assert {k: (p["gpu_s"], p["mark"]) for k, p in cs["parts"].items()} == {
        "prove": (480.0, "measured"), "staging": (20.0, "measured"), "replay": (240.0, "measured"),
        "interior": (20.0, "measured"), "serving_charge": (200.0, "measured")}
    # the part at the measured share is the part the command reads at that share given; the given 10%'s is beside it
    at = E.build(share=0.12, store=s.store, root=REPO)["value"]
    assert cs["GB"] == pytest.approx(at, rel=1e-12) and cs["inside"] and cs["no_GB"] is None
    assert cs["given"] == {"share": 0.1, "mark": "assumed", "GB": doc["value"]} and doc["value"] == pytest.approx(YEAR_GB)
    by = {i["input"]: i for i in doc["inputs"]}
    assert {k: by[f"served_{k}_gpu_s"]["mark"] for k in (*E.SERVED_PARTS, "serving")} == dict.fromkeys(
        (*E.SERVED_PARTS, "serving"), "measured")
    assert (by["served_compute_share"]["mark"], by["served_compute_share"]["value"]) == ("measured", cs["value"])
    assert by["compute_share"]["mark"] == "assumed" and SERVED["run"] in by["compute_share"]["source"]
    text = E.render(doc)
    assert (f"**The audit's share of compute: 12.000% (measured)**, on {SERVED['run']}: the audit's GPU-seconds over the "
            "window's serving, 8,000 GPU-seconds, in parts prove 6.000%, staging 0.250%, replay 3.000%, interior 0.250%, "
            f"serving charge 2.500%. At that share, β = {E._part(cs['beta'])}, the compute audit's part over the world's "
            f"year is {E._n(at)} GB, beside 14,240 GB at the given 10% of total compute (assumed)") in text
    assert f"| serving_charge | {E.SERVED_PARTS['serving_charge'][0]} | 200 | 2.500% | measured | the served GPU-seconds less" in text

    # with no stock serve of the load, the serving charge is the serve tax's cost of the arm over stock, modeled
    cost = {k: v for k, v in SERVED.items() if k != "stock"} | {
        "serving_cost": 0.025 / 0.975, "serving_cost_source": "r20261110-000000-0000, load chat"}
    cs = E.build(served=cost, store=s.store, root=REPO)["compute_share"]
    assert cs["value"] == pytest.approx(0.12, rel=1e-12) and cs["mark"] == "modeled"
    assert cs["parts"]["serving_charge"]["mark"] == "modeled" and "r20261110-000000-0000, load chat" in cs["parts"][
        "serving_charge"]["source"]


def test_a_field_the_record_lacks_names_its_gap_and_no_share_is_claimed(tmp_path):
    s = recorded(tmp_path)
    lacking = {k: v for k, v in SERVED.items() if k not in ("staging", "stock")} | {
        "prove": [{"seconds": 300.0}, {"seconds": 180.0, "gpus": 1}]}
    doc = E.build(served=lacking, store=s.store, root=REPO)
    cs = doc["compute_share"]
    assert (cs["value"], cs["mark"], cs["GB"]) == (None, None, None) and doc["value"] == pytest.approx(YEAR_GB)
    assert {g["field"]: g["lacks"] for g in cs["missing"]} == {
        "prove": ["prove.gpus"], "staging": ["staging"], "serving_charge": ["stock"]}
    assert cs["why"] == "not measured: " + "; ".join(
        f"{SERVED['run']}'s record lacks {lacks} ({E.SERVED_PARTS[k][1]})"
        for k, lacks in (("prove", "`prove.gpus`"), ("staging", "`staging`"), ("serving_charge", "`stock`")))
    assert {k: p["mark"] for k, p in cs["parts"].items()} == {
        "prove": "missing", "staging": "missing", "replay": "measured", "interior": "measured", "serving_charge": "missing"}
    by = {i["input"]: i for i in doc["inputs"]}
    assert {k: by[k]["mark"] for k in ("served_prove_gpu_s", "served_staging_gpu_s", "served_replay_gpu_s",
                                       "served_compute_share")} == {
        "served_prove_gpu_s": "missing", "served_staging_gpu_s": "missing", "served_replay_gpu_s": "measured",
        "served_compute_share": "missing"}
    assert E.SERVED_PARTS["staging"][1] in by["served_staging_gpu_s"]["source"]
    text = E.render(doc)
    assert f"\n\n**The audit's share of compute: {cs['why']}.** Every figure below is read at the given 10%" in text
    assumed = sum(i["mark"] == "assumed" for i in doc["inputs"])
    assert f"modeled, 1 simulated, {assumed} assumed and 4 missing." in text
    assert "| staging | " + E.SERVED_PARTS["staging"][0] + " | none | none | missing | `staging`; lacks `staging`: " in text
    # the window's serving lacking: every part is unclaimed as a share, and the gap is the serve's
    cs = E.served_share({k: v for k, v in SERVED.items() if k != "serving"})
    assert cs["value"] is None and cs["missing"][0] == {"field": "serving", "lacks": ["serving"], "why": E.SERVING_GAP}
    assert {p["share"] for p in cs["parts"].values()} == {None}

    # a part on 0 GPUs started no process on a card: it is read, and adds no GPU-seconds
    cs = E.served_share(SERVED | {"staging": [{"seconds": 20.0, "gpus": 0}]})
    assert cs["missing"] == [] and cs["parts"]["staging"]["gpu_s"] == 0.0 and cs["value"] == pytest.approx(0.1175)
    for bad, why in (({"replay": [{"seconds": 30.0, "gpus": -1}]}, "holds replay at 30.0 seconds on -1 GPUs"),
                     ({"serving": {"seconds": 1000.0, "gpus": 0}}, "holds the window's serving at 0.0 GPU-seconds"),
                     ({"serving": {"seconds": 0.0, "gpus": 8}}, "holds the window's serving at 0.0 GPU-seconds"),
                     ({"stock": None, "serving_cost": -1}, "holds a serving cost of -1: a fraction over -1")):
        with pytest.raises(ValueError, match=f"{SERVED['run']}'s record {why}"):
            E.served_share(SERVED | bad)
    with pytest.raises(ValueError, match="a workload's record is one JSON object, not list"):
        E.served_share([SERVED])


def test_the_proving_seconds_source_marks_the_proving_and_the_share(tmp_path):
    """`prove_seconds_source`: `busy`, the proof server's busy seconds a card, is measured; `session`, a session's
    seconds with its wait included, prints the proving and the share as measured, at most (an upper bound); absent,
    `prove` lacks it and no share is claimed; any other value is refused by name. Both read seconds held times GPUs
    held."""

    s = recorded(tmp_path)
    busy = E.build(served=SERVED, store=s.store, root=REPO)
    cs = busy["compute_share"]
    assert (cs["mark"], cs["parts"]["prove"]["mark"], cs["parts"]["prove"]["source"]) == ("measured", "measured", BUSY)
    assert f"**The audit's share of compute: 12.000% (measured)**, on {SERVED['run']}" in E.render(busy)

    twice = SERVED | {"prove": [{"seconds": 150.0, "gpus": 2}, {"seconds": 180.0, "gpus": 1}]}
    doc = E.build(served=twice | {"prove_seconds_source": "session"}, store=s.store, root=REPO)
    cs = doc["compute_share"]
    assert (cs["value"], cs["mark"]) == (pytest.approx(0.12, rel=1e-12), "measured, at most")
    assert (cs["parts"]["prove"]["gpu_s"], cs["parts"]["prove"]["mark"]) == (480.0, "measured, at most")
    assert {p["mark"] for k, p in cs["parts"].items() if k != "prove"} == {"measured"}
    by = {i["input"]: i for i in doc["inputs"]}
    assert (by["served_prove_gpu_s"]["mark"], by["served_compute_share"]["mark"]) == ("measured, at most",) * 2
    text = E.render(doc)
    assert f"**The audit's share of compute: 12.000% (measured, at most)**, on {SERVED['run']}" in text
    assert (f"| prove | {E.SERVED_PARTS['prove'][0]} | 480 | 6.000% | measured, at most | `prove`, a session's seconds, "
            "its wait included: an upper bound (`prove_seconds_source`: `session`) |") in text
    assert "2 measured upper bounds" in text
    modeled = {k: v for k, v in SERVED.items() if k != "stock"} | {
        "serving_cost": 0.025 / 0.975, "serving_cost_source": "r20261110-000000-0000, load chat",
        "prove_seconds_source": "session"}
    assert E.served_share(modeled)["mark"] == "modeled, at most"

    absent = E.build(served={k: v for k, v in SERVED.items() if k != "prove_seconds_source"}, store=s.store, root=REPO)
    cs = absent["compute_share"]
    assert (cs["value"], cs["mark"], cs["GB"]) == (None, None, None)
    assert cs["missing"] == [{"field": "prove", "lacks": ["prove_seconds_source"], "why": E.PROVE_SOURCE_GAP}]
    assert cs["why"] == f"not measured: {SERVED['run']}'s record lacks `prove_seconds_source` ({E.PROVE_SOURCE_GAP})"
    assert (cs["parts"]["prove"]["gpu_s"], cs["parts"]["prove"]["mark"]) == (480.0, "missing")
    assert f"\n\n**The audit's share of compute: {cs['why']}.** Every figure below" in E.render(absent)
    both = E.served_share({k: v for k, v in SERVED.items() if k != "prove_seconds_source"}
                          | {"prove": [{"seconds": 300.0}]})
    assert both["missing"] == [{"field": "prove", "lacks": ["prove.gpus", "prove_seconds_source"],
                                "why": f"{E.SERVED_PARTS['prove'][1]}; {E.PROVE_SOURCE_GAP}"}]

    with pytest.raises(ValueError, match=re.escape(
            f"{SERVED['run']}'s record holds `prove_seconds_source` 'timer': one of busy (the proof server's busy "
            "seconds a card), session (a session's seconds, its wait included: an upper bound)")):
        E.served_share(SERVED | {"prove_seconds_source": "timer"})


def test_a_part_on_cpus_alone_adds_no_gpu_seconds_and_says_so_beside_its_0(tmp_path):
    """The audit's driver writes 0 GPUs for a step that started no process on a card (staging, the interior commitment,
    a session proved on a CPU): it is read and adds no GPU-seconds, and a part whose entries are all on 0 GPUs is
    printed `ON_CPUS` beside its 0 wherever its 0 is printed, so the row doesn't read as GPU time given free. A part
    with one entry on a GPU is not marked; a count that is not a whole number is refused."""

    cpu = SERVED | {"staging": [{"seconds": 20.0, "gpus": 0}], "interior": [{"seconds": 5.0, "gpus": 0}],
                    "prove": [{"seconds": 300.0, "gpus": 1}, {"seconds": 180.0, "gpus": 0}]}
    doc = E.build(served=cpu, store=recorded(tmp_path).store, root=REPO)
    cs = doc["compute_share"]
    assert {k: (p["gpu_s"], p["on_cpus"]) for k, p in cs["parts"].items()} == {
        "prove": (300.0, False), "staging": (0.0, True), "replay": (240.0, False), "interior": (0.0, True),
        "serving_charge": (200.0, False)}
    assert cs["missing"] == [] and cs["value"] == pytest.approx(740 / 8000)
    text = E.render(doc)
    assert (f"in parts prove 3.750%, staging 0.000% ({E.ON_CPUS}), replay 3.000%, interior 0.000% ({E.ON_CPUS}), "
            "serving charge 2.500%.") in text
    assert f"| staging | {E.SERVED_PARTS['staging'][0]} | 0, {E.ON_CPUS} | 0.000% | measured | `staging` |" in text
    assert f"| prove | {E.SERVED_PARTS['prove'][0]} | 300 | 3.750% | measured | {BUSY} |" in text
    by = {i["input"]: i for i in doc["inputs"]}
    assert by["served_interior_gpu_s"]["source"].endswith(f"`interior`; {E.ON_CPUS}")
    assert E.ON_CPUS not in by["served_prove_gpu_s"]["source"] and E.ON_CPUS not in E.render(
        E.build(served=SERVED, store=recorded(tmp_path / "gpus").store, root=REPO))
    for bad in (True, 1.0, "1"):
        with pytest.raises(ValueError, match=re.escape(f"holds staging at 20.0 seconds on {bad!r} GPUs")):
            E.served_share(SERVED | {"staging": [{"seconds": 20.0, "gpus": bad}]})
