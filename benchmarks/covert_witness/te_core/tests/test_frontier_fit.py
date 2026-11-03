"""te_core.frontier_fit builds `training/frontier-fit`: per data, L(C) = E + A·C^(−γ) by least squares in relative
loss over its unconstrained frontier from the least start's C0 up, one point per 1/20 decade of C; a data with too few
points, or too short a span, has no fit, and so has one with no scored start and no ladder point; effective compute
inverts the fit, so a point on it has gain 1; the ladder's three files give v3's from-scratch points; each row carries
its record's family whichever quantities score it; a checkpoint is on its start's data whether or not its record says
so; and the panel has a fit row per data, its points with their residuals and the starts with their own gains."""
import json
import math

import pytest

pytest.importorskip("matplotlib")

from te_core import algorithm_policy as AP
from te_core import console, frontier, panels, plots, rescore
from te_core import frontier_fit as FF
from te_core.gates import per_mac
from test_algorithm_policy import C0, LAW, RATE, SCRATCH, SHAPE2, V3_SHAPE, args, law, ledger_rows
from test_plots import SHAPE


def on(f: dict, shares) -> list[tuple]:
    return [(s * C0, law(s * C0, f), {"source": f"{s:g}"}) for s in shares]


@pytest.mark.parametrize("f", [LAW, {"E": 0.5, "A": 0.5 * C0 ** 0.6, "gamma": 0.6},
                               {"E": 0.0, "A": 1.2 * C0 ** 0.08, "gamma": 0.08}])
def test_the_fit_recovers_a_law_from_points_on_it(f):
    got = FF.fit(on(f, (1, 1.7, 3, 5, 9, 16, 30, 55, 100)))
    assert got["form"] == FF.FORM and got["points"] == 9 and (got["C_lo"], got["C_hi"]) == pytest.approx((C0, 100 * C0))
    assert (got["E"], got["A"], got["gamma"]) == pytest.approx((f["E"], f["A"], f["gamma"]), rel=1e-5, abs=1e-6)
    assert got["rms_pct"] < 1e-3 and got["max_abs_pct"] < 1e-3
    assert all(float(f"{got[k]:.{FF.DIGITS}g}") == got[k] for k in ("E", "A", "gamma"))


def test_a_fit_takes_three_points_spanning_a_decade_and_is_provisional_on_fewer_than_five_from_scratch():
    """Five points from scratch is the target, not a gate (coordinator 07:58Z): a fit on fewer is provisional."""
    assert FF.fit(on(LAW, (1, 100))) is None and FF.fit(on(LAW, (1, 2, 3, 5, 7, 9))) is None
    three = FF.fit(on(LAW, (1, 10, 30)))
    assert three is not None and (three["points"], three["scratch"], three["provisional"]) == (3, 0, True)
    scratch = [(x, y, r | {"method_key": panels.SCRATCH}) for x, y, r in on(LAW, (1, 3, 10, 30, 60, 100))]
    assert [FF.fit(scratch[:n])["provisional"] for n in (4, 5, 6)] == [True, False, False]
    assert FF.fit(scratch[:4] + on(LAW, (200, 400)))["provisional"]


def test_the_fit_reads_the_frontier_from_the_least_c0_up_one_point_a_bin():
    """The staircase (the least compute at each loss) from c_lo up, and in each 1/20 decade of C its least loss."""
    pts = [(0.5 * C0, 1.3, "below c_lo"), (C0, 1.2, "start"), (1.01 * C0, 1.19, "same bin, lower"),
           (1.02 * C0, 1.195, "dominated"), (2 * C0, 1.15, "a"), (3 * C0, 1.16, "dominated too"), (4 * C0, 1.1, "b")]
    assert [r for *_, r in FF.used(pts, C0)] == ["same bin, lower", "a", "b"]
    assert [r for *_, r in FF.used(pts, 0.4 * C0)] == ["below c_lo", "same bin, lower", "a", "b"]


def test_effective_compute_inverts_the_fit():
    for c in (C0, 3 * C0, 1e3 * C0):
        assert FF.gain(LAW, FF.loss_at(LAW, c), c) == pytest.approx(1)
        assert FF.gain(LAW, FF.loss_at(LAW, 2 * c), c) == pytest.approx(2)
    assert FF.c_eff(LAW, LAW["E"]) == FF.c_eff(LAW, 0.5) == math.inf


def test_a_starts_targets_are_daniels_gain_on_the_fit_below_its_loss_with_the_fits_rms_in_compute():
    """A gain from l0 to L is C(L)/C(l0) on the fit (TRN-40): `target` inverts `gain_from`, is the fit's loss at
    g·C_eff(l0) whatever the start's own gain, and lies below l0 at every gain; a start at or below E has none.
    `c_spread` is the fit's rms in loss over its slope in log compute there."""
    for l0 in (1.0, 1.2, 1.5):
        for g in AP.GAINS:
            t = FF.target(LAW, l0, g)
            assert t < l0 and FF.gain_from(LAW, l0, t) == pytest.approx(g)
            assert t == pytest.approx(FF.loss_at(LAW, g * FF.c_eff(LAW, l0)))
    assert FF.target(LAW, LAW["E"], 2) is None and FF.gain_from(LAW, 1.2, LAW["E"]) == math.inf
    with pytest.raises(ValueError, match="at or below the fit's E"):
        FF.gain_from(LAW, LAW["E"], 1.0)
    f, h = LAW | {"rms_pct": 1.0}, 1e-6
    t = FF.target(f, 1.2, 1.1)
    c = FF.c_eff(f, t)
    slope = (FF.loss_at(f, c * (1 - h)) - FF.loss_at(f, c * (1 + h))) / (2 * h)
    assert FF.c_spread(f, t) == pytest.approx(0.01 * t / slope, rel=1e-6)
    text = FF.targets(f, 1.2, AP.GAINS)
    assert text.startswith(f"targets, C(L1)/C(L0) = g: 1.1x {t:.5f} ({100 * (t / 1.2 - 1):+.2f}%), 2x ")
    assert text.endswith(f"; the fit's rms is {100 * FF.c_spread(f, t):.3g}% in effective compute at 1.1x")
    assert FF.targets(f, 0.9, AP.GAINS) == "no targets: its loss is at or below the fit's E (0.9)"


def test_the_ladders_files_give_v3s_from_scratch_points(tmp_path):
    """Its annealed runs at their counted MACs (F/v1, or report/v2's F) and FIT.json's compute-optimal models at their
    train MACs, all at the bf16 GEMM price unless FIT.json names its own."""
    annealed = [{"branch": "s", "macs": 1e18, "bpb": 1.4}, {"size": "m", "macs": 3e18, "bpb": 1.3},
                {"size": "l", "macs": None, "bpb": 1.2}]
    optimal = {"costs": {"gates_per_mac": 2.5},
               "points": [{"train_macs": 5e18, "loss": 1.25, "non_embedding_params": 4e7}]}
    files = {"F.json": {"schema": "te-ladder/F/v1", "annealed": annealed},
             "F2.json": {"schema": "te-ladder/F/v2", "annealed": annealed},
             "report.json": {"schema": "te-ladder/report/v2", "F": {"annealed": annealed}, "compute_optimal": optimal},
             "report3.json": {"schema": "te-ladder/report/v3", "F": {"annealed": annealed}, "compute_optimal": optimal},
             "FIT.json": {"schema": "te-ladder/fit/v1"} | optimal, "FIT2.json": {"schema": "te-ladder/fit/v2"} | optimal,
             "empty.json": {"schema": "te-ladder/F/v1"}}
    for name, doc in files.items():
        (tmp_path / name).write_text(json.dumps(doc))
    gates = per_mac("gemm/bf16")
    f = [(x, y) for x, y, _ in FF.ladder_points(tmp_path / "F.json")]
    assert f == [(1e18 * gates, 1.4), (3e18 * gates, 1.3)]
    assert [r["ladder"] for *_, r in FF.ladder_points(tmp_path / "F.json")] == ["s", "m"]
    assert [(x, y) for x, y, _ in FF.ladder_points(tmp_path / "F2.json")] == f
    for name in ("report.json", "report3.json"):
        assert [(x, y) for x, y, _ in FF.ladder_points(tmp_path / name)] == f + [(5e18 * 2.5, 1.25)]
    fit = FF.ladder_points(tmp_path / "FIT.json")
    assert [(x, y) for x, y, _ in fit] == [(5e18 * 2.5, 1.25)] and fit[0][2]["data"] == "v3"
    assert fit[0][2]["source"] == "ladder fit, N 4e+07 optimal" and FF.ladder_points(tmp_path / "FIT2.json") == fit
    with pytest.raises(SystemExit, match="holds no ladder point"):
        FF.ladder_points(tmp_path / "empty.json")


def test_a_row_takes_its_records_family_from_the_analysis_whatever_the_quantities(monkeypatch):
    """A row without a family takes its record's from the analysis; the rows te_core.formalism refuses are `refused`,
    with their reasons, and a row left out for any other reason isn't."""
    rows = [{"id": "a-1", "run_id": "a"}, {"id": "b-1", "run_id": "b", "family": "full"}, {"id": "c-1", "run_id": "c"}]
    monkeypatch.setattr(rescore, "load", lambda a: {"rows": rows, "pin": ("pin",), "shapes": {}, "keep": None,
                                                    "ladder": None, "blocks": {("a", 7): ({}, "adapter"),
                                                                               ("b", 5): ({}, "seed")}})
    monkeypatch.setattr(frontier, "valid", lambda rows, pin: rows)
    monkeypatch.setattr(frontier, "formalism_rows", lambda rows, *rest: (
        rows, {"d-1": f"{frontier.REFUSED}data-dependent: not measured", "e-1": "unmeasured"}))
    d, got = FF.scored_rows(type("A", (), {"pin": ["pin"]})())
    assert [r.get("family") for r in got] == ["adapter", "full", None]
    assert d["refused"] == {"d-1": "data-dependent: not measured"}
    assert FF.refusals({"x": "data-dependent: a", "y": "read-backs unchecked: b", "z": "data-dependent: c"}) == \
        "2 data-dependent, 1 read-backs unchecked"


def test_a_checkpoint_is_on_its_starts_data_whether_or_not_its_record_says_so():
    """The scorer's measured copy of a traced record can lack the record's `data`: every checkpoint from t1, a start
    with v3's tokenizer, is on v3 in t1's one start, and the pilot's starts' checkpoints are on the pilot."""
    shapes = {"s1": SHAPE, "s2": SHAPE2, "t1": V3_SHAPE}
    s1 = [r for r in ledger_rows() if r["start"] == "s1"]
    v3 = [r | {"id": f"v3-{r['id']}", "run_id": f"v3-{r['run_id']}", "start": "t1", "start_bpb": 1.5}
          | ({"data": "v3"} if i % 2 else {}) for i, r in enumerate(s1)]
    scored = plots.load(ledger_rows() + v3, ("pin",), shapes)[0]
    n = {s: sum(r["start"] == s for r in scored) for s in shapes}
    assert n["t1"] == len(v3) and any("data" not in r for r in v3)
    assert [(s.name, s.data, len(s.rows)) for s in FF.starts(scored, shapes, RATE)] == [
        ("t1", "v3", n["t1"]), ("s1", "v2.3-pilot", n["s1"]), ("s2", "v2.3-pilot", n["s2"])]


def test_the_panel_is_each_datas_fit_its_points_and_the_starts(tmp_path, monkeypatch):
    """A fit row per data (v3, with no point yet, says why it has none), a row per point it read with the fit's loss
    there, the residual and the point's gain, and a row per start at (C0, l0) with its own gain; --publish PUTs it."""
    sent = []
    monkeypatch.setattr(console, "put", lambda q: sent.append(q) or "200")
    assert FF.main(args(tmp_path, monkeypatch) + ["--publish"]) == 0
    out = tmp_path / "out"
    p = json.loads((out / f"{FF.NAME}.json").read_text())
    assert (p["id"], p["kind"], p["columns"]) == (FF.PANEL, "table", FF.COLUMNS) and p["source"] and p["updated_at"]
    assert [q["id"] for q in sent] == [FF.PANEL] and console.body(sent[0]) == (out / "console" / f"{FF.NAME}.json").read_bytes()
    assert len(console.body(p)) <= console.BODY_MAX and not console.refused(p) and len(p["note"]) <= console.NOTE_MAX
    assert p["note"] == FF.note() and "0 ladder points)" in p["source"]
    ds = plots.frame(p)
    fits = {d["data"]: d for d in ds if d["row"] == "fit"}
    assert fits["v3"]["E_bpb"] is None and fits["v3"]["status"].startswith("no fit: no start has a scored checkpoint")
    pilot = fits["v2.3-pilot"]
    f = {"E": pilot["E_bpb"], "A": pilot["A"], "gamma": pilot["gamma"]}
    assert (f["E"], f["gamma"]) == pytest.approx((LAW["E"], LAW["gamma"]), rel=0.05)
    pts = [d for d in ds if d["row"] == "point"]
    assert pilot["points"] == len(pts) >= len([s for s in SCRATCH if s >= 1])
    assert {d["data"] for d in pts} == {"v2.3-pilot"} and min(d["C_gates"] for d in pts) == pytest.approx(C0, rel=1e-5)
    assert (pts[0]["loss_bpb"], pts[0]["source"]) == (1.2, "start s1")
    for d in pts:
        assert d["fit_bpb"] == pytest.approx(FF.loss_at(f, d["C_gates"]), abs=1e-5)
        assert d["residual_pct"] == pytest.approx(100 * (d["loss_bpb"] / FF.loss_at(f, d["C_gates"]) - 1), abs=1e-3)
        assert d["gain"] == pytest.approx(FF.c_eff(f, d["loss_bpb"]) / d["C_gates"], rel=1e-3)
    rms = math.sqrt(sum(d["residual_pct"] ** 2 for d in pts) / len(pts))
    assert pilot["rms_pct"] == pytest.approx(rms, rel=0.01, abs=1e-3)
    starts = [(d["source"].split(" (")[0], d["C_gates"], d["loss_bpb"]) for d in ds if d["row"] == "start"]
    assert starts == [("start s1", pytest.approx(C0, rel=1e-5), 1.2), ("start s2", pytest.approx(2 * C0, rel=1e-5), 1.3)]
    whole = f | {"rms_pct": pilot["rms_pct"]}
    assert [d["status"] for d in ds if d["row"] == "start"] == [FF.targets(whole, l0, AP.GAINS) for l0 in (1.2, 1.3)]
    spread = sorted(100 * FF.c_spread(whole, FF.target(whole, l0, 1.1)) for l0 in (1.2, 1.3))
    assert pilot["status"].endswith(f"; its rms in loss is {spread[0]:.3g}% to {spread[1]:.3g}% in effective compute "
                                    "at its starts' 1.1x targets")
    every = json.loads((out / "frontier-points.json").read_text())
    assert set(every) == {"v3", "v2.3-pilot"} and every["v3"] == [] and len(every["v2.3-pilot"]) >= len(pts)


def test_a_data_with_no_scored_start_has_no_fit(tmp_path, monkeypatch):
    """Every row refused (traced before te_core 4.2's read-back lists): the pilot's from-scratch run is still there, but
    with no start nothing converts its losses or prices C0, so the pilot has no fit rather than one on stubs."""
    rows = [r | {"formalism": r["formalism"] | {"trace": {"affected_share": 0.5, "u1_bits": 1e9}}}
            for r in ledger_rows()]
    assert FF.main(args(tmp_path, monkeypatch, index=rows)) == 0
    p = json.loads((tmp_path / "out" / f"{FF.NAME}.json").read_text())
    fits = [d for d in plots.frame(p) if d["row"] == "fit"]
    assert len(fits) == 2 and all(d["E_bpb"] is None and d["points"] == 0 for d in fits)
    assert all(d["status"].startswith("no fit: no start has a scored checkpoint") for d in fits)
    assert f"; {len(rows)} refused; " in p["source"]


def test_the_source_fits_the_route_its_pins_at_7_characters_where_12_dont(tmp_path, monkeypatch):
    """The panels route answers 400 to a source over 300 characters, as it did at 09:30Z to the scorer's eight pins
    beside the ladder's manifest: each command's source names its pins at 12 characters where that fits, else at 7."""
    tree = tmp_path / "ladder"
    tree.mkdir()
    (tree / FF.LADDER_FILE).write_text(json.dumps({"schema": "te-ladder/F/v1",
                                                   "annealed": [{"branch": "s", "macs": 1e18, "bpb": 1.4}] * 5}))
    monkeypatch.setattr(FF, "LADDER", "art:" + "b" * 64)
    monkeypatch.setattr(panels, "fetch", lambda art: tree)
    monkeypatch.setattr(panels, "pretraining", lambda path: [])
    a = type("A", (), {"pretraining": tmp_path, "ladder_points": [], "ladder_fit": None, "source": None, "pin": True})()
    for n, width in ((8, 12), (9, 12), (12, 7)):
        d = {"pin": [f"{i:x}" * 40 for i in range(n)], "quantities": "measured, t literal",
             "refused": dict.fromkeys(map(str, range(3956)))}
        for command in ("te_core.frontier_fit", "te_core.algorithm_policy"):
            source = FF.inputs(a, [{}] * 2958, {}, d, command)[2]
            assert len(source) <= console.SOURCE_MAX and f"; pin {'0' * width} + {'1' * width} + " in source
            assert source.endswith("; 3956 refused; 5 ladder points (art:bbbbbbbbbbbb))")
