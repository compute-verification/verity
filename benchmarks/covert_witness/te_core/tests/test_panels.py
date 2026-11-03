"""te_core.panels builds `training/frontier-branches`: the frontier over the from-scratch runs and every start's
continuations, each start's honest and throttled curves at total compute C0 + after, S = W*/H >= 1, `on_frontier`
when no from-scratch point beats the start, the dial cells whose curve is the honest one left out, the reference and
the served demo's block marked, every curve on one data with the console showing v3 once it has points, the scoring
cap, each row's training/methods key, the candidates whose columns give the throttled rows' ρ, and a console copy
that fits the panels route."""
import json
import math

import pytest

pytest.importorskip("matplotlib")

from te_core import algorithm_policy, console, formalism, frontier, panels, plots, policy, rescore
from test_algorithm_policy import dial
from test_plots import SHAPE, rows

RATIO = 1.03
L_PARAMS = next(iter(panels.FORWARD_MACS))
C = {c: i for i, c in enumerate(panels.COLUMNS)}
W = policy.w_fwd(SHAPE)


def ledger_rows(unbounded=False):
    return [r | ({"mechanism": "adamw-all"} if r["direction"] == "ref-optim" else {}) for r in rows(unbounded)]


V3_SHAPE = SHAPE | {"tokenizer": policy.V3_TOKENIZER}


def with_v3(start="t1"):
    """The pilot's rows and the same records again on v3's data, from `start` (default t1, a v3 start) at l0 = 1.3."""
    pilot = ledger_rows()
    return pilot + [r | {"id": f"{r['id']}-v3", "run_id": f"{r['run_id']}-v3", "data": "v3", "start": start,
                         "start_bpb": 1.3, "val_bpb": r["val_bpb"] + 0.1} for r in pilot]


def pretraining(root, beaten: bool):
    """An s run through the start's checkpoint (l0 = 1.2 there, so the ratio is `RATIO`) and an l run whose one cheap
    point lies below the start (`beaten`) or above it, and a deep one beyond every continuation."""
    def run(name, size, params, pts):
        d = root / name
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"args": {"size": size}, "counts": {"total": params}}))
        (d / "curve.json").write_text(json.dumps([{"step": i, "tokens": t, "val_nats_per_byte": b * math.log(2)}
                                                  for i, (t, b) in enumerate(pts)]))
    run("lf-s", "s", SHAPE["parameters"], [(SHAPE["tokens"] // 2, 1.30), (SHAPE["tokens"], 1.2 / RATIO)])
    run("lf-l", "l", L_PARAMS, [(1000, (1.19 if beaten else 1.25) / RATIO), (10 ** 12, 1.0 / RATIO)])
    return root


def build(tmp_path, monkeypatch, beaten=False, extra=(), index=None, unbounded=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    man = tmp_path / "starts_store.json"
    man.write_text(json.dumps({"starts": {"s1": {"shape": SHAPE}, "t1": {"shape": V3_SHAPE}}}))
    monkeypatch.setattr(rescore, "Ledger", lambda root: type("L", (), {"index": staticmethod(
        index or (lambda: ledger_rows(unbounded))), "root": tmp_path})())
    out = tmp_path / "out"
    code = panels.main([panels.NAME, "--out", str(out), "--pin", "pin", "--manifest", str(man), "--stubs",
                        "--pretraining", str(pretraining(tmp_path / "lf", beaten)), *extra])
    return code, out


@pytest.mark.parametrize("beaten", [False, True])
def test_the_frontier_and_its_branches(tmp_path, monkeypatch, beaten):
    code, out = build(tmp_path, monkeypatch, beaten)
    assert code == 0 and (out / f"{panels.NAME}.png").stat().st_size > 10_000
    p = json.loads((out / f"{panels.NAME}.json").read_text())
    assert (p["id"], p["kind"], p["columns"]) == (panels.PANEL, "table", panels.COLUMNS) and p["source"]
    assert all(len(r) == len(p["columns"]) for r in p["rows"])
    ds = plots.frame(p)
    assert {d["curve"] for d in ds} == {"frontier", "honest", "throttled", "candidate"}
    assert {d["data"] for d in ds} == {"v2.3-pilot"}
    fr = [d for d in ds if d["curve"] == "frontier"]
    assert all(a["compute_total_gates"] < b["compute_total_gates"] and a["loss"] > b["loss"] for a, b in zip(fr, fr[1:]))
    assert fr[-1]["loss"] == pytest.approx(1.0) and fr[0]["method"].endswith("loss converted")
    on = [d for d in fr if d["method"].endswith("the start s1")]
    assert bool(on) != beaten and {d["on_frontier"] for d in ds if d["start"]} == {"no" if beaten else "yes"}
    branch = [d for d in ds if d["start"]]
    assert {d["start_loss"] for d in branch} == {1.2} and {d["start"] for d in branch} == {plots.start_label(SHAPE)}
    if on:
        assert (on[0]["compute_total_gates"], on[0]["loss"]) == (branch[0]["start_compute_gates"], 1.2)
    assert all(d["compute_total_gates"] == pytest.approx(d["start_compute_gates"] + d["compute_after_gates"])
               for d in branch)
    honest = [(d["compute_after_gates"], d["run"]) for d in branch if d["curve"] == "honest"]
    assert [d["loss"] for d in branch if d["curve"] == "honest"] == [1.17, 1.15]
    cells: dict[tuple, list] = {}
    for d in branch:
        if d["curve"] == "throttled":
            cells.setdefault((d["c_tokens_per_block"], d["k_bits"]), []).append(d)
    assert cells and all([(d["compute_after_gates"], d["run"]) for d in g] != honest for g in cells.values())
    h = {d["loss"]: d["compute_after_gates"] for d in branch if d["curve"] == "honest"}
    for g in cells.values():
        for d in g:
            least = min(x for y, x in h.items() if y <= d["loss"])
            assert d["S"] == pytest.approx(d["compute_after_gates"] / least) and d["S"] >= 1
            assert d["basis"] in {"fit", "target"}
            assert d["c_gates"] == pytest.approx(d["c_tokens_per_block"] * W)
    marked = {d["mark"]: (d["c_tokens_per_block"], d["c_gates"], d["k_bits"]) for d in branch if d["mark"]}
    assert set(marked) == {"reference", "served demo"}
    assert marked["reference"][::2] == panels.DEFAULT
    assert marked["served demo"] == pytest.approx((panels.SERVED[0] / W, panels.SERVED[0], panels.SERVED[1]))
    q = json.loads((out / "console" / f"{panels.NAME}.json").read_text())
    assert len(console.body(q)) <= console.BODY_MAX and len(q["note"]) <= console.NOTE_MAX and not console.refused(q)
    assert f"x{RATIO:.4f}" in q["note"] and "best unconstrained/constrained learner" in q["note"]
    assert "(v2.3-pilot)" in q["note"] and "candidate: their checkpoints, with rho's inputs" in q["note"]


def test_v3_is_the_headline_once_it_has_points(tmp_path, monkeypatch):
    """Each data has its own branch points and frontier (a loss on one is none on the other); the uncut panel keeps
    both and the console copy and figure show v3's."""
    code, out = build(tmp_path, monkeypatch, index=with_v3)
    assert code == 0
    ds = plots.frame(json.loads((out / f"{panels.NAME}.json").read_text()))
    assert {d["data"] for d in ds if d["curve"] == "frontier"} == {"v2.3-pilot", "v3"}
    v3 = [d for d in ds if d["data"] == "v3"]
    assert {d["start_loss"] for d in v3 if d["start"]} == {1.3} and {d["on_frontier"] for d in v3 if d["start"]} == {"yes"}
    assert not any("learning_frontier" in d["method"] for d in v3)
    assert [d["loss"] for d in v3 if d["curve"] == "honest"] == pytest.approx([1.27, 1.25])
    pilot = [d for d in ds if d["data"] == "v2.3-pilot" and d["curve"] == "honest"]
    assert [d["loss"] for d in pilot] == [1.17, 1.15]
    q = json.loads((out / "console" / f"{panels.NAME}.json").read_text())
    assert {r[C["data"]] for r in q["rows"]} == {"v3"} and "(v3)" in q["note"] and "From scratch: the starts" in q["note"]


def test_a_v3_record_from_a_pilot_start_is_no_v3_branch_point(tmp_path, monkeypatch):
    """A record on v3's data from a start whose shape has no v3 tokenizer (the pilot's FineWeb-Edu starts) is left out
    (`pilot-start`), so the panel has no v3 branch point and stays on the pilot."""
    rows = with_v3("s1")
    _, skipped = frontier.formalism_rows(rows, {"s1": SHAPE})
    assert skipped == {r["id"]: "pilot-start" for r in rows if r.get("data") == "v3"}
    code, out = build(tmp_path, monkeypatch, index=lambda: rows)
    ds = plots.frame(json.loads((out / f"{panels.NAME}.json").read_text()))
    assert code == 0 and {d["data"] for d in ds} == {"v2.3-pilot"}


def test_the_scoring_cap(tmp_path, monkeypatch):
    """With no ladder fit a checkpoint whose model's forward is over twice its start's is left out (`over-cap`), so a
    grown model up to that is scored; where the fit has a value at its loss the cap is the greater of the start's and
    the fit's, and outside the fit's span it is twice the start's again. One with no count has the start's."""
    grown = [r | ({"model": {"forward_gates_per_token": 2.5 * W}} if r["id"] == "r1" else {}) for r in ledger_rows()]
    fit = tmp_path / "fit.json"
    fit.write_text(json.dumps({"points": [{"loss": 1.0, "forward_gates_per_token": 8 * W},
                                          {"loss": 1.3, "forward_gates_per_token": W}]}))
    ladder = policy.ladder_fit(fit)
    assert ladder(1.15) == pytest.approx(8 ** 0.5 * W) and ladder(0.9) is None and ladder(1.31) is None
    assert policy.CAP_FALLBACK == 2
    assert policy.eligible(None, W, 1.15) and policy.eligible(2 * W, W, 1.15) and not policy.eligible(2.5 * W, W, 1.15)
    assert policy.eligible(2.5 * W, W, 1.15, ladder) and not policy.eligible(1.5 * W, W, 1.29, ladder)
    assert policy.eligible(2 * W, W, 0.9, ladder) and not policy.eligible(2.5 * W, W, 0.9, ladder)
    _, skipped = frontier.formalism_rows(grown, {"s1": SHAPE})
    assert skipped == {"r1": "over-cap"}
    assert frontier.formalism_rows(grown, {"s1": SHAPE}, ladder)[1] == {}
    runs = {}
    for name, extra in (("capped", ()), ("fit", ("--ladder-fit", str(fit)))):
        _, out = build(tmp_path / name, monkeypatch, extra=extra, index=lambda: grown)
        runs[name] = {d["run"] for d in plots.frame(json.loads((out / f"{panels.NAME}.json").read_text()))}
    assert "r1@9" not in runs["capped"] and "r1@9" in runs["fit"]


def test_an_infinite_rho_is_on_no_throttled_curve(tmp_path, monkeypatch):
    """r4, the deepest record, reads its whole update (capped at Δ = 8 Mbit) at once: its ρ is infinite in every dial
    cell with k < Δ (at k = Δ, x0 is 1), where it is on no throttled curve, while it stays on the honest one and the
    panel strict JSON."""
    code, out = build(tmp_path, monkeypatch, unbounded=True)
    ds = plots.frame(json.loads((out / f"{panels.NAME}.json").read_text()))
    assert code == 0 and "r4@9" in {d["run"] for d in ds if d["curve"] == "honest"}
    low = [d for d in ds if d["curve"] == "throttled" and d["k_bits"] < 8e6]
    assert low and "r4@9" not in {d["run"] for d in low}


def test_dense_adamw_sets_the_rate_c0_is_counted_at(tmp_path, monkeypatch):
    """No counted training gates in the start's shape: C0 is its tokens at the median gates per token of the AdamW
    checkpoints, in forward passes of the model."""
    _, out = build(tmp_path, monkeypatch)
    p = plots.frame(json.loads((out / f"{panels.NAME}.json").read_text()))
    scored, _ = plots.load(ledger_rows(), ("pin",), {"s1": SHAPE})
    rate, how = panels.dense_rate(scored, {"s1": SHAPE})
    assert how.startswith("counted: the median of 2 ") and rate > 0
    assert p[-1]["start_compute_gates"] == pytest.approx(rate * W * SHAPE["tokens"], rel=1e-3)


def test_every_row_has_its_methods_key(tmp_path, monkeypatch):
    """`method_key` is training/methods' key: a continuation's direction/mechanism (its method with no ' from
    <start>'), and scratch/adamw at every from-scratch point, a start's own included."""
    _, out = build(tmp_path, monkeypatch)
    ds = plots.frame(json.loads((out / f"{panels.NAME}.json").read_text()))
    for d in ds:
        scratch = d["curve"] == "frontier" and "from scratch" in d["method"]
        assert d["method_key"] == (panels.SCRATCH if scratch else d["method"].split(" from ")[0])
    assert {d["method_key"] for d in ds} == {panels.SCRATCH, "ref-optim/adamw-all", "lowrank/lowrank", "sparse/sparse"}
    assert any(d["method"].endswith("the start s1") for d in ds if d["method_key"] == panels.SCRATCH)


def test_the_candidates_give_every_throttled_row(tmp_path, monkeypatch):
    """Every checkpoint on a throttled curve at some cell is a candidate at C0 + |C|, every one in the console's copy,
    and the draft formula on its console columns alone (test_algorithm_policy's `dial`, written apart from
    te_core.formalism) gives each throttled row's W* = |C|·ρ and basis, so those rows check the page."""
    _, out = build(tmp_path, monkeypatch)
    uncut = plots.frame(json.loads((out / f"{panels.NAME}.json").read_text()))
    ds = plots.frame(json.loads((out / "console" / f"{panels.NAME}.json").read_text()))
    cand = {d["run"]: d for d in ds if d["curve"] == "candidate"}
    throttled = [d for d in ds if d["curve"] == "throttled"]
    assert len(cand) == len([d for d in uncut if d["curve"] == "candidate"])
    assert {d["run"] for d in throttled} <= set(cand) <= {d["run"] for d in throttled} | {d["run"] for d in uncut
                                                                                          if d["curve"] == "honest"}
    for d in cand.values():
        assert set(json.loads(d["rho_inputs"])) == set(panels.PACKED) and len(d["rho_inputs"]) <= console.CELL_MAX
        assert d["compute_total_gates"] == pytest.approx(d["start_compute_gates"] + d["compute_after_gates"], rel=1e-3)
        assert (d["c_tokens_per_block"], d["k_bits"], d["S"], d["basis"]) == (None,) * 4
    for t in throttled:
        d = cand[t["run"]]
        rho, basis = dial(panels.page_inputs(d), t["c_gates"], t["k_bits"])
        assert basis == t["basis"] and d["compute_after_gates"] * rho == pytest.approx(t["compute_after_gates"], rel=2e-3)


def test_a_candidate_whose_columns_miss_rho_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(algorithm_policy, "page_rho", lambda d, c, k: (math.pi, "target"))
    with pytest.raises(ValueError, match="cells where a candidate's columns don't give te_core.formalism.at's rho"):
        build(tmp_path, monkeypatch)


def curve(kind, n, start="", c=None, k=None, mark=None, top=1.2, step=0.001):
    return [panels.row(curve=kind, data="v2.3-pilot", start=start, c_tokens_per_block=c, k_bits=k, mark=mark,
                       compute_total_gates=float(i), loss=top * (1 - step) ** i, run="r", method="m")
            for i in range(n)]


def at(rows, col):
    return [r[C[col]] for r in rows]


def test_the_cut_thins_the_unmarked_cells_first():
    """The frontier, honest and the marked cells keep points `FRONTIER_STEP` apart; every other cell the first point
    to reach each of n + 1 evenly spaced losses, its ends kept."""
    fine = [curve("frontier", 40), curve("honest", 40, "s"), curve("throttled", 40, "s", *panels.DEFAULT, "reference"),
            curve("throttled", 40, "s", 600.0, panels.SERVED[1], "served demo")]
    off = curve("throttled", 40, "s", 1, 80_000)
    got = panels.thinned([r for g in fine for r in g] + off, 3)
    for g in fine:
        kept = [r for r in got if (r[C["curve"]], r[C["mark"]]) == (g[0][C["curve"]], g[0][C["mark"]])
                and r[C["c_tokens_per_block"]] == g[0][C["c_tokens_per_block"]]]
        assert (kept[0], kept[-1]) == (g[0], g[-1])
        assert at(kept, "compute_total_gates") == [float(i) for i in range(0, 40, 3)]
        assert all(b <= a * (1 - panels.FRONTIER_STEP) for a, b in zip(at(kept, "loss"), at(kept, "loss")[1:]))
    assert at([r for r in got if r[C["c_tokens_per_block"]] == 1], "compute_total_gates") == [0.0, 13.0, 26.0, 39.0]
    assert panels.thinned(off, None) == off
    every = curve("candidate", 40, "s")
    assert panels.thinned(every, 1, panels.STEPS[-1]) == every
    branch = panels.row(curve="honest", data="v2.3-pilot", start="s", start_compute_gates=4.0,
                        start_loss=fine[0][4][C["loss"]], compute_total_gates=5.0, loss=1.0)
    kept = at([r for r in panels.thinned(fine[0] + [branch], 3) if r[C["curve"]] == "frontier"], "compute_total_gates")
    assert kept == [0.0, 3.0] + [float(i) for i in range(4, 40, 3)] + [39.0]


def test_the_cut_fits_the_route_and_says_how_many_points(monkeypatch):
    p = {"id": panels.PANEL, "note": "", "columns": panels.COLUMNS,
         "rows": curve("frontier", 10) + [r for kb in panels.K_BYTES for r in curve("throttled", 60, "s", 1, 8 * kb)]}
    whole = panels.cut(p, RATIO, "v2.3-pilot")
    assert len(whole["rows"]) == len(p["rows"]) and whole["note"] == panels.note(RATIO, None, "v2.3-pilot")
    assert panels.cut(p, RATIO, "v3")["rows"] == []
    monkeypatch.setattr(console, "BODY_MAX", len(console.body(whole)) // 4)
    q = panels.cut(p, RATIO, "v2.3-pilot")
    n = next(n for n in panels.POINTS if n and q["note"].endswith(f": {n + 1} points."))
    assert len(console.body(q)) <= console.BODY_MAX and len(q["note"]) <= console.NOTE_MAX
    assert len([r for r in q["rows"] if r[C["curve"]] == "throttled" and r[C["k_bits"]] == 80_000]) == n + 1
    monkeypatch.setattr(console, "BODY_MAX", 100)
    with pytest.raises(ValueError, match="no cut fits"):
        panels.cut(p, RATIO, "v2.3-pilot")
    p["rows"][0][C["on_frontier"]] = True
    with pytest.raises(ValueError, match="refuses 1 cells .row 0's on_frontier is True"):
        panels.cut(p, RATIO, "v2.3-pilot")


def test_the_cut_coarsens_the_fine_curves_last(monkeypatch):
    """When the unmarked cells at two points each don't fit, the cut keeps the other curves' points further apart in
    loss (`STEPS`), and its note says how far."""
    p = {"id": panels.PANEL, "note": "", "columns": panels.COLUMNS,
         "rows": curve("frontier", 200, step=1.01 * panels.FRONTIER_STEP) + curve("throttled", 60, "s", 1, 80_000)}
    finest = p | {"rows": [panels.rounded(r) for r in panels.thinned(p["rows"], 1)]}
    monkeypatch.setattr(console, "BODY_MAX", len(console.body(finest)) - 1)
    q = panels.cut(p, RATIO, "v2.3-pilot")
    assert len(console.body(q)) <= console.BODY_MAX and len(q["note"]) <= console.NOTE_MAX
    assert q["note"].endswith(f"rest {200 * panels.FRONTIER_STEP:g}% apart.")
    assert len([r for r in q["rows"] if r[C["curve"]] == "frontier"]) < 200


def test_the_cut_keeps_the_most_candidates_that_fit(monkeypatch):
    """Where every candidate doesn't fit at the coarsest cut, it keeps the most that do, those on the most cells'
    throttled curves first, and its note says how many."""
    cand = [panels.row(curve="candidate", data="v2.3-pilot", start="s", compute_after_gates=float(i), loss=1.0,
                       run=f"r{i}", rho_inputs="{}") for i in range(40)]
    on = [panels.row(curve="throttled", data="v2.3-pilot", start="s", c_tokens_per_block=float(n), k_bits=8e4,
                     compute_total_gates=1.0, loss=1.1, run="r39") for n in range(3)]
    p = {"id": panels.PANEL, "note": "", "columns": panels.COLUMNS, "rows": curve("frontier", 5) + on + cand}
    assert panels.ranked(p["rows"])[0] is cand[-1] and panels.ranked(p["rows"])[1:] == cand[:-1]
    whole = panels.cut(p, RATIO, "v2.3-pilot")
    monkeypatch.setattr(console, "BODY_MAX", len(console.body(whole)) - 2000)
    q = panels.cut(p, RATIO, "v2.3-pilot")
    kept = [r[C["run"]] for r in q["rows"] if r[C["curve"]] == "candidate"]
    assert len(console.body(q)) <= console.BODY_MAX and 0 < len(kept) < 40 and "r39" in kept
    assert kept == sorted(kept, key=lambda r: int(r[1:])) and q["note"].endswith(f"Candidates: {len(kept)} of 40.")
    one_more = q | {"rows": q["rows"] + [panels.rounded(next(r for r in cand if r[C["run"]] not in kept))]}
    assert len(console.body(one_more)) > console.BODY_MAX


def test_a_candidate_from_parts_that_differ_gives_rho_at_every_cell():
    """te_core 4.x's Fit per part and ᾱ/α_max: a record of two parts with different w, affected counts and Δ_grad (one
    unmeasured, then measured), and a u_in of 200 steps, gives at every cell of the dial on its candidate's columns
    alone (test_algorithm_policy's `dial`) te_core.formalism.at's ρ and basis, every cell within the route's."""
    from te_core.formalism import Measured, Part

    steps = tuple(p for i in range(200) for p in ((i / 200, 3e3 * i), (i / 200, 3e3 * (i + 1))))
    for dg in (None, 2e4):
        m = Measured((Part(4e15, 1e4, 3e15, 2e15, delta_grad_bits=1e4), Part(1e15, 1e4, 5e14, 1e14, delta_grad_bits=dg)),
                     16.0, 12288.0, ((0.0, 0.0), (0.4, 0.0), (1.0, 6e5)), steps + ((1.0, 6e5),), 5e5, 0.0)
        assert m.w_fit == 5e10 and m.alpha_ratio == pytest.approx(0.525) and m.fit_delta_grad_bits == (dg and 2e4)
        d = {"run": "r", "delta_bits": m.delta_bits, "w_fwd_gates": W, **panels.rho_columns(m, W)}
        assert all(len(v) <= console.CELL_MAX for v in d.values() if isinstance(v, str))
        assert len(algorithm_policy.profile(d["u_in_points"])) <= len(panels.K_LEVELS) + 2
        fits = set()
        for (n, k) in panels.cells(W):
            want = formalism.at(m, n * W, k)
            rho, basis = dial(panels.page_inputs(d), n * W, k)
            assert basis == ("fit" if want["fit"] else "target") and rho == pytest.approx(want["rho"], rel=1e-4)
            fits.add(basis)
        assert fits == ({"target"} if dg is None else {"fit", "target"})
        assert panels.mismatches(d, panels.dial(m, W), W) == []


def test_every_note_fits():
    assert max(len(panels.note(RATIO, n, d, s, (999, 1000))) for n in panels.POINTS for d in panels.DATA
               for s in panels.STEPS) <= console.NOTE_MAX


def test_publish_puts_the_console_copy(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(console, "put", lambda p: sent.append(p) or "201")
    code, out = build(tmp_path, monkeypatch, False, ("--publish",))
    assert code == 0 and len(sent) == 1
    assert console.body(sent[0]) == (out / "console" / f"{panels.NAME}.json").read_bytes()
    monkeypatch.setattr(console, "put", lambda p: "403 forbidden")
    assert build(tmp_path / "again", monkeypatch, False, ("--publish",))[0] == 1
