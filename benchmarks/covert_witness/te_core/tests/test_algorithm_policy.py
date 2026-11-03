"""te_core.algorithm_policy builds `training/algorithm-policy`: a row per run, start and gain target, at its first
checkpoint whose gain (effective compute under te_core.frontier_fit's fit, over C0) reaches the target, else at its
last; H the least compute from the start reaching it, every checkpoint pooled, and p = C/H; the dial's formula on a
row's columns alone is te_core.formalism.at's at every cell; the coverage per gain target, data and decade of ρ; and
the console's cut, which keeps the coverage's rows and each group's (p, ρ) front, naming the stored tables' art id."""
import json
import math

import pytest

pytest.importorskip("matplotlib")

from te_core import algorithm_policy as AP
from te_core import console, formalism, panels, plots, policy, rescore
from te_core import frontier_fit as FF
from te_core.gates import per_mac
from test_plots import SHAPE

SHAPE2 = SHAPE | {"tokens": 2 * SHAPE["tokens"], "tokens_per_param": 6.84}
V3_SHAPE = SHAPE | {"tokenizer": policy.V3_TOKENIZER}
#: a token block's gates: the forward and a backward of twice its cost
W = 3 * policy.w_fwd(SHAPE)
#: dense AdamW's gates a training token in forward passes, as te_core.panels' `dense_rate` reads it off the
#: ref-optim/adamw runs: `block`'s token blocks are `W` gates and σ = 0.8 of the gates
RATE = 3 / 0.8
C0 = panels.c0(SHAPE, RATE)
#: the pilot's scaling law in these tests, s1's start on it (L(C0) = 1.2), and the from-scratch losses' ratio
LAW = {"E": 0.9, "A": 0.3 * C0 ** 0.3, "gamma": 0.3}
RATIO = 1.03
#: the from-scratch run's evaluations, in s1's training tokens (none at s2's twice, whose start it would be)
SCRATCH = (0.5, 1, 3, 6, 12, 25, 50, 100)
#: (start, run, direction/mechanism, family, Δ, Δ_grad, Δ_t, tail share, [(step, |C| / C0, bpb), ...]), each above the
#: law at its total compute; from s1 the law's targets are 1.1915 at gain 1.1, 1.1437 at 2 and 1.0460 at 11
RUNS = [("s1", "ref", "ref-optim/adamw", "full", 9e8, 3e9, None, 0.0, [(1, 0.05, 1.1975), (2, 0.3, 1.185),
                                                                      (3, 1.5, 1.138)]),
        ("s1", "lora", "lowrank/lora", "adapter", 8e6, 2e7, None, 0.3, [(1, 0.2, 1.187), (2, 0.6, 1.17)]),
        ("s1", "mezo", "seed-mezo/mezo", "seed", 1e3, 3e4, None, 0.0, [(1, 0.3, 1.199), (2, 1.2, 1.197)]),
        ("s1", "qat", "quant/qat", "update-code", 1e3, 3e4, 6e4, 0.0, [(1, 0.4, 1.183)]),
        ("s1", "never", "sparse/lines", "partial", 1e6, 4e6, None, 0.0, [(1, 0.05, 1.1995)]),
        ("s2", "ref2", "ref-optim/adamw", "full", 9e8, 3e9, None, 0.0, [(1, 0.5, 1.28), (2, 1.0, 1.25)])]
#: adapters from s1 that reach no target, so that every tier of the cut drops rows: lora's attempts beat theirs
DOMINATED = [("s1", f"lora{i}", "lowrank/lora", "adapter", 8e6, 2e7, None, 0.3, [(1, 0.2 + 0.1 * i, 1.196)])
             for i in range(6)]
L0 = {"s1": 1.2, "s2": 1.3, "t1": 1.5}


def law(c: float, f: dict = LAW) -> float:
    return f["E"] + f["A"] * c ** -f["gamma"]


def block(gates, delta, dgrad, delta_t=None, tail=0.0, sigma=0.8) -> dict:
    """One part whose token blocks are `W` gates each, labelled as te_core.rescore labels the analysis's blocks."""
    u, u_in = formalism.tail(delta, tail)
    return {"parts": [{"gates": gates, "tokens": sigma * gates / W, "token_gates": sigma * gates,
                       "affected_gates": 0.5 * sigma * gates, "steps": 1, "delta_grad_bits": dgrad}],
            "tau_bits": 16, "a_bits": 8192, "delta_bits": delta, "delta_t_bits": delta_t, "u1_bits": delta,
            "u": [list(x) for x in u], "u_in": [list(x) for x in u_in], "tail_share": tail, "decoder_gates": None,
            "labels": rescore.LABELS | (rescore.TAIL if tail else rescore.EVEN)}


def ledger_rows(runs=RUNS) -> list[dict]:
    out = []
    for start, run, method, family, delta, dgrad, delta_t, tail, cps in runs:
        direction, mechanism = method.split("/")
        for step, share, bpb in cps:
            out.append({"id": f"{run}-{step}", "run_id": run, "checkpoint_step": step, "start": start,
                        "start_bpb": L0[start], "val_bpb": bpb, "direction": direction, "mechanism": mechanism,
                        "family": family, "measurement_commit": "pin", "output": {"e_bytes": delta / 8},
                        "written_ts": len(out), "formalism": block(share * C0, delta, dgrad, delta_t, tail)})
    return out


def pretraining(root):
    """learning_frontier's s run on the law, through s1's start (so the loss ratio is `RATIO`)."""
    d = root / "lf-s"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"args": {"size": "s"}, "counts": {"total": SHAPE["parameters"]}}))
    (d / "curve.json").write_text(json.dumps([{"step": i, "tokens": round(f * SHAPE["tokens"]),
                                               "val_nats_per_byte": law(f * C0) / RATIO * math.log(2)}
                                              for i, f in enumerate(SCRATCH)]))
    return root


def args(tmp_path, monkeypatch, runs=RUNS, index=None) -> list[str]:
    """te_core.algorithm_policy's and te_core.frontier_fit's options over a ledger of `runs` (or the rows `index`), a
    manifest of s1, s2 and v3's t1 and the law's from-scratch run, scored on the stubs."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    man = tmp_path / "starts_store.json"
    man.write_text(json.dumps({"starts": {"s1": {"shape": SHAPE}, "s2": {"shape": SHAPE2}, "t1": {"shape": V3_SHAPE}}}))
    rows = ledger_rows(runs) if index is None else index
    monkeypatch.setattr(rescore, "Ledger", lambda root: type("L", (), {"index": staticmethod(lambda: rows),
                                                                      "root": tmp_path})())
    return ["--out", str(tmp_path / "out"), "--pin", "pin", "--manifest", str(man), "--stubs",
            "--pretraining", str(pretraining(tmp_path / "lf"))]


def run(tmp_path, monkeypatch, *extra, runs=RUNS, index=None) -> tuple[int, dict]:
    code = AP.main(args(tmp_path, monkeypatch, runs, index) + list(extra))
    out = tmp_path / "out"
    read = {k: json.loads((out / "tables" / f"{name}.json").read_text())
            for k, name in (("panel", AP.NAME), ("fit", FF.NAME), ("coverage", "coverage"))}
    return code, read | {"out": out, "console": json.loads((out / "console" / f"{AP.NAME}.json").read_text())}


def fitted(fit_panel: dict, data: str = "v2.3-pilot") -> dict:
    d = next(d for d in plots.frame(fit_panel) if d["row"] == "fit" and d["data"] == data)
    return {"E": d["E_bpb"], "A": d["A"], "gamma": d["gamma"]}


def info(fit: dict | None = LAW, runs=RUNS) -> dict:
    """te_core.frontier_fit's `info` for `runs`: their starts, the pilot's fit `fit` and none for v3."""
    shapes = {"s1": SHAPE, "s2": SHAPE2}
    scored = plots.load(ledger_rows(runs), ("pin",), shapes)[0]
    return {"starts": FF.starts(scored, shapes, RATE), "fits": {"v3": None, "v2.3-pilot": fit}}


def dial(d: dict, c: float, k: float) -> tuple[float, str]:
    """ρ's formula on a row's columns alone, with te_core 4.x's per-part Fit (w_fit_gates) and ᾱ/α_max (alpha_ratio)
    where the row has them, written apart from te_core.formalism: Fit for the training, else Slowdown's bound from u
    and u_in's points."""
    s, dg = c / d["w_gates"], d["delta_grad_bits"]
    s_fit = c / d.get("w_fit_gates", d["w_gates"])
    if d["delta_bits"] + s_fit * d["tau_bits"] <= k and dg is not None and 2 * dg <= k and d["delta_t_bits"] + dg <= k:
        return c / (c - d["d_gates"]), "fit"
    u, u_in = AP.profile(d["u_points"]), AP.profile(d["u_in_points"])
    x0 = 0.0
    for (xa, ua), (xb, ub) in zip(u_in, u_in[1:]):
        if ub <= k:
            x0 = xb
            continue
        if ua <= k and xb > xa:
            x0 = xa + (xb - xa) * (k - ua) / (ub - ua)
        break
    best = 0.0
    for (xa, ua), (xb, ub) in zip(u, u[1:]):
        xs = [xa, xb]
        if xb > xa and ub > ua:
            m = (ub - ua) / (xb - xa)
            xs.append(min(max((k - ua + m * xa) / (2 * m), xa), xb))
        for x in xs:
            ux = ua if xb == xa else ua + (ub - ua) * (x - xa) / (xb - xa)
            if x > 0 and ux <= k:
                best = max(best, x * (k - ux))
    lead = d["sigma"] * (1 - x0) * s * d.get("alpha_ratio", 1.0)
    return (1.0 if lead == 0 else max(1.0, lead * d["a_bits"] / best) if best > 0 else math.inf), "target"


def test_every_run_has_a_row_at_each_gain_target_at_its_first_checkpoint_reaching_it_else_at_its_last(tmp_path,
                                                                                                    monkeypatch):
    """At gain g a start's target loss is the fit's at g·C0; each run from the start has one row there: reached at its
    first checkpoint at or below it, else not at its last, with C its |C|, the gain C_eff(loss)/C0, H the least |C| of
    any checkpoint from the start reaching it and p = C/H (null without H)."""
    code, got = run(tmp_path, monkeypatch)
    assert code == 0 and (got["out"] / f"{AP.NAME}.png").stat().st_size > 10_000
    p, f = got["panel"], fitted(got["fit"])
    assert (p["id"], p["kind"], p["columns"]) == (AP.PANEL, "table", AP.COLUMNS) and p["source"] and p["updated_at"]
    ds, rows = plots.frame(p), ledger_rows()
    assert {d["data"] for d in ds} == {"v2.3-pilot"} and {d["gain_target"] for d in ds} == set(AP.GAINS)
    for start, shape in (("s1", SHAPE), ("s2", SHAPE2)):
        c0, mine = panels.c0(shape, RATE), [r for r in rows if r["start"] == start]
        for g in AP.GAINS:
            target = FF.loss_at(f, g * c0)
            at = {d["record"].split("@")[0]: d for d in ds
                  if (d["start"], d["gain_target"]) == (plots.start_label(shape), g)}
            assert set(at) == {r["run_id"] for r in mine}
            h = min((r["formalism"]["parts"][0]["gates"] for r in mine if r["val_bpb"] <= target), default=None)
            for run_id, d in at.items():
                cps = [r for r in mine if r["run_id"] == run_id]
                hit = next((r for r in cps if r["val_bpb"] <= target), None)
                r = hit or cps[-1]
                assert (d["reached"], d["record"]) == ("yes" if hit else "no", f"{run_id}@{r['checkpoint_step']}")
                assert (d["start_loss"], d["end_loss"]) == (L0[start], r["val_bpb"])
                assert (d["start_compute_gates"], d["C_gates"]) == pytest.approx((c0, r["formalism"]["parts"][0]["gates"]))
                assert d["gain"] == pytest.approx(FF.c_eff(f, r["val_bpb"]) / c0) and (d["gain"] >= g) == bool(hit)
                assert d["H_gates"] == (h and pytest.approx(h))
                assert d["p"] == (h and pytest.approx(d["C_gates"] / h))
                assert d["family"] == next(x[3] for x in RUNS if x[1] == run_id)
                assert d["method_key"] == next(x[2] for x in RUNS if x[1] == run_id)
                assert d["w_fwd_gates"] == pytest.approx(policy.w_fwd(shape))
    s1 = {(d["record"], d["gain_target"]): (d["reached"], d["p"]) for d in ds if d["start"] == plots.start_label(SHAPE)}
    assert s1[("lora@1", 1.1)] == ("yes", 1) and s1[("qat@1", 1.1)] == ("yes", 2) and s1[("ref@3", 2)] == ("yes", 1)
    assert s1[("mezo@2", 1.1)][0] == s1[("lora@2", 2)][0] == s1[("ref@3", 11)][0] == "no"
    assert s1[("ref@3", 11)][1] is None and s1[("mezo@2", 2)][1] == pytest.approx(1.2 / 1.5)
    q = got["console"]
    assert q["rows"] == [AP.rounded(r) for r in p["rows"]] and not console.refused(q)
    assert len(console.body(q)) <= console.BODY_MAX and len(q["note"]) <= console.NOTE_MAX
    assert all(not isinstance(v, bool) for r in q["rows"] for v in r)


def test_the_dial_formula_on_a_rows_columns_is_te_cores_at_every_cell(tmp_path, monkeypatch):
    """rho_default and basis_default are the dial's formula on the published columns, and te_core.panels' dial on them
    gives formalism.at's ρ and basis at every cell; Fit's update application reads delta_t (qat's latents fail it where
    mezo, the same but for them, fits)."""
    _, got = run(tmp_path, monkeypatch)
    by = {r["id"]: r for r in ledger_rows()}
    ds = plots.frame(got["console"])
    for d in ds:
        page = panels.page_inputs(d)
        rho, basis = dial(page, *AP.default(d["w_fwd_gates"]))
        assert basis == d["basis_default"] and rho == pytest.approx(d["rho_default"], rel=1e-3)
        r = by[d["record"].replace("@", "-")]
        m = formalism.measure(r, {"s1": SHAPE, "s2": SHAPE2}[r["start"]])
        assert panels.mismatches(d | {"run": d["record"]}, panels.dial(m, d["w_fwd_gates"]), d["w_fwd_gates"]) == []
    at = {d["record"]: d for d in ds if d["gain_target"] == 1.1}
    assert (at["mezo@2"]["basis_default"], at["mezo@2"]["rho_default"], at["qat@1"]["basis_default"]) == ("fit", 1,
                                                                                                         "target")
    assert json.loads(at["qat@1"]["rho_inputs"])["delta_t_bits"] == 6e4 and at["mezo@2"]["delta_bits"] == 1e3
    assert json.loads(at["mezo@2"]["rho_inputs"])["delta_t_bits"] == 1e3


@pytest.mark.parametrize("u1, dgrad, delta_t, tail, decoder", [
    (1e9, 3e9, None, 0.0, None), (1e9, 3e9, None, 0.3, None), (1e3, 3e4, None, 0.0, None), (1e3, 3e4, 6e4, 0.0, None),
    (2e4, 1e4, 3e4, 0.05, 1e9), (5e5, None, None, 0.2, None), (5e4, 2e4, None, 0.0, None)])
def test_te_core_at_is_the_dial_formula(u1, dgrad, delta_t, tail, decoder):
    """Over decades of c and k, te_core.formalism.at's rho and basis are the draft note's formula on the inputs this
    panel publishes for the record (te_core's ρ is the score; the dial reruns it in the page)."""
    m = formalism.from_json(block(1e15, u1, dgrad, delta_t, tail) | {"decoder_gates": decoder, "labels": {}})
    d = dict(zip(AP.INPUTS, AP.inputs(m, policy.w_fwd(SHAPE))))
    for c in (1e10, 1e12, 1e14, 1e16):
        for k in (1e3, 1e4, 8e4, 1e6, 1e8, 1e10):
            v = formalism.at(m, c, k)
            rho, basis = dial(d, c, k)
            assert basis == ("fit" if v["fit"] else "target"), (c, k)
            assert rho == pytest.approx(v["rho"], rel=1e-4), (c, k)
            assert AP.page_rho(d, c, k) == (pytest.approx(rho, rel=1e-4), basis)


def test_a_profile_capped_at_delta_loses_its_redundant_points_and_nothing_else():
    """A profile capped at Δ has long flat runs at Δ; `simplified` keeps the function (room and the bottom share at
    every k) and fits the route's cell."""
    u = ((0.0, 0.0), (0.8, 0.0), *((0.8 + i / 100, 5e8 + i * 4e6) for i in range(15)), (1.0, 6e8))
    stair = tuple(p for (x, b), (x1, _) in zip(u, u[1:] + ((1.0, 0.0),)) for p in ((x, b), (x1, b)))
    capped = formalism.capped(stair, 1864.0)
    s = AP.simplified(capped)
    assert s == ((0.0, 0.0), (0.8, 0.0), (0.8, 1864.0), (1.0, 1864.0)) and len(AP.points(s)) <= console.CELL_MAX
    for v in (u, stair, capped, ((0.0, 0.0), (0.5, 0.0), (0.5, 2.0), (0.5, 3.0), (1.0, 3.0), (1.0, 3.0))):
        for k in (0.0, 1.0, 1864.0, 1e4, 5.1e8, 1e9):
            assert formalism.room(AP.simplified(v), k) == pytest.approx(formalism.room(v, k), abs=1e-9)
            assert formalism.bottom_share(AP.simplified(v), k) == formalism.bottom_share(v, k)


def test_estimated_names_the_inputs_that_arent_measured():
    m = formalism.from_json(block(1e15, 1e6, 3e6, None, 0.3))
    text = AP.estimated(m)
    assert text == ("w_gates:est tau_bits:stub a_bits:bound u_points:A1 u_in_points:A1' delta_bits:bound "
                    "delta_t_bits:bound d_gates:stub delta_grad_bits:est")
    latent = formalism.from_json(block(1e15, 1e6, 3e6, 4e6, 0.3))
    assert "delta_t_bits:est" in AP.estimated(latent)
    counted = formalism.from_json(block(1e15, 1e6, 3e6) | {"labels": {
        "gates": "counted MACs in gates (te_core.gates)", "tokens": "measured: the steps' token counts",
        "token_gates": "counted: the steps' forward and backward in gates (te_core.gates)", "tau_bits": "measured: ids",
        "a_bits": "measured: cut", "u": "measured: u", "u_in": "measured: u_in", "u1_bits": "measured: u(1)",
        "delta_bits": "measured: E", "decoder_gates": "measured: d", "delta_grad_bits": "measured: the cut"}})
    assert AP.estimated(counted) == ""
    worst = formalism.from_json(block(1e15, 1e6, 3e6, 4e6) | {"labels": {}})
    assert len(AP.estimated(worst)) <= console.CELL_MAX and AP.estimated(worst).startswith("C_gates:est w_gates:est")


def test_profiles_are_compact_strings():
    assert AP.points(((0.0, 0.0), (0.25, 1.2e6), (1.0, 4.8e6))) == "0:0;0.25:1.2e6;1:4.8e6"
    u = ((0.0, 0.0), (0.7586553743, 0.0), (1.0, 24601880.0))
    back = AP.profile(AP.points(u))
    assert [v for xb in back for v in xb] == pytest.approx([v for xb in u for v in xb], rel=1e-6)
    assert AP.num(1.234567e-7) == "1.23457e-7"


def test_a_target_the_start_already_meets_has_no_rows_and_a_data_without_a_fit_none():
    rich = LAW | {"A": LAW["A"] * 1.5 ** 0.3}
    assert FF.gain(rich, 1.2, C0) == pytest.approx(1.5)
    ds = plots.frame(AP.build(info(rich), "test"))
    assert {d["gain_target"] for d in ds if d["start"] == plots.start_label(SHAPE)} == {2, 11}
    assert AP.build(info(None), "test")["rows"] == []


def row(start, g, family, reached, p, rho, gain, rec, data="v2.3-pilot", gates=1.0) -> list:
    d = dict.fromkeys(AP.COLUMNS) | {"start": start, "gain_target": g, "family": family, "reached": reached, "p": p,
                                    "rho_default": rho, "gain": gain, "record": rec, "data": data, "C_gates": gates}
    return [d[c] for c in AP.COLUMNS]


def test_the_coverage_is_the_least_p_reached_row_else_the_best_failed_attempt_else_empty():
    """Per gain target, data and decade of ρ (an infinite ρ in the last): the reached row with the least p, else the
    failed attempt with the greatest gain, else EMPTY, which the table and the board line both say."""
    rows = [row("s", 2, "adapter", "yes", 3.0, 50.0, 2.1, "a"), row("s", 2, "full", "yes", 1.0, 20.0, 2.0, "b"),
            row("s", 2, "adapter", "no", 0.5, 2.0, 1.5, "c"), row("s", 2, "adapter", "no", 0.2, 3.0, 1.7, "d"),
            row("s", 2, "seed", "no", 0.9, None, 1.1, "e"), row("s", 2, "full", "yes", 2.0, 9.0, 2.5, "f"),
            row("s", 11, "adapter", "no", None, 5e3, 1.9, "g")]
    cov = AP.coverage({"columns": AP.COLUMNS, "rows": rows})
    assert len(cov) == len(AP.GAINS) * len(panels.DATA) * len(AP.DECADES)
    at = {(c["gain_target"], c["data"], c["rho_decade"]): c for c in cov}
    pick = {k[::2]: (c["state"], c.get("record"), c["rows"]) for k, c in at.items() if c["state"] != "empty"}
    assert pick == {(2, "1"): ("reached", "f", 3), (2, "10"): ("reached", "b", 2), (2, "1e5+"): ("failed", "e", 1),
                    (11, "1e3"): ("failed", "g", 1)}
    assert at[(2, "v2.3-pilot", "1")]["row"] == 5 and all(c["state"] == "empty" for c in cov if c["data"] == "v3")
    fits = {"v3": None, "v2.3-pilot": LAW}
    md = AP.coverage_md(cov, fits, "T")
    table = [line for line in md.splitlines() if line.startswith("| ") and "gain target" not in line]
    assert len(table) == len(AP.GAINS) * len(panels.DATA) and sum(t.count(f"**{AP.EMPTY}**") for t in table) == 32
    assert "| 2 | v3 (no fit) | " + " | ".join([f"**{AP.EMPTY}**"] * len(AP.DECADES)) + " |" in table
    assert "| 2 | v2.3-pilot | yes p 1, rho 2.5" not in md and md.endswith("32 of 36 cells EMPTY; 2 hold only failed "
                                                                          "attempts.\n")
    assert "| 2 | v2.3-pilot | yes p 2, rho 9 (s, None, f) | yes p 1, rho 20 (s, None, b) | **EMPTY** |" in md
    line = AP.board_line(cov, fits, "T", "art:x")
    assert "g 2 v3: no fit, all 6 EMPTY" in line and "g 11 v2.3-pilot: EMPTY | EMPTY | EMPTY | no gain 1.9 | " in line
    assert line.endswith("32 of 36 cells EMPTY.")


def test_kept_is_the_coverage_rows_and_each_groups_p_rho_front_and_best_failed_attempt():
    rows = [row("s", 2, "adapter", "yes", 1.0, 50.0, 2.1, "a"), row("s", 2, "adapter", "yes", 2.0, 10.0, 2.0, "b"),
            row("s", 2, "adapter", "yes", 3.0, 20.0, 2.0, "dominated"), row("s", 2, "adapter", "yes", 5.0, 5.0, 2.2, "c"),
            row("s", 2, "adapter", "no", 0.5, 1.0, 1.5, "near"), row("s", 2, "adapter", "no", 0.2, 1.0, 1.2, "far"),
            row("s", 2, "full", "yes", 1.5, 90.0, 2.0, "f"), row("s", 2, "full", "no", 0.1, 1.0, 1.9, "f-near"),
            row("s", 11, "adapter", "no", None, 1.0, 1.9, "no-H"), row("t", 2, "adapter", "yes", 4.0, None, 2.0, "inf")]
    rec = AP.COLUMNS.index("record")
    assert [r[rec] for r in AP.kept(rows, "family", set())] == ["a", "b", "c", "near", "f", "f-near", "no-H", "inf"]
    assert [r[rec] for r in AP.kept(rows, "start", {5})] == ["a", "b", "c", "far", "f-near", "no-H", "inf"]
    assert [r[rec] for r in AP.kept(rows, "coverage", {1, 7})] == ["b", "f-near"]


def test_the_cut_is_the_table_when_it_fits_else_the_first_tier_that_does(monkeypatch):
    p = AP.build(info(runs=RUNS + DOMINATED), "test")
    art = "art:" + "a" * 64
    assert AP.cut(p, art)["rows"] == [AP.rounded(r) for r in p["rows"]] and AP.cut(p, art)["note"] == AP.note(art)
    cov = {c["row"] for c in AP.coverage(p) if c["row"] is not None}
    size = {t: len(console.body(AP.checked(p | {"rows": [AP.rounded(r) for r in AP.kept(p["rows"], t, cov)],
                                                "note": AP.note(art, t)}))) for t in AP.TIERS}
    assert len(console.body(AP.cut(p, art))) > size["family"] > size["start"] > size["coverage"]
    for t in AP.TIERS:
        monkeypatch.setattr(console, "BODY_MAX", size[t])
        q = AP.cut(p, art)
        assert q["note"] == AP.note(art, t) and len(q["note"]) <= console.NOTE_MAX and not console.refused(q)
        assert {p["rows"][i][AP.COLUMNS.index("record")] for i in cov} <= {r[AP.COLUMNS.index("record")] for r in q["rows"]}
    monkeypatch.setattr(console, "BODY_MAX", size["coverage"] - 1)
    with pytest.raises(ValueError, match="the coverage's rows alone"):
        AP.cut(p, art)


def test_publish_stores_the_tables_and_puts_both_panels_the_cut_naming_their_art_id(tmp_path, monkeypatch):
    """--publish puts tables/ in the evidence store (the uncut panel, its coverage and the fit's panel and points) and
    PUTs training/frontier-fit, then the cut, whose note names the art id; a refused PUT fails the command."""
    sent, stored = [], []
    monkeypatch.setattr(console, "put", lambda q: sent.append(q) or "200")
    monkeypatch.setattr(AP, "store", lambda path, meta: stored.append(sorted(f.name for f in path.iterdir()))
                        or "art:" + "a" * 64)
    runs = RUNS + DOMINATED
    assert run(tmp_path / "dry", monkeypatch, runs=runs)[0] == 0 and not stored and not sent
    code, got = run(tmp_path / "fits", monkeypatch, "--publish", runs=runs)
    assert code == 0 and stored == [["algorithm-policy.json", "coverage.json", "coverage.md", "frontier-fit.json",
                                     "frontier-points.json"]]
    assert [q["id"] for q in sent] == [FF.PANEL, AP.PANEL] and console.body(sent[1]) == console.body(got["console"])
    assert len(sent[1]["rows"]) == len(got["panel"]["rows"]) and sent[1]["note"].endswith(f" Tables art:{'a' * 64}.")
    monkeypatch.setattr(console, "BODY_MAX", len(console.body(sent[1])) - 1)
    code, got = run(tmp_path / "cut", monkeypatch, "--publish", runs=runs)
    assert code == 0 and len(got["console"]["rows"]) < len(got["panel"]["rows"])
    assert " Cut: coverage rows + per start, g, family: " in got["console"]["note"]
    monkeypatch.setattr(console, "put", lambda q: "403 forbidden")
    assert run(tmp_path / "refused", monkeypatch, "--publish", runs=runs)[0] == 1


def test_v3_rows_come_with_v3s_fit_from_the_ladder_and_lead_the_figure(tmp_path, monkeypatch):
    """A loss on harness v3's data is no loss on the pilot's: v3's fit reads the ladder's from-scratch runs and its own
    starts and their checkpoints, none of the pilot's, and a v3 start's rows have their own l0, targets and H; the
    figure shows v3 once it has rows."""
    v3_law = LAW | {"E": 1.2}
    ladder = tmp_path / "F.json"
    ladder.write_text(json.dumps({"schema": "te-ladder/F/v1", "annealed": [
        {"branch": f"x{f}", "macs": f * C0 / per_mac("gemm/bf16"), "bpb": law(f * C0, v3_law)} for f in (2, 4, 8, 16, 32)]}))
    v3 = [r | {"id": f"v3-{r['id']}", "run_id": f"v3-{r['run_id']}", "data": "v3", "start": "t1", "start_bpb": 1.5,
               "val_bpb": r["val_bpb"] + 0.3} for r in ledger_rows() if r["start"] == "s1"]
    code, got = run(tmp_path, monkeypatch, "--ladder-points", str(ladder), index=ledger_rows() + v3)
    assert code == 0 and AP.headline(got["panel"]) == "v3"
    read = [d["source"] for d in plots.frame(got["fit"]) if (d["row"], d["data"]) == ("point", "v3")]
    assert read[0] == "start t1" and {f"ladder x{f}, annealed" for f in (2, 4, 8, 16, 32)} <= set(read)
    assert all(s.startswith(("start t1", "ladder ")) or " from t1 (v3-" in s for s in read)
    assert fitted(got["fit"], "v3")["E"] == pytest.approx(1.2, abs=0.05)
    assert fitted(got["fit"])["E"] == pytest.approx(LAW["E"], abs=0.05)
    ds = [d for d in plots.frame(got["panel"]) if d["data"] == "v3"]
    assert {d["start"] for d in ds} == {plots.start_label(V3_SHAPE)} and {d["start_loss"] for d in ds} == {1.5}
    assert {(d["record"], d["gain_target"]) for d in ds if d["reached"] == "yes"} >= {("v3-qat@1", 1.1)}
    assert any(c["state"] != "empty" for c in got["coverage"] if c["data"] == "v3")


def test_a_record_its_columns_cant_reproduce_is_refused():
    """The context charge (te_core.rescore's --charge-context) changes s* in a way the dial's columns don't carry, so a
    row scored with it would show the console a ρ its own dial doesn't give."""
    m = formalism.from_json(block(C0, 1e6, 3e6, None, 0.3) | {"context": formalism.context(["blocks.2.wq"], SHAPE)})
    r = {"id": "x-1", "run_id": "x", "checkpoint_step": 1, "start": "s1", "start_bpb": 1.2, "val_bpb": 1.19,
         "gates": m.gates, "measured": m, "direction": "d", "mechanism": "m"}
    st = FF.Start("s1", SHAPE, "v2.3-pilot", [r], RATE)
    with pytest.raises(ValueError, match=r"cells where a row's columns don't give .* has no context charge"):
        AP.build({"starts": [st], "fits": {"v2.3-pilot": LAW}}, "test")
    plain = formalism.from_json(block(C0, 1e6, 3e6, None, 0.3))
    assert AP.build({"starts": [FF.Start("s1", SHAPE, "v2.3-pilot", [r | {"measured": plain}], RATE)],
                     "fits": {"v2.3-pilot": LAW}}, "test")["rows"]
