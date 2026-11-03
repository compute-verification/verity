"""te_core.algorithm_policy builds `training/algorithm-policy`: a row per run, start and gain target, at its first
checkpoint whose gain (C(L1)/C(L0) under te_core.frontier_fit's fit, from the start's loss) reaches the target, else at
its last; H the least compute from the start reaching it, every checkpoint pooled, and p = C/H; the dial's formula on a
row's columns alone is te_core.formalism.at's at every cell; the coverage per gain target, data and decade of ρ; and
the console's cut, which keeps the coverage's rows and each group's (p, ρ) front, naming the stored tables' art id."""
import json
import math

import pytest

pytest.importorskip("matplotlib")

from te_core import algorithm_policy as AP
from te_core import console, formalism, gates, panels, plots, policy, rescore
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
    """One part whose token blocks are `W` gates each, labelled as te_core.rescore labels the analysis's blocks but
    with tau_bits as a trace measures it (the analysis's is a stub, which would mark every row `STUB`)."""
    u, u_in = formalism.tail(delta, tail)
    return {"parts": [{"gates": gates, "tokens": sigma * gates / W, "token_gates": sigma * gates,
                       "affected_gates": 0.5 * sigma * gates, "steps": 1, "delta_grad_bits": dgrad}],
            "tau_bits": 16, "a_bits": 8192, "delta_bits": delta, "delta_t_bits": delta_t, "u1_bits": delta,
            "u": [list(x) for x in u], "u_in": [list(x) for x in u_in], "tail_share": tail,
            "labels": rescore.LABELS | {"tau_bits": formalism.TRACED["tau_bits"]}
            | (rescore.TAIL if tail else rescore.EVEN)}


def ledger_rows(runs=RUNS) -> list[dict]:
    out = []
    for start, run, method, family, delta, dgrad, delta_t, tail, cps in runs:
        direction, mechanism = method.split("/")
        for step, share, bpb in cps:
            out.append({"id": f"{run}-{step}", "run_id": run, "checkpoint_step": step, "start": start,
                        "start_bpb": L0[start], "val_bpb": bpb, "direction": direction, "mechanism": mechanism,
                        "family": family, "measurement_commit": "pin", "output": {"e_bytes": delta / 8, "rebuild_macs": 0},
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
    monkeypatch.setattr(FF, "LADDER", None)
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
    counts = d["d_gates"] is not None and d["d_gates"] < c
    if counts and d["delta_bits"] + s_fit * d["tau_bits"] <= k and dg is not None and 2 * dg <= k \
            and d["delta_t_bits"] + dg <= k:
        return c / (c - d["d_gates"]), "fit"
    u, u_in = AP.profile(d["u_points"]), AP.profile(d["u_in_points"])
    if counts:
        u, u_in = formalism.capped(u, d["delta_bits"]), formalism.capped(u_in, d["delta_bits"])
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
    """At gain g a start's target loss is E + (l0 − E)·g^(−γ) on the fit; each run from the start has one row there:
    reached at its first checkpoint at or below it, else not at its last, with C its |C|, the gain C(L1)/C(l0) on the
    fit, H the least |C| of any checkpoint from the start reaching it and p = C/H (null without H)."""
    code, got = run(tmp_path, monkeypatch)
    assert code == 0 and (got["out"] / f"{AP.NAME}.png").stat().st_size > 10_000
    p, f = got["panel"], fitted(got["fit"])
    assert (p["id"], p["kind"], p["columns"]) == (AP.PANEL, "table", AP.COLUMNS) and p["source"] and p["updated_at"]
    cell = {c: [r[AP.COLUMNS.index(c)] for r in p["rows"]]
            for c in ("reading", "te_core", "rho_console", "binding_kind", "rests_on", "kinds")}
    assert set(cell["reading"]) == {"B"} and set(cell["te_core"]) == {AP.VERSION} and p["rows"]
    assert all(v == "inf" or v >= 1 for v in cell["rho_console"])
    assert set(cell["binding_kind"]) <= {"fit", "target"}
    assert all(formalism.RESTS_ON_CELL[b] == r for b, r in zip(cell["binding_kind"], cell["rests_on"]))
    assert all(set(json.loads(x)) == {"bound", "sensitivity"} for x in cell["kinds"])
    ds, rows = plots.frame(p), ledger_rows()
    assert {d["data"] for d in ds} == {"v2.3-pilot"} and {d["gain_target"] for d in ds} == set(AP.GAINS)
    for start, shape in (("s1", SHAPE), ("s2", SHAPE2)):
        c0, mine = panels.c0(shape, RATE), [r for r in rows if r["start"] == start]
        for g in AP.GAINS:
            target = FF.target(f, L0[start], g)
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
                assert d["start_effective_gates"] == pytest.approx(FF.c_eff(f, L0[start]))
                assert d["gain"] == pytest.approx(FF.c_eff(f, r["val_bpb"]) / FF.c_eff(f, L0[start]))
                assert (d["gain"] >= g) == bool(hit) and target < L0[start]
                assert d["H_gates"] == (h and pytest.approx(h))
                assert d["p"] == (h and pytest.approx(d["C_gates"] / h))
                assert d["family"] == next(x[3] for x in RUNS if x[1] == run_id)
                assert d["method_key"] == next(x[2] for x in RUNS if x[1] == run_id)
                assert panels.page_inputs(d)["w_fwd_gates"] == pytest.approx(policy.w_fwd(shape))
    s1 = {(d["record"], d["gain_target"]): (d["reached"], d["p"]) for d in ds if d["start"] == plots.start_label(SHAPE)}
    assert s1[("lora@1", 1.1)] == ("yes", 1) and s1[("qat@1", 1.1)] == ("yes", 2) and s1[("ref@3", 2)] == ("yes", 1)
    assert s1[("mezo@2", 1.1)][0] == s1[("lora@2", 2)][0] == s1[("ref@3", 11)][0] == "no"
    assert s1[("ref@3", 11)][1] is None and s1[("mezo@2", 2)][1] == pytest.approx(1.2 / 1.5)
    q = got["console"]
    assert q["columns"] == AP.CONSOLE_COLUMNS and len(q["columns"]) <= console.COLUMNS_MAX
    assert {"reading", "te_core", "rho_console", "rho_default", "rho_served"} <= set(q["columns"])
    assert [d["rho_default"] for d in ds] == [None if v == "inf" else v for v in cell["rho_console"]]
    assert q["rows"] == AP.console_rows(p["rows"]) and not console.refused(q)
    assert len(console.body(q)) <= console.BODY_MAX and len(q["note"]) <= console.NOTE_MAX
    assert all(not isinstance(v, bool) for r in q["rows"] for v in r)
    assert q["title"] == p["title"] and all(x["title"].endswith(f"{gates.UNIT})") for x in (p, got["fit"]))


def test_the_dial_formula_on_a_rows_columns_is_te_cores_at_every_cell(tmp_path, monkeypatch):
    """rho_default and basis_default are the dial's formula on the published columns at the console's default, and at
    the registered serving block rho_served is, so u_in is exact at both k; te_core.panels' dial on them gives
    formalism.at's ρ and basis at every cell; Fit's update application reads delta_t (qat's latents fail it where
    mezo, the same but for them, fits)."""
    _, got = run(tmp_path, monkeypatch)
    by = {r["id"]: r for r in ledger_rows()}
    ds = plots.frame(got["console"])
    assert {AP.CONSOLE[1], panels.SERVED[1]} <= set(panels.K_LEVELS)
    for d in ds:
        page = panels.page_inputs(d)
        assert not set(AP.SPLIT) & set(d) and set(AP.SPLIT) <= set(page)
        rho, basis = dial(page, *AP.CONSOLE)
        assert basis == d["basis_default"] and rho == pytest.approx(d["rho_default"], rel=1e-3)
        assert dial(page, *panels.SERVED)[0] == pytest.approx(d["rho_served"], rel=1e-3)
        r = by[d["record"].replace("@", "-")]
        m = formalism.measure(r, {"s1": SHAPE, "s2": SHAPE2}[r["start"]])
        w = page["w_fwd_gates"]
        assert panels.mismatches(d | {"run": d["record"]}, panels.dial(m, w), w) == []
    at = {d["record"]: d for d in ds if d["gain_target"] == 1.1}
    assert (at["mezo@2"]["basis_default"], at["mezo@2"]["rho_default"], at["qat@1"]["basis_default"]) == ("fit", 1,
                                                                                                         "target")
    assert json.loads(at["qat@1"]["rho_inputs"])["delta_t_bits"] == 6e4
    assert json.loads(at["mezo@2"]["rho_inputs"])["delta_bits"] == 1e3
    assert json.loads(at["mezo@2"]["rho_inputs"])["delta_t_bits"] == 1e3


@pytest.mark.parametrize("u1, dgrad, delta_t, tail, decoder", [
    (1e9, 3e9, None, 0.0, 0.0), (1e9, 3e9, None, 0.3, 0.0), (1e3, 3e4, None, 0.0, 0.0), (1e3, 3e4, 6e4, 0.0, 0.0),
    (2e4, 1e4, 3e4, 0.05, 1e9), (5e5, None, None, 0.2, 0.0), (5e4, 2e4, None, 0.0, 0.0), (1e3, 3e4, None, 0.3, None),
    (2e4, 1e4, None, 0.3, 1e13)])
def test_te_core_at_is_the_dial_formula(u1, dgrad, delta_t, tail, decoder):
    """Over decades of c and k, te_core.formalism.at's rho and basis are the draft note's formula on the inputs this
    panel publishes for the record (te_core's ρ is the score; the dial reruns it in the page), with d = 0, d below
    some c and above others, and d unmeasured (None: no block fits)."""
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
    m = formalism.from_json(block(1e15, 1e6, 3e6, None, 0.3) | {"labels": rescore.LABELS | rescore.TAIL})
    text = AP.estimated(m)
    assert text == ("w_gates:est tau_bits:stub a_bits:bound u_points:A1 u_in_points:A1' delta_bits:bound "
                    "delta_t_bits:bound d_gates:est delta_grad_bits:est")
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


def test_a_start_ahead_of_the_fit_has_every_target_below_its_loss_and_one_at_or_below_e_none():
    """Gain is Daniel's C(L1)/C(L0) on the fit (coordinator 11:16Z): a start the fit puts ahead of the frontier
    (its own gain C_eff(l0)/C0 1.5) has rows at every target, each below its l0, so a run that only holds l0 reaches
    none (ref-data's 11:09Z LOOPHOLE); a start at or below the fit's E has no rows, and a data without a fit none."""
    rich = LAW | {"A": LAW["A"] * 1.5 ** 0.3}
    assert FF.gain(rich, 1.2, C0) == pytest.approx(1.5)
    ds = [d for d in plots.frame(AP.build(info(rich), "test")) if d["start"] == plots.start_label(SHAPE)]
    assert {d["gain_target"] for d in ds} == set(AP.GAINS)
    for d in ds:
        target = FF.target(rich, 1.2, d["gain_target"])
        assert target < 1.2 and (d["reached"] == "yes") == (d["end_loss"] <= target)
        assert d["gain"] == pytest.approx(FF.gain_from(rich, 1.2, d["end_loss"]))
    assert {d["start"] for d in plots.frame(AP.build(info(LAW | {"E": 1.25}), "test"))} == {plots.start_label(SHAPE2)}
    assert AP.build(info(None), "test")["rows"] == []


def row(start, g, family, reached, p, rho, gain, rec, data="v2.3-pilot", gates=1.0, basis=None, served=None) -> list:
    d = dict.fromkeys(AP.COLUMNS) | {"start": start, "gain_target": g, "family": family, "reached": reached, "p": p,
                                    "rho_default": rho, "gain": gain, "record": rec, "data": data, "C_gates": gates,
                                    "basis_default": basis, "rho_served": served}
    return [d[c] for c in AP.COLUMNS]


def test_the_coverage_is_the_least_p_reached_row_else_the_best_failed_attempt_else_empty():
    """Per gain target, data and decade of ρ (an infinite ρ in the last): the reached row with the least p, else the
    failed attempt with the greatest gain, else EMPTY, which the table and the board line both say, with the reached
    cells' ρ at the registered serving block beside the console's default."""
    rows = [row("s", 2, "adapter", "yes", 3.0, 50.0, 2.1, "a"),
            row("s", 2, "full", "yes", 1.0, 20.0, 2.0, "b", served=1.5),
            row("s", 2, "adapter", "no", 0.5, 2.0, 1.5, "c"), row("s", 2, "adapter", "no", 0.2, 3.0, 1.7, "d"),
            row("s", 2, "seed", "no", 0.9, None, 1.1, "e"), row("s", 2, "full", "yes", 2.0, 9.0, 2.5, "f", served=1.0),
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
    served = ("Reached cells' rho: v2.3-pilot 9 to 20 at the console's default, 1 to 1.5 at the registered serving "
              "block (c 3.01e+13 gates, k 145,568 bits).")
    assert "| 2 | v2.3-pilot | yes p 1, rho 2.5" not in md and md.endswith("32 of 36 cells EMPTY; 2 hold only failed "
                                                                          f"attempts.\n{served}\n")
    assert "| 2 | v2.3-pilot | yes p 2, rho 9, served 1 (s, None, f) | yes p 1, rho 20, served 1.5 (s, None, b) | " in md
    line = AP.board_line(cov, fits, "T", "art:x")
    assert "g 2 v3: no fit, all 6 EMPTY" in line and "g 11 v2.3-pilot: EMPTY | EMPTY | EMPTY | no gain 1.9 | " in line
    assert line.endswith(f"32 of 36 cells EMPTY. {served}")


def test_an_unmeasured_rho_is_in_no_decade_and_an_infinite_one_in_the_last():
    """Under reading A a row whose ρ no kind valid on its own measures has rho_default null with basis `unmeasured`:
    the coverage puts it in no decade, never with reading B's null, an infinite ρ in the last, and the table and the
    board line count it."""
    rows = [row("s", 2, "adapter", "yes", 1.0, None, 2.1, "unmeasured", basis=AP.UNMEASURED),
            row("s", 2, "full", "no", 0.5, None, 1.5, "inf", basis="target"),
            row("s", 2, "full", "yes", 2.0, 37.5, 2.2, "decoding", basis="decoding")]
    p = {"columns": AP.COLUMNS, "rows": rows}
    cov = AP.coverage(p)
    held = {c["record"]: (c["rho_decade"], c["rows"]) for c in cov if c["state"] != "empty"}
    assert held == {"inf": ("1e5+", 1), "decoding": ("10", 1)} and AP.unmeasured_rows(p) == 1
    assert [AP.decade(None), AP.decade(None, "target"), AP.decade(None, AP.UNMEASURED)] == [5, 5, None]
    fits = {"v3": None, "v2.3-pilot": LAW}
    assert AP.coverage_md(cov, fits, "T", unmeasured=1).endswith(
        "1 row with no measured rho under reading A (rho_default null, basis_default unmeasured), in no decade.\n")
    assert "reading A" not in AP.coverage_md(cov, fits, "T") + AP.board_line(cov, fits, "T", None)
    assert AP.board_line(cov, fits, "T", None, unmeasured=2).endswith(" 2 rows unmeasured under reading A, in no "
                                                                     "decade.")


def test_a_row_resting_on_a_stub_is_marked_and_kept_out_of_the_coverage_the_cuts_fronts_and_the_figure(tmp_path,
                                                                                                    monkeypatch):
    """A row whose ρ rests on a stub input (`estimated`'s stub tag; coordinator 08:09Z item 2) has basis_default
    `stub`, whatever its ρ: no decade holds it, the cut's fronts leave it out at every tier and the figure doesn't plot
    it, while the stored table keeps it and the table and board line count it."""
    stub = {"tau_bits": "stub: no count of the update's token-side bits"}
    rows = [r | {"formalism": r["formalism"] | {"labels": r["formalism"]["labels"] | stub}} if r["run_id"] == "lora"
            else r for r in ledger_rows()]
    shapes = {"s1": SHAPE, "s2": SHAPE2}
    scored = plots.load(rows, ("pin",), shapes)[0]
    p = AP.build({"starts": FF.starts(scored, shapes, RATE), "fits": {"v3": None, "v2.3-pilot": LAW}}, "test")
    ds = plots.frame(p)
    lora = [d for d in ds if d["record"].startswith("lora@")]
    assert len(lora) == len(AP.GAINS) and all(d["rho_default"] is not None for d in lora)
    assert all(d["basis_default"] == AP.STUB and "tau_bits:stub" in d["estimated"] for d in lora)
    assert all(d["basis_default"] != AP.STUB for d in ds if d not in lora)
    assert AP.decade(2.0, AP.STUB) is None and AP.unmeasured_rows(p, AP.STUB) == len(lora)
    cov = AP.coverage(p)
    assert sum(c["rows"] for c in cov) == len(ds) - len(lora)
    assert not any((c.get("record") or "").startswith("lora@") for c in cov)
    rec, held = AP.COLUMNS.index("record"), {c["row"] for c in cov if c["row"] is not None}
    for tier in AP.TIERS:
        assert not any(r[rec].startswith("lora@") for r in AP.kept(p["rows"], tier, held))
    fits = {"v3": None, "v2.3-pilot": LAW}
    assert "3 rows rest on a stub input (basis_default stub), in no decade." in AP.coverage_md(cov, fits, "T", stubs=3)
    assert AP.board_line(cov, fits, "T", None, stubs=3).endswith(" 3 rows rest on a stub, in no decade.")
    plotted, real = set(), AP.plt.Axes.plot

    def plot(ax, *a, **k):
        if len(a) >= 2 and all(isinstance(x, float) for x in a[:2]):
            plotted.add(a[:2])
        return real(ax, *a, **k)

    monkeypatch.setattr(AP.plt.Axes, "plot", plot)
    AP.figure(p, tmp_path)
    points = {(d["p"], d["rho_default"]) for d in ds if d["p"] is not None}
    assert plotted and plotted <= points and not plotted & {(d["p"], d["rho_default"]) for d in lora}


def test_a_rows_rho_is_published_at_the_consoles_default_and_the_serving_blocks_and_follows_published_to_reading_a(
        monkeypatch):
    """rho_default and basis_default are te_core.formalism's `published` at the console's default (`AP.CONSOLE`), the
    one rho_console is at, and rho_served is it at the registered serving block (te_core.panels' `SERVED`): reading
    B's token form or Fit's until formalism.READING_A, then the one bound's max with its binding kind, null with basis
    `unmeasured` where no kind valid on its own is measured. A stand-in `published` reads A on two runs: nothing
    measured on lora's, decoding binding on never's."""
    real, at = formalism.published, []

    def published(m, c=None, k=None, reading=None):
        at.append(AP.CONSOLE[1] if k is None else k)
        one = {8e6: {"rho": None, "binding_kind": "none", "rests_on": "unmeasured"},
               1e6: {"rho": 37.5, "binding_kind": "decoding", "rests_on": formalism.RESTS_ON_CELL["decoding"]}}
        return real(m, c, k, reading) | ({"reading": "A"} | one[m.delta_bits] if m.delta_bits in one else {})

    monkeypatch.setattr(formalism, "published", published)
    p = AP.build(info(), "test")
    ds, by = plots.frame(p), {r["id"]: r for r in ledger_rows()}
    assert set(at) == {AP.CONSOLE[1], panels.SERVED[1]} and {"lora", "never"} <= {d["record"].split("@")[0] for d in ds}
    for d in ds:
        run = d["record"].split("@")[0]
        got = (d["reading"], d["rho_default"], d["basis_default"], d["rho_served"])
        if run == "lora":
            assert got == ("A", None, AP.UNMEASURED, None) and d["rho_console"] is None
        elif run == "never":
            assert got == ("A", 37.5, "decoding", 37.5) and d["rho_console"] == 37.5
        else:
            r = by[d["record"].replace("@", "-")]
            m = formalism.measure(r, {"s1": SHAPE, "s2": SHAPE2}[r["start"]])
            v, s = formalism.at(m, *AP.CONSOLE), formalism.at(m, *panels.SERVED)
            assert got == ("B", AP.finite(v["rho"]) and pytest.approx(v["rho"]), "fit" if v["fit"] else "target",
                           AP.finite(s["rho"]) and pytest.approx(s["rho"]))
    cov, lora = AP.coverage(p), sum(d["record"].startswith("lora@") for d in ds)
    assert lora == len(AP.GAINS) == AP.unmeasured_rows(p) and sum(c["rows"] for c in cov) == len(ds) - lora
    assert not any((c.get("record") or "").startswith("lora@") for c in cov)


def test_a_partial_row_is_drawn_marked_but_in_no_cell_on_no_others_front_and_never_the_least_s(tmp_path, monkeypatch):
    """A row `published` marks partial (a measured kind's term over 1 with a valid kind unmeasured, so ρ is only a lower
    bound; coordinator 12:20Z) has basis_default `partial`: no decade holds it, the cut picks the partial rows among
    themselves alone, so the others' picks are as if they weren't there, and the figure draws each with a caret but
    never rings one as the least S. A stand-in `published` makes the adapters' rows (lora's and the dominated ones')
    partial at ρ 1.01, which would put lora's at the least S at gain 1.1."""
    real = formalism.published

    def published(m, c=None, k=None, reading=None):
        one = real(m, c, k, reading)
        return one if m.delta_bits != 8e6 else one | {"reading": "A", "rho": 1.01, "binding_kind": "token",
                                                      "rests_on": formalism.RESTS_ON_CELL["token"], "partial": True}

    monkeypatch.setattr(formalism, "published", published)
    p = AP.build(info(runs=RUNS + DOMINATED), "test")
    ds = plots.frame(p)
    part = [d["record"].startswith("lora") for d in ds]
    assert any(part) and all((d["basis_default"], d["rho_default"], d["rho_console"], d["binding_kind"])
                             == (AP.PARTIAL, 1.01, 1.01, "token") and json.loads(d["kinds"])["partial"]
                             for d, x in zip(ds, part) if x)
    assert all(d["basis_default"] != AP.PARTIAL for d, x in zip(ds, part) if not x)
    cov = AP.coverage(p)
    assert sum(c["rows"] for c in cov) == len(ds) - sum(part) == len(ds) - AP.unmeasured_rows(p, AP.PARTIAL)
    assert not any((c.get("record") or "").startswith("lora") for c in cov)
    rest = p | {"rows": [r for r, x in zip(p["rows"], part) if not x]}
    rec, g = AP.COLUMNS.index("record"), AP.COLUMNS.index("gain_target")
    for tier in AP.TIERS:
        got, alone = ({(r[rec], r[g]) for r in AP.kept(q["rows"], tier, {c["row"] for c in AP.coverage(q)
                                                                           if c["row"] is not None})} for q in (p, rest))
        assert {x for x in got if not x[0].startswith("lora")} == alone
        assert (("lora@1", 1.1) in got) == (tier != "coverage")
    fits = {"v3": None, "v2.3-pilot": LAW}
    assert ("2 rows are partial (basis_default partial: a valid kind unmeasured, so rho is only a lower bound), in no "
            "decade.") in AP.coverage_md(cov, fits, "T", partial=2)
    assert AP.board_line(cov, fits, "T", None, partial=1).endswith(" 1 row partial (rho a lower bound), in no decade.")
    assert "(1, 1.01)" not in AP.summary(p, info(runs=RUNS + DOMINATED))
    calls, real_plot = [], AP.plt.Axes.plot

    def plot(ax, *a, **k):
        calls.append((a[:2], k))
        return real_plot(ax, *a, **k)

    monkeypatch.setattr(AP.plt.Axes, "plot", plot)
    AP.figure(p, tmp_path)
    marked = {(d["p"], d["rho_default"]) for d, x in zip(ds, part) if x and d["p"] is not None}
    assert marked and marked <= {xy for xy, k in calls if k.get("marker") == 6}
    rings = {xy for xy, k in calls if k.get("mec") == "red"}
    assert rings and not rings & marked


def test_the_output_leads_with_the_coverage_table(tmp_path, monkeypatch, capsys):
    """The scorer's refresh posts the first Markdown table under the first line of the output naming coverage: that is
    tables/coverage.md's table, every gain target and data a row."""
    assert run(tmp_path, monkeypatch)[0] == 0
    lines = capsys.readouterr().out.splitlines()
    at = next(i for i, x in enumerate(lines) if "coverage" in x.lower())
    first = next(i for i in range(at, len(lines)) if lines[i].startswith("|"))
    table = lines[first:next(i for i in range(first, len(lines)) if not lines[i].startswith("|"))]
    md = (tmp_path / "out" / "tables" / "coverage.md").read_text().splitlines()
    assert table == [x for x in md if x.startswith("|")] and len(table) == 2 + len(AP.GAINS) * len(panels.DATA)


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
    assert AP.cut(p, art)["rows"] == AP.console_rows(p["rows"]) and AP.cut(p, art)["note"] == AP.note(art)
    cov = {c["row"] for c in AP.coverage(p) if c["row"] is not None}
    size = {t: len(console.body(AP.checked(p | {"columns": AP.CONSOLE_COLUMNS, "note": AP.note(art, t),
                                                "rows": AP.console_rows(AP.kept(p["rows"], t, cov))}))) for t in AP.TIERS}
    assert len(console.body(AP.cut(p, art))) > size["family"] > size["start"] > size["coverage"]
    for t in AP.TIERS:
        monkeypatch.setattr(console, "BODY_MAX", size[t])
        q = AP.cut(p, art)
        assert q["note"] == AP.note(art, t) and len(q["note"]) <= console.NOTE_MAX and not console.refused(q)
        assert {p["rows"][i][AP.COLUMNS.index("record")] for i in cov} <= {r[AP.CONSOLE_COLUMNS.index("record")]
                                                                          for r in q["rows"]}
    monkeypatch.setattr(console, "BODY_MAX", size["coverage"] - 1)
    with pytest.raises(ValueError, match="the coverage's rows alone"):
        AP.cut(p, art)


def test_publish_stores_the_tables_and_puts_both_panels_the_cut_naming_their_art_id(tmp_path, monkeypatch, capsys):
    """--publish puts tables/ in the evidence store (the uncut panel, its coverage and the fit's panel and points) and
    PUTs training/frontier-fit, then the cut, whose note names the art id, a line each; a refused PUT fails the
    command."""
    sent, stored = [], []
    monkeypatch.setattr(console, "put", lambda q: sent.append(q) or "200")
    monkeypatch.setattr(console, "served", lambda: {})
    monkeypatch.setattr(AP, "store", lambda path, meta: stored.append(sorted(f.name for f in path.iterdir()))
                        or "art:" + "a" * 64)
    runs = RUNS + DOMINATED
    assert run(tmp_path / "dry", monkeypatch, runs=runs)[0] == 0 and not stored and not sent
    capsys.readouterr()
    code, got = run(tmp_path / "fits", monkeypatch, "--publish", runs=runs)
    assert code == 0 and stored == [["algorithm-policy.json", "coverage.json", "coverage.md", "frontier-fit.json",
                                     "frontier-points.json", "refused.json"]]
    puts = [x for x in capsys.readouterr().out.splitlines() if x.startswith("PUT ")]
    assert puts == [f"PUT {console.PANELS_URL}/{FF.PANEL}: 200", f"PUT {console.PANELS_URL}/{AP.PANEL}: 200"]
    assert [q["id"] for q in sent] == [FF.PANEL, AP.PANEL] and console.body(sent[1]) == console.body(got["console"])
    assert len(sent[1]["rows"]) == len(got["panel"]["rows"]) and sent[1]["note"].endswith(f" Tables art:{'a' * 64}.")
    monkeypatch.setattr(console, "BODY_MAX", len(console.body(sent[1])) - 1)
    code, got = run(tmp_path / "cut", monkeypatch, "--publish", runs=runs)
    assert code == 0 and len(got["console"]["rows"]) < len(got["panel"]["rows"])
    assert " Cut: coverage + per start, g, family: " in got["console"]["note"]
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
    assert not any("gain:provisional" in d["estimated"] for d in plots.frame(got["panel"]))


def served_as(p: dict, version: str | None, rows: list | None = None, source: str = "") -> dict:
    """`p` (uncut, or a console copy) as the route would serve it, its rows (or `rows`) priced at te_core `version`;
    with None, from before the te_core cell (its version in `source` alone)."""
    te = p["columns"].index("te_core")
    rows = p["rows"] if rows is None else rows
    if version is None:
        return p | {"columns": p["columns"][:te] + p["columns"][te + 1:], "rows": [r[:te] + r[te + 1:] for r in rows],
                    "source": source}
    return p | {"rows": [r[:te] + [version] + r[te + 1:] for r in rows]}


def test_a_version_switch_never_empties_a_populated_panel():
    """A data's served rows stay where they are at a newer te_core, or at an older one with a run there this build has
    no row for (coordinator 07:58Z); the same te_core's are replaced, and an older one's where this build has every
    run, though its coverage fills fewer cells (every rho 1 here; coordinator 11:51Z).
    A data with no served rows takes this build's. A panel from before the te_core cell is at the version its source
    names."""
    p = AP.build(info(runs=RUNS + DOMINATED), "test")
    mine = tuple(map(int, AP.VERSION.split(".")))
    newer, older = f"{mine[0] + 1}.0", f"{mine[0] - 1}.9"
    one = [r for r in p["rows"] if r[AP.COLUMNS.index("record")] == "never@1"][:1]
    assert one and AP.runs(p, "v2.3-pilot") > AP.runs({"columns": AP.COLUMNS, "rows": one}, "v2.3-pilot") == {"never"}
    sparse = p | {"rows": one}
    rho = AP.COLUMNS.index("rho_default")
    flat = p | {"rows": [r[:rho] + [1.0] + r[rho + 1:] for r in p["rows"]]}
    cells = [sum(c["state"] != "empty" for c in AP.coverage(q) if c["data"] == "v2.3-pilot") for q in (flat, p)]
    assert cells[0] < cells[1] and AP.kept_data(flat, served_as(p, older)) == []
    assert AP.kept_data(p, None) == AP.kept_data(p, served_as(p, AP.VERSION)) == []
    assert AP.kept_data(sparse, served_as(p, AP.VERSION)) == []
    assert AP.kept_data(p, served_as(sparse, newer)) == ["v2.3-pilot"]
    assert AP.kept_data(sparse, served_as(p, older)) == ["v2.3-pilot"]
    assert AP.kept_data(p, served_as(sparse, older)) == AP.kept_data(p, served_as(p, older)) == []
    legacy = served_as(p, None, source=f"python -m te_core.algorithm_policy (te_core {older}; pin x)")
    assert AP.version_of(legacy, "v2.3-pilot") == tuple(map(int, older.split("."))) and not AP.version_of(legacy, "v3")
    assert AP.kept_data(sparse, legacy) == ["v2.3-pilot"]
    fp = {"id": FF.PANEL, "columns": ["row", "data", "E_bpb"], "rows": [["fit", "v3", 1.0], ["fit", "v2.3-pilot", 0.9]]}
    fit = fp | {"rows": [["fit", "v2.3-pilot", 0.8]]}
    served = {AP.PANEL: served_as(sparse, newer), FF.PANEL: fit}
    v3 = [r[:AP.COLUMNS.index("data")] + ["v3"] + r[AP.COLUMNS.index("data") + 1:] for r in p["rows"][:2]]
    pub, fpub, held = AP.guarded(p | {"rows": v3 + p["rows"]}, fp, served)
    assert held == f"v2.3-pilot (te_core {newer})"
    assert pub["rows"][:2] == v3 and pub["rows"][2:] == [[d.get(c) for c in AP.COLUMNS]
                                                         for d in plots.frame(served_as(sparse, newer))]
    assert fpub["rows"] == [["fit", "v3", 1.0], ["fit", "v2.3-pilot", 0.8]]
    assert AP.guarded(p, fp, {AP.PANEL: served_as(p, AP.VERSION)}) == (p, fp, "")
    assert AP.cut(pub, "art:x", held)["note"].endswith(f" Kept as served: v2.3-pilot (te_core {newer}). Tables art:x.")


def test_publish_puts_nothing_without_the_routes_panels_and_keeps_what_a_newer_te_core_serves(tmp_path, monkeypatch,
                                                                                             capsys):
    """--publish GETs the route's panels first: when that fails it stores and PUTs nothing, and when the route serves
    the pilot at a newer te_core (its console copy, without CONSOLE_DROP's columns) both PUTs keep its rows there,
    which the note and the board line name."""
    sent, stored = [], []
    monkeypatch.setattr(console, "put", lambda q: sent.append(q) or "200")
    monkeypatch.setattr(AP, "store", lambda path, meta: stored.append(path) or "art:" + "a" * 64)

    def down():
        raise OSError("the route is down")

    monkeypatch.setattr(console, "served", down)
    assert AP.main(args(tmp_path / "down", monkeypatch) + ["--publish"]) == 1 and not sent and not stored
    assert "not PUT: GET " in capsys.readouterr().err
    _, dry = run(tmp_path / "dry", monkeypatch)
    capsys.readouterr()
    mine = tuple(map(int, AP.VERSION.split(".")))
    newer = f"{mine[0] + 1}.0"
    ap = served_as(dry["console"], newer, dry["console"]["rows"][:3])
    monkeypatch.setattr(console, "served", lambda: {AP.PANEL: ap, FF.PANEL: dry["fit"]})
    code, got = run(tmp_path / "kept", monkeypatch, "--publish")
    assert code == 0 and [q["id"] for q in sent] == [FF.PANEL, AP.PANEL] and len(stored) == 1
    assert sent[1]["rows"] == ap["rows"] and got["console"] == json.loads(console.body(sent[1]))
    assert f" Kept as served: v2.3-pilot (te_core {newer}). Tables art:" in sent[1]["note"]
    assert len(got["panel"]["rows"]) == len(dry["panel"]["rows"])
    board = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("board: "))
    assert board.endswith(f" Kept as served: v2.3-pilot (te_core {newer}).")


def test_without_ladder_points_v3_reads_the_manifests_and_a_fit_on_four_from_scratch_is_provisional(tmp_path,
                                                                                                   monkeypatch):
    """With no --ladder-points v3 reads te_core.frontier_fit's LADDER, the ladder's starts built from its manifest:
    t1's start and three of them on v3's frontier are four from scratch, under five, so v3's fit is provisional, its
    status, the fit panel's note (its points with their residuals, the ladder's runs it doesn't read with theirs, and
    where the points come from), every v3 row's `estimated` and the console copy's note say so; the pilot's fit, on
    its many, isn't."""
    v3_law = LAW | {"E": 1.2}
    tree = tmp_path / "ladder"
    tree.mkdir()
    near = 16 * (1 - 1e-4)
    assert math.floor(FF.PER_DECADE * math.log10(near * C0)) == math.floor(FF.PER_DECADE * math.log10(16 * C0))
    annealed = [{"branch": f"x{f:g}", "macs": f * C0 / per_mac("gemm/bf16"), "bpb": law(f * C0, v3_law)}
                for f in (2, 4, 16)]
    annealed += [{"branch": "over", "macs": 8 * C0 / per_mac("gemm/bf16"), "bpb": law(4 * C0, v3_law) + 0.01},
                 {"branch": "binned", "macs": near * C0 / per_mac("gemm/bf16"), "bpb": law(16 * C0, v3_law) + 0.01}]
    (tree / FF.LADDER_FILE).write_text(json.dumps({"schema": "te-ladder/F/v1", "annealed": annealed}))
    v3 = [r | {"id": f"v3-{r['id']}", "run_id": f"v3-{r['run_id']}", "data": "v3", "start": "t1", "start_bpb": 1.5,
               "val_bpb": 1.5 + 0.001 * r["checkpoint_step"]} for r in ledger_rows() if r["start"] == "s1"]
    a = args(tmp_path, monkeypatch, index=ledger_rows() + v3)
    monkeypatch.setattr(FF, "LADDER", "art:" + "b" * 64)
    monkeypatch.setattr(panels, "fetch", lambda art: {FF.LADDER: tree}[art])
    assert AP.main(a) == 0
    tables = tmp_path / "out" / "tables"
    fp, p = (json.loads((tables / f"{name}.json").read_text()) for name in (FF.NAME, AP.NAME))
    fits = {d["data"]: d for d in plots.frame(fp) if d["row"] == "fit"}
    assert fits["v3"]["status"].startswith("provisional fit: 4 of its points from scratch, under 5; ")
    assert fits["v2.3-pilot"]["status"].startswith("fit; ")
    assert fp["source"].endswith(f"; 5 ladder points ({FF.LADDER[:16]}))")
    assert p["source"] == fp["source"].replace("te_core.frontier_fit", "te_core.algorithm_policy", 1)
    assert fp["note"].startswith(FF.note() + " ") and len(fp["note"]) <= console.NOTE_MAX
    f, gates = fitted(fp, "v3"), per_mac("gemm/bf16")
    assert f["E"] == pytest.approx(1.2, abs=1e-3)
    off = {p["branch"]: f"{round(100 * (p['bpb'] / FF.loss_at(f, p['macs'] * gates) - 1), 2):+.2f}%"
           for p in annealed[3:]}
    assert fp["note"][len(FF.note()) + 1:] == (
        "v3 provisional (4 from scratch, 5 the target): t1 +0.00%, x2 +0.00%, x4 +0.00%, x16 +0.00%; off it over "
        f"{off['over']}, binned {off['binned']}. Ladder points built from its manifest.")
    assert float(off["over"][:-1]) > 3 and float(off["binned"][:-1]) > 0.5
    ds = plots.frame(p)
    assert {d["data"] for d in ds} == {"v3", "v2.3-pilot"}
    assert all(("gain:provisional" in d["estimated"]) == (d["data"] == "v3") for d in ds)
    q = json.loads((tmp_path / "out" / "console" / f"{AP.NAME}.json").read_text())
    assert " v3: provisional fit." in q["note"] and len(q["note"]) <= console.NOTE_MAX


def test_the_rows_te_core_refuses_are_listed_and_counted_never_scored(tmp_path, monkeypatch, capsys):
    """te_core.formalism's refusals: mezo's capture lists a clip on a read-back loss (data-dependent) and never's was
    traced before te_core 4.2 (no read-back lists). Neither has a row, so no coverage cell holds them; refused.json gives
    each checkpoint's reason, and the coverage table, the board line, both panels' source and the output count them."""
    trace = {"affected_share": 0.5, "u1_bits": 1e9}
    clip = {"kind": "a comparison of a read-back float", "where": "train.py:399", "steps": [0, 4], "count": 5}
    traced = {"mezo": trace | {"data_dependent": [clip], "carried_scalars": []}, "never": trace}
    rows = [r | {"formalism": r["formalism"] | {"trace": traced[r["run_id"]]}} if r["run_id"] in traced else r
            for r in ledger_rows()]
    code, got = run(tmp_path, monkeypatch, index=rows)
    refused = json.loads((got["out"] / "tables" / "refused.json").read_text())
    assert code == 0 and refused == {r["id"]: formalism.refusal(r) for r in rows if r["run_id"] in traced}
    assert refused["mezo-1"].startswith("data-dependent") and refused["never-1"].startswith("read-backs unchecked")
    assert not {d["record"].split("@")[0] for d in plots.frame(got["panel"])} & set(traced)
    assert not {c.get("record", "").split("@")[0] for c in got["coverage"]} & set(traced)
    kinds = "2 data-dependent, 1 read-backs unchecked"
    assert f"3 valid checkpoints are refused by te_core.formalism, never scored, so in no cell ({kinds}; " in \
        (got["out"] / "tables" / "coverage.md").read_text()
    assert "; 3 refused; " in got["panel"]["source"] and "; 3 refused; " in got["fit"]["source"]
    out = capsys.readouterr().out
    assert f"3 valid checkpoints refused, never scored: {kinds}" in out
    assert f" 3 valid checkpoints refused by te_core.formalism, never scored ({kinds})." in \
        next(x for x in out.splitlines() if x.startswith("board: "))


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
