"""te_core.algorithm_policy builds `training/algorithm-policy`: one row per run and end loss it reaches, at its first
checkpoint reaching it, with p = C/H (H the start's honest envelope there) beside the inputs of its rho; the dial's
formula on a row's columns alone is te_core.formalism.at's (Fit's update application reading delta_t); the inputs
that aren't measured are named; and the console's cut keeps each family's (p, rho) front and its least-p and
least-delta runs, naming the full table's art id."""
import json
import math

import pytest

pytest.importorskip("matplotlib")

from te_core import algorithm_policy as AP
from te_core import console, formalism, panels, plots, rescore
from test_plots import SHAPE

SHAPE2 = SHAPE | {"tokens": 2 * SHAPE["tokens"], "tokens_per_param": 6.84}
#: a token block's gates: the forward and a backward of twice its cost
W = 3 * panels.w_fwd(SHAPE)
C_DEFAULT, K_DEFAULT = AP.default(panels.w_fwd(SHAPE))
#: (start, run, direction/mechanism, family, Δ, Δ_grad, Δ_t, tail share, [(step, gates, bpb), ...])
RUNS = [("s1", "ref", "ref-optim/adamw", "full", 9e8, 3e9, None, 0.0, [(1, 1e15, 1.193), (2, 2e15, 1.185),
                                                                      (3, 4e15, 1.17)]),
        ("s1", "lora", "lowrank/lora", "adapter", 8e6, 2e7, None, 0.3, [(1, 5e14, 1.195), (2, 3e15, 1.187)]),
        ("s1", "mezo", "seed-mezo/mezo", "seed", 1e3, 3e4, None, 0.0, [(1, 2e15, 1.193), (2, 8e15, 1.192)]),
        ("s1", "qat", "quant/qat", "update-code", 1e3, 3e4, 6e4, 0.0, [(1, 3e15, 1.193)]),
        ("s1", "never", "sparse/lines", "partial", 1e6, 4e6, None, 0.0, [(1, 1e15, 1.199)]),
        ("s2", "ref2", "ref-optim/adamw", "full", 9e8, 3e9, None, 0.0, [(1, 1e14, 1.28)])]
L0 = {"s1": 1.2, "s2": 1.3}


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
        for step, gates, bpb in cps:
            out.append({"id": f"{run}-{step}", "run_id": run, "checkpoint_step": step, "start": start,
                        "start_bpb": L0[start], "val_bpb": bpb, "direction": direction, "mechanism": mechanism,
                        "family": family, "measurement_commit": "pin", "output": {"e_bytes": delta / 8},
                        "written_ts": len(out), "formalism": block(gates, delta, dgrad, delta_t, tail)})
    return out


def run(tmp_path, monkeypatch, runs=RUNS, *extra) -> tuple[int, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    man = tmp_path / "starts_store.json"
    man.write_text(json.dumps({"starts": {"s1": {"shape": SHAPE}, "s2": {"shape": SHAPE2}}}))
    monkeypatch.setattr(rescore, "Ledger", lambda root: type("L", (), {"index": staticmethod(lambda: ledger_rows(runs)),
                                                                      "root": tmp_path})())
    out = tmp_path / "out"
    code = AP.main(["--out", str(out), "--pin", "pin", "--manifest", str(man), "--stubs", *extra])
    return code, {"out": out, "panel": json.loads((out / f"{AP.NAME}.json").read_text()),
                  "console": json.loads((out / "console" / f"{AP.NAME}.json").read_text())}


def dial(d: dict, c: float, k: float) -> tuple[float, str]:
    """The draft note's formula (note:training-20261102T2334Z-draft-rho-inputs-and-formula) on a row's columns alone,
    written apart from te_core.formalism: Fit for the training, else Slowdown's bound from u and u_in's points."""
    s, dg = c / d["w_gates"], d["delta_grad_bits"]
    if d["delta_bits"] + s * d["tau_bits"] <= k and dg is not None and 2 * dg <= k and d["delta_t_bits"] + dg <= k:
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
    lead = d["sigma"] * (1 - x0) * s
    return (1.0 if lead == 0 else max(1.0, lead * d["a_bits"] / best) if best > 0 else math.inf), "target"


def test_one_row_per_run_and_end_loss_at_its_first_checkpoint(tmp_path, monkeypatch):
    """Each run reaching a target gives one row there, at its first checkpoint reaching it; H is the least gates of any
    checkpoint from the same start reaching it, so p >= 1 and the setter's is 1; a run reaching no target has none."""
    code, got = run(tmp_path, monkeypatch)
    assert code == 0 and (got["out"] / f"{AP.NAME}.png").stat().st_size > 10_000
    p = got["panel"]
    assert (p["id"], p["kind"], p["columns"]) == (AP.PANEL, "table", AP.COLUMNS) and p["source"] and p["updated_at"]
    ds = plots.frame(p)
    at = {(d["record"], d["target_pct"]): d for d in ds}
    assert set(at) == {("ref@1", -0.5), ("lora@2", -0.5), ("mezo@1", -0.5), ("qat@1", -0.5), ("ref@2", -1),
                       ("lora@2", -1), ("ref@3", -2), ("ref2@1", -0.5), ("ref2@1", -1)}
    for (rec, tp), d in at.items():
        start = "s1" if not rec.startswith("ref2") else "s2"
        end = L0[start] * (1 + tp / 100)
        reach = [r for r in ledger_rows() if r["start"] == start and r["val_bpb"] <= end]
        assert d["end_loss"] == pytest.approx(end) and d["start_loss"] == L0[start]
        assert d["H_gates"] == min(r["formalism"]["parts"][0]["gates"] for r in reach)
        assert d["p"] == pytest.approx(d["C_gates"] / d["H_gates"]) and d["p"] >= 1
        assert d["start"] == plots.start_label(SHAPE if start == "s1" else SHAPE2) and d["data"] == "v2.3-pilot"
        assert d["w_fwd_gates"] == pytest.approx(panels.w_fwd(SHAPE)) and d["w_gates"] == pytest.approx(W)
    assert [at[("lora@2", -0.5)][k] for k in ("C_gates", "p", "method", "family")] == [3e15, 3, "lowrank/lora", "adapter"]
    assert at[("ref@2", -1)]["p"] == 1 and at[("ref2@1", -1)]["H_gates"] == 1e14
    q = got["console"]
    assert q["rows"] == [AP.rounded(r) for r in p["rows"]] and not console.refused(q)
    assert len(console.body(q)) <= console.BODY_MAX and len(q["note"]) <= console.NOTE_MAX
    assert all(not isinstance(v, bool) for r in q["rows"] for v in r)


def test_the_dial_formula_on_a_rows_columns_is_te_cores(tmp_path, monkeypatch):
    """rho_default and basis_default are the dial's formula on the published columns; Fit's update application reads
    delta_t (qat's latents fail it where mezo, the same but for them, fits)."""
    _, got = run(tmp_path, monkeypatch)
    ds = {d["record"]: d for d in plots.frame(got["console"]) if d["target_pct"] == -0.5}
    for d in ds.values():
        rho, basis = dial(d, *AP.default(d["w_fwd_gates"]))
        assert basis == d["basis_default"] and rho == pytest.approx(d["rho_default"], rel=1e-3)
    assert (ds["mezo@1"]["basis_default"], ds["mezo@1"]["rho_default"], ds["qat@1"]["basis_default"]) == ("fit", 1, "target")
    assert ds["qat@1"]["delta_t_bits"] == 6e4 and ds["mezo@1"]["delta_t_bits"] == ds["mezo@1"]["delta_bits"] == 1e3


@pytest.mark.parametrize("u1, dgrad, delta_t, tail, decoder", [
    (1e9, 3e9, None, 0.0, None), (1e9, 3e9, None, 0.3, None), (1e3, 3e4, None, 0.0, None), (1e3, 3e4, 6e4, 0.0, None),
    (2e4, 1e4, 3e4, 0.05, 1e9), (5e5, None, None, 0.2, None), (5e4, 2e4, None, 0.0, None)])
def test_te_core_at_is_the_dial_formula(u1, dgrad, delta_t, tail, decoder):
    """Over decades of c and k, te_core.formalism.at's rho and basis are the draft note's formula on the inputs this
    panel publishes for the record (te_core's ρ is the score; the dial reruns it in the page)."""
    m = formalism.from_json(block(1e15, u1, dgrad, delta_t, tail) | {"decoder_gates": decoder, "labels": {}})
    d = dict(zip(AP.INPUTS, AP.inputs(m, panels.w_fwd(SHAPE))))
    for c in (1e10, 1e12, 1e14, 1e16):
        for k in (1e3, 1e4, 8e4, 1e6, 1e8, 1e10):
            v = formalism.at(m, c, k)
            rho, basis = dial(d, c, k)
            assert basis == ("fit" if v["fit"] else "target"), (c, k)
            assert rho == pytest.approx(v["rho"], rel=1e-4), (c, k)
            assert AP.page_rho(d, c, k) == (pytest.approx(rho, rel=1e-4), basis)


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


def row(start, tp, family, p, rho, delta, rec) -> list:
    d = dict.fromkeys(AP.COLUMNS) | {"start": start, "target_pct": tp, "family": family, "p": p, "rho_default": rho,
                                    "delta_bits": delta, "record": rec}
    return [d[c] for c in AP.COLUMNS]


def test_the_cut_keeps_each_familys_front_and_its_least_p_and_least_delta():
    """Per start, target and family: the lower-left (p, rho) front, plus the least-p and the least-delta run."""
    rows = [row("s", -0.5, "adapter", 1.0, 50.0, 8e6, "a"), row("s", -0.5, "adapter", 2.0, 10.0, 8e6, "b"),
            row("s", -0.5, "adapter", 3.0, 20.0, 8e6, "dominated"), row("s", -0.5, "adapter", 4.0, 60.0, 1e5, "tiny"),
            row("s", -0.5, "adapter", 5.0, 5.0, 8e6, "c"), row("s", -0.5, "full", 1.5, 90.0, 9e8, "f"),
            row("s", -1, "adapter", 3.0, 20.0, 8e6, "dominated"), row("t", -0.5, "adapter", 3.0, 20.0, 8e6, "dominated")]
    assert [r[AP.COLUMNS.index("record")] for r in AP.kept(rows)] == ["a", "b", "tiny", "c", "f", "dominated",
                                                                      "dominated"]


def test_a_cut_names_the_full_tables_art_id_and_publish_stores_it(tmp_path, monkeypatch):
    """When the whole table doesn't fit the route, --publish puts it in the evidence store and PUTs the cut, whose note
    names its art id; when it fits, the store isn't touched and the cut is the table."""
    extra = [("s1", f"lora{i}", "lowrank/lora", "adapter", 8e6, 2e7, None, 0.3, [(1, (4 + i) * 1e15, 1.19)])
             for i in range(6)]
    sent, stored = [], []
    monkeypatch.setattr(console, "put", lambda q: sent.append(q) or "200")
    monkeypatch.setattr(AP, "store", lambda path, meta: stored.append(json.loads(path.read_text())) or "art:" + "a" * 64)
    code, got = run(tmp_path / "fits", monkeypatch, RUNS + extra, "--publish")
    assert code == 0 and not stored and len(sent) == 1 and len(sent[0]["rows"]) == len(got["panel"]["rows"])
    monkeypatch.setattr(console, "BODY_MAX", len(console.body(sent[0])) - 1)
    code, got = run(tmp_path / "cut", monkeypatch, RUNS + extra, "--publish")
    q, p = got["console"], got["panel"]
    assert code == 0 and stored == [p] and console.body(sent[-1]) == console.body(q)
    assert len(q["rows"]) == len(p["rows"]) - 6 and q["note"].endswith(f"full table art:{'a' * 64}.")
    assert len(console.body(q)) <= console.BODY_MAX and len(q["note"]) <= console.NOTE_MAX
    monkeypatch.setattr(console, "put", lambda q: "403 forbidden")
    assert run(tmp_path / "refused", monkeypatch, RUNS, "--publish")[0] == 1


def test_a_record_its_columns_cant_reproduce_is_refused():
    """Token blocks that differ between parts take Slowdown's affected-count ratio, which one w and sigma can't carry."""
    parts = (formalism.Part(1e15, 1e3, 0.9e15, 0.8e15, 1, 3e9), formalism.Part(1e15, 1e3, 0.5e15, 0.1e15, 1, 3e9))
    m = formalism.Measured(parts, 16, 8192, formalism.even(1e9), formalism.even(1e9), 1e9)
    r = {"id": "x-1", "run_id": "x", "checkpoint_step": 1, "start": "s1", "start_bpb": 1.2, "val_bpb": 1.19,
         "gates": m.gates, "measured": m, "direction": "d", "mechanism": "m"}
    with pytest.raises(ValueError, match="per-part columns"):
        AP.build([r], {"s1": SHAPE}, "test")
