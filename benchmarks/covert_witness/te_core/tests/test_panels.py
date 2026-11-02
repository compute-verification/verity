"""te_core.panels builds `training/frontier-branches`: the frontier over the from-scratch runs and every start's
continuations, each start's honest and throttled curves at total compute C0 + after, S = W*/H >= 1, `on_frontier`
when no from-scratch point beats the start, the dial cells whose curve is the honest one left out, and a console copy
that fits the panels route."""
import json
import math

import pytest

pytest.importorskip("matplotlib")

from te_core import console, panels, plots, rescore
from test_plots import SHAPE, rows

RATIO = 1.03
L_PARAMS = next(iter(panels.FORWARD_MACS))


def ledger_rows():
    return [r | ({"mechanism": "adamw-all"} if r["direction"] == "ref-optim" else {}) for r in rows()]


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


def build(tmp_path, monkeypatch, beaten=False, *extra):
    tmp_path.mkdir(parents=True, exist_ok=True)
    man = tmp_path / "starts_store.json"
    man.write_text(json.dumps({"starts": {"s1": {"shape": SHAPE}}}))
    monkeypatch.setattr(rescore, "Ledger", lambda root: type("L", (), {"index": staticmethod(ledger_rows),
                                                                      "root": tmp_path})())
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
    assert {d["curve"] for d in ds} == {"frontier", "honest", "throttled"}
    fr = [d for d in ds if d["curve"] == "frontier"]
    assert all(a["compute_total_gates"] < b["compute_total_gates"] and a["loss"] > b["loss"] for a, b in zip(fr, fr[1:]))
    assert fr[-1]["loss"] == pytest.approx(1.0) and fr[0]["method"].endswith("loss converted")
    on = [d for d in fr if d["method"].endswith("the start s1")]
    assert bool(on) != beaten and {d["on_frontier"] for d in ds if d["start"]} == {not beaten}
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
    q = json.loads((out / "console" / f"{panels.NAME}.json").read_text())
    assert len(console.body(q)) <= console.BODY_MAX and len(q["note"]) <= console.NOTE_MAX
    assert f"x{RATIO:.4f}" in q["note"] and "best unconstrained / constrained learner" in q["note"]


def test_dense_adamw_sets_the_rate_c0_is_counted_at(tmp_path, monkeypatch):
    """No counted training gates in the start's shape: C0 is its tokens at the median gates per token of the AdamW
    checkpoints, in forward passes of the model."""
    _, out = build(tmp_path, monkeypatch)
    p = plots.frame(json.loads((out / f"{panels.NAME}.json").read_text()))
    scored, _ = plots.load(ledger_rows(), ("pin",), {"s1": SHAPE})
    rate, how = panels.dense_rate(scored, {"s1": SHAPE})
    assert how.startswith("counted: the median of 2 ") and rate > 0
    assert p[-1]["start_compute_gates"] == pytest.approx(rate * panels.w_fwd(SHAPE) * SHAPE["tokens"], rel=1e-3)


def curve(kind, n, start="", c=None, k=None, top=1.2, step=0.001):
    return [[kind, start, None, None, None, c, k, float(i), None, top * (1 - step) ** i, None, "r", "m", None, None]
            for i in range(n)]


def test_the_cut_thins_the_off_default_cells_first():
    """The frontier, honest and the default cell keep points `FRONTIER_STEP` apart; every other cell the first point
    to reach each of n + 1 evenly spaced losses, its ends kept."""
    c, k = panels.DEFAULT[0], 8 * panels.DEFAULT[1]
    fine = [curve("frontier", 40), curve("honest", 40, "s"), curve("throttled", 40, "s", c, k)]
    off = curve("throttled", 40, "s", 1, 80_000)
    got = panels.thinned([r for g in fine for r in g] + off, 3)
    for g in fine:
        kept = [r for r in got if r[0] == g[0][0] and r[5] == g[0][5]]
        assert (kept[0], kept[-1]) == (g[0], g[-1]) and [r[7] for r in kept] == [float(i) for i in range(0, 40, 3)]
        assert all(b[9] <= a[9] * (1 - panels.FRONTIER_STEP) for a, b in zip(kept, kept[1:]))
    assert [r[7] for r in got if r[5] == 1] == [0.0, 13.0, 26.0, 39.0]
    assert panels.thinned(off, None) == off
    branch = fine[1][0][:2] + [4.0, fine[0][4][9]] + fine[1][0][4:]
    kept = [r[7] for r in panels.thinned(fine[0] + [branch], 3) if r[0] == "frontier"]
    assert kept == [0.0, 3.0] + [float(i) for i in range(4, 40, 3)] + [39.0]


def test_the_cut_fits_the_route_and_says_how_many_points(monkeypatch):
    p = {"id": panels.PANEL, "note": "", "columns": panels.COLUMNS,
         "rows": curve("frontier", 10) + [r for kb in panels.K_BYTES for r in curve("throttled", 60, "s", 1, 8 * kb)]}
    whole = panels.cut(p, RATIO)
    assert len(whole["rows"]) == len(p["rows"]) and whole["note"] == panels.note(RATIO, None)
    monkeypatch.setattr(console, "BODY_MAX", len(console.body(whole)) // 4)
    q = panels.cut(p, RATIO)
    n = next(n for n in panels.POINTS if n and q["note"].endswith(f": {n + 1} points."))
    assert len(console.body(q)) <= console.BODY_MAX and len(q["note"]) <= console.NOTE_MAX
    assert len([r for r in q["rows"] if r[0] == "throttled" and r[6] == 80_000]) == n + 1
    monkeypatch.setattr(console, "BODY_MAX", 100)
    with pytest.raises(ValueError, match="no cut fits"):
        panels.cut(p, RATIO)


def test_publish_puts_the_console_copy(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(console, "put", lambda p: sent.append(p) or "201")
    code, out = build(tmp_path, monkeypatch, False, "--publish")
    assert code == 0 and len(sent) == 1
    assert console.body(sent[0]) == (out / "console" / f"{panels.NAME}.json").read_bytes()
    monkeypatch.setattr(console, "put", lambda p: "403 forbidden")
    assert build(tmp_path / "again", monkeypatch, False, "--publish")[0] == 1
