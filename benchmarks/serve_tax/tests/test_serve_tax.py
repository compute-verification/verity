"""The driver's metrics, table, tick calibration and server command on synthetic records."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import serve_tax as S


def rec(i: int, send: float, ttft: float, gap: float, n: int, status: int = 200) -> dict:
    chunks = [[round(ttft + k * gap, 6), 1] for k in range(n)]
    return {"id": f"x-{i:05d}", "i": i, "status": status, "send_unix": 1000.0 + send, "first_token_s": ttft,
            "done_s": chunks[-1][0], "output_ids": list(range(n)), "chunks": chunks}


def test_load_metrics_counts_goodput_against_both_slo_bounds():
    recs = [rec(0, 0.0, 0.5, 0.05, 11), rec(1, 1.0, 3.0, 0.05, 11), rec(2, 2.0, 0.5, 0.2, 11)]
    m = S.load_metrics(recs, (2.0, 0.08))
    assert m["ok"] == 3 and m["output_tokens"] == 33
    assert m["span_s"] == pytest.approx(2.0 + 0.5 + 10 * 0.2)
    assert m["goodput_req_per_s"] == pytest.approx(1 / m["span_s"])          # only request 0 meets both
    assert m["slo_attained"] == pytest.approx(1 / 3)
    assert m["ttft_p50_s"] == pytest.approx(0.5) and m["itl_p50_s"] == pytest.approx(0.05)


def test_failed_requests_are_sent_but_not_served():
    bad = rec(1, 0.5, 0.5, 0.05, 3, status=500)
    m = S.load_metrics([rec(0, 0.0, 0.5, 0.05, 3), bad], (2.0, 0.08))
    assert (m["sent"], m["ok"], m["slo_attained"]) == (2, 1, 0.5)


def test_pct_interpolates_between_order_statistics():
    assert S.pct([], 0.5) is None
    assert S.pct([4.0, 1.0, 3.0, 2.0], 0.5) == pytest.approx(2.5)
    assert S.pct(list(range(101)), 0.99) == pytest.approx(99.0)


def arm(name: str, parent: str | None, tputs: list[float]) -> dict:
    return {"arm": name, "parent": parent, "what": name, "run_id": f"r-{name}", "startup_s": 1.0, "error": None,
            "tick": None, "slo": {"ttft_s": 2.0, "tpot_s": 0.08},
            "loads": {"saturated": [{"rep": i + 1, "rate": 10.0, "output_tok_per_s": t, "goodput_req_per_s": t / 100,
                                     "req_per_s": t / 100, "slo_attained": 1.0} for i, t in enumerate(tputs)]}}


def test_tax_is_compute_per_token_over_the_reference():
    t = S.table([arm("compiled", None, [1000, 1000, 1000]), arm("eager", "compiled", [800, 800, 800]),
                 arm("record", "compiled", [990, 1000, 1010])])
    rows = {r["arm"]: r for r in t["rows"]}
    assert "tax_vs_compiled" not in rows["compiled"]
    assert rows["eager"]["tax_vs_compiled"]["throughput"]["value"] == pytest.approx(0.25)
    rec_tax = rows["record"]["tax_vs_compiled"]["throughput"]
    assert rec_tax["value"] == pytest.approx(0.0)
    assert rec_tax["ci95"][0] < 0 < rec_tax["ci95"][1]
    md = S.markdown(t)
    assert "+25.0%" in md and "r-eager" in md


def test_busy_cycle_counts_only_cycles_with_a_request_in_the_engine():
    recs = [{"kind": "header"}, {"kind": "arrive", "request": "a", "t": 0.0}, {"kind": "step", "t": 1.0, "finished": {}},
            {"kind": "step", "t": 1.2, "finished": {"a": {}}}, {"kind": "arrive", "request": "b", "t": 5.0},
            {"kind": "step", "t": 6.0, "finished": {}}, {"kind": "step", "t": 6.4, "finished": {"b": {}}}]
    c = S.busy_cycle(recs)
    assert c["cycles"] == 2 and c["idle"] == 1
    assert c["mean_us"] == pytest.approx(0.3e6)


def test_window_trace_keeps_the_header_and_the_load():
    tr = [{"kind": "header"}, {"kind": "step", "t": 1.0}, {"kind": "step", "t": 5.0}, {"kind": "step", "t": 9.0}]
    assert [r.get("t") for r in S.in_window(tr, 4.0, 6.0)] == [None, 5.0]


def test_plugin_site_loads_the_tick_after_this_plugin(tmp_path: Path):
    ep = (S.plugin_site(tmp_path, tick=True) / "verity_serve_tax-0.0.dist-info" / "entry_points.txt").read_text().splitlines()
    assert ep[0] == "[vllm.general_plugins]"
    assert ep[1].endswith("servetax_plugin:register") and ep[2].endswith("tick_plugin:register")
    ep = (S.plugin_site(tmp_path / "plain", tick=False) / "verity_serve_tax-0.0.dist-info" / "entry_points.txt").read_text()
    assert "tick_plugin" not in ep


def test_arms_differ_only_by_their_change():
    m = S.MODELS["moonlight"]
    argv = {a: S.server_argv(S.ARMS[a], m, Path("/ck"), 8000) for a in S.ARMS}
    assert "--enforce-eager" not in argv["compiled"] and "--enforce-eager" in argv["eager"]
    assert argv["record"] == argv["compiled"] == argv["release-order"]
    assert argv["tick"] == argv["compiled"] + ["--middleware", S.TICK_MIDDLEWARE]
    for name, a in S.ARMS.items():
        assert a.parent is None or a.parent in S.ARMS, name
        if a.parent and name != "eager":
            assert set(S.ARMS[a.parent].plugin) <= set(a.plugin) and S.ARMS[a.parent].tick <= a.tick, name


def test_server_env_names_the_changes(tmp_path: Path):
    env = S.server_env(S.ARMS["tokens"], tmp_path, tmp_path, {"bucket_us": 100000, "sat_us": 9000, "free_us": 11000},
                       600.0, 30.0)
    cfg = json.loads(env[S.PLUGIN_ENV])
    assert cfg["record"] and cfg["tokens"] and not cfg.get("release_order") and cfg["kv_window_s"] == 30.0
    assert json.loads(env["VERITY_TICK"])["sat_us"] == 9000
    assert S.PLUGIN_ENV not in S.server_env(S.ARMS["compiled"], tmp_path, tmp_path, None, 600.0, 600.0)
    with pytest.raises(SystemExit):
        S.server_env(S.ARMS["tick"], tmp_path, tmp_path, None, 600.0, 600.0)


def test_parse_rates_keeps_their_order():
    assert list(S.parse_rates("light=1.5,medium=4,saturated=9")) == ["light", "medium", "saturated"]


def test_tick_arms_refuse_a_tree_without_the_tick(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(S, "TICK_PLUGIN", tmp_path / "tick_plugin.py")
    with pytest.raises(SystemExit, match="the tick arms need the certified tick"):
        S.tick_args(None, "23.2,28.3")
