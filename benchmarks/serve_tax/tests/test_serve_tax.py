"""The driver's metrics, table, tick calibration and server command on synthetic records."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace as NS

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


def arm(name: str, parent: str | None, tputs: list[float], ok: int = 100) -> dict:
    return {"arm": name, "parent": parent, "what": name, "run_id": f"r-{name}", "startup_s": 1.0, "error": None,
            "tick": None, "slo": {"ttft_s": 2.0, "tpot_s": 0.08},
            "loads": {"saturated": [{"rep": i + 1, "rate": 10.0, "sent": 100, "ok": ok, "output_tok_per_s": t,
                                     "goodput_req_per_s": t / 100, "req_per_s": t / 100, "slo_attained": 1.0}
                                    for i, t in enumerate(tputs)]}}


def test_tax_is_compute_per_token_over_the_reference():
    t = S.table([arm("compiled", None, [1000, 1000, 1000]), arm("eager", "compiled", [800, 800, 800], ok=99),
                 arm("record", "compiled", [990, 1000, 1010])])
    rows = {r["arm"]: r for r in t["rows"]}
    assert "tax_vs_compiled" not in rows["compiled"]
    assert rows["eager"]["tax_vs_compiled"]["throughput"]["value"] == pytest.approx(0.25)
    rec_tax = rows["record"]["tax_vs_compiled"]["throughput"]
    assert rec_tax["value"] == pytest.approx(0.0)
    assert rec_tax["ci95"][0] < 0 < rec_tax["ci95"][1]
    assert (rows["eager"]["unserved"], rows["eager"]["sent"], rows["compiled"]["unserved"]) == (3, 300, 0)
    md = S.markdown(t)
    assert "+25.0%" in md and "r-eager" in md and "| 3 of 300 |" in md


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
    assert argv["record"] == argv["compiled"] == argv["release-order"] == argv["bi-compiled"]
    assert argv["bi-eager"] == argv["eager"]
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


def test_the_boundary_rows_arm_holds_the_server_until_its_commits_account_for_every_finished_request(tmp_path: Path):
    """`await_commits` polls the sink lines (a failed class's line counts its spans too) until they hold the finished
    requests' spans, or its wait runs out."""
    sink, now = tmp_path / "boundary.7.jsonl", [0.0]
    sink.write_text('{"window":0,"spans":8}\n')

    def sleep(s: float) -> None:
        now[0] += s
        if now[0] == 3.0:
            with sink.open("a") as fh:
                fh.write('{"window":1,"spans":7}\n{"window":1,"spans":1,"error":"OSError: full"}\n')

    got = S.await_commits(tmp_path, 16, 60.0, clock=lambda: now[0], sleep=sleep)
    assert got == {"spans": 16, "finished": 16, "waited_s": 3.0, "complete": True}
    later = [0.0]
    out = S.await_commits(tmp_path, 17, 5.0, clock=lambda: later[0], sleep=lambda s: later.__setitem__(0, later[0] + s))
    assert out == {"spans": 16, "finished": 17, "waited_s": 5.0, "complete": False}


def test_the_boundary_rows_arm_turns_prefix_caching_off_and_hands_the_plugin_its_object(tmp_path: Path):
    m = S.MODELS["glm53"]
    argv = S.server_argv(S.ARMS["boundary-rows"], m, Path("/ck"), 8000)
    assert argv == S.server_argv(S.ARMS["record"], m, Path("/ck"), 8000) + ["--no-enable-prefix-caching"]
    for missing in (None, {}, {"per_output": 1}):
        with pytest.raises(SystemExit, match="--boundary-rows"):
            S.server_env(S.ARMS["boundary-rows"], tmp_path, tmp_path, None, 30.0, 600.0, boundary_rows=missing)
    rows = {"check": "t", "per_output": 1, "drained": True}
    cfg = json.loads(S.server_env(S.ARMS["boundary-rows"], tmp_path, tmp_path, None, 30.0, 600.0, boundary_rows=rows)[S.PLUGIN_ENV])
    assert cfg["record"] and cfg["boundary_rows"] == {"window_s": 30.0, **rows}
    record = S.server_env(S.ARMS["record"], tmp_path, tmp_path, None, 30.0, 600.0, boundary_rows=rows)
    assert "boundary_rows" not in json.loads(record[S.PLUGIN_ENV])
    sh = (S.HERE / "run.sh").read_text()
    assert '${BOUNDARY_ROWS:+--boundary-rows "$BOUNDARY_ROWS"}' in sh


def test_only_the_restore_arms_take_edges_and_only_restore_dumps_there(tmp_path: Path):
    edges = {"edges_s": [180.0, 360.0], "save_kv": [1], "dump": False}
    cfg = {a: json.loads(S.server_env(S.ARMS[a], tmp_path, tmp_path, None, 600.0, 600.0, edges)[S.PLUGIN_ENV])
           for a in ("restore-base", "restore-pause", "restore")}
    assert "edges_s" not in cfg["restore-base"] and cfg["restore-pause"]["edges_s"] == [180.0, 360.0]
    assert [S.ARMS[a].edge_dump for a in ("restore-pause", "restore")] == [False, True]
    assert [S.ARMS[a].edge_busy for a in ("restore-pause", "restore", "restore-unseeded")] == [False, True, True]
    assert {S.ARMS[a].parent for a in ("restore-pause", "restore")} == {"restore-base"}
    assert S.ARMS["restore-unseeded"].sampling == S.UNSEEDED and S.ARMS["restore-unseeded"].edges
    m = S.MODELS["moonlight"]
    argv = S.server_argv(S.ARMS["restore"], m, Path("/ck"), 8000, 65536)
    assert argv[argv.index("--num-gpu-blocks-override") + 1] == "65536"
    assert argv == S.server_argv(S.ARMS["restore-base"], m, Path("/ck"), 8000, 65536)


def test_unseeded_requests_carry_the_seeded_sampling_without_the_seed(monkeypatch: pytest.MonkeyPatch):
    def body(served: str, r: dict, rid: str, sampling: str) -> dict:
        return {"prompt": r["p"], "temperature": 0.8 if sampling == "seeded" else 0.0} | (
            {"seed": 7} if sampling == "seeded" else {})

    monkeypatch.setattr(S, "BODY_EXTRA", {"logprobs": 0})
    assert S.serve_body(body, "m", {"p": [1]}, "x", "seeded") == {"prompt": [1], "temperature": 0.8, "seed": 7,
                                                                  "ignore_eos": True, "logprobs": 0}
    assert S.serve_body(body, "m", {"p": [1]}, "x", S.UNSEEDED) == {"prompt": [1], "temperature": 0.8, "ignore_eos": True,
                                                                    "logprobs": 0}
    assert S.serve_body(body, "m", {"p": [1]}, "x", "greedy")["temperature"] == 0.0


def test_every_engine_runs_with_inductors_rblock_pinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    lines = (S.HERE / "run.sh").read_text().splitlines()
    # unindented and before the leased branch: both the served arms and the replays inherit it
    assert lines.index("export TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0") < lines.index('if [ "${1:-}" = leased ]; then')
    monkeypatch.setenv("TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK", "0")
    for arm in ("compiled", "record"):
        env = S.server_env(S.ARMS[arm], tmp_path, tmp_path, None, 600.0, 600.0)
        assert env["TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK"] == "0", arm


def test_only_the_batch_invariant_arms_run_vllms_batch_invariant_kernels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")   # a job's environment doesn't turn it on for the stock arms
    on = {a for a in S.ARMS if S.server_env(S.ARMS[a], tmp_path, tmp_path, {"bucket_us": 1, "sat_us": 1, "free_us": 1},
                                            600.0, 600.0, boundary_rows={"check": "t"})["VLLM_BATCH_INVARIANT"] == "1"}
    assert on == {"bi-compiled", "bi-eager"}
    assert (S.ARMS["bi-compiled"].parent, S.ARMS["bi-eager"].parent) == ("compiled", "eager")


def test_run_shs_replay_runs_at_the_models_width_and_captures_what_it_is_told():
    sh = (S.HERE / "run.sh").read_text()
    call = sh[sh.index('"$PY" "$here/replay.py"'):]
    call = call[:call.index("rc=1; }")]
    assert '--tp "$GPUS"' in call and '${REPLAY_CAPTURE:+--capture "$REPLAY_CAPTURE"}' in call


def test_run_sh_serves_every_model():
    cases = [re.match(r"  ([a-z0-9-]+(?:\|[a-z0-9-]+)*)\)", line) for line in (S.HERE / "run.sh").read_text().splitlines()]
    assert {name for m in cases if m for name in m[1].split("|")} == set(S.MODELS)


def test_checkpoint_is_the_pinned_snapshot_under_hf_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert S.main(["checkpoint", "--model-id", "qwen3-30b"]) == 0
    assert capsys.readouterr().out.strip() == str(
        tmp_path / "hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39")


METRICS = """\
# HELP vllm:iteration_tokens_total Histogram of number of tokens per engine_step.
# TYPE vllm:iteration_tokens_total histogram
vllm:iteration_tokens_total_bucket{engine="0",le="1.0",model_name="m"} 3.0
vllm:iteration_tokens_total_count{engine="0",model_name="m"} 40.0
vllm:iteration_tokens_total_sum{engine="0",model_name="m"} 5000.0
vllm:iteration_tokens_total_count{engine="1",model_name="m"} 10.0
vllm:iteration_tokens_total_sum{engine="1",model_name="m"} 1000.0
vllm:iteration_tokens_total_created{engine="0",model_name="m"} 1.79e+09
vllm:generation_tokens_total{engine="0",model_name="m"} 1200.0
vllm:prompt_tokens_total{engine="0",model_name="m"} 4800.0
vllm:prompt_tokens_cached_total{engine="0",model_name="m"} 1200.0
vllm:num_preemptions_total{engine="0",model_name="m"} 0.0
vllm:num_requests_running{engine="0",model_name="m"} 7.0
"""


def test_counters_sum_each_over_its_label_sets_and_the_schedule_reads_them():
    c = S.counters(METRICS)
    assert c == {"steps": 50.0, "step_tokens": 6000.0, "generation_tokens": 1200.0, "prompt_tokens": 4800.0,
                 "prompt_tokens_cached": 1200.0, "preemptions": 0.0}
    s = S.schedule_metrics(c)
    assert (s["steps"], s["tokens_per_step"], s["rows_per_step"], s["prompt_cached_share"]) == (50.0, 120.0, 24.0, 0.25)
    assert S.schedule_metrics(None) == {} == S.schedule_metrics(dict(c, steps=0.0))


def test_the_table_carries_each_loads_schedule():
    a = arm("compiled", None, [1000, 1000])
    for r, steps in zip(a["loads"]["saturated"], (500.0, 520.0)):
        r |= {"steps": steps, "tokens_per_step": 6e5 / steps, "rows_per_step": 30.0}
    row = S.table([a])["rows"][0]
    assert row["steps"]["mean"] == pytest.approx(510.0) and row["rows_per_step"]["mean"] == pytest.approx(30.0)
    assert "| 510 | 1177 / 30 |" in S.markdown(S.table([a]))


def test_parse_rates_keeps_their_order():
    assert list(S.parse_rates("light=1.5,medium=4,saturated=9")) == ["light", "medium", "saturated"]


def test_tick_arms_refuse_a_tree_without_the_tick(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(S, "TICK_PLUGIN", tmp_path / "tick_plugin.py")
    with pytest.raises(SystemExit, match="the tick arms need the certified tick"):
        S.tick_args(None, "23.2,28.3")


def test_drive_load_counts_what_the_client_sent_and_the_server_did_not_serve(monkeypatch: pytest.MonkeyPatch,
                                                                             tmp_path: Path):
    class Corpus:
        class traffic:
            @staticmethod
            def arrivals(plan, rate):
                return [r["i"] / rate for r in plan]

        @staticmethod
        async def drive(port, served, reqs, arr, ids, out, drain_s, sampling):
            recs = [rec(r["i"], a, 0.1, 0.01, 2) for r, a in zip(reqs, arr)]
            recs[1] |= {"status": None, "error": "ClientConnectorError: [Too many open files]", "done_s": None}
            out.write_text("".join(json.dumps(x) + "\n" for x in recs))
            return {"sent": len(recs), "unfinished": 0}

    monkeypatch.setattr(S, "corpus", lambda: Corpus)
    m = S.drive_load(0, "m", [{"i": i} for i in range(4)], 10.0, 1.0, "light1", tmp_path / "rep-1", 1.0, None)
    assert (m["requests"], m["unserved"]) == (4, 1)


def test_step_traces_are_read_plain_or_gzipped(tmp_path: Path):
    import gzip

    (tmp_path / "server").mkdir()
    lines = "".join(json.dumps(r) + "\n" for r in ({"kind": "header"}, {"kind": "step", "t": 1.0}))
    (tmp_path / "server" / "step_trace.1.jsonl").write_text(lines)
    with gzip.open(tmp_path / "server" / "step_trace.2.jsonl.gz", "wt") as f:
        f.write(lines)
    plain, gz = S.step_traces(tmp_path)
    assert S.trace_records(plain) == S.trace_records(gz) == [{"kind": "header"}, {"kind": "step", "t": 1.0}]


def test_edge_cost_reads_the_stall_across_an_edge_and_the_ttft_around_it(tmp_path: Path):
    def rec(send: float, chunks: list[float]) -> dict:
        return {"status": 200, "send_s": send, "send_unix": 100 + send, "first_token_s": chunks[0], "done_s": chunks[-1],
                "chunks": [[c, 1] for c in chunks], "output_ids": [0] * len(chunks)}

    recs = [rec(0.0, [0.1, 0.2, 3.5, 3.6]), rec(1.0, [0.1, 0.2]), rec(2.5, [1.2, 1.3]), rec(9.0, [0.1, 0.2])]
    (e,) = S.edge_cost(recs, [(2.0, 1.0)], before_s=1.0)
    assert (e["in_flight"], e["gap_max_s"], e["sent"], e["ttft_max_s"]) == (1, pytest.approx(3.3), 2, 1.2)
    (e,) = S.edge_cost(recs, [(5.0, 0.5)])
    assert (e["in_flight"], e["gap_max_s"], e["sent"]) == (0, None, 3)
    d = tmp_path / "restore"
    (d / "medium" / "rep-1").mkdir(parents=True)
    (d / "server").mkdir()
    (d / "medium" / "rep-1" / "meta.json").write_text(json.dumps({"t0_unix": 100.0}))
    (d / "server" / "edges.7.jsonl").write_text("\n".join(json.dumps(r) for r in (
        {"hash_build_s": 9.0}, {"edge": 1, "next_step": 9, "drained_unix": 130.0, "pause_s": 1.5},
        {"edge": 0, "next_step": 4, "drained_unix": 110.0, "pause_s": 1.0}, {"edge": 0, "state_write_s": 0.2})))
    assert S.edge_windows(d, "medium") == [(10.0, 1.0), (30.0, 1.5)]


def test_a_plan_request_with_its_own_sampling_is_sent_with_it_and_stops_at_its_end_of_text(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(S, "BODY_EXTRA", {"logprobs": 0})
    monkeypatch.setattr(S, "corpus", lambda: NS(request_seed=len))
    r = {"prompt_ids": [1, 2], "max_tokens": 9, "sampling": {"temperature": 0.7, "top_p": 0.9}, "seeded": True}
    assert S.serve_body(None, "m", r, "light1-00003", S.UNSEEDED) == {
        "model": "m", "prompt": [1, 2], "max_tokens": 9, "temperature": 0.7, "top_p": 0.9, "stream": True,
        "stream_options": {"include_usage": True}, "return_token_ids": True, "user": "light1-00003", "seed": 12,
        "logprobs": 0}
    assert "seed" not in S.serve_body(None, "m", dict(r, seeded=False), "x", "seeded")


def test_max_model_len_replaces_the_base_context_limit():
    argv = S.server_argv(S.ARMS["compiled"], S.MODELS["moonlight"], Path("/ck"), 8000, max_model_len=40960)
    assert argv.count("--max-model-len") == 1 and argv[argv.index("--max-model-len") + 1] == "40960"
    assert S.base_args() == S.BASE_ARGS and len(S.base_args(40960)) == len(S.BASE_ARGS)


def test_the_boundary_arm_takes_edges_every_period_or_at_set_times_and_keeps_no_kv_file_at_tp():
    def ns(**kw) -> argparse.Namespace:
        return argparse.Namespace(**({"arm": "boundary", "model_id": "glm53", "edges_s": None, "edges_every_s": None,
                                      "save_kv": None, "kv_dir": None, "kv_copy": "first"} | kw))

    glm, moon, rates = S.MODELS["glm53"], S.MODELS["moonlight"], {"light": 1.0, "medium": 2.0}
    assert (S.tp_size(glm), S.tp_size(moon)) == (8, 1)
    assert S.edge_config(S.ARMS["tokens-only"], glm, ns(edges_every_s=30.0), rates) is None
    for bad in (ns(), ns(edges_every_s=30.0, edges_s="180")):
        with pytest.raises(SystemExit, match="one of them"):
            S.edge_config(S.ARMS["boundary"], glm, bad, rates)
    assert S.edge_config(S.ARMS["boundary"], glm, ns(edges_every_s=30.0), rates) == {
        "every_s": 30.0, "save_kv": [], "kv_dir": None, "kv_copy": "first", "dump": False, "busy": False}
    with pytest.raises(SystemExit, match="TP 8"):
        S.edge_config(S.ARMS["boundary"], glm, ns(edges_every_s=30.0, save_kv="1"), rates)
    c = S.edge_config(S.ARMS["restore"], moon, ns(arm="restore", model_id="moonlight", edges_s="180,360", kv_dir="/kv"),
                      rates)
    assert (c["edges_s"], c["edge_tag"], c["save_kv"], c["kv_dir"]) == ([180.0, 360.0], "-light1-", [0], "/kv/restore")
    assert S.ARMS["boundary"].parent == "tokens-only" and S.ARMS["tokens-only"].parent == "record"
    assert not S.ARMS["tokens-only"].tick and not S.ARMS["tokens-only"].edges


def test_kv_copy_choices_are_the_plugins():
    import boundary as B

    assert S.KV_COPY == B.KV_COPY


class Nvml:
    def __init__(self):
        self.util = {0: 0, 1: 0, "GPU-x": 0}

    def nvmlDeviceGetCount(self):
        return 2

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetHandleByUUID(self, u):
        return u

    def nvmlDeviceGetUtilizationRates(self, h):
        return NS(gpu=self.util[h])


def test_busy_gpu_seconds_integrate_each_visible_gpus_utilization_between_reads():
    lib, now = Nvml(), [0.0]
    b = S.GpuBusy(lib, "1,GPU-x", clock=lambda: now[0])
    assert b.handles == [1, "GPU-x"] and S.GpuBusy(lib).handles == [0, 1]
    b.sample()
    lib.util[1], lib.util["GPU-x"], now[0] = 50, 100, 2.0
    b.sample()
    lib.util[1], now[0] = 100, 3.0
    b.sample()
    assert b.result() == {"gpus": 2, "busy_gpu_s": 5.0, "busy_s_per_gpu": [2.0, 3.0], "reads": 3, "period_s": 0.2}
    with S.GpuBusy(None) as none:
        assert none.thread is None
    assert none.result() is None


def test_reps_that_fit_count_the_startup_the_warm_up_and_the_longest_rep():
    meta = {"startup_s": 500.0, "loads": [{"rep": 1, "wall_s": 100.0}, {"rep": 1, "wall_s": 200.0},
                                          {"rep": 2, "wall_s": 310.0}]}
    # fixed (500 + 50 + 60) * 1.1 = 671 s; a rep max(310, 315) * 1.1 = 346.5 s
    assert [S.reps_that_fit(meta, 50.0, b, 3) for b in (2000.0, 1300.0, 600.0)] == [3, 1, 0]
    assert S.reps_that_fit({"loads": []}, 50.0, 10.0, 3) == 3 == S.reps_that_fit(dict(meta, startup_s=None), 0, 1, 3)


def test_fit_prints_the_reps_left_for_the_next_arm(tmp_path: Path, capsys: pytest.CaptureFixture):
    (tmp_path / "warm").mkdir()
    (tmp_path / "meta.json").write_text(json.dumps({"startup_s": 500.0, "loads": [{"rep": 1, "wall_s": 310.0}]}))
    (tmp_path / "warm" / "meta.json").write_text(json.dumps({"wall_s": 50.0}))
    assert S.main(["fit", str(tmp_path), "--budget-s", "1300", "--reps", "3"]) == 0
    assert capsys.readouterr().out.strip() == "1"


def edge(k: int, load: str, drained: float, pause: float, roots_equal: bool = True) -> dict:
    return {"edge": k, "load": load, "next_step": 10 * k, "drained_unix": drained, "drain_s": 0.1, "pause_s": pause,
            "state_s": 0.01, "kv_s": pause - 0.05, "kv": {"root": "aa", "roots_equal": roots_equal, "bytes": 3e10,
                                                         "copied_bytes": 4e9, "hash_kernel_ms": 80.0, "copy_ms": 300.0,
                                                         "runner_s": 0.02}}


def test_an_arms_metrics_carry_its_busy_gpu_seconds_and_its_edges_outside_the_warm_up(tmp_path: Path):
    d = tmp_path / "boundary"
    (d / "light" / "rep-1").mkdir(parents=True)
    (d / "server").mkdir()
    (d / "meta.json").write_text(json.dumps({"arm": "boundary", "parent": "tokens-only", "model_id": "moonlight",
                                             "rates": {"light": 1.0}, "run_id": "r-b"}))
    (d / "light" / "rep-1" / "client.jsonl").write_text("".join(json.dumps(rec(i, i, 0.2, 0.05, 50)) + "\n"
                                                                for i in range(4)))
    (d / "light" / "rep-1" / "meta.json").write_text(json.dumps(
        {"rate": 1.0, "t0_unix": 1000.0, "wall_s": 10.0, "gpu": {"busy_gpu_s": 4.0}}))
    (d / "server" / "edges.1.jsonl").write_text("\n".join(json.dumps(r) for r in (
        {"hash_build_s": 2.0}, edge(0, "warm", 990.0, 5.0), edge(1, "light", 1003.0, 0.5),
        edge(2, "light", 1006.0, 0.7, roots_equal=False))))
    m = S.arm_metrics(d)
    (load,) = m["loads"]["light"]
    assert (load["busy_gpu_s"], load["busy_gpu_s_per_ktok"]) == (4.0, pytest.approx(20.0))
    assert (load["edges"], load["edge_stall_max_s"], load["edge_stall_s"]) == (2, pytest.approx(0.8), pytest.approx(1.4))
    e = m["edges"]
    assert (e["edges"], e["pause_max_s"], e["hash_kernel_ms_p50"], e["copy_ms_p50"], e["roots_equal"]) == (
        2, 0.7, 80.0, 300.0, False)
    assert e["pause_p50_s"] == pytest.approx(0.6) and e["stall_p50_s"] == pytest.approx(0.7)
    md = S.markdown(S.table([m]))
    assert "| 4 (20) | 2 (0.7 / 0.8) |" in md
    assert "| boundary | 2 | 0.6 / 0.698 / 0.7 |" in md and "| 30 / 4 | NO |" in md


def test_a_given_plan_is_served_as_it_is_and_a_short_one_refused(tmp_path: Path):
    f = tmp_path / "plan.jsonl"
    f.write_text("".join(json.dumps({"i": i, "sampling": {"temperature": 0.7}}) + "\n" for i in range(5)))
    assert [r["i"] for r in S.plan_of(f, tmp_path / "no-c4", tmp_path, 3, 10, 0)] == [0, 1, 2]
    with pytest.raises(SystemExit, match="holds 5 requests and these loads may send 6"):
        S.plan_of(f, tmp_path / "no-c4", tmp_path, 6, 10, 0)
    assert len(f.read_text().splitlines()) == 5


def test_nvml_says_why_a_run_has_no_busy_gpu_seconds(monkeypatch: pytest.MonkeyPatch):
    import importlib.machinery
    import importlib.util
    import sys
    import types

    fake = types.ModuleType("pynvml")
    fake.__spec__ = importlib.machinery.ModuleSpec("pynvml", None)

    def no_driver():
        raise RuntimeError("Driver Not Loaded")

    fake.nvmlInit = no_driver
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda m, *a: None if m == "vllm" else real(m, *a))
    assert S.nvml() == (None, "pynvml.nvmlInit: RuntimeError('Driver Not Loaded')")
    fake.nvmlInit = lambda: None
    assert S.nvml() == (fake, None)
    monkeypatch.setattr(importlib.util, "find_spec", lambda m, *a: None)
    assert S.nvml() == (None, "no NVML bindings (vllm.third_party.pynvml, pynvml)")


def test_a_tax_against_one_rep_has_no_interval():
    one = S.tax(S.summary([1000.0, 1010.0, 990.0]), S.summary([900.0]))
    assert one["value"] == pytest.approx(1000 / 900 - 1) and one["n"] == [3, 1] and "ci95" not in one
    assert "ci95" in S.tax(S.summary([1000.0, 1010.0, 990.0]), S.summary([900.0, 910.0]))
    assert S._tax(one) == "+11.1%"


class Server:
    """A server that ignores SIGINT: `stop` waits its grace, kills it, then kills its group."""

    pid = 4242

    def __init__(self):
        self.calls = []

    def poll(self):
        return None

    def send_signal(self, s):
        self.calls.append(("signal", s))

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        if timeout is not None:
            raise S.subprocess.TimeoutExpired("vllm", timeout)

    def kill(self):
        self.calls.append(("kill",))


def test_a_terminated_arm_kills_its_servers_group_within_the_runners_grace(monkeypatch):
    groups = []
    monkeypatch.setattr(S.os, "killpg", lambda pid, s: groups.append((pid, s)))
    server = Server()
    assert S.stop(server, S.TERM_GRACE_S) == "killed"
    assert server.calls == [("signal", S.signal.SIGINT), ("wait", S.TERM_GRACE_S), ("kill",), ("wait", None)]
    assert groups == [(4242, S.signal.SIGKILL)] and S.TERM_GRACE_S < 20
