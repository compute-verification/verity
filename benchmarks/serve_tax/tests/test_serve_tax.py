"""The driver's metrics, table, tick calibration and server command on synthetic records."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
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


def site_entries(d: Path, arm: str) -> list[str]:
    return (S.plugin_site(d / arm, S.ARMS[arm]) / "verity_serve_tax-0.0.dist-info" / "entry_points.txt").read_text().splitlines()


def test_plugin_site_declares_the_arms_plugins_the_tick_after_this_one(tmp_path: Path):
    ep = site_entries(tmp_path, "tick")
    assert ep[0] == "[vllm.general_plugins]"
    assert ep[1].endswith("servetax_plugin:register") and ep[2].endswith("tick_plugin:register")
    assert site_entries(tmp_path, "record") == ep[:2] and site_entries(tmp_path, "compiled") == ep[:1]
    # the canonical top-k's own arm loads its plugin alone; the two-stage arm loads it after this module's
    topk = "verity_dsa_topk = dsa_topk:register"
    assert site_entries(tmp_path, "canonical-topk") == [ep[0], topk]
    assert site_entries(tmp_path, "two-stage") == [*ep[:2], topk]
    # the launch line under the tick loads all three: this module's, the tick's after it, then the top-k's
    assert site_entries(tmp_path, "two-stage-tick") == [*ep[:3], topk]


def test_the_two_stage_arm_is_the_serves_launch_line_and_nothing_else(tmp_path: Path):
    """The serve's line as data: prefix caching off, which its history tables need, the step record and the boundary
    commitment with the model's check template and one attention layer's tables, and the canonical top-k on the path
    and switched on."""
    glm, two, rows = S.MODELS["glm53"], S.ARMS["two-stage"], S.ARMS["boundary-rows"]
    assert S.server_argv(two, glm, Path("/ck"), 8000) == [*S.server_argv(rows, glm, Path("/ck"), 8000),
                                                          "--no-enable-prefix-caching"]
    assert (two.parent, two.plugin, S.ARMS["canonical-topk"].parent) == ("boundary-rows", rows.plugin, "compiled")
    private = tmp_path / "private"
    env = S.server_env(two, tmp_path, tmp_path / "site", None, 3600.0, 600.0, check=glm.check, private=private)
    cfg = json.loads(env[S.PLUGIN_ENV])
    assert cfg["boundary_rows"] == {
        "window_s": 3600.0, "check": "TokenSelect_v4{V=154880}", "inline": True, "per_prompt": 0, "per_output": 1,
        "drained": True, "tables": {"layers": 1, "private": str(private)}}
    assert (env["VERITY_DSA_TOPK"], env["VERITY_DSA_TOPK_LOG"]) == ("canonical", f"{private}/topk-log")
    assert str(S.ROOT / "benchmarks/dsa_topk") in env["PYTHONPATH"].split(S.os.pathsep)
    # a boundary commitment's classes hold the requests' tokens and the salts: the plugin is told the private
    # directory, and the run's directory, which is published, gets only what the plugin writes to ``out``
    assert (cfg["out"], cfg["private"]) == (str(tmp_path), str(private))
    # the arms it is made of carry the same pieces: the boundary commitment without the tables, the top-k alone
    cfg = json.loads(S.server_env(rows, tmp_path, tmp_path, None, 3600.0, 600.0, check=glm.check, private=private)[S.PLUGIN_ENV])
    assert cfg["boundary_rows"] == {"window_s": 3600.0, "check": glm.check, **S.ROWS} and "tables" not in S.ROWS
    assert cfg["private"] == str(private)
    alone = S.server_env(S.ARMS["canonical-topk"], tmp_path, tmp_path / "site", None, 600.0, 600.0, private=private)
    assert S.PLUGIN_ENV not in alone and alone["VERITY_DSA_TOPK"] == "canonical"
    assert [S.keeps_private(S.ARMS[a]) for a in ("compiled", "record", "boundary-rows", "canonical-topk", "two-stage")] == [
        False, False, True, True, True]
    assert "private" not in json.loads(S.server_env(S.ARMS["record"], tmp_path, tmp_path, None, 1.0, 1.0)[S.PLUGIN_ENV])
    for keeps in (two, rows):                 # no place on the node to keep them: nothing starts
        with pytest.raises(SystemExit, match="--private DIR"):
            S.server_env(keeps, tmp_path, tmp_path, None, 600.0, 600.0, check=glm.check)
    # the arm's data names its modules by path: each is a file of this tree
    for a in S.ARMS.values():
        for entry, path in a.general:
            assert (S.ROOT / path / f"{entry.split('=')[1].split(':')[0].strip()}.py").is_file(), entry


def test_what_an_arm_keeps_on_the_node_is_outside_every_runs_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seen = []
    monkeypatch.setattr(S, "serve", lambda a, private: seen.append((private, private is None or private.is_dir())) or 0)
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path / "run"))
    out = tmp_path / "run" / "out" / "two-stage"
    for inside in (out / "private", tmp_path / "run" / "private"):
        with pytest.raises(SystemExit, match="which leaves the node"):
            S.run(NS(out=str(out), private=str(inside), arm="two-stage"))
    assert not seen and not (tmp_path / "run").exists()
    kept = tmp_path / "kept"
    assert S.run(NS(out=str(out), private=str(kept), arm="two-stage", client=S.CORPUS)) == 0
    assert seen == [(kept, True)] and kept.stat().st_mode & 0o777 == 0o700
    # none given, under vLLM's client (a cost's run, whose kept files nothing reads afterwards): a fresh directory for an
    # arm that keeps something, gone when the arm stops; none for the others
    bench = {"out": str(out), "private": None, "client": S.BENCH}
    assert S.run(NS(arm="two-stage", **bench)) == 0 == S.run(NS(arm="compiled", **bench))
    (fresh, was_there), stock = seen[1], seen[2]
    assert was_there and not fresh.exists() and not fresh.is_relative_to(tmp_path / "run") and stock == (None, True)
    # none given, under the corpus client (a serve, whose later stages read them): refused, so no serve finds its
    # classes gone with a directory it never named; an arm that keeps nothing serves as before
    for keeps in ("boundary-rows", "two-stage", "canonical-topk"):
        with pytest.raises(SystemExit, match=r"a serve's later stages read it: --private DIR.*PRIVATE"):
            S.run(NS(out=str(out), private=None, arm=keeps, client=S.CORPUS))
    assert len(seen) == 3 and S.run(NS(out=str(out), private=None, arm="record", client=S.CORPUS)) == 0


def test_a_jobs_environment_cannot_drop_an_arms_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_PLUGINS", "something_else")
    env = S.server_env(S.ARMS["record"], tmp_path, tmp_path, None, 600.0, 600.0)
    assert "VLLM_PLUGINS" not in env and "VLLM_PLUGINS" not in S.server_env(S.ARMS["compiled"], tmp_path, tmp_path, None, 1.0, 1.0)


def test_no_server_writes_a_requests_or_a_responses_text_to_its_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in S.LOGS_TEXT_ENV:
        monkeypatch.setenv(name, "1")
    for arm in ("compiled", "record", "two-stage"):
        env = S.server_env(S.ARMS[arm], tmp_path, tmp_path, None, 600.0, 600.0, check=S.MODELS["glm53"].check,
                           private=tmp_path / "private")
        assert not set(S.LOGS_TEXT_ENV) & set(env)
    for model in S.MODELS.values():
        for arm in S.ARMS.values():
            assert not set(S.LOGS_TEXT_ARGS) & set(S.server_argv(arm, model, tmp_path, 8000))


def test_an_arms_plugins_are_loaded_once_before_its_server_starts(tmp_path: Path):
    """vLLM serves on without a plugin whose import fails; `check_plugins` refuses before the server starts."""
    site = tmp_path / "site"
    (site / "mod").mkdir(parents=True)
    (site / "mod" / "good_plugin.py").write_text("def register():\n    raise SystemExit('loaded, never called')\n")
    good = S.replace(S.ARMS["compiled"], general=(("good = good_plugin:register", str(site / "mod")),))
    bad = S.replace(good, general=(*good.general, ("absent = absent_plugin:register", str(site / "mod"))))
    for arm, why in ((good, None), (bad, "absent_plugin")):
        env = S.server_env(arm, tmp_path, S.plugin_site(site, arm), None, 600.0, 600.0)
        if why is None:
            S.check_plugins(env, arm)
        else:
            with pytest.raises(SystemExit, match=f"doesn't load in the server's environment.*{why}"):
                S.check_plugins(env, arm)
    S.check_plugins({}, S.ARMS["compiled"])        # a stock arm loads nothing, so nothing is run


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
    assert got == {"spans": 16, "finished": 16, "waited_s": 3.0, "complete": True, "failed": 1, "tables": 0}
    later = [0.0]
    out = S.await_commits(tmp_path, 17, 5.0, clock=lambda: later[0], sleep=lambda s: later.__setitem__(0, later[0] + s))
    assert out == {"spans": 16, "finished": 17, "waited_s": 5.0, "complete": False, "failed": 1, "tables": 0}
    # a class committed with its history tables says so on its line: the count is of what was committed, not asked
    with sink.open("a") as fh:
        fh.write('{"window":2,"spans":3,"tables":{"layers":[0],"capacity":131072}}\n'
                 '{"window":2,"spans":2,"tables":{"layers":[0]},"error":"ValueError: rank 0 built no history tables"}\n')
    assert S.await_commits(tmp_path, 21, 5.0, clock=lambda: 0.0)["tables"] == 3


def test_the_two_stage_tick_arm_is_the_launch_line_with_the_tick_on(tmp_path: Path):
    """The launch line under the certified tick is the two-stage arm's data with the tick: its flags with the tick's
    middleware, its boundary commitment and top-k, and τ in the server's environment. Its engine has the launch line's
    changes and the tick, so of the target's parts only the edge snapshot's is off; and vLLM's own client, which sends
    no certified feed, can't drive it."""
    glm, two, tick = S.MODELS["glm53"], S.ARMS["two-stage"], S.ARMS["two-stage-tick"]
    assert replace(tick, parent=two.parent, what=two.what, tick=False) == two and tick.parent == "two-stage"
    assert S.server_argv(tick, glm, Path("/ck"), 8000) == [*S.server_argv(two, glm, Path("/ck"), 8000), "--middleware",
                                                          S.TICK_MIDDLEWARE]
    tau, private = {"bucket_us": 100000, "sat_us": 9000, "free_us": 11000}, tmp_path / "private"
    env = S.server_env(tick, tmp_path, tmp_path / "site", tau, 600.0, 600.0, check=glm.check, private=private)
    plain = S.server_env(two, tmp_path, tmp_path / "site", None, 600.0, 600.0, check=glm.check, private=private)
    assert {k: v for k, v in env.items() if k != "VERITY_TICK"} == plain
    assert {k: json.loads(env["VERITY_TICK"])[k] for k in tau} == tau
    with pytest.raises(SystemExit, match="the tick arms need τ"):
        S.server_env(tick, tmp_path, tmp_path / "site", None, 600.0, 600.0, check=glm.check, private=private)
    assert S.changes_of("two-stage-tick") == [*S.changes_of("two-stage"), "admission on the certified tick"]
    assert [(x["name"], x["off"]) for x in S.target_parts(glm, "two-stage-tick")] == [
        ("the certified tick", []), ("the tie order", []), ("the snapshot bundle", ["the edge snapshot"])]
    with pytest.raises(SystemExit, match="two-stage-tick needs the corpus client"):
        S.main(["passes", "--loads", "chat", "--arms", "two-stage-tick", "--model-id", "glm53"])


def test_the_boundary_rows_arm_keeps_the_prefix_cache_on_and_hands_the_plugin_its_object(tmp_path: Path):
    m = S.MODELS["glm53"]
    argv = S.server_argv(S.ARMS["boundary-rows"], m, Path("/ck"), 8000)
    assert argv == S.server_argv(S.ARMS["record"], m, Path("/ck"), 8000)
    assert S.server_argv(S.ARMS["no-prefix-cache"], m, Path("/ck"), 8000) == argv + ["--no-enable-prefix-caching"]
    # the object is the arm's own with the model's check template; a model that names none needs one given
    kept = tmp_path / "kept"
    for missing in (None, {}, {"per_output": 1}):
        with pytest.raises(SystemExit, match="--boundary-rows"):
            S.server_env(S.ARMS["boundary-rows"], tmp_path, tmp_path, None, 30.0, 600.0, boundary_rows=missing, private=kept)
    # an object given stands in place of the arm's, whole: its keys alone, with the model's check where it names none
    rows = {"check": "t", "per_output": 1, "drained": True}
    cfg = json.loads(S.server_env(S.ARMS["boundary-rows"], tmp_path, tmp_path, None, 30.0, 600.0, boundary_rows=rows,
                                  check="the model's", private=kept)[S.PLUGIN_ENV])
    assert cfg["record"] and cfg["boundary_rows"] == {"window_s": 30.0, **rows}
    given = S.server_env(S.ARMS["two-stage"], tmp_path, tmp_path, None, 30.0, 600.0, check="the model's", private=kept,
                         boundary_rows={"per_output": 2, "tables": {"layers": [3], "private": S.PRIVATE}})
    assert json.loads(given[S.PLUGIN_ENV])["boundary_rows"] == {
        "window_s": 30.0, "check": "the model's", "per_output": 2, "tables": {"layers": [3], "private": str(kept)}}
    record = S.server_env(S.ARMS["record"], tmp_path, tmp_path, None, 30.0, 600.0, boundary_rows=rows)
    assert "boundary_rows" not in json.loads(record[S.PLUGIN_ENV])
    # each arm of a job keeps its own under the job's private directory
    sh = (S.HERE / "run.sh").read_text()
    assert '${BOUNDARY_ROWS:+--boundary-rows "$BOUNDARY_ROWS"} ${PRIVATE:+--private "$PRIVATE/$dir"}' in sh


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
                                            600.0, 600.0, boundary_rows={"check": "t"}, private=tmp_path)["VLLM_BATCH_INVARIANT"] == "1"}
    assert on == {"bi-compiled", "bi-eager"}
    assert (S.ARMS["bi-compiled"].parent, S.ARMS["bi-eager"].parent) == ("compiled", "eager")


def test_run_shs_replay_runs_at_the_models_width_and_captures_what_it_is_told():
    sh = (S.HERE / "run.sh").read_text()
    call = sh[sh.index('"$PY" "$here/replay.py"'):]
    call = call[:call.index("rc=1; }")]
    assert '--tp "$GPUS"' in call and '${REPLAY_CAPTURE:+--capture "$REPLAY_CAPTURE"}' in call
    assert '${PRIVATE:+--private "$PRIVATE/$dir"}' in call          # where the served arm's other plugins keep their files


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
                 "prompt_tokens_cached": 1200.0, "preemptions": 0.0, "drafts": 0.0, "draft_tokens": 0.0,
                 "accepted_tokens": 0.0}
    s = S.schedule_metrics(c)
    assert (s["steps"], s["tokens_per_step"], s["rows_per_step"], s["prompt_cached_share"]) == (50.0, 120.0, 24.0, 0.25)
    assert "accepted_per_draft" not in s and "draft_accepted_share" not in s
    assert S.schedule_metrics(None) == {} == S.schedule_metrics(dict(c, steps=0.0))
    # an engine that drafts: its drafts, the tokens they proposed and the accepted ones, and not the per-position counter
    drafting = S.counters(METRICS + 'vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 400.0\n'
                          'vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} 2000.0\n'
                          'vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 700.0\n'
                          'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="0"} 300.0\n')
    assert (drafting["drafts"], drafting["draft_tokens"], drafting["accepted_tokens"]) == (400.0, 2000.0, 700.0)
    assert S.schedule_metrics(drafting)["accepted_per_draft"] == 1.75
    assert S.schedule_metrics(drafting)["draft_accepted_share"] == 0.35


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


def test_the_published_arm_starts_the_server_as_the_models_recipe_does(tmp_path, monkeypatch):
    """The arm's line is stock's with the recipe's flags added and the runner's limit on the requests at once dropped,
    and nothing else, at the job's context limit, and at the recipe's, not the runner's, where the job names none; a
    model with no recipe here is refused. Behind the Rust front end the same flags go to ``vllm serve`` with the
    recipe's own tokenizer mode, under the variable that starts it, and no other arm's server has it."""
    glm = S.MODELS["glm53"]
    context = S.published_context("published", "glm53", None)
    assert context == glm.published_context == 262144 and S.AUTO == "auto"
    assert [S.published_context("published", "glm53", n) for n in (131072, 492672, "auto")] == [131072, 492672, "auto"]
    stock = S.server_argv(S.ARMS["compiled"], glm, Path("/ck"), 8000, max_model_len=context)
    argv = S.server_argv(S.ARMS["published"], glm, Path("/ck"), 8000, max_model_len=context)
    seqs = stock.index("--max-num-seqs")
    at = stock.index("--max-model-len")
    assert argv == [*stock[:at], *glm.published, "--tokenizer-mode", "auto", *stock[at:seqs], *stock[seqs + 2:]]
    assert argv.count("--max-model-len") == 1 and argv[argv.index("--max-model-len") + 1] == "262144"
    # a job that names vLLM's auto gets it on the line in the number's place, and nothing else changes
    named = S.server_argv(S.ARMS["published"], glm, Path("/ck"), 8000,
                          max_model_len=S.published_context("published", "glm53", S.AUTO))
    limit = argv.index("--max-model-len") + 1
    assert named == [*argv[:limit], "auto", *argv[limit + 1:]]
    # the recipe's command for the cards, flag for flag, with the front end's fallback for a build without it
    for flag, value in (("--served-model-name", "glm-5.3"), ("--tensor-parallel-size", "8"), ("--tokenizer-mode", "auto"),
                        ("--kv-cache-dtype", "fp8"), ("--speculative-config.method", "mtp"),
                        ("--speculative-config.num_speculative_tokens", "5"), ("--tool-call-parser", "glm47"),
                        ("--reasoning-parser", "glm47")):
        assert argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value, flag
    assert "--enable-auto-tool-choice" in argv and "--max-num-seqs" not in argv
    # the Rust front end: vLLM's serve command with the checkpoint as its argument and the recipe's tokenizer mode,
    # every other flag as behind the Python one; the variable is the published arm's alone, whatever the job's says
    rust = S.server_argv(S.ARMS["published"], glm, Path("/ck"), 8000, max_model_len=context, front=S.RUST)
    assert rust[:5] == [argv[0], "-m", "vllm.entrypoints.cli.main", "serve", "/ck"]
    assert argv[:5] == [argv[0], "-m", "vllm.entrypoints.openai.api_server", "--model", "/ck"]
    mode = argv.index("--tokenizer-mode") + 1
    assert rust[5:] == [*argv[5:mode], "glm5_next", *argv[mode + 1:]]
    assert [S.front_end(S.ARMS["published"], g) for g in (None, "python", "rust")] == ["python", "python", "rust"]
    assert S.front_end(S.ARMS["compiled"], "rust") == "python" and S.FRONT_ENDS == ("python", "rust")
    monkeypatch.setenv("VLLM_USE_RUST_FRONTEND", "1")
    env = S.server_env(S.ARMS["published"], tmp_path, tmp_path, None, 1.0, 1.0)
    assert "VLLM_USE_RUST_FRONTEND" not in env
    assert S.server_env(S.ARMS["published"], tmp_path, tmp_path, None, 1.0, 1.0, front=S.RUST)["VLLM_USE_RUST_FRONTEND"] == "1"
    assert S.base_args(without=S.UNPUBLISHED) == ("--max-model-len", "4096", "--seed", "0", "--disable-uvicorn-access-log")
    # auto is the engine's to resolve: no load is checked against it, and a number still is, the recipe's among them
    assert S.context_of({"long": (1,)}, "auto") == "auto" and S.context_arg("auto") == "auto" and S.context_arg("4096") == 4096
    assert S.context_of({"long": (1,)}, context) == 262144
    with pytest.raises(ValueError):
        S.context_arg("automatic")
    with pytest.raises(SystemExit, match="need a context of 16384"):
        S.context_of({"long": (1,)}, 4096)
    for model_id, why in (("moonlight", "no recipe for moonlight"), (None, "no recipe for no model")):
        with pytest.raises(SystemExit, match=why + r" \(Model.published; it holds glm53's\)"):
            S.published_context("published", model_id, None)
    # the line of a model with no recipe, which a job never starts, adds nothing to stock's
    plain = S.server_argv(S.ARMS["compiled"], S.MODELS["moonlight"], Path("/ck"), 8000)
    seqs = plain.index("--max-num-seqs")
    assert S.server_argv(S.ARMS["published"], S.MODELS["moonlight"], Path("/ck"), 8000) == plain[:seqs] + plain[seqs + 2:]
    # a job's own flags of vLLM's go last on the recipe's line and on no other arm's; one the line sets already,
    # or one named twice, is refused, and so is anything but --FLAG=VALUE
    tuning = S.published_args("--max-num-seqs=256,--gpu-memory-utilization=0.95,--max-num-batched-tokens=8192")
    assert tuning == ("--max-num-seqs=256", "--gpu-memory-utilization=0.95", "--max-num-batched-tokens=8192")
    assert S.server_argv(S.ARMS["published"], glm, Path("/ck"), 8000, max_model_len=context, added=tuning) == [*argv, *tuning]
    assert S.server_argv(S.ARMS["compiled"], glm, Path("/ck"), 8000, max_model_len=context, added=tuning) == stock
    assert S.published_args(None) == S.published_args("") == ()
    for twice in (("--kv-cache-dtype=auto",), ("--max-model-len=262144",), ("--max-num-seqs=8", "--max-num-seqs=16")):
        with pytest.raises(SystemExit, match=twice[0].split("=")[0] + ": the arm's line sets it already"):
            S.server_argv(S.ARMS["published"], glm, Path("/ck"), 8000, max_model_len=context, added=twice)
    for bad in ("--max-num-seqs 256", "--max-num-seqs", "max-num-seqs=256", "--a=1=2", "--a="):
        with pytest.raises(SystemExit, match="each flag is --FLAG=VALUE"):
            S.published_args(bad)
    # a recipe is its flags, its context limit and its tokenizer mode behind each front end together, and one arm
    # alone is started by it
    for name, m in S.MODELS.items():
        assert bool(m.published) == (m.published_context is not None) == (set(m.published_tokenizer) == set(S.FRONT_ENDS)), name
    assert [n for n, a in S.ARMS.items() if a.published] == ["published", "prod", "prod-again", "prod-record", "prod-seeded",
                                                             "prod-trace"]
    assert S.ARMS["published"].parent == "compiled" and S.changes_of("published") == []


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


def test_an_arm_starts_no_pass_it_cannot_expect_to_finish_by_its_stop_time():
    """`stop_before` goes by the pass's own longest wall time, or any pass's where it has none, with the pause
    between passes and the server's stop; an arm with no stop time, and one with enough left, serve on."""
    served = [{"load": "chat@4", "rep": 1, "wall_s": 600.0}, {"load": "chat@8", "rep": 1, "wall_s": 300.0},
              {"load": "chat@4", "rep": 2, "wall_s": 620.0}]
    assert S.stop_before(served, "chat@4", None) is None
    # chat@4's own: (620 + 5) * 1.1 + 60 = 747.5 s; chat@8's own: (300 + 5) * 1.1 + 60 = 395.5 s
    assert [S.stop_before(served, "chat@4", left) for left in (748.0, 747.0)] == [None, 747.5]
    assert [S.stop_before(served, "chat@8", left) for left in (396.0, 395.0)] == [None, 395.5]
    # a pass not served yet goes by the longest of any, and an arm's first by the stop alone
    assert S.stop_before(served, "long@1", 700.0) == 747.5 and S.stop_before(served, "long@1", 800.0) is None
    assert S.stop_before([], "chat@4", 61.0) is None and S.stop_before([], "chat@4", 59.0) == 60.0
    assert S.parse_stop_at(None) is None
    assert S.parse_stop_at("2030-01-01T00:00:00Z") == S.parse_stop_at("2030-01-01T01:00:00+01:00") == 1893456000.0
    for bad in ("2030-01-01T00:00:00", "soon", "1893456000"):
        with pytest.raises(SystemExit, match="a time with its zone"):
            S.parse_stop_at(bad)


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


# ------------------------------------------------------------------------------------- vLLM's own client and the cost

def result(completed: int = 200, failed: int = 0, duration: float = 100.0, good: float | None = 150.0, tok: float = 300.0) -> dict:
    """A result as `vllm bench serve --save-result` writes it (its keys at the integration's pin), ``good`` requests
    within the SLO."""
    return {"date": "20260101-000000", "endpoint_type": "openai", "backend": "openai", "label": None, "model_id": "glm-5.3",
            "tokenizer_id": "/ck", "num_prompts": completed + failed,
            "request_rate": "inf", "burstiness": 1.0, "max_concurrency": 8, "duration": duration, "completed": completed,
            "failed": failed, "total_input_tokens": 51200, "total_output_tokens": int(tok * duration),
            "request_throughput": completed / duration, "request_goodput": None if good is None else good / duration,
            "output_throughput": tok, "total_token_throughput": tok + 512.0, "latencies": [4.0] * completed,
            "queue_times": [0.001] * completed, "max_output_tokens_per_s": 2 * tok,
            "max_concurrent_requests": 8, "rtfx": 0.0, "mean_ttft_ms": 410.0, "median_ttft_ms": 400.0, "std_ttft_ms": 20.0,
            "p99_ttft_ms": 900.0, "mean_tpot_ms": 31.0, "median_tpot_ms": 30.0, "std_tpot_ms": 2.0, "p99_tpot_ms": 45.0,
            "mean_itl_ms": 30.5, "median_itl_ms": 29.0, "std_itl_ms": 3.0, "p99_itl_ms": 70.0, "mean_e2el_ms": 4000.0,
            "median_e2el_ms": 3900.0, "std_e2el_ms": 300.0, "p99_e2el_ms": 5200.0}


def test_a_pass_of_vllms_client_holds_a_concurrency_and_counts_goodput_against_the_models_slo(tmp_path: Path):
    glm, chat = S.MODELS["glm53"], S.LOADS["chat"]
    argv = S.bench_argv(glm, Path("/ck"), 8123, chat, 16, tmp_path, "chat@16-r2", dataset="/data/sharegpt.json")
    assert argv[1:] == [
        "-m", "vllm.entrypoints.cli.main", "bench", "serve", "--backend", "openai", "--base-url", "http://127.0.0.1:8123",
        "--endpoint", "/v1/completions", "--model", "glm-5.3", "--tokenizer", "/ck",
        "--dataset-name", "sharegpt", "--dataset-path", "/data/sharegpt.json", "--num-prompts", "200", "--seed", "0",
        "--request-rate", "inf", "--max-concurrency", "16", "--ignore-eos", "--temperature", "0",
        "--goodput", "ttft:2000", "tpot:80", "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "99",
        "--request-id-prefix", "chat@16-r2-", "--ready-check-timeout-sec", "0", "--disable-tqdm",
        "--save-result", "--save-detailed", "--result-dir", str(tmp_path), "--result-filename", "result.json"]
    # the client's tokenizer loads as the server's does, and a load with no file names none; long context is the
    # client's synthetic dataset at one long length, its goodput against the time per output token alone
    moon = S.bench_argv(S.MODELS["moonlight"], Path("/ck"), 1, S.LOADS["long"], 4, tmp_path, "warm", prompts=8)
    assert "--trust-remote-code" in moon and "--dataset-path" not in moon
    assert moon[moon.index("--num-prompts") + 1] == "8" and moon[moon.index("--dataset-name") + 1] == "random"
    assert moon[moon.index("--random-input-len") + 1] == "8192" and moon[moon.index("--goodput") + 1:][:2] == [
        "tpot:80", "--percentile-metrics"]
    # the release-order arm reads request k's release number from the id the client sends
    plugin = pytest.importorskip("servetax_plugin")
    assert plugin.release_key("cmpl-chat@16-r2-7-0-0123abcd") == ("chat@16-r2", 7)


def test_loads_are_named_with_their_rungs_and_the_engines_context_fits_them():
    assert S.parse_loads("chat") == {"chat": S.LOADS["chat"].rungs}
    assert S.parse_loads("long=2+4, chat=12+16+24") == {"long": (2, 4), "chat": (12, 16, 24)}
    assert (S.pass_name("chat", 16), S.load_tag("chat@16", 2, True), S.load_tag("light", 2, False)) == (
        "chat@16", "chat@16-r2", "light2")
    for bad in ("", "sonnet", "random", "chat=0", "chat=8+x"):
        with pytest.raises((SystemExit, ValueError)):
            S.parse_loads(bad)
    # chat fits every engine's context as it is; long context raises it, and a given limit it doesn't fit is refused
    chat, both = S.parse_loads("chat"), S.parse_loads("chat,long")
    assert (S.context_of(chat, None), S.context_of(both, None), S.context_of(both, 131072)) == (None, 16384, 131072)
    argv = S.server_argv(S.ARMS["compiled"], S.MODELS["glm53"], Path("/ck"), 1, None, S.context_of(both, None))
    assert S.context_of(chat, 4096) == 4096 and argv[argv.index("--max-model-len") + 1] == "16384"
    with pytest.raises(SystemExit, match="load long's requests need a context of 16384"):
        S.context_of(both, 8192)
    # each long request, its output and the tokens the client's alignment may add, fits the context the load names
    args = dict(zip(S.LOADS["long"].dataset[::2], S.LOADS["long"].dataset[1::2]))
    assert int(args["--random-input-len"]) + int(args["--random-output-len"]) + 64 <= S.LOADS["long"].context


def test_the_load_that_repeats_a_prefix_is_four_planned_requests_sent_one_at_a_time(tmp_path: Path):
    """Two prompts near the context limit, each sent again with another ending. The client's file gives each request
    its lengths and an id a block, and the second of a pair carries the first's ids for the blocks they share, so the
    client draws both the same tokens there and no others. One request in flight: a second request starts when the
    prompt it repeats has been computed."""
    load, plan = S.LOADS["prefix"], S.PAIRS
    assert (load.plan, load.prompts, load.rungs, load.number, load.file) == (plan, 4, (1,), False, None)
    assert [n for n, x in S.LOADS.items() if x.number] == ["chat", "long"]
    assert S.planned(S.trace_lines(plan)) == {"requests": 4, "prompt_tokens": 519647, "repeated_tokens": 259712,
                                              "span_s": 0.0}
    # every request with its output fits the context the load names, and a pair shares whole blocks both prompts have
    assert all(r.prompt + r.output <= load.context for r in plan)
    assert [(r.after, r.blocks * S.BLOCK) for r in plan] == [(None, 0), (0, 130752), (None, 0), (2, 128960)]
    f = Path(S.write_trace(tmp_path / "prefix.trace.jsonl", S.trace_lines(plan)))
    lines = [json.loads(x) for x in f.read_text().splitlines()]
    assert lines == S.trace_lines(plan) and all(set(x) == {"timestamp", "input_length", "output_length", "hash_ids"} for x in lines)
    assert [(x["input_length"], x["output_length"]) for x in lines] == [(r.prompt, r.output) for r in plan]
    a, b, c, d = (x["hash_ids"] for x in lines)
    assert [len(x) for x in (a, b, c, d)] == [2044, 2044, 2016, 2016]         # an id a block, the last block short
    assert b[:2043] == a[:2043] and d[:2015] == c[:2015]                      # the blocks a pair shares
    assert len({*a, *b[2043:], *c, *d[2015:]}) == 2044 + 1 + 2016 + 1         # and no other block is shared
    for bad in ((S.Planned(0.0, 640, 8, after=0, blocks=1),), (plan[0], S.Planned(0.0, 640, 8, after=0, blocks=11)),
                (plan[0], S.Planned(0.0, 640, 8, after=0))):
        with pytest.raises(ValueError, match="an earlier request"):
            S.trace_lines(bad)
    # the client sends the file's lines in its order, one in flight, each prompt as token ids of its planned length
    argv = S.bench_argv(S.MODELS["glm53"], Path("/ck"), 1, load, 1, tmp_path, "prefix@1-r1", dataset=str(f))
    at = argv.index("--dataset-name")
    assert argv[at:at + 12] == ["--dataset-name", "timed_trace", "--timed-trace-chunk-hash-size", "64",
                                "--dataset-path", str(f), "--num-prompts", "4", "--seed", "0", "--no-self-timed",
                                "--request-rate"]
    assert argv[argv.index("--max-concurrency") + 1] == "1" and argv[argv.index("--goodput") + 1:][:2] == [
        "tpot:80", "--percentile-metrics"]
    # its engine's context is its own, so it is served in a run of its own
    assert S.context_of(S.parse_loads("prefix"), None) == 131072
    # and at its own concurrency: with more in flight a repeat would arrive before the prompt it repeats is computed
    assert S.parse_loads("prefix") == {"prefix": (1,)}
    with pytest.raises(SystemExit, match="'prefix=2': prefix's requests are sent as planned, at concurrency 1"):
        S.parse_loads("prefix=2")
    with pytest.raises(SystemExit, match="prefix is served in a run of its own"):
        S.parse_loads("chat,prefix")


#: a table of planned requests: the second begins as the first does, on two whole blocks, and the fourth on one
TABLE = [{"at_s": 0.0, "prompt": 200, "output": 8},
         {"at_s": 2.5, "prompt": 130, "output": 8, "after": 0, "blocks": 2},
         {"at_s": 30.0, "prompt": 64, "output": 8},
         {"at_s": 30.0, "prompt": 100, "output": 8, "after": 0, "blocks": 1}]


def test_a_table_of_planned_requests_is_checked_and_sent_as_its_requests_arrive(tmp_path: Path, capsys):
    """``--dataset prefix=FILE`` serves a table of planned requests in place of the four: arrival times and lengths,
    with the earlier line a request repeats, and nothing else; each request is sent at its arrival time, and what the
    table plans is read from its lines."""
    load, f = S.LOADS["prefix"], tmp_path / "table.jsonl"
    f.write_text("".join(json.dumps(x) + "\n" for x in TABLE))
    plan = S.read_plan("prefix", f)
    assert plan == (S.Planned(0.0, 200, 8), S.Planned(2.5, 130, 8, 0, 2), S.Planned(30.0, 64, 8), S.Planned(30.0, 100, 8, 0, 1))
    lines = S.trace_lines(plan)
    # what repeats is the whole leading blocks an earlier line begins with: 128 tokens of the second, 64 of the fourth
    assert S.planned(lines) == {"requests": 4, "prompt_tokens": 494, "repeated_tokens": 192, "span_s": 30.0}
    # an id an earlier line has repeats nothing unless it is where both prompts begin
    assert S.planned([{"timestamp": 0.0, "input_length": 200, "output_length": 8, "hash_ids": [1, 2, 3, 4]},
                      {"timestamp": 1.0, "input_length": 64, "output_length": 8, "hash_ids": [2]},
                      {"timestamp": 2.0, "input_length": 100, "output_length": 8, "hash_ids": [1, 3]}])["repeated_tokens"] == 64
    loads = S.parse_loads("prefix")
    files = S.parse_datasets([f"prefix={f}"], loads)
    assert files == {"prefix": str(f)}
    # the given table is sent as its requests arrive; with none, the four are, one in flight
    assert S.passes_of(loads, files) == ({"prefix": (S.ARRIVING,)}, {"prefix": lines})
    assert S.passes_of(loads, {}) == ({"prefix": (1,)}, {"prefix": S.trace_lines(S.PAIRS)})
    assert S.pass_name("prefix", S.ARRIVING) == "prefix@0"
    # a pass sends every line; a warm-up the lines that arrive in its seconds, one at least, or two a request in flight
    assert [S.pass_prompts(load, S.ARRIVING, lines, w) for w in (None, 20.0, 30.0, 0.0)] == [4, 2, 4, 1]
    assert S.pass_prompts(load, S.ARRIVING, lines[1:], 1.0) == 1
    assert [S.pass_prompts(x, c, None, w) for x, c, w in ((load, 1, None), (load, 1, 20.0), (S.LOADS["chat"], 8, None),
                                                          (S.LOADS["chat"], 8, 20.0))] == [4, 2, 200, 16]
    # a given pass size (--pass-prompts) is every rung's alike; a warm-up stays within the load's own prompts
    chat = S.LOADS["chat"]
    assert [S.pass_prompts(chat, c, None, prompts=1024) for c in (8, 96, 256, 384)] == [1024] * 4
    assert [S.pass_prompts(chat, c, None, 20.0, 1024) for c in (8, 96, 256)] == [16, 192, 200]
    assert S.pass_prompts(chat, 96, None, 20.0, 100) == 100
    # Kx: K prompts for each request the rung holds, the load's own at least; its warm-up is the load's
    assert [S.pass_prompts(chat, c, None, prompts="8x") for c in (4, 25, 26, 96, 256)] == [200, 200, 208, 768, 2048]
    assert [S.pass_prompts(chat, c, None, 20.0, "8x") for c in (8, 96, 256)] == [16, 192, 200]
    both = {"chat": (96, 256), "long": (1, 2)}
    assert S.parse_pass_prompts("chat=1024, long=64", both) == {"chat": 1024, "long": 64}
    assert S.parse_pass_prompts("chat=08x,long=64", both) == {"chat": "8x", "long": 64}
    assert S.parse_pass_prompts(None, both) == {} == S.parse_pass_prompts("", both)
    for text, given, why in (("chat=1024", {"long": (1,)}, "'chat' is not a load of --loads"),
                             ("prefix=8", {"prefix": (1,)}, "sent as planned"), ("chat=0", both, "1 or more"),
                             ("chat", both, "NAME=N"), ("chat=0x", both, "NAME=Kx"), ("chat=x4", both, "NAME=N"),
                             ("chat=4xx", both, "NAME=N")):
        with pytest.raises(SystemExit, match=why):
            S.parse_pass_prompts(text, given)
    # the client then sends each at its arrival: nothing holds a concurrency, and nothing turns the file's times off;
    # the file it reads is the one the arm wrote from the table, never the table
    trace = S.write_trace(tmp_path / "prefix.trace.jsonl", lines)
    assert [json.loads(x) for x in Path(trace).read_text().splitlines()] == lines
    assert S.client_files({"chat": "chat.json", **files}, {"prefix": trace}) == {"chat": "chat.json", "prefix": trace}
    assert S.client_files({"chat": "chat.json"}, {}) == {"chat": "chat.json"}
    argv = S.bench_argv(S.MODELS["glm53"], Path("/ck"), 1, load, S.ARRIVING, tmp_path, "prefix@0-r1", dataset=trace, prompts=4)
    at = argv.index("--dataset-name")
    assert argv[at:at + 11] == ["--dataset-name", "timed_trace", "--timed-trace-chunk-hash-size", "64",
                                "--dataset-path", trace, "--num-prompts", "4", "--seed", "0", "--ignore-eos"]
    assert not {"--no-self-timed", "--request-rate", "--max-concurrency"} & set(argv)
    # the guards of the four stand: no rung is named for it, and it is served in a run of its own
    with pytest.raises(SystemExit, match="'prefix=0': each rung is a concurrency of 1 or more"):
        S.parse_loads("prefix=0")
    with pytest.raises(SystemExit, match="prefix is served in a run of its own"):
        S.parse_datasets([f"prefix={f}"], S.parse_loads("long,prefix"))
    with pytest.raises(SystemExit, match=r"--dataset prefix=.*absent\.jsonl: no such file"):
        S.parse_datasets([f"prefix={tmp_path / 'absent.jsonl'}"], loads)
    # a line that is not a request under the table's names is refused, so a table holds no token and no text
    one = TABLE[0]
    bad = tmp_path / "bad.jsonl"
    for line, why in ((one | {"text": "a text"}, "line 2: a request is at_s, prompt, output, with after and blocks where it "
                                                 "begins as an earlier one does, and nothing else"),
                      (one | {"token_ids": [1, 2, 3, 4]}, "and nothing else"),
                      (one | {"after": 0}, "and nothing else"),
                      ([1, 2], "and nothing else"),
                      (one | {"prompt": 0}, "a prompt and an output of whole tokens"),
                      (one | {"output": 8.5}, "a prompt and an output of whole tokens"),
                      (one | {"prompt": True}, "a prompt and an output of whole tokens"),
                      (one | {"output": 131072 - 199}, "131,072 at most together"),
                      (one | {"after": 1, "blocks": 1}, "line 2: after: an earlier line, counted from 0; blocks: whole "
                                                        "blocks of 64 tokens both prompts have"),
                      (one | {"after": 0, "blocks": 4}, "after: an earlier line"),
                      (one | {"after": 0, "blocks": 0}, "after: an earlier line"),
                      (one | {"after": 0, "blocks": 1.0}, "after: an earlier line"),
                      (one | {"at_s": -1.0}, "an arrival in seconds, from zero on and no earlier than the line's before it"),
                      (one | {"at_s": "0"}, "an arrival in seconds"),
                      ("{", "line 2: not a JSON object")):
        bad.write_text(json.dumps(one) + "\n" + (line if isinstance(line, str) else json.dumps(line)) + "\n")
        with pytest.raises(SystemExit, match=why):
            S.read_plan("prefix", bad)
    bad.write_text(json.dumps(TABLE[2]) + "\n" + json.dumps(TABLE[0]) + "\n")       # the table's order is the order sent
    with pytest.raises(SystemExit, match="line 2: an arrival in seconds, from zero on and no earlier than the line's before it"):
        S.passes_of(loads, {"prefix": str(bad)})
    bad.write_text("\n")
    with pytest.raises(SystemExit, match="no request"):
        S.read_plan("prefix", bad)
    # the command a job runs before it holds a GPU reads the table and says what it plans
    capsys.readouterr()
    assert S.main(["passes", "--loads", "prefix", "--dataset", f"prefix={f}"]) == 0
    assert capsys.readouterr().out == (
        "[serve-tax] load prefix: a table's planned requests, each sent at its arrival time, of synthetic token ids "
        f"(vLLM's timed_trace dataset); 4 prompts a pass as they arrive, from {f} (SHA-256 {S.file_digest(f)[:16]}...), "
        "192 of their 494 prompt tokens repeated, arriving over 30.0 s\n")
    assert S.main(["passes", "--loads", "prefix"]) == 0
    assert capsys.readouterr().out.endswith("(vLLM's timed_trace dataset); 4 prompts a pass at concurrency 1, 259,712 of "
                                            "their 519,647 prompt tokens repeated\n")


def test_the_full_size_table_is_the_one_its_source_names_and_its_maker_makes(tmp_path: Path):
    """`prefix_plan.jsonl` is the table `prefix_plan.source.json` records, it reads as a planned load, and its two
    repeats are the four requests' own; `prefix_plan.table` makes such a table from a serve plan's rows and meta."""
    import prefix_plan as M

    source = json.loads(M.SOURCE.read_text())
    table = M.HERE / source["table"]
    assert S.file_digest(table) == source["sha256"] and source["plan"].startswith("art:") and len(source["files"]) == 2
    plan = S.read_plan("prefix", table)
    assert S.planned(S.trace_lines(plan)) == {"requests": 651, "prompt_tokens": 2235146, "repeated_tokens": 259712,
                                              "span_s": 1486.955}
    repeats = [(plan[r.after], r) for r in plan if r.after is not None]
    assert [(x.prompt, y.prompt, y.blocks) for x, y in repeats] == [
        (x.prompt, y.prompt, y.blocks) for x, y in ((S.PAIRS[0], S.PAIRS[1]), (S.PAIRS[2], S.PAIRS[3]))]
    assert all(x.at_s < y.at_s for x, y in repeats) and M.BLOCK == S.BLOCK
    # the maker: the rows sent within the span, arrivals the running sum of the draws over the rate, and a long
    # request's hit as the earlier long request of its book and the whole blocks the two share
    rows = [{"i": 0, "kind": "chat", "prompt_len": 20, "max_tokens": 8, "gap_draw": 0.2},
            {"i": 1, "kind": "long128k", "book": "b", "prompt_len": 200, "max_tokens": 8, "gap_draw": 0.4},
            {"i": 2, "kind": "long128k-followup", "book": "b", "prompt_len": 190, "max_tokens": 8, "gap_draw": 0.4},
            {"i": 3, "kind": "chat", "prompt_len": 5, "max_tokens": 8, "gap_draw": 1.0}]
    meta = {"rate": 0.4, "max_model_len": 4096, "sent_within_span": 3,
            "long_rows": [{"i": 1, "model": {"prefix_hit": 0}}, {"i": 2, "model": {"prefix_hit": 128}}]}
    lines = M.table(rows, meta)
    assert lines == [{"at_s": 0.5, "prompt": 20, "output": 8}, {"at_s": 1.5, "prompt": 200, "output": 8},
                     {"at_s": 2.5, "prompt": 190, "output": 8, "after": 1, "blocks": 2}]
    f = tmp_path / "made.jsonl"
    f.write_text(M.text(lines))
    assert S.read_plan("prefix", f) == (S.Planned(0.5, 20, 8), S.Planned(1.5, 200, 8), S.Planned(2.5, 190, 8, 1, 2))
    meta["long_rows"][1]["model"]["prefix_hit"] = 100
    with pytest.raises(SystemExit, match="row 2: its hit of 100 is not whole blocks of one earlier row of its book"):
        M.table(rows, meta)
    with pytest.raises(SystemExit, match="row 1 does not fit the context of 100"):
        M.table(rows, meta | {"max_model_len": 100})


@pytest.fixture
def pinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[S.Dataset, bytes, Path]:
    """Chat's dataset pinned to a few bytes, under a node's Hugging Face cache in ``tmp_path``: the pin, the bytes,
    and where the node's copy lies."""
    import hashlib

    data = b'[{"id": "a", "conversations": [{"value": "hello"}, {"value": "hi there"}]}]'
    ds = S.Dataset("org/set", "f" * 40, "set.json", len(data), hashlib.sha512(data).hexdigest())
    monkeypatch.setitem(S.LOADS, "chat", S.replace(S.LOADS["chat"], file=ds))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    at = tmp_path / "hf" / "hub" / "datasets--org--set" / "snapshots" / ("f" * 40) / "set.json"
    assert ds.path() == at and ds.url() == f"https://huggingface.co/datasets/org/set/resolve/{'f' * 40}/set.json"
    return ds, data, at


class Body:
    """A download's body, as `urlopen` returns it."""

    def __init__(self, data: bytes) -> None:
        import io

        self.body = io.BytesIO(data)

    def __enter__(self) -> Body:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, n: int) -> bytes:
        return self.body.read(n)


def test_a_loads_dataset_file_is_the_nodes_copy_of_the_pinned_one_or_nothing(tmp_path: Path, pinned, capsys):
    ds, data, at = pinned
    loads = S.parse_loads("chat=4,long=4")
    assert S.parse_datasets(None, S.parse_loads("long")) == {}          # a load of no file needs none
    # no copy on the node: refused with the command that downloads it, and nothing is downloaded unasked
    with pytest.raises(SystemExit, match=r"no dataset file at .*set\.json: `serve_tax.py passes --loads chat --fetch` downloads"):
        S.parse_datasets(None, loads, open_url=lambda url: pytest.fail("downloaded unasked"))
    # a download that is not the pinned file leaves nothing behind, in place or beside it
    with pytest.raises(SystemExit, match=r"is not the pinned file: its SHA-512 is .* \(nothing kept\)"):
        S.parse_datasets(None, loads, fetch_missing=True, open_url=lambda url: Body(data + b" "))
    assert list(at.parent.iterdir()) == []
    urls = []
    got = S.parse_datasets(None, loads, fetch_missing=True, open_url=lambda url: urls.append(url) or Body(data))
    assert got == {"chat": str(at)} and urls == [ds.url()] and at.read_bytes() == data
    assert [p.name for p in at.parent.iterdir()] == ["set.json"]
    # it is there now: read again, never downloaded again
    assert S.parse_datasets(None, loads, fetch_missing=True, open_url=lambda url: pytest.fail("downloaded twice")) == got
    # the node's copy changed, or another file given as a copy: refused by its SHA-512, and a given file is never replaced
    other = tmp_path / "other.json"
    other.write_bytes(b"[]")
    with pytest.raises(SystemExit, match=r"other\.json is not the pinned dataset file \(org/set's set\.json at ffffffffffff\)"):
        S.parse_datasets([f"chat={other}"], loads, fetch_missing=True, open_url=lambda url: pytest.fail("replaced a given file"))
    with pytest.raises(SystemExit, match="no dataset file at"):
        S.parse_datasets([f"chat={tmp_path / 'absent.json'}"], loads, fetch_missing=True,
                         open_url=lambda url: pytest.fail("downloaded to a given path"))
    copy = tmp_path / "copy.json"
    copy.write_bytes(data)
    assert S.parse_datasets([f"chat={copy}"], loads) == {"chat": str(copy)}
    at.write_bytes(data + b"\n")
    with pytest.raises(SystemExit, match="is not the pinned dataset file"):
        S.parse_datasets(None, loads)
    with pytest.raises(SystemExit, match="long: not a load of --loads that reads a file"):
        S.parse_datasets([f"chat={copy}", f"long={copy}"], loads)
    # the command a job runs before it holds a GPU prints the checked file; `dataset` prints its path alone
    at.write_bytes(data)
    capsys.readouterr()
    assert S.main(["passes", "--loads", "chat=4+8,long", "--arms", "compiled two-stage", "--model-id", "glm53"]) == 0
    text = capsys.readouterr().out
    assert f"200 prompts a pass at concurrency 4, 8, from {at} (SHA-512 {ds.sha512[:16]}...)" in text
    assert "load long: 8,192 random tokens in and 128 out" in text and "24 prompts a pass at concurrency 1, 2, 4, 8\n" in text
    # a given pass size is printed, with the load's own beside it
    assert S.main(["passes", "--loads", "chat=256+384+96,long=1+2", "--pass-prompts", "chat=1024"]) == 0
    text = capsys.readouterr().out
    assert (f"; 1,024 prompts a pass at concurrency 256, 384, 96 (--pass-prompts: the load's own are 200), from {at} "
            in text)
    assert "24 prompts a pass at concurrency 1, 2\n" in text
    # sized by the rung, each rung's is printed
    assert S.main(["passes", "--loads", "chat=12+48+96", "--pass-prompts", "chat=8x"]) == 0
    assert "; 200, 384, 768 prompts a pass at concurrency 12, 48, 96 (--pass-prompts: " in capsys.readouterr().out
    with pytest.raises(SystemExit, match="'long' is not a load of --loads"):
        S.main(["passes", "--loads", "chat", "--pass-prompts", "long=64"])
    assert S.main(["dataset", "chat"]) == 0 and capsys.readouterr().out == f"{at}\n"
    for argv, why in ((["passes", "--loads", "chat", "--arms", "compiled absent"], "absent"),
                      (["passes", "--loads", "chat", "--arms", "tick"], "corpus client"),
                      (["passes", "--loads", "chat", "--arms", "two-stage", "--model-id", "moonlight"], "check template"),
                      (["passes", "--loads", "long", "--max-model-len", "4096"], "need a context of 16384"),
                      (["passes", "--loads", "chat", "--arms", "published", "--model-id", "moonlight"], "no recipe for"),
                      (["passes", "--loads", "long", "--arms", "compiled published", "--model-id", "glm53",
                        "--max-model-len", "4096"], "need a context of 16384"),
                      (["dataset", "long"], None)):                # a load that reads no file is no choice of `dataset`
        with pytest.raises(SystemExit, match=why):
            S.main(argv)
    # the published arm is served at the job's context limit, a number or vLLM's auto, like the arms beside it
    for limit in ("131072", "262144", "auto"):
        assert S.main(["passes", "--loads", "long", "--arms", "compiled published", "--model-id", "glm53",
                       "--max-model-len", limit]) == 0
    assert S.main(["passes", "--loads", "long", "--arms", "published", "--model-id", "glm53"]) == 0
    # a model with no check template of its own takes the job's object
    assert S.main(["passes", "--loads", "long", "--arms", "two-stage", "--model-id", "moonlight", "--boundary-rows",
                   '{"check": "t"}']) == 0


def test_the_pinned_dataset_is_a_whole_sha512_and_a_commit_of_its_repository():
    ds = S.LOADS["chat"].file
    assert re.fullmatch(r"[0-9a-f]{128}", ds.sha512) and re.fullmatch(r"[0-9a-f]{40}", ds.revision) and ds.size > 0
    assert [n for n, load in S.LOADS.items() if load.file] == ["chat", "agentic", "agentic-full", "agentic-256k"]
    assert list(S.LOADS) == ["chat", "long", "prefix", "agentic", "agentic-full", "agentic-256k"]


def test_drive_bench_keeps_the_clients_result_and_counts_its_failed_requests(tmp_path: Path,
                                                                             monkeypatch: pytest.MonkeyPatch):
    """The run keeps the client's detailed result, each request's own figures, without the generated texts and the
    inter-token times, which the client saved outside the run's directory, in a directory gone when the pass ends."""
    now, calls, seeds, saved = [0.0], [], [], []
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(S.tempfile, "tempdir", str(tmp_path / "tmp"))
    tmp_path = tmp_path / "run"
    detail = {"input_lens": [8192] * 197, "output_lens": [128] * 197, "ttfts": [0.4] * 197, "start_times": [1.0] * 197,
              "errors": [None] * 197}

    def run(argv, stdout, stderr, env):
        calls.append(argv)
        seeds.append((env.get("PYTHONHASHSEED"), "PATH" in env))
        stdout.write(b"Starting main benchmark run...\n")
        d = Path(argv[argv.index("--result-dir") + 1])
        saved.append(d)
        (d / argv[argv.index("--result-filename") + 1]).write_text(json.dumps(
            result(completed=197, failed=3) | detail | {"generated_texts": ["a served answer"] * 197,
                                                         "itls": [[0.03] * 127] * 197}))
        now[0] += 64.25
        return NS(returncode=0)

    m = S.drive_bench(8000, S.MODELS["glm53"], Path("/ck"), S.LOADS["long"], 8, "long@8-r1", tmp_path / "rep-1",
                      run=run, clock=lambda: now[0], now=lambda: 1700.0)
    assert (m["requests"], m["unfinished"], m["unserved"], m["concurrency"], m["rate"]) == (200, 3, 3, 8, None)
    assert (m["wall_s"], m["t0_unix"], m["client"], m["tag"]) == (64.25, 1700.0, "vllm-bench", "long@8-r1")
    assert calls == [S.bench_argv(S.MODELS["glm53"], Path("/ck"), 8000, S.LOADS["long"], 8, saved[0], "long@8-r1")]
    assert not saved[0].exists() and saved[0].parent == (tmp_path.parent / "tmp").resolve()
    kept = json.loads((tmp_path / "rep-1" / "result.json").read_text())
    assert kept == result(completed=197, failed=3) | detail and not S.UNKEPT & kept.keys()
    assert "a served answer" not in "".join(f.read_text() for f in (tmp_path / "rep-1").iterdir())
    assert seeds == [("0", True)]        # the job's environment, with the seed a planned load's tokens are drawn under
    assert json.loads((tmp_path / "rep-1" / "meta.json").read_text()) == m
    assert (tmp_path / "rep-1" / "bench.log").read_bytes() == b"Starting main benchmark run...\n"
    for rc, why in ((1, "exited 1"), (0, "saved no result")):
        with pytest.raises(RuntimeError, match=why):
            S.drive_bench(8000, S.MODELS["glm53"], Path("/ck"), S.LOADS["long"], 8, "x", tmp_path / f"bad-{rc}",
                          run=lambda argv, stdout, stderr, env, rc=rc: NS(returncode=rc), clock=lambda: 0.0,
                          now=lambda: 0.0)
    # a temporary directory inside the run's would leave the texts there if the pass were stopped: refused, unsent
    monkeypatch.setattr(S.tempfile, "tempdir", str(tmp_path / "rep-1"))
    with pytest.raises(RuntimeError, match="inside the run's directory"):
        S.drive_bench(8000, S.MODELS["glm53"], Path("/ck"), S.LOADS["long"], 8, "x", tmp_path / "rep-1",
                      run=lambda *a, **k: pytest.fail("sent"), clock=lambda: 0.0, now=lambda: 0.0)


def test_a_pass_metrics_are_the_clients_own_figures():
    m = S.bench_metrics(result(completed=197, failed=3, duration=100.0, good=150.0, tok=300.0))
    assert (m["sent"], m["ok"], m["met"], m["output_tokens"], m["span_s"]) == (200, 197, 150, 30000, 100.0)
    assert (m["output_tok_per_s"], m["total_tok_per_s"], m["req_per_s"], m["goodput_req_per_s"]) == (300.0, 812.0, 1.97, 1.5)
    # the prompt's tokens beside the output's, which a reader converting tokens a second to compute needs apart
    assert (m["prompt_tokens"], m["prompt_tok_per_s"]) == (51200, 512.0)
    assert m["slo_attained"] == pytest.approx(0.75)
    assert (m["ttft_p50_s"], m["ttft_p99_s"], m["itl_p50_s"], m["itl_p99_s"], m["tpot_p50_s"], m["tpot_p99_s"]) == (
        0.4, 0.9, 0.029, 0.07, 0.03, 0.045)
    none = S.bench_metrics(result(good=None))        # a pass the client counted no goodput for
    assert none["goodput_req_per_s"] is None and none["met"] is None and none["slo_attained"] is None


def bench_arm(d: Path, name: str, passes: dict[str, list[dict]], model_id: str = "glm53", **meta: object) -> Path:
    """An arm's run dir as `run` leaves it under vLLM's client: each pass's reps' result.json and meta.json."""
    d = d / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({
        "arm": name, "parent": S.ARMS[name].parent, "what": S.ARMS[name].what, "model_id": model_id, "rates": {},
        "client": "vllm-bench", "run_id": "r-1", "commit": "c" * 40, "versions": {"vllm": "0.1+gd9105ea8"},
        "passes": {p: {"load": p.rpartition("@")[0], "concurrency": int(p.rpartition("@")[2])} for p in passes}, **meta}))
    for p, reps in passes.items():
        for i, r in enumerate(reps, 1):
            (d / p / f"rep-{i}").mkdir(parents=True)
            (d / p / f"rep-{i}" / "result.json").write_text(json.dumps(r))
            (d / p / f"rep-{i}" / "meta.json").write_text(json.dumps(
                {"rate": None, "concurrency": int(p.rpartition("@")[2]), "unfinished": r["failed"], "t0_unix": 1000.0,
                 "wall_s": r["duration"] + 5, "gpu": {"busy_gpu_s": 8 * r["duration"]}}))
    return d


def test_an_arm_served_by_vllms_client_reads_each_pass_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    d = bench_arm(tmp_path, "compiled", {"chat@8": [result(good=150.0), result(good=160.0)], "chat@16": [result(good=40.0)]})
    m = S.arm_metrics(d)
    assert list(m["loads"]) == ["chat@8", "chat@16"] and [r["rep"] for r in m["loads"]["chat@8"]] == [1, 2]
    one = m["loads"]["chat@8"][0]
    assert (one["concurrency"], one["rate"], one["goodput_req_per_s"], one["met"], one["busy_gpu_s"]) == (8, None, 1.5, 150, 800.0)
    assert (m["commit"], m["vllm"], m["run_id"]) == ("c" * 40, "0.1+gd9105ea8", "r-1")
    md = S.markdown(S.table([m]))
    assert "| compiled | chat@8 | concurrency 8 | 0 of 400 | 300 | 2 | 1.55 | 77.5% | 400 / 900 | 29 / 70 |" in md
    # the client counted goodput against the SLO it ran with: another can't be read from its result
    with pytest.raises(SystemExit, match="serve it again"):
        S.arm_metrics(d, (1.0, 0.05))
    # a server lifetime of an arm (run --lifetime, --leave-out) says which it is, what it leaves out and its reps
    assert (m["lifetime"], m["left_out"], m["reps"], m["seeded"]) == (None, 0, None, False)
    # a sampled arm says whether its client sent a seed; one served before the arms differed in it was sent one
    sampled = {"chat@8": [result(good=150.0)]}
    assert [S.arm_metrics(bench_arm(tmp_path / str(i), "compiled", sampled, **meta))["seeded"] for i, meta in enumerate(
        ({"sampled": S.SAMPLED, "seeded": False}, {"sampled": S.SAMPLED, "seeded": True}, {"sampled": S.SAMPLED}))] == [
            False, True, True]
    passes = {"chat@8": [result(good=150.0), result(good=160.0), result(good=155.0)]}
    two = S.arm_metrics(bench_arm(tmp_path / "second", "compiled", passes, lifetime=2, left_out=1, reps=3))
    assert (two["lifetime"], two["left_out"], two["reps"]) == (2, 1, 3)
    (one,) = S.lifetimes([two])
    assert [(r["rep"], r["lifetime"], r["pass"]) for r in one["loads"]["chat@8"]] == [(5, 2, 2), (6, 2, 3)]
    assert [r["rep"] for r in one["warm"]["chat@8"]] == [4] and one["lifetimes"] == [2]
    seen = []
    monkeypatch.setattr(S, "run", lambda a: seen.append((a.lifetime, a.leave_out)) or 0)
    run = ["run", "--arm", "prod", "--model-id", "glm53", "--out", str(tmp_path / "o")]
    assert S.main(run) == 0 == S.main([*run, "--lifetime", "2", "--leave-out", "1"]) and seen == [(None, 0), (2, 1)]
    sh = (S.HERE / "run.sh").read_text()
    # the job gives each lifetime its number and a directory of its own, and every arm what it leaves out
    assert 'dir=$arm.${item##*:}; life=(--lifetime "${item##*:}")' in sh and '${LEAVE_OUT:+--leave-out "$LEAVE_OUT"}' in sh
    assert '--private "$PRIVATE/$dir"} --out "$OUT/$dir"' in sh


def cost_arm(name: str, rate: dict[int, list[float]], load: str = "chat", good: dict[int, list[float]] | None = None) -> dict:
    """An arm's metrics with ``load`` served at each rung of ``rate``, a rep each value: the requests a second it
    served, 200 a rep, each of 256 prompt tokens and 150 output tokens, every one within the SLO unless ``good`` gives
    the rung's goodputs."""
    def rep(c: int, i: int, r: float) -> dict:
        g = (good or rate)[c][i]
        return {"rep": i + 1, "rate": None, "concurrency": c, "sent": 200, "ok": 200, "met": round(200 * g / r) if r else 0,
                "goodput_req_per_s": g, "req_per_s": r, "output_tok_per_s": 150 * r, "prompt_tok_per_s": 256 * r,
                "output_tokens": 30000, "prompt_tokens": 51200, "slo_attained": g / r if r else None}

    return {"arm": name, "parent": S.ARMS[name].parent, "what": name, "run_id": "r-1", "commit": "c" * 40,
            "vllm": "0.1+gd9105ea8", "model_id": "glm53", "slo": {"ttft_s": 2.0, "tpot_s": 0.08},
            "loads": {S.pass_name(load, c): [rep(c, i, r) for i, r in enumerate(rs)] for c, rs in rate.items()}}


def test_the_serving_cost_is_the_kits_paired_ratio_of_tokens_a_second_where_the_reference_is_saturated():
    """Both sides are read at one rung, the one where the reference's tokens a second are highest. The cost is the
    median over reps of the reference's tokens a second over the arm's in the same rep, less one."""
    ref = cost_arm("compiled", {8: [1.0] * 6, 16: [1.5, 1.6, 1.4, 1.5, 1.55, 1.45], 32: [1.2, 1.1, 1.3, 1.2, 1.1, 1.3]})
    ours = cost_arm("boundary-rows", {8: [1.3] * 6, 16: [1.2, 1.25, 1.15, 1.2, 1.25, 1.1]})
    assert S.saturated(ref, ours, "chat") == 16 and S.saturated(ref, ours, "long") is None
    c = S.serving_cost(ref, ours, "chat")
    ratios = sorted(x / y for x, y in zip([1.5, 1.6, 1.4, 1.5, 1.55, 1.45], [1.2, 1.25, 1.15, 1.2, 1.25, 1.1]))
    assert c["value"] == pytest.approx((ratios[2] + ratios[3]) / 2 - 1) == pytest.approx(0.25)
    assert (c["reps"], c["rung"], c["reference"]["concurrency"], c["arm"]["concurrency"]) == (6, 16, 16, 16)
    # six reps are the fewest whose ratios bound their median at 95%: the interval is then their least and greatest
    assert c["ci95"] == [pytest.approx(ratios[0] - 1), pytest.approx(ratios[5] - 1)]
    assert c["ci95"][0] < c["value"] < c["ci95"][1]
    # one slow rep moves the interval's end and not the median: the cost isn't the ratio of the sides' means
    assert c["value"] != pytest.approx(c["reference"]["output_tok_per_s"] / c["arm"]["output_tok_per_s"] - 1)
    # each side's tokens a second there, the prompt's apart from the output's, and the tokens a request it served
    assert c["reference"]["output_tok_per_s"] == pytest.approx(225.0) and c["reference"]["prompt_tok_per_s"] == pytest.approx(384.0)
    assert (c["arm"]["prompt_tokens_per_request"], c["arm"]["output_tokens_per_request"]) == (256.0, 150.0)
    # below the reference's capacity a ratio reads what was sent, not compute: there this arm would look faster than stock
    assert S.serving_cost(ref, ours, "chat", 8)["value"] == pytest.approx(1.0 / 1.3 - 1)
    # the rung is one both served: the reference's 32 is not the arm's, and an arm with one rung is read there
    assert S.serving_cost(ref, ours, "chat", 32) is None
    low = S.serving_cost(ref, cost_arm("boundary-rows", {8: [0.8] * 2}), "chat")
    assert low["rung"] == 8
    # the reference's own rungs show it saturated at 16: it served 32 too, and no more tokens a second anywhere
    assert (c["saturated"], c["peak"]) == (True, 16) and S.peak(ref, "chat", 16) == (16, True)
    # at 8 they show it is not, since it served more at 16; and at its highest rung nothing shows it
    assert (low["saturated"], low["peak"]) == (False, 16) and S.peak(ref, "chat", 32) == (16, False)
    assert S.peak(cost_arm("compiled", {8: [1.0] * 6, 16: [1.5] * 6}), "chat", 16) == (16, False)
    assert S.peak(cost_arm("compiled", {16: [1.5] * 6}), "chat", 16) == (16, False)
    assert S._at(c, "compiled") == "at concurrency 16, where compiled is saturated"
    assert S._at(low, "compiled") == ("at concurrency 8, where compiled is not shown saturated (it served more tokens "
                                      "a second at concurrency 16)")
    # the reps both have are the ones compared; five have no interval; a load one side didn't serve has no cost
    five = S.serving_cost(ref, cost_arm("boundary-rows", {16: [1.2, 1.25, 1.15, 1.2, 1.25]}), "chat")
    assert five["reps"] == 5 and five["value"] == pytest.approx(0.25) and "ci95" not in five
    one = S.serving_cost(ref, cost_arm("boundary-rows", {16: [1.2]}), "chat")
    assert one["reps"] == 1 and one["value"] == pytest.approx(1.5 / 1.2 - 1) and "ci95" not in one
    assert S.serving_cost(ref, cost_arm("boundary-rows", {16: [1.0]}, load="long"), "chat") is None
    # the estimator is the kit's own, read from the tree's one copy of it
    paired = S.kit()[0].paired
    assert paired.__module__ == "verity_bench.overhead" and paired([1 / 1.2], [1 / 1.5])["gamma"] == pytest.approx(0.25)


def test_the_cost_within_the_slo_reads_each_sides_best_goodput_and_is_not_the_cost():
    """Within the SLO each arm's best is its own rung: the reference's goodput peaks at concurrency 16 and the arm's at
    8. Both serve the same tokens a second at every rung, so the cost is nothing while a quarter more is needed
    within the SLO: the two are one number only where every request meets the SLO on both sides."""
    rate = {8: [2.0] * 6, 16: [2.0] * 6}
    ref = cost_arm("compiled", rate, good={8: [1.0] * 6, 16: [1.5, 1.6, 1.4, 1.5, 1.55, 1.45]})
    ours = cost_arm("boundary-rows", rate, good={8: [1.2, 1.25, 1.15, 1.2, 1.25, 1.1], 16: [1.0, 0.9, 1.1, 1.0, 0.9, 1.1]})
    assert S.best_pass(ref, "chat")[0] == 16 and S.best_pass(ours, "chat")[0] == 8 and S.best_pass(ref, "long") is None
    c = S.serving_cost(ref, ours, "chat")
    assert c["value"] == pytest.approx(0.0) and c["rung"] == 8          # the lowest of the rungs where stock is fastest
    w = c["within"]
    ratios = sorted(x / y for x, y in zip([1.5, 1.6, 1.4, 1.5, 1.55, 1.45], [1.2, 1.25, 1.15, 1.2, 1.25, 1.1]))
    assert w["value"] == pytest.approx(0.25) and w["ci95"] == [pytest.approx(ratios[0] - 1), pytest.approx(ratios[5] - 1)]
    assert (w["reps"], w["reference"]["concurrency"], w["arm"]["concurrency"], w["arm"]["sent"]) == (6, 16, 8, 1200)
    assert S.within_slo(ref, ours, "long") is None
    # where every request meets the SLO on both sides the two are the same number
    all_met = S.serving_cost(cost_arm("compiled", {8: [2.0] * 6}), cost_arm("boundary-rows", {8: [1.9, 1.95, 1.9, 1.85, 1.9, 1.95]}), "chat")
    assert all_met["value"] == pytest.approx(all_met["within"]["value"]) and all_met["ci95"] == pytest.approx(all_met["within"]["ci95"])


def test_an_arm_that_meets_the_slo_for_no_request_has_a_cost_and_none_within_the_slo():
    ref = cost_arm("compiled", {8: [2.0, 2.0], 16: [3.0, 3.0]}, good={8: [1.0, 1.0], 16: [1.5, 1.5]})
    none = cost_arm("boundary-rows", {8: [1.0, 1.0], 16: [2.0, 2.0]}, good={8: [0.0, 0.0], 16: [0.0, 0.0]})
    c = S.serving_cost(ref, none, "chat")
    # the cost reads no SLO; with no goodput anywhere the arm's best pass within the SLO is its fastest
    assert c["value"] == pytest.approx(0.5) and c["within"]["value"] is None and c["within"]["arm"]["concurrency"] == 16
    r = S.cost_report(rowed([ref, none]), arm="boundary-rows", target=5.0)
    assert (r["value"], r["ci95"], r["verdict"]) == (50.0, None, "cannot tell")          # two reps have no interval
    assert ("no finite cost (boundary-rows met it for 0 of 400 requests at its best rung, compiled for 200 of 400)"
            in S.cost_text(r))
    # a rep with no goodput leaves no interval within the SLO
    some = S.within_slo(ref, cost_arm("boundary-rows", {8: [1.0, 1.0]}, good={8: [1.0, 0.0]}), "chat")
    assert some["value"] == pytest.approx(2.0) and "ci95" not in some
    # an arm that served no token has no finite cost, and its verdict cannot be told
    dead = S.serving_cost(ref, cost_arm("boundary-rows", {16: [0.0] * 6}), "chat")
    assert dead["value"] is None and S.verdict(dead, 0.05) == "cannot tell"
    r = S.cost_report(rowed([ref, cost_arm("boundary-rows", {16: [0.0] * 6})]), arm="boundary-rows", target=5.0)
    assert (r["value"], r["verdict"]) == (None, "cannot tell")
    assert "no finite cost: a side served no token at concurrency 16" in S.cost_text(r)


def test_a_loads_line_says_whether_the_references_rungs_show_it_saturated_and_no_verdict_reads_it():
    """A cost is the tax where stock is saturated only where stock's passes show that: a rung above with no more
    tokens a second. A job that serves one rung, or whose highest is its fastest, says it is not shown."""
    ours = cost_arm("boundary-rows", {8: [0.9] * 6, 16: [1.2, 1.25, 1.15, 1.2, 1.25, 1.1]})
    shown = S.cost_report([cost_arm("compiled", {8: [1.0] * 6, 16: [1.5] * 6, 32: [1.4] * 6}), ours],
                          arm="boundary-rows", target=50.0)
    top = S.cost_report([cost_arm("compiled", {8: [1.0] * 6, 16: [1.5] * 6}), ours], arm="boundary-rows", target=50.0)
    assert [(r["loads"][0]["rung"], r["loads"][0]["saturated"], r["loads"][0]["peak"]) for r in (shown, top)] == [
        (16, True, 16), (16, False, 16)]
    assert "over 6 reps, meets; at concurrency 16, where compiled is saturated: compiled 225.0 output" in S.cost_text(shown)
    assert ("over 6 reps, meets; at concurrency 16, where compiled is not shown saturated (it served no rung above): "
            "compiled 225.0 output") in S.cost_text(top)
    # the cost, its interval and its verdict are the same either way: the line says what the passes show
    same = ("value", "ci95", "verdict")
    assert {k: shown["loads"][0][k] for k in same} == {k: top["loads"][0][k] for k in same}
    assert "where compiled serves the most output tokens a second" in S.cost_text(top).splitlines()[2]


def test_the_verdict_is_one_of_three_and_the_noise_floor_widens_the_interval():
    cost = {"value": 0.03, "ci95": [0.02, 0.04]}
    assert S.verdict(cost, None) is None
    assert [S.verdict(cost, t) for t in (0.05, 0.04, 0.03, 0.02, 0.01)] == [
        "meets", "meets", "cannot tell", "cannot tell", "misses"]
    assert [S.verdict(cost, t, 0.015) for t in (0.06, 0.05, 0.01, 0.004)] == ["meets", "cannot tell", "cannot tell", "misses"]
    assert S.verdict({k: v for k, v in cost.items() if k != "ci95"}, 0.05) == "cannot tell"


def both(name: str, chat: list[float], long: list[float]) -> dict:
    """An arm's metrics with chat served at concurrency 8 and long context at 4: six reps, each list twice over."""
    a = cost_arm(name, {8: chat * 2})
    a["loads"] |= cost_arm(name, {4: long * 2}, load="long")["loads"]
    return a


#: what an arm's directory says of the load that repeats a prefix: its plan's counts and its file's digest
PLANNED = {"prefix": S.planned(S.trace_lines(S.PAIRS)) | {"sha256": "5" * 64}}
#: the same of a table of planned requests that arrive over a thousand seconds, a quarter of their prompt tokens repeated
ARRIVE = {"prefix": {"requests": 10, "prompt_tokens": 1000, "repeated_tokens": 250, "span_s": 1000.0, "sha256": "8" * 64,
                     "table_sha256": "9" * 64}}


def repeats(a: dict, rate: list[float], cached: float | None = None, ttft: float | None = None,
            gap: float | None = None) -> dict:
    """``a`` with the load that repeats a prefix served too, one request in flight: a rep each value of ``rate``, the
    requests a second it served, 4 a rep; the share of the prompt tokens its prefix cache served; its time to the
    first token and its 99th-percentile gap between tokens."""
    served = cost_arm(a["arm"], {1: rate}, load="prefix")["loads"]
    for r in served["prefix@1"]:
        r |= {"sent": 4, "ok": 4, "prompt_cached_share": cached, "ttft_p50_s": ttft, "itl_p99_s": gap}
    return a | {"loads": a["loads"] | served, "planned": PLANNED}


def arrives(a: dict, tok: list[float], spans: list[float], cached: float, ttft: list[float], gap: float) -> dict:
    """``a`` with a table of planned requests (`ARRIVE`) served as they arrive: a rep each value of ``spans``, the
    seconds its pass took, of ``tok``, its output tokens a second, and of ``ttft``, its time to the first token; 10
    requests a rep; the share of the prompt tokens its prefix cache served; its 99th-percentile gap between tokens."""
    served = cost_arm(a["arm"], {S.ARRIVING: [1.0] * len(spans)}, load="prefix")["loads"]
    for r, span, rate, first in zip(served["prefix@0"], spans, tok, ttft):
        r |= {"sent": 10, "ok": 10, "output_tok_per_s": rate, "span_s": span, "prompt_cached_share": cached,
              "ttft_p50_s": first, "itl_p99_s": gap}
    return a | {"loads": a["loads"] | served, "planned": ARRIVE}


def arriving(last: float = 1100.0) -> list[dict]:
    """Stock, stock with its prefix cache off and the launch line, each with the number's loads and a table of planned
    requests served as they arrive over two reps: stock's first pass ends 10 s after the last arrival, and its second
    ``last`` seconds after the first one."""
    return [arrives(both("compiled", [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]), [100.0, 100.0], [1010.0, last], 0.25, [0.5, 0.5], 0.05),
            arrives(both("no-prefix-cache", [2.0, 1.99, 2.0], [1.0, 0.99, 1.0]), [99.5, 100.0], [1030.0] * 2, 0.0, [0.55, 0.5], 0.05),
            arrives(both("two-stage", [1.95, 1.96, 1.97], [0.94, 0.935, 0.93]), [99.0, 98.0], [1020.0] * 2, 0.0, [0.6, 0.65], 0.055)]


def rowed(arms: list[dict]) -> list[dict]:
    """``arms`` with the load that repeats a prefix served by each, so a report of them gives the number: stock at
    twice the others' rate."""
    return [repeats(a, [0.02 if a["arm"] == "compiled" else 0.01] * 6) for a in arms]


def cost_arms() -> list[dict]:
    """Stock, stock again, and one arm for each change up to the two-stage serve's launch line. Stock, stock with its
    prefix cache off and the launch line served the load that repeats a prefix too."""
    return [repeats(both("compiled", [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]), [0.02] * 6, 0.4998, 45.0, 0.05),
            both("compiled-again", [1.98, 2.0, 2.02], [1.0, 1.0, 1.0]),
            both("record", [1.98, 1.99, 2.0], [0.99, 0.985, 0.98]),
            repeats(both("no-prefix-cache", [2.0, 1.99, 2.0], [1.0, 0.99, 1.0]), [0.0104] * 6, 0.0, 87.3, 0.05),
            both("boundary-rows", [1.96, 1.97, 1.98], [0.96, 0.955, 0.95]),
            both("canonical-topk", [2.0, 2.0, 1.99], [0.98, 0.975, 0.97]),
            repeats(both("two-stage", [1.95, 1.96, 1.97], [0.94, 0.935, 0.93]), [0.0101, 0.01, 0.0102] * 2, 0.0, 90.0, 0.051)]


#: an arm with every change the model's target has a part for and no other: stock with the tick, the canonical top-k
#: and the edge snapshot
WHOLE = "whole"


@pytest.fixture
def whole(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """`cost_arms` with the measured arm's passes under an arm that has every part of the target on and no change the
    target has no part for."""
    monkeypatch.setitem(S.ARMS, WHOLE, replace(S.ARMS["compiled"], parent="compiled", what="every part of the target",
                                               env={"VERITY_DSA_TOPK": "canonical"}, tick=True, edges=True))
    return [a | {"arm": WHOLE, "parent": "compiled"} if a["arm"] == "two-stage" else a for a in cost_arms()]


def test_a_models_target_is_the_sum_of_its_parts_each_of_changes_an_arm_can_have_on():
    glm = S.MODELS["glm53"]
    assert [(p.name, p.points, p.changes, p.basis) for p in glm.target_parts] == [
        ("the certified tick", 1.63, ("admission on the certified tick",), "modelled"),
        ("the tie order", 0.25, ("the canonical top-k",), "modelled"),
        ("the snapshot bundle", 0.17, ("the edge snapshot",), "modelled")]
    assert glm.target == 2.05 and all(not m.target_parts for m in S.MODELS.values() if m.target is None)
    with pytest.raises(ValueError, match="the target's parts sum to 1.88, and the target is 2.05"):
        replace(glm, target_parts=glm.target_parts[:2])
    with pytest.raises(ValueError, match="names no change of CHANGES: the tick"):
        replace(glm, target_parts=(S.Part("all", 2.05, ("the tick",)),))
    # an arm's changes are read from its data, and an arm this file doesn't define has none
    assert S.changes_of("compiled") == S.changes_of("compiled-again") == S.changes_of("no such arm") == []
    assert S.changes_of("two-stage") == ["the step record", "prefix caching off", "the two-stage boundary commitment",
                                         "the history tables", "the canonical top-k"]
    # the cache goes off with the history tables, not with the boundary commitment
    assert S.changes_of("boundary-rows") == ["the step record", "the two-stage boundary commitment"]
    assert S.changes_of("tick") == ["the step record", "admission on the certified tick"]
    assert S.changes_of("boundary")[-1] == "the edge snapshot"
    # the launch line leaves off the tick and the edge snapshot, which the snapshot bundle is of
    assert [(p["name"], p["points"], p["off"]) for p in S.target_parts(glm, "two-stage")] == [
        ("the certified tick", 1.63, ["admission on the certified tick"]), ("the tie order", 0.25, []),
        ("the snapshot bundle", 0.17, ["the edge snapshot"])]
    assert all(p["off"] == list(p["changes"]) for p in S.target_parts(glm, "no such arm"))
    # and it turns on four changes the target has no part for: the cost model keeps the prefix cache on
    assert S.not_in_target(glm, "two-stage") == ["the step record", "prefix caching off",
                                                 "the two-stage boundary commitment", "the history tables"]
    assert S.not_in_target(glm, "no-prefix-cache") == ["prefix caching off"]
    assert S.not_in_target(glm, "canonical-topk") == S.not_in_target(glm, "compiled") == []


def test_a_cost_has_a_verdict_against_the_models_target_only_where_its_arm_is_of_the_targets_changes(
        whole: list[dict], monkeypatch: pytest.MonkeyPatch):
    """The target counts changes the launch line leaves off and has no part for changes it turns on, so its cost
    neither meets nor misses it: the verdict is a fourth value, which is not the "cannot tell" of an interval that
    straddles the target."""
    r = S.cost_report(cost_arms())
    assert r["verdict"] == S.NOT_COMPARABLE == "not comparable" and (r["target"], r["target_basis"]) == (2.05, "modelled")
    assert [c["verdict"] for c in r["loads"]] == ["not comparable", "not comparable"]
    assert [(p["name"], p["off"]) for p in r["target_parts"] if p["off"]] == [
        ("the certified tick", ["admission on the certified tick"]), ("the snapshot bundle", ["the edge snapshot"])]
    assert r["not_in_target"] == ["the step record", "prefix caching off", "the two-stage boundary commitment",
                                  "the history tables"]
    text = S.cost_text(r)
    assert text.splitlines()[0].endswith("target ≤ 2.05% (modelled)   verdict: not comparable")
    assert ("  the target is the sum of its parts: the certified tick 1.63% (modelled), the tie order 0.25% (modelled), "
            "the snapshot bundle 0.17% (modelled; the pause to copy and hash the cache at each edge and the carried "
            "pages' hash, with the prefix cache on)\n"
            "  not comparable: two-stage leaves off admission on the certified tick (the certified tick, 1.63%) and "
            "the edge snapshot (the snapshot bundle, 0.17%), which the target counts, and turns on the step record, "
            "prefix caching off, the two-stage boundary commitment and the history tables, which the target has no "
            "part for, so no cost of it meets or misses the target\n") in text
    # no target is made of the parts that are on
    assert "0.42" not in text and "0.25%)" not in text.split("\n\n")[0].replace("0.25% (modelled)", "")
    # one load alone is not comparable either, and neither is a result's entry
    alone = S.cost_report([a for a in cost_arms() if a["arm"] in ("compiled", "two-stage")])
    assert alone["verdict"] == "not comparable"
    # the row on a repeated prefix has no target, so its entry has no verdict of any kind
    assert {m["name"]: m["verdict"] for m in S.result_of(r)[3]} == {
        "chat": "not comparable", "long": "not comparable", "prefix": None, "vllm/serving-cost-glm53": "not comparable"}
    # an arm with every part on and nothing else has one of the three, the largest load's: here it misses
    r = S.cost_report(whole, arm=WHOLE)
    assert not any(p["off"] for p in r["target_parts"]) and r["not_in_target"] == []
    assert "not comparable" not in S.cost_text(r)
    assert [c["verdict"] for c in r["loads"]] == ["cannot tell", "misses"] and r["verdict"] == "misses"
    assert "the target is the sum of its parts: the certified tick 1.63% (modelled)" in S.cost_text(r)
    # every part on is not enough: the launch line with the tick and the edge snapshot still has four changes on that
    # the target has no part for
    monkeypatch.setitem(S.ARMS, "wider", replace(S.ARMS["two-stage"], tick=True, edges=True))
    wider = S.cost_report([a | {"arm": "wider"} if a["arm"] == "two-stage" else a for a in cost_arms()], arm="wider")
    assert not any(p["off"] for p in wider["target_parts"]) and len(wider["not_in_target"]) == 4
    assert wider["verdict"] == "not comparable" and {c["verdict"] for c in wider["loads"]} == {"not comparable"}
    assert ("  not comparable: wider turns on the step record, prefix caching off, the two-stage boundary commitment "
            "and the history tables, which the target has no part for, so no cost of it meets or misses the target\n"
            ) in S.cost_text(wider)
    # a target that is given has no parts: the cost is compared with it as it is
    given = S.cost_report(cost_arms(), target=10.0)
    assert (given["verdict"], given["target_parts"]) == ("meets", []) and "sum of its parts" not in S.cost_text(given)
    assert all(x["in_target"] is None for x in given["arms"]) and "| share of the cost |\n" in S.cost_text(given)


def test_each_arms_row_says_what_its_step_turns_on_and_the_targets_part_for_it():
    r = S.cost_report(cost_arms())
    rows = {x["arm"]: x["in_target"] for x in r["arms"]}
    # the tie order alone is one part, whole: its points are beside the step that measures it
    assert rows["canonical-topk"] == [{"changes": ["the canonical top-k"], "part": "the tie order", "points": 0.25,
                                       "whole": True}]
    # the cost model has no part for the step record, prefix caching off, the two-stage boundary commitment or the
    # history tables
    assert rows["record"] == [{"changes": ["the step record"], "part": None, "points": None, "whole": False}]
    assert rows["no-prefix-cache"] == [{"changes": ["prefix caching off"], "part": None, "points": None, "whole": False}]
    assert rows["boundary-rows"] == [{"changes": ["the two-stage boundary commitment"], "part": None, "points": None,
                                      "whole": False}]
    assert rows["two-stage"] == [
        {"changes": ["the canonical top-k"], "part": "the tie order", "points": 0.25, "whole": True},
        {"changes": ["prefix caching off", "the history tables"], "part": None, "points": None, "whole": False}]
    text = S.cost_text(r)
    assert ("| arm | cost over compiled | over | cost over that arm | share of the cost | what the step turns on, and "
            "the target's part for it |\n|---|---:|---|---:|---:|---|\n"
            "| record | 1.52% [1.01%, 2.04%] | compiled | 1.52% [1.01%, 2.04%] | 22% | the step record: not in the "
            "model |\n"
            "| no-prefix-cache | 0.00% [0.00%, 1.01%] | compiled | 0.00% [0.00%, 1.01%] | — | prefix caching off: not "
            "in the model |\n") in text
    assert "| 3.14% [3.12%, 3.16%] | 45% | the two-stage boundary commitment: not in the model |\n" in text
    assert "| canonical-topk | 2.56% [2.04%, 3.09%] | compiled | 2.56% [2.04%, 3.09%] | — | the canonical top-k: the " \
           "tie order, 0.25% |\n" in text
    assert "| 31% | the canonical top-k: the tie order, 0.25%; prefix caching off, the history tables: not in the " \
           "model |\n" in text
    # a step that turns on none of our changes has nothing to say
    assert S.in_target(S.MODELS["glm53"], "eager", "compiled") == [] and S._in_target([]) == "—"


def test_the_cost_report_is_the_largest_loads_cost_with_the_consoles_keys_first(whole: list[dict]):
    arms = whole
    r = S.cost_report(arms, arm=WHOLE)
    assert list(r)[:6] == ["value", "unit", "commit", "run", "prefix", "label"] and r["label"] == "no repeated prefix"
    assert (r["unit"], r["commit"], r["run"], r["metric"], r["better"]) == (
        "percent of stock's serving compute", "c" * 40, "r-1", "serving_cost", "lower")
    # long context costs more than chat, so it is the number; the model's target applies where none is given
    assert [(c["load"], c["value"], c["verdict"]) for c in r["loads"]] == [("chat", 2.04, "cannot tell"), ("long", 6.95, "misses")]
    assert (r["load"], r["value"], r["target"], r["model"], r["arm"], r["reference"], r["not_run"]) == (
        "long", 6.95, S.MODELS["glm53"].target, "glm-5.3", WHOLE, "compiled", [])
    # the number is the largest, so it misses where one load misses, whatever the other does
    assert r["ci95"][0] < r["value"] < r["ci95"][1] and r["verdict"] == "misses"
    # each load is read at the rung where stock serves the most tokens a second, an engine on the model's GPUs; these
    # arms served one rung a load, so nothing shows stock saturated there
    assert [c["rung"] for c in r["loads"]] == [8, 4] and (r["gpus"], r["vllm"]) == (8, "0.1+gd9105ea8")
    assert [(c["saturated"], c["peak"]) for c in r["loads"]] == [(False, 8), (False, 4)]
    # the cost within the SLO is beside each load's cost, with no verdict: each load's goodput counts against its own
    # bounds, since long context's first token takes seconds on any engine
    assert [c["within"]["latency_target_ms"] for c in r["loads"]] == [{"ttft": 2000, "tpot": 80}, {"tpot": 80}]
    assert [c["within"]["value"] for c in r["loads"]] == [2.04, 6.95] and "verdict" not in r["loads"][0]["within"]
    assert "latency_target_ms" not in r
    # stock again is the noise floor, not a change; each other arm's cost is over stock and over the arm above it
    assert r["noise_floor"] == {"arm": "compiled-again", "value": 0.0, "ci95": [0.0, 0.0]}
    assert [c["noise_floor"] for c in r["loads"]] == [0.0, 0.0]
    rows = {x["arm"]: x for x in r["arms"]}
    assert list(rows) == ["record", "no-prefix-cache", "boundary-rows", "canonical-topk", WHOLE]
    assert [rows[a]["over"] for a in rows] == ["compiled", "compiled", "record", "compiled", "compiled"]
    assert rows["record"]["value"] == rows["record"]["step"]["value"] == 1.52
    assert rows["boundary-rows"]["value"] == 4.71 and rows["boundary-rows"]["step"]["value"] == 3.14
    # this arm is over stock itself, so it is the whole of its cost and no other arm is a step of it
    assert rows[WHOLE]["value"] == rows[WHOLE]["step"]["value"] == 6.95
    assert [rows[a]["share"] for a in rows] == [None, None, None, None, 1.0]
    assert rows["canonical-topk"]["value"] == rows["canonical-topk"]["step"]["value"] == 2.56
    assert r["no_arm"] == list(S.NO_ARM) and r["commitment"] is None and json.loads(json.dumps(r)) == r
    # the launch line's arms from the measured one up to stock split the whole: the record's step, the boundary
    # commitment's and the launch line's last; prefix caching off and the canonical top-k alone are beside that chain
    ours = S.cost_report(cost_arms())
    steps = {x["arm"]: x for x in ours["arms"]}
    assert [steps[a]["over"] for a in steps] == ["compiled", "compiled", "record", "compiled", "boundary-rows"]
    assert steps["two-stage"]["value"] == 6.95 and steps["two-stage"]["step"]["value"] == 2.14
    assert [steps[a]["share"] for a in steps] == [pytest.approx(0.219, abs=1e-3), None, pytest.approx(0.452, abs=1e-3),
                                                  None, pytest.approx(0.308, abs=1e-3)]
    table = S.cost_text(ours)
    assert "| two-stage | 6.95% " in table and "| boundary-rows | 2.14% " in table and "| record | 3.14% " in table
    lines = S.cost_text(r).splitlines()
    assert lines[0] == ("serving cost, no repeated prefix, glm-5.3, whole over compiled: 6.95% [6.38%, 7.53%]   "
                        "target ≤ 2.05% (modelled)   verdict: misses")
    # the row on a repeated prefix is directly under the number, before anything else of the report
    assert lines[1].startswith("  on a repeated prefix, scaled down, ") and lines[4].startswith("    no-prefix-cache, ")
    assert lines[5:7] == [
        "  the largest of the costs of its loads (chat, long), load long's",
        "  in percent of stock's serving compute, lower is better: how much more GPU time whole needs a token than "
        "compiled, an engine on 8 GPUs, both serving the same tokens at the concurrency, of those both served, where "
        "compiled serves the most output tokens a second"]
    assert (r["id"], r["target_basis"]) == ("vllm/serving-cost-glm53", "modelled")
    text = "\n".join(lines)
    # each load's line: its cost, then both sides' output and prompt tokens a second at its rung and the tokens a request
    assert ("  load chat, ShareGPT's first turns, each decoding its recorded answer's length (vLLM's sharegpt dataset): "
            "2.04% [1.52%, 2.56%] over 6 reps, cannot tell; at concurrency 8, where compiled is not shown saturated (it "
            "served no rung above): compiled 300.0 output and 512.0 prompt "
            "tokens/s, whole 294.0 output and 501.8 prompt tokens/s; 256 prompt and 150 output tokens a request\n"
            "    within the latency target (TTFT ≤ 2000 ms, TPOT ≤ 80 ms), which no target or verdict is of: 2.04% "
            "[1.52%, 2.56%] over 6 reps; best goodput 2 req/s at concurrency 8 (compiled), 1.96 req/s at concurrency 8 "
            "(whole)\n") in text
    assert ("  load long, 8,192 random tokens in and 128 out (vLLM's random dataset at one long length): 6.95% [6.38%, "
            "7.53%] over 6 reps, misses; at concurrency 4, where compiled is not shown saturated (it served no rung "
            "above): compiled 150.0 output and 256.0 prompt tokens/s, ") in text
    assert "    within the latency target (TPOT ≤ 80 ms), which no target or verdict is of: 6.95% [6.38%, 7.53%]" in text
    assert "noise floor (compiled-again over compiled, two server lifetimes of one engine, load long): 0.00%" in text
    assert "| whole | 6.95% [6.38%, 7.53%] | compiled | 6.95% [6.38%, 7.53%] | 100% |" in text
    assert "changes with no arm" in text and "'s boundary commitment" not in text
    # a target given replaces the model's, and without stock again the output says the floor was not measured
    arms = cost_arms()
    two = S.cost_report([arms[0], arms[-1]], target=10.0)
    assert (two["target"], two["target_basis"], two["verdict"], two["noise_floor"]) == (10.0, "given", "meets", None)
    assert two["prefix"]["cache_off"] is None                       # stock with its prefix cache off did not serve
    assert [(x["arm"], x["over"], x["share"]) for x in two["arms"]] == [("two-stage", "compiled", 1.0)]
    assert "noise floor: not measured" in S.cost_text(two) and "target ≤ 10.00% (given)   verdict: meets" in S.cost_text(two)
    assert S.cost_report(rowed([arms[0], arms[2]])[:2], arm="record", target=0.1)["verdict"] == "misses"


@pytest.mark.usefixtures("whole")
def test_one_load_alone_is_named_and_never_meets_the_target_for_the_number():
    """The number is the largest of both loads' costs: a line from one load says which, and that it is not the number
    yet; it can miss on one load, and cannot meet."""
    chat = [cost_arm(name, {8: gs * 2}) for name, gs in (("compiled", [2.0, 2.0, 2.0]), (WHOLE, [1.95, 1.96, 1.97]))]
    # a job of its own served the load that repeats a prefix, whose row every line of the number stands over
    row = rowed([cost_arm(name, {}) for name in ("compiled", WHOLE)])
    r = S.cost_report([*chat, *row], arm=WHOLE)
    assert (r["load"], r["value"], r["not_run"], r["verdict"]) == ("chat", 2.04, ["long"], "cannot tell")
    assert S.cost_report([*chat, *row], arm=WHOLE, target=3.0)["loads"][0]["verdict"] == "meets"       # chat's own cost is within 3%,
    assert S.cost_report([*chat, *row], arm=WHOLE, target=3.0)["verdict"] == "cannot tell"             # and the number still waits for long
    lines = S.cost_text(r).splitlines()
    assert lines[0] == ("serving cost, no repeated prefix, glm-5.3, whole over compiled, load chat alone: 2.04% [1.52%, "
                        "2.56%]   target ≤ 2.05% (modelled)   verdict: cannot tell")
    assert lines[1].startswith("  on a repeated prefix, scaled down, ") and lines[4] == (
        "  not the number yet: it is the largest of the costs of its loads (chat, long), and load long has not run")
    long = [cost_arm(name, {4: gs * 2}, load="long")
            for name, gs in (("compiled", [1.0, 1.0, 1.0]), (WHOLE, [0.94, 0.935, 0.93]))]
    r = S.cost_report([*long, *row], arm=WHOLE)
    assert (r["load"], r["value"], r["not_run"], r["verdict"]) == ("long", 6.95, ["chat"], "misses")
    assert ", load long alone: 6.95% [" in S.cost_text(r) and "and load chat has not run" in S.cost_text(r)
    # two jobs' directories of an arm, each a load of its own, are one arm: the number is then the larger of the two,
    # under the commit and run of the job that served its load, each load's job named
    commits = {"spans": 600, "finished": 600, "failed": 0, "tables": 600}
    chat[1]["boundary_commits"] = commits
    long = [a | {"run_id": "r-2", "commit": "d" * 40, "boundary_commits": commits | {"spans": 72, "finished": 80}} for a in long]
    row = [a | {"run_id": "r-3"} for a in row]
    r = S.cost_report([*chat, *long, *row], arm=WHOLE)
    assert [(c["load"], c["value"], c["run"]) for c in r["loads"]] == [("chat", 2.04, "r-1"), ("long", 6.95, "r-2")]
    assert (r["prefix"]["run"], r["prefix"]["commit"]) == ("r-3", "c" * 40)
    assert (r["load"], r["not_run"], r["verdict"], r["run"], r["commit"]) == ("long", [], "misses", "r-2", "d" * 40)
    assert r["commitment"] == {"finished": 680, "committed": 672, "failed": 0, "tables": 1200, "tables_asked": False,
                               "window_s": None}
    text = S.cost_text(r)
    assert text.splitlines()[4] == "  the largest of the costs of its loads (chat, long), load long's"
    assert (f"load chat: commit {'c' * 40}, run r-1\nload long: commit {'d' * 40}, run r-2\n"
            f"load prefix: commit {'c' * 40}, run r-3\nvLLM 0.1+gd9105ea8\n") in text
    assert S.cost_report([*long, *chat], arm=WHOLE)["loads"][0]["load"] == "chat"          # in the loads' order, whichever came first
    for twice, why in (([*chat, *chat], "two directories of arm compiled both served chat: give one of them"),
                       ([*chat, *(a | {"vllm": "0.2"} for a in long)], "two directories of arm compiled differ in vllm")):
        with pytest.raises(SystemExit, match=why):
            S.cost_report(twice)
    # passes of another size (--pass-prompts on one side) are other requests: no cost across them
    def sized(a: dict, n: int) -> dict:
        return a | {"loads": {p: [r | {"sent": n} for r in reps] for p, reps in a["loads"].items()}}

    longer = [sized(a, 1024) if a["arm"] == "compiled" else a for a in chat]
    with pytest.raises(SystemExit, match=r"compiled and whole sent passes of different sizes at load chat, at "
                                         r"concurrency \d+ \(1024 and 200 prompts in rep 1; --pass-prompts\)"):
        S.cost_report([*longer, *long, *row], arm=WHOLE)
    same = S.cost_report([*(sized(a, 1024) for a in chat), *long, *row], arm=WHOLE)
    assert [c["value"] for c in same["loads"]] == [2.04, 6.95]


def test_the_row_under_the_number_is_the_cost_on_a_repeated_prefix_with_no_target_and_no_verdict():
    """The number's loads repeat no prefix, so it says nothing of what serving without the prefix cache costs: the row
    under it does, on the load that repeats one, for the arm and for stock with its prefix cache off alone."""
    r = S.cost_report(cost_arms())
    row = r["prefix"]
    assert (row["load"], row["label"], row["scaled_down"], row["rung"], row["reps"]) == (
        "prefix", "on a repeated prefix", True, 1, 6)
    assert "target" not in row and "verdict" not in row
    # the cost: with one request in flight, stock's output tokens a second over the arm's, less one
    assert (row["value"], row["ci95"]) == (98.02, [96.08, 100.0])
    assert (row["reference"]["output_tok_per_s"], row["arm"]["output_tok_per_s"]) == (pytest.approx(3.0), pytest.approx(1.515))
    # beside it, how much longer the first token and the 99th-percentile gap between tokens take
    assert (row["ttft"], row["itl_p99"]) == ({"value": 100.0, "ci95": [100.0, 100.0]}, {"value": 2.0, "ci95": [2.0, 2.0]})
    slow = [{"ttft_p50_s": x} for x in (3.0, 3.0, 3.3, 3.0, 2.7, 3.0)]
    assert S.latency_cost([{"ttft_p50_s": 2.0}] * 6, slow, "ttft_p50_s") == {
        "value": pytest.approx(0.5), "ci95": [pytest.approx(0.35), pytest.approx(0.65)]}
    assert S.latency_cost([{"ttft_p50_s": 2.0}, {}], slow[:2], "ttft_p50_s") == {"value": None}
    # the share planned to repeat is the load's own; the share each side's prefix cache served is its engine's count
    assert row["planned"] == PLANNED["prefix"] | {"repeated_share": 49.98}
    assert [(row[k]["arm"], row[k]["cached_share"], row[k]["served_all"]) for k in ("reference", "arm")] == [
        ("compiled", 49.98, True), ("two-stage", 0.0, True)]
    # stock with its prefix cache off, over stock: what the cache alone is worth on this load
    off = row["cache_off"]
    assert (off["arm"]["arm"], off["value"], off["ttft"]["value"], off["itl_p99"]["value"], off["arm"]["cached_share"]) == (
        "no-prefix-cache", 92.31, 94.0, 0.0, 0.0)
    assert S.cost_text(r).splitlines()[1:5] == [
        "  on a repeated prefix, scaled down, two prompts near the context limit, each sent again with another ending, "
        "of synthetic token ids (vLLM's timed_trace dataset); no target, and it judges nothing: 98.02% [96.08%, 100.00%] "
        "over 6 reps, one request in flight, where the cost is compiled's output tokens a second over two-stage's, less "
        "one: compiled 3.0 output tokens/s, two-stage 1.5",
        "    4 requests, 49.98% of their prompt tokens planned to repeat; the prefix cache served 49.98% (compiled) and "
        "0.00% (two-stage); every request served",
        "    two-stage over compiled: the time to the first token 100.00% [100.00%, 100.00%] longer, the 99th-percentile "
        "gap between tokens 2.00% [2.00%, 2.00%] longer",
        "    no-prefix-cache, compiled with its prefix cache off: 92.31% [92.31%, 92.31%] over 6 reps, 1.6 output "
        "tokens/s; the time to the first token 94.00% [94.00%, 94.00%] longer, the 99th-percentile gap between tokens "
        "0.00% [0.00%, 0.00%] longer; its prefix cache served 0.00%; every request served"]
    # the number and its verdict read nothing of the row: the same costs with another row under them
    arms = cost_arms()
    for x in arms[-1]["loads"]["prefix@1"]:
        x["output_tok_per_s"] /= 4
    worse = S.cost_report(arms)
    assert worse["prefix"]["value"] > 600 and {k: worse[k] for k in ("value", "ci95", "verdict", "loads")} == {
        k: r[k] for k in ("value", "ci95", "verdict", "loads")}
    # a rep that left a request unfinished and a rep whose engine's counters were not read are both said
    arms = cost_arms()
    arms[-1]["loads"]["prefix@1"][2]["ok"] = 3
    del arms[0]["loads"]["prefix@1"][0]["prompt_cached_share"]
    text = S.cost_text(S.cost_report(arms))
    assert ("the prefix cache served not recorded (compiled) and 0.00% (two-stage); two-stage did not serve every "
            "request\n") in text
    # the arms of a row served one plan: the same requests, from the same file
    arms = cost_arms()
    arms[-1]["planned"] = {"prefix": PLANNED["prefix"] | {"sha256": "6" * 64}}
    with pytest.raises(SystemExit, match="compiled and two-stage served load prefix from different plans"):
        S.cost_report(arms)
    arms = cost_arms()
    del arms[3]["planned"]
    with pytest.raises(SystemExit, match="compiled and no-prefix-cache served load prefix from different plans, or "
                                         "no-prefix-cache's directory names none"):
        S.cost_report(arms)
    arms = cost_arms()
    del arms[0]["planned"]
    with pytest.raises(SystemExit, match="compiled's directory that served load prefix names no plan"):
        S.cost_report(arms)
    # a row of another plan than the four requests is not labelled scaled down, and its share is its own
    other = {"prefix": {"requests": 4, "prompt_tokens": 1000000, "repeated_tokens": 250000, "sha256": "7" * 64}}
    longer = S.cost_report([a | {"planned": other} if "planned" in a else a for a in cost_arms()])["prefix"]
    assert (longer["scaled_down"], longer["planned"]["repeated_share"]) == (False, 25.0)
    assert S.prefix_text(longer)[0].startswith("  on a repeated prefix, two prompts near")
    # with the reference's cache off as the arm, no second line repeats it
    assert S.cost_report(cost_arms(), arm="no-prefix-cache")["prefix"]["cache_off"] is None


def test_the_row_of_requests_sent_as_they_arrive_says_whether_stock_kept_up_and_which_figure_is_the_cost():
    """A table's requests arrive at their own times. Where stock keeps up with them, each arm's tokens a second
    follows the arrivals and is no cost: the row's cost is the two latencies. Where stock does not, the row is read as
    the number is. It prints tokens a second and both latencies either way, each as its readings, one a rep, with no
    mean and no interval."""
    r = S.cost_report(arriving())
    row = r["prefix"]
    assert (row["load"], row["what"], row["scaled_down"], row["rung"], row["reps"]) == ("prefix", S.GIVEN, False, S.ARRIVING, 2)
    assert "target" not in row and "verdict" not in row
    assert row["planned"] == ARRIVE["prefix"] | {"repeated_share": 25.0}
    # stock served all that arrived and its longest pass ended 100 s after the last arrival, a tenth of the 1,000 s
    # they arrive over: it kept up
    assert (row["reference"]["tail_s"], row["arm"]["tail_s"], row["kept_up"], row["why"], S.KEPT_UP) == (
        100.0, 20.0, True, "compiled kept up", 0.1)
    # so the cost is the two latencies; every figure is its two readings, and the row has no value
    assert (row["cost"], row["value"], row["ci95"]) == (S.LATENCY, None, None)
    assert (row["ttft"], row["itl_p99"]) == ({"value": None, "ci95": None, "readings": [20.0, 30.0]},
                                             {"value": None, "ci95": None, "readings": [10.0, 10.0]})
    assert row["tokens"] == {"value": None, "ci95": None, "readings": [1.01, 2.04]}
    assert [(row[k]["arm"], row[k]["cached_share"], row[k]["served_all"]) for k in ("reference", "arm")] == [
        ("compiled", 25.0, True), ("two-stage", 0.0, True)]
    off = row["cache_off"]
    assert (off["cost"], off["kept_up"], off["value"], off["tokens"]["readings"], off["ttft"]["readings"],
            off["arm"]["tail_s"]) == (S.LATENCY, True, None, [0.5, 0.0], [10.0, 0.0], 30.0)
    assert S.cost_text(r).splitlines()[1:6] == [
        "  on a repeated prefix, a table's planned requests, each sent at its arrival time, of synthetic token ids "
        "(vLLM's timed_trace dataset); no target, and it judges nothing: the time to the first token 20.00%, 30.00% "
        "longer, the 99th-percentile gap between tokens 10.00%, 10.00% longer, a reading a rep over 2 reps, as they "
        "arrive; compiled kept up, so the cost is the two latencies, two-stage's over compiled's, less one: at "
        "arrivals compiled keeps up with, tokens a second follows the arrivals and is not a cost",
        "    compiled kept up where it served all that arrived and each of its passes ended within a tenth of the time "
        "the requests arrive over after the last one arrived: they arrive over 1,000.0 s, and its longest pass ended "
        "100.0 s after the last arrived, two-stage's 20.0 s after",
        "    10 requests, 25.00% of their prompt tokens planned to repeat; the prefix cache served 25.00% (compiled) and "
        "0.00% (two-stage); compiled served all that arrived, two-stage served all that arrived",
        "    compiled's output tokens a second over two-stage's, less one, which is no cost here: 1.01%, 2.04%; "
        "compiled 100.0 output tokens/s, two-stage 98.5",
        "    no-prefix-cache, compiled with its prefix cache off: in output tokens a second, which is no cost here, "
        "0.50%, 0.00%, a reading a rep over 2 reps, 99.8 output tokens/s; the time to the first token 10.00%, 0.00% "
        "longer, the 99th-percentile gap between tokens 0.00%, 0.00% longer; its prefix cache served 0.00%; "
        "compiled served all that arrived, no-prefix-cache served all that arrived; its longest pass ended 30.0 s "
        "after the last arrival"]
    # the number is given with this row under it as with the four's, and reads nothing of it
    assert (r["value"], r["label"], r["prefix"]["cost"]) == (6.95, "no repeated prefix", S.LATENCY)
    # the run's result: the row's measurement has no value, its cost named and why, the readings beside it, and stands
    ok, _, what, ms = S.result_of(r)
    under = next(m for m in ms if m["name"] == "prefix")
    assert ok and what["loads"] == ["chat", "long", "prefix"] and ms[-1]["prefix"].items() <= under.items()
    assert (under["value"], under["ci95"], under["cost"], under["kept_up"], under["why"], under["ttft"]["readings"],
            under["itl_p99"]["readings"], under["tokens"]["readings"], under["scaled_down"], under["rung"],
            under["target"], under["verdict"]) == (
        None, None, S.LATENCY, True, "compiled kept up", [20.0, 30.0], [10.0, 10.0], [1.01, 2.04], False, 0, None, None)
    # a row whose cost is the latencies does not stand where a rep has no reading of one
    arms = arriving()
    arms[-1]["loads"]["prefix@0"][1]["ttft_p50_s"] = None
    gone = S.cost_report(arms)
    assert gone["prefix"]["ttft"]["readings"] == [20.0, None] and not S.result_of(gone)[0]
    assert "the time to the first token 20.00%, — longer" in S.cost_text(gone)
    # one pass of stock's ended later than a tenth of the span after the last arrival: it did not keep up, and the
    # row is read as the number is, by tokens a second, still as its readings
    late = S.cost_report(arriving(last=1100.1))
    row = late["prefix"]
    assert (row["kept_up"], row["cost"], row["value"], row["ci95"], row["tokens"]["readings"]) == (
        False, S.TOKENS, None, None, [1.01, 2.04])
    assert row["why"] == ("compiled's longest pass ended 100.1 s after the last arrival, over a tenth of the 1,000.0 s "
                          "the requests arrive over")
    assert S.cost_text(late).splitlines()[1:5] == [
        "  on a repeated prefix, a table's planned requests, each sent at its arrival time, of synthetic token ids "
        "(vLLM's timed_trace dataset); no target, and it judges nothing: 1.01%, 2.04%, a reading a rep over 2 reps, as "
        "they arrive; compiled's longest pass ended 100.1 s after the last arrival, over a tenth of the 1,000.0 s the "
        "requests arrive over, so the row is read as the number is: the cost is compiled's output tokens a second over "
        "two-stage's, less one: compiled 100.0 output tokens/s, two-stage 98.5",
        "    compiled kept up where it served all that arrived and each of its passes ended within a tenth of the time "
        "the requests arrive over after the last one arrived: they arrive over 1,000.0 s, and its longest pass ended "
        "100.1 s after the last arrived, two-stage's 20.0 s after",
        "    10 requests, 25.00% of their prompt tokens planned to repeat; the prefix cache served 25.00% (compiled) and "
        "0.00% (two-stage); compiled served all that arrived, two-stage served all that arrived",
        "    two-stage over compiled: the time to the first token 20.00%, 30.00% longer, the 99th-percentile gap "
        "between tokens 10.00%, 10.00% longer"]
    ok, _, _, ms = S.result_of(late)
    under = next(m for m in ms if m["name"] == "prefix")
    assert ok and (under["value"], under["cost"], under["kept_up"], under["tokens"]["readings"]) == (
        None, S.TOKENS, False, [1.01, 2.04])
    # stock that left a request of a rep unserved did not keep up either, however early its passes ended
    arms = arriving()
    arms[0]["loads"]["prefix@0"][1]["ok"] = 9
    short = S.cost_report(arms)["prefix"]
    assert (short["kept_up"], short["cost"], short["why"], short["reference"]["tail_s"]) == (
        False, S.TOKENS, "compiled did not serve all that arrived", 100.0)
    # an arm that left a request of a rep unserved is said by name, and stock still kept up
    arms = arriving()
    arms[-1]["loads"]["prefix@0"][1]["ok"] = 9
    assert S.cost_report(arms)["prefix"]["kept_up"]
    assert ("compiled served all that arrived, two-stage did not serve all that arrived\n") in S.cost_text(S.cost_report(arms))
    # the four requests' row says none of it: one request in flight, its cost in tokens a second, a value with its
    # interval over the reps
    four = S.cost_report(cost_arms())["prefix"]
    assert (four["cost"], four["kept_up"], four["why"], four["tokens"], four["reference"]["tail_s"]) == (
        S.TOKENS, None, None, {"value": four["value"], "ci95": four["ci95"]}, None)
    assert four["value"] is not None and "readings" not in four["ttft"] and four["scaled_down"]
    # a directory whose plan names no span of arrivals, as one in flight needs none, is of the four requests all the same
    arms = [a | {"planned": {"prefix": {k: v for k, v in a["planned"]["prefix"].items() if k != "span_s"}}}
            if "planned" in a else a for a in cost_arms()]
    assert S.cost_report(arms)["prefix"]["scaled_down"]


def test_the_cost_says_what_the_arms_boundary_commitment_did_and_whether_its_tables_were_built():
    """What the arm asked for is not what its engine did: the line counts the classes whose own lines carry tables."""
    arms = [cost_arms()[0], cost_arms()[-1]]
    arms[1] |= {"boundary_rows": {"window_s": 30.0, "check": "t", **S.ARMS["two-stage"].rows},
                "boundary_commits": {"spans": 4200, "finished": 4200, "failed": 0, "tables": 0, "complete": True}}
    # the window the arm committed at is on the line: a commit only counts in a pass it falls inside
    line = "  two-stage's boundary commitment, a window every 30 s: 4,200 of 4,200 finished requests committed, 0 failed, "
    r = S.cost_report(arms)
    assert r["commitment"] == {"finished": 4200, "committed": 4200, "failed": 0, "tables": 0, "tables_asked": True,
                               "window_s": 30.0}
    assert line + ("none with history tables: the arm asks for them and this tree's boundary commitment builds none, "
                   "so this cost is without them\n") in S.cost_text(r)
    arms[1]["boundary_commits"]["tables"] = 4200
    assert line + "4,200 of them with their history tables\n" in S.cost_text(S.cost_report(arms))
    # a commit that failed is counted as failed, and nothing is said of why no tables were built
    arms[1]["boundary_commits"] |= {"tables": 0, "failed": 8}
    assert ("4,192 of 4,200 finished requests committed, 8 failed, none with history tables\n"
            in S.cost_text(S.cost_report(arms)))
    # an arm that asks for none says only that there are none
    arms[1] |= {"boundary_rows": {"check": "t", **S.ROWS}, "boundary_commits": {"spans": 10, "finished": 10}}
    assert "10 of 10 finished requests committed, 0 failed, none with history tables\n" in S.cost_text(S.cost_report(arms))


def test_the_cost_needs_both_arms_and_a_load_they_share():
    ref = cost_arm("compiled", {8: [2.0, 2.0]})
    with pytest.raises(SystemExit, match="no directory of arm two-stage"):
        S.cost_report([ref])
    with pytest.raises(SystemExit, match=r"share no load \(chat, long, prefix\) sent by vLLM's client"):
        S.cost_report([ref, cost_arm("two-stage", {8: [1.9, 1.9]}, load="long")])
    with pytest.raises(SystemExit, match=r"share no load \(chat, long, prefix\) sent by vLLM's client"):       # the corpus client's fixed rates
        S.cost_report([arm("compiled", None, [1000.0]) | {"model_id": "glm53"},
                       arm("two-stage", "boundary-rows", [900.0]) | {"model_id": "glm53"}])


def test_the_cost_command_prints_the_number_or_its_json_alone(tmp_path: Path, capsys: pytest.CaptureFixture):
    commits = {"spans": 1200, "finished": 1200, "waited_s": 2.0, "complete": True, "failed": 0, "tables": 1200}
    dirs = [bench_arm(tmp_path, "compiled", {"chat@8": [result(good=g, tok=2 * g) for g in (150.0, 152.0, 151.0) * 2]}),
            bench_arm(tmp_path, "two-stage", {"chat@8": [result(good=g, tok=2 * g) for g in (147.0, 149.0, 148.0) * 2]},
                      boundary_commits=commits, boundary_rows={"check": "t", **S.ARMS["two-stage"].rows})]
    # without the row on a repeated prefix no number is printed, as text or as JSON: chat's cost is on its own line
    assert S.main(["cost", *map(str, dirs)]) == 0
    text = capsys.readouterr().out
    assert text.startswith("serving cost, no repeated prefix, glm-5.3, two-stage over compiled, load chat alone: not "
                           "given without the row on a repeated prefix under it, and load prefix has not run   target")
    assert "  load chat, ShareGPT's first turns" in text and ": 2.03% [2.01%, 2.04%] over 6 reps" in text
    assert S.main(["cost", *map(str, dirs), "--json"]) == 0
    bare = json.loads(capsys.readouterr().out)
    assert (bare["value"], bare["ci95"], bare["prefix"], bare["loads"][0]["value"]) == (None, None, None, 2.03)
    # a job of its own served that load, four requests a rep and one in flight; its row is given without the number
    row = [bench_arm(tmp_path / "row", name, {"prefix@1": [result(completed=4, duration=400.0, tok=t)] * 6}, planned=PLANNED)
           for name, t in (("compiled", 3.0), ("two-stage", 1.5))]
    assert S.main(["cost", *map(str, row)]) == 0
    first, under = capsys.readouterr().out.splitlines()[:2]
    assert first.startswith("serving cost, no repeated prefix, glm-5.3, two-stage over compiled: none of its loads has run")
    assert under.startswith("  on a repeated prefix, scaled down, two prompts near the context limit")
    dirs += row
    assert S.main(["cost", *map(str, dirs), "--out", str(tmp_path / "out")]) == 0
    text = capsys.readouterr().out
    assert text.startswith("serving cost, no repeated prefix, glm-5.3, two-stage over compiled, load chat alone: 2.03% "
                           "[2.01%, 2.04%]")
    assert text.splitlines()[1].startswith("  on a repeated prefix, scaled down, ") and ": 100.00% [100.00%, 100.00%] over 6 reps" in text
    assert "1,200 of 1,200 finished requests committed, 0 failed, 1,200 of them with their history tables\n" in text
    assert text == (tmp_path / "out" / "serving_cost.txt").read_text()
    assert S.main(["cost", *map(str, dirs), "--json", "--target", "3"]) == 0
    (line,) = capsys.readouterr().out.splitlines()
    r = json.loads(line)
    # chat's cost is within the target, and the number's verdict waits for the load that has not run
    assert (r["value"], r["target"], r["verdict"], r["load"], r["not_run"]) == (2.03, 3.0, "cannot tell", "chat", ["long"])
    assert list(r)[:6] == ["value", "unit", "commit", "run", "prefix", "label"] and r["prefix"]["value"] == 100.0
    assert r["loads"][0]["verdict"] == "meets" and r["commitment"]["tables"] == 1200
    # what was saved is against the model's target, whose parts the arm leaves off: the same costs, not comparable
    saved = json.loads((tmp_path / "out" / "serving_cost.json").read_text())
    assert (saved["target"], saved["target_basis"], saved["verdict"], saved["loads"][0]["verdict"]) == (
        2.05, "modelled", "not comparable", "not comparable")
    assert r["target_parts"] == [] and [p["name"] for p in saved["target_parts"] if p["off"]] == [
        "the certified tick", "the snapshot bundle"]
    assert r["not_in_target"] == [] and len(saved["not_in_target"]) == 4
    of_target = ("target", "target_basis", "target_parts", "not_in_target", "verdict", "loads", "arms")
    assert {k: v for k, v in saved.items() if k not in of_target} == {k: v for k, v in r.items() if k not in of_target}
    assert saved["loads"][0] | {"verdict": "meets"} == r["loads"][0]


def test_a_runs_result_holds_the_number_under_its_metric_id_only_once_every_load_has_run(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, whole: list[dict]):
    """What a reader of results takes the number from is the run's result.json, the entry named by the metric's id: a
    job that ran one load writes that load's cost and no such entry, so one load's cost is never read as the number."""
    r = S.cost_report(whole, arm=WHOLE)
    ok, detail, what, ms = S.result_of(r)
    assert ok and detail == S.cost_text(r).splitlines()[0] and what["loads"] == ["chat", "long", "prefix"]
    assert [m["name"] for m in ms] == ["chat", "long", "prefix", "vllm/serving-cost-glm53"]
    chat, long, under, number = ms
    about = {"unit": "percent of stock's serving compute", "target": 2.05, "target_basis": "modelled", "better": "lower"}
    assert all(m | about == m and m["commit"] == "c" * 40 and m["run"] == "r-1" for m in (chat, long, number))
    # the row on a repeated prefix is a measurement with no target and no verdict, and the number's entry carries it
    assert under | about | {"target": None, "target_basis": None} == under and under["verdict"] is None
    assert (under["value"], under["ci95"], under["label"], under["scaled_down"], under["reps"], under["rung"]) == (
        98.02, [96.08, 100.0], "on a repeated prefix", True, 6, 1)
    assert number["label"] == "no repeated prefix" and number["prefix"].items() <= under.items()
    assert number["prefix"]["value"] == 98.02 and list(number["prefix"])[:3] == ["load", "label", "scaled_down"]
    assert (chat["value"], chat["ci95"], chat["verdict"], chat["reps"]) == (2.04, [1.52, 2.56], "cannot tell", 6)
    assert (number["value"], number["ci95"], number["verdict"], number["load"]) == (6.95, [6.38, 7.53], "misses", "long")
    # the number names the loads it is the largest of, each with its cost beside it, and its target's parts
    assert number["inputs"] == ["chat", "long"] and number["target_parts"] == r["target_parts"] != []
    assert [(c["load"], c["value"], c["verdict"]) for c in number["loads"]] == [
        ("chat", 2.04, "cannot tell"), ("long", 6.95, "misses")]
    # each load's entry says the rung it was read at and whether stock's rungs show it saturated there
    assert [(m["rung"], m["saturated"]) for m in (chat, long)] == [(8, False), (4, False)]
    assert [(c["rung"], c["saturated"]) for c in number["loads"]] == [(8, False), (4, False)]
    # the launch line's own result carries the fourth verdict, with the changes it leaves off, and still stands
    ours = S.result_of(S.cost_report(cost_arms()))
    assert ours[0] and {m["verdict"] for m in ours[3] if m["name"] != "prefix"} == {"not comparable"}
    assert [p["off"] for p in ours[3][-1]["target_parts"]] == [["admission on the certified tick"], [], ["the edge snapshot"]]
    assert ours[3][-1]["not_in_target"] == ["the step record", "prefix caching off", "the two-stage boundary commitment",
                                            "the history tables"] and number["not_in_target"] == []
    # one load alone: its cost, and no entry under the metric's id
    alone = S.cost_report([cost_arm(name, {8: gs * 2}) for name, gs in (("compiled", [2.0] * 3), ("two-stage", [1.95, 1.96, 1.97]))])
    assert [m["name"] for m in S.result_of(alone)[3]] == ["chat"] and S.result_of(alone)[0]
    # every load of the number and no row under it: each load's cost, and no entry under the metric's id either
    bare = S.cost_report([a | {"loads": {k: v for k, v in a["loads"].items() if not k.startswith("prefix@")}} for a in whole],
                         arm=WHOLE)
    assert (bare["value"], bare["ci95"], bare["not_run"], bare["verdict"]) == (None, None, [], "misses")
    assert [m["name"] for m in S.result_of(bare)[3]] == ["chat", "long"] and S.result_of(bare)[0]
    assert "whole over compiled: not given without the row on a repeated prefix under it" in S.result_of(bare)[1]
    # the row alone: its measurement, and no number
    only = S.cost_report([a | {"loads": {k: v for k, v in a["loads"].items() if k.startswith("prefix@")}} for a in whole],
                         arm=WHOLE)
    assert (only["value"], only["load"], only["loads"], only["arms"], only["not_run"]) == (None, None, [], [], ["chat", "long"])
    assert [m["name"] for m in S.result_of(only)[3]] == ["prefix"] and S.result_of(only)[0]
    assert S.result_of(only)[2]["loads"] == ["prefix"] and "noise floor" not in S.cost_text(only)
    # a cost with no finite value, or a boundary commitment that left a finished request out, is not a passed result
    none = cost_arm("two-stage", {8: [0.0] * 6})
    assert not S.result_of(S.cost_report([cost_arm("compiled", {8: [2.0] * 6}), none]))[0]
    arms = [cost_arms()[0], cost_arms()[-1] | {"boundary_commits": {"spans": 90, "finished": 100, "failed": 0}}]
    assert not S.result_of(S.cost_report(arms))[0]
    # the command writes it through the kit's writer, in the run's directory, citing the report beside it
    dirs = [bench_arm(tmp_path, name, {f"{load}@8": [result(good=g, tok=2 * g) for g in gs * 2] for load in ("chat", "long")}
                      | {"prefix@1": [result(completed=4, duration=400.0, tok=t)] * 6}, planned=PLANNED)
            for name, gs, t in (("compiled", (150.0, 152.0, 151.0), 3.0), ("two-stage", (147.0, 149.0, 148.0), 1.5))]
    out = tmp_path / "run" / "out"
    with pytest.raises(SystemExit, match="with --out DIR"):
        S.main(["cost", *map(str, dirs), "--result"])
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    with pytest.raises(SystemExit, match="run it under research run"):
        S.main(["cost", *map(str, dirs), "--out", str(out), "--result"])
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-9")
    assert S.main(["cost", *map(str, dirs), "--out", str(out), "--result"]) == 0
    saved = json.loads((tmp_path / "run" / "result.json").read_text())
    assert (saved["schema"], saved["run_id"], saved["validation"]["status"]) == ("research/result/v0.1", "r-9", "passed")
    assert saved["validation"]["evidence"] == saved["measurement_files"] == ["out/serving_cost.json"]
    assert saved["validation"]["detail"] == capsys.readouterr().out.splitlines()[0]
    assert [(m["name"], m["value"], m["unit"]) for m in saved["measurements"]] == [
        (name, cost, "percent of stock's serving compute")
        for name, cost in (("chat", 2.03), ("long", 2.03), ("prefix", 100.0), ("vllm/serving-cost-glm53", 2.03))]
    assert saved["workload_fingerprint"] == {"benchmark": "serve_tax", "model": "glm-5.3", "arm": "two-stage",
                                             "reference": "compiled", "loads": ["chat", "long", "prefix"],
                                             "vllm": "0.1+gd9105ea8"}


def test_only_an_arm_that_loads_a_plugin_has_this_tree_on_its_servers_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    rows = {"check": "t"}
    paths = {a: S.server_env(S.ARMS[a], tmp_path, tmp_path / "site", {"bucket_us": 1, "sat_us": 1, "free_us": 1}, 600.0, 600.0,
                             boundary_rows=rows, private=tmp_path)["PYTHONPATH"].split(S.os.pathsep) for a in S.ARMS}
    for a, arm_ in S.ARMS.items():
        tree = [p for p in paths[a] if p in (str(S.HERE), str(S.ROOT), str(S.ROOT / "adapters/vllm"))]
        assert bool(tree) == bool(arm_.plugin or arm_.tick), a
        assert (str(tmp_path / "site") in paths[a]) == bool(S.entry_points(arm_)), a
        # nothing else of this tree is on it but the directory of each other plugin the arm names
        assert [p for p in paths[a] if p.startswith(str(S.ROOT))] == [*tree, *(str(S.ROOT / d) for _, d in arm_.general)], a
    assert paths["compiled"] == paths["compiled-again"] == [""]
    assert S.same_engine("compiled", "compiled-again") and not S.same_engine("compiled", "no-prefix-cache")
    assert not S.same_engine("compiled", "record") and not S.same_engine("compiled", "absent")
    assert not S.same_engine("compiled", "canonical-topk") and not S.same_engine("boundary-rows", "two-stage")


def bash(argv: list[str], env: dict[str, str]) -> tuple[int, str]:
    """What bash prints for ``argv`` under ``env`` alone, and its exit code."""
    import subprocess

    p = subprocess.run(["bash", *argv], capture_output=True, text=True, env={"PATH": S.os.environ["PATH"], **env})
    return p.returncode, p.stdout + p.stderr


def run_sh(*flags: str, **env: str) -> tuple[int, str]:
    """run.sh's flags, read before it asks for the research run it needs: what it refuses, and its exit code."""
    return bash([str(S.HERE / "run.sh"), *flags], env)


def served_order(arms: str, **env: str) -> list[str]:
    """The server lifetimes run.sh serves, in order, each ARM:K: its own lines that make the order, run by bash."""
    lines = (S.HERE / "run.sh").read_text().splitlines()
    first = lines.index("  order=()")
    loop = "\n".join(lines[first:lines.index("  done", first) + 1])
    rc, out = bash(["-c", loop + '\necho "${order[@]}"'], {"ARMS": arms, **env})
    assert rc == 0, out
    return out.split()


def test_run_sh_serves_the_arms_lifetimes_in_the_order_named_then_turned_round_or_the_turned_order_first():
    assert served_order("prod prod-record") == ["prod:1", "prod-record:1"]
    assert served_order("prod prod-record", LIFETIMES="3") == ["prod:1", "prod-record:1", "prod-record:2", "prod:2",
                                                               "prod:3", "prod-record:3"]
    # TURNED=1: the last arm named is served first, and the lifetimes of one number are still neighbours
    assert served_order("prod prod-record", LIFETIMES="2", TURNED="1") == ["prod-record:1", "prod:1", "prod:2", "prod-record:2"]
    assert served_order("prod prod-record", TURNED="1") == ["prod-record:1", "prod:1"]
    # both are read before the run, as is the warm-up's seconds, which reach every arm where they are given
    assert run_sh("--model", "glm53", TURNED="2") == (
        2, "serve-tax: TURNED=2: 1 begins with the arms' order turned round, 0 with the order named\n")
    assert run_sh("--model", "glm53", WARM_S="none") == (2, "serve-tax: WARM_S=none: the warm-up's seconds, 0 for none\n")
    rc, out = run_sh("--model", "glm53", "--arms", "prod prod-record", LIFETIMES="2", LEAVE_OUT="1", TURNED="1", WARM_S="0")
    assert rc != 0 and "run this under research run" in out
    sh = (S.HERE / "run.sh").read_text()
    assert '[ -n "${WARM_S:-}" ] && count+=(--warm-s "$WARM_S")' in sh and sh.count('"${count[@]}"') == 1
    # so is how long the corpus client waits for a load's last answers: a request at the context limit needs more
    assert run_sh(ARMS="prod", RATES="limit=1000", DRAIN_S="long") == (
        2, "serve-tax: DRAIN_S=long: the seconds the corpus client waits for answers after a load's last arrival\n")
    rc, out = run_sh(ARMS="prod", RATES="limit=1000", DRAIN_S="2400")
    assert rc != 0 and "run this under research run" in out
    assert '[ -n "${DRAIN_S:-}" ] && count+=(--drain-s "$DRAIN_S")' in sh


def test_run_sh_takes_the_serving_costs_flags_and_refuses_before_it_holds_anything():
    assert run_sh("--bogus") == (2, "serve-tax: unknown flag --bogus (--model, --arms, --loads, --dataset NAME=FILE, --reps, "
                                    "--target, --json)\n")
    assert run_sh("--model") == (2, "serve-tax: --model takes a value\n")
    # the default arms end in the two-stage serve's launch line, which is the arm's data: the job names no object
    for flags in (("--model", "glm53"), ("--model", "glm53", "--arms", "compiled compiled-again", "--json")):
        rc, out = run_sh(*flags)
        assert rc != 0 and "run this under research run" in out    # past the flags: only the run is missing
    rc, out = run_sh(ARMS="compiled", RATES="light=1")             # the corpus client's form is untouched
    assert rc != 0 and "run this under research run" in out
    # the arms' lifetimes and what each leaves out are read before the run too
    assert run_sh("--model", "glm53", LIFETIMES="0") == (2, "serve-tax: LIFETIMES=0: how many times the arms are served, 1 or more\n")
    assert run_sh("--model", "glm53", LEAVE_OUT="6") == (
        2, "serve-tax: LEAVE_OUT=6: the passes each lifetime leaves out, fewer than its REPS=6\n")
    rc, out = run_sh("--model", "glm53", "--arms", "prod prod-record", LIFETIMES="6", LEAVE_OUT="1")
    assert rc != 0 and "run this under research run" in out
    sh = (S.HERE / "run.sh").read_text()
    assert 'cost "${dirs[@]}" --arm "${ARMS##* }"' in sh and '--client vllm-bench --loads "$LOADS"' in sh
    # a job that ends on a stock arm has no cost: no arm of ours is over it
    assert all(f'[ "${{ARMS##* }}" != {stock} ]' in sh for stock in ("compiled", "published"))
    assert S.ARMS["published"].plugin == S.ARMS["compiled"].plugin and not S.ARMS["published"].args
    # an earlier job's arms join the cost, after this job's table is made of this job's arms alone
    assert sh.index('table "${dirs[@]}"') < sh.index('for o in ${WITH:-}; do for d in "$o"/*/') < sh.index('cost "${dirs[@]}"')
    assert 'ARMS=${ARMS:-compiled compiled-again two-stage}; LOADS=${LOADS:-chat,long}' in sh
    # six reps give the cost its interval, a Commit period shorter than a pass puts the commits inside the passes, and
    # the cost is written as the run's result
    assert 'REPS=${REPS:-6}; WINDOW_S=${WINDOW_S:-30}' in sh and '"${cost[@]}" --out "$OUT" --result' in sh
    # a budget's time left is shared equally by the arms still to serve: the measured arm is served last, and a cost
    # compares only the reps both of its arms have
    assert 'left=$(((BUDGET_MIN * 60 - ($(date +%s) - t_leased)) / to_serve))' in sh
    assert sh.count("to_serve=$((to_serve - 1))") == 1 and "to_serve=${#order[@]}" in sh
    # --arms all is every arm vLLM's client can drive from stock up to the launch line, each after the arm it is over
    every = re.search(r'ARMS = all \] && ARMS="([^"]+)"', sh.replace("\"$ARMS\"", "ARMS")).group(1).split()
    assert every[:2] == ["compiled", "compiled-again"] and every[-1] == S.OURS and set(every) <= set(S.ARMS)
    assert all(S.ARMS[a].parent in every[:i] for i, a in enumerate(every) if S.ARMS[a].parent)
    # before it holds a GPU the job checks its arms and downloads the pinned dataset where the node has none
    assert 'passes --loads "$LOADS" "${ds[@]}" --fetch --arms "$ARMS" --model-id "$MODEL_ID"' in sh
    # a given pass size reaches that check and every arm alike: the check prints the passes the arms then serve
    assert sh.count('${PASS_PROMPTS:+--pass-prompts "$PASS_PROMPTS"}') == 2
