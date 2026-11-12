"""The report on synthetic arm and replay directories, shaped as the serving cost benchmark leaves them, and the runner
against a stand-in for that benchmark's runner."""
from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import prod_repro as P
import pytest
import replay as R
import serve_tax as S

LOAD = "burst"


def rid(tag: str, i: int) -> str:
    return f"cmpl-{tag}-{i:05d}-0-0123abcd"


def corpus_arm(out: Path, name: str, tok_per_s: list[float], *, argv: tuple[str, ...] = (), trace: bool = False,
               arrivals_s: float = 0.001, run: str = "r-1", unserved: int = 0) -> Path:
    """An arm's directory as `serve_tax.py run` leaves it under the corpus client: each rep two requests of two tokens,
    sent together and done so that the rep's output tokens a second is ``tok_per_s``'s; with ``trace``, a step trace in
    which rep 1's second request begins from a 16-token prefix-cache hit."""
    d = out / name
    (d / "server").mkdir(parents=True)
    ck = out / "ck"
    ck.mkdir(exist_ok=True)
    (ck / "config.json").write_text(json.dumps({"max_position_embeddings": 131072, "n_positions": 262144}))
    (out / "plan.jsonl").write_text("".join(json.dumps({"i": i, "prompt_ids": [i, i + 1]}) + "\n" for i in range(2)))
    (d / "meta.json").write_text(json.dumps({
        "arm": name, "parent": None, "what": name, "model_id": "moonlight", "model": "repo@rev", "client": "corpus",
        "rates": {LOAD: 1000.0}, "run_id": run, "commit": "c" * 40, "checkpoint": str(ck), "plan": str(out / "plan.jsonl"),
        "server_argv": ["-m", "vllm.entrypoints.openai.api_server", "--max-model-len", "4096", "--seed", "0", *argv]}))
    for k, tps in enumerate(tok_per_s, 1):
        rep, done = d / LOAD / f"rep-{k}", 4 / tps
        rep.mkdir(parents=True)
        (rep / "client.jsonl").write_text("".join(json.dumps(
            {"id": f"{LOAD}{k}-{i:05d}", "i": i, "status": 200 if i >= unserved else 500, "send_unix": 1000.0,
             "first_token_s": done / 2, "done_s": done, "output_ids": [7, i], "chunks": [[done / 2, 1], [done, 1]]}) + "\n"
            for i in range(2)))
        (rep / "meta.json").write_text(json.dumps({
            "tag": f"{LOAD}{k}", "rate": 1000.0, "requests": 2, "arrivals_span_s": arrivals_s, "t0_unix": 1000.0,
            "wall_s": done, "unfinished": 0, "schedule": {"steps": 2, "step_tokens": 104, "generation_tokens": 4,
                                                          "prompt_tokens": 100, "prompt_tokens_cached": 40, "preemptions": 0}}))
    if trace:
        a, b = rid(f"{LOAD}1", 0), rid(f"{LOAD}1", 1)
        recs = [{"kind": "header", "config": {"enable_prefix_caching": "--no-enable-prefix-caching" not in argv, "max_model_len": 4096}},
                {"kind": "arrive", "request": a, "before_step": 0, "sampling": {"max_tokens": 2}, "t": 0.0},
                {"kind": "arrive", "request": b, "before_step": 0, "sampling": {"max_tokens": 2}, "t": 0.0},
                {"kind": "step", "step": 0, "rows": [[a, 0, 50], [b, 16, 34]], "cache_hit": {b: 16}, "t": 0.1},
                {"kind": "step", "step": 1, "rows": [[a, 50, 1], [b, 50, 1]], "cache_hit": {}, "t": 0.2}]
        with gzip.open(d / "server" / "step_trace.9.jsonl.gz", "wt") as f:
            f.write("".join(json.dumps(r) + "\n" for r in recs))
    return d


def replayed(out: Path, ours: str = "record", *, tokens: dict[str, list[int]] | None = None, steps: list[dict] | None = None,
             **over: object) -> Path:
    """A replay's directory as `replay.py` leaves it for rep 1 of ``ours``: by default every step and token as served."""
    w = R.window(out / ours, LOAD, 1)
    got = {"steps": steps if steps is not None else [dict(s) for s in w["steps"]],
           "tokens": {r["engine_id"]: list(r["served"]) for r in w["requests"]} | (tokens or {})}
    d = out / "replay-compiled"
    d.mkdir()
    (d / "replay.json").write_text(json.dumps({
        "arm_dir": str(out / ours), "served_run_id": "r-1", "run_id": "r-1", "load": LOAD, "rep": 1, "mode": "compiled",
        "served_mode": "compiled", "served_window_s": 0.2, "replay_wall_s": 0.3, "engine_ready_s": 9.0,
        **R.compare(w, got), **over}))
    R.write_tokens(d, got["tokens"])
    return d


def job(tmp_path: Path, ours: list[float] = (99.0, 100.0, 101.0), **kw: object) -> Path:
    """A job's out/: stock, stock again and the step record, ours 1% slower than stock unless told."""
    out = tmp_path / "out"
    corpus_arm(out, "compiled", [100.0, 101.0, 102.0])
    corpus_arm(out, "compiled-again", [100.5, 101.0, 101.5])
    corpus_arm(out, "record", list(ours), trace=True, **kw)
    return out


def run(out: Path, *argv: str) -> tuple[int, str, dict]:
    rc = P.main(["report", str(out), "--load", LOAD, *argv])
    return rc, (out / "prod_repro.txt").read_text(), json.loads((out / "prod_repro.json").read_text())


def test_a_replayed_window_prints_the_settings_and_the_three_lines(tmp_path: Path, capsys: pytest.CaptureFixture):
    out = job(tmp_path)
    replayed(out)
    rc, txt, r = run(out)
    assert rc == 0 and capsys.readouterr().out == txt and r["failed"] == []
    assert txt.startswith("prod_repro: load burst; stock compiled, stock again compiled-again, ours record; run r-1, commit "
                          + "c" * 40 + f"\nmodel: repo@rev, as the benchmark names it; the checkpoint served: {out / 'ck'}\n")
    assert ("tokens: reproduced: 2 requests replayed, 4 served tokens compared with 4 replayed, 0 differ in 0 requests\n"
            "  2 engine steps, 0 with other rows than the served schedule; 1 requests began from a prefix-cache hit, the "
            "longest 16 tokens\n"
            "  served compiled, replayed compiled; the window: every request of the load's rep") in txt
    # the cost is the benchmark's tax of the arms' output tokens a second over reps, with its interval and its floor
    ref, mine = (S.summary(x) for x in ([100.0, 101.0, 102.0], [99.0, 100.0, 101.0]))
    want = S.tax(ref, mine)
    assert r["cost"]["value"] == pytest.approx(want["value"]) and r["cost"]["ci95"] == pytest.approx(want["ci95"])
    assert r["cost"]["how"]["client"] == "corpus" and r["cost"]["noise_floor"]["arm"] == "compiled-again"
    assert f"cost: record over compiled {100 * want['value']:+.2f}% of stock's serving time [" in txt
    assert "]; under 1%: cannot tell\n  95% interval over 3 and 3 reps, at 1000 requests a second, which stock did not keep up with" in txt
    assert "\n  noise floor: compiled-again over compiled" in txt
    assert "advice: not measured: nothing in this job computes it" in txt
    assert txt.endswith("prod_repro: 2 of 3 lines measured\n")


def test_a_replayed_token_that_differs_fails_the_job_naming_the_request_and_the_position(tmp_path: Path):
    out = job(tmp_path)
    b = rid(f"{LOAD}1", 1)
    replayed(out, tokens={b: [7, 9]})
    rc, txt, r = run(out)
    assert rc == 1 and r["tokens"]["reproduces"] is False
    assert ("tokens: FAILED: 2 requests replayed, 4 served tokens compared with 4 replayed, 1 differ in 1 requests\n"
            f"  request {b} position 1 (served 2 tokens, replayed 2; 1 positions differ)\n") in txt
    assert r["failed"] == [f"request {b} position 1 (served 2 tokens, replayed 2; 1 positions differ)"]
    assert txt.endswith("prod_repro: 2 of 3 lines measured; FAILED: the replay differs from what was served\n")


def test_a_replay_that_stops_short_or_runs_another_schedule_fails_too(tmp_path: Path):
    out = job(tmp_path)
    a, b = rid(f"{LOAD}1", 0), rid(f"{LOAD}1", 1)
    replayed(out, tokens={a: [7]})
    rc, txt, _ = run(out)
    assert rc == 1 and f"request {a} position 1 (served 2 tokens, replayed 1; 1 positions differ)" in txt
    shutil.rmtree(out / "replay-compiled")
    # the same tokens from other rows: the replay did not run the served schedule
    replayed(out, steps=[{"step": 0, "rows": [[a, 0, 50], [b, 0, 50]]}, {"step": 1, "rows": [[a, 50, 1], [b, 50, 1]]}])
    rc, txt, r = run(out)
    assert rc == 1 and r["tokens"]["tokens_differing"] == 0
    assert "1 of 2 steps' rows differ from the served schedule, the first step 0" in txt


def test_the_kept_tokens_are_compared_not_the_replays_own_count(tmp_path: Path):
    out = job(tmp_path)
    d = replayed(out, tokens={rid(f"{LOAD}1", 1): [7, 9]})
    rec = json.loads((d / "replay.json").read_text())
    (d / "replay.json").write_text(json.dumps(rec | {"requests_tokens_differing": 0, "token_positions_differing": 0,
                                                     "bit_exact": True, "requests_differing": []}))
    rc, txt, r = run(out)
    assert rc == 1 and r["tokens"]["record_disagrees"] and "1 differ in 1 requests" in txt
    assert "the replay's record (replay.json) and the tokens it kept disagree" in txt


def test_without_a_replay_of_this_window_the_tokens_are_not_measured(tmp_path: Path):
    out = job(tmp_path)
    rc, txt, r = run(out)
    assert rc == 0 and r["tokens"] == {"measured": False, "why": f"no replay ran (no replay.json in {out / 'replay-compiled'})"}
    assert txt.endswith("prod_repro: 1 of 3 lines measured\n")
    # a replay of another job's serve of the same arm, or of another rep, is not this window's
    for other in ({"served_run_id": "r-0"}, {"rep": 2}, {"arm_dir": str(out / "compiled")}):
        shutil.rmtree(out / "replay-compiled", ignore_errors=True)
        replayed(out, **other)
        rc, txt, r = run(out)
        assert rc == 0 and not r["tokens"]["measured"] and "tokens: not measured: the replay in" in txt
        assert "is of another window" in txt


def test_a_setting_the_serve_did_not_turn_on_is_off(tmp_path: Path):
    out = tmp_path / "out"
    corpus_arm(out, "compiled", [100.0, 101.0])
    corpus_arm(out, "record", [99.0, 100.0], trace=True, argv=("--no-enable-prefix-caching", "--enforce-eager"))
    _, txt, r = run(out)
    mine, stock = r["settings"]["record"], r["settings"]["compiled"]
    assert (mine["prefix_caching"]["state"], mine["speculative_decoding"]["state"], mine["compiled"]["state"]) == ("off",) * 3
    # stock kept no step trace: its prefix cache is on by what its engine's counters say it served
    assert stock["prefix_caching"] == {"state": "on", "by": "the cache served 40.0% of the load's prompt tokens", "cached_share": 0.4}
    assert "  prefix caching: off (--no-enable-prefix-caching); DIFFERS; compiled: on (" in txt
    assert "  speculative decoding: off (no --speculative-config in the launch line); compiled: off (" in txt
    assert "  context limit: 4,096 tokens, BELOW the model's own 131,072 (config.json max_position_embeddings)\n" in txt
    assert "  circuit compiled ahead of serving: no: nothing in this job compiles the settings' circuit" in txt
    # a cost between arms at different settings is printed, and no verdict is given on it
    assert "; under 1%: not said, the arms are not at the same settings\n" in txt
    assert ("  NOT at the same settings: record and compiled differ in prefix caching, torch.compile and CUDA graphs, so "
            "the figure is not our changes' cost\n") in txt
    assert r["cost"]["settings_differ"] == ["prefix caching", "torch.compile and CUDA graphs"]
    assert r["cost"]["verdict"] == P.UNLIKE


def test_speculative_decoding_and_the_models_own_limit_are_read_from_the_launch_line(tmp_path: Path):
    out = tmp_path / "out"
    spec = ("--speculative-config.method", "mtp", "--speculative-config.num_speculative_tokens", "5")
    corpus_arm(out, "compiled", [100.0, 101.0], argv=(*spec, "--max-model-len=131072"))
    corpus_arm(out, "record", [100.0, 101.0], argv=(*spec, "--kv-cache-dtype", "fp8", "--max-model-len=131072"))
    _, txt, r = run(out)
    assert r["settings"]["record"]["speculative_decoding"] == {"state": "on", "by": " ".join(spec)}
    assert ("  speculative decoding: on (--speculative-config.method mtp --speculative-config.num_speculative_tokens 5); "
            "compiled: on (") in txt
    # the KV cache's type is the launch line's, and an arm that does not share it is not at the same settings
    assert ("  KV cache: fp8 (--kv-cache-dtype); DIFFERS; compiled: the engine's default (no --kv-cache-dtype in the "
            "launch line)\n") in txt
    assert r["cost"]["settings_differ"] == ["the KV cache's type"]
    assert P.flags(["--speculative-config={}", "--seed", "0"], ("--speculative",)) == "--speculative-config={}"
    assert r["settings"]["record"]["context_limit"] == {
        "tokens": 131072, "model": {"tokens": 131072, "key": "max_position_embeddings"}, "models_own": True}
    assert "  context limit: 131,072 tokens, the model's own 131,072 (config.json max_position_embeddings)\n" in txt
    # an arm with neither a step trace nor a counted hit: nothing confirms vLLM's default
    for rep in (out / "record" / LOAD).glob("rep-*/meta.json"):
        rep.write_text(json.dumps(json.loads(rep.read_text()) | {"schedule": None}))
    (out / "ck" / "config.json").unlink()
    _, txt, r = run(out)
    assert r["settings"]["record"]["prefix_caching"]["state"] == "not confirmed"
    assert "131,072 tokens, the model's own limit not read (no config.json at the checkpoint on this machine)" in txt


def test_tokens_a_second_are_no_cost_where_stock_kept_up_or_an_arm_left_requests_unserved(tmp_path: Path):
    out = tmp_path / "out"
    corpus_arm(out, "compiled", [100.0, 101.0], arrivals_s=0.039)    # each rep ends within 0.001 s of its last arrival
    corpus_arm(out, "record", [99.0, 100.0], trace=True, arrivals_s=0.039)
    _, txt, r = run(out)
    assert not r["cost"]["measured"] and r["cost"]["how"]["kept_up"] and r["cost"]["reading"]["value"] == pytest.approx(0.0101, abs=1e-3)
    assert "cost: not measured: compiled kept up with the arrivals" in txt and "send faster" in txt
    shutil.rmtree(out)
    corpus_arm(out, "compiled", [100.0, 101.0])
    corpus_arm(out, "record", [99.0, 100.0], trace=True, unserved=1)
    _, txt, r = run(out)
    assert r["cost"] == {"measured": False, "why": "the arms did not serve the same tokens: record left 2 requests unserved"}
    shutil.rmtree(out / "compiled")
    _, txt, r = run(out)
    assert "cost: not measured: compiled did not serve load burst" in txt and "; compiled was not served" in txt


def result(tok: float) -> dict:
    """A pass's result as vLLM's client saves it, with the keys the benchmark reads."""
    return {"completed": 200, "failed": 0, "duration": 100.0, "total_input_tokens": 51200, "total_output_tokens": int(tok * 100),
            "request_throughput": 2.0, "request_goodput": 1.5, "output_throughput": tok, "total_token_throughput": tok + 512.0,
            "median_ttft_ms": 400.0, "p99_ttft_ms": 900.0, "median_tpot_ms": 30.0, "p99_tpot_ms": 45.0,
            "median_itl_ms": 29.0, "p99_itl_ms": 70.0}


def bench_arm(out: Path, name: str, passes: dict[str, list[float]]) -> None:
    d = out / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({
        "arm": name, "parent": None, "what": name, "model_id": "glm53", "model": "repo@rev", "client": "vllm-bench", "rates": {},
        "run_id": "r-1", "commit": "c" * 40, "server_argv": ["--max-model-len", "4096"],
        "passes": {p: {"load": p.rpartition("@")[0], "concurrency": int(p.rpartition("@")[2])} for p in passes}}))
    for p, toks in passes.items():
        for i, tok in enumerate(toks, 1):
            (d / p / f"rep-{i}").mkdir(parents=True)
            (d / p / f"rep-{i}" / "result.json").write_text(json.dumps(result(tok)))
            (d / p / f"rep-{i}" / "meta.json").write_text(json.dumps(
                {"rate": None, "concurrency": int(p.rpartition("@")[2]), "unfinished": 0, "t0_unix": 1000.0, "wall_s": 105.0}))


def test_by_vllms_client_the_cost_is_the_benchmarks_paired_cost_and_nothing_is_replayed(tmp_path: Path):
    out = tmp_path / "out"
    stock = [300.0, 301.0, 299.0, 300.5, 299.5, 300.0]
    bench_arm(out, "compiled", {"chat@8": [200.0] * 6, "chat@16": stock, "chat@32": [290.0] * 6})
    bench_arm(out, "compiled-again", {"chat@16": [x + 0.3 for x in stock]})
    bench_arm(out, "record", {"chat@16": [x / 1.004 for x in stock]})
    rc, txt, r = P.main(["report", str(out), "--load", "chat"]), (out / "prod_repro.txt").read_text(), json.loads(
        (out / "prod_repro.json").read_text())
    want = S.serving_cost(S.arm_metrics(out / "compiled"), S.arm_metrics(out / "record"), "chat")
    assert rc == 0 and r["cost"]["value"] == pytest.approx(0.004) == pytest.approx(want["value"]) and r["cost"]["ci95"] == want["ci95"]
    assert r["cost"]["how"] == {"client": "vllm-bench", "rung": 16, "reps": 6, "saturated": True, "by": r["cost"]["how"]["by"]}
    assert r["cost"]["verdict"] == "meets"
    assert ("cost: record over compiled +0.40% of stock's serving time [+0.40%, +0.40%]; under 1%: yes\n"
            "  95% interval over 6 paired reps, at concurrency 16, where stock's rungs show it saturated: ") in txt
    assert ("tokens: not measured: no replay ran: a pass sent by vLLM's client keeps neither its prompts' ids nor the "
            "served ids") in txt


def test_the_arms_are_the_published_settings_arms_once_the_benchmark_has_them(monkeypatch: pytest.MonkeyPatch):
    assert P.default_arms() == (P.PROD if set(P.PROD) <= set(S.ARMS) else P.TODAY)
    assert all(a in S.ARMS for a in P.TODAY)
    monkeypatch.setattr(S, "ARMS", {**S.ARMS, **dict.fromkeys(P.PROD, S.ARMS["compiled"])})
    assert P.default_arms() == ("prod", "prod-again", "prod-record")
    monkeypatch.setattr(S, "ARMS", {k: v for k, v in S.ARMS.items() if k != "prod-record"})
    assert P.default_arms() == ("compiled", "compiled-again", "record")


# --------------------------------------------------------------------------------------------------------- the runner

#: a stand-in for benchmarks/serve_tax/run.sh: it says what it was started with and ends as told
STAND_IN = """#!/usr/bin/env bash
{ echo "argv=$*"; for k in ARMS RATES REPS REQUESTS SPAN_S REPLAY REPLAY_FROM REPLAY_MODES MODEL_ID CLIENT MAX_MODEL_LEN; do
    echo "$k=${!k-unset}"; done; } > "$RESEARCH_RUN_DIR/started"
exit "${STAND_IN_RC:-0}"
"""


def runner(tmp_path: Path) -> Path:
    """A tree with this directory's files and, in the serving cost benchmark's place, its two modules and the stand-in."""
    here, tax = tmp_path / "tree/benchmarks/prod_repro", tmp_path / "tree/benchmarks/serve_tax"
    here.mkdir(parents=True)
    tax.mkdir()
    for f in ("run.sh", "prod_repro.py"):
        shutil.copy(Path(P.__file__).parent / f, here / f)
    for f in ("serve_tax.py", "replay.py"):
        (tax / f).symlink_to(Path(S.__file__).parent / f)
    (tax / "run.sh").write_text(STAND_IN)
    return here / "run.sh"


def bash(script: Path, *argv: str, run_dir: Path | None = None, **env: str) -> subprocess.CompletedProcess:
    base = {k: v for k, v in os.environ.items() if k not in ("RESEARCH_RUN_DIR", "LOAD", "REPS", "REQUESTS", "MODEL_ID")}
    return subprocess.run(["bash", str(script), *argv], capture_output=True, text=True, check=False, env={
        **base, "VENV": str(Path(sys.executable).parents[1]), **({"RESEARCH_RUN_DIR": str(run_dir)} if run_dir else {}), **env})


def started(run_dir: Path) -> dict[str, str]:
    return dict(x.split("=", 1) for x in (run_dir / "started").read_text().splitlines())


def test_the_runner_serves_the_three_arms_a_burst_and_replays_ours_then_reports(tmp_path: Path):
    script, run_dir = runner(tmp_path), tmp_path / "run"
    replayed(job(run_dir))
    p = bash(script, "--max-model-len", "8192", run_dir=run_dir)
    assert p.returncode == 0, p.stdout + p.stderr
    assert started(run_dir) == {
        "argv": "", "ARMS": "compiled compiled-again record", "RATES": "burst=1000", "REPS": "3", "REQUESTS": "800",
        "SPAN_S": "unset", "REPLAY": "burst", "REPLAY_FROM": str(run_dir / "out/record"), "REPLAY_MODES": "compiled",
        "MODEL_ID": "moonlight", "CLIENT": "corpus", "MAX_MODEL_LEN": "8192"}
    assert "tokens: reproduced: 2 requests replayed" in p.stdout and p.stdout.endswith("prod_repro: 2 of 3 lines measured\n")
    assert (run_dir / "out/prod_repro.txt").read_text() in p.stdout
    # an arm or a replay that failed fails the job, and the report still says what the files hold
    p = bash(script, run_dir=run_dir, STAND_IN_RC="1")
    assert p.returncode == 1 and "the serving cost benchmark's runner ended 1" in p.stdout and "tokens: reproduced" in p.stdout
    # a replayed token that differs fails a job whose serve and replay ended well
    shutil.rmtree(run_dir / "out/replay-compiled")
    replayed(run_dir / "out", tokens={rid(f"{LOAD}1", 0): [8, 0]})
    p = bash(script, run_dir=run_dir)
    assert p.returncode == 1 and f"request {rid(f'{LOAD}1', 0)} position 0" in p.stdout


def test_the_runner_hands_vllms_client_the_load_and_asks_for_no_replay(tmp_path: Path):
    script, run_dir = runner(tmp_path), tmp_path / "run"
    stock = [300.0, 301.0, 299.0, 300.5, 299.5, 300.0]
    for name in ("compiled", "compiled-again", "record"):
        bench_arm(run_dir / "out", name, {"chat@16": stock})
    p = bash(script, "--client", "vllm-bench", "--load", "chat=16", "--model", "glm53", run_dir=run_dir)
    assert p.returncode == 0, p.stdout + p.stderr
    s = started(run_dir)
    assert s["argv"] == "--model glm53 --arms compiled compiled-again record --loads chat=16 --reps 6"
    assert (s["REPLAY"], s["REPLAY_FROM"], s["CLIENT"]) == ("unset", "unset", "vllm-bench")
    assert "tokens: not measured: no replay ran" in p.stdout and "cost: record over compiled +0.00%" in p.stdout


def test_the_runner_reads_an_earlier_job_again_and_refuses_what_it_cannot_run(tmp_path: Path):
    script = runner(tmp_path)
    out = job(tmp_path)
    replayed(out)
    p = bash(script, "--report", str(out), "--load", LOAD)
    assert p.returncode == 0 and p.stdout == (out / "prod_repro.txt").read_text() and "tokens: reproduced" in p.stdout
    for argv, said in ((("--report", str(out)), "--report OUT --load NAME"), (("--sample", "3"), "unknown flag --sample"),
                       (("--load",), "--load takes a value"), (("--client", "curl"), "--client corpus or vllm-bench"),
                       ((), "run this under research run")):
        p = bash(script, *argv)
        assert p.returncode != 0 and said in p.stdout + p.stderr, argv
    # a job that served nothing has no report
    (tmp_path / "empty").mkdir()
    p = bash(script, run_dir=tmp_path / "empty", STAND_IN_RC="2")
    assert p.returncode == 2 and "nothing was served" in p.stdout
