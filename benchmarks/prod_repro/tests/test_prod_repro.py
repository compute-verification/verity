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
#: the arms of every job here, named to the report and the runner: their defaults are the benchmark's arms at the
#: published settings where it has them
ARMS = ("--stock", "compiled", "--again", "compiled-again", "--ours", "record")


def rid(tag: str, i: int) -> str:
    return f"cmpl-{tag}-{i:05d}-0-0123abcd"


def step_trace(out: Path, d: Path, tag: str, config: dict) -> None:
    """What the replay reads of an arm beside a rep's own files: the plan and the arm's step trace, in which the rep
    tagged ``tag`` is two requests of two tokens, the second begun from a 16-token prefix-cache hit."""
    (out / "plan.jsonl").write_text("".join(json.dumps({"i": i, "prompt_ids": [i, i + 1]}) + "\n" for i in range(2)))
    a, b = rid(tag, 0), rid(tag, 1)
    recs = [{"kind": "header", "config": config},
            {"kind": "arrive", "request": a, "before_step": 0, "sampling": {"max_tokens": 2}, "t": 0.0},
            {"kind": "arrive", "request": b, "before_step": 0, "sampling": {"max_tokens": 2}, "t": 0.0},
            {"kind": "step", "step": 0, "rows": [[a, 0, 50], [b, 16, 34]], "cache_hit": {b: 16}, "emitted": {a: 1, b: 1}, "t": 0.1},
            {"kind": "step", "step": 1, "rows": [[a, 50, 1], [b, 50, 1]], "cache_hit": {}, "emitted": {a: 1, b: 1}, "t": 0.2}]
    (d / "server").mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "server" / "step_trace.9.jsonl.gz", "wt") as f:
        f.write("".join(json.dumps(r) + "\n" for r in recs))


def corpus_arm(out: Path, name: str, tok_per_s: list[float], *, argv: tuple[str, ...] = (), trace: bool = False,
               arrivals_s: float = 0.001, run: str = "r-1", unserved: int = 0, header: dict | None = None,
               **record: object) -> Path:
    """An arm's directory as `serve_tax.py run` leaves it under the corpus client: each rep two requests of two tokens,
    sent together and done so that the rep's output tokens a second is ``tok_per_s``'s; with ``trace``, a step trace in
    which rep 1's second request begins from a 16-token prefix-cache hit, its header the engine's configuration with
    ``header``'s keys; ``record``'s keys are the arm's record's (an ``error``, ``stopped``)."""
    d = out / name
    (d / "server").mkdir(parents=True)
    ck = out / "ck"
    ck.mkdir(exist_ok=True)
    (ck / "config.json").write_text(json.dumps({"max_position_embeddings": 131072, "n_positions": 262144}))
    (out / "plan.jsonl").write_text("".join(json.dumps({"i": i, "prompt_ids": [i, i + 1]}) + "\n" for i in range(2)))
    (d / "meta.json").write_text(json.dumps({
        "arm": name, "parent": None, "what": name, "model_id": "moonlight", "model": "repo@rev", "client": "corpus",
        "rates": {LOAD: 1000.0}, "run_id": run, "commit": "c" * 40, "checkpoint": str(ck), "plan": str(out / "plan.jsonl"),
        "server_argv": ["-m", "vllm.entrypoints.openai.api_server", "--max-model-len", "4096", "--seed", "0", *argv],
        "startup_s": 12.5, "loads": [{"load": LOAD, "rep": k, "rate": 1000.0, "requests": 2, "unfinished": 0}
                                     for k in range(1, len(tok_per_s) + 1)], **record}))
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
        step_trace(out, d, f"{LOAD}1", {"enable_prefix_caching": "--no-enable-prefix-caching" not in argv,
                                        "max_model_len": 4096, **(header or {})})
    return d


def replayed(out: Path, ours: str = "record", *, tokens: dict[str, list[int]] | None = None, steps: list[dict] | None = None,
             load: str = LOAD, keep_steps: bool = True, **over: object) -> Path:
    """A replay's directory as `replay.py` leaves it for rep 1 of ``ours``'s pass ``load``: by default every step and
    token as served."""
    w = R.window(out / ours, load, 1)
    got = {"steps": steps if steps is not None else [dict(s) for s in w["steps"]],
           "tokens": {r["engine_id"]: list(r["served"]) for r in w["requests"]} | (tokens or {})}
    d = out / "replay-compiled"
    d.mkdir()
    (d / "replay.json").write_text(json.dumps({
        "arm_dir": str(out / ours), "served_run_id": "r-1", "run_id": "r-1", "load": load, "rep": 1, "mode": "compiled",
        "served_mode": "compiled", "served_window_s": 0.2, "replay_wall_s": 0.3, "engine_ready_s": 9.0,
        **R.compare(w, got), **over}))
    R.write_tokens(d, got["tokens"])
    if keep_steps:
        (d / "replay_steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in got["steps"]))
    return d


def job(tmp_path: Path, ours: list[float] = (99.0, 100.0, 101.0), **kw: object) -> Path:
    """A job's out/: stock, stock again and the step record, ours 1% slower than stock unless told."""
    out = tmp_path / "out"
    corpus_arm(out, "compiled", [100.0, 101.0, 102.0])
    corpus_arm(out, "compiled-again", [100.5, 101.0, 101.5])
    corpus_arm(out, "record", list(ours), trace=True, **kw)
    return out


def run(out: Path, *argv: str) -> tuple[int, str, dict]:
    rc = P.main(["report", str(out), "--load", LOAD, *ARMS, *argv])
    return rc, (out / "prod_repro.txt").read_text(), json.loads((out / "prod_repro.json").read_text())


def test_a_replayed_window_prints_the_settings_and_the_three_lines(tmp_path: Path, capsys: pytest.CaptureFixture):
    out = job(tmp_path)
    replayed(out)
    rc, txt, r = run(out)
    assert rc == 0 and capsys.readouterr().out == txt and r["failed"] == []
    assert txt.startswith("prod_repro: load burst; stock compiled, stock again compiled-again, ours record; run r-1, commit "
                          + "c" * 40 + f"\nmodel: repo@rev, as the benchmark names it; the checkpoint served: {out / 'ck'}\n")
    # how far each arm and the replay got comes first, from their own records
    assert ("what ran:\n"
            "  compiled: up in 12.5 s; the load served 3 times, 6 requests, every one finished\n"
            "  compiled-again: up in 12.5 s; the load served 3 times, 6 requests, every one finished\n"
            "  record: up in 12.5 s; the load served 3 times, 6 requests, every one finished\n"
            "  the replay: ended in 0 s, its engine ready in 9 s; the served window took 0 s\n"
            "settings of record, with compiled's beside:\n") in txt
    assert ("tokens: reproduced: 2 requests replayed, 4 served tokens compared with 4 replayed, 0 differ in 0 requests\n"
            "  2 engine steps, 0 with other rows than the served schedule; 1 requests began from a prefix-cache hit, the "
            "longest 16 tokens\n"
            "  served compiled, replayed compiled; the window's requests: 2 greedy\n"
            "  the window: every request of the load's rep") in txt
    # the cost is the benchmark's tax of the arms' output tokens a second over reps, with its interval and its floor
    ref, mine = (S.summary(x) for x in ([100.0, 101.0, 102.0], [99.0, 100.0, 101.0]))
    want = S.tax(ref, mine)
    assert r["cost"]["value"] == pytest.approx(want["value"]) and r["cost"]["ci95"] == pytest.approx(want["ci95"])
    assert r["cost"]["how"]["client"] == "corpus" and r["cost"]["noise_floor"]["arm"] == "compiled-again"
    assert f"cost: record over compiled {100 * want['value']:+.2f}% of stock's serving time [" in txt
    assert "]; under 1%: cannot tell\n  95% interval over 3 and 3 reps, at 1000 requests a second, which stock did not keep up with" in txt
    assert "\n  noise floor: compiled-again over compiled" in txt
    assert "advice: not measured: no trace directory of a certified serve was given (ADVICE_TRACE=DIR)" in txt
    assert txt.endswith("prod_repro: 2 of 3 lines measured\n")


def test_a_replayed_token_that_differs_fails_the_job_naming_the_token_and_the_step(tmp_path: Path):
    out = job(tmp_path)
    b = rid(f"{LOAD}1", 1)
    replayed(out, tokens={b: [7, 9]})
    rc, txt, r = run(out)
    assert rc == 1 and r["tokens"]["reproduces"] is False
    # the first token that differs: its request and position, the ids on each side, and the step that emitted it
    assert ("tokens: FAILED: 2 requests replayed, 4 served tokens compared with 4 replayed, 1 differ in 1 requests\n"
            f"  the first token that differs: request {b} position 1: served id 1, replayed id 9; serving emitted it in "
            "step 1 of the window\n"
            f"  request {b} position 1 (served 2 tokens, replayed 2; 1 positions differ)\n") in txt
    assert r["tokens"]["differing"][0] | {"served_id": 1, "replayed_id": 9, "step": 1} == r["tokens"]["differing"][0]
    assert r["failed"] == [f"request {b} position 1 (served 2 tokens, replayed 2; 1 positions differ)"]
    assert txt.endswith("prod_repro: 2 of 3 lines measured; FAILED: the replay differs from what was served\n")
    # of two requests that differ, the first is the one whose token serving emitted in the earlier step
    shutil.rmtree(out / "replay-compiled")
    a = rid(f"{LOAD}1", 0)
    replayed(out, tokens={a: [7, 5], b: [3, 1]})
    _, txt, r = run(out)
    assert [d["request"] for d in r["tokens"]["differing"]] == [b, a]
    assert f"the first token that differs: request {b} position 0: served id 7, replayed id 3; serving emitted it in step 0" in txt


def test_a_replay_that_stops_short_or_runs_another_schedule_fails_too(tmp_path: Path):
    out = job(tmp_path)
    a, b = rid(f"{LOAD}1", 0), rid(f"{LOAD}1", 1)
    replayed(out, tokens={a: [7]})
    rc, txt, _ = run(out)
    assert rc == 1 and f"request {a} position 1 (served 2 tokens, replayed 1; 1 positions differ)" in txt
    assert f"the first token that differs: request {a} position 1: served id 0, the replay ended before it; serving" in txt
    shutil.rmtree(out / "replay-compiled")
    # the same tokens from other rows: the replay did not run the served schedule, and the first step says how
    other = [{"step": 0, "rows": [[a, 0, 50], [b, 0, 50]]}, {"step": 1, "rows": [[a, 50, 1], [b, 50, 1]]}]
    replayed(out, steps=other)
    rc, txt, r = run(out)
    assert rc == 1 and r["tokens"]["tokens_differing"] == 0
    assert "1 of 2 steps' rows differ from the served schedule, the first step 0" in txt
    assert (f'  the first step that differs: step 0 of the window\'s 2: serving ran [["{b}",16,34]] where the replay ran '
            f'[["{b}",0,50]] (a row: request, first position, tokens; the first 3 of each side\'s own)\n') in txt
    assert r["tokens"]["first_step"] == {"step": 0, "steps": 2, "served": [[b, 16, 34]], "replayed": [[b, 0, 50]],
                                         "replay_kept_steps": True}
    # a replay that ran fewer steps, and one that kept none
    shutil.rmtree(out / "replay-compiled")
    replayed(out, steps=[{"step": 0, "rows": [[a, 0, 50], [b, 16, 34]]}])
    txt = run(out)[1]
    assert "the first step that differs: step 1 of the window's 2: serving ran" in txt
    assert "where the replay ran no such step" in txt
    shutil.rmtree(out / "replay-compiled")
    replayed(out, steps=other, keep_steps=False)
    assert "where the replay kept no steps to say what it ran" in run(out)[1]


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


def test_what_ran_names_the_arm_that_failed_the_one_that_stopped_and_a_replay_that_did_not_end(tmp_path: Path):
    out = tmp_path / "out"
    corpus_arm(out, "compiled", [], error="RuntimeError: the server exited 1 before it was ready", startup_s=None)
    corpus_arm(out, "record", [99.0], trace=True, stopped={"before": LOAD, "rep": 2, "need_s": 310.0, "stop_at": "11:20Z"})
    (out / "replay-compiled.log").write_text("Traceback (most recent call last):\n  ...\nSystemExit: steps [3] hold rows of another load\n\n")
    rc = P.main(["report", str(out), "--load", LOAD, "--again", "", "--stock", "compiled", "--ours", "record", "--expect-replay"])
    txt, r = (out / "prod_repro.txt").read_text(), json.loads((out / "prod_repro.json").read_text())
    log = out / "replay-compiled.log"
    assert ("prod_repro: load burst; stock compiled, stock again not served, ours record; run r-1, commit " + "c" * 40) in txt
    assert ("what ran:\n"
            "  compiled: FAILED: RuntimeError: the server exited 1 before it was ready (its record's error; its server's "
            "log: compiled/server/server.log); NOTHING of the load was served\n"
            "  record: STOPPED before burst rep 2 (--stop-at 11:20Z: the pass needed 310.0 s); up in 12.5 s; the load "
            "served 1 time, 2 requests, every one finished\n"
            f"  the replay: DID NOT END; its log ends: SystemExit: steps [3] hold rows of another load ({log})\n") in txt
    # a replay that was asked for and did not end fails the job, and the tokens line says where it stopped
    assert rc == 1 and r["failed"] == ["a replay was asked for and the tokens line is not measured"]
    assert (f"tokens: not measured: the replay did not end (no replay.json in {out / 'replay-compiled'}); its log ends: "
            "SystemExit: steps [3] hold rows of another load") in txt
    assert "cost: not measured: compiled did not serve load burst" in txt
    assert txt.endswith("prod_repro: 0 of 3 lines measured; FAILED: a replay was asked for and the tokens line is not "
                        "measured; not served whole: compiled, record\n")
    assert r["ran"]["arms"]["compiled"] | {"started": True, "ok": False, "passes": 0} == r["ran"]["arms"]["compiled"]
    # an arm with no record never started; a replay with no log never did either; unfinished requests are counted
    shutil.rmtree(out / "compiled")
    log.unlink()
    meta = json.loads((out / "record/meta.json").read_text())
    (out / "record/meta.json").write_text(json.dumps(meta | {"stopped": None, "loads": [
        {"load": LOAD, "rep": 1, "requests": 2, "unfinished": 1}]}))
    rc = P.main(["report", str(out), "--load", LOAD, "--again", "", "--stock", "compiled", "--ours", "record", "--expect-replay"])
    txt = (out / "prod_repro.txt").read_text()
    assert rc == 1 and "  compiled: NOT SERVED: the job holds no record of it (no compiled/meta.json): it did not start\n" in txt
    assert "  record: up in 12.5 s; the load served 1 time, 2 requests, 1 UNFINISHED\n" in txt
    assert "  the replay: DID NOT END: it left no log, so it did not start\n" in txt
    assert f"tokens: not measured: the replay did not end (no replay.json in {out / 'replay-compiled'}); it left no log\n" in txt
    # the job with no replay asked for: the same files are no failure
    rc = P.main(["report", str(out), "--load", LOAD, "--again", "", "--stock", "compiled", "--ours", "record"])
    assert rc == 0 and "  the replay: none was asked for\n" in (out / "prod_repro.txt").read_text()
    # ours never served: every line says so and the report stands
    shutil.rmtree(out / "record")
    rc = P.main(["report", str(out), "--load", LOAD, "--again", "", "--stock", "compiled", "--ours", "record", "--expect-replay"])
    txt = (out / "prod_repro.txt").read_text()
    assert rc == 1 and "tokens: not measured: record was not served\n" in txt and "  prefix caching: not read (record was not served)" in txt
    assert "model: not read, as the benchmark names it; the checkpoint served: not read\n" in txt


def test_stock_served_once_has_no_noise_floor(tmp_path: Path):
    out = tmp_path / "out"
    corpus_arm(out, "compiled", [100.0, 101.0, 102.0])
    corpus_arm(out, "record", [99.0, 100.0, 101.0], trace=True)
    replayed(out)
    rc, txt, r = run(out, "--again", "")
    assert rc == 0 and r["arms"] == {"stock": "compiled", "again": None, "ours": "record", "served": ["compiled", "record"]}
    assert "stock compiled, stock again not served, ours record;" in txt
    assert r["cost"]["measured"] and r["cost"]["noise_floor"] is None
    assert "\n  noise floor: not measured (stock was not served a second time)\n" in txt
    assert list(r["ran"]["arms"]) == ["compiled", "record"] and txt.endswith("prod_repro: 2 of 3 lines measured\n")


def test_files_the_benchmarks_readers_refuse_end_their_lines_and_not_the_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = job(tmp_path)
    replayed(out)
    real = S.arm_metrics

    def refuses(d: Path, *a: object) -> dict:
        if d.name == "compiled":
            raise SystemExit("compiled/burst/rep-1: vLLM's client counted this pass's goodput against another SLO")
        return real(d, *a)

    monkeypatch.setattr(S, "arm_metrics", refuses)
    rc, txt, r = run(out)
    why = ("compiled's metrics can't be read from the job's files: SystemExit: compiled/burst/rep-1: vLLM's client counted "
           "this pass's goodput against another SLO")
    assert rc == 0 and r["ran"]["unread"] == {"compiled": why} and f"\n  {why}\n" in txt
    assert "  prefix caching: on (the engine's configuration; the cache served 40.0% of the load's prompt tokens); compiled's files can't be read\n" in txt
    assert "cost: not measured: compiled did not serve load burst" in txt and "tokens: reproduced" in txt
    # a record that is not JSON, a step trace cut short: the part that reads it says so
    monkeypatch.setattr(S, "arm_metrics", real)
    (out / "record/server/step_trace.9.jsonl.gz").write_bytes(b"not gzip")
    rc, txt, r = run(out)
    assert rc == 0 and "record's settings can't be read from the job's files: BadGzipFile" in r["ran"]["unread"]["record"]
    assert "tokens: not measured: the served window or the replay's tokens can't be read back: BadGzipFile" in txt
    assert "  prefix caching: not read (record's files can't be read)\n" in txt and "cost: record over compiled" in txt


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
    assert ("  context limit: 4,096 tokens (the engine took it; --max-model-len 4096), BELOW the model's own 131,072 "
            "(config.json max_position_embeddings); compiled: 4,096 tokens (--max-model-len 4096), BELOW the model's own "
            "131,072 (config.json max_position_embeddings)\n") in txt
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
    assert r["settings"]["record"]["speculative_decoding"] == {"state": "on", "by": " ".join(spec),
                                                               "draft_accepted_share": None, "tokens_per_draft": None}
    assert ("  speculative decoding: on (--speculative-config.method mtp --speculative-config.num_speculative_tokens 5); "
            "compiled: on (") in txt
    # the KV cache's type is the launch line's, and an arm that does not share it is not at the same settings
    assert ("  KV cache: fp8 (--kv-cache-dtype); DIFFERS; compiled: the engine's default (no --kv-cache-dtype in the "
            "launch line)\n") in txt
    assert r["cost"]["settings_differ"] == ["the KV cache's type"]
    assert P.flags(["--speculative-config={}", "--seed", "0"], ("--speculative",)) == "--speculative-config={}"
    assert r["settings"]["record"]["context_limit"] == {
        "tokens": 131072, "given": "131072", "engine": False,
        "model": {"tokens": 131072, "key": "max_position_embeddings"}, "models_own": True}
    assert ("  context limit: 131,072 tokens (--max-model-len 131072), the model's own 131,072 (config.json "
            "max_position_embeddings); compiled: 131,072 tokens (") in txt
    # an arm with neither a step trace nor a counted hit: nothing confirms vLLM's default
    for rep in (out / "record" / LOAD).glob("rep-*/meta.json"):
        rep.write_text(json.dumps(json.loads(rep.read_text()) | {"schedule": None}))
    (out / "ck" / "config.json").unlink()
    _, txt, r = run(out)
    assert r["settings"]["record"]["prefix_caching"]["state"] == "not confirmed"
    assert ("131,072 tokens (--max-model-len 131072), the model's own limit not read (no config.json at the checkpoint "
            "on this machine)") in txt


def test_a_limit_left_to_vllm_is_the_engines_own_and_the_engine_says_what_the_launch_line_does_not(tmp_path: Path):
    out = tmp_path / "out"
    corpus_arm(out, "compiled", [100.0, 101.0], argv=("--max-model-len", "auto"))
    corpus_arm(out, "record", [100.0, 101.0], argv=("--max-model-len", "auto"), trace=True, header={
        "max_model_len": 202752, "kv_cache_dtype": "fp8", "speculative": {"method": "mtp", "num_speculative_tokens": 5}})
    rc, txt, r = run(out)
    # `auto` is resolved by the engine alone: ours kept its engine's configuration, stock kept none
    assert rc == 0 and r["settings"]["record"]["context_limit"] | {"tokens": 202752, "given": "auto", "engine": True} == \
        r["settings"]["record"]["context_limit"]
    assert r["settings"]["compiled"]["context_limit"]["tokens"] is None
    assert ("  context limit: 202,752 tokens (the engine took it; --max-model-len auto), the model's own 131,072 "
            "(config.json max_position_embeddings); compiled: not resolved (--max-model-len auto; the arm kept no step "
            "trace to say what its engine took)\n") in txt
    # two launch lines that both leave it to vLLM are at the same limit; the other settings are the engine's word
    assert "the context limit" not in r["cost"]["settings_differ"]
    assert "  KV cache: fp8 (the engine's configuration); DIFFERS; compiled: the engine's default (" in txt
    assert "  speculative decoding: on (the engine's configuration: method mtp, num_speculative_tokens 5); DIFFERS; " in txt
    mine = r["settings"]["record"]
    other = {**mine, "context_limit": {**mine["context_limit"], "tokens": None, "given": "65536"}}
    assert P.settings_differ(mine, other) == ["the context limit"]
    assert P.settings_differ(mine, {**mine, "context_limit": {**mine["context_limit"], "tokens": 4096}}) == ["the context limit"]
    # the drafts' acceptance is the benchmark's counters', beside the launch line that turned them on
    reps = {"loads": {LOAD: [{"prompt_cached_share": 0.4, "draft_accepted_share": 0.5, "tokens_per_draft": 3.5}]}}
    spec = P.settings(out / "record", reps, LOAD)["speculative_decoding"]
    assert spec["by"].endswith("; 50.0% of its draft tokens accepted, 3.50 tokens a draft") and spec["tokens_per_draft"] == 3.5


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


def bench_arm(out: Path, name: str, passes: dict[str, list[float]], ids: str | None = None) -> None:
    """An arm's directory as `serve_tax.py run` leaves it under vLLM's client; with ``ids``, rep 1 of that pass also
    holds what the replay reads: each request's ids in client.jsonl's form, beside the arm's step trace and the plan."""
    d = out / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({
        "arm": name, "parent": None, "what": name, "model_id": "glm53", "model": "repo@rev", "client": "vllm-bench", "rates": {},
        "run_id": "r-1", "commit": "c" * 40, "server_argv": ["--max-model-len", "4096"], "startup_s": 300.0,
        "passes": {p: {"load": p.rpartition("@")[0], "concurrency": int(p.rpartition("@")[2])} for p in passes},
        "loads": [{"load": p, "rep": i, "rate": None, "requests": 200, "unfinished": 0}
                  for p, toks in passes.items() for i in range(1, len(toks) + 1)]}))
    for p, toks in passes.items():
        for i, tok in enumerate(toks, 1):
            tag = f"{p}-r{i}"
            (d / p / f"rep-{i}").mkdir(parents=True)
            (d / p / f"rep-{i}" / "result.json").write_text(json.dumps(result(tok)))
            (d / p / f"rep-{i}" / "meta.json").write_text(json.dumps(
                {"tag": tag, "rate": None, "concurrency": int(p.rpartition("@")[2]), "unfinished": 0, "t0_unix": 1000.0,
                 "wall_s": 105.0}))
            if (p, i) == (ids, 1):
                (d / p / "rep-1" / "client.jsonl").write_text("".join(json.dumps(
                    {"id": f"{tag}-{k:05d}", "i": k, "status": 200, "output_ids": [7, k]}) + "\n" for k in range(2)))
                step_trace(out, d, tag, {"enable_prefix_caching": True, "max_model_len": 4096})


def test_by_vllms_client_the_cost_is_the_benchmarks_paired_cost_and_nothing_is_replayed(tmp_path: Path):
    out = tmp_path / "out"
    stock = [300.0, 301.0, 299.0, 300.5, 299.5, 300.0]
    bench_arm(out, "compiled", {"chat@8": [200.0] * 6, "chat@16": stock, "chat@32": [290.0] * 6})
    bench_arm(out, "compiled-again", {"chat@16": [x + 0.3 for x in stock]})
    bench_arm(out, "record", {"chat@16": [x / 1.004 for x in stock]})
    rc, txt, r = P.main(["report", str(out), "--load", "chat", *ARMS]), (out / "prod_repro.txt").read_text(), json.loads(
        (out / "prod_repro.json").read_text())
    want = S.serving_cost(S.arm_metrics(out / "compiled"), S.arm_metrics(out / "record"), "chat")
    assert rc == 0 and r["cost"]["value"] == pytest.approx(0.004) == pytest.approx(want["value"]) and r["cost"]["ci95"] == want["ci95"]
    assert r["cost"]["how"] == {"client": "vllm-bench", "rung": 16, "reps": 6, "saturated": True, "by": r["cost"]["how"]["by"]}
    assert r["cost"]["verdict"] == "meets"
    assert ("cost: record over compiled +0.40% of stock's serving time [+0.40%, +0.40%]; under 1%: yes\n"
            "  95% interval over 6 paired reps, at concurrency 16, where stock's rungs show it saturated: ") in txt
    assert ("tokens: not measured: no replay ran: a pass sent by vLLM's client keeps neither its prompts' ids nor the "
            "served ids") in txt


def test_a_pass_vllms_client_sent_is_replayed_where_ours_kept_its_ids(tmp_path: Path):
    out = tmp_path / "out"
    stock = [300.0, 301.0]
    bench_arm(out, "prod", {"chat@16": stock})
    bench_arm(out, "prod-record", {"chat@16": [x / 1.004 for x in stock]}, ids="chat@16")
    argv = ["report", str(out), "--load", "chat", "--stock", "prod", "--again", "", "--ours", "prod-record",
            "--replayed", "chat@16", "--expect-replay"]
    replayed(out, "prod-record", load="chat@16")
    rc, txt = P.main(argv), (out / "prod_repro.txt").read_text()
    r = json.loads((out / "prod_repro.json").read_text())
    # the cost is of the load, the replay of one pass of it, named as the benchmark names it
    assert rc == 0 and r["tokens"]["window"] | {"arm": "prod-record", "load": "chat@16", "rep": 1} == r["tokens"]["window"]
    assert "tokens: reproduced: 2 requests replayed, 4 served tokens compared with 4 replayed, 0 differ in 0 requests\n" in txt
    assert "cost: prod-record over prod +0.40% of stock's serving time" in txt and r["cost"]["how"]["client"] == "vllm-bench"
    assert "  prod-record: up in 300 s; the load served 2 times, 400 requests, every one finished\n" in txt
    assert txt.endswith("prod_repro: 2 of 3 lines measured\n")
    # a replay of the load's name alone is of another window
    assert P.main([a for a in argv if a not in ("--replayed", "chat@16")]) == 1
    assert "tokens: not measured: the replay in" in (out / "prod_repro.txt").read_text()
    # a token that differs fails it, with the first token's ids
    shutil.rmtree(out / "replay-compiled")
    replayed(out, "prod-record", load="chat@16", tokens={rid("chat@16-r1", 0): [7, 4]})
    assert P.main(argv) == 1
    assert (f"the first token that differs: request {rid('chat@16-r1', 0)} position 1: served id 0, replayed id 4; serving "
            "emitted it in step 1 of the window") in (out / "prod_repro.txt").read_text()


# --------------------------------------------------------------------------------------------------------- the advice

#: a stand-in for the tree's reader of a certified serve's advice by class: `report` gives what the trace directory's
#: row.json says, and `OF` each class's events
READER = """import json
from pathlib import Path

OF = {"hits": ("lookups_hit", "admissions_cut"), "admission": ("waited",), "aborts": ()}


def report(label, path, stop_ids, T=600, ingress="client"):
    row = json.loads((Path(path) / "row.json").read_text())
    return {**row, "stops_seen": list(stop_ids)}
"""


def certified(trace: Path, **over: object) -> None:
    """A certified serve's trace directory, as the stand-in reader reads it: the reader's row for it."""
    trace.mkdir(exist_ok=True)
    row = {"trace": "advice", "T": 600, "tick": {"tau_b": 1}, "prefix_caching": True, "requests": 200, "steps": 900,
           "windows": 2, "round_trip": True, "refused": None, "first_step_that_differs": None,
           "classes": {"hits": {"standing": "charged", "rules": []}, "admission": {"standing": "closed under the tick", "rules": []},
                       "aborts": {"standing": "no field", "rules": []}},
           "total": {"charge": 700, "classes": {"hits": 700, "admission": 0},
                     "events": {"lookups_hit": 120, "admissions_cut": 3, "waited": 0}}}
    (trace / "row.json").write_text(json.dumps(row | over))


def test_the_advice_line_is_the_readers_split_of_the_certified_serves_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out, trace, reader = job(tmp_path), tmp_path / "trace", tmp_path / "by_class.py"
    replayed(out)
    reader.write_text(READER)
    monkeypatch.setattr(P, "BY_CLASS", reader)
    certified(trace)
    rc, txt, r = run(out, "--advice-trace", str(trace), "--advice-stops", "154820,154827")
    assert rc == 0 and r["advice"]["measured"] and r["advice"]["bits_a_request"] == 3.5
    assert ("advice: 3.500 bits a request: 700 bits over 200 requests, under the certified tick; 2 windows of T = 600 buckets; "
            "it decodes back to the trace\n"
            "  hits: 700 bits, 3.500 a request (charged; lookups hit 120, admissions cut 3)\n"
            "  admission: 0 bits, not exercised: none of its events arose (closed under the tick; waited 0)\n"
            "  aborts: no field in the code\n"
            f"  the trace: {trace}; stop ids 154820,154827 (ADVICE_STOPS)\n") in txt
    assert txt.endswith("prod_repro: 3 of 3 lines measured\n")
    # the reader's whole row is kept beside the report, under the stop ids it was read with
    kept = json.loads((out / "advice_by_class.json").read_text())
    assert kept["stops"] == [154820, 154827] == kept["row"]["stops_seen"] and kept["trace"] == str(trace)
    # without ADVICE_STOPS the stop ids are the served checkpoint's own
    (out / "ck" / "generation_config.json").write_text(json.dumps({"eos_token_id": [154820, 154827, 154829]}))
    _, txt, r = run(out, "--advice-trace", str(trace))
    assert r["advice"]["stops"] == [154820, 154827, 154829]
    assert f"stop ids 154820,154827,154829 (eos_token_id of {out / 'ck' / 'generation_config.json'})" in txt
    (out / "ck" / "generation_config.json").unlink()
    _, txt, r = run(out, "--advice-trace", str(trace))
    assert ("advice: not measured: the registered stop ids are not known (this machine has no generation_config.json of "
            "the checkpoint the job served): ADVICE_STOPS=ID,..., EOS first\n") in txt
    # a trace the reader refuses, one whose advice does not decode back, and a serve off the tick
    certified(trace, error="the trace holds speculative decoding, which the advice code has no field for")
    _, txt, r = run(out, "--advice-trace", str(trace), "--advice-stops", "1")
    assert (f"advice: not measured: the reader refuses the trace {trace}: the trace holds speculative decoding, which the "
            "advice code has no field for\n") in txt and txt.endswith("prod_repro: 2 of 3 lines measured\n")
    certified(trace, round_trip=False, refused="window 1: a symbol past the book", first_step_that_differs={"step": 41})
    _, txt, _ = run(out, "--advice-trace", str(trace), "--advice-stops", "1")
    assert (f"advice: not measured: the advice of {trace} does not decode back to it: the decoder refuses it: window 1: a "
            "symbol past the book; its replay's schedule differs from the engine's from step 41\n") in txt
    certified(trace, tick=None)
    assert "admitted on arrival, NOT under the certified tick" in run(out, "--advice-trace", str(trace), "--advice-stops", "1")[1]
    # no directory, a directory the reader can't read, and a tree without the reader
    (trace / "row.json").unlink()
    _, txt, _ = run(out, "--advice-trace", str(trace), "--advice-stops", "1")
    assert "advice: not measured: the advice can't be read from the job's files: FileNotFoundError" in txt
    _, txt, _ = run(out, "--advice-trace", str(tmp_path / "nowhere"), "--advice-stops", "1")
    assert f"advice: not measured: ADVICE_TRACE={tmp_path / 'nowhere'} is not a directory on this machine\n" in txt
    monkeypatch.setattr(P, "BY_CLASS", tmp_path / "no_reader.py")
    _, txt, _ = run(out, "--advice-trace", str(trace), "--advice-stops", "1")
    assert (f"advice: not measured: this tree has no reader of a certified serve's advice by class (no "
            f"{tmp_path / 'no_reader.py'}); the trace is {trace}\n") in txt


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
{ echo "argv=$*"; for k in ARMS RATES REPS REQUESTS SPAN_S REPLAY REPLAY_FROM REPLAY_MODES MODEL_ID CLIENT MAX_MODEL_LEN \\
    PASS_PROMPTS STOP_AT; do
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
    base = {k: v for k, v in os.environ.items() if k not in (
        "RESEARCH_RUN_DIR", "LOAD", "REPS", "REQUESTS", "MODEL_ID", "STOCK", "AGAIN", "OURS", "RATE", "SPAN_S", "TARGET",
        "MAX_MODEL_LEN", "REPLAY", "REPLAY_FROM", "REPLAY_MODES", "PASS_PROMPTS", "STOP_AT", "ADVICE_TRACE", "ADVICE_STOPS")}
    return subprocess.run(["bash", str(script), *argv], capture_output=True, text=True, check=False, env={
        **base, "VENV": str(Path(sys.executable).parents[1]), **({"RESEARCH_RUN_DIR": str(run_dir)} if run_dir else {}), **env})


def started(run_dir: Path) -> dict[str, str]:
    return dict(x.split("=", 1) for x in (run_dir / "started").read_text().splitlines())


def test_the_runner_serves_the_three_arms_a_burst_and_replays_ours_then_reports(tmp_path: Path):
    script, run_dir = runner(tmp_path), tmp_path / "run"
    replayed(job(run_dir))
    p = bash(script, *ARMS, "--max-model-len", "8192", run_dir=run_dir)
    assert p.returncode == 0, p.stdout + p.stderr
    assert started(run_dir) == {
        "argv": "", "ARMS": "compiled compiled-again record", "RATES": "burst=1000", "REPS": "3", "REQUESTS": "800",
        "SPAN_S": "unset", "REPLAY": "burst", "REPLAY_FROM": str(run_dir / "out/record"), "REPLAY_MODES": "compiled",
        "MODEL_ID": "moonlight", "CLIENT": "corpus", "MAX_MODEL_LEN": "8192", "PASS_PROMPTS": "unset", "STOP_AT": "unset"}
    assert "tokens: reproduced: 2 requests replayed" in p.stdout and p.stdout.endswith("prod_repro: 2 of 3 lines measured\n")
    assert (run_dir / "out/prod_repro.txt").read_text() in p.stdout
    # an arm or a replay that failed fails the job, and the report still says what the files hold
    p = bash(script, *ARMS, run_dir=run_dir, STAND_IN_RC="1")
    assert p.returncode == 1 and "the serving cost benchmark's runner ended 1" in p.stdout and "tokens: reproduced" in p.stdout
    # a replayed token that differs fails a job whose serve and replay ended well
    shutil.rmtree(run_dir / "out/replay-compiled")
    replayed(run_dir / "out", tokens={rid(f"{LOAD}1", 0): [8, 0]})
    p = bash(script, *ARMS, run_dir=run_dir)
    assert p.returncode == 1 and f"request {rid(f'{LOAD}1', 0)} position 0: served id 7, replayed id 8" in p.stdout
    # a replay that left nothing fails it too: the corpus client's job always asks for one, unless told not to
    shutil.rmtree(run_dir / "out/replay-compiled")
    p = bash(script, *ARMS, run_dir=run_dir)
    assert p.returncode == 1 and "the replay: DID NOT END: it left no log, so it did not start" in p.stdout
    p = bash(script, *ARMS, "--replay", "none", "--again", "none", run_dir=run_dir)
    s = started(run_dir)
    assert p.returncode == 0 and (s["REPLAY"], s["REPLAY_FROM"], s["ARMS"]) == ("unset", "unset", "compiled record")
    assert "the replay: none was asked for" in p.stdout and "stock again not served" in p.stdout


def test_the_runner_hands_vllms_client_the_load_and_asks_for_no_replay(tmp_path: Path):
    script, run_dir = runner(tmp_path), tmp_path / "run"
    stock = [300.0, 301.0, 299.0, 300.5, 299.5, 300.0]
    for name in ("compiled", "compiled-again", "record"):
        bench_arm(run_dir / "out", name, {"chat@16": stock})
    p = bash(script, *ARMS, "--client", "vllm-bench", "--load", "chat=16", "--model", "glm53", run_dir=run_dir)
    assert p.returncode == 0, p.stdout + p.stderr
    s = started(run_dir)
    assert s["argv"] == "--model glm53 --arms compiled compiled-again record --loads chat=16 --reps 6"
    assert (s["REPLAY"], s["REPLAY_FROM"], s["CLIENT"]) == ("unset", "unset", "vllm-bench")
    assert "tokens: not measured: no replay ran" in p.stdout and "cost: record over compiled +0.00%" in p.stdout


def test_the_runner_asks_for_the_replay_of_a_named_pass_of_vllms_client_and_reads_the_advice_trace(tmp_path: Path):
    script, run_dir = runner(tmp_path), tmp_path / "run"
    bench_arm(run_dir / "out", "prod", {"chat@16": [300.0]})
    bench_arm(run_dir / "out", "prod-record", {"chat@16": [299.0]}, ids="chat@16")
    argv = ("--client", "vllm-bench", "--model", "glm53", "--stock", "prod", "--again", "none", "--ours", "prod-record",
            "--load", "chat=16", "--reps", "1", "--replay", "chat@16", "--pass-prompts", "chat=8", "--stop-at", "2030-01-01T00:00Z")
    replayed(run_dir / "out", "prod-record", load="chat@16")
    p = bash(script, *argv, run_dir=run_dir, ADVICE_TRACE=str(tmp_path / "nowhere"))
    assert p.returncode == 0, p.stdout + p.stderr
    s = started(run_dir)
    assert s["argv"] == "--model glm53 --arms prod prod-record --loads chat=16 --reps 1"
    assert (s["REPLAY"], s["REPLAY_FROM"], s["REPLAY_MODES"]) == ("chat@16", str(run_dir / "out/prod-record"), "compiled")
    assert (s["PASS_PROMPTS"], s["STOP_AT"]) == ("chat=8", "2030-01-01T00:00Z")
    assert "stock prod, stock again not served, ours prod-record" in p.stdout and "tokens: reproduced: 2 requests replayed" in p.stdout
    assert f"advice: not measured: ADVICE_TRACE={tmp_path / 'nowhere'} is not a directory on this machine" in p.stdout
    # the serve ended well and the replay left nothing: the job fails, and says the replay did not end
    shutil.rmtree(run_dir / "out/replay-compiled")
    (run_dir / "out/replay-compiled.log").write_text("serve-tax: no arrival of load chat@16 rep 1\n")
    p = bash(script, *argv, run_dir=run_dir)
    assert p.returncode == 1 and "the replay: DID NOT END; its log ends: serve-tax: no arrival of load chat@16 rep 1" in p.stdout
    assert p.stdout.endswith("FAILED: a replay was asked for and the tokens line is not measured\n")
    # read again without a GPU, the earlier job fails the same way
    p = bash(script, "--report", str(run_dir / "out"), *argv[4:12], "--replay", "chat@16")
    assert p.returncode == 1 and "FAILED: a replay was asked for and the tokens line is not measured" in p.stdout


def test_the_runner_reads_an_earlier_job_again_and_refuses_what_it_cannot_run(tmp_path: Path):
    script = runner(tmp_path)
    out = job(tmp_path)
    replayed(out)
    p = bash(script, "--report", str(out), "--load", LOAD, *ARMS)
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
