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
    token as served, by an engine under nothing of the arm's (``served_under``)."""
    w = R.window(out / ours, load, 1)
    got = {"steps": steps if steps is not None else [dict(s) for s in w["steps"]],
           "tokens": {r["engine_id"]: list(r["served"]) for r in w["requests"]} | (tokens or {})}
    d = out / "replay-compiled"
    d.mkdir()
    (d / "replay.json").write_text(json.dumps({
        "arm_dir": str(out / ours), "served_run_id": "r-1", "run_id": "r-1", "load": load, "rep": 1, "mode": "compiled",
        "served_mode": "compiled", "served_window_s": 0.2, "replay_wall_s": 0.3, "engine_ready_s": 9.0,
        "served_under": None, **R.compare(w, got), **over}))
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
    # first the context limit the arms were started at
    assert txt.startswith("prod_repro: context limit 4,096 tokens, given outright to record and compiled\n"
                          "prod_repro: load burst; stock compiled, stock again compiled-again, ours record; run r-1, commit "
                          + "c" * 40 + f"\nmodel: repo@rev, as the benchmark names it; the checkpoint served: {out / 'ck'}\n")
    # how far each arm and the replay got comes first, from their own records
    assert ("what ran:\n"
            "  compiled: up in 12.5 s; the load served 3 times, 6 requests, every one finished\n"
            "  compiled-again: up in 12.5 s; the load served 3 times, 6 requests, every one finished\n"
            "  record: up in 12.5 s; the load served 3 times, 6 requests, every one finished\n"
            "  the replay: ended in 0 s, its engine ready in 9 s; the served window took 0 s\n"
            "  the replay's engine: vLLM's own (record serves with no other vLLM plugin of ours than its record)\n"
            "settings of record, with compiled's beside:\n") in txt
    assert ("tokens: reproduced: 2 requests replayed, 4 served tokens compared with 4 replayed, 0 differ in 0 requests\n"
            "  2 engine steps, 0 with other rows than the served schedule; 1 requests began from a prefix-cache hit, the "
            "longest 16 tokens\n"
            "  served compiled, replayed compiled; the window's requests: 2 greedy\n"
            "  the window: every request of the load's rep") in txt
    # the cost is the benchmark's tax of the arms' output tokens a second over reps, with its interval and its floor
    ref, mine = (S.summary(x) for x in ([100.0, 101.0, 102.0], [99.0, 100.0, 101.0]))
    want = S.tax(ref, mine)
    assert r["cost"]["value"] == pytest.approx(want["value"]) and r["cost"]["passes"]["ci95"] == pytest.approx(want["ci95"])
    assert r["cost"]["how"]["client"] == "corpus" and r["cost"]["noise_floor"]["arm"] == "compiled-again"
    # one lifetime an arm: no interval over lifetime pairs, so no verdict; the interval printed is over the passes
    assert (f"cost: record over compiled {100 * want['value']:+.2f}% of stock's serving time; under 1%: not shown, each "
            "arm has one server lifetime, and the verdict reads the interval over lifetime pairs\n"
            "  passes taken as independent, 95% interval [") in txt
    assert "]: over 3 and 3 reps, at 1000 requests a second, which stock did not keep up with" in txt
    assert r["cost"]["verdict"] == "not shown" and r["cost"]["lifetimes"] == {"pairs": 1, "served": 1, "ci95": None}
    assert "\n  noise floor, beside and not added: compiled-again over compiled" in txt
    assert ", passes taken as independent\n" in txt and r["cost"]["noise_floor"]["leaves_out_zero"] is False
    assert "STOCK DIFFERS FROM STOCK" not in txt
    # this synthetic trace is not the adapter's step trace, and the scheduler replay's loader refuses it
    assert "advice: not measured: the scheduler replay refuses the pass: not a verity-vllm/step-trace/v1 trace\n" in txt
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


def end_step(out: Path, ended: bool) -> str:
    """Add a third step to the record arm's served trace, one that emits nothing: the second request scheduled once
    more, after both requests ``ended`` in the step before it or while that request still runs. The request."""
    f, b = out / "record" / "server" / "step_trace.9.jsonl.gz", rid(f"{LOAD}1", 1)
    with gzip.open(f, "rt") as lines:
        recs = [json.loads(x) for x in lines]
    recs[-1]["finished"] = {rid(f"{LOAD}1", 0): "length", b: "length"} if ended else {rid(f"{LOAD}1", 0): "length"}
    recs.append({"kind": "step", "step": 2, "rows": [[b, 51, 6]], "cache_hit": {}, "emitted": {}, "finished": {}, "t": 0.3})
    with gzip.open(f, "wt") as lines:
        lines.write("".join(json.dumps(r) + "\n" for r in recs))
    return b


def test_a_step_after_a_requests_end_that_emitted_nothing_is_counted_and_is_no_difference(tmp_path: Path):
    def served_twice(out: Path) -> list[dict]:
        return [dict(s) for s in R.window(out / "record", LOAD, 1)["steps"][:2]]

    # serving planned one more step for a request before it learned the request had ended, and dropped its output;
    # the replay, which ends with its last request, ran no such step
    out = job(tmp_path)
    end_step(out, ended=True)
    replayed(out, steps=served_twice(out))
    rc, txt, r = run(out)
    assert json.loads((out / "replay-compiled" / "replay.json").read_text()) | {
        "bit_exact": False, "steps_differing": 1, "first_differing_step": 2} == json.loads(
            (out / "replay-compiled" / "replay.json").read_text())
    assert rc == 0 and r["tokens"]["reproduces"] is True and r["failed"] == []
    assert r["tokens"] | {"steps_served": 3, "steps_differing": 0, "first_differing_step": None, "steps_after_end": [2],
                          "first_step": None, "record_disagrees": False} == r["tokens"]
    assert ("tokens: reproduced: 2 requests replayed, 4 served tokens compared with 4 replayed, 0 differ in 0 requests\n"
            "  1 step after a request's end, no token, not replayed: step 2 of the window's 3 (serving had planned each "
            "before it learned its requests had ended, and dropped what it computed: counted here, and no difference)\n"
            "  3 engine steps, 0 with other rows than the served schedule; ") in txt
    assert txt.endswith("prod_repro: 2 of 3 lines measured\n")
    # a step the replay ran that serving did not still fails, beside the counted one
    shutil.rmtree(out / "replay-compiled")
    replayed(out, steps=[*served_twice(out), {"step": 3, "rows": [[rid(f"{LOAD}1", 0), 51, 6]]}])
    rc, txt, r = run(out)
    assert rc == 1 and r["tokens"]["steps_after_end"] == [2] and r["tokens"]["steps_differing"] == 1
    assert "the first step that differs: step 3 of the window's 3: serving ran no such step where the replay ran" in txt
    # the same no-token step of a request that had not ended is a missing step, and fails
    out = job(tmp_path / "live")
    b = end_step(out, ended=False)
    replayed(out, steps=served_twice(out))
    rc, txt, r = run(out)
    assert rc == 1 and r["tokens"]["reproduces"] is False and r["tokens"]["steps_after_end"] == []
    assert "tokens: FAILED: 2 requests replayed, 4 served tokens compared with 4 replayed, 0 differ in 0 requests\n" in txt
    assert "1 of 3 steps' rows differ from the served schedule, the first step 2" in txt
    assert f'the first step that differs: step 2 of the window\'s 3: serving ran [["{b}",51,6]] where the replay ran no such step' in txt
    assert "after a request's end" not in txt
    # and so is a replay that kept no steps to say which it ran: the record's count stands
    out = tmp_path / "out"
    shutil.rmtree(out / "replay-compiled")
    replayed(out, steps=served_twice(out), keep_steps=False)
    rc, txt, r = run(out)
    assert rc == 1 and r["tokens"]["steps_differing"] == 1 and r["tokens"]["steps_after_end"] == []


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
    # where the failed arm's server left a log, the line names the first exception in it: the record's error is the
    # last thing that failed, and the cause is above it
    (out / "compiled/server").mkdir(parents=True, exist_ok=True)
    (out / "compiled/server/server.log").write_text(
        "INFO 11-12 [model.py:1] Using max model len 262144\n"
        "(EngineCore pid=7) ERROR 11-12 [core.py:9] torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 24.00 GiB.\n"
        "RuntimeError: Engine core initialization failed. See root cause above.\n")
    P.main(["report", str(out), "--load", LOAD, "--again", "", "--stock", "compiled", "--ours", "record", "--expect-replay"])
    assert ("  compiled: FAILED: RuntimeError: the server exited 1 before it was ready (its record's error; the first "
            "exception its server's log names: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 24.00 GiB. "
            "(compiled/server/server.log)); NOTHING of the load was served\n") in (out / "prod_repro.txt").read_text()
    (out / "compiled/server/server.log").write_text("INFO 11-12 [model.py:1] Using max model len 262144\n")
    P.main(["report", str(out), "--load", LOAD, "--again", "", "--stock", "compiled", "--ours", "record", "--expect-replay"])
    assert "its server's log names no exception: compiled/server/server.log" in (out / "prod_repro.txt").read_text()
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
    assert "\n  noise floor, beside and not added: not measured (stock was not served a second time)\n" in txt
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
    # a cost between arms at different settings is printed, and shows nothing of our changes
    assert "; under 1%: not shown, the arms are not at the same settings\n" in txt
    assert ("  NOT at the same settings: record and compiled differ in prefix caching, torch.compile and CUDA graphs, so "
            "the figure is not our changes' cost\n") in txt
    assert r["cost"]["settings_differ"] == ["prefix caching", "torch.compile and CUDA graphs"]
    assert r["cost"]["verdict"] == "not shown"


def test_the_value_settings_in_force_are_the_adapters_record_as_each_server_and_the_replay_kept_it(tmp_path: Path):
    """An arm's record and the replay's hold the adapter's record of the value settings in force (the benchmark's
    ``settings``): the report keeps each whole and names the variables ours's differs from stock's in and the
    replay's from ours's. A job whose directories hold none says not recorded."""
    out = job(tmp_path)
    replayed(out)
    _, txt, r = run(out)
    assert "\n  value settings in force: not recorded (no arm's record and no replay's holds the adapter's record of them)\n" in txt
    assert r["value_settings"] == {"arms": {"compiled": {"compiled": None}, "record": {"record": None}}, "replay": None,
                                   "lifetimes_differ": [], "differ": {"ours_from_stock": None, "replay_from_ours": None}}
    # the records as the benchmark makes them (`serve_tax.value_settings`): stock's server's, ours's with the canonical
    # top-k besides, and a replay's engine's, which is given another variable and lacks one the servers have
    base = {"NCCL_P2P_DISABLE": "1", "VLLM_API_KEY": "a credential's value", "HOME": "/no/value/setting"}
    served = base | {"VLLM_BATCH_INVARIANT": "0"}
    stock, ours = S.value_settings(served), S.value_settings(served | {"VERITY_DSA_TOPK": "canonical"})
    replay = S.value_settings(base | {"VERITY_DSA_TOPK": "canonical", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
    assert (len(stock["values"]), len(ours["values"]), len(replay["values"])) == (3, 4, 4)
    out = tmp_path / "kept" / "out"
    corpus_arm(out, "compiled", [100.0, 101.0, 102.0], settings=stock)
    corpus_arm(out, "compiled-again", [100.5, 101.0, 101.5], settings=stock)
    corpus_arm(out, "record", [99.0, 100.0, 101.0], trace=True, settings=ours)
    replayed(out, settings=replay)
    _, txt, r = run(out)
    assert ("\n  circuit compiled ahead of serving: no: nothing in this job compiles the settings' circuit\n"
            "  value settings in force, the adapter's record of each server's environment (serve_tax.value_settings): "
            f"compiled 3 set, SHA-512 {stock['digest'][:12]}; record 4 set, SHA-512 {ours['digest'][:12]}, differing from "
            f"compiled's in VERITY_DSA_TOPK; the replay's engine 4 set, SHA-512 {replay['digest'][:12]}, differing from "
            "record's in VLLM_BATCH_INVARIANT, VLLM_ENABLE_V1_MULTIPROCESSING\ntokens: ") in txt
    assert r["value_settings"] == {
        "arms": {"compiled": {"compiled": stock}, "record": {"record": ours}}, "replay": replay, "lifetimes_differ": [],
        "differ": {"ours_from_stock": ["VERITY_DSA_TOPK"],
                   "replay_from_ours": ["VLLM_BATCH_INVARIANT", "VLLM_ENABLE_V1_MULTIPROCESSING"]}}
    # a credential is in the record by its name alone, and the difference is in no verdict
    assert "a credential's value" not in txt + json.dumps(r) and "settings_differ" not in r["cost"]
    # a job in lifetimes: each lifetime's record is kept; an arm whose lifetimes hold different ones says so, and a
    # side with no record is not recorded and compared with nothing
    out = tmp_path / "lifetimes" / "out"
    for k, mine in enumerate((ours, replay), 1):
        corpus_arm(out, f"compiled.{k}", [100.0, 101.0], arm="compiled", lifetime=k, reps=2)
        corpus_arm(out, f"record.{k}", [99.0, 100.0], trace=True, arm="record", lifetime=k, reps=2, settings=mine)
    replayed(out, "record.1", settings=ours)
    _, txt, r = run(out, "--again", "")
    assert ("(serve_tax.value_settings): compiled not recorded; "
            f"record 4 set, SHA-512 {ours['digest'][:12]}; the replay's engine 4 set, SHA-512 {ours['digest'][:12]}, the "
            "same as record's\n"
            "  NOT ONE RECORD: record's lifetimes hold different value settings; the line above reads its first "
            "lifetime's\ntokens: ") in txt
    assert r["value_settings"]["arms"] == {"compiled": {"compiled.1": None, "compiled.2": None},
                                           "record": {"record.1": ours, "record.2": replay}}
    assert r["value_settings"]["lifetimes_differ"] == ["record"]
    assert r["value_settings"]["differ"] == {"ours_from_stock": None, "replay_from_ours": []}
    # a record that is not in the adapter's form ends this line and not the report
    (out / "record.1" / "meta.json").write_text(json.dumps(
        json.loads((out / "record.1" / "meta.json").read_text()) | {"settings": {"schema": "another"}}))
    _, txt, r = run(out, "--again", "")
    assert "\n  value settings in force: the adapter's record of them can't be read from the job's files: KeyError: " in txt
    assert r["value_settings"]["unread"] and "tokens: reproduced" in txt


def test_speculative_decoding_and_the_models_own_limit_are_read_from_the_launch_line(tmp_path: Path):
    out = tmp_path / "out"
    spec = ("--speculative-config.method", "mtp", "--speculative-config.num_speculative_tokens", "5")
    corpus_arm(out, "compiled", [100.0, 101.0], argv=(*spec, "--max-model-len=131072"))
    corpus_arm(out, "record", [100.0, 101.0], argv=(*spec, "--kv-cache-dtype", "fp8", "--max-model-len=131072"))
    _, txt, r = run(out)
    assert r["settings"]["record"]["speculative_decoding"] == {"state": "on", "by": " ".join(spec),
                                                               "draft_accepted_share": None, "accepted_per_draft": None}
    assert ("  speculative decoding: on (--speculative-config.method mtp --speculative-config.num_speculative_tokens 5); "
            "compiled: on (") in txt
    # the KV cache's type is the launch line's, and an arm that does not share it is not at the same settings
    assert ("  KV cache: fp8 (--kv-cache-dtype); DIFFERS; compiled: the engine's default (no --kv-cache-dtype in the "
            "launch line)\n") in txt
    assert r["cost"]["settings_differ"] == ["the KV cache's type"]
    assert P.flags(["--speculative-config={}", "--seed", "0"], ("--speculative",)) == "--speculative-config={}"
    assert r["settings"]["record"]["context_limit"] == {
        "tokens": 131072, "given": "131072", "by": None,
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
    assert rc == 0 and r["settings"]["record"]["context_limit"] | {
        "tokens": 202752, "given": "auto", "by": "the engine took it"} == r["settings"]["record"]["context_limit"]
    assert r["settings"]["compiled"]["context_limit"]["tokens"] is None
    assert ("  context limit: 202,752 tokens (the engine took it; --max-model-len auto), the model's own 131,072 "
            "(config.json max_position_embeddings); compiled: not resolved (--max-model-len auto; the arm kept no step "
            "trace and its server's log does not say what its engine took)\n") in txt
    # two launch lines that both leave it to vLLM are at the same limit; the other settings are the engine's word
    assert "the context limit" not in r["cost"]["settings_differ"]
    assert "  KV cache: fp8 (the engine's configuration: fp8); DIFFERS; compiled: the engine's default (" in txt
    assert "  speculative decoding: on (the engine's configuration: method mtp, num_speculative_tokens 5); DIFFERS; " in txt
    # stock keeps no step trace: its limit is the one vLLM's own log says it fitted, and one that differs is said
    log = out / "compiled/server/server.log"
    log.write_text("INFO 11-12 kv_cache_utils.py:1 Auto-fit max_model_len: reduced from 1048576 to 202752 to fit in "
                   "available GPU memory (61.2 GiB available for KV cache)\n")
    _, txt, r = run(out)
    assert r["settings"]["compiled"]["context_limit"]["tokens"] == 202752
    assert ("; compiled: 202,752 tokens (its server's log: Auto-fit max_model_len: reduced from 1048576 to 202752; "
            "--max-model-len auto), the model's own 131,072 (") in txt and "the context limit" not in r["cost"]["settings_differ"]
    log.write_text("x\nINFO Auto-fit max_model_len: full model context length 65536 fits in available GPU memory\n")
    _, txt, r = run(out)
    assert "; DIFFERS; compiled: 65,536 tokens (its server's log: Auto-fit max_model_len: full model context length 65536 fits; " in txt
    assert "BELOW the model's own 131,072" in txt and "the context limit" in r["cost"]["settings_differ"]
    log.unlink()
    _, txt, r = run(out)
    mine = r["settings"]["record"]
    other = {**mine, "context_limit": {**mine["context_limit"], "tokens": None, "given": "65536"}}
    assert P.settings_differ(mine, other) == ["the context limit"]
    assert P.settings_differ(mine, {**mine, "context_limit": {**mine["context_limit"], "tokens": 4096}}) == ["the context limit"]
    # the drafts' acceptance is the benchmark's counters', beside the launch line that turned them on
    reps = {"loads": {LOAD: [{"prompt_cached_share": 0.4, "draft_accepted_share": 0.5, "accepted_per_draft": 2.5}]}}
    spec = P.settings(out / "record", reps, LOAD)["speculative_decoding"]
    assert spec["by"].endswith("; 50.0% of its draft tokens accepted, 2.50 a draft") and spec["accepted_per_draft"] == 2.5
    # a setting the launch line turns on and the step trace's header does not hold: the replay's engine would lack it
    shutil.rmtree(out / "record")
    line = ("--speculative-config.method", "mtp", "--kv-cache-dtype", "fp8")
    corpus_arm(out, "record", [100.0, 101.0], argv=line, trace=True)
    _, txt, r = run(out)
    assert ("  speculative decoding: on (--speculative-config.method mtp; the step trace's header holds none, and the "
            "replay's engine is built from the header); DIFFERS; compiled: off (") in txt
    assert "  KV cache: fp8 (--kv-cache-dtype; the step trace's header holds none, and the replay's engine is built from the header); " in txt
    shutil.rmtree(out / "record")
    corpus_arm(out, "record", [100.0, 101.0], argv=line, trace=True,
               header={"kv_cache_dtype": "fp8_ds_mla", "speculative": {"method": "mtp"}})
    _, txt, r = run(out)
    assert "  speculative decoding: on (--speculative-config.method mtp; the engine's configuration: method mtp); DIFFERS; " in txt
    assert "  KV cache: fp8 (--kv-cache-dtype; the engine's configuration: fp8_ds_mla); DIFFERS; " in txt


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


def bench_arm(out: Path, name: str, passes: dict[str, list[float]], ids: str | None = None,
              argv: tuple[str, ...] = ("--max-model-len", "4096")) -> None:
    """An arm's directory as `serve_tax.py run` leaves it under vLLM's client, which keeps no request's ids; with
    ``ids``, the arm's server kept them for rep 1 of that pass (a line a request, gzipped as the runner leaves them),
    beside its step trace."""
    d = out / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({
        "arm": name, "parent": None, "what": name, "model_id": "glm53", "model": "repo@rev", "client": "vllm-bench", "rates": {},
        "run_id": "r-1", "commit": "c" * 40, "server_argv": list(argv), "startup_s": 300.0,
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
                step_trace(out, d, tag, {"enable_prefix_caching": True, "max_model_len": 4096})
                with gzip.open(d / "server" / "ids.9.jsonl.gz", "wt") as f:
                    f.write("".join(json.dumps({"id": rid(tag, k), "prompt_ids": [k, k + 1], "output_ids": [7, k]}) + "\n"
                                    for k in range(2)))


def test_by_vllms_client_the_cost_is_the_benchmarks_paired_cost_and_nothing_is_replayed(tmp_path: Path):
    out = tmp_path / "out"
    stock = [300.0, 301.0, 299.0, 300.5, 299.5, 300.0]
    bench_arm(out, "compiled", {"chat@8": [200.0] * 6, "chat@16": stock, "chat@32": [290.0] * 6})
    bench_arm(out, "compiled-again", {"chat@16": [x + 0.3 for x in stock]})
    bench_arm(out, "record", {"chat@16": [x / 1.004 for x in stock]})
    rc, txt, r = P.main(["report", str(out), "--load", "chat", *ARMS]), (out / "prod_repro.txt").read_text(), json.loads(
        (out / "prod_repro.json").read_text())
    want = S.serving_cost(S.arm_metrics(out / "compiled"), S.arm_metrics(out / "record"), "chat")
    assert rc == 0 and r["cost"]["value"] == pytest.approx(0.004) == pytest.approx(want["value"])
    assert r["cost"]["passes"] == {"value": want["value"], "ci95": want["ci95"]}
    assert r["cost"]["how"] == {"client": "vllm-bench", "rung": 16, "reps": 6, "saturated": True, "sessions": False,
                                "by": r["cost"]["how"]["by"]}
    # six passes that all sit under the target show nothing: they are one lifetime an arm, and the verdict reads the
    # interval over lifetime pairs, which a job does not have
    assert r["cost"]["verdict"] == "not shown" and r["cost"]["lifetimes"] == {"pairs": 1, "served": 1, "ci95": None}
    assert ("cost: record over compiled +0.40% of stock's serving time; under 1%: not shown, each arm has one server "
            "lifetime, and the verdict reads the interval over lifetime pairs\n"
            "  passes taken as independent, 95% interval [+0.40%, +0.40%]: over 6 paired reps, at concurrency 16, where "
            "stock's rungs show it saturated: ") in txt
    # the floor is beside and not added; stock's second lifetime is slower than its first in every pass here, so its
    # interval leaves out 0 and the report says stock differs from stock
    assert r["cost"]["noise_floor"]["leaves_out_zero"] is True
    assert "\n  noise floor, beside and not added: compiled-again over compiled -0.10%, 95% interval [" in txt
    assert (", passes taken as independent\n"
            "  STOCK DIFFERS FROM STOCK: the noise floor's interval leaves out 0, so two servers of compiled differ by "
            "more than their passes do\n") in txt
    assert "tokens: not measured: no replay ran: none was asked for (--replay PASS names the pass of vLLM's client" in txt


def sessions_arm(name: str, rates: list[float]) -> dict:
    """An arm's metrics with the agentic load served by four users, a rep each rate in thousands of output tokens a
    second, every request served and 94% of each rep's prompt tokens from the prefix cache."""
    plan = {"requests": 96, "prompt_tokens": 8000000, "repeated_tokens": 7600000, "span_s": 0.0, "sha256": "a" * 64, "users": 4}
    reps = [{"rep": i + 1, "rate": None, "concurrency": 4, "sent": 96, "ok": 96, "met": 96, "goodput_req_per_s": 0.5,
             "req_per_s": 0.5, "output_tok_per_s": 1000 * r, "prompt_tok_per_s": 50000.0, "output_tokens": 60000,
             "prompt_tokens": 8000000, "prompt_cached_share": 0.94, "total_tok_per_s": 52000.0, "tpot_p90_s": 0.025}
            for i, r in enumerate(rates)]
    return {"arm": name, "parent": S.ARMS[name].parent, "what": name, "run_id": "r-1", "commit": "c" * 40,
            "vllm": "0.1+gd9105ea8", "model_id": "glm53", "slo": {"ttft_s": 2.0, "tpot_s": 0.08},
            "loads": {S.pass_name(S.AGENTIC, 4): reps}, "planned": {S.AGENTIC: plan}}


def lifetimes_of(name: str, rates: list[list[float]]) -> list[dict]:
    """An arm's metrics a server lifetime (`sessions_arm`), as the benchmark reads each lifetime's directory."""
    return [sessions_arm(name, x) | {"lifetime": k, "reps": len(x), "left_out": 0} for k, x in enumerate(rates, 1)]


def lifetimes_cost(tmp_path: Path, stock: list[list[float]], ours: list[list[float]]) -> tuple[dict, dict]:
    """The cost line and the benchmark's own cost of two arms served in lifetimes, joined as the report joins them."""
    arms = S.lifetimes(lifetimes_of("prod", stock) + lifetimes_of("prod-record", ours))
    return (P.cost(tmp_path, {a["arm"]: a for a in arms}, "prod", "prod-record", None, S.AGENTIC, 1.0),
            S.session_cost(arms, "prod-record", "prod"))


def test_a_sessions_loads_cost_is_the_benchmarks_own_and_its_verdict_reads_the_interval_over_lifetimes(tmp_path: Path):
    ours = [1.99, 1.988, 1.992, 1.99, 1.989, 1.991]
    by = {"prod": sessions_arm("prod", [2.0] * 6), "prod-record": sessions_arm("prod-record", ours)}
    c = P.cost(tmp_path, by, "prod", "prod-record", None, S.AGENTIC, 1.0)
    own = S.session_cost(list(by.values()), "prod-record", "prod")
    assert c["measured"] and round(100 * c["value"], 2) == own["value"] == 0.5 and c["how"]["sessions"] and c["how"]["rung"] == 4
    # one lifetime an arm: the benchmark can't tell and the line shows nothing, though the interval over its six passes
    # taken as independent sits under the target
    assert (own["verdict"], own["lifetimes"]) == ("cannot tell", 1) and c["passes"]["ci95"][1] < 0.01
    assert (c["same_work"], c["verdict"], P._verdict(c)) == (True, "not shown", "not shown, " + P.ONE_LIFETIME)
    assert c["lifetimes"]["pairs"] == c["lifetimes"]["served"] == 1 and c["lifetimes"]["ci95"] is None and c["noise_floor"] is None
    # two lifetimes an arm, each leaving out its first pass, and ours's second stopped before the pass it counts (the
    # benchmark's STOP_AT): one pair of lifetimes has a pass both counted, so no interval, and the line says which it is
    cut = S.lifetimes([sessions_arm(name, x) | {"lifetime": k, "reps": 2, "left_out": 1} for name, rates in (
        ("prod", [[2.0, 2.0], [2.0, 2.0]]), ("prod-record", [[1.99, 1.99], [1.99]])) for k, x in enumerate(rates, 1)])
    c = P.cost(tmp_path, {a["arm"]: a for a in cut}, "prod", "prod-record", None, S.AGENTIC, 1.0)
    assert (c["lifetimes"]["served"], c["lifetimes"]["pairs"], c["lifetimes"]["ci95"]) == (2, 1, None)
    assert [(x["lifetime"], x["reps"]) for x in c["lifetimes"]["each"]] == [(1, 1)] and round(100 * c["value"], 2) == 0.5
    assert (c["verdict"], P._verdict(c)) == ("not shown", "not shown, " + P.ONE_PAIR)
    assert P.ONE_PAIR.startswith("only one pair of the arms' lifetimes has a pass both arms counted")
    # the same where ours was not served its second lifetime at all (its turn came after the stop time): stock has two
    short = S.lifetimes([sessions_arm(name, x) | {"lifetime": k, "reps": 2, "left_out": 1} for name, rates in (
        ("prod", [[2.0, 2.0], [2.0, 2.0]]), ("prod-record", [[1.99, 1.99]])) for k, x in enumerate(rates, 1)])
    c = P.cost(tmp_path, {a["arm"]: a for a in short}, "prod", "prod-record", None, S.AGENTIC, 1.0)
    assert (c["lifetimes"]["served"], c["lifetimes"]["pairs"], P._verdict(c)) == (2, 1, "not shown, " + P.ONE_PAIR)
    # three lifetimes an arm whose costs agree: the interval over them is the benchmark's, and its upper end is under
    c, own = lifetimes_cost(tmp_path, [[2.0, 2.0], [2.01, 2.01], [1.99, 1.99]], [[1.99, 1.99], [1.998, 1.998], [1.981, 1.981]])
    assert own["verdict"] == "meets" and (c["verdict"], P._verdict(c)) == ("under", "yes")
    assert c["value"] == own["value"] / 100 and c["lifetimes"]["ci95"] == [x / 100 for x in own["ci95"]]
    assert c["lifetimes"]["pairs"] == 3 and c["lifetimes"]["ci95"][1] < 0.01
    assert [round(100 * x["value"], 2) for x in c["lifetimes"]["each"]] == [x["value"] for x in own["by_lifetime"]] == [0.5, 0.6, 0.45]
    # the line that says how the cost is taken ends with the benchmark's own words on each request's length; no arm
    # here kept its served ids, so the benchmark has no line on the requests that went on past an end-of-text token
    # and the report says they are not counted
    assert c["lifetimes"]["past_end"] == {"said": [], "sides": {"prod": None, "prod-record": None}}
    assert P._lifetimes(c) == [
        "  the cost is the mean of the lifetimes' costs, a lifetime's being ours's seconds for the passes both arms counted "
        "in it over stock's, less one, a pass's seconds from its users' first send to the last one's last answer "
        "(serve_tax.session_cost); each request decoded to its recorded length, past any end-of-text token",
        "  past an end-of-text token: not counted, no arm's passes hold a count of the requests that went on past one",
        "  by lifetime: +0.50% (2 passes), +0.60% (2 passes), +0.45% (2 passes)",
        f"  the cost's spread (standard deviations): {own['spread']['between']:.2f} points between lifetimes, 0.00 points "
        "between a lifetime's passes"]
    # with no twin of stock the floor is stock's own lifetimes, the second over the first: one pair, so no interval
    assert c["noise_floor"] == {"arm": "prod", "over": "lifetimes", "value": pytest.approx(-0.005), "ci95": None, "pairs": 1,
                                "own": True, "leaves_out_zero": False}
    assert P._noise(c["noise_floor"], "prod") == ("prod's own lifetimes, each second one over the one before it, -0.50% (no "
                                                  "interval: one pair), over 1 pair of server lifetimes")
    # lifetimes whose costs spread: the upper end is not under the target, and nothing is shown
    c, own = lifetimes_cost(tmp_path, [[2.0, 2.0]] * 3, [[1.99, 1.99], [1.97, 1.97], [1.999, 1.999]])
    assert own["verdict"] == "cannot tell" and c["lifetimes"]["ci95"][1] > 0.01
    assert (c["verdict"], P._verdict(c)) == ("not shown", "not shown, " + P.NOT_UNDER)
    # lifetimes that all cost more than the target: the lower end is over it, and the line says no
    c, own = lifetimes_cost(tmp_path, [[2.0, 2.0]] * 3, [[1.96, 1.96], [1.959, 1.959], [1.961, 1.961]])
    assert own["verdict"] == "misses" and c["verdict"] == "over"
    assert P._verdict(c) == "NO, the lower end of the interval over lifetime pairs is over it"
    # one pass an arm has no interval over passes either
    one = {n: sessions_arm(n, x) for n, x in (("prod", [2.0]), ("prod-record", ours[:1]))}
    c = P.cost(tmp_path, one, "prod", "prod-record", None, S.AGENTIC, 1.0)
    assert c["measured"] and c["passes"]["ci95"] is None and (c["same_work"], c["verdict"]) == (True, "not shown")
    # a rep whose prefix cache served another share of the prompts: the benchmark's own cost has no verdict, nor this
    by["prod-record"]["loads"]["agentic@4"][2]["prompt_cached_share"] = 0.9
    c = P.cost(tmp_path, by, "prod", "prod-record", None, S.AGENTIC, 1.0)
    assert c["measured"] and (c["same_work"], c["verdict"]) == (False, "not shown")
    assert P._verdict(c) == ("not shown, the arms did not do the same work (one left a request unserved, they served "
                             "another count of output tokens, or their prefix caches served different shares of the prompts "
                             "in some rep)")
    # so does a rep in which the arms served another count of output tokens, as sampled passes can
    by["prod-record"]["loads"]["agentic@4"][2] |= {"prompt_cached_share": 0.94, "output_tokens": 59000}
    c = P.cost(tmp_path, by, "prod", "prod-record", None, S.AGENTIC, 1.0)
    assert c["measured"] and (c["same_work"], c["verdict"]) == (False, "not shown")
    by["prod-record"]["loads"]["agentic@4"][2]["output_tokens"] = 60000
    # beside the cost and in no verdict: the benchmark's other readings of the same passes, and each side's engine
    # steps and accepted drafts, where the passes hold them
    assert c["lifetimes"]["readings"] == {"busy_gpu_s": None, "users_s": None, "users_rates": None}
    assert not any("other readings" in x or "beside the cost" in x for x in P._lifetimes(c))
    for name, busy, steps, share in (("prod", 240.0, 2000, 0.75), ("prod-record", 247.2, 2100, 0.7)):
        for rep in by[name]["loads"]["agentic@4"]:
            rep |= {"busy_gpu_s": busy, "users_s": 4 * busy, "users_output_tok_per_s": 60000 / busy, "steps": steps,
                    "draft_accepted_share": share, "accepted_per_draft": 4 * share}
    c = P.cost(tmp_path, by, "prod", "prod-record", None, S.AGENTIC, 1.0)
    own = S.session_cost(list(by.values()), "prod-record", "prod")
    assert c["lifetimes"]["readings"] == {k: v / 100 for k, v in own["readings"].items()} and own["readings"]["busy_gpu_s"] == 3.0
    assert c["lifetimes"]["sides"] == {
        "prod": {"steps": 2000, "draft_accepted_share": 75.0, "tokens_per_draft": 4.0},
        "prod-record": {"steps": 2100, "draft_accepted_share": 70.0, "tokens_per_draft": pytest.approx(3.8)}}
    assert P._lifetimes(c)[-2:] == [
        "  other readings of the same passes, in no verdict: +3.00% by the GPUs' busy seconds, +3.00% by the users' seconds "
        "summed, +3.00% by the users' own rates summed",
        "  beside the cost, which is adjusted for neither: prod 2,000 engine steps a pass, 75.0% of its draft tokens "
        "accepted, 4.00 tokens a draft; prod-record 2,100 engine steps a pass, 70.0% of its draft tokens accepted, 3.80 "
        "tokens a draft"]
    assert P._verdict(c | {"settings_differ": ["the context limit"]}) == "not shown, the arms are not at the same settings"
    # where ours's server kept the served ids, the benchmark's own line counts its requests that served an end-of-text
    # token and went on, word for word under the line that says how the cost is taken; stock's server keeps none
    assert P._lifetimes(c)[0].endswith(f"(serve_tax.session_cost); {S.TO_LENGTH}")
    for rep in by["prod-record"]["loads"]["agentic@4"]:
        rep |= {"ids_kept": 96, "past_end": 12, "after_end_tokens": 6000}
    c = P.cost(tmp_path, by, "prod", "prod-record", None, S.AGENTIC, 1.0)
    own = S.session_cost(list(by.values()), "prod-record", "prod")
    assert c["lifetimes"]["past_end"] == {"said": S.past_end_text(own), "sides": {
        "prod": None, "prod-record": {"requests": 72, "of": 576, "tokens_after_share": 10.0}}}
    assert P._lifetimes(c)[1] == (
        "  past an end-of-text token, in the passes counted: 72 of prod-record's 576 requests served an end-of-text token "
        "before their end and went on, 10.00% of its output tokens after one; prod's server keeps no ids, so the count is "
        "prod-record's alone")
    # a twin of stock is the floor, a lifetime pair its unit too
    by = {n: sessions_arm(n, x) for n, x in (("prod", [2.0] * 6), ("prod-again", [1.998] * 6), ("prod-record", ours))}
    c = P.cost(tmp_path, by, "prod", "prod-record", "prod-again", S.AGENTIC, 1.0)
    assert c["noise_floor"] == {"arm": "prod-again", "over": "lifetimes", "value": pytest.approx(0.001), "ci95": None,
                                "pairs": 1, "own": False, "leaves_out_zero": False}
    assert P._noise(c["noise_floor"], "prod") == ("prod-again over prod +0.10% (no interval: one pair), over 1 pair of "
                                                  "server lifetimes")
    # arms the benchmark's own cost refuses are not measured, with its reason
    by["prod-record"] = sessions_arm("prod-record", ours) | {"vllm": "0.2"}
    c = P.cost(tmp_path, by, "prod", "prod-record", None, S.AGENTIC, 1.0)
    assert not c["measured"] and "refuses these arms: cost: the arms were served by different builds of vLLM" in c["why"]


def test_a_job_served_in_lifetimes_is_read_from_each_lifetimes_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The arms of a job the benchmark served in lifetimes are ``<arm>.K``: every lifetime is in what ran, an arm's
    reps are all its lifetimes', and the tokens are those of the lifetime the replay's record names."""
    out = tmp_path / "out"
    for k, (stock, ours) in enumerate((([100.0, 101.0], [99.0, 100.0]), ([102.0, 103.0], [101.0, 102.0])), 1):
        corpus_arm(out, f"compiled.{k}", stock, arm="compiled", lifetime=k, reps=2)
        corpus_arm(out, f"record.{k}", ours, trace=True, arm="record", lifetime=k, reps=2)
    replayed(out, "record.1")
    rc, txt, r = run(out, "--again", "")
    assert rc == 0, txt
    assert ("what ran:\n"
            "  compiled.1: up in 12.5 s; the load served 2 times, 4 requests, every one finished\n"
            "  compiled.2: up in 12.5 s; the load served 2 times, 4 requests, every one finished\n"
            "  record.1: up in 12.5 s; the load served 2 times, 4 requests, every one finished\n"
            "  record.2: up in 12.5 s; the load served 2 times, 4 requests, every one finished\n"
            "  the replay: ended") in txt
    assert r["arms"]["served"] == ["compiled", "record"] and P.default_arms(out=out) == P.TODAY
    assert "tokens: reproduced: 2 requests replayed, 4 served tokens compared with 4 replayed" in txt
    assert r["tokens"]["window"]["lifetime"] == 1 and "\n  the window: of lifetime 1 of record; every request of the load's rep" in txt
    # the cost of a load that is not a sessions load is over the four passes an arm, and has no verdict
    ref, mine = (S.summary(x) for x in ([100.0, 101.0, 102.0, 103.0], [99.0, 100.0, 101.0, 102.0]))
    assert r["cost"]["value"] == pytest.approx(S.tax(ref, mine)["value"]) and r["cost"]["how"]["reps"] == [4, 4]
    assert r["cost"]["lifetimes"] == {"pairs": 2, "served": 2, "ci95": None}
    assert "; under 1%: not shown, the interval over lifetime pairs is the benchmark's for a sessions load" in txt
    # a replay of the second lifetime's pass is compared with that lifetime's window
    shutil.rmtree(out / "replay-compiled")
    replayed(out, "record.2")
    monkeypatch.setattr(P, "encodings", Loader)
    rc, txt, r = run(out, "--again", "")
    assert rc == 0 and r["tokens"]["reproduces"] and r["tokens"]["window"]["lifetime"] == 2
    # and the advice is of that lifetime's pass
    assert r["advice"]["window"]["lifetime"] == 2 and "\n  the pass: lifetime 2 of record's burst rep 1, the one the replay" in txt
    # a replay of a directory that is none of ours's lifetimes is of another window
    shutil.rmtree(out / "replay-compiled")
    replayed(out, "record.2", arm_dir=str(out / "record.3"))
    rc, txt, r = run(out, "--again", "")
    assert "tokens: not measured: the replay in " in txt and "is of another window (arm record.3, load burst, rep 1" in txt
    # a lifetime with no record is in what ran as not served, and `served` says the arm was not served whole
    assert P.main(["served", str(out), "record", "--load", LOAD]) == 0
    shutil.rmtree(out / "record.2")
    (out / "record.2").mkdir()
    rc, txt, r = run(out, "--again", "")
    assert "  record.2: NOT SERVED: the job holds no record of it (no record.2/meta.json): it did not start\n" in txt
    assert txt.endswith("; not served whole: record.2\n") and P.main(["served", str(out), "record", "--load", LOAD]) == 1


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


# ------------------------------------------------------------------------------------------------- the replay's engine

def topk_log(d: Path, pid: int, calls: int, rows: tuple[int, int, int]) -> None:
    """One process's counters as the canonical top-k's plugin keeps them (`dsa_topk.log_to`)."""
    d.mkdir(parents=True, exist_ok=True)
    (d / f"dsa_topk.{pid}.json").write_text(json.dumps({
        "pid": pid, "calls": {"top_k_per_row_decode": calls}, "launches": {"top_k_per_row_decode": calls // 2},
        "rows": {"cuda:0": dict(zip(("rows_long", "rows_tied", "rows_bisected"), rows))}}))


#: what a replay of an arm that served on the canonical top-k says its engine ran under (`replay.served_under`)
TOPK_UNDER = {"plugins": ["verity_dsa_topk = dsa_topk:register"], "by": "the replay", "env": {"VERITY_DSA_TOPK": "canonical"}}


def test_the_report_says_what_the_replays_engine_ran_under_and_what_the_top_k_counted(tmp_path: Path):
    out, private = tmp_path / "out", tmp_path / "private"
    stock = [300.0, 301.0]
    bench_arm(out, "prod", {"chat@16": stock})
    bench_arm(out, "prod-record", {"chat@16": stock}, ids="chat@16")
    argv = ["report", str(out), "--load", "chat", "--stock", "prod", "--again", "", "--ours", "prod-record", "--replayed", "chat@16"]

    def said(*more: str) -> tuple[str, dict]:
        assert P.main([*argv, *more]) == 0
        return (out / "prod_repro.txt").read_text(), json.loads((out / "prod_repro.json").read_text())["ran"]["replay_engine"]

    # a replay that does not say (an earlier build of the benchmark's) is given no line
    d = replayed(out, "prod-record", load="chat@16")
    rec = json.loads((d / "replay.json").read_text())
    (d / "replay.json").write_text(json.dumps({k: v for k, v in rec.items() if k != "served_under"}))
    txt, engine = said()
    assert engine == {"engine": None} and "the replay's engine" not in txt
    # the arm's is what `serve_tax` gives it: the plugin and the switch the replay's own record names
    assert S.ARMS["prod-record"].env["VERITY_DSA_TOPK"] == "canonical" and "{private}" in S.ARMS["prod-record"].env["VERITY_DSA_TOPK_LOG"]
    (d / "replay.json").write_text(json.dumps(rec | {"served_under": TOPK_UNDER}))
    txt, engine = said()
    assert engine == {"engine": {"arm": "prod-record", "under": TOPK_UNDER, "topk": {"kept": False, "replay": None, "served": None}}}
    assert ("  the replay's engine: under what prod-record served with beside its record (set by the replay): the vLLM "
            "plugin verity_dsa_topk; VERITY_DSA_TOPK=canonical\n"
            "  the canonical top-k's counters were not kept (the job named no PRIVATE directory): nothing here shows how "
            "many rows it rewrote\n") in txt
    # with the job's private directory: serving's processes kept theirs under the arm's name, the replay's under its
    # directory's; a replay none of whose processes kept any shows nothing of the top-k having run there
    topk_log(private / "prod-record/topk-log", 21, 9000, (60, 3, 1))
    txt, engine = said("--private", str(private))
    assert engine["engine"]["topk"] == {"kept": True, "replay": None, "served": {"processes": 1, "calls": 9000, "rows_long": 60}}
    assert ("  the canonical top-k in the replay: NO process kept counters: nothing shows it ran there\n"
            "  the canonical top-k while serving: 1 process kept counters: at least 9,000 calls of the indexer's top-k, "
            "60 rows of more than its keys rewritten\n") in txt
    topk_log(private / "replay-compiled/topk-log", 11, 5000, (40, 2, 1))
    topk_log(private / "replay-compiled/topk-log", 12, 3000, (10, 0, 0))
    txt, engine = said("--private", str(private))
    assert engine["engine"]["topk"]["replay"] == {"processes": 2, "calls": 8000, "rows_long": 50}
    assert ("  the canonical top-k in the replay: 2 processes kept counters: at least 8,000 calls of the indexer's top-k, "
            "50 rows of more than its keys rewritten\n") in txt
    # the report keeps the counts in OUT, where a later reading of the job finds them once the node's directory is gone
    kept = json.loads((out / P.TOPK_KEPT).read_text())
    assert kept == {"replay": {"processes": 2, "calls": 8000, "rows_long": 50}, "served": {"processes": 1, "calls": 9000, "rows_long": 60}}
    shutil.rmtree(private)
    assert said()[1]["engine"]["topk"] == {"kept": True, **kept}
    # counters that can't be read end that part alone
    (out / P.TOPK_KEPT).write_text("not json")
    assert P.main(argv) == 0 and "  the replay's engine: the replay's engine can't be read from the job's files: JSONDecodeError" in (
        out / "prod_repro.txt").read_text()


def test_the_first_line_says_the_limit_and_what_is_known_of_leaving_it_to_vllm(tmp_path: Path):
    held = json.loads(P.LIMITS.read_text())
    assert held["glm53"]["auto"]["starts"] is False
    assert P.limits("glm53") == held["glm53"] and P.limits("moonlight") == P.limits(None) == {}
    # the limit a job that names none serves at is the benchmark's own for the arm: the file holds no second copy
    assert "max_model_len" not in held["glm53"] and S.published_context("prod", "glm53", None) == 262144
    # a job on a model the file holds: its first line says the limit its arms were given and that auto does not start
    out = tmp_path / "out"
    bench_arm(out, "prod", {"chat@16": [300.0]}, argv=("--max-model-len", "262144"))
    bench_arm(out, "prod-record", {"chat@16": [299.0]}, argv=("--max-model-len", "262144"))
    argv = ["report", str(out), "--load", "chat", "--stock", "prod", "--again", "", "--ours", "prod-record"]
    assert P.main(argv) == 0
    auto = held["glm53"]["auto"]
    assert (out / "prod_repro.txt").read_text().startswith(
        "prod_repro: context limit 262,144 tokens, given outright to prod-record and prod; with the limit left to it "
        f"(--max-model-len auto), at {auto['at']}, vLLM does not start this model on these cards at the pin: "
        f"{auto['why']} (run {auto['run']})\n"
        "prod_repro: load chat; stock prod, stock again not served, ours prod-record; run r-1, commit ")
    # arms given different limits, or one left to vLLM, are said as they were started
    shutil.rmtree(out / "prod")
    bench_arm(out, "prod", {"chat@16": [300.0]}, argv=("--max-model-len", "auto"))
    assert P.main(argv) == 0
    assert (out / "prod_repro.txt").read_text().startswith(
        "prod_repro: context limit: prod-record --max-model-len 262144, prod --max-model-len auto; with the limit left")
    shutil.rmtree(out / "prod-record")
    assert P.main(argv) == 0
    assert (out / "prod_repro.txt").read_text().startswith("prod_repro: context limit left to vLLM by prod (--max-model-len auto)\n")
    shutil.rmtree(out / "prod")
    assert P.main(argv) == 0
    assert (out / "prod_repro.txt").read_text().startswith("prod_repro: context limit: not read (no record of ours or of stock)\n")


def test_the_tokens_line_counts_the_requests_past_the_indexers_top_k(tmp_path: Path):
    out = job(tmp_path)
    replayed(out)
    assert run(out)[2]["tokens"]["past_topk"] is None             # a model with no such top-k
    cfg = json.loads((out / "ck/config.json").read_text())
    # each request is two prompt ids and two served ids
    (out / "ck/config.json").write_text(json.dumps(cfg | {"index_topk": 4}))
    assert run(out)[2]["tokens"]["past_topk"] == {"keys": 4, "requests": 0, "differing": 0}
    (out / "ck/config.json").write_text(json.dumps(cfg | {"index_topk": 3}))
    _, txt, r = run(out)
    assert r["tokens"]["past_topk"] == {"keys": 3, "requests": 2, "differing": 0}
    assert ("  2 of the window's requests passed 3 ids of prompt and output, past which attention reads the keys the DSA "
            "indexer's top-k selects\n") in txt
    shutil.rmtree(out / "replay-compiled")
    replayed(out, tokens={rid(f"{LOAD}1", 0): [8, 0]})
    rc, txt, r = run(out)
    assert rc == 1 and r["tokens"]["past_topk"]["differing"] == 1
    assert "top-k selects; 1 of the 1 differing requests are among them\n" in txt


# --------------------------------------------------------------------------------------------------------- the advice

class Loader:
    """A stand-in for the scheduler replay's loader: it keeps what the report hands it and gives the figures of a
    served pass of 32 requests; told to, it refuses the pass, or reads it as served under the certified tick."""

    def __init__(self, refuse: str | None = None, tick: bool = False) -> None:
        self.refuse, self.tick = refuse, tick

    def span(self, records: list[dict], mine: object) -> list[dict]:
        self.span_ids = sorted(r["request"] for r in records if r["kind"] == "arrive" and mine(r["request"]))
        return list(records)

    def load_step_trace(self, recs: list[dict], *, tokens: dict[str, list[int]]) -> str:
        self.tokens = tokens
        return "trace"

    def draft_table(self, trace: str) -> dict:
        if self.refuse:
            raise ValueError(self.refuse)
        return {"requests": 32, "steps": 2040, "tick": self.tick, "arrivals": {"plain": 230},
                "kept_count": {"K": 5, "rows": 5143, "counts": [665, 553, 442, 257, 150, 3076], "entropy": 1.8400717745206507,
                               "max": 2.584962500721156, "alone": 9463.489136359707, "charged": 6997,
                               "ideal": 6995.721295001329, "derived": 0}}


def test_the_advice_line_is_the_replayed_passs_schedule_as_the_scheduler_replays_loader_charges_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = job(tmp_path)
    replayed(out)
    # the tree's loader is there, loaded once, with the three functions the line is made of
    E = P.encodings()
    assert all(callable(getattr(E, name)) for name in ("span", "load_step_trace", "draft_table")) and P.encodings() is E
    loader = Loader()
    monkeypatch.setattr(P, "encodings", lambda: loader)
    rc, txt, r = run(out)
    assert rc == 0 and r["advice"]["measured"]
    # the loader is handed the replayed pass's requests alone, each with its prompt ids and then its served ids
    a, b = rid(f"{LOAD}1", 0), rid(f"{LOAD}1", 1)
    assert loader.span_ids == [a, b] and loader.tokens == {a: [0, 1, 7, 0], b: [1, 2, 7, 1]}
    # the charged figure first; the kept count's own figures as the loader gives them
    assert ("advice: 225.8 bits a request charged (32 requests): the arrivals 7.2 (0 under the certified tick, which this "
            "serve is not under), the kept count 218.7 (5,143 rows at 1.360 bits, its code decoded back; each count alone "
            "295.7, not sent)\n"
            "  the kept count over the pass, as the scheduler replay's loader gives it: K 5, rows 5,143, counts [665, 553, "
            "442, 257, 150, 3076], entropy 1.840, max 2.585, alone 9,463.489, charged 6,997, ideal 6,995.721, derived 0\n"
            "  the kept count is 0 where the audited block derives it, and no Definition holds the accept test yet\n"
            "  no request was sampled, so no seed is read\n"
            "  the pass: record's burst rep 1, the one the replay re-ran; 2,040 steps, each reproduced by the scheduler "
            "replay\n") in txt
    assert txt.endswith("prod_repro: 3 of 3 lines measured\n")
    v = r["advice"]
    assert v["bits"] == 230 + 6997 and v["kept_count"]["counts"][5] == 3076 and v["arrivals_sent"] == 230
    assert v["window"] == {"arm": "record", "lifetime": None, "load": LOAD, "rep": 1, "run": "r-1", "replayed": True}
    assert v["seeds"] == {"sampled": 0, "unseeded": 0, "by_client": False}
    # the seeds are the pass's own arrivals': the client's, in an arrival, or in no record
    w = {"requests": [{"sampling": {"temperature": 0.6, "seed": 3}}, {"sampling": {"temperature": 0.6, "seed": None}},
                      {"sampling": {"max_tokens": 2}}]}
    assert P.seeds(w, {"seeded": True}) == {"sampled": 2, "unseeded": 1, "by_client": True}
    assert P._advice(v | {"seeds": {"sampled": 32, "unseeded": 0, "by_client": True}})[3] == "  32 sampled requests, each seed the client's"
    assert P._advice(v | {"seeds": {"sampled": 32, "unseeded": 0, "by_client": False}})[3] == \
        "  32 sampled requests, each with a seed in its arrival"
    assert P._advice(v | {"seeds": {"sampled": 32, "unseeded": 5, "by_client": True}})[3] == (
        "  5 of the 32 sampled requests carry NO seed in their arrival: what each was sampled with is in no record, so "
        "its kept count is not derived and its tokens cannot be replayed")
    # a serve under the certified tick sends no arrivals: the kept count alone is charged
    monkeypatch.setattr(P, "encodings", lambda: Loader(tick=True))
    assert ("advice: 218.7 bits a request charged (32 requests): the arrivals 0.0 (under the certified tick, which sends "
            "none), the kept count 218.7 (5,143 rows at ") in run(out)[1]
    # a pass the scheduler replay does not reproduce, or a serve with no draft, is the loader's refusal
    monkeypatch.setattr(P, "encodings", lambda: Loader(refuse="the replay diverges from the trace at step 41 (41/2040 steps match)"))
    _, txt, r = run(out)
    assert ("advice: not measured: the scheduler replay refuses the pass: the replay diverges from the trace at step 41 "
            "(41/2040 steps match)\n") in txt and txt.endswith("prod_repro: 2 of 3 lines measured\n")
    # a pass ours did not serve has no schedule to read, and a replay of another pass is not this pass's
    monkeypatch.setattr(P, "encodings", Loader)
    _, txt, _ = run(out, "--replayed", "other")
    assert ("advice: not measured: record holds no pass other rep 1, the one the replay re-runs: the advice is that pass's "
            "schedule's (--replay PASS names it)\n") in txt
    _, txt, _ = run(out, "--rep", "2")
    assert (f"advice: not measured: the replay in {out / 'replay-compiled'} is of another pass (burst rep 1), not of "
            "record's burst rep 2\n") in txt
    # without a replay the line is of the pass a replay would re-run, and says none ran
    shutil.rmtree(out / "replay-compiled")
    _, txt, r = run(out)
    assert r["advice"]["measured"] and "\n  the pass: record's burst rep 1, the one a replay re-runs, and none ran; 2,040 steps" in txt


def test_under_research_run_the_runs_result_names_what_the_report_rests_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = job(tmp_path)
    replayed(out)
    monkeypatch.setattr(P, "encodings", Loader)
    # not under research run, and another job's out/ read again under it: no result is written
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    run(out)
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path / "elsewhere"))
    run(out)
    assert not (tmp_path / "result.json").exists() and not (tmp_path / "elsewhere").exists()
    # the run's own out/: a job with no result gets one, whose validation is the tokens line's
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-1")
    run(out)
    doc = json.loads((tmp_path / "result.json").read_text())
    files = ["out/prod_repro.json", "out/prod_repro.txt", "out/replay-compiled/replay.json"]
    assert doc["artifacts"] == files and (doc["schema"], doc["run_id"]) == ("research/result/v0.1", "r-1")
    assert doc["validation"] == {
        "status": "passed", "checker": "verity bench prod_repro", "evidence": ["out/prod_repro.json"],
        "detail": "tokens: reproduced: 2 requests replayed, 4 served tokens compared with 4 replayed, 0 differ in 0 requests"}
    assert all((tmp_path / f).is_file() for f in files)
    # a result the benchmark's cost wrote keeps what it says, with the files added to its own
    (tmp_path / "result.json").write_text(json.dumps({"schema": "research/result/v0.1", "run_id": "r-1", "validation": {"status": "failed"},
                                                      "measurement_files": ["out/serving_cost.json"], "artifacts": ["out/kept"]}))
    run(out)
    doc = json.loads((tmp_path / "result.json").read_text())
    assert doc["artifacts"] == ["out/kept", *files] and doc["validation"] == {"status": "failed"}
    assert doc["measurement_files"] == ["out/serving_cost.json"]
    # replayed tokens that differ fail the validation of a result the report writes
    (tmp_path / "result.json").unlink()
    shutil.rmtree(out / "replay-compiled")
    replayed(out, tokens={rid(f"{LOAD}1", 0): [8, 0]})
    assert run(out)[0] == 1 and json.loads((tmp_path / "result.json").read_text())["validation"]["status"] == "failed"


def test_the_arms_are_the_published_settings_arms_for_a_model_whose_recipe_the_benchmark_holds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    assert all(a in S.ARMS for a in P.PROD + P.TODAY) and all(S.ARMS[a].published for a in P.PROD)
    # the benchmark refuses an arm at the published settings for a model with no recipe: such a job serves the others
    assert P.default_arms("glm53") == ("prod", "prod-again", "prod-record") and S.MODELS["glm53"].published
    assert P.default_arms("moonlight") == P.default_arms("no-model") == P.default_arms() == ("compiled", "compiled-again", "record")
    with pytest.raises(SystemExit, match="holds no recipe for moonlight"):
        S.published_context("prod", "moonlight", None)
    assert P.main(["arms", "--model", "glm53"]) == 0 and capsys.readouterr().out == "prod prod-again prod-record\n"
    monkeypatch.setattr(S, "ARMS", {k: v for k, v in S.ARMS.items() if k != "prod-record"})
    assert P.default_arms("glm53") == P.TODAY
    # an earlier job's out/ is read as the arms it served
    assert P.default_arms("glm53", tmp_path) == P.TODAY
    (tmp_path / "prod-record").mkdir()
    assert P.default_arms("moonlight", tmp_path) == P.PROD
    assert P.main(["arms", "--out", str(tmp_path)]) == 0 and capsys.readouterr().out == "prod prod-again prod-record\n"


# --------------------------------------------------------------------------------------------------------- the runner

#: a stand-in for benchmarks/serve_tax/run.sh: it says what it was started with and ends as told; told to, it leaves
#: one process's counters where a replay on the canonical top-k keeps them, or ends the job's time (SIGTERM)
STAND_IN = """#!/usr/bin/env bash
{ echo "argv=$*"; for k in ARMS RATES REPS REQUESTS SPAN_S REPLAY REPLAY_FROM REPLAY_MODES MODEL_ID CLIENT MAX_MODEL_LEN \\
    PASS_PROMPTS STOP_AT SAMPLED PRIVATE LIFETIMES LEAVE_OUT; do echo "$k=${!k-unset}"; done; } > "$RESEARCH_RUN_DIR/started"
if [ -n "${STAND_IN_TOPK:-}" ]; then
  mkdir -p "$PRIVATE/replay-compiled/topk-log"
  echo '{"pid": 1, "calls": {"persistent_topk": 7}, "launches": {"persistent_topk": 3}, "rows": {"cuda:0": {"rows_long": 5}}}' \\
    > "$PRIVATE/replay-compiled/topk-log/dsa_topk.1.json"
fi
[ -z "${STAND_IN_TERM:-}" ] || kill -TERM "$PPID"
exit "${STAND_IN_RC:-0}"
"""


def runner(tmp_path: Path) -> Path:
    """A tree with this directory's files and, in the serving cost benchmark's place, its two modules and the stand-in;
    the scheduler replay's loader and the adapter it reads are the tree's own."""
    here, tax = tmp_path / "tree/benchmarks/prod_repro", tmp_path / "tree/benchmarks/serve_tax"
    here.mkdir(parents=True)
    tax.mkdir()
    (tmp_path / "tree" / P.ENCODINGS.parent.relative_to(P.ROOT)).mkdir(parents=True)
    (tmp_path / "tree" / P.ENCODINGS.relative_to(P.ROOT)).symlink_to(P.ENCODINGS)
    (tmp_path / "tree/adapters").mkdir()
    (tmp_path / "tree/adapters/vllm").symlink_to(P.ROOT / "adapters/vllm")
    for f in ("run.sh", "prod_repro.py", "limits.json"):
        shutil.copy(Path(P.__file__).parent / f, here / f)
    for f in ("serve_tax.py", "replay.py"):
        (tax / f).symlink_to(Path(S.__file__).parent / f)
    (tax / "run.sh").write_text(STAND_IN)
    return here / "run.sh"


def bash(script: Path, *argv: str, run_dir: Path | None = None, **env: str) -> subprocess.CompletedProcess:
    base = {k: v for k, v in os.environ.items() if k not in (
        "RESEARCH_RUN_DIR", "LOAD", "REPS", "REQUESTS", "MODEL_ID", "STOCK", "AGAIN", "OURS", "RATE", "SPAN_S", "TARGET",
        "MAX_MODEL_LEN", "REPLAY", "REPLAY_FROM", "REPLAY_MODES", "PASS_PROMPTS", "STOP_AT",
        "SAMPLED", "PRIVATE", "ARMS", "LIFETIMES", "LEAVE_OUT")}
    if run_dir:
        (run_dir / "started").unlink(missing_ok=True)
    return subprocess.run(["bash", str(script), *argv], capture_output=True, text=True, check=False, env={
        **base, "VENV": str(Path(sys.executable).parents[1]), **({"RESEARCH_RUN_DIR": str(run_dir)} if run_dir else {}), **env})


def started(run_dir: Path) -> dict[str, str]:
    return dict(x.split("=", 1) for x in (run_dir / "started").read_text().splitlines())


def test_the_runner_serves_the_three_arms_a_burst_and_replays_ours_then_reports(tmp_path: Path):
    script, run_dir = runner(tmp_path), tmp_path / "run"
    replayed(job(run_dir))
    p = bash(script, *ARMS, "--max-model-len", "8192", run_dir=run_dir)
    assert p.returncode == 0, p.stdout + p.stderr
    # one call of the benchmark's runner: the arms, then the replay of ours under the same lease
    assert started(run_dir) == {
        "argv": "", "ARMS": "compiled compiled-again record", "RATES": "burst=1000", "REPS": "3", "REQUESTS": "800",
        "SPAN_S": "unset", "REPLAY": "burst", "REPLAY_FROM": str(run_dir / "out/record"), "REPLAY_MODES": "compiled",
        "MODEL_ID": "moonlight", "CLIENT": "corpus", "MAX_MODEL_LEN": "8192", "PASS_PROMPTS": "unset", "STOP_AT": "unset",
        "SAMPLED": "unset", "PRIVATE": "unset", "LIFETIMES": "unset", "LEAVE_OUT": "unset"}
    assert "the replay's engine: vLLM's own (record serves with no other vLLM plugin of ours" in p.stdout
    assert "tokens: reproduced: 2 requests replayed" in p.stdout and p.stdout.endswith("prod_repro: 2 of 3 lines measured\n")
    assert (run_dir / "out/prod_repro.txt").read_text() in p.stdout
    # a model the command holds no limit for is served at the benchmark's, and its arms are the benchmark's own
    p = bash(script, run_dir=run_dir)
    assert p.returncode == 0 and started(run_dir)["MAX_MODEL_LEN"] == "unset" and started(run_dir)["ARMS"] == "compiled compiled-again record"
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
    assert p.returncode == 0 and (s["ARMS"], s["REPLAY"], s["REPLAY_FROM"]) == ("compiled record", "unset", "unset")
    assert "the replay: none was asked for" in p.stdout and "stock again not served" in p.stdout
    assert "the replay's engine" not in p.stdout


def test_the_runner_hands_vllms_client_the_load_and_asks_for_no_replay(tmp_path: Path):
    script, run_dir = runner(tmp_path), tmp_path / "run"
    stock = [300.0, 301.0, 299.0, 300.5, 299.5, 300.0]
    for name in ("compiled", "compiled-again", "record"):
        bench_arm(run_dir / "out", name, {"chat@16": stock})
    p = bash(script, *ARMS, "--client", "vllm-bench", "--load", "chat=16", "--model", "glm53", run_dir=run_dir)
    assert p.returncode == 0, p.stdout + p.stderr
    s = started(run_dir)
    assert s["argv"] == "--model glm53 --arms compiled compiled-again record --loads chat=16 --reps 6"
    assert (s["REPLAY"], s["REPLAY_FROM"], s["CLIENT"], s["SAMPLED"]) == ("unset", "unset", "vllm-bench", "unset")
    # the limit is the benchmark's own for each arm unless the job names one, which is then every arm's
    assert s["MAX_MODEL_LEN"] == "unset"
    assert "tokens: not measured: no replay ran" in p.stdout and "cost: record over compiled +0.00%" in p.stdout
    p = bash(script, *ARMS, "--client", "vllm-bench", "--load", "chat=16", "--model", "glm53", "--max-model-len", "131072",
             run_dir=run_dir)
    assert p.returncode == 0 and started(run_dir)["MAX_MODEL_LEN"] == "131072"


def test_the_runner_serves_the_arms_in_lifetimes_and_replays_ours_first(tmp_path: Path):
    script, run_dir = runner(tmp_path), tmp_path / "run"
    (run_dir / "out").mkdir(parents=True)
    argv = ("--client", "vllm-bench", "--model", "glm53", "--again", "none", "--load", "agentic=4", "--replay", "agentic@4")
    p = bash(script, *argv, "--lifetimes", "3", "--leave-out", "1", run_dir=run_dir)
    s = started(run_dir)
    assert s["argv"] == "--model glm53 --arms prod prod-record --loads agentic=4 --reps 6", p.stdout + p.stderr
    assert (s["LIFETIMES"], s["LEAVE_OUT"], s["REPLAY_FROM"]) == ("3", "1", str(run_dir / "out/prod-record.1"))
    # nothing of it was served here, and the report says so of each arm; a replay was asked for, so the job fails
    assert p.returncode == 1 and "  prod: NOT SERVED" in p.stdout and "  prod-record: NOT SERVED" in p.stdout
    # one lifetime an arm is the arm's own directory
    bash(script, *argv, run_dir=run_dir)
    s = started(run_dir)
    assert (s["LIFETIMES"], s["LEAVE_OUT"], s["REPLAY_FROM"]) == ("unset", "unset", str(run_dir / "out/prod-record"))
    p = bash(script, *argv, "--lifetimes", "0", run_dir=run_dir)
    assert p.returncode == 2 and "--lifetimes 0: how many times the arms are served, 1 or more" in p.stdout
    p = bash(script, *argv, "--replay-of", str(tmp_path), "--lifetimes", "2", run_dir=run_dir)
    assert p.returncode == 2 and "--replay-of serves nothing: --lifetimes is of a job that serves its arms" in p.stdout


def test_the_runner_replays_a_named_pass_of_vllms_client_at_the_recipes_arms(tmp_path: Path):
    script, run_dir, private = runner(tmp_path), tmp_path / "run", tmp_path / "private"
    limit = ("--max-model-len", "262144")
    bench_arm(run_dir / "out", "prod", {"chat@16": [300.0]}, argv=limit)
    bench_arm(run_dir / "out", "prod-record", {"chat@16": [299.0]}, ids="chat@16", argv=limit)
    # the arms are the recipe's by default for a model whose recipe the benchmark holds
    argv = ("--client", "vllm-bench", "--model", "glm53", "--again", "none", "--load", "chat=16", "--reps", "1",
            "--replay", "chat@16", "--pass-prompts", "chat=8", "--sampled", "--stop-at", "9999-01-01T00:00Z")
    replayed(run_dir / "out", "prod-record", load="chat@16", served_under=TOPK_UNDER)
    topk_log(private / "prod-record/topk-log", 21, 9000, (60, 0, 0))
    p = bash(script, *argv, run_dir=run_dir, PRIVATE=str(private), STAND_IN_TOPK="1")
    assert p.returncode == 0, p.stdout + p.stderr
    s = started(run_dir)
    assert s["argv"] == "--model glm53 --arms prod prod-record --loads chat=16 --reps 1"
    assert (s["REPLAY"], s["REPLAY_FROM"], s["REPLAY_MODES"]) == ("chat@16", str(run_dir / "out/prod-record"), "compiled")
    assert (s["PASS_PROMPTS"], s["STOP_AT"], s["SAMPLED"]) == ("chat=8", "9999-01-01T00:00Z", "1")
    assert (s["MAX_MODEL_LEN"], s["PRIVATE"]) == ("unset", str(private))
    assert p.stdout.startswith("prod_repro: context limit 262,144 tokens, given outright to prod-record and prod; with the "
                               "limit left to it (--max-model-len auto), at its recipe's flags and vLLM's own limit of "
                               "requests at once, vLLM does not start this model")
    # the job's report reads the top-k's counters where its processes kept them, and keeps the counts in the run's
    assert ("  the canonical top-k in the replay: 1 process kept counters: at least 7 calls of the indexer's top-k, 5 rows "
            "of more than its keys rewritten\n"
            "  the canonical top-k while serving: 1 process kept counters: at least 9,000 calls of the indexer's top-k, 60 "
            "rows of more than its keys rewritten\n") in p.stdout
    assert json.loads((run_dir / "out" / P.TOPK_KEPT).read_text())["replay"] == {"processes": 1, "calls": 7, "rows_long": 5}
    assert "stock prod, stock again not served, ours prod-record" in p.stdout and "tokens: reproduced: 2 requests replayed" in p.stdout
    assert "advice: not measured: the scheduler replay refuses the pass: " in p.stdout
    # read again without a GPU and without the node's directory, the earlier job says the same
    shutil.rmtree(private)
    again = bash(script, "--report", str(run_dir / "out"), "--load", "chat=16", "--replay", "chat@16", "--again", "none",
                 PRIVATE=str(private))
    assert again.returncode == 0 and again.stdout == p.stdout
    # the serve ended well and the replay left nothing: the job fails, and says the replay did not end
    shutil.rmtree(run_dir / "out/replay-compiled")
    (run_dir / "out/replay-compiled.log").write_text("serve-tax: no arrival of load chat@16 rep 1\n")
    p = bash(script, *argv, run_dir=run_dir)
    assert p.returncode == 1 and "the replay: DID NOT END; its log ends: serve-tax: no arrival of load chat@16 rep 1" in p.stdout
    assert p.stdout.endswith("FAILED: a replay was asked for and the tokens line is not measured\n")
    p = bash(script, "--report", str(run_dir / "out"), "--load", "chat=16", "--replay", "chat@16", "--again", "none")
    assert p.returncode == 1 and "FAILED: a replay was asked for and the tokens line is not measured" in p.stdout
    # a job whose time ends (SIGTERM) still reports what was served and replayed by then
    p = bash(script, *argv, run_dir=run_dir, STAND_IN_TERM="1", TERM_WAIT_S="0")
    assert p.returncode == 143 and "ended early (SIGTERM): the report is of what was served and replayed by then" in p.stdout
    assert "  prod: up in 300 s; the load served 1 time, 200 requests" in p.stdout and "the replay: DID NOT END" in p.stdout


def test_the_runner_serves_ours_alone_and_takes_the_benchmarks_missing_cost_as_no_failure(tmp_path: Path):
    script, run_dir = runner(tmp_path), tmp_path / "run"
    limit = ("--max-model-len", "262144")
    bench_arm(run_dir / "out", "prod-record", {"chat@16": [299.0]}, ids="chat@16", argv=limit)
    replayed(run_dir / "out", "prod-record", load="chat@16", served_under=TOPK_UNDER)
    argv = ("--client", "vllm-bench", "--model", "glm53", "--stock", "none", "--load", "chat=16", "--reps", "1",
            "--replay", "chat@16", "--pass-prompts", "chat=8")
    # the benchmark's runner ends 1 on its cost step, which has no stock arm to be over: the job's exit is the report's
    p = bash(script, *argv, run_dir=run_dir, STAND_IN_RC="1")
    assert p.returncode == 0, p.stdout + p.stderr
    s = started(run_dir)
    assert s["argv"] == "--model glm53 --arms prod-record --loads chat=16 --reps 1"
    assert (s["REPLAY"], s["REPLAY_FROM"]) == ("chat@16", str(run_dir / "out/prod-record"))
    assert "prod_repro: context limit 262,144 tokens, given outright to prod-record; with the limit left to it" in p.stdout
    assert "prod_repro: load chat; stock not served, stock again not served, ours prod-record; run r-1" in p.stdout
    assert "settings of prod-record:\n  prefix caching: on (the engine's configuration)\n" in p.stdout
    assert "  context limit: 4,096 tokens (the engine took it; --max-model-len 262144), the model's own limit not read" in p.stdout
    assert "tokens: reproduced: 2 requests replayed" in p.stdout
    assert "cost: not measured: no stock arm was served: the job serves prod-record alone\n" in p.stdout
    assert "prod_repro: no stock arm was served (--stock none), so the benchmark's own cost step had nothing to be over" in p.stdout
    # read again, the job says the same of its arms
    again = bash(script, "--report", str(run_dir / "out"), "--load", "chat=16", "--replay", "chat@16", "--stock", "none")
    assert again.returncode == 0 and "stock not served, stock again not served, ours prod-record" in again.stdout
    # where ours ended on an error the runner's 1 stands, and so it does where the replay differs or did not end
    meta = run_dir / "out/prod-record/meta.json"
    kept = meta.read_text()
    meta.write_text(json.dumps(json.loads(kept) | {"error": "RuntimeError: the server exited 1"}))
    p = bash(script, *argv, run_dir=run_dir, STAND_IN_RC="1")
    assert p.returncode == 1 and "prod-record: FAILED: RuntimeError: the server exited 1" in p.stdout
    assert "the job's exit is the report's" not in p.stdout
    meta.write_text(kept)
    shutil.rmtree(run_dir / "out/replay-compiled")
    p = bash(script, *argv, run_dir=run_dir, STAND_IN_RC="1")
    assert p.returncode == 1 and p.stdout.endswith("FAILED: a replay was asked for and the tokens line is not measured\n")
    # any other exit of the runner is the job's
    p = bash(script, *argv, run_dir=run_dir, STAND_IN_RC="76")
    assert p.returncode == 76
    # the corpus client's runner has no cost step: its exit stands
    assert bash(script, "--stock", "none", "--ours", "record", "--requests", "8", run_dir=run_dir, STAND_IN_RC="1").returncode == 1
    assert started(run_dir)["ARMS"] == "record"


def test_the_runner_replays_an_earlier_jobs_pass_and_serves_nothing(tmp_path: Path):
    script, run_dir, earlier = runner(tmp_path), tmp_path / "run", tmp_path / "earlier/out"
    bench_arm(earlier, "prod-record", {"chat@16": [299.0]}, ids="chat@16", argv=("--max-model-len", "262144"))
    argv = ("--client", "vllm-bench", "--model", "glm53", "--replay-of", str(earlier), "--load", "chat=16", "--replay", "chat@16")
    # the benchmark's runner is given no arms, the pass and ours's directory, copied into this job's out/; where it
    # leaves no replay (the stand-in replays nothing) the job fails for the replay it asked for
    p = bash(script, *argv, run_dir=run_dir)
    s = started(run_dir)
    assert (s["argv"], s["ARMS"], s["CLIENT"], s["MODEL_ID"]) == ("", "unset", "corpus", "glm53"), p.stdout + p.stderr
    assert (s["REPLAY"], s["REPLAY_FROM"], s["REPLAY_MODES"]) == ("chat@16", str(run_dir / "out/prod-record"), "compiled")
    assert (run_dir / "out/prod-record/meta.json").read_bytes() == (earlier / "prod-record/meta.json").read_bytes()
    assert p.returncode == 1 and p.stdout.endswith("FAILED: a replay was asked for and the tokens line is not measured\n")
    # with the replay the runner leaves, the report is of ours alone: the tokens, and no cost
    replayed(run_dir / "out", "prod-record", load="chat@16", served_under=TOPK_UNDER)
    p = bash(script, *argv, run_dir=run_dir)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "prod_repro: load chat; stock not served, stock again not served, ours prod-record; run r-1" in p.stdout
    assert "tokens: reproduced: 2 requests replayed" in p.stdout
    assert "cost: not measured: no stock arm was served: the job serves prod-record alone\n" in p.stdout
    # the top-k's counters of the arm's serving are the earlier job's, carried; the replay's are this job's own
    served = {"processes": 9, "calls": 65536, "rows_long": 20496064}
    (earlier / "topk_counters.json").write_text(json.dumps({"replay": {"processes": 9, "calls": 1, "rows_long": 1}, "served": served}))
    p = bash(script, *argv, run_dir=run_dir, PRIVATE=str(tmp_path / "private"), STAND_IN_TOPK="1")
    assert p.returncode == 0, p.stdout + p.stderr
    assert ("  the canonical top-k in the replay: 1 process kept counters: at least 7 calls of the indexer's top-k, 5 rows "
            "of more than its keys rewritten\n"
            "  the canonical top-k while serving: 9 processes kept counters: at least 65,536 calls of the indexer's top-k, "
            "20,496,064 rows of more than its keys rewritten\n") in p.stdout
    assert json.loads((run_dir / "out/topk_counters.json").read_text()) == {
        "replay": {"processes": 1, "calls": 7, "rows_long": 5}, "served": served}
    # and where this job's replay kept none, the earlier job's replay's are not shown as its own
    p = bash(script, *argv, run_dir=run_dir, PRIVATE=str(tmp_path / "private-2"))
    assert "  the canonical top-k in the replay: NO process kept counters: nothing shows it ran there\n" in p.stdout
    assert "  the canonical top-k while serving: 9 processes kept counters" in p.stdout
    # a replay that fails is the job's failure, whatever --stock says
    shutil.rmtree(run_dir / "out/replay-compiled")
    p = bash(script, *argv, "--stock", "none", run_dir=run_dir, STAND_IN_RC="1")
    assert p.returncode == 1 and "the job's exit is the report's" not in p.stdout
    # what it refuses: no pass named, an out/ without ours, and a job directory that holds another job's arm
    p = bash(script, "--model", "glm53", "--replay-of", str(earlier), "--load", "chat=16", run_dir=tmp_path / "other")
    assert p.returncode == 2 and "--replay-of OUT --load NAME --replay PASS" in p.stdout
    p = bash(script, "--model", "glm53", "--replay-of", str(tmp_path), "--load", "chat=16", "--replay", "chat@16", run_dir=tmp_path / "other")
    assert p.returncode == 2 and f"--replay-of {tmp_path} holds no prod-record/meta.json" in p.stdout
    meta = earlier / "prod-record/meta.json"
    meta.write_text(json.dumps(json.loads(meta.read_text()) | {"run_id": "r-2"}))
    p = bash(script, *argv, run_dir=run_dir)
    assert p.returncode == 2 and "is there already and is not" in p.stdout


def test_the_runner_reads_an_earlier_job_again_and_refuses_what_it_cannot_run(tmp_path: Path):
    script = runner(tmp_path)
    out = job(tmp_path)
    replayed(out)
    p = bash(script, "--report", str(out), "--load", LOAD, *ARMS)
    assert p.returncode == 0 and p.stdout == (out / "prod_repro.txt").read_text() and "tokens: reproduced" in p.stdout
    # with no arm named, an earlier job is read as the arms it served
    assert bash(script, "--report", str(out), "--load", LOAD).stdout == p.stdout
    assert P.main(["report", str(out), "--load", LOAD]) == 0 and (out / "prod_repro.txt").read_text() == p.stdout
    for argv, said in ((("--report", str(out)), "--report OUT --load NAME"), (("--sample", "3"), "unknown flag --sample"),
                       (("--load",), "--load takes a value"), (("--client", "curl"), "--client corpus or vllm-bench"),
                       (("--sampled",), "--sampled is a pass of vLLM's client"), ((), "run this under research run")):
        p = bash(script, *argv)
        assert p.returncode != 0 and said in p.stdout + p.stderr, argv
    p = bash(script, "--stop-at", "soon", run_dir=tmp_path / "empty")
    assert p.returncode == 2 and "--stop-at soon is not a time" in p.stdout
    # a job that served nothing has no report
    (tmp_path / "empty").mkdir()
    p = bash(script, run_dir=tmp_path / "empty", STAND_IN_RC="2")
    assert p.returncode == 2 and "nothing was served" in p.stdout
