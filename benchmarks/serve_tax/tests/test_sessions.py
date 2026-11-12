"""The arms at production settings and the sessions loads: their data, the requests a corpus of sessions gives each
user, a pass sent by one client a user, and the cost on such a load, on synthetic records."""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import serve_tax as S

GLM = S.MODELS["glm53"]


def test_the_production_arms_are_the_recipes_settings_and_are_costed_over_prod():
    prod, again, record = (S.ARMS[n] for n in ("prod", "prod-again", "prod-record"))
    assert prod.parent is None and again.parent == record.parent == "prod"
    assert prod.args == again.args == record.args == S.PROD and record.plugin == ("record",) and not prod.plugin
    argv = S.server_argv(prod, GLM, Path("/ck"), 8000, max_model_len=131072)
    assert S.server_argv(record, GLM, Path("/ck"), 8000, max_model_len=131072) == argv
    # the recipe's settings, as the pin spells them, on the engine every arm starts
    for flag, value in (("--kv-cache-dtype", "fp8"), ("--gpu-memory-utilization", "0.95"), ("--tool-call-parser", "glm47"),
                        ("--reasoning-parser", "glm47"), ("--tensor-parallel-size", "8"), ("--max-num-seqs", "256"),
                        ("--max-model-len", "131072")):
        assert argv[argv.index(flag) + 1] == value, flag
    assert "--enable-prefix-caching" in argv and "--enable-auto-tool-choice" in argv
    assert "--no-enable-prefix-caching" not in argv and "--enforce-eager" not in argv
    assert json.loads(argv[argv.index("--speculative-config") + 1]) == {"method": "mtp", "num_speculative_tokens": 5}
    # a second lifetime starts the same engine, the record another; and stock at vLLM's defaults is not this engine
    assert S.same_engine("prod", "prod-again") and not S.same_engine("prod", "prod-record")
    assert not S.same_engine("prod", "compiled") and not S.same_engine("prod-again", "compiled-again")
    assert S.changes_of("prod") == S.changes_of("prod-again") == [] and S.changes_of("prod-record") == ["the step record"]
    # an arm is costed over the stock engine it is made from
    assert [S.root_of(a) for a in ("prod", "prod-again", "prod-record")] == ["prod"] * 3
    assert {S.root_of(a) for a in S.ARMS if not a.startswith("prod")} == {"compiled"} == {S.root_of("no such arm")}
    # the settings are one model's recipe
    for arm in ("prod", "prod-again", "prod-record"):
        S.check_model(arm, "glm53")
        with pytest.raises(SystemExit, match=f"arm {arm} holds vLLM's recipe for glm53.*the model is moonlight"):
            S.check_model(arm, "moonlight")
    S.check_model("compiled", "moonlight")


def test_the_pinned_corpus_is_a_whole_sha512_and_a_commit_of_its_repository():
    for name in (S.AGENTIC, S.AGENTIC_FULL):
        load = S.LOADS[name]
        assert load.file is S.AGENTX and load.sessions and not load.number and not load.plan and load.dataset == S.TIMED_TRACE
    ds = S.AGENTX
    assert re.fullmatch(r"[0-9a-f]{128}", ds.sha512) and re.fullmatch(r"[0-9a-f]{40}", ds.revision) and ds.size > 0
    assert (S.LOADS[S.AGENTIC].context, S.LOADS[S.AGENTIC_FULL].context) == (131072, 1048576)


def turn(n_in: int, n_out: int, ids: list[int]) -> dict:
    return {"t": 0.0, "model": "m", "in": n_in, "out": n_out, "hash_ids": ids, "api_time": 1.0, "type": "s"}


def chain(sizes: list[tuple[int, int]], first: int = 0) -> list[dict]:
    """A session's own requests: each prompt of whole blocks begins with the blocks of the one before it."""
    return [turn(n, m, list(range(first, first + n // S.BLOCK))) for n, m in sizes]


#: four sessions. The first holds a subagent's group between its turns; the second's second turn is 640 tokens; the
#: third's fourth is 448
SESSIONS = [
    [*chain([(128, 8), (256, 8)]),
     {"t": 1.0, "type": "subagent", "requests": [turn(64, 4, [900]), turn(128, 4, [900, 901])], "models": ["h"]},
     *chain([(320, 8)])],
    chain([(64, 8), (640, 8)], first=10),
    chain([(64, 4), (192, 4), (256, 4), (448, 4)], first=20),
    chain([(64, 4), (128, 4)], first=30),
]


def corpus(d: Path, sessions: list[list[dict]] = SESSIONS, block: int = 64) -> Path:
    f = d / "traces.jsonl"
    f.write_text("".join(json.dumps({"id": f"s{k}", "models": ["m"], "block_size": block, "hash_id_scope": "local",
                                     "requests": rs}) + "\n" for k, rs in enumerate(sessions)))
    return f


def lengths(lines: list[dict], user: int) -> list[tuple[int, int]]:
    return [(x["input_length"], x["output_length"]) for x in lines if x["user"] == user]


def test_a_sessions_loads_requests_are_each_users_sessions_turns_in_order(tmp_path: Path):
    """A session is its own chain without its subagents' groups, cut at the first request that does not fit the
    context; sessions go in the file's order to the user with the fewest turns, and each user sends the pass's
    turns."""
    f = corpus(tmp_path)
    lines = S.session_lines(f, users=2, turns=3, context=400)
    assert all(set(x) == {"timestamp", "input_length", "output_length", "hash_ids", "user"} and x["timestamp"] == 0.0
               for x in lines)
    # the first session is the first user's, without its subagent's requests; the second has one turn that fits and is
    # left out; the third is the second user's, cut before the turn of 448 tokens
    assert lengths(lines, 0) == [(128, 8), (256, 8), (320, 8)] and lengths(lines, 1) == [(64, 4), (192, 4), (256, 4)]
    # a block's id is its session's: a turn begins with the blocks of the turn before it, and no session shares one
    a, b = ([x["hash_ids"] for x in lines if x["user"] == j] for j in (0, 1))
    assert a == [[0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 4]] and b[2][:3] == b[1] == [2 << 32 | h for h in (20, 21, 22)]
    assert S.planned(lines) == {"requests": 6, "prompt_tokens": 1216, "repeated_tokens": 128 + 256 + 64 + 192, "span_s": 0.0}
    # a user whose session ends takes the next, and its last is cut at the pass's turns
    one = S.session_lines(f, users=1, turns=5, context=400)
    assert lengths(one, 0) == [(128, 8), (256, 8), (320, 8), (64, 4), (192, 4)]
    # with room for it the second session is whole, and the third goes to the user it leaves short
    wide = S.session_lines(f, users=2, turns=3, context=131072)
    assert lengths(wide, 0) == [(128, 8), (256, 8), (320, 8)] and lengths(wide, 1) == [(64, 8), (640, 8), (64, 4)]
    with pytest.raises(SystemExit, match="too few sessions for 2 users of 4 turns each within a context of 400 tokens: "
                                         "the shortest has 3"):
        S.session_lines(f, users=2, turns=4, context=400)
    with pytest.raises(SystemExit, match="line 1: a session in blocks of 32 tokens, not 64"):
        S.session_lines(corpus(tmp_path, block=32), 1, 2, 400)


def test_a_sessions_loads_passes_are_its_users_and_the_turns_each_sends(tmp_path: Path):
    f, load = corpus(tmp_path), S.LOADS[S.AGENTIC]
    assert S.parse_loads("agentic") == {"agentic": (4,)} and S.parse_loads("agentic=2+8") == {"agentic": (2, 8)}
    with pytest.raises(SystemExit, match="agentic is served in a run of its own"):
        S.parse_loads("chat,agentic")
    loads = S.parse_loads("agentic=1+2")
    assert S.context_of(loads, None) == 131072 and S.parse_pass_prompts("agentic=3", loads) == {"agentic": 3}
    # the lines are of the highest rung's users, each with the turns given; the rungs are the load's own
    assert S.passes_of(loads, {"agentic": str(f)}, {"agentic": 3}) == (loads, {"agentic": S.session_lines(f, 2, 3, 131072)})
    # a pass's prompts are the turns each user sends; a warm-up's are each user's first two
    assert [S.pass_prompts(load, 4, None, w, n) for w, n in ((None, None), (None, 3), (20.0, None), (20.0, 1))] == [24, 3, 2, 1]
    # a user's client sends its file in order with one request in flight, whatever the file's arrivals
    argv = S.bench_argv(GLM, Path("/ck"), 1, load, 1, tmp_path, "agentic@4-r1-u0", dataset="/u0.jsonl", prompts=24)
    at = argv.index("--dataset-name")
    assert argv[at:at + 12] == ["--dataset-name", "timed_trace", "--timed-trace-chunk-hash-size", "64", "--dataset-path",
                                "/u0.jsonl", "--num-prompts", "24", "--seed", "0", "--no-self-timed", "--request-rate"]
    assert argv[argv.index("--max-concurrency") + 1] == "1" and argv[argv.index("--request-id-prefix") + 1] == "agentic@4-r1-u0-"
    assert argv[argv.index("--goodput") + 1:][:2] == ["tpot:80", "--percentile-metrics"]


def test_each_user_has_a_file_of_its_own_written_once(tmp_path: Path):
    lines = S.session_lines(corpus(tmp_path), 2, 3, 400)
    whole = S.write_trace(tmp_path / "agentic.trace.jsonl", lines)
    files = S.user_files(whole, 2)
    assert files == [str(tmp_path / f"agentic.trace.user-{j}.jsonl") for j in (0, 1)]
    for j, f in enumerate(files):
        assert [json.loads(x) for x in Path(f).read_text().splitlines()] == [x for x in lines if x["user"] == j]
    Path(files[0]).write_text("kept\n")
    assert S.user_files(whole, 1) == files[:1] and Path(files[0]).read_text() == "kept\n"
    with pytest.raises(SystemExit, match="holds 2 users' requests, and the pass is of 3"):
        S.user_files(whole, 3)


def result(completed: int, duration: float, tok: float, good: float | None, failed: int = 0) -> dict:
    """What `vllm bench serve --save-result` writes of one user's pass, the keys a pass's metrics read."""
    return {"completed": completed, "failed": failed, "duration": duration, "total_input_tokens": 1000 * completed,
            "total_output_tokens": int(tok * duration), "request_throughput": completed / duration,
            "request_goodput": None if good is None else good / duration, "output_throughput": tok,
            "total_token_throughput": tok + 500.0, "median_ttft_ms": 400.0, "p99_itl_ms": 70.0}


def test_a_pass_of_a_sessions_load_is_one_client_a_user_and_its_rates_are_the_users_sum(tmp_path: Path):
    whole = S.write_trace(tmp_path / "agentic.trace.jsonl", S.session_lines(corpus(tmp_path), 2, 3, 400))
    users = {"u0": result(3, 60.0, 100.0, 3.0), "u1": result(2, 80.0, 50.0, 1.0, failed=1)}
    calls = []

    def run(argv, stdout, stderr, env):
        calls.append(argv)
        assert env["PYTHONHASHSEED"] == "0"
        d = Path(argv[argv.index("--result-dir") + 1])
        user = argv[argv.index("--request-id-prefix") + 1].split("-")[-2]
        (d / S.RESULT).write_text(json.dumps(users[user]))
        return NS(returncode=0)

    d = tmp_path / "agentic@2" / "rep-1"
    m = S.drive_bench(8000, GLM, Path("/ck"), S.LOADS[S.AGENTIC], 2, "agentic@2-r1", d, dataset=whole, prompts=3, run=run,
                      clock=iter([0.0, 90.0]).__next__, now=lambda: 1700.0)
    files = S.user_files(whole, 2)
    want = [S.bench_argv(GLM, Path("/ck"), 8000, S.LOADS[S.AGENTIC], 1, d / f"user-{j}", f"agentic@2-r1-u{j}", files[j], 3)
            for j in (0, 1)]
    assert sorted(calls) == sorted(want) and m["argv"] == want[0][1:]
    assert (m["requests"], m["unfinished"], m["concurrency"], m["wall_s"], m["tag"]) == (6, 1, 2, 90.0, "agentic@2-r1")
    r = json.loads((d / S.RESULT).read_text())
    assert r == S.merged([users["u0"], users["u1"]]) and (d / "user-1" / "bench.log").exists()
    assert (r["users"], r["completed"], r["failed"], r["duration"], r["total_output_tokens"]) == (2, 5, 1, 80.0, 10000)
    # a user's rate is over its own time, and the pass's is their sum
    assert (r["output_throughput"], r["total_token_throughput"]) == (150.0, 1150.0)
    assert r["request_throughput"] == pytest.approx(3 / 60 + 2 / 80) and r["request_goodput"] == pytest.approx(3 / 60 + 1 / 80)
    b = S.bench_metrics(r)
    assert (b["sent"], b["ok"], b["met"], b["output_tok_per_s"], b["prompt_tok_per_s"]) == (6, 5, 4, 150.0, 1000.0)
    assert b["ttft_p50_s"] is None and b["itl_p99_s"] is None and b["slo_attained"] == pytest.approx(4 / 6)
    assert S.merged([result(3, 60.0, 100.0, None), users["u1"]])["met"] is None
    # one user's client that fails fails the pass
    with pytest.raises(RuntimeError, match="exited 1"):
        S.drive_bench(8000, GLM, Path("/ck"), S.LOADS[S.AGENTIC], 2, "x", tmp_path / "bad", dataset=whole, prompts=3,
                      run=lambda argv, stdout, stderr, env: NS(returncode=1), clock=lambda: 0.0, now=lambda: 0.0)


#: what a job's arms planned for load agentic: four users of 24 turns each
PLAN = {"requests": 96, "prompt_tokens": 8000000, "repeated_tokens": 7600000, "span_s": 0.0, "sha256": "a" * 64, "users": 4}


def served(name: str, rates: list[float], load: str = S.AGENTIC, cached: float = 0.94, ok: int = 96, plan: dict = PLAN) -> dict:
    """An arm's metrics with a sessions load served by four users, a rep each rate, in thousands of output tokens a
    second, each rep with the share of its prompt tokens the prefix cache served."""
    reps = [{"rep": i + 1, "rate": None, "concurrency": 4, "sent": 96, "ok": ok, "met": ok, "goodput_req_per_s": 0.5,
             "req_per_s": 0.5, "output_tok_per_s": 1000 * r, "prompt_tok_per_s": 50000.0, "output_tokens": 60000,
             "prompt_tokens": 8000000, "prompt_cached_share": cached} for i, r in enumerate(rates)]
    return {"arm": name, "parent": S.ARMS[name].parent, "what": name, "run_id": "r-1", "commit": "c" * 40,
            "vllm": "0.1+gd9105ea8", "model_id": "glm53", "slo": {"ttft_s": 2.0, "tpot_s": 0.08},
            "loads": {S.pass_name(load, 4): reps}, "planned": {load: plan}}


OURS = [1.99, 1.988, 1.992, 1.99, 1.989, 1.991]


def arms() -> list[dict]:
    return [served("prod", [2.0] * 6), served("prod-record", OURS), served("prod-again", [1.998] * 6)]


def test_the_cost_on_a_sessions_load_is_over_stock_at_the_same_settings_with_its_noise_floor():
    r = S.session_cost(arms(), "prod-record", "prod")
    ratios = sorted(2.0 / x - 1 for x in OURS)
    assert r["value"] == round(100 * (ratios[2] + ratios[3]) / 2, 2) == 0.5
    assert r["ci95"] == [round(100 * ratios[0], 2), round(100 * ratios[5], 2)] == [0.4, 0.6]
    assert (r["metric"], r["load"], r["users"], r["reps"], r["arm"], r["reference"]) == ("session_cost", "agentic", 4, 6,
                                                                                    "prod-record", "prod")
    # the second lifetime's cost over the first is the noise floor, which widens the interval before the verdict
    assert r["noise_floor"] == {"arm": "prod-again", "value": 0.1, "ci95": [0.1, 0.1], "reps": 6}
    assert (r["target"], r["verdict"], r["same_work"]) == (1.0, "meets", True)
    assert S.session_cost(arms(), "prod-record", "prod", target=0.65)["verdict"] == "cannot tell"      # 0.6 + 0.1 is over
    assert S.session_cost(arms(), "prod-record", "prod", target=0.25)["verdict"] == "misses"           # 0.4 - 0.1 is over
    assert r["planned"]["repeated_share"] == 95.0 and r["sides"]["arm"]["cached_share"] == 94.0
    assert r["sides"]["reference"]["served_all"] and r["sides"]["reference"]["output_tok_per_s"] == pytest.approx(2000.0)
    text = S.session_text(r)
    assert text.splitlines()[0] == ("serving cost on load agentic: prod-record over prod: 0.50% [0.40%, 0.60%] of stock's "
                                    "serving compute, 6 reps, 4 users; target 1%: meets")
    assert "noise floor: prod-again over prod: 0.10% [0.10%, 0.10%], 6 reps" in text and "95.00% of them planned to repeat" in text
    # with no second lifetime the floor is not measured, and the cost of the second lifetime itself has none
    alone = S.session_cost(arms()[:2], "prod-record", "prod")
    assert alone["noise_floor"] is None and alone["verdict"] == "meets" and "noise floor: not measured" in S.session_text(alone)
    again = S.session_cost(arms(), "prod-again", "prod")
    assert again["value"] == 0.1 and again["noise_floor"] is None
    # fewer than six reps have no interval, so no verdict but "cannot tell"
    few = S.session_cost([served("prod", [2.0] * 3), served("prod-record", OURS[:3])], "prod-record", "prod")
    assert few["ci95"] is None and few["verdict"] == "cannot tell"


def test_a_sessions_cost_has_no_verdict_where_the_sides_did_not_do_the_same_work():
    ref = served("prod", [2.0] * 6)
    evicted = served("prod-record", OURS)
    evicted["loads"]["agentic@4"][2]["prompt_cached_share"] = 0.9
    r = S.session_cost([ref, evicted], "prod-record", "prod")
    assert (r["value"], r["same_work"], r["verdict"]) == (0.5, False, "cannot tell")
    assert "did not do the same work" in S.session_text(r)
    short = S.session_cost([ref, served("prod-record", OURS, ok=95)], "prod-record", "prod")
    assert not short["sides"]["arm"]["served_all"] and short["verdict"] == "cannot tell"
    assert "prod-record: 1,990.0 output tokens a second, 94.00% of its prompt tokens from its prefix cache, NOT every" in S.session_text(short)
    with pytest.raises(SystemExit, match="prod and prod-record served load agentic from different plans"):
        S.session_cost([ref, served("prod-record", OURS, plan=PLAN | {"sha256": "b" * 64})], "prod-record", "prod")
    with pytest.raises(SystemExit, match="prod served load agentic and prod-record did not"):
        S.session_cost([ref], "prod-record", "prod")
    with pytest.raises(SystemExit, match="no directory of arm prod "):
        S.session_cost([evicted], "prod-record", "prod")


def test_cost_prints_a_sessions_loads_cost_over_the_arms_own_stock(tmp_path: Path, capsys: pytest.CaptureFixture):
    files = []
    for a in arms():
        files.append(tmp_path / f"{a['arm']}.json")
        files[-1].write_text(json.dumps(a))
    assert S.main(["cost", *map(str, files), "--arm", "prod-record", "--out", str(tmp_path / "out")]) == 0
    text = capsys.readouterr().out
    assert text.startswith("serving cost on load agentic: prod-record over prod: 0.50% [0.40%, 0.60%]")
    assert (tmp_path / "out" / "serving_cost.txt").read_text() == text
    r = json.loads((tmp_path / "out" / "serving_cost.json").read_text())
    assert list(r)[:4] == ["value", "unit", "commit", "run"] and r["reference"] == "prod"
    ok, first, what, measurements = S.result_of(r)
    assert ok and first == text.splitlines()[0] and what["loads"] == ["agentic"] and what["arm"] == "prod-record"
    assert [(m["name"], m["value"], m["ci95"], m["target"], m["verdict"], m["users"]) for m in measurements] == [
        ("agentic", 0.5, [0.4, 0.6], 1.0, "meets", 4)]
    # arms that served no sessions load are costed as before, over compiled
    assert S.session_cost([served("compiled", [2.0] * 6, load="chat"), served("record", OURS, load="chat")], "record",
                          "compiled") is None
