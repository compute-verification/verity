"""The arms at the recipe's settings and the sessions loads: their data, the requests a corpus of sessions gives each
user, a pass sent by one client a user, and the cost on such a load, on synthetic records."""
from __future__ import annotations

import gzip
import hashlib
import inspect
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import serve_tax as S

GLM = S.MODELS["glm53"]


def test_the_prod_arms_start_the_published_arms_server_and_are_costed_over_prod():
    prod, again, record = (S.ARMS[n] for n in ("prod", "prod-again", "prod-record"))
    assert prod.parent == "published" and again.parent == record.parent == "prod"
    assert prod.args == again.args == record.args == () and record.plugin == ("record", "ids") and not prod.plugin
    # ours serves on the canonical top-k, as the two-stage serve does: stock loads no plugin and sets nothing
    assert (record.env, record.general) == (S.TOPK_ENV, S.TOPK) == (S.ARMS["two-stage"].env, S.ARMS["two-stage"].general)
    assert S.entry_points(record) == ["verity_serve_tax = servetax_plugin:register", "verity_dsa_topk = dsa_topk:register"]
    assert not prod.env and not again.env and not S.entry_points(prod) and not S.entry_points(again)
    assert prod.published and again.published and record.published
    context = S.published_context("prod", "glm53", None)
    argv = S.server_argv(prod, GLM, Path("/ck"), 8000, max_model_len=context)
    # the published arm's line, the recipe's speculative decoding in it once, in every one of the three
    assert argv == S.server_argv(S.ARMS["published"], GLM, Path("/ck"), 8000, max_model_len=context)
    assert S.server_argv(record, GLM, Path("/ck"), 8000, max_model_len=context) == argv
    for flag, value in (("--kv-cache-dtype", "fp8"), ("--speculative-config.method", "mtp"),
                        ("--speculative-config.num_speculative_tokens", "5"), ("--tool-call-parser", "glm47"),
                        ("--reasoning-parser", "glm47"), ("--tensor-parallel-size", "8"), ("--max-model-len", "262144")):
        assert argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value, flag
    # what the recipe leaves to vLLM stays vLLM's: the requests at once, each card's memory, the prefix cache, the mode
    assert not {"--max-num-seqs", "--gpu-memory-utilization", "--no-enable-prefix-caching", "--enforce-eager"} & set(argv)
    # the published arm and a second lifetime start prod's engine, the record another
    assert S.same_engine("prod", "prod-again") and S.same_engine("prod", "published")
    assert not S.same_engine("prod", "prod-record") and not S.same_engine("prod-again", "compiled-again")
    assert S.changes_of("prod") == S.changes_of("prod-again") == []
    assert S.changes_of("prod-record") == ["the step record", "the served ids", "the canonical top-k"]
    # an arm is costed over the nearest stock arm above it: at the recipe's settings, stock at those settings
    assert [S.stock_of(a) for a in ("prod-again", "prod-record", "prod", "published")] == ["prod", "prod", "published", "compiled"]
    assert {S.stock_of(a) for a in S.ARMS if a not in ("prod", "prod-again", "prod-record")} == {"compiled"}
    assert S.stock_of("no such arm") == "compiled" and all(n in S.ARMS and not S.changes_of(n) for n in S.STOCK)
    # the recipe is one model's; its arms serve at the job's context limit, the same for the three
    for arm in ("prod", "prod-again", "prod-record"):
        with pytest.raises(SystemExit, match=f"arm {arm} starts the server as vLLM's recipe.*no recipe for moonlight"):
            S.published_context(arm, "moonlight", None)
        assert S.published_context(arm, "glm53", 131072) == 131072


def test_an_arms_meta_holds_the_value_settings_in_force_in_the_environment_its_server_is_started_with(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A variable that changes what the engine computes can come from the job, from run.sh or from the arm; the launch
    line and the arm's own ``env`` show only the last. The record is of the dict the server is started with."""
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")                    # the job's, and a value setting by its family
    monkeypatch.setenv("VLLM_API_KEY", "a credential's value")    # named, never held
    monkeypatch.setenv("VLLM_PLUGINS", "none")                     # what `server_env` unsets is not in force
    monkeypatch.setenv("HF_HOME", "/where/not/what")               # no value setting
    monkeypatch.delenv("VERITY_DSA_TOPK", raising=False)
    site, private = tmp_path / "site", tmp_path / "private"
    env = {n: S.server_env(S.ARMS[n], tmp_path, site, None, 600.0, 600.0, private=private)
           for n in ("prod", "prod-again", "prod-record")}
    stock, again, ours = (S.value_settings(env[n]) for n in ("prod", "prod-again", "prod-record"))
    assert stock["schema"] == "verity-vllm/settings/v1" and len(stock["digest"]) == 128
    for name, value in (("NCCL_P2P_DISABLE", "1"), ("VLLM_API_KEY", "<redacted>"), ("VLLM_BATCH_INVARIANT", "0"),
                        ("VLLM_SERVER_DEV_MODE", "1")):
        assert stock["values"][name] == ours["values"][name] == value, name
    assert not {"VLLM_PLUGINS", "HF_HOME", "PYTHONPATH"} & set(stock["values"]) and "a credential's value" not in json.dumps([stock, ours])
    # stock's second lifetime is started with what its first was; ours differs from stock by the canonical top-k alone,
    # and where the top-k keeps its counters is no value setting
    assert again == stock and ours["digest"] != stock["digest"]
    assert {k: v for k, v in ours["values"].items() if stock["values"].get(k) != v} == {"VERITY_DSA_TOPK": "canonical"}
    assert set(stock["values"]) < set(ours["values"]) and env["prod-record"]["VERITY_DSA_TOPK_LOG"] == f"{private}/topk-log"
    # a record of no dict is of this process's environment, which is how a replay takes its engine's
    assert S.value_settings()["values"]["NCCL_P2P_DISABLE"] == "1"
    # `serve` records the dict it starts the server with
    assert '"settings": value_settings(env)' in inspect.getsource(S.serve)



def test_the_pinned_corpus_is_a_whole_sha512_and_a_commit_of_its_repository():
    for name in (S.AGENTIC, S.AGENTIC_FULL):
        load = S.LOADS[name]
        assert load.file is S.AGENTX and load.sessions and not load.number and not load.plan and load.dataset == S.TIMED_TRACE
    for ds in (S.AGENTX, S.AGENTX_256K):
        assert re.fullmatch(r"[0-9a-f]{128}", ds.sha512) and re.fullmatch(r"[0-9a-f]{40}", ds.revision) and ds.size > 0
    assert (S.LOADS[S.AGENTIC].context, S.LOADS[S.AGENTIC_FULL].context) == (131072, 1048576)
    # the publisher's own cut of the sessions for the recipe's context is a file of its own, at the recipe's limit: no
    # session of it is cut by ours
    at_limit = S.LOADS[S.AGENTIC_256K]
    assert at_limit.file is S.AGENTX_256K and S.AGENTX_256K.repo == S.AGENTX.repo + "-256k" and S.AGENTX_256K != S.AGENTX
    assert at_limit.context == 262144 == S.MODELS["glm53"].published_context and at_limit.sessions and not at_limit.number
    assert at_limit.dataset == S.TIMED_TRACE and S.context_of({S.AGENTIC_256K: (4,)}, 262144) == 262144
    # half the other loads' turns a user, at the same users: what the recipe's server's KV cache holds of these sessions
    assert (at_limit.prompts, at_limit.rungs) == (S.LOADS[S.AGENTIC].prompts // 2, S.LOADS[S.AGENTIC].rungs) == (12, (4,))
    with pytest.raises(SystemExit, match="load agentic-256k's requests need a context of 262144"):
        S.context_of({S.AGENTIC: (4,), S.AGENTIC_256K: (4,)}, 131072)


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
    # sessions loads share a run, its engine's context the largest of theirs; with any other load they do not
    both = S.parse_loads("agentic=4,agentic-256k")
    assert both == {"agentic": (4,), "agentic-256k": (4,)} and S.context_of(both, None) == 262144
    with pytest.raises(SystemExit, match="agentic is served in a run of its own"):
        S.parse_loads("agentic,agentic-256k,chat")
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


def test_a_pass_of_a_sessions_load_is_one_client_a_user_and_its_rates_are_over_the_pass_seconds(tmp_path: Path, monkeypatch):
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(S.tempfile, "tempdir", str(tmp_path / "tmp"))
    whole = S.write_trace(tmp_path / "agentic.trace.jsonl", S.session_lines(corpus(tmp_path), 2, 3, 400))
    # the client's detailed result: one user's requests take 20 ms a token after the first, the other's 40 ms, and
    # its last request has one token, which has no time a token
    users = {"u0": result(3, 60.0, 100.0, 3.0) | {"ttfts": [0.4] * 3, "latencies": [0.4 + 0.02 * 99] * 3, "output_lens": [100] * 3},
             "u1": result(2, 80.0, 50.0, 1.0, failed=1) | {"ttfts": [0.8] * 2, "latencies": [0.8 + 0.04 * 49, 0.8],
                                                           "output_lens": [50, 1]}}
    calls, saved = [], []

    def run(argv, stdout, stderr, env):
        calls.append(argv)
        assert env["PYTHONHASHSEED"] == "0"
        d = Path(argv[argv.index("--result-dir") + 1])
        saved.append(d)
        user = argv[argv.index("--request-id-prefix") + 1].split("-")[-2]
        (d / S.RESULT).write_text(json.dumps(users[user] | {"generated_texts": ["a served answer"], "itls": [[0.02]]}))
        return NS(returncode=0)

    d = tmp_path / "run" / "agentic@2" / "rep-1"
    m = S.drive_bench(8000, GLM, Path("/ck"), S.LOADS[S.AGENTIC], 2, "agentic@2-r1", d, dataset=whole, prompts=3, run=run,
                      clock=iter([0.0, 90.0]).__next__, now=lambda: 1700.0)
    files = S.user_files(whole, 2)
    held = sorted(saved)
    # each client saved its result outside the run's directory, in a directory gone when the pass ends
    assert [x.name for x in held] == ["user-0", "user-1"] and held[0].parent == held[1].parent
    assert held[0].parent.parent == (tmp_path / "tmp").resolve() and not held[0].parent.exists()
    want = [S.bench_argv(GLM, Path("/ck"), 8000, S.LOADS[S.AGENTIC], 1, held[j], f"agentic@2-r1-u{j}", files[j], 3)
            for j in (0, 1)]
    assert sorted(calls) == sorted(want) and m["argv"] == want[0][1:]
    assert (m["requests"], m["unfinished"], m["concurrency"], m["wall_s"], m["tag"]) == (6, 1, 2, 90.0, "agentic@2-r1")
    each = [json.loads((d / f"user-{j}" / S.RESULT).read_text()) for j in (0, 1)]
    r = json.loads((d / S.RESULT).read_text())
    assert each == [users["u0"], users["u1"]] and r == S.merged(each) and (d / "user-1" / "bench.log").exists()
    assert "a served answer" not in "".join(f.read_text() for f in d.rglob("*") if f.is_file())
    assert (r["users"], r["completed"], r["failed"], r["duration"], r["total_output_tokens"]) == (2, 5, 1, 80.0, 10000)
    # the pass's rates are over its seconds, the longest user's: 10,000 output and 5,000 prompt tokens, 5 requests
    # and 4 within the SLO in 80 s. The users' own rates summed (one ended after 60 s) and their seconds are beside
    assert (r["output_throughput"], r["total_token_throughput"]) == (125.0, 187.5)
    assert (r["request_throughput"], r["request_goodput"], r["met"]) == (5 / 80, 4 / 80, 4)
    assert (r["users_output_throughput"], r["users_duration"]) == (150.0, 140.0)
    # the times are over every user's requests: first tokens after 0.4 s (three) and 0.8 s; 20, 20, 20 and 40 ms a token
    assert (r["median_ttft_ms"], r["p90_ttft_ms"]) == (pytest.approx(400.0), pytest.approx(800.0))
    assert (r["median_tpot_ms"], r["p90_tpot_ms"], r["p99_tpot_ms"]) == tuple(map(pytest.approx, (20.0, 34.0, 39.4)))
    b = S.bench_metrics(r)
    assert (b["sent"], b["ok"], b["met"], b["output_tok_per_s"], b["prompt_tok_per_s"]) == (6, 5, 4, 125.0, 62.5)
    assert (b["users_output_tok_per_s"], b["users_s"]) == (150.0, 140.0) and "users_s" not in S.bench_metrics(users["u0"])
    assert b["ttft_p50_s"] == pytest.approx(0.4) and b["tpot_p90_s"] == pytest.approx(0.034)
    assert b["itl_p99_s"] is None and b["slo_attained"] == pytest.approx(4 / 6)
    # a result that is not the detailed one gives no time, and one with no goodput no count within the SLO
    bare = S.merged([result(3, 60.0, 100.0, None), users["u1"]])
    assert bare["met"] is None and "p90_tpot_ms" not in bare and "tpot_p90_s" not in S.bench_metrics(bare)
    # where the clients kept each request's start, the pass's seconds are from the first send of any user to the last
    # answer of any: the user that took 80 s sent first and was answered last 79.5 s later, the other began 0.5 s
    # after it; the users' own seconds are summed as they are
    timed = S.merged([users["u0"] | {"start_times": [1000.5, 1020.0, 1040.0]}, users["u1"] | {"start_times": [1000.0, 1078.7]}])
    assert timed["duration"] == pytest.approx(79.5) and timed["output_throughput"] == pytest.approx(10000 / 79.5)
    assert (timed["users_output_throughput"], timed["users_duration"]) == (150.0, 140.0)
    # one user's client that fails fails the pass
    with pytest.raises(RuntimeError, match="exited 1"):
        S.drive_bench(8000, GLM, Path("/ck"), S.LOADS[S.AGENTIC], 2, "x", tmp_path / "bad", dataset=whole, prompts=3,
                      run=lambda argv, stdout, stderr, env: NS(returncode=1), clock=lambda: 0.0, now=lambda: 0.0)


def test_a_sampled_pass_sends_the_recipes_sampling_and_each_users_name_and_ours_its_seed(tmp_path: Path, monkeypatch):
    load, user = S.LOADS[S.AGENTIC], "agentic@4-r1-u0"
    greedy = S.bench_argv(GLM, Path("/ck"), 1, load, 1, tmp_path, user, dataset="/u0.jsonl", prompts=24)
    stock = S.bench_argv(GLM, Path("/ck"), 1, load, 1, tmp_path, user, dataset="/u0.jsonl", prompts=24, sampled=True)
    argv = S.bench_argv(GLM, Path("/ck"), 1, load, 1, tmp_path, user, dataset="/u0.jsonl", prompts=24, sampled=True,
                        seeded=True)
    at = greedy.index("--temperature")
    assert greedy[at:at + 2] == ["--temperature", "0"] and not {"--extra-body", "--top-p"} & set(greedy)
    assert argv[at:at + 5] == ["--temperature", "1.0", "--top-p", "0.95", "--extra-body"]
    assert json.loads(argv[at + 5]) == {"user": user, "seed": S.user_seed(user)}
    # a stock arm's requests name their user and carry no seed, as a served request carries none; the arms that keep
    # the record, which a replay samples again from, are sent one
    assert json.loads(stock[at + 5]) == {"user": user} and stock[:at + 5] == argv[:at + 5] and stock[at + 6:] == argv[at + 6:]
    assert [S.sends_seed(S.ARMS[a]) for a in ("prod", "prod-again", "published", "compiled", "prod-record", "record")] == [
        False, False, False, False, True, True]
    # the rest of the pass is the greedy one's: its requests, their lengths, its ids
    assert argv[:at] == greedy[:at] and argv[at + 6:] == greedy[at + 2:] and "--ignore-eos" in argv
    # a seed is its user's own, 63 bits of SHA-512 over the user's name
    seeds = [S.user_seed(f"agentic@4-r{r}-u{j}") for r in (1, 2) for j in range(4)]
    assert len(set(seeds)) == 8 and all(0 <= x < 2 ** 63 for x in seeds)
    assert seeds[0] == int.from_bytes(hashlib.sha512(b"serve-tax/user-seed/v1/" + user.encode()).digest()[:8], "big") >> 1
    # and its server lifetime's: the users' names repeat in every lifetime, and under one seed the arm's lifetimes
    # would serve one sample of text, whose chance no interval over the lifetimes holds
    lives = [S.user_seed(user, k) for k in (None, 1, 2, 3)]
    assert len(set(lives)) == 4 and lives[0] == seeds[0] and all(0 <= x < 2 ** 63 for x in lives)
    assert lives[2] == int.from_bytes(hashlib.sha512(b"serve-tax/user-seed/v1/" + user.encode() + b"/lifetime-2")
                                      .digest()[:8], "big") >> 1
    second = S.bench_argv(GLM, Path("/ck"), 1, load, 1, tmp_path, user, dataset="/u0.jsonl", prompts=24, sampled=True,
                          seeded=True, lifetime=2)
    assert json.loads(second[at + 5]) == {"user": user, "seed": lives[2]} and second[:at + 5] == argv[:at + 5]
    assert S.bench_argv(GLM, Path("/ck"), 1, load, 1, tmp_path, user, dataset="/u0.jsonl", prompts=24, sampled=True,
                        lifetime=2) == stock
    # each user's client names its own user
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(S.tempfile, "tempdir", str(tmp_path / "tmp"))
    whole = S.write_trace(tmp_path / "agentic.trace.jsonl", S.session_lines(corpus(tmp_path), 2, 3, 400))
    bodies = []

    def run(argv, stdout, stderr, env):
        bodies.append(json.loads(argv[argv.index("--extra-body") + 1]))
        (Path(argv[argv.index("--result-dir") + 1]) / S.RESULT).write_text(json.dumps(result(3, 60.0, 100.0, 3.0)))
        return NS(returncode=0)

    S.drive_bench(8000, GLM, Path("/ck"), load, 2, "agentic@2-r1", tmp_path / "rep-1", dataset=whole, prompts=3,
                  sampled=True, seeded=True, run=run, clock=lambda: 0.0, now=lambda: 0.0)
    assert sorted(bodies, key=lambda b: b["user"]) == [{"user": u, "seed": S.user_seed(u)}
                                                       for u in ("agentic@2-r1-u0", "agentic@2-r1-u1")]
    del bodies[:]
    S.drive_bench(8000, GLM, Path("/ck"), load, 2, "agentic@2-r1", tmp_path / "rep-1-life-3", dataset=whole, prompts=3,
                  sampled=True, seeded=True, lifetime=3, run=run, clock=lambda: 0.0, now=lambda: 0.0)
    assert sorted(bodies, key=lambda b: b["user"]) == [{"user": u, "seed": S.user_seed(u, 3)}
                                                       for u in ("agentic@2-r1-u0", "agentic@2-r1-u1")]
    del bodies[:]
    S.drive_bench(8000, GLM, Path("/ck"), load, 2, "agentic@2-r1", tmp_path / "rep-1-stock", dataset=whole, prompts=3,
                  sampled=True, run=run, clock=lambda: 0.0, now=lambda: 0.0)
    assert sorted(bodies, key=lambda b: b["user"]) == [{"user": u} for u in ("agentic@2-r1-u0", "agentic@2-r1-u1")]


#: what a job's arms planned for load agentic: four users of 24 turns each
PLAN = {"requests": 96, "prompt_tokens": 8000000, "repeated_tokens": 7600000, "span_s": 0.0, "sha256": "a" * 64, "users": 4}


def served(name: str, rates: list[float], load: str = S.AGENTIC, cached: float = 0.94, ok: int = 96, plan: dict = PLAN) -> dict:
    """An arm's metrics with a sessions load served by four users, a rep each rate, in thousands of output tokens a
    second, each rep with the share of its prompt tokens the prefix cache served, 52,000 prompt and output tokens a
    second, 25 ms a token at the P90, 2,000 engine steps and half its draft tokens accepted."""
    reps = [{"rep": i + 1, "rate": None, "concurrency": 4, "sent": 96, "ok": ok, "met": ok, "goodput_req_per_s": 0.5,
             "req_per_s": 0.5, "output_tok_per_s": 1000 * r, "prompt_tok_per_s": 50000.0, "output_tokens": 60000,
             "prompt_tokens": 8000000, "prompt_cached_share": cached, "total_tok_per_s": 52000.0, "tpot_p90_s": 0.025,
             "steps": 2000, "draft_accepted_share": 0.5, "accepted_per_draft": 2.5} for i, r in enumerate(rates)]
    return {"arm": name, "parent": S.ARMS[name].parent, "what": name, "run_id": "r-1", "commit": "c" * 40,
            "vllm": "0.1+gd9105ea8", "model_id": "glm53", "slo": {"ttft_s": 2.0, "tpot_s": 0.08},
            "loads": {S.pass_name(load, 4): reps}, "planned": {load: plan}}


OURS = [1.99, 1.988, 1.992, 1.99, 1.989, 1.991]


def arms() -> list[dict]:
    return [served("prod", [2.0] * 6), served("prod-record", OURS), served("prod-again", [1.998] * 6)]


def test_the_cost_on_a_sessions_load_is_over_stock_at_the_same_settings_with_its_noise_floor():
    r = S.session_cost(arms(), "prod-record", "prod")
    ratios = sorted(2.0 / x - 1 for x in OURS)
    # one server lifetime an arm: the cost is ours's seconds for its passes over stock's, less one (stock's passes
    # took the same time each, so the mean of the passes' ratios), with no interval over lifetimes, so no verdict
    assert r["value"] == round(100 * sum(ratios) / 6, 2) == 0.5 and r["ci95"] is None and r["lifetimes"] == 1
    assert (r["target"], r["verdict"], r["same_work"]) == (1.0, "cannot tell", True)
    assert all(S.session_cost(arms(), "prod-record", "prod", target=t)["verdict"] == "cannot tell" for t in (0.25, 5.0))
    # beside it, the passes taken as independent, the median over them with its interval, and the spread among them
    assert r["passes"] == {"value": round(100 * (ratios[2] + ratios[3]) / 2, 2),
                           "ci95": [round(100 * ratios[0], 2), round(100 * ratios[5], 2)], "reps": 6}
    assert r["passes"] == {"value": 0.5, "ci95": [0.4, 0.6], "reps": 6}
    assert r["spread"] == {"between": None, "within": 0.07} and r["by_lifetime"] is None
    assert (r["metric"], r["load"], r["users"], r["reps"], r["arm"], r["reference"]) == ("session_cost", "agentic", 4, 6,
                                                                                    "prod-record", "prod")
    # the second lifetime's cost over the first is the noise floor, beside the cost: one pair of lifetimes, no interval
    assert r["noise_floor"] == {"arm": "prod-again", "value": 0.1, "ci95": None, "reps": 6, "pairs": 1, "differs": None}
    assert r["planned"]["repeated_share"] == 95.0 and r["sides"]["arm"]["cached_share"] == 94.0
    stock = r["sides"]["reference"]
    assert stock["served_all"] and stock["output_tok_per_s"] == pytest.approx(2000.0)
    # beside the rate: tokens a second a GPU of the model's eight, a user's tokens a second at the P90 time a token,
    # the engine steps a pass took and what the engine's drafts gave
    assert (stock["tok_per_s_a_gpu"], stock["user_tok_per_s_p90"]) == (6500.0, pytest.approx(40.0)) and r["gpus"] == 8
    assert (stock["steps"], stock["draft_accepted_share"], stock["tokens_per_draft"]) == (2000, 50.0, 3.5)
    text = S.session_text(r)
    assert text.splitlines()[0] == ("serving cost on load agentic: prod-record over prod: 0.50% of stock's serving "
                                    "compute over 1 server lifetime an arm, 4 users; target 1%: cannot tell")
    assert ("  the cost is the mean of the lifetimes' costs, a lifetime's being prod-record's seconds for the passes "
            "both arms counted in it over prod's, less one, a pass's seconds from its users' first send to the last "
            "one's last answer; one lifetime an arm gives no interval and so no verdict\n"
            "  passes taken as independent: 0.50% [0.40%, 0.60%], 6 reps, the median over them with its interval; in no "
            "verdict\n"
            "  the spread of the cost (standard deviations): not measured between lifetimes, 0.07 points between a "
            "lifetime's passes\n"
            "  noise floor: prod-again over prod: 0.10% over 1 pair of lifetimes, 6 reps; beside the cost, not added to "
            "its interval\n") in text
    assert "95.00% of them planned to repeat" in text and "differs from" not in text
    assert ("  prod: 2,000.0 output tokens a second (6,500.0 prompt and output tokens a second a GPU), 40.0 output tokens "
            "a second a user at the P90 time a token, 94.00% of its prompt tokens from its prefix cache, 2,000 engine "
            "steps a pass, 50.00% of its draft tokens accepted, 3.50 tokens a draft, every request served\n") in text
    # with no second lifetime the floor is not measured, and the cost of the second lifetime itself has none
    alone = S.session_cost(arms()[:2], "prod-record", "prod")
    assert alone["noise_floor"] is None and alone["value"] == 0.5 and "noise floor: not measured" in S.session_text(alone)
    again = S.session_cost(arms(), "prod-again", "prod")
    assert again["value"] == 0.1 and again["noise_floor"] is None
    # fewer than six passes have no interval of their own either
    few = S.session_cost([served("prod", [2.0] * 3), served("prod-record", OURS[:3])], "prod-record", "prod")
    assert few["ci95"] is None and few["passes"] == {"value": 0.5, "ci95": None, "reps": 3} and few["verdict"] == "cannot tell"


def test_a_sessions_cost_has_the_other_readings_of_its_passes_beside_and_in_no_verdict():
    # a pass of ours takes 1% longer by its rate; its GPUs were busy 2% longer, its users' seconds sum to 3% more,
    # and its users' own rates sum to the same as stock's
    ref, ours = served("prod", [2.0] * 3), served("prod-record", [2.0 / 1.01] * 3)
    for arm, busy, seconds in ((ref, 800.0, 300.0), (ours, 816.0, 309.0)):
        for rep in arm["loads"]["agentic@4"]:
            rep |= {"busy_gpu_s": busy, "users_s": seconds, "users_output_tok_per_s": 2100.0}
    r = S.session_cost([ref, ours], "prod-record", "prod")
    assert r["value"] == 1.0 and r["readings"] == {"busy_gpu_s": 2.0, "users_s": 3.0, "users_rates": 0.0}
    assert ("  other readings of the same passes, in no verdict: 2.00% by the GPUs' busy seconds, 3.00% by the users' "
            "seconds summed, 0.00% by the users' own rates summed\n") in S.session_text(r)
    # a reading a pass lacks is not printed, and passes with none have no such line
    del ours["loads"]["agentic@4"][1]["users_s"]
    r = S.session_cost([ref, ours], "prod-record", "prod")
    assert r["readings"] == {"busy_gpu_s": 2.0, "users_s": None, "users_rates": 0.0}
    assert "in no verdict: 2.00% by the GPUs' busy seconds, 0.00% by the users' own rates summed\n" in S.session_text(r)
    bare = S.session_cost(arms(), "prod-record", "prod")
    assert set(bare["readings"].values()) == {None} and "other readings" not in S.session_text(bare)


def test_an_arms_sessions_pass_is_read_from_its_users_own_results(tmp_path: Path):
    d = tmp_path / "prod"
    rep = d / "agentic@2" / "rep-1"
    users = [result(3, 60.0, 100.0, 3.0), result(2, 80.0, 50.0, 1.0)]
    for j, u in enumerate(users):
        (rep / f"user-{j}").mkdir(parents=True)
        (rep / f"user-{j}" / S.RESULT).write_text(json.dumps(u))
    # the pass's own file as a job that summed its users' rates left it
    (rep / S.RESULT).write_text(json.dumps(S.merged(users) | {"output_throughput": 150.0}))
    (rep / "meta.json").write_text(json.dumps({"rate": None, "concurrency": 2, "unfinished": 0}))
    (d / "meta.json").write_text(json.dumps({"arm": "prod", "model_id": "glm53", "rates": {},
                                             "passes": {"agentic@2": {"load": "agentic", "concurrency": 2}}}))
    (one,) = S.arm_metrics(d)["loads"]["agentic@2"]
    assert (one["output_tok_per_s"], one["users_output_tok_per_s"], one["users_s"]) == (125.0, 150.0, 140.0)


def kept_ids(d: Path, lines: list[tuple[str, list[int]]], gz: bool = False) -> None:
    """An arm's server's kept ids (`servetax_plugin.ServedIds`'s lines): each request's id and the ids it was served."""
    (d / "server").mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps({"id": rid, "prompt_ids": [1, 2, 3], "output_ids": out}) + "\n" for rid, out in lines)
    if gz:
        with gzip.open(d / "server" / "ids.7.jsonl.gz", "wt", encoding="utf-8") as f:
            f.write(text)
    else:
        (d / "server" / "ids.7.jsonl").write_text(text)


def test_the_requests_that_went_on_past_an_end_of_text_token_are_counted_where_the_server_kept_its_ids(tmp_path: Path):
    # a checkpoint's end-of-text ids: its generation configuration's, one or a list, else its model configuration's
    ck = tmp_path / "ck"
    ck.mkdir()
    assert S.end_of_text_ids(ck) == []
    (ck / "config.json").write_text(json.dumps({"eos_token_id": 9}))
    assert S.end_of_text_ids(ck) == [9]
    (ck / "generation_config.json").write_text(json.dumps({"eos_token_id": [9, 7, 8]}))
    assert S.end_of_text_ids(ck) == [7, 8, 9]
    (ck / "generation_config.json").write_text(json.dumps({"temperature": 1.0}))
    assert S.end_of_text_ids(ck) == [9]
    # a pass's requests by its tag: one that served an end-of-text id and went on for three tokens, one whose last
    # token is one, which did not go on, and one that served none; another pass's and the warm-up's are not its own
    d = tmp_path / "prod-record"
    lines = [("cmpl-agentic@2-r1-u0-0-0-aa", [5, 9, 5, 7, 5]), ("cmpl-agentic@2-r1-u1-0-0-bb", [5, 5, 8]),
             ("cmpl-agentic@2-r1-u1-1-0-cc", [5, 5, 5, 5]), ("cmpl-agentic@2-r10-u0-0-0-dd", [7, 5]),
             ("cmpl-warm-u0-0-0-ee", [9, 5])]
    kept_ids(d, lines)
    want = {"agentic@2-r1": {"ids_kept": 3, "past_end": 1, "after_end_tokens": 3},
            "agentic@2-r10": {"ids_kept": 1, "past_end": 1, "after_end_tokens": 1}}
    assert S.past_end(d, ["agentic@2-r1", "agentic@2-r10"], [7, 8, 9]) == want
    assert S.past_end(d, ["agentic@2-r1"], [7, 8, 9]) == {"agentic@2-r1": want["agentic@2-r1"]}
    # no end-of-text id named, or no ids kept: nothing; a gzipped file reads as a plain one
    assert S.past_end(d, ["agentic@2-r1"], []) == {} == S.past_end(tmp_path / "prod", ["agentic@2-r1"], [9])
    (d / "server" / "ids.7.jsonl").unlink()
    kept_ids(d, lines, gz=True)
    assert S.past_end(d, ["agentic@2-r1", "agentic@2-r10"], [7, 8, 9]) == want
    # an arm's metrics have them for each rep whose ids its server kept, where its meta names the ids
    rep = d / "agentic@2" / "rep-1"
    users = [result(1, 60.0, 100.0, 1.0), result(2, 80.0, 50.0, 1.0)]
    for j, u in enumerate(users):
        (rep / f"user-{j}").mkdir(parents=True)
        (rep / f"user-{j}" / S.RESULT).write_text(json.dumps(u))
    (rep / S.RESULT).write_text(json.dumps(S.merged(users)))
    (rep / "meta.json").write_text(json.dumps({"rate": None, "concurrency": 2, "unfinished": 0, "tag": "agentic@2-r1"}))
    meta = {"arm": "prod-record", "model_id": "glm53", "rates": {}, "passes": {"agentic@2": {"load": "agentic", "concurrency": 2}}}
    (d / "meta.json").write_text(json.dumps(meta | {"end_of_text_ids": [7, 8, 9]}))
    (one,) = S.arm_metrics(d)["loads"]["agentic@2"]
    assert (one["ids_kept"], one["past_end"], one["after_end_tokens"]) == (3, 1, 3)
    (d / "meta.json").write_text(json.dumps(meta))            # a directory served before its meta named them
    assert "past_end" not in S.arm_metrics(d)["loads"]["agentic@2"][0]


def test_a_sessions_cost_says_how_many_requests_went_on_past_an_end_of_text_token_for_the_side_that_kept_its_ids():
    ref, ours = served("prod", [2.0] * 3), served("prod-record", [2.0 / 1.01] * 3)
    for rep, past in zip(ours["loads"]["agentic@4"], (20, 24, 28)):
        rep |= {"ids_kept": 96, "past_end": past, "after_end_tokens": 12000}
    r = S.session_cost([ref, ours], "prod-record", "prod")
    assert r["sides"]["arm"]["past_end"] == {"requests": 72, "of": 288, "tokens_after_share": 20.0}
    assert r["sides"]["reference"]["past_end"] is None
    assert ("; each greedy; each request decoded to its recorded length, past any end-of-text token\n"
            "  past an end-of-text token, in the passes counted: 72 of prod-record's 288 requests served an end-of-text "
            "token before their end and went on, 20.00% of its output tokens after one; prod's server keeps no ids, so "
            "the count is prod-record's alone\n") in S.session_text(r)
    # a side that kept the ids of some of its counted passes only has no count, and with none kept there is no line
    del ours["loads"]["agentic@4"][1]["ids_kept"]
    r = S.session_cost([ref, ours], "prod-record", "prod")
    assert r["sides"]["arm"]["past_end"] is None and "in the passes counted" not in S.session_text(r)
    assert S.past_end_text(r) == []
    # both sides' where both kept them
    for arm in (ref, ours):
        for rep in arm["loads"]["agentic@4"]:
            rep |= {"ids_kept": 96, "past_end": 24, "after_end_tokens": 6000}
    r = S.session_cost([ref, ours], "prod-record", "prod")
    assert S.past_end_text(r) == [
        "  past an end-of-text token, in the passes counted: 72 of prod's 288 requests served an end-of-text token "
        "before their end and went on, 10.00% of its output tokens after one; 72 of prod-record's 288 requests served "
        "an end-of-text token before their end and went on, 10.00% of its output tokens after one"]
    assert S.past_end_text(r)[0] + "\n" in S.session_text(r)


def test_a_cost_over_lifetimes_is_the_mean_of_theirs_with_students_t_over_them():
    # three pairs of lifetimes of two passes each: ours costs 0.4%, 0.5% and 0.6% in them, a pass 0.1 points either side
    pairs = [[(1.0, 1 / (1 + c + d)) for d in (-0.001, 0.001)] for c in (0.004, 0.005, 0.006)]
    r = S.over_lifetimes(pairs)
    assert r["value"] == pytest.approx(0.005) and r["each"] == pytest.approx([0.004, 0.005, 0.006])
    assert r["between"] == pytest.approx(0.001) and r["within"] == pytest.approx(0.001 * math.sqrt(2))
    half = 4.303 * 0.001 / math.sqrt(3)
    assert S.T95[2] == 4.303 and r["ci95"] == pytest.approx([0.005 - half, 0.005 + half])
    # one pair has a cost and no interval, and one pass a pair no spread within
    one = S.over_lifetimes(pairs[:1])
    assert one["value"] == pytest.approx(0.004) and "ci95" not in one and "between" not in one and "within" in one
    assert "within" not in S.over_lifetimes([p[:1] for p in pairs])
    # a pair's cost is the arm's seconds for its passes over the reference's, not the mean of the passes' ratios (0
    # and 0.5): two passes of 300 tokens each take 3 s and 1 s under stock's rates and 3 s and 1.5 s under ours's
    assert S.over_lifetimes([[(100.0, 100.0), (300.0, 200.0)]])["each"] == pytest.approx([4.5 / 4 - 1])
    # no pair, a pair with no pass, or a pass a side has no rate in: no value
    assert S.over_lifetimes([]) == S.over_lifetimes([[]]) == S.over_lifetimes([[(1.0, 0.0)]]) == {"value": None}
    # degrees of freedom the table lacks read the entry below them, a little wide, as the table's means do
    eight = S.over_lifetimes([[(1 + 0.001 * (k % 2), 1.0)] for k in range(8)])
    assert eight["ci95"][1] - eight["value"] == pytest.approx(S.T95[6] * eight["between"] / math.sqrt(8)) and 7 not in S.T95


def test_a_sessions_cost_has_no_verdict_where_the_sides_did_not_do_the_same_work():
    ref = served("prod", [2.0] * 6)
    evicted = served("prod-record", OURS)
    evicted["loads"]["agentic@4"][2]["prompt_cached_share"] = 0.9
    r = S.session_cost([ref, evicted], "prod-record", "prod")
    assert (r["value"], r["same_work"], r["verdict"]) == (0.5, False, "cannot tell")
    assert "did not do the same work" in S.session_text(r)
    short = S.session_cost([ref, served("prod-record", OURS, ok=95)], "prod-record", "prod")
    assert not short["sides"]["arm"]["served_all"] and short["verdict"] == "cannot tell"
    assert "3.50 tokens a draft, NOT every request served" in S.session_text(short)
    # nor where a rep of one side served another count of output tokens; the token ids are not compared
    fewer = served("prod-record", OURS)
    fewer["loads"]["agentic@4"][0]["output_tokens"] -= 1
    r = S.session_cost([ref, fewer], "prod-record", "prod")
    assert (r["same_work"], r["verdict"]) == (False, "cannot tell")
    assert ("  the two sides did not do the same work (a request unserved, another count of output tokens, or their "
            "prefix caches served different tokens in a rep): no verdict\n") in S.session_text(r)
    # an engine that drafts nothing, and a pass without the client's per-request record, say so and give no figure
    plain = [served(n, x) for n, x in (("prod", [2.0] * 6), ("prod-record", OURS))]
    for a in plain:
        for rep in a["loads"]["agentic@4"]:
            del rep["draft_accepted_share"], rep["accepted_per_draft"], rep["tpot_p90_s"]
    text = S.session_text(S.session_cost(plain, "prod-record", "prod"))
    assert "a GPU), 94.00% of its prompt tokens from its prefix cache, 2,000 engine steps a pass, no drafts, every request served" in text
    # a cost is over stock from the same environment
    with pytest.raises(SystemExit, match=r"different builds of vLLM \(prod: 0.1\+gd9105ea8, prod-record: 0.2\)"):
        S.session_cost([ref, served("prod-record", OURS) | {"vllm": "0.2"}], "prod-record", "prod")
    with pytest.raises(SystemExit, match="prod and prod-record served load agentic from different plans"):
        S.session_cost([ref, served("prod-record", OURS, plan=PLAN | {"sha256": "b" * 64})], "prod-record", "prod")
    # nor of a sampled side over a greedy one: the plan each served names its sampling, and the cost's text says which
    sampled = PLAN | {"sampled": S.SAMPLED}
    with pytest.raises(SystemExit, match="prod and prod-record served load agentic from different plans"):
        S.session_cost([ref, served("prod-record", OURS, plan=sampled)], "prod-record", "prod")
    both = [served("prod", [2.0] * 6, plan=sampled), served("prod-record", OURS, plan=sampled) | {"seeded": True}]
    cost = S.session_cost(both, "prod-record", "prod")
    assert cost["seeded"] == {"reference": False, "arm": True}
    # sampled or greedy, a request is decoded to its recorded length, and the cost's line on the load says so
    assert S.TO_LENGTH == "each request decoded to its recorded length, past any end-of-text token"
    past = f"; {S.TO_LENGTH}\n"
    assert ("planned to repeat; each sampled at temperature 1 and top-p 0.95, prod-record's under a seed its client "
            "sends, one a user, and prod's under none" + past) in S.session_text(cost)
    both[0]["seeded"] = True                              # arms served when every sampled arm was sent a seed
    assert ("planned to repeat; each sampled at temperature 1 and top-p 0.95 under a seed its client sends, one a user"
            + past in S.session_text(S.session_cost(both, "prod-record", "prod")))
    assert "planned to repeat; each greedy" + past in S.session_text(
        S.session_cost([ref, served("prod-record", OURS)], "prod-record", "prod"))
    with pytest.raises(SystemExit, match="prod served load agentic and prod-record did not"):
        S.session_cost([ref], "prod-record", "prod")
    with pytest.raises(SystemExit, match="no directory of arm prod "):
        S.session_cost([evicted], "prod-record", "prod")


def lifetime(name: str, k: int | None, rates: list[float], left_out: int = 1) -> dict:
    """One server lifetime of an arm (`served`): its number in the job, the reps it was to serve and how many of its
    first ones it leaves out."""
    return served(name, rates) | {"lifetime": k, "left_out": left_out, "reps": len(rates)}


#: stock's two lifetimes, pass by pass, each with a warm-up first that is far off the rest; ours costs 0.5% in every
#: pass of every lifetime, so any other pairing than the same pass of the same lifetime gives another figure
STOCK = {1: [1.0, 2.0, 2.1, 2.2], 2: [1.2, 2.002, 2.102, 2.202]}


def in_lifetimes() -> list[dict]:
    """A job's directories in the order it served them: stock, ours, ours, stock."""
    ours = {k: [0.75 * rates[0], *(x / 1.005 for x in rates[1:])] for k, rates in STOCK.items()}
    return [lifetime("prod", 1, STOCK[1]), lifetime("prod-record", 1, ours[1]), lifetime("prod-record", 2, ours[2]),
            lifetime("prod", 2, STOCK[2])]


def test_a_cost_over_server_lifetimes_pairs_the_same_pass_of_the_same_lifetime_and_leaves_the_warm_up_out():
    stock, ours = S.lifetimes(in_lifetimes())
    assert (stock["arm"], stock["lifetimes"], ours["arm"], ours["lifetimes"]) == ("prod", [1, 2], "prod-record", [1, 2])
    # lifetime k's pass r is rep (k - 1) * 4 + r; each lifetime's first pass is kept apart from the reps
    assert [(r["rep"], r["lifetime"], r["pass"]) for r in stock["loads"]["agentic@4"]] == [
        (2, 1, 2), (3, 1, 3), (4, 1, 4), (6, 2, 2), (7, 2, 3), (8, 2, 4)]
    assert [(r["rep"], r["output_tok_per_s"]) for r in stock["warm"]["agentic@4"]] == [(1, 1000.0), (5, 1200.0)]
    assert S.lifetimes([stock, ours]) == [stock, ours]                      # read once
    r = S.session_cost(in_lifetimes(), "prod-record", "prod")
    assert (r["value"], r["ci95"], r["reps"], r["lifetimes"]) == (0.5, [0.5, 0.5], 6, 2)
    assert r["by_lifetime"] == [{"lifetime": 1, "value": 0.5, "reps": 3}, {"lifetime": 2, "value": 0.5, "reps": 3}]
    assert r["left_out"] == {"passes": 1, "warm": [2, 2]} and r["spread"] == {"between": 0.0, "within": 0.0}
    # no twin of stock was served: the noise floor is stock's second lifetime over its first, pass by pass, one pair
    assert r["noise_floor"] == {"arm": "prod", "value": -0.1, "ci95": None, "reps": 3, "pairs": 1, "differs": None,
                                "lifetimes": [[1, 2]]}
    # the verdict is the interval's upper end against the target, and the floor is not added to it
    assert r["verdict"] == "meets" and S.session_cost(in_lifetimes(), "prod-record", "prod", target=0.55)["verdict"] == "meets"
    assert S.session_cost(in_lifetimes(), "prod-record", "prod", target=0.45)["verdict"] == "misses"
    text = S.session_text(r)
    assert text.splitlines()[0] == ("serving cost on load agentic: prod-record over prod: 0.50% [0.50%, 0.50%] of stock's "
                                    "serving compute over 2 server lifetimes an arm, 4 users; target 1%: meets")
    assert ("last answer; its interval is Student's t over the lifetimes, and the verdict is its upper end against the target\n"
            "  passes taken as independent: 0.50% [0.50%, 0.50%], 6 reps, the median over them with its interval; in no "
            "verdict\n"
            "  by lifetime: 0.50% (3 reps), 0.50% (3 reps); the first pass of each lifetime, its warm-up, served and "
            "recorded and left out (2 of prod's, 2 of prod-record's)\n"
            "  the spread of the cost (standard deviations): 0.00 points between lifetimes, 0.00 points between a "
            "lifetime's passes\n"
            "  noise floor: prod's own lifetimes, each second one over the one before it: -0.10% over 1 pair of "
            "lifetimes, 3 reps; beside the cost, not added to its interval\n") in text
    # a twin of stock, where one served, is still the floor, a pair of lifetimes its unit too; its interval leaves out
    # zero here, and the cost says in words that stock differs from stock
    twin = S.session_cost([*in_lifetimes(), *(lifetime("prod-again", k, [x / 1.002 for x in STOCK[k]]) for k in (1, 2))],
                          "prod-record", "prod")
    assert twin["noise_floor"] == {"arm": "prod-again", "value": 0.2, "ci95": [0.2, 0.2], "reps": 6, "pairs": 2,
                                   "differs": True}
    assert ("  noise floor: prod-again over prod: 0.20% [0.20%, 0.20%] over 2 pairs of lifetimes, 6 reps; beside the "
            "cost, not added to its interval\n"
            "  prod differs from prod: the noise floor's interval leaves out zero, so two lifetimes of the same engine "
            "did not serve at the same rate\n") in S.session_text(twin)
    assert twin["verdict"] == "meets" and "differs from" not in text


def spread(costs: tuple[float, ...], again: tuple[float, ...] = ()) -> list[dict]:
    """A job of a lifetime an arm for each of ``costs``, ours costing that much in every pass of it; stock's lifetime
    k + 1 costs ``again[k]`` over the first one's rates where it is given."""
    out = []
    for k, c in enumerate(costs, 1):
        stock = [x / (1 + dict(enumerate(again)).get(k - 1, 0.0)) for x in STOCK[1]]
        out += [lifetime("prod", k, stock), lifetime("prod-record", k, [0.75, *(x / (1 + c) for x in stock[1:])])]
    return out


def test_the_verdict_reads_the_interval_over_lifetimes_by_its_upper_end_alone():
    # three lifetimes an arm costing 0.4%, 0.5% and 0.6%: 0.5% with Student's t over three, 4.303 * 0.1 / sqrt(3)
    r = S.session_cost(spread((0.004, 0.005, 0.006)), "prod-record", "prod")
    assert (r["value"], r["ci95"], r["lifetimes"], r["reps"], r["verdict"]) == (0.5, [0.25, 0.75], 3, 9, "meets")
    assert [x["value"] for x in r["by_lifetime"]] == [0.4, 0.5, 0.6] and r["spread"] == {"between": 0.1, "within": 0.0}
    assert (r["passes"]["value"], r["passes"]["reps"]) == (0.5, 9)
    told = {t: S.session_cost(spread((0.004, 0.005, 0.006)), "prod-record", "prod", target=t)["verdict"] for t in (0.8, 0.7, 0.2)}
    assert told == {0.8: "meets", 0.7: "cannot tell", 0.2: "misses"}
    # stock's lifetimes were the same, so its floor is zero over one pair (the third has no second): no interval
    assert r["noise_floor"] == {"arm": "prod", "value": 0.0, "ci95": None, "reps": 3, "pairs": 1, "differs": None,
                                "lifetimes": [[1, 2]]}
    # two lifetimes an arm that spread as much have an interval Student's t makes wide: 12.706 * 0.1414 / sqrt(2)
    two = S.session_cost(spread((0.004, 0.006)), "prod-record", "prod")
    assert (two["value"], two["ci95"], two["verdict"]) == (0.5, [-0.77, 1.77], "cannot tell")
    # stock's second lifetimes 1% and 1.02% slower than the ones before them: a floor whose interval leaves out zero,
    # said in words, and in no verdict
    four = S.session_cost(spread((0.005,) * 4, (0.0, 0.01, 0.0, 0.0102)), "prod-record", "prod")
    assert four["noise_floor"] == {"arm": "prod", "value": 1.01, "ci95": [0.88, 1.14], "reps": 6, "pairs": 2,
                                   "differs": True, "lifetimes": [[1, 2], [3, 4]]}
    assert (four["value"], four["verdict"]) == (0.5, "meets")
    assert ("  noise floor: prod's own lifetimes, each second one over the one before it: 1.01% [0.88%, 1.14%] over 2 "
            "pairs of lifetimes, 6 reps; beside the cost, not added to its interval\n"
            "  prod differs from prod: the noise floor's interval leaves out zero, so two lifetimes of the same engine "
            "did not serve at the same rate\n") in S.session_text(four)
    # the table has the lifetimes' counted passes as the arm's reps, and says what it left out
    t = S.table(S.lifetimes(in_lifetimes()))
    assert [(x["arm"], x["reps"], x["lifetimes"], x["left_out"]) for x in t["rows"]] == [("prod", 6, 2, 2), ("prod-record", 6, 2, 2)]
    assert ("which are served and recorded and in no figure here: prod agentic@4: 2 server lifetimes, 2 left out; "
            "prod-record agentic@4: 2 server lifetimes, 2 left out.") in S.markdown(t)
    assert "left out" not in S.markdown(S.table(arms()))


def test_one_lifetime_can_leave_its_warm_up_out_and_lifetimes_that_are_not_one_arms_are_refused():
    # a job of one lifetime an arm: the first pass left out, five reps counted, which have no interval
    one = S.session_cost([lifetime("prod", None, [1.0] + [2.0] * 5), lifetime("prod-record", None, [1.5] + OURS[1:])],
                         "prod-record", "prod")
    assert (one["reps"], one["ci95"], one["verdict"]) == (5, None, "cannot tell")
    assert one["left_out"] == {"passes": 1, "warm": [1, 1]} and one["noise_floor"] is None
    assert "over 1 server lifetime an arm, 4 users" in S.session_text(one)
    assert "  by lifetime: 0.50% (5 reps); the first pass of each lifetime, its warm-up" in S.session_text(one)
    two = S.session_cost([lifetime("prod", 1, [2.0] * 3, 2), lifetime("prod-record", 1, OURS[:3], 2)], "prod-record", "prod")
    assert two["reps"] == 1 and "the first 2 passes of each lifetime" in S.session_text(two)
    a, b = in_lifetimes()[:2]
    with pytest.raises(SystemExit, match="two directories of arm prod are both its lifetime 1: give one of them"):
        S.session_cost([a, a, b], "prod-record", "prod")
    with pytest.raises(SystemExit, match="two directories of arm prod are lifetimes that leave out 1 and 2 passes"):
        S.session_cost([a, lifetime("prod", 2, STOCK[2], 2), b], "prod-record", "prod")
    with pytest.raises(SystemExit, match="two directories of arm prod are lifetimes of which one names no number"):
        S.session_cost([a, lifetime("prod", None, STOCK[2]), b], "prod-record", "prod")
    with pytest.raises(SystemExit, match="two lifetimes of arm prod differ in vllm: they are not one arm's"):
        S.session_cost([a, lifetime("prod", 2, STOCK[2]) | {"vllm": "0.2"}, b], "prod-record", "prod")
    # two directories of an arm that are no lifetimes of it are refused as before: nothing pairs their reps
    with pytest.raises(SystemExit, match="two directories of arm prod both served agentic: give one of them"):
        S.session_cost([served("prod", [2.0] * 6), served("prod", [2.0] * 6), served("prod-record", OURS)], "prod-record", "prod")
    # a lifetime the job cut short pairs the passes both arms have
    cut = in_lifetimes()
    cut[3]["loads"]["agentic@4"] = cut[3]["loads"]["agentic@4"][:3]
    short = S.session_cost(cut, "prod-record", "prod")
    assert short["reps"] == 5 and [x["reps"] for x in short["by_lifetime"]] == [3, 2]


def test_cost_prints_a_sessions_loads_cost_over_the_arms_own_stock(tmp_path: Path, capsys: pytest.CaptureFixture):
    files = []
    for a in arms():
        files.append(tmp_path / f"{a['arm']}.json")
        files[-1].write_text(json.dumps(a))
    assert S.main(["cost", *map(str, files), "--arm", "prod-record", "--out", str(tmp_path / "out")]) == 0
    text = capsys.readouterr().out
    assert text.startswith("serving cost on load agentic: prod-record over prod: 0.50% of stock's serving compute over 1 ")
    assert (tmp_path / "out" / "serving_cost.txt").read_text() == text
    r = json.loads((tmp_path / "out" / "serving_cost.json").read_text())
    assert list(r)[:4] == ["value", "unit", "commit", "run"] and r["reference"] == "prod"
    ok, first, what, measurements = S.result_of(r)
    assert ok and first == text.splitlines()[0] and what["loads"] == ["agentic"] and what["arm"] == "prod-record"
    assert [(m["name"], m["value"], m["ci95"], m["target"], m["verdict"], m["users"], m["lifetimes"]) for m in measurements] == [
        ("agentic", 0.5, None, 1.0, "cannot tell", 4, 1)]
    # a job that served two sessions loads has a cost of each: the first's is the report, the second's is beside it
    two = []
    for a, slower in zip(arms(), (1.0, 1.01, 1.0)):
        other = served(a["arm"], [x["output_tok_per_s"] / 1000 / slower for x in a["loads"]["agentic@4"]], load=S.AGENTIC_256K)
        two.append(a | {"loads": a["loads"] | other["loads"], "planned": a["planned"] | other["planned"]})
        files.append(tmp_path / f"{a['arm']}.two.json")
        files[-1].write_text(json.dumps(two[-1]))
    assert S.session_loads(two, "prod") == ["agentic", "agentic-256k"] and S.session_loads(two, "no such arm") == []
    assert S.session_cost(two, "prod-record", "prod", load=S.AGENTIC_256K)["value"] == 1.51
    assert S.session_cost(two, "prod-record", "prod")["load"] == "agentic" and S.session_cost(two, "prod-record", "prod", load="chat") is None
    assert S.main(["cost", *map(str, files[3:]), "--arm", "prod-record", "--out", str(tmp_path / "two")]) == 0
    first, second = capsys.readouterr().out.split("serving cost on load ")[1:]
    assert first.startswith("agentic: prod-record over prod: 0.50% of stock's") and "independent: 0.50% [0.40%, 0.60%]" in first
    assert second.startswith("agentic-256k: prod-record over prod: 1.51% of stock's") and "independent: 1.51% [1.41%, 1.61%]" in second
    assert (tmp_path / "two" / "serving_cost.txt").read_text() == "serving cost on load " + first
    assert (tmp_path / "two" / "serving_cost.agentic-256k.txt").read_text() == "serving cost on load " + second
    assert json.loads((tmp_path / "two" / "serving_cost.agentic-256k.json").read_text())["load"] == "agentic-256k"
    # arms that served no sessions load are costed as before, over compiled
    assert S.session_cost([served("compiled", [2.0] * 6, load="chat"), served("record", OURS, load="chat")], "record",
                          "compiled") is None
