"""The arms at the recipe's settings and the sessions loads: their data, the requests a corpus of sessions gives each
user, a pass sent by one client a user, and the cost on such a load, on synthetic records."""
from __future__ import annotations

import hashlib
import json
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


def test_a_pass_of_a_sessions_load_is_one_client_a_user_and_its_rates_are_the_users_sum(tmp_path: Path, monkeypatch):
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
    # a user's rate is over its own time, and the pass's is their sum
    assert (r["output_throughput"], r["total_token_throughput"]) == (150.0, 1150.0)
    assert r["request_throughput"] == pytest.approx(3 / 60 + 2 / 80) and r["request_goodput"] == pytest.approx(3 / 60 + 1 / 80)
    # the times are over every user's requests: first tokens after 0.4 s (three) and 0.8 s; 20, 20, 20 and 40 ms a token
    assert (r["median_ttft_ms"], r["p90_ttft_ms"]) == (pytest.approx(400.0), pytest.approx(800.0))
    assert (r["median_tpot_ms"], r["p90_tpot_ms"], r["p99_tpot_ms"]) == tuple(map(pytest.approx, (20.0, 34.0, 39.4)))
    b = S.bench_metrics(r)
    assert (b["sent"], b["ok"], b["met"], b["output_tok_per_s"], b["prompt_tok_per_s"]) == (6, 5, 4, 150.0, 1000.0)
    assert b["ttft_p50_s"] == pytest.approx(0.4) and b["tpot_p90_s"] == pytest.approx(0.034)
    assert b["itl_p99_s"] is None and b["slo_attained"] == pytest.approx(4 / 6)
    # a result that is not the detailed one gives no time, and one with no goodput no count within the SLO
    bare = S.merged([result(3, 60.0, 100.0, None), users["u1"]])
    assert bare["met"] is None and "p90_tpot_ms" not in bare and "tpot_p90_s" not in S.bench_metrics(bare)
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
    S.drive_bench(8000, GLM, Path("/ck"), load, 2, "agentic@2-r1", tmp_path / "rep-1-stock", dataset=whole, prompts=3,
                  sampled=True, run=run, clock=lambda: 0.0, now=lambda: 0.0)
    assert sorted(bodies, key=lambda b: b["user"]) == [{"user": u} for u in ("agentic@2-r1-u0", "agentic@2-r1-u1")]


#: what a job's arms planned for load agentic: four users of 24 turns each
PLAN = {"requests": 96, "prompt_tokens": 8000000, "repeated_tokens": 7600000, "span_s": 0.0, "sha256": "a" * 64, "users": 4}


def served(name: str, rates: list[float], load: str = S.AGENTIC, cached: float = 0.94, ok: int = 96, plan: dict = PLAN) -> dict:
    """An arm's metrics with a sessions load served by four users, a rep each rate, in thousands of output tokens a
    second, each rep with the share of its prompt tokens the prefix cache served, 52,000 prompt and output tokens a
    second, 25 ms a token at the P90, and half its draft tokens accepted."""
    reps = [{"rep": i + 1, "rate": None, "concurrency": 4, "sent": 96, "ok": ok, "met": ok, "goodput_req_per_s": 0.5,
             "req_per_s": 0.5, "output_tok_per_s": 1000 * r, "prompt_tok_per_s": 50000.0, "output_tokens": 60000,
             "prompt_tokens": 8000000, "prompt_cached_share": cached, "total_tok_per_s": 52000.0, "tpot_p90_s": 0.025,
             "draft_accepted_share": 0.5, "accepted_per_draft": 2.5} for i, r in enumerate(rates)]
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
    stock = r["sides"]["reference"]
    assert stock["served_all"] and stock["output_tok_per_s"] == pytest.approx(2000.0)
    # beside the rate: tokens a second a GPU of the model's eight, a user's tokens a second at the P90 time a token,
    # and what the engine's drafts gave
    assert (stock["tok_per_s_a_gpu"], stock["user_tok_per_s_p90"]) == (6500.0, pytest.approx(40.0)) and r["gpus"] == 8
    assert (stock["draft_accepted_share"], stock["tokens_per_draft"]) == (50.0, 3.5)
    text = S.session_text(r)
    assert text.splitlines()[0] == ("serving cost on load agentic: prod-record over prod: 0.50% [0.40%, 0.60%] of stock's "
                                    "serving compute, 6 reps, 4 users; target 1%: meets")
    assert "noise floor: prod-again over prod: 0.10% [0.10%, 0.10%], 6 reps" in text and "95.00% of them planned to repeat" in text
    assert ("  prod: 2,000.0 output tokens a second (6,500.0 prompt and output tokens a second a GPU), 40.0 output tokens "
            "a second a user at the P90 time a token, 94.00% of its prompt tokens from its prefix cache, 50.00% of its "
            "draft tokens accepted, 3.50 tokens a draft, every request served\n") in text
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
    assert "3.50 tokens a draft, NOT every request served" in S.session_text(short)
    # an engine that drafts nothing, and a pass without the client's per-request record, say so and give no figure
    plain = [served(n, x) for n, x in (("prod", [2.0] * 6), ("prod-record", OURS))]
    for a in plain:
        for rep in a["loads"]["agentic@4"]:
            del rep["draft_accepted_share"], rep["accepted_per_draft"], rep["tpot_p90_s"]
    text = S.session_text(S.session_cost(plain, "prod-record", "prod"))
    assert "a GPU), 94.00% of its prompt tokens from its prefix cache, no drafts, every request served" in text
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
    assert ("planned to repeat; each sampled at temperature 1 and top-p 0.95, prod-record's under a seed its client "
            "sends, one a user, and prod's under none\n") in S.session_text(cost)
    both[0]["seeded"] = True                              # arms served when every sampled arm was sent a seed
    assert ("planned to repeat; each sampled at temperature 1 and top-p 0.95 under a seed its client sends, one a user\n"
            in S.session_text(S.session_cost(both, "prod-record", "prod")))
    assert "planned to repeat; each greedy\n" in S.session_text(S.session_cost([ref, served("prod-record", OURS)], "prod-record", "prod"))
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
    assert (r["value"], r["ci95"], r["reps"]) == (0.5, [0.5, 0.5], 6)
    assert r["by_lifetime"] == [{"lifetime": 1, "value": 0.5, "ci95": None, "reps": 3},
                                {"lifetime": 2, "value": 0.5, "ci95": None, "reps": 3}]
    assert r["left_out"] == {"passes": 1, "warm": [2, 2]}
    # no twin of stock was served: the noise floor is stock's second lifetime over its first, pass by pass
    assert r["noise_floor"] == {"arm": "prod", "value": -0.1, "ci95": None, "reps": 3, "lifetimes": [[1, 2]]}
    assert r["verdict"] == "meets" and S.session_cost(in_lifetimes(), "prod-record", "prod", target=0.55)["verdict"] == "cannot tell"
    text = S.session_text(r)
    assert ("  the reps are the passes both arms counted, in 2 server lifetimes an arm; the first pass of each lifetime, "
            "its warm-up, served and recorded and left out (2 of prod's, 2 of prod-record's); by lifetime: 0.50% (3 reps), "
            "0.50% (3 reps)\n") in text
    assert ("  noise floor: prod's own lifetimes, each second one over the one before it: -0.10%, 3 reps; the verdict's "
            "interval is widened by its size\n") in text
    # a twin of stock, where one served, is still the floor
    twin = [*in_lifetimes(), *(lifetime("prod-again", k, [x / 1.002 for x in STOCK[k]]) for k in (1, 2))]
    assert S.session_cost(twin, "prod-record", "prod")["noise_floor"] == {"arm": "prod-again", "value": 0.2,
                                                                          "ci95": [0.2, 0.2], "reps": 6}
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
    assert "in 1 server lifetime an arm; the first pass of each lifetime" in S.session_text(one)
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
