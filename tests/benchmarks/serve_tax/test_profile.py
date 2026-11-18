"""A serving profile on serve_tax's own job (`run.sh --profile`, `serve_tax.profile_run`): its server and client what
`benchmarks/production.toml` says, the job as it was without one, what it refuses, `cannot run` with its second
line, `--find-load`'s line, and the result's three lines in order. Stand-ins for the server and the client: no GPU."""
from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace as NS
from dataclasses import replace
from pathlib import Path

import pytest

import serve_tax as S

S.profiles()
from verity_bench import profiles as P  # noqa: E402 (benchmarks/harness, put on sys.path by S.profiles)

#: what a request without a seed gets under the certified tick (`verity_vllm.pipeline.tick_plugin.unseeded`)
UNSEEDED = ("under the certified tick a request sampled at the server's default temperature needs an integer seed, "
            "not null: send one, or temperature 0")
SOURCE_COMMIT = S.source_commit
FEATURE = S.feature


@pytest.fixture(autouse=True)
def an_h200(monkeypatch: pytest.MonkeyPatch):
    """The host's card as nvidia-smi names it (`card`), read through the card files; the tree's commit given; and,
    since no arm serves a registered scheme yet (`feature`), each arm of ours over stock standing in for the arm its
    scheme resolves to, named in its place."""
    monkeypatch.setattr(S, "card", lambda: "NVIDIA H200")
    monkeypatch.setattr(S, "source_commit", lambda: "c0ffee")
    monkeypatch.setattr(S, "feature", lambda name, stock: name if name in S.profile_features(stock) else FEATURE(name, stock))


def test_feature_resolves_a_registered_scheme_to_the_arm_that_serves_it_through_the_one_table(
        monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(SystemExit, match="--feature record: --protocol 'record' is no registered compute scheme; "
                                         "registered: dither, pearl-plus"):
        FEATURE("record", "compiled")
    with pytest.raises(NotImplementedError, match="^benchmarks/serving-profiles-first-number: the schemes' table$"):
        FEATURE("pearl-plus", "prod")
    monkeypatch.setitem(S.ARMS, "prod-record", replace(S.ARMS["prod-record"], scheme="pearl-plus"))
    assert FEATURE("pearl-plus", "prod") == "prod-record"
    with pytest.raises(NotImplementedError):
        FEATURE("pearl-plus", "compiled")


def test_without_a_profile_the_server_and_client_lines_are_mains():
    """main's lines (8f18a3246ae), sys.executable written PY: no flag of the profile's reaches a job without one."""
    ck, r = Path("/ck"), Path("/r")
    got = {"server_1b": S.server_argv(S.ARMS["compiled"], S.MODELS["llama32-1b"], ck, 8000),
           "server_glm_prod": S.server_argv(S.ARMS["prod"], S.MODELS["glm53"], ck, 8000, max_model_len=262144),
           "chat": S.bench_argv(S.MODELS["llama32-1b"], ck, 8000, S.LOADS["chat"], 8, r, "chat@8-r1", "/data"),
           "long_sampled": S.bench_argv(S.MODELS["glm53"], ck, 8000, S.LOADS["long"], 4, r, "long@4-r2", sampled=True,
                                        seeded=True, lifetime=2)}
    got = {k: [x.replace(sys.executable, "PY") for x in v] for k, v in got.items()}
    serve = ["PY", "-m", "vllm.entrypoints.openai.api_server", "--model", "/ck", "--served-model-name"]
    client = ["PY", "-m", "vllm.entrypoints.cli.main", "bench", "serve", "--backend", "openai", "--base-url",
              "http://127.0.0.1:8000", "--endpoint", "/v1/completions", "--model"]
    tail = ["--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "99", "--request-id-prefix"]
    saved = ["--ready-check-timeout-sec", "0", "--disable-tqdm", "--save-result", "--save-detailed", "--result-dir", "/r",
             "--result-filename", "result.json"]
    assert got == {
        "server_1b": [*serve, "llama-3.2-1b", "--host", "127.0.0.1", "--port", "8000", "--dtype", "bfloat16",
                      "--max-model-len", "4096", "--max-num-seqs", "256", "--seed", "0", "--disable-uvicorn-access-log"],
        "server_glm_prod": [*serve, "glm-5.3", "--host", "127.0.0.1", "--port", "8000", "--tensor-parallel-size", "8",
                            "--kv-cache-dtype", "fp8", "--speculative-config.method", "mtp",
                            "--speculative-config.num_speculative_tokens", "5", "--tool-call-parser", "glm47",
                            "--reasoning-parser", "glm47", "--enable-auto-tool-choice", "--tokenizer-mode", "auto",
                            "--max-model-len", "262144", "--seed", "0", "--disable-uvicorn-access-log"],
        "chat": [*client, "llama-3.2-1b", "--tokenizer", "/ck", "--dataset-name", "sharegpt", "--dataset-path", "/data",
                 "--num-prompts", "200", "--seed", "0", "--request-rate", "inf", "--max-concurrency", "8", "--ignore-eos",
                 "--temperature", "0", "--goodput", "ttft:2000", "tpot:80", *tail, "chat@8-r1-", *saved],
        "long_sampled": [*client, "glm-5.3", "--tokenizer", "/ck", "--dataset-name", "random", "--random-input-len",
                         "8192", "--random-output-len", "128", "--num-prompts", "24", "--seed", "0", "--request-rate",
                         "inf", "--max-concurrency", "4", "--ignore-eos", "--temperature", "1.0", "--top-p", "0.95",
                         "--extra-body", '{"user": "long@4-r2", "seed": 4015412303577097265}', "--goodput", "tpot:80",
                         *tail, "long@4-r2-", *saved]}


def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *flags: str, datasets: dict | None = None) -> tuple[list, list]:
    """``serve_tax.py run FLAGS`` on stand-ins: the server's command and each pass's client command, as `serve` and
    `drive_bench` build them."""
    servers, clients = [], []
    for name, stub in {"signal": NS(signal=lambda *a: None, SIGTERM=15), "open_files": lambda: 4096,
                       "source_commit": lambda: "c0ffee", "check_plugins": lambda env, arm: None,
                       "wait_ready": lambda server, port, startup_s: None, "nvml": lambda: (None, "no NVML here"),
                       "scrape": lambda port: dict.fromkeys(S.SCHEDULE, 0.0), "versions": lambda: {},
                       "stop": lambda server, grace_s: "exited", "end_of_text_ids": lambda ck: [1],
                       "reset_prefix_cache": lambda port: True, "corpus": lambda: NS(free_port=lambda: 8000)}.items():
        monkeypatch.setattr(S, name, stub)
    if datasets is not None:
        monkeypatch.setattr(S, "parse_datasets", lambda given, loads, fetch_missing=False: datasets)
    monkeypatch.setattr(S.subprocess, "Popen", lambda argv, **how: servers.append(argv) or NS(returncode=0, poll=lambda: 0))

    def client(argv, stdout, stderr, env):
        clients.append(argv)
        d = Path(argv[argv.index("--result-dir") + 1])
        (d / S.RESULT).write_text(json.dumps({"completed": 1, "failed": 0}))
        return subprocess.CompletedProcess(argv, 0)

    real = S.drive_bench
    monkeypatch.setattr(S, "drive_bench", lambda *args: real(*args, run=client))
    assert S.main(["run", "--checkpoint", "/ckpt", "--client", "vllm-bench", "--reps", "1", "--warm-s", "0",
                   "--out", str(tmp_path / "out"), *flags]) == 0
    return servers, clients


def the_files_client(client: list, profile: str, model: str, in_flight: int, prompts: int) -> None:
    """``client`` is ``profile``'s on ``model`` as the file says it: its client, dataset and lengths, its prompts and
    requests in flight, the goodput of its bounds, and no sampling flag, so the server samples by the model's defaults."""
    doc = P.load()
    prof = doc.profiles[profile]
    flag = {k: v for k, v in zip(client, client[1:], strict=False) if k.startswith("--")}
    assert not {"--temperature", "--top-p", "--top-k"} & set(client)
    assert (flag["--backend"], flag["--endpoint"], flag["--model"]) == (doc.client.backend, doc.client.endpoint,
                                                                      doc.models[model].served)
    assert flag["--dataset-name"] == prof.dataset_name
    if prof.dataset_name == "random":
        assert (flag["--random-input-len"], flag["--random-output-len"]) == (str(prof.input), str(prof.output))
    if prof.dataset_name == "timed_trace":
        assert flag["--timed-trace-chunk-hash-size"] == str(prof.block)
    assert (flag["--num-prompts"], flag["--max-concurrency"]) == (str(prompts), str(in_flight))
    i = client.index("--goodput") + 1
    assert client[i:i + len(prof.within)] == [f"{P.GOODPUT[b][0]}:{round(getattr(doc.bounds, P.GOODPUT[b][1]) * 1e3)}"
                                              for b in prof.within]
    assert "--ignore-eos" in client


@pytest.mark.parametrize(("profile", "model", "load", "rung"), [
    ("fixed-size", "llama32-1b", "long", 8), ("fixed-size", "glm53", "long", 4),
    ("chat", "llama32-1b", "chat", 16), ("chat", "glm53", "chat", 12)])
def test_a_profiles_server_and_client_are_the_files(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str, model: str, load: str, rung: int):
    stock = S.profile_stock(profile, model)
    data = {"chat": "/data"} if load == "chat" else {}
    servers, clients = served(tmp_path, monkeypatch, "--arm", stock, "--model-id", model, "--loads", f"{load}={rung}",
                              "--profile", profile, datasets=data)
    doc = P.load()
    m = doc.models[model]
    (server,), (client,) = servers, clients
    assert server[server.index("--model") + 2:] == [
        "--served-model-name", m.served, "--host", "127.0.0.1", "--port", "8000", *m.serve,
        *(["--max-model-len", str(m.max_model_len)] if m.max_model_len else []), *doc.server.fixed]
    the_files_client(client, profile, model, rung, doc.profiles[profile].prompts)
    meta = json.loads((tmp_path / "out" / "meta.json").read_text())
    assert meta["profile"] == {"name": profile, "version": doc.profiles[profile].version,
                               "digest": P.digest(doc, profile), "ours": [], "seeded": False}


def test_a_stock_launch_off_the_files_is_refused(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(S.ARMS, "compiled", replace(S.ARMS["compiled"], args=(*S.ARMS["compiled"].args, "--x", "1")))
    with pytest.raises(SystemExit, match="compiled's server line .* is not production.toml's stock launch"):
        S.profile_stock("chat", "llama32-1b")


@pytest.mark.parametrize(("model", "load"), [("glm53", "agentic-256k"), ("llama32-1b", "agentic")])
def test_a_coding_agent_users_client_is_the_files(model: str, load: str):
    """A sessions pass's client is a user's (`drive_bench`), its file the user's and its prompts the cut's turns."""
    assert S.profile_load("coding-agent", model) == load
    turns = S.pass_prompts(S.LOADS[load], 1, None)
    ours = S.bench_argv(S.MODELS[model], Path("/ckpt"), 8000, S.as_profile(S.LOADS[load], "coding-agent"), 1,
                        Path("/r"), f"{load}@4-r1-u0", "/u0.jsonl", turns, defaults=True)
    assert turns == P.cut(P.load(), "coding-agent", model).turns
    the_files_client(ours, "coding-agent", model, 1, turns)


def run_args(**kw) -> NS:
    base = {"profile": "fixed-size", "model_id": "llama32-1b", "arm": "compiled", "client": S.BENCH, "loads": "long=8",
            "sampled": False, "max_model_len": None, "pass_prompts": None, "published_args": None,
            "published_frontend": None, "ours": None, "boundary_rows": None}
    return NS(**(base | kw))


@pytest.mark.parametrize(("kw", "refusal"), [
    ({"sampled": True}, "--profile fixed-size fixes the sampling: none is sent, so the server samples by the model's "
                        "generation_config (production.toml): drop --sampled"),
    ({"max_model_len": 4096}, "fixes the context limit: the file's (production.toml): drop --max-model-len"),
    ({"pass_prompts": "long=8"}, "drop --pass-prompts"),
    ({"published_args": "--seed=1"}, "fixes the stock launch (production.toml): drop --published-args"),
    ({"client": S.CORPUS}, "is sent by vLLM's client: --client vllm-bench"),
    ({"loads": "chat=8"}, "--profile fixed-size on llama32-1b serves load long alone: --loads long=RUNG"),
    ({"ours": "seeded"}, "--ours on compiled: stock is never changed"),
    ({"arm": "record", "ours": "greedy"}, "--ours greedy: the changes on our side are seeded, prefix-cache-off"),
    ({"arm": "prod-record"}, "serves compiled or one arm of ours beside it: no-prefix-cache, eager"),
    ({"arm": "compiled-again"}, "serves compiled or one arm of ours beside it"),
    ({"arm": "restore"}, "arm restore cannot run under --profile fixed-size: arm restore needs the corpus client, not "
                         "vLLM's: its sampling is the corpus client's (--client corpus)"),
])
def test_run_refuses_what_the_profile_fixes(kw: dict, refusal: str):
    with pytest.raises(SystemExit, match=refusal.replace("(", r"\(").replace(")", r"\)").replace("+", r"\+")):
        S.profile_run(run_args(**kw))


@pytest.fixture
def refusing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three arms over prod that refuse what a profile has on: one on the certified tick whose client sends no seed,
    one with the boundary commitment's history tables and prefix caching on, and one whose sampling is the corpus
    client's."""
    arms = dict(S.ARMS)
    arms["tick-unseeded"] = S.Arm("prod", "the tick, no seed sent", tick=True, plugin=("idle_reset",), published=True)
    arms["rows-tables"] = S.Arm("prod", "history tables, the cache on", plugin=("boundary_rows",),
                                rows={**S.ROWS, "tables": S.TABLES}, published=True)
    arms["prod-restore"] = replace(S.ARMS["restore"], parent="prod", published=True)
    monkeypatch.setattr(S, "ARMS", arms)


def test_a_refusal_has_the_refusing_codes_words_and_the_change_on_our_side_that_clears_it(refusing: None):
    assert S.refusals("tick-unseeded") == [{"piece": "sampling", "words": UNSEEDED, "change": "seeded"}]
    (rows,) = S.refusals("rows-tables")
    assert (rows["piece"], rows["change"]) == ("the cache", "prefix-cache-off")
    assert rows["words"].startswith("the boundary commitment's history tables need prefix caching off "
                                    "(--no-enable-prefix-caching)")
    assert S.refusals("prod-restore") == [{"piece": "vLLM's client", "change": None, "words":
                                           "arm prod-restore needs the corpus client, not vLLM's: its sampling is the "
                                           "corpus client's (--client corpus)"}]
    assert S.refusals("tick-certifier")[0]["words"] == S.SERVED_GREEDY.format(arm="tick-certifier")
    # the arms that send each user's seed run as they are; no arm refuses draft tokens
    assert S.refusals("prod-tick") == S.refusals("prod-all-on") == S.refusals("prod-record") == []
    plan = S.profile_plan("coding-agent", "glm53", "tick-unseeded", False)
    assert (plan["ARMS"], plan["OURS"], plan["LOADS"]) == ("prod tick-unseeded", "seeded", "agentic-256k=4")
    assert S.profile_plan("coding-agent", "glm53", "rows-tables", False)["OURS"] == "prefix-cache-off"
    assert S.profile_plan("coding-agent", "glm53", "prod-restore", False)["ARMS"] == ""
    run = S.profile_run(run_args(profile="coding-agent", model_id="glm53", arm="rows-tables", loads="agentic-256k=4",
                                 ours="prefix-cache-off"))
    assert (run["ours"], run["args"], run["seeded"]) == (["prefix cache off for ours"], ["--no-enable-prefix-caching"],
                                                         False)
    with pytest.raises(SystemExit, match=r"cannot run under --profile coding-agent: under the certified tick .* "
                                         r"\(--ours seeded: seeded for ours\)"):
        S.profile_run(run_args(profile="coding-agent", model_id="glm53", arm="tick-unseeded", loads="agentic-256k=4"))


def rep(r: int, tok: float, ttft: float = 0.2, tpot: float = 0.02, met: int = 90) -> dict:
    return {"rep": r, "rate": None, "concurrency": 4, "sent": 96, "ok": 96, "met": met, "output_tok_per_s": tok,
            "req_per_s": tok / 100, "prompt_tok_per_s": tok * 10, "output_tokens": 9600, "prompt_tokens": 96000,
            "goodput_req_per_s": tok / 100 * met / 96, "slo_attained": met / 96, "ttft_p50_s": ttft,
            "ttft_p99_s": ttft * 4, "tpot_p50_s": tpot, "tpot_p99_s": tpot * 1.5, "prompt_cached_share": 0.8}


def arm_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load: str, sides: dict[str, list[list[dict]]],
             stopped: str | None = None) -> list[Path]:
    """An arm's directory for each lifetime of each side, its metrics ``sides[arm][k - 1]`` (`arm_metrics` stood in)."""
    metrics, dirs = {}, []
    for arm, lives in sides.items():
        for k, reps in enumerate(lives, 1):
            d = tmp_path / f"{arm}.{k}"
            d.mkdir()
            stop = {"stopped": {"before": f"{load}@4", "rep": 3}} if stopped == d.name else {}
            (d / "meta.json").write_text(json.dumps({"arm": arm, **stop}))
            metrics[d.name] = {"arm": arm, "model_id": "glm53", "slo": {"ttft_s": 2.0, "tpot_s": 0.08},
                               "vllm": "0.1+gd9105ea8", "lifetime": k if len(lives) > 1 else None, "reps": 3,
                               "left_out": 0, "loads": {f"{load}@{reps[0]['concurrency']}": reps} if reps else {},
                               "commit": "c0ffee", "run_id": f"run-{arm}", "planned": {load: PLANNED}}
            dirs.append(d)
    monkeypatch.setattr(S, "arm_metrics", lambda d, slo=None: metrics[d.name])
    return dirs


STOCK = [[rep(1, 100), rep(2, 101), rep(3, 99)], [rep(1, 100, 0.21), rep(2, 100.5), rep(3, 99.5)]]
#: a sessions load's plan as `arm_metrics` reads it: four users, 24 turns each, the 96 requests `rep` sends at rung 4
PLANNED = {"requests": 96, "users": 4, "prompt_tokens": 96000, "repeated_tokens": 80000}


def test_a_feature_that_refuses_is_cannot_run_and_its_second_line_is_served_with_the_change_on_our_side(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusing: None):
    ours = [[rep(1, 90, 0.3, 0.025, 80), rep(2, 91), rep(3, 89)], [rep(1, 90), rep(2, 90.5), rep(3, 89.5)]]
    dirs = arm_dirs(tmp_path, monkeypatch, "agentic-256k", {"prod": STOCK, "tick-unseeded": ours})
    rc, r, text = S.profile_result(dirs, "coding-agent", "glm53", "tick-unseeded", False, tmp_path / "out")
    assert rc == 1
    assert r["cannot_run"] == [{"piece": "sampling", "words": UNSEEDED, "change": "seeded"}]
    assert r["second"]["ours"] == ["seeded for ours"] and r["second"]["throughput"]["verdict"] == "ours slower"
    lines = text.splitlines()
    assert lines[0] == (f"1. cannot run: {UNSEEDED} (arm tick-unseeded refuses sampling, which the profile has on)")
    assert lines[1] == "the second line, the same prompts with on our side alone seeded for ours (stock unchanged):"
    assert [x[:5] for x in lines[2:5]] == ["  1. ", "  2. ", "  3. "]
    result = json.loads((tmp_path / "out" / "result.json").read_text())
    assert not (tmp_path / "out" / "serving_profile.json").exists()
    assert (result["kind"], result["validation"]["status"]) == ("serve-tax-profile/v1", "failed")
    assert result["workload_fingerprint"] | {"source_sha": None} == {
        "profile": "coding-agent", "version": 2, "digest": r["digest"], "model": "glm53", "card": "h200", "stock": "prod",
        "feature": "tick-unseeded", "arm": "tick-unseeded", "load": "agentic-256k", "point": 4, "source_sha": None}
    got = {(m["name"], m["member"]): m for m in result["measurements"]}
    ours = "glm-5.3-h200-tick-unseeded-over-prod-coding-agent"
    refused = got["vllm/serving-cost-glm53", ours]
    assert (refused["value"], refused["why"]) == (None, f"cannot run: {UNSEEDED}")
    second = got["vllm/serving-cost-glm53", f"{ours}-seeded"]
    assert (second["unit"], second["mark"], second["label"], second["reading"]) == (
        "percent", "measured", "our extra card time a token", "ours slower")
    assert second["value"] == pytest.approx(100 * r["second"]["throughput"]["value"], abs=0.01)
    assert second["ci95"] == [pytest.approx(100 * x, abs=0.01) for x in r["second"]["throughput"]["ci95"]]
    assert got["ttft_p50_s", f"{ours}-seeded"]["value"] == r["second"]["latency"]["ours"]["ttft_p50_s"]
    assert got["within", "glm-5.3-h200-prod-coding-agent"]["value"] == r["second"]["within"]["stock"]["within"]
    assert len(got) == 2 + 2 * len(S.SIDE_FIGURES)
    # where no change on our side lets the profile through, nothing is served and there is no second line
    rc, r, text = S.profile_result([], "coding-agent", "glm53", "prod-restore", False)
    assert rc == 1 and "second" not in r and r["cannot_run"][0]["change"] is None
    assert text.splitlines()[1] == "no second line: no change on our side alone lets the profile's prompts through"


def test_the_results_three_lines_come_first_in_order_then_the_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ours = [[rep(1, 90, 0.3, 0.025, 80), rep(2, 91), rep(3, 89)], [rep(1, 90), rep(2, 90.5), rep(3, 89.5)]]
    dirs = arm_dirs(tmp_path, monkeypatch, "agentic-256k", {"prod": STOCK, "prod-record": ours})
    rc, r, text = S.profile_result(dirs, "coding-agent", "glm53", "prod-record", False, tmp_path / "res")
    assert rc == 0 and list(r)[:3] == ["throughput", "latency", "within"]
    result = json.loads((tmp_path / "res" / "result.json").read_text())
    cost = next(m for m in result["measurements"] if m["name"] == "vllm/serving-cost-glm53")
    assert (result["validation"]["status"], cost["member"], cost["cost_unit"], cost["value"]) == (
        "passed", "glm-5.3-h200-prod-record-over-prod-coding-agent", "lifetime", round(100 * r["throughput"]["value"], 2))
    t = r["throughput"]
    assert (t["label"], t["unit"], t["reps"]) == ("our extra card time a token", "lifetime", 2)
    assert t["verdict"] == "ours slower" and t["ci95"][0] > t["floor"]
    each = [sum(1 / x["output_tok_per_s"] for x in o) / sum(1 / x["output_tok_per_s"] for x in s) - 1
            for s, o in zip(STOCK, ours, strict=True)]
    assert t["value"] == pytest.approx(sum(each) / 2, abs=1e-4)
    assert r["latency"]["ours"]["ttft_p50_s"] == 0.2 and r["latency"]["stock"]["tpot_p99_s"] == pytest.approx(0.03)
    assert r["within"]["ours"] == {"met": 530, "sent": 576, "within": 530 / 576}
    assert (r["profile"], r["version"], r["model_id"], r["quick_check"]) == ("coding-agent", 2, "glm53", False)
    assert r["vllm"] == P.load().vllm and r["served_vllm"] == ["0.1+gd9105ea8"]
    lines = text.splitlines()
    assert lines[0].startswith("1. our extra card time a token, our seconds for the same requests over stock's, less "
                               "one: +")
    assert "(mean of 2 server lifetimes' costs, Student's t 95% interval)" in lines[0]
    assert lines[0].endswith(": ours slower")
    assert lines[1].startswith("2. time to first token, median / P99: ours 200.0 ms / 800.0 ms, stock ")
    assert lines[2] == "3. within the bounds (each token within 80 ms): ours 92.0% of 576 requests, stock 93.8% of 576 requests"
    assert lines[3] == "prod-record (prod-record) beside prod at agentic-256k, 4 users, 2 lifetimes of 3 reps each"
    assert lines[4].startswith("profile coding-agent v2 (digest ")
    # an interval the noise floor straddles, or a side the cap cut short, cannot tell
    same = [[rep(1, 100.4), rep(2, 101.2), rep(3, 99.2)], [rep(1, 99.6), rep(2, 100.3), rep(3, 99.3)]]
    for k in (0, 1):
        sub = tmp_path / f"t{k}"
        sub.mkdir()
        dirs = arm_dirs(sub, monkeypatch, "agentic-256k", {"prod": STOCK, "prod-record": same if k == 0 else ours},
                        stopped=None if k == 0 else "prod-record.2")
        t = S.profile_result(dirs, "coding-agent", "glm53", "prod-record", False)[1]["throughput"]
        assert t["verdict"] == "cannot tell"
        assert t["why"] == ("the interval widened by the noise floor holds zero" if k == 0 else
                            "the job's 120 minutes ran out: prod-record stopped before agentic-256k@4 rep 3")


def test_on_requests_in_flight_the_first_line_is_the_kits_median_of_paired_reps(tmp_path: Path,
                                                                              monkeypatch: pytest.MonkeyPatch):
    ours = [[rep(1, 90), rep(2, 91), rep(3, 89)], [rep(1, 90), rep(2, 90.5), rep(3, 89.5)]]
    by = S.joined([S.arm_metrics(d) for d in arm_dirs(tmp_path, monkeypatch, "chat", {"prod": STOCK, "prod-record": ours})])
    t = S.three_lines(by["prod"], by["prod-record"], "chat", 4, None)["throughput"]
    assert (t["label"], t["unit"], t["reps"], t["verdict"]) == ("our extra card time a token", "rep", 6, "ours slower")
    assert t["value"] == pytest.approx(S.statistics.median(s["output_tok_per_s"] / o["output_tok_per_s"] - 1
                                                           for s, o in zip(sum(STOCK, []), sum(ours, []))), abs=1e-3)
    assert S.lines_text(S.three_lines(by["prod"], by["prod-record"], "chat", 4, None), "")[0].startswith(
        "1. our extra card time a token, stock's output tokens a second at full load over ours, less one: +")
    ours[1][2] = {**ours[1][2], "output_tokens": 9599}
    (tmp_path / "again").mkdir()
    by = S.joined([S.arm_metrics(d) for d in arm_dirs(tmp_path / "again", monkeypatch, "chat",
                                                      {"prod": STOCK, "prod-record": ours})])
    with pytest.raises(SystemExit, match="--profile: prod and prod-record served 9600 and 9599 output tokens in rep 6 "
                                         "at load chat, at concurrency 4"):
        S.three_lines(by["prod"], by["prod-record"], "chat", 4, None)


@pytest.mark.parametrize(("name", "card"), [("NVIDIA H200", "h200"), ("NVIDIA H100 80GB HBM3", "h100")])
def test_find_load_prints_the_files_line_the_version_to_raise_and_the_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                           name: str, card: str):
    monkeypatch.setattr(S, "card", lambda: name)
    plan = S.profile_plan("chat", "llama32-1b", None, True)
    assert plan == {"STOCK": "compiled", "OURS": "", "CAP_MIN": "20", "CARD": card, "ARMS": "compiled",
                    "LOADS": "chat=4+8+12+16+24+32+48", "REPS": "3", "LIFETIMES": "1"}
    assert S.profile_plan("fixed-size", "llama32-1b", None, True)["LOADS"] == "long=1+2+4+8"
    assert S.profile_plan("coding-agent", "llama32-1b", None, True)["LOADS"] == "agentic=1+2+4+8+16+32+64"
    with pytest.raises(SystemExit, match=f"on llama32-1b.{card}: the load is not found yet: run --find-load first"):
        S.profile_plan("chat", "llama32-1b", "record", False)
    tok = {4: 100, 8: 190, 12: 260, 16: 300, 24: 300, 32: 298, 48: 295}
    reps = {c: [{**rep(r, v), "concurrency": c} for r in (1, 2, 3)] for c, v in tok.items()}
    metrics = {"arm": "compiled", "model_id": "llama32-1b", "slo": {}, "vllm": "v", "lifetime": None, "reps": 3,
               "left_out": 0, "loads": {f"chat@{c}": x for c, x in reps.items()}}
    (tmp_path / "compiled").mkdir()
    (tmp_path / "compiled" / "meta.json").write_text(json.dumps({"arm": "compiled"}))
    monkeypatch.setattr(S, "arm_metrics", lambda d, slo=None: metrics)
    rc, r, text = S.profile_result([tmp_path / "compiled"], "chat", "llama32-1b", None, True, tmp_path / "out")
    text_ = P.PATH.read_text(encoding="utf-8")
    table = text_.index("[profiles.chat.in_flight]")
    old = f'llama32-1b.{card} = "not found yet"'
    at = text_.index(old, table)
    edited = tmp_path / "production.toml"
    edited.write_text(text_[:at] + f"llama32-1b.{card} = 16" + text_[at + len(old):], encoding="utf-8")
    digest = P.digest(P.load(edited), "chat")
    assert rc == 0 and (r["verdict"], r["peak"], r["raise_version"], r["pin"], r["card"]) == (
        "saturated", 16, 2, [2, digest], card)
    assert text.splitlines()[0].startswith(f"llama32-1b stock on {card} at chat: 4 in flight 100 tok/s")
    assert text.splitlines()[2:5] == [f"in benchmarks/production.toml, under [profiles.chat.in_flight]: llama32-1b.{card} = 16",
                                      "and under [profiles.chat]: version = 2 (from 1)",
                                      f'in tests/benchmarks/harness/test_profiles.py, PINS: "chat": (2, "{digest}"),']
    assert json.loads((tmp_path / "out" / "find_load.json").read_text()) == r
    result = json.loads((tmp_path / "out" / "result.json").read_text())
    assert (result["kind"], result["validation"]["status"], result["measurements"]) == (
        "serve-tax-profile/v1", "passed", [])
    assert P.PATH.read_text(encoding="utf-8") == text_
    # stock still rising at its highest rung: no line
    metrics["loads"] = {f"chat@{c}": x for c, x in reps.items() if c <= 12}
    rc, r, text = S.profile_result([tmp_path / "compiled"], "chat", "llama32-1b", None, True)
    assert (r["verdict"], "line" in r) == ("cannot tell", False)
    assert text.splitlines()[1] == "cannot tell: stock still rises at its highest rung served (12): no line"


def test_find_load_on_sessions_takes_the_fewest_users_past_which_the_rise_is_under_the_ladders_gain(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(S, "card", lambda: "NVIDIA H200")
    tok = {1: 100, 2: 190, 4: 300, 8: 310, 16: 330, 32: 331, 64: 331}
    reps = {c: [{**rep(r, v), "concurrency": c} for r in (1, 2, 3)] for c, v in tok.items()}
    metrics = {"arm": "compiled", "model_id": "llama32-1b", "slo": {}, "vllm": "v", "lifetime": None, "reps": 3,
               "left_out": 0, "loads": {f"agentic@{c}": x for c, x in reps.items()}}
    (tmp_path / "compiled").mkdir()
    (tmp_path / "compiled" / "meta.json").write_text(json.dumps({"arm": "compiled"}))
    monkeypatch.setattr(S, "arm_metrics", lambda d, slo=None: metrics)
    rc, r, text = S.profile_result([tmp_path / "compiled"], "coding-agent", "llama32-1b", None, True)
    assert rc == 0 and (r["verdict"], r["peak"], r["key"], r["line"], r["raise_version"]) == (
        "saturated", 4, "users", "llama32-1b.h200 = 4", 3)
    assert text.splitlines()[1] == ("saturated: output tokens a second rise by less than 5% past 4 "
                                    "(1+2+4+8+16+32+64 served)")
    # rising by 5% or more at every rung served: no line
    metrics["loads"] = {f"agentic@{c}": x for c, x in reps.items() if c <= 4}
    rc, r, text = S.profile_result([tmp_path / "compiled"], "coding-agent", "llama32-1b", None, True)
    assert (r["verdict"], "line" in r) == ("cannot tell", False)
    assert text.splitlines()[1] == "cannot tell: stock still rises by 5% or more at each rung served, up to 4: no line"


def bash(argv: list[str], env: dict[str, str]) -> tuple[int, str]:
    p = subprocess.run(["bash", *argv], capture_output=True, text=True, env={"PATH": S.os.environ["PATH"], **env})
    return p.returncode, p.stdout + p.stderr


def run_sh(*flags: str, **env: str) -> tuple[int, str]:
    return bash([str(S.HERE / "run.sh"), *flags], env)


def test_run_sh_takes_a_profile_and_refuses_what_it_fixes_before_it_holds_anything():
    one = ("--model", "llama32-1b", "--profile", "chat", "--find-load")
    for flags, env in ((one, {}), (("--model", "glm53", "--profile", "coding-agent", "--feature", "prod-record"), {})):
        rc, out = run_sh(*flags, **env)
        assert rc != 0 and "run this under research run" in out
    assert run_sh("--model", "llama32-1b", "--profile", "chat") == (
        2, "serve-tax: --profile chat takes --feature SCHEME or --find-load, one of them\n")
    assert run_sh(*one, "--feature", "record")[0] == 2
    assert run_sh("--find-load") == (2, "serve-tax: --feature and --find-load are a profile's: --profile NAME\n")
    for env in ({"PROFILE": "chat"}, {"FEATURE": "record"}, {"FIND_LOAD": "1"}):
        assert run_sh("--model", "llama32-1b", **env) == (
            2, "serve-tax: PROFILE, FEATURE and FIND_LOAD are not read from the environment: --profile NAME with "
               "--feature SCHEME or --find-load\n")
    fixed = {("--loads", "chat=4"): "--loads/LOADS", ("--arms", "compiled record"): "--arms/ARMS",
             ("--reps", "6"): "--reps/REPS and LIFETIMES", ("--target", "5"): "--target/TARGET and WITH"}
    for flags, drop in fixed.items():
        rc, out = run_sh(*one, *flags)
        assert rc == 2 and out.startswith("serve-tax: --profile chat fixes ") and out.endswith(f": drop {drop}\n"), out
    for env, drop in (({"SAMPLED": "1"}, "SAMPLED"), ({"MAX_MODEL_LEN": "8192"}, "MAX_MODEL_LEN"),
                      ({"LIFETIMES": "3"}, "--reps/REPS and LIFETIMES"), ({"PASS_PROMPTS": "chat=8"}, "PASS_PROMPTS"),
                      ({"PUBLISHED_ARGS": "--seed=1"}, "PUBLISHED_ARGS and PUBLISHED_FRONTEND")):
        rc, out = run_sh(*one, **env)
        assert rc == 2 and out.endswith(f": drop {drop}\n"), out
    sh = (S.HERE / "run.sh").read_text()
    # the plan before the GPUs, the cap as the time the arms stop by, the profile and our side's changes on each run,
    # and the result in place of the cost
    assert 'profile plan --profile "$PROFILE" --model-id "$MODEL_ID"' in sh and 'eval "$plan"' in sh
    assert "cap_at=$((t_leased + CAP_MIN * 60))" in sh
    assert '[ "$arm" != "$STOCK" ] && [ -n "${OURS:-}" ] && profile+=(--ours "$OURS")' in sh
    assert sh.index('profile result "${dirs[@]}"') < sh.index('cost=("$PY" "$here/serve_tax.py" cost')
    # the second entry takes the profile as a flag, as the first does: nothing of it is exported
    assert sh.count('run.sh" leased ${PROFILE:+--profile "$PROFILE"}') == 2 and "export PROFILE" not in sh
    assert run_sh("leased", "--feature", "record") == (2, "serve-tax: leased takes --profile NAME alone\n")


def test_the_card_is_the_driver_names_card_file_and_the_commit_needs_no_git(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(S, "card", lambda: "NVIDIA H100 80GB HBM3")
    assert S.profile_card("llama32-1b") == "h100"
    with pytest.raises(SystemExit, match="this host's card is h100, and production.toml serves glm53 on h200: add h100"):
        S.profile_plan("coding-agent", "glm53", None, True)
    monkeypatch.setattr(S, "card", lambda: "none")
    with pytest.raises(SystemExit, match="nvidia-smi names no card on this host"):
        S.profile_card("llama32-1b")
    monkeypatch.setattr(S, "card", lambda: "NVIDIA A100-SXM4-80GB")
    with pytest.raises(SystemExit, match="0 card records .* have 'NVIDIA A100-SXM4-80GB'"):
        S.profile_card("llama32-1b")
    monkeypatch.setattr(S, "source_commit", lambda: None)
    with pytest.raises(SystemExit, match="no commit names this tree: neither RESEARCH_SOURCE_SHA .* nor "
                                         "RESEARCH_SOURCE_COMMIT .* and git gives no HEAD here"):
        S.profile_plan("chat", "llama32-1b", None, True)
    monkeypatch.delenv("RESEARCH_SOURCE_SHA", raising=False)
    monkeypatch.setenv("RESEARCH_SOURCE_COMMIT", "abc123")
    assert SOURCE_COMMIT() == "abc123"
    monkeypatch.delenv("RESEARCH_SOURCE_COMMIT")
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr(S, "ROOT", Path("/nonexistent"))
    assert SOURCE_COMMIT() is None


def test_run_sh_on_a_machine_not_ours_names_venv_and_takes_the_hubs_own_place(tmp_path: Path):
    """No /workspace of ours: VENV refused by name where it holds no environment, and HF_HOME the hub's own place under
    XDG_CACHE_HOME; on our nodes, where /workspace's are, both are as they were."""
    env = {"RESEARCH_RUN_DIR": str(tmp_path / "run"), "HOME": str(tmp_path / "home"), "XDG_CACHE_HOME": str(tmp_path / "c")}
    rc, out = run_sh("--model", "llama32-1b", "--profile", "chat", "--find-load", VENV=str(tmp_path / "none"), **env)
    assert (rc, out) == (2, f"serve-tax: no environment at VENV={tmp_path / 'none'}: on a machine that is not ours (no "
                            "/workspace/venv312), set VENV to one holding vLLM d9105ea80\n")
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    (tmp_path / "venv" / "bin" / "python").symlink_to(sys.executable)
    rc, out = run_sh("--model", "llama32-1b", "--profile", "chat", "--find-load", VENV=str(tmp_path / "venv"), **env)
    assert (rc, out) == (2, "serve-tax: CUDA_HOME is not set: set it to the CUDA toolkit of torch's CUDA version (a node's placement "
                            "entry names it)\n")
    rc, out = run_sh("--model", "llama32-1b", "--profile", "chat", "--find-load", VENV=str(tmp_path / "venv"),
                     CUDA_HOME=str(tmp_path / "cuda"), **env)
    ours = next((d for d in ("/workspace/cp/hf", "/workspace/hf") if Path(d).is_dir()), None)
    hub = Path(ours or tmp_path / "c" / "huggingface") / "hub"
    m = S.MODELS["llama32-1b"]
    assert (rc, out) == (2, f"serve-tax: no checkpoint at {hub}/models--unsloth--Llama-3.2-1B/snapshots/{m.revision}\n")
    sh = (S.HERE / "run.sh").read_text()
    assert "[ -d /workspace/cp/venv312 ] && venv=/workspace/cp/venv312 || venv=/workspace/venv312" in sh
    assert "[ -d /workspace/cp/hf ] && hf=/workspace/cp/hf || hf=/workspace/hf\n[ -d \"$hf\" ] || hf=" in sh
