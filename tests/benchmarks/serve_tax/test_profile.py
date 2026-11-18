"""A serving profile on serve_tax's own job (`run.sh --profile`, `serve_tax.profile_run`): its server and client the
lines `verity_bench.profiles` builds, the job as it was without one, what it refuses, `cannot run` with its second
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

CLIENT = (sys.executable, "-m", "vllm.entrypoints.cli.main")
#: what a request without a seed gets under the certified tick (`verity_vllm.pipeline.tick_plugin.unseeded`)
UNSEEDED = ("under the certified tick a request sampled at the server's default temperature needs an integer seed, "
            "not null: send one, or temperature 0")


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


@pytest.mark.parametrize(("profile", "model", "load", "rung"), [
    ("fixed-size", "llama32-1b", "long", 8), ("fixed-size", "glm53", "long", 4),
    ("chat", "llama32-1b", "chat", 16), ("chat", "glm53", "chat", 12)])
def test_a_profiles_server_and_client_are_the_lines_verity_bench_profiles_builds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str, model: str, load: str, rung: int):
    stock = S.profile_stock(profile, model)
    data = {"chat": "/data"} if load == "chat" else {}
    servers, clients = served(tmp_path, monkeypatch, "--arm", stock, "--model-id", model, "--loads", f"{load}={rung}",
                              "--profile", profile, datasets=data)
    doc = P.load()
    (server,), (client,) = servers, clients
    assert server[server.index("--model") + 2:] == P.serve_argv(doc, model, "/ckpt", port=8000)[3:]
    want = P.bench_argv(doc, profile, model, "/ckpt", port=8000, result_dir="R", tag=f"{load}@{rung}-r1",
                        dataset_path=data.get(load), in_flight=rung, vllm=CLIENT)
    i = client.index("--result-dir") + 1
    assert [*client[:i], "R", *client[i + 1:]] == want
    meta = json.loads((tmp_path / "out" / "meta.json").read_text())
    assert meta["profile"] == {"name": profile, "version": doc.profiles[profile].version,
                               "digest": P.digest(doc, profile), "ours": [], "seeded": False}


@pytest.mark.parametrize(("model", "load"), [("glm53", "agentic-256k"), ("llama32-1b", "agentic")])
def test_a_coding_agent_users_client_is_the_line_verity_bench_profiles_builds(model: str, load: str):
    """A sessions pass's client is a user's (`drive_bench`), its file the user's and its prompts the cut's turns."""
    assert S.profile_load("coding-agent", model) == load
    turns = S.pass_prompts(S.LOADS[load], 1, None)
    ours = S.bench_argv(S.MODELS[model], Path("/ckpt"), 8000, S.as_profile(S.LOADS[load], "coding-agent"), 1,
                        Path("/r"), f"{load}@4-r1-u0", "/u0.jsonl", turns, defaults=True)
    assert ours == P.bench_argv(P.load(), "coding-agent", model, "/ckpt", port=8000, result_dir="/r",
                                tag=f"{load}@4-r1-u0", dataset_path="/u0.jsonl", vllm=CLIENT)


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
            "ttft_p99_s": ttft * 4, "tpot_p50_s": tpot, "tpot_p99_s": tpot * 1.5}


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
                               "left_out": 0, "loads": {f"{load}@{reps[0]['concurrency']}": reps} if reps else {}}
            dirs.append(d)
    monkeypatch.setattr(S, "arm_metrics", lambda d, slo=None: metrics[d.name])
    return dirs


STOCK = [[rep(1, 100), rep(2, 101), rep(3, 99)], [rep(1, 100, 0.21), rep(2, 100.5), rep(3, 99.5)]]


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
    assert json.loads((tmp_path / "out" / "serving_profile.json").read_text()) == r
    assert (tmp_path / "out" / "serving_profile.txt").read_text() == text + "\n"
    # where no change on our side lets the profile through, nothing is served and there is no second line
    rc, r, text = S.profile_result([], "coding-agent", "glm53", "prod-restore", False)
    assert rc == 1 and "second" not in r and r["cannot_run"][0]["change"] is None
    assert text.splitlines()[1] == "no second line: no change on our side alone lets the profile's prompts through"


def test_the_results_three_lines_come_first_in_order_then_the_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ours = [[rep(1, 90, 0.3, 0.025, 80), rep(2, 91), rep(3, 89)], [rep(1, 90), rep(2, 90.5), rep(3, 89.5)]]
    dirs = arm_dirs(tmp_path, monkeypatch, "agentic-256k", {"prod": STOCK, "prod-record": ours})
    rc, r, text = S.profile_result(dirs, "coding-agent", "glm53", "prod-record", False)
    assert rc == 0 and list(r)[:3] == ["throughput", "latency", "within"]
    t = r["throughput"]
    assert t["reps"] == 6 and t["verdict"] == "ours slower" and t["ci95"][0] > t["floor"]
    assert t["value"] == pytest.approx(sorted(s["output_tok_per_s"] / o["output_tok_per_s"] - 1
                                              for s, o in zip(sum(STOCK, []), sum(ours, [])))[2:4][0], abs=0.02)
    assert r["latency"]["ours"]["ttft_p50_s"] == 0.2 and r["latency"]["stock"]["tpot_p99_s"] == pytest.approx(0.03)
    assert r["within"]["ours"] == {"met": 530, "sent": 576, "within": 530 / 576}
    assert (r["profile"], r["version"], r["model_id"], r["quick_check"]) == ("coding-agent", 1, "glm53", False)
    assert r["vllm"] == P.load().vllm and r["served_vllm"] == ["0.1+gd9105ea8"]
    lines = text.splitlines()
    assert lines[0].startswith("1. output tokens a second at full load, stock's over ours less one: +")
    assert lines[0].endswith(": ours slower")
    assert lines[1].startswith("2. time to first token, median / P99: ours 200.0 ms / 800.0 ms, stock ")
    assert lines[2] == "3. within the bounds (each token within 80 ms): ours 92.0% of 576 requests, stock 93.8% of 576 requests"
    assert lines[3] == "prod-record beside prod at agentic-256k, 4 users, 2 lifetimes of 3 reps each"
    assert lines[4].startswith("profile coding-agent v1 (digest ")
    # an interval the noise floor straddles, or a side the cap cut short, cannot tell
    same = [[rep(1, 100.2), rep(2, 100.6), rep(3, 99.1)], [rep(1, 99.8), rep(2, 100.4), rep(3, 99.7)]]
    for k in (0, 1):
        sub = tmp_path / f"t{k}"
        sub.mkdir()
        dirs = arm_dirs(sub, monkeypatch, "agentic-256k", {"prod": STOCK, "prod-record": same if k == 0 else ours},
                        stopped=None if k == 0 else "prod-record.2")
        t = S.profile_result(dirs, "coding-agent", "glm53", "prod-record", False)[1]["throughput"]
        assert t["verdict"] == "cannot tell"
        assert t["why"] == ("the interval widened by the noise floor holds zero" if k == 0 else
                            "the job's 120 minutes ran out: prod-record stopped before agentic-256k@4 rep 3")


def test_find_load_prints_the_files_line_the_version_to_raise_and_the_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    plan = S.profile_plan("chat", "llama32-1b", None, True)
    assert plan == {"STOCK": "compiled", "OURS": "", "CAP_MIN": "20", "ARMS": "compiled",
                    "LOADS": "chat=4+8+12+16+24+32+48", "REPS": "3", "LIFETIMES": "1"}
    assert S.profile_plan("fixed-size", "llama32-1b", None, True)["LOADS"] == "long=4+8+12+16+24"
    assert S.profile_plan("coding-agent", "llama32-1b", None, True)["LOADS"] == "agentic=1+2+4+8"
    with pytest.raises(SystemExit, match="the load is not found yet: run --find-load first"):
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
    at = text_.index('llama32-1b = "not found yet"', table)
    edited = tmp_path / "production.toml"
    edited.write_text(text_[:at] + "llama32-1b = 16" + text_[at + len('llama32-1b = "not found yet"'):], encoding="utf-8")
    digest = P.digest(P.load(edited), "chat")
    assert rc == 0 and (r["verdict"], r["peak"], r["raise_version"], r["pin"]) == ("saturated", 16, 2, [2, digest])
    assert text.splitlines()[2:5] == ["in benchmarks/production.toml, under [profiles.chat.in_flight]: llama32-1b = 16",
                                      "and under [profiles.chat]: version = 2 (from 1)",
                                      f'in tests/benchmarks/harness/test_profiles.py, PINS: "chat": (2, "{digest}"),']
    assert (tmp_path / "out" / "find_load.txt").read_text() == text + "\n"
    assert P.PATH.read_text(encoding="utf-8") == text_
    # stock still rising at its highest rung: no line
    metrics["loads"] = {f"chat@{c}": x for c, x in reps.items() if c <= 12}
    rc, r, text = S.profile_result([tmp_path / "compiled"], "chat", "llama32-1b", None, True)
    assert (r["verdict"], "line" in r) == ("cannot tell", False)
    assert text.splitlines()[1] == "cannot tell: stock still rises at its highest rung served (12): no line"


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
        2, "serve-tax: --profile chat takes --feature ARM or --find-load, one of them\n")
    assert run_sh(*one, "--feature", "record")[0] == 2
    assert run_sh("--find-load") == (2, "serve-tax: --feature and --find-load are a profile's: --profile NAME\n")
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
