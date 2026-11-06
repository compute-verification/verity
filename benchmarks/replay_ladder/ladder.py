"""The replay ladder: on each model of a ladder of pinned checkpoints, does replaying a saturated served window under its
recorded schedule give the served tokens, compiled, run after run, and how far do the tokens depend on the batch?

    ladder.py serve   --model ID --c4 C4.json.gz --out DIR [--span-s 60] [--over 1.45] [--burst 800]
    ladder.py replay  --model ID --out DIR --index I
    ladder.py alone   --model ID --out DIR [--requests 64]
    ladder.py summary DIR... --out DIR

A model (`LADDER`) is a role of the integration's checkpoints.json, served at the integration's vLLM pin by serve_tax's
engine: vLLM's own OpenAI server, compiled with CUDA graphs, chunked prefill, prefix caching and async scheduling
(serve_tax.BASE_ARGS), BF16 activations, at the model's tensor-parallel width.

- ``serve`` measures the model's capacity with one burst of ``--burst`` requests (serve_tax's ``compiled`` arm), then
  serves serve_tax's workload (C4 prompts, most behind one of eight shared prefixes, Poisson arrivals, every request
  decoding to its cap) at ``--over`` times that capacity for ``--span-s`` seconds with the scheduler's step trace
  (the ``record`` arm): a saturated window of mixed arrivals, prefill chunks beside decodes.
- ``replay`` replays that window in a fresh process under its recorded schedule, compiled (serve_tax's `replay`), and
  compares its schedule and every request's tokens with serving's, and from the second replay on its tokens with the
  first replay's: run-to-run variation at a fixed schedule.
- ``alone`` serves an evenly spaced sample of the window's requests one at a time, each on an empty prefix cache, in the
  same compiled engine, and compares their tokens with serving's: how far a request's tokens depend on its batch, which
  is why the replay needs the recorded schedule.
- ``summary`` reads each model's directory into summary.json and summary.md. A row of output is one request's token at
  one position; its share exact is the window's output tokens less those that differ, over the window's output tokens.
"""
from __future__ import annotations

import argparse
import functools
import gzip
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SERVE_TAX = ROOT / "benchmarks/serve_tax"
CHECKPOINTS = ROOT / "integrations/vllm/manifests/checkpoints.json"
sys.path.insert(0, str(SERVE_TAX))
import replay as R  # noqa: E402 (serve_tax's, put on sys.path above)
import serve_tax as S  # noqa: E402
SCHEMA = "verity-bench/replay-ladder/v0"
#: the load each model's window is served at, and the one each replay and the alone sample read
LOAD = "saturated"


@dataclass(frozen=True)
class Rung:
    #: the checkpoint's role in the integration's checkpoints.json
    role: str
    served: str
    tp: int = 1


LADDER = {
    "qwen3-8b": Rung("QWEN3_8B", "qwen3-8b"),
    "llama31-8b": Rung("LLAMA31_8B", "llama-3.1-8b"),
    "r1-distill-llama-8b": Rung("R1_DISTILL_LLAMA_8B", "r1-distill-llama-8b"),
    "gemma2-9b": Rung("GEMMA2_9B", "gemma-2-9b"),
    "phi4": Rung("PHI4", "phi-4"),
    "olmoe-1b-7b": Rung("OLMOE", "olmoe-1b-7b"),
    "qwen3-30b-a3b": Rung("QWEN3_30B_A3B", "qwen3-30b-a3b"),
    "qwen3-30b-a3b-fp8": Rung("QWEN3_30B_A3B_FP8", "qwen3-30b-a3b-fp8"),
    "llama31-70b": Rung("LLAMA31_70B", "llama-3.1-70b", tp=2),
    "qwen3-235b-a22b-fp8": Rung("QWEN3_235B_A22B_FP8", "qwen3-235b-a22b-fp8", tp=4),
}


def entry(model: str, manifest: Path = CHECKPOINTS) -> dict:
    """The model's checkpoints.json entry (its repo, revision and local path)."""
    role = LADDER[model].role
    for e in json.loads(manifest.read_text())["checkpoints"]:
        if e.get("role") == role and e.get("local_path"):
            return e
    raise SystemExit(f"no checkpoints.json entry with role {role} and a local path")


def register(model: str, manifest: Path = CHECKPOINTS) -> Path:
    """Add the model to serve_tax's models and return its checkpoint directory."""
    rung, e = LADDER[model], entry(model, manifest)
    args = ("--dtype", "bfloat16", *(("--tensor-parallel-size", str(rung.tp)) if rung.tp > 1 else ()))
    S.MODELS[model] = S.Model(e["repo"], e["revision"], rung.served, args, (2.0, 0.08))
    return Path(e["local_path"])


def tp_engine(tp: int) -> Any:
    """serve_tax's replay engine, at tensor-parallel width ``tp``."""
    make = R.engine

    def engine(checkpoint: Path, header: dict, eager: bool) -> Any:
        if tp == 1:
            return make(checkpoint, header, eager)
        import vllm

        real = vllm.EngineArgs
        vllm.EngineArgs = functools.partial(real, tensor_parallel_size=tp)
        try:
            return make(checkpoint, header, eager)
        finally:
            vllm.EngineArgs = real
    return engine


# ------------------------------------------------------------------------------------------------------------ serve

def native_tokens(tok: Any) -> tuple[int, int]:
    """The tokenizer's document start (its BOS, else its EOS) and blank line, which the traffic plan, written for Llama 3's
    ids, puts at the head of every prompt."""
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    blank = tok.encode("\n\n", add_special_tokens=False)
    if start is None or len(blank) != 1:
        raise SystemExit(f"the tokenizer has no document-start token ({start}) or no one-token blank line ({blank})")
    return int(start), int(blank[0])


def serve(a: argparse.Namespace) -> int:
    ck = register(a.model)
    from transformers import AutoTokenizer

    traffic = S.corpus().traffic
    traffic.BOS, traffic.BLANK_LINE = native_tokens(AutoTokenizer.from_pretrained(str(ck), trust_remote_code=True))
    out = Path(a.out)
    cap = out / "capacity"
    base = ["--model-id", a.model, "--checkpoint", str(ck), "--c4", a.c4, "--reps", "1", "--warm-s", "10"]
    if S.main(["run", "--arm", "compiled", *base, "--rates", "burst=1000", "--requests", str(a.burst), "--out", str(cap)]):
        return 1
    burst = S.arm_metrics(cap)["loads"]["burst"][0]
    capacity = burst["req_per_s"]
    rate = round(a.over * capacity, 2)
    print(f"[ladder] {a.model}: capacity {capacity:.2f} req/s, {burst['output_tok_per_s']:.0f} out tok/s; "
          f"serving the window at {rate} req/s for {a.span_s:g} s", flush=True)
    rc = S.main(["run", "--arm", "record", *base, "--rates", f"{LOAD}={rate}", "--span-s", str(a.span_s),
                 "--plan", str(out / "plan.jsonl"), "--out", str(out / "record")])
    rec = S.arm_metrics(out / "record")
    (out / "serve.json").write_text(json.dumps({
        "schema": SCHEMA + "/serve", "model": a.model, "checkpoint": f"{S.MODELS[a.model].repo}@{S.MODELS[a.model].revision}",
        "tp": LADDER[a.model].tp, "bos": traffic.BOS, "blank_line": traffic.BLANK_LINE, "capacity": burst, "over": a.over,
        "rate": rate, "span_s": a.span_s,
        "window": (rec["loads"].get(LOAD) or [None])[0], "error": rec.get("error"), "run_id": S.SHARED.RESEARCH_RUN_ID,
    }, indent=1))
    return rc


# ----------------------------------------------------------------------------------------------------------- replay

def replay(a: argparse.Namespace) -> int:
    ck = register(a.model)
    R.engine = tp_engine(LADDER[a.model].tp)
    out = Path(a.out)
    argv = ["--arm-dir", str(out / "record"), "--load", LOAD, "--checkpoint", str(ck), "--out", str(out / f"replay-{a.index}")]
    if a.index > 1:
        argv += ["--against", str(out / "replay-1")]
    return R.main(argv)


# ------------------------------------------------------------------------------------------------------------ alone

def sample(requests: list[dict], k: int) -> list[dict]:
    """``k`` of the window's requests, evenly spaced in arrival order."""
    reqs = sorted(requests, key=lambda r: (r["before_step"], r["engine_id"]))
    k = min(k, len(reqs))
    return [reqs[i * len(reqs) // k] for i in range(k)]


def alone(a: argparse.Namespace) -> int:
    if sys.flags.hash_randomization:
        raise SystemExit("run with PYTHONHASHSEED=0, as the served engine did")
    ck = register(a.model)
    out = Path(a.out)
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_DISABLE_REQUEST_ID_RANDOMIZATION="1")
    w = R.window(out / "record", LOAD, 1)
    picks = sample(w["requests"], a.requests)
    t0 = time.monotonic()
    eng = tp_engine(LADDER[a.model].tp)(ck, w["header"], False)
    ready = time.monotonic() - t0
    tokens: dict[str, list[int]] = {}
    t0 = time.monotonic()
    for r in picks:
        if not any(eng.reset_prefix_cache() or time.sleep(0.1) for _ in range(50)):
            raise SystemExit(f"the prefix cache could not be reset before {r['engine_id']}")
        eng.add_request(r["engine_id"], {"prompt_token_ids": r["prompt_ids"]}, R.params(r))
        while eng.has_unfinished_requests():
            for o in eng.step():
                if o.finished:
                    tokens[o.request_id] = [int(t) for t in o.outputs[0].token_ids]
    wall = time.monotonic() - t0
    served = {r["engine_id"]: r["served"] for r in picks}
    d = R.token_diff(served, tokens)
    res = {"schema": SCHEMA + "/alone", "model": a.model, "run_id": S.SHARED.RESEARCH_RUN_ID, "load": LOAD,
           "requests": len(picks), "tokens_served": sum(map(len, served.values())), "requests_tokens_differing": len(d),
           "token_positions_differing": sum(x["positions"] for x in d), "requests_differing": d,
           "engine_ready_s": ready, "wall_s": wall}
    first = out / "replay-1"
    if (first / "replay_tokens.jsonl.gz").exists():
        rep = R.read_tokens(first)
        dr = R.token_diff({k: rep[k] for k in served if k in rep}, tokens)
        res["against_replay"] = {"replay": str(first), "requests_tokens_differing": len(dr),
                                 "token_positions_differing": sum(x["positions"] for x in dr)}
    (out / "alone").mkdir(parents=True, exist_ok=True)
    (out / "alone" / "alone.json").write_text(json.dumps(res, indent=1))
    with gzip.open(out / "alone" / "alone_tokens.jsonl.gz", "wt") as f:
        for rid in sorted(tokens):
            f.write(json.dumps({"request": rid, "tokens": tokens[rid]}, separators=(",", ":")) + "\n")
    print(json.dumps({k: res[k] for k in ("model", "requests", "requests_tokens_differing", "token_positions_differing",
                                          "wall_s")}))
    return 0


# ---------------------------------------------------------------------------------------------------------- summary

#: what vLLM logs at its pin when a process compiles its graphs, and when it loads them from its compile cache
COMPILE_FRESH = re.compile(r"Compiling a graph for compile range|saved AOT compiled function")
COMPILE_LOAD = re.compile(r"Directly load AOT compilation|Directly load the compiled graph")


def compile_mode(log: Path) -> str | None:
    """``compile`` when the process compiled its graphs, ``load`` when it read them from vLLM's compile cache."""
    if not log.exists():
        return None
    text = log.read_text(errors="replace")
    return "compile" if COMPILE_FRESH.search(text) else "load" if COMPILE_LOAD.search(text) else None


def model_row(d: Path) -> dict:
    """One model's directory: its window, its replays and its alone sample."""
    serve_meta = json.loads((d / "serve.json").read_text()) if (d / "serve.json").exists() else {}
    row: dict[str, Any] = {"model": d.name, "checkpoint": serve_meta.get("checkpoint"), "tp": serve_meta.get("tp"),
                           "serve_run_id": serve_meta.get("run_id"), "capacity_req_s": (serve_meta.get("capacity") or {}).get("req_per_s"),
                           "rate_req_s": serve_meta.get("rate"), "replays": [], "alone": None, "error": serve_meta.get("error"),
                           "serve_compile": compile_mode(d / "record" / "server" / "server.log")}
    win = serve_meta.get("window") or {}
    row["window"] = {"sent": win.get("sent"), "served": win.get("ok"), "output_tokens": win.get("output_tokens"),
                     "span_s": win.get("span_s"), "req_s": win.get("req_per_s")}
    for rd in sorted((p for p in d.glob("replay-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[1])):
        f = rd / "replay.json"
        if not f.is_file():
            row["replays"].append({"index": int(rd.name.split("-")[1]), "error": "no replay.json"})
            continue
        r = json.loads(f.read_text())
        tokens = r["tokens_served"] or 1
        row["replays"].append({
            "index": int(rd.name.split("-")[1]), "run_id": r.get("run_id"), "requests": r["requests"],
            "steps": r["steps_served"], "rows_scheduled": r["rows_served"], "steps_differing": r["steps_differing"],
            "rows_differing": r["rows_differing"], "tokens_served": r["tokens_served"],
            "requests_tokens_differing": r["requests_tokens_differing"],
            "token_positions_differing": r["token_positions_differing"],
            "rows_exact_share": 1 - r["token_positions_differing"] / tokens, "bit_exact": r["bit_exact"],
            "against_first_requests_differing": (r.get("against") or {}).get("requests_tokens_differing"),
            "replay_over_served": r["replay_over_served"], "compile": compile_mode(d / f"{rd.name}.log")})
    af = d / "alone" / "alone.json"
    if af.is_file():
        x = json.loads(af.read_text())
        row["alone"] = {k: x.get(k) for k in ("run_id", "requests", "tokens_served", "requests_tokens_differing",
                                              "token_positions_differing")} | {
            "against_replay_requests_differing": (x.get("against_replay") or {}).get("requests_tokens_differing")}
    return row


def markdown(rows: list[dict]) -> str:
    head = ("| model | TP | window: requests, steps, output tokens | replays exact | rows exact (worst replay) | "
            "requests differing from serving, per replay | requests differing from replay 1 | alone: requests differing |")
    lines = [head, "|" + "---|" * 8]
    for r in rows:
        reps = [x for x in r["replays"] if "bit_exact" in x]
        w = reps[0] if reps else {}
        exact = f"{sum(x['bit_exact'] for x in reps)} of {len(r['replays'])}" if r["replays"] else "—"
        worst = f"{min(x['rows_exact_share'] for x in reps):.6%}" if reps else "—"
        per = ", ".join(f"{x['requests_tokens_differing']}{' (fresh compile)' if x['compile'] == 'compile' else ''}"
                        for x in reps) or "—"
        first = ", ".join(str(x["against_first_requests_differing"]) for x in reps[1:]) or "—"
        al = r["alone"]
        alone_s = f"{al['requests_tokens_differing']} of {al['requests']}" if al else "—"
        win = (f"{w.get('requests', '—')}, {w.get('steps', '—')}, {w.get('tokens_served', '—')}" if w else "—")
        lines.append(f"| {r['model']} | {r['tp'] or '—'} | {win} | {exact} | {worst} | {per} | {first} | {alone_s} |")
    return "\n".join(lines) + "\n"


def summary(a: argparse.Namespace) -> int:
    rows = [model_row(d) for d in a.dirs if (d / "serve.json").exists() or (d / "record").is_dir()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"schema": SCHEMA + "/summary", "models": rows}, indent=1))
    md = markdown(rows)
    (out / "summary.md").write_text(md)
    print(md, end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--model", required=True, choices=sorted(LADDER))
    s.add_argument("--c4", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--span-s", type=float, default=60.0)
    s.add_argument("--over", type=float, default=1.45, help="the window's arrival rate over the measured capacity")
    s.add_argument("--burst", type=int, default=800, help="the capacity burst's requests")
    r = sub.add_parser("replay")
    r.add_argument("--model", required=True, choices=sorted(LADDER))
    r.add_argument("--out", required=True)
    r.add_argument("--index", type=int, required=True)
    al = sub.add_parser("alone")
    al.add_argument("--model", required=True, choices=sorted(LADDER))
    al.add_argument("--out", required=True)
    al.add_argument("--requests", type=int, default=64)
    sm = sub.add_parser("summary")
    sm.add_argument("dirs", nargs="+", type=Path)
    sm.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"serve": serve, "replay": replay, "alone": alone, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
