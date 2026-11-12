"""Replay one served window of a ``record`` arm (`serve_tax`) under its recorded schedule, and say how close it came.

    replay.py --arm-dir OUT/record --load NAME [--rep 1] [--checkpoint DIR] [--eager] [--against EARLIER_OUT] [--tp N]
              [--capture ID,... [--capture-template T]] --out DIR

The window is the load's requests: its step trace's steps from the first one's arrival to the last one's finish, all of them
the load's own (each load starts from an empty prefix cache and ends drained). The replay builds the served engine again
in this process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) from the trace's header (its batch caps, chunked prefill, prefix
caching, async scheduling, block size and block count, and where the header holds them the long-prefill threshold, the KV
cache's format and the speculative decoding: `HEADER_ARGS`), compiled unless ``--eager``, and adds each request, under its served
engine id with its served prompt and sampling, right before the scheduled step it arrived before in the trace (a wrapper of
the scheduler's ``schedule``); the scheduler decides the rest as it did while serving. Its own step trace (`StepTrace`) is
then compared with the served one, step by step: the rows (request, first position, tokens) that differ and the first step
that does, and each request's sampled tokens against the ones the client received. replay.json has the counts, the
differing requests (each one's lengths and first differing position) and the replay's wall time beside the served window's
(first arrival to last step); replay_tokens.jsonl.gz has every request's replayed tokens, which ``--against`` compares a
later replay's with, to tell a replay that differs from serving from one that differs from itself. ``--capture`` takes the
named requests' logits rows and ids at every rank's sampler (`verity_vllm.engine.sampler_capture`) into OUT/capture, a
token-select span each, and replay.json says whether each one's ids are the ones served.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "config"))
sys.path.append(str(ROOT))
from verity_config.settings import SHARED  # noqa: E402 (from the tree's tools/config, put on sys.path above)
import serve_tax as S  # noqa: E402 (beside this file)


def window(arm_dir: Path, load: str, rep: int) -> dict[str, Any]:
    """The load's served window: its requests (engine id, plan entry, client tokens), arrivals by step and steps, with
    step indices from the window's first."""
    meta = json.loads((arm_dir / load / f"rep-{rep}" / "meta.json").read_text())
    tag = meta["tag"]
    named = json.loads((arm_dir / "meta.json").read_text()).get("plan")
    tried = [p for p in (Path(named) if named else None, arm_dir / "plan.jsonl", arm_dir.parent / "plan.jsonl") if p]
    plan_file = next((p for p in tried if p.is_file()), None)
    if plan_file is None:
        raise SystemExit(f"no plan at {', '.join(map(str, tried))} (an evicted run's: `research data fetch` puts it back)")
    plan = {r["i"]: r for r in (json.loads(x) for x in plan_file.read_text().splitlines() if x.strip())}
    client = {r["id"]: r for r in (json.loads(x) for x in (arm_dir / load / f"rep-{rep}" / "client.jsonl").read_text().splitlines()
                                   if x.strip())}
    (trace,) = S.step_traces(arm_dir)
    recs = S.trace_records(trace)
    mine = lambda rid: f"-{tag}-" in rid   # noqa: E731
    arrive = [r for r in recs if r["kind"] == "arrive" and mine(r["request"])]
    if not arrive:
        raise SystemExit(f"{trace}: no arrival of load {load} rep {rep} ({tag})")
    first = arrive[0]["before_step"]
    steps = [r for r in recs if r["kind"] == "step" and r["step"] >= first and any(mine(row[0]) for row in r["rows"])]
    foreign = [r["step"] for r in steps if not all(mine(row[0]) for row in r["rows"])]
    if foreign:
        raise SystemExit(f"steps {foreign[:5]} hold rows of another load: the window is not its own")
    last = steps[-1]["step"]
    if [s["step"] for s in steps] != list(range(first, last + 1)):
        raise SystemExit(f"the window's steps {first}..{last} are not contiguous")
    reqs = []
    for a in arrive:
        rid = a["request"]
        http = rid.split("-", 1)[1].rsplit("-", 2)[0]       # cmpl-{X-Request-Id}-0-{8 hex}
        c = client[http]
        reqs.append({"engine_id": rid, "before_step": a["before_step"] - first, "prompt_ids": plan[c["i"]]["prompt_ids"],
                     "max_tokens": a["sampling"].get("max_tokens"), "sampling": a["sampling"], "served": c["output_ids"]})
    for s in steps:
        s["step"] -= first
    return {"header": recs[0], "requests": reqs, "steps": steps, "tag": tag,
            "served_s": steps[-1]["t"] - arrive[0]["t"], "served_t0": arrive[0]["t"]}


#: the engine argument for each entry of the header's config that is passed only where the header holds it: a header holds
#: the KV cache's format and the speculative decoding only where the served engine left vLLM's defaults
HEADER_ARGS = {"long_prefill_token_threshold": "long_prefill_token_threshold", "kv_cache_dtype": "kv_cache_dtype",
               "speculative": "speculative_config"}


def engine(checkpoint: Path, header: dict, eager: bool, tp: int = 1) -> Any:
    """The served engine again, from the trace's header. Its context limit is the header's, launched as the served one was:
    where that launch left the limit to vLLM (the header's ``max_model_len_auto``), at -1 on the header's block count, so
    that vLLM builds the model at the checkpoint's own limit and fits the served limit to the pool, as it did while serving
    (`replay` refuses an engine it fitted another to)."""
    from vllm import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    cfg = header["config"]
    args = EngineArgs(model=str(checkpoint), dtype="bfloat16", seed=0, trust_remote_code=True, enforce_eager=eager,
                      **({"tensor_parallel_size": tp} if tp > 1 else {}),
                      max_model_len=-1 if cfg.get("max_model_len_auto") else cfg["max_model_len"], max_num_seqs=cfg["max_num_seqs"],
                      max_num_batched_tokens=cfg["max_num_batched_tokens"], enable_chunked_prefill=cfg["enable_chunked_prefill"],
                      enable_prefix_caching=cfg["enable_prefix_caching"], async_scheduling=cfg["async_scheduling"],
                      block_size=cfg["block_size"], num_gpu_blocks_override=cfg["num_gpu_blocks"], disable_log_stats=True,
                      **{arg: cfg[key] for key, arg in HEADER_ARGS.items() if key in cfg})
    return LLMEngine.from_engine_args(args)


def params(r: dict, logprobs: bool = False) -> Any:
    from vllm import SamplingParams

    s = r["sampling"]
    return SamplingParams(n=1, temperature=s.get("temperature", 0.0), top_p=s.get("top_p", 1.0), top_k=s.get("top_k", 0),
                          seed=s.get("seed"), max_tokens=r["max_tokens"], ignore_eos=bool(s.get("ignore_eos")), detokenize=False,
                          logprobs=0 if logprobs else None)


def replay(w: dict, checkpoint: Path, eager: bool, logprobs: bool = False, tp: int = 1,
           capture: tuple[list[str], Path] | None = None) -> dict:
    sys.path.insert(0, str(ROOT / "adapters/vllm"))
    from verity_vllm.engine import sampler_capture as SC
    from verity_vllm.engine.hooks import Hooks
    from verity_vllm.engine.step_trace import StepTrace

    t0 = time.monotonic()
    eng = engine(checkpoint, w["header"], eager, tp)
    if capture:
        eng.collective_rpc(SC.install, args=(capture[0],))
    ready = time.monotonic() - t0
    trace = StepTrace().install(eng)
    limit, served = trace.records[0]["config"].get("max_model_len"), w["header"]["config"].get("max_model_len")
    if limit != served:
        raise SystemExit(f"the replay's engine has a context limit of {limit}, the served engine's was {served}: not the "
                         f"engine that served this window")
    pending = sorted(w["requests"], key=lambda r: r["before_step"])
    tokens: dict[str, list[int]] = {}
    lps: dict[str, list[float]] = {}

    def add_due() -> None:
        while pending and pending[0]["before_step"] == trace.scheduled:
            r = pending.pop(0)
            eng.add_request(r["engine_id"], {"prompt_token_ids": r["prompt_ids"]}, params(r, logprobs))
        if pending and pending[0]["before_step"] < trace.scheduled:
            raise SystemExit(f"the replay passed step {pending[0]['before_step']} before {pending[0]['engine_id']} arrived")

    def schedule(orig):
        def wrapped(*a, **kw):
            add_due()
            return orig(*a, **kw)
        return wrapped

    hooks = Hooks(tag="serve-tax-replay")
    hooks.wrap(trace.sched, "schedule", schedule)
    start = time.monotonic()
    while pending or eng.has_unfinished_requests():
        if not eng.has_unfinished_requests():
            add_due()
            if not eng.has_unfinished_requests():
                raise SystemExit(f"the replay is idle at step {trace.scheduled} with {pending[0]['engine_id']} due at "
                                 f"step {pending[0]['before_step']}")
        for o in eng.step():
            if o.finished:
                c = o.outputs[0]
                tokens[o.request_id] = [int(t) for t in c.token_ids]
                if logprobs:
                    lps[o.request_id] = [float(c.logprobs[i][t].logprob) for i, t in enumerate(c.token_ids)]
    wall = time.monotonic() - start
    takes = eng.collective_rpc(SC.take, args=(str(capture[1]),)) if capture else None
    hooks.uninstall()
    trace.uninstall()
    return {"ready_s": ready, "wall_s": wall, "steps": [r for r in trace.records if r["kind"] == "step"], "tokens": tokens,
            "logprobs": lps, "takes": takes}


def emitted_at(steps: list[dict]) -> dict[str, list[int]]:
    """Each request's step of each output position, from the steps' emitted counts."""
    at: dict[str, list[int]] = {}
    for s in steps:
        for rid, n in (s.get("emitted") or {}).items():
            at.setdefault(rid, []).extend([s["step"]] * int(n))
    return at


def logprob_diff(want: dict[str, list[float]], got: dict[str, list[float]], steps: list[dict]) -> dict:
    """Where two replays' sampled-token logprobs first differ, bit for bit: per request its first position and that
    position's step, and the earliest step over all requests. A logprob is the sampled token's logit less the row's
    log-sum-exp, so a difference in any logit of the row almost surely shows."""
    at = emitted_at(steps)
    out = []
    for rid, a in want.items():
        b = got.get(rid, [])
        i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if i is None and len(a) == len(b):
            continue
        i = min(len(a), len(b)) if i is None else i
        pos = at.get(rid, [])
        out.append({"request": rid, "first_position": i, "step": pos[i] if i < len(pos) else None,
                    "positions": sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b)),
                    "want": a[i] if i < len(a) else None, "got": b[i] if i < len(b) else None})
    out.sort(key=lambda d: (d["step"] is None, d["step"]))
    return {"requests_logprobs_differing": len(out), "first_step": out[0]["step"] if out else None,
            "requests_differing": out[:DIFFERING_MAX]}


def read_logprobs(replay_dir: Path) -> dict[str, list[float]]:
    with gzip.open(replay_dir / "replay_logprobs.jsonl.gz", "rt") as f:
        return {r["request"]: r["logprobs"] for r in map(json.loads, f) if r}


def write_logprobs(replay_dir: Path, lps: dict[str, list[float]]) -> None:
    with gzip.open(replay_dir / "replay_logprobs.jsonl.gz", "wt") as f:
        for rid in sorted(lps):
            f.write(json.dumps({"request": rid, "logprobs": lps[rid]}, separators=(",", ":")) + "\n")


#: the differing requests replay.json lists, each with its lengths and first differing position
DIFFERING_MAX = 100


def token_diff(want: dict[str, list[int]], got: dict[str, list[int]]) -> list[dict]:
    """Each request of ``want`` whose tokens in ``got`` differ: both lengths, the first position that differs and how many do."""
    out = []
    for rid, a in want.items():
        b = got.get(rid, [])
        if a != b:
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            out.append({"request": rid, "tokens": len(a), "tokens_replayed": len(b), "first_position": first,
                        "positions": sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))})
    return out


def compare(w: dict, got: dict) -> dict:
    served = {s["step"]: s["rows"] for s in w["steps"]}
    mine = {s["step"]: s["rows"] for s in got["steps"]}
    steps = sorted(set(served) | set(mine))
    differ = [s for s in steps if served.get(s) != mine.get(s)]
    rows = sum(len(set(map(tuple, served.get(s, ()))) ^ set(map(tuple, mine.get(s, ())))) for s in differ)
    reqs = {r["engine_id"]: r["served"] for r in w["requests"]}
    tok_differ = token_diff(reqs, got["tokens"])
    return {"steps_served": len(served), "steps_replayed": len(mine), "steps_differing": len(differ),
            "rows_served": sum(map(len, served.values())), "rows_differing": rows, "first_differing_step": differ[0] if differ else None,
            "requests": len(reqs), "requests_tokens_differing": len(tok_differ),
            "token_positions_differing": sum(d["positions"] for d in tok_differ),
            "tokens_served": sum(map(len, reqs.values())), "bit_exact": not differ and not tok_differ,
            "requests_differing": tok_differ[:DIFFERING_MAX]}


def read_tokens(replay_dir: Path) -> dict[str, list[int]]:
    with gzip.open(replay_dir / "replay_tokens.jsonl.gz", "rt") as f:
        return {r["request"]: r["tokens"] for r in map(json.loads, f) if r}


def write_tokens(replay_dir: Path, tokens: dict[str, list[int]]) -> None:
    with gzip.open(replay_dir / "replay_tokens.jsonl.gz", "wt") as f:
        for rid in sorted(tokens):
            f.write(json.dumps({"request": rid, "tokens": tokens[rid]}, separators=(",", ":")) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm-dir", type=Path, required=True)
    ap.add_argument("--load", required=True)
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--against", type=Path, help="an earlier replay's --out, whose tokens this one's are compared with too")
    ap.add_argument("--logprobs", action="store_true",
                    help="record each sampled token's logprob too, and compare them with --against's, step by step")
    ap.add_argument("--tp", type=int, default=1, help="the tensor-parallel width the window was served at")
    ap.add_argument("--capture", help="engine ids: take their logits rows and ids at the sampler (`sampler_capture`) into "
                                      "OUT/capture, each a span of --capture-template's units")
    ap.add_argument("--capture-template", default="TokenSelect_v4{V=154880}")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if sys.flags.hash_randomization:
        raise SystemExit("run with PYTHONHASHSEED=0, as the served engine did: the scheduler iterates sets of request ids")
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_DISABLE_REQUEST_ID_RANDOMIZATION="1")
    meta = json.loads((a.arm_dir / "meta.json").read_text())
    checkpoint = a.checkpoint or Path(meta["checkpoint"])
    w = window(a.arm_dir, a.load, a.rep)
    capture = (a.capture.split(","), a.out / "capture" / "ranks") if a.capture else None
    got = replay(w, checkpoint, a.eager, a.logprobs, a.tp, capture)
    out = {"schema": "verity-bench/serve-tax/replay/v0", "arm_dir": str(a.arm_dir), "served_run_id": meta.get("run_id"),
           "run_id": SHARED.RESEARCH_RUN_ID, "load": a.load, "rep": a.rep, "mode": "eager" if a.eager else "compiled",
           "served_mode": "eager" if "--enforce-eager" in meta["server_argv"] else "compiled", "requests": len(w["requests"]),
           "served_window_s": w["served_s"], "replay_wall_s": got["wall_s"], "replay_over_served": got["wall_s"] / w["served_s"],
           "engine_ready_s": got["ready_s"], **compare(w, got)}
    if a.against:
        d = token_diff(read_tokens(a.against), got["tokens"])
        out["against"] = {"replay": str(a.against), "requests_tokens_differing": len(d), "requests_differing": d[:DIFFERING_MAX]}
        if a.logprobs and (a.against / "replay_logprobs.jsonl.gz").is_file():
            out["against"]["logprobs"] = logprob_diff(read_logprobs(a.against), got["logprobs"], got["steps"])
    if capture:
        from verity_vllm.engine import sampler_capture as SC
        spans = SC.spans(got["takes"], got["steps"])
        SC.write(a.out / "capture", a.capture_template, spans)
        served = {r["engine_id"]: r["served"] for r in w["requests"]}
        out["capture"] = {rid: {"tokens": len(s["tokens"]), "as_served": s["tokens"] == served[rid], "program": s["program"]}
                          for rid, s in spans.items()}
        out["capture_missing"] = sorted(set(capture[0]) - set(spans))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "replay.json").write_text(json.dumps(out, indent=1))
    write_tokens(a.out, got["tokens"])
    if a.logprobs:
        write_logprobs(a.out, got["logprobs"])
    with open(a.out / "replay_steps.jsonl", "w") as f:
        for s in got["steps"]:
            f.write(json.dumps(s, separators=(",", ":")) + "\n")
    print(json.dumps({**{k: out[k] for k in ("mode", "requests", "steps_served", "steps_differing", "rows_differing",
                                             "requests_tokens_differing", "bit_exact", "served_window_s", "replay_wall_s")},
                      **({"against_requests_differing": out["against"]["requests_tokens_differing"]} if a.against else {})}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
