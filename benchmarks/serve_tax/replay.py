"""Replay one served window of a ``record`` arm (`serve_tax`) under its recorded schedule, and say how close it came.

    replay.py --arm-dir OUT/record --load NAME [--rep 1] [--checkpoint DIR] [--eager] --out DIR

The window is the load's requests: its step trace's steps from the first one's arrival to the last one's finish, all of them
the load's own (each load starts from an empty prefix cache and ends drained). The replay builds the served engine again
in this process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) from the trace's header (its batch caps, chunked prefill, prefix
caching, async scheduling, block size and block count), compiled unless ``--eager``, and adds each request, under its served
engine id with its served prompt and sampling, right before the scheduled step it arrived before in the trace (a wrapper of
the scheduler's ``schedule``); the scheduler decides the rest as it did while serving. Its own step trace (`StepTrace`) is
then compared with the served one, step by step: the rows (request, first position, tokens) that differ and the first step
that does, and each request's sampled tokens against the ones the client received. replay.json has the counts and the
replay's wall time beside the served window's (first arrival to last step).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def window(arm_dir: Path, load: str, rep: int) -> dict[str, Any]:
    """The load's served window: its requests (engine id, plan entry, client tokens), arrivals by step and steps, with
    step indices from the window's first."""
    meta = json.loads((arm_dir / load / f"rep-{rep}" / "meta.json").read_text())
    tag = meta["tag"]
    plan_file = Path(json.loads((arm_dir / "meta.json").read_text()).get("plan") or arm_dir / "plan.jsonl")
    plan = {r["i"]: r for r in (json.loads(x) for x in plan_file.read_text().splitlines() if x.strip())}
    client = {r["id"]: r for r in (json.loads(x) for x in (arm_dir / load / f"rep-{rep}" / "client.jsonl").read_text().splitlines()
                                   if x.strip())}
    (trace,) = sorted((arm_dir / "server").glob("step_trace.*.jsonl"))
    recs = [json.loads(x) for x in trace.read_text().splitlines() if x.strip()]
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


def engine(checkpoint: Path, header: dict, eager: bool) -> Any:
    from vllm import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    cfg = header["config"]
    args = EngineArgs(model=str(checkpoint), dtype="bfloat16", seed=0, trust_remote_code=True, enforce_eager=eager,
                      max_model_len=cfg["max_model_len"], max_num_seqs=cfg["max_num_seqs"],
                      max_num_batched_tokens=cfg["max_num_batched_tokens"], enable_chunked_prefill=cfg["enable_chunked_prefill"],
                      enable_prefix_caching=cfg["enable_prefix_caching"], async_scheduling=cfg["async_scheduling"],
                      block_size=cfg["block_size"], num_gpu_blocks_override=cfg["num_gpu_blocks"], disable_log_stats=True)
    return LLMEngine.from_engine_args(args)


def params(r: dict) -> Any:
    from vllm import SamplingParams

    s = r["sampling"]
    return SamplingParams(n=1, temperature=s.get("temperature", 0.0), top_p=s.get("top_p", 1.0), top_k=s.get("top_k", 0),
                          seed=s.get("seed"), max_tokens=r["max_tokens"], ignore_eos=bool(s.get("ignore_eos")), detokenize=False)


def replay(w: dict, checkpoint: Path, eager: bool) -> dict:
    sys.path.insert(0, str(ROOT / "integrations/vllm"))
    from verity_vllm.engine.hooks import Hooks
    from verity_vllm.engine.step_trace import StepTrace

    t0 = time.monotonic()
    eng = engine(checkpoint, w["header"], eager)
    ready = time.monotonic() - t0
    trace = StepTrace().install(eng)
    pending = sorted(w["requests"], key=lambda r: r["before_step"])
    tokens: dict[str, list[int]] = {}

    def add_due() -> None:
        while pending and pending[0]["before_step"] == trace.scheduled:
            r = pending.pop(0)
            eng.add_request(r["engine_id"], {"prompt_token_ids": r["prompt_ids"]}, params(r))
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
                tokens[o.request_id] = [int(t) for t in o.outputs[0].token_ids]
    wall = time.monotonic() - start
    hooks.uninstall()
    trace.uninstall()
    return {"ready_s": ready, "wall_s": wall, "steps": [r for r in trace.records if r["kind"] == "step"], "tokens": tokens}


def compare(w: dict, got: dict) -> dict:
    served = {s["step"]: s["rows"] for s in w["steps"]}
    mine = {s["step"]: s["rows"] for s in got["steps"]}
    steps = sorted(set(served) | set(mine))
    differ = [s for s in steps if served.get(s) != mine.get(s)]
    rows = sum(len(set(map(tuple, served.get(s, ()))) ^ set(map(tuple, mine.get(s, ())))) for s in differ)
    reqs = {r["engine_id"]: r["served"] for r in w["requests"]}
    tok_differ = [rid for rid, t in reqs.items() if got["tokens"].get(rid) != t]
    positions = sum(sum(a != b for a, b in zip(got["tokens"].get(rid, []), reqs[rid])) +
                    abs(len(got["tokens"].get(rid, [])) - len(reqs[rid])) for rid in tok_differ)
    return {"steps_served": len(served), "steps_replayed": len(mine), "steps_differing": len(differ),
            "rows_served": sum(map(len, served.values())), "rows_differing": rows, "first_differing_step": differ[0] if differ else None,
            "requests": len(reqs), "requests_tokens_differing": len(tok_differ), "token_positions_differing": positions,
            "tokens_served": sum(map(len, reqs.values())), "bit_exact": not differ and not tok_differ,
            "first_differing_request": tok_differ[0] if tok_differ else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm-dir", type=Path, required=True)
    ap.add_argument("--load", required=True)
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if sys.flags.hash_randomization:
        raise SystemExit("run with PYTHONHASHSEED=0, as the served engine did: the scheduler iterates sets of request ids")
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_DISABLE_REQUEST_ID_RANDOMIZATION="1")
    meta = json.loads((a.arm_dir / "meta.json").read_text())
    checkpoint = a.checkpoint or Path(meta["checkpoint"])
    w = window(a.arm_dir, a.load, a.rep)
    got = replay(w, checkpoint, a.eager)
    out = {"schema": "verity-bench/serve-tax/replay/v0", "arm_dir": str(a.arm_dir), "served_run_id": meta.get("run_id"),
           "run_id": os.environ.get("RESEARCH_RUN_ID"), "load": a.load, "rep": a.rep, "mode": "eager" if a.eager else "compiled",
           "served_mode": "eager" if "--enforce-eager" in meta["server_argv"] else "compiled", "requests": len(w["requests"]),
           "served_window_s": w["served_s"], "replay_wall_s": got["wall_s"], "replay_over_served": got["wall_s"] / w["served_s"],
           "engine_ready_s": got["ready_s"], **compare(w, got)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "replay.json").write_text(json.dumps(out, indent=1))
    with open(a.out / "replay_steps.jsonl", "w") as f:
        for s in got["steps"]:
            f.write(json.dumps(s, separators=(",", ":")) + "\n")
    print(json.dumps({k: out[k] for k in ("mode", "requests", "steps_served", "steps_differing", "rows_differing",
                                          "requests_tokens_differing", "bit_exact", "served_window_s", "replay_wall_s")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
