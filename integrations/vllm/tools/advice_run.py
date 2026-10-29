"""A run under varied traffic with the scheduler's step trace on, and its advice: the input `verity-vllm advice` reads, from a GPU.

One in-process engine (`verity_vllm.LLM`, the pinned model) serves `--requests` requests whose prompts run from 16 to
`--max-prompt` tokens (log-uniform), whose `max_tokens` vary and stop at the model's EOS before it, a share of them sampled at
temperature 0.8 (half of those with a client seed, the rest under the served default, `engine.served_seed.DEFAULT_SEED`),
arriving at seeded engine steps while others decode, with `--gaps` idle gaps (each a window boundary).  A small
`--max-batched-tokens` chunks the long prompts; `--kv-bytes` shrinks the KV cache until the scheduler preempts.
`engine.step_trace` records every step.  It writes, under `--out`:

    step_trace.jsonl   the scheduler's step trace
    requests.jsonl     each request's id, the `user` it was sent as, its prompt and output token ids and finish: the engine's own
                       outputs, standing in for the Commit record until a committing server writes one (no committer is installed)
    run.json           the engine record, the traffic's parameters and the run's counts
    advice/            `verity-vllm advice` on the above

    cd integrations/vllm; PYTHONPATH=.:../..:../../tools/research/src /workspace/jobs/venv312/bin/python tools/advice_run.py \\
        --out DIR [--model REPO --revision SHA] [--requests 48] [--seed 0] [--execution bi-eager]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

FILLER = ("The river runs past the old mill, where the miller keeps a ledger of every sack of grain that comes in and every sack of "
          "flour that goes out. In spring the water rises and the wheel turns faster; in late summer it slows, and the miller "
          "mends the gears. Travellers stop to ask the way to the city, and he tells them to follow the road along the water. ")
QUESTIONS = ["Summarize the text above in one sentence.", "What does the miller keep? Answer in a few words.",
             "Answer with one word: what turns the wheel?", "Name three things mentioned above.", "Is the river fast in summer? Yes or no.",
             "Write a short poem about the mill.", "What is 17 + 25? Reply with the number only.", "List the seasons mentioned.",
             "Explain in two sentences why the wheel slows.", "Say hello.", "Translate 'the mill' into French.",
             "Continue the story for a few sentences."]


def traffic(tok, n: int, rng: random.Random, max_prompt: int, gaps: int, caps: list[int]) -> list[SimpleNamespace]:
    filler = tok.encode(FILLER * (max_prompt // 40 + 2), add_special_tokens=False)
    gap_at = set(rng.sample(range(n // 4, n), gaps)) if gaps else set()
    out, at = [], 0
    for i in range(n):
        want = int(math.exp(rng.uniform(math.log(16), math.log(max_prompt))))
        q = rng.choice(QUESTIONS)
        body = tok.decode(filler[:max(want - 24, 0)]) + "\n\n" + q
        if getattr(tok, "chat_template", None):
            # enable_thinking is Qwen3's switch (other templates ignore it): off, answers end at EOS within max_tokens
            text = tok.apply_chat_template([{"role": "user", "content": body}], add_generation_prompt=True, tokenize=False,
                                           enable_thinking=False)
            ids = tok.encode(text, add_special_tokens=False)
        else:
            ids = tok.encode(body + "\nAnswer:")
        stoch = rng.random() < 0.3
        at += int(rng.expovariate(1 / 3))
        out.append(SimpleNamespace(request_id=f"r{i:03d}", user=f"user-{i:03d}", prompt_token_ids=list(ids), arrive_step=at,
                                   after_idle=i in gap_at,
                                   max_tokens=rng.choice(caps), temperature=0.8 if stoch else 0.0,
                                   seed=rng.randrange(2**31) if stoch and rng.random() < 0.5 else None))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    ap.add_argument("--requests", type=int, default=48)
    ap.add_argument("--max-prompt", type=int, default=1024)
    ap.add_argument("--gaps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seqs", type=int, default=8)
    ap.add_argument("--max-batched-tokens", type=int, default=256)
    ap.add_argument("--kv-bytes", type=int, default=None)
    ap.add_argument("--execution", default=None, help="the engine's execution label (engine.served.EXECUTIONS), such as bi-eager")
    ap.add_argument("--max-tokens", default="8,16,32,64,128,200", help="the max_tokens values requests draw from")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from verity_vllm import LLM
    from verity_vllm.engine.engine_driver import external_id
    from verity_vllm.engine.served_seed import DEFAULT_SEED
    from verity_vllm.engine.step_trace import StepTrace

    kw = {"max_num_seqs": a.max_seqs, "max_num_batched_tokens": a.max_batched_tokens}
    if a.kv_bytes:
        kw["kv_cache_memory_bytes"] = a.kv_bytes
    if a.execution:
        kw["execution"] = a.execution
    llm = LLM(a.model, revision=a.revision, **kw)
    from vllm import SamplingParams
    tok = llm.vllm.get_tokenizer()
    reqs = traffic(tok, a.requests, random.Random(a.seed), a.max_prompt, a.gaps, [int(x) for x in a.max_tokens.split(",")])
    for r in reqs:
        if r.temperature and r.seed is None:
            r.seed = DEFAULT_SEED
    trace = StepTrace(out / "step_trace.jsonl", request_id=external_id, clock=time.time).install(llm)
    done: dict[str, object] = {}

    def on_step(outs):
        for o in outs:
            if o.finished:
                done[o.request_id] = o

    params = lambda r: SamplingParams(temperature=r.temperature, top_p=0.95 if r.temperature else 1.0, seed=r.seed,  # noqa: E731
                                      max_tokens=r.max_tokens, ignore_eos=False)
    eng = llm.vllm.llm_engine
    pending = list(reqs)
    shift = 0
    t0 = time.perf_counter()
    try:
        while pending or eng.has_unfinished_requests():
            # a request marked after_idle waits until the engine drains, and the arrivals after it keep their spacing from it
            while pending and (pending[0].after_idle and not eng.has_unfinished_requests()
                               or not pending[0].after_idle and pending[0].arrive_step + shift <= trace.scheduled):
                r = pending.pop(0)
                if r.after_idle:
                    shift = trace.scheduled - r.arrive_step
                eng.add_request(request_id=r.request_id, prompt={"prompt_token_ids": r.prompt_token_ids}, params=params(r))
            if not eng.has_unfinished_requests():
                if pending and not pending[0].after_idle:
                    shift = trace.scheduled - pending[0].arrive_step      # an idle engine takes the next arrival now
                continue
            on_step(eng.step())
    finally:
        trace.uninstall()
    timing = {"wall_s": time.perf_counter() - t0}
    with (out / "requests.jsonl").open("w") as f:
        for r in reqs:
            o = done[r.request_id]
            c = o.outputs[0]
            f.write(json.dumps({"request": r.request_id, "user": r.user, "prompt_ids": list(o.prompt_token_ids), "output_ids": list(c.token_ids),
                                "finish_reason": str(c.finish_reason), "stop_reason": c.stop_reason, "arrive_step": r.arrive_step,
                                "max_tokens": r.max_tokens, "temperature": r.temperature, "seed": r.seed}) + "\n")
    finishes = {}
    for o in done.values():
        finishes[str(o.outputs[0].finish_reason)] = finishes.get(str(o.outputs[0].finish_reason), 0) + 1
    from verity_vllm.program.frontend.target_profile import TargetProfile
    (out / "run.json").write_text(json.dumps({
        "model": a.model, "revision": a.revision, "engine_kwargs": kw, "engine_record": llm.engine_record, "args": vars(a),
        "target": TargetProfile.from_device().to_json(), "execution": a.execution, "default_seed": DEFAULT_SEED,
        "commit_record": "requests.jsonl: the engine's own outputs; no committer installed", "finishes": finishes,
        "steps_traced": trace.executed, "wall_s": timing.get("wall_s") if isinstance(timing, dict) else None}, indent=1, default=str) + "\n")
    print(f"[advice-run] {len(done)} requests finished {finishes}, {trace.executed} steps traced -> {out}", flush=True)
    from verity_vllm.pipeline import advice
    return advice.main(advice.Options(run=str(out)))


if __name__ == "__main__":
    sys.exit(main())
