"""A chat workload from ShareGPT served by vLLM on one GPU (in process), with every KV block's allocation, prefix hit
and release logged (`lifetimes.Recorder`) and each step's step-model quantities, for the KV-credit frontier.

Workloads (`--workload`), each from ShareGPT_V3_unfiltered_cleaned_split.json (`--sharegpt`), conversations drawn at
random with `--seed`, each message in Llama-3.1's chat format (header tokens around it, no system prompt):
  single  vLLM's ShareGPT benchmark: a conversation's first user message is the prompt and its first reply's length
          the output length; pairs with prompt or output under 4 tokens, a prompt over 1024 or a total over 2048
          are skipped. Requests arrive as a Poisson process at the phase's rate (a second).
  multi   sessions: a conversation's turns in order, each turn's prompt the previous prompt, the tokens generated
          for it and the next user message, its output length the next reply's (at least 4, at most 2048), so the
          prefix cache can hit the whole history. Sessions start as a Poisson process at the phase's rate; a turn
          is sent a think time after the previous one finished (exponential, mean `--think-s`); a session ends when
          its turns run out or the next would pass `--max-model-len`.
Output lengths are forced (`ignore_eos`, `max_tokens` the reply's length).

`--phases RATE:SECONDS,...` runs the phases in order on one engine; at each phase's end the unfinished requests are
aborted, the pending turns dropped and the prefix cache emptied. A step is timed from its `schedule` to its
`update_from_output` (sync scheduling), and records T (token rows), X (sampled rows), Q (the token rows' keys), K (the
scheduled requests' keys after the step), n (requests), n0 (requests' first steps), the running and waiting requests
and the pool's free blocks, on the clock of `lifetimes.Recorder` (seconds from the first phase's start). A request's
blocks carry its origin (`lifetimes.Recorder.allocating`): its submit time, lowered by the prefix hits it takes, so a
turn that reuses cached history inherits the history's origin while a re-prefilled one starts fresh.

Writes to `--out`: events.npz (the recorder's arrays and the steps), serve.json (arguments, engine facts, each
phase's throughput, latency and occupancy) and serve.log.
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import random
import signal
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import lifetimes  # noqa: E402

LOGF = None


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + "\n")
        LOGF.flush()


def conversations(path: Path) -> list[list[str]]:
    """Each conversation's messages from its first human message on, alternating human and gpt, cut at the first
    break in the alternation."""
    out = []
    for c in json.loads(path.read_text()):
        msgs, want = [], "human"
        for m in c.get("conversations", []):
            if m.get("from") != want:
                if msgs:
                    break
                continue
            msgs.append(m.get("value", ""))
            want = "gpt" if want == "human" else "human"
        if len(msgs) >= 2:
            out.append(msgs)
    return out


class Chat:
    """Llama-3.1's chat format over a tokenizer."""

    def __init__(self, tok) -> None:
        self.tok = tok
        enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
        self.bos = enc("<|begin_of_text|>")
        self.user = enc("<|start_header_id|>user<|end_header_id|>\n\n")
        self.asst = enc("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        self.eot = enc("<|eot_id|>")
        self.enc = enc

    def turn(self, text: str, cap: int) -> list[int]:
        return self.user + self.enc(text)[:cap] + self.asst


def main() -> int:
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sharegpt", type=Path, required=True)
    ap.add_argument("--model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--quantization", default=None, help="vLLM's quantization (default: the checkpoint's own; verity_int8 is registered first)")
    ap.add_argument("--workload", choices=("single", "multi"), default="single")
    ap.add_argument("--phases", required=True, help="RATE:SECONDS,... (requests or sessions a second)")
    ap.add_argument("--prefix-caching", choices=("on", "off"), default="off")
    ap.add_argument("--think-s", type=float, default=30.0)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--mem", type=float, default=0.90)
    ap.add_argument("--max-seqs", type=int, default=256)
    ap.add_argument("--max-batched-tokens", type=int, default=8192)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    phases = [(float(r), float(s)) for r, s in (p.split(":") for p in a.phases.split(","))]
    a.out.mkdir(parents=True, exist_ok=True)
    LOGF = (a.out / "serve.log").open("a")

    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from verity_vllm.engine.engine_profile import get_scheduler

    t = time.perf_counter()
    convs = conversations(a.sharegpt)
    log(f"ShareGPT: {len(convs)} conversations from {a.sharegpt.name} in {time.perf_counter() - t:.1f} s")
    kw = {"model": a.model, "max_num_seqs": a.max_seqs, "enable_prefix_caching": a.prefix_caching == "on",
          "seed": 0, "max_model_len": a.max_model_len, "max_num_batched_tokens": a.max_batched_tokens,
          "gpu_memory_utilization": a.mem, "tensor_parallel_size": a.tp, "async_scheduling": False}
    if a.quantization:
        kw["quantization"] = a.quantization
    qc = json.loads((Path(a.model) / "config.json").read_text()).get("quantization_config") or {}
    if (a.quantization or qc.get("quant_method")) == "verity_int8":
        from verity_vllm.engine.int8_vllm import register
        register()
    t = time.perf_counter()
    llm = LLM(**kw)
    log(f"engine up in {time.perf_counter() - t:.1f} s")
    eng = llm.llm_engine
    sched = get_scheduler(llm)
    pool = sched.kv_cache_manager.block_pool
    cache = eng.vllm_config.cache_config
    hf = eng.vllm_config.model_config.hf_config
    heads = getattr(hf, "num_key_value_heads", hf.num_attention_heads)
    head_dim = getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads
    kv_dtype_bytes = 1 if "fp8" in str(cache.cache_dtype) else 2
    token_bytes = hf.num_hidden_layers * 2 * heads * head_dim * kv_dtype_bytes
    block_bytes = int(cache.block_size) * token_bytes
    chat = Chat(llm.get_tokenizer())
    res = {"format": "verity-kv-lifetime/serve/v0", "args": {k: str(v) for k, v in vars(a).items()},
           "vllm": vllm.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
           "kv_blocks": int(cache.num_gpu_blocks), "block_size": int(cache.block_size),
           "kv_bytes_per_token": token_bytes, "block_bytes": block_bytes, "cache_dtype": str(cache.cache_dtype),
           "pool_bytes": int(cache.num_gpu_blocks) * block_bytes, "phases": []}
    log(f"pool {res['kv_blocks']} blocks of {res['block_size']} tokens, {block_bytes} B a block (all ranks), "
        f"{res['pool_bytes'] / 1e9:.1f} GB")

    warm = SamplingParams(max_tokens=8, ignore_eos=True)
    llm.generate([TokensPrompt(prompt_token_ids=[1000 + i for i in range(64)])], warm, use_tqdm=False)
    llm.reset_prefix_cache()

    t_base = time.perf_counter()
    clock = lambda: time.perf_counter() - t_base  # noqa: E731
    rec = lifetimes.Recorder(clock)
    rec.install(pool)
    steps: list[tuple] = []
    cur: dict = {}
    seen: set[str] = set()
    s_orig, u_orig = sched.schedule, sched.update_from_output

    def schedule(*args, **kwargs):
        t0 = clock()
        so = s_orig(*args, **kwargs)
        T = X = Q = K = n = n0 = 0
        for rid, s in so.num_scheduled_tokens.items():
            r = sched.requests.get(rid)
            if r is None:
                continue
            c = int(r.num_computed_tokens) - s
            T += s
            K += c + s
            Q += s * c + s * (s + 1) // 2
            n += 1
            if rid not in seen:
                seen.add(rid)
                n0 += 1
            X += int(c + s >= r.num_prompt_tokens)
        cur["row"] = [t0, 0.0, T, X, Q, K, n, n0, len(sched.running), len(sched.waiting)]
        return so

    def update_from_output(*args, **kwargs):
        out = u_orig(*args, **kwargs)
        row = cur.pop("row", None)
        if row is not None and row[2] > 0:
            row[1] = clock()
            steps.append((*row, pool.get_num_free_blocks()))
        return out

    sched.schedule, sched.update_from_output = schedule, update_from_output
    origin_by_rid: dict[str, float] = {}
    alloc0 = sched.kv_cache_manager.allocate_slots

    def allocate_slots(request, *args, **kwargs):
        rid = request.request_id
        rec.allocating(origin_by_rid.get(rid, clock()))
        try:
            return alloc0(request, *args, **kwargs)
        finally:
            origin_by_rid[rid] = rec.origin
            rec.allocating(None)

    sched.kv_cache_manager.allocate_slots = allocate_slots

    stop = [False]  # a SIGTERM (a preemption) ends the current phase early and writes what was recorded
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    rng = random.Random(a.seed)
    order = list(range(len(convs)))
    rng.shuffle(order)
    nxt_conv = [0]

    def draw_single():
        while True:
            msgs = convs[order[nxt_conv[0] % len(order)]]
            nxt_conv[0] += 1
            p = chat.bos + chat.turn(msgs[0], 1 << 30)
            o = len(chat.enc(msgs[1]))
            if 4 <= len(p) <= 1024 and o >= 4 and len(p) + o <= 2048:
                return p, o

    def draw_session():
        msgs = convs[order[nxt_conv[0] % len(order)]]
        nxt_conv[0] += 1
        return {"msgs": msgs, "turn": 0, "history": list(chat.bos)}

    def turn_request(sess):
        i = sess["turn"]
        if 2 * i + 1 >= len(sess["msgs"]):
            return None
        p = sess["history"] + chat.turn(sess["msgs"][2 * i], 2048)
        o = min(2048, max(4, len(chat.enc(sess["msgs"][2 * i + 1]))))
        if len(p) + o > a.max_model_len:
            return None
        return p, o

    for pi, (rate, seconds) in enumerate(phases):
        p_start = clock()
        arrivals: list = []  # (time, seq, kind, payload)
        seq = [0]

        def push(tt, kind, payload):
            heapq.heappush(arrivals, (tt, seq[0], kind, payload))
            seq[0] += 1

        tt = p_start
        prng = random.Random(a.seed * 1000 + pi)
        while True:
            tt += prng.expovariate(rate)
            if tt >= p_start + seconds:
                break
            push(tt, "new", None)
        live: dict[str, dict] = {}
        sessions: dict[str, dict] = {}
        ttft, tpot, e2e = [], [], []
        out_tok = in_tok = done_n = 0
        idle = 0.0
        rid_n = [0]

        def submit(prompt, olen, sess=None):
            rid = f"p{pi}-{rid_n[0]}"
            rid_n[0] += 1
            live[rid] = {"t": clock(), "first": None, "olen": olen, "plen": len(prompt)}
            origin_by_rid[rid] = live[rid]["t"]
            eng.add_request(rid, TokensPrompt(prompt_token_ids=prompt),
                            SamplingParams(max_tokens=olen, ignore_eos=True, temperature=0.0))
            if sess is not None:
                sessions[rid] = sess

        while True:
            now = clock()
            if now >= p_start + seconds or stop[0]:
                break
            while arrivals and arrivals[0][0] <= now:
                _, _, kind, payload = heapq.heappop(arrivals)
                if a.workload == "single":
                    submit(*draw_single())
                else:
                    sess = payload if kind == "turn" else draw_session()
                    req = turn_request(sess)
                    if req is not None:
                        sess["prompt"] = req[0]
                        submit(*req, sess=sess)
            if eng.has_unfinished_requests():
                outs = eng.step()
                t1 = clock()
                for o in outs:
                    st = live.get(o.request_id)
                    if st is None:
                        continue
                    if st["first"] is None and o.outputs and len(o.outputs[0].token_ids):
                        st["first"] = t1
                        ttft.append(t1 - st["t"])
                    if o.finished:
                        live.pop(o.request_id)
                        origin_by_rid.pop(o.request_id, None)
                        done_n += 1
                        gen = list(o.outputs[0].token_ids)
                        out_tok += len(gen)
                        in_tok += st["plen"]
                        e2e.append(t1 - st["t"])
                        if st["first"] is not None and st["olen"] > 1:
                            tpot.append((t1 - st["first"]) / (st["olen"] - 1))
                        sess = sessions.pop(o.request_id, None)
                        if sess is not None:
                            sess["history"] = sess["prompt"] + gen + chat.eot
                            sess["turn"] += 1
                            push(t1 + prng.expovariate(1.0 / a.think_s), "turn", sess)
            else:
                i0 = clock()
                wait = (arrivals[0][0] - i0) if arrivals else (p_start + seconds - i0)
                time.sleep(max(0.0, min(0.05, wait, p_start + seconds - i0)))
                idle += clock() - i0
        p_end = clock()
        unfinished = list(live)
        if unfinished:
            eng.abort_request(unfinished)
        while eng.has_unfinished_requests():
            eng.step()
        if a.prefix_caching == "on":
            llm.reset_prefix_cache()
        ph_steps = [s for s in steps if s[0] >= p_start and s[1] <= p_end]
        q = lambda xs, f: sorted(xs)[int(f * (len(xs) - 1))] if xs else None  # noqa: E731
        busy = sum(s[1] - s[0] for s in ph_steps)
        ph = {"rate": rate, "seconds": seconds, "t0": p_start, "t1": p_end, "stopped": stop[0], "requests_done": done_n,
              "aborted": len(unfinished), "input_tokens_per_s": in_tok / (p_end - p_start),
              "output_tokens_per_s": out_tok / (p_end - p_start),
              "token_rows_per_s": sum(s[2] for s in ph_steps) / (p_end - p_start),
              "ttft_s": {"p50": q(ttft, 0.5), "p90": q(ttft, 0.9), "p99": q(ttft, 0.99)},
              "tpot_ms": {"p50": (q(tpot, 0.5) or 0) * 1e3, "p90": (q(tpot, 0.9) or 0) * 1e3},
              "e2e_s": {"p50": q(e2e, 0.5), "p90": q(e2e, 0.9)}, "idle_s": idle, "busy_s": busy,
              "steps": len(ph_steps),
              "running_mean": statistics.fmean(s[8] for s in ph_steps) if ph_steps else 0,
              "waiting_mean": statistics.fmean(s[9] for s in ph_steps) if ph_steps else 0,
              "used_blocks_mean": (res["kv_blocks"] - statistics.fmean(s[10] for s in ph_steps)) if ph_steps else 0,
              "sessions_open_at_end": len({id(s) for s in sessions.values()})}
        res["phases"].append(ph)
        log(f"phase {pi} rate {rate}/s: {done_n} done, {len(unfinished)} aborted, in {ph['input_tokens_per_s']:.0f} "
            f"out {ph['output_tokens_per_s']:.0f} tok/s, rows {ph['token_rows_per_s']:.0f}/s, TTFT p50 "
            f"{ph['ttft_s']['p50']} p90 {ph['ttft_s']['p90']}, TPOT p50 {ph['tpot_ms']['p50']:.1f} ms, running "
            f"{ph['running_mean']:.1f} waiting {ph['waiting_mean']:.1f}, used blocks {ph['used_blocks_mean']:.0f} of "
            f"{res['kv_blocks']}, idle {idle:.1f} s")
        (a.out / "serve.json").write_text(json.dumps(res, indent=1) + "\n")
        if stop[0]:
            log("SIGTERM: stopped in this phase")
            break
    ev = rec.arrays()
    st = np.asarray(steps, dtype=np.float64).reshape(-1, len(lifetimes.STEP_FIELDS))
    np.savez_compressed(a.out / "events.npz", **ev, steps=st, t_end=np.float64(clock()))
    res["events"] = {lifetimes.KINDS[k]: int((ev["kind"] == k).sum()) for k in range(len(lifetimes.KINDS))}
    (a.out / "serve.json").write_text(json.dumps(res, indent=1) + "\n")
    log(f"events {res['events']}, {len(steps)} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
