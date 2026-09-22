"""Stock production recorder (contract §3 "Production"): one fresh process per invocation.

    python -m verity_vllm_sampler.production --workload workload.json --variant topk_topp --seed 1 \
        --max-num-seqs 64 --rep 0 --out prod.json [--order reversed|shuffled:<n>] [--no-record] \
        [--eos-probes eos_probes.json] [--abort <request index>:<after_tokens>]

Engine: `vllm.LLM(...)` under the profile (VLLM_BATCH_INVARIANT=1 from the environment, stock
defaults otherwise) driven through its `LLMEngine` so that request ids, submission order and
mid-generation aborts are under our control.  Nothing is instrumented: outputs come from
`RequestOutput` only (token ids, finish_reason, stop_reason, per-token logprob of the chosen
token when `logprobs=1`), plus `SchedulerStats.num_running_reqs` per step for the achieved
concurrency.

`--no-record` runs the same requests without logprobs and without keeping outputs (the "stock"
baseline for Table 4 production overhead); timings are written either way.

Prefix-cache witness (contract §15.4 / §15.8): every record carries `num_cached_tokens` (the
engine's own `RequestOutput.num_cached_tokens`, i.e. how many prompt tokens were served from the
prefix cache instead of being prefilled) and `cached_prefix = num_cached_tokens > 0`; the run
carries the scheduler's `prefix_cache_stats` (queries/hits) per step.  A recorded request that hit
the cache is NOT a fresh-prefill execution and is labelled as such.  `--warmup-throwaway N` runs N
throwaway prompts (seeded random ids, verified to share no 16-token block with the workload) to
completion before the recorded pass, `--only/--exclude` restrict the pass to some request indices
(the workload's requests 11 and 13 share their first 17 tokens, so in one engine one of them always
hits the other's first block; a fresh-prefill record runs the second one in its own engine), and
`--assert-no-cached` makes the process exit 4 (after writing the JSON) if any recorded request had
`num_cached_tokens > 0`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from typing import Any

from .workload import SampledRequest, Workload, build_eos_probes, load_workload


def engine_kwargs(model: str, max_num_seqs: int, max_model_len: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    kw: dict[str, Any] = dict(
        model=model,
        dtype="bfloat16",
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        # Fixed KV cache (8 GiB ~ 300k tokens, far above 16 x 4096) instead of gpu_memory_utilization:
        # skips vLLM's free-memory profiling, which aborts when another engine (the replay dump)
        # shares the 24 GB card; two engines fit side by side.
        kv_cache_memory_bytes=8 * 1024**3,
        disable_log_stats=False,  # scheduler stats -> achieved concurrency
        enable_prefix_caching=True,
    )
    if extra:
        kw.update(extra)
    return kw


def profile_snapshot(llm) -> dict[str, Any]:
    """Record what the engine actually resolved (for the manifest / Table 1 row)."""
    import torch
    import vllm

    cfg = llm.llm_engine.vllm_config
    comp = cfg.compilation_config
    ver = vllm.__version__
    snap: dict[str, Any] = {
        "vllm": ver,
        "vllm_commit": ver.split("+g", 1)[1] if "+g" in ver else None,  # nightly wheels carry the short sha
        "vllm_file": os.path.dirname(vllm.__file__),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "model": cfg.model_config.model,
        "dtype": str(cfg.model_config.dtype),
        "max_num_seqs": cfg.scheduler_config.max_num_seqs,
        "max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens,
        "max_model_len": cfg.model_config.max_model_len,
        "enable_chunked_prefill": cfg.scheduler_config.enable_chunked_prefill,
        "enable_prefix_caching": cfg.cache_config.enable_prefix_caching,
        "async_scheduling": getattr(cfg.scheduler_config, "async_scheduling", None),
        "compilation_mode": str(getattr(comp, "mode", None)),
        "cudagraph_mode": str(getattr(comp, "cudagraph_mode", None)),
        "enforce_eager": cfg.model_config.enforce_eager,
        "use_v2_model_runner": bool(cfg.use_v2_model_runner),
        "engine_seed": cfg.model_config.seed,
        "env": {k: os.environ.get(k) for k in (
            "VLLM_BATCH_INVARIANT", "VLLM_USE_V2_MODEL_RUNNER", "VLLM_ENABLE_V1_MULTIPROCESSING",
            "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_ATTENTION_BACKEND")},
        "python": platform.python_version(),
    }
    try:
        snap["attention_backend"] = str(cfg.attention_config.backend)
    except Exception:  # noqa: BLE001
        pass
    try:
        import triton
        snap["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import flashinfer
        snap["flashinfer"] = flashinfer.__version__
    except Exception:  # noqa: BLE001
        snap["flashinfer"] = None
    return snap


def make_sampling_params(sreq: SampledRequest, record: bool, extra: dict[str, Any] | None = None):
    from vllm import SamplingParams

    kw = dict(sreq.sampling)
    if record:
        kw["logprobs"] = 1
    if extra:
        kw.update(extra)
    return SamplingParams(**kw)


def run_requests(llm, sreqs: list[SampledRequest], record: bool, abort: tuple[str, int] | None = None,
                 sampling_extra: dict[str, Any] | None = None, id_map: dict[str, str] | None = None,
                 params_override: dict[str, Any] | None = None) -> dict[str, Any]:
    """Submit all requests up-front in the given order and step the engine to completion.

    `id_map` (internal -> external request id) is filled as requests are added: vLLM's
    InputProcessor.assign_request_id rewrites every id to `<external>-<8 random hex>` and that is
    what the worker's InputBatch.req_ids carries (the outputs use the external id again).
    `params_override` maps external request id -> SamplingParams to use instead of the default
    (trace replay on the nightly wheel: `trace_decode_token_ids`)."""
    engine = llm.llm_engine
    core = engine.engine_core
    peak = 0
    steps = 0
    running_trace: list[int] = []
    waiting_trace: list[int] = []
    prefix_cache_trace: list[dict[str, int]] = []  # per step: scheduler's PrefixCacheStats (queries/hits since last read)
    internal_ids: dict[str, str] = {}
    orig_get_output = core.get_output

    def get_output_traced():
        nonlocal peak
        out = orig_get_output()
        st = getattr(out, "scheduler_stats", None)
        if st is not None:
            running_trace.append(int(st.num_running_reqs))
            waiting_trace.append(int(getattr(st, "num_waiting_reqs", 0) or 0))
            peak = max(peak, int(st.num_running_reqs))
            pcs = getattr(st, "prefix_cache_stats", None)
            if pcs is not None:
                prefix_cache_trace.append({"requests": int(getattr(pcs, "requests", 0) or 0),
                                           "queries": int(getattr(pcs, "queries", 0) or 0),
                                           "hits": int(getattr(pcs, "hits", 0) or 0)})
        return out

    core.get_output = get_output_traced  # type: ignore[method-assign]
    try:
        t0 = time.perf_counter()
        for s in sreqs:
            params = (params_override or {}).get(s.request_id) or make_sampling_params(s, record, sampling_extra)
            internal = engine.add_request(
                request_id=s.request_id,
                prompt={"prompt_token_ids": list(s.prompt_token_ids)},
                params=params,
            )
            internal_ids[s.request_id] = internal
            if id_map is not None:
                id_map[internal] = s.request_id
        t_submit = time.perf_counter() - t0
        finals: dict[str, Any] = {}
        partial: dict[str, Any] = {}
        arrivals: dict[str, list[int]] = {}  # request_id -> step index at which each token arrived
        aborted_at: dict[str, int] = {}
        cached: dict[str, int] = {}  # request_id -> engine-reported num_cached_tokens (max over its outputs)
        while engine.has_unfinished_requests():
            outs = engine.step()
            steps += 1
            for o in outs:
                c = o.outputs[0]
                arr = arrivals.setdefault(o.request_id, [])
                while len(arr) < len(c.token_ids):
                    arr.append(steps)
                partial[o.request_id] = o
                nct = getattr(o, "num_cached_tokens", None)
                if nct is not None:
                    cached[o.request_id] = max(cached.get(o.request_id, 0), int(nct))
                if o.finished:
                    finals[o.request_id] = o
            if abort is not None:
                rid, after = abort
                if rid not in finals and rid not in aborted_at and rid in partial \
                        and len(partial[rid].outputs[0].token_ids) >= after:
                    engine.abort_request([rid])
                    aborted_at[rid] = steps
        t_total = time.perf_counter() - t0
    finally:
        core.get_output = orig_get_output  # type: ignore[method-assign]

    records = []
    for s in sreqs:
        o = finals.get(s.request_id) or partial.get(s.request_id)
        rec: dict[str, Any] = {
            "request_id": s.request_id,
            "base_request_id": s.base_request_id,
            "index": s.index,
            "variant": s.variant,
            "seed": s.seed,
            "sampling": dict(s.sampling),
            "prompt_len": s.prompt_len,
            "max_tokens": s.max_tokens,
            "internal_request_id": internal_ids.get(s.request_id),
        }
        if record:
            rec["prompt_token_ids"] = list(s.prompt_token_ids)
        if o is None:
            rec.update({"output_token_ids": [], "finish_reason": "abort" if s.request_id in aborted_at else "missing",
                        "stop_reason": None, "num_output_tokens": 0})
        else:
            c = o.outputs[0]
            rec.update({
                "output_token_ids": list(c.token_ids),
                "num_prompt_tokens": len(o.prompt_token_ids) if o.prompt_token_ids is not None else s.prompt_len,
                "num_output_tokens": len(c.token_ids),
                "finish_reason": c.finish_reason if s.request_id not in aborted_at else "abort",
                "stop_reason": c.stop_reason,
                "engine_finished": bool(o.finished),
                "token_arrival_step": arrivals.get(s.request_id, []),
                # prefix-cache witness: prompt tokens served from the cache instead of prefilled
                "num_cached_tokens": cached.get(s.request_id),
                "cached_prefix": (cached.get(s.request_id) or 0) > 0,
            })
            if s.request_id in aborted_at:
                rec["aborted_at_step"] = aborted_at[s.request_id]
                rec["client_finish_reason"] = "abort"
            if record and c.logprobs:
                lp = []
                for tok, d in zip(c.token_ids, c.logprobs):
                    e = d.get(tok)
                    lp.append(None if e is None else {"logprob": float(e.logprob), "rank": e.rank})
                rec["chosen_logprobs"] = lp
            # output-presence record per slot (contract §7/§8 N14): token id or None after finish
            pres = list(c.token_ids) + [None] * (s.max_tokens - len(c.token_ids))
            rec["output_presence"] = pres
        rec["output_sha256"] = hashlib.sha256(json.dumps(rec.get("output_token_ids", [])).encode()).hexdigest()
        records.append(rec)
    return {
        "records": records,
        "peak_concurrency_achieved": peak,
        "running_trace": running_trace,
        "waiting_trace": waiting_trace,
        "prefix_cache_trace": prefix_cache_trace,
        "prefix_cache_totals": {k: sum(d[k] for d in prefix_cache_trace) for k in ("requests", "queries", "hits")},
        "num_cached_tokens_by_index": {str(s.index): cached.get(s.request_id) for s in sreqs},
        "requests_with_cached_prefix": sorted(s.index for s in sreqs if (cached.get(s.request_id) or 0) > 0),
        "engine_steps": steps,
        "timing": {"submit_s": t_submit, "generate_s": t_total},
    }


BLOCK = 16  # vLLM default KV block size (prefix-cache granularity)


def shared_prefix_pairs(sreqs: list[SampledRequest], block: int = BLOCK) -> list[dict[str, int]]:
    """Pairs of requests whose prompts share >= one full leading block: in a single engine with prefix
    caching on, the later-scheduled one of each pair is served `blocks*block` tokens from the cache."""
    out = []
    for i, a in enumerate(sreqs):
        for b in sreqs[i + 1:]:
            pa, pb = a.prompt_token_ids, b.prompt_token_ids
            n = 0
            lim = min(len(pa), len(pb))
            while n < lim and pa[n] == pb[n]:
                n += 1
            if n >= block:
                out.append({"first_index": a.index, "second_index": b.index, "common_prefix_tokens": n,
                            "full_blocks": n // block})
    return out


def throwaway_prompts(sreqs: list[SampledRequest], n: int, vocab_size: int = 151643, seed: int = 20260907,
                      lengths: tuple[int, ...] = (48, 200, 640, 1040), block: int = BLOCK) -> list[list[int]]:
    """Seeded random prompts that share no leading `block` tokens with any workload prompt (or each
    other), so warming an engine on them cannot put a workload block into the prefix cache."""
    rng = random.Random(seed)
    heads = {tuple(s.prompt_token_ids[:block]) for s in sreqs}
    out: list[list[int]] = []
    while len(out) < n:
        L = lengths[len(out) % len(lengths)]
        ids = [rng.randrange(1, vocab_size) for _ in range(L)]
        if tuple(ids[:block]) in heads:
            continue
        heads.add(tuple(ids[:block]))
        out.append(ids)
    return out


def warmup_throwaway(llm, sreqs: list[SampledRequest], n: int, max_tokens: int = 8) -> dict[str, Any]:
    """Run `n` throwaway prompts to completion in this engine BEFORE the recorded pass (contract §15.4:
    the recorded pass is not the engine's first work), without touching any workload block."""
    from vllm import SamplingParams

    engine = llm.llm_engine
    prompts = throwaway_prompts(sreqs, n)
    t0 = time.perf_counter()
    for i, ids in enumerate(prompts):
        engine.add_request(request_id=f"warmup-{i}", prompt={"prompt_token_ids": ids},
                           params=SamplingParams(temperature=0.0, max_tokens=max_tokens))
    outs: dict[str, Any] = {}
    steps = 0
    while engine.has_unfinished_requests():
        for o in engine.step():
            outs[o.request_id] = o
        steps += 1
    return {"requests": n, "prompt_lens": [len(p) for p in prompts], "max_tokens": max_tokens, "steps": steps,
            "seconds": time.perf_counter() - t0,
            "num_cached_tokens": {rid: getattr(o, "num_cached_tokens", None) for rid, o in outs.items()},
            "output_lens": {rid: len(o.outputs[0].token_ids) for rid, o in outs.items()},
            "shares_block_with_workload": False}


def order_requests(sreqs: list[SampledRequest], order: str) -> list[SampledRequest]:
    if order == "natural":
        return list(sreqs)
    if order == "reversed":
        return list(reversed(sreqs))
    if order.startswith("shuffled:"):
        r = random.Random(int(order.split(":", 1)[1]))
        out = list(sreqs)
        r.shuffle(out)
        return out
    if order == "longest_first":
        return sorted(sreqs, key=lambda s: -s.prompt_len)
    raise ValueError(order)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--variant", default="greedy", help="greedy | temp | topk_topp | probes (EOS probes)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--order", default="natural")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--abort", default=None, help="<request index>:<after_tokens> abort mid-generation")
    ap.add_argument("--probe-max-tokens", type=int, default=32)
    ap.add_argument("--stop-token-ids", default=None, help="comma list; adds stop_token_ids to every request")
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--engine-json", default=None, help="extra LLM kwargs as JSON")
    ap.add_argument("--only", default=None, help="comma list of request indices to run (others skipped)")
    ap.add_argument("--exclude", default=None, help="comma list of request indices to skip")
    ap.add_argument("--warmup-throwaway", type=int, default=0,
                    help="run N throwaway prompts (no shared 16-token block with the workload) to completion first")
    ap.add_argument("--assert-no-cached", action="store_true",
                    help="exit 4 (after writing the record) if any recorded request had num_cached_tokens > 0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wl = load_workload(args.workload)
    if args.variant == "probes":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(wl.model)
        probes = build_eos_probes(tok, model=wl.model, max_tokens=args.probe_max_tokens)
        base = Workload(schema=wl.schema, model=wl.model, tokenizer_revision=wl.tokenizer_revision,
                        seed=wl.seed, requests=probes, variants=wl.variants, seeds=wl.seeds)
        variant = "greedy" if args.seed is None else "temp"
        sreqs = base.expand(variant, args.seed)
    else:
        sreqs = wl.expand(args.variant, args.seed)
    sreqs = order_requests(sreqs, args.order)
    if args.only:
        keep = {int(x) for x in args.only.split(",")}
        sreqs = [s for s in sreqs if s.index in keep]
    if args.exclude:
        drop = {int(x) for x in args.exclude.split(",")}
        sreqs = [s for s in sreqs if s.index not in drop]
    record = not args.no_record
    extra_sp: dict[str, Any] = {}
    if args.stop_token_ids:
        extra_sp["stop_token_ids"] = [int(x) for x in args.stop_token_ids.split(",")]
    if args.ignore_eos:
        extra_sp["ignore_eos"] = True
    abort = None
    if args.abort:
        idx, after = args.abort.split(":")
        target = [s for s in sreqs if s.index == int(idx)][0]
        abort = (target.request_id, int(after))

    from vllm import LLM

    t0 = time.perf_counter()
    kw = engine_kwargs(wl.model, args.max_num_seqs, args.max_model_len,
                       json.loads(args.engine_json) if args.engine_json else None)
    llm = LLM(**kw)
    t_init = time.perf_counter() - t0
    snap = profile_snapshot(llm)
    warm = warmup_throwaway(llm, sreqs, args.warmup_throwaway) if args.warmup_throwaway > 0 else None
    result = run_requests(llm, sreqs, record, abort, extra_sp or None)
    cached_idx = result["requests_with_cached_prefix"]
    result.update({
        "schema": "verity-vllm/production-record/v1",
        "mode": "production",
        "recording": record,
        "variant": args.variant,
        "seed": args.seed,
        "rep": args.rep,
        "order": args.order,
        "only": args.only, "exclude": args.exclude,
        "max_num_seqs": args.max_num_seqs,
        "abort": args.abort,
        "sampling_extra": extra_sp,
        "engine_kwargs": {k: v for k, v in kw.items()},
        "profile": snap,
        "timing_init_s": t_init,
        "pid": os.getpid(),
        "argv": sys.argv,
        "eos_token_id": llm.get_tokenizer().eos_token_id,
        "warmup": warm,
        "recorded_pass": "first pass of these prompts in this engine" + (" (after throwaway warm-up)" if warm else ""),
        "shared_prefix_pairs_in_pass": shared_prefix_pairs(sreqs),
        "any_cached_prefix": bool(cached_idx),
        "assert_no_cached": args.assert_no_cached,
        "assert_no_cached_failed": bool(args.assert_no_cached and cached_idx),
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"PRODUCTION_DONE {args.out} peak={result['peak_concurrency_achieved']} "
          f"steps={result['engine_steps']} gen_s={result['timing']['generate_s']:.2f} init_s={t_init:.1f} "
          f"cached_prefix_indices={cached_idx} prefix_cache_totals={result['prefix_cache_totals']}", flush=True)
    shutdown_engine(llm)
    if args.assert_no_cached and cached_idx:
        print(f"ASSERT_NO_CACHED_FAILED indices={cached_idx}", file=sys.stderr, flush=True)
        os._exit(4)


def shutdown_engine(llm, grace_s: float = 60.0) -> None:
    """Tear the engine down explicitly.  Left to interpreter finalisation, multiprocessing's
    exit hook can `join()` the EngineCore child before vLLM's own shutdown hook signals it
    (observed once: probes run, `transformers` imported before `vllm`), hanging forever with the
    results already on disk.  A watchdog force-exits if finalisation still stalls."""
    import threading

    def _force_exit():
        print("SHUTDOWN_WATCHDOG: forcing exit after results were written", file=sys.stderr, flush=True)
        os._exit(0)

    t = threading.Timer(grace_s, _force_exit)
    t.daemon = True
    t.start()
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception as e:  # noqa: BLE001
        print(f"engine shutdown raised {type(e).__name__}: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
