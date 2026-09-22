"""Production recorder (CONTRACT.md §3, mode "Production").

Runs STOCK vLLM under the profile (compiled, CUDA graphs, prefix caching, chunked prefill;
declared flags in ``profile.DECLARED_ENV``) on the §11 workload and records per request:
request id, prompt token ids, SamplingParams, output token ids, finish reason, chosen-token
logprobs, ``num_cached_tokens``, and the scheduler's step trace. Also measures production
overhead (recording on vs off, three repetitions each) into ``timings.json`` and writes the
profile manifest (§2.1).

**The recorded pass is the engine's FIRST pass over the workload** (§15.4). The first recorder
(adapter revisions <= e2399cd) warmed up on ``requests[:2]`` and ran unrecorded passes over the
whole workload in the same engine before the recorded one, so with stock prefix caching every
recorded request was a full prefix-cache hit (``num_cached_tokens = prompt_len - 16``): the
recorded tokens came from cached-prefix attention under a different batching, not from the
prefill the replay reproduces. Now: warm-up (compile / CUDA-graph capture, cold start) runs
only on throwaway random-token prompts that share no ``block_size``-token block with any
workload prompt (asserted), then the workload is recorded, and the recorder fails loudly if
any recorded request reports ``num_cached_tokens != 0`` or the step trace shows a cache hit at
admission. Prefix caching stays at the stock default (on) and the effective flag is recorded.
Timing passes for the overhead ratio run after the recorded pass, each behind
``LLM.reset_prefix_cache()`` so they are fresh prefills like the recorded pass; one extra pass
*without* the reset (every prompt a cache hit) records whether stock cached-prefix production
reproduces the fresh-prefill tokens (``timings.cached_prefix_pass``).

Run ON THE POD::

    /workspace/venv/bin/python -m verity_vllm_adapter.production --case B0 \
        --checkpoints /workspace/veritor/manifests/checkpoints.json \
        --workload /workspace/results/B0/workload.json --out /workspace/results/B0/production \
        --adapter-revision <git sha> --wheel-sha256 <sha> --vllm-commit <sha>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from verity_vllm_adapter import profile as prof

ENV_FLAGS = prof.apply_env()

import numpy as np  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from verity_vllm_adapter import kernel_identity, state_rows  # noqa: E402
from verity_vllm_adapter.driver import run_requests  # noqa: E402
from verity_vllm_adapter.steptrace import StepTrace  # noqa: E402
from verity_vllm_sampler.workload import load_workload  # noqa: E402

# Warm-up prompts come from a different corpus seed than any workload (workload.json seeds are
# small integers; the sampler's corpus_seed values are 32-bit); the disjointness is asserted anyway.
WARMUP_SEED = 0x5EED_7A9B
WARMUP_LENGTHS = (32, 128)
WARMUP_MAX_TOKENS = 8


def greedy_params(max_tokens: int, logprobs: int | None) -> SamplingParams:
    return SamplingParams(temperature=0.0, top_p=1.0, top_k=0, max_tokens=max_tokens, seed=None, logprobs=logprobs)


class _Throwaway:
    def __init__(self, i: int, ids: list[int], max_tokens: int) -> None:
        self.request_id = f"warmup-{i}"
        self.prompt_token_ids = ids
        self.prompt_len = len(ids)
        self.max_tokens = max_tokens


def aligned_blocks(ids: list[int], block_size: int) -> set[tuple[int, ...]]:
    """Every full ``block_size``-aligned token block of a prompt (what the prefix cache hashes;
    a hit at block j also requires blocks 0..j-1 to match, so disjoint aligned blocks suffice.
    We check every aligned block, not only the first, to be conservative)."""
    return {tuple(ids[i : i + block_size]) for i in range(0, len(ids) - block_size + 1, block_size)}


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def explainable_cached_tokens(requests, block_size: int) -> dict[str, dict]:
    """Per request, the most prefix tokens that a *stock* prefix cache can legitimately have hit
    within the recorded pass itself: full aligned blocks shared with another workload prompt
    (the B0 workload has a chance 17-token shared prefix between two prompts, one block). Any
    hit beyond this bound can only come from outside the pass (warm-up / earlier passes) and is
    contamination."""
    out: dict[str, dict] = {}
    for r in requests:
        best, sources = 0, []
        for s in requests:
            if s is r:
                continue
            blocks = min(common_prefix_len(list(r.prompt_token_ids), list(s.prompt_token_ids)), len(s.prompt_token_ids)) // block_size
            if blocks > 0:
                sources.append({"request_id": s.request_id, "shared_aligned_blocks": blocks})
                best = max(best, blocks * block_size)
        out[r.request_id] = {"max_explainable_cached_tokens": best, "sources": sources}
    return out


def build_warmup(requests, vocab: int, block_size: int) -> tuple[list[_Throwaway], dict]:
    """Throwaway warm-up prompts sharing no aligned block with any workload prompt (§15.4)."""
    workload_blocks: set[tuple[int, ...]] = set()
    for r in requests:
        workload_blocks |= aligned_blocks(list(r.prompt_token_ids), block_size)
    rng = np.random.default_rng(WARMUP_SEED)
    warm: list[_Throwaway] = []
    for i, n in enumerate(WARMUP_LENGTHS):
        for _attempt in range(100):
            ids = rng.integers(3, vocab - 1, size=n).tolist()
            if not (aligned_blocks(ids, block_size) & workload_blocks):
                break
        else:  # pragma: no cover - 16 random tokens colliding 100 times is not a thing
            raise SystemExit("could not draw a warm-up prompt disjoint from the workload")
        warm.append(_Throwaway(i, ids, WARMUP_MAX_TOKENS))
    shared = sum(len(aligned_blocks(w.prompt_token_ids, block_size) & workload_blocks) for w in warm)
    assert shared == 0, shared
    return warm, {
        "requests": len(warm),
        "prompt_lens": [w.prompt_len for w in warm],
        "max_tokens": WARMUP_MAX_TOKENS,
        "seed": WARMUP_SEED,
        "block_size": block_size,
        "aligned_blocks_shared_with_workload": shared,
        "workload_aligned_blocks": len(workload_blocks),
    }


def params_dict(sp: SamplingParams) -> dict:
    return {
        "temperature": sp.temperature,
        "top_p": sp.top_p,
        "top_k": sp.top_k,
        "max_tokens": sp.max_tokens,
        "seed": sp.seed,
        "logprobs": sp.logprobs,
        "n": sp.n,
        "ignore_eos": sp.ignore_eos,
        "min_tokens": sp.min_tokens,
        "repetition_penalty": sp.repetition_penalty,
        "presence_penalty": sp.presence_penalty,
        "frequency_penalty": sp.frequency_penalty,
    }


def run_once(llm: LLM, requests, logprobs: int | None, record: bool, scheduler=None, first_boundary: str = "designated", boundary_out=None):
    """One pass over the workload. ``record`` = scheduler step-trace wrapper + per-step output
    bookkeeping on; off = the bare add_request/step loop. Same engine, same kernels.

    ``first_boundary`` (MINREC 2026-09-09, opt-in; default ``designated`` = unchanged behaviour): for a multi-RU partition
    (``layer`` | ``block``) additionally hash the plan's first-boundary rows online (``boundary.BoundaryRecorder``) so the
    pre-``r_q`` commitment can bind the CUs' outgoing boundaries (instance ``partition.first_boundary.rule = 'outgoing
    boundary runs of every CU'``); the commitment summary is written to ``boundary_out``."""
    trace = None
    brec = None
    if record and scheduler is not None:
        trace = StepTrace()
        trace.install(scheduler)
    if record and first_boundary != "designated":
        from verity_vllm_adapter.boundary import BoundaryRecorder

        brec = BoundaryRecorder(runner=prof.get_model_runner(llm), model=prof.get_model_runner(llm).model, plan=first_boundary,
                                out_dir=boundary_out, sink="hash", log=lambda *a: None)
    seen_steps: list[int] = []
    outs_by_id, timing = run_requests(
        llm, requests, lambda r: greedy_params(r.max_tokens, logprobs),
        on_step=(lambda outs: seen_steps.append(len(outs))) if record else None,
    )
    if brec is not None:
        timing["first_boundary"] = {k: v for k, v in brec.close().items() if k != "per_step"}
    if trace is not None:
        trace.uninstall()
    outs = [outs_by_id[r.request_id] for r in requests]
    return outs, timing, trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(prof.CASES))
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter-revision", required=True)
    ap.add_argument("--wheel-sha256", default=None)
    ap.add_argument("--vllm-commit", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--logprobs", type=int, default=1)
    ap.add_argument("--skip-verify-weights", action="store_true")
    ap.add_argument("--enforce-eager", action="store_true", help="declared flag: eager production profile (new profile row, §15.5)")
    ap.add_argument("--custom-ops", default=None, choices=["all", "none"],
                    help="declared flag: compilation_config.custom_ops (e.g. 'all' keeps vLLM's CUDA custom ops inside the compiled forward); new profile row")
    ap.add_argument("--compare-tokens", default=None, help="optional production.jsonl of another profile: report token agreement per request (diagnostic only)")
    ap.add_argument("--order", default="workload", choices=["workload", "reversed", "interleaved"],
                    help="declared: submission order of the workload's requests (varied batch composition, M2)")
    ap.add_argument("--max-num-batched-tokens", type=int, default=None,
                    help="declared scheduler flag (varied batch composition, M2); replay reads it from the manifest")
    ap.add_argument("--engine-arg", action="append", default=[], metavar="KEY=VALUE",
                    help="state rows (R2/R3/R4): declared LLM(...) override, e.g. max_num_batched_tokens=256, num_gpu_blocks_override=96, "
                         "enable_prefix_caching=false; recorded in the manifest (engine_overrides) and applied to the replay engine")
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="declared parallelism flag of the profile (a port's parallelism.tensor_parallel_size, lane TP, axis 2): "
                         "shard the model over N GPUs; replay reads it from the manifest")
    ap.add_argument("--first-boundary", default="designated", choices=["designated", "layer", "block"],
                    help="[MINREC 2026-09-09, opt-in; default unchanged] first-boundary rule committed online before r_q: 'designated' = designated "
                         "outputs only (tokens/params/logprobs; whole-request RUs; today's recorder) | 'layer' | 'block' = additionally hash the plan's "
                         "outgoing CU boundary rows online (boundary.BoundaryRecorder); commitment summary in production/boundary/ and timings.json")
    ap.add_argument("--quantization", default=None,
                    help="[QZ, additive] declared quantization scheme of the profile (a port's `quantization`, lane QZ, port b1-fp8): "
                         "LLM(quantization=...), e.g. fp8 (vLLM online per-tensor weight quant + dynamic per-token activation quant); "
                         "recorded in the manifest's `quantization` field; replay reads it from the manifest")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cp = prof.checkpoint_for(args.case, args.checkpoints)
    if not args.skip_verify_weights:
        prof.verify_local_weights(cp)
    wl = load_workload(args.workload)
    requests = list(wl.requests)
    if args.order == "reversed":
        requests = requests[::-1]
    elif args.order == "interleaved":
        # long and short prompts alternate: (last, first, second-last, second, ...)
        n = len(requests)
        order: list[int] = []
        for i in range(n):
            j = n - 1 - i // 2 if i % 2 == 0 else i // 2
            if j not in order:
                order.append(j)
        requests = [requests[j] for j in order]

    kw = prof.engine_kwargs(cp, replay=False, enforce_eager=args.enforce_eager, custom_ops=args.custom_ops,
                            max_num_batched_tokens=args.max_num_batched_tokens, tensor_parallel_size=args.tensor_parallel_size,
                            quantization=args.quantization)
    # state rows (W10): declared KEY=VALUE overrides, recorded in the manifest (engine_overrides)
    engine_overrides = state_rows.parse_engine_args(args.engine_arg)
    kw.update(engine_overrides)
    t_build0 = time.perf_counter()
    llm = LLM(**kw)
    build_s = time.perf_counter() - t_build0
    scheduler = prof.get_scheduler(llm)
    cache_cfg = llm.llm_engine.vllm_config.cache_config
    block_size = int(cache_cfg.block_size)
    prefix_caching = bool(cache_cfg.enable_prefix_caching)
    vocab = int(getattr(llm.llm_engine.vllm_config.model_config.hf_config, "vocab_size", 32000))

    # Warm-up (compilation / cudagraph capture is cold-start; §12 separates it from steady state)
    # on throwaway prompts only: a workload prompt warmed here would make the recorded pass a
    # prefix-cache hit (§15.4, the trap the first recorder fell into).
    warm, warm_info = build_warmup(requests, vocab, block_size)
    _, warm_timing, _ = run_once(llm, warm, args.logprobs, record=False)
    warm_info["timing"] = warm_timing

    # --- THE recorded pass: the engine's first pass over the workload ---
    canonical_outs, t_rec, canonical_trace = run_once(llm, requests, args.logprobs, record=True, scheduler=scheduler,
                                                      first_boundary=args.first_boundary, boundary_out=out / "boundary")
    cached = {r.request_id: int(getattr(o, "num_cached_tokens", 0) or 0) for r, o in zip(requests, canonical_outs, strict=True)}
    admission_hits = [
        nr for s in canonical_trace.steps if "error" not in s for nr in s["new_reqs"] if nr["num_computed_tokens_at_admission"] > 0
    ]
    # Contamination check. Ideal: num_cached_tokens == 0 everywhere. Impossible to demand of the
    # frozen workload under stock prefix caching: two B0 prompts share their first 16-token block,
    # so the later one legitimately hits the earlier one's block *inside* the recorded pass (the
    # first B0/p2 attempt failed the strict assertion exactly there: bb0e17f8, 16 tokens). The
    # invariant enforced is therefore: every cached token is explained by an aligned block of
    # another workload prompt; anything beyond that bound can only be warm-up / earlier-pass
    # state and aborts the recording. In-workload hits are recorded, not hidden (§15.8).
    explainable = explainable_cached_tokens(requests, block_size)
    contaminated = {k: v for k, v in cached.items() if v > explainable[k]["max_explainable_cached_tokens"]}
    if contaminated:
        raise SystemExit(
            f"RECORDED PASS CONTAMINATED: {len(contaminated)} request(s) report more cached tokens than any other workload "
            f"prompt can explain ({[(k, v, explainable[k]['max_explainable_cached_tokens']) for k, v in list(contaminated.items())[:4]]}); "
            f"{len(admission_hits)} cache hit(s) at admission in the step trace. The recorded pass must be the engine's "
            f"first pass over the workload (§15.4); no production.jsonl written."
        )
    if not prefix_caching and any(cached.values()):
        raise SystemExit(f"num_cached_tokens != 0 with prefix caching disabled?! {cached}")
    in_workload_hits = [
        {"request_id": k, "num_cached_tokens": v, "prompt_len": next(r.prompt_len for r in requests if r.request_id == k),
         "explained_by": [s for s in explainable[k]["sources"] if s["shared_aligned_blocks"] * block_size >= v]}
        for k, v in cached.items() if v
    ]

    # --- overhead ratio (§12): recording on vs off, same profile; every pass a fresh prefill like
    # the recorded one (prefix cache reset before each pass; the reset is a public LLM API and
    # touches no kernel), interleaved so clocks/thermal drift hit both equally.
    timings = {
        "engine_build_s": build_s,
        "warmup": warm_info,
        "recorded_pass": t_rec,
        "recording_off": [],
        "recording_on": [],
        "prefix_cache_reset_before_each_timing_pass": True,
        "first_boundary": args.first_boundary,
    }
    resets_ok = []
    for _ in range(args.reps):
        resets_ok.append(bool(llm.reset_prefix_cache()))
        _, t, _ = run_once(llm, requests, args.logprobs, record=False)
        timings["recording_off"].append(t)
        resets_ok.append(bool(llm.reset_prefix_cache()))
        _, t, _ = run_once(llm, requests, args.logprobs, record=True, scheduler=scheduler, first_boundary=args.first_boundary,
                           boundary_out=out / "boundary-timing")
        timings["recording_on"].append(t)
    timings["prefix_cache_resets_succeeded"] = all(resets_ok)
    off = sum(t["wall_s"] for t in timings["recording_off"]) / args.reps
    on = sum(t["wall_s"] for t in timings["recording_on"]) / args.reps
    timings["production_overhead_ratio"] = on / off
    timings["note"] = (
        "recording_on = per-request output recording + scheduler step trace wrapper; "
        "recording_off = plain generate; both in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0); "
        "the recorded pass itself is the first pass over the workload (recorded_pass), before these."
    )

    # Reproducibility (stock vs stock, §15.5 diagnostic order step 1): two fresh-prefill passes
    # (prefix cache reset) and one cached-prefix pass (no reset: every prompt but its last block is
    # a cache hit, as in the contaminated first recorder) against the recorded tokens.
    base_tokens = [list(x.outputs[0].token_ids) for x in canonical_outs]

    def pass_tokens(reset: bool) -> tuple[list[list[int]], list[int]]:
        if reset:
            resets_ok.append(bool(llm.reset_prefix_cache()))
        o, _, _ = run_once(llm, requests, args.logprobs, record=False)
        return [list(x.outputs[0].token_ids) for x in o], [int(getattr(x, "num_cached_tokens", 0) or 0) for x in o]

    fresh_runs = [pass_tokens(reset=True) for _ in range(2)]
    stock_repro = all(toks == base_tokens for toks, _ in fresh_runs)
    timings["stock_vs_stock_token_identical_runs"] = {
        "compared": 2,
        "identical": sum(toks == base_tokens for toks, _ in fresh_runs),
        "num_cached_tokens_seen": sorted({c for _, cs in fresh_runs for c in cs}),
        "note": "fresh prefill (prefix cache reset before each pass) vs the recorded first pass",
    }
    timings["prefix_cache_resets_succeeded"] = all(resets_ok)
    cached_toks, cached_counts = pass_tokens(reset=False)
    diffs = []
    for r, a, b in zip(requests, base_tokens, cached_toks, strict=True):
        if a != b:
            k = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y) if any(x != y for x, y in zip(a, b)) else min(len(a), len(b))
            diffs.append({"request_id": r.request_id, "prompt_len": r.prompt_len, "first_differing_output_index": k,
                          "position": r.prompt_len - 1 + k, "recorded": a[k] if k < len(a) else None, "cached_prefix": b[k] if k < len(b) else None})
    timings["cached_prefix_pass"] = {
        "requests": len(requests),
        "tokens_identical_to_recorded": sum(a == b for a, b in zip(base_tokens, cached_toks)),
        "num_cached_tokens_per_request": dict(zip([r.request_id for r in requests], cached_counts, strict=True)),
        "differences": diffs,
        "note": "stock production, same engine, every prompt a prefix-cache hit except its last block, vs the recorded fresh-prefill pass "
                "(the configuration the first recorder unintentionally recorded)",
    }

    sampling_desc = {"greedy": True, "temperature": 0.0, "top_p": 1.0, "top_k": 0, "seeds": [], "logprobs": args.logprobs}
    manifest = prof.build_manifest(
        case=args.case,
        cp=cp,
        llm=llm,
        wheel_sha256=args.wheel_sha256,
        vllm_commit=args.vllm_commit,
        adapter_revision=args.adapter_revision,
        # W1cF F9: what actually runs in production — no tensor interception at all. The recorder observes
        # the scheduler (StepTrace), the engine's outputs/logprobs and, afterwards, the compile caches.
        interception="none (production uninstrumented: StepTrace scheduler observer + engine outputs; Inductor/Triton caches read after the run)",
        sampling=sampling_desc,
        env_flags=ENV_FLAGS,
        enforce_eager=args.enforce_eager,
        custom_ops=args.custom_ops,
        max_num_batched_tokens=args.max_num_batched_tokens,
        request_order=args.order,
        engine_overrides=engine_overrides,
        tensor_parallel_size=args.tensor_parallel_size,
        quantization=args.quantization,
    )
    # §16.1: identity of the kernels Inductor generated for this profile (content hashes of the
    # generated Triton sources / wrapper programs), recorded in the manifest next to the effective
    # compilation mode; the full listing is kernels.json. Eager production has none (count 0).
    kid = kernel_identity.collect(kernel_identity.inductor_cache_dir(), kernel_identity.triton_cache_dir(),
                                  vllm_cache_dir=str(getattr(llm.llm_engine.vllm_config.compilation_config, "cache_dir", "") or ""))
    (out / "kernels.json").write_text(json.dumps(kid, indent=1) + "\n")
    manifest["compilation"]["generated_kernels"] = {k: kid["summary"][k] for k in ("wrappers", "distinct_kernel_sources", "kernel_set_sha256", "call_set_sha256")}
    # W1cF F10: the normalised PTX of every compiled kernel variant, exported for the bundle (the replay
    # ships production/ptx/ as kernels/<sha>.ptx); its set hash is bound into the manifest.
    ptx_index = kernel_identity.export_ptx(kernel_identity.triton_cache_dir(), out / "ptx", gpu=manifest.get("gpu"))
    manifest["compilation"]["generated_kernels"]["ptx_code_set_sha256"] = ptx_index["ptx_code_set_sha256"]
    manifest["compilation"]["generated_kernels"]["ptx_variants"] = ptx_index["variants"]
    profile_id = prof.profile_id_of(manifest)
    (out / "manifest.json").write_bytes(prof.canonical_json_bytes(manifest))
    (out / "manifest.pretty.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with (out / "production.jsonl").open("w") as f:
        for r, o in zip(requests, canonical_outs, strict=True):
            c = o.outputs[0]
            lps = []
            margins = []  # NOTE 15:20 item 1: top-2 logprob margin per emitted token (chosen - best other);
            # needs --logprobs >= 2, else null per token.  Additive: the boundary binds chosen logprobs only.
            if c.logprobs:
                for tok, lp in zip(c.token_ids, c.logprobs, strict=False):
                    entry = lp.get(tok)
                    lps.append(entry.logprob if entry is not None else None)
                    others = [e.logprob for t2, e in lp.items() if t2 != tok]
                    margins.append((entry.logprob - max(others)) if (entry is not None and others) else None)
            rec = {
                "request_id": r.request_id,
                "profile_id": profile_id,
                "prompt_len": r.prompt_len,
                "prompt_token_ids": list(r.prompt_token_ids),
                "sampling_params": params_dict(greedy_params(r.max_tokens, args.logprobs)),
                "output_token_ids": list(c.token_ids),
                "finish_reason": c.finish_reason,
                "stop_reason": c.stop_reason,
                "chosen_logprobs": lps,
                "top2_margins": margins,
                "num_cached_tokens": cached[r.request_id],
            }
            f.write(json.dumps(rec) + "\n")

    summary = canonical_trace.summary()
    (out / "steptrace.json").write_text(json.dumps({"summary": summary, "steps": canonical_trace.steps}, indent=1) + "\n")
    (out / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    # §15.8 witness of activation for the recorded pass (also copied into Table 1 by verify.py)
    workload_exercised = {
        "recorded_pass": "first pass of a fresh engine over the workload (warm-up on throwaway prompts only)",
        "enable_prefix_caching": prefix_caching,
        "block_size": block_size,
        "num_cached_tokens_per_request": cached,
        "requests_with_cached_tokens": sum(1 for v in cached.values() if v),
        "in_workload_prefix_cache_hits": in_workload_hits,
        "cached_tokens_from_outside_the_recorded_pass": 0,
        "prefix_cache_hits_at_admission": len(admission_hits),
        "peak_concurrency_achieved": summary["peak_concurrency_achieved"],
        "engine_steps": summary["engine_steps"],
        "preemptions": summary["preemptions"],
        "prefill_chunks_per_request": summary["prefill_chunks_per_request"],
        "max_tokens_in_a_step": summary["max_tokens_in_a_step"],
        "warmup": {k: v for k, v in warm_info.items() if k != "timing"},
        "effective_execution": prof.effective_compilation(llm),
    }
    (out / "workload_exercised.json").write_text(json.dumps(workload_exercised, indent=2) + "\n")
    # optional cross-profile token comparison (stock vs stock under two declared profiles, §15.5 step 1)
    token_cmp = None
    if args.compare_tokens:
        other = {json.loads(l)["request_id"]: json.loads(l) for l in Path(args.compare_tokens).read_text().splitlines() if l.strip()}
        per = []
        for r, toks in zip(requests, base_tokens, strict=True):
            o = other.get(r.request_id)
            if o is None:
                continue
            ot = list(o["output_token_ids"])
            k = next((i for i, (x, y) in enumerate(zip(toks, ot)) if x != y), None)
            per.append({"request_id": r.request_id, "prompt_len": r.prompt_len, "identical": toks == ot,
                        "first_differing_output_index": k, "position": (r.prompt_len - 1 + k) if k is not None else None})
        token_cmp = {"other_production": str(args.compare_tokens), "other_profile_id": next(iter(other.values()))["profile_id"] if other else None,
                     "requests_compared": len(per), "identical": sum(p["identical"] for p in per), "per_request": per}
        (out / "token_comparison.json").write_text(json.dumps(token_cmp, indent=2) + "\n")
    (out / "run.json").write_text(
        json.dumps(
            {
                "profile_id": profile_id,
                "case": args.case,
                "argv": sys.argv,
                "engine_kwargs": {k: v for k, v in kw.items()},
                "stock_vs_stock_identical": stock_repro,
                "cached_prefix_pass_identical": not diffs,
                "workload_exercised": workload_exercised,
                "utc": prof.utc_stamp(),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"profile_id={profile_id} overhead={timings['production_overhead_ratio']:.4f} stock_repro={stock_repro} "
        f"cached_tokens_in_recorded_pass={sum(cached.values())} cached_prefix_pass_identical={not diffs} "
        f"({len(diffs)} request(s) differ)"
        + (f" tokens_identical_to_{token_cmp['other_profile_id']}={token_cmp['identical']}/{token_cmp['requests_compared']}" if token_cmp else "")
    )


if __name__ == "__main__":
    main()
