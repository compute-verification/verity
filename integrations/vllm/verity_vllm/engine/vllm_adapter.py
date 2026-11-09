"""vLLM adapter (runtime side): load a workload, and drive its requests through a built engine.

The engine is built in `engine.build`, the run's profile manifest and versions in `engine.run_facts`, and code identity in
`engine.code_identity`; the stages reach all of them through this module's names.  The capture and its run header are
`observe.capture`'s, above the engine, and its callers import them from there.  `load_workload` and what it calls stay in this
file: tests execute them from its source, without importing torch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from verity_vllm.engine import static_schedule
from verity_vllm.config import RowEngine, workload_file
from verity_vllm.program import prefix_plan as PP
from verity_vllm.engine.arrivals import run_requests_arrivals  # step-indexed arrivals (torch-free module, unit-tested)
from verity_vllm.engine import chunks as _chunks  # prefill-chunk / token-step attribution (torch-free, unit-tested)
from verity_vllm.engine.build import (CASE, EXECUTION_ENFORCE_EAGER, Engine, build_engine, case_of, compilation_facts,  # noqa: F401
                                      effective_execution_doc, engine_kwargs_for, execution_label_of, observed_flash_attn_version,
                                      parse_engine_args, parse_target_arg, prepare_compiled_process_cache, resolved_async_scheduling,
                                      target_mismatches, target_of_workload, vllm_has_flashinfer)
from verity_vllm.engine.code_identity import (_VERITY_INDUCTOR_DIR_MARK, CUBIN_CODE_IDENTITY_RULE, CUBIN_DENY_LIST,  # noqa: F401
                                              cubin_code_sha256, cubin_code_sha256_normalized, cubin_section_denied,
                                              cubin_section_detail, cubin_sections, generated_kernels_of_this_process,
                                              loaded_code_objects, loaded_code_objects_delta)
from verity_vllm.engine.run_facts import (ACCEPTED, ACCEPTED_PROFILE_ID, ACCEPTED_RECORD, host_doc, model_config_subset,  # noqa: F401
                                          profile_manifest, versions_doc)


def load_workload(path: str | Path, max_tokens: int | None = None) -> dict:
    """The workload file (`workloads/workload_32x16_1req.json`, `workload_qwen15_32x16_1req.json`) ->
    {"requests": [...], "sampling": {...}, "case": <checkpoint role or None>, "accepted": {request_id: accepted token ids}}.
    The accepted tokens come from the file's `accepted_token_ids` per request (M6) or `ACCEPTED_TOKENS` of the case."""
    doc = json.loads(workload_file(path).read_text())
    eng = RowEngine.of(doc)
    # an optional top-level `sampling` block selects the variant and the request seeds / fan-out:
    # {"variant": "greedy", "seed": null, "n": 1} (defaults = the greedy runs); a request may override `seed` / `n`.
    samp = dict(doc.get("sampling") or {})
    variant = str(samp.get("variant", "greedy"))
    params = (doc.get("variants") or {}).get(variant)
    if params is None:
        if variant != "greedy":
            raise KeyError(f"workload {path}: sampling variant {variant!r} is not in `variants`")
        params = {"temperature": 0.0, "top_p": 1.0, "top_k": 0}
    seed = None if samp.get("seed") is None else int(samp["seed"])
    n = int(samp.get("n", 1))
    reqs = []
    accepted: dict[str, list[int]] = {}
    per_request: list[dict] = []
    plan = PP.plan(doc)
    for i, r in enumerate(doc["requests"]):
        rs = r.get("seed", seed)
        req = {"request_id": r["request_id"], "prompt_token_ids": [int(t) for t in r["prompt_token_ids"]],
               "max_tokens": int(max_tokens if max_tokens is not None else r["max_tokens"]),
               "seed": None if rs is None else int(rs), "n": int(r.get("n", n)),
               # serving profile B8-mixed: the engine step at which the request is admitted (0 = with the first
               # step, the historical behaviour).  Step-indexed, not wall-clock, so the observer and control runs see the same
               # batch membership at every step.
               "arrive_step": int(r.get("arrive_step") or 0),
               # the prompt rows the prefix cache serves it and the request that computed each cached block (prefix_plan): its Program's
               # shape is (LP, T, C), and the Commit's replay reads those rows from what the producers committed
               "cached": plan[i]["cached"], "cached_from": [str(doc["requests"][q]["request_id"]) for q in plan[i]["producers"]]}
        if r.get("variant") is not None:
            # a request may pin ANOTHER variant (mixed batch: one greedy row next to a sampled row); the
            # request record then carries its own temperature / top_p / top_k and the manifest lists them per request
            rp = (doc.get("variants") or {}).get(str(r["variant"]))
            if rp is None:
                raise KeyError(f"workload {path}: request {r['request_id']} variant {r['variant']!r} is not in `variants`")
            req.update({"variant": str(r["variant"]), "temperature": float(rp.get("temperature", 0.0)),
                        "top_p": float(rp.get("top_p", 1.0)), "top_k": int(rp.get("top_k", 0))})
            per_request.append({"request_id": req["request_id"], "variant": req["variant"], "temperature": req["temperature"],
                                "top_p": req["top_p"], "top_k": req["top_k"], "seed": req["seed"]})
        reqs.append(req)
        if r.get("accepted_token_ids") is not None:
            accepted[r["request_id"]] = [int(t) for t in r["accepted_token_ids"]]
    for rid, toks in ACCEPTED_TOKENS.items():
        accepted.setdefault(rid, list(toks))
    temperature = float(params.get("temperature", 0.0))
    sampling = {"temperature": temperature, "top_p": float(params.get("top_p", 1.0)), "top_k": int(params.get("top_k", 0)),
                "logprobs": None, "seed": seed, "n": n, "variant": variant,
                "greedy": temperature == 0.0 and all(p["temperature"] == 0.0 for p in per_request)}
    if per_request:
        sampling["per_request"] = per_request
    if samp.get("ignore_eos"):
        sampling["ignore_eos"] = True           # the static schedule (engine.static_schedule): every request runs its cap
    return {"requests": reqs, "sampling": sampling, "source": str(path), "model": doc.get("model"), "revision": eng.revision, "case": doc.get("case"),
            "accepted": accepted,
            # declared scheduler knobs the workload cannot run without (`engine.args`, e.g. max_num_batched_tokens for an
            # unchunked 8-request prefill); run_config passes them as --engine-arg, an explicit CLI value wins
            "engine_args_required": dict(eng.args),
            "serving_profile": doc.get("serving_profile"), "context_class": doc.get("context_class"),
            "expected_engine_steps": doc.get("expected_engine_steps"),
            # the workload's row id and the execution it declares (row-id suffix / explicit field): the
            # GPU stages hand `execution` to build_engine so enforce_eager is the workload's, never the harness's literal
            "row_id": (doc.get("sweep") or {}).get("row_id") if isinstance(doc.get("sweep"), dict) else doc.get("row_id"),
            "execution": execution_of_workload(doc),
            # the workload's declared TargetProfile overrides (`engine.target`, written by pipeline/workload.py --target): the channel
            # `target_of_workload` documents -- rc3 dropped it here, so every GPU stage built its engine with target=None (engine ARGS
            # still travelled through `engine.args`; environment knobs such as VLLM_USE_DEEP_GEMM have no other channel, and the header
            # carried no `profile.target`).  None when the workload declares none.
            "target": dict(eng.target) if eng.target else None,
            # the workload's whole engine block: what a served Commit holds its engine's switches to (`engine.served.declared`)
            "engine": eng.to_json(),
            # each enabled protocol's settings (`protocol_options.load_settings`), never hashed: the capture and the Commit install from it
            "protocols_config": dict(doc["protocols_config"]) if isinstance(doc.get("protocols_config"), dict) else None,
            "sha256": hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


#: The accepted record's generated tokens for the B0 workload's request (its file predates `accepted_token_ids`):
#: out/scale/e8/live/V5-N32x16-1req-bex0-inc-sc1/adapter/selective/ru/7cc7c3bb…/boundary.json
ACCEPTED_TOKENS: dict[str, list[int]] = {
    "7cc7c3bb-adf5-58cd-87fc-9f1b8e7c34d6": [359, 253, 1123, 1183, 282, 253, 1772, 338, 553, 7527, 288, 11492, 351, 260, 4340, 1357],
}


def execution_of_workload(workload: dict | None) -> str | None:
    """The execution label a workload manifest declares: an explicit top-level `execution` (or `sweep.execution`) when the
    schema carries one, else the last `__` token of `sweep.row_id` (`…__greedy__bi-eager` -> 'bi-eager'); None for a workload outside
    a sweep (no row id: the caller's default applies and the header records `declared_execution: null`)."""
    if not isinstance(workload, dict):
        return None
    sw = workload.get("sweep") if isinstance(workload.get("sweep"), dict) else {}
    for v in (workload.get("execution"), sw.get("execution")):
        if isinstance(v, str) and v:
            return v
    rid = sw.get("row_id") or workload.get("row_id")
    if isinstance(rid, str) and "__" in rid:
        return rid.rsplit("__", 1)[1]
    return None


def sampling_params(req: dict, sampling: dict) -> Any:
    from vllm import SamplingParams

    seed = req.get("seed") if req.get("seed") is not None else sampling.get("seed")     # per-request seed / n
    # a request with its own variant (mixed batch) carries temperature / top_p / top_k; else the workload's
    return SamplingParams(temperature=float(req.get("temperature", sampling.get("temperature", 0.0))),
                          top_p=float(req.get("top_p", sampling.get("top_p", 1.0))),
                          top_k=int(req.get("top_k", sampling.get("top_k", 0))), max_tokens=int(req["max_tokens"]),
                          seed=None if seed is None else int(seed), n=int(req.get("n") or sampling.get("n") or 1),
                          logprobs=sampling.get("logprobs"), ignore_eos=static_schedule.declared_static(sampling))


def fresh_prefix_cache(engine: Engine) -> None:
    """Empty vLLM's prefix cache before a workload run of an engine that serves with it: a run's prefix-cache hits are its own requests'
    (`program.prefix_plan`), never those of a run before it on the same engine (a warm-up, the other arm of a pair, a capture)."""
    if engine.kwargs.get("enable_prefix_caching") and engine.llm.reset_prefix_cache() is False:
        raise RuntimeError("vLLM refused to reset its prefix cache (a request is still running)")


def run_requests(engine: Engine, requests: list[dict], sampling: dict) -> tuple[dict[str, list[int]], dict]:
    """Add every request with *our* request ids and step the engine until all finish (`arrivals.run_requests_arrivals`, at
    each request's declared `arrive_step`).  `timing["requests"]` carries the realized per-request steps (D87)."""
    fresh_prefix_cache(engine)
    class _R:
        def __init__(self, d: dict, order: int) -> None:
            self.request_id = d["request_id"]
            self.prompt_token_ids = list(d["prompt_token_ids"])
            self.max_tokens = int(d["max_tokens"])
            self.seed = d.get("seed")
            self.n = int(d.get("n") or 1)
            self.arrive_step = int(d.get("arrive_step") or 0)
            self._order = order
            # per-request variant (mixed batch): the request's own temperature / top_p / top_k, if pinned
            self.params = {k: d[k] for k in ("temperature", "top_p", "top_k") if k in d}

    # n > 1: vLLM forks child requests f"{index}_{request_id}" (v1/engine/parallel_sampling.py) — the ids the model
    # runner and the log see.  With the default CUMULATIVE output kind each step's RequestOutput carries only the children that produced
    # a token that step (the final one need not hold every child), so collect every CompletionOutput by (parent, index) across steps.
    static = static_schedule.declared_static(sampling)
    if static:
        pc = (engine.effective or {}).get("enable_prefix_caching")
        refusal = static_schedule.host_prefix_cache_refusal(engine.kwargs.get("enable_prefix_caching") if pc is None else pc)
        if refusal:
            raise RuntimeError(refusal)
    by_child: dict[str, dict[int, list[int]]] = {}
    finish: dict[str, dict] = {}
    cached: dict[str, int | None] = {}
    step_no = [0]
    # prefill-chunk attribution: the in-process scheduler's per-request `num_computed_tokens` is read
    # after every step() (pure reads; nothing patched, nothing on the device), the returned tokens are counted per iteration; after
    # the run `chunks.attribute` adds prefill_steps / prefill_chunks / partial_prefill_steps / first_token_step /
    # token_steps to every `timing["requests"][rid]` and `timing["chunk_attribution"]` run-wide.  Every existing key is unchanged.
    chunks = _chunks.Collector(_chunks.scheduler_of(engine.llm))

    def _collect(outs) -> None:
        for o in outs:
            for c in o.outputs:
                by_child.setdefault(o.request_id, {})[int(c.index)] = [int(t) for t in c.token_ids]
            if getattr(o, "num_cached_tokens", None) is not None:
                cached[o.request_id] = int(o.num_cached_tokens)
            if o.finished and o.request_id not in finish:
                fr = [getattr(c, "finish_reason", None) for c in o.outputs]
                finish[o.request_id] = {"finish_step": step_no[0], "finish_reason": str(fr[0]) if len(fr) == 1 else [str(x) for x in fr]}
        step_no[0] += 1
        chunks.on_step(outs)
    rs = [_R(r, i) for i, r in enumerate(requests)]
    params_for = lambda r: sampling_params({"max_tokens": r.max_tokens, "seed": r.seed, "n": r.n, **r.params}, sampling)  # noqa: E731
    # one loop for every profile: with all arrive_steps 0 `run_requests_arrivals` adds every request before the first step, and
    # its `engine_steps` counts forwards (the observer's / gates' unit) rather than step() iterations (vLLM V1 returns the previous
    # forward's sample, so iterations = forwards + 1); the raw iteration count is its `step_iterations`
    outs, timing = run_requests_arrivals(engine.llm, rs, params_for, on_step=_collect)
    timing["arrivals"] = any(r.arrive_step > 0 for r in rs)
    try:
        # the attribution receives the RESOLVED async_scheduling as the lag of record (the same fact tokens.json's
        # `lag.value` and the RunHeader attest): a schedule-delta inference that contradicts it never rewrites the loop's step fields
        _asched = resolved_async_scheduling(engine.llm)
        _chunks.attribute(timing, chunks, {r.request_id: len(r.prompt_token_ids) for r in rs},
                          resolved_lag=None if _asched is None else (1 if _asched else 0))
    except Exception as e:  # noqa: BLE001  -- attribution is a record, never a reason to lose the run
        timing["chunk_attribution"] = {"source": chunks.source, "error": repr(e)[:300], "consistent": False}
    if static:
        timing["static_schedule"] = static_schedule.check(requests, timing, cached_tokens=cached,
                                                          preempted=static_schedule.preemptions(chunks.computed))
    tokens = {}
    for rid, o in outs.items():
        kids = by_child.get(rid) or {0: [int(t) for t in o.outputs[0].token_ids]}
        tokens[rid] = kids[min(kids)]                    # the parent key keeps child 0 (n == 1: the request itself; unchanged shape)
        if len(kids) > 1:
            for i, toks in sorted(kids.items()):
                tokens[f"{i}_{rid}"] = toks
    return tokens, timing


#: Per-request sampling fields a warm-up request keeps from the request it stands in for.  The collector stages per-request sampler
#: tensors only under some of them (`runner.sampler/seed` only for an explicitly seeded batch), so a bounded warm-up that dropped them
#: learned plans every seeded record step overran (SmolLM2-135M Gumbel sm_120: +8 B past a chunk boundary = +256 B, failed closed).
THROWAWAY_SAMPLING_FIELDS = ("seed", "n", "variant", "temperature", "top_p", "top_k")
#: The warm-up prompts' token seed for a single warm-up pass.
WARMUP_SEED = 0x5EED_7A9B


def throwaway_requests(engine: Engine, like: list[dict], seed: int) -> list[dict]:
    """Warm-up prompts of the same lengths, random tokens (prefix caching is off; they share no block anyway), with the same per-request
    sampling fields (`THROWAWAY_SAMPLING_FIELDS`): the warm-up stages the same layout as the run it stands in for."""
    from verity_vllm.commit import challenge

    vocab = int(getattr(engine.llm.llm_engine.vllm_config.model_config.hf_config, "vocab_size", 32000))
    rng = challenge.generator(seed)
    out = []
    for i, r in enumerate(like):
        w = {"request_id": f"warmup-{i}", "prompt_token_ids": rng.integers(3, vocab - 1, size=len(r["prompt_token_ids"])).tolist(),
             "max_tokens": int(r["max_tokens"]), "arrive_step": int(r.get("arrive_step") or 0)}   # same schedule shape (D87)
        w.update({k: r[k] for k in THROWAWAY_SAMPLING_FIELDS if k in r})
        out.append(w)
    return out
