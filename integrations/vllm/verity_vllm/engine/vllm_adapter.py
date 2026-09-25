"""vLLM adapter (runtime side): load a workload, and drive its requests through a built engine.

The engine is built in `engine.build`, the capture and its run header are in `engine.capture`, the run's profile manifest and
versions in `engine.run_facts`, and code identity in `engine.code_identity`; the stages reach all of them through this module's
names.  `load_workload` and what it calls stay in this file: tests execute them from its source, without importing torch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm.engine.arrivals import run_requests_arrivals  # [coverage D87] step-indexed arrivals (torch-free module, unit-tested)
from verity_vllm.observe import chunks as _chunks  # [chunked 2026-09-17] prefill-chunk / token-step attribution (torch-free, unit-tested)
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
from verity_vllm.engine.pinned import PinnedArena, spanned_bytes, storage_bytes_view  # noqa: F401
from verity_vllm.engine.capture import SCHEMA, Capture, ObserverConfig, RunHeader, make_header  # noqa: F401


def load_workload(path: str | Path, max_tokens: int | None = None) -> dict:
    """The workload file (`workloads/workload_32x16_1req.json`, `workload_qwen15_32x16_1req.json`) ->
    {"requests": [...], "sampling": {...}, "case": <checkpoint role or None>, "accepted": {request_id: accepted token ids}}.
    The accepted tokens come from the file's `accepted_token_ids` per request (M6) or `ACCEPTED_TOKENS` of the case."""
    doc = json.loads(Path(path).read_text())
    # [GEN sampling] an optional top-level `sampling` block selects the variant and the request seeds / fan-out:
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
    for r in doc["requests"]:
        rs = r.get("seed", seed)
        req = {"request_id": r["request_id"], "prompt_token_ids": [int(t) for t in r["prompt_token_ids"]],
               "max_tokens": int(max_tokens if max_tokens is not None else r["max_tokens"]),
               "seed": None if rs is None else int(rs), "n": int(r.get("n", n)),
               # [coverage D87] serving profile B8-mixed: the engine step at which the request is admitted (0 = with the first
               # step, the historical behaviour).  Step-indexed, not wall-clock, so the observer and control runs see the same
               # batch membership at every step.
               "arrive_step": int(r.get("arrive_step") or 0)}
        if r.get("variant") is not None:
            # [ov-sampling] a request may pin ANOTHER variant (mixed batch: one greedy row next to a sampled row); the
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
    return {"requests": reqs, "sampling": sampling, "source": str(path), "model": doc.get("model"), "case": doc.get("case"),
            "accepted": accepted,
            # [coverage D87] declared scheduler knobs the workload cannot run without (e.g. max_num_batched_tokens for an
            # unchunked 8-request prefill); run_config passes them as --engine-arg, an explicit CLI value wins
            "engine_args_required": dict(doc.get("engine_args_required") or {}),
            "serving_profile": doc.get("serving_profile"), "context_class": doc.get("context_class"),
            "expected_engine_steps": doc.get("expected_engine_steps"),
            # [R15 EXEC-02] the sweep row this workload belongs to and the execution it declares (row-id suffix / explicit field): the
            # GPU stages hand `execution` to build_engine so enforce_eager is the row's, never the harness's literal (COORD M-0063 (2))
            "row_id": (doc.get("sweep") or {}).get("row_id") if isinstance(doc.get("sweep"), dict) else doc.get("row_id"),
            "execution": execution_of_workload(doc),
            # [R15 fp8] the row's declared TargetProfile overrides (`target` block, written by sweep/workload.py --target): the channel
            # `target_of_workload` documents ("else the workload manifest's target block") -- rc3 dropped it here, so every GPU stage
            # built its engine with target=None (engine ARGS still travelled through engine_args_required; environment knobs such as
            # VLLM_USE_DEEP_GEMM have no other channel, and the header carried no `profile.target`).  None when the row declares none.
            "target": dict(doc["target"]) if isinstance(doc.get("target"), dict) and doc.get("target") else None,
            "sha256": hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


#: The accepted record's generated tokens for the B0 workload's request (its file predates `accepted_token_ids`):
#: out/scale/e8/live/V5-N32x16-1req-bex0-inc-sc1/adapter/selective/ru/7cc7c3bb…/boundary.json
ACCEPTED_TOKENS: dict[str, list[int]] = {
    "7cc7c3bb-adf5-58cd-87fc-9f1b8e7c34d6": [359, 253, 1123, 1183, 282, 253, 1772, 338, 553, 7527, 288, 11492, 351, 260, 4340, 1357],
}


def execution_of_workload(workload: dict | None) -> str | None:
    """[R15 EXEC-02] The execution label a workload manifest declares: an explicit top-level `execution` (or `sweep.execution`) when the
    schema carries one, else the last `__` token of `sweep.row_id` (`…__greedy__bi-eager` -> 'bi-eager'); None for a workload that is not
    a sweep row (no row id: the caller's default applies and the header records `declared_execution: null`)."""
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

    seed = req.get("seed") if req.get("seed") is not None else sampling.get("seed")     # [GEN sampling] per-request seed / n
    # [ov-sampling] a request with its own variant (mixed batch) carries temperature / top_p / top_k; else the workload's
    return SamplingParams(temperature=float(req.get("temperature", sampling.get("temperature", 0.0))),
                          top_p=float(req.get("top_p", sampling.get("top_p", 1.0))),
                          top_k=int(req.get("top_k", sampling.get("top_k", 0))), max_tokens=int(req["max_tokens"]),
                          seed=None if seed is None else int(seed), n=int(req.get("n") or sampling.get("n") or 1),
                          logprobs=sampling.get("logprobs"))


def run_requests(engine: Engine, requests: list[dict], sampling: dict) -> tuple[dict[str, list[int]], dict]:
    """Add every request with *our* request ids and step the engine until all finish (driver.run_requests; with declared
    `arrive_step`s, `run_requests_arrivals`).  `timing["requests"]` carries the realized per-request steps (D87)."""
    class _R:
        def __init__(self, d: dict, order: int) -> None:
            self.request_id = d["request_id"]
            self.prompt_token_ids = list(d["prompt_token_ids"])
            self.max_tokens = int(d["max_tokens"])
            self.seed = d.get("seed")
            self.n = int(d.get("n") or 1)
            self.arrive_step = int(d.get("arrive_step") or 0)
            self._order = order
            # [ov-sampling] per-request variant (mixed batch): the request's own temperature / top_p / top_k, if pinned
            self.params = {k: d[k] for k in ("temperature", "top_p", "top_k") if k in d}

    # [GEN sampling] n > 1: vLLM forks child requests f"{index}_{request_id}" (v1/engine/parallel_sampling.py) — the ids the model
    # runner and the log see.  With the default CUMULATIVE output kind each step's RequestOutput carries only the children that produced
    # a token that step (the final one need not hold every child), so collect every CompletionOutput by (parent, index) across steps.
    by_child: dict[str, dict[int, list[int]]] = {}
    finish: dict[str, dict] = {}
    step_no = [0]
    # [chunked 2026-09-17, additive] prefill-chunk attribution: the in-process scheduler's per-request `num_computed_tokens` is read
    # after every step() (pure reads; nothing patched, nothing on the device), the returned tokens are counted per iteration; after
    # the run `chunks.attribute` adds prefill_steps / prefill_chunks / partial_prefill_steps / first_token_step /
    # token_steps to every `timing["requests"][rid]` and `timing["chunk_attribution"]` run-wide.  Every existing key is unchanged.
    chunks = _chunks.Collector(_chunks.scheduler_of(engine.llm))

    def _collect(outs) -> None:
        for o in outs:
            for c in o.outputs:
                by_child.setdefault(o.request_id, {})[int(c.index)] = [int(t) for t in c.token_ids]
            if o.finished and o.request_id not in finish:
                fr = [getattr(c, "finish_reason", None) for c in o.outputs]
                finish[o.request_id] = {"finish_step": step_no[0], "finish_reason": str(fr[0]) if len(fr) == 1 else [str(x) for x in fr]}
        step_no[0] += 1
        chunks.on_step(outs)
    rs = [_R(r, i) for i, r in enumerate(requests)]
    params_for = lambda r: sampling_params({"max_tokens": r.max_tokens, "seed": r.seed, "n": r.n, **r.params}, sampling)  # noqa: E731
    # [D87] one loop for every profile: with all arrive_steps 0 `run_requests_arrivals` is the driver's loop, and its `engine_steps`
    # counts forwards (the observer's / gates' unit) rather than step() iterations (vLLM V1 returns the previous forward's sample, so
    # iterations = forwards + 1); verity_vllm.engine.engine_driver.run_requests stays available for callers that want the raw iteration count
    outs, timing = run_requests_arrivals(engine.llm, rs, params_for, on_step=_collect)
    timing["arrivals"] = any(r.arrive_step > 0 for r in rs)
    try:
        # [rev F-r17-16 / R17-1] the attribution receives the RESOLVED async_scheduling as the lag of record (the same fact tokens.json's
        # `lag.value` and the RunHeader attest): a schedule-delta inference that contradicts it never rewrites the loop's step fields
        _asched = resolved_async_scheduling(engine.llm)
        _chunks.attribute(timing, chunks, {r.request_id: len(r.prompt_token_ids) for r in rs},
                          resolved_lag=None if _asched is None else (1 if _asched else 0))
    except Exception as e:  # noqa: BLE001  -- attribution is a record, never a reason to lose the run
        timing["chunk_attribution"] = {"source": chunks.source, "error": repr(e)[:300], "consistent": False}
    tokens = {}
    for rid, o in outs.items():
        kids = by_child.get(rid) or {0: [int(t) for t in o.outputs[0].token_ids]}
        tokens[rid] = kids[min(kids)]                    # the parent key keeps child 0 (n == 1: the request itself; unchanged shape)
        if len(kids) > 1:
            for i, toks in sorted(kids.items()):
                tokens[f"{i}_{rid}"] = toks
    return tokens, timing


def throwaway_requests(engine: Engine, like: list[dict], seed: int = 0x5EED_7A9B) -> list[dict]:
    """Warm-up prompts of the same lengths, random tokens (prefix caching is off; they share no block anyway)."""
    vocab = int(getattr(engine.llm.llm_engine.vllm_config.model_config.hf_config, "vocab_size", 32000))
    rng = np.random.default_rng(seed)
    out = []
    for i, r in enumerate(like):
        out.append({"request_id": f"warmup-{i}", "prompt_token_ids": rng.integers(3, vocab - 1, size=len(r["prompt_token_ids"])).tolist(),
                    "max_tokens": int(r["max_tokens"]), "arrive_step": int(r.get("arrive_step") or 0)})   # same schedule shape (D87)
    return out
