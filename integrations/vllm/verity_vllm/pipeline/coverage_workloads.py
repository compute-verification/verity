"""bench/gen_coverage_workloads.py — the coverage campaign's workloads (D87): serving profile × context class, seeded lengths.

    verity-vllm coverage-workloads --case B0 --tokenizer <dir-or-tokenizer.json> \\
        [--profiles B1,B8-static,B8-mixed] [--ctx C256,C1024] [--seed 0] [--out-dir workloads]

Dimensions (out/gen/COVERAGE-20260917.md):
  serving profile   B1         one request, admitted at step 0
                    B8-static  eight requests admitted together at step 0 (max_num_seqs 8)
                    B8-mixed   eight requests with seeded arrival steps: request 0 at step 0, the others spread over the first
                               ~60 % of the run so requests enter and leave the active batch while others decode
  context class     C256       prompt_len ~ U{32..256},   max_tokens = max(2, prompt_len // 10)  (<= 25)
                    C1024      prompt_len ~ U{128..1024}, max_tokens = max(4, prompt_len // 10)  (<= 102)
The lengths depend on (profile, ctx, seed) ONLY — the same draws for every model, so the model dimension is isolated.
Prompts are natural text (verity_vllm/engine/profiles/corpus_coverage.json: three public-domain Gutenberg books), each request
starting at a different seeded paragraph, tokenized with the served model's tokenizer (add_special_tokens=False) and truncated to
the drawn prompt length.  `expected_engine_steps` is the upper bound max_r(arrive_r + max_tokens_r) (every request runs to
max_tokens; a request stopping at EOS earlier shortens the run — gates compare against the control run's realized steps, D87).
`engine.args.max_num_batched_tokens` keeps an 8-request prefill in ONE step (chunked prefill is its own feature column:
`--variant chunked=<N>` declares a small budget that FORCES several prefill chunks per request, suffix `_ck<N>`; see the block below).
`--variant dtype=float16` (the fp16 column): the same requests, plus `engine.args.dtype = "float16"` so the harness builds
`LLM(dtype="float16")` (file suffix `_f16`; `dtype` / `numerical_format` keys on the document).  Feature variants are `key=value`
items of one `--variant` (comma-separated or the flag repeated) and compose: `chunked=64,dtype=float16` -> suffix `_ck64_f16`.

Writes <out-dir>/workload_cov_<case-lower>_<profile>_<ctx>[_r<r>][_t1][_ck<N>][_f16].json (a row file, `config.RowConfig`, + the D87 keys).
"""
from __future__ import annotations

import hashlib
import json
import sys
import uuid
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from verity_vllm.commit import challenge as CH
from verity_vllm.config import CHECKPOINTS, VLLM, WORKLOAD_SCHEMA, WORKLOADS, option
from verity_vllm.engine.profiles.coverage_roles import ROLE_REPOS

CORPUS = resources.files("verity_vllm") / "engine" / "profiles" / "corpus_coverage.json"
SCHEMA = WORKLOAD_SCHEMA
PROFILES = ("B1", "B8-static", "B8-mixed")
CTX = {"C256": {"lo": 32, "cap": 256, "min_out": 2}, "C1024": {"lo": 128, "cap": 1024, "min_out": 4}}
#: max_num_batched_tokens that keeps the whole 8-request prefill in one engine step (+ decode rows)
BATCHED_TOKENS = {"C256": 2048 + 64, "C1024": 8192 + 64}


class Tok:
    """`tokenizers` tokenizer (tokenizer.json) with the decode side; `add_special_tokens=False` like the batch workloads."""

    def __init__(self, path: str | Path) -> None:
        from tokenizers import Tokenizer

        p = Path(path)
        if p.is_dir():
            p = p / "tokenizer.json"
        self.path = p
        self.tk = Tokenizer.from_file(str(p))
        self.sha256 = hashlib.sha256(p.read_bytes()).hexdigest()

    def encode(self, text: str) -> list[int]:
        return list(self.tk.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=False)


def draw_lengths(profile: str, ctx: str, seed: int) -> list[dict]:
    """The (prompt_len, max_tokens, arrive_step) draws of one (profile, ctx, seed) -- model independent."""
    c = CTX[ctx]
    n = 1 if profile == "B1" else 8
    rng = CH.generator([seed, PROFILES.index(profile), sorted(CTX).index(ctx), 0xC0FFEE])
    plens = [int(x) for x in rng.integers(c["lo"], c["cap"] + 1, size=n)]
    mts = [max(c["min_out"], pl // 10) for pl in plens]
    if profile == "B8-mixed":
        horizon = max(1, int(round(0.6 * (c["cap"] // 10))))            # arrivals over the first ~60 % of a full-length run
        arr = sorted(int(x) for x in rng.integers(1, horizon + 1, size=n - 1))
        arrive = [0] + arr
    else:
        arrive = [0] * n
    return [{"prompt_len": pl, "max_tokens": mt, "arrive_step": a} for pl, mt, a in zip(plens, mts, arrive)]


def simulate_schedule(reqs: list[dict]) -> dict:
    """Active set per step if every request runs to max_tokens: expected steps, peak concurrency, membership changes."""
    spans = [(r["arrive_step"], r["arrive_step"] + r["max_tokens"] - 1) for r in reqs]
    hi = max(b for _, b in spans)
    active = [frozenset(i for i, (a, b) in enumerate(spans) if a <= s <= b) for s in range(hi + 1)]
    changes = sum(1 for s in range(1, len(active)) if active[s] != active[s - 1])
    return {"expected_engine_steps": hi + 1, "peak_concurrent": max(len(x) for x in active),
            "membership_changes": changes, "steps_with_mixed_prefill_decode":
            sum(1 for s in range(1, len(active)) if any(a == s for a, _ in spans) and any(a < s <= b for a, b in spans))}


def corpus_paragraphs() -> list[dict]:
    doc = json.loads(CORPUS.read_text())
    out = []
    for b in doc["books"]:
        for i, p in enumerate(b["paragraphs"]):
            out.append({"text": p, "source": b["source"], "index": i})
    return out


def build_requests(tok: Tok, draws: list[dict], *, case: str, profile: str, ctx: str, seed: int, prompt_seed: int | None) -> list[dict]:
    """The requests: lengths from `draws` (a function of (profile, ctx, seed) only), text from the corpus chosen by `prompt_seed`
    (None: the length seed).  A replicate keeps the draws and changes the prompt seed -- same structural configuration, other prompts."""
    paras = corpus_paragraphs()
    ps = seed if prompt_seed is None else prompt_seed
    rng = CH.generator([ps, PROFILES.index(profile), sorted(CTX).index(ctx), 0xC0DE])
    starts = rng.choice(len(paras), size=len(draws), replace=False)
    reqs = []
    for i, (d, s0) in enumerate(zip(draws, starts)):
        ids: list[int] = []
        j = int(s0)
        used = []
        while len(ids) < d["prompt_len"]:
            p = paras[j % len(paras)]
            ids += tok.encode((" " if ids else "") + p["text"])
            used.append(f"{p['source']} ¶{p['index']}")
            j += 1
        ids = ids[: d["prompt_len"]]
        tag = f"{seed}/{i}" if ps == seed else f"{seed}/p{ps}/{i}"
        rid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"verity-coverage/{case}/{profile}/{ctx}/{tag}"))
        reqs.append({"request_id": rid, "index": i, "prompt_len": d["prompt_len"], "max_tokens": d["max_tokens"],
                     "arrive_step": d["arrive_step"], "prompt_token_ids": ids, "prompt_text": tok.decode(ids),
                     "origin": {"corpus": CORPUS.name, "paragraphs": used}})
    return reqs


VARIANTS = {"greedy": {"temperature": 0.0, "top_p": 1.0, "top_k": 0},
            "t1": {"temperature": 1.0, "top_p": 1.0, "top_k": 0}}       # temp=1 column: seeded sampling, no truncation

#: `--variant dtype=<vllm dtype>`: the numerical-format column (COVERAGE §1).  The workload then REQUIRES the engine to be built with
#: `LLM(dtype=<x>)` through `engine.args` (run_config passes it as `--engine-arg dtype="float16"`, `vllm_adapter.build_engine`
#: applies it after the profile's pinned `"dtype": "bfloat16"`, so the served activations / weights / KV cache are that dtype -- no
#: silent cast).  File suffix per dtype; only the dtypes listed here are accepted (bf16 is the default column, not a variant).
DTYPE_VARIANTS = {"float16": "f16", "half": "f16"}


# =========================================================================================================
# feature variant `--variant chunked=<max_num_batched_tokens>`
#
# vLLM V1 always has chunked prefill enabled, but the campaign's `engine.args.max_num_batched_tokens` (2048+64 /
# 8192+64) keeps every prefill in ONE engine step, so no coverage cell so far has exercised partial-prefill attention.  The
# chunked variant keeps the (profile, ctx, seed) length draws and the prompts of the base cell and FORCES several prefill
# chunks per request by declaring a small token budget: prompt 155 at budget 64 -> chunks 64 / 64 / 27 (three prefill
# forwards, the first two partial: vLLM samples a token at each and discards it), then max_tokens - 1 decode forwards.
# `expected_engine_steps` is simulated under the V1 scheduler policy (`simulate_schedule_chunked`): running requests first
# (1 token per decode, the remaining prompt for a prefill in progress), then waiting requests in arrival order with the
# remaining budget, no preemption; every request to max_tokens (an upper bound, like the unchunked simulation).  File suffix
# `_ck<N>`.  `parse_variant` is the one entry point: a sampling variant name (greedy, t1) or `key=value` feature variants,
# comma-separated; the fp16 owner adds `dtype=...` to FEATURE_VARIANTS in its own block -- keep this block self-contained.
# =========================================================================================================
def _dtype_variant(v: str) -> str:
    """[fp16] the value of `dtype=<x>`: one of `DTYPE_VARIANTS`, validated at parse time."""
    if v not in DTYPE_VARIANTS:
        raise KeyError(f"unknown dtype variant {v!r}; known: {sorted(DTYPE_VARIANTS)}")
    return v


FEATURE_VARIANTS = {"chunked": int,           # key -> value parser; the value is recorded under `feature_variant`
                    "dtype": _dtype_variant}  # [fp16 via integ] the numerical-format column; recorded under `dtype` / `numerical_format`
DEFAULT_MAX_NUM_SEQS = 8                      # cov_pod.sh runs every cell with --max-num-seqs 8


def parse_variant(spec: str) -> tuple[str, dict]:
    """`"greedy"` / `"t1"` -> (name, {});  `"chunked=64"` (optionally `"t1,chunked=64"`) -> (sampling variant, {"chunked": 64})."""
    samp = "greedy"
    feats: dict = {}
    for part in (spec or "greedy").split(","):
        part = part.strip()
        if not part:
            continue
        k, sep, v = part.partition("=")
        if sep:
            if k not in FEATURE_VARIANTS:
                raise KeyError(f"unknown feature variant {k!r}; known: {sorted(FEATURE_VARIANTS)}")
            feats[k] = FEATURE_VARIANTS[k](v)
        elif part in VARIANTS:
            samp = part
        else:
            raise KeyError(f"unknown sampling variant {part!r}; known: {sorted(VARIANTS)}")
    return samp, feats


def variant_suffix(samp: str, feats: dict) -> str:
    """File-name suffix of a variant: `_t1`, `_ck64`, `_t1_ck64`, `_f16`, `_ck64_f16`; empty for plain greedy."""
    s = f"_{samp}" if samp != "greedy" else ""
    if "chunked" in feats:
        s += f"_ck{int(feats['chunked'])}"
    if feats.get("dtype"):
        s += f"_{DTYPE_VARIANTS[feats['dtype']]}"       # [fp16] `_f16`, after the chunked suffix
    return s


def simulate_schedule_chunked(reqs: list[dict], budget: int, max_num_seqs: int = DEFAULT_MAX_NUM_SEQS) -> dict:
    """The vLLM V1 schedule of `reqs` under `max_num_batched_tokens = budget` if every request runs to max_tokens.

    Per step: RUNNING requests in admission order take 1 token (decode) or min(remaining prompt, budget) (a prefill in progress);
    then WAITING requests in arrival order take min(prompt_len, remaining budget) while the budget and `max_num_seqs` allow (a
    request whose first chunk does not fit waits).  A request samples its first token at the step that completes its prompt (the
    engine discards the samples of the earlier, partial chunks), then one token per step.  Arrivals follow `arrive_step` with the
    idle-gap rule of `arrivals.run_requests_arrivals` (an idle engine admits the next arrival at once).  Returns the step count and
    the per-request chunk plan, plus the keys of `simulate_schedule` computed on the realized spans."""
    n = len(reqs)
    order = sorted(range(n), key=lambda i: (int(reqs[i]["arrive_step"]), i))
    pending = list(order)
    waiting: list[int] = []
    running: list[int] = []
    computed = [0] * n
    generated = [0] * n
    chunks: list[list[dict]] = [[] for _ in range(n)]
    first_token_step: list[int | None] = [None] * n
    finish_step: list[int | None] = [None] * n
    admitted_step: list[int | None] = [None] * n
    step = 0
    steps_with_partial = 0
    steps_mixed = 0
    while pending or waiting or running:
        while pending and int(reqs[pending[0]]["arrive_step"]) <= step:
            i = pending.pop(0)
            waiting.append(i)
            admitted_step[i] = step
        if not waiting and not running:
            i = pending.pop(0)                      # idle gap: the next arrival is brought forward
            waiting.append(i)
            admitted_step[i] = step
        left = int(budget)
        sched: list[tuple[int, int]] = []
        for i in list(running):
            if left <= 0:
                break
            plen = int(reqs[i]["prompt_len"])
            t = min(plen - computed[i], left) if computed[i] < plen else 1
            sched.append((i, t))
            left -= t
        for i in list(waiting):
            if left <= 0 or len(running) >= max_num_seqs:
                break
            t = min(int(reqs[i]["prompt_len"]), left)
            waiting.remove(i)
            running.append(i)
            sched.append((i, t))
            left -= t
        if not sched:
            raise RuntimeError(f"chunked schedule stalled at step {step} (budget {budget}, max_num_seqs {max_num_seqs})")
        any_prefill = any_decode = any_partial = False
        for i, t in sched:
            plen = int(reqs[i]["prompt_len"])
            if computed[i] < plen:
                before = computed[i]
                computed[i] += t
                complete = computed[i] >= plen
                chunks[i].append({"step": step, "tokens": t, "computed_before": before, "prompt_complete": complete})
                any_prefill = True
                any_partial = any_partial or not complete
                if complete:
                    generated[i] = 1
                    first_token_step[i] = step
            else:
                computed[i] += t
                generated[i] += 1
                any_decode = True
            if generated[i] >= int(reqs[i]["max_tokens"]):
                finish_step[i] = step
                running.remove(i)
        steps_with_partial += 1 if any_partial else 0
        steps_mixed += 1 if (any_prefill and any_decode) else 0
        step += 1
    spans = [(admitted_step[i], finish_step[i]) for i in range(n)]
    active = [frozenset(i for i, (a, b) in enumerate(spans) if a <= s <= b) for s in range(step)]
    return {"expected_engine_steps": step, "peak_concurrent": max(len(x) for x in active),
            "membership_changes": sum(1 for s in range(1, step) if active[s] != active[s - 1]),
            "steps_with_mixed_prefill_decode": steps_mixed, "steps_with_partial_prefill": steps_with_partial,
            "max_num_batched_tokens": int(budget), "max_num_seqs": int(max_num_seqs),
            "prefill_chunks": [[c["tokens"] for c in chunks[i]] for i in range(n)],
            "partial_prefill_steps": [[c["step"] for c in chunks[i] if not c["prompt_complete"]] for i in range(n)],
            "first_token_step": first_token_step, "finish_step": finish_step, "admitted_step": admitted_step,
            "policy": "vLLM V1: running first (1 decode token / remaining prompt), then waiting in arrival order with the remaining "
                      "budget; first token sampled at the step that completes the prompt; no preemption; every request to max_tokens"}
# ========================================================================================= end of the chunked block


def workload_doc(*, case: str, model: str, tok: Tok, profile: str, ctx: str, seed: int, tokenizer_revision: str | None,
                 sampling_seed: int | None, replicate: int = 0, variant: str = "greedy", chunked: int | None = None,
                 dtype: str | None = None) -> dict:
    """replicate r > 0: same length draws (same structural configuration), prompts from prompt seed seed + 1000 r.  variant "t1":
    the temp=1 column -- every request sampled at temperature 1.0 with `sampling_seed` (vLLM's per-request seeded sampler).
    chunked = N: the chunked-prefill feature variant (max_num_batched_tokens N forces several prefill chunks; see the block above).
    dtype (`DTYPE_VARIANTS`): the numerical-format column -- `engine.args.dtype` makes the harness build `LLM(dtype=...)`;
    the requests (ids, token ids, lengths, arrivals) are identical to the bf16 workload of the same (profile, ctx, seed, replicate).
    The two feature variants compose: `chunked=64,dtype=float16` is one workload with both keys, suffix `_ck64_f16`."""
    draws = draw_lengths(profile, ctx, seed)
    prompt_seed = seed + 1000 * replicate if replicate else None
    reqs = build_requests(tok, draws, case=case, profile=profile, ctx=ctx, seed=seed, prompt_seed=prompt_seed)
    sim = simulate_schedule(reqs)
    eng: dict = {}
    if profile != "B1":
        eng["max_num_batched_tokens"] = BATCHED_TOKENS[ctx]
    if variant not in VARIANTS:
        raise KeyError(f"unknown sampling variant {variant!r}; known: {sorted(VARIANTS)}")
    if dtype is not None:
        if dtype not in DTYPE_VARIANTS:
            raise KeyError(f"unknown dtype variant {dtype!r}; known: {sorted(DTYPE_VARIANTS)}")
        eng["dtype"] = dtype
    samp = {"variant": variant, "seed": (sampling_seed if variant != "greedy" else None), "n": 1}
    # [chunked] the feature variant replaces the one-step-prefill budget and the schedule simulation; declared as its own key
    feature: dict = {}
    if chunked is not None:
        eng = {"max_num_batched_tokens": int(chunked), "enable_chunked_prefill": True, **({"dtype": dtype} if dtype is not None else {})}
        sim = simulate_schedule_chunked(reqs, int(chunked), DEFAULT_MAX_NUM_SEQS)
        feature = {"chunked": int(chunked)}
    note = (f"coverage D87: {profile} / {ctx}, seed {seed}{f', replicate {replicate} (prompt seed {prompt_seed})' if replicate else ''}"
            f"{f', dtype {dtype} (engine.args.dtype; the same requests as the bf16 workload)' if dtype else ''}: "
            f"{len(reqs)} request(s), prompt lens {[r['prompt_len'] for r in reqs]}, "
            f"max_tokens {[r['max_tokens'] for r in reqs]}, arrive steps {[r['arrive_step'] for r in reqs]}; every request to max_tokens => "
            f"{sim['expected_engine_steps']} engine steps, peak {sim['peak_concurrent']} concurrent, {sim['membership_changes']} batch-membership changes, "
            f"{sim['steps_with_mixed_prefill_decode']} steps with a prefill next to decodes.  "
            f"{'Greedy.' if variant == 'greedy' else f'Sampling variant {variant} {VARIANTS[variant]} seed {sampling_seed}.'}  "
            + (f"CHUNKED PREFILL: max_num_batched_tokens {chunked} (enable_chunked_prefill) forces prefill chunks "
               f"{sim['prefill_chunks']} (partial-prefill steps {sim['partial_prefill_steps']}: vLLM samples and discards a token at each), "
               f"first served token at steps {sim['first_token_step']}.  " if chunked is not None else "")
            + "The control run's realized step count is the gates' reference (EOS may end a request early).")
    return {"schema": SCHEMA, "model": model, "case": case, "engine": {"revision": tokenizer_revision, "vllm": VLLM, "args": eng}, "seed": 0,
            "variants": {"greedy": VARIANTS["greedy"], **({variant: VARIANTS[variant]} if variant != "greedy" else {})}, "seeds": [],
            "sampling": samp,
            "serving_profile": profile, "context_class": ctx, "length_seed": seed, "prompt_seed": prompt_seed if replicate else seed,
            "replicate": replicate,
            **({"feature_variant": feature} if feature else {}),
            **({"dtype": dtype, "numerical_format": DTYPE_VARIANTS[dtype]} if dtype else {}),     # the fp16 column (COVERAGE §1); absent = bf16
            "length_rule": {"prompt_len": f"U{{{CTX[ctx]['lo']}..{CTX[ctx]['cap']}}}", "max_tokens": f"max({CTX[ctx]['min_out']}, prompt_len // 10)",
                            "arrive_step": "0 | sorted U{1..round(0.6*cap/10)} (B8-mixed)", "depends_on": "(profile, ctx, seed) only"},
            "expected_engine_steps": sim["expected_engine_steps"], "schedule_if_full_length": sim,
            "tokenizer_note": f"prompt_token_ids from {tok.path.name} sha256 {tok.sha256[:16]} (tokenizers, add_special_tokens=False), "
                              f"natural text from {CORPUS.name}, truncated to prompt_len; prompt_text is the decode of the ids (informational)",
            "prompt_lengths": sorted({r["prompt_len"] for r in reqs}), "max_tokens": sorted({r["max_tokens"] for r in reqs}),
            "requests": reqs, "note": note}


@dataclass(frozen=True, kw_only=True)
class Options:
    case: str = option("--case", help="checkpoint role (B0, B1, ...: a role of checkpoints.json)")
    model: str | None = option("--model", default=None,
                               help="HF repo of the served model (default: from checkpoints.json by role)")
    tokenizer: str = option("--tokenizer", help="HF snapshot dir or tokenizer.json of the served model")
    tokenizer_revision: str | None = option("--tokenizer-revision", default=None)
    profiles: str = option("--profiles", default=",".join(PROFILES))
    ctx: str = option("--ctx", default=",".join(sorted(CTX)))
    seed: int = option("--seed", type=int, default=0, help="length seed (the draws depend on (profile, ctx, seed) only)")
    out_dir: str = option("--out-dir", default=str(WORKLOADS))
    checkpoints: str = option("--checkpoints", default=str(CHECKPOINTS))
    replicate: int = option("--replicate", type=int, default=0,
                            help="r > 0: same length draws, prompts from prompt seed seed+1000r; file suffix _r<r>")
    variant: list[str] = option("--variant", action="append", default=[],
                                help="sampling variant ('greedy' | 't1' = temperature 1.0, file suffix _t1) and/or feature variants key=value, "
                                f"known: {sorted(FEATURE_VARIANTS)} (e.g. 'chunked=64' -> suffix _ck64, 'dtype=float16' -> suffix _f16: "
                                "engine.args.dtype = float16); comma-separated or repeated; see parse_variant")
    sampling_seed: int = option("--sampling-seed", type=int, default=20260917,
                                help="per-request sampler seed for a non-greedy variant")
    dry_run: bool = option("--dry-run", default=False, action="store_true", help="print the draws only")


def checkpoint_repo(path: str | Path, role: str) -> str | None:
    """The repo of the first checkpoint of `role` in a checkpoints.json; None when the file is unreadable, isn't the
    `{"checkpoints": [{...}, ...]}` form up to that entry, or that entry carries no repo."""
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    cks = doc.get("checkpoints") if isinstance(doc, dict) else None
    for c in cks if isinstance(cks, list) else []:
        if not isinstance(c, dict):
            return None
        if c.get("role") == role:
            return c.get("repo")
    return None


def main(a: Options) -> int:
    # [chunked] one parse of --variant: the sampling variant name + the feature variants (fp16's dtype= lands in the same dict);
    # [integ] the flag is repeatable (fp16's calling style) -- the values are joined and parsed once
    samp_variant, feats = parse_variant(",".join(a.variant) if a.variant else "greedy")
    chunked = feats.get("chunked")
    dtype = feats.get("dtype")

    model = a.model
    if model is None:
        model = checkpoint_repo(a.checkpoints, a.case)
    if model is None:
        model = ROLE_REPOS.get(a.case)
    if a.dry_run:
        for prof in a.profiles.split(","):
            for ctx in a.ctx.split(","):
                d = draw_lengths(prof, ctx, a.seed)
                sim = simulate_schedule(d) if chunked is None else simulate_schedule_chunked(d, chunked)   # [chunked]
                print(prof, ctx, [(r["prompt_len"], r["max_tokens"], r["arrive_step"]) for r in d], sim)
        return 0
    tok = Tok(a.tokenizer)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for prof in a.profiles.split(","):
        for ctx in a.ctx.split(","):
            doc = workload_doc(case=a.case, model=model or "unknown", tok=tok, profile=prof, ctx=ctx, seed=a.seed,
                               tokenizer_revision=a.tokenizer_revision, replicate=a.replicate, variant=samp_variant, sampling_seed=a.sampling_seed,
                               chunked=chunked, dtype=dtype)
            suffix = (f"_r{a.replicate}" if a.replicate else "") + variant_suffix(samp_variant, feats)
            p = out_dir / f"workload_cov_{a.case.lower()}_{prof.lower()}_{ctx.lower()}{suffix}.json"
            p.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
            print(f"{p.name}: {doc['note'][:200]}", file=sys.stderr)
    return 0
