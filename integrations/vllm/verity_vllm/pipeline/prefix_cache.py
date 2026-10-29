"""The `verity-vllm prefix-cache` command: `gen` writes the controlled workload pair below; `extract` (the linkage record) and
`evidence` (warm run vs cold control) run `observe/prefix_cache.py`.

The controlled workload pair (`gen`):  `cov_b0_prefix_c256` (warm) / `cov_b0_prefix_c256_cold` (control)
       request A (cold)  = P (PREFIX_TOKENS shared prefix) + X (CONT_TOKENS), arrive_step 0
       request B (warm)  = P + Y (CONT_TOKENS, Y != X),      arrive_step WARM_ARRIVE_STEP (> 0: A's prefill has completed,
                                                             A is still decoding when B is admitted)
   identical requests in both files; the warm file declares `engine.args: {"enable_prefix_caching": true}`, the cold one
   `false`.  With block_size 16 the shared prefix is exactly PREFIX_TOKENS // 16 full blocks, so a warm B computes only
   CONT_TOKENS prompt rows (`StepStart.num_computed[B] == PREFIX_TOKENS` at its first step).  The format is
   `pipeline/coverage_workloads.py`'s (a row file, `config.RowConfig`, + the D87 keys); that module is imported read-only.

    verity-vllm prefix-cache gen --tokenizer <dir-or-tokenizer.json> [--out-dir verity_vllm/engine/profiles]
    verity-vllm prefix-cache extract <run-dir | capture/log.jsonl.gz> [--out <path>] [--instances instances.jsonl]
"""
from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.config import VLLM, WORKLOAD_SCHEMA, checkpoint_entries, option, workload_arg
from verity_vllm.observe import prefix_cache as PC

#: shared prefix length (tokens); a multiple of vLLM's block_size 16 so every prefix block is a full, cacheable block
PREFIX_TOKENS = 128
#: continuation length (tokens) of each request after the shared prefix
CONT_TOKENS = 40
#: generated tokens per request
MAX_TOKENS = 10
#: engine step before which the warm request is admitted (A prefills at step 0 and decodes at 1..3; B's prefill runs at step 4
#: next to A's decode row)
WARM_ARRIVE_STEP = 4
#: vLLM's default block size for the pinned profile (the log's `StepStart.block_size` is the realized value)
BLOCK_SIZE_ASSUMED = 16
PROFILE_LABEL = "B2-prefix"
CTX_LABEL = "C256"
WORKLOAD_STEM = "cov_b0_prefix_c256"


def _paragraph_tokens(tok: Any, paras: list[dict], start: int, n: int) -> tuple[list[int], list[str]]:
    """>= n tokens of consecutive corpus paragraphs from `start` (the generator's rule), truncated to n."""
    ids: list[int] = []
    used: list[str] = []
    j = start
    while len(ids) < n:
        p = paras[j % len(paras)]
        ids += tok.encode((" " if ids else "") + p["text"])
        used.append(f"{p['source']} ¶{p['index']}")
        j += 1
    return ids[:n], used


def prefix_requests(tok: Any, *, case: str = "B0", seed: int, prefix_tokens: int = PREFIX_TOKENS, cont_tokens: int = CONT_TOKENS,
                    max_tokens: int = MAX_TOKENS, warm_arrive_step: int = WARM_ARRIVE_STEP) -> list[dict]:
    """The two requests: P+X (arrive 0) and P+Y (arrive `warm_arrive_step`), P shared, X != Y.  Text from the coverage corpus at
    seeded paragraph starts (numpy RNG like the generator), tokenized with the served model's tokenizer."""
    import numpy as np

    from verity_vllm.pipeline import coverage_workloads as gw

    paras = gw.corpus_paragraphs()
    rng = np.random.default_rng([seed, 0xF1E, 0xC0DE])
    starts = [int(x) for x in rng.choice(len(paras), size=3, replace=False)]
    P, used_p = _paragraph_tokens(tok, paras, starts[0], prefix_tokens)
    X, used_x = _paragraph_tokens(tok, paras, starts[1], cont_tokens)
    Y, used_y = _paragraph_tokens(tok, paras, starts[2], cont_tokens)
    if X == Y:
        raise RuntimeError("continuations coincide; change the seed")
    reqs = []
    for i, (cont, used, arrive, role) in enumerate(((X, used_x, 0, "cold: computes the shared prefix"),
                                                     (Y, used_y, warm_arrive_step, "warm: reuses the shared prefix's KV blocks"))):
        ids = P + cont
        rid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"verity-coverage/{case}/{PROFILE_LABEL}/{CTX_LABEL}/{seed}/{i}"))
        reqs.append({"request_id": rid, "index": i, "prompt_len": len(ids), "max_tokens": max_tokens, "arrive_step": arrive,
                     "prompt_token_ids": ids, "prompt_text": tok.decode(ids),
                     "prefix_role": role, "shared_prefix_tokens": prefix_tokens,
                     "origin": {"corpus": gw.CORPUS.name, "prefix_paragraphs": used_p, "continuation_paragraphs": used}})
    return reqs


def prefix_workload_docs(tok: Any, *, case: str = "B0", model: str, tokenizer_revision: str | None = None,
                         seed: int, block_size: int = BLOCK_SIZE_ASSUMED) -> dict[str, dict]:
    """{stem: doc} for the warm and the cold file (identical requests; only `engine.args` and the labels differ)."""
    from verity_vllm.pipeline import coverage_workloads as gw

    reqs = prefix_requests(tok, case=case, seed=seed)
    sim = gw.simulate_schedule(reqs)
    shared = reqs[0]["shared_prefix_tokens"]
    if shared % block_size:
        raise ValueError(f"shared prefix {shared} is not a multiple of block_size {block_size}")
    out: dict[str, dict] = {}
    for stem, warm in ((WORKLOAD_STEM, True), (f"{WORKLOAD_STEM}_cold", False)):
        expect = {"cached_blocks_if_hit": shared // block_size if warm else 0,
                  "warm_prefill_rows": (reqs[1]["prompt_len"] - shared) if warm else reqs[1]["prompt_len"],
                  "cold_prefill_rows": reqs[0]["prompt_len"],
                  "num_computed_at_first_step": {reqs[0]["request_id"]: 0, reqs[1]["request_id"]: shared if warm else 0}}
        note = (f"prefix caching {'ON' if warm else 'OFF (control)'}: request 0 = P({shared}) + X({reqs[0]['prompt_len'] - shared}) at step 0, "
                f"request 1 = P + Y({reqs[1]['prompt_len'] - shared}) at step {reqs[1]['arrive_step']} (after request 0's prefill; request 0 still "
                f"decoding); max_tokens {reqs[0]['max_tokens']} each => {sim['expected_engine_steps']} engine steps if both run to max_tokens.  "
                + (f"Expected: request 1's first StepStart has num_computed == {shared} and {expect['warm_prefill_rows']} prompt rows; the "
                   f"{shared // block_size} prefix blocks in its block table are request 0's step-0 blocks.  "
                   if warm else f"Expected: request 1 computes all {reqs[1]['prompt_len']} prompt rows at its first step.  ")
                + "Greedy.  The control run's realized step count is the gates' reference.")
        out[stem] = {"schema": WORKLOAD_SCHEMA, "model": model, "case": case,
                     "engine": {"revision": tokenizer_revision, "vllm": VLLM, "args": {"enable_prefix_caching": bool(warm)}}, "seed": 0,
                     "variants": {"greedy": gw.VARIANTS["greedy"]}, "seeds": [],
                     "sampling": {"variant": "greedy", "seed": None, "n": 1},
                     "serving_profile": PROFILE_LABEL, "context_class": CTX_LABEL, "length_seed": seed, "prompt_seed": seed, "replicate": 0,
                     "feature": {"prefix_caching": warm, "shared_prefix_tokens": shared, "block_size_assumed": block_size,
                                 "control_of": WORKLOAD_STEM if not warm else None, "expect": expect},
                     # `depends_on` is recorded in the pinned workload pair (workloads/, the gen test's sha256): it keeps naming the
                     # constants' module as those files do
                     "length_rule": {"prompt_len": f"{shared} shared + {reqs[0]['prompt_len'] - shared} continuation", "max_tokens": str(reqs[0]["max_tokens"]),
                                     "arrive_step": f"0 | {reqs[1]['arrive_step']}", "depends_on": "constants of verity_vllm.observe.prefix_cache"},
                     "expected_engine_steps": sim["expected_engine_steps"], "schedule_if_full_length": sim,
                     "tokenizer_note": f"prompt_token_ids from {tok.path.name} sha256 {tok.sha256[:16]} (tokenizers, add_special_tokens=False), "
                                       f"natural text from {gw.CORPUS.name}; prompt_text is the decode of the ids (informational)",
                     "prompt_lengths": sorted({r["prompt_len"] for r in reqs}), "max_tokens": sorted({r["max_tokens"] for r in reqs}),
                     "requests": reqs, "note": note}
    return out


def write_workloads(tok: Any, out_dir: Path, **kw: Any) -> list[Path]:
    docs = prefix_workload_docs(tok, **kw)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for stem, doc in docs.items():
        p = out_dir / f"workload_{stem}.json"
        p.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        paths.append(p)
    return paths


@dataclass(frozen=True, kw_only=True)
class GenOptions:
    "write the warm/cold workload pair"

    tokenizer: str = option("--tokenizer")
    tokenizer_revision: str | None = option("--tokenizer-revision", default=None)
    case: str = option("--case", default="B0")
    model: str | None = option("--model", default=None, help="HF repo of the served model (default: the case's pin in checkpoints.json)")
    seed: int = option("--seed", type=int, default=0)
    out_dir: str = option("--out-dir", default=str(Path(__file__).resolve().parents[1] / "engine" / "profiles"))
    cmd: str = field(default="gen", init=False)


@dataclass(frozen=True, kw_only=True)
class ExtractOptions:
    "write capture/prefix_linkage.json for a run dir or a log"

    run_or_log: str = option("run_or_log")
    out: str | None = option("--out", default=None)
    instances: str | None = option("--instances", default=None)
    workload: str | None = option("--workload", type=workload_arg, default=None)
    cmd: str = field(default="extract", init=False)


@dataclass(frozen=True, kw_only=True)
class EvidenceOptions:
    "warm vs cold comparison from two run dirs -> <warm>/prefix_evidence.json + text"

    warm_dir: str = option("warm_dir")
    cold_dir: str | None = option("cold_dir", nargs="?", default=None)
    out: str | None = option("--out", default=None)
    cmd: str = field(default="evidence", init=False)


COMMANDS = {"gen": GenOptions, "extract": ExtractOptions, "evidence": EvidenceOptions}


def main(a: GenOptions | ExtractOptions | EvidenceOptions) -> int:
    if a.cmd == "evidence":
        doc = PC.evidence(Path(a.warm_dir), Path(a.cold_dir) if a.cold_dir else None, out=Path(a.out) if a.out else None)
        print(PC.evidence_text(doc))
        return 0
    if a.cmd == "gen":
        from verity_vllm.pipeline import coverage_workloads as gw

        tok = gw.Tok(a.tokenizer)
        model = a.model or next((c["repo"] for c in checkpoint_entries() if c.get("role") == a.case), None)
        if model is None:
            raise SystemExit(f"case {a.case!r}: no checkpoints.json pin has that role; pass --model")
        for p in write_workloads(tok, Path(a.out_dir), case=a.case, model=model, tokenizer_revision=a.tokenizer_revision, seed=a.seed):
            d = json.loads(p.read_text())
            print(f"{p.name}: {d['note'][:240]}", file=sys.stderr)
        return 0
    rec, dst = PC.extract_run(Path(a.run_or_log), out=Path(a.out) if a.out else None,
                           instances=Path(a.instances) if a.instances else None, workload=Path(a.workload) if a.workload else None)
    print(PC.summarize(rec))
    print(f"-> {dst}")
    return 0
