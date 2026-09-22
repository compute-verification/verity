"""[PAD-INT, stage 0] Find B0 prompts that terminate EARLY (EOS strictly before the decode horizon) under the
port's greedy batch-invariant profile, and emit a capture workload with early-terminating AND horizon requests.

Why: the certified record `INTPAD2-B0-20260909T013519Z` has four requests that all ended by `length`, so the
inactive (post-termination) path of the padded construction has never been exercised on a real production record
(GUIDANCE §44.2 PAD-INT stage 0; `docs/vllm-poc/correspondence-gap.md` G7).

Constraints the workload must keep so the record links to the certified program `Serve_v1{C=B0, LP=32, STEPS=15}`
(digest d4f4b1f1…): every prompt EXACTLY `--lp` (32) tokens, `max_tokens` = `--max-tokens` (16) → `STEPS = 15`.  Every
candidate is therefore `filler ‖ trigger`: a trigger suffix (few-shot / Q&A / "The capital of X is") after which a base
LM tends to emit `<|endoftext|>`, left-padded with corpus sentences and trimmed to exactly LP tokens.

Engine: the SAME construction as `verity_vllm_adapter.production` (`profile.apply_env()`, `profile.engine_kwargs(cp,
replay=False, enforce_eager=True)`, `SamplingParams(temperature=0, top_p=1, top_k=0, max_tokens, logprobs=2)`), so the
search's finish reasons are those of port `b0-eager` on this GPU.  One `generate` over all candidates (the profile is
batch-invariant; the capture itself is a separate run of `run_composed`).

Run ON THE POD (vLLM venv)::

    /workspace/padint-env/venv-cu129/bin/python scripts/padint_prompt_search.py \
        --checkpoints manifests/checkpoints.json --out /workspace/padint-results/prompt-search

Outputs: `candidates.jsonl` (every candidate: prompt, ids, output ids, finish_reason, EOS position, top-2 margins),
`search_summary.json`, and `workload-B0-padint.json` (schema `verity-vllm/workload/v1`; `--n-stop` early-terminating +
`--n-length` horizon requests, all LP tokens / max_tokens 16).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
for p in (str(POC), str(POC.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from verity_vllm_adapter import profile as prof  # noqa: E402

ENV_FLAGS = prof.apply_env()

from vllm import LLM, SamplingParams  # noqa: E402

from verity_vllm_sampler.workload import CORPUS, SCHEMA, VARIANTS, SEEDS, base_request_id  # noqa: E402

# Trigger suffixes: each ends where a base LM is likely to produce a short completion and then <|endoftext|>.
# `{eos}` is the tokenizer's EOS text (few-shot separators, as `workload.EOS_PROBE_TEMPLATES` does).
TRIGGERS: list[tuple[str, str]] = [
    ("fewshot-capital", "The capital of France is Paris.{eos}The capital of Germany is Berlin.{eos}The capital of Italy is"),
    ("fewshot-capital-2", "The capital of Spain is Madrid.{eos}The capital of Japan is Tokyo.{eos}The capital of Canada is"),
    ("fewshot-arith", "2 + 2 = 4{eos}3 + 5 = 8{eos}7 + 6 ="),
    ("fewshot-arith-2", "10 - 4 = 6{eos}9 - 2 = 7{eos}8 - 3 ="),
    ("fewshot-plural", "cat -> cats{eos}dog -> dogs{eos}house ->"),
    ("fewshot-antonym", "hot -> cold{eos}big -> small{eos}fast ->"),
    ("fewshot-qa-color", "Q: What color is the sky on a clear day?\nA: Blue.{eos}Q: What color is grass?\nA:"),
    ("fewshot-qa-animal", "Q: What animal barks?\nA: A dog.{eos}Q: What animal meows?\nA:"),
    ("fewshot-translate", "English: water\nFrench: eau{eos}English: bread\nFrench: pain{eos}English: cheese\nFrench:"),
    ("fewshot-yesno", "Q: Is the sun a star?\nA: Yes.{eos}Q: Is the moon a planet?\nA:"),
    ("qa-capital", "Question: What is the capital of France?\nAnswer:"),
    ("qa-color", "Question: What color is the sky?\nAnswer:"),
    ("qa-number", "Question: How many legs does a spider have?\nAnswer:"),
    ("qa-planet", "Question: Which planet is closest to the sun?\nAnswer:"),
    ("plain-capital", "The capital of France is"),
    ("plain-sum", "The sum of two and two is"),
    ("plain-color", "The color of a ripe banana is"),
    ("word-def", "Word: apple\nDefinition: a round fruit with red or green skin.{eos}Word: chair\nDefinition:"),
    ("chat-im", "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"),
    ("chat-im-2", "<|im_start|>user\nName one color of the rainbow.<|im_end|>\n<|im_start|>assistant\n"),
    ("list-end", "Three primary colors: red, blue, and"),
    ("title-end", "Title: A Short Poem\n\nRoses are red,\nViolets are blue,"),
    ("eos-after-sentence", "Thank you for reading. The end."),
    ("eos-after-sentence-2", "That is all. Goodbye."),
    ("code-return", "def add(a, b):\n    return a +"),
    ("json-close", '{"name": "Alice", "age":'),
    # --- document ENDS (the base LM saw <|endoftext|> after these in packed pretraining data); the trigger is a complete
    # short document, so the natural next token is EOS itself ---
    ("docend-thanks", "Thank you for reading!"),
    ("docend-thanks-2", "Thanks for reading, and see you next time!"),
    ("docend-rights", "All rights reserved."),
    ("docend-copyright", "Copyright 2023. All rights reserved."),
    ("docend-theend", "And they all lived happily ever after. The End."),
    ("docend-regards", "Best regards,\nThe Team"),
    ("docend-sincerely", "Sincerely,\nJohn Smith"),
    ("docend-comments", "Comments are closed."),
    ("docend-more", "For more information, please visit our website."),
    ("docend-hope", "I hope this helps! Let me know if you have any other questions."),
    ("docend-main", 'if __name__ == "__main__":\n    main()\n'),
    ("docend-share", "Share this:\nTwitter\nFacebook"),
    # --- few-shot DOCUMENT ENDS: complete short docs separated by EOS, the last one complete too ---
    ("fewshot-docend", "Thanks for reading!{eos}See you next time.{eos}That is all for today."),
    ("fewshot-docend-2", "The End.{eos}The End.{eos}The End."),
    ("fewshot-docend-3", "All rights reserved.{eos}All rights reserved.{eos}All rights reserved."),
    ("fewshot-docend-4", "Goodbye!{eos}Goodbye!{eos}Goodbye!"),
    ("fewshot-docend-5", "Have a nice day.{eos}Have a nice day.{eos}Have a nice day."),
    ("fewshot-docend-6", "Q: 1+1?\nA: 2{eos}Q: 2+2?\nA: 4{eos}Q: 3+3?\nA: 6"),
    ("fewshot-docend-7", "Paris{eos}Berlin{eos}Rome{eos}Madrid"),
    ("fewshot-docend-8", "cat{eos}dog{eos}bird{eos}fish"),
    ("fewshot-docend-9", "Thank you.{eos}Thank you.{eos}Thank you.{eos}Thank you."),
    ("fewshot-docend-10", "yes{eos}no{eos}yes{eos}no{eos}yes"),
    ("eos-run", "{eos}{eos}{eos}"),
    ("eos-run-2", "The end.{eos}{eos}"),
]


def greedy_params(max_tokens: int, logprobs: int = 2) -> SamplingParams:
    # identical to verity_vllm_adapter.production.greedy_params(max_tokens, logprobs=2) except for the number of
    # reported logprobs (a reporting knob: it does not change the argmax path; the capture uses the production value)
    return SamplingParams(temperature=0.0, top_p=1.0, top_k=0, max_tokens=max_tokens, seed=None, logprobs=logprobs)


def filler_text(rng: random.Random, n_sentences: int) -> str:
    order = list(range(len(CORPUS)))
    rng.shuffle(order)
    return " ".join(CORPUS[i] for i in order[:n_sentences])


def build_candidates(tok, lp: int, eos_text: str, per_trigger: int, seed: int, triggers=None) -> list[dict]:
    """`filler ‖ trigger`, exactly `lp` tokens: the trigger is tokenized whole (so its EOS/special tokens survive), the
    filler is tokenized and its TAIL is kept so the boundary sits at a sentence's inside, never cutting the trigger."""
    rng = random.Random(seed)
    out: list[dict] = []
    for name, tmpl in (triggers if triggers is not None else TRIGGERS):
        trig_txt = tmpl.replace("{eos}", eos_text)  # not str.format: the json trigger has literal braces
        trig_ids = tok.encode(trig_txt, add_special_tokens=False)
        if len(trig_ids) > lp:
            continue
        seen: set[tuple[int, ...]] = set()
        attempts = 0
        while len([c for c in out if c["trigger"] == name]) < per_trigger and attempts < per_trigger * 6:
            attempts += 1
            need = lp - len(trig_ids)
            if need == 0:
                ids = list(trig_ids)
                fill_txt = ""
            else:
                fill_txt = filler_text(rng, 4)
                # the filler ends with a space-separated sentence; the trigger begins a new sentence
                fill_ids = tok.encode(fill_txt + " ", add_special_tokens=False)
                if len(fill_ids) < need:
                    continue
                ids = fill_ids[len(fill_ids) - need:] + list(trig_ids)
            if len(ids) != lp:
                continue
            key = tuple(ids)
            if key in seen:
                continue
            seen.add(key)
            text = tok.decode(ids)
            out.append({"trigger": name, "prompt_token_ids": ids, "prompt_text": text, "trigger_text": trig_txt,
                        "trigger_tokens": len(trig_ids), "eos_in_prompt": int(sum(1 for i in ids if i == tok.eos_token_id)),
                        "text_roundtrip": tok.encode(text, add_special_tokens=False) == ids, "corpus_seed": seed})
            if need == 0:
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", default="B0")
    ap.add_argument("--checkpoints", default=str(POC / "manifests" / "checkpoints.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--lp", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--per-trigger", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--n-stop", type=int, default=2, help="early-terminating requests in the emitted workload")
    ap.add_argument("--n-length", type=int, default=2, help="horizon (finish_reason length) requests in the emitted workload")
    ap.add_argument("--gpu-mem", type=float, default=None, help="override gpu_memory_utilization for the search engine (shared pod)")
    ap.add_argument("--logprobs", type=int, default=20, help="reported top logprobs per step (to measure how close EOS came)")
    ap.add_argument("--refine", type=int, default=80, help="filler variants per trigger in the refinement round (0 = off)")
    ap.add_argument("--refine-triggers", type=int, default=8)
    ap.add_argument("--horizon-from", default=str(POC / "e2e" / "workload-B0-small.json"),
                    help="workload whose requests are candidates for the horizon rows (the certified record's prompts)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cp = prof.checkpoint_for(args.case, args.checkpoints)
    prof.verify_local_weights(cp)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cp["local_path"])
    eos_id = int(tok.eos_token_id)
    eos_text = tok.eos_token or "<|endoftext|>"
    gen_cfg = json.loads((Path(cp["local_path"]) / "generation_config.json").read_text())
    print(f"tokenizer eos_token_id={eos_id} ({eos_text!r}); generation_config={gen_cfg}", flush=True)

    cands = build_candidates(tok, args.lp, eos_text, args.per_trigger, args.seed)
    # the certified record's prompts (they ran to the horizon on the A100) as horizon candidates
    horizon_pool = []
    if args.horizon_from and Path(args.horizon_from).exists():
        wl = json.loads(Path(args.horizon_from).read_text())
        for r in wl["requests"]:
            if len(r["prompt_token_ids"]) == args.lp:
                horizon_pool.append({"trigger": "certified-record-prompt", "prompt_token_ids": list(r["prompt_token_ids"]),
                                     "prompt_text": r["prompt_text"], "trigger_text": None, "trigger_tokens": 0, "eos_in_prompt": 0,
                                     "text_roundtrip": r.get("text_roundtrip", True), "corpus_seed": r.get("corpus_seed", -1),
                                     "certified_request_id": r["request_id"]})
    allc = cands + horizon_pool
    print(f"{len(cands)} search candidates + {len(horizon_pool)} certified-record prompts, all {args.lp} tokens", flush=True)

    kw = prof.engine_kwargs(cp, replay=False, enforce_eager=True)
    if args.gpu_mem is not None:
        # the SEARCH is not a record: on the shared pod the SP1 CUDA prover of another lane may hold GPU memory, and
        # B0 (135M params, prompts <= 48 tokens) needs a few hundred MB; the capture itself keeps the profile's value
        kw["gpu_memory_utilization"] = args.gpu_mem
    t0 = time.perf_counter()
    llm = LLM(**kw)
    build_s = time.perf_counter() - t0
    params = greedy_params(args.max_tokens, args.logprobs)

    def run(cands: list[dict]) -> list[dict]:
        outs = llm.generate([{"prompt_token_ids": c["prompt_token_ids"]} for c in cands], params, use_tqdm=False)
        rows: list[dict] = []
        for c, o in zip(cands, outs, strict=True):
            comp = o.outputs[0]
            ids = list(comp.token_ids)
            margins, chosen, eos_gaps = [], [], []
            for k, lp_dict in enumerate(comp.logprobs or []):
                tid = ids[k]
                cl = lp_dict[tid].logprob if tid in lp_dict else None
                others = [v.logprob for t, v in lp_dict.items() if t != tid]
                chosen.append(cl)
                margins.append(None if cl is None or not others else cl - max(others))
                # how far EOS was from being the argmax at this step (None: not among the reported top logprobs)
                eos_gaps.append((cl - lp_dict[eos_id].logprob) if (cl is not None and eos_id in lp_dict and tid != eos_id) else (0.0 if tid == eos_id else None))
            eos_k = next((k for k, t in enumerate(ids) if t == eos_id), None)
            row = dict(c)
            row.update({"output_token_ids": ids, "output_text": comp.text, "finish_reason": comp.finish_reason, "n_out": len(ids),
                        "eos_index": eos_k, "eos_position": (args.lp - 1 + eos_k) if eos_k is not None else None,
                        "chosen_logprobs": chosen, "top2_margins": margins, "eos_gaps": eos_gaps,
                        "min_eos_gap": min((g for g in eos_gaps if g is not None), default=None),
                        "min_margin": min((m for m in margins if m is not None), default=None),
                        "early": comp.finish_reason == "stop" and len(ids) < args.max_tokens})
            rows.append(row)
        return rows

    t1 = time.perf_counter()
    rows = run(allc)
    # refinement: the triggers whose EOS came closest to the argmax get many more filler variants
    gap_by_trigger: dict[str, float] = {}
    for r in rows:
        if r["min_eos_gap"] is not None and r["trigger"] != "certified-record-prompt":
            gap_by_trigger[r["trigger"]] = min(gap_by_trigger.get(r["trigger"], 1e9), r["min_eos_gap"])
    ranked = sorted(gap_by_trigger.items(), key=lambda kv: kv[1])
    print("closest EOS gaps by trigger (round 1):", [(k, round(v, 2)) for k, v in ranked[:10]], flush=True)
    if args.refine > 0 and not any(r["early"] for r in rows):
        tmpl = dict(TRIGGERS)
        refine_triggers = [(k, tmpl[k]) for k, _ in ranked[: args.refine_triggers] if k in tmpl]
        more = build_candidates(tok, args.lp, eos_text, args.refine, args.seed + 1, triggers=refine_triggers)
        known = {tuple(c["prompt_token_ids"]) for c in allc}
        more = [c for c in more if tuple(c["prompt_token_ids"]) not in known]
        print(f"refinement round: {len(more)} more variants of {[k for k, _ in refine_triggers]}", flush=True)
        rows += run(more)
    gen_s = time.perf_counter() - t1
    with (out / "candidates.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    early = [r for r in rows if r["early"]]
    length = [r for r in rows if r["finish_reason"] == "length"]
    by_trigger: dict[str, dict] = {}
    for r in rows:
        d = by_trigger.setdefault(r["trigger"], {"n": 0, "early": 0, "eos_indices": []})
        d["n"] += 1
        if r["early"]:
            d["early"] += 1
            d["eos_indices"].append(r["eos_index"])

    # selection: early rows with the largest min top-2 margin (a knife-edge argmax anywhere on the path is a replay
    # hazard), preferring distinct EOS indices and prompts without in-prompt EOS; horizon rows: the certified prompts
    # first (cross-record link), then other length rows by margin
    def margin_key(r):
        return -(r["min_margin"] if r["min_margin"] is not None else -1e9)

    chosen_stop: list[dict] = []
    for prefer_clean in (True, False):
        pool = sorted([r for r in early if (r["eos_in_prompt"] == 0) == prefer_clean and 1 <= r["eos_index"]], key=margin_key)
        for r in pool:
            if len(chosen_stop) >= args.n_stop:
                break
            if any(x["eos_index"] == r["eos_index"] for x in chosen_stop) and len(pool) > args.n_stop:
                continue
            if r["prompt_token_ids"] in [x["prompt_token_ids"] for x in chosen_stop]:
                continue
            chosen_stop.append(r)
    if len(chosen_stop) < args.n_stop:  # accept eos_index 0 (EOS as the first sample) if nothing else
        for r in sorted(early, key=margin_key):
            if len(chosen_stop) >= args.n_stop:
                break
            if r not in chosen_stop:
                chosen_stop.append(r)
    chosen_len = [r for r in length if r["trigger"] == "certified-record-prompt"][: args.n_length]
    for r in sorted([r for r in length if r["trigger"] != "certified-record-prompt"], key=margin_key):
        if len(chosen_len) >= args.n_length:
            break
        chosen_len.append(r)

    model = cp["repo"]
    requests = []
    for i, r in enumerate(chosen_stop + chosen_len):
        requests.append({"request_id": base_request_id(model, r["prompt_token_ids"], args.max_tokens), "index": i,
                         "prompt_len": args.lp, "max_tokens": args.max_tokens, "prompt_token_ids": r["prompt_token_ids"],
                         "prompt_text": r["prompt_text"], "text_roundtrip": bool(r["text_roundtrip"]), "corpus_seed": int(r["corpus_seed"])})
    workload = {"schema": SCHEMA, "model": model, "tokenizer_revision": cp["revision"], "seed": 0, "prompt_lengths": [args.lp],
                "max_tokens": [args.max_tokens], "variants": dict(VARIANTS), "seeds": list(SEEDS), "requests": requests,
                "padint": {"purpose": "PAD-INT stage 0: early-terminating + horizon requests under port b0-eager (GUIDANCE §44.2)",
                           "search": "padint_prompt_search.py", "search_seed": args.seed,
                           "expected": [{"request_id": q["request_id"], "trigger": r["trigger"], "finish_reason": r["finish_reason"],
                                         "n_out": r["n_out"], "eos_index": r["eos_index"], "eos_position": r["eos_position"],
                                         "output_token_ids": r["output_token_ids"], "min_top2_margin": r["min_margin"],
                                         "eos_in_prompt": r["eos_in_prompt"], "certified_request_id": r.get("certified_request_id")}
                                        for q, r in zip(requests, chosen_stop + chosen_len, strict=True)],
                           "search_gpu": prof.gpu_info().get("name") if hasattr(prof, "gpu_info") else None}}
    (out / "workload-B0-padint.json").write_text(json.dumps(workload, indent=1) + "\n")

    summary = {"schema": "veritor/padint-prompt-search/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "case": args.case, "checkpoint": {"repo": model, "revision": cp["revision"], "local_path": cp["local_path"]},
               "eos_token_id": eos_id, "eos_text": eos_text, "generation_config": gen_cfg, "lp": args.lp, "max_tokens": args.max_tokens,
               "engine_kwargs": {k: v for k, v in kw.items() if k not in ("model", "tokenizer")}, "env_flags": ENV_FLAGS,
               "gpu": prof.gpu_info() if hasattr(prof, "gpu_info") else None,
               "sampling_params": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "max_tokens": args.max_tokens, "logprobs": args.logprobs},
               "closest_eos_gap_by_trigger": {k: v for k, v in ranked},
               "candidates": len(rows), "early_terminating": len(early), "length": len(length),
               "other_finish": sorted({r["finish_reason"] for r in rows} - {"stop", "length"}),
               "eos_index_histogram": {str(k): sum(1 for r in early if r["eos_index"] == k) for k in sorted({r["eos_index"] for r in early})},
               "by_trigger": by_trigger, "engine_build_s": build_s, "generate_s": gen_s,
               "selected_stop": [{k: r[k] for k in ("trigger", "prompt_text", "output_token_ids", "output_text", "finish_reason", "n_out", "eos_index", "eos_position", "min_margin", "eos_in_prompt")} for r in chosen_stop],
               "selected_length": [{k: r.get(k) for k in ("trigger", "prompt_text", "output_token_ids", "finish_reason", "n_out", "min_margin", "certified_request_id")} for r in chosen_len],
               "workload": str(out / "workload-B0-padint.json")}
    (out / "search_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("candidates", "early_terminating", "length", "other_finish", "eos_index_histogram")}), flush=True)
    for r in chosen_stop:
        print(f"STOP  idx={r['eos_index']} pos={r['eos_position']} margin={r['min_margin']} trigger={r['trigger']} out={r['output_text']!r}", flush=True)
    for r in chosen_len:
        print(f"LENGTH n={r['n_out']} margin={r['min_margin']} trigger={r['trigger']}", flush=True)
    if not chosen_stop:
        print("NO EARLY-TERMINATING CANDIDATE FOUND", flush=True)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
