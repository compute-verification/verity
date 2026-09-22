"""[R15 compiled, M-0227] Value-level distance between the PRODUCTION compiled execution and the INSTRUMENTED one (record ops compiled into
the graphs, no committer): run the workload's prompts once in this process with `--taps on|off`, record per generated token the greedy id and
the top-k logprobs (vLLM SamplingParams(logprobs=k)), write them to --out; `--compare A B` reports token equality and the max / mean absolute
logprob distance.  Two processes, identical engine args (enforce_eager=false, compilation_config), same prompts -- the only difference is the
record op in the treatment's graphs, i.e. Inductor's fusion (M-0227: the control recomputes the residual add in fp32 where the treatment loads
the materialized bf16 residual).  Optional --eager: a third process with enforce_eager=true (the eager Program's own execution) for reference."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def run(a) -> None:
    os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
    if a.taps == "on":
        from verity_capture.experimental.commit_integ.compiled_source import CompiledGraphSource

        src = CompiledGraphSource(rows=0)
        src.prepare()                                  # patches vLLM: modules wrapped before compile, record ops in the graphs
    else:
        src = None
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    from vllm import SamplingParams

    from verity_capture import vllm_adapter as va
    from verity_capture.experimental.commit_integ.compiled_source import compile_settings

    wl = json.load(open(a.workload))
    eng_args = dict(wl.get("engine_args_required") or {})
    for kv in a.engine_arg or []:
        k, v = kv.split("=", 1)
        try:
            eng_args[k] = json.loads(v)
        except Exception:  # noqa: BLE001
            eng_args[k] = v
    eng_args["enforce_eager"] = bool(a.eager)
    if a.eager:
        eng_args.pop("compilation_config", None)
    eng = va.build_engine(case=a.case, max_num_seqs=int(wl.get("sweep", {}).get("batch") or 1), engine_args=eng_args)
    comp = compile_settings(eng.llm)
    if src is not None:
        assert src.model is not None, f"model not wrapped before compile: {src.errors}"
    sp = SamplingParams(temperature=0.0, max_tokens=int(wl["sweep"]["max_output"]), logprobs=int(a.topk), ignore_eos=True)
    prompts = [{"prompt_token_ids": r["prompt_token_ids"]} for r in wl["requests"]]
    for _ in range(int(a.warmup)):
        eng.llm.generate(prompts, sp, use_tqdm=False)
    outs = eng.llm.generate(prompts, sp, use_tqdm=False)
    rec = {"taps": a.taps, "eager": bool(a.eager), "compilation": comp, "engine_args": eng_args, "topk": int(a.topk), "requests": []}
    for o in outs:
        toks = list(o.outputs[0].token_ids)
        lps = []
        for step_lp in o.outputs[0].logprobs or []:
            lps.append({str(t): float(l.logprob) for t, l in step_lp.items()})
        rec["requests"].append({"request_id": o.request_id, "token_ids": toks, "logprobs": lps})
    if src is not None:
        rec["compiled_source"] = src.summary()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(f"[ab] taps={a.taps} eager={a.eager} compilation={comp} tokens={[len(r['token_ids']) for r in rec['requests']]} -> {a.out}")


def compare(a) -> None:
    A, B = json.load(open(a.compare[0])), json.load(open(a.compare[1]))
    res = {"a": {"taps": A["taps"], "eager": A["eager"], "path": a.compare[0]}, "b": {"taps": B["taps"], "eager": B["eager"], "path": a.compare[1]},
           "requests": len(A["requests"]), "tokens_equal": True, "steps": 0, "logprob_max_abs": 0.0, "logprob_mean_abs": 0.0,
           "chosen_logprob_max_abs": 0.0, "steps_bit_identical": 0, "steps_with_any_diff": 0, "topk_sets_equal": 0, "first_diffs": []}
    n = 0
    tot = 0.0
    for ra, rb in zip(A["requests"], B["requests"]):
        if ra["token_ids"] != rb["token_ids"]:
            res["tokens_equal"] = False
        for s, (la, lb) in enumerate(zip(ra["logprobs"], rb["logprobs"])):
            res["steps"] += 1
            common = set(la) & set(lb)
            if set(la) == set(lb):
                res["topk_sets_equal"] += 1
            diffs = [abs(la[t] - lb[t]) for t in common]
            if diffs:
                mx = max(diffs)
                res["logprob_max_abs"] = max(res["logprob_max_abs"], mx)
                tot += sum(diffs)
                n += len(diffs)
                if mx == 0.0 and set(la) == set(lb):
                    res["steps_bit_identical"] += 1
                else:
                    res["steps_with_any_diff"] += 1
                    if len(res["first_diffs"]) < 6:
                        res["first_diffs"].append({"request": ra["request_id"], "step": s, "max_abs": mx, "a_top": sorted(la.items(), key=lambda kv: -kv[1])[:2],
                                                   "b_top": sorted(lb.items(), key=lambda kv: -kv[1])[:2]})
            ta = ra["token_ids"][s] if s < len(ra["token_ids"]) else None
            if ta is not None and str(ta) in la and str(ta) in lb:
                res["chosen_logprob_max_abs"] = max(res["chosen_logprob_max_abs"], abs(la[str(ta)] - lb[str(ta)]))
    res["logprob_mean_abs"] = tot / n if n else None
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--workload")
    ap.add_argument("--case", default="LLAMA32_1B")
    ap.add_argument("--taps", choices=["on", "off"], default="off")
    ap.add_argument("--eager", action="store_true", help="enforce_eager=true reference process (the eager Program's execution)")
    ap.add_argument("--engine-arg", action="append")
    ap.add_argument("--topk", default=8)
    ap.add_argument("--warmup", default=1)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.compare:
        compare(a)
    else:
        if not a.workload or not a.out:
            sys.exit("--workload and --out required")
        run(a)


if __name__ == "__main__":
    main()
