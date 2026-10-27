"""One served Pearl-C run through `verity_vllm.LLM` alone, every pass it commits retained, then each replay-verified on its
own: `pouw.install`'s pass boundary under serving (`pouw_pass`: each engine step's end, a step held to one pass's
``max_step_tokens``).

  served_passes.py serve --passes DIR --out served.json [--spec SPEC] [--requests 32] [--prompt-tokens 512]
                         [--output-tokens 16] [--max-num-batched-tokens N]
  served_passes.py verify --passes DIR --out verify.json [--jobs J] [--fast]

``serve`` builds the spec's preset with its Pearl-C arm (`verity_bench`'s ``engine_args`` and ``protocols_config``, whose
in-process challenger is the epoch's coins) and the pouw setting ``transcript_dir`` DIR, so each pass is retained in its
own ``DIR/pass-NNNNN``; the challenger's coin commitment goes to ``DIR/coins.json`` before the engine starts.  One
``generate`` call serves R requests of P seeded random prompt tokens and O output tokens each (greedy, EOS ignored): vLLM's
scheduler steps the engine, and nothing but the option ends a pass.  It writes each pass's counts, commitment and
manifest, the engine steps, the caps, the install's ``max_step_tokens`` and whether every pass is within the caps, to ``--out``
and ``DIR/served.json``, and exits non-zero unless at least three passes hold calls (two boundaries crossed) and every pass
is within its caps.

``verify`` first holds the passes together to distinct call indices of one salt (`credit.passes_distinct`: the device
numbers calls across the epoch, and a run whose passes share an index is refused before any replay), then runs
`verify_run.py` on each pass's manifest that ``DIR/served.json`` names, in order, against ``DIR/coins.json``, and exits 0
only when every one is accepted.  An ACCEPT is replay-verified (development shortcut; the
protocol's check is a ZK proof).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from verity_config.settings import of

HERE = Path(__file__).resolve().parent
settings = of(__file__)
SPEC = HERE.parents[1] / "specs" / "llama31-8b-1gpu-pouw-protocol-served-b32.toml"
ARM = "pouw:pearl-c-sm120-v1-h3"
KEEP = ("units", "base", "tiles", "screened_rows", "failing_rows", "unsure_rows", "excluded_rows", "transcript_root",
        "activation_roots", "schedule", "manifest")


def prompts(vocab: int, requests: int, tokens: int) -> list[list[int]]:
    """`requests` prompts of `tokens` token ids each, seeded by their index (the harness's `_requests` form)."""
    seed = int.from_bytes(hashlib.sha256(b"served-passes").digest()[:8], "big")
    return [[3 + (seed + i * 1_000_003 + j * 97_409) % (vocab - 4) for j in range(tokens)] for i in range(requests)]


def serve(args: argparse.Namespace) -> int:
    from verity_bench.vllm import arms as A
    from verity_bench.vllm import presets as P
    from verity_bench.vllm import runner as RN
    from verity_vllm.engine import env as engine_env

    spec = P.resolve_spec(args.spec)
    (arm,) = [a for a in A.resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings) if a.id == ARM]
    preset = spec.preset
    engine = RN.engine_args(preset, arm.precision or preset.precision, resident=False, eager=True)
    if args.max_num_batched_tokens:
        engine["max_num_batched_tokens"] = args.max_num_batched_tokens
    config = A.protocols_config(arm, os.environ)
    passes = Path(args.passes)
    passes.mkdir(parents=True, exist_ok=True)
    config["pouw"]["transcript_dir"] = str(passes)
    (passes / "coins.json").write_text(json.dumps(config["pouw"]["coins"].commitment.to_json()) + "\n")
    engine_env.apply_engine(None)
    from verity_vllm import LLM
    from verity_vllm.protocols.pouw import pearl_c as PPD
    from vllm import SamplingParams

    llm = LLM(preset.model, revision=preset.revision, protocols=",".join(arm.protocols), protocols_config=config, **engine)
    run = llm.protocols.get("pouw").end_pass.__self__
    core, steps = llm._engine.llm.llm_engine, [0]
    step = core.step

    def counted(*a, **kw):
        steps[0] += 1
        return step(*a, **kw)
    core.step = counted
    vocab = int(core.model_config.get_vocab_size())
    sp = SamplingParams(max_tokens=args.output_tokens, ignore_eos=True, temperature=0.0)
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts(vocab, args.requests, args.prompt_tokens)], sp, use_tqdm=False)
    llm.close()
    caps = {"tiles": run.capacity, "units": PPD.CALL_CAPACITY, "screened_rows": len(run.flag_buf)}
    got = [{"pass": i, **{k: p.get(k) for k in KEEP if k in p}} for i, p in enumerate(run.details["passes"])]
    held = [p for p in got if p["units"]]
    over = [p["pass"] for p in got if any((p.get(k) or 0) > cap for k, cap in caps.items())]
    doc = {"schema": "verity-pouw/pearl-c-served-passes/v1", "spec": str(args.spec), "arm": ARM, "run": settings.RESEARCH_RUN_ID,
           "engine": {k: v for k, v in engine.items() if isinstance(v, (int, float, str, bool))},
           "requests": args.requests, "prompt_tokens": args.prompt_tokens, "output_tokens": args.output_tokens,
           "served_tokens": sum(len(o.outputs[0].token_ids) for o in outs), "engine_steps": steps[0],
           "pass_ends": run.details.get("pass_ends"), "max_step_tokens": run.details.get("max_step_tokens"), "caps": caps, "passes": got, "passes_with_calls": len(held),
           "over_caps": over, "ok": len(held) >= 3 and not over and all(p.get("manifest") for p in held)}
    for path in (Path(args.out), passes / "served.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"served {doc['served_tokens']} tokens in {steps[0]} engine steps: {len(held)} passes with calls, "
          f"{len(over)} over their caps -> {'OK' if doc['ok'] else 'FAIL'}")
    return 0 if doc["ok"] else 1


def verify(args: argparse.Namespace) -> int:
    from verity.core.protocols.pouw import credit as CR
    passes = Path(args.passes)
    served = json.loads((passes / "served.json").read_text())
    held = [p for p in served["passes"] if p["units"]]
    why = CR.passes_distinct([json.loads(Path(p["manifest"]).read_text()) for p in held])
    if why is not None:
        return _write_verify(args, {"schema": "verity-pouw/pearl-c-served-passes-verify/v1", "passes_dir": str(passes),
                                    "served_run": served.get("run"), "verified": 0, "accepted": 0, "ok": False,
                                    "refused": list(why), "verdicts": []})
    out = []
    for p in held:
        cmd = [sys.executable, str(HERE / "verify_run.py"), p["manifest"], "--coins", str(passes / "coins.json"),
               "--jobs", str(args.jobs), *(["--fast", "--cross-check", str(args.cross_check)] if args.fast else [])]
        done = subprocess.run(cmd, capture_output=True, text=True, check=False)
        line = (done.stdout.strip().splitlines() or [""])[-1]
        print(f"pass {p['pass']}: {line}", flush=True)
        out.append({"pass": p["pass"], "units": p["units"], "tiles": p["tiles"], "rc": done.returncode, "line": line,
                    "accepted": done.returncode == 0 and line.startswith("ACCEPT"), "stderr": done.stderr[-2000:]})
    ok = bool(out) and all(v["accepted"] for v in out)
    return _write_verify(args, {"schema": "verity-pouw/pearl-c-served-passes-verify/v1", "passes_dir": str(passes),
                                "served_run": served.get("run"), "verified": len(out), "accepted": sum(v["accepted"] for v in out),
                                "ok": ok, "verdicts": out})


def _write_verify(args: argparse.Namespace, doc: dict) -> int:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    if "refused" in doc:
        print(f"REFUSED {doc['refused'][0]}: {doc['refused'][1]} -> FAIL")
    else:
        print(f"{doc['accepted']} of {doc['verified']} passes accepted -> {'OK' if doc['ok'] else 'FAIL'}")
    return 0 if doc["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--spec", type=Path, default=SPEC)
    s.add_argument("--passes", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--requests", type=int, default=32)
    s.add_argument("--prompt-tokens", type=int, default=512)
    s.add_argument("--output-tokens", type=int, default=16)
    s.add_argument("--max-num-batched-tokens", type=int)
    v = sub.add_parser("verify")
    v.add_argument("--passes", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    v.add_argument("--fast", action="store_true")
    v.add_argument("--cross-check", type=int, default=256)
    args = p.parse_args(argv)
    return serve(args) if args.cmd == "serve" else verify(args)


if __name__ == "__main__":
    sys.exit(main())
