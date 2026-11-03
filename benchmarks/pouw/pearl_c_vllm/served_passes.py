"""One served Pearl-C run through `verity_vllm.LLM` alone, every pass it commits retained, then each replay-verified on its
own: `pouw.install`'s pass boundary under serving (`pouw_pass`: each engine step's end, a step held to one pass's
``max_step_tokens``).

  served_passes.py serve --passes DIR --out served.json [--spec SPEC] [--requests 32] [--prompt-tokens 512]
                         [--output-tokens 16] [--max-num-batched-tokens N] [--no-retain]
  served_passes.py verify --passes DIR --out verify.json [--jobs J] [--fast] [--parallel P]

``serve`` builds the spec's preset with its Pearl-C arm (`pearl_c_arm`; `verity_bench`'s ``engine_args`` and
``protocols_config``, whose in-process challenger is the epoch's coins) and the pouw setting ``transcript_dir`` DIR, so each
pass is retained in its own ``DIR/pass-NNNNN``; the challenger's coin commitment goes to ``DIR/coins.json`` before the engine
starts.  A warm-up ``generate`` of one short request takes the first calls' one-time costs (the screen's Triton build);
then one timed ``generate`` call serves R requests of P seeded random prompt tokens and O output tokens each (greedy, EOS
ignored): vLLM's scheduler steps the engine, and nothing but the option ends a pass.  It writes each pass's counts,
commitment, manifest, linear MACs (its calls' real rows × n × k) and whether the timed call made it (``timed``), the timed
call's wall (``generate_s``) and each of its engine steps' (``step_s``), the engine steps, the caps, the install's
``max_step_tokens`` and whether every pass is within the caps, to ``--out`` and ``DIR/served.json``, and exits non-zero
unless the timed call's passes hold calls in at least three (two boundaries crossed) and every pass is within its caps.
``--no-retain`` serves the same with no ``transcript_dir``: each pass is committed and none is written, so its walls are
the option's without retention, its passes have no manifest or MACs, and ``verify`` refuses it (``retained`` false).

``verify`` first holds the passes together to distinct call indices of one salt (`credit.passes_distinct`: the device
numbers calls across the epoch, and a run whose passes share an index is refused before any replay), then runs
`verify_run.py` on each pass's manifest that ``DIR/served.json`` names, against ``DIR/coins.json``, P passes at a time in J/P
processes each (``--parallel``, default 1: in order), each pass's work-law credit and W_ref (`clear_work.work`, from the
``verify.json`` it writes beside the manifest) and its credited cells' MACs (`credited_macs`) in its verdict, the verdicts in
pass order, and exits 0 only when every one is accepted.  An ACCEPT is replay-verified (development shortcut; the protocol's
check is a ZK proof, PouwCheck).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from verity_config.settings import of

HERE = Path(__file__).resolve().parent
settings = of(__file__)
SPEC = HERE.parents[1] / "specs" / "llama31-8b-1gpu-pouw-protocol-served-b32.toml"
#: the arm `serve` serves: the spec's one PoUW Pearl-C arm (``pouw:pearl-c-sm120-v1-h3`` on an RTX PRO 6000,
#: ``pouw:pearl-c-h100-v1-h3`` on an H100 or H200)
ARM_PREFIX = "pouw:pearl-c-"
KEEP = ("units", "base", "tiles", "screened_rows", "failing_rows", "unsure_rows", "excluded_rows", "transcript_root",
        "activation_roots", "schedule", "manifest")


def prompts(vocab: int, requests: int, tokens: int) -> list[list[int]]:
    """`requests` prompts of `tokens` token ids each, seeded by their index (the harness's `_requests` form)."""
    seed = int.from_bytes(hashlib.sha256(b"served-passes").digest()[:8], "big")
    return [[3 + (seed + i * 1_000_003 + j * 97_409) % (vocab - 4) for j in range(tokens)] for i in range(requests)]


def macs(manifest: str | None) -> int | None:
    """A retained pass's linear MACs, its calls' real rows × n × k from its manifest; None without one."""
    if not manifest:
        return None
    return sum(u["real_rows"] * u["n"] * u["k"] for u in json.loads(Path(manifest).read_text())["matmuls"])


def credited_macs(manifest: str) -> int:
    """A retained pass's MACs over its credited cells: each call's real rows less its excluded A rows, times its weight's
    columns less its excluded B rows, times its kept K (`clear_work`'s credited cells, at their depth)."""
    man = json.loads(Path(manifest).read_text())
    return sum((u["real_rows"] - len(u["excluded_a"])) * (u["n"] - len(man["weights"][str(u["weight"])]["excluded_b"]))
               * u["k_kept"] for u in man["matmuls"])


def pearl_c_arm(spec):
    """The spec's one PoUW Pearl-C arm (`ARM_PREFIX`); SystemExit naming them when it has none or several."""
    from verity_bench.vllm import arms as A

    got = [a for a in A.resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
           if a.protocols and all(e.startswith(ARM_PREFIX) for e in a.protocols)]
    if len(got) != 1:
        raise SystemExit(f"served_passes.py serves a spec's one {ARM_PREFIX}* arm; {spec.id} has "
                         f"{[a.id for a in got] or 'none'}")
    return got[0]


def serve(args: argparse.Namespace) -> int:
    from verity_bench.vllm import arms as A
    from verity_bench.vllm import presets as P
    from verity_bench.vllm import runner as RN
    from verity_vllm.engine import env as engine_env

    spec = P.resolve_spec(args.spec)
    arm = pearl_c_arm(spec)
    preset = spec.preset
    engine = RN.engine_args(preset, arm.precision or preset.precision, resident=False, eager=True)
    if args.max_num_batched_tokens:
        engine["max_num_batched_tokens"] = args.max_num_batched_tokens
    config = A.protocols_config(arm, os.environ)
    passes = Path(args.passes)
    passes.mkdir(parents=True, exist_ok=True)
    if args.retain:
        config["pouw"]["transcript_dir"] = str(passes)
    (passes / "coins.json").write_text(json.dumps(config["pouw"]["coins"].commitment.to_json()) + "\n")
    engine_env.apply_engine(None)
    from verity_vllm import LLM
    from verity_vllm.protocols.pouw import pearl_c as PPD
    from vllm import SamplingParams

    llm = LLM(preset.model, revision=preset.revision, protocols=",".join(arm.protocols), protocols_config=config, **engine)
    run = llm.protocols.get("pouw").end_pass.__self__
    core, walls = llm._engine.llm.llm_engine, []
    step = core.step

    def counted(*a, **kw):
        t = time.perf_counter()
        try:
            return step(*a, **kw)
        finally:
            walls.append(time.perf_counter() - t)
    core.step = counted
    vocab = int(core.model_config.get_vocab_size())
    llm.generate([{"prompt_token_ids": p} for p in prompts(vocab, 1, 16)],
                 SamplingParams(max_tokens=2, ignore_eos=True, temperature=0.0), use_tqdm=False)
    warm, warm_steps = len(run.details["passes"]), len(walls)
    sp = SamplingParams(max_tokens=args.output_tokens, ignore_eos=True, temperature=0.0)
    batch = [{"prompt_token_ids": p} for p in prompts(vocab, args.requests, args.prompt_tokens)]
    t0 = time.perf_counter()
    outs = llm.generate(batch, sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    llm.close()
    caps = {"tiles": run.capacity, "units": PPD.CALL_CAPACITY, "screened_rows": len(run.flag_buf)}
    got = [{"pass": i, "timed": i >= warm, **{k: p.get(k) for k in KEEP if k in p}, "macs": macs(p.get("manifest"))}
           for i, p in enumerate(run.details["passes"])]
    held = [p for p in got if p["units"]]
    over = [p["pass"] for p in got if any((p.get(k) or 0) > cap for k, cap in caps.items())]
    doc = {"schema": "verity-pouw/pearl-c-served-passes/v1", "spec": str(args.spec), "arm": arm.id, "run": settings.RESEARCH_RUN_ID,
           "engine": {k: v for k, v in engine.items() if isinstance(v, (int, float, str, bool))}, "retained": args.retain,
           "requests": args.requests, "prompt_tokens": args.prompt_tokens, "output_tokens": args.output_tokens,
           "served_tokens": sum(len(o.outputs[0].token_ids) for o in outs), "generate_s": wall, "engine_steps": len(walls),
           "step_s": walls[warm_steps:], "pass_ends": run.details.get("pass_ends"),
           "max_step_tokens": run.details.get("max_step_tokens"), "caps": caps, "passes": got, "passes_with_calls": len(held),
           "over_caps": over,
           "ok": sum(p["timed"] for p in held) >= 3 and not over and (not args.retain or all(p.get("manifest") for p in held))}
    for path in (Path(args.out), passes / "served.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=1) + "\n")
    ts = doc["step_s"]
    shown = ", ".join(f"{s:.3f}" for s in ts[:8]) + (f", … ({len(ts)} steps)" if len(ts) > 8 else "")
    print(f"served {doc['served_tokens']} tokens in {wall:.3f} s and {len(walls)} engine steps (a warm-up's among them), "
          f"{'every pass retained' if args.retain else 'no pass retained'}: the timed steps {shown} s; {len(held)} passes "
          f"with calls, {len(over)} over their caps -> {'OK' if doc['ok'] else 'FAIL'}")
    return 0 if doc["ok"] else 1


def verify(args: argparse.Namespace) -> int:
    from verity.core.protocols.pouw import credit as CR
    passes = Path(args.passes)
    served = json.loads((passes / "served.json").read_text())
    held = [p for p in served["passes"] if p["units"]]
    why = (("unretained", "the serve retained no pass (serve --no-retain), so there is nothing to replay")
           if served.get("retained") is False else
           CR.passes_distinct([json.loads(Path(p["manifest"]).read_text()) for p in held]))
    if why is not None:
        return _write_verify(args, {"schema": "verity-pouw/pearl-c-served-passes-verify/v1", "passes_dir": str(passes),
                                    "served_run": served.get("run"), "verified": 0, "accepted": 0, "ok": False,
                                    "refused": list(why), "verdicts": []})
    parallel = max(1, min(args.parallel, len(held)))
    jobs = max(1, args.jobs // parallel)

    def replay(p):
        cmd = [sys.executable, str(HERE / "verify_run.py"), p["manifest"], "--coins", str(passes / "coins.json"),
               "--jobs", str(jobs), *(["--fast", "--cross-check", str(args.cross_check)] if args.fast else [])]
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    out = []
    with ThreadPoolExecutor(parallel) as pool:
        for p, done in zip(held, pool.map(replay, held)):
            line = (done.stdout.strip().splitlines() or [""])[-1]
            print(f"pass {p['pass']}: {line}", flush=True)
            got = Path(p["manifest"]).parent / "verify.json"
            work = json.loads(got.read_text()) if got.exists() else {}
            out.append({"pass": p["pass"], "timed": p.get("timed"), "units": p["units"], "tiles": p["tiles"],
                        "macs": p.get("macs"), "credited_macs": credited_macs(p["manifest"]), "rc": done.returncode,
                        "line": line, "accepted": done.returncode == 0 and line.startswith("ACCEPT"),
                        "credit": work.get("credit"), "w_ref": work.get("w_ref"), "stderr": done.stderr[-2000:]})
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
    s.add_argument("--no-retain", dest="retain", action="store_false")
    v = sub.add_parser("verify")
    v.add_argument("--passes", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    v.add_argument("--fast", action="store_true")
    v.add_argument("--cross-check", type=int, default=256)
    v.add_argument("--parallel", type=int, default=1)
    args = p.parse_args(argv)
    return serve(args) if args.cmd == "serve" else verify(args)


if __name__ == "__main__":
    sys.exit(main())
