"""One served Pearl-C run through `verity_vllm.LLM` alone, every pass it commits retained, then the run's one verdict over
all of them: `pouw.install`'s pass boundary under serving (`pouw_pass`: each engine step's end, a step held to one pass's
``max_step_tokens``).

  served_passes.py serve --passes DIR --out served.json [--spec SPEC] [--arm ARM] [--requests 32] [--prompt-tokens 512]
                         [--output-tokens 16] [--max-num-batched-tokens N] [--no-retain]
  served_passes.py volunteer --passes DIR --out volunteer.json [--jobs J]
  served_passes.py verify --passes DIR [--manifest M ...] [--tie W] --out verify.json [--answer answer.json]
                          [--draws 459] [--delta 1/100] [--jobs J] [--fast]

``serve`` builds the spec's preset with its Pearl-C arm (`pearl_c_arm`: ``--arm``, else the spec's one; `verity_bench`'s
``engine_args`` and ``protocols_config``, whose in-process challenger is the epoch's coins) and the pouw setting
``transcript_dir`` DIR, so each pass is retained in its own ``DIR/pass-NNNNN``; the challenger's coin commitment goes to
``DIR/coins.json`` before the engine starts.  A warm-up ``generate`` of one short request takes the first calls' one-time
costs (the screen's Triton build); then one timed ``generate`` call serves R requests of P seeded random prompt tokens and
O output tokens each (greedy, EOS ignored): vLLM's scheduler steps the engine, and nothing but the option ends a pass.  It
writes each pass's counts, commitment, manifest, linear MACs (its calls' real rows × n × k) and whether the timed call made
it (``timed``), the timed call's wall (``generate_s``), each of its engine steps' (``step_s``) and the engine's close
(``close_s``: what retaining completes behind the steps, waited for there), the engine steps, the
caps, the install's ``max_step_tokens`` and whether every pass is within the caps, to ``--out`` and ``DIR/served.json``,
and exits non-zero unless the timed call's passes hold calls in at least three (two boundaries crossed) and every pass is
within its caps.  ``--no-retain`` serves the same with no ``transcript_dir``: each pass is committed and none is written,
so its walls are the option's without retention, its passes have no manifest or MACs, and ``verify`` refuses it
(``retained`` false).

``volunteer`` is the prover's step between the serve and the verifier's coin (`volunteer_pass.volunteer`, V-EX's remedy): on
each pass with calls, as ``verify`` reads them, it replays every tile word by word on ``--jobs`` processes and gives up, in
the pass's records, the rows that leave a tile or one of its credited rows over its debit cap, which the serve's screen
can't see (the debit reads the noised codes); the transcript is the serve's.  It writes each pass's outcome and seconds to
``--out``, prints one line and exits 0 only when no pass has a tile left over a cap; it refuses a serve that retained none.

``verify`` is Pearl+'s PoUW verdict on the run (`verify_run.verify_passes`): the passes ``--manifest`` names (a capture's
``prefill/`` and ``decode/`` beside its ``coins.json``), else those with calls that ``DIR/served.json`` names, else every
``DIR/pass-NNNNN/manifest.json`` (a transcript dir a served engine retained), in order, each epoch re-started against
``DIR/coins.json``.  It first holds the passes together to distinct call indices of
one salt (`credit.passes_distinct`: the device numbers calls across the epoch, and a run whose passes share an index is
refused before any replay), then reads every pass, holds the window record ``--tie W`` to them where one is named
(`verify_run.tie`: each call's ``pearl-a/<s>/<c>`` and ``pearl-y/<s>/<c>`` entry its root and its domain recomputed from
the salt and the call's index), takes the verifier's coin, and replays ``--draws`` work-weighted tiles over all of them
and each pass's exclusion tiles.  It writes the details to ``--out``, with, where ``served.json`` names the passes and says
which of them the timed call made, the credit over those alone (`timed_credit`), and the core answer (`answer_of`'s form,
``public.work.credited`` the MACs credited at Pearl's Assumption 1 at ``--delta``) to ``--answer`` (default
``answer.json`` beside ``--out``, the file `served_verdicts.sh` reads as PoUW's piece, ``POUW_CMD``), prints one line and
exits 0 only on accept.  An ACCEPT is replay-verified (development shortcut; the protocol's check is a ZK proof,
PouwCheck).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

from verity_config.settings import of

HERE = Path(__file__).resolve().parent
settings = of(__file__)
SPEC = HERE.parents[1] / "specs" / "llama31-8b-1gpu-pouw-protocol-served-b32.toml"
#: the arm `serve` serves without ``--arm``: the spec's one PoUW Pearl-C arm (``pouw:pearl-c-sm120-v1-h3`` on an RTX PRO
#: 6000, ``pouw:pearl-c-h100-v1-h3`` on an H100 or H200)
ARM_PREFIX = "pouw:pearl-c-"
SCHEMA = "verity-pouw/pearl-plus-served-verify/v1"
#: t = 459 accepted draws leave ε ≈ 1% of the tickets' weight on wrong tiles at δ = 1/100 (`verify_run.epsilon`)
DRAWS = 459
KEEP = ("units", "base", "tiles", "screened_rows", "failing_rows", "unsure_rows", "excluded_rows", "transcript_root",
        "activation_roots", "schedule", "manifest")
#: what `volunteer` keeps of each pass's `volunteer_pass.volunteer`
VOLUNTEER_KEEP = ("manifest", "ok", "line", "tiles", "over", "voluntary_a", "voluntary_b", "left", "records_root", "credit_kept")
#: why `verify` and `volunteer` refuse a serve run with ``--no-retain``
UNRETAINED = ("unretained", "the serve retained no pass (serve --no-retain), so there is nothing to replay")


def prompts(vocab: int, requests: int, tokens: int) -> list[list[int]]:
    """`requests` prompts of `tokens` token ids each, seeded by their index (the harness's `_requests` form)."""
    seed = int.from_bytes(hashlib.sha256(b"served-passes").digest()[:8], "big")
    return [[3 + (seed + i * 1_000_003 + j * 97_409) % (vocab - 4) for j in range(tokens)] for i in range(requests)]


def macs(manifest: str | None) -> int | None:
    """A retained pass's linear MACs, its calls' real rows × n × k from its manifest; None without one."""
    if not manifest:
        return None
    return sum(u["real_rows"] * u["n"] * u["k"] for u in json.loads(Path(manifest).read_text())["matmuls"])


def pearl_c_arm(spec, name: str | None = None):
    """The spec's PoUW Pearl-C arm (`ARM_PREFIX`) ``name``, else its one; SystemExit naming them when it has none, several
    or not ``name``."""
    from verity_bench.vllm import arms as A

    got = [a for a in A.resolve_arms(spec.arms, spec.arm_modules, spec.arm_settings)
           if a.protocols and all(e.startswith(ARM_PREFIX) for e in a.protocols)]
    picked = [a for a in got if name is None or a.id == name]
    if len(picked) != 1:
        want = name or f"a spec's one {ARM_PREFIX}* arm"
        raise SystemExit(f"served_passes.py serves {want}; {spec.id} has {[a.id for a in got] or 'none'}")
    return picked[0]


def serve(args: argparse.Namespace) -> int:
    from verity_bench.vllm import arms as A
    from verity_bench.vllm import presets as P
    from verity_bench.vllm import runner as RN
    from verity_vllm.engine import env as engine_env

    spec = P.resolve_spec(args.spec)
    arm = pearl_c_arm(spec, args.arm)
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
    close = time.perf_counter() - t0 - wall
    caps = {"tiles": run.capacity, "units": PPD.CALL_CAPACITY, "screened_rows": len(run.flag_buf)}
    got = [{"pass": i, "timed": i >= warm, **{k: p.get(k) for k in KEEP if k in p}, "macs": macs(p.get("manifest"))}
           for i, p in enumerate(run.details["passes"])]
    held = [p for p in got if p["units"]]
    over = [p["pass"] for p in got if any((p.get(k) or 0) > cap for k, cap in caps.items())]
    doc = {"schema": "verity-pouw/pearl-c-served-passes/v1", "spec": str(args.spec), "arm": arm.id, "run": settings.RESEARCH_RUN_ID,
           "engine": {k: v for k, v in engine.items() if isinstance(v, (int, float, str, bool))}, "retained": args.retain,
           "requests": args.requests, "prompt_tokens": args.prompt_tokens, "output_tokens": args.output_tokens,
           "served_tokens": sum(len(o.outputs[0].token_ids) for o in outs), "generate_s": wall, "engine_steps": len(walls),
           "step_s": walls[warm_steps:], "close_s": close, "pass_ends": run.details.get("pass_ends"),
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


def manifests(passes: Path) -> tuple[list[Path], str | None]:
    """The run's passes with calls, in order, and its run id: those ``served.json`` names, else every
    ``pass-NNNNN/manifest.json`` the install retained under the transcript dir."""
    if (passes / "served.json").exists():
        served = json.loads((passes / "served.json").read_text())
        return [Path(p["manifest"]) for p in served["passes"] if p["units"]], served.get("run")
    return sorted(passes.glob("pass-*/manifest.json")), None


def timed_credit(served: dict, work: dict, coins) -> dict:
    """An accepted run's credit over the passes its timed call made (``served.json``'s ``timed``; a pass without the flag
    is timed), the work the timed call's wall bought: the run's credited MACs less the other passes' credited cells' MACs
    at (1 − ρ), rounded up, so all of the run's deduction for ε falls on the timed passes."""
    sys.path.insert(0, str(HERE))
    import verify_run as VR

    held = [p for p in served["passes"] if p["units"]]
    warm = sum(VR.pass_macs(VR.load(Path(p["manifest"]), coins)) for p in held if not p.get("timed", True))
    return {"passes": sum(1 for p in held if p.get("timed", True)), "macs": work["macs"] - warm,
            "credited": max(0, work["credited"] - math.ceil((1 - Fraction(work["rho"])) * warm))}


def volunteer(args: argparse.Namespace) -> int:
    from verity.core.protocols.pouw.coins import CoinCommitment
    sys.path.insert(0, str(HERE))
    import volunteer_pass as VP

    passes = Path(args.passes)
    served = json.loads((passes / "served.json").read_text()) if (passes / "served.json").exists() else None
    if served is not None and served.get("retained") is False:
        print(f"volunteer FAIL: {UNRETAINED[1]}", flush=True)
        return 1
    paths, _ = manifests(passes)
    coins = CoinCommitment.from_json(json.loads((passes / "coins.json").read_text()))
    got = []
    for path in paths:
        t = time.perf_counter()
        r = VP.volunteer(path, coins, args.jobs)
        got.append({**{k: r[k] for k in VOLUNTEER_KEEP}, "s": time.perf_counter() - t})
        print(f"{path.parent.name}: {r['line']}", flush=True)
    total = {k: sum(len(r[k]) for r in got) for k in ("over", "left")}
    given = {k: sum(len(rows) for r in got for rows in r[k].values()) for k in ("voluntary_a", "voluntary_b")}
    doc = {"schema": "verity-pouw/pearl-plus-served-volunteer/v1", "passes_dir": str(passes), "served_run": (served or {}).get("run"),
           "jobs": args.jobs, "passes": got, "tiles": sum(r["tiles"] for r in got), **total, **given,
           "s": sum(r["s"] for r in got), "ok": bool(got) and all(r["ok"] for r in got)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"volunteer {'PASS' if doc['ok'] else 'FAIL'}: {len(got)} passes, {doc['tiles']} tiles replayed in {doc['s']:.0f} s on "
          f"{args.jobs} processes, {doc['over']} over a cap; gave up {doc['voluntary_a']} A rows and {doc['voluntary_b']} B rows; "
          f"{doc['left']} tiles left over a cap", flush=True)
    return 0 if doc["ok"] else 1


def verify(args: argparse.Namespace) -> int:
    from verity.core.protocols.pouw.coins import CoinCommitment
    sys.path.insert(0, str(HERE))
    import verify_run as VR

    passes = Path(args.passes)
    served = (json.loads((passes / "served.json").read_text())
              if not args.manifest and (passes / "served.json").exists() else None)
    doc = {"schema": SCHEMA, "passes_dir": str(passes), "served_run": (served or {}).get("run"), "tie_record": args.tie}
    if served is not None and served.get("retained") is False:
        doc.update(refused=list(UNRETAINED), **VR.rejected(": ".join(UNRETAINED), args.commit))
    else:
        coins = CoinCommitment.from_json(json.loads((passes / "coins.json").read_text()))
        window = json.loads(Path(args.tie).read_text()) if args.tie else None
        doc.update(VR.verify_passes(args.manifest or manifests(passes)[0], args.draws, args.jobs, args.commit, coins,
                                    fast=args.fast, cross_check=args.cross_check, delta=args.delta, window=window))
        if served is not None and doc["answer"]["verdict"] == "accept":
            doc["timed"] = timed_credit(served, doc["answer"]["public"]["work"], coins)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    (Path(args.answer) if args.answer else out.parent / "answer.json").write_text(json.dumps(doc["answer"], indent=1) + "\n")
    print(doc["line"], flush=True)
    return 0 if doc["answer"]["verdict"] == "accept" else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--spec", type=Path, default=SPEC)
    s.add_argument("--arm", help=f"the spec's {ARM_PREFIX}* arm to serve (default: its one)")
    s.add_argument("--passes", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--requests", type=int, default=32)
    s.add_argument("--prompt-tokens", type=int, default=512)
    s.add_argument("--output-tokens", type=int, default=16)
    s.add_argument("--max-num-batched-tokens", type=int)
    s.add_argument("--no-retain", dest="retain", action="store_false")
    o = sub.add_parser("volunteer")
    o.add_argument("--passes", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    v = sub.add_parser("verify")
    v.add_argument("--passes", required=True)
    v.add_argument("--manifest", action="append", type=Path,
                   help="a pass's manifest, in order, repeated (default: those DIR/served.json names, else DIR/pass-*/)")
    v.add_argument("--tie", metavar="W", help="the window record Pearl+ wrote at the window's close: each call's A and y "
                   "entry must be its root and recomputed domain (verify_run.tie)")
    v.add_argument("--out", required=True)
    v.add_argument("--answer", help="where the core answer goes (default: answer.json beside --out)")
    v.add_argument("--draws", type=int, default=DRAWS)
    v.add_argument("--delta", type=Fraction, default=Fraction(1, 100))
    v.add_argument("--commit", default=settings.RESEARCH_SOURCE_SHA or "unknown")
    v.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    v.add_argument("--fast", action="store_true")
    v.add_argument("--cross-check", type=int, default=256)
    args = p.parse_args(argv)
    return {"serve": serve, "volunteer": volunteer, "verify": verify}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
