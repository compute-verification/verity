"""One served Pearl-C run through `verity_vllm.LLM` alone, every pass it commits retained, then the run's one verdict over
all of them: `pouw.install`'s pass boundary under serving (`pouw_pass`: each engine step's end, a step held to one pass's
``max_step_tokens``).

  served_passes.py serve --passes DIR --out served.json [--spec SPEC] [--arm ARM] [--requests 32] [--prompt-tokens 512]
                         [--output-tokens 16] [--max-num-batched-tokens N] [--no-retain [--seeded]]
  served_passes.py same A.json B.json --out same.json
  served_passes.py volunteer --passes DIR --out volunteer.json [--jobs J] [--device DEBIT_CUBIN [--device-check N]]
  served_passes.py volunteer-needed [--over-cap reject|uncredited] [--verdict run|pass]
  served_passes.py retain --passes DIR [--volunteer volunteer.json] [--drop ROUND ...] [--window-record W]
  served_passes.py verify --passes DIR [--manifest M ...] [--tie W [--class S,P,O]] --out verify.json [--answer answer.json]
                          [--draws 459] [--delta 1/100] [--jobs J] [--fast] [--coin-record R]
  served_passes.py lean --passes DIR --verify verify.json --exe pouw-pearlplus --out pearlplus-lean.json [--check C]
                        [--timeout S]
  served_passes.py case --passes DIR --verify verify.json --lean pearlplus-lean.json --out CASE_DIR
  served_passes.py arm --out config.json [--spec SPEC] [--arm ARM]

``arm`` writes the spec's Pearl-C arm's pouw settings (`pearl_c_arm`, each ``"$NAME"`` read from the environment the
spec's setup exports) as serve-openai's ``--protocols-config`` takes them for the same arm (`serve_pearl.config_of`): the
keys of `serve_pearl.CONFIG` alone, the server drawing each window's coins and naming its transcript dir itself.

``serve`` builds the spec's preset with its Pearl-C arm (`pearl_c_arm`: ``--arm``, else the spec's one; `verity_bench`'s
``engine_args`` and ``protocols_config``, whose in-process challenger is the epoch's coins) and the pouw setting
``transcript_dir`` DIR, so each pass is retained in its own ``DIR/pass-NNNNN``; the challenger's coin commitment goes to
``DIR/coins.json`` before the engine starts.  With ``PEARLC_CHALLENGER_GATE`` and ``PEARLC_CHALLENGER_PEER`` set the epoch's
salt coin is a live session's on the auditor host instead (`bench_arm.live_coins`: a salt-only session reserved under
``PEARLC_CHALLENGER_LANE`` before the engine starts, which the install opens on its salt epoch's root and closes at
``close``), and ``DIR/coins.json`` is that session's commitment, written once the install has opened it; the document's
``coins`` says which (`pearl_c.window.coins_of`: the source, the session without its secret, and how it closed).  A warm-up
``generate`` of one short request takes the first calls' one-time
costs (the screen's Triton build); then one timed ``generate`` call serves R requests of P seeded random prompt tokens and
O output tokens each (greedy, EOS ignored): vLLM's scheduler steps the engine, and nothing but the option ends a pass.  It
writes each pass's counts, commitment, manifest, linear MACs (its calls' real rows × n × k) and whether the timed call made
it (``timed``), the timed call's wall (``generate_s``), each of its engine steps' (``step_s``) and the engine's close
(``close_s``: what retaining completes behind the steps, waited for there), the engine steps, the
caps, the install's ``max_step_tokens`` and whether every pass is within the caps, to ``--out`` and ``DIR/served.json``,
and exits non-zero unless the timed call's passes hold calls in at least three (two boundaries crossed) and every pass is
within its caps.  ``--no-retain`` serves the same with no ``transcript_dir``: each pass is committed and none is written,
so its walls are the option's without retention, its passes have no manifest or MACs, and ``verify`` refuses it
(``retained`` false).  With ``--seeded`` too (`seed_install`; refused with retention or the challenger gate) the
in-process challenger's coins and the install's hm96 salt key are drawn from fixed streams (`seeded`), so its
commitments hide nothing and bind nothing a verdict may read: two seeded serves of one load draw the same salt and key.

``same`` holds two seeded serves of one load (their ``served.json``) to the same passes, field by field (`KEEP`: each
pass's transcript root, activation roots and counts), which they give only where the served path commits the same bytes
for the same load.  It writes the passes compared and the fields of each that differ, by name and never by value, to
``--out``, prints one line and exits 0 only when every pass is the same; it refuses serves that aren't both seeded, that
served different loads or that drew different salts, and a serve with nothing committed to compare (no pass with calls, or
one without its roots).

``volunteer`` is the prover's step between the serve and the verifier's coin (`volunteer_pass.volunteer`, V-EX's remedy): on
each pass with calls, as ``verify`` reads them, it replays every tile word by word on ``--jobs`` processes and gives up, in
the pass's records, the rows that leave a tile or one of its credited rows over its debit cap, which the serve's screen
can't see (the debit reads the noised codes); the transcript is the serve's.  With ``--device`` (pearl_c_sm90's
debit_sm90.cubin, on this node's H100 or H200) the words' counts are the card's (`volunteer_pass.words_device`), and
every ``--device-check``-th tile (default `volunteer_pass.CROSS_CHECK`) and each tile with a word the card marks
non-finite are replayed on the CPU too and held equal.  It writes each pass's outcome and seconds to ``--out`` as each
pass ends (``done`` once all have), with the node it ran on and, on the device, the cubin, that rate and what it
replayed (``device``), prints one line and exits 0 only when no pass has a tile left over a cap; it refuses a serve that
retained none, and any once the verifier's coin is drawn (``DIR/coin.json``).

``volunteer-needed`` prints ``yes`` and the over-cap rule (`over_cap_rule`) where the verdict a pipeline calls (``run``,
the default: ``verify``'s, `verify_run.verify_passes`; ``pass``: `verify_run.verify`, a pass at a time) rejects a run on
a drawn tile with a credited row or tile over its debit cap under ``--over-cap`` (default: the verdict's own rule), and
``no`` and the rule where it credits such a row nothing instead.  The pipelines (`served_pearlplus.sh`,
`pearl_c_sm90/h200.sh`) ask it before they serve: they run ``volunteer`` exactly when it says ``yes``, and refuse to skip
it then.

``retain`` is the prover's last step before the coin: it writes ``DIR/retention.json``, the rounds (places among
``served.json``'s passes with calls) the verdict is to read, dropping each ``--drop`` round and, with ``--volunteer``, each
pass that step didn't finish or left over a cap.  A dropped pass credits nothing and its MACs stay in what the run
declared.  It refuses once the coin is drawn.  With ``--window-record W`` (a tied run) it drops the warm-up's rounds too
(``timed`` false), which a tied run's W doesn't list, and after ``DIR/retention.json`` writes Pearl+'s window record W
over the rounds it retains, each under its round (`pearl_c.window.write`), for ``verify --tie W`` and the one-stage
audit tied to it.

``verify`` is Pearl+'s PoUW verdict on the run (`verify_run.verify_passes`): the passes ``--manifest`` names (a capture's
``prefill/`` and ``decode/`` beside its ``coins.json``), else those with calls that ``DIR/served.json`` names, each at its
round, less the rounds ``DIR/retention.json`` drops, with the verifier's coin recorded in ``DIR/coin.json``
(``--coin-record``; a verdict over other passes or records after it is refused, ``after-coin``), else every
``DIR/pass-NNNNN/manifest.json`` (a transcript dir a served engine retained), in order, each epoch re-started against
``DIR/coins.json``.  It first holds the passes together to distinct call indices of one salt (`credit.passes_distinct`:
the device numbers calls across the epoch, and a run whose passes share an index is refused before any replay), then
reads every pass, holds the window record ``--tie W`` (``retain --window-record W`` writes it) to them where one is named
(`verify_run.tie`: each call's ``pearl-a/<s>/<c>`` and ``pearl-y/<s>/<c>`` entry its root and its domain recomputed from
the salt and the call's index, and each weight's ``pearl-b/<w>`` its epoch's weight root and `audit.weight_binding`'s
domain; a W that states no version, written before W registered the weights, ties the calls alone, its weights named
untied, `verify_run.WEIGHTS_UNTIED`), and with ``--class SLOTS,PROMPT,OUTPUTS`` (the window's class, read only beside
``--tie``) refuses passes whose rows by round aren't that class's fixed load (`verify_run.held_to_load`), then
takes the verifier's coin (with ``PEARLC_CHALLENGER_GATE`` and ``PEARLC_CHALLENGER_PEER``
set, a draw session's on the auditor host opened on the run's commitment, `bench_arm.LiveDraw`, reserved under
``PEARLC_CHALLENGER_LANE`` once that lane has no period open, and as its first reserve after the Pearl-C period the salt
session joined closed (``served.json``'s ``coins.reserved.period``); else the node's, `verify_run.node_draw`; the format is
`verify_run`'s "The coin record"), and replays ``--draws`` work-weighted tiles over all of them and
each pass's exclusion tiles.  A tile that passes every check but its inert-work bound is credited nothing under
``--over-cap uncredited`` (the default, CMP-311 option A; the credit takes the draws on such tiles, X_O) and rejects the run
under ``--over-cap reject``; the verdict records which (``over_cap``).  Under ``--class`` the work an accept publishes
is in the grid form (`verify_run.grid_work`: the class's structural credit less the give-up padded up to H1's window
grid), and the exact credit is the profile's ``exact_work``.  The run of record takes ``--over-cap uncredited``
and volunteers no row (``volunteer-needed`` says ``no``, so the pipelines run no ``volunteer``): a volunteered A row
uncredits its whole band of 64 rows, and a B row its block of 64 columns (`pearl_c_work.UnitRecord.credited`), a third
or more of an honest run's credit, where the over-cap tiles it would spare cost under half a percent.  The passes are
of one salt and, but under ``--test-instance`` (a test instance's passes, whose verdict credits nothing), of the served instance, k_min 1,024 at 64 x 64 tiles.  It writes the details to ``--out``, with, where
``served.json`` says which passes the timed call made and no ``--manifest`` names others, the credit over those retained
alone (`timed_credit`), and the core answer (`answer_of`'s form, ``public.work.credited`` the MACs credited at Pearl's Assumption 1 at ``--delta``) to ``--answer``
(default ``answer.json`` beside ``--out``, the file `served_verdicts.sh` reads as a piece's core answer), prints one line
and exits 0 only on accept.  An ACCEPT is replay-verified (development shortcut; the protocol's check is a ZK proof,
PouwCheck).

``lean`` is the Lean verdict of record on the run `verify` judged (`PouwCheck.PearlPlus`, ``--exe`` its ``pouw-pearlplus``):
the passes ``verify --out``'s document (``--verify``) names, read as `verify` reads them and held to the names and SHA-256s
that document gives them, the epoch's ``DIR/coins.json``, that document's draws, over-cap rule and test instance, the
verifier's coin record it wrote (``DIR/coin.json``), which pins the Lean's source to the Python's, the W ``verify --tie``
held the passes to with the class ``verify --class`` held them at, and, on a live session's coins, ``--of-record``, under which the Lean takes only a challenger draw
(`lean_argv`, the one place the Lean's command line is written). It holds the Lean's printout to the Python's verdict on
every field both give (`LEAN_HELD`: the verdict, its reason and refusal, the run's commitment and source, the over-cap
rule, each opened tile's verdict, an accepted run's work and, where a class is held, the exact credit its grid form
replaces, ``exact_work``), and its ``tie`` to the Python's: the same W at the same class's load, or none on both. Where they agree it writes the Lean's answer to ``--out`` (`lean_answer`: the
printout with W at ``tie``), the file `served_verdicts.pearlplus_of_record` reads; where they don't, or the Lean doesn't
answer within ``--timeout`` seconds, it writes none and removes any there. Either way both verdicts, the Lean's command
line and the executable's SHA-256 go to ``--check`` (default `LEAN_CHECK` beside ``--out``). Prints one line and exits
0 on a written accept, 1 on a written reject and 2 when it wrote no answer.

``case`` writes, from the Lean verdict of record ``lean`` wrote (``--lean``), the parts of the run's case in the form
remote's composite of record reads it (`PouwCheck.PearlPlus.Protocol`, ``verity/pouw/pearl-plus/v1``) that need no
registration record, to ``--out`` (0700, its files 0600: the manifests carry the passes' salt key, and the files their
values): ``parts.json`` (`CASE_SCHEMA`: the params, of the epoch's ``DIR/coins.json``, the document's draws and its coin
record's run commitment and source; the retained manifests, in order; the evidence, each pass's files the Lean's opened
tiles read (`tile_files`), by path, under the SHA-512 of their bytes; the Lean's verdict, reason, tiles drawn and work;
and the document's δ) and ``blobs/<SHA-512>``, each of those files' bytes once. `served_cases.pearlplus_case` makes the
case of them once the window's registration record R is made, its commitment naming R's digest. It refuses (exit 2)
without a Lean verdict or a coin record, on retained passes other than those the document names, and on a Lean verdict
over another run or source.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import itertools
import json
import math
import os
import socket
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

from verity_config.settings import of
from verity_vllm.protocols.pouw.pearl_c.window import KEEP, SERVED_SCHEMA, coins_of, served_passes as passes_of

HERE = Path(__file__).resolve().parent
settings = of(__file__)
SPEC = HERE.parents[1] / "specs" / "llama31-8b-1gpu-pouw-protocol-served-b32.toml"
#: the arm `serve` serves without ``--arm``: the spec's one PoUW Pearl-C arm (``pouw:pearl-c-sm120-v1-h3`` on an RTX PRO
#: 6000, ``pouw:pearl-c-h100-v1-h3`` on an H100 or H200)
ARM_PREFIX = "pouw:pearl-c-"
SCHEMA = "verity-pouw/pearl-plus-served-verify/v1"
#: t = 459 accepted draws leave ε ≈ 1% of the tickets' weight on wrong tiles at δ = 1/100 (`verify_run.epsilon`)
DRAWS = 459
#: what `volunteer` keeps of each pass's `volunteer_pass.volunteer` (``device`` on the device alone)
VOLUNTEER_KEEP = ("manifest", "ok", "line", "tiles", "over", "voluntary_a", "voluntary_b", "left", "records_root",
                  "credit_kept", "device")
#: what `volunteer`'s document sums of its passes' ``device``
DEVICE_SUMS = ("replayed", "bad_tiles", "form_s", "device_s", "replay_s")
#: Pearl+'s rules for a drawn tile with a credited row or tile over its debit cap (``verify --over-cap``, `verify_run`'s
#: ``over_cap``): the verdict rejects the run, or credits the row nothing; ``volunteer`` is the prover's under the first
OVER_CAP = ("reject", "uncredited")
#: why `verify` and `volunteer` refuse a serve run with ``--no-retain``
UNRETAINED = ("unretained", "the serve retained no pass (serve --no-retain), so there is nothing to replay")
#: the verifier's record of its coin (`verify_run.verify_passes`' ``record``) and the prover's retention, in DIR
COIN, RETENTION = "coin.json", "retention.json"
#: the fields of the Lean `PouwCheck.PearlPlus` printout `lean` holds to the Python's verdict, as its difftest does
#: (`benchmarks/pouw/pearlplus_lean`'s ``answer``)
LEAN_HELD = ("verdict", "reason", "refused", "commitment", "source", "over_cap", "verdicts", "work", "exact_work")
#: `lean`'s record of both verdicts, beside its answer
LEAN_CHECK = "pearlplus-lean-check.json"
LEAN_SCHEMA = "verity-pouw/pearl-plus-lean-check/v1"
#: why the step refuses where the two verdicts disagree
LEAN_DISAGREES = "the Lean verdict and the Python's disagree, so no Lean answer is written"
#: `case`'s parts of the run's case, and its directory of the files they name, each under the SHA-512 of its bytes
CASE_SCHEMA = "verity-pouw/pearl-plus-case-parts/v1"
CASE_PARTS, CASE_BLOBS = "parts.json", "blobs"
#: why `retain` and `volunteer` refuse once the verifier's coin is drawn
DRAWN = "the verifier's coin is drawn (DIR/coin.json), so the passes and their records are fixed"
#: why ``retain --window-record`` drops a warm-up pass (`pearl_c.window`)
WARM_UP = "the warm-up's (timed false): a tied run's W lists only the timed call's rounds, so it credits nothing"
#: why `serve` refuses ``--seeded`` beside retention or a live session's coins
SEEDED_ALONE = ("serve --seeded fixes the coins and the hm96 salt key, so its commitments hide nothing and bind nothing a "
                "verdict may read: it takes --no-retain and no PEARLC_CHALLENGER_GATE")
#: what makes two serves' documents one load in `same`, beside the engine's batched tokens
LOAD = ("spec", "arm", "requests", "prompt_tokens", "output_tokens")


def prompts(vocab: int, requests: int, tokens: int) -> list[list[int]]:
    """`requests` prompts of `tokens` token ids each, seeded by their index (the harness's `_requests` form)."""
    seed = int.from_bytes(hashlib.sha256(b"served-passes").digest()[:8], "big")
    return [[3 + (seed + i * 1_000_003 + j * 97_409) % (vocab - 4) for j in range(tokens)] for i in range(requests)]


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


def seeded(label: str):
    """`serve --seeded`'s entropy for `label`: SHA-512 over the label and a counter, its whole outputs taken in order, so
    every stream of one label gives the same bytes however they are drawn."""
    buf, count = bytearray(), itertools.count()

    def draw(n: int) -> bytes:
        while len(buf) < n:
            buf.extend(hashlib.sha512(f"served-passes seeded {label}".encode() + next(count).to_bytes(8, "big")).digest())
        out = bytes(buf[:n])
        del buf[:n]
        return out
    return draw


def seed_install(pouw: dict) -> None:
    """`serve --seeded`: the PoUW settings' in-process challenger drawing from `seeded` coins, and the hm96 salt key the
    install's weight commitment takes (`pearl_c.new_salt_key`) a `seeded` one, in this process."""
    from verity.protocols.pouw.verifier.coins import Challenger
    from verity_vllm.protocols.pouw import pearl_c as PPD

    pouw["coins"] = Challenger(0, seeded("coins"))
    key = seeded("salt key")(32)
    PPD.new_salt_key = lambda scheme: key


def serve(args: argparse.Namespace) -> int:
    if args.seeded and (args.retain or settings.PEARLC_CHALLENGER_GATE):
        raise SystemExit(SEEDED_ALONE)
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
    sys.path.insert(0, str(HERE))
    import bench_arm as BA

    config["pouw"], reserved = BA.live_coins(config["pouw"], settings.PEARLC_CHALLENGER_GATE, settings.PEARLC_CHALLENGER_PEER,
                                             settings.PEARLC_BENCH_EPOCH, settings.PEARLC_CHALLENGER_LANE)
    if args.seeded:
        seed_install(config["pouw"])
    if reserved is None:
        (passes / "coins.json").write_text(json.dumps(config["pouw"]["coins"].commitment.to_json()) + "\n")
    engine_env.apply_engine(None)
    from verity_vllm import LLM
    from verity_vllm.protocols.pouw import pearl_c as PPD
    from vllm import SamplingParams

    llm = LLM(preset.model, revision=preset.revision, protocols=",".join(arm.protocols), protocols_config=config, **engine)
    option = llm.protocols.get("pouw")
    run = option.end_pass.__self__
    if reserved is not None:
        (passes / "coins.json").write_text(json.dumps(run.commits["coins"]) + "\n")
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
    got = passes_of(run.details["passes"], warm)
    held = [p for p in got if p["units"]]
    over = [p["pass"] for p in got if any((p.get(k) or 0) > cap for k, cap in caps.items())]
    doc = {"schema": SERVED_SCHEMA, "spec": str(args.spec), "arm": arm.id, "run": settings.RESEARCH_RUN_ID,
           "engine": {k: v for k, v in engine.items() if isinstance(v, (int, float, str, bool))}, "retained": args.retain,
           "seeded": args.seeded, "requests": args.requests, "prompt_tokens": args.prompt_tokens, "output_tokens": args.output_tokens,
           "served_tokens": sum(len(o.outputs[0].token_ids) for o in outs), "generate_s": wall, "engine_steps": len(walls),
           "step_s": walls[warm_steps:], "close_s": close, "pass_ends": run.details.get("pass_ends"),
           "max_step_tokens": run.details.get("max_step_tokens"), "caps": caps, "passes": got, "passes_with_calls": len(held),
           "over_caps": over,
           "ok": sum(p["timed"] for p in held) >= 3 and not over and (not args.retain or all(p.get("manifest") for p in held))}
    doc["coins"] = {**coins_of(option, reserved), **({"source": "seeded"} if args.seeded else {})}
    for path in (Path(args.out), passes / "served.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=1) + "\n")
    ts = doc["step_s"]
    shown = ", ".join(f"{s:.3f}" for s in ts[:8]) + (f", … ({len(ts)} steps)" if len(ts) > 8 else "")
    coins = doc["coins"]
    session = (coins["session"] or {}).get("session")
    whose = f" (session {session}, closed {coins['closed']})" if session else ""
    print(f"served {doc['served_tokens']} tokens in {wall:.3f} s and {len(walls)} engine steps (a warm-up's among them), "
          f"{'every pass retained' if args.retain else 'no pass retained'}: the timed steps {shown} s; {len(held)} passes "
          f"with calls, {len(over)} over their caps; coins {coins['source']}{whose} -> {'OK' if doc['ok'] else 'FAIL'}")
    return 0 if doc["ok"] else 1


def same(args: argparse.Namespace) -> int:
    missing = [str(p) for p in (args.a, args.b) if not Path(p).is_file()]
    docs = [{} if str(p) in missing else json.loads(Path(p).read_text()) for p in (args.a, args.b)]

    def load(d: dict) -> list:
        return [d.get(k) for k in LOAD] + [(d.get("engine") or {}).get("max_num_batched_tokens")]

    def bare(d: dict) -> bool:
        called = [p for p in d.get("passes") or [] if p.get("units")]
        return not called or not all(p.get("transcript_root") and p.get("activation_roots") for p in called)
    a, b = (d.get("passes") or [] for d in docs)
    refused = next((why for why, bad in (
        (f"a serve wrote no document ({', '.join(missing)})", missing),
        ("the serves aren't both seeded (serve --no-retain --seeded), so each committed under a salt and salt key of its "
         "own", not all(d.get("seeded") and d.get("retained") is False for d in docs)),
        ("the serves served different loads", load(docs[0]) != load(docs[1])),
        ("the serves drew different salts, so they weren't seeded alike",
         (docs[0].get("coins") or {}).get("salt") != (docs[1].get("coins") or {}).get("salt")),
        ("a serve committed nothing to compare: no pass with calls, or one without its transcript root and activation "
         "roots", any(map(bare, docs)))) if bad), None)
    differ = [] if refused else [{"pass": i, "fields": f} for i, (p, q) in enumerate(zip(a, b))
                                 if (f := [k for k in KEEP if p.get(k) != q.get(k)])]
    n, called, calls = len(a), sum(1 for p in a if p.get("units")), sum(p.get("units") or 0 for p in a)
    doc = {"schema": "verity-pouw/pearl-plus-served-same/v1", "serves": [str(args.a), str(args.b)],
           "load": dict(zip((*LOAD, "max_num_batched_tokens"), load(docs[0]))), "passes": [len(a), len(b)], "calls": calls,
           "refused": refused, "differ": differ, "ok": refused is None and len(a) == len(b) and not differ}
    r, p, o, t = (doc["load"][k] for k in ("requests", "prompt_tokens", "output_tokens", "max_num_batched_tokens"))
    head = f"same: two seeded serves of one load ({r} requests of {p} tokens, {o} out, {t} a step)"
    if refused:
        doc["line"] = f"same: refused: {refused} -> FAIL"
    elif doc["ok"]:
        doc["line"] = (f"{head}: {n} passes ({called} with calls), {calls} calls, every pass's transcript root, activation "
                       f"roots and counts equal, so the served path committed the same bytes for the same load -> OK")
    else:
        first = (f"; first pass {differ[0]['pass']}: {', '.join(differ[0]['fields'])}" if differ else "")
        doc["line"] = (f"{head}: {len(differ)} of {min(len(a), len(b))} passes differ{first}"
                       + (f"; {len(a)} passes against {len(b)}" if len(a) != len(b) else "") + " -> FAIL")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(doc["line"], flush=True)
    return 0 if doc["ok"] else 1


def manifests(passes: Path) -> tuple[list[Path], str | None]:
    """The run's passes with calls, in order, and its run id: those ``served.json`` names, else every
    ``pass-NNNNN/manifest.json`` the install retained under the transcript dir."""
    if (passes / "served.json").exists():
        served = json.loads((passes / "served.json").read_text())
        return [Path(p["manifest"]) for p in served["passes"] if p["units"]], served.get("run")
    return sorted(passes.glob("pass-*/manifest.json")), None


def retention(passes: Path, served: dict) -> tuple[list[Path], list[int], tuple[int, ...]]:
    """The passes the verdict reads, their rounds (each one's place among ``served.json``'s passes with calls) and the
    rounds dropped: ``DIR/retention.json``'s (`retain`), else every pass with calls.  ValueError on a retention whose
    rounds are not in order, distinct, and the served run's."""
    held = [p for p in served["passes"] if p["units"]]
    rounds = list(range(len(held)))
    if (passes / RETENTION).exists():
        got = json.loads((passes / RETENTION).read_text())["rounds"]
        if not (all(type(s) is int for s in got) and got == sorted(set(got)) and set(got) <= set(rounds)):
            raise ValueError(f"the retention's rounds {got} are not distinct rounds of the served run's {len(held)}, in order")
        rounds = got
    return [Path(held[s]["manifest"]) for s in rounds], rounds, tuple(s for s in range(len(held)) if s not in rounds)


def timed_credit(served: dict, work: dict, coins, rounds: list[int] | None = None) -> dict:
    """An accepted run's credit over the passes its timed call made (``served.json``'s ``timed``; a pass without the flag
    is timed), the work the timed call's wall bought: the run's credited MACs less the other retained passes' (``rounds``,
    default all) credited cells' MACs at (1 − ρ), rounded up, so all of the run's deduction for ε falls on the timed
    passes.  A dropped pass is in neither.  Where the work is in the grid form (`verify_run.grid_work`, a run held to its
    class's fixed load, which names its ``give_up``), the other passes' MACs are their whole records'
    (`verify_run.whole_pass`): the class's, and at least their credited cells', so the timed credit reads no more of the
    retained values than the work does and is at most the exact one."""
    sys.path.insert(0, str(HERE))
    import verify_run as VR

    held = [p for p in served["passes"] if p["units"]]
    held = held if rounds is None else [held[s] for s in rounds]
    form = VR.whole_pass if "give_up" in work else (lambda p: p)
    warm = sum(VR.pass_macs(form(VR.load(Path(p["manifest"]), coins))) for p in held if not p.get("timed", True))
    return {"passes": sum(1 for p in held if p.get("timed", True)), "macs": work["macs"] - warm,
            "credited": max(0, work["credited"] - math.ceil((1 - Fraction(work["rho"])) * warm))}


def volunteer(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(HERE))
    import verify_run as VR
    import volunteer_pass as VP

    passes = Path(args.passes)
    served = json.loads((passes / "served.json").read_text()) if (passes / "served.json").exists() else None
    if served is not None and served.get("retained") is False:
        print(f"volunteer FAIL: {UNRETAINED[1]}", flush=True)
        return 1
    if (passes / COIN).exists():
        print(f"volunteer FAIL: {DRAWN}", flush=True)
        return 1
    paths, _ = manifests(passes)
    coins = VR.epoch_coins(passes / "coins.json")
    out, got = Path(args.out), []
    out.parent.mkdir(parents=True, exist_ok=True)
    device, check = (Path(args.device) if args.device else None), args.device_check or VP.CROSS_CHECK

    def written(done: bool) -> dict:
        total = {k: sum(len(r[k]) for r in got) for k in ("over", "left")}
        given = {k: sum(len(rows) for r in got for rows in r[k].values()) for k in ("voluntary_a", "voluntary_b")}
        on = None if device is None else {
            "cubin": str(device), "cubin_sha256": hashlib.sha256(device.read_bytes()).hexdigest(), "cross_check": check,
            **{k: sum(r["device"][k] for r in got) for k in DEVICE_SUMS}}
        doc = {"schema": "verity-pouw/pearl-plus-served-volunteer/v1", "passes_dir": str(passes),
               "served_run": (served or {}).get("run"), "node": socket.gethostname().split(".")[0], "jobs": args.jobs,
               "device": on, "done": done, "of": len(paths), "passes": got,
               "tiles": sum(r["tiles"] for r in got), **total, **given, "s": sum(r["s"] for r in got),
               "ok": done and bool(got) and all(r["ok"] for r in got)}
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        os.replace(tmp, out)
        return doc
    for path in paths:
        t = time.perf_counter()
        r = VP.volunteer(path, coins, args.jobs, device, check)
        got.append({**{k: r[k] for k in VOLUNTEER_KEEP if k in r}, "s": time.perf_counter() - t})
        print(f"{path.parent.name}: {r['line']}", flush=True)
        written(False)
    doc = written(True)
    how = "replayed" if device is None else f"counted on the device ({doc['device']['replayed']} replayed on the CPU too)"
    print(f"volunteer {'PASS' if doc['ok'] else 'FAIL'}: {len(got)} passes, {doc['tiles']} tiles {how} in {doc['s']:.0f} s "
          f"on {args.jobs} processes, {doc['over']} over a cap; gave up {doc['voluntary_a']} A rows and "
          f"{doc['voluntary_b']} B rows; {doc['left']} tiles left over a cap", flush=True)
    return 0 if doc["ok"] else 1


def over_cap_rule(verdict, rule: str | None = None) -> str:
    """The rule `verdict` (`verify_run.verify_passes` or `verify_run.verify`) takes on a drawn tile with a credited row or
    tile over its debit cap, given ``over_cap=rule`` where `rule` isn't None: its ``over_cap`` default, or ``reject``
    where it takes no ``over_cap``, a verdict without it having that one rule."""
    took = inspect.signature(verdict).parameters.get("over_cap")
    if took is None:
        if rule is not None:
            raise ValueError(f"verify_run.{verdict.__name__} takes no over_cap (its one rule is reject), so no --over-cap")
        return OVER_CAP[0]
    rule = took.default if rule is None else rule
    if rule not in OVER_CAP:
        raise ValueError(f"verify_run.{verdict.__name__}'s over-cap rule {rule!r} is none of {', '.join(OVER_CAP)}")
    return rule


def volunteer_needed(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(HERE))
    import verify_run as VR

    verdict = VR.verify_passes if args.verdict == "run" else VR.verify
    try:
        rule = over_cap_rule(verdict, args.over_cap)
    except ValueError as e:
        print(f"volunteer-needed FAIL: {e}", file=sys.stderr, flush=True)
        return 2
    print("yes" if rule == OVER_CAP[0] else "no", rule, flush=True)
    return 0


def retain(args: argparse.Namespace) -> int:
    passes = Path(args.passes)
    served = json.loads((passes / "served.json").read_text())
    if served.get("retained") is False or (passes / COIN).exists():
        print(f"retain FAIL: {UNRETAINED[1] if served.get('retained') is False else DRAWN}", flush=True)
        return 1
    held = [p for p in served["passes"] if p["units"]]
    if bad := [s for s in args.drop if not 0 <= s < len(held)]:
        print(f"retain FAIL: no round {bad[0]} among the served run's {len(held)} passes with calls", flush=True)
        return 1
    why = {s: "dropped by the prover (--drop)" for s in args.drop}
    if args.volunteer:
        vol = {p["manifest"]: p for p in json.loads(Path(args.volunteer).read_text())["passes"]}
        for s, p in enumerate(held):
            v = vol.get(p["manifest"])
            if v is None:
                why.setdefault(s, "its replay did not finish before the coin (served_passes.py volunteer)")
            elif not v["ok"]:
                why.setdefault(s, f"{len(v['left'])} tiles left over a cap after its give-up")
    if args.window_record:
        for s, p in enumerate(held):
            if not p.get("timed", True):
                why.setdefault(s, WARM_UP)
    rounds = [s for s in range(len(held)) if s not in why]
    doc = {"schema": "verity-pouw/pearl-plus-retention/v1", "served_run": served.get("run"), "volunteer": args.volunteer,
           "rounds": rounds, "dropped": [{"round": s, "pass": held[s].get("pass"), "manifest": held[s]["manifest"],
                                          "macs": held[s].get("macs"), "why": why[s]} for s in sorted(why)]}
    (passes / RETENTION).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"retain {'PASS' if rounds else 'FAIL'}: {len(rounds)} of {len(held)} passes with calls retained, "
          f"{len(why)} dropped before the coin (credit nothing, declared still)"
          + "".join(f"; round {d['round']}: {d['why']}" for d in doc["dropped"][:8]), flush=True)
    if not rounds:
        return 1
    if args.window_record:
        from verity.protocols.spot.verifier.one_stage import registration as REG
        from verity_vllm.protocols.pouw.pearl_c import window as PWR

        try:
            w = PWR.write(args.window_record, served["passes"], rounds)
        except (KeyError, OSError, ValueError) as e:
            print(f"retain FAIL: W: {type(e).__name__}: {e}", flush=True)
            return 1
        print(f"retain W: {args.window_record}, rounds {rounds[0]}..{rounds[-1]} ({len(rounds)}), {len(w['registered'])} "
              f"values, digest {REG.digest(w)[:16]}…", flush=True)
    return 0


def verify(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(HERE))
    import bench_arm as BA
    import verify_run as VR

    passes = Path(args.passes)
    served = json.loads((passes / "served.json").read_text()) if (passes / "served.json").exists() else None
    record = Path(args.coin_record) if args.coin_record else (passes / COIN if served is not None and not args.manifest else None)
    doc = {"schema": SCHEMA, "passes_dir": str(passes), "served_run": (served or {}).get("run"), "tie_record": args.tie,
           "class": None if args.cls is None else list(args.cls), "coin_record": None if record is None else str(record)}
    paths, rounds, dropped = None, None, ()
    if served is not None and served.get("retained") is False:
        doc.update(refused=list(UNRETAINED), **VR.rejected(": ".join(UNRETAINED), args.commit))
    elif served is not None and not args.manifest:
        try:
            paths, rounds, dropped = retention(passes, served)
        except (KeyError, TypeError, ValueError) as e:
            doc.update(VR.rejected(f"the retention: {type(e).__name__}: {e}", args.commit))
    else:
        paths = args.manifest or manifests(passes)[0]
    if paths is not None:
        coins = VR.epoch_coins(passes / "coins.json")
        window = json.loads(Path(args.tie).read_text()) if args.tie else None
        period = (((served or {}).get("coins") or {}).get("reserved") or {}).get("period")
        draw = BA.live_draw(settings.PEARLC_CHALLENGER_GATE, settings.PEARLC_CHALLENGER_PEER,
                            settings.PEARLC_CHALLENGER_LANE, period=period) or VR.node_draw
        doc.update(VR.verify_passes(paths, args.draws, args.jobs, args.commit, coins, fast=args.fast,
                                    cross_check=args.cross_check, delta=args.delta, window=window, rounds=rounds,
                                    dropped=dropped, record=record, draw=draw,
                                    over_cap=over_cap_rule(VR.verify_passes, args.over_cap),
                                    test_instance=args.test_instance, window_class=args.cls))
        if served is not None and not args.manifest and doc["answer"]["verdict"] == "accept":
            doc["timed"] = timed_credit(served, doc["answer"]["public"]["work"], coins, rounds)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    (Path(args.answer) if args.answer else out.parent / "answer.json").write_text(json.dumps(doc["answer"], indent=1) + "\n")
    print(doc["line"], flush=True)
    return 0 if doc["answer"]["verdict"] == "accept" else 1


def arm(args: argparse.Namespace) -> int:
    from verity_bench.vllm import arms as A
    from verity_bench.vllm import presets as P
    from verity_vllm.pipeline import serve_pearl as SPL

    picked = pearl_c_arm(P.resolve_spec(args.spec), args.arm)
    if tuple(picked.protocols) != (SPL.ENTRY,):
        raise SystemExit(f"served_passes.py arm: serve-openai serves {SPL.ENTRY} alone; {picked.id} is {list(picked.protocols)}")
    # read with no protocol enabled, so no in-process challenger draws coins the server draws per window itself
    pouw = A.protocols_config(dataclasses.replace(picked, protocols=()), os.environ).get("pouw", {})
    doc = SPL.config_of(json.dumps({"pouw": {k: v for k, v in pouw.items() if k in SPL.CONFIG}}))
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"arm {picked.id}: --protocols-config {args.out}, pouw {sorted(doc['pouw'])}", flush=True)
    return 0


def window_class(text: str) -> tuple[int, int, int]:
    """``--class``'s SLOTS,PROMPT,OUTPUTS, each a positive integer (`verify_run.load_class`'s form)."""
    try:
        cls = tuple(int(x) for x in text.split(","))
    except ValueError:
        cls = ()
    if len(cls) != 3 or min(cls) < 1:
        raise argparse.ArgumentTypeError(f"a window class is SLOTS,PROMPT,OUTPUTS, each a positive integer, not {text!r}")
    return cls


def lean_argv(exe: Path, coins: Path, draws: int, record: Path, manifests: list[Path], tie: Path | None,
              over_cap: str = OVER_CAP[1], test_instance: bool = False, of_record: bool = False,
              load: list[int] | None = None) -> list[str]:
    """``pouw-pearlplus``'s command line on the run's retained passes (`manifests`, in order), the epoch's coin commitment,
    the draws, the verifier's coin record, which pins the Lean's source to the one the Python drew, the over-cap rule and
    test instance the Python ran under, the window record W (`tie`) the passes are held to where there is one, with
    the class whose fixed load they are held at (`load`, ``verify --class``'s), and ``--of-record`` (`of_record`, a run
    on a live session's coins), under which the Lean takes only a challenger draw:
    ``--coins COINS --draws T --record RECORD --over-cap RULE [--test-instance] [--tie W [--class S,P,O]] [--of-record]
    MANIFEST...``."""
    return [str(exe), "--coins", str(coins), "--draws", str(draws), "--record", str(record), "--over-cap", over_cap,
            *(["--test-instance"] if test_instance else []), *(["--tie", str(tie)] if tie is not None else []),
            *(["--class", ",".join(map(str, load))] if load is not None else []),
            *(["--of-record"] if of_record else []), *map(str, manifests)]


def lean_answer(got: dict) -> dict:
    """The Lean's printout as `served_verdicts.pearlplus_of_record` reads it: as printed, with the W it held the passes to
    at ``tie`` (``{"record": digest(W), ...}``), from its ``tie`` or its ``public.tie``, else None."""
    tie = got.get("tie") or (got.get("public") or {}).get("tie")
    return {**got, "tie": {"record": tie} if isinstance(tie, str) else tie}


def python_answer(doc: dict) -> dict:
    """`verify`'s document in the Lean's terms (`LEAN_HELD`)."""
    a = doc["answer"]
    return {"verdict": a["verdict"], "reason": a["reason"], "refused": doc.get("refused"),
            "commitment": doc.get("commitment"), "source": doc.get("source"), "over_cap": doc.get("over_cap"),
            "verdicts": doc.get("verdicts", []),
            "work": (a.get("public") or {}).get("work"), "exact_work": (a.get("profile") or {}).get("exact_work")}


def lean(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(HERE))
    import verify_run as VR

    passes, out, exe = Path(args.passes), Path(args.out), Path(args.exe)
    check = Path(args.check) if args.check else out.with_name(LEAN_CHECK)
    out.unlink(missing_ok=True)
    doc = json.loads(Path(args.verify).read_text())
    rec = {"schema": LEAN_SCHEMA, "verify": str(args.verify), "exe": str(exe), "exe_sha256": None, "argv": None, "rc": None,
           "stderr": None, "lean": None, "python": python_answer(doc), "differ": None, "answer": None, "reason": None}

    def done(reason: str | None, rc: int) -> int:
        rec["reason"] = reason
        check.parent.mkdir(parents=True, exist_ok=True)
        check.write_text(json.dumps(rec, indent=1) + "\n")
        if reason is None:
            print(f"lean {rec['lean']['verdict'].upper()} {VR.PROTOCOL}: the Lean PouwCheck.PearlPlus verdict is the "
                  f"Python's on {', '.join(LEAN_HELD)}; run {str(rec['lean']['commitment'])[:16]}, W "
                  f"{str((rec['answer']['tie'] or {}).get('record'))[:16]} -> {out.name}", flush=True)
        else:
            print(f"lean REFUSED: {reason}; both verdicts in {check.name}", flush=True)
        return rc

    record = Path(doc["coin_record"]) if doc.get("coin_record") else None
    if record is None or not record.is_file():
        return done("the Python verdict recorded no coin (verify's coin record), so nothing pins the Lean's source", 2)
    served = json.loads((passes / "served.json").read_text()) if (passes / "served.json").exists() else None
    try:
        paths = retention(passes, served)[0] if served is not None else manifests(passes)[0]
        named = [VR.read(p)[2] for p in paths]
    except (KeyError, OSError, TypeError, ValueError) as e:
        return done(f"the retained passes can't be read: {type(e).__name__}: {e}", 2)
    if named != doc.get("passes"):
        return done("the retained passes are not the ones the Python verdict read, by name and SHA-256", 2)
    if not exe.is_file():
        return done(f"no Lean executable at {exe}", 2)
    rec["exe_sha256"] = hashlib.sha256(exe.read_bytes()).hexdigest()
    try:
        live = isinstance(VR.epoch_coins(passes / "coins.json"), VR.LiveCommitment)
    except (OSError, TypeError, ValueError) as e:
        return done(f"the epoch's coins can't be read: {type(e).__name__}: {e}", 2)
    rec["argv"] = lean_argv(exe, passes / "coins.json", doc["draws"], record, paths,
                            Path(doc["tie_record"]) if doc.get("tie_record") else None, doc.get("over_cap", OVER_CAP[0]),
                            doc.get("test_instance", False), of_record=live, load=doc.get("class"))
    try:
        p = subprocess.run(rec["argv"], capture_output=True, text=True, timeout=args.timeout, check=False)
    except subprocess.TimeoutExpired:
        return done(f"the Lean gave no verdict within {args.timeout} s", 2)
    rec["rc"], rec["stderr"] = p.returncode, p.stderr[-2000:] or None
    said = p.stdout.strip().splitlines()
    try:
        got = json.loads(said[-1]) if said else None
    except ValueError as e:
        rec["unparsed"] = f"the Lean's last line isn't JSON: {e}"
        got = None
    if not isinstance(got, dict) or p.returncode != (got.get("verdict") != "accept"):
        return done(f"the Lean exited {p.returncode} without a verdict: {(p.stdout + p.stderr).strip()[-400:]}", 2)
    rec["lean"] = got
    answer, want = lean_answer(got), rec["python"]
    differ = [k for k in LEAN_HELD if got.get(k) != want[k]]
    if got.get("drawn") not in (None, doc["draws"]):
        differ.append("drawn")
    tie = answer["tie"]
    if (tie is not None and not isinstance(tie, dict)) or any((tie or {}).get(k) != (doc.get("tie") or {}).get(k)
                                                              for k in ("record", "weights", "load")):
        differ.append("tie")
    rec["differ"] = differ
    if differ:
        return done(f"{LEAN_DISAGREES}: they differ in {', '.join(differ)}", 2)
    rec["answer"] = answer
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(answer, indent=1) + "\n")
    return done(None, 0 if got["verdict"] == "accept" else 1)


def tile_files(p, tile: int) -> list[dict]:
    """The manifest entries the opening of pass ``p``'s tile ``tile`` reads (`verify_run.Opened`'s ``open`` and
    ``open_y``, the Lean's ``filesOf``): its call's rows and tree, its weight's, the pass's leaves and transcript tree, and
    its call's y where the call commits one."""
    pos, man = p.verifier.refs[tile].matmul, p.man
    u, wt = man["matmuls"][pos], man["weights"][str(p.epoch.workload.matmuls[pos].weight)]
    return [u["rows"], u["tree"], wt["rows"], wt["tree"], man["leaves"], man["tree_t"],
            *((u["y"]["rows"], u["y"]["tree"]) if "y" in u else ())]


def _private(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.chmod(path, 0o600)


def case(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(HERE))
    import verify_run as VR

    passes, out = Path(args.passes), Path(args.out)

    def refuse(reason: str) -> int:
        print(f"case REFUSED: {reason}", flush=True)
        return 2

    doc = json.loads(Path(args.verify).read_text())
    try:
        lean = json.loads(Path(args.lean).read_text())
    except (OSError, ValueError) as e:
        return refuse(f"no Lean verdict of record ({type(e).__name__}: {e})")
    record = Path(doc["coin_record"]) if doc.get("coin_record") else None
    if record is None or not record.is_file():
        return refuse("the Python verdict recorded no coin (verify's coin record)")
    coin = json.loads(record.read_text())
    served = json.loads((passes / "served.json").read_text()) if (passes / "served.json").exists() else None
    paths = retention(passes, served)[0] if served is not None else manifests(passes)[0]
    if [VR.read(p)[2] for p in paths] != doc.get("passes"):
        return refuse("the retained passes are not the ones the Python verdict read, by name and SHA-256")
    if (lean.get("commitment"), lean.get("source")) != (coin["commitment"], coin["source"]):
        return refuse("the Lean verdict is not on the coin record's run and source")
    coins = VR.epoch_coins(passes / "coins.json")
    loaded = [VR.load(p, coins) for p in paths]
    files: list[dict[str, str]] = [{} for _ in paths]
    held: dict[str, int] = {}
    blobs = out / CASE_BLOBS
    blobs.mkdir(parents=True, exist_ok=True, mode=0o700)
    for d in (out, blobs):
        os.chmod(d, 0o700)
    for v in lean.get("verdicts") or []:
        i = v["pass"]
        for spec in tile_files(loaded[i], v["tile"]):
            if spec["path"] in files[i]:
                continue
            data = (paths[i].parent / spec["path"]).read_bytes()
            files[i][spec["path"]] = h = hashlib.sha512(data).hexdigest()
            if h not in held:
                _private(blobs / h, data)
                held[h] = len(data)
    parts = {"schema": CASE_SCHEMA,
             "params": {"coins": json.loads((passes / "coins.json").read_text()), "draws": doc["draws"],
                        "record": {"commitment": coin["commitment"], "source": coin["source"]}},
             "manifests": [json.loads(p.read_text()) for p in paths], "evidence": {"files": files},
             "lean": {k: lean.get(k) for k in ("verdict", "reason", "drawn", "work")}, "delta": doc["delta"]}
    _private(out / CASE_PARTS, (json.dumps(parts, sort_keys=True) + "\n").encode())
    print(f"case {lean.get('verdict')}: {len(paths)} passes, {sum(map(len, files))} files their "
          f"{len(lean.get('verdicts') or [])} opened tiles read, {len(held)} blobs of {sum(held.values()):,} bytes -> {out}",
          flush=True)
    return 0


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
    s.add_argument("--seeded", action="store_true", help="with --no-retain: coins and the hm96 salt key from fixed streams "
                   "(`seeded`), for `same`")
    m = sub.add_parser("same")
    m.add_argument("a", type=Path, help="a seeded serve's served.json")
    m.add_argument("b", type=Path, help="another of the same load")
    m.add_argument("--out", required=True)
    o = sub.add_parser("volunteer")
    o.add_argument("--passes", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    o.add_argument("--device", help="pearl_c_sm90's debit_sm90.cubin: the words' counts on this node's H100 or H200")
    o.add_argument("--device-check", type=int, metavar="N",
                   help="with --device, replay every Nth tile on the CPU too (default volunteer_pass.CROSS_CHECK)")
    n = sub.add_parser("volunteer-needed")
    n.add_argument("--over-cap", choices=OVER_CAP, help="the rule verify is to take (default: the verdict's own)")
    n.add_argument("--verdict", choices=("run", "pass"), default="run",
                   help="run: verify's (verify_run.verify_passes); pass: verify_run.verify's, a pass at a time")
    r = sub.add_parser("retain")
    r.add_argument("--passes", required=True)
    r.add_argument("--volunteer", help="volunteer's document: drop each pass it didn't finish or left over a cap")
    r.add_argument("--drop", type=int, action="append", default=[], metavar="ROUND",
                   help="drop this round (its place among served.json's passes with calls), repeated")
    r.add_argument("--window-record", metavar="W", help="a tied run: drop the warm-up's rounds too and write Pearl+'s window "
                   "record W over the rounds retained, after the retention (pearl_c.window), for verify --tie W")
    v = sub.add_parser("verify")
    v.add_argument("--passes", required=True)
    v.add_argument("--manifest", action="append", type=Path,
                   help="a pass's manifest, in order, repeated (default: those DIR/served.json names, else DIR/pass-*/)")
    v.add_argument("--tie", metavar="W", help="the window record retain --window-record wrote: each retained round's "
                   "calls' A and y entry must be its root and recomputed domain (verify_run.tie)")
    v.add_argument("--class", dest="cls", type=window_class, metavar="SLOTS,PROMPT,OUTPUTS",
                   help="with --tie, the window's class: passes whose rows by round aren't its fixed load are refused "
                        "(verify_run.held_to_load)")
    v.add_argument("--out", required=True)
    v.add_argument("--answer", help="where the core answer goes (default: answer.json beside --out)")
    v.add_argument("--draws", type=int, default=DRAWS)
    v.add_argument("--delta", type=Fraction, default=Fraction(1, 100))
    v.add_argument("--commit", default=settings.RESEARCH_SOURCE_SHA or "unknown")
    v.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    v.add_argument("--fast", action="store_true")
    v.add_argument("--cross-check", type=int, default=256)
    v.add_argument("--coin-record", help=f"the verifier's coin record (default: DIR/{COIN} where served.json names the "
                   "passes, none with --manifest)")
    v.add_argument("--over-cap", choices=OVER_CAP, help="the rule on a drawn tile over its inert-work bound: uncredited "
                   "credits it nothing, reject rejects the run (default: verify_run.verify_passes' own, uncredited)")
    v.add_argument("--test-instance", action="store_true", help="the passes are a test instance's: their tile may be "
                   "other than 64 x 64, and the verdict credits nothing")
    a = sub.add_parser("arm")
    a.add_argument("--spec", type=Path, default=SPEC)
    a.add_argument("--arm", help=f"the spec's {ARM_PREFIX}* arm (default: its one)")
    a.add_argument("--out", required=True)
    n = sub.add_parser("lean")
    n.add_argument("--passes", required=True)
    n.add_argument("--verify", required=True, help="verify's --out document")
    n.add_argument("--exe", required=True, help="PouwCheck.PearlPlus's pouw-pearlplus, built")
    n.add_argument("--out", required=True, help="where the Lean's answer goes where it agrees with the Python's")
    n.add_argument("--check", help=f"where both verdicts go (default: {LEAN_CHECK} beside --out)")
    n.add_argument("--timeout", type=float, default=3600, help="seconds the Lean has to answer (default 3,600)")
    c = sub.add_parser("case")
    c.add_argument("--passes", required=True)
    c.add_argument("--verify", required=True, help="verify's --out document")
    c.add_argument("--lean", required=True, help="lean's --out answer: the Lean verdict of record")
    c.add_argument("--out", required=True, help="the case's parts and blobs (0700)")
    args = p.parse_args(argv)
    if args.cmd == "verify" and args.cls is not None and not args.tie:
        p.error("verify --class holds the passes --tie W ties to its class's fixed load: it is read only beside --tie")
    return {"serve": serve, "same": same, "volunteer": volunteer, "volunteer-needed": volunteer_needed, "retain": retain,
            "verify": verify, "lean": lean, "case": case, "arm": arm}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
