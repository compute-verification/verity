"""PoUW's MVP end to end on the RTX PRO 6000 (sm_120): Pearl-C (`pearl-c-sm120-v1-h3`) commits every linear call of a vLLM model
on the device, timed against stock vLLM on the same card at the panel's two shapes, rep by rep.

  python benchmarks/pouw/pearl_c_vllm/e2e.py --model DIR --kernel benchmarks/pouw/pearl_c_sm120 --cubin SHIP/pearl_c_sm120.cubin \\
      --out RUN_DIR/e2e.json [--retain DIR] [--reps 3]

The run:
- **Model:** Llama-3.1-8B-Instruct, TP 1, eager, prefix caching off.  Every linear is in the kernels' domain: k is 4,096 or
  14,336, and n is 6,144, 4,096 or 28,672.
- **Two engines stay resident, each in its own process on the one GPU:**
  - ``base``: stock vLLM ``quantization="fp8"``, the baseline the panel carries;
  - ``arm``: the BF16 model with every LinearBase apply wrapped.  Per rep the wrapper is off (``bf16``, stock vLLM) or runs
    the `PearlCDeviceRun` without hashing (``pearlc-nohash``) or with it (``pearlc``).
  The controller sends one command at a time, so each engine is idle while the other times.  The reps alternate
  (`CYCLE`: fp8, pearlc, fp8, pearlc-nohash, fp8, bf16), each rep on the same prompts in every mode.  If the second engine
  can't start beside the first, the run falls back to ABAB blocks (`BLOCKS` engine lifetimes, alternating), and records it.
- **Prefill** (the panel's m = 8,192): one prompt of 8,192 random tokens with max_tokens 1 and a batch budget of 8,192 tokens,
  so each linear call has m = 8,192.  ``--prefill P`` sets the prompt's tokens and the batch budget (`configure`); off 8,192
  its key is ``prefill_<P>``, so `panel_rows.py` (``prefill_8192``) never reads it as the panel's prefill row.
- **Decode** (the panel's m = 32): 32 prompts of 64 tokens.  The step is (t(1 + D tokens) − t(1 token)) / D over D = 64
  decode steps, whose linear calls have m = 32.  ``--batch B`` and ``--prompt P`` set the decode batch and its prompts'
  length (a sweep's point; `configure`): its key is ``decode_b<B>_step``, with ``_p<P>`` before ``_step`` off 64 tokens.
- **The split:** ``pearlc`` over ``fp8``, the headline, and over ``bf16``; the GEMM (``gemm_pearlc``'s kernel time in
  ``pearlc-nohash``), hashing and records (``pearlc`` less ``pearlc-nohash``), and everything else.
- **Gates, in the arm's process, before any timing:** at every distinct linear shape and both m, served y is held to the
  float product x·Wᵀ (relative RMS error under 8%, the scheme's tested bound), with hashing and without.  With hashing, the
  same call made twice, each from a fresh pass on poisoned outputs, must give the same A root and tile root, and the
  negative control, the call once more with its tile leaves omitted, must not.
- **The verify passes** (``--retain``): after the timing, the first timed prefill prompt and the decode prompts are replayed
  in the arm, each a retained pass on poisoned buffers, whose commitment must equal its timed pass's.  Then the negative
  control: one decode prompt as a retained pass writing none of the commitment (`CONTROL_OMITS`).  The three manifests go to
  `verify_run.py` after the window: the first two must be replay-verified (development shortcut; the protocol's check is
  a ZK proof, PROTOCOL.md) and the control rejected (the panel's rule).
- **The panel's shapes:** `SHAPES`, under ``pearl-c-sm120`` v1-h3; `panel_rows.py` writes the rows.

``--whole-step`` (with ``--graphs``) builds the arm engine with vLLM's FULL decode graphs (no compile, the decode batch's
size only, or ``--capture``'s sizes), and each Pearl-C arm's decode forward is captured whole with its calls' graph steps inside
(`pouw_pearl_c_whole`): a decode step is one graph replay after one slot upload.  Prefill runs eagerly; a retained decode
replays its own capture, which also keeps each call's A words and tree.  Its gate (`whole_check`) holds a forward's calls
replayed whole to the eager calls, and a retained pass replayed whole to an eager one, bit for bit; the decode response
counts the steps replayed and run eagerly.  vLLM's eager decode isn't its graph's bit for bit, so the verify pass's eager
check compares Pearl-C's graphed calls with its eager ones both under vLLM's eager decode.  ``--whole-defer`` (with
``--whole-step``) captures each call's tile hashing beside the next call (`pouw_pearl_c_graphs`), ``--whole-defer-a``
(with ``--whole-defer``) each call's A tree beside its own forming and GEMM, ``--whole-tree-sm`` (with
``--whole-defer-a``) that GEMM leaving the tree's CTA its SM, and ``--whole-fork-stats`` (with ``--whole-defer``) each
call's A scales beside its A commitment, under the same gate.

``--statement CUBIN`` (with ``--whole-step``; hm96_rows.cu built for sm_120a) adds the mode ``pearlc-zk``: ``pearlc`` with
each decode forward replayed whole also committing its calls' A rows and its tile rows in the ZK statement's format,
hm96-sha512 row leaves of frame-v3-sha512 trees, once after the forward (served_zk/statement.py), beside the scheme's own
-h3 trees.  Its gate (`Statement.check`) holds the device's roots to Python's over the rows a whole replay staged, and
the no-write control's not; prefill's rows are not committed in it.

Every rep records the SM clock (the panel's clock rule).  ``--smoke`` (`smoke`) is the untimed check before a window: both
engines resident, the arm's install and gates, and each engine's first sampling calls.  ``--perplexity`` (`perplexity`) is
the untimed quality run: WikiText-2's test split through bf16, stock fp8, and Pearl-C v1 and v2.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import ClassVar

from verity_config.settings import of

settings = of(__file__)
settings.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SCHEME = "pearl-c-sm120-v1-h3"
MODES = ("bf16", "fp8", "pearlc-nohash", "pearlc")
PEARLC = ("pearlc-nohash", "pearlc", "pearlc-zk")
GROUP = {"fp8": "base", "fp8-graphs": "base-graphs", "bf16": "arm", "pearlc-nohash": "arm", "pearlc": "arm", "pearlc-zk": "arm"}
#: ``--statement``: ``pearlc`` with each decode forward's rows committed in the ZK statement's format too, once after the
#: forward (served_zk/statement.py), timed after each ``pearlc`` rep
ZK = ("pearlc-zk",)
#: ``--fp8-graphs``: stock vLLM FP8 as vLLM serves it, compiled with its CUDA graphs (the decode headline's baseline),
#: in an engine of its own beside the eager one, timed after each eager ``fp8`` rep
GRAPHED = ("fp8-graphs",)
#: the graphed engine's CUDA-graph batch sizes at the default decode batch; `configure` keeps those under ``--batch`` and adds it
CAPTURE_32 = [1, 2, 4, 8, 16, 24, 32]
CAPTURE_SIZES = CAPTURE_32
#: one rep of every mode, the baseline between each other mode's
CYCLE = ("fp8", "pearlc", "fp8", "pearlc-nohash", "fp8", "bf16")
#: the fallback's engine lifetimes: base, arm, base, arm, each timing its share of the reps
BLOCKS = 4
#: the decode batch and its prompts' length are ``--batch`` and ``--prompt`` (`configure`); these are their defaults, the panel's
PREFILL, BATCH, PROMPT, STEPS = 8192, 32, 64, 64
PREFILL_DEFAULT, BATCH_DEFAULT, PROMPT_DEFAULT = PREFILL, BATCH, PROMPT
#: the served y's bound against the float product (the scheme's tested bound, verity/core/protocols/pouw's sm120 useful-output test)
ERR = 0.08
#: the negative control's omitted steps: every write of the commitment (A's row digests and row tree, the tile leaves)
CONTROL_OMITS = frozenset({"pad_a", "rowdig_a", "hash_rows_a", "tree_a", "hash_leaf"})
#: the verify's no-write controls, each a retained pass the reference replay must reject: every commitment write omitted, and (under
#: ``--graphs``, whose leaves reach the pass through a copy) the tile leaves alone
CONTROLS = {"control": CONTROL_OMITS, "control-leaves": frozenset({"hash_leaf"})}
#: ``--smoke --retain``'s served requests, each one chat prompt (vllm_bench's PROMPTS, by index) as a retained pass: the
#: request whose calls a ZK proof opens (served_zk), and another request, whose commitment is that proof's foreign control.
#: Each is served at the fixed length SERVED_TOKENS under ignore_eos, and how it was served goes
#: beside its pass as SERVED_RECORD (``{"ignore_eos", "max_tokens", "served"}``), which served_zk's committer registers
SERVED, SERVED_TOKENS = {"request": 0, "other": 1}, 4
SERVED_RECORD = "served.json"
#: the quality run's schemes: Pearl-C v1 (G = 4) and v2 (no promotion), each in an arm engine of its own
QUALITY = {"pearlc-v1": "pearl-c-sm120-v1-h3", "pearlc-v2": "pearl-c-sm120-unpromoted-v1-h3"}
#: WikiText-2 (raw), its test split, at the revision cached on node 2
WIKITEXT = ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3", "wikitext-2-raw-v1/test-00000-of-00001.parquet")
#: the quality run's windows: BOS and 2,047 tokens each, non-overlapping; its batch budget, one window a step
WINDOW, PPL_BATCH, PPL_WINDOWS_A_PASS = 2048, 2048, 8
#: the panel's shape names for the two phases
SHAPES = {"prefill_8192": "e2e-llama31-8b-vllm-m8192", "decode_b32_step": "e2e-llama31-8b-vllm-m32"}


def configure(args) -> None:
    """``--batch``, ``--prompt`` and ``--prefill`` in this process (the controller's, and each engine's: spawn re-imports this
    module).  Off the defaults, the decode key names both, ``decode_b<B>_step`` with ``_p<P>`` before ``_step`` when the prompt
    isn't 64 tokens, and the prefill key its length, ``prefill_<P>``, so `panel_rows.py` (``decode_b32_step``,
    ``prefill_8192``) never reads a sweep point as the panel's row."""
    global BATCH, PROMPT, PREFILL, CAPTURE_SIZES
    BATCH, PROMPT = getattr(args, "batch", BATCH_DEFAULT), getattr(args, "prompt", PROMPT_DEFAULT)
    PREFILL = getattr(args, "prefill", PREFILL_DEFAULT)
    CAPTURE_SIZES = [s for s in CAPTURE_32 if s < BATCH] + [BATCH]
    for phase in ("prefill_", "decode_"):
        SHAPES.pop(next((k for k in SHAPES if k.startswith(phase)), None), None)
    SHAPES[prefill_key()] = f"e2e-llama31-8b-vllm-m{PREFILL}"
    SHAPES[decode_key() + "_step"] = f"e2e-llama31-8b-vllm-m{BATCH}" + (f"-p{PROMPT}" if PROMPT != PROMPT_DEFAULT else "")


def prefill_key() -> str:
    return f"prefill_{PREFILL}"


def decode_key() -> str:
    return f"decode_b{BATCH}" + (f"_p{PROMPT}" if PROMPT != PROMPT_DEFAULT else "")


def kv_tokens(llm) -> int | None:
    """The engine's KV cache in tokens (vLLM's block count times block size), read in-process; None where vLLM hides it."""
    try:
        cc = llm.llm_engine.vllm_config.cache_config
        return int(cc.num_gpu_blocks) * int(cc.block_size)
    except (AttributeError, TypeError):
        return None


def pass_capacity(tile: int, ns: list[int]) -> tuple[int, int]:
    """The tiles and screened rows one decode generate's pass holds at the decode batch and prompt length, each a power of two
    and at least the device run's defaults: the prompts' prefill in chunks of `PREFILL` tokens, then 1 + `STEPS` forwards of the
    batch, each forward's rows padded to `ROW_ALIGN` and its tiles to whole tile rows, with as many forwards again for
    vLLM's mixed prefill-and-decode steps.  Every linear (`ns`, their n) runs once per forward."""
    from verity_vllm.protocols.pouw import pearl_c as PPD
    cdiv = lambda a, b: -(-a // b)  # noqa: E731
    prompt_rows, forwards = BATCH * PROMPT, 2 * (cdiv(BATCH * PROMPT, PREFILL) + 1 + STEPS)
    tile_rows = cdiv(prompt_rows, tile) + forwards * cdiv(BATCH, tile)
    rows = prompt_rows + forwards * cdiv(BATCH, PPD.ROW_ALIGN) * PPD.ROW_ALIGN
    pow2 = lambda x: 1 << max(0, x - 1).bit_length()  # noqa: E731
    return (max(PPD.LEAF_CAPACITY, pow2(tile_rows * sum(n // tile for n in ns))), max(PPD.FLAG_ROWS, pow2(rows * len(ns))))


def engine(args, group: str):
    from vllm import LLM
    kw = {"model": args.model, "enforce_eager": True,
          "gpu_memory_utilization": args.mem_base if group in ("base", "base-graphs") else args.mem_arm,
          "max_num_seqs": BATCH, "max_model_len": max(PREFILL + 64, PROMPT + 1 + STEPS),
          "max_num_batched_tokens": getattr(args, "batch_tokens", None) or PREFILL, "enable_prefix_caching": False,
          "seed": 0}
    if args.revision:
        kw["revision"] = args.revision
    if group in ("base", "base-graphs"):
        kw["quantization"] = "fp8"
    if group == "base-graphs":
        kw["enforce_eager"] = False
        return LLM(**kw, compilation_config={"cudagraph_capture_sizes": CAPTURE_SIZES})
    if group == "arm" and getattr(args, "whole_step", False):
        kw["enforce_eager"] = False
        sizes = getattr(args, "capture", None) or [BATCH]
        return LLM(**kw, compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": sizes})
    try:
        return LLM(**kw, compilation_config={"mode": 0})
    except TypeError:
        return LLM(**kw)


def install(model, args, coins):
    """Every LinearBase layer's apply, found by type (`verity_vllm.protocols.linear.vllm_site`), through one `Stack` whose executor is a
    `PearlCDeviceRun` over the layers' own weights, attached through `engine.hooks`.  The epoch's salt is `coins`' salt coin,
    asked for once the weights are committed (`Coins`).  Returns the run, the weights by id and the stack; the stack starts
    inactive (vLLM's own apply), and `arm_of` gives each mode's arm."""
    from verity.core.protocols.pouw.audit import weights_root
    from verity_vllm.engine.hooks import Hooks
    from verity_vllm.protocols.linear import vllm_site
    from verity_vllm.protocols.linear.api import Stack
    from verity_vllm.protocols.pouw import pearl_c as PPD

    sites = vllm_site.discover(model)
    weights = {site.weight_id: layer.weight.data for site, layer in sites}
    scheme = getattr(args, "scheme", None) or SCHEME
    bad = PPD.refusals(scheme, {w: tuple(t.shape) for w, t in weights.items()})
    if bad:
        raise SystemExit(f"{scheme} refuses the model: " + "; ".join(bad))
    kernel = PPD.Kernel.load(args.kernel, args.cubin)
    salt_key = PPD.new_salt_key(scheme)
    roots = PPD.weight_roots(kernel, weights, scheme, salt_key=salt_key)
    tiles, rows = pass_capacity(kernel.run.TILE, [t.shape[0] for t in weights.values()])
    opening = coins.open("salt", weights_root(roots))
    run = PPD.PearlCDeviceRun(PPD.scheme_of(scheme), f"pouw:{scheme}", kernel, weights, roots, coins.commitment, opening, time_gemm=True,
                              schedule=getattr(args, "schedule", "serial"), graphs=getattr(args, "graphs", False),
                              whole_defer=getattr(args, "whole_defer", False), capacity=tiles, flag_rows=rows, salt_key=salt_key,
                              whole_defer_a=getattr(args, "whole_defer_a", False),
                              whole_fork_stats=getattr(args, "whole_fork_stats", False),
                              whole_tree_sm=getattr(args, "whole_tree_sm", False))
    if getattr(args, "statement", None):
        import importlib.util
        spec = importlib.util.spec_from_file_location("served_zk_statement", HERE.parent / "served_zk" / "statement.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        run.statement = mod.Statement(run, args.statement)
    stack = Stack([site for site, _ in sites], PPD.PearlCExecutor(run))
    bad = stack.refusals()
    if bad:
        raise SystemExit(f"{scheme} refuses the model: " + "; ".join(bad))
    vllm_site.attach(stack, sites, Hooks(tag=f"pouw:{scheme}"))
    return run, weights, stack


def whole_step(llm, run, stack):
    """``--whole-step``: Pearl-C's decode forwards replayed as vLLM's own graphs (`pouw_pearl_c_whole.WholeStep`), on the
    engine's in-process model runner."""
    from verity_vllm.engine import handle
    from verity_vllm.engine.hooks import Hooks
    from verity_vllm.protocols.pouw.pearl_c.whole import WholeStep
    return WholeStep(run, stack, handle.model_runner(llm), Hooks(tag=f"{run.entry}:whole-step"))


def arm_of(mode: str, run, args):
    """The arm engine's arm for `mode`: ``bf16`` is vLLM's own apply with the stack attached (so the wrapper's cost is inside
    the baseline), ``pearlc-nohash`` and ``pearlc`` the run without and with hashing, under the run's schedule, and
    ``pearlc-zk`` ``pearlc`` with its statement enabled (``--statement``)."""
    from verity_vllm.protocols.linear.api import Arm, PassSpec
    if getattr(run, "statement", None) is not None:
        run.statement.enabled = mode in ZK
    if mode not in PEARLC:
        return Arm(mode, "bf16")
    return Arm(mode, "bf16", run.entry, spec=PassSpec(hashing=mode != "pearlc-nohash", schedule=getattr(args, "schedule", "serial"),
                                                      run_id=settings.RESEARCH_RUN_ID))


def gate(run, weights: dict, whole: bool = False) -> list[dict]:
    """Served y against x·Wᵀ, and with hashing the same call's commitment twice and the negative control, at every
    distinct (n, k) and both m; with `whole` (``--whole-step``), `whole_check` too.  The checks that compare passes repeat
    one pass's call indices (`PearlCDeviceRun.begin`'s ``at``), since the epoch numbers calls across passes."""
    import torch
    out, seen = [], set()
    gen = torch.Generator(device=next(iter(weights.values())).device).manual_seed(20260930)
    for w, W in weights.items():
        if tuple(W.shape) in seen:
            continue
        seen.add(tuple(W.shape))
        for m in (PREFILL, BATCH):
            x = torch.randn(m, W.shape[1], device=W.device, generator=gen).to(torch.bfloat16)
            commits, served, at = [], None, run.epoch_calls()
            for omit in ((), (), ("hash_leaf",)) if run.hashing else ((),):
                run.begin(at=at)
                run.poison()
                run.omit = frozenset(omit)
                y = run.gemm(w, x, W, None)
                served = y if served is None else served
                commits.append({k: v for k, v in run.end().items() if k in ("transcript_root", "activation_roots")})
            run.omit = frozenset()
            ref = x.float() @ W.float().T
            err = float(((served.float() - ref).pow(2).sum() / ref.pow(2).sum()).sqrt())
            same = len(commits) == 1 or commits[0] == commits[1]
            control = len(commits) == 1 or commits[2]["transcript_root"] != commits[0]["transcript_root"]
            out.append({"check": "call", "weight": w, "n": W.shape[0], "k": W.shape[1], "m": m, "hashing": run.hashing, "rel_rms": err,
                        "commitment_repeats": same, "omitted_leaves_rejected": control, "ok": err < ERR and same and control})
    if run.hashing:
        out.append(screen_check(run, weights))
    if run.hashing and run.schedule.side:
        out.append(schedule_check(run, weights))
    if run.hashing and getattr(run, "graphs", False):
        out += graphs_check(run, weights)
    if whole:
        out.append(whole_check(run, weights))
    if whole and run.hashing and getattr(run, "statement", None) is not None:
        out.append(run.statement.check(weights))
    run.details["passes"].clear()
    run.gemm_ms()
    return out


def whole_check(run, weights: dict, m: int | None = None) -> dict:
    """With ``--whole-step``, a forward's calls (every weight once, `m` rows, default `BATCH`, in weight order) made eagerly
    twice in one pass, then captured whole in one CUDA graph after a warm-up outside it (as vLLM captures a decode forward)
    and replayed twice in each of two fresh passes on poisoned outputs, between `whole_replay` and `whole_replayed`, as vLLM's
    decode steps replay it: each pass must give the eager pass's A roots, tile root, screen flags and outputs, bit for bit,
    and each pass's second replay must upload the slots the first one staged (with hashing).  A retained capture keeps a
    stage for each weight and m₁₆, so the check runs at any m beside the sizes vLLM's graphs captured."""
    import torch
    m = BATCH if m is None else m
    dev = next(iter(weights.values())).device
    gen = torch.Generator(device=dev).manual_seed(20261002)
    xs = {w: torch.randn(m, W.shape[1], device=dev, generator=gen).to(torch.bfloat16) for w, W in weights.items()}
    rows = 2 * m * len(weights) if run.hashing else 0
    def forward():
        return [run.gemm(w, xs[w], W, None) for w, W in weights.items()]
    keep = ("transcript_root", "activation_roots", "units", "tiles")

    def ended(ys):
        return [y.clone() for y in ys], run.flag_buf[:rows].clone(), {k: v for k, v in run.end().items() if k in keep}

    run.graphs = False
    run.begin()
    at = run.cur.base
    run.poison()
    forward()
    want = ended(forward())
    run.graphs = True
    run.begin()
    graph, side = torch.cuda.CUDAGraph(), torch.cuda.Stream(device=dev)
    run.whole_capture_begin()
    try:
        forward()
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.graph(graph, stream=side):
            outs = forward()
    finally:
        calls = run.whole_capture_end()
    run.end()
    got, staged = [], []
    for _ in range(2):
        run.begin(at=at)
        run.poison()
        for _ in range(2):
            staged.append(run.whole_replay(calls))
            graph.replay()
            run.whole_replayed(calls)
        got.append(ended(outs))
    same = all(g[2] == want[2] and torch.equal(g[1], want[1]) and all(torch.equal(a, b) for a, b in zip(g[0], want[0], strict=True))
               for g in got)
    kept = whole_retained_check(run, forward, side) if run.hashing else None
    return {"check": "whole", "hashing": run.hashing, "calls": len(calls), "eager": want[2], "replays": [g[2] for g in got],
            "replay_equals_eager": same, "staged": staged, "retained_equals_eager": kept,
            "ok": (same and len(calls) == len(weights) and kept is not False
                   and staged == ([False, True] * 2 if run.hashing else [False] * 4))}


def whole_retained_check(run, forward, side) -> bool:
    """A retained pass of `forward` made eagerly, and one replayed from a whole capture of a retained pass's kind: the same
    records, call files (each call's rows and A tree) and commitment, byte for byte.  Each pass drops its retain directory
    before it ends, so neither writes the weights' files, which the two would share."""
    import shutil
    import tempfile

    import torch
    keep, dirs = ("transcript_root", "activation_roots", "units", "tiles"), [Path(tempfile.mkdtemp(prefix="whole-kept-")) for _ in "ab"]

    def ended() -> tuple:
        recs, files = list(run.cur.records), list(run.cur.files)
        run.cur.retain = None
        return recs, files, {k: v for k, v in run.end().items() if k in keep}

    try:
        run.graphs = False
        run.begin(retain=str(dirs[0]))
        at = run.cur.base
        forward()
        want = ended()
        run.graphs = True
        run.begin()
        graph = torch.cuda.CUDAGraph()
        run.whole_capture_begin(run.whole_key(True))
        try:
            forward()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(graph, stream=side):
                forward()
        finally:
            calls = run.whole_capture_end()
        run.end()
        run.begin(retain=str(dirs[1]), at=at)
        run.whole_replay(calls)
        graph.replay()
        run.whole_replayed(calls)
        got = ended()
        files = sorted(p.relative_to(dirs[0]) for p in (dirs[0] / "calls").iterdir())
        return (got == want and files == sorted(p.relative_to(dirs[1]) for p in (dirs[1] / "calls").iterdir())
                and all((dirs[0] / f).read_bytes() == (dirs[1] / f).read_bytes() for f in files))
    finally:
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)


def schedule_check(run, weights: dict) -> dict:
    """The run's schedule on the card: the same calls (m = 32 on the 64 x 64 path, m = 128 on the 128 x 128 one), each
    from a fresh pass on poisoned outputs, under the serial schedule and under the run's, must commit the same A roots and
    tile roots.  The run's schedule is its own again after."""
    import torch
    w, W = next(iter(weights.items()))
    gen = torch.Generator(device=W.device).manual_seed(20260932)
    xs = [torch.randn(m, W.shape[1], device=W.device, generator=gen).to(torch.bfloat16) for m in (32, 128, 32)]
    commits, mine, at = {}, run.schedule.name, run.epoch_calls()
    for name in ("serial", mine):
        run.begin(schedule=name, at=at)
        run.poison()
        for x in xs:
            run.gemm(w, x, W, None)
        commits[name] = {k: v for k, v in run.end().items() if k in ("transcript_root", "activation_roots", "tiles")}
    run.begin(schedule=mine)
    return {"check": "schedule", "schedule": mine, "priorities": run.details["schedule"], "serial": commits["serial"],
            mine: commits[mine], "ok": commits["serial"] == commits[mine]}


def graphs_check(run, weights: dict) -> list[dict]:
    """With ``--graphs``, at every distinct (n, k): one m = 32 call made eagerly, then twice as a graph replay, each from a
    fresh pass on poisoned outputs, must give the same A roots, tile root, screen flags and y, bit for bit.  And the screen
    of an m = 8,192 call read from its BF16 x must give the flags it gives from A's FP32 words."""
    import torch
    from verity_vllm.protocols.pouw import pearl_c as PPD
    out, seen = [], set()
    gen = torch.Generator(device=next(iter(weights.values())).device).manual_seed(20261001)
    for w, W in weights.items():
        if tuple(W.shape) in seen:
            continue
        seen.add(tuple(W.shape))
        x = torch.randn(BATCH, W.shape[1], device=W.device, generator=gen).to(torch.bfloat16)
        got, at = [], run.epoch_calls()
        for graphs in (False, True, True):
            run.graphs = graphs
            run.begin(at=at)
            run.poison()
            y = run.gemm(w, x, W, None).clone()
            flags = run.flag_buf[:BATCH].clone()
            c = run.end()
            got.append((c["transcript_root"], c["activation_roots"], flags, y))
        run.graphs = True
        same = all(g[0] == got[0][0] and g[1] == got[0][1] and torch.equal(g[2], got[0][2]) and torch.equal(g[3], got[0][3])
                   for g in got[1:])
        out.append({"check": "graphs", "weight": w, "n": W.shape[0], "k": W.shape[1], "m": BATCH, "eager_equals_replay": same,
                    "ok": same})
    w, W = next(iter(weights.items()))
    x = torch.randn(PREFILL, W.shape[1], device=W.device, generator=gen)
    x[::97] = 0
    x[1::89, 1::8] = 1e4
    x = x.to(torch.bfloat16)
    run.begin()
    run.gemm(w, x, W, None)
    sh = run.shapes[(PREFILL, W.shape[0], W.shape[1])]
    a, b = torch.zeros((PREFILL, 2), dtype=torch.uint8, device=W.device), torch.zeros((PREFILL, 2), dtype=torch.uint8, device=W.device)
    PPD.screen_into(a, x, sh.b["alpha_a"], sh.b["aux_a"], run.fused_screen)
    PPD.screen_into(b, sh.xa[:PREFILL], sh.b["alpha_a"], sh.b["aux_a"], run.fused_screen)
    run.end()
    same = bool(torch.equal(a, b))
    out.append({"check": "screen_bf16", "rows": PREFILL, "k": W.shape[1], "flagged": int(a.sum()), "flags_equal": same, "ok": same})
    return out


def screen_check(run, weights: dict) -> dict:
    """The records' screen on the card, at the rules' edges: one call of 16 rows (a zero row, rows with 16, 17 and 40 dead
    entries, one over the noise floor's bound, the rest ordinary) whose flags, from the run's screen, must be the torch
    screen's on the same rows and the kernel's own α and ρ."""
    import torch
    from verity_vllm.protocols.pouw import pearl_c as PPD
    w, W = next(iter(weights.items()))
    k = W.shape[1]
    gen = torch.Generator(device=W.device).manual_seed(20260931)
    x = torch.randn(16, k, device=W.device, generator=gen)
    x[1] = 0
    odd = [p for p in range(k) if p % 8]
    for r, count in ((2, 17), (3, 16), (4, 40)):
        x[r, odd[:count]] = 1e4
    x[5, 3] = 200.0
    run.begin()
    run.gemm(w, x.to(torch.bfloat16), W, None)
    sh = run.shapes[(16, W.shape[0], k)]
    got = run.flag_buf[:16].clone()
    ref = torch.zeros_like(got)
    PPD.screen_into(ref, sh.xa[:16], sh.b["alpha_a"], sh.b["aux_a"], fused=False)
    same = PPD.excluded(got[:, 0].bool(), got[:, 1].bool(), sh.xa[:16]) == PPD.excluded(ref[:, 0].bool(), ref[:, 1].bool(), sh.xa[:16])
    run.end()
    return {"check": "screen", "screen": run.details["host"]["screen"], "flags_equal": bool(torch.equal(got, ref)),
            "excluded_equal": same, "ok": same}


def gpu() -> str:
    """The GPU nvidia-smi should read: window.sh's E2E_GPU_UUID (the lease's), since NVML ignores CUDA_VISIBLE_DEVICES."""
    return settings.E2E_GPU_UUID


def clocks() -> dict:
    """The SM and memory clocks now, from nvidia-smi (the panel's per-rep clock record)."""
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=uuid,clocks.sm,clocks.mem,power.draw", "--format=csv,noheader,nounits",
                            "-i", gpu()], capture_output=True, text=True, timeout=10, check=False).stdout.strip().split(", ")
        return {"uuid": q[0], "sm_mhz": int(q[1]), "mem_mhz": int(q[2]), "watts": float(q[3])}
    except Exception as ex:  # noqa: BLE001 - a missing reading is recorded, not fatal
        return {"error": f"{type(ex).__name__}: {ex}"}


# ---- one resident engine --------------------------------------------------------------------------------------------

def serve(conn, args, group: str, coins: dict) -> None:
    """One engine, resident in its own process: ``base`` (stock FP8) or ``arm`` (BF16 with the Pearl-C wrapper), answering
    the controller's commands one at a time.  `coins` is the epoch's published coin commitment; the arm's install asks the
    controller, which holds the challenger, for the salt coin (`PipeCoins`)."""
    try:
        configure(args)
        import torch
        from vllm import SamplingParams
        from vllm_bench import PROMPTS, find_model

        llm = engine(args, group)
        run = stack = whole = None
        ready: dict = {"group": group, "kv_tokens": kv_tokens(llm)}
        if group == "arm":
            run, weights, stack = install(find_model(llm), args, PipeCoins(conn, coins))
            ws = getattr(args, "whole_step", False)
            ready["gate"] = gate(run, weights, ws)
            run.hashing = False
            ready["gate"] += gate(run, weights, ws)
            run.hashing = True
            ready.update(committer=run.details["committer"], excluded_b=sum(run.details["excluded_b"].values()))
            if ws:
                whole = whole_step(llm, run, stack)
        conn.send(ready)
        one = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        many = SamplingParams(temperature=0.0, max_tokens=1 + STEPS, ignore_eos=True)
        tok = llm.get_tokenizer()
        # at most a decode batch of them, so a small batch's chat (its prompts one after another) stays within one pass
        chat = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                for p in PROMPTS[:BATCH]]

        def timed(prompts, sp, retain=None, controls=()):
            """One timed generate; under a Pearl-C arm it is one pass (`Stack.pass_`), whose end is inside the time."""
            import dataclasses
            pearl = stack is not None and stack.executing
            spec = dataclasses.replace(stack.active.spec, retain=retain, controls=tuple(controls)) if pearl else None
            with stack.pass_(spec) if pearl else contextlib.nullcontext():
                if pearl:
                    run.gemm_ms()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                outs = llm.generate(prompts, sp, use_tqdm=False)
            commit = stack.passes[-1] if pearl else {}
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            return dt, (run.gemm_ms()[0] / 1e3 if pearl else 0.0), commit, outs

        while True:
            cmd = conn.recv()
            op, mode = cmd["op"], cmd.get("mode")
            if op == "exit":
                conn.send({"ok": True})
                return
            if mode is not None:
                if GROUP[mode] != group:
                    raise ValueError(f"mode {mode} is not the {group} engine's")
                if stack is not None:
                    stack.activate(arm_of(mode, run, args))
                if whole is not None:
                    whole.prepare()
                    whole.take()
            if op == "chat":
                *_, outs = timed(chat, SamplingParams(temperature=0.0, max_tokens=48))
                zk = run.statement.take() if run is not None and run.statement is not None and run.statement.enabled else None
                conn.send({"tokens": [list(o.outputs[0].token_ids) for o in outs], "texts": [o.outputs[0].text for o in outs[:4]],
                           **({"statement": zk} if zk else {}), **({"whole_step": whole.take()} if whole is not None else {})})
            elif op == "prefill":
                dt, gemm, commit, _ = timed([{"prompt_token_ids": cmd["prompt"]}], one)
                conn.send({"s": dt, "gemm_s": gemm, "commitment": commit})
            elif op == "decode":
                p = [{"prompt_token_ids": x} for x in cmd["prompts"]]
                t1, g1, _, _ = timed(p, one)
                tn, gn, cn, _ = timed(p, many)
                zk = run.statement.take() if run is not None and run.statement is not None and run.statement.enabled else None
                conn.send({"step_s": (tn - t1) / STEPS, "gemm_step_s": (gn - g1) / STEPS, "t1_s": t1, "tn_s": tn, "commitment": cn,
                           **({"whole_step": whole.take()} if whole is not None else {}), **({"statement": zk} if zk else {})})
            elif op == "ppl":
                conn.send(prompt_logprobs(llm, SamplingParams, timed, cmd["windows"]))
            elif op == "verify":
                from verity_vllm.protocols.linear.api import Control
                stack.activate(arm_of("pearlc", run, args))
                out = {}
                if whole is not None:
                    whole.take()
                for name, prompts, sp in (("prefill", [cmd["prefill"]], one), ("decode", cmd["decode"], many)):
                    want = cmd["want"][name]
                    run.repeat_at = want.get("base")     # the timed pass's call indices, so its commitment can repeat
                    _, _, got, _ = timed([{"prompt_token_ids": x} for x in prompts], sp, retain=str(Path(args.retain) / name))
                    same = all(got.get(k) == want.get(k) for k in ("transcript_root", "activation_roots", "units", "tiles"))
                    out[name] = {"manifest": got.get("manifest"), "commitment": got, "same_commitment_as_timed": same,
                                 **({"whole_step": whole.take()} if whole is not None else {})}
                    if run.graphs:
                        p, ref = [{"prompt_token_ids": x} for x in prompts], want
                        if whole is not None:    # vLLM's eager decode isn't its graph's: both sides on vLLM's eager path
                            whole.enabled = False
                        try:
                            if whole is not None:
                                run.repeat_at = want.get("base")
                                _, _, ref, _ = timed(p, sp)
                            run.graphs = False
                            run.repeat_at = want.get("base")
                            _, _, eager, _ = timed(p, sp)
                        finally:
                            run.graphs = True
                            if whole is not None:
                                whole.enabled = True
                        out[name]["same_commitment_eager"] = all(eager.get(k) == ref.get(k) for k in
                                                                 ("transcript_root", "activation_roots", "units", "tiles"))
                for label, omits in CONTROLS.items():
                    _, _, got, _ = timed([{"prompt_token_ids": cmd["decode"][0]}], one, retain=str(Path(args.retain) / label),
                                         controls=(Control("omit", omits),))
                    out[label] = {"manifest": got.get("manifest"), "commitment": got, "omitted": sorted(omits)}
                conn.send(out)
            elif op == "served":
                stack.activate(arm_of("pearlc", run, args))
                sp = SamplingParams(temperature=0.0, max_tokens=SERVED_TOKENS, ignore_eos=True)
                out = {}
                if whole is not None:
                    whole.take()
                for name, i in SERVED.items():
                    _, _, got, outs = timed([chat[i]], sp, retain=str(Path(args.retain) / name))
                    tokens = list(outs[0].outputs[0].token_ids)
                    length = {"ignore_eos": sp.ignore_eos, "max_tokens": [sp.max_tokens], "served": [len(tokens)]}
                    (Path(args.retain) / name / SERVED_RECORD).write_text(json.dumps(length, sort_keys=True) + "\n")
                    out[name] = {"prompt": PROMPTS[i], "tokens": tokens, "length": length,
                                 "text": outs[0].outputs[0].text, "manifest": got.get("manifest"), "commitment": got,
                                 **({"whole_step": whole.take()} if whole is not None else {})}
                conn.send(out)
            else:
                raise ValueError(f"unknown command {op!r}")
    except Exception as ex:  # noqa: BLE001 - the controller reports it
        conn.send({"error": f"{group}: {type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-4000:]})


def prompt_logprobs(llm, sampling_params, timed, windows: list[list[int]]) -> dict:
    """Each window's prompt log-probs (vLLM's ``prompt_logprobs=1``), a pass per `PPL_WINDOWS_A_PASS` windows: the summed
    negative log-likelihood of every token after the window's first, their count, and each position's top-1 token."""
    sp = sampling_params(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    nll, count, top1 = 0.0, 0, []
    for i in range(0, len(windows), PPL_WINDOWS_A_PASS):
        batch = windows[i:i + PPL_WINDOWS_A_PASS]
        *_, outs = timed([{"prompt_token_ids": w} for w in batch], sp)
        for w, o in zip(batch, outs, strict=True):
            for pos in range(1, len(w)):
                d = o.prompt_logprobs[pos]
                nll -= d[w[pos]].logprob
                top1.append(next(t for t, lp in d.items() if lp.rank == 1))
                count += 1
    return {"nll": nll, "tokens": count, "top1": top1}


class Coins:
    """The challenger's side of one epoch in the controller: its coins (`coins.Challenger`), committed before any engine
    starts, drawn from `entropy` (by default the Lean verifier of record's, `pouw.verifier.coins`, `IO.getRandomBytes`),
    which a claim instance must name (`challenger.served_claim`). A coin opens once; an engine started again in the same
    epoch (the ABAB blocks) asking for it after the same record gets the one opening again, and after any other record a
    refusal, so nothing redraws."""

    def __init__(self, epoch: int, entropy=None):
        from verity.core.protocols.pouw import verifier
        from verity.core.protocols.pouw.coins import Challenger
        from verity.core.protocols.sampled_proofs.challenger import served_claim
        entropy = verifier.coins if entropy is None else entropy
        served_claim(entropy)
        self.challenger, self.opened = Challenger(epoch, entropy), {}
        self.commitment = self.challenger.commitment

    def open(self, label: str, after: bytes):
        from verity.core.protocols.pouw.protocol import DomainError
        if label in self.opened:
            if self.opened[label].after != after:
                raise DomainError(f"epoch {self.commitment.epoch}'s {label!r} coin is already open, after another record")
            return self.opened[label]
        self.opened[label] = self.challenger.open(label, after)
        return self.opened[label]


class PipeCoins:
    """The engine's side: the epoch's commitment, and each coin asked of the controller's `Coins` over the pipe."""

    def __init__(self, conn, commitment: dict):
        from verity.core.protocols.pouw.coins import CoinCommitment
        self.conn, self.commitment = conn, CoinCommitment.from_json(commitment)

    def open(self, label: str, after: bytes):
        from verity.core.protocols.pouw.coins import CoinOpening
        from verity.core.protocols.pouw.protocol import DomainError
        self.conn.send({"coin": label, "after": after.hex()})
        msg = self.conn.recv()
        if "refused" in msg:
            raise DomainError(msg["refused"])
        return CoinOpening.from_json(msg["opening"])


class Worker:
    started = 0
    live: ClassVar[list[Worker]] = []

    def __init__(self, ctx, args, group: str, coins: Coins):
        Worker.started += 1
        self.group, self.serial, (self.conn, child) = group, Worker.started, ctx.Pipe()
        self.coins = coins
        self.proc = ctx.Process(target=serve, args=(child, args, group, coins.commitment.to_json()))
        Worker.live.append(self)
        self.proc.start()
        try:
            self.ready = self.recv(1200)
        except RuntimeError:
            self.proc.join(30)
            if self.proc.is_alive():
                self.proc.terminate()
            raise
        need, have = BATCH * (PROMPT + 1 + STEPS), self.ready.get("kv_tokens")
        if have is not None and have < need:
            raise SystemExit(f"{group}: its KV cache holds {have} tokens and the decode batch needs {need} ({BATCH} x "
                             f"({PROMPT} + 1 + {STEPS})): vLLM would queue or preempt requests inside the timed step; raise --mem-*")

    def recv(self, timeout: float = 1200) -> dict:
        """The engine's answer; a coin it asks for on the way is answered from the epoch's `Coins`."""
        from verity.core.protocols.pouw.protocol import DomainError
        while True:
            if not self.conn.poll(timeout):
                raise RuntimeError(f"{self.group}: no answer in {timeout:.0f} s")
            try:
                msg = self.conn.recv()
            except EOFError:
                raise RuntimeError(f"{self.group}: the engine's process exited (code {self.proc.exitcode})") from None
            if "coin" not in msg:
                break
            try:
                self.conn.send({"opening": self.coins.open(msg["coin"], bytes.fromhex(msg["after"])).to_json()})
            except DomainError as e:
                self.conn.send({"refused": str(e)})
        if "error" in msg:
            raise RuntimeError(msg["error"] + "\n" + msg.get("trace", ""))
        return msg

    def ask(self, **cmd) -> dict:
        self.conn.send(cmd)
        return self.recv()

    def close(self) -> None:
        try:
            self.ask(op="exit")
        except Exception as ex:  # noqa: BLE001 - it may already be gone; the join below decides
            print(f"e2e: {self.group} did not answer exit: {ex}", file=sys.stderr)
        self.proc.join(120)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(30)
        if self in Worker.live:
            Worker.live.remove(self)


# ---- the controller -------------------------------------------------------------------------------------------------

def prompts(reps: int) -> dict:
    """Every mode's prompts, the same in both engines: the prefill warm-up and one prompt per rep, and the decode batch."""
    rng = random.Random(8192)
    def ids(n):
        return [rng.randrange(100, 20000) for _ in range(n)]
    return {"warm": ids(PREFILL), "prefill": [ids(PREFILL) for _ in range(reps)], "decode": [ids(PROMPT) for _ in range(BATCH)]}


def schedule(reps: int, interleave: str, graphed: bool = False, zk: bool = False) -> list[list[str]]:
    """The timed modes in order, as one segment per engine lifetime: with both engines resident one segment, the reps
    alternating by `CYCLE` (with `graphed`, each ``fp8`` rep followed by an ``fp8-graphs`` one; with `zk`, each ``pearlc``
    rep by a ``pearlc-zk`` one); in ``blocks``, `BLOCKS` segments alternating base and arm, each with its share of the reps."""
    after = {"fp8": ("fp8-graphs",) if graphed else (), "pearlc": ZK if zk else ()}
    cycle = [x for m in CYCLE for x in (m, *after.get(m, ()))]
    if interleave == "rep":
        return [cycle * reps]
    share = [reps // 2 + reps % 2, reps // 2]
    arm = [m for m in cycle if GROUP[m] == "arm"]
    return [["fp8"] * share[i // 2] if i % 2 == 0 else arm * share[i // 2] for i in range(BLOCKS)]


def timing(records: list[dict]) -> dict:
    """Each mode's medians over its reps, and its reps, from the controller's records."""
    out, p, d = {}, prefill_key(), decode_key()
    for mode in MODES + GRAPHED + ZK:
        pre = [r for r in records if r["mode"] == mode and r["phase"] == "prefill"]
        dec = [r for r in records if r["mode"] == mode and r["phase"] == "decode"]
        if not pre or not dec:
            continue
        out[mode] = {"timing": {f"{p}_s": statistics.median(r["s"] for r in pre),
                                f"{p}_gemm_s": statistics.median(r["gemm_s"] for r in pre),
                                f"{d}_step_s": statistics.median(r["step_s"] for r in dec),
                                f"{d}_step_gemm_s": statistics.median(r["gemm_step_s"] for r in dec)},
                     "reps": {p: pre, d: dec}}
    return out


def measure(workers: dict, segment: list[str], p: dict, records: list[dict], warm: set) -> None:
    """The segment's reps in order, prefill then decode; each mode warmed once per engine lifetime before its first rep."""
    for phase in ("prefill", "decode"):
        for mode in segment:
            w = workers[GROUP[mode]]
            if (w.serial, mode, phase) not in warm:
                w.ask(op=phase, mode=mode, prompt=p["warm"], prompts=p["decode"])
                warm.add((w.serial, mode, phase))
            i = sum(1 for r in records if r["mode"] == mode and r["phase"] == phase)
            got = w.ask(op=phase, mode=mode, prompt=p["prefill"][i % len(p["prefill"])], prompts=p["decode"])
            records.append({"mode": mode, "phase": phase, "rep": i, **got, "clock": clocks()})


def split(results: dict) -> dict:
    """Per phase: ``pearlc`` against ``fp8`` (the headline) and ``bf16``, and its time split into the GEMM (the no-hash mode's
    kernel), hashing and records (``pearlc`` less the no-hash mode) and everything else."""
    t = {m: r["timing"] for m, r in results.items() if "timing" in r}
    out = {}
    for key in (prefill_key(), decode_key() + "_step"):
        if not all(m in t for m in MODES):
            continue
        full, nohash, gemm = t["pearlc"][f"{key}_s"], t["pearlc-nohash"][f"{key}_s"], t["pearlc-nohash"][f"{key}_gemm_s"]
        row = {"bf16_s": t["bf16"][f"{key}_s"], "fp8_s": t["fp8"][f"{key}_s"], "pearlc_s": full, "nohash_s": nohash, "gemm_s": gemm,
               "hashing_s": full - nohash, "else_s": nohash - gemm}
        if "fp8-graphs" in t:
            row["fp8_graphs_s"] = t["fp8-graphs"][f"{key}_s"]
            row["slowdown_vs_fp8_graphs"] = full / row["fp8_graphs_s"]
        if "pearlc-zk" in t:
            row["pearlc_zk_s"] = t["pearlc-zk"][f"{key}_s"]
            row["statement_s"], row["zk_over_pearlc"] = row["pearlc_zk_s"] - full, row["pearlc_zk_s"] / full
        for base in ("fp8", "bf16"):
            b = row[f"{base}_s"]
            row.update({f"slowdown_vs_{base}": full / b, f"gemm_x_{base}": gemm / b, f"hashing_x_{base}": (full - nohash) / b,
                        f"else_x_{base}": (nohash - gemm) / b})
        out[key] = {"shape": SHAPES[key], **row}
    return out


def state_entropy(path: str):
    """The challenger's entropy held in the file `path` (readable only by its owner; when absent, created from the Lean
    verifier of record's coins, `pouw.verifier.coins`, before any engine starts): the same coins for the controller here
    and for the challenger process that later draws the window's units (`served_commit.py challenger --state`), so named
    by the Lean's claim instance (`verifier.CLAIM`)."""
    from verity.core.protocols.pouw import verifier
    from verity.core.protocols.pouw.coins import COIN_BYTES, HM96, LABELS
    p = Path(path)
    if not p.exists():
        entropy = verifier.coins(len(LABELS) * (COIN_BYTES + HM96.salt_bytes))
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(entropy)
    data, at = p.read_bytes(), [0]

    def take(n: int) -> bytes:
        at[0] += n
        return data[at[0] - n:at[0]]
    take.claim = verifier.CLAIM
    return take


def epoch_coins(args, epoch: int) -> Coins:
    """The run's epoch coins, committed before any engine starts; with ``--retain`` the commitment is published beside the
    passes (``coins.json``), where `verify_run.py --coins` reads it.  ``POUW_COIN_STATE`` names a file holding the
    challenger's entropy (`state_entropy`)."""
    state = settings.POUW_COIN_STATE
    coins = Coins(epoch, state_entropy(state)) if state else Coins(epoch)
    if getattr(args, "retain", None):
        Path(args.retain).mkdir(parents=True, exist_ok=True)
        (Path(args.retain) / "coins.json").write_text(json.dumps(coins.commitment.to_json()) + "\n")
    return coins


def memory_mib() -> int | None:
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", gpu()],
                           capture_output=True, text=True, timeout=10, check=False)
        return int(q.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def smoke(args) -> None:
    """Untimed: both engines resident on the GPU, and each one's first sampling calls.  The stock FP8 engine starts and
    samples; then the Pearl-C engine starts beside it (its install and gates) and samples as stock BF16 and as ``pearlc``.
    Writes each step's time and the GPU's memory use after it, the gates, and the greedy tokens' agreement with BF16.  With
    ``--retain``, the arm then serves `SERVED`'s requests, each a retained ``pearlc`` pass under its own name."""
    ctx, coins = mp.get_context("spawn"), epoch_coins(args, 0)
    out: dict = {"schema": "verity-pouw/pearl-c-vllm-smoke/v1", "coins": coins.commitment.to_json(), "scheme": getattr(args, "scheme", SCHEME),
                 "schedule": getattr(args, "schedule", "serial"),
                 "whole_step": getattr(args, "whole_step", False), "whole_defer": getattr(args, "whole_defer", False),
                 "whole_defer_a": getattr(args, "whole_defer_a", False),
                 "whole_fork_stats": getattr(args, "whole_fork_stats", False),
                 "whole_tree_sm": getattr(args, "whole_tree_sm", False),
                 "capture": getattr(args, "capture", None), "statement": getattr(args, "statement", None), "run": settings.RESEARCH_RUN_ID, "steps": [], "resident": False}

    def step(name, f):
        t0 = time.perf_counter()
        got = f()
        out["steps"].append({"step": name, "s": round(time.perf_counter() - t0, 2), "memory_mib": memory_mib()})
        return got

    try:
        base = step("start the fp8 engine", lambda: Worker(ctx, args, "base", coins))
        fp8 = step("fp8: first sampling calls", lambda: base.ask(op="chat", mode="fp8"))
        arm = step("start the pearl-c engine beside it (install, gates)", lambda: Worker(ctx, args, "arm", coins))
        out["resident"], out["gate"] = True, arm.ready["gate"]
        out["gate_ok"] = all(g["ok"] for g in arm.ready["gate"])
        bf16 = step("bf16: first sampling calls", lambda: arm.ask(op="chat", mode="bf16"))
        pearlc = step("pearlc: first sampling calls", lambda: arm.ask(op="chat", mode="pearlc"))
        out["texts"] = {"fp8": fp8["texts"][0], "bf16": bf16["texts"][0], "pearlc": pearlc["texts"][0]}
        out["token_agreement_vs_bf16"] = {"fp8": agree(fp8["tokens"], bf16["tokens"]), "pearlc": agree(pearlc["tokens"], bf16["tokens"])}
        if getattr(args, "statement", None):
            zk = step("pearlc-zk: first sampling calls", lambda: arm.ask(op="chat", mode="pearlc-zk"))
            out["texts"]["pearlc-zk"] = zk["texts"][0]
            out["token_agreement_vs_bf16"]["pearlc-zk"] = agree(zk["tokens"], bf16["tokens"])
            out["statement_forwards"] = zk.get("statement")
        if getattr(args, "retain", None):
            out["served"] = step("served: retained requests", lambda: arm.ask(op="served"))
    except (RuntimeError, SystemExit) as ex:
        out["error"] = str(ex)[:4000]
    finally:
        for w in list(Worker.live):
            w.close()
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "gate"}), flush=True)
    import run_result
    ok = "error" not in out and out.get("gate_ok", False)
    run_result.write(args.out, ok, "pearl-c vLLM e2e smoke: both engines resident, the arm gated, each one's first sampling calls",
                     json.dumps({k: v for k, v in out.items() if k != "gate"}), [(s["step"], s["s"], "s") for s in out["steps"]])


def wikitext_windows(model: str, limit: int | None) -> list[list[int]]:
    """WikiText-2's test split (`WIKITEXT`, from the Hugging Face cache) joined with blank lines, tokenized, and cut into
    non-overlapping windows of BOS and `WINDOW` − 1 tokens; the last partial window is dropped, and `limit` keeps the first.
    BOS is the tokenizer's, else the model config's `bos_token_id` (Qwen3's tokenizer names none; its config 151643)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoTokenizer
    repo, rev, name = WIKITEXT
    text = "\n\n".join(pq.read_table(hf_hub_download(repo, name, repo_type="dataset", revision=rev)).column("text").to_pylist())
    tok = AutoTokenizer.from_pretrained(model)
    bos = tok.bos_token_id if tok.bos_token_id is not None else AutoConfig.from_pretrained(model).bos_token_id
    if bos is None:
        raise ValueError(f"{model}: neither its tokenizer nor its config names a BOS token")
    ids = tok(text, add_special_tokens=False)["input_ids"]
    body = WINDOW - 1
    out = [[bos, *ids[i:i + body]] for i in range(0, len(ids) - body + 1, body)]
    return out[:limit] if limit else out


def quality_summary(parts: dict) -> dict:
    """Each mode's perplexity, and its agreement with bf16: top-1 teacher-forced over WikiText-2, and greedy on the chat prompts."""
    import math
    ref = parts.get("bf16", {})
    out = {}
    for mode, r in parts.items():
        row = {"scheme": r.get("scheme"), "perplexity": math.exp(r["nll"] / r["tokens"]), "tokens": r["tokens"]}
        if ref:
            row["top1_agreement_vs_bf16"] = sum(a == b for a, b in zip(r["top1"], ref["top1"], strict=True)) / len(ref["top1"])
            row["greedy_token_agreement_vs_bf16"] = agree(r["chat"]["tokens"], ref["chat"]["tokens"])
        out[mode] = row
    return out


def perplexity(args) -> int:
    """The untimed quality run, in groups (one engine each) whose results are kept beside --out as they finish, so a
    restart (a preempted or a continued fill chunk) skips them: stock fp8; bf16 and Pearl-C v1 in one arm engine (bf16 with the
    wrapper off); Pearl-C v2 (`QUALITY`).  Each mode's WikiText-2 prompt log-probs and its greedy tokens on the chat prompts;
    Pearl-C's arm is gated first.  A chunk runs at least one group, and starts no other once --chunk-min has passed.  Returns
    0 when every group is done, 99 when groups remain."""
    import copy
    ctx, t0 = mp.get_context("spawn"), time.perf_counter()
    d = args.out.parent
    windows = wikitext_windows(args.model, args.ppl_windows)
    groups = [("fp8", "base", None, ("fp8",)), ("v1", "arm", QUALITY["pearlc-v1"], ("bf16", "pearlc-v1")),
              ("v2", "arm", QUALITY["pearlc-v2"], ("pearlc-v2",))]
    ran = False
    for epoch, (name, group, scheme, modes) in enumerate(groups):
        path = d / f"quality-{name}.json"
        if path.exists():
            continue
        if ran and time.perf_counter() - t0 > 60 * args.chunk_min:
            return 99
        ran = True
        a = copy.copy(args)
        a.scheme, a.batch_tokens = scheme, PPL_BATCH
        w = Worker(ctx, a, group, Coins(epoch))
        try:
            res = {"gate": w.ready.get("gate"), "committer": w.ready.get("committer"), "modes": {}}
            if group == "arm" and not all(g["ok"] for g in w.ready["gate"]):
                raise SystemExit(f"{scheme}: the arm's gate failed")
            for mode in modes:
                engine_mode = "pearlc" if mode.startswith("pearlc") else mode
                got = w.ask(op="ppl", mode=engine_mode, windows=windows)
                got["chat"] = w.ask(op="chat", mode=engine_mode)
                res["modes"][mode] = {"scheme": scheme if engine_mode == "pearlc" else None, **got}
        finally:
            w.close()
        path.write_text(json.dumps(res))
    parts = {m: r for name, *_ in groups for m, r in json.loads((d / f"quality-{name}.json").read_text())["modes"].items()}
    summary = quality_summary(parts)
    gates = {name: all(g["ok"] for g in (json.loads((d / f"quality-{name}.json").read_text())["gate"] or [{"ok": True}]))
             for name, *_ in groups}
    doc = {"schema": "verity-pouw/pearl-c-vllm-quality/v1", "run": settings.RESEARCH_RUN_ID or settings.FILL_JOB,
           "model": args.model, "dataset": {"repo": WIKITEXT[0], "revision": WIKITEXT[1], "file": WIKITEXT[2]},
           "windows": len(windows), "window_tokens": WINDOW, "gates": gates, "summary": summary}
    args.out.write_text(json.dumps(doc, indent=1))
    print(json.dumps({"summary": summary, "gates": gates}), flush=True)
    return 0


def agree(a, b) -> float:
    same = sum(x == y for s, u in zip(a, b) for x, y in zip(s, u))
    return same / max(1, sum(min(len(s), len(u)) for s, u in zip(a, b)))


def main() -> None:
    from verity.core.protocols.pouw.serving import SCHEDULES
    from verity_vllm.protocols.pouw import pearl_c as PPD
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--kernel", default=str(HERE.parent / "pearl_c_sm120"), help="GPU 1's kernel directory (run.py, fixture.py)")
    ap.add_argument("--cubin", help="build.sh's pearl_c_sm120.cubin (default: the kernel directory's)")
    ap.add_argument("--out", type=Path, default=Path(settings.RESEARCH_RUN_DIR or ".") / "e2e.json")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=BATCH_DEFAULT, help="the decode batch: its prompts, and each decode call's m")
    ap.add_argument("--prompt", type=int, default=PROMPT_DEFAULT, help="each decode prompt's tokens (the context decode starts at)")
    ap.add_argument("--prefill", type=int, default=PREFILL_DEFAULT, help="the prefill prompt's tokens, and the engines' batch budget")
    ap.add_argument("--mem-base", type=float, default=0.2, help="the FP8 engine's gpu_memory_utilization (weights about 8 GB)")
    ap.add_argument("--mem-arm", type=float, default=0.3, help="the BF16 engine's; Pearl-C's buffers (about 13 GB) sit outside it")
    ap.add_argument("--interleave", choices=["rep", "blocks"], default="rep", help="blocks: skip co-residency, go straight to ABAB")
    ap.add_argument("--retain", help="the verify passes' directory")
    ap.add_argument("--scheme", default=SCHEME, choices=sorted(PPD.SCHEMES), help="the Pearl-C scheme, and so its hashing format")
    ap.add_argument("--schedule", choices=sorted(SCHEDULES), default="serial",
                    help="deferred: each call's tile hashing on a side lane at the least priority and the rest at the greatest, "
                         "as measured (the decode plan's step 2); deferred-caller: the rest on vLLM's stream at its own")
    ap.add_argument("--fp8-graphs", action="store_true",
                    help="also time stock FP8 as vLLM serves it (compiled, CUDA graphs), a third engine, beside the eager one")
    ap.add_argument("--graphs", action="store_true", help="decode's calls (m <= GRAPH_ROWS) replay their steps as one CUDA graph each")
    ap.add_argument("--whole-step", action="store_true",
                    help="(implies --graphs) the arm engine runs vLLM's FULL decode graphs, Pearl-C's captured inside them "
                         "(pouw_pearl_c_whole); retained passes run eagerly")
    ap.add_argument("--capture", type=lambda s: sorted({int(x) for x in s.split(",")}), metavar="SIZES",
                    help="(with --whole-step) the arm engine's CUDA-graph batch sizes, comma-separated (default the decode batch)")
    ap.add_argument("--whole-defer", action="store_true",
                    help="(with --whole-step) a whole capture runs each call's tile hashing beside the next call's A tree")
    ap.add_argument("--whole-defer-a", action="store_true",
                    help="(with --whole-defer) and each call's A tree beside its own forming and GEMM, on a kernel whose "
                         "pipeline splits A's commitment")
    ap.add_argument("--whole-tree-sm", action="store_true",
                    help="(with --whole-defer-a) the GEMM beside each call's A tree launching one persistent CTA fewer, "
                         "leaving the tree's CTA its SM")
    ap.add_argument("--whole-fork-stats", action="store_true",
                    help="(with --whole-defer) and each call's A scales beside its A commitment")
    ap.add_argument("--statement", metavar="CUBIN",
                    help="(with --whole-step) hm96_rows.cu's cubin: adds pearlc-zk, each decode forward's rows committed in the "
                         "ZK statement's format too, once after the forward (served_zk/statement.py)")
    ap.add_argument("--smoke", action="store_true", help="untimed: both engines resident and each one's first sampling calls")
    ap.add_argument("--perplexity", action="store_true", help="untimed: WikiText-2 perplexity and agreement for bf16, fp8, v1, v2")
    ap.add_argument("--ppl-windows", type=int, help="the quality run's first N windows (default all)")
    ap.add_argument("--chunk-min", type=float, default=7.0, help="the quality run starts no group after this many minutes")
    args = ap.parse_args()
    if args.batch < 1 or args.prompt < 1 or args.prefill < 1:
        ap.error("--batch, --prompt and --prefill are positive")
    for size in (args.batch, *(args.capture or ())) if args.whole_step else ():
        if not 0 < -(-size // PPD.ROW_ALIGN) * PPD.ROW_ALIGN <= PPD.GRAPH_ROWS:
            ap.error(f"--whole-step captures the decode batch whole: a batch of {size}'s calls, filler rows to a multiple of "
                     f"{PPD.ROW_ALIGN} included, are not 1 to {PPD.GRAPH_ROWS} rows (pouw_pearl_c_whole)")
    if args.capture and not args.whole_step:
        ap.error("--capture sets the whole-step arm's graph sizes: it needs --whole-step")
    configure(args)
    args.graphs = args.graphs or args.whole_step
    if args.whole_defer and not args.whole_step:
        ap.error("--whole-defer defers within a whole capture: it needs --whole-step")
    if args.whole_defer_a and not args.whole_defer:
        ap.error("--whole-defer-a defers A's trees beside the deferred tile hashing: it needs --whole-defer")
    if args.whole_tree_sm and not args.whole_defer_a:
        ap.error("--whole-tree-sm leaves a deferred A tree its SM: it needs --whole-defer-a")
    if args.whole_fork_stats and not args.whole_defer:
        ap.error("--whole-fork-stats forks A's scales within a deferred whole capture: it needs --whole-defer")
    if args.statement and not args.whole_step:
        ap.error("--statement commits the forwards vLLM replays whole: it needs --whole-step")
    if args.smoke:
        smoke(args)
        return
    if args.perplexity:
        sys.exit(perplexity(args))
    ctx, coins, p = mp.get_context("spawn"), epoch_coins(args, 0), prompts(args.reps)
    records: list[dict] = []
    out: dict = {"schema": "verity-pouw/pearl-c-vllm-e2e/v2", "scheme": args.scheme, "schedule": args.schedule, "graphs": args.graphs, "fp8_graphs": args.fp8_graphs,
                 "decode": {"batch": BATCH, "prompt": PROMPT, "steps": STEPS, "key": decode_key()},
                 "whole_step": args.whole_step, "whole_defer": args.whole_defer, "whole_defer_a": args.whole_defer_a,
                 "whole_fork_stats": args.whole_fork_stats, "whole_tree_sm": args.whole_tree_sm, "capture": args.capture,
                 "statement": args.statement, "run": settings.RESEARCH_RUN_ID,
                 "coins": coins.commitment.to_json(), "interleave": args.interleave, "resident_failure": None}
    ready, tokens, verify = {}, {}, {}

    def arm_steps(arm: Worker, last: bool) -> None:
        ready["arm"] = arm.ready
        if not all(g["ok"] for g in arm.ready["gate"]):
            raise SystemExit("the arm's gate failed: " + json.dumps([g for g in arm.ready["gate"] if not g["ok"]]))
        for mode in ("bf16", "pearlc-nohash", "pearlc", *(ZK if getattr(args, "statement", None) else ())):
            if mode not in tokens:
                tokens[mode] = arm.ask(op="chat", mode=mode)
        if last and args.retain:
            want = {ph: next(r["commitment"] for r in records if r["mode"] == "pearlc" and r["phase"] == ph and r["rep"] == 0)
                    for ph in ("prefill", "decode")}
            verify.update(arm.ask(op="verify", prefill=p["prefill"][0], decode=p["decode"], want=want))

    try:
        run_workers(ctx, args, coins, p, records, out, arm_steps)
    except (RuntimeError, SystemExit) as ex:
        out["error"] = str(ex)
    finally:
        for w in list(Worker.live):
            w.close()
    finish(args, out, records, ready, tokens, verify)


def run_workers(ctx, args, coins: Coins, p: dict, records: list[dict], out: dict, arm_steps) -> None:
    """Both engines resident, the reps alternating; or, if the arm can't start beside the base, ABAB blocks."""
    warm: set = set()
    base = Worker(ctx, args, "base", coins)
    tokens_fp8 = base.ask(op="chat", mode="fp8")
    out["tokens_fp8"] = tokens_fp8
    graphed = None
    if getattr(args, "fp8_graphs", False):
        graphed = Worker(ctx, args, "base-graphs", coins)
        out["tokens_fp8_graphs"] = graphed.ask(op="chat", mode="fp8-graphs")
    arm = None
    if args.interleave == "rep":
        try:
            arm = Worker(ctx, args, "arm", coins)
        except RuntimeError as ex:
            out["interleave"], out["resident_failure"] = "blocks", str(ex).splitlines()[0]
    if out["interleave"] == "rep":
        arm_steps(arm, last=False)
        zk = bool(getattr(args, "statement", None))
        measure({"base": base, "base-graphs": graphed, "arm": arm}, schedule(args.reps, "rep", graphed is not None, zk)[0], p,
                records, warm)
        arm_steps(arm, last=True)
        arm.close()
        base.close()
        if graphed is not None:
            graphed.close()
    else:
        segments = schedule(args.reps, "blocks", zk=bool(getattr(args, "statement", None)))
        for i, segment in enumerate(segments):
            group = GROUP[segment[0]]
            w = base if group == "base" and base is not None else Worker(ctx, args, group, coins)
            if group == "arm":
                arm_steps(w, last=False)
            measure({group: w}, segment, p, records, warm)
            if group == "arm" and i == len(segments) - 1:
                arm_steps(w, last=True)
            w.close()
            base = None


def finish(args, out: dict, records: list[dict], ready: dict, tokens: dict, verify: dict) -> None:
    tokens["fp8"] = out.pop("tokens_fp8", {})
    if "tokens_fp8_graphs" in out:
        tokens["fp8-graphs"] = out.pop("tokens_fp8_graphs")
    results = timing(records)
    for mode, r in results.items():
        r.update(tokens.get(mode, {}))
    summary = {"split": split(results), "interleave": out["interleave"], "resident_failure": out["resident_failure"],
               "token_agreement_vs_bf16": {m: agree(r["tokens"], results["bf16"]["tokens"]) for m, r in results.items()
                                           if "tokens" in r and "tokens" in results.get("bf16", {})},
               "gates": {"arm": all(g["ok"] for g in ready.get("arm", {}).get("gate", [{"ok": False}]))},
               "verify_pass": {k: {"manifest": v["manifest"], "same_commitment_as_timed": v.get("same_commitment_as_timed"),
                                   "same_commitment_eager": v.get("same_commitment_eager")}
                               for k, v in verify.items()}}
    out.update(summary=summary, arm=ready.get("arm"), results=results)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(summary), flush=True)
    import run_result
    ok = (not out.get("error") and all(m in results for m in MODES) and summary["gates"]["arm"]
          and all(v["same_commitment_as_timed"] and v.get("same_commitment_eager") is not False
                  for k, v in summary["verify_pass"].items() if not k.startswith("control")))
    ms = [(f"{m}.{k}", v, "s") for m, r in results.items() for k, v in r["timing"].items()]
    run_result.write(args.out, ok, "pearl-c vLLM e2e: gated, fp8 interleaved with the Pearl-C engine, and each verify pass repeats "
                     "its timed commitment", json.dumps(summary), ms)


if __name__ == "__main__":
    main()
