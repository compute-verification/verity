"""PoUW's MVP end to end on the RTX PRO 6000 (sm_120): Pearl-C (`pearl-c-sm120-v1-h1`) commits every linear call of a vLLM model
on the device, timed against stock vLLM on the same card at the panel's two shapes, rep by rep.

  python benchmarks/pouw/pearl_c_vllm/e2e.py --model DIR --kernel benchmarks/pouw/pearl_c_sm120 --cubin SHIP/pearl_c_sm120.cubin \\
      --out RUN_DIR/e2e.json [--retain DIR] [--reps 3]

The run (research store ``docs/pouw/mvp-e2e.md``, "RTX PRO 6000"):
- **Model:** Llama-3.1-8B-Instruct, TP 1, eager, prefix caching off.  Every linear is in the kernels' domain: k is 4,096 or
  14,336, and n is 6,144, 4,096 or 28,672.
- **Two engines stay resident, each in its own process on the one GPU** (the root's ruling, 12:11Z):
  - ``base``: stock vLLM ``quantization="fp8"``, the baseline the panel carries;
  - ``arm``: the BF16 model with every LinearBase apply wrapped.  Per rep the wrapper is off (``bf16``, stock vLLM) or runs
    the `PearlCDeviceRun` without hashing (``pearlc-nohash``) or with it (``pearlc``).
  The controller sends one command at a time, so each engine is idle while the other times.  The reps alternate
  (`CYCLE`: fp8, pearlc, fp8, pearlc-nohash, fp8, bf16), each rep on the same prompts in every mode.  If the second engine
  can't start beside the first, the run falls back to ABAB blocks (`BLOCKS` engine lifetimes, alternating), and records it.
- **Prefill** (the panel's m = 8,192): one prompt of 8,192 random tokens with max_tokens 1 and a batch budget of 8,192 tokens,
  so each linear call has m = 8,192.
- **Decode** (the panel's m = 32): 32 prompts of 64 tokens.  The step is (t(1 + D tokens) − t(1 token)) / D over D = 64
  decode steps, whose linear calls have m = 32.
- **The split:** ``pearlc`` over ``fp8``, the headline, and over ``bf16``; the GEMM (``gemm_pearlc``'s kernel time in
  ``pearlc-nohash``), hashing and records (``pearlc`` less ``pearlc-nohash``), and everything else.
- **Gates, in the arm's process, before any timing:** at every distinct linear shape and both m, served y is held to the
  float product x·Wᵀ (relative RMS error under 8%, the scheme's tested bound), with hashing and without.  With hashing, the
  same call made twice, each from a fresh pass on poisoned outputs, must give the same A root and tile root, and the
  negative control, the call once more with its tile leaves omitted, must not.
- **The verify passes** (``--retain``): after the timing, the first timed prefill prompt and the decode prompts are replayed
  in the arm, each a retained pass on poisoned buffers, whose commitment must equal its timed pass's.  Then the negative
  control: one decode prompt as a retained pass writing none of the commitment (`CONTROL_OMITS`).  The three manifests go to
  `verify_run.py` after the window: the first two must be accepted and the control rejected (the panel's rule of 11:50Z).
- **The panel's shapes** (the coordinator, 11:50Z): `SHAPES`, under ``pearl-c-sm120`` v1-h1; `panel_rows.py` writes the rows.

Every rep records the SM clock (the panel's clock rule).  ``--smoke`` (`smoke`) is the untimed check before a window: both
engines resident, the arm's install and gates, and each engine's first sampling calls.
"""
from __future__ import annotations

import argparse
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

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SCHEME = "pearl-c-sm120-v1-h1"
MODES = ("bf16", "fp8", "pearlc-nohash", "pearlc")
PEARLC = ("pearlc-nohash", "pearlc")
GROUP = {"fp8": "base", "bf16": "arm", "pearlc-nohash": "arm", "pearlc": "arm"}
#: one rep of every mode, the baseline between each other mode's
CYCLE = ("fp8", "pearlc", "fp8", "pearlc-nohash", "fp8", "bf16")
#: the fallback's engine lifetimes: base, arm, base, arm, each timing its share of the reps
BLOCKS = 4
PREFILL, BATCH, PROMPT, STEPS = 8192, 32, 64, 64
#: the served y's bound against the float product (the scheme's tested bound, protocols/pouw's sm120 useful-output test)
ERR = 0.08
#: the negative control's omitted steps: every write of the commitment (A's row tree, the digests, the tile leaves)
CONTROL_OMITS = frozenset({"pad_a", "node_keys_a", "hash_rows_a", "tree_a", "hash_msg", "hash_leaf"})
#: the panel's shape names for the two phases
SHAPES = {"prefill_8192": "e2e-llama31-8b-vllm-m8192", "decode_b32_step": "e2e-llama31-8b-vllm-m32"}


def engine(args, group: str):
    from vllm import LLM
    kw = {"model": args.model, "enforce_eager": True, "gpu_memory_utilization": args.mem_base if group == "base" else args.mem_arm,
          "max_num_seqs": BATCH, "max_model_len": PREFILL + 64, "max_num_batched_tokens": PREFILL, "enable_prefix_caching": False,
          "seed": 0}
    if args.revision:
        kw["revision"] = args.revision
    if group == "base":
        kw["quantization"] = "fp8"
    try:
        return LLM(**kw, compilation_config={"mode": 0})
    except TypeError:
        return LLM(**kw)


def install(model, args, beacon: bytes, state: dict):
    """Every LinearBase layer's apply, found by type, through a `PearlCDeviceRun` over the layers' own weights while
    ``state["active"]``, else stock."""
    from verity_pouw.audit import Epoch, Workload
    from verity_pouw.protocol import Matmul, Shape
    from verity_vllm.protocol_options import pouw_pearl_c_device as PPD
    from vllm.model_executor.layers.linear import LinearBase

    layers = [m for m in model.modules() if isinstance(m, LinearBase)]
    weights = {w: layer.weight.data for w, layer in enumerate(layers)}
    bad = PPD.refusals(SCHEME, {w: tuple(t.shape) for w, t in weights.items()})
    if bad:
        raise SystemExit(f"{SCHEME} refuses the model: " + "; ".join(bad))
    kernel = PPD.Kernel.load(args.kernel, args.cubin)
    roots = PPD.weight_roots(kernel, weights)
    declared = Workload(tuple(Matmul(w, Shape(PPD.ROW_ALIGN, t.shape[1], t.shape[0]), w) for w, t in weights.items()))
    salt = Epoch.start(declared, roots, beacon).salt
    run = PPD.PearlCDeviceRun(PPD.scheme_of(SCHEME), f"pouw:{SCHEME}", kernel, weights, roots, salt, beacon, time_gemm=True)

    def make(original):
        def apply(layer, x, bias=None):
            w = getattr(layer, "_pouw_w", None)
            return run.gemm(w, x, layer.weight, bias) if state["active"] and w is not None else original(layer, x, bias)
        return apply

    for w, layer in enumerate(layers):
        layer._pouw_w = w
    for qm in {id(layer.quant_method): layer.quant_method for layer in layers}.values():
        qm.apply = make(type(qm).apply.__get__(qm))
    return run, weights


def gate(run, weights: dict) -> list[dict]:
    """Served y against x·Wᵀ, and with hashing the same call's commitment twice and the negative control, at every
    distinct (n, k) and both m."""
    import torch
    out, seen = [], set()
    gen = torch.Generator(device=next(iter(weights.values())).device).manual_seed(20260930)
    for w, W in weights.items():
        if tuple(W.shape) in seen:
            continue
        seen.add(tuple(W.shape))
        for m in (PREFILL, BATCH):
            x = torch.randn(m, W.shape[1], device=W.device, generator=gen).to(torch.bfloat16)
            commits, served = [], None
            for omit in ((), (), ("hash_leaf",)) if run.hashing else ((),):
                run.begin()
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
            out.append({"weight": w, "n": W.shape[0], "k": W.shape[1], "m": m, "hashing": run.hashing, "rel_rms": err,
                        "commitment_repeats": same, "omitted_leaves_rejected": control, "ok": err < ERR and same and control})
    run.details["passes"].clear()
    run.gemm_ms()
    return out


def clocks() -> dict:
    """The SM and memory clocks now, from nvidia-smi (the panel's per-rep clock record)."""
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=uuid,clocks.sm,clocks.mem,power.draw", "--format=csv,noheader,nounits",
                            "-i", "0"], capture_output=True, text=True, timeout=10, check=False).stdout.strip().split(", ")
        return {"uuid": q[0], "sm_mhz": int(q[1]), "mem_mhz": int(q[2]), "watts": float(q[3])}
    except Exception as ex:  # noqa: BLE001 - a missing reading is recorded, not fatal
        return {"error": f"{type(ex).__name__}: {ex}"}


# ---- one resident engine --------------------------------------------------------------------------------------------

def serve(conn, args, group: str, beacon: bytes) -> None:
    """One engine, resident in its own process: ``base`` (stock FP8) or ``arm`` (BF16 with the Pearl-C wrapper), answering
    the controller's commands one at a time."""
    try:
        import torch
        from vllm import SamplingParams
        from vllm_bench import PROMPTS, find_model

        llm = engine(args, group)
        state, run = {"active": False}, None
        ready: dict = {"group": group}
        if group == "arm":
            run, weights = install(find_model(llm), args, beacon, state)
            ready["gate"] = gate(run, weights)
            run.hashing = False
            ready["gate"] += gate(run, weights)
            run.hashing = True
            ready.update(committer=run.details["committer"], excluded_b=sum(run.details["excluded_b"].values()))
        conn.send(ready)
        one = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        many = SamplingParams(temperature=0.0, max_tokens=1 + STEPS, ignore_eos=True)
        tok = llm.get_tokenizer()
        chat = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in PROMPTS]

        def timed(prompts, sp, retain=None):
            pearl = state["active"]
            if pearl:
                run.begin(retain, os.environ.get("RESEARCH_RUN_ID"))
                run.gemm_ms()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            commit = run.end() if pearl else {}
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
                state["active"] = mode in PEARLC
                if run:
                    run.hashing = mode == "pearlc"
            if op == "chat":
                *_, outs = timed(chat, SamplingParams(temperature=0.0, max_tokens=48))
                conn.send({"tokens": [list(o.outputs[0].token_ids) for o in outs], "texts": [o.outputs[0].text for o in outs[:4]]})
            elif op == "prefill":
                dt, gemm, commit, _ = timed([{"prompt_token_ids": cmd["prompt"]}], one)
                conn.send({"s": dt, "gemm_s": gemm, "commitment": commit})
            elif op == "decode":
                p = [{"prompt_token_ids": x} for x in cmd["prompts"]]
                t1, g1, _, _ = timed(p, one)
                tn, gn, cn, _ = timed(p, many)
                conn.send({"step_s": (tn - t1) / STEPS, "gemm_step_s": (gn - g1) / STEPS, "t1_s": t1, "tn_s": tn, "commitment": cn})
            elif op == "verify":
                state["active"], run.hashing = True, True
                out = {}
                for name, prompts, sp in (("prefill", [cmd["prefill"]], one), ("decode", cmd["decode"], many)):
                    _, _, got, _ = timed([{"prompt_token_ids": x} for x in prompts], sp, retain=str(Path(args.retain) / name))
                    want = cmd["want"][name]
                    same = all(got.get(k) == want.get(k) for k in ("transcript_root", "activation_roots", "units", "tiles"))
                    out[name] = {"manifest": got.get("manifest"), "commitment": got, "same_commitment_as_timed": same}
                run.omit = CONTROL_OMITS
                _, _, got, _ = timed([{"prompt_token_ids": cmd["decode"][0]}], one, retain=str(Path(args.retain) / "control"))
                run.omit = frozenset()
                out["control"] = {"manifest": got.get("manifest"), "commitment": got, "omitted": sorted(CONTROL_OMITS)}
                conn.send(out)
            else:
                raise ValueError(f"unknown command {op!r}")
    except Exception as ex:  # noqa: BLE001 - the controller reports it
        conn.send({"error": f"{group}: {type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-4000:]})


class Worker:
    started = 0
    live: ClassVar[list[Worker]] = []

    def __init__(self, ctx, args, group: str, beacon: bytes):
        Worker.started += 1
        self.group, self.serial, (self.conn, child) = group, Worker.started, ctx.Pipe()
        self.proc = ctx.Process(target=serve, args=(child, args, group, beacon))
        Worker.live.append(self)
        self.proc.start()
        try:
            self.ready = self.recv(1200)
        except RuntimeError:
            self.proc.join(30)
            if self.proc.is_alive():
                self.proc.terminate()
            raise

    def recv(self, timeout: float = 1200) -> dict:
        if not self.conn.poll(timeout):
            raise RuntimeError(f"{self.group}: no answer in {timeout:.0f} s")
        try:
            msg = self.conn.recv()
        except EOFError:
            raise RuntimeError(f"{self.group}: the engine's process exited (code {self.proc.exitcode})") from None
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
    ids = lambda n: [rng.randrange(100, 20000) for _ in range(n)]
    return {"warm": ids(PREFILL), "prefill": [ids(PREFILL) for _ in range(reps)], "decode": [ids(PROMPT) for _ in range(BATCH)]}


def schedule(reps: int, interleave: str) -> list[list[str]]:
    """The timed modes in order, as one segment per engine lifetime: with both engines resident one segment, the reps
    alternating by `CYCLE`; in ``blocks``, `BLOCKS` segments alternating base and arm, each with its share of the reps."""
    if interleave == "rep":
        return [list(CYCLE) * reps]
    share = [reps // 2 + reps % 2, reps // 2]
    arm = [m for m in CYCLE if GROUP[m] == "arm"]
    return [["fp8"] * share[i // 2] if i % 2 == 0 else arm * share[i // 2] for i in range(BLOCKS)]


def timing(records: list[dict]) -> dict:
    """Each mode's medians over its reps, and its reps, from the controller's records."""
    out = {}
    for mode in MODES:
        pre = [r for r in records if r["mode"] == mode and r["phase"] == "prefill"]
        dec = [r for r in records if r["mode"] == mode and r["phase"] == "decode"]
        if not pre or not dec:
            continue
        out[mode] = {"timing": {"prefill_8192_s": statistics.median(r["s"] for r in pre),
                                "prefill_8192_gemm_s": statistics.median(r["gemm_s"] for r in pre),
                                "decode_b32_step_s": statistics.median(r["step_s"] for r in dec),
                                "decode_b32_step_gemm_s": statistics.median(r["gemm_step_s"] for r in dec)},
                     "reps": {"prefill_8192": pre, "decode_b32": dec}}
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
    for key in ("prefill_8192", "decode_b32_step"):
        if not all(m in t for m in MODES):
            continue
        full, nohash, gemm = t["pearlc"][f"{key}_s"], t["pearlc-nohash"][f"{key}_s"], t["pearlc-nohash"][f"{key}_gemm_s"]
        row = {"bf16_s": t["bf16"][f"{key}_s"], "fp8_s": t["fp8"][f"{key}_s"], "pearlc_s": full, "nohash_s": nohash, "gemm_s": gemm,
               "hashing_s": full - nohash, "else_s": nohash - gemm}
        for base in ("fp8", "bf16"):
            b = row[f"{base}_s"]
            row.update({f"slowdown_vs_{base}": full / b, f"gemm_x_{base}": gemm / b, f"hashing_x_{base}": (full - nohash) / b,
                        f"else_x_{base}": (nohash - gemm) / b})
        out[key] = {"shape": SHAPES[key], **row}
    return out


def memory_mib() -> int | None:
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"], capture_output=True,
                           text=True, timeout=10, check=False)
        return int(q.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def smoke(args) -> None:
    """Untimed: both engines resident on the GPU, and each one's first sampling calls.  The stock FP8 engine starts and
    samples; then the Pearl-C engine starts beside it (its install and gates) and samples as stock BF16 and as ``pearlc``.
    Writes each step's time and the GPU's memory use after it, the gates, and the greedy tokens' agreement with BF16."""
    ctx, beacon = mp.get_context("spawn"), os.urandom(32)
    out: dict = {"schema": "verity-pouw/pearl-c-vllm-smoke/v1", "scheme": SCHEME, "run": os.environ.get("RESEARCH_RUN_ID"),
                 "steps": [], "resident": False}

    def step(name, f):
        t0 = time.perf_counter()
        got = f()
        out["steps"].append({"step": name, "s": round(time.perf_counter() - t0, 2), "memory_mib": memory_mib()})
        return got

    try:
        base = step("start the fp8 engine", lambda: Worker(ctx, args, "base", beacon))
        fp8 = step("fp8: first sampling calls", lambda: base.ask(op="chat", mode="fp8"))
        arm = step("start the pearl-c engine beside it (install, gates)", lambda: Worker(ctx, args, "arm", beacon))
        out["resident"], out["gate"] = True, arm.ready["gate"]
        out["gate_ok"] = all(g["ok"] for g in arm.ready["gate"])
        bf16 = step("bf16: first sampling calls", lambda: arm.ask(op="chat", mode="bf16"))
        pearlc = step("pearlc: first sampling calls", lambda: arm.ask(op="chat", mode="pearlc"))
        out["texts"] = {"fp8": fp8["texts"][0], "bf16": bf16["texts"][0], "pearlc": pearlc["texts"][0]}
        out["token_agreement_vs_bf16"] = {"fp8": agree(fp8["tokens"], bf16["tokens"]), "pearlc": agree(pearlc["tokens"], bf16["tokens"])}
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


def agree(a, b) -> float:
    same = sum(x == y for s, u in zip(a, b) for x, y in zip(s, u))
    return same / max(1, sum(min(len(s), len(u)) for s, u in zip(a, b)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--kernel", default=str(HERE.parent / "pearl_c_sm120"), help="GPU 1's kernel directory (run.py, fixture.py)")
    ap.add_argument("--cubin", help="build.sh's pearl_c_sm120.cubin (default: the kernel directory's)")
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "e2e.json")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--mem-base", type=float, default=0.2, help="the FP8 engine's gpu_memory_utilization (weights about 8 GB)")
    ap.add_argument("--mem-arm", type=float, default=0.3, help="the BF16 engine's; Pearl-C's buffers (about 13 GB) sit outside it")
    ap.add_argument("--interleave", choices=["rep", "blocks"], default="rep", help="blocks: skip co-residency, go straight to ABAB")
    ap.add_argument("--retain", help="the verify passes' directory")
    ap.add_argument("--smoke", action="store_true", help="untimed: both engines resident and each one's first sampling calls")
    args = ap.parse_args()
    if args.smoke:
        smoke(args)
        return
    ctx, beacon, p = mp.get_context("spawn"), os.urandom(32), prompts(args.reps)
    records: list[dict] = []
    out: dict = {"schema": "verity-pouw/pearl-c-vllm-e2e/v2", "scheme": SCHEME, "run": os.environ.get("RESEARCH_RUN_ID"),
                 "beacon": beacon.hex(), "interleave": args.interleave, "resident_failure": None}
    ready, tokens, verify = {}, {}, {}

    def arm_steps(arm: Worker, last: bool) -> None:
        ready["arm"] = arm.ready
        if not all(g["ok"] for g in arm.ready["gate"]):
            raise SystemExit("the arm's gate failed: " + json.dumps([g for g in arm.ready["gate"] if not g["ok"]]))
        for mode in ("bf16", "pearlc-nohash", "pearlc"):
            if mode not in tokens:
                tokens[mode] = arm.ask(op="chat", mode=mode)
        if last and args.retain:
            want = {ph: next(r["commitment"] for r in records if r["mode"] == "pearlc" and r["phase"] == ph and r["rep"] == 0)
                    for ph in ("prefill", "decode")}
            verify.update(arm.ask(op="verify", prefill=p["prefill"][0], decode=p["decode"], want=want))

    try:
        run_workers(ctx, args, beacon, p, records, out, arm_steps)
    except (RuntimeError, SystemExit) as ex:
        out["error"] = str(ex)
    finally:
        for w in list(Worker.live):
            w.close()
    finish(args, out, records, ready, tokens, verify)


def run_workers(ctx, args, beacon: bytes, p: dict, records: list[dict], out: dict, arm_steps) -> None:
    """Both engines resident, the reps alternating; or, if the arm can't start beside the base, ABAB blocks."""
    warm: set = set()
    base = Worker(ctx, args, "base", beacon)
    tokens_fp8 = base.ask(op="chat", mode="fp8")
    out["tokens_fp8"] = tokens_fp8
    arm = None
    if args.interleave == "rep":
        try:
            arm = Worker(ctx, args, "arm", beacon)
        except RuntimeError as ex:
            out["interleave"], out["resident_failure"] = "blocks", str(ex).splitlines()[0]
    if out["interleave"] == "rep":
        arm_steps(arm, last=False)
        measure({"base": base, "arm": arm}, schedule(args.reps, "rep")[0], p, records, warm)
        arm_steps(arm, last=True)
        arm.close()
        base.close()
    else:
        segments = schedule(args.reps, "blocks")
        for i, segment in enumerate(segments):
            group = GROUP[segment[0]]
            w = base if group == "base" and base is not None else Worker(ctx, args, group, beacon)
            if group == "arm":
                arm_steps(w, last=False)
            measure({group: w}, segment, p, records, warm)
            if group == "arm" and i == len(segments) - 1:
                arm_steps(w, last=True)
            w.close()
            base = None


def finish(args, out: dict, records: list[dict], ready: dict, tokens: dict, verify: dict) -> None:
    tokens["fp8"] = out.pop("tokens_fp8", {})
    results = timing(records)
    for mode, r in results.items():
        r.update(tokens.get(mode, {}))
    summary = {"split": split(results), "interleave": out["interleave"], "resident_failure": out["resident_failure"],
               "token_agreement_vs_bf16": {m: agree(r["tokens"], results["bf16"]["tokens"]) for m, r in results.items()
                                           if "tokens" in r and "tokens" in results.get("bf16", {})},
               "gates": {"arm": all(g["ok"] for g in ready.get("arm", {}).get("gate", [{"ok": False}]))},
               "verify_pass": {k: {"manifest": v["manifest"], "same_commitment_as_timed": v.get("same_commitment_as_timed")}
                               for k, v in verify.items()}}
    out.update(summary=summary, arm=ready.get("arm"), results=results)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(summary), flush=True)
    import run_result
    ok = (not out.get("error") and all(m in results for m in MODES) and summary["gates"]["arm"]
          and all(v["same_commitment_as_timed"] for k, v in summary["verify_pass"].items() if k != "control"))
    ms = [(f"{m}.{k}", v, "s") for m, r in results.items() for k, v in r["timing"].items()]
    run_result.write(args.out, ok, "pearl-c vLLM e2e: gated, fp8 interleaved with the Pearl-C engine, and each verify pass repeats "
                     "its timed commitment", json.dumps(summary), ms)


if __name__ == "__main__":
    main()
