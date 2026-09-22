"""Compiled-vs-eager mismatch reproducer for one request (CONTRACT §16.8 item 2; W1d).

For request ``--request`` of a recorded production run, replays its trajectory (trace replay, prefix
caching off) under up to three execution profiles in *separate engines of the same process*, hooking
``model.compute_logits`` (eager in every profile: it runs outside the compiled/captured graph), and
saves the logits row of every sampled position as ``.npy`` (bf16 bits as int16 plus float32), the
argmax, and production's committed token:

* ``compiled_graphs``  : production's configuration (Inductor + CUDA graphs FULL_AND_PIECEWISE, custom_ops none);
* ``compiled_nograph`` : the same compiled graph with ``cudagraph_mode=NONE`` (W1d's replay profile, no record pass);
* ``eager``            : ``enforce_eager=True`` (the eager replay profile of the B0/B2 rows).

Kernel identities: for the compiled profiles the content hashes of the generated Triton kernels of a
fresh ``TORCHINDUCTOR_CACHE_DIR`` (``compiled_fx.kernels``); for eager the runtime_impl names of the
eager adapter (recorded in the eager rows' description, quoted in the README).

Usage (pod)::

    TORCHINDUCTOR_CACHE_DIR=... VLLM_CACHE_ROOT=... python -m verity_vllm_adapter.compiled_fx.mismatch_repro \\
        --case B0 --checkpoints manifests/checkpoints.json --production <prod dir> --request 784c37e1 \\
        --positions 511,529 --out /workspace/results/B0/mismatch-t511
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verity_vllm_adapter import profile as prof
from verity_vllm_adapter.compiled_fx import kernels
from verity_vllm_adapter.driver import run_requests
from verity_vllm_adapter.replay import Req, load_production, replay_params

PROFILES = {
    "compiled_graphs": {"enforce_eager": False, "cudagraph_mode": None},
    "compiled_nograph": {"enforce_eager": False, "cudagraph_mode": "NONE"},
    "eager": {"enforce_eager": True, "cudagraph_mode": None},
}


def engine_kwargs(cp: dict[str, Any], prod_manifest: dict[str, Any], profile: str) -> dict[str, Any]:
    spec = PROFILES[profile]
    kw = prof.engine_kwargs(cp, replay=True)
    if spec["enforce_eager"]:
        kw["enforce_eager"] = True
    else:
        kw.pop("enforce_eager", None)
        cc = dict(prod_manifest.get("engine_kwargs", {}).get("compilation_config") or {})
        if spec["cudagraph_mode"]:
            cc["cudagraph_mode"] = spec["cudagraph_mode"]
        if cc:
            kw["compilation_config"] = cc
    return kw


def run_profile(profile: str, cp: dict[str, Any], prod_manifest: dict[str, Any], req: Req, positions: list[int], out: Path, cache_dir: Path, log) -> dict[str, Any]:
    from vllm import LLM

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    vroot = Path(os.environ["VLLM_CACHE_ROOT_BASE"]) / profile
    if vroot.exists():
        shutil.rmtree(vroot)
    os.environ["VLLM_CACHE_ROOT"] = str(vroot)
    kw = engine_kwargs(cp, prod_manifest, profile)
    t0 = time.perf_counter()
    llm = LLM(**kw)
    build_s = time.perf_counter() - t0
    runner = prof.get_model_runner(llm)
    model = runner.get_model()
    effective = prof.effective_compilation(llm)
    log(f"[{profile}] built in {build_s:.1f}s effective={ {k: effective.get(k) for k in ('mode', 'cudagraph_mode', 'enforce_eager', 'custom_ops')} }")

    plen = len(req.prompt_token_ids)
    captured: list[torch.Tensor] = []
    orig = model.compute_logits

    def compute_logits(hidden_states, *a, **kw2):
        out_ = orig(hidden_states, *a, **kw2)
        if out_ is not None and capture_on[0]:
            captured.append(out_.detach().clone())
        return out_

    capture_on = [False]
    model.compute_logits = compute_logits
    # warm-up on an unrelated prompt (compiled profiles: the engine already ran its own warm-up)
    rng = np.random.default_rng(1)
    vocab = int(getattr(model.config, "vocab_size", 32000))
    warm = [Req({"request_id": "warmup-0", "prompt_token_ids": rng.integers(3, vocab - 1, size=64).tolist(),
                 "output_token_ids": rng.integers(3, vocab - 1, size=4).tolist(), "sampling_params": {}, "finish_reason": None})]
    run_requests(llm, warm, lambda r: replay_params(r, 1))
    capture_on[0] = True
    torch.cuda.synchronize()
    outs, timing = run_requests(llm, [req], lambda r: replay_params(r, 1))
    torch.cuda.synchronize()
    capture_on[0] = False
    o = outs[req.request_id] if isinstance(outs, dict) else outs[0]
    gen = list(o.outputs[0].token_ids)
    # one compute_logits call per engine step for this single request -> step i is position plen-1+i
    rows: dict[int, dict[str, Any]] = {}
    for i, lg in enumerate(captured):
        t = plen - 1 + i
        if lg.shape[0] != 1:
            continue
        if t in positions:
            row = lg[0]
            f32 = row.float().cpu().numpy()
            np.save(out / f"logits_{profile}_t{t}.npy", f32)
            if row.dtype == torch.bfloat16:
                np.save(out / f"logits_{profile}_t{t}_bf16bits.npy", row.view(torch.int16).cpu().numpy())
            top = np.argsort(-f32)[:5]
            rows[t] = {"argmax": int(f32.argmax()), "top5": [(int(k), float(f32[k])) for k in top], "dtype": str(row.dtype).replace("torch.", ""),
                       "production_token": int(req.output_token_ids[t - plen + 1]) if 0 <= t - plen + 1 < len(req.output_token_ids) else None,
                       "logit_of_production_token": float(f32[int(req.output_token_ids[t - plen + 1])]) if 0 <= t - plen + 1 < len(req.output_token_ids) else None}
    kid = None
    if not PROFILES[profile]["enforce_eager"]:
        ks, ws = kernels.collect(cache_dir)
        by = {}
        for k in ks:
            by.setdefault(k.norm_hash, k.name)
        kid = {"distinct_kernels": len(by), "kernels": {h: kernels._KNAME.sub(lambda m: __import__('re').sub(r'_\d+$', '', m.group(0)), n) for h, n in sorted(by.items())}, "wrappers": len(ws)}
    doc = {"profile": profile, "engine_kwargs": kw, "effective_execution": effective, "build_s": build_s, "run_s": timing.get("wall_s"),
           "trajectory_followed": gen == list(req.output_token_ids), "generated_tokens": gen, "positions": rows, "kernel_identity": kid,
           "compute_logits_calls": len(captured)}
    del llm
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--production", required=True)
    ap.add_argument("--request", required=True, help="request id or unique prefix")
    ap.add_argument("--positions", default="511,529")
    ap.add_argument("--profiles", default="compiled_graphs,compiled_nograph,eager")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "repro.log", "a")  # noqa: SIM115

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    os.environ.setdefault("VLLM_CACHE_ROOT_BASE", str(out / "vllm_cache"))
    cp = prof.checkpoint_for(args.case, args.checkpoints)
    prod_manifest, prod_rows = load_production(Path(args.production))
    rows = [r for r in prod_rows if r["request_id"].startswith(args.request)]
    if len(rows) != 1:
        raise SystemExit(f"request prefix {args.request!r} matches {len(rows)} rows")
    req = Req(rows[0])
    positions = [int(x) for x in args.positions.split(",")]
    prev = json.loads((out / "mismatch.json").read_text()) if (out / "mismatch.json").exists() else {}
    result: dict[str, Any] = {"case": args.case, "request_id": req.request_id, "prompt_len": len(req.prompt_token_ids), "positions": positions,
                              "production_profile_id": prof.profile_id_of(prod_manifest),
                              "production_effective_execution": prod_manifest.get("effective_execution"), "device": torch.cuda.get_device_name(0),
                              "argv": sys.argv, "profiles": dict(prev.get("profiles", {}))}
    for profile in args.profiles.split(","):  # one engine per process is the safe way: run the profiles as separate invocations
        result["profiles"][profile] = run_profile(profile, cp, prod_manifest, req, positions, out, out / "inductor_cache" / profile, log)
        (out / "mismatch.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    # cross-profile summary
    summ = {}
    for t in positions:
        row = {}
        for p, d in result["profiles"].items():
            pr = d["positions"].get(t) or d["positions"].get(str(t))
            if pr:
                row[p] = {"argmax": pr["argmax"], "production_token": pr["production_token"]}
        fs = {p: np.load(out / f"logits_{p}_t{t}.npy") for p in result["profiles"] if (out / f"logits_{p}_t{t}.npy").exists()}
        names = sorted(fs)
        row["pairwise"] = {f"{a}~{b}": {"equal": bool(np.array_equal(fs[a], fs[b])), "max_abs_diff": float(np.max(np.abs(fs[a] - fs[b]))), "n_diff": int(np.sum(fs[a] != fs[b]))}
                           for i, a in enumerate(names) for b in names[i + 1:]}
        summ[str(t)] = row
    result["summary"] = summ
    (out / "mismatch.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    log(json.dumps(summ, indent=1))
    shutil.rmtree(out / "vllm_cache", ignore_errors=True)


if __name__ == "__main__":
    main()
