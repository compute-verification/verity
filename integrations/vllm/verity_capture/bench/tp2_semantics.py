"""R13 lane tp, P4 step 1 — the TP2 numerical-semantics question.

Runs ONE workload through the pinned BI vLLM at `--tp N` with the per-rank `worker_extension_cls` seam of `tp2_worker.py`
and records, on every rank, the rank-local inputs / partial GEMM outputs / collective outputs of the parallel layers of
`--layers` (+ `embed_tokens`, `logits_processor`) for the first `--calls` forward calls (call 0 = the prefill step of the
real request, then decode steps), plus the rank's weight shards.  `--tp 1` records the same modules in the in-process
TP1 engine (the reference execution).  `tp2_semantics_analyze.py` then recomputes both semantics offline and compares.

    HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1 NCCL_P2P_DISABLE=1 PYTHONPATH=. \
      /workspace/venv-cu129/bin/python -m verity_capture.bench.tp2_semantics --workload <manifest> --case LLAMA32_1B --tp 2 \
        --out /workspace/cp/tp2sem/tp2 --layers 0,15 --calls 3 --engine-arg disable_custom_all_reduce=true --engine-arg gpu_memory_utilization=0.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from verity_vllm_adapter import profile as prof

prof.apply_env()

from verity_capture import vllm_adapter as va
from verity_capture.bench.tp2_capture import parse_engine_args
from verity_capture.bench.tp2_worker import EXTENSION


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workload", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--case", default=None)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--layers", default="0,15")
    ap.add_argument("--calls", type=int, default=3)
    ap.add_argument("--engine-arg", action="append", default=[])
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--rpc-timeout", type=float, default=1800.0)
    args = ap.parse_args()

    from vllm import LLM

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wl = va.load_workload(args.workload, None)
    case = va.case_of(wl, args.case)
    env_flags = prof.apply_env()
    cp = prof.checkpoint_for(case, args.checkpoints)
    kw = prof.engine_kwargs(cp, replay=False, enforce_eager=True, tensor_parallel_size=args.tp)
    kw["enable_prefix_caching"] = False
    kw["max_num_seqs"] = int(args.max_num_seqs)
    kw.update(parse_engine_args(args.engine_arg))
    kw["worker_extension_cls"] = EXTENSION
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    print(f"[tp2sem] tp={args.tp} case={case} layers={layers} calls={args.calls} kwargs={ {k: v for k, v in kw.items() if k not in ('model', 'tokenizer')} }", flush=True)
    t0 = time.perf_counter()
    llm = LLM(**kw)
    build_s = time.perf_counter() - t0
    eng = va.Engine(llm=llm, runner=None, model=None, cp=cp, kwargs=kw, env_flags=env_flags, build_s=build_s, case=case)
    topo = llm.collective_rpc("tp2_probe", timeout=args.rpc_timeout)
    (out / "topology.json").write_text(json.dumps({"tp": args.tp, "build_s": round(build_s, 1), "engine_kwargs": {k: v for k, v in kw.items() if k not in ("model", "tokenizer")},
                                                    "env": {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "VLLM_", "CUBLAS"))},
                                                    "ranks": topo}, indent=1, default=str) + "\n")
    print(f"[tp2sem] engine {build_s:.1f}s; ranks={[(r.get('pid'), (r.get('tp') or {}).get('rank'), (r.get('device') or {}).get('current')) for r in topo]}; "
          f"comm={[(r.get('tp') or {}).get('device_communicator') for r in topo][:1]} pynccl={[(r.get('tp') or {}).get('pynccl_comm') for r in topo][:1]}", flush=True)
    if not args.no_warmup:
        warm = va.throwaway_requests(eng, wl["requests"])
        _, tw = va.run_requests(eng, warm, wl["sampling"])
        print(f"[tp2sem] warm-up {tw['wall_s']:.2f}s, {tw['engine_steps']} steps", flush=True)
    inst = llm.collective_rpc("tp2_sem_install", timeout=args.rpc_timeout, kwargs={"out_dir": str(out), "layers": layers, "max_calls": int(args.calls)})
    print(f"[tp2sem] hooks installed: {[(i['rank'], len(i['hooked']), len(i['weights'])) for i in inst]}", flush=True)
    t0 = time.perf_counter()
    tokens, timing = va.run_requests(eng, wl["requests"], wl["sampling"])
    run_s = time.perf_counter() - t0
    closed = llm.collective_rpc("tp2_sem_close", timeout=args.rpc_timeout)
    for c in closed:
        print(f"[tp2sem] rank {c['rank']}: {len(c['records'])} records: " + ", ".join(f"{r['name'].split('.')[-1]}#{r.get('call')}(coll {r.get('n_coll')})" for r in c["records"] if not r.get("missing")), flush=True)
    doc = {"tp": args.tp, "case": case, "model": cp["repo"], "run_s": round(run_s, 3), "timing": timing, "argv": sys.argv,
           "requests": [{"request_id": r["request_id"], "prompt_token_ids": r["prompt_token_ids"], "max_tokens": r["max_tokens"],
                         "generated_token_ids": tokens.get(r["request_id"], [])} for r in wl["requests"]]}
    (out / "tokens.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[tp2sem] run {run_s:.2f}s {timing.get('engine_steps')} steps; tokens -> {out / 'tokens.json'}", flush=True)


if __name__ == "__main__":
    main()
