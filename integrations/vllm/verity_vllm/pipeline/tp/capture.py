"""Lane tp2 (mechanism prototype): the M1 capture driven through vLLM's multiprocess executor at `tensor_parallel_size > 1`.

    HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 PYTHONPATH=. verity-vllm tp-capture \
        --workload workloads/workload_qwen15_32x16_1req.json --case B1 --tp 2 --out /vol/cp/tp2/runs/<name>/capture \
        [--snapshot-steps all|0,1,2,3] [--no-observer] [--census] [--verify-weights] [--engine-arg key=value ...]

Same κ as `bench/m1_capture.py` (`profile.apply_env()`, `engine_kwargs(cp, enforce_eager=True)`, `enable_prefix_caching=False`,
uninstrumented throwaway warm-up, our request ids) with two differences forced by the process topology:

  * `tensor_parallel_size` (and any `--engine-arg`) is passed to `LLM(...)` directly — `vllm_adapter.build_engine` exposes no
    engine-argument passthrough and ends in `get_model_runner(llm)`, which has no in-process runner to find under the
    multiprocess executor (PROPOSED: `build_engine(engine_args=...)`, `m1_capture/run_config --engine-arg`);
  * the observer is installed in every rank's process through `worker_extension_cls` = `tp2_worker.EXTENSION` and driven with
    `llm.collective_rpc("tp2_install" | "tp2_close")`; the run directory holds one raw log per rank (`log.rank<r>.jsonl.gz`)
    plus `rank<r>/{values,snapshots.json,versions.json,cos_sin_cache.npy}`, `stats.rank<r>.json`, `tokens.rank<r>.json`, and the
    driver's `tokens.json` (what the engine returned = rank 0's sampler), `topology.json` (per-rank probe), `ranks.json`.

The workload's declared target (`target_of_workload`) is applied as `build_engine` applies it (its environment knobs, then its engine
arguments under the CLI's), every rank's engine is checked against it before anything runs, and each rank's header attests it
(`pipeline.tp.target`).

`--inventory-out PATH`: every rank's module inventory of the loaded engine, before the warm-up, as one document
(`meta_inventory.live_document` of `rank_inventory`), what `capture-map check --inventory` reads, so a capture-map check on the
target costs no engine load of its own.

`--no-observer`: the identical engine without the extension (the non-interference CONTROL; `tokens.json` only).  `--census`:
no observer; one `torch.profiler` session per rank over the workload (`census.rank<r>.json` — the protected `census.py`
profiles the driver process, which launches no kernel under TP).  `--tp 1` reproduces the in-process M1 topology with this
driver (sanity of the driver itself).
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.config import option
from verity_vllm.engine import engine_profile as prof

prof.apply_env()

from verity_vllm.engine import vllm_adapter as va
from verity_vllm.engine.rank_worker import EXTENSION
from verity_vllm.pipeline.tp import target as TT


def parse_engine_args(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--engine-arg expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def parse_steps(s: str | None, max_tokens: int | None) -> list[int]:
    if not s:
        return []
    if s.strip().lower() == "all":
        return list(range(int(max_tokens or 4)))
    return sorted({int(x) for x in s.split(",") if x.strip() != ""})


@dataclass(frozen=True, kw_only=True)
class Options:
    workload: str = option("--workload", default="workloads/workload_qwen15_32x16_1req.json")
    out: str = option("--out")
    checkpoints: str = option("--checkpoints", default="manifests/checkpoints.json")
    case: str | None = option("--case", default=None)
    tp: int = option("--tp", type=int, default=2, help="tensor_parallel_size")
    engine_arg: list[str] = option("--engine-arg", action="append", default=[],
                                   help="extra LLM(...) kwarg key=value (JSON value)")
    snapshot_steps: str = option("--snapshot-steps", default="all",
                                 help="`all` = every step of the workload, or a comma list; empty = none")
    snapshot_max_bytes: int = option("--snapshot-max-bytes", type=int, default=32 << 20)
    no_observer: bool = option("--no-observer", default=False, action="store_true",
                               help="control: same engine, no extension, no observer")
    census: bool = option("--census", default=False, action="store_true",
                          help="no observer; per-rank torch.profiler kernel census")
    no_reenter: bool = option("--no-reenter", default=False, action="store_true")
    no_triton: bool = option("--no-triton", default=False, action="store_true")
    no_module_hooks: bool = option("--no-module-hooks", default=False, action="store_true")
    max_tokens: int | None = option("--max-tokens", type=int, default=None)
    no_warmup: bool = option("--no-warmup", default=False, action="store_true")
    max_num_seqs: int = option("--max-num-seqs", type=int, default=8)
    verify_weights: bool = option("--verify-weights", default=False, action="store_true")
    inventory_out: str | None = option("--inventory-out", default=None,
                                       help="every rank's live module inventory, before the warm-up -> PATH (`capture-map check --inventory`)")
    param_hashes: bool = option("--param-hashes", default=False, action="store_true",
                                help="capture mode: sha256 of every rank's parameter shards -> param_hashes.rank<r>.json")
    record_collectives: bool = option("--record-collectives", default=False, action="store_true",
                                      help="[R17 tp D90] capture mode: also run tp2_match's collective recorder in the same engine run -> <out>/coll.rank<r>.jsonl (+ .pt of --store-steps)")
    store_steps: int = option("--store-steps", type=int, default=1 << 30,
                              help="--record-collectives: store the partial/received tensors of steps < N (default all)")
    adapter_revision: str | None = option("--adapter-revision", default=None)
    wheel_sha256: str = option("--wheel-sha256", default="7aa52ac71500caf74643135a340eb2d03ae0386d4282fc0b23846940029fbbb3")
    vllm_commit: str = option("--vllm-commit", default="d9105ea80")
    rpc_timeout: float = option("--rpc-timeout", type=float, default=1800.0)


def main(args: Options) -> None:
    from vllm import LLM

    from verity_vllm.pipeline.tp import lease as TPL

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wl = va.load_workload(args.workload, args.max_tokens)
    TPL.acquire()        # a leased row's Match leg: every GPU of the task from here to the exit (`row_tp.TpRow.match`); nothing reads the device before
    target = va.target_of_workload(wl)
    target_env, target_kw = TT.apply(target)       # its env knobs before vLLM reads them; its engine args under the CLI's
    case = va.case_of(wl, args.case)
    revision = args.adapter_revision
    if revision is None:
        try:
            revision = json.loads(Path("EXPORT.json").read_text()).get("commit", "unknown")
        except Exception:  # noqa: BLE001
            revision = "unknown"
    env_flags = {**prof.apply_env(), **target_env}
    cp = prof.checkpoint_for(case, args.checkpoints)
    if args.verify_weights:
        prof.verify_local_weights(cp)
    kw = prof.engine_kwargs(cp, replay=False, enforce_eager=True, tensor_parallel_size=args.tp)
    kw["enable_prefix_caching"] = False
    kw["max_num_seqs"] = int(args.max_num_seqs)
    kw.update(target_kw)
    kw.update(parse_engine_args(args.engine_arg))
    observer = not (args.no_observer or args.census)
    with_extension = observer or args.census
    if with_extension:
        kw["worker_extension_cls"] = EXTENSION
    mode = "control" if args.no_observer else ("census" if args.census else "capture")
    print(f"[tp2] mode={mode} tp={args.tp} case={case} kwargs={ {k: v for k, v in kw.items() if k not in ('model', 'tokenizer')} }", flush=True)
    t0 = time.perf_counter()
    llm = LLM(**kw)
    build_s = time.perf_counter() - t0
    eng = va.Engine(llm=llm, runner=None, model=None, cp=cp, kwargs=kw, env_flags=env_flags, build_s=build_s, case=case, target=target)
    TT.require_ranks(llm, target, args.rpc_timeout)
    core = getattr(llm.llm_engine, "engine_core", None)
    inner = getattr(core, "engine_core", None) or core
    executor = getattr(inner, "model_executor", None)
    topo: dict[str, Any] = {"mode": mode, "tp": args.tp, "driver_pid": os.getpid(), "build_s": round(build_s, 2),
                            "engine_core_class": f"{type(core).__module__}.{type(core).__name__}" if core is not None else None,
                            "executor_class": f"{type(executor).__module__}.{type(executor).__name__}" if executor is not None else None,
                            "engine_kwargs": {k: v for k, v in kw.items() if k not in ("model", "tokenizer")}, "env_flags": env_flags,
                            "in_process_runner": None, "ranks": None}
    try:
        prof.get_model_runner(llm)
        topo["in_process_runner"] = True
    except Exception as e:  # noqa: BLE001
        topo["in_process_runner"] = f"none: {e}"
    versions = va.versions_doc(eng)
    versions["gpu_count"] = __import__("torch").cuda.device_count()
    versions["target"] = TT.as_json(target)
    (out / "versions.json").write_text(json.dumps(versions, indent=1) + "\n")
    if with_extension:
        topo["ranks"] = llm.collective_rpc("tp2_probe", timeout=args.rpc_timeout)
    (out / "topology.json").write_text(json.dumps(topo, indent=1, default=str) + "\n")
    print(f"[tp2] engine built in {build_s:.1f}s; executor={topo['executor_class']}; in-process runner: {topo['in_process_runner']}; "
          f"ranks={[(r.get('pid'), (r.get('tp') or {}).get('rank'), (r.get('device') or {}).get('current')) for r in (topo['ranks'] or [])]}", flush=True)
    if args.inventory_out:
        from verity_vllm.engine.meta_inventory import write_live

        inv = write_live(llm, Path(args.inventory_out), timeout=args.rpc_timeout, repo=cp["repo"], revision=cp.get("revision"),
                         tp=args.tp, engine_kwargs=topo["engine_kwargs"])
        print(f"[tp2] live inventory of ranks {sorted(inv['ranks'])} -> {args.inventory_out}", flush=True)

    if not args.no_warmup:
        warm = va.throwaway_requests(eng, wl["requests"])
        _, tw = va.run_requests(eng, warm, wl["sampling"])
        print(f"[tp2] warm-up (uninstrumented, {len(warm)} throwaway requests): {tw['wall_s']:.2f}s, {tw['engine_steps']} steps", flush=True)

    ranks_doc: dict[str, Any] = {}
    if observer:
        steps = parse_steps(args.snapshot_steps, max(r["max_tokens"] for r in wl["requests"]))
        inst = llm.collective_rpc("tp2_install", timeout=args.rpc_timeout, kwargs={
            "out_dir": str(out), "workload": wl, "case": case, "cp": cp, "engine_kwargs": {k: v for k, v in kw.items()},
            "env_flags": env_flags, "snapshot_steps": steps, "snapshot_max_bytes": args.snapshot_max_bytes, "adapter_revision": revision,
            "wheel_sha256": args.wheel_sha256, "vllm_commit": args.vllm_commit, "reenter": not args.no_reenter, "triton": not args.no_triton,
            "module_hooks": not args.no_module_hooks, "weights_verified": bool(args.verify_weights), "argv": sys.argv,
            "target": target})
        ranks_doc["install"] = inst
        print(f"[tp2] observer installed on {len(inst)} ranks: " + "; ".join(f"rank {r['rank']} pid {r['pid']} roots {r['roots']} pid={r['profile_id']}" for r in inst), flush=True)
        if args.param_hashes:
            ranks_doc["param_hashes"] = llm.collective_rpc("tp2_param_hashes", timeout=args.rpc_timeout, kwargs={"out_dir": str(out)})
            print(f"[tp2] param shard hashes: {ranks_doc['param_hashes']}", flush=True)
        if args.record_collectives:
            # [R17 tp D90] the collective recorder of tp2_match's capture arm in the SAME engine run as the observer: every collective of
            # every rank in call order (step, k, kind, module, partial + received sha256; tensors of the first --store-steps steps), the
            # rows the cross-rank leg of the per-rank fold Match (tp2_fold_match / tp.rank_match) keys by the same engine step as the fold.
            # Its bookkeeping is hidden from the observer (tp2_worker.coll_wrap pauses the recorder's aten observer around the copies).
            # (rows land beside the logs -- the layout of tp2_match's capture arm, so `tp2_match --arm check` runs unchanged over this dir)
            ranks_doc["coll_install"] = llm.collective_rpc("tp2_coll_install", timeout=args.rpc_timeout,
                                                           kwargs={"out_dir": str(out), "store_steps": int(args.store_steps)})
            print(f"[tp2] collective recorder installed (same run as the observer): {ranks_doc['coll_install']}", flush=True)
    elif args.census:
        ranks_doc["profiler_start"] = llm.collective_rpc("tp2_profiler_start", timeout=args.rpc_timeout)

    t0 = time.perf_counter()
    tokens, timing = va.run_requests(eng, wl["requests"], wl["sampling"])
    run_s = time.perf_counter() - t0
    print(f"[tp2] {mode} run {run_s:.2f}s, {timing.get('engine_steps')} engine steps", flush=True)

    if observer:
        if args.record_collectives:
            ranks_doc["coll_close"] = llm.collective_rpc("tp2_coll_close", timeout=args.rpc_timeout)
            print(f"[tp2] collective recorder closed: {ranks_doc['coll_close']}", flush=True)
        closed = llm.collective_rpc("tp2_close", timeout=args.rpc_timeout)
        ranks_doc["close"] = closed
        for c in closed:
            print(f"[tp2] rank {c['rank']}: {c['events_written']} events, {c['steps']} steps, snapshot {c['snapshot_bytes']} B, errors {c['n_errors']}", flush=True)
            if c["errors"]:
                print("[tp2]   ERRORS:", json.dumps(c["errors"][:3])[:1500], flush=True)
    elif args.census:
        ranks_doc["profiler_stop"] = llm.collective_rpc("tp2_profiler_stop", timeout=args.rpc_timeout, kwargs={"out_dir": str(out)})
        for c in ranks_doc["profiler_stop"]:
            print(f"[tp2] rank {c['rank']}: {c['n_kernels']} device kernels, {c['distinct']} distinct; collectives {c['collective_kernels']}", flush=True)
    if ranks_doc:
        (out / "ranks.json").write_text(json.dumps(ranks_doc, indent=1, default=str) + "\n")

    doc = {"requests": [], "timing": timing, "observer": observer, "mode": mode, "tp": args.tp, "case": case, "model": cp["repo"],
           "run_s": round(run_s, 3), "engine_kwargs": {k: v for k, v in kw.items() if k not in ("model", "tokenizer")}}
    all_equal = True
    for r in wl["requests"]:
        rid = r["request_id"]
        acc = wl["accepted"].get(rid)
        gen = tokens.get(rid, [])
        eq = (gen == acc) if acc is not None else None
        all_equal = all_equal and bool(eq)
        doc["requests"].append({"request_id": rid, "prompt_token_ids": r["prompt_token_ids"], "max_tokens": r["max_tokens"],
                                "generated_token_ids": gen, "accepted_record_token_ids": acc, "equals_accepted": eq})
        print(f"[tp2] {rid}: generated {gen}\n[tp2]   accepted (TP=1 RTX 4090 record) {acc}  equal={eq}", flush=True)
    doc["all_equal_accepted"] = all_equal
    (out / "tokens.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[tp2] done -> {out}", flush=True)
