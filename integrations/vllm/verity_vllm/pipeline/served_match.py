"""The Match of a served-compiled row on live data.

vLLM serves the workload at its defaults (execution label ``compiled``) with every value of every engine step committed
(`ServedCompiledSource`: graph-safe acquisition, the native collector retaining the values on the host); the served Build is
derived from that process's own compile artifact (`acquire.served`: its Inductor wrappers, vLLM's split graph, the kernels'
launch configs, and each GEMM's cuBLAS route probed on this GPU); and the Build's Program is checked word for word against the
committed values (`check.replay.served_check`), with the weights the engine served.

    verity-vllm served-match --case LLAMA32_1B --workload W.json --out DIR [--build-of-record BUILD.json] [--max-num-seqs 8]

DIR gets match.json (the verdict), build.json (the Build), launch_configs.json, routes.json and facts.json.  With
--build-of-record, the Build derived here must carry its program digest.  Exit 0 when the Match passes, 1 when it fails, 3 when
the Build refuses the artifact.
"""
from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from verity_vllm.config import UsageError, option


@dataclass(frozen=True, kw_only=True)
class Options:
    case: str = option("--case", help="the checkpoint role (verity_vllm.config.CHECKPOINTS)")
    workload: str = option("--workload", help="sweep workload json: requests with prompt_token_ids, max_tokens, arrive_step")
    out: str = option("--out", help="the output directory")
    build_of_record: str | None = option("--build-of-record", default=None, help="a served Build whose program digest this one must equal")
    max_num_seqs: int = option("--max-num-seqs", type=int, default=8)
    gpu_memory_utilization: float = option("--gpu-memory-utilization", type=float, default=0.3)
    kv_cache_gib: int = option("--kv-cache-gib", type=int, default=4, help="the KV cache's size (kv_cache_memory_bytes), GiB")
    rows: int = option("--rows", type=int, default=16, help="the rows of a request a GEMM or attention op is checked on (served_check.ROWS)")


def _words(t: Any) -> Any:
    import numpy as np
    import torch

    nd = {torch.bfloat16: np.uint16, torch.float16: np.uint16, torch.float32: np.uint32}.get(t.dtype)
    t = t.detach().contiguous().cpu()
    if nd is None:
        return t.numpy()
    return t.view({np.uint16: torch.int16, np.uint32: torch.int32}[nd]).numpy().view(nd)


def main(a: Options) -> int:
    os.makedirs(a.out, exist_ok=True)
    out = os.path.abspath(a.out)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{out}/inductor"
    os.environ["VLLM_CACHE_ROOT"] = f"{out}/vllm_cache"
    os.environ["VERITY_RETAIN"] = "host"
    os.environ.setdefault("VERITY_WINDOW_MB", "256")
    os.environ.setdefault("VERITY_WINDOW_SLOTS", "8")
    from verity_vllm.engine import env as engine_env

    os.environ[engine_env.INDUCTOR_DIR_MARK] = "1"
    engine_env.apply_compile_fresh()

    import torch
    from vllm import SamplingParams

    from verity_vllm import config
    from verity_vllm.acquire import compiled as C
    from verity_vllm.acquire import served as SV
    from verity_vllm.acquire.inductor_templates import Unmodelled
    from verity_vllm.acquire.sources import served_compiled_source as S
    from verity_vllm.check.replay import served_check as SC
    from verity_vllm.commit.binding import logical_request
    from verity_vllm.commit.committer.native_collect import NativeCollectCommitter
    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.engine.arrivals import run_requests_arrivals

    t0 = time.perf_counter()
    log = lambda s: print(f"[served-match {time.perf_counter() - t0:7.1f}s] {s}", flush=True)
    wl = json.load(open(a.workload))
    schedule = C.schedule_of_workload(wl)
    steps = max(r["arrive"] + r["max_tokens"] for r in schedule)
    reqs = [SimpleNamespace(request_id=s["request_id"], prompt_token_ids=[int(t) for t in r["prompt_token_ids"]],
                            max_tokens=s["max_tokens"], arrive_step=s["arrive"]) for s, r in zip(schedule, wl["requests"])]
    prompts = {r.request_id: r.prompt_token_ids for r in reqs}

    src = S.ServedCompiledSource()
    src.install_hooks()
    eng = va.build_engine(config.CHECKPOINTS, max_num_seqs=a.max_num_seqs, case=a.case, execution="compiled",
                          engine_args={"gpu_memory_utilization": a.gpu_memory_utilization, "kv_cache_memory_bytes": a.kv_cache_gib << 30})
    src.note_engine(eng.model)
    vc = eng.llm.llm_engine.vllm_config
    cc = vc.compilation_config
    facts = {"case": a.case, "effective": eng.effective, "cudagraph_capture_sizes": list(cc.cudagraph_capture_sizes or []),
             "max_num_seqs": vc.scheduler_config.max_num_seqs, "max_num_batched_tokens": vc.scheduler_config.max_num_batched_tokens,
             "cache_dir": cc.cache_dir, "device": torch.cuda.get_device_name(),
             "sms": torch.cuda.get_device_properties(0).multi_processor_count,
             "versions": {"torch": torch.__version__, "cuda": torch.version.cuda}}
    log(f"engine {a.case}: {json.dumps(eng.effective, default=str)[:400]}")

    com = NativeCollectCommitter(eng, run_id="served-match", arena_bytes=None, wrap=True, gather=True, native_hooks=True,
                                 auto_blocks=True, native_worker=True)
    src.attach(com)
    asz = com.size_arena({"batch": a.max_num_seqs, "max_prompt": max(len(p) for p in prompts.values()),
                          "max_new": max(r.max_tokens for r in reqs), "max_num_batched_tokens": facts["max_num_batched_tokens"],
                          "max_model_len": vc.model_config.max_model_len})
    if com.windowed:
        com.enable_windows(asz)
    com.reset()
    src.begin_run()
    com.install()
    params = lambda r: SamplingParams(temperature=0.0, max_tokens=r.max_tokens, ignore_eos=True, detokenize=False)
    fin, _ = run_requests_arrivals(eng.llm, reqs, params)
    rc = com.finalize()
    facts.update(run_root=bytes(rc.run_root).hex(), tokens={rid: list(o.outputs[0].token_ids) for rid, o in fin.items()},
                 linkage=S.linkage(com), source=src.summary(), errors=(rc.extra or {}).get("errors"))
    log(f"served {len(fin)} requests, {len(com._layouts)} steps committed, root {facts['run_root'][:16]}, linkage {facts['linkage']}")

    configs = src.launch_configs()
    json.dump(configs, open(f"{out}/launch_configs.json", "w"), indent=1)
    graphs = glob.glob(os.path.join(cc.cache_dir, "**", "computation_graph.py"), recursive=True)
    if len(graphs) != 1:
        raise UsageError(f"{len(graphs)} computation_graph.py under {cc.cache_dir}: one split graph is required")
    kw = dict(split_graph=graphs[0], launch_configs=configs, schedule=schedule, steps=steps, capture_sizes=facts["cudagraph_capture_sizes"],
              max_num_seqs=facts["max_num_seqs"], sms=facts["sms"])
    try:
        build = SV.build(os.environ["TORCHINDUCTOR_CACHE_DIR"], routes={}, **kw)
        routes = SV.probe_routes(build["routes_missing"])
        json.dump(routes, open(f"{out}/routes.json", "w"), indent=1)
        build = SV.build(os.environ["TORCHINDUCTOR_CACHE_DIR"], routes=routes, **kw)
    except Unmodelled as e:
        log(f"Build refused: {e}")
        json.dump({"ok": False, "build_refused": str(e), "facts": facts}, open(f"{out}/match.json", "w"), indent=1, default=str)
        return 3
    json.dump(build, open(f"{out}/build.json", "w"), indent=1)
    json.dump(facts, open(f"{out}/facts.json", "w"), indent=1, default=str)
    log(f"Build: program {build['program_digest'][:16]} manifest {build['manifest_digest'][:16]} identities {len(build['identities'])} "
        f"complete {build['complete']}")

    order = {r.request_id: r.request_id for r in reqs}
    named = {**dict(eng.model.named_buffers(remove_duplicate=False)), **dict(eng.model.named_parameters(remove_duplicate=False))}
    cache: dict[str, Any] = {}

    def weight(name: str) -> Any:
        if name not in cache:
            cache[name] = _words(named[name])
        return cache[name]

    check = SC.ServedCheck(build, weight, rows=a.rows, prompts=prompts, log=log)
    for s in sorted(com._layouts):
        values = {m.name: _words(S._committed(com, s, m.name)) for m in com._layouts[s]}
        check.step(SC.Step(s, [(logical_request(rid, order), lo, hi) for rid, lo, hi in com.step_rows.get(s) or []], values))
    com.uninstall()
    rep = check.report()
    of_record = None
    if a.build_of_record:
        want = json.load(open(a.build_of_record))["program_digest"]
        of_record = {"program_digest": want, "equal": want == build["program_digest"]}
    ok = bool(rep["ok"] and build["complete"] and not facts["errors"] and (of_record is None or of_record["equal"]))
    json.dump({"ok": ok, "build": {k: build[k] for k in ("program_digest", "manifest_digest", "complete")}, "build_of_record": of_record,
               "check": rep, "facts": facts}, open(f"{out}/match.json", "w"), indent=1, default=str)
    log(f"Match {'PASS' if ok else 'FAIL'}: tally {json.dumps(rep['tally'])}")
    return 0 if ok else 1
