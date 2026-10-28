"""The Match of a served-compiled row on live data.

vLLM serves the workload at its defaults (execution label ``compiled``), with the row's ``engine.args`` laid over them and its
switches judged as a Commit judges them (`engine.served`: each one away from vLLM's default declared with its ``engine.reasons``),
with every value of every engine step committed
(`ServedCompiledSource`: graph-safe acquisition, the native collector retaining the values on the host); the served Build is
derived from that process's own compile artifact (`acquire.served`: its Inductor wrappers, vLLM's split graph, the kernels'
launch configs, and each GEMM's cuBLAS route probed on this GPU); and the Build's Program is checked word for word against the
committed values (`check.replay.served_check`), with the weights the engine served.

    verity-vllm served-match --case LLAMA32_1B --workload W.json --out DIR [--build-of-record BUILD.json] [--max-num-seqs 8]

DIR gets match.json (the verdict), build.json (the Build, its compile artifact pinned: `acquire.served_artifact`),
launch_configs.json, routes.json and facts.json.  With --build-of-record, the process's fresh Inductor dir is seeded with that
Build's autotune winners and the Build derived here must carry its program digest.  Exit 0 when the Match passes, 1 when it fails,
3 when the Build refuses the artifact, the device or the served switches.

The process's own compile dirs are under DIR (`engine.env.own_compile_dirs`); the committer retains on the host
(``VERITY_RETAIN=host``, which the run exports).

The Commit's served mode (`pipeline.commit --compiled-taps served`, `ops/served_commit.sh`) runs the same derivation and check in
its treatment process through `ServedCommit`, on the Build of record's pinned artifact: its Inductor dir seeded with the Build's
winners, and its own Build's artifact the Build of record's or the Commit refuses (rc 3) before it serves a measured run.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

from verity_vllm.config import UsageError, option, workload_arg
from verity_vllm.engine import env as engine_env
from verity_vllm.query.manifest.format import SERVED_IDS


@dataclass(frozen=True, kw_only=True)
class Options:
    case: str = option("--case", help="the checkpoint role (verity_vllm.config.CHECKPOINTS)")
    workload: str = option("--workload", type=workload_arg, help="sweep workload json: requests with prompt_token_ids, max_tokens, arrive_step")
    out: str = option("--out", help="the output directory")
    build_of_record: str | None = option("--build-of-record", default=None, help="a served Build whose autotune winners seed this process "
                                         "and whose program digest and compile artifact this one must equal")
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


def schedule_of(wl: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """The workload's request schedule (`acquire.compiled.schedule_of_workload`) and the engine steps it spans."""
    from verity_vllm.acquire import compiled as C

    schedule = C.schedule_of_workload(wl)
    return schedule, max(r["arrive"] + r["max_tokens"] for r in schedule)


def facts_of(eng: Any) -> dict[str, Any]:
    """The serving engine's dispatch facts the Build reads (capture sizes, max_num_seqs, max_model_len, the GPU's compute capability
    and multiprocessor count, the attention kernels its layers selected, the fused-MoE layers)."""
    import torch
    from vllm.model_executor.layers.attention.attention import Attention

    vc = eng.llm.llm_engine.vllm_config
    cc = vc.compilation_config
    attention = sorted({(m.get_attn_backend().get_name(), getattr(m.impl, "vllm_flash_attn_version", None))
                        for m in eng.model.modules() if isinstance(m, Attention)}, key=str)
    return {"case": eng.case, "effective": eng.effective, "cudagraph_capture_sizes": list(cc.cudagraph_capture_sizes or []),
            "max_num_seqs": vc.scheduler_config.max_num_seqs, "max_num_batched_tokens": vc.scheduler_config.max_num_batched_tokens,
            "max_model_len": vc.model_config.max_model_len, "cache_dir": cc.cache_dir, "device": torch.cuda.get_device_name(),
            "cc": list(torch.cuda.get_device_capability()), "sms": torch.cuda.get_device_properties(0).multi_processor_count,
            "attention": [{"backend": b, "fa_version": v} for b, v in attention],
            "versions": {"torch": torch.__version__, "cuda": torch.version.cuda}, "moe": moe_facts(eng.model)}


def device_of(facts: dict[str, Any]) -> Any:
    """The served Build's device (`served_device.device_of`) from the engine's facts; `Unmodelled` names every binding it lacks."""
    from verity_vllm.acquire import served_device as SD

    return SD.device_of(facts["cc"], facts["sms"], attention=facts["attention"], max_model_len=facts["max_model_len"])


def moe_facts(model: Any) -> dict[str, dict[str, Any]]:
    """Each fused-MoE layer by its layer name, which is all the opaque `vllm.moe_forward` op the compiled graph calls names: the served
    names (the checkpoint's, the weights of record's) and shapes of its gate, w13 and w2 weights, its router's top_k, renormalize and values per thread, and what the expansion of
    the op into Calls (`acquire.served_moe`) requires of the block, each a flag: the runner holds the gate and does not fuse it with a
    shared-expert gate, no shared experts and no routed scaling, the softmax top-k router, modular Triton experts without bias or
    clamp, SiLU, bf16 weights.  A flag it cannot read is None (refused)."""
    import torch
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    runners = [m for _n, m in model.named_modules() if isinstance(m, MoERunner)]
    if not runners:
        return {}
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter

    from verity_vllm.observe.fold.profiles.generic import router_vpt_bf16

    def flag(test: Callable[[], Any]) -> bool | None:
        try:
            return bool(test())
        except AttributeError:
            return None

    # each tensor under its first registration, the name the loader fills from the checkpoint: the runner re-registers the block's
    # gate as its own `gate`
    named = {id(p): n for n, p in model.named_parameters()}
    out: dict[str, dict[str, Any]] = {}
    for m in runners:
        ex, r, gate = m.routed_experts, m.router, getattr(m, "gate", None)
        experts = getattr(getattr(ex.quant_method, "moe_kernel", None), "fused_experts", None)
        w = {"gate": getattr(gate, "weight", None), "w13": ex.w13_weight, "w2": ex.w2_weight}
        E = int(ex.w13_weight.shape[0])
        out[str(m.layer_name)] = {
            **{k: {"weight": named.get(id(t)), "shape": list(t.shape)} if t is not None else None for k, t in w.items()},
            "E": E, "top_k": int(r.top_k), "renormalize": flag(lambda: r.renormalize), "VPT": router_vpt_bf16(E),
            "gate_held": gate is not None, "fused_gate": flag(lambda: m._fse_fuse_gate),
            "shared_experts": flag(lambda: m.shared_experts is not None), "routed_scaling": flag(lambda: m.routed_scaling_factor != 1.0),
            "softmax_router": flag(lambda: type(r) is FusedTopKRouter and r.scoring_func == "softmax"),
            "triton_experts": flag(lambda: type(experts) is TritonExperts and not ex.quant_method.is_monolithic),
            "bias": flag(lambda: experts.w1_bias is not None or experts.w2_bias is not None),
            "clamp": flag(lambda: experts.activation_config.clamp_limit is not None),
            "silu": flag(lambda: m.activation == MoEActivation.SILU),
            "bf16": all(t is not None and t.dtype == torch.bfloat16 for t in w.values())}
    return out


def build_here(src: Any, facts: dict[str, Any], schedule: list[dict[str, Any]], steps: int, out: str) -> dict[str, Any]:
    """The served Build of this process's compile artifact, its GEMMs' cuBLAS routes probed on this GPU; launch_configs.json,
    routes.json and, with fused-MoE layers, moe.json (the process's MoE record, `served_moe.expand`) go to `out`.  Raises
    `Unmodelled` when the Build refuses the artifact."""
    from verity_vllm.acquire import served as SV

    configs = src.launch_configs()
    json.dump(configs, open(f"{out}/launch_configs.json", "w"), indent=1)
    graphs = glob.glob(os.path.join(facts["cache_dir"], "**", "computation_graph.py"), recursive=True)
    if len(graphs) != 1:
        raise UsageError(f"{len(graphs)} computation_graph.py under {facts['cache_dir']}: one split graph is required")
    dirs = engine_env.compile_dirs()
    if dirs is None:
        raise UsageError("this process was not given its own compile dirs (engine.env.own_compile_dirs, ops/served_commit.sh)")
    kw = dict(split_graph=graphs[0], launch_configs=configs, schedule=schedule, steps=steps, capture_sizes=facts["cudagraph_capture_sizes"],
              max_num_seqs=facts["max_num_seqs"], device=device_of(facts))
    if facts.get("moe"):
        kw["moe"] = {"layers": facts["moe"], "pieces": dict(src.moe_pieces), "configs": src.moe_configs, "cc": facts["cc"]}
        json.dump(kw["moe"], open(f"{out}/moe.json", "w"), indent=1)
    build = SV.build(dirs["inductor"], routes={}, **kw)
    routes = SV.probe_routes(build["routes_missing"])
    json.dump(routes, open(f"{out}/routes.json", "w"), indent=1)
    return SV.build(dirs["inductor"], routes=routes, **kw)


class _Experts:
    """A stacked expert weight [E, N, K] whose words are read one expert at a time (`slab[e]`): the whole stack of every layer is
    more than the host holds."""

    def __init__(self, t: Any) -> None:
        self.t, self.shape = t, tuple(t.shape)

    def __getitem__(self, e: int) -> Any:
        return _words(self.t[int(e)])


def served_weights(model: Any) -> Callable[[str], Any]:
    """A served weight's words by its served name (buffers and parameters, tied ones under each name), read once; a stacked expert
    weight (3-d) per expert on each use (`_Experts`)."""
    named = {**dict(model.named_buffers(remove_duplicate=False)), **dict(model.named_parameters(remove_duplicate=False))}
    cache: dict[str, Any] = {}

    def weight(name: str) -> Any:
        if named[name].dim() == 3:
            return _Experts(named[name])
        if name not in cache:
            cache[name] = _words(named[name])
        return cache[name]

    return weight


def check_committed(build: dict[str, Any], com: Any, weight: Callable[[str], Any], prompts: dict[str, list[int]], order: dict[str, str], *,
                    seed: int | None, rows: int | None = None, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """`served_check` of the Build's Program over every step `com` committed (its retained values), requests by `order`'s logical ids."""
    from verity_vllm.acquire.sources import served_compiled_source as S
    from verity_vllm.check.replay import served_check as SC
    from verity_vllm.commit.binding import logical_request

    check = SC.ServedCheck(build, weight, rows=SC.ROWS if rows is None else rows, prompts=prompts, log=log, seed=seed)
    for s in sorted(com._layouts):
        values = {m.name: _words(S._committed(com, s, m.name)) for m in com._layouts[s]}
        check.step(SC.Step(s, [(logical_request(rid, order), lo, hi) for rid, lo, hi in com.step_rows.get(s) or []], values))
    return check.report()


class ServedCommit:
    """The served row in a Commit's treatment process (`pipeline.commit --compiled-taps served`): the source's hooks before vLLM
    compiles, the Build of record's value modules once the engine is up, this process's own Build after the warm-ups
    (build_rebuilt.json, which `compiled verify` holds the Build of record to), and each run's C2: the Build's Program on the
    committed values, each request's rows a sample of the seed of a challenge chosen after the run root is fixed (the Commit is its
    run's own auditor), and the linkage that is the check's input rule (the prompt at a request's first step, then its previous
    step's sampled id)."""

    def __init__(self, build: dict[str, Any], fault: str | None) -> None:
        from verity_vllm.acquire.sources.served_compiled_source import ServedCompiledSource

        from verity_vllm.acquire import served_artifact as SA

        if engine_env.compile_dirs() is None:
            raise UsageError("--compiled-taps served: the Build reads this process's own compile artifact; export its own "
                             f"TORCHINDUCTOR_CACHE_DIR and VLLM_CACHE_ROOT with {engine_env.INDUCTOR_DIR_MARK}=1 (ops/served_commit.sh)")
        self.pin = SA.pinned(build)
        engine_env.apply_compile_fresh()
        self.build, self.facts, self.weight = build, {}, None
        self.src = ServedCompiledSource(fault=fault)
        self.src.install_hooks()

    def engine(self, eng: Any) -> dict[str, Any]:
        """After the engine is built: what versions.json records of the source and the engine."""
        self.src.note_engine(eng.model)
        self.src.use_build(self.build)
        self.facts, self.weight = facts_of(eng), served_weights(eng.model)
        if self.src.errors or not self.src.slots:
            raise UsageError(f"--compiled-taps served: no captured slots / errors {self.src.errors}")
        return {"compiled_source": self.src.summary(), "served_facts": self.facts}

    def attach(self, com: Any) -> None:
        if not hasattr(com, "_nc") or not hasattr(com, "attach_source"):
            raise UsageError(f"--compiled-taps served needs the native_collect committer (got {com.name})")
        self.src.attach(com)

    def rebuild(self, workload: dict[str, Any], out: str) -> dict[str, Any]:
        """This process's own Build (build_rebuilt.json); exits 3 (`[delta] REFUSE`) when the compile artifact it read is not the
        Build of record's (artifact_check.json names each kernel and winner that differs; a refused Build read no kernels, and
        its process's winners are checked)."""
        from verity_vllm.acquire import served_artifact as SA
        from verity_vllm.acquire.inductor_templates import Unmodelled

        schedule, steps = schedule_of(workload)
        try:
            b = build_here(self.src, self.facts, schedule, steps, out)
        except Unmodelled as e:
            b = {"program_digest": None, "manifest_digest": None, "complete": False, "unmodelled": {"build_refused": str(e)},
                 "populations": {"identities": 0}, "identities": []}
        json.dump(b, open(f"{out}/build_rebuilt.json", "w"), indent=1)
        loaded = b.get("artifact") or {"winners": SA.winners(engine_env.compile_dirs()["inductor"])}
        diff = SA.differences(self.pin, loaded, SA.PARTS if "artifact" in b else ("winners",))
        json.dump({"pinned": self.pin["digest"], "loaded": loaded.get("digest"), "equal": not diff, "differences": diff},
                  open(f"{out}/artifact_check.json", "w"), indent=1)
        if diff:
            print(f"[delta] REFUSE: this process loaded compile artifact {str(loaded.get('digest', 'of a refused Build'))[:16]}, not the Build of record's "
                  f"{self.pin['digest'][:16]} ({len(diff)} differences: {diff[:3]})", flush=True)
            sys.exit(3)
        print(f"[delta] compile artifact {self.pin['digest'][:16]}: this process's is the Build of record's", flush=True)
        return b

    def c2(self, com: Any, rc: Any, reqs: list[dict[str, Any]], challenge: str, replay_seed: int | None) -> dict[str, Any]:
        """One run's C2: ``components`` (sampled_replay, boundary_linkage), the ``value_correspondence`` fields of record
        (``compiled_check`` among them), ``ok`` and a log ``line``."""
        from verity_vllm.acquire.sources import served_compiled_source as S
        from verity_vllm.check.replay import served_check as SC
        from verity_vllm.pipeline import c2_replay as C2

        seed, seed_src, seed_form, chal = C2.seed_of(SimpleNamespace(replay_seed=replay_seed, replay_challenge=challenge), rc)
        order = {str(q["request_id"]): f"r{i}" for i, q in enumerate(reqs)}
        prompts = {f"r{i}": [int(t) for t in q["prompt_token_ids"]] for i, q in enumerate(reqs)}
        try:
            rep = check_committed(self.build, com, self.weight, prompts, order, seed=seed)
        except Exception as e:  # noqa: BLE001  -- recorded: the run's C2 fails with the error
            rep = {"ok": False, "error": repr(e)[:300], "traceback": traceback.format_exc()[-1200:], "tally": {}, "steps": {}, "failures": []}
        n, exact = sum(t[1] for t in rep["tally"].values()), sum(t[0] for t in rep["tally"].values())
        ok = bool(rep.get("ok")) and not rep.get("error") and not self.src.errors
        pairs, unlinked, horizon, prompt_linked = 0, [], {}, {}
        for p in sorted(self.build["steps_plan"], key=lambda p: int(p["step"])):
            s = int(p["step"])
            ids_ok = bool(((rep["steps"].get(str(s)) or {}).get(SERVED_IDS) or {}).get("ok"))
            for rq in p["requests"]:
                pairs += 1
                horizon.setdefault(rq["request_id"], [s, s])[1] = s
                prompt_linked.setdefault(rq["request_id"], ids_ok)
                if not ids_ok:
                    unlinked.append({"step": s, "request": rq["request_id"]})
        chain = S.linkage(com)
        linked_ok = bool(pairs and not unlinked and chain["passed"])
        pop = self.build["populations"]
        # one process serves the row (build_here refuses a second split graph): the check reads rank 0's committed values
        sampled = {"result": ok, "seed": seed, "seed_source": seed_src, "seed_form": seed_form,
                   "sample": {"seed": seed, "seed_source": seed_src, "seed_form": seed_form, "challenge": chal,
                              "rows_per_request": rep.get("rows_per_request"), "picks": rep.get("picks")},
                   "population": {"n": pop["identities"], "digest": self.build["manifest_digest"], "not_evaluable": {},
                                  **SC.population(self.build, rep, ranks=[0])},
                   "n": n, "exact": exact, "tally": rep.get("tally"), "steps_checked": rep.get("steps_checked"),
                   "steps_unrun": rep.get("steps_unrun"), "mismatch": rep.get("failures"), "errors": self.src.errors,
                   "why": rep.get("error") or ("every value replays word for word from the committed values" if ok else "see mismatch/errors"),
                   "method": "sampled-exact-replay: the served Build's Program (acquire.served) on the committed values word for word "
                             "(served_check.py): every call of every piece on every row; each GEMM and attention op on a seeded draw of "
                             "each request's rows; the logits and the sampled ids of every request"}
        linkage = {"result": linked_ok, "pairs": pairs, "unlinked_n": len(unlinked), "linked": pairs - len(unlinked), "unlinked": unlinked[:16],
                   "horizon": horizon, "prompt_linked": prompt_linked, "embed_op_path": SERVED_IDS, "committed_chain": chain,
                   "method": "per (engine step, request): the committed ids are the declared prompt at the request's first step and its "
                             "previous step's sampled id at every later step (served_check inputs); committed sampled(s) == committed ids(s+1)",
                   "why": None if linked_ok else f"{len(unlinked)} (step, request) boundaries unlinked; committed chain {chain}"}
        check = {"ok": ok, "checked": n, "equal": exact, "launches_checked": n, "launches_exact": exact, "error": rep.get("error"),
                 "seed": seed, "tally": rep.get("tally"), "failures": rep.get("failures"), "steps_checked": rep.get("steps_checked"),
                 "steps_unrun": rep.get("steps_unrun")}
        line = (f"{'OK' if ok else 'FAIL'} {exact}/{n} values exact over steps {rep.get('steps_checked')} (unrun {rep.get('steps_unrun')}), "
                f"seed {seed_form} {seed}; linkage {pairs - len(unlinked)}/{pairs}, committed chain {chain}; source stats "
                f"{self.src.summary()['stats']}; first failures {rep.get('failures', [])[:2]} {rep.get('error') or ''}"
                + (f"\n{rep['traceback']}" if rep.get("traceback") else ""))
        return {"components": {"sampled_replay": sampled, "boundary_linkage": linkage}, "ok": ok and linked_ok, "line": line,
                "value_correspondence": {"population": pop, "n": n, "seed": seed, "compiled_check": check}}


def main(a: Options) -> int:
    os.makedirs(a.out, exist_ok=True)
    out = os.path.abspath(a.out)
    dirs = engine_env.own_compile_dirs(out)
    from verity_vllm.acquire import served_artifact as SA

    of_record = json.load(open(a.build_of_record)) if a.build_of_record else None
    if of_record is not None:
        SA.seed(SA.pinned(of_record), dirs["TORCHINDUCTOR_CACHE_DIR"])

    from vllm import SamplingParams

    from verity_vllm import config
    from verity_vllm.acquire.inductor_templates import Unmodelled
    from verity_vllm.acquire.sources import served_compiled_source as S
    from verity_vllm.commit.committer.native_collect import NativeCollectCommitter
    from verity_vllm.engine import served as SW
    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.engine.arrivals import run_requests_arrivals

    t0 = time.perf_counter()
    log = lambda s: print(f"[served-match {time.perf_counter() - t0:7.1f}s] {s}", flush=True)
    wl = json.load(open(a.workload))
    schedule, steps = schedule_of(wl)
    reqs = [SimpleNamespace(request_id=s["request_id"], prompt_token_ids=[int(t) for t in r["prompt_token_ids"]],
                            max_tokens=s["max_tokens"], arrive_step=s["arrive"]) for s, r in zip(schedule, wl["requests"])]
    prompts = {r.request_id: r.prompt_token_ids for r in reqs}

    src = S.ServedCompiledSource()
    src.install_hooks()
    eng = va.build_engine(config.CHECKPOINTS, max_num_seqs=a.max_num_seqs, case=a.case, execution="compiled",
                          engine_args={**config.RowEngine.of(wl).args, "gpu_memory_utilization": a.gpu_memory_utilization,
                                       "kv_cache_memory_bytes": a.kv_cache_gib << 30})
    src.note_engine(eng.model)
    facts = facts_of(eng)
    facts["served"] = SW.record(list(eng.llm.collective_rpc(SW.read_rank)), SW.declared({**wl, "execution": "compiled"}))
    log(f"engine {a.case}: {json.dumps(eng.effective, default=str)[:400]}")
    try:
        if not facts["served"]["ok"]:
            raise Unmodelled("served switches: " + "; ".join(facts["served"]["problems"]))
        device_of(facts)
    except Unmodelled as e:
        log(f"Build refused: {e}")
        json.dump({"ok": False, "build_refused": str(e), "facts": facts}, open(f"{out}/match.json", "w"), indent=1, default=str)
        return 3

    com = NativeCollectCommitter(eng, run_id="served-match", arena_bytes=None, wrap=True, gather=True, native_hooks=True,
                                 auto_blocks=True, native_worker=True)
    if com.retain != "host":
        raise UsageError(f"VERITY_RETAIN={com.retain}: the check reads the committed values the collector retains on the host (export VERITY_RETAIN=host)")
    src.attach(com)
    asz = com.size_arena({"batch": a.max_num_seqs, "max_prompt": max(len(p) for p in prompts.values()),
                          "max_new": max(r.max_tokens for r in reqs), "max_num_batched_tokens": facts["max_num_batched_tokens"],
                          "max_model_len": facts["max_model_len"]})
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

    try:
        build = build_here(src, facts, schedule, steps, out)
    except Unmodelled as e:
        log(f"Build refused: {e}")
        json.dump({"ok": False, "build_refused": str(e), "facts": facts}, open(f"{out}/match.json", "w"), indent=1, default=str)
        return 3
    json.dump(build, open(f"{out}/build.json", "w"), indent=1)
    json.dump(facts, open(f"{out}/facts.json", "w"), indent=1, default=str)
    log(f"Build: program {build['program_digest'][:16]} manifest {build['manifest_digest'][:16]} identities {len(build['identities'])} "
        f"complete {build['complete']}")

    rep = check_committed(build, com, served_weights(eng.model), prompts, {r.request_id: r.request_id for r in reqs}, seed=None, rows=a.rows, log=log)
    com.uninstall()
    if of_record is not None:
        diff = SA.differences(SA.pinned(of_record), build["artifact"])
        of_record = {"program_digest": of_record["program_digest"], "artifact": of_record["artifact"]["digest"], "artifact_differences": diff[:20],
                     "equal": of_record["program_digest"] == build["program_digest"] and not diff}
    ok = bool(rep["ok"] and build["complete"] and not facts["errors"] and (of_record is None or of_record["equal"]))
    json.dump({"ok": ok, "build": {**{k: build[k] for k in ("program_digest", "manifest_digest", "complete")}, "artifact": build["artifact"]["digest"]},
               "build_of_record": of_record,
               "check": rep, "facts": facts}, open(f"{out}/match.json", "w"), indent=1, default=str)
    log(f"Match {'PASS' if ok else 'FAIL'}: tally {json.dumps(rep['tally'])}")
    return 0 if ok else 1
