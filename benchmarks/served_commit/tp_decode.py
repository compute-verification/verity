"""Commit under a TP engine in eager mode (GLM-5.3 TP8): plain served vLLM (`stock`) against one Commit arm in the same engine,
run by run interleaved, batch by batch (B1, B8, B32: a Commit run follows the plans only of the committed run before it,
`native_plans.h`, so each batch's warm-up learns them right before its pairs), each rank's committer made as `tp-commit` makes it
(`pipeline.tp.rank_worker.tp2_commit_make`, the native collector on windowed staging) and its root read as the rank's: no root
combination, openings or verdict. Measurement only.

    bash benchmarks/served_commit/tp_decode.sh      (ARM, RETAIN, BATCHES, PAIRS; its header has the rest)

One engine process per arm (``--arm``), since the row size is a module constant set on each rank before its committer is built
(`prof_arm.Rows4kArm`):

- ``rows``: rows-sha512's step at 256 B rows (whose roots are no scheme's), with
  `tp-commit`'s acquisition: the acquired classes' modules, the MoE block
  stream, every TP collective's rank-local partial, the capture map's routes and attention taps, and the FP8 linears'
  per-scale-group input quantization (`acquisition_manifest` names its family; the vocabulary-range tap, one small tensor a step, is off);
- ``rows-layer``: rows-sha512 acquiring only what a per-layer proof block commits (`per_layer`): each decoder layer's residual stream
  as its fused add-RMSNorm leaves it (`LayerStreamSource`), the MLA cache entry and the DSA indexer's cache row and scale each
  attention layer writes, the embedding's ids and output, the final norm, the logits and the sampled ids;
- ``rows-4k`` and ``rows-layer-4k``: the same at 4 KiB rows, rows-sha512's row size.

A batch's requests are the workload's request with its prompt rotated by ``--rotate`` tokens per request, at the workload's
prompt length
and cap, all arriving at once; every request decodes to its cap (``ignore_eos``), so a batch's runs have the same steps. Per run,
`runs.jsonl` has the step walls and outputs, and `phases` reads TTFT (the mean over the requests, and the time to the batch's last
first token) and the decode TPOT (the mean step wall once every request decodes) from them. A Commit run passes its gate when its
tokens equal the batch's first control run's, every rank has a 64-byte root, no collector error and no step that bounded staging
left without a root, and, on the arms with `tp-commit`'s acquisition, every rank's identity check passed. `summary.json` has the
committer's value retention as it took it (``retain``: ``RETAIN``, none by default) and, per batch, each arm's median TPOT and TTFT
and the paired ratios.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

#: arm -> (row bytes, per-layer acquisition)
ARMS = {"rows": (256, False), "rows-4k": (4096, False), "rows-layer": (256, True), "rows-layer-4k": (4096, True)}
#: the modules a per-layer proof block keeps of the committer's selection: the embedding (its ids and output), the final norm,
#: the logits, and the sampler where the runner has one as a module (the sampled ids come from the committer's `runner.sample` wrap either way)
UNIT_MODULES = ("model.embed_tokens", "model.norm", "logits_processor", "runner.sampler")
UNIT_REQUIRED = UNIT_MODULES[:3]
ROOT_BYTES = hashlib.sha512().digest_size


def requests_at(wl: dict[str, Any], batch: int, rotate: int) -> list[dict[str, Any]]:
    """``batch`` requests from the workload's first: its prompt rotated by ``rotate`` tokens per request, its cap, all at step 0."""
    base = wl["requests"][0]
    ids = [int(t) for t in base["prompt_token_ids"]]
    keep = {k: base[k] for k in ("max_tokens", "temperature", "top_p", "top_k", "seed") if k in base}
    return [{**keep, "request_id": f"b{batch}-r{i}", "index": i, "arrive_step": 0,
             "prompt_token_ids": ids[(rotate * i) % len(ids):] + ids[:(rotate * i) % len(ids)]} for i in range(batch)]


def phases(walls: list[float], outs: list[int], batch: int) -> dict[str, Any]:
    """One run's prefill and decode from its step walls and per-step outputs (`pipeline.commit.run_workload`). A request's first
    token is at the first step it has an output at: TTFT is the mean over the requests of the run's time at that step, and
    ``prefill_s`` the time at which the last request has its first. The decode steps are the later steps at which every request
    has an output; TPOT is their mean wall."""
    t, started, ttft, prefill, decode = 0.0, 0, 0.0, None, []
    for w, n in zip(walls, outs):
        t += w
        if started == batch:
            if n == batch:
                decode.append(w)
            continue
        new = max(0, min(n, batch) - started)
        ttft += new * t
        started += new
        if started == batch:
            prefill = t
    return {"ttft_mean_s": ttft / batch if started == batch else None, "prefill_s": prefill, "decode_steps": len(decode),
            "tpot_s": statistics.fmean(decode) if decode else None, "tpot_median_s": statistics.median(decode) if decode else None}


def acquisition_manifest(world: int) -> dict[str, Any]:
    """The rows each rank's taps read (`acquire.sources.taps.attach_rank`): a `DeepGemmFp8Linear_v1` Call per rank, so every rank
    attaches the FP8 linears' per-scale-group input quantization as a GLM-5.3 Commit's manifest has it do. No Program, no population."""
    return {"ranks": world, "identities": [{"rank": r, "spec": "DeepGemmFp8Linear_v1{}"} for r in range(world)]}


class LayerStreamSource:
    """Each decoder layer's residual stream as the layer's fused add-RMSNorm leaves it, `input_layernorm`'s second output
    (`<layer>.input_layernorm/1`; the first layer's norm returns one tensor, and its stream is the embedding's output), flushed
    at once: the layer's `post_attention_layernorm` adds into it in place."""

    short_name = "layer_stream"

    def __init__(self, com: Any, model: Any) -> None:
        from verity_vllm.engine.hooks import Hooks

        self.c = com
        self.norms = [(n, m) for n, m in model.named_modules() if n.endswith(".input_layernorm")]
        self._hooks = Hooks()
        self.stats: dict[str, Any] = {"layers": len(self.norms), "bound": 0}

    def install(self) -> None:
        for name, m in self.norms:
            self._hooks.wrap(m, "forward", self._stream(f"{name}/1"))

    def uninstall(self) -> None:
        self._hooks.uninstall()

    def on_step_begin(self) -> None:
        pass

    def begin_run(self) -> None:
        self.stats["bound"] = 0

    def declaration_classes(self) -> list[dict]:
        return []

    def _stream(self, name: str) -> Any:
        me = self

        def make(orig: Any) -> Any:
            def forward(*a: Any, **kw: Any) -> Any:
                out = orig(*a, **kw)
                job = me.c._cur
                if job is not None and isinstance(out, tuple) and len(out) > 1:
                    me.c._acquire(name, out[1], job)
                    me.c._nc.flush()
                    me.stats["bound"] += 1
                return out
            return forward
        return make


def per_layer(com: Any, model: Any) -> dict[str, Any]:
    """Keep of `com`'s acquisition what a per-layer proof block commits: of its modules `UNIT_MODULES`, of its sources the attention
    source with only the cache rows each attention layer writes, and `LayerStreamSource`."""
    from verity_vllm.acquire.sources import attention_source as AS

    missing = sorted(set(UNIT_REQUIRED) - {n for n, _m in com._modules})
    attn = [s for s in com.sources if isinstance(s, AS.AttentionSource)]
    if missing or not attn:
        raise RuntimeError(f"per-layer arm: the committer selects no {missing} or has no attention source ({len(attn)})")
    com._modules = [(n, m) for n, m in com._modules if n in UNIT_MODULES]
    for s in attn:
        s.names = {t: v for t, v in s.names.items() if t in (AS.MLA_ROWS, AS.INDEX_ROWS, AS.INDEX_SCALE)}
    stream = LayerStreamSource(com, model)
    com.sources = [*attn, stream]
    com.name = "+".join([com.name.split("+", 1)[0], *([com.commit_scheme] if f"+{com.commit_scheme}+" in f"{com.name}+" else []),
                         "layer_stream", "attention_rows"])
    return {"modules": [n for n, _m in com._modules], "taps": {t: len(v) for s in attn for t, v in s.names.items()}, "layers": len(stream.norms)}


def make(worker: Any, *, arm: str, **kw: Any) -> dict[str, Any]:
    """On a rank (`LLM.collective_rpc` with a callable): the arm's committer as `tp-commit` makes it (`tp2_commit_make`), under
    rows-sha512 at the arm's row size; on a per-layer arm, then only the per-layer proof block's acquisition (`per_layer`)."""
    from verity_vllm.acquire.sources import taps as TAPS
    from verity_vllm.commit import step_rows
    from verity_vllm.commit.committer import host_retention, native_collect, native_host

    row_bytes, layer = ARMS[arm]
    step_rows.ROW_BYTES = native_host.GPU_CHUNK_BYTES = native_collect.GPU_CHUNK_BYTES = host_retention.GPU_CHUNK_BYTES = row_bytes
    res = worker.tp2_commit_make(kind="native_collect_v2b", commit_scheme=step_rows.NAME, tp_partials=not layer,
                                 tap_sources=TAPS.for_ranks(TAPS.TapSources()), **kw)
    st = worker._tp2_commit
    if layer:
        res["per_layer"] = per_layer(st["com"], st["eng"].model)
    res.update(committer=st["com"].name, commit_scheme=st["com"].commit_scheme, row_bytes=step_rows.ROW_BYTES, retain=st["com"].retain,
               sources=[getattr(s, "short_name", type(s).__name__) for s in st["com"].sources], modules=len(st["com"]._modules))
    return res


def finalize(worker: Any) -> dict[str, Any]:
    """On a rank: the run's commitment (the committer's finalize, its hooks off), and what the gate reads of it."""
    from verity_vllm.pipeline.commit import assert_pristine

    st = worker._tp2_commit
    com = st["com"]
    t0 = time.perf_counter()
    rc = st["rc"] = com.finalize()
    fin_s = time.perf_counter() - t0
    com.uninstall()
    assert_pristine(st["eng"], st["base"])
    ex = rc.extra or {}
    ic, bs = ex.get("identity_check") or {}, ex.get("bounded_staging") or {}
    return {"rank": st["rank"], "run_root": bytes(rc.run_root).hex(), "steps": len(rc.steps), "tensors": sum(s.n_tensors for s in rc.steps),
            "bytes_bound": sum(s.n_bytes for s in rc.steps), "leaves": sum(s.n_leaves for s in rc.steps),
            "errors_n": ex.get("errors_n", len(ex.get("errors") or [])), "errors": [str(e)[:200] for e in (ex.get("errors") or [])[:3]],
            "identity_ok": ic.get("ok"), "identity_violations": [str(v)[:200] for v in (ic.get("violations") or [])[:2]],
            "learned_only_steps": len(bs.get("learned_only_steps") or ()), "plan_mismatches": bs.get("plan_mismatches"),
            "root_ready_delay_s": rc.root_ready_delay_s, "finalize_s": round(fin_s, 3)}


def gate(tokens_equal: bool, fin: list[dict[str, Any]], world: int, identity: bool) -> dict[str, Any]:
    why = [] if tokens_equal else ["tokens differ from the batch's first control run"]
    if sorted(int(f["rank"]) for f in fin) != list(range(world)):
        why.append(f"ranks {sorted(int(f['rank']) for f in fin)} finalized of {world}")
    for f in sorted(fin, key=lambda f: int(f["rank"])):
        r = f["rank"]
        if len(bytes.fromhex(f["run_root"])) != ROOT_BYTES:
            why.append(f"rank {r}: a root of {len(f['run_root']) // 2} bytes")
        if f["errors_n"]:
            why.append(f"rank {r}: {f['errors_n']} collector errors {f['errors'][:1]}")
        if f["learned_only_steps"]:
            why.append(f"rank {r}: {f['learned_only_steps']} steps without a root")
        if identity and f["identity_ok"] is not True:
            why.append(f"rank {r}: identity check {f['identity_ok']} {f['identity_violations'][:1]}")
    return {"passed": not why, "why": why[:8]}


def summarize(rows: list[dict[str, Any]], arm: str, batches: list[int]) -> dict[str, Any]:
    """Per batch: each arm's median TPOT, TTFT and prefill over its runs, and the arm's paired ratios over control (median, min, max)."""
    out: dict[str, Any] = {}
    for b in batches:
        by = {a: {r["pair"]: r["phases"] for r in rows if r["batch"] == b and r["arm"] == a} for a in ("control", arm)}
        pairs = sorted(set(by["control"]) & set(by[arm]))
        doc: dict[str, Any] = {}
        for a in ("control", arm):
            doc[a] = {k: statistics.median(p[k] for p in by[a].values()) if by[a] and all(p[k] is not None for p in by[a].values()) else None
                      for k in ("tpot_s", "ttft_mean_s", "prefill_s")}
        for k in ("tpot_s", "ttft_mean_s", "prefill_s"):
            q = [by[arm][p][k] / by["control"][p][k] for p in pairs if by[arm][p][k] and by["control"][p][k]]
            doc[f"{k[:-2]}_ratio"] = {"median": statistics.median(q), "min": min(q), "max": max(q), "pairs": len(q)} if q else None
        ms = [r for r in rows if r["batch"] == b and r["arm"] == arm]
        doc["gates_passed"] = bool(ms) and all(r["gate"]["passed"] for r in ms)
        doc["control_tokens_equal"] = all(r["tokens_equal_ref"] for r in rows if r["batch"] == b and r["arm"] == "control")
        doc["decode_steps"] = sorted({p["decode_steps"] for a in by for p in by[a].values()})
        out[str(b)] = doc
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="stock against one rows-sha512 Commit arm in one TP engine (eager), B1/B8/B32 interleaved")
    ap.add_argument("--workload", required=True, help="the row (its first request is every batch's prompt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--case", default=None)
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--rotate", type=int, default=8)
    ap.add_argument("--engine-arg", action="append", default=[])
    ap.add_argument("--rpc-timeout", type=float, default=3600.0)
    a = ap.parse_args()

    from verity_vllm.engine import engine_profile as prof
    prof.apply_env()
    from vllm import LLM

    from verity_vllm import protocols as PO
    from verity_vllm.engine import served as SV
    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.pipeline.tp.rank_worker import EXTENSION
    from verity_vllm.pipeline.commit import git_head, run_workload
    from verity_vllm.pipeline.tp import phases as PH
    from verity_vllm.pipeline.tp import target as TT
    from verity_vllm.pipeline.tp import trace as TRACE
    from verity_vllm.pipeline.tp import warmup as WU
    from verity_vllm.pipeline.tp.capture import parse_engine_args

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    batches = [int(b) for b in a.batches.split(",")]
    wl = va.load_workload(a.workload, None)
    target = PO.commit_guard(va.target_of_workload(wl), world=a.tp, installs=False)
    target_env, target_kw = TT.apply(target, wl["execution"])
    case = va.case_of(wl, a.case)
    env_flags = {**prof.apply_env(), **target_env}
    cp = prof.checkpoint_for(case, a.checkpoints)
    kw = prof.engine_kwargs(cp, replay=False, seed=prof.ENGINE_SEED, enforce_eager=True, tensor_parallel_size=a.tp)
    kw.update({"enable_prefix_caching": False, "max_num_seqs": max(batches), **target_kw})
    kw.update({k: v for k, v in (wl.get("engine_args_required") or {}).items() if k != "tensor_parallel_size"})
    kw.update(parse_engine_args(a.engine_arg), worker_extension_cls=EXTENSION)
    llm, build_s, topo = PH.engine(lambda: LLM(**kw), a.tp, a.rpc_timeout, model=cp["repo"], enforce_eager=True, case=case, max_num_seqs=max(batches))
    TT.require_ranks(llm, target, a.rpc_timeout)
    served = SV.require(llm, wl, target, out, a.rpc_timeout)
    eng = va.Engine(llm=llm, runner=None, model=None, cp=cp, kwargs=kw, env_flags=env_flags, build_s=build_s, case=case, target=target)
    world = len(topo)
    to_cap = va.sampling_params

    def capped(req: dict[str, Any], sampling: dict[str, Any]) -> Any:
        p = to_cap(req, sampling)
        p.ignore_eos = True
        return p

    va.sampling_params = capped
    sampling = wl["sampling"]
    reqs = {b: requests_at(wl, b, a.rotate) for b in batches}
    engine_kwargs = {k: v for k, v in kw.items() if k not in ("model", "tokenizer", "worker_extension_cls")}
    versions = {"git_head": git_head(), "args": vars(a), "tp": a.tp, "build_s": round(build_s, 1),
                "engine_kwargs": engine_kwargs, "target": TT.as_json(target), "served": served, "ranks": topo}
    (out / "versions.json").write_text(json.dumps(versions, indent=1, default=str) + "\n")
    _row_bytes, layer = ARMS[a.arm]
    manifest = None
    if not layer:
        manifest = out / "acquisition_manifest.json"
        manifest.write_text(json.dumps(acquisition_manifest(world)) + "\n")
    big = reqs[max(batches)]
    envelope = {"batch": max(batches), "max_prompt": max(len(r["prompt_token_ids"]) for r in big), "max_new": max(int(r["max_tokens"]) for r in big),
                "max_num_batched_tokens": kw.get("max_num_batched_tokens"), "max_model_len": kw.get("max_model_len")}
    t0 = time.perf_counter()
    made = llm.collective_rpc(make, timeout=a.rpc_timeout,
                              kwargs={"arm": a.arm, "run_id": f"tp-decode/{a.arm}", "cp": cp, "case": case, "hidden_so": None, "hidden_ns": None,
                                      "envelope": envelope, "n_requests": max(batches), "engine_kwargs": engine_kwargs,
                                      "required_manifest": str(manifest) if manifest else None})
    if len(made) != world:
        raise SystemExit(f"fail-closed: committer made on {len(made)}/{world} ranks")
    (out / "committer_make.json").write_text(json.dumps(made, indent=1, default=str) + "\n")
    m0 = min(made, key=lambda m: int(m["rank"]))
    print(f"[tp-decode] {a.arm}: {m0['committer']} scheme {m0['commit_scheme']} rows {m0['row_bytes']} B, retain {m0['retain']}, "
          f"{m0['modules']} modules, sources "
          f"{m0['sources']}{' ' + json.dumps(m0['per_layer']) if layer else ''}; made in {time.perf_counter() - t0:.1f}s", flush=True)
    rows: list[dict[str, Any]] = []
    ref: dict[int, Any] = {}
    with (out / "runs.jsonl").open("w") as f:
        for b in batches:
            (out / f"b{b}").mkdir(exist_ok=True)
            PH.warmups(1, lambda i, b=b: run_workload(eng, va.throwaway_requests(eng, reqs[b], seed=100 + i), sampling))
            WU.run(llm, lambda b=b: run_workload(eng, va.throwaway_requests(eng, reqs[b], seed=200), sampling), out / f"b{b}", a.arm, a.rpc_timeout)
            for pair in range(a.pairs):
                for k, arm in enumerate(("control", a.arm) if pair % 2 == 0 else (a.arm, "control")):
                    gc.collect()
                    if arm != "control":
                        llm.collective_rpc("tp2_commit_install", timeout=a.rpc_timeout)
                    cpu0 = llm.collective_rpc(TRACE.thread_cpu, timeout=a.rpc_timeout)
                    r = run_workload(eng, reqs[b], sampling)
                    cpu = TRACE.cpu_delta(cpu0, llm.collective_rpc(TRACE.thread_cpu, timeout=a.rpc_timeout))
                    ref.setdefault(b, r["tokens"])
                    row = {"arm": arm, "batch": b, "pair": pair, "order_in_pair": k, "wall_s": r["wall_s"], "generated_tokens": r["generated_tokens"],
                           "phases": phases(r["step_wall_s"], r["step_tokens_out"], b), "tokens_equal_ref": r["tokens"] == ref[b],
                           "step_wall_s": r["step_wall_s"], "step_tokens_out": r["step_tokens_out"], "thread_cpu_s_rank0": cpu.get(0)}
                    if arm != "control":
                        t1 = time.perf_counter()
                        fin = llm.collective_rpc(finalize, timeout=a.rpc_timeout)
                        row.update(finalize_wall_s=round(time.perf_counter() - t1, 3), ranks=fin,
                                   gate=gate(row["tokens_equal_ref"], fin, world, identity=not layer))
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    ph = row["phases"]
                    print(f"[tp-decode] B{b} pair {pair} {arm}: TPOT {1e3 * (ph['tpot_s'] or 0):.2f} ms over {ph['decode_steps']} steps, TTFT "
                          f"{1e3 * (ph['ttft_mean_s'] or 0):.1f} ms, prefill {1e3 * (ph['prefill_s'] or 0):.1f} ms, wall {r['wall_s']:.2f}s"
                          + (f"; finalize {row['finalize_wall_s']}s, gate {row['gate']}" if arm != "control" else
                             f"; tokens equal {row['tokens_equal_ref']}"), flush=True)
    summary = {"arm": a.arm, "committer": m0["committer"], "row_bytes": m0["row_bytes"], "retain": m0["retain"], "world": world,
               "batches": summarize(rows, a.arm, batches), "gates_passed": all(r["gate"]["passed"] for r in rows if r["arm"] != "control")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for b, d in summary["batches"].items():
        print(f"[tp-decode] B{b}: TPOT control {1e3 * (d['control']['tpot_s'] or 0):.2f} ms, {a.arm} {1e3 * (d[a.arm]['tpot_s'] or 0):.2f} ms, "
              f"ratio {d['tpot_ratio']}; TTFT ratio {d['ttft_mean_ratio']}; prefill ratio {d['prefill_ratio']}; gates {d['gates_passed']}", flush=True)
    if not summary["gates_passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    from served_commit.tp_decode import main as _main

    _main()
