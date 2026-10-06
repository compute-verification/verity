"""How much does committing to proof-unit outputs slow down stock, compiled vLLM?  Llama-3.1-8B BF16 on one GPU,
vLLM's production defaults (torch.compile + CUDA graphs, no batch invariance, prefix caching on), ``max_num_seqs=256``.

Arms (``pu_commit.ARMS``): ``control`` (no op), ``coarse`` (each layer's residual stream), ``fine`` (every linear and
op output of each layer), ``retain`` (fine, plus every hashed byte stored to a pinned host ring) and ``staged`` (fine's
commitment, small sites copied in stream and hashed once per step: ``pu_commit``'s docstring).  Each arm runs in
its own engine process; a round runs every arm once, in a rotated order, on one GPU.

    python pu_bench.py run [--rounds 0,1] [--micro] [--verify]   # everything below, in order, into $RESEARCH_RUN_DIR/out
    python pu_bench.py micro --out OUT                  # hash kernel GB/s, tree fold, D2H/H2D pinned GB/s, per-site latency
    python pu_bench.py verify --out OUT [--arm retain]  # a prefill and a decode step's leaves and root against hashlib
    python pu_bench.py arm --arm A --round R --out OUT  # one engine: throughput workload, then batch-1 latency
    python pu_bench.py summarize OUT... > summary.md    # arm x workload table over every round found
    python pu_bench.py sweep --deadline 2026-10-06T13:55Z  # model x granularity x batch x seed cells, resumable via --state
    python pu_bench.py summarize-cells DIR...           # their table

Workloads: ``vllm bench throughput``'s random dataset run through the offline API (random token ids, ``ignore_eos``,
temperature 1, top_p 1): input 1024, output 256, ``--num-prompts`` at once; then batch 1, input 1024, output 256,
TPOT = (median latency at 256 output tokens - median latency at 1) / 255, distinct prompts so prefix caching never hits.
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
ARMS = ("control", "coarse", "fine", "retain", "staged")


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def _versions() -> dict:
    import torch
    import vllm
    return {"vllm": vllm.__version__, "torch": torch.__version__, "cuda": torch.version.cuda}


def gpu_info() -> dict:
    q = "name,driver_version,uuid,memory.total,clocks.max.sm,power.limit,pcie.link.gen.max,pcie.link.width.max"
    sel = ["-i", os.environ["GPU_LEASE_UUID"]] if os.environ.get("GPU_LEASE_UUID") else []
    r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader", *sel], capture_output=True, text=True)
    vals = [v.strip() for v in r.stdout.strip().splitlines()[0].split(",")] if r.returncode == 0 and r.stdout.strip() else []
    return dict(zip(q.split(","), vals))


def _cuda_time(fn, reps: int = 5) -> float:
    """Median seconds of ``fn`` between CUDA events, after one warm-up call."""
    import torch
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / 1e3)
    return statistics.median(ts)


def micro(args) -> int:
    import torch
    import pu_commit as P
    dev = torch.device("cuda")
    check = _self_check(P, dev)
    big = 2 << 30
    x = torch.randint(-(1 << 15), 1 << 15, (big // P.CHUNK, P.CHUNK // 2), dtype=torch.int16, device=dev)
    n = big // P.CHUNK
    st = P.State(ring_leaves=n + 1024, keep_bytes=0)
    t_leaf = _cuda_time(lambda: st.leaf(x, 0, 0))
    t_fold = _cuda_time(lambda: st.finalize(n, big))
    keep = P.State(ring_leaves=(1 << 30) // P.CHUNK + 1024, keep_bytes=1 << 30)
    x1 = x[: (1 << 30) // P.CHUNK]
    t_keep = _cuda_time(lambda: keep.leaf(x1, 0, 0))
    src = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
    pinned = torch.empty(1 << 30, dtype=torch.uint8, pin_memory=True)
    t_d2h = _cuda_time(lambda: pinned.copy_(src, non_blocking=True))
    t_h2d = _cuda_time(lambda: src.copy_(pinned, non_blocking=True))
    check_staged = _self_check_staged(P, dev)
    check["ok"] = check["ok"] and check_staged["ok"]
    out = {"self_check": check, "self_check_staged": check_staged, "gpu": gpu_info(), "versions": _versions(), "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
           "hash_leaves": {"bytes": big, "seconds": t_leaf, "GBps": big / t_leaf / 1e9},
           "tree_fold": {"leaves": n, "seconds": t_fold, "leaves_per_s": n / t_fold},
           "hash_and_fold_GBps": big / (t_leaf + t_fold) / 1e9,
           "hash_leaves_with_mapped_host_store": {"bytes": 1 << 30, "seconds": t_keep, "GBps": (1 << 30) / t_keep / 1e9},
           "d2h_pinned_dma_GBps": (1 << 30) / t_d2h / 1e9, "h2d_pinned_dma_GBps": (1 << 30) / t_h2d / 1e9}
    # per-site cost inside a CUDA graph: 288 sites (Llama-8B's fine arm) of one decode step at batch 1 (8 KiB) and 256 (2 MiB)
    per_site = {}
    for label, nbytes in (("8KiB", 8192), ("2MiB", 2 << 20)):
        xs = x[: nbytes // P.CHUNK]
        cnt = nbytes // P.CHUNK
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(288):
                st.leaf(xs, i, i * cnt)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            for i in range(288):
                st.leaf(xs, i, i * cnt)
        t = _cuda_time(g.replay, reps=20)
        per_site[label] = {"us_per_site": t / 288 * 1e6, "step_288_sites_ms": t * 1e3}
        fold_n = 288 * cnt
        per_site[label]["fold_step_ms"] = _cuda_time(lambda: st.finalize(fold_n, fold_n * P.CHUNK)) * 1e3
    out["graph_per_site"] = per_site
    out["graph_step"] = _graph_step(P, dev)
    _write(Path(args.out) / "micro.json", out)
    print(json.dumps(out, indent=1))
    return 0 if check["ok"] else 1


LLAMA8B_FINE = (("in_norm", 4096), ("qkv", 6144), ("attn", 4096), ("o_proj", 4096), ("post_norm", 4096),
                ("resid_post", 4096), ("gate_up", 28672), ("act", 14336), ("down", 4096))


def _llama8b_layout(P, tokens: int):
    return P.Layout([(f"{l}.{n}", 2 * w) for l in range(32) for n, w in LLAMA8B_FINE], tokens)


def _graph_step(P, dev) -> dict:
    """One decode step's commitment alone, captured in a CUDA graph: Llama-8B's 288 fine sites at their real sizes, then
    the fold, for fine's per-site hashing and for staged's copies + one hash."""
    import torch
    out = {}
    for tokens in (1, 256):
        lay = _llama8b_layout(P, tokens)
        xs = [torch.randn(tokens, w // 2, device=dev, dtype=torch.bfloat16) for w in lay.bytes_per_token]
        n = tokens * lay.chunks_per_token
        for arm, st in (("fine", P.State(n + 1024, 0)), ("staged", P.State(n + 1024, 0, lay))):
            def step():
                for s, x in enumerate(xs):
                    st.leaf(x, s, tokens * lay.chunk_offset[s])
                st.finalize(n, n * P.CHUNK)
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                step()
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(g):
                step()
            out[f"{arm}_b{tokens}_ms"] = _cuda_time(g.replay, reps=20) * 1e3
            del g, st
    return out


def _self_check_staged(P, dev) -> dict:
    """The staged path (sites staged, hashed directly, and zero-padded to whole chunks: an fp32 site of 50 elements) against
    hashlib."""
    import torch
    keep_min = P.DIRECT_MIN
    P.DIRECT_MIN = 10
    try:
        specs = [("0.a", 256, torch.bfloat16), ("0.b", 768, torch.bfloat16), ("0.c", 200, torch.float32),
                 ("1.a", 256, torch.bfloat16), ("1.b", 768, torch.bfloat16), ("1.c", 200, torch.float32)]
        lay = P.Layout([(n, w) for n, w, _ in specs], 8)
        T = 5
        n = T * lay.chunks_per_token
        st = P.State(80, 0, lay)
        st.ctr[0] = 41
        xs = []
        for s, (_, w, dt) in enumerate(specs):
            x = torch.randint(0, 256, (T, w), dtype=torch.uint8, device=dev).view(dt)
            xs.append(x)
            st.leaf(x, s, T * lay.chunk_offset[s])
        st.finalize(n, n * P.CHUNK)
        torch.cuda.synchronize()
        leaves = []
        for s, x in enumerate(xs):
            c = lay.site_chunks[s]
            b = b"".join(r.tobytes() + bytes(c * P.CHUNK - len(r.tobytes())) for r in x.cpu().view(torch.uint8).numpy())
            cnt = T * c
            leaves += [P.leaf(s, ci, cnt, b[ci * P.CHUNK:(ci + 1) * P.CHUNK]) for ci in range(cnt)]
        root = st.record(0)[0]
        return {"ok": root == P.fold(leaves), "root": root.hex(), "leaves": n, "staged_sites": "a, c (5 chunks)",
                "direct_sites": "b (15 chunks)", "padded_sites": "c (200 B per token)"}
    finally:
        P.DIRECT_MIN = keep_min


def _self_check(P, dev) -> dict:
    """The kernel's step root and retained bytes against hashlib, on sites of odd chunk counts across a ring wrap."""
    import torch
    counts = (7, 1, 300)
    st = P.State(ring_leaves=400, keep_bytes=500 * P.CHUNK)
    st.ctr[0], st.ctr[1] = 350, 450
    xs = [torch.randint(-(1 << 15), 1 << 15, (c, P.CHUNK // 2), dtype=torch.int16, device=dev) for c in counts]
    off = 0
    for site, x in enumerate(xs):
        st.leaf(x, site, off)
        off += x.numel() * 2 // P.CHUNK
    st.finalize(off, off * P.CHUNK)
    torch.cuda.synchronize()
    data = b"".join(x.cpu().numpy().tobytes() for x in xs)
    leaves = [P.leaf(site, ci, c, x.cpu().numpy().tobytes()[ci * P.CHUNK:(ci + 1) * P.CHUNK])
              for site, (c, x) in enumerate(zip(counts, xs)) for ci in range(c)]
    root = st.record(0)[0]
    keep = st.keep.numpy().tobytes()
    kept = b"".join(keep[((450 + i) % 500) * P.CHUNK:][:P.CHUNK] for i in range(off))
    ok = root == P.fold(leaves) and kept == data
    return {"ok": ok, "root": root.hex(), "leaves": off, "counts": counts}


def _llm(args):
    from vllm import LLM
    return LLM(model=args.model, max_num_seqs=args.max_num_seqs, worker_extension_cls="pu_ext.PUExt", seed=0,
               max_model_len=args.max_model_len)


def _prompts(seed: int, n: int, length: int, vocab: int) -> list[dict]:
    import numpy as np
    rng = np.random.default_rng(seed)
    return [{"prompt_token_ids": rng.integers(1000, vocab - 1000, length).tolist()} for _ in range(n)]


def _engine(llm, stats: dict) -> dict:
    vc = llm.llm_engine.vllm_config
    return {"model": vc.model_config.model, "dtype": str(vc.model_config.dtype), "max_num_seqs": vc.scheduler_config.max_num_seqs,
            "max_num_batched_tokens": vc.scheduler_config.max_num_batched_tokens,
            "enable_prefix_caching": vc.cache_config.enable_prefix_caching, "enforce_eager": vc.model_config.enforce_eager,
            "VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT"),
            **{k: stats.get(k) for k in ("cudagraph_mode", "compilation_mode", "cudagraph_capture_sizes", "num_gpu_blocks", "block_size")}}


def _window(a: dict, b: dict) -> dict:
    if not b.get("steps") and not a.get("steps"):
        return {}
    return {k: b[k] - a.get(k, 0) for k in ("steps", "bytes", "leaves", "leaf_calls", "finalize_calls")} | \
        {"step_tokens": b.get("step_tokens")}


def arm(args) -> int:
    from vllm import SamplingParams
    t0 = time.perf_counter()
    llm = _llm(args)
    rpc = lambda m, *a: llm.collective_rpc(m, args=a)[0]  # noqa: E731
    init_s = time.perf_counter() - t0
    vocab = llm.get_tokenizer().vocab_size
    s_init = rpc("pu_stats")
    llm.generate(_prompts(100 + args.round, args.max_num_seqs, args.input_len, vocab),
                 SamplingParams(max_tokens=32, ignore_eos=True), use_tqdm=False)
    s0 = rpc("pu_stats")
    sp = SamplingParams(n=1, temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=args.output_len)
    t = time.perf_counter()
    outs = llm.generate(_prompts(1, args.num_prompts, args.input_len, vocab), sp, use_tqdm=False)
    wall = time.perf_counter() - t
    s1 = rpc("pu_stats", s0.get("steps"))
    out_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    in_tok = sum(len(o.prompt_token_ids) for o in outs)
    tp = {"wall_s": wall, "requests": len(outs), "prompt_tokens": in_tok, "output_tokens": out_tok,
          "output_tps": out_tok / wall, "total_tps": (in_tok + out_tok) / wall, "requests_per_s": len(outs) / wall,
          "commit": _window(s0, s1)}
    if tp["commit"]:
        tp["commit"]["bytes_per_output_token"] = tp["commit"]["bytes"] / out_tok
    lat_full, lat_one = [], []
    p_full = _prompts(2, args.lat_reqs, args.input_len, vocab)
    p_one = _prompts(3, args.lat_reqs, args.input_len, vocab)
    for i in range(args.lat_reqs):
        t = time.perf_counter()
        o = llm.generate([p_full[i]], sp, use_tqdm=False)
        lat_full.append(time.perf_counter() - t)
        assert len(o[0].outputs[0].token_ids) == args.output_len
        t = time.perf_counter()
        llm.generate([p_one[i]], SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=1), use_tqdm=False)
        lat_one.append(time.perf_counter() - t)
    s2 = rpc("pu_stats", s1.get("steps"))
    lat = {"latency_s": statistics.median(lat_full), "ttft_proxy_s": statistics.median(lat_one),
           "tpot_ms": (statistics.median(lat_full) - statistics.median(lat_one)) / (args.output_len - 1) * 1e3,
           "latencies_s": lat_full, "one_token_s": lat_one, "commit": _window(s1, s2)}
    if lat["commit"]:
        lat["commit"]["bytes_per_output_token"] = lat["commit"]["bytes"] / (args.lat_reqs * (args.output_len + 1))
    res = {"arm": args.arm, "round": args.round, "gpu": gpu_info(), "versions": _versions(), "engine": _engine(llm, s0),
           "init_s": init_s, "layout": s0.get("layout"), "ring_bytes": s0.get("ring_bytes"), "keep_bytes": s0.get("keep_bytes"),
           "commit_counters_after_init": {k: s_init.get(k) for k in ("steps", "leaf_calls", "finalize_calls")},
           "workload": {"input_len": args.input_len, "output_len": args.output_len, "num_prompts": args.num_prompts,
                        "lat_reqs": args.lat_reqs},
           "throughput": tp, "batch1": lat}
    _write(Path(args.out) / f"arm-r{args.round}-{args.arm}.json", res)
    print(json.dumps({k: res[k] for k in ("arm", "round", "throughput", "batch1")}, indent=1, default=str))
    return 0


def verify(args) -> int:
    """The retain (or staged) arm in the production-compiled engine: a few requests whose prefill (400 tokens, a piecewise
    graph) and decode (a full graph) steps are recomputed with hashlib from the retained bytes (staged: the last step's,
    from the staging buffer)."""
    from vllm import SamplingParams
    llm = _llm(args)
    rpc = lambda m, *a: llm.collective_rpc(m, args=a)[0]  # noqa: E731
    vocab = llm.get_tokenizer().vocab_size
    llm.generate(_prompts(7, 4, 100, vocab), SamplingParams(max_tokens=8, ignore_eos=True), use_tqdm=False)
    s0 = rpc("pu_stats")
    outs = llm.generate(_prompts(8, 4, 100, vocab), SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True), use_tqdm=False)
    s1 = rpc("pu_stats", s0["steps"])
    res = {"gpu": gpu_info(), "versions": _versions(), "engine": _engine(llm, s0), "window": _window(s0, s1),
           "output_tokens": sum(len(o.outputs[0].token_ids) for o in outs),
           "first_step": rpc("pu_verify", s0["steps"]), "last_step": rpc("pu_verify", s1["steps"] - 1)}
    if args.arm == "staged":
        res["pass"] = (res["last_step"].get("root_from_staged_bytes_matches") is True and
                       all(res[k]["root_from_device_leaves_matches"] and res[k]["nonzero_leaves"] == res[k]["n_leaves"]
                           for k in ("first_step", "last_step")))
    else:
        res["pass"] = all(res[k].get("root_from_retained_bytes_matches") and res[k]["nonzero_leaves"] == res[k]["n_leaves"]
                          for k in ("first_step", "last_step"))
    # what Inductor made of the ops: their call lines, and two whole output-code files to read for copies of the token
    cache = Path(os.environ.get("VLLM_CACHE_ROOT", "~/.cache/vllm")).expanduser()
    lines, kept = [], 0
    for f in sorted(cache.rglob("*.py")):
        try:
            text = f.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if "verity_pu" in text:
            lines += [f"{f.relative_to(cache)}: {ln.strip()}" for ln in text.splitlines() if "verity_pu" in ln]
            if kept < 2:
                (Path(args.out) / f"inductor-{args.arm}_{kept}_{f.name}").write_text(text)
                kept += 1
    (Path(args.out) / f"inductor-{args.arm}_verity_pu_lines.txt").write_text("\n".join(lines) + "\n")
    res["inductor_lines_with_verity_pu"] = len(lines)
    _write(Path(args.out) / f"verify-{args.arm}.json", res)
    print(json.dumps(res, indent=1, default=str))
    return 0 if res["pass"] else 1


def _child(args, mode: list[str], arm_name: str, log: Path) -> int:
    env = dict(os.environ)
    env.pop("VLLM_BATCH_INVARIANT", None)
    env.update(VLLM_PU_ARM=arm_name, PYTHONPATH=f"{HERE}{os.pathsep}{env.get('PYTHONPATH', '')}",
               VLLM_CACHE_ROOT=str(Path(args.scratch) / f"vllm-{arm_name}"), VLLM_DISABLE_COMPILE_CACHE="1",
               TORCH_EXTENSIONS_DIR=str(Path(args.scratch) / "torch-ext"), TRITON_CACHE_DIR=str(Path(args.scratch) / "triton"))
    cmd = [sys.executable, str(HERE / "pu_bench.py"), *mode, "--out", args.out, "--model", args.model,
           "--max-num-seqs", str(args.max_num_seqs), "--input-len", str(args.input_len), "--output-len", str(args.output_len),
           "--num-prompts", str(args.num_prompts), "--lat-reqs", str(args.lat_reqs), "--max-model-len", str(args.max_model_len)]
    print(f"[pu_bench] {' '.join(mode)} arm={arm_name} -> {log}", flush=True)
    t = time.time()
    with log.open("w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"[pu_bench] rc={rc} in {time.time() - t:.0f} s", flush=True)
    return rc


def run(args) -> int:
    out = Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    rc = 0
    if args.micro:
        rc |= _child(args, ["micro"], "control", out / "logs" / "micro.log")
    if args.verify:
        for a in args.verify_arms.split(","):
            rc |= _child(args, ["verify", "--arm", a], a, out / "logs" / f"verify-{a}.log")
    arms = args.arms.split(",")
    for r in args.rounds:
        k = r % len(arms)
        order = arms[k:] + arms[:k]
        if r % 2 and len(arms) > 2:
            order.reverse()
        for a in order:
            rc |= _child(args, ["arm", "--arm", a, "--round", str(r)], a, out / "logs" / f"arm-r{r}-{a}.log")
    summary = summarize_dirs([out])
    (out / "summary.md").write_text(summary)
    print(summary)
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    if run_dir:
        rows = [json.loads(p.read_text()) for p in sorted(out.glob("arm-r*.json"))]
        vers = {p.stem: json.loads(p.read_text())["pass"] for p in sorted(out.glob("verify-*.json"))}
        ver = {"pass": all(vers.values()), "arms": vers} if vers else None
        result = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", "local"), "kind": "bench-result/v1",
                  "workload_fingerprint": {"workload": "vllm_pu_commit", "model": args.model, "arms": args.arms, "rounds": args.rounds,
                                           "input_len": args.input_len, "output_len": args.output_len,
                                           "num_prompts": args.num_prompts, "max_num_seqs": args.max_num_seqs},
                  "validation": {"status": "passed" if rc == 0 and (ver is None or ver["pass"]) else "failed",
                                 "checker": "every engine process exited 0" + ("; verify: a prefill and a decode step's leaves and "
                                                                               "root recomputed with hashlib" if ver else ""),
                                 "detail": f"rc={rc}" + (f", verify {ver['arms']}" if ver else ""), "evidence": []},
                  "measurements": [m for r in rows for m in (
                      {"name": f"r{r['round']}.{r['arm']}.output_tps", "value": r["throughput"]["output_tps"], "unit": "tok/s"},
                      {"name": f"r{r['round']}.{r['arm']}.total_tps", "value": r["throughput"]["total_tps"], "unit": "tok/s"},
                      {"name": f"r{r['round']}.{r['arm']}.tpot_b1", "value": r["batch1"]["tpot_ms"], "unit": "ms"})],
                  "measurement_files": [os.path.relpath(p, run_dir) for p in [*sorted(out.glob("*.json")), out / "summary.md"]],
                  "artifacts": []}
        (Path(run_dir) / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    return rc


MODELS = {"llama31-8b": "unsloth/Meta-Llama-3.1-8B-Instruct", "qwen3-8b": "Qwen/Qwen3-8B",
          "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B", "glm47-flash": "zai-org/GLM-4.7-Flash"}
GRANS = {"none": "control", "coarse": "coarse", "fine": "fine", "staged": "staged"}
HUBS = ("/workspace/hf/hub", "/workspace/jobs/hf/hub")


def resolve(model_id: str) -> Path | None:
    """The complete local snapshot of ``model_id`` (every shard its index names), or None."""
    hubs = [Path(os.environ["HF_HOME"]) / "hub"] if os.environ.get("HF_HOME") else []
    for hub in [*hubs, *map(Path, HUBS)]:
        snaps = hub / f"models--{model_id.replace('/', '--')}" / "snapshots"
        for snap in sorted(snaps.glob("*")) if snaps.is_dir() else []:
            idx = snap / "model.safetensors.index.json"
            shards = set(json.loads(idx.read_text())["weight_map"].values()) if idx.exists() else {"model.safetensors"}
            if (snap / "config.json").exists() and all((snap / s).exists() for s in shards):
                return snap
    return None


def _cell_name(model: str, gran: str, b: int, seed: int) -> str:
    return f"{model}__{gran}__b{b}__s{seed}.json"


def _gen(llm, prompts, sp) -> float:
    t = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t
    assert all(len(o.outputs[0].token_ids) == sp.max_tokens for o in outs), "a request stopped early"
    return dt


def cells(args) -> int:
    """One engine (``--model-key``, ``--gran``, ``--seed``): for each batch b of ``--batches`` not yet in the state dir,
    waves of b concurrent requests (distinct random prompts, ``ignore_eos``, temperature 1): ``reps`` waves of
    ``output_len`` tokens and ``reps`` of one token, interleaved.  TPOT at b = (median full wave - median one-token wave) /
    (output_len - 1); output tok/s = b * output_len / median full wave.  Each finished cell is written at once."""
    from vllm import LLM, SamplingParams
    state, out = Path(args.state) / "cells", Path(args.out) / "cells"
    state.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    llm = LLM(model=args.model, max_num_seqs=max(args.batches), max_model_len=args.max_model_len, seed=args.seed,
              gpu_memory_utilization=args.gpu_mem, worker_extension_cls="pu_ext.PUExt")
    init_s = time.perf_counter() - t0
    rpc = lambda m, *a: llm.collective_rpc(m, args=a)[0]  # noqa: E731
    vocab = llm.get_tokenizer().vocab_size
    s_init = rpc("pu_stats")
    eng = _engine(llm, s_init)
    blocks, bs = s_init["num_gpu_blocks"], s_init["block_size"]
    rc = 0
    for b in args.batches:
        name = _cell_name(args.model_key, args.gran, b, args.seed)
        if (state / name).exists():
            continue
        in_len = args.input_len
        while in_len > 64 and b * -(-(in_len + args.output_len + 1) // bs) > blocks:
            in_len -= 64
        reps = 5 if b == 1 else 3
        try:
            _gen(llm, _prompts([args.seed, b, 99], b, in_len, vocab), SamplingParams(max_tokens=8, ignore_eos=True))
            s0 = rpc("pu_stats")
            full, one = [], []
            sp = SamplingParams(temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=args.output_len)
            sp1 = SamplingParams(temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=1)
            for r in range(reps):
                full.append(_gen(llm, _prompts([args.seed, b, r, 0], b, in_len, vocab), sp))
                one.append(_gen(llm, _prompts([args.seed, b, r, 1], b, in_len, vocab), sp1))
            s1 = rpc("pu_stats", s0.get("steps"))
        except Exception as e:  # noqa: BLE001 - a cell that cannot run is recorded and the engine moves on
            rc = 1
            _write(out / f"failed__{name}", {"error": repr(e)[:4000], "batch": b, "input_len": in_len})
            print(f"[cells] {name} failed: {e!r}", flush=True)
            continue
        lf, lo = statistics.median(full), statistics.median(one)
        tpot = (lf - lo) / (args.output_len - 1)
        res = {"model": args.model_key, "model_path": args.model, "gran": args.gran, "arm": GRANS[args.gran], "batch": b,
               "seed": args.seed, "input_len": in_len, "input_len_requested": args.input_len, "output_len": args.output_len,
               "reps": reps, "full_wave_s": full, "one_token_wave_s": one, "full_wave_median_s": lf,
               "one_token_wave_median_s": lo, "tpot_ms": tpot * 1e3, "output_tps": b * args.output_len / lf,
               "decode_tps": b / tpot, "kv_tokens": blocks * bs, "commit": _window(s0, s1), "gpu": gpu_info(),
               "versions": _versions(), "engine": eng, "init_s": init_s, "sites": s_init.get("sites"),
               "layout": s1.get("layout"), "untraced": s1.get("untraced"), "ring_bytes": s1.get("ring_bytes"),
               "stage_bytes": s1.get("stage_bytes"), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if GRANS[args.gran] != "control" and not (res["commit"] and res["commit"]["steps"] > 0 and res["sites"]):
            rc = 1
            _write(out / f"failed__{name}", {"error": "no commitment steps recorded", **res})
            continue
        _write(out / name, res)
        _write(state / name, res)
        print(f"[cells] {name}: {res['output_tps']:.1f} tok/s, TPOT {res['tpot_ms']:.2f} ms, in {in_len}", flush=True)
    return rc


def _cells_child(args, model_key: str, path: Path, gran: str, seed: int, batches: list[int], log: Path,
                 timeout: float | None = None) -> int:
    env = dict(os.environ)
    env.pop("VLLM_BATCH_INVARIANT", None)
    scratch = Path(args.scratch)
    env.update(VLLM_PU_ARM=GRANS[gran], PYTHONPATH=f"{HERE}{os.pathsep}{env.get('PYTHONPATH', '')}",
               VLLM_CACHE_ROOT=str(scratch / f"vllm-{model_key}-{gran}"), VLLM_DISABLE_COMPILE_CACHE="1",
               TORCH_EXTENSIONS_DIR=str(Path(args.state) / "torch-ext"), TRITON_CACHE_DIR=str(Path(args.state) / "triton"))
    cmd = [sys.executable, str(HERE / "pu_bench.py"), "cells", "--out", args.out, "--state", args.state, "--model", str(path),
           "--model-key", model_key, "--gran", gran, "--seed", str(seed), "--batches", ",".join(map(str, batches)),
           "--input-len", str(args.input_len), "--output-len", str(args.output_len), "--max-model-len", str(args.max_model_len),
           "--gpu-mem", str(args.gpu_mem)]
    print(f"[sweep] {model_key} {gran} seed {seed} batches {batches} -> {log}", flush=True)
    t = time.time()
    with log.open("w") as f:
        try:
            rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    print(f"[sweep] rc={rc} in {time.time() - t:.0f} s", flush=True)
    return rc


def sweep(args) -> int:
    """Every (model, granularity, batch, seed) cell not yet in ``--state``: per model, a staged verify once, then per seed
    one engine per granularity (order rotated by seed, reversed on odd seeds) running that seed's missing batches.  Stops
    starting engines when the next one would not finish before ``--deadline``; a lane resumed later skips finished cells."""
    out, st = Path(args.out), Path(args.state)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    st.mkdir(parents=True, exist_ok=True)
    deadline = calendar.timegm(time.strptime(args.deadline, "%Y-%m-%dT%H:%MZ")) if args.deadline else None
    grans = list(args.grans)
    engine_s: list[float] = []
    rc, notes = 0, []
    for model_key in args.models:
        path = resolve(MODELS[model_key])
        if path is None:
            notes.append(f"{model_key}: no complete local snapshot of {MODELS[model_key]}; skipped")
            print(f"[sweep] {notes[-1]}", flush=True)
            continue
        if args.verify and not (st / f"verify-{model_key}-staged.json").exists():
            vargs = argparse.Namespace(**{**vars(args), "model": str(path), "max_num_seqs": 16})
            vrc = _child(vargs, ["verify", "--arm", "staged"], "staged", out / "logs" / f"verify-{model_key}.log")
            v = out / "verify-staged.json"
            if v.exists():
                v.rename(out / f"verify-{model_key}-staged.json")
                if vrc == 0:
                    (st / f"verify-{model_key}-staged.json").write_bytes((out / f"verify-{model_key}-staged.json").read_bytes())
            rc |= vrc
        for seed in args.seeds:
            k = seed % len(grans)
            order = grans[k:] + grans[:k]
            if seed % 2:
                order.reverse()
            for gran in order:
                missing = [b for b in args.batches if not (st / "cells" / _cell_name(model_key, gran, b, seed)).exists()]
                if not missing:
                    continue
                need = max(engine_s) if engine_s else args.engine_min * 60
                if deadline and time.time() + need > deadline:
                    notes.append(f"stopped before {model_key} {gran} seed {seed}: the deadline")
                    print(f"[sweep] {notes[-1]}", flush=True)
                    _sweep_result(args, rc, notes)
                    return rc
                t = time.time()
                crc = _cells_child(args, model_key, path, gran, seed, missing,
                                   out / "logs" / f"{model_key}__{gran}__s{seed}.log",
                                   deadline + 180 - time.time() if deadline else None)
                engine_s.append(time.time() - t)
                rc |= crc
                if crc:
                    notes.append(f"{model_key} {gran} seed {seed}: engine rc={crc}")
                _sweep_result(args, rc, notes)
    _sweep_result(args, rc, notes)
    return rc


def _sweep_result(args, rc: int, notes: list[str]) -> None:
    out = Path(args.out)
    rows = [json.loads(p.read_text()) for p in sorted((out / "cells").glob("*__s*.json")) if not p.name.startswith("failed")]
    summary = summarize_cells([Path(args.state) / "cells"])
    (out / "summary.md").write_text(summary + "".join(f"\n- {n}" for n in notes) + "\n")
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    if not run_dir:
        return
    result = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", "local"), "kind": "bench-result/v1",
              "workload_fingerprint": {"workload": "vllm_pu_commit_sweep", "models": args.models, "grans": args.grans,
                                       "batches": args.batches, "seeds": args.seeds, "input_len": args.input_len,
                                       "output_len": args.output_len, "max_model_len": args.max_model_len},
              "validation": {"status": "passed" if rc == 0 else "failed",
                             "checker": "every engine process exited 0 and every non-control cell recorded commitment steps"
                                        + ("; staged verify per model" if args.verify else ""),
                             "detail": f"rc={rc}; {len(rows)} cells this run; " + "; ".join(notes), "evidence": []},
              "measurements": [m for r in rows for m in (
                  {"name": f"{r['model']}.{r['gran']}.b{r['batch']}.s{r['seed']}.output_tps", "value": r["output_tps"], "unit": "tok/s"},
                  {"name": f"{r['model']}.{r['gran']}.b{r['batch']}.s{r['seed']}.tpot", "value": r["tpot_ms"], "unit": "ms"})],
              "measurement_files": [os.path.relpath(p, run_dir) for p in [*sorted(out.rglob("*.json")), out / "summary.md"]],
              "artifacts": []}
    (Path(run_dir) / "result.json").write_text(json.dumps(result, indent=1) + "\n")


def summarize_cells(dirs: list[Path]) -> str:
    """Per model, granularity and batch: medians over seeds of output tok/s and TPOT, and the overhead against the control
    of the same seed (extra time: control_tps / tps - 1, tpot / control_tpot - 1), median and min..max over seeds."""
    rows = {}
    for d in dirs:
        for p in sorted(Path(d).rglob("*__s*.json")):
            if not p.name.startswith("failed"):
                r = json.loads(p.read_text())
                rows[(r["model"], r["gran"], r["batch"], r["seed"])] = r
    lines = ["| model | granularity | batch | output tok/s | TPOT ms | throughput overhead | TPOT overhead | seeds | input |",
             "|---|---|---|---|---|---|---|---|---|"]
    med = statistics.median
    for model in MODELS:
        for gran in GRANS:
            for b in sorted({k[2] for k in rows}):
                rs = [r for k, r in rows.items() if k[:3] == (model, gran, b)]
                if not rs:
                    continue
                ot, tp = [], []
                for r in rs:
                    c = rows.get((model, "none", b, r["seed"]))
                    if c and gran != "none":
                        ot.append((c["output_tps"] / r["output_tps"] - 1) * 100)
                        tp.append((r["tpot_ms"] / c["tpot_ms"] - 1) * 100)
                fmt = lambda v: f"{med(v):+.2f}% ({min(v):+.2f}..{max(v):+.2f})" if v else "-"  # noqa: E731
                lines.append(f"| {model} | {gran} | {b} | {med([r['output_tps'] for r in rs]):.1f} | "
                             f"{med([r['tpot_ms'] for r in rs]):.2f} | {fmt(ot)} | {fmt(tp)} | {len(rs)} | "
                             f"{'/'.join(sorted({str(r['input_len']) for r in rs}))} |")
    return "\n".join(lines) + "\n"


def summarize_dirs(dirs: list[Path]) -> str:
    rows = [json.loads(p.read_text()) for d in dirs for p in sorted(Path(d).rglob("arm-r*.json"))]
    by_round: dict[tuple, dict] = {}
    for r in rows:
        by_round.setdefault((r["round"], r["gpu"].get("uuid")), {})[r["arm"]] = r
    metrics = {"decode-heavy output tok/s": lambda r: r["throughput"]["output_tps"],
               "decode-heavy total tok/s": lambda r: r["throughput"]["total_tps"],
               "batch-1 TPOT ms": lambda r: r["batch1"]["tpot_ms"]}
    # Overhead is extra time: control_tps / arm_tps - 1 for throughput, arm_tpot / control_tpot - 1 for TPOT. Bytes are those
    # hashed per output token: for the decode-heavy workload every prompt token's bytes count too (all hashed over output
    # tokens); for batch 1 a decode step processes one token, so it is the bytes of one token through every site.
    lines = ["| arm | metric | mean | min..max | overhead vs control (mean, min..max over rounds) | bytes hashed / output token | rounds |",
             "|---|---|---|---|---|---|---|"]
    for a in ARMS:
        for name, f in metrics.items():
            vals, ovh, bpt = [], [], []
            for arms in by_round.values():
                if a not in arms or "control" not in arms:
                    continue
                v, c = f(arms[a]), f(arms["control"])
                vals.append(v)
                ovh.append((v / c - 1) * 100 if "TPOT" in name else (c / v - 1) * 100)
                if "TPOT" in name:
                    bpt.append((arms[a].get("layout") or {}).get("bytes_per_token_step", 0))
                else:
                    bpt.append((arms[a]["throughput"].get("commit") or {}).get("bytes_per_output_token", 0))
            if not vals:
                continue
            lines.append(f"| {a} | {name} | {statistics.mean(vals):.1f} | {min(vals):.1f}..{max(vals):.1f} | "
                         f"{statistics.mean(ovh):+.2f}% ({min(ovh):+.2f}..{max(ovh):+.2f}) | "
                         f"{statistics.mean(bpt) / 1e6:.2f} MB | {len(vals)} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ints = lambda s: [int(x) for x in s.split(",")]  # noqa: E731
    ap.add_argument("mode", choices=("run", "micro", "verify", "arm", "summarize", "sweep", "cells", "summarize-cells"))
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--out", default=str(Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out"))
    ap.add_argument("--scratch", default=None, help="compile caches and the extension build (default: a temporary directory)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--arm", choices=ARMS, default="control")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--verify-arms", default="retain,staged")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--rounds", type=lambda s: [int(x) for x in s.split(",")], default=[0])
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--num-prompts", type=int, default=1000)
    ap.add_argument("--lat-reqs", type=int, default=5)
    ap.add_argument("--micro", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--models", type=lambda s: s.split(","), default=list(MODELS))
    ap.add_argument("--grans", type=lambda s: s.split(","), default=list(GRANS))
    ap.add_argument("--batches", type=ints, default=[1, 8, 32, 128])
    ap.add_argument("--seeds", type=ints, default=[0, 1, 2])
    ap.add_argument("--state", default="/workspace/jobs/cco-sweep", help="finished cells (resumable across runs), caches")
    ap.add_argument("--deadline", default=None, help="UTC YYYY-MM-DDTHH:MMZ: start no engine that would end after it")
    ap.add_argument("--engine-min", type=float, default=12.0, help="minutes one engine is assumed to take before one has run")
    ap.add_argument("--model-key", default=None)
    ap.add_argument("--gran", choices=tuple(GRANS), default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    args = ap.parse_args(argv)
    if args.mode == "summarize":
        sys.stdout.write(summarize_dirs([Path(d) for d in args.dirs]))
        return 0
    if args.mode == "summarize-cells":
        sys.stdout.write(summarize_cells([Path(d) for d in args.dirs]))
        return 0
    if args.mode in ("run", "sweep") and not args.scratch:
        with tempfile.TemporaryDirectory(prefix="pu_bench-") as d:
            args.scratch = d
            return (run if args.mode == "run" else sweep)(args)
    return {"run": run, "micro": micro, "verify": verify, "arm": arm, "sweep": sweep, "cells": cells}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
