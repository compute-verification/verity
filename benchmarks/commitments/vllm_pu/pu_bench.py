"""How much does committing to proof-unit outputs slow down stock, compiled vLLM?  Llama-3.1-8B BF16 on one GPU,
vLLM's production defaults (torch.compile + CUDA graphs, no batch invariance, prefix caching on), ``max_num_seqs=256``.

Arms (``pu_commit.ARMS``): ``control`` (no op), ``coarse`` (each layer's residual stream), ``fine`` (every linear and
op output of each layer) and ``retain`` (fine, plus every hashed byte stored to a pinned host ring).  Each arm runs in
its own engine process; a round runs every arm once, in a rotated order, on one GPU.

    python pu_bench.py run [--rounds 0,1] [--micro] [--verify]   # everything below, in order, into $RESEARCH_RUN_DIR/out
    python pu_bench.py micro --out OUT                  # hash kernel GB/s, tree fold, D2H/H2D pinned GB/s, per-site latency
    python pu_bench.py verify --out OUT                 # retain arm: a step's digests and root against hashlib
    python pu_bench.py arm --arm A --round R --out OUT  # one engine: throughput workload, then batch-1 latency
    python pu_bench.py summarize OUT... > summary.md    # arm x workload table over every round found

Workloads: ``vllm bench throughput``'s random dataset run through the offline API (random token ids, ``ignore_eos``,
temperature 1, top_p 1): input 1024, output 256, ``--num-prompts`` at once; then batch 1, input 1024, output 256,
TPOT = (median latency at 256 output tokens - median latency at 1) / 255, distinct prompts so prefix caching never hits.
"""

from __future__ import annotations

import argparse
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
ARMS = ("control", "coarse", "fine", "retain")


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
    big = 2 << 30
    x = torch.randint(-(1 << 15), 1 << 15, (big // 2,), dtype=torch.int16, device=dev)
    n = big // P.CHUNK
    st = P.State(ring_leaves=n + 1024, keep_bytes=0)
    t_leaf = _cuda_time(lambda: st.leaf(x, 0, 0))
    t_fold = _cuda_time(lambda: st.finalize(n, big))
    keep = P.State(ring_leaves=(1 << 30) // P.CHUNK + 1024, keep_bytes=1 << 30)
    x1 = x[: (1 << 30) // 2]
    t_keep = _cuda_time(lambda: keep.leaf(x1, 0, 0))
    src = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
    pinned = torch.empty(1 << 30, dtype=torch.uint8, pin_memory=True)
    t_d2h = _cuda_time(lambda: pinned.copy_(src, non_blocking=True))
    t_h2d = _cuda_time(lambda: src.copy_(pinned, non_blocking=True))
    out = {"gpu": gpu_info(), "versions": _versions(), "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
           "hash_leaves": {"bytes": big, "seconds": t_leaf, "GBps": big / t_leaf / 1e9},
           "tree_fold": {"leaves": n, "seconds": t_fold, "leaves_per_s": n / t_fold},
           "hash_and_fold_GBps": big / (t_leaf + t_fold) / 1e9,
           "hash_leaves_with_mapped_host_store": {"bytes": 1 << 30, "seconds": t_keep, "GBps": (1 << 30) / t_keep / 1e9},
           "d2h_pinned_dma_GBps": (1 << 30) / t_d2h / 1e9, "h2d_pinned_dma_GBps": (1 << 30) / t_h2d / 1e9}
    # per-site cost inside a CUDA graph: 288 sites (Llama-8B's fine arm) of one decode step at batch 1 (8 KiB) and 256 (2 MiB)
    per_site = {}
    for label, nbytes in (("8KiB", 8192), ("2MiB", 2 << 20)):
        xs = x[: nbytes // 2]
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
    _write(Path(args.out) / "micro.json", out)
    print(json.dumps(out, indent=1))
    return 0


def _llm(args):
    from vllm import LLM
    return LLM(model=args.model, max_num_seqs=args.max_num_seqs, worker_extension_cls="pu_ext.PUExt", seed=0)


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
    """The retain arm in the production-compiled engine: a few requests whose prefill (400 tokens, a piecewise graph) and
    decode (a full graph) steps are recomputed with hashlib from the retained bytes."""
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
                (Path(args.out) / f"inductor_{kept}_{f.name}").write_text(text)
                kept += 1
    (Path(args.out) / "inductor_verity_pu_lines.txt").write_text("\n".join(lines) + "\n")
    res["inductor_lines_with_verity_pu"] = len(lines)
    _write(Path(args.out) / "verify.json", res)
    print(json.dumps(res, indent=1, default=str))
    return 0 if res["pass"] else 1


def _child(args, mode: list[str], arm_name: str, log: Path) -> int:
    env = dict(os.environ)
    env.pop("VLLM_BATCH_INVARIANT", None)
    env.update(VLLM_PU_ARM=arm_name, PYTHONPATH=f"{HERE}{os.pathsep}{env.get('PYTHONPATH', '')}",
               VLLM_CACHE_ROOT=str(Path(args.scratch) / f"vllm-{arm_name}"),
               TORCH_EXTENSIONS_DIR=str(Path(args.scratch) / "torch-ext"), TRITON_CACHE_DIR=str(Path(args.scratch) / "triton"))
    cmd = [sys.executable, str(HERE / "pu_bench.py"), *mode, "--out", args.out, "--model", args.model,
           "--max-num-seqs", str(args.max_num_seqs), "--input-len", str(args.input_len), "--output-len", str(args.output_len),
           "--num-prompts", str(args.num_prompts), "--lat-reqs", str(args.lat_reqs)]
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
        rc |= _child(args, ["verify"], "retain", out / "logs" / "verify.log")
    arms = args.arms.split(",")
    for r in args.rounds:
        k = r % len(arms)
        order = arms[k:] + arms[:k]
        if r % 2:
            order.reverse()
        for a in order:
            rc |= _child(args, ["arm", "--arm", a, "--round", str(r)], a, out / "logs" / f"arm-r{r}-{a}.log")
    summary = summarize_dirs([out])
    (out / "summary.md").write_text(summary)
    print(summary)
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    if run_dir:
        rows = [json.loads(p.read_text()) for p in sorted(out.glob("arm-r*.json"))]
        ver = json.loads((out / "verify.json").read_text()) if (out / "verify.json").exists() else None
        result = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", "local"), "kind": "bench-result/v1",
                  "workload_fingerprint": {"workload": "vllm_pu_commit", "model": args.model, "arms": args.arms, "rounds": args.rounds,
                                           "input_len": args.input_len, "output_len": args.output_len,
                                           "num_prompts": args.num_prompts, "max_num_seqs": args.max_num_seqs},
                  "validation": {"status": "passed" if rc == 0 and (ver is None or ver["pass"]) else "failed",
                                 "checker": "every engine process exited 0" + ("; verify: a prefill and a decode step's leaves and "
                                                                               "root recomputed with hashlib" if ver else ""),
                                 "detail": f"rc={rc}" + (f", verify pass={ver['pass']}" if ver else ""), "evidence": []},
                  "measurements": [m for r in rows for m in (
                      {"name": f"r{r['round']}.{r['arm']}.output_tps", "value": r["throughput"]["output_tps"], "unit": "tok/s"},
                      {"name": f"r{r['round']}.{r['arm']}.total_tps", "value": r["throughput"]["total_tps"], "unit": "tok/s"},
                      {"name": f"r{r['round']}.{r['arm']}.tpot_b1", "value": r["batch1"]["tpot_ms"], "unit": "ms"})],
                  "measurement_files": [os.path.relpath(p, run_dir) for p in [*sorted(out.glob("*.json")), out / "summary.md"]],
                  "artifacts": []}
        (Path(run_dir) / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    return rc


def summarize_dirs(dirs: list[Path]) -> str:
    rows = [json.loads(p.read_text()) for d in dirs for p in sorted(Path(d).rglob("arm-r*.json"))]
    by_round: dict[tuple, dict] = {}
    for r in rows:
        by_round.setdefault((r["round"], r["gpu"].get("uuid")), {})[r["arm"]] = r
    metrics = {"decode-heavy output tok/s": lambda r: r["throughput"]["output_tps"],
               "decode-heavy total tok/s": lambda r: r["throughput"]["total_tps"],
               "batch-1 TPOT ms": lambda r: r["batch1"]["tpot_ms"]}
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
                w = arms[a]["batch1" if "TPOT" in name else "throughput"].get("commit") or {}
                bpt.append(w.get("bytes_per_output_token", 0))
            if not vals:
                continue
            lines.append(f"| {a} | {name} | {statistics.mean(vals):.1f} | {min(vals):.1f}..{max(vals):.1f} | "
                         f"{statistics.mean(ovh):+.2f}% ({min(ovh):+.2f}..{max(ovh):+.2f}) | "
                         f"{statistics.mean(bpt) / 1e6:.2f} MB | {len(vals)} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "micro", "verify", "arm", "summarize"))
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--out", default=str(Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out"))
    ap.add_argument("--scratch", default=None, help="compile caches and the extension build (default: a temporary directory)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--arm", choices=ARMS, default="control")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--rounds", type=lambda s: [int(x) for x in s.split(",")], default=[0])
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--num-prompts", type=int, default=1000)
    ap.add_argument("--lat-reqs", type=int, default=5)
    ap.add_argument("--micro", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if args.mode == "summarize":
        sys.stdout.write(summarize_dirs([Path(d) for d in args.dirs]))
        return 0
    if args.mode == "run" and not args.scratch:
        with tempfile.TemporaryDirectory(prefix="pu_bench-") as d:
            args.scratch = d
            return run(args)
    return {"run": run, "micro": micro, "verify": verify, "arm": arm}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
