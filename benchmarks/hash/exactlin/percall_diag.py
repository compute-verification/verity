"""Why sbench's stock-percall arm (`percall.py`) can serve other tokens than stock.  Each engine is sbench's stock engine
(compressed-tensors W8A8 over the int8 pin, Triton int8 GEMM, T = max_num_batched_tokens) in its own spawned process, and
serves sbench's warm-up, gate (B32 greedy, 16 tokens) and decode (B32 greedy at N1, then N2 tokens: the N2 tokens are
compared, as sbench's decode reps compare them).  A scenario brings its engines up one at a time and keeps them resident
together, as sbench does, then serves them in order; its engines share one Inductor, Triton and vLLM cache directory
(`shared`, as served.sh gives sbench's arms) or each have their own (`own`).

Every engine logs each Inductor kernel's autotune: its name, its source hash, the config it ran and, where it had more than
one, the timings it was chosen by (`CachingAutotuner.benchmark_all_configs`, which picks the fastest).  An engine's PICK
replaces that choice: `first` or `last` takes the first or last config untimed (`own` caches, so no `.best_config` of
another engine's is read), `det` sets TORCHINDUCTOR_DETERMINISTIC=1 (Inductor's mode that times no config that moves
numerics).  With graphs `none` (`cudagraph_mode` NONE, still compiled) every int8 GEMM call of the N2 generation is
recorded (s_a's bits, an exact per-row checksum of xq and of y) and the scenario's two engines' calls are compared in order:
the first whose xq or s_a differ (the GEMM's inputs) and the first whose y differs on equal inputs.

OUT/<scenario>-<i>-<arm>.json is each engine's own (tokens, autotune log, counters); OUT/percall_diag.json the comparisons.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PIN = "/workspace/hf/hub/models--verity-int8--Llama-3.1-8B-INT8/snapshots/723d342ac443b040931242c768df342b0f6f15a7"
B, N1, N2, GATE_TOKENS = 32, 32, 160, 16
CT_CONFIG = {
    "quant_method": "compressed-tensors", "format": "int-quantized", "quantization_status": "compressed",
    "config_groups": {"group_0": {
        "targets": ["Linear"],
        "weights": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "channel", "dynamic": False,
                    "group_size": None, "observer": "minmax", "actorder": None, "block_structure": None},
        "input_activations": {"num_bits": 8, "type": "int", "symmetric": True, "strategy": "token", "dynamic": True,
                              "group_size": None, "observer": None, "actorder": None, "block_structure": None}}},
    "ignore": ["lm_head"], "kv_cache_scheme": None}
# Llama-3.1-8B's int8 linears in call order within a layer, (K, N)
LINEARS = (("qkv", 4096, 6144), ("o", 4096, 4096), ("gate_up", 4096, 28672), ("down", 14336, 4096))
# name: (graphs, caches, ((arm, pick), ...)); sbench is served.sh's sbench (stock, then stock-percall, resident, shared)
SCENARIOS = {
    "sbench": ("full", "shared", (("stock", None), ("stock-percall", None))),
    "first": ("full", "own", (("stock", "first"), ("stock-percall", "first"))),
    "last": ("full", "own", (("stock", "last"), ("stock-percall", "last"))),
    "det": ("full", "shared", (("stock", "det"), ("stock-percall", "det"))),
    "sbench-none": ("none", "shared", (("stock", None), ("stock-percall", None))),
    "pick-none": ("none", "own", (("stock", "first"), ("stock", "last"))),
}


def prompts(n: int) -> list[str]:
    from tests.engine.int8_exact_gpu import PROMPTS
    return [PROMPTS[i % len(PROMPTS)] for i in range(n)]


def stock_checkpoint(scratch: Path) -> Path:
    """The pin's files linked into SCRATCH, its config's quantization_config replaced by compressed-tensors W8A8's."""
    d = scratch / "stock-ckpt"
    d.mkdir(parents=True, exist_ok=True)
    for f in Path(PIN).iterdir():
        if f.name != "config.json" and not (d / f.name).exists():
            (d / f.name).symlink_to(f.resolve())
    cfg = json.loads((Path(PIN) / "config.json").read_text())
    cfg["quantization_config"] = CT_CONFIG
    (d / "config.json").write_text(json.dumps(cfg, indent=1))
    return d


class Recorder:
    """Per int8 GEMM call while `on`: (K, N, M) and M s_a bits, M xq checksums, M y checksums (int64, exact)."""

    def __init__(self):
        self.on = False
        self.kn: list[tuple[int, int, int]] = []
        self.data: list[np.ndarray] = []
        self.short = 0
        self.wv = None

    def take(self, xq, s_a, y) -> None:
        import torch
        m, k = xq.shape
        n = y.shape[1]
        if self.wv is None:
            g = torch.Generator().manual_seed(1)
            self.wv = torch.randint(1, 1 << 20, (32768,), generator=g, dtype=torch.int64).to(xq.device)
        sa = s_a.reshape(-1).contiguous().view(torch.int32).to(torch.int64)
        sa = sa.expand(m) if sa.numel() == 1 else sa
        xc = (xq.to(torch.int64) * self.wv[:k]).sum(1)
        yc = (y.contiguous().view(torch.int16).to(torch.int64) * self.wv[:n]).sum(1)
        self.kn.append((k, n, m))
        self.data.append(torch.cat([sa, xc, yc]).cpu().numpy())

    def install_stock(self) -> None:
        """Stock's GEMM is the traced `scaled_mm_kernel`, launched from Inductor's code (`CachingAutotuner.run`)."""
        from torch._inductor.runtime import triton_heuristics as th
        orig, rec = th.CachingAutotuner.run, self

        def run(self_, *args, **kw):
            out = orig(self_, *args, **kw)
            if rec.on and "scaled_mm_kernel" in getattr(self_.fn, "__name__", ""):
                import torch
                ts = [a for a in args if isinstance(a, torch.Tensor)]
                if len(ts) >= 5 and ts[0].dtype == torch.int8 and ts[2].dtype == torch.float32 and ts[4].dtype == torch.bfloat16:
                    rec.take(ts[0], ts[2], ts[4])
                else:
                    rec.short += 1
            return out

        th.CachingAutotuner.run = run

    def install_percall(self) -> None:
        from vllm.model_executor.layers.quantization.compressed_tensors import triton_scaled_mm as M
        orig, rec = M.triton_scaled_mm, self

        def wrapped(x_q, w_q, scale_a, scale_b, out_dtype, *a, **kw):
            y = orig(x_q, w_q, scale_a, scale_b, out_dtype, *a, **kw)
            if rec.on:
                rec.take(x_q, scale_a, y)
            return y

        M.triton_scaled_mm = wrapped

    def save(self, path: Path) -> dict:
        off = np.cumsum([0] + [len(d) for d in self.data])
        np.savez(path, kn=np.array(self.kn, dtype=np.int64).reshape(-1, 3), off=off,
                 data=np.concatenate(self.data) if self.data else np.zeros(0, np.int64))
        return {"calls": len(self.kn), "short_args": self.short}


def install_tuning_log(pick: str | None) -> list[dict]:
    """Each Inductor kernel's autotune, in order of first launch; with PICK first|last its choice made untimed.  A kernel
    with one config (or one read from the autotune cache) is never timed: its entry has no `timings_ms`.  The log never
    raises into Inductor: an entry it can't read records the error."""
    from torch._inductor.runtime import triton_heuristics as th
    log: list[dict] = []
    seen: dict[int, tuple] = {}   # id -> (autotuner, entry): the autotuner kept, so its id is never reused
    orig_run, orig_bench = th.CachingAutotuner.run, th.CachingAutotuner.benchmark_all_configs

    def entry(self_) -> dict:
        got = seen.get(id(self_))
        if got is None:
            try:
                fn = getattr(self_, "fn", None)
                src = getattr(fn, "src", "") or ""
                e = {"kernel": (getattr(self_, "inductor_meta", None) or {}).get("kernel_name") or getattr(fn, "__name__", "?"),
                     "src": hashlib.sha1(src.encode()).hexdigest()[:10]}
            except Exception as x:  # noqa: BLE001
                e = {"kernel": "?", "error": repr(x)}
            got = seen[id(self_)] = (self_, e)
            log.append(e)
        return got[1]

    def bench(self_, *args, **kw):
        launchers = list(getattr(self_, "launchers", None) or ())
        if pick in ("first", "last") and launchers:
            n = len(launchers)
            t = {ln: float(i if pick == "first" else n - i) for i, ln in enumerate(launchers)}
        else:
            t = orig_bench(self_, *args, **kw)
        try:
            entry(self_)["timings_ms"] = [[str(ln.config), round(float(v), 5)] for ln, v in t.items()]
        except Exception as x:  # noqa: BLE001
            entry(self_)["error"] = repr(x)
        return t

    def run(self_, *args, **kw):
        out = orig_run(self_, *args, **kw)
        try:
            e = entry(self_)
            launchers = getattr(self_, "launchers", None) or ()
            if "ran" not in e and len(launchers) == 1:
                e["ran"] = str(launchers[0].config)
        except Exception as x:  # noqa: BLE001
            entry(self_)["error"] = repr(x)
        return out

    th.CachingAutotuner.benchmark_all_configs = bench
    th.CachingAutotuner.run = run
    return log


def worker(spec: dict, conn) -> None:
    """One engine: SPEC's env set before torch is imported, built, then served on the parent's `go`."""
    os.environ.update(spec["env"])
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_DISABLED_KERNELS"] = "CutlassInt8ScaledMMLinearKernel"
    sys.path.insert(0, str(HERE))
    try:
        import torch
        from vllm import LLM, SamplingParams
        tuning = install_tuning_log(spec["pick"])
        rec = Recorder() if spec["graphs"] == "none" else None
        if spec["arm"] == "stock-percall":
            import percall
            percall.install()
        if rec is not None:
            (rec.install_stock if spec["arm"] == "stock" else rec.install_percall)()
        # no debug_dump_path: it runs depyf.prepare_debug, whose load_by_key_path patch this torch's Inductor refuses
        cc: dict = {"cudagraph_mode": "NONE"} if spec["graphs"] == "none" else {}
        t = spec["t"]
        llm = LLM(model=str(stock_checkpoint(Path(spec["scratch"]))), compilation_config=cc, max_num_seqs=min(256, t),
                  max_num_batched_tokens=t, gpu_memory_utilization=spec["gpu_mem"], enable_prefix_caching=False,
                  max_model_len=2048, seed=0, kernel_config={"enable_flashinfer_autotune": False})
        from vllm.compilation.counter import compilation_counter
        counters = dataclasses.asdict(compilation_counter)
        import torch._inductor.config as icfg
        conn.send({"ready": True})
    except Exception as e:  # noqa: BLE001
        conn.send({"error": f"build: {type(e).__name__}: {e}"})
        return
    try:
        assert conn.recv() == "go"
        ps = prompts(B)

        def gen(reqs, n):
            sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True, seed=0)
            return [list(o.outputs[0].token_ids) for o in llm.generate(reqs, sp, use_tqdm=False)]

        gen(ps, 8)
        gen(prompts(64), 16)
        gate = gen(ps, GATE_TOKENS)
        gen(ps, N1)
        if rec is not None:
            rec.on = True
        toks = gen(ps, N2)
        torch.cuda.synchronize()
        doc = {**{k: spec[k] for k in ("scenario", "index", "arm", "pick", "graphs", "caches", "t", "gpu_mem")},
               "tokens": toks, "gate_tokens": gate, "counters": counters, "tuning": tuning,
               "torch": torch.__version__, "inductor_deterministic": getattr(icfg, "deterministic", "absent"),
               "gpu": torch.cuda.get_device_name()}
        if spec["arm"] == "stock-percall":
            import percall
            doc["percall"] = percall.summary()
        if rec is not None:
            rec.on = False
            doc["record"] = rec.save(Path(spec["record"]))
        Path(spec["json"]).write_text(json.dumps(doc, indent=1, default=str))
        conn.send({"done": True})
        assert conn.recv() == "stop"
    except Exception as e:  # noqa: BLE001
        conn.send({"error": f"serve: {type(e).__name__}: {e}"})


def token_diff(a: list, b: list) -> dict:
    first = [next((i for i, (x, y) in enumerate(zip(p, q)) if x != y), None) for p, q in zip(a, b)]
    return {"requests_equal": sum(f is None for f in first), "requests": len(first),
            "first_diff": {str(r): f for r, f in enumerate(first) if f is not None}}


def tuning_diff(a: list[dict], b: list[dict]) -> dict:
    """Kernels (by name, in first-launch order) whose two engines ran other configs; and which had more than one."""
    ra, rb = {}, {}
    for e in a:
        ra.setdefault(e["kernel"], []).append(e.get("ran"))
    for e in b:
        rb.setdefault(e["kernel"], []).append(e.get("ran"))
    differ = {k: [ra[k], rb[k]] for k in ra if k in rb and ra[k] != rb[k]}
    timed = sorted({e["kernel"] for e in a + b if len(e.get("timings_ms", ())) > 1})
    return {"kernels_ran_other_config": differ, "kernels_with_configs": timed,
            "shared_source": sorted({e["kernel"] for e in a} & {e["kernel"] for e in b}
                                    & {e["kernel"] for e in a if any(e["src"] == f["src"] for f in b)})}


def where(i: int) -> dict:
    per = len(LINEARS)
    return {"call": i, "step": i // (32 * per), "layer": (i // per) % 32, "linear": LINEARS[i % per][0]}


def compare_records(pa: Path, pb: Path) -> dict:
    """The first call (in order) whose GEMM inputs (s_a, xq) differ, and the first whose y differs on equal inputs."""
    a, b = dict(np.load(pa)), dict(np.load(pb))
    n = min(len(a["kn"]), len(b["kn"]))
    out: dict = {"calls": [int(len(a["kn"])), int(len(b["kn"]))], "first_inputs": None, "first_y_equal_inputs": None,
                 "calls_inputs_differ": 0, "calls_y_differ": 0, "calls_y_differ_equal_inputs": 0}
    for i in range(n):
        if not np.array_equal(a["kn"][i], b["kn"][i]):
            out["schedule_diverges"] = {**where(i), "kn": [a["kn"][i].tolist(), b["kn"][i].tolist()]}
            break
        m = int(a["kn"][i][2])
        da, db = a["data"][a["off"][i]:a["off"][i + 1]], b["data"][b["off"][i]:b["off"][i + 1]]
        sa = np.nonzero(da[:m] != db[:m])[0]
        xq = np.nonzero(da[m:2 * m] != db[m:2 * m])[0]
        yy = np.nonzero(da[2 * m:] != db[2 * m:])[0]
        inputs = bool(len(sa) or len(xq))
        if inputs:
            out["calls_inputs_differ"] += 1
            if out["first_inputs"] is None:
                def f32(v):
                    return float(np.array([v], np.int64).astype(np.int32).view(np.float32)[0])
                out["first_inputs"] = {**where(i), "m": m, "rows_s_a": sa.tolist()[:32], "rows_xq": xq.tolist()[:32],
                                       "s_a": [[f32(da[r]), f32(db[r])] for r in sa.tolist()[:4]]}
        if len(yy):
            out["calls_y_differ"] += 1
            if not inputs:
                out["calls_y_differ_equal_inputs"] += 1
                if out["first_y_equal_inputs"] is None:
                    out["first_y_equal_inputs"] = {**where(i), "m": m, "rows_y": yy.tolist()[:32]}
    return out


def run_scenario(name: str, a, out: Path, scratch: Path) -> dict:
    """NAME's engines up one at a time and resident together, then served in order; each's rc-like status."""
    graphs, caches, engines = SCENARIOS[name]
    ctx = mp.get_context("spawn")
    procs, conns, status = [], [], {}
    for i, (arm, pick) in enumerate(engines):
        tag = f"{name}-{i}-{arm}"
        cdir = scratch / name / ("shared" if caches == "shared" else tag)
        env = {"VLLM_CACHE_ROOT": str(cdir / "vllm"), "TORCHINDUCTOR_CACHE_DIR": str(cdir / "inductor"),
               "TRITON_CACHE_DIR": str(cdir / "triton"), "VLLM_DISABLE_COMPILE_CACHE": "1"}
        if pick == "det":
            env["TORCHINDUCTOR_DETERMINISTIC"] = "1"
        spec = {"scenario": name, "index": i, "arm": arm, "pick": pick, "graphs": graphs, "caches": caches, "t": a.t,
                "gpu_mem": a.gpu_mem, "env": env, "scratch": str(scratch), "json": str(out / f"{tag}.json"),
                "record": str(scratch / f"{tag}.npz")}
        here, there = ctx.Pipe()
        p = ctx.Process(target=worker, args=(spec, there), daemon=True)
        p.start()
        there.close()
        msg = _recv(here)
        status[tag] = msg.get("error", "ready")
        print(f"percall_diag: {tag} {status[tag]}", flush=True)
        procs.append(p)
        conns.append(here)
        if "error" in msg:
            break
    for (tag, st), c in zip(list(status.items()), conns):
        if st == "ready":
            c.send("go")
            msg = _recv(c)
            status[tag] = msg.get("error", "done")
            print(f"percall_diag: {tag} {status[tag]}", flush=True)
    for c, p in zip(conns, procs):
        try:
            c.send("stop")
        except (BrokenPipeError, OSError):
            pass
        p.join()
    return status


def _recv(conn) -> dict:
    try:
        return conn.recv()
    except EOFError:
        return {"error": "the worker exited"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--t", type=int, default=256, help="max_num_batched_tokens (sbench's SBENCH_T)")
    ap.add_argument("--gpu-mem", type=float, default=0.4, help="each resident engine's (sbench's default)")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    a = ap.parse_args()
    out, scratch = Path(a.out), Path(a.scratch)
    out.mkdir(parents=True, exist_ok=True)
    status = {}
    for name in a.scenarios.split(","):
        status.update(run_scenario(name, a, out, scratch))
    docs = {p.stem: json.loads(p.read_text()) for p in sorted(out.glob("*-[0-9]-*.json"))}
    summary: dict = {"t": a.t, "gpu_mem": a.gpu_mem, "status": status, "pairs": {}, "tokens": {}}
    for name in a.scenarios.split(","):
        tags = [t for t in docs if docs[t]["scenario"] == name]
        if len(tags) == 2:
            x, y = (docs[t] for t in tags)
            summary["pairs"][name] = {"engines": tags, "tokens": token_diff(x["tokens"], y["tokens"]),
                                      "gate_tokens": token_diff(x["gate_tokens"], y["gate_tokens"])["requests_equal"],
                                      "tuning": tuning_diff(x["tuning"], y["tuning"]),
                                      "counters_differ": {k: [x["counters"].get(k), y["counters"].get(k)] for k in x["counters"]
                                                          if x["counters"].get(k) != y["counters"].get(k)}}
            pa, pb = (scratch / f"{t}.npz" for t in tags)
            if SCENARIOS[name][0] == "none" and pa.exists() and pb.exists():
                summary["pairs"][name]["records"] = compare_records(pa, pb)
    tags = sorted(docs)
    for i, x in enumerate(tags):
        for y in tags[i + 1:]:
            if docs[x]["graphs"] == docs[y]["graphs"]:
                summary["tokens"][f"{x} vs {y}"] = token_diff(docs[x]["tokens"], docs[y]["tokens"])["requests_equal"]
    summary["versions"] = {t: [d["torch"], d["inductor_deterministic"]] for t, d in docs.items()}
    (out / "percall_diag.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary["status"]), flush=True)
    print(json.dumps({k: {"tokens": v["tokens"]["requests_equal"], "ran_other": len(v["tuning"]["kernels_ran_other_config"])}
                      for k, v in summary["pairs"].items()}), flush=True)
    return 0 if all(s == "done" for s in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
