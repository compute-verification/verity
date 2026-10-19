"""Why sbench's stock-percall arm (`percall.py`) can serve other tokens than stock: each arm (stock, stock-percall) builds
sbench's stock engine (compressed-tensors W8A8 over the int8 pin, Triton int8 GEMM, T = max_num_batched_tokens) in its own
process, once with vLLM's default CUDA graphs and once without (`cudagraph_mode` NONE, still compiled), and serves sbench's
decode gate (its warm-up, then B32 greedy at N1 and N2 tokens; the N2 tokens are compared).

Without graphs every int8 GEMM call is recorded during the N2 generation: s_a's bits and an exact per-row checksum of xq
and of y.  Stock's GEMM is the traced `scaled_mm_kernel`, launched from Inductor's code (`CachingAutotuner.run`), and its y
is also recomputed eagerly from the same xq and s_a, at the heuristic's tile for the call's M (as stock-percall runs it)
and at the tile frozen at the traced M; stock-percall's is the op body's `triton_scaled_mm`.  The calls are compared in
order: the first whose xq or s_a differ (the GEMM's inputs) and the first whose y differs on equal inputs (the GEMM).

Also per arm: vLLM's compilation counters, and counts of the ops and kernels in Inductor's output code and vLLM's graph
dump.  OUT/percall_diag.json has the comparisons; OUT/<arm>-<graphs>.json each process's own.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PIN = "/workspace/hf/hub/models--verity-int8--Llama-3.1-8B-INT8/snapshots/723d342ac443b040931242c768df342b0f6f15a7"
B, N1, N2 = 32, 32, 160
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
CODE_TOKENS = ("scaled_mm_kernel", "exactlin_stock.scaled_mm", "dynamic_scaled_int8_quant", "rms_norm", "fused_add_rms_norm",
               "silu_and_mul", "rotary_embedding", "unified_attention", "def triton_poi_", "def triton_red_",
               "def triton_per_", "extern_kernels.mm", "user_autotune")


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


def vllm_tile(m: int, n: int) -> tuple[int, int, int]:
    """triton_scaled_mm's heuristic tile at d9105ea80."""
    p = max(32, 1 << (m - 1).bit_length())
    if p <= 32:
        return (64, 64, 256) if n < 8192 else (64, 128, 256)
    return (64, 64, 256) if p <= 64 else (64, 128, 128) if p <= 128 else (128, 128, 128)


class Recorder:
    """Per int8 GEMM call while `on`: (K, N, M) and M s_a bits, M xq checksums, M y checksums (int64, exact)."""

    def __init__(self, t: int, eager: bool):
        self.on, self.t, self.eager = False, t, eager
        self.kn: list[tuple[int, int, int]] = []
        self.data: list[np.ndarray] = []
        self.eager_diff: list[tuple[int, int, int]] = []   # (call, elements != heuristic-tile y, != frozen-tile y)
        self.names: set[str] = set()
        self.short = 0
        self.wv = None

    def take(self, xq, w, s_a, s_w, y) -> None:
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
        if self.eager:
            import kgate
            from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import triton_scaled_mm
            yh = triton_scaled_mm(xq, w, s_a, s_w, torch.bfloat16)
            yf = kgate.scaled_mm_at(xq, w, s_a, s_w, vllm_tile(self.t, n))
            dh = int((yh.view(torch.int16) != y.view(torch.int16)).sum())
            df = int((yf.view(torch.int16) != y.view(torch.int16)).sum())
            if dh or df:
                self.eager_diff.append((len(self.kn), dh, df))
        self.kn.append((k, n, m))
        self.data.append(torch.cat([sa, xc, yc]).cpu().numpy())

    def install_stock(self) -> None:
        from torch._inductor.runtime import triton_heuristics as th
        orig, rec = th.CachingAutotuner.run, self

        def run(self_, *args, **kw):
            out = orig(self_, *args, **kw)
            if rec.on:
                name = f"{getattr(self_.fn, '__name__', '')}/{(getattr(self_, 'inductor_meta', None) or {}).get('kernel_name', '')}"
                rec.names.add(name)
                if "scaled_mm_kernel" in name:
                    import torch
                    ts = [a for a in args if isinstance(a, torch.Tensor)]
                    if len(ts) >= 5 and ts[0].dtype == torch.int8 and ts[2].dtype == torch.float32 and ts[4].dtype == torch.bfloat16:
                        rec.take(ts[0], ts[1], ts[2], ts[3], ts[4])
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
                rec.take(x_q, w_q, scale_a, scale_b, y)
            return y

        M.triton_scaled_mm = wrapped

    def save(self, path: Path) -> dict:
        off = np.cumsum([0] + [len(d) for d in self.data])
        np.savez(path, kn=np.array(self.kn, dtype=np.int64).reshape(-1, 3), off=off,
                 data=np.concatenate(self.data) if self.data else np.zeros(0, np.int64))
        return {"calls": len(self.kn), "short_args": self.short, "kernel_names": sorted(self.names)[:40],
                "eager_diff_calls": len(self.eager_diff), "eager_diff_first": self.eager_diff[:8],
                "eager_diff_elements": [sum(d[1] for d in self.eager_diff), sum(d[2] for d in self.eager_diff)]}


def code_stats(root: Path, tar: Path | None) -> dict:
    """Counts of CODE_TOKENS over the .py files under ROOT; Inductor's wrappers (`def call(`) counted apart and, with TAR, kept."""
    allc = dict.fromkeys(CODE_TOKENS, 0)
    wrap = dict.fromkeys(CODE_TOKENS, 0)
    files, wrappers = 0, []
    for p in sorted(root.rglob("*.py")) if root.is_dir() else []:
        try:
            s = p.read_text(errors="replace")
        except OSError:
            continue
        files += 1
        is_wrap = "def call(" in s
        if is_wrap:
            wrappers.append(p)
        for t in CODE_TOKENS:
            c = s.count(t)
            allc[t] += c
            if is_wrap:
                wrap[t] += c
    if tar is not None and wrappers and sum(p.stat().st_size for p in wrappers) < 30 << 20:
        with tarfile.open(tar, "w:gz") as tf:
            for p in wrappers:
                tf.add(p, arcname=str(p.relative_to(root)))
    return {"files": files, "wrappers": len(wrappers), "all": allc, "wrappers_tokens": wrap}


def worker(a) -> int:
    tag = f"{a.arm}-{a.graphs}"
    s = Path(a.scratch) / tag
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_DISABLED_KERNELS"] = "CutlassInt8ScaledMMLinearKernel"
    sys.path.insert(0, str(HERE))
    import torch
    from vllm import LLM, SamplingParams
    rec = Recorder(a.t, eager=a.arm == "stock") if a.graphs == "none" else None
    if a.arm == "stock-percall":
        import percall
        percall.install()
    if rec is not None:
        (rec.install_stock if a.arm == "stock" else rec.install_percall)()
    cc: dict = {"debug_dump_path": str(s / "dump")}
    if a.graphs == "none":
        cc["cudagraph_mode"] = "NONE"
    llm = LLM(model=str(stock_checkpoint(Path(a.scratch))), compilation_config=cc, max_num_seqs=min(256, a.t),
              max_num_batched_tokens=a.t, gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=False,
              max_model_len=2048, seed=0, kernel_config={"enable_flashinfer_autotune": False})
    from vllm.compilation.counter import compilation_counter
    counters = dataclasses.asdict(compilation_counter)
    ps = prompts(B)

    def gen(reqs, n):
        sp = SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True, seed=0)
        return llm.generate(reqs, sp, use_tqdm=False)

    gen(ps, 8)
    gen(prompts(64), 16)
    gen(ps, N1)
    if rec is not None:
        rec.on = True
    outs = gen(ps, N2)
    torch.cuda.synchronize()
    if rec is not None:
        rec.on = False
    doc = {"arm": a.arm, "graphs": a.graphs, "t": a.t, "tokens": [list(o.outputs[0].token_ids) for o in outs],
           "counters": counters, "inductor": code_stats(Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]), Path(a.out) / f"code-{tag}.tar.gz"),
           "dump": code_stats(s / "dump", None), "gpu": torch.cuda.get_device_name()}
    if a.arm == "stock-percall":
        import percall
        doc["percall"] = percall.summary()
    if rec is not None:
        doc["record"] = rec.save(s / "record.npz")
    (Path(a.out) / f"{tag}.json").write_text(json.dumps(doc, indent=1, default=str))
    print(f"{tag}: {len(doc['tokens'])} requests, record {doc.get('record', {}).get('calls')}")
    return 0


def token_diff(a: list, b: list) -> dict:
    first = [next((i for i, (x, y) in enumerate(zip(p, q)) if x != y), None) for p, q in zip(a, b)]
    return {"requests_equal": sum(f is None for f in first), "requests": len(first),
            "first_diff": {str(r): f for r, f in enumerate(first) if f is not None}}


def where(i: int) -> dict:
    per = len(LINEARS)
    name, k, n = LINEARS[i % per]
    return {"call": i, "step": i // (32 * per), "layer": (i // per) % 32, "linear": name}


def compare_records(pa: Path, pb: Path) -> dict:
    """The first call (in order) whose GEMM inputs (s_a, xq) differ, and the first whose y differs on equal inputs."""
    a, b = np.load(pa), np.load(pb)
    n = min(len(a["kn"]), len(b["kn"]))
    out: dict = {"calls": [int(len(a["kn"])), int(len(b["kn"]))], "first_inputs": None, "first_y_equal_inputs": None,
                 "calls_inputs_differ": 0, "calls_y_differ": 0}
    for i in range(n):
        if not np.array_equal(a["kn"][i], b["kn"][i]):
            out["schedule_diverges"] = {**where(i), "kn": [a["kn"][i].tolist(), b["kn"][i].tolist()]}
            break
        m = int(a["kn"][i][2])
        da, db = a["data"][a["off"][i]:a["off"][i + 1]], b["data"][b["off"][i]:b["off"][i + 1]]
        sa = np.nonzero(da[:m] != db[:m])[0]
        xq = np.nonzero(da[m:2 * m] != db[m:2 * m])[0]
        yy = np.nonzero(da[2 * m:] != db[2 * m:])[0]
        if len(sa) or len(xq):
            out["calls_inputs_differ"] += 1
            if out["first_inputs"] is None:
                f32 = lambda v: float(np.array([v], np.int64).astype(np.int32).view(np.float32)[0])
                out["first_inputs"] = {**where(i), "m": m, "rows_s_a": sa.tolist()[:32], "rows_xq": xq.tolist()[:32],
                                       "s_a": [[f32(da[r]), f32(db[r])] for r in sa.tolist()[:4]]}
        if len(yy):
            out["calls_y_differ"] += 1
            if out["first_y_equal_inputs"] is None and not (len(sa) or len(xq)):
                out["first_y_equal_inputs"] = {**where(i), "m": m, "rows_y": yy.tolist()[:32]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--t", type=int, default=256, help="max_num_batched_tokens (sbench's SBENCH_T)")
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    ap.add_argument("--arms", default="stock,stock-percall")
    ap.add_argument("--graphs", default="full,none", help="full: vLLM's default CUDA graphs; none: cudagraph_mode NONE, recorded")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--arm")
    a = ap.parse_args()
    if a.worker:
        return worker(a)
    out, scratch = Path(a.out), Path(a.scratch)
    out.mkdir(parents=True, exist_ok=True)
    arms, graphs = a.arms.split(","), a.graphs.split(",")
    rcs = {}
    for g in graphs:
        for arm in arms:
            tag = f"{arm}-{g}"
            s = scratch / tag
            env = {**os.environ, "VLLM_CACHE_ROOT": str(s / "vllm-cache"), "TORCHINDUCTOR_CACHE_DIR": str(s / "inductor"),
                   "TRITON_CACHE_DIR": str(s / "triton"), "VLLM_DISABLE_COMPILE_CACHE": "1"}
            cmd = [sys.executable, __file__, "--worker", "--arm", arm, "--graphs", g, "--out", str(out), "--scratch", str(scratch),
                   "--t", str(a.t), "--gpu-mem", str(a.gpu_mem)]
            with open(out / f"{tag}.log", "w") as log:
                rcs[tag] = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            print(f"percall_diag: {tag} rc {rcs[tag]}", flush=True)
    docs = {t: json.loads((out / f"{t}.json").read_text()) for t in rcs if (out / f"{t}.json").exists()}
    summary: dict = {"t": a.t, "rcs": rcs, "tokens": {}, "counters_differ": {}, "inductor": {}, "dump": {}}
    tags = sorted(docs)
    for i, x in enumerate(tags):
        for y in tags[i + 1:]:
            summary["tokens"][f"{x} vs {y}"] = token_diff(docs[x]["tokens"], docs[y]["tokens"])
    for g in graphs:
        p, q = docs.get(f"stock-{g}"), docs.get(f"stock-percall-{g}")
        if p and q:
            summary["counters_differ"][g] = {k: [p["counters"].get(k), q["counters"].get(k)] for k in p["counters"]
                                             if p["counters"].get(k) != q["counters"].get(k)}
    for t, d in docs.items():
        summary["inductor"][t] = d["inductor"]["wrappers_tokens"]
        summary["dump"][t] = d["dump"]["all"]
        if "record" in d:
            summary.setdefault("record", {})[t] = d["record"]
    ra, rb = scratch / "stock-none" / "record.npz", scratch / "stock-percall-none" / "record.npz"
    if ra.exists() and rb.exists():
        summary["records"] = compare_records(ra, rb)
    (out / "percall_diag.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("rcs", "tokens")}, default=str)[:2000])
    print(json.dumps(summary.get("records"), default=str)[:1500])
    return 0 if all(r == 0 for r in rcs.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
