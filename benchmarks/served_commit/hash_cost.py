"""The hiding-leaf commitment's cost on one GPU, in the served kernels (native_rows.cu's split form, the H200's: rows_leaf_salt_kernel,
then rows_leaf_data_kernel in VERITY_ROWS_PHASES launches): hm96-sha512 rows of each ROWS size over one stream of TOTAL_GB, alone; the
Merkle tree over those leaves in two forms (rows_level_kernel a level a launch, rows_levels_kernel ten levels a launch); and the leaves
beside the served int8 GEMM, Llama-3.2-1B's 64 linears of one forward at T tokens (int8_fused.scaled_mm at the served path's own config,
replayed from one CUDA graph on a high-priority stream), at a range of hash shares: CTA sizes THREADS (the source's THREADS, with
min blocks 512 / THREADS so every form keeps the 128-register cap) and caps of CTAs a launch. Each form's root of a small padded domain
is checked against step_rows before it is timed, and the two tree forms' roots against each other.  Measurement only; JSON to
$OUT/hash_cost.json.  No scheme or committed byte changes: THREADS changes only the launch geometry.

    bash benchmarks/served_commit/hash_cost.sh   (ROWS default 4096 16384 65536; TOTAL_GB 16; THREADS 256 128 64; WARPS, the hash warps an
                                                  SM each cap aims at, default 1 2 4 8 (and 0, the whole grid); T 8192; HASH_MS 30)

Beside a GEMM loop of N forwards, one event after each, the hash's H bytes start with the loop (one event releases both streams) and
must finish before it: hash GB/s = H / its time; the GEMM's loss is delta = loop time - N x its time alone; kappa, the GEMM's rate in the
hash's window over its rate alone, is the forwards done by the hash's end (interpolated between events) over that window's length in
forwards.  h, the MAC-times a committed byte costs, is delta x the ruler (4,096 MACs an SM-clock x SMs x the SM clock) / H beside, and
the hash's own time x the ruler / H alone.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from verity_vllm.commit import step_rows
from verity_vllm.commit.committer.native_jit import CUDA_CFLAGS

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "integrations/vllm/verity_vllm/commit/committer/native_rows.cu"
LAYER = (("qkv", 3072, 2048), ("o", 2048, 2048), ("gate_up", 16384, 2048), ("down", 2048, 8192))   # Llama-3.2-1B (N, K)
LAYERS = 16
MACS_SM_CLK = 4096                                     # int8 dense MACs an SM-clock (the P1a ruler)
c_p, c_u64 = ctypes.c_void_p, ctypes.c_uint64


def source(threads: int) -> str:
    """native_rows.cu at `threads` a CTA, every kernel's min blocks scaled so its register cap stays 65536 / 512 = 128."""
    s = SRC.read_text()
    if threads == 256:
        return s
    minb = 512 // threads
    for old, new in (("constexpr int THREADS = 256;", f"constexpr int THREADS = {threads};"),
                     ("__launch_bounds__(THREADS, 2)\n    rows_leaf_salt_kernel", f"__launch_bounds__(THREADS, {minb})\n    rows_leaf_salt_kernel")):
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    return s


def build(tmp: Path, arch: str, threads: int) -> dict[str, Any]:
    cu = tmp / f"native_rows_t{threads}.cu"
    cu.write_text(source(threads))
    so = tmp / f"rows_t{threads}.so"
    p = subprocess.run(["nvcc", *CUDA_CFLAGS, "-std=c++17", "-shared", "-Xcompiler", "-fPIC", f"-gencode=arch=compute_{arch},code=sm_{arch}",
                        "-Xptxas", "-v", f"-DVERITY_ROWS_MIN_BLOCKS={512 // threads}", f"-I{SRC.parent}", str(cu), "-o", str(so)],
                       capture_output=True, text=True, check=False)
    log = p.stderr + p.stdout
    regs = {}
    cur = None
    for line in log.splitlines():
        if "Compiling entry function" in line or "Function properties for" in line:
            cur = next((k for k in ("rows_leaf_salt_kernel", "rows_leaf_data_kernel", "rows_level_kernel", "rows_levels_kernel")
                        if k in line), None)
        elif cur and "Used" in line and "registers" in line:
            regs.setdefault(cur, line.strip())
        elif cur and "spill" in line:
            regs.setdefault(cur + ":spill", line.strip())
    return {"rc": p.returncode, "so": str(so), "ptxas": regs, "log": log[-1500:] if p.returncode else ""}


class Rows:
    """native_rows.cu's extern "C" entry points through ctypes."""

    def __init__(self, so: str, threads: int) -> None:
        L = ctypes.CDLL(so)
        L.verity_rows_params_bytes.restype = ctypes.c_int
        L.verity_rows_build.argtypes = [ctypes.c_char_p] * 6 + [ctypes.c_int, ctypes.c_uint32, c_p]
        L.verity_rows_leaves_sf_launch.argtypes = [c_p, c_p, c_u64, c_u64, c_p, c_p, c_u64, c_u64, c_u64, c_p, c_p, c_p, c_p, c_u64, c_p]
        L.verity_rows_level_launch.argtypes = [c_p, c_p, c_p, c_u64, ctypes.c_int, c_u64, c_u64, c_p]
        L.verity_rows_levels_launch.argtypes = [c_p, c_p, c_p, c_u64, ctypes.c_int, c_u64, ctypes.c_int, c_p]
        self.L, self.threads = L, threads
        self.stride0 = False                # STRIDE0: every thread hashes row 0 (the same compressions, no row traffic past L1)
        self.tn = torch.frombuffer(bytearray(step_rows.t_nibbles()), dtype=torch.uint8).cuda()

    def params(self, dom: Any, ks: bytes, rb: int) -> ctypes.Array:
        buf = ctypes.create_string_buffer(self.L.verity_rows_params_bytes())
        schema = step_rows.SCHEMA.encode()
        rc = self.L.verity_rows_build(dom.domain_id, ks, step_rows.row_prefix(8 * rb), step_rows.INST.salt_prefix,
                                      step_rows.INST.leaf_prefix(step_rows.KEY), schema, len(schema), rb, buf)
        assert rc == 0, rc
        return buf

    def leaves(self, P, rows: torch.Tensor, rb: int, n_rows: int, m: int, out: torch.Tensor, scratch: torch.Tensor,
               s: torch.cuda.Stream, cap: int = 0) -> None:
        rc = self.L.verity_rows_leaves_sf_launch(P, rows.data_ptr(), 0 if self.stride0 else rb, n_rows, None, self.tn.data_ptr(), 0, m,
                                                 n_rows, out.data_ptr(), None, None, scratch.data_ptr(), cap, s.cuda_stream)
        assert rc == 0, rc

    def tree_level(self, P, lv: torch.Tensor, m: int, s: torch.cuda.Stream, cap: int = 0) -> None:
        """lv [2m - 1, 64]: the m leaves, then each level; one launch a level."""
        off, n, level = 0, m, 0
        while n > 1:
            rc = self.L.verity_rows_level_launch(P, lv[off].data_ptr(), lv[off + n].data_ptr(), n // 2, level, 0, cap, s.cuda_stream)
            assert rc == 0, rc
            off, n, level = off + n, n // 2, level + 1

    def tree_levels(self, P, lv: torch.Tensor, m: int, s: torch.cuda.Stream) -> None:
        """The same levels, ten a launch (rows_levels_kernel): the last launch takes what is left."""
        off, n, level = 0, m, 0
        while n > 1:
            k = min(10, n.bit_length() - 1)
            rc = self.L.verity_rows_levels_launch(P, lv[off].data_ptr(), lv[off + n].data_ptr(), n, level, 0, k, s.cuda_stream)
            assert rc == 0, rc
            for _ in range(k):
                off, n, level = off + n, n // 2, level + 1


def domain(n: int, rb: int) -> Any:
    return step_rows.domain(b"p" * 32, b"c" * 32, b"g" * 32, n, 8 * rb)


def check(R: Rows, rb: int) -> dict[str, bool]:
    """A 5-row domain (width 8, three pads): the split form's leaves under both tree forms give step_rows' root."""
    n, ks = 5, os.urandom(step_rows.KEY_BYTES)
    data = os.urandom(n * rb)
    rows = [data[i * rb:(i + 1) * rb] for i in range(n)]
    dom = domain(n, rb)
    want = step_rows.levels(dom, rows, ks, 8 * rb)[-1][0]
    P, s = R.params(dom, ks, rb), torch.cuda.current_stream()
    d = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()
    got = {}
    for form in ("level", "levels"):
        lv = torch.zeros(15, 64, dtype=torch.uint8, device="cuda")
        sc = torch.zeros(8 * 192, dtype=torch.uint8, device="cuda")
        rc = R.L.verity_rows_leaves_sf_launch(P, d.data_ptr(), rb, n, None, R.tn.data_ptr(), 0, 8, n, lv.data_ptr(), None, None,
                                              sc.data_ptr(), 0, s.cuda_stream)
        assert rc == 0, rc
        (R.tree_level if form == "level" else R.tree_levels)(P, lv, 8, s)
        torch.cuda.synchronize()
        got[form] = bytes(lv[-1].cpu().numpy()) == want
    return got


def timed(fn, reps: int, s: torch.cuda.Stream) -> float:
    """Median ms of fn on s, each rep behind a spin so the host's launches are off the clock."""
    ts = []
    for r in range(reps + 1):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(s):
            torch.cuda._sleep(20_000_000)
            e0.record(s)
            fn(s)
            e1.record(s)
        e1.synchronize()
        if r:
            ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


def compressions_leaf(rb: int) -> int:
    """A leaf's SHA-512 compressions: the row after P's midstate, salt 3, c 2, h 2, frame leaf 2."""
    return math.ceil((rb + 17) / 128) + 9


class Gemm:
    """One forward's int8 linears at T tokens: the served kernel (int8_fused.scaled_mm at _plan's config), 16 layers of distinct weights,
    captured once into a CUDA graph."""

    def __init__(self, T: int) -> None:
        from verity_vllm.engine import int8_fused as F
        dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(7)
        self.T, self.calls, self.cfgs = T, [], {}
        acts = {k: (torch.randint(-127, 128, (T, k), dtype=torch.int8, device=dev, generator=g),
                    torch.rand(T, dtype=torch.float32, device=dev, generator=g) + 0.5) for k in {k for _, _, k in LAYER}}
        outs = {n: torch.empty(T, n, dtype=torch.bfloat16, device=dev) for _, n, _ in LAYER}
        self.macs = 0
        for _ in range(LAYERS):
            for name, n, k in LAYER:
                w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device=dev, generator=g)
                sw = torch.rand(n, dtype=torch.float32, device=dev, generator=g) + 0.5
                cfg, work = F._plan(T, n, k, dev)
                self.cfgs[name] = str(cfg)
                self.calls.append((acts[k][0], acts[k][1], w, sw, outs[n], cfg, work))
                self.macs += T * n * k
        self.F = F
        for c in [*self.calls, self.calls[0]]:                # compile every config outside the capture; layer 0's qkv last
            F.scaled_mm(*c[:5], c[5], c[6])
        torch.cuda.synchronize()
        xq, sa, w, sw, _, cfg, work = self.calls[0]
        ref = (xq[:64].float() @ w[:64].float().t()) * sa[:64, None] * sw[None, :64]
        got = self.calls[0][4][:64, :64].float()
        self.close = bool(torch.allclose(got, ref.to(torch.bfloat16).float(), rtol=1e-2, atol=1e-2))
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=s):
            for c in self.calls:
                F.scaled_mm(*c[:5], c[5], c[6])
        torch.cuda.synchronize()

    def replay(self) -> None:
        self.graph.replay()


def beside(R: Rows, P, data: torch.Tensor, rb: int, n_rows: int, lv: torch.Tensor, scratch: torch.Tensor, cap: int, G: Gemm,
           n_fwd: int, hs: torch.cuda.Stream, gs: torch.cuda.Stream, reps: int) -> dict[str, Any]:
    """The GEMM loop (n_fwd forwards, an event after each) on gs and n_rows leaves on hs, released by one event; medians over reps.
    Each rep runs the loop alone (A), beside the hash (B), and alone again (A'), each behind the same spin, so the power-capped clock
    starts each leg alike: B's baseline is (A + A') / 2, and A' - A is the noise floor (aa_ms)."""
    main = torch.cuda.current_stream()

    def leg(hashing: bool) -> tuple[float, dict[str, Any] | None]:
        e0, eh, eh0, eg0 = (torch.cuda.Event(enable_timing=True) for _ in range(4))
        ef = [torch.cuda.Event(enable_timing=True) for _ in range(n_fwd)]
        torch.cuda.synchronize()
        torch.cuda._sleep(60_000_000)
        e0.record(main)
        gs.wait_event(e0)
        with torch.cuda.stream(gs):
            eg0.record(gs)
            for i in range(n_fwd):
                G.replay()
                ef[i].record(gs)
        if hashing:
            hs.wait_event(e0)
            with torch.cuda.stream(hs):
                eh0.record(hs)
                R.leaves(P, data, rb, n_rows, n_rows, lv, scratch, hs, cap)
                eh.record(hs)
        torch.cuda.synchronize()
        done = [e0.elapsed_time(e) for e in ef]
        g0 = e0.elapsed_time(eg0)
        if not hashing:
            return done[-1] - g0, None
        th = e0.elapsed_time(eh)
        k = sum(1 for t in done if t <= th)
        prev = done[k - 1] if k else g0
        frac = k + ((th - prev) / (done[k] - prev) if k < n_fwd and done[k] > prev else 0.0)
        return done[-1] - g0, {"hash_ms": th, "gemm_ms": done[-1], "hash_start_ms": e0.elapsed_time(eh0), "gemm_start_ms": g0,
                               "fwd_in_hash_window": frac, "overlapped": th < done[-1]}

    rows = []
    for r in range(reps + 1):
        a, _ = leg(False)
        b, row = leg(True)
        a2, _ = leg(False)
        if not r:
            continue
        base = (a + a2) / 2
        row.update(loop_a_ms=a, loop_a2_ms=a2, delta_ms=b - base, aa_ms=a2 - a,
                   kappa=row["fwd_in_hash_window"] * base / n_fwd / row["hash_ms"], slowdown=b / base)
        rows.append(row)
    med = {k: statistics.median(x[k] for x in rows) for k in ("hash_ms", "gemm_ms", "hash_start_ms", "gemm_start_ms", "fwd_in_hash_window")}
    H = n_rows * rb
    loop_ms = statistics.median((x["loop_a_ms"] + x["loop_a2_ms"]) / 2 for x in rows)
    delta = statistics.median(x["delta_ms"] for x in rows)
    return {"cap": cap, "n_rows": n_rows, "bytes": H, "n_fwd": n_fwd, "loop_alone_ms": round(loop_ms, 4),
            "overlapped": all(x["overlapped"] for x in rows),
            "hash_gbps": round(H / (med["hash_ms"] - med["hash_start_ms"]) * 1e-6, 2), "gemm_delta_ms": round(delta, 4),
            "kappa_window": round(statistics.median(x["kappa"] for x in rows), 4),
            "gemm_slowdown_total": round(statistics.median(x["slowdown"] for x in rows), 4),
            "aa_ms": round(statistics.median(abs(x["aa_ms"]) for x in rows), 4),
            "delta_ms_range": [round(min(x["delta_ms"] for x in rows), 4), round(max(x["delta_ms"] for x in rows), 4)],
            "delta_ms_per_gb": round(delta / (H * 1e-9), 4), **{k: round(v, 4) for k, v in med.items()}, "reps": rows}


def main() -> None:
    out_dir = Path(os.environ.get("OUT") or os.path.join(os.environ.get("RESEARCH_RUN_DIR", "."), "out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    reps = int(os.environ.get("REPS", "5"))
    total = int(float(os.environ.get("TOTAL_GB", "16")) * (1 << 30))
    sizes = [int(x) for x in os.environ.get("ROWS", "4096 16384 65536").replace("+", " ").split()]
    threads_forms = [int(x) for x in os.environ.get("THREADS", "256 128 64").replace("+", " ").split()]
    warps = [int(x) for x in os.environ.get("WARPS", "1 2 4 8").replace("+", " ").split()]
    T = int(os.environ.get("T", "8192"))
    hash_ms_target = float(os.environ.get("HASH_MS", "30"))
    caps_given = [int(x) for x in os.environ["CAPS"].replace("+", " ").split()] if os.environ.get("CAPS") else None
    tree = os.environ.get("TREE", "1") == "1"
    stride0 = os.environ.get("STRIDE0", "0") == "1"
    gprio = int(os.environ.get("GPRIO", "-100"))
    build_dir = Path(os.environ.get("BUILD_DIR") or tempfile.mkdtemp(prefix="hash-cost-"))
    prop = torch.cuda.get_device_properties(0)
    arch, sms = f"{prop.major}{prop.minor}", prop.multi_processor_count
    res: dict[str, Any] = {"device": prop.name, "sms": sms, "arch": arch, "total_bytes": total, "T": T,
                           "phases": os.environ.get("VERITY_ROWS_PHASES", "4"), "form": os.environ.get("VERITY_ROWS_FORM", "split"),
                           "stride0": stride0, "gemm_priority": gprio, "cuda_cflags": list(CUDA_CFLAGS), "forms": {}, "clock_marks": []}

    def mark(tag: str) -> None:
        res["clock_marks"].append([tag, time.time()])

    hs, gs = torch.cuda.Stream(priority=0), torch.cuda.Stream(priority=gprio)   # -100: clamped to the highest
    mark("gemm_build")
    G = Gemm(T)
    fwd_ms = timed(lambda s: G.replay(), max(reps, 9), gs)
    res["gemm"] = {"macs_per_forward": G.macs, "configs": G.cfgs, "spot_check_close": G.close, "forward_ms": round(fwd_ms, 4),
                   "tmacs_per_s": round(G.macs / fwd_ms * 1e-9, 1)}
    mark("gemm_alone")
    print("gemm", json.dumps(res["gemm"]), flush=True)

    data = torch.randint(0, 256, (total,), dtype=torch.uint8, device="cuda")
    for threads in threads_forms:
        b = build(build_dir, arch, threads) if not (build_dir / f"rows_t{threads}.so").exists() else \
            {"rc": 0, "so": str(build_dir / f"rows_t{threads}.so"), "ptxas": json.loads((build_dir / f"rows_t{threads}.json").read_text())
             if (build_dir / f"rows_t{threads}.json").exists() else {}}
        form: dict[str, Any] = {"build_rc": b["rc"], "ptxas": b["ptxas"], "sizes": {}}
        res["forms"][f"t{threads}"] = form
        if b["rc"]:
            form["log"] = b["log"]
            print(f"t{threads} build failed", b["log"][-800:], flush=True)
            continue
        R = Rows(b["so"], threads)
        for rb in sizes:
            entry: dict[str, Any] = {"check": check(R, rb)}
            form["sizes"][rb] = entry
            if not all(entry["check"].values()):
                print(f"t{threads} rows {rb} check failed {entry['check']}", flush=True)
                continue
            m = total // rb
            P = R.params(domain(m, rb), os.urandom(step_rows.KEY_BYTES), rb)
            lv = torch.empty(2 * m - 1, 64, dtype=torch.uint8, device="cuda")
            scratch = torch.empty(192 * m, dtype=torch.uint8, device="cuda")
            R.stride0 = stride0
            mark(f"t{threads}:{rb}:alone")
            leaf_ms = timed(lambda s: R.leaves(P, data, rb, m, m, lv, scratch, s), reps, hs)
            entry.update(rows=m, leaf_ms=round(leaf_ms, 4), leaf_gbps=round(total / leaf_ms * 1e-6, 2),
                         leaf_gcomp_per_s=round(m * compressions_leaf(rb) / leaf_ms * 1e-6, 3))
            if threads == 256 and tree:
                pw = m & (m - 1) == 0
                tl = timed(lambda s: R.tree_level(P, lv, m, s), reps, hs)
                root_l = bytes(lv[-1].cpu().numpy())
                tm = timed(lambda s: R.tree_levels(P, lv, m, s), reps, hs) if pw else None
                root_m = bytes(lv[-1].cpu().numpy()) if pw else None
                entry["tree"] = {"leaves": m, "nodes": m - 1, "level_form_ms": round(tl, 4), "levels10_form_ms": tm and round(tm, 4),
                                 "roots_equal": root_l == root_m if pw else None,
                                 "level_form_gnodes_per_s": round((m - 1) / tl * 1e-6, 3),
                                 "levels10_form_gnodes_per_s": tm and round((m - 1) / tm * 1e-6, 3)}
                best_tree = min(x for x in (tl, tm) if x)
                entry["commit_ms"] = round(leaf_ms + best_tree, 4)
                entry["commit_gbps"] = round(total / (leaf_ms + best_tree) * 1e-6, 2)
            caps = caps_given if caps_given is not None else [*sorted({max(1, round(sms * w * 32 / threads)) for w in warps}), 0]
            entry["capped_alone"] = {}
            entry["beside"] = []
            for cap in caps:
                # the hash's bytes: about HASH_MS alone at this cap, whole launches of cap CTAs, at most the stream
                chunk = (cap or 10 ** 9) * threads
                probe = min(m, max(chunk, min(threads * sms, 8 * chunk)))
                probe -= probe % min(chunk, probe)
                pm = timed(lambda s: R.leaves(P, data, rb, probe, probe, lv, scratch, s, cap), 3, hs)
                rate = probe / pm                                   # rows a ms, alone at this cap
                n_rows = int(min(m, max(probe, rate * hash_ms_target)))
                if cap:
                    n_rows = max(chunk, n_rows - n_rows % chunk) if n_rows >= chunk else n_rows
                alone_ms = timed(lambda s: R.leaves(P, data, rb, n_rows, n_rows, lv, scratch, s, cap), reps, hs)
                entry["capped_alone"][cap] = {"n_rows": n_rows, "ms": round(alone_ms, 4), "gbps": round(n_rows * rb / alone_ms * 1e-6, 2)}
                n_fwd = max(2, math.ceil(2.5 * alone_ms / fwd_ms))
                mark(f"t{threads}:{rb}:cap{cap}:beside")
                for _ in range(3):
                    r = beside(R, P, data, rb, n_rows, lv, scratch, cap, G, n_fwd, hs, gs, reps)
                    if r["overlapped"]:
                        break
                    n_fwd *= 2
                r["alone_gbps"] = entry["capped_alone"][cap]["gbps"]
                entry["beside"].append(r)
                print(f"t{threads} rows {rb} cap {cap}: alone {r['alone_gbps']} GB/s, beside {r['hash_gbps']} GB/s, kappa "
                      f"{r['kappa_window']}, delta {r['gemm_delta_ms']} ms over {r['n_fwd']} fwd, overlapped {r['overlapped']}", flush=True)
            mark(f"t{threads}:{rb}:done")
            print(f"t{threads} rows {rb}", json.dumps({k: v for k, v in entry.items() if k not in ("beside",)}), flush=True)
            lv = scratch = P = None
            torch.cuda.empty_cache()
            (out_dir / "hash_cost.json").write_text(json.dumps(res, indent=1))
    mark("end")
    res["gemm"]["forward_ms_end"] = round(timed(lambda s: G.replay(), max(reps, 9), gs), 4)
    (out_dir / "hash_cost.json").write_text(json.dumps(res, indent=1))


def build_all() -> None:
    """`hash_cost.py build`: every THREADS form's library into BUILD_DIR for ARCH (default 90), no GPU needed."""
    d = Path(os.environ["BUILD_DIR"])
    d.mkdir(parents=True, exist_ok=True)
    for threads in [int(x) for x in os.environ.get("THREADS", "256 128 64").replace("+", " ").split()]:
        b = build(d, os.environ.get("ARCH", "90"), threads)
        (d / f"rows_t{threads}.json").write_text(json.dumps(b["ptxas"]))
        print(f"t{threads} build rc {b['rc']}", json.dumps(b["ptxas"]), b["log"], flush=True)
        if b["rc"]:
            raise SystemExit(3)


if __name__ == "__main__":
    import sys
    build_all() if sys.argv[1:] == ["build"] else main()
