"""The rows-sha512 floor on one GPU: the windowed worker's kernels (native_rows.cu, built here with nvcc and the JIT's CUDA flags into
one shared library per leaf-kernel __launch_bounds__ form) over one window of rows (the leaves and the window's whole tree), alone and
beside a busy stream (bf16 GEMM loops, a square one and decode's skinny one, a device copy loop; at default and at the highest stream
priority, and in launches capped at a number of CTAs), and COMMIT-GPU's raw SHA-512 compression rate on the same device.  Each form's
root of a small padded domain is checked against step_rows before it is timed.  Measurement only; JSON to $OUT/rows_floor.json.

    bash benchmarks/served_commit/rows_floor.sh     (ROWS: row bytes, default 256 1024 4096; WINDOW_MB, default 256; REPS, default 9;
                                                     MINB: the leaf kernel's min CTAs per SM, default 1 2 3;
                                                     CAPS: CTAs per launch to time, default none; BESIDE: the row
                                                     sizes timed beside the busy loops, default the worker's)
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from verity_vllm.commit import step_rows
from verity_vllm.commit.committer.native_jit import CUDA_CFLAGS

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "integrations/vllm/verity_vllm/commit/committer/native_rows.cu"
COMMIT_GPU = ROOT / "benchmarks/pouw/commit_gpu/commit_gpu.cu"
c_p, c_u64 = ctypes.c_void_p, ctypes.c_uint64


def nvcc(args: list[str]) -> tuple[int, str]:
    p = subprocess.run(["nvcc", *args], capture_output=True, text=True, check=False)
    return p.returncode, p.stderr + p.stdout


def ptxas(log: str) -> dict[str, dict[str, int]]:
    """Registers, stack and spills per kernel from ptxas -v."""
    out: dict[str, dict[str, int]] = {}
    cur = None
    for line in log.splitlines():
        if m := re.search(r"(?:Compiling entry function|Function properties for) .*?(rows_(?:leaf|level)_kernel)", line):
            cur = out.setdefault(m.group(1), {})
        elif cur is not None and (m := re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)):
            cur.update(stack=int(m.group(1)), spill_stores=int(m.group(2)), spill_loads=int(m.group(3)))
        elif cur is not None and (m := re.search(r"Used (\d+) registers", line)):
            cur["registers"] = int(m.group(1))
    return out


class Rows:
    """native_rows.cu's extern "C" entry points through ctypes."""

    def __init__(self, so: Path) -> None:
        L = ctypes.CDLL(str(so))
        L.verity_rows_params_bytes.restype = ctypes.c_int
        L.verity_rows_build.argtypes = [ctypes.c_char_p] * 6 + [ctypes.c_int, ctypes.c_uint32, c_p]
        L.verity_rows_leaves_launch.argtypes = [c_p, c_p, c_u64, c_u64, c_p, c_p, c_u64, c_u64, c_u64, c_p, c_p, c_p, c_u64, c_p]
        L.verity_rows_level_launch.argtypes = [c_p, c_p, c_p, c_u64, ctypes.c_int, c_u64, c_u64, c_p]
        self.L = L
        self.tn = torch.frombuffer(bytearray(step_rows.t_nibbles()), dtype=torch.uint8).cuda()

    def params(self, dom: Any, ks: bytes, row_bytes: int) -> ctypes.Array:
        buf = ctypes.create_string_buffer(self.L.verity_rows_params_bytes())
        schema = step_rows.SCHEMA.encode()
        rc = self.L.verity_rows_build(dom.domain_id, ks, step_rows.row_prefix(8 * row_bytes), step_rows.INST.salt_prefix,
                                      step_rows.INST.leaf_prefix(step_rows.KEY), schema, len(schema), row_bytes, buf)
        assert rc == 0, rc
        return buf

    def leaves(self, P: ctypes.Array, rows: torch.Tensor, row_bytes: int, n_rows: int, m: int, n_total: int, out: torch.Tensor,
               stream: torch.cuda.Stream, cap: int = 0) -> None:
        rc = self.L.verity_rows_leaves_launch(P, rows.data_ptr(), row_bytes, n_rows, None, self.tn.data_ptr(), 0, m, n_total,
                                              out.data_ptr(), None, None, cap, stream.cuda_stream)
        assert rc == 0, rc

    def tree(self, P: ctypes.Array, lv: torch.Tensor, m: int, stream: torch.cuda.Stream, cap: int = 0) -> None:
        """lv: [2m - 1, 64], the m leaves first, then each level above them."""
        off, n, level = 0, m, 0
        while n > 1:
            rc = self.L.verity_rows_level_launch(P, lv[off].data_ptr(), lv[off + n].data_ptr(), n // 2, level, 0, cap, stream.cuda_stream)
            assert rc == 0, rc
            off, n, level = off + n, n // 2, level + 1


def domain(n: int, row_bytes: int) -> Any:
    return step_rows.domain(b"p" * 32, b"c" * 32, b"g" * 32, n, 8 * row_bytes)


def check(R: Rows, row_bytes: int) -> bool:
    """A 5-row domain (width 8, three pads): the kernels' root is step_rows'."""
    n, ks = 5, os.urandom(step_rows.KEY_BYTES)
    data = os.urandom(n * row_bytes)
    rows = [data[i * row_bytes:(i + 1) * row_bytes] for i in range(n)]
    dom = domain(n, row_bytes)
    want = step_rows.levels(dom, rows, ks, 8 * row_bytes)[-1][0]
    P, s = R.params(dom, ks, row_bytes), torch.cuda.current_stream()
    lv = torch.zeros(15, 64, dtype=torch.uint8, device="cuda")
    R.leaves(P, torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda(), row_bytes, n, 8, n, lv, s)
    R.tree(P, lv, 8, s)
    torch.cuda.synchronize()
    return bytes(lv[-1].cpu().numpy()) == want


def timed(fn: Callable[[torch.cuda.Stream], None], reps: int, stream: torch.cuda.Stream) -> float:
    """Median ms of fn on `stream`, each rep launched behind a ~1 ms spin so the host's launch time is off the clock."""
    ts = []
    for r in range(reps + 1):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(2_000_000)
            e0.record(stream)
            fn(stream)
            e1.record(stream)
        e1.synchronize()
        if r:
            ts.append(e0.elapsed_time(e1))
    return statistics.median(ts)


def beside(hash_fn: Callable, busy_fn: Callable, n_busy: int, hs: torch.cuda.Stream, bs: torch.cuda.Stream, reps: int) -> dict:
    """The busy loop (n_busy calls) on bs, then the commit on hs, both released by one event: when each is done, median ms."""
    main = torch.cuda.current_stream()
    hd, bd = [], []
    for r in range(reps + 1):
        e0, eh, eb = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        torch.cuda.synchronize()
        torch.cuda._sleep(4_000_000)
        e0.record(main)
        bs.wait_event(e0)
        hs.wait_event(e0)
        with torch.cuda.stream(bs):
            for _ in range(n_busy):
                busy_fn()
            eb.record(bs)
        with torch.cuda.stream(hs):
            hash_fn(hs)
            eh.record(hs)
        torch.cuda.synchronize()
        if r:
            hd.append(e0.elapsed_time(eh))
            bd.append(e0.elapsed_time(eb))
    return {"commit_done_ms": round(statistics.median(hd), 4), "busy_done_ms": round(statistics.median(bd), 4)}


def raw_rate(tmp: Path, arch: str) -> dict:
    """COMMIT-GPU's `sha` mode (each thread chains compressions) built for this device."""
    exe = tmp / "commit_gpu"
    stubs = [f"-L{os.environ['CUDA_HOME']}/lib64/stubs"] if os.environ.get("CUDA_HOME") else []
    rc, log = nvcc(["-O3", "-std=c++17", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-o", str(exe), str(COMMIT_GPU), *stubs,
                    "-lcublasLt", "-lnvidia-ml"])
    if rc:
        return {"build_rc": rc, "log": log[-2000:]}
    p = subprocess.run([str(exe), "sha", "20"], capture_output=True, text=True, check=False)
    return json.loads(p.stdout.strip().splitlines()[-1])["sha"] if p.returncode == 0 else {"rc": p.returncode, "log": p.stderr[-2000:]}


def compressions(m: int, row_bytes: int) -> int:
    """SHA-512 compressions of m leaves (row after P's midstate, salt 3, c 2, h 2, frame leaf 2) and their m - 1 nodes (2 each)."""
    return m * (math.ceil((row_bytes + 17) / 128) + 9) + 2 * (m - 1)


def main() -> None:
    out_dir = Path(os.environ.get("OUT") or os.path.join(os.environ.get("RESEARCH_RUN_DIR", "."), "out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    reps = int(os.environ.get("REPS", "9"))
    window = int(os.environ.get("WINDOW_MB", "256")) << 20
    sizes = [int(x) for x in os.environ.get("ROWS", "256 1024 4096").split()]
    forms = [int(x) for x in os.environ.get("MINB", "1 2 3").split()]
    caps = [int(x) for x in os.environ.get("CAPS", "").split()]
    beside_rows = [int(x) for x in os.environ.get("BESIDE", str(step_rows.ROW_BYTES)).split()]
    prop = torch.cuda.get_device_properties(0)
    arch = f"{prop.major}{prop.minor}"
    res: dict[str, Any] = {"device": prop.name, "sms": prop.multi_processor_count, "arch": arch, "window_bytes": window,
                           "cuda_cflags": list(CUDA_CFLAGS), "forms": {}}
    tmp = Path(tempfile.mkdtemp(prefix="rows-floor-"))
    res["raw"] = raw_rate(tmp, arch)
    print("raw", json.dumps(res["raw"]), flush=True)

    hs, hs_hi, bs = torch.cuda.Stream(), torch.cuda.Stream(priority=-100), torch.cuda.Stream()   # -100: clamped to the highest
    a = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
    c = torch.empty_like(a)
    src, dst = torch.empty(1 << 30, dtype=torch.uint8, device="cuda"), torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    x, w = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda"), torch.randn(4096, 12288, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(32, 12288, dtype=torch.bfloat16, device="cuda")
    busy = {"gemm_bf16_8192": lambda: torch.matmul(a, b, out=c), "copy_1GiB": lambda: dst.copy_(src),
            "gemm_bf16_32x4096x12288": lambda: torch.matmul(x, w, out=y)}
    busy_ms = {k: timed(lambda _s, f=f: f(), reps, bs) for k, f in busy.items()}
    res["busy_alone_ms"] = {k: round(v, 4) for k, v in busy_ms.items()}

    data = torch.randint(0, 256, (window,), dtype=torch.uint8, device="cuda")
    for minb in forms:
        so = tmp / f"rows_minb{minb}.so"
        rc, log = nvcc([*CUDA_CFLAGS, "-std=c++17", "-shared", "-Xcompiler", "-fPIC", f"-gencode=arch=compute_{arch},code=sm_{arch}",
                        "-Xptxas", "-v", f"-DVERITY_ROWS_MIN_BLOCKS={minb}", str(SRC), "-o", str(so)])
        form: dict[str, Any] = {"build_rc": rc, "ptxas": ptxas(log)}
        res["forms"][f"minb{minb}"] = form
        if rc:
            form["log"] = log[-2000:]
            continue
        R = Rows(so)
        form["sizes"] = {}
        for rb in sizes:
            form["sizes"][rb] = entry = size_floor(R, rb, data, reps, (hs, hs_hi, bs), busy, busy_ms, caps, beside_rows)
            print(f"minb{minb} rows {rb}", json.dumps(entry), flush=True)
    (out_dir / "rows_floor.json").write_text(json.dumps(res, indent=1))


def size_floor(R: Rows, rb: int, data: torch.Tensor, reps: int, streams: tuple, busy: dict, busy_ms: dict,
               caps: list[int], beside_rows: list[int]) -> dict[str, Any]:
    """One window of rb-byte rows: its leaves, its tree and both, alone and in launches of at most each cap's CTAs; at the row
    sizes of beside_rows also all-pad leaves and the commit (whole-grid at both priorities, and capped) beside each busy loop."""
    hs, hs_hi, bs = streams
    window = data.numel()
    m = window // rb
    entry: dict[str, Any] = {"leaves": m, "matches_step_rows": check(R, rb)}
    if not entry["matches_step_rows"]:
        return entry
    P = R.params(domain(m, rb), os.urandom(step_rows.KEY_BYTES), rb)
    lv = torch.empty(2 * m - 1, 64, dtype=torch.uint8, device="cuda")

    def leaves(s: torch.cuda.Stream, n_rows: int = m, cap: int = 0) -> None:
        R.leaves(P, data, rb, n_rows, m, m, lv, s, cap)

    def commit(s: torch.cuda.Stream, cap: int = 0) -> None:
        leaves(s, cap=cap)
        R.tree(P, lv, m, s, cap)

    leaf_ms, tree_ms = timed(leaves, reps, hs), timed(lambda s: R.tree(P, lv, m, s), reps, hs)
    total_ms, comps = timed(commit, reps, hs), compressions(m, rb)
    entry.update(leaf_ms=round(leaf_ms, 4), tree_ms=round(tree_ms, 4), total_ms=round(total_ms, 4),
                 gb_per_s=round(window / total_ms * 1e-6, 2), ns_per_leaf=round(total_ms * 1e6 / m, 3),
                 compressions=comps, gcomp_per_s=round(comps / total_ms * 1e-6, 3))
    entry["capped_ms"] = {c: round(timed(lambda s, c=c: commit(s, c), reps, hs), 4) for c in caps}
    if rb not in beside_rows:
        return entry
    entry["pads_ms"] = round(timed(lambda s: leaves(s, 0), reps, hs), 4)
    entry["beside"] = {}
    for k, f in busy.items():
        n_busy = max(1, math.ceil(2 * total_ms / busy_ms[k]))
        runs = [(tag, stream, commit) for tag, stream in (("default", hs), ("high_priority", hs_hi))]
        runs += [(f"cap{c}", hs, lambda s, c=c: commit(s, c)) for c in caps]
        for tag, stream, fn in runs:
            r = beside(fn, f, n_busy, stream, bs, reps)
            r.update(n_busy=n_busy, busy_alone_ms=round(n_busy * busy_ms[k], 4),
                     busy_slowdown=round(r["busy_done_ms"] / (n_busy * busy_ms[k]), 4),
                     commit_slowdown=round(r["commit_done_ms"] / total_ms, 4))
            entry["beside"][f"{k}:{tag}"] = r
    return entry


if __name__ == "__main__":
    main()
