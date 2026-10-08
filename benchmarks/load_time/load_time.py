"""Time a large model to engine-ready under vLLM at TP8, by page-cache, compile-cache and loader state.

Each configuration runs in a fresh process; the driver records the wall time from spawn to a constructed `vllm.LLM`
and to its first 8-token generation, the phases vLLM logs (weights load, torch.compile, graph capture, engine init),
and the fraction of the model's files resident in the page cache before and after (mincore). Eviction is
posix_fadvise(DONTNEED) on every shard, which needs no root and leaves pages another process maps. mincore reports every
page of a file resident to a process that neither owns it nor may write it (Linux >= 5.0), so the residency fractions mean
something only when the job owns the model's files.

  python load_time.py --model DIR --out OUT_DIR [--plan default|quick|fastsafetensors|fastsafetensors-warm|fastsafetensors-restart] [--tp 8]

writes OUT_DIR/summary.json after each configuration and OUT_DIR/<config>.log with the child's output.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import ctypes.util
import json
import mmap
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

#: phase -> pattern over rank 0's and the engine core's lines; a phase logged more than once (CUDA graphs are captured once to
#: profile memory and once for serving) keeps every value
PHASES = {
    "weights_load_s": r"Loading weights took ([\d.]+) s",
    "model_load_s": r"Model loading took [\d.]+ GiB (?:memory )?and ([\d.]+) s",
    "dynamo_s": r"Dynamo bytecode transform time: ([\d.]+) s",
    "compile_graph_s": r"Compiling a graph for .* takes ([\d.]+) s",
    "compile_total_s": r"torch\.compile (?:takes|took) ([\d.]+) s in total",
    "profile_run_s": r"Initial profiling/warmup run took ([\d.]+) s",
    "graph_capture_s": r"Graph capturing finished in ([\d.]+) secs?",
    "init_engine_s": r"init engine \(profile, create kv cache, warmup model\) took ([\d.]+) s",
}
CACHE_HIT = r"[Dd]irectly load|[Ll]oad(?:ed|ing)? (?:the )?(?:AOT )?compiled .* from|from the cache"
RANK0 = re.compile(r"^(?:\((?:Worker_TP0|EngineCore)[ \w=]*\) )?(?:INFO|WARNING|DEBUG)", re.M)


def model_files(model: Path) -> list[Path]:
    seen, out = set(), []
    for p in sorted(model.iterdir()):
        r = p.resolve()
        if r.is_file() and r not in seen:
            seen.add(r)
            out.append(r)
    return out


_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]


def resident(files: list[Path]) -> float:
    page = mmap.PAGESIZE
    total = hit = 0
    for f in files:
        size = f.stat().st_size
        if not size:
            continue
        fd = os.open(f, os.O_RDONLY)
        try:
            addr = _libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
            if addr in (None, ctypes.c_void_p(-1).value):
                raise OSError(ctypes.get_errno(), f"mmap {f}")
            pages = (size + page - 1) // page
            vec = ctypes.create_string_buffer(pages)
            if _libc.mincore(addr, size, vec) != 0:
                raise OSError(ctypes.get_errno(), f"mincore {f}")
            hit += sum(b & 1 for b in vec.raw)
            total += pages
            _libc.munmap(addr, size)
        finally:
            os.close(fd)
    return hit / total if total else 0.0


def evict(files: list[Path]) -> None:
    for f in files:
        fd = os.open(f, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def read_all(files: list[Path], threads: int) -> dict:
    def one(f: Path) -> int:
        n, buf = 0, memoryview(bytearray(16 << 20))
        with open(f, "rb", buffering=0) as h:
            while k := h.readinto(buf):
                n += k
        return n

    t = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(threads) as ex:
        n = sum(ex.map(one, files))
    s = time.monotonic() - t
    return {"bytes": n, "seconds": round(s, 2), "gb_per_s": round(n / s / 1e9, 3), "threads": threads}


def child(ns: argparse.Namespace) -> None:
    t0 = time.monotonic()
    from vllm import LLM, SamplingParams
    t_import = time.monotonic()
    llm = LLM(model=ns.model, tensor_parallel_size=ns.tp, load_format=ns.load_format, enforce_eager=ns.eager,
              max_model_len=4096, gpu_memory_utilization=0.85, seed=0)
    t_ready = time.monotonic()
    out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0.0))
    t_gen = time.monotonic()
    print("LOADTIME " + json.dumps({"import_s": round(t_import - t0, 2), "construct_s": round(t_ready - t_import, 2),
                                    "first_generate_s": round(t_gen - t_ready, 2),
                                    "text": out[0].outputs[0].text}), flush=True)
    shutdown = getattr(getattr(llm.llm_engine, "engine_core", None), "shutdown", None)
    if shutdown:
        shutdown()


def parse(log: str) -> dict:
    phases = {}
    lines = "\n".join(line for line in log.splitlines() if RANK0.match(line))
    for k, rx in PHASES.items():
        vals = [float(v) for v in re.findall(rx, lines)]
        if vals:
            phases[k] = vals[0] if len(vals) == 1 else vals
    phases["compile_cache_hit"] = bool(re.search(CACHE_HIT, lines))
    m = re.findall(r"^LOADTIME (.*)$", log, re.M)
    if m:
        phases.update(json.loads(m[-1]))
    return phases


def cache_env(root: Path) -> dict:
    d = {"VLLM_CACHE_ROOT": "vllm", "TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "inductor",
         "XDG_CACHE_HOME": "xdg", "CUDA_CACHE_PATH": "nv", "FLASHINFER_WORKSPACE_BASE": "flashinfer"}
    for sub in d.values():
        (root / sub).mkdir(parents=True, exist_ok=True)
    return {k: str(root / v) for k, v in d.items()}


def cudart_dir(root: Path) -> Path:
    """A directory holding only `libcudart.so`, linked to the CUDA runtime torch loads.

    fastsafetensors dlopens the unversioned name, which the CUDA wheels don't ship and the job's image doesn't put on the
    loader path; putting all of CUDA_HOME's lib64 there instead would also put its cuBLAS and NCCL ahead of torch's.
    """
    import nvidia.cuda_runtime

    lib = next(Path(p) / "lib" / "libcudart.so.12" for p in nvidia.cuda_runtime.__path__)
    d = root / "cudart"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "libcudart.so").exists():
        (d / "libcudart.so").symlink_to(lib)
    return d


def run_config(ns, files, name, *, page, caches: Path, load_format="auto", eager=False) -> dict:
    if page == "cold":
        evict(files)
    rec = {"config": name, "page_cache": page, "compile_cache_dir": caches.name, "load_format": load_format,
           "enforce_eager": eager, "resident_before": round(resident(files), 4)}
    if page == "cold" and rec["resident_before"] > 0.05:
        rec["evict_failed"] = True
    env = {**os.environ, **cache_env(caches), "HF_HUB_OFFLINE": "1", "VLLM_LOGGING_LEVEL": "INFO", "PYTHONUNBUFFERED": "1"}
    if load_format == "fastsafetensors":
        env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(cudart_dir(caches.parent)), env.get("LD_LIBRARY_PATH")]))
    cmd = [sys.executable, __file__, "--child", "--model", ns.model, "--tp", str(ns.tp), "--load-format", load_format]
    if eager:
        cmd.append("--eager")
    log_path = Path(ns.out) / f"{name}.log"
    t = time.monotonic()
    # its own session, so the engine core and TP workers it forks are reaped with it; the log is a file, not a pipe, so an
    # orphan holding it open can't stall the wait
    with open(log_path, "w") as log_file:
        p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=ns.child_timeout_s)
        except subprocess.TimeoutExpired:
            rc = "timeout"
    rec["wall_s"] = round(time.monotonic() - t, 2)
    rec["rc"] = rc
    rec["reaped"] = reap(p.pid)
    rec["gpu_free_wait_s"] = wait_gpus_free()
    rec.update(parse(log_path.read_text(errors="replace")))
    rec["resident_after"] = round(resident(files), 4)
    return rec


def reap(pgid: int) -> int:
    """Kill what's left of the child's process group (an engine core or TP worker it didn't shut down); how many were left."""
    left = [int(p) for p in os.listdir("/proc") if p.isdigit() and _pgid(int(p)) == pgid]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        for _ in range(30):
            if not any(_pgid(p) == pgid for p in left):
                return len(left)
            time.sleep(1)
    return len(left)


def _pgid(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except ProcessLookupError:
        return None


def wait_gpus_free(limit_s: int = 300) -> float:
    t = time.monotonic()
    while time.monotonic() - t < limit_s:
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.split()
        if all(int(u) < 1024 for u in used):
            break
        time.sleep(2)
    return round(time.monotonic() - t, 1)


def driver(ns: argparse.Namespace) -> None:
    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    model = Path(ns.model)
    files = model_files(model)
    weights = [f for f in files if f.suffix == ".safetensors"] or files
    scratch = Path(tempfile.mkdtemp(prefix="loadtime-"))
    summary = {"model": str(model), "tp": ns.tp, "files": len(files),
               "bytes": sum(f.stat().st_size for f in files), "host": platform.node(), "configs": []}
    try:
        import torch
        import vllm
        summary["versions"] = {"vllm": vllm.__version__, "torch": torch.__version__, "python": platform.python_version()}
        summary["gpus"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                         capture_output=True, text=True).stdout.strip().splitlines()
    except Exception as e:  # noqa: BLE001 - record what's missing, keep timing
        summary["versions_error"] = repr(e)

    def save():
        (out / "summary.json").write_text(json.dumps(summary, indent=1))

    evict(files)
    summary["resident_after_evict"] = round(resident(files), 4)
    summary["cold_read"] = read_all(weights, ns.read_threads)
    summary["resident_after_read"] = round(resident(files), 4)
    save()

    warm = scratch / "caches-a"
    plan = [("warm-page_cold-compile_graphs", dict(page="warm", caches=warm))]
    if ns.plan == "fastsafetensors-warm":
        plan = [("warm-page_cold-compile_graphs_fastsafetensors", dict(page="warm", caches=warm, load_format="fastsafetensors"))]
    elif ns.plan == "fastsafetensors-restart":
        plan += [(f"warm-page_warm-compile_graphs_fastsafetensors{s}", dict(page="warm", caches=warm, load_format="fastsafetensors"))
                 for s in ("", "_repeat")]
    elif ns.plan == "fastsafetensors":
        plan += [("warm-page_warm-compile_graphs_fastsafetensors", dict(page="warm", caches=warm, load_format="fastsafetensors")),
                 ("cold-page_warm-compile_graphs_fastsafetensors", dict(page="cold", caches=warm, load_format="fastsafetensors"))]
    else:
        plan += [("warm-page_warm-compile_graphs", dict(page="warm", caches=warm)),
                 ("warm-page_eager", dict(page="warm", caches=warm, eager=True))]
    if ns.plan == "default":
        plan += [("warm-page_warm-compile_graphs_fastsafetensors", dict(page="warm", caches=warm, load_format="fastsafetensors")),
                 ("cold-page_warm-compile_graphs", dict(page="cold", caches=warm)),
                 ("cold-page_warm-compile_graphs_fastsafetensors", dict(page="cold", caches=warm, load_format="fastsafetensors")),
                 ("warm-page_warm-compile_graphs_repeat", dict(page="warm", caches=warm))]
    for name, kw in plan:
        if kw["page"] == "warm" and resident(files) < 0.99:
            read_all(weights, ns.read_threads)
        rec = run_config(ns, files, name, **kw)
        summary["configs"].append(rec)
        save()
        print(json.dumps(rec), flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    failed = [r["config"] for r in summary["configs"] if r["rc"] != 0]
    if failed:
        sys.exit(f"load_time: failed configs {failed}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--plan", choices=["default", "quick", "fastsafetensors", "fastsafetensors-warm", "fastsafetensors-restart"], default="default")
    ap.add_argument("--read-threads", type=int, default=16)
    ap.add_argument("--child-timeout-s", type=int, default=2700)
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--load-format", default="auto", help=argparse.SUPPRESS)
    ap.add_argument("--eager", action="store_true", help=argparse.SUPPRESS)
    ns = ap.parse_args()
    if ns.child:
        child(ns)
    elif not ns.out:
        ap.error("--out is required")
    else:
        driver(ns)


if __name__ == "__main__":
    main()
