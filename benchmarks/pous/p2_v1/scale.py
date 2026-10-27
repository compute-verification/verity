"""P2 at device scale: what a store of B labels costs to build, check, hold beside a served model and audit, on one GPU
(`pod.sh IMPL=p2-scale` runs the GPU part; `cpu` and `setup-check` need no GPU).

    scale.py cpu --out DIR --cpus 124,126 [--siblings 125,127] [--blocks 2]
    scale.py setup-check --out DIR --blocks 4096 --workers 4 [--lean-exe pous-setup]
    scale.py gpu --out DIR --p2-lib LIB --build-dir BUILD [--batches 1,1024,4096,16384,65536]
    scale.py audit --out DIR --p2-lib LIB --build-dir BUILD --log2 19,25 --verifier-cpu V --responder-cpu R \
        [--workers N] [--fit-python PY --fit-model ID --fit-cpus LIST]

- `cpu`: one label's encode, `E(m) = ρ(σ_K(ρ(m + t)))`, on one core: `reference.encode` with gmpy2's `powmod` (two
  exponentiations by `(p + 1) / 4`), and the special-form chain (`w − 2` squarings with two folds each, times the inverse
  of `y^((c0 − 1)/4)`) on gmpy2; one root by Python's own `pow` and by `schemes.p2`'s pure-Python field. Then the chain's
  aggregate rate with one process per listed core, and per listed SMT sibling too. Every label decodes back by
  `reference.decode`.
- `setup-check`: a store of `--blocks` labels (one segment) under `schemes.p2`'s framing, encoded on the CPU, then the
  public-encoder check two ways: Python's `setup.verifier_key_from_encoding` and the Lean `pous-setup check`.
- `gpu`: the device's encode throughput (`p2v1_encode`, two roots a label) at growing batches, each batch decoded back to
  its payloads and sampled labels held to `reference.decode`; and the decode rate, the setup check's device step.
- `audit`: for each B = 2^e, a store of B uniformly random canonical labels on the device (every canonical label is the
  encoding of its decode, and an answer's time does not depend on its bytes), the leaves the verifier keeps hashed by a
  process pool, then the harness's honest timed audit (`p2_v1/memory_challenger.py`: k = 105, Δ = 0.5 ms plus the loopback
  allowance fixed before it). Beside it: the device decode of all of `C` (the setup check's device step at B), the cost of
  gathering a v4 RESPONSE's authentication path (log2 B siblings of 64 bytes) from a tree held in host memory, and a
  loopback round trip at the sizes of a RESPONSE without and with that path. With `--fit-python`, vLLM serves the model in
  FP8 (`quantization="fp8"`) in its own process beside the largest store, at growing batches, and the audit runs again
  while it serves.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BPS = 1 << 16
WIDTH, C0, PAYLOAD = 16_448, 21_065, 2_054
NB = WIDTH // 8
DIGEST = 64


def quantiles(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    if not s:
        return {}
    at = lambda q: s[min(len(s) - 1, int(q * len(s)))]  # noqa: E731
    return {"n": len(s), "min": s[0], "p50": at(0.5), "p99": at(0.99), "p99.9": at(0.999), "max": s[-1]}


def dump(out: Path, name: str, doc: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(doc, indent=1) + "\n")


# --- the CPU's encode ---------------------------------------------------------------------------------------------------

class Chain:
    """`y^((p+1)/4)` for `p = 2^w − c0` on gmpy2: `w − 2` squarings, each folded twice at bit `w`, then times the inverse of
    `y^small` with `small = (c0 − 1)/4`, since `(p + 1)/4 = 2^(w−2) − small`."""

    def __init__(self, w: int = WIDTH, c0: int = C0) -> None:
        import gmpy2

        self.g, self.w, self.c0 = gmpy2, w, c0
        self.p, self.mask, self.small = gmpy2.mpz((1 << w) - c0), gmpy2.mpz((1 << w) - 1), (c0 - 1) // 4

    def root(self, y: int) -> int:
        g, w, c0, mask, p = self.g, self.w, self.c0, self.mask, self.p
        x = g.mpz(y)
        for _ in range(w - 2):
            x = x * x
            x = (x & mask) + c0 * (x >> w)
            x = (x & mask) + c0 * (x >> w)
        return int(x * g.invert(g.powmod(g.mpz(y), self.small, p), p) % p)

    def rho(self, y: int) -> int:
        """`reference.rho` on this root: the parity-canonical root of ±y."""
        p = int(self.p)
        if y == 0:
            return 0
        s = self.root(y)
        sq = s * s % p
        if sq == y:
            return s if s % 2 == 0 else p - s
        if sq == (-y) % p:
            return s if s % 2 else p - s
        raise ArithmeticError("p is not a prime congruent to 3 mod 4")

    def encode(self, P: Any, m: int, k: int, t: int) -> int:
        from p2_v1 import reference as R

        return self.rho(R.sigma(P, self.rho((m + t) % P.p), k))


def _block_inputs(seed: int, i: int) -> tuple[Any, int, int, int]:
    import hashlib

    from p2_v1 import reference as R

    P = R.DEPLOYMENT
    prefix = R.expand_prefix(hashlib.sha256(b"scale/salt").digest(), P, hashlib.sha512(b"scale/model").digest())
    m = int.from_bytes(hashlib.shake_256(f"scale/m/{seed}/{i}".encode()).digest(PAYLOAD), "big")
    k, t, _ = R.keys_after(prefix, P, i // BPS, i % BPS)
    return P, m, k, t


def _cpu_worker(job: tuple[int, str, int, int]) -> dict[str, Any]:
    cpu, mode, blocks, seed = job
    from p2_v1 import reference as R

    os.sched_setaffinity(0, {cpu})
    chain = Chain()
    times, ok = [], []
    for i in range(blocks):
        P, m, k, t = _block_inputs(seed, i)
        t0 = time.perf_counter()
        c = chain.encode(P, m, k, t) if mode == "gmp-chain" else R.encode(P, m, k, t)
        times.append(time.perf_counter() - t0)
        ok.append(R.decode(P, c, k, t) == m)
    return {"cpu": cpu, "mode": mode, "block_s": times, "round_trip": all(ok)}


def cpu_cmd(a: argparse.Namespace) -> int:
    import multiprocessing as mp

    import gmpy2

    from p2_v1 import reference as R
    from verity.core.protocols.pous.schemes import p2 as S

    out = Path(a.out)
    cpus = [int(x) for x in a.cpus.split(",") if x]
    siblings = [int(x) for x in a.siblings.split(",") if x]
    os.sched_setaffinity(0, {cpus[0]})
    doc: dict[str, Any] = {"cpus": cpus, "siblings": siblings, "gmpy2": gmpy2.version(), "python": sys.version,
                           "uname": " ".join(os.uname()), "lscpu_model": _lscpu_model()}
    P, m, k, t = _block_inputs(a.seed, 0)
    chain = Chain()
    agree = chain.encode(P, m, k, t) == R.encode(P, m, k, t)
    doc["chain_equals_reference"] = agree
    y = (m + t) % P.p
    t0 = time.perf_counter()
    pow(y, (P.p + 1) // 4, P.p)
    doc["root_s_python_pow"] = time.perf_counter() - t0
    f = S._Field(WIDTH, C0)
    t0 = time.perf_counter()
    f._sqrt(y)
    doc["root_s_schemes_p2_field"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    chain.root(y)
    doc["root_s_gmp_chain"] = time.perf_counter() - t0
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        doc["one_core"] = {mode: pool.map(_cpu_worker, [(cpus[0], mode, a.blocks, a.seed)])[0]
                           for mode in ("gmp-powmod", "gmp-chain")}
    dump(out, "cpu.json", doc)
    for name, lst in (("physical", cpus), ("smt", cpus + siblings)):
        if name == "smt" and not siblings:
            continue
        t0 = time.perf_counter()
        with ctx.Pool(len(lst)) as pool:
            res = pool.map(_cpu_worker, [(c, "gmp-chain", a.blocks, a.seed + 1 + j) for j, c in enumerate(lst)])
        wall = time.perf_counter() - t0
        per = [x for r in res for x in r["block_s"]]
        doc[f"aggregate_{name}"] = {"processes": len(lst), "blocks": len(per), "wall_s": wall,
                                    "block_s": quantiles(per), "blocks_per_s": len(lst) / statistics.median(per),
                                    "round_trip": all(r["round_trip"] for r in res)}
        dump(out, "cpu.json", doc)
    one = statistics.median(doc["one_core"]["gmp-chain"]["block_s"])
    doc["summary"] = {"block_s_one_core_gmp_chain": one,
                      "block_s_one_core_gmp_powmod": statistics.median(doc["one_core"]["gmp-powmod"]["block_s"]),
                      "core_hours_2^25": (1 << 25) * one / 3600, "core_hours_2^26": (1 << 26) * one / 3600}
    dump(out, "cpu.json", doc)
    print(json.dumps(doc["summary"]))
    return 0 if agree and all(r["round_trip"] for r in doc["one_core"].values()) else 1


def _lscpu_model() -> str:
    try:
        return next((ln.split(":", 1)[1].strip() for ln in subprocess.run(["lscpu"], capture_output=True, text=True).stdout
                     .splitlines() if ln.startswith("Model name")), "")
    except OSError:
        return ""


# --- the setup check -----------------------------------------------------------------------------------------------------

def _setup_encode(job: tuple[Any, bytes, bytes, int, int]) -> list[bytes]:
    q, salt, w_seg, lo, hi = job
    from verity.core.protocols.pous.schemes import p2 as S

    chain, keys, p = Chain(q.w, q.c0), S.segment_keys(q, salt, 0), (1 << q.w) - q.c0
    pb, nb, out = q.payload_bytes, q.block_bytes, []
    for j in range(lo, hi):
        k, t = keys(j)
        m = int.from_bytes(w_seg[j * pb:(j + 1) * pb], "big")
        r = chain.rho((m + t) % p)
        x = r ^ k
        out.append(chain.rho(x if x < p else r).to_bytes(nb, "big"))
    return out


def setup_check_cmd(a: argparse.Namespace) -> int:
    import multiprocessing as mp

    from verity.core.protocols.pous import setup
    from verity.core.protocols.pous.schemes import p2 as S
    from verity.ml.commitments.identity import canonical_json_bytes

    out = Path(a.out)
    scratch = Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    q = S.P2Params(WIDTH, C0, a.blocks, 1)
    w = os.urandom(q.w_bytes)
    commitment = setup.commit(q, w)
    salt = setup.draw_salt(os.urandom(32), commitment)
    step = -(-a.blocks // a.workers)
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        parts = pool.map(_setup_encode, [(q, salt, w, lo, min(lo + step, a.blocks)) for lo in range(0, a.blocks, step)])
    encode_s = time.perf_counter() - t0
    blocks = [b for part in parts for b in part]
    doc: dict[str, Any] = {"blocks": a.blocks, "workers": a.workers, "encode_wall_s": encode_s,
                           "manifest": q.manifest()}
    t0 = time.perf_counter()
    vk = setup.verifier_key_from_encoding(q, None, commitment, salt, [blocks])
    doc["python_check_s"] = time.perf_counter() - t0
    doc["python_root"] = vk.root.hex() if hasattr(vk, "root") else None
    params, comm, ev = scratch / "params.json", scratch / "commitment.bin", scratch / "evidence.bin"
    params.write_bytes(canonical_json_bytes(q.manifest()))
    comm.write_bytes(commitment + salt)
    ev.write_bytes(len([blocks]).to_bytes(4, "big") + len(blocks).to_bytes(4, "big")
                   + b"".join(len(b).to_bytes(4, "big") + b for b in blocks))
    doc["evidence_bytes"] = ev.stat().st_size
    if a.lean_exe:
        runs = []
        for _ in range(a.lean_reps):
            t0 = time.perf_counter()
            r = subprocess.run([a.lean_exe, "check", str(params), str(comm), str(ev)], capture_output=True, text=True)
            runs.append({"s": time.perf_counter() - t0, "rc": r.returncode, "stdout": r.stdout.strip()[:2000],
                         "stderr": r.stderr.strip()[-2000:]})
        doc["lean_check"] = runs
        accepted = [x for x in runs if x["rc"] == 0]
        doc["lean_check_s"] = min(x["s"] for x in accepted) if accepted else None
        lean_root = next((json.loads(x["stdout"]).get("public", {}).get("root") for x in accepted), None)
        doc["lean_root_equals_python"] = lean_root == doc["python_root"] if lean_root else None
    per = {k: doc[k] / a.blocks for k in ("python_check_s", "lean_check_s") if doc.get(k)}
    doc["per_block_s"] = per
    doc["extrapolated_2^25_s"] = {k: v * (1 << 25) for k, v in per.items()}
    dump(out, "setup-check.json", doc)
    print(json.dumps({"per_block_s": per, "extrapolated_2^25_s": doc["extrapolated_2^25_s"]}))
    return 0 if not a.lean_exe or doc.get("lean_check_s") else 1


# --- the device's encode and decode ---------------------------------------------------------------------------------------

def _timer(torch: Any):
    def timed(fn) -> tuple[float, Any]:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        r = fn()
        e.record()
        e.synchronize()
        return s.elapsed_time(e) / 1e3, r
    return timed


def _smi(query: str) -> list[str]:
    gpu = os.environ.get("GPU_LEASE_UUID") or (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    try:
        r = subprocess.run(["nvidia-smi", f"--id={gpu}", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return [x.strip() for x in r.stdout.strip().split(",")]
    except (OSError, subprocess.TimeoutExpired):
        return []


def gpu_cmd(a: argparse.Namespace) -> int:
    import torch

    from p2_v1 import reference as R
    from p2_v1.gpu import Kernels

    out = Path(a.out)
    batches = [int(x) for x in a.batches.split(",") if x]
    P = R.DEPLOYMENT.with_segments(max(1, -(-max(batches) // BPS)))
    salt, model = os.urandom(32), os.urandom(64)
    prefix = R.expand_prefix(salt, P, model)
    K = Kernels(a.p2_lib, a.build_dir, P, prefix)
    dev, timed = K.device, _timer(torch)
    props = torch.cuda.get_device_properties(dev)
    doc: dict[str, Any] = {"device": props.name, "sms": props.multi_processor_count, "total_memory": props.total_memory,
                           "smi": _smi("name,uuid,clocks.sm,clocks.max.sm,power.draw"), "encode": [], "torch": torch.__version__}

    def payload_rows(n: int, seed: int) -> Any:
        g = torch.Generator(device=dev).manual_seed(seed)
        m = torch.randint(0, 256, (n, NB), dtype=torch.uint8, device=dev, generator=g)
        m[:, :NB - PAYLOAD] = 0
        return m

    K.encode(payload_rows(64, 0), 0)
    torch.cuda.synchronize()
    K.raised()
    ok = True
    for j, n in enumerate(batches):
        m = payload_rows(n, 1 + j)
        reps = [timed(lambda: K.encode(m, 0)) for _ in range(3 if n <= 16 else 1)]
        s, c = min(reps, key=lambda x: x[0])
        flags = K.raised()
        rt = bool(torch.equal(K.decode(c, 0), m))
        picks = sorted({0, n - 1, n // 2})
        ref = []
        for i in picks:
            k, t, _ = R.keys_after(prefix, P, i // BPS, i % BPS)
            ci = int.from_bytes(bytes(c[i].cpu().numpy()), "big")
            ref.append(R.decode(P, ci, k, t) == int.from_bytes(bytes(m[i].cpu().numpy()), "big"))
        row = {"batch": n, "encode_s": s, "reps_s": [x[0] for x in reps], "blocks_per_s": n / s, "flags": flags,
               "round_trip": rt, "reference_decode": ref, "sm_clock_after": _smi("clocks.sm")}
        ok &= rt and all(ref) and flags == 0
        doc["encode"].append(row)
        dump(out, "gpu.json", doc)
        del m, c
    rows = a.decode_rows
    c = torch.randint(0, 256, (rows, NB), dtype=torch.uint8, device=dev)
    c[:, 0] &= 0x7F
    K.decode(c, 0)
    reps = [timed(lambda: K.decode(c, 0))[0] for _ in range(5)]
    K.raised()
    doc["decode"] = {"rows": rows, "reps_s": reps, "stored_GBps": rows * NB / statistics.median(reps) / 1e9,
                     "blocks_per_s": rows / statistics.median(reps)}
    best = max(doc["encode"], key=lambda r: r["blocks_per_s"])
    doc["summary"] = {"encode_blocks_per_s": best["blocks_per_s"], "at_batch": best["batch"],
                      "hours_2^25_one_gpu": (1 << 25) / best["blocks_per_s"] / 3600,
                      "hours_2^26_one_gpu": (1 << 26) / best["blocks_per_s"] / 3600,
                      "decode_stored_GBps": doc["decode"]["stored_GBps"], "gates": ok}
    dump(out, "gpu.json", doc)
    print(json.dumps(doc["summary"]))
    return 0 if ok else 1


# --- the store at scale: leaves, the timed audit, paths, and a served model beside it ----------------------------------

def _segment(seed: int, s: int) -> Any:
    """Segment `s` of the store: uniformly random 2,056-byte blocks with the top bit clear, so each is below p."""
    import numpy as np

    rng = np.random.Generator(np.random.PCG64([seed, s]))
    seg = np.frombuffer(bytearray(rng.bytes(BPS * NB)), dtype=np.uint8).reshape(BPS, NB)
    seg[:, 0] &= 0x7F
    return seg


def _leaves_job(job: tuple[int, int, bytes]) -> tuple[int, bytes]:
    seed, s, salt = job
    from p2_v1 import reference as R

    raw = _segment(seed, s).tobytes()
    return s, b"".join(R.leaf(salt, s, v, raw[v * NB:(v + 1) * NB]) for v in range(BPS))


def leaves_file(path: Path, seed: int, segments: int, salt: bytes, workers: int) -> float:
    import multiprocessing as mp

    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(workers) as pool, path.open("wb") as f:
        for _, leaves in pool.imap(_leaves_job, [(seed, s, salt) for s in range(segments)]):
            f.write(leaves)
    return time.perf_counter() - t0


def path_cost(blocks: int, reps: int, seed: int) -> dict[str, Any]:
    """A v4 RESPONSE's path (log2 B siblings, leaf toward root) gathered from a full tree of 64-byte nodes in host memory:
    per answer in Python, and amortized over a vectorized gather (the memory's own cost)."""
    import numpy as np

    h = blocks.bit_length() - 1
    rng = np.random.Generator(np.random.PCG64([seed, 1 << 40]))
    levels = [np.frombuffer(rng.bytes((blocks >> lv) * DIGEST), dtype=np.uint8).reshape(-1, DIGEST) for lv in range(h)]
    idx = rng.integers(0, blocks, reps)
    per = []
    for i in idx[:min(reps, 20_000)].tolist():
        t0 = time.perf_counter_ns()
        b"".join(levels[lv][(i >> lv) ^ 1].tobytes() for lv in range(h))
        per.append((time.perf_counter_ns() - t0) / 1e9)
    sib = (idx[:, None] >> np.arange(h)) ^ 1
    t0 = time.perf_counter_ns()
    np.concatenate([levels[lv][sib[:, lv]] for lv in range(h)], axis=1)
    vec = (time.perf_counter_ns() - t0) / 1e9 / reps
    return {"siblings": h, "path_bytes": h * DIGEST, "tree_bytes": sum(x.nbytes for x in levels) + DIGEST,
            "python_per_answer_s": quantiles(per), "vectorized_per_answer_s": vec}


def _echo_server(port_file: str, size: int, cpu: int) -> None:
    import socket

    os.sched_setaffinity(0, {cpu})
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    Path(port_file).write_text(str(srv.getsockname()[1]))
    conn, _ = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    buf = bytes(size)
    while True:
        req = conn.recv(16, socket.MSG_WAITALL)
        if len(req) < 16 or req[0] == 3:
            break
        conn.sendall(buf)
    conn.close()


def loopback_sizes(sizes: list[int], n: int, server_cpu: int, client_cpu: int, scratch: Path) -> dict[str, Any]:
    """Round trips of a 16-byte request answered by `size` bytes, Python on both ends, on the audit's two cores: the
    difference between sizes is what a path adds on the wire."""
    import multiprocessing as mp
    import socket

    ctx, out = mp.get_context("spawn"), {}
    here = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {client_cpu})
    try:
        for size in sizes:
            pf = scratch / f"echo-{size}.port"
            pf.unlink(missing_ok=True)
            proc = ctx.Process(target=_echo_server, args=(str(pf), size, server_cpu))
            proc.start()
            while not pf.exists() or not pf.read_text():
                time.sleep(0.01)
            s = socket.create_connection(("127.0.0.1", int(pf.read_text())))
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            req, rtt = bytes(16), []
            for j in range(n + 1000):
                t0 = time.perf_counter_ns()
                s.sendall(req)
                s.recv(size, socket.MSG_WAITALL)
                if j >= 1000:
                    rtt.append((time.perf_counter_ns() - t0) / 1e9)
            s.sendall(b"\x03" + bytes(15))
            s.close()
            proc.join()
            out[str(size)] = quantiles(rtt)
    finally:
        os.sched_setaffinity(0, here)
    return out


class MemSampler:
    """The GPU's used memory by nvidia-smi, twice a second, while a phase runs."""

    def __init__(self) -> None:
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            v = _smi("memory.used,memory.total")
            if v and v[0].isdigit():
                self.samples.append((time.time(), int(v[0])))
            self._stop.wait(0.5)

    def __enter__(self) -> MemSampler:
        self._t.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._t.join()

    def summary(self) -> dict[str, Any]:
        used = [m for _, m in self.samples]
        return {"samples": len(used), "peak_mib": max(used) if used else None, "series_mib": used[::4]}


VLLM_LINES = re.compile(r"(Model loading took|model weights take|KV cache|Maximum concurrency|Graph capturing|"
                        r"activation peak|non_torch|Memory profiling|gpu_memory_utilization|CUDA graph|torch.compile|"
                        r"Available|Free memory|Error|error)", re.I)


def fit_driver_cmd(a: argparse.Namespace) -> int:
    """vLLM's side of the fit (run under vLLM's interpreter): the model in FP8, then each batch of prompts of
    `--prompt-tokens` random tokens generating `--output-tokens` each, then a steady load until `--stop` appears."""
    import random

    from vllm import LLM, SamplingParams

    out = Path(a.out)
    doc: dict[str, Any] = {"model": a.model, "util": a.util, "max_model_len": a.max_model_len, "max_num_seqs": a.max_seqs,
                           "quantization": "fp8", "kv_cache_dtype": a.kv_cache_dtype, "batches": []}
    t0 = time.perf_counter()
    llm = LLM(model=a.model, quantization="fp8", gpu_memory_utilization=a.util, max_model_len=a.max_model_len,
              max_num_seqs=a.max_seqs, kv_cache_dtype=a.kv_cache_dtype, seed=0, enable_prefix_caching=False)
    doc["init_s"] = time.perf_counter() - t0
    dump(out, "fit-driver.json", doc)
    rng = random.Random(0)
    sp = SamplingParams(max_tokens=a.output_tokens, ignore_eos=True, temperature=0.0)

    def prompts(b: int) -> list[dict[str, list[int]]]:
        return [{"prompt_token_ids": [rng.randrange(1000, 120_000) for _ in range(a.prompt_tokens)]} for _ in range(b)]

    for b in [int(x) for x in a.batches.split(",") if x]:
        ps = prompts(b)
        t0 = time.perf_counter()
        res = llm.generate(ps, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        gen = sum(len(r.outputs[0].token_ids) for r in res)
        doc["batches"].append({"batch": b, "s": dt, "generated_tokens": gen, "output_tokens_per_s": gen / dt,
                               "complete": all(len(r.outputs[0].token_ids) == a.output_tokens for r in res)})
        dump(out, "fit-driver.json", doc)
    if a.serving:
        ps = prompts(a.load_batch)
        Path(a.serving).write_text("1")
        loads = 0
        while not Path(a.stop).exists():
            llm.generate(ps, sp, use_tqdm=False)
            loads += 1
        doc["load"] = {"batch": a.load_batch, "generate_calls": loads}
        dump(out, "fit-driver.json", doc)
    return 0


def audit_cmd(a: argparse.Namespace) -> int:
    import numpy as np  # noqa: F401  (the leaf pool's)

    from p2_v1 import reference as R

    out, scratch = Path(a.out), Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    exps = sorted(int(x) for x in a.log2.split(",") if x)
    doc: dict[str, Any] = {"log2": exps, "store_seed": a.store_seed, "stores": {}}
    stores: dict[int, dict[str, Any]] = {}
    for e in exps:
        blocks = 1 << e
        segments = blocks // BPS
        salt, model = os.urandom(32), os.urandom(64)
        leaves = scratch / f"leaves-2^{e}.bin"
        hash_s = leaves_file(leaves, a.store_seed + e, segments, salt, a.workers)
        stores[e] = {"blocks": blocks, "segments": segments, "salt": salt, "model": model, "leaves": leaves}
        doc["stores"][str(e)] = {"blocks": blocks, "store_bytes": blocks * NB, "leaf_hash_s": hash_s,
                                 "leaf_hash_workers": a.workers, "leaf_hash_blocks_per_s": blocks / hash_s}
        dump(out, "audit.json", doc)

    import torch

    from band_gpu import harness as H
    from p2_v1 import harness as P2H

    backend = P2H.gpu(a.p2_lib, a.build_dir)
    dev = backend.device
    timed = _timer(torch)
    free0, total = torch.cuda.mem_get_info(dev)
    doc["device"] = {**backend.facts(), "total_bytes": total, "free_before_bytes": free0}
    H.leave_reserved(a)
    script = P2H.MEMORY_CHALLENGER
    c = None
    for e in exps:
        st, rec = stores[e], doc["stores"][str(e)]
        if c is not None:
            del c
            torch.cuda.empty_cache()
        blocks, segments = st["blocks"], st["segments"]
        P = R.DEPLOYMENT.with_segments(segments)
        t0 = time.perf_counter()
        c = torch.empty((blocks, NB), dtype=torch.uint8, device=dev)
        for s in range(segments):
            c[s * BPS:(s + 1) * BPS].copy_(torch.from_numpy(_segment(a.store_seed + e, s)))
        torch.cuda.synchronize()
        rec["fill_s"] = time.perf_counter() - t0
        rec["free_after_store_bytes"] = torch.cuda.mem_get_info(dev)[0]
        bound = backend.bind(P, R.expand_prefix(st["salt"], P, st["model"]))
        chunk = min(blocks, 1 << 20)
        bound.decode(c[:chunk], 0)
        torch.cuda.synchronize()
        dec = 0.0
        for lo in range(0, blocks, chunk):
            s_, m_ = timed(lambda lo=lo: bound.decode(c[lo:lo + chunk], lo))
            dec += s_
            del m_
        bound.raised()
        rec["decode_all_s"] = dec
        rec["decode_stored_GBps"] = blocks * NB / dec / 1e9
        rec["path"] = path_cost(blocks, a.path_reps, a.store_seed)
        dump(out, "audit.json", doc)
        fields = {"salt": st["salt"].hex(), "model": st["model"].hex(), "manifest": P.manifest(), "payload_bytes": PAYLOAD}
        start = lambda: backend.responder(c, a.responder_cpu)  # noqa: E731
        cal = H.calibrate_limit(start, fields, script, out, a, st["leaves"], f"calibrate-2^{e}")
        limit = int(cal.get("limit_ns") or 0)
        honest = H.audit_round(start, fields, script, st["leaves"], out, f"honest-2^{e}", a, None, limit)
        rec["calibration"] = {k: cal.get(k) for k in ("limit_ns", "rtt_allowance_ns", "allowance") if k in cal}
        rec["honest"] = {k: honest.get(k) for k in ("rc", "k", "limit_ns", "audits", "audits_accepted", "answers", "late",
                                                     "wrong", "elapsed_s", "margin_s", "device_s", "server_s", "root",
                                                     "rtt_allowance_ns")}
        dump(out, "audit.json", doc)
    h_max = exps[-1]
    sizes = sorted({NB + 24} | {NB + 24 + 2 + DIGEST * e for e in exps})
    doc["loopback_by_size"] = loopback_sizes(sizes, a.loopback_reps, a.responder_cpu, a.verifier_cpu, scratch)
    dump(out, "audit.json", doc)
    if a.fit_python:
        doc["fit"] = fit_beside(a, backend, c, stores[h_max], out, scratch, script)
        dump(out, "audit.json", doc)
    accepted = all(r.get("honest", {}).get("audits_accepted") == r.get("honest", {}).get("audits") for r in doc["stores"].values())
    if a.fit_python:
        beside = doc["fit"].get("audit_beside") or {}
        accepted &= doc["fit"].get("driver_rc") == 0 and bool(beside) and beside.get("audits_accepted") == beside.get("audits")
    print(json.dumps({e: {k: doc["stores"][e]["honest"].get(k) for k in ("audits_accepted", "late")} for e in doc["stores"]}))
    return 0 if accepted else 1


def fit_beside(a: argparse.Namespace, backend: Any, c: Any, st: dict[str, Any], out: Path, scratch: Path,
               script: Path) -> dict[str, Any]:
    import torch

    from band_gpu import harness as H
    from p2_v1 import reference as R

    # the decode passes' cached labels would otherwise count against vLLM: only the store and the responder stay
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(backend.device)
    util = max(0.05, int((free - a.headroom_gib * (1 << 30)) / total * 1000) / 1000)
    rec: dict[str, Any] = {"store_blocks": st["blocks"], "store_bytes": c.numel(), "free_bytes": free, "total_bytes": total,
                           "gpu_memory_utilization": util, "headroom_gib": a.headroom_gib}
    serving, stop = scratch / "fit.serving", scratch / "fit.stop"
    for f in (serving, stop):
        f.unlink(missing_ok=True)
    cmd = [a.fit_python, str(Path(__file__).resolve()), "fit-driver", "--out", str(out), "--model", a.fit_model,
           "--util", str(util), "--max-model-len", str(a.max_model_len), "--max-seqs", str(a.max_seqs),
           "--batches", a.fit_batches, "--prompt-tokens", str(a.prompt_tokens), "--output-tokens", str(a.output_tokens),
           "--kv-cache-dtype", a.kv_cache_dtype, "--serving", str(serving), "--stop", str(stop),
           "--load-batch", str(a.load_batch)]
    if a.fit_cpus:
        cmd = ["taskset", "-c", a.fit_cpus, *cmd]
    rec["fit_cpus"] = a.fit_cpus or None
    # vLLM's own environment: this tree's packages on PYTHONPATH would shadow the venv's
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONPATH": ""}
    log = out / "fit.log"
    with MemSampler() as mem, log.open("w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        while proc.poll() is None and not serving.exists():
            time.sleep(0.5)
        if serving.exists():
            P = R.DEPLOYMENT.with_segments(st["segments"])
            fields = {"salt": st["salt"].hex(), "model": st["model"].hex(), "manifest": P.manifest(), "payload_bytes": PAYLOAD}
            start = lambda: backend.responder(c, a.responder_cpu)  # noqa: E731
            cal = H.calibrate_limit(start, fields, script, out, a, st["leaves"], "calibrate-beside")
            beside = H.audit_round(start, fields, script, st["leaves"], out, "honest-beside", a, None,
                                   int(cal.get("limit_ns") or 0))
            rec["audit_beside"] = {k: beside.get(k) for k in ("rc", "limit_ns", "audits", "audits_accepted", "answers", "late",
                                                               "wrong", "elapsed_s", "margin_s", "device_s", "server_s")}
            stop.write_text("1")
        rc = proc.wait()
    rec["driver_rc"] = rc
    rec["memory"] = mem.summary()
    text = log.read_text(errors="replace")
    rec["vllm_lines"] = [ln.strip()[-400:] for ln in text.splitlines() if VLLM_LINES.search(ln)][-80:]
    drv = out / "fit-driver.json"
    rec["driver"] = json.loads(drv.read_text()) if drv.exists() else None
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cpu")
    p.add_argument("--out", required=True)
    p.add_argument("--cpus", required=True, help="one process per listed CPU (distinct physical cores)")
    p.add_argument("--siblings", default="", help="their SMT siblings, for the SMT aggregate")
    p.add_argument("--blocks", type=int, default=2, help="blocks each process encodes")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("setup-check")
    p.add_argument("--out", required=True)
    p.add_argument("--scratch", required=True)
    p.add_argument("--blocks", type=int, default=4096)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lean-exe", default="")
    p.add_argument("--lean-reps", type=int, default=2)
    for name in ("gpu", "audit"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--p2-lib", required=True)
        p.add_argument("--build-dir", required=True)
    p = sub.choices["gpu"]
    p.add_argument("--batches", default="1,1024,4096,16384,65536")
    p.add_argument("--decode-rows", type=int, default=1 << 20)
    p = sub.choices["audit"]
    p.add_argument("--scratch", required=True)
    p.add_argument("--log2", default="19,25")
    p.add_argument("--store-seed", type=int, default=20261012,
                   help="the synthetic store's blocks (the memory challenger draws its own coins)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--audits", type=int, default=20)
    p.add_argument("--pings", type=int, default=5000)
    p.add_argument("--k", type=int, default=0)
    p.add_argument("--allowance", default="p99.9", choices=["p50", "p99.9", "max"])
    p.add_argument("--deadline-ns", type=int, default=0)
    p.add_argument("--clock", default="wall", choices=["wall", "logical"])
    p.add_argument("--verifier-cpu", type=int, required=True)
    p.add_argument("--responder-cpu", type=int, required=True)
    p.add_argument("--path-reps", type=int, default=200_000)
    p.add_argument("--loopback-reps", type=int, default=20_000)
    p.add_argument("--fit-python", default="", help="vLLM's interpreter: serve --fit-model beside the largest store")
    p.add_argument("--fit-model", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--fit-cpus", default="", help="taskset list for vLLM's processes (default: the job's)")
    p.add_argument("--fit-batches", default="1,8,32,64,128")
    p.add_argument("--headroom-gib", type=float, default=2.0)
    p.add_argument("--load-batch", type=int, default=32)
    for name in ("audit", "fit-driver"):
        p = sub.choices.get(name) or sub.add_parser(name)
        p.add_argument("--max-model-len", type=int, default=8192)
        p.add_argument("--max-seqs", type=int, default=128)
        p.add_argument("--prompt-tokens", type=int, default=1024)
        p.add_argument("--output-tokens", type=int, default=256)
        p.add_argument("--kv-cache-dtype", default="auto")
    p = sub.choices["fit-driver"]
    p.add_argument("--out", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--util", type=float, required=True)
    p.add_argument("--batches", default="1,8,32,64,128")
    p.add_argument("--serving", default="")
    p.add_argument("--stop", default="")
    p.add_argument("--load-batch", type=int, default=32)
    a = ap.parse_args(argv)
    return {"cpu": cpu_cmd, "setup-check": setup_check_cmd, "gpu": gpu_cmd, "audit": audit_cmd,
            "fit-driver": fit_driver_cmd}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
