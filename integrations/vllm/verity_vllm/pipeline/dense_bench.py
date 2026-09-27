"""`verity-vllm pous-dense-bench`: the frozen dense-decode benchmark (POUS `docs/dense-decode-benchmark.md`).

    verity-vllm pous-dense-bench --out DIR --shapes workloads/dense_decode_shapes.json --build-dir DIR [--models M] [--reps 3]

Scheme: `verity_pous.dense.DENSE` at `dense.SECURE_FIRST` (512 blocks of 64 KiB per segment).  Workload: each model's linear
weights (vLLM's merged shapes, bf16) laid out back to back into segments; one model pass decodes every tile's blocks of `C`
(`dense.decode_blocks`, on the device clock), then runs each shape's `y = x W^T` at T = 1, 8, 32 and 2048 tokens on its fresh
weight, against the same GEMMs from plaintext weights.  `C` is pseudo-random bytes (any `C` is the encoding of its decode) under a fixed salt.

Gates, failing the run closed: `vectors`, the reference known-answer vectors `dense/small/blocks` and `dense/64kb/prefix16`
(GPU encode, then decode back); `reference_decode`, GPU against reference decode of sampled blocks (0, 1, B - 1 and 5 uniform
ones) of `--gate-segments` segments.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import statistics
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from verity_vllm.config import option

KECCAK_PER_CALL = 10 * 2 * 483
LANE_INSTR_PER_KECCAK = 3000
HBM_PEAK = 864e9
BATCHES = (1, 8, 32, 2048)

@dataclasses.dataclass(frozen=True, kw_only=True)
class Options:
    """the frozen dense-decode microbenchmark: encoded tile -> plaintext -> GEMM, against plaintext GEMMs"""

    out: str = option("--out", required=True, help="where dense-bench.json goes")
    shapes: str = option("--shapes", required=True, help="the workload's shapes (integrations/vllm/workloads/dense_decode_shapes.json)")
    build_dir: str = option("--build-dir", required=True, help="parent directory of the pous_native JIT build")
    models: str = option("--models", default="", help="models of workloads/dense_decode_shapes.json, comma separated (default all)")
    reps: int = option("--reps", type=int, default=3, help="timed repetitions after one warm-up; medians are reported")
    thread_rows: int = option("--thread-rows", type=int, default=4096, help="rows from which a Feistel launch runs a thread per row")
    word_io: bool = option("--word-io", action="store_true", default=False, help="one-thread Π₂ launches use 64-bit word I/O")
    gate_segments: int = option("--gate-segments", type=int, default=2, help="segments whose sampled blocks the reference decodes")
    workers: int = option("--workers", type=int, default=12, help="processes for the reference side of G2")
    result: str | None = option("--result", default=None, help="also write the research result.json here")
    run_id: str | None = option("--run-id", default=None, help="the research run id result.json names")


def models(path: str) -> dict[str, dict[str, Any]]:
    return json.loads(Path(path).read_text())["models"]


def model_shapes(m: dict[str, Any]) -> list[tuple[str, int, int]]:
    """The model's measured linear weights in the order served: `(name, out, in)`."""
    shapes = [(f"layers.{i}.{s}", o, k) for i in range(m["measured_layers"]) for s, o, k in m["layer"]]
    return shapes + [tuple(m["head"])]


def placements(shapes: list[tuple[str, int, int]]) -> list[tuple[str, int, int, int, int]]:
    """Back to back from byte 0: `(name, out, in, offset, bytes)`."""
    out, at = [], 0
    for name, o, k in shapes:
        out.append((name, o, k, at, o * k * 2))
        at += o * k * 2
    return out


def _vectors_module():
    import verity_pous

    path = Path(verity_pous.__file__).resolve().parents[1] / "tests" / "pous_vectors.py"
    spec = importlib.util.spec_from_file_location("pous_vectors", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, json.loads((path.parent / "fixtures" / "vectors.json").read_text())


def gate_vectors() -> dict[str, Any]:
    """The GPU encode reproduces the reference's dense known-answer vectors, and decodes back."""
    import torch
    from verity_pous import dense as ref

    from verity_vllm.engine.pous import codec, dense

    mod, fixtures = _vectors_module()
    small = ref.DenseParams(8, 64, 2)
    w, salt = mod._bytes("dense-small-w", small.w_bytes), mod._bytes("salt", 24)
    wt = torch.tensor(list(w), dtype=torch.uint8, device="cuda").view(2, 8, 8)
    tags = codec.segment_tags(salt, [0, 1], "cuda")
    c = dense.encode(wt, tags)
    small_ok = [bytes(r) for r in c.view(-1, 8).cpu().tolist()] == [bytes.fromhex(x) for x in fixtures["dense/small/blocks"]]
    small_back = torch.equal(dense.decode(c, tags), wt)
    sf = ref.SECURE_FIRST
    w16 = mod._bytes("dense-w", 16 * sf.block_bytes)
    wt = torch.tensor(list(w16), dtype=torch.uint8, device="cuda").view(1, 16, sf.block_bytes)
    tags = codec.segment_tags(mod._bytes("salt", 24), [0], "cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = dense.encode(wt, tags)
    torch.cuda.synchronize()
    enc_s = time.perf_counter() - t0
    prefix_ok = hashlib.sha256(c.cpu().numpy().tobytes()).hexdigest() == fixtures["dense/64kb/prefix16"]
    prefix_back = torch.equal(dense.decode(c, tags), wt)
    calls = 16 * 17 // 2
    per_call = enc_s / calls
    return {"small_blocks": small_ok, "small_decodes_back": small_back, "prefix16": prefix_ok, "prefix16_decodes_back": prefix_back,
            "prefix16_encode_s": enc_s, "encode_per_call_s": per_call,
            "encode_segment_s_extrapolated": per_call * sf.blocks_per_segment * (sf.blocks_per_segment + 1) // 2,
            "passed": small_ok and small_back and prefix_ok and prefix_back}


def _reference_block(job: tuple[bytes, int, int, list[bytes]]) -> bytes:
    from verity_pous import dense as ref
    from verity_pous.labels import salt_value, tag
    from verity_pous.oracle import KeyQuery
    from verity_pous.oracle import run as run_prog
    from verity_pous.primitives import key_program

    salt, s, i, blocks = job
    params = ref.SECURE_FIRST
    prims = ref.reference(params)
    ints = [int.from_bytes(b, "little") for b in blocks]
    q = KeyQuery(tag(salt_value(salt), s), 0, i, tuple((u, ints[u]) for u in range(i)))
    key, _ = run_prog(key_program(prims.h, q), prims)
    return (key ^ ints[i]).to_bytes(params.block_bytes, "little")


def gate_reference(c_flat, tags, salt: bytes, segments: list[int], workers: int) -> dict[str, Any]:
    """The GPU decode of sampled blocks of `segments` equals the reference's."""
    import torch

    from verity_vllm.engine.pous import dense

    b = c_flat.shape[0] // tags.shape[0]
    picks = []
    for s in segments:
        rng = hashlib.sha256(f"g2/{s}".encode()).digest()
        picks += [(s, i) for i in sorted({0, 1, b - 1, *(int.from_bytes(rng[2 * k:2 * k + 2], "little") % b for k in range(5))})]
    segs = torch.tensor([p[0] for p in picks], device=c_flat.device)
    blks = torch.tensor([p[1] for p in picks], device=c_flat.device)
    gpu = dense.decode_blocks(c_flat, tags, segs, blks, b).cpu().numpy()
    host = {s: [bytes(r) for r in c_flat[s * b:(s + 1) * b].cpu().numpy()] for s in segments}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        ref = list(pool.map(_reference_block, [(salt, s, i, host[s][:i + 1]) for s, i in picks]))
    same = [gpu[k].tobytes() == ref[k] for k in range(len(picks))]
    return {"blocks": [list(p) for p in picks], "equal": same, "passed": all(same)}


def _timed(fn, reps: int, warmup: bool = True) -> list[float]:
    import torch

    if warmup:
        fn()
    out = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) / 1e3)
    return out


def fill(c, label: str) -> None:
    """`C` as pseudo-random rows: row `r` is SHAKE256(sha256(label) || u24(r))."""
    import torch

    from verity_vllm.engine.pous import native

    seed = torch.tensor(list(hashlib.sha256(label.encode()).digest()), dtype=torch.uint8, device=c.device)
    shifts = torch.tensor([0, 8, 16], device=c.device)
    for k in range(0, c.shape[0], 1024):
        rows = c[k:k + 1024]
        idx = torch.arange(k, k + rows.shape[0], device=c.device).view(-1, 1)
        prefix = torch.cat([seed.expand(rows.shape[0], -1), (idx >> shifts).to(torch.uint8)], 1).contiguous()
        rows.zero_()
        native.ext().shake256_rows(prefix, prefix.new_empty((rows.shape[0], 0)), rows, True)


def run_model(name: str, m: dict[str, Any], reps: int, salt: bytes) -> dict[str, Any]:
    """One model pass per repetition: decode every tile's blocks from `C` (`dense.decode_blocks`, all at once), then each
    shape's GEMM at each T on its freshly decoded weight; and the same GEMMs from plaintext weights.  Also the latency of one
    small tile and of one full-segment tile decoded alone."""
    import torch

    from verity_pous import dense as ref

    from verity_vllm.engine.pous import codec, dense

    params = ref.SECURE_FIRST
    b, width = params.blocks_per_segment, params.block_bytes
    shapes = placements(model_shapes(m))
    total = shapes[-1][3] + shapes[-1][4]
    segments = -(-total // params.segment_bytes)
    c = torch.empty(segments * b, width, dtype=torch.uint8, device="cuda")
    fill(c, f"dense-bench/{name}")
    tags = codec.segment_tags(salt, list(range(segments)), "cuda")
    nblocks = -(-total // width)
    blocks = torch.arange(nblocks, device="cuda")
    segs, blks = blocks // b, blocks % b
    calls = int((blks + 1).sum())

    def views(flat):
        return [flat[off:off + nbytes].view(torch.bfloat16).view(o, k) for _, o, k, off, nbytes in shapes]

    held = []

    def decode_pass():
        held[:] = [None]
        held[:] = [dense.decode_blocks(c, tags, segs, blks, b).view(-1)]

    torch.cuda.synchronize()
    decode_reps = _timed(decode_pass, reps, warmup=False)
    weights = views(held.pop())
    xs = {t: [torch.randn(t, k, dtype=torch.bfloat16, device="cuda") for _, _, k, _, _ in shapes] for t in BATCHES}
    gemm = {t: statistics.median(_timed(lambda: [torch.nn.functional.linear(x, w) for x, w in zip(xs[t], weights)], reps))
            for t in BATCHES}
    plain = [w.clone() for w in weights]
    del weights
    base = {t: statistics.median(_timed(lambda: [torch.nn.functional.linear(x, w) for x, w in zip(xs[t], plain)], reps))
            for t in BATCHES}
    del plain
    torch.cuda.empty_cache()
    _, o, k, off, nbytes = shapes[0]
    small = torch.arange(off // width, (off + nbytes - 1) // width + 1, device="cuda")
    seg0 = torch.arange(b, device="cuda")
    latency = {"first_tile_blocks": int(small.numel()),
               "first_tile_s": _timed(lambda: dense.decode_blocks(c, tags, small // b, small % b, b), 1, warmup=False)[0],
               "segment_s": _timed(lambda: dense.decode_blocks(c, tags, seg0 // b, seg0 % b, b), 1, warmup=False)[0]}
    print(json.dumps({"model": name, "decode_reps_s": [round(x, 2) for x in decode_reps], **latency}), flush=True)
    return {"model": name, "segments": segments, "blocks": nblocks, "calls": calls, "weight_bytes": total,
            "decode_reps_s": decode_reps, "gemm_s": gemm, "plain_gemm_s": base, "latency": latency,
            **summarize(m, total, statistics.median(decode_reps), calls, gemm, base)}


def summarize(m: dict[str, Any], nbytes: int, decode: float, calls: int, gemm: dict, base: dict) -> dict[str, Any]:
    """EWB(T) = weight bytes / pass time (decode + the GEMMs), POUS and plaintext, and the Keccak rate.  Where only some layers
    are measured, the rest repeat them, so the ratio is the same and the model's weight bytes scale."""
    head = m["head"][1] * m["head"][2] * 2
    scale = ((nbytes - head) * m["layers"] / m["measured_layers"] + head) / nbytes
    out = {"model_weight_bytes": nbytes * scale, "decode_s": decode, "ewb": {}, "plain_ewb": {}, "op_s": {}}
    for t in BATCHES:
        out["op_s"][t] = decode + gemm[t]
        out["ewb"][t] = nbytes / out["op_s"][t]
        out["plain_ewb"][t] = nbytes / base[t]
    out["keccak_per_s"] = calls * KECCAK_PER_CALL / decode
    return out


def device_facts() -> dict[str, Any]:
    import torch

    q = "name,driver_version,clocks.sm,clocks.max.sm"
    smi = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1:]
    props = torch.cuda.get_device_properties(0)
    return {"nvidia_smi": smi, "torch": torch.__version__, "cuda": torch.version.cuda, "nvcc": nvcc, "sms": props.multi_processor_count}


def main(a: Options) -> int:
    from verity_vllm.engine.pous import codec, native

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    native.load(a.build_dir)
    codec.THREAD_ROWS[0] = a.thread_rows
    codec.WORD_IO[0] = a.word_io
    doc: dict[str, Any] = {"options": dataclasses.asdict(a), "device": device_facts(), "gates": {"vectors": gate_vectors()}}
    salt = bytes(range(24))
    table = models(a.shapes)
    doc["models"] = {}
    for name in (a.models.split(",") if a.models else list(table)):
        doc["models"][name] = run_model(name, table[name], a.reps, salt)
    import torch

    from verity_pous import dense as ref

    c_probe = torch.empty(a.gate_segments * ref.SECURE_FIRST.blocks_per_segment, ref.SECURE_FIRST.block_bytes, dtype=torch.uint8,
                          device="cuda")
    fill(c_probe, "dense-bench/G2")
    doc["gates"]["reference_decode"] = gate_reference(c_probe, codec.segment_tags(salt, list(range(a.gate_segments)), "cuda"), salt,
                                        list(range(a.gate_segments)), a.workers)
    clock_mhz = float(doc["device"]["nvidia_smi"].split(",")[3].split()[0])
    for m in doc["models"].values():
        m["integer_share"] = m["keccak_per_s"] * LANE_INSTR_PER_KECCAK / (doc["device"]["sms"] * 64 * clock_mhz * 1e6)
    passed = all(g["passed"] for g in doc["gates"].values())
    (out / "dense-bench.json").write_text(json.dumps(doc, indent=1, default=str))
    for name, m in doc["models"].items():
        print(json.dumps({"model": name, "ewb_MBps": {t: round(v / 1e6, 2) for t, v in m["ewb"].items()},
                          "plain_ewb_GBps": {t: round(v / 1e9, 1) for t, v in m["plain_ewb"].items()},
                          "keccak_per_s": round(m["keccak_per_s"] / 1e6), "integer_share": round(m["integer_share"], 3)}), flush=True)
    if a.result:
        rows = [{"name": f"{n}_ewb_T{t}", "value": v, "unit": "B/s"} for n, r in doc["models"].items() for t, v in r["ewb"].items()]
        rows += [{"name": f"{n}_keccak_per_s", "value": r["keccak_per_s"], "unit": "1/s"} for n, r in doc["models"].items()]
        Path(a.result).write_text(json.dumps({
            "schema": "research/result/v0.1", "run_id": a.run_id,
            "workload_fingerprint": {"benchmark": "dense-decode/v1", "models": list(doc["models"]), "thread_rows": a.thread_rows,
                                     "word_io": a.word_io},
            "validation": {"status": "passed" if passed else "failed", "checker": "verity-vllm pous-dense-bench gates",
                           "detail": json.dumps({k: v["passed"] for k, v in doc["gates"].items()}), "evidence": ["dense-bench.json"]},
            "measurements": rows, "measurement_files": ["dense-bench.json"], "artifacts": ["dense-bench.json"]}, indent=1))
    return 0 if passed else 1
