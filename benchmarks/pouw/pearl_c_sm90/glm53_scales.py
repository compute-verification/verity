"""GLM-5.3's block-FP8 B, as Pearl-C on the H200 would widen it: is a weight's e4m3 code times its 128 x 128 block scale exact in
FP32, and what does vLLM's load-time ue8m0 requant (`verity_vllm.check.fp8_ue8m0.requant`, run when DeepGEMM's e8m0 scales are in
use) change?

Reads the pinned checkpoint (zai-org/GLM-5.3 at `REVISION`) from the Hub by HTTP range requests: every `weight_scale_inv`, and the
whole weights of `SAMPLES`.  Writes one JSON report:

* per role, over every block scale: how many, how many are powers of two, the scale significand's significant bits, and how many
  blocks are exact for every e4m3 code (the odd part of 15 · the scale's odd part fits 24 bits, and no product is subnormal);
* per sampled weight, over its actual codes: elements whose code x scale rounds in FP32, elements whose ue8m0-requantized value
  (code' x 2^e) differs from code x scale, and whether code' x 2^e is exact in FP32.

  uv run python benchmarks/pouw/pearl_c_sm90/glm53_scales.py OUT.json [--threads 32]
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import socket
import struct
import sys
import threading
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

REVISION = "aca966e4e02791568aa6a4ced368624b3d897f42"
BASE = f"https://huggingface.co/zai-org/GLM-5.3/resolve/{REVISION}/"
SHARDS = 141
G = 128
#: one weight of each GEMM kind Pearl-C would run at TP8 (dense layer 0, MoE layer 40)
SAMPLES = ("model.layers.0.self_attn.q_a_proj.weight", "model.layers.0.self_attn.q_b_proj.weight",
           "model.layers.0.self_attn.kv_a_proj_with_mqa.weight", "model.layers.0.self_attn.kv_b_proj.weight",
           "model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.down_proj.weight",
           "model.layers.40.mlp.shared_experts.up_proj.weight", "model.layers.40.mlp.experts.7.gate_proj.weight",
           "model.layers.40.mlp.experts.7.down_proj.weight", "model.layers.40.self_attn.indexer.wq_b.weight")


_resolved: dict[tuple, list] = {}
_getaddrinfo = socket.getaddrinfo


def _cached_getaddrinfo(host, port, *a, **kw):
    """Each host resolved once (59,044 requests; this VM's resolver drops some under load)."""
    key = (host, port, a, tuple(sorted(kw.items())))
    if key not in _resolved:
        for t in range(8):
            try:
                _resolved[key] = _getaddrinfo(host, port, *a, **kw)
                break
            except socket.gaierror:
                if t == 7:
                    raise
                threading.Event().wait(2.0 ** t)
    return _resolved[key]


socket.getaddrinfo = _cached_getaddrinfo
_conns = threading.local()


def get(url: str, a: int, b: int, tries: int = 6) -> bytes:
    """Bytes [a, b) of url over this thread's keep-alive connection to its host."""
    u = urllib.parse.urlsplit(url)
    path = u.path + ("?" + u.query if u.query else "")
    for t in range(tries):
        pool = getattr(_conns, "pool", None)
        if pool is None:
            pool = _conns.pool = {}
        try:
            c = pool.get(u.netloc)
            if c is None:
                c = pool[u.netloc] = http.client.HTTPSConnection(u.netloc, timeout=120)
            c.request("GET", path, headers={"Range": f"bytes={a}-{b - 1}"})
            r = c.getresponse()
            data = r.read()
            if r.status != 206 or len(data) != b - a:
                raise IOError(f"{u.netloc}: status {r.status}, {len(data)} bytes for range {a}-{b}")
            return data
        except (OSError, http.client.HTTPException):
            pool.pop(u.netloc, None)
            if t == tries - 1:
                raise
            threading.Event().wait(2.0 ** t)
    raise AssertionError


def header(url: str) -> tuple[str, int, dict]:
    """The shard's final URL (the Hub's redirect, resolved once), its data start and its header."""
    for t in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-7"}), timeout=120) as r:  # noqa: S310  (BASE's https URL)
                final, n = r.geturl(), struct.unpack("<Q", r.read())[0]
            break
        except OSError:
            if t == 5:
                raise
            threading.Event().wait(2.0 ** t)
    return final, 8 + n, json.loads(get(final, 8, 8 + n))


def role(name: str) -> str:
    name = re.sub(r"^model\.layers\.\d+\.", "", name.removesuffix(".weight_scale_inv").removesuffix(".weight"))
    name = re.sub(r"experts\.\d+\.", "experts.*.", name)
    return name


def scale_stats(s: np.ndarray) -> dict:
    """Over f32 block scales: powers of two, significant bits of the significand, blocks exact for every e4m3 code."""
    u = s.view(np.uint32).astype(np.int64)
    exp, man = (u >> 23) & 0xFF, u & 0x7FFFFF
    normal = (exp > 0) & (exp < 255)
    sig = np.where(normal, man | 0x800000, man)
    tz = np.where(sig > 0, np.log2(np.maximum(sig & -sig, 1)).astype(np.int64), 0)
    bits = np.where(sig > 0, 24 - tz, 0)
    odd = np.where(sig > 0, sig >> tz, 0)
    fits = np.ceil(np.log2(np.maximum(odd * 15, 1) + 1)) <= 24
    smallest = s.astype(np.float64) * 2.0 ** -9
    exact_all = normal & fits & (smallest >= 2.0 ** -126) & (s.astype(np.float64) * 448 < 3.4e38)
    return {"blocks": int(s.size), "pow2": int(((man == 0) & normal).sum()), "exact_every_code": int(exact_all.sum()),
            "significant_bits": {int(k): int(v) for k, v in sorted(Counter(bits.tolist()).items())},
            "min": float(s.min()), "max": float(s.max())}


def weight_stats(w: np.ndarray, s: np.ndarray) -> dict:
    from verity_vllm.check.fp8_ue8m0 import requant
    from verity.ml.definitions.numerics.fp8_twins import np_e4m3_to_f32
    N, K = w.shape
    full = np.repeat(np.repeat(s, G, axis=0), G, axis=1)[:N, :K].astype(np.float64)
    code = np_e4m3_to_f32(w).astype(np.float64)
    exact = code * full
    f32 = exact.astype(np.float32)
    w2, s2 = requant(w, s.astype(np.float32).view(np.uint32), G)
    full2 = np.repeat(np.repeat(s2.view(np.float32), G, axis=0), G, axis=1)[:N, :K].astype(np.float64)
    served = np_e4m3_to_f32(w2).astype(np.float64) * full2
    nonzero = code != 0
    return {"shape": [N, K], "elements": int(w.size), "nonzero": int(nonzero.sum()),
            "inexact_in_f32": int((f32.astype(np.float64) != exact).sum()),
            "blocks_with_an_inexact_element": int(((f32.astype(np.float64) != exact).reshape(
                -(-N // G), G, -(-K // G), G).any(axis=(1, 3))).sum()) if N % G == 0 and K % G == 0 else None,
            "requant_changes_value": int((served != exact).sum()),
            "requant_changes_code": int((w2 != w).sum()),
            "requant_exact_in_f32": bool((served.astype(np.float32).astype(np.float64) == served).all()),
            "requant_scale_pow2": bool(((s2 & 0x7FFFFF) == 0).all()),
            "max_rel_change": float(np.max(np.abs(served - exact)[nonzero] / np.abs(exact)[nonzero])) if nonzero.any() else 0.0}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args(argv)
    urls = [BASE + f"model-{i:05d}-of-{SHARDS:05d}.safetensors" for i in range(1, SHARDS + 1)]
    with ThreadPoolExecutor(args.threads) as ex:
        heads = list(ex.map(header, urls))
    jobs, where = [], {}
    for url, start, h in heads:
        for name, t in h.items():
            if name == "__metadata__":
                continue
            a, b = t["data_offsets"]
            where[name] = (url, start + a, start + b, t["dtype"], t["shape"])
            if name.endswith(".weight_scale_inv"):
                jobs.append(name)
    print(f"{len(jobs)} block-scale tensors in {SHARDS} shards", file=sys.stderr)

    def fetch(name: str) -> tuple[str, np.ndarray]:
        url, a, b, dtype, shape = where[name]
        if dtype != "F32":
            raise ValueError(f"{name}: {dtype}")
        return name, np.frombuffer(get(url, a, b), dtype="<f4").reshape(shape)

    scales: dict[str, list[np.ndarray]] = defaultdict(list)
    kept: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(args.threads) as ex:
        for name, s in ex.map(fetch, jobs):
            scales[role(name)].append(s.ravel())
            if name.removesuffix("_scale_inv") in SAMPLES:
                kept[name] = s
    roles = {r: scale_stats(np.concatenate(v)) for r, v in sorted(scales.items())}
    every = {k: sum(r[k] for r in roles.values()) for k in ("blocks", "pow2", "exact_every_code")}
    every["significant_bits"] = dict(sorted(sum((Counter(r["significant_bits"]) for r in roles.values()), Counter()).items()))
    every["min"], every["max"] = min(r["min"] for r in roles.values()), max(r["max"] for r in roles.values())
    samples = {}
    for name in SAMPLES:
        if name not in where:
            samples[name] = "absent"
            continue
        url, a, b, dtype, shape = where[name]
        w = np.frombuffer(get(url, a, b), dtype=np.uint8).reshape(shape)
        samples[name] = weight_stats(w, kept[name + "_scale_inv"])
        print(name, samples[name], file=sys.stderr)
    report = {"checkpoint": f"zai-org/GLM-5.3@{REVISION}", "block": G, "all": every, "roles": roles, "samples": samples}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({"all": every, "samples": {k: (v if isinstance(v, str) else {x: v[x] for x in (
        "inexact_in_f32", "requant_changes_value", "requant_exact_in_f32")}) for k, v in samples.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
