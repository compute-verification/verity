"""A `p2-16448/v4` store encoded on the device, one shard (one segment of 65,536 blocks) at a time, by one worker per visible
GPU (`pod.sh IMPL=p2-encode` builds `libp2v1.so` and runs it):

    encode_store.py --p2-lib LIB --salt HEX --segments 1038 --shards A:B [--descending] [--blocks N]
        [--weights fixture|MODULE:FUNC] [--write DIR] [--stop-file PATH] --out DIR

- The scheme is `verity.core.protocols.pous.schemes.p2`'s, unforked: the keys are `p2v1_keys` off `p2.expand_prefix(params,
  salt)` (24-byte salt, the 128-byte prefix), block `i = 65,536·s + v` keyed by `(s, v)`, and the encode is `p2v1_encode`.
- Shards `A..B−1` are claimed in order, from `A` upward or with `--descending` from `B−1` downward, under a lock on
  `OUT/claims`, so two encoders on two nodes can start from the two ends of the store. Before each claim a worker reads
  `--stop-file` when it exists: the nearest shard the other encoder has produced, at which it stops.
- `--blocks N` encodes only each shard's first `N` blocks (a dry run); the default is the whole segment.
- `--weights fixture` is segment `s` of W as `SHAKE256("verity/p2-encode/fixture-w/v1\\0" ‖ u64be(s))`, so any node
  regenerates it; `MODULE:FUNC` is `FUNC(s) -> bytes`, segment `s` of W (2,054 bytes a block).
- Every shard is checked before its record is written: the kernels raised no flag, every block is canonical (below `p`) and
  decodes back on the device to its payload, and `--ref-blocks` of them (the first, the last, and pseudo-random ones) equal
  `reference.encode` byte for byte. A failed check stops that worker, and the run exits nonzero.
- Each shard's record (`OUT/shards/seg-SSSS.json`) has the SHA-512 of its encoded blocks in order (of the segment file,
  for a whole segment) and of its W bytes, its leaf root (`verifier.MerkleTree` over its blocks' `verifier.leaf_hash`: for a
  whole segment, its aligned 2^16-leaf subtree of the store's tree, so the store's root is `MerkleTree(segment roots).root`), the
  encode's device time and its rate. `OUT/manifest.json` gathers them with each GPU's rate.
- `--write DIR` keeps each shard's blocks as `DIR/seg-SSSS.bin` (134,742,016 bytes for a whole segment)."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FIXTURE_TAG = b"verity/p2-encode/fixture-w/v1\0"


def fixture_segment(s: int, nbytes: int) -> bytes:
    return hashlib.shake_256(FIXTURE_TAG + s.to_bytes(8, "big")).digest(nbytes)


def weights_source(spec: str, segment_bytes: int) -> Callable[[int], bytes]:
    if spec == "fixture":
        return lambda s: fixture_segment(s, segment_bytes)
    module, _, func = spec.partition(":")
    if not func:
        raise SystemExit(f"--weights is 'fixture' or MODULE:FUNC, not {spec!r}")
    return getattr(importlib.import_module(module), func)


def store_root(segment_roots: list[bytes]) -> bytes:
    """The store's root from its whole segments' subtree roots, in order: segments are aligned 2^16-leaf subtrees of the
    RFC 6962 tree, whose splits over a multiple of 2^16 leaves fall on segment boundaries."""
    from verity.core.protocols.pous.verifier import MerkleTree
    return MerkleTree(segment_roots).root


def shard_order(lo: int, hi: int, descending: bool) -> list[int]:
    return list(range(hi - 1, lo - 1, -1)) if descending else list(range(lo, hi))


def read_stop(path: str | None) -> int | None:
    if not path or not os.path.exists(path):
        return None
    text = Path(path).read_text().strip()
    return int(text) if text else None


def stopped(shard: int, stop: int | None, descending: bool) -> bool:
    return stop is not None and (shard <= stop if descending else shard >= stop)


def claim(out: Path, order: list[int], stop_file: str | None, descending: bool) -> int | None:
    """The next unclaimed shard of `order`, or None when none is left or the stop file reaches it."""
    with open(out / "claims", "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        taken = int(f.read().strip() or 0)
        if taken >= len(order) or stopped(order[taken], read_stop(stop_file), descending):
            return None
        f.seek(0)
        f.truncate()
        f.write(str(taken + 1))
        return order[taken]


def ref_picks(s: int, n: int, k: int) -> list[int]:
    picks = [0, n - 1]
    seed = hashlib.sha512(b"verity/p2-encode/ref-picks/v1\0" + s.to_bytes(8, "big")).digest()
    for j in range(max(0, k - 2)):
        picks.append(int.from_bytes(seed[8 * j:8 * j + 8], "big") % n)
    return sorted(set(picks))[:max(1, k)]


def smi(gpu: str, query: str) -> list[str]:
    try:
        r = subprocess.run(["nvidia-smi", f"--id={gpu}", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10, check=False)
        return [x.strip() for x in r.stdout.strip().split(",")]
    except (OSError, subprocess.TimeoutExpired):
        return []


def worker(a: argparse.Namespace) -> int:
    import numpy as np
    import torch

    from p2_v1 import reference as R
    from p2_v1.gpu import Kernels
    from verity.core.protocols.pous.schemes import p2
    from verity.core.protocols.pous.verifier import MerkleTree, leaf_hash

    out, salt = Path(a.out), bytes.fromhex(a.salt)
    params = p2.P2Params(*p2.POINT, segments=a.segments)
    rp = R.DEPLOYMENT.with_segments(a.segments)
    bps, nb, pb = params.blocks_per_segment, params.block_bytes, params.payload_bytes
    n = a.blocks or bps
    if not 0 < n <= bps:
        raise SystemExit(f"--blocks is 1..{bps}")
    prefix = p2.expand_prefix(params, salt)
    K = Kernels(a.p2_lib, "", rp, prefix)
    weights = weights_source(a.weights, params.segment_bytes)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    props = torch.cuda.get_device_properties(K.device)
    device = {"name": props.name, "sms": props.multi_processor_count, "visible": gpu,
              "smi": smi(gpu, "index,uuid,pci.bus_id,clocks.max.sm")}
    lo, hi = (int(x) for x in a.shards.split(":"))
    order = shard_order(lo, hi, a.descending)
    shards = out / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    K.encode(torch.zeros(64, nb, dtype=torch.uint8, device=K.device), 0)
    torch.cuda.synchronize()
    K.raised()
    over = torch.tensor([255] * (nb - 3), dtype=torch.uint8, device=K.device)
    while (s := claim(out, order, a.stop_file, a.descending)) is not None:
        t0 = time.time()
        w = weights(s)
        if len(w) != params.segment_bytes:
            raise SystemExit(f"--weights gave {len(w)} bytes for segment {s}, not {params.segment_bytes}")
        w = w[:n * pb]
        m = torch.zeros(n, nb, dtype=torch.uint8)
        m[:, nb - pb:] = torch.from_numpy(np.frombuffer(bytearray(w), dtype=np.uint8).reshape(n, pb))
        m = m.to(K.device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        c = K.encode(m, s * bps)
        end.record()
        end.synchronize()
        encode_s = start.elapsed_time(end) / 1e3
        flags = K.raised()
        round_trip = bool(torch.equal(K.decode(c, s * bps), m))
        flags |= K.raised()
        near_p = (c[:, :nb - 3] == over).all(dim=1).nonzero().flatten().tolist()
        raw = c.cpu().numpy().tobytes()
        blocks = [raw[v * nb:(v + 1) * nb] for v in range(n)]
        canonical = all(int.from_bytes(blocks[v], "big") < params.p for v in near_p)
        ref = []
        for v in ref_picks(s, n, a.ref_blocks):
            k, t, _ = R.keys_after(prefix, rp, s, v)
            mv = int.from_bytes(w[v * pb:(v + 1) * pb], "big")
            ref.append({"block": v, "equal": R.encode(rp, mv, k, t).to_bytes(nb, "big") == blocks[v]})
        leaves = [leaf_hash(params, salt, s * bps + v, b) for v, b in enumerate(blocks)]
        rec = {"shard": s, "first_block": s * bps, "blocks": n, "whole": n == bps,
               "c_sha512": hashlib.sha512(raw).hexdigest(), "w_sha512": hashlib.sha512(w).hexdigest(),
               "leaf_root": MerkleTree(leaves).root.hex(), "encode_s": encode_s, "blocks_per_s": n / encode_s,
               "wall_s": None, "flags": flags, "round_trip": round_trip, "canonical": canonical, "near_p": len(near_p),
               "reference_encode": ref, "device": device, "sm_clock_after": smi(gpu, "clocks.sm")}
        ok = flags == 0 and round_trip and canonical and all(r["equal"] for r in ref)
        if a.write and ok:
            d = Path(a.write)
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f".seg-{s:04d}.bin.{os.getpid()}"
            tmp.write_bytes(raw)
            os.replace(tmp, d / f"seg-{s:04d}.bin")
        rec["wall_s"] = time.time() - t0
        rec["ok"] = ok
        (shards / f"seg-{s:04d}.json").write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps({k: rec[k] for k in ("shard", "blocks", "c_sha512", "blocks_per_s", "ok")}), flush=True)
        if not ok:
            return 1
        del m, c
    return 0


def gather(a: argparse.Namespace, rcs: list[int]) -> dict[str, Any]:
    recs = sorted((json.loads(p.read_text()) for p in (Path(a.out) / "shards").glob("seg-*.json")), key=lambda r: r["shard"])
    per_gpu: dict[str, dict[str, Any]] = {}
    for r in recs:
        g = per_gpu.setdefault(r["device"]["visible"], {"device": r["device"], "shards": 0, "blocks": 0, "encode_s": 0.0})
        g["shards"] += 1
        g["blocks"] += r["blocks"]
        g["encode_s"] += r["encode_s"]
    for g in per_gpu.values():
        g["blocks_per_s"] = g["blocks"] / g["encode_s"] if g["encode_s"] else None
    whole = [r for r in recs if r["whole"]]
    contiguous = bool(whole) and [r["shard"] for r in whole] == list(range(whole[0]["shard"], whole[-1]["shard"] + 1))
    doc = {"schema": "verity-p2-encode-store/v1", "scheme": "p2-16448/v4", "segments": a.segments, "shards": a.shards,
           "descending": a.descending, "blocks_per_shard": a.blocks or 65_536, "weights": a.weights,
           "salt_sha512": hashlib.sha512(bytes.fromhex(a.salt)).hexdigest(), "stop_file": a.stop_file,
           "stop_at": read_stop(a.stop_file), "worker_rcs": rcs, "ok": all(r["ok"] for r in recs) and not any(rcs),
           "produced": [r["shard"] for r in recs], "per_gpu": per_gpu,
           "store_root": store_root([bytes.fromhex(r["leaf_root"]) for r in whole]).hex()
           if contiguous and len(whole) == a.segments else None,
           "records": [{k: r[k] for k in ("shard", "blocks", "c_sha512", "w_sha512", "leaf_root", "ok")} for r in recs]}
    (Path(a.out) / "manifest.json").write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def visible_gpus() -> list[str]:
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if v:
        return [x for x in v.split(",") if x]
    r = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, check=False)
    return [x.strip() for x in r.stdout.splitlines() if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p2-lib", required=True, help="the directory holding libp2v1.so (v4 keys)")
    ap.add_argument("--salt", required=True, help="the setup's 24-byte salt, hex")
    ap.add_argument("--segments", type=int, required=True)
    ap.add_argument("--shards", required=True, help="A:B, segments A..B-1")
    ap.add_argument("--descending", action="store_true")
    ap.add_argument("--blocks", type=int, default=0, help="each shard's first N blocks (default: the whole segment)")
    ap.add_argument("--weights", default="fixture")
    ap.add_argument("--write", default="")
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--ref-blocks", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if len(bytes.fromhex(a.salt)) != 24:
        raise SystemExit("--salt is the registered scheme's 24-byte salt (setup.draw_salt), 48 hex digits")
    lo, hi = (int(x) for x in a.shards.split(":"))
    if not 0 <= lo < hi <= a.segments:
        raise SystemExit(f"--shards A:B needs 0 <= A < B <= {a.segments}")
    if a.worker:
        return worker(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "claims").write_text("0")
    gpus = visible_gpus()
    argv_w = [sys.executable, __file__, *(argv if argv is not None else sys.argv[1:]), "--worker"]
    with contextlib.ExitStack() as logs:
        procs = [subprocess.Popen(argv_w, env={**os.environ, "CUDA_VISIBLE_DEVICES": g}, stderr=subprocess.STDOUT,
                                  stdout=logs.enter_context(open(out / f"worker-{g}.log", "w")))
                 for g in gpus]
        rcs = [p.wait() for p in procs]
    doc = gather(a, rcs)
    print(json.dumps({k: doc[k] for k in ("ok", "produced", "worker_rcs", "store_root")}))
    for g, d in doc["per_gpu"].items():
        print(json.dumps({"gpu": g, "name": d["device"]["name"], "blocks": d["blocks"], "blocks_per_s": d["blocks_per_s"]}))
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
