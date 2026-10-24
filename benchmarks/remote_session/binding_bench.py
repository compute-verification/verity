"""The store binding's payload root at a store's size, timed through each verifier: a synthetic pinned model whose
arena is `--weights` blocks (random digests) then zeros to `--arena-blocks`, `--engines` arenas of it, then `--host`
host blocks, under one payload key. The Lean verifier of record (`remote-audit-verifier`'s `bind`, then `hold` of a
hello carrying the root, its subtrees on `--workers` threads) and `store.expected_root` (in `--workers` processes, as
`Binding.hold` runs it) each compute the root. The Lean's `hold` must bind the hello whose root is the Python's, with
that root as its expected root, and refuse one whose root is a byte off (a second timing of the Lean's root). Writes
OUT/binding_bench.json after each phase; exits 1 unless both hold.

    python binding_bench.py --out OUT [--engines 8 --arena-blocks 16250000 --weights 4000000 --host 0 --workers N]

The defaults are a whole node's 130M blocks: eight arenas of an 8B BF16 model's 4M weights blocks.

    research run --on vy-nebius-cpu-1 --pool research --as LANE --project verity --source . --cwd clone -- bash -lc \
      'uv sync -q --all-packages --extra torch-cpu && uv run python benchmarks/remote_session/binding_bench.py --out "$OUT"'
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from verity.core.protocols.remote_bandwidth import store  # noqa: E402

import lean_verifier as lv  # noqa: E402

ROLE = "SYNTHETIC_BF16"
NODE = b"binding-bench-node"


def pin_data() -> bytes:
    files = {n: hashlib.sha512(n.encode()).hexdigest() for n in ("config.json", "model.safetensors")}
    return json.dumps({"name": ROLE, "checkpoint": files, "checkpoint_sha512": store.checkpoint_sha512(files)}).encode()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--engines", type=int, default=8)
    ap.add_argument("--arena-blocks", type=int, default=16_250_000)
    ap.add_argument("--weights", type=int, default=4_000_000)
    ap.add_argument("--host", type=int, default=0)
    ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--no-python", action="store_true", help="the Lean's root alone, with no reference to hold it to")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    blocks = a.engines * a.arena_blocks + a.host
    out: dict = {"engines": a.engines, "arena_blocks": a.arena_blocks, "weights_blocks": a.weights, "host_blocks": a.host,
                 "blocks": blocks, "workers": a.workers, "chunk_log2": store.CHUNK_LOG2, "cpus": len(os.sched_getaffinity(0))}

    def write() -> None:
        (a.out / "binding_bench.json").write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out), flush=True)

    key = hashlib.sha256(b"binding-bench key").digest()
    digests = hashlib.shake_256(b"binding-bench weights").digest(64 * a.weights)
    lean = lv.Lean(lv.build(), lambda _msg: None)
    t0 = time.monotonic()
    bound = lean.call({"op": "bind", "bytes": len(digests), "pin": pin_data().hex(), "path": "pin.json", "role": ROLE,
                       "engines": a.engines, "arena_blocks": a.arena_blocks, "workers": a.workers}, digests)
    out["lean_bind_s"] = round(time.monotonic() - t0, 3)
    out["lean_bind"] = {k: v for k, v in bound.items() if k != "pin"}
    write()

    def hold(root: bytes) -> tuple[dict, float]:
        t = time.monotonic()
        got = lean.call({"op": "hold", "node": NODE.hex(), "blocks": blocks, "payload_root": root.hex(), "key": key.hex(),
                         "layout": None})
        return got, round(time.monotonic() - t, 3)

    if a.no_python:
        got, out["lean_root_s"] = hold(bytes(64))
        out["lean_root"] = got.get("expected_root")
        write()
        lean.close()
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "weights.sha512rows"
        path.write_bytes(digests)
        t0 = time.monotonic()
        s = store.Store(store.Arena("", (), a.weights), a.engines, a.arena_blocks, a.host)
        want = store.expected_root(s, NODE, key, path, a.workers)
        out["python_root_s"], out["python_root"] = round(time.monotonic() - t0, 3), want.hex()
    write()
    got, out["lean_root_s"] = hold(want)
    out["lean_root"] = got.get("expected_root")
    out["lean_bound"] = "regions" in got and got["expected_root"] == want.hex()
    write()
    off = bytes([want[0] ^ 1]) + want[1:]
    got, out["lean_refused_root_s"] = hold(off)
    out["lean_refused"] = "refused" in got and got["expected_root"] == want.hex()
    write()
    lean.close()
    return 0 if out["lean_bound"] and out["lean_refused"] else 1


if __name__ == "__main__":
    sys.exit(main())
