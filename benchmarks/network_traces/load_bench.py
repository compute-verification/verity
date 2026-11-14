"""load-bench: the time the load's relay takes, by its workers.

It times the relay (`load_relay.relay`) over the first ``--bytes`` of the H200 manifest's plain link, from random bytes
(the commit's cost doesn't depend on them) to no sink, at each ``--workers``, and extrapolates the whole load; with
``load_commit.p2_decode``'s per-block decode for the ``p2`` link.

    python benchmarks/network_traces/load_bench.py --out DIR [--shape h200] [--bytes N] [--workers 1,16]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from verity.protocols.accounting.certifier.verifier.load_check import SHAPES, Shape, manifest

sys.path.append(str(Path(__file__).resolve().parents[2]))  # the root, for benchmarks.network_traces

from benchmarks.network_traces.load_relay import SALT_BYTES, first_values, relay, store_params, verifier_domain


class RandomSource(io.RawIOBase):
    """`n` bytes of the operating system's generator, read as a stream."""

    def __init__(self, n: int) -> None:
        self.left = n

    def read(self, size: int = -1) -> bytes:
        size = self.left if size < 0 else min(size, self.left)
        self.left -= size
        return os.urandom(size)


def bench(shape: Shape, nbytes: int, workers: Sequence[int], witness_dir: Path | None) -> dict[str, Any]:
    import load_commit as LC
    values = first_values(manifest(shape), nbytes)
    rows = sum(v.leaves for v in values)
    total_rows, total_bytes = sum(v.leaves for v in manifest(shape)), sum(v.nbytes for v in manifest(shape))
    dom = verifier_domain("00" * 64)
    out: dict[str, Any] = {"values": len(values), "bytes": sum(v.nbytes for v in values), "rows": rows, "runs": []}
    for n in workers:
        wit = None if witness_dir is None else witness_dir / f"witness-{n}.bin"
        r = relay(values, RandomSource(out["bytes"]), None, dom, workers=n, witness=wit)
        if wit is not None:
            wit.unlink()
        us = r.seconds / rows * 1e6
        out["runs"].append({"workers": n, "seconds": r.seconds, "row_us_wall": us, "mb_s": out["bytes"] / r.seconds / 1e6,
                            "load_s": total_rows * us / 1e6, "witness": wit is not None})
    params = store_params(shape)
    t0, blocks = time.perf_counter(), 64
    LC.p2_decode(os.urandom(blocks * params.block_bytes), os.urandom(24))
    dec = (time.perf_counter() - t0) / blocks
    out["p2"] = {"decode_block_us": dec * 1e6, "blocks": params.blocks, "decode_core_s": params.blocks * dec,
                 "by_workers": {n: params.blocks * dec / n for n in workers}}
    out["shape"] = {"rows": total_rows, "bytes": total_bytes, "witness_bytes": SALT_BYTES * total_rows}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="load_bench.py", description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--shape", choices=sorted(SHAPES), default="h200")
    ap.add_argument("--bytes", type=int, default=1 << 31)
    ap.add_argument("--workers", default="1")
    ap.add_argument("--witness-dir", help="write each run's salts there (default: none)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    host = {"node": platform.node(), "cpus": os.cpu_count(), "affinity": len(os.sched_getaffinity(0)),
            "python": sys.version.split()[0]}
    res = bench(SHAPES[a.shape], a.bytes, [int(x) for x in a.workers.split(",")],
                None if a.witness_dir is None else Path(a.witness_dir)) | {"host": host}
    (out / "load-relay-bench.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    for run in res["runs"]:
        print(f"LOADRELAY workers {run['workers']}: {res['bytes'] / 2**30:.2f} GiB, {res['rows']:,} rows in "
              f"{run['seconds']:.1f} s ({run['mb_s']:.0f} MB/s), whole load {run['load_s']:,.0f} s", flush=True)
    p = res["p2"]
    print(f"LOADRELAY p2 decode {p['decode_block_us']:.0f} µs/block, {p['decode_core_s']:,.0f} core-s; "
          + ", ".join(f"{n} workers {s:,.0f} s" for n, s in p["by_workers"].items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
