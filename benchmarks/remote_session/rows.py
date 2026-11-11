"""The session's pass at each sketch row count in turn over one store on the visible GPUs, with no verifier: what a
pass at 64 rows costs beside one at 16, each over the plain read (`rows.sh` builds the libraries and runs it under a
lease). The registration of each row count is drawn here, as `audit.register` draws the verifier's.

For each of --rows, in the order given, on the same store (`device.GpuBackend`; its bytes are encoded once, at the
first):

    gates      the device's gates at that registration (`GpuBackend.gate`), over a gate store on the same GPUs: its
               payload root, encode and decode, a pass's receipts and root against the reference's
               (`receipts.receipter` at the rows, `receipts.tree`), a session accepted, its openings checked by
               `audit.AuditRun`, and one that skips a sixteenth refused at a receipt
    openings   over the store itself, a pass's openings at --openings positions, the first and last block of every
               segment among them: each opened block's receipt in its leaf equal to the reference's, and the leaf
               under the pass's root (`receipts.verify_leaf`)
    calibrate  the plain read, --calibration-passes chained passes queued as a session queues them, the chained pass
               against the plain pass at the same seed over the whole store, the device's signs against the
               reference's and a window's openings (`GpuBackend.calibrate`): tau is the pass over the read

Writes OUT/rows.json after each row count: the gates, the openings and the calibration of each, in order.

    python rows.py --lib session.so --kv kv.so --out OUT --rows 16 64 16 64 [--gib 496 --gpus 8, `device.py`'s options]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import device  # noqa: E402
from device import audit, rb  # noqa: E402


def openings(backend: device.GpuBackend, rows: int, draws: int, rng: np.random.Generator) -> dict:
    """One pass over the backend's store and its openings at `draws` positions against the reference at `rows`."""
    s = backend.store
    seed = rb.seed(rng.bytes(rb.CHALLENGE_BYTES), 0, bytes(rb.DIGEST_BYTES))
    root, handle = s.run_pass(seed)
    edges = {j for q in range(len(s.n)) for j in (s.base[q], s.base[q] + s.n[q] - 1)}
    ps = sorted(edges | set(map(int, rng.integers(0, s.blocks, draws))))
    r = rb.receipter(rb.SKETCH_SHA512, seed, rows)
    receipts = paths = 0
    for p, (blk, leaf, salt, path, *_payload) in zip(ps, s.open(handle, ps)):
        receipts += rb.leaf_receipt(leaf, p, s.blocks) == r(p, blk)
        paths += rb.verify_leaf(seed, s.plan, root, p, leaf, salt, path)
    return {"opened": len(ps), "receipts": receipts, "paths": paths,
            "ok": receipts == len(ps) and paths == len(ps)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, nargs="+", default=[rb.ROWS[1], rb.ROWS[0]], choices=rb.ROWS,
                    help="the row counts, in the order they run")
    ap.add_argument("--openings", type=int, default=256, help="the positions opened on the store at each row count")
    ap.add_argument("--node", default="rows")
    ap.add_argument("--seed", type=int, default=20261111)
    device.add_arguments(ap)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sys.setswitchinterval(2e-4)

    def log(m: str) -> None:
        print(m, flush=True)

    backend = device.from_arguments(a, log=log)
    rng = np.random.default_rng(a.seed)
    source, node = rng.bytes(32), a.node.encode()
    backend.set_payload_key(rng.bytes(32))
    root = backend.payload_root(node)
    out = {"schema": "verity/remote-session/rows/v0", "blocks": backend.blocks, "bytes": backend.blocks * device.BLOCK,
           "rows": []}
    ok = True
    for k, rows in enumerate(a.rows):
        reg = audit.register(source, node, backend.blocks, root, rows=rows, nonce=rng.bytes(audit.NONCE_BYTES))
        row = {"rows": rows}
        out["rows"].append(row)
        try:
            if k == 0:
                row["encode"] = backend.encode(reg)
            else:
                backend.store.set_rows(rows)
                backend.gates = backend.gate(reg)
                if not backend.gates["ok"]:
                    raise RuntimeError(f"a gate failed, nothing is timed: {backend.gates}")
            row["openings"] = openings(backend, rows, a.openings, rng)
            if not row["openings"]["ok"]:
                raise RuntimeError(f"the store's openings are not the reference's: {row['openings']}")
            row["calibration"] = backend.calibrate()
        except RuntimeError as e:
            row["failed"], ok = str(e), False
        row["gates"] = backend.gates
        cal = row.get("calibration", {})
        log(f"rows {rows}: gates {json.dumps(backend.gates, default=str)}")
        log(f"rows {rows}: openings {row.get('openings')} tau {cal.get('tau')} tau_gpu {cal.get('tau_gpu')} "
            f"read_ms {cal.get('read_ms')} pass_gpu_ms {cal.get('pass_gpu_ms_median')} failed {row.get('failed')}")
        out["ok"] = ok
        (a.out / "rows.json").write_text(json.dumps(out, indent=1, default=str))
        if not ok:
            break
    print(json.dumps({"ok": ok, "rows": [(r["rows"], r.get("calibration", {}).get("tau")) for r in out["rows"]]}),
          flush=True)
    # As prover.py ends: tearing down a whole-node store kept the process past its last output.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())
