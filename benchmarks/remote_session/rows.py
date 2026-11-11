"""The session's pass at each sketch row count in turn over one store on the visible GPUs, with no verifier: what a
pass at 64 rows costs beside one at 16, each over the plain read (`rows.sh` builds the libraries and runs it under a
lease). The registration of each row count is drawn here, as `audit.register` draws the verifier's.

For each of --rows, in the order given, on the same store (`device.GpuBackend`; its bytes are encoded once, at the
first), and at a row count for each of its receipt CTA counts in turn (--ctas-at; `--sketch-ctas` alone unless given),
the gates once and the rest at each count:

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

Writes OUT/rows.json after each: the gates, the openings and the calibration of each row count and CTA count, in
order. Prints for each the plain read's and the pass's milliseconds and GB/s.

    python rows.py --lib session.so --kv kv.so --out OUT --rows 16 64 16 64 [--gib 496 --gpus 8, `device.py`'s options]
    python rows.py ... --rows 64 --ctas-at 64=96,128,264      which CTA count the 64-row pass is fastest at
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


def ctas_at(words: list[str], rows: list[int], default: int) -> dict[int, list[int]]:
    """Each row count's receipt CTA counts from --ctas-at's ROWS=N[,N...] words, `default` alone where none names it."""
    at = {r: [default] for r in rows}
    for word in words:
        r, _, counts = word.partition("=")
        if not r.isdigit() or int(r) not in at or not counts or not all(c.isdigit() for c in counts.split(",")):
            raise SystemExit(f"--ctas-at {word}: ROWS=N[,N...] with ROWS one of --rows ({rows}) and N a CTA count")
        at[int(r)] = [int(c) for c in counts.split(",")]
    return at


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, nargs="+", default=[rb.ROWS[1], rb.ROWS[0]], choices=rb.ROWS,
                    help="the row counts, in the order they run")
    ap.add_argument("--ctas-at", nargs="+", default=[], metavar="ROWS=N[,N...]",
                    help="the receipt grid's CTA counts at a row count, each in turn (default: --sketch-ctas alone)")
    ap.add_argument("--openings", type=int, default=256, help="the positions opened on the store each time")
    ap.add_argument("--node", default="rows")
    ap.add_argument("--seed", type=int, default=20261111)
    device.add_arguments(ap)
    a = ap.parse_args()
    at = ctas_at(a.ctas_at, a.rows, a.sketch_ctas)
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
        for i, ctas in enumerate(at[rows]):
            row = {"rows": rows, "sketch_ctas": ctas}
            out["rows"].append(row)
            try:
                if i == 0 and k == 0:
                    row["encode"] = backend.encode(reg)
                elif i == 0:
                    backend.store.set_rows(rows)
                    backend.gates = backend.gate(reg)
                    if not backend.gates["ok"]:
                        raise RuntimeError(f"a gate failed, nothing is timed: {backend.gates}")
                backend.set_sketch_ctas(ctas)
                row["openings"] = openings(backend, rows, a.openings, rng)
                if not row["openings"]["ok"]:
                    raise RuntimeError(f"the store's openings are not the reference's: {row['openings']}")
                row["calibration"] = backend.calibrate()
            except RuntimeError as e:
                row["failed"], ok = str(e), False
            row["gates"] = dict(backend.gates)
            cal = row.get("calibration")
            if i == 0:
                log(f"rows {rows}: gates {json.dumps(backend.gates, default=str)}")
            log(f"rows {rows} ctas {ctas}: openings {row.get('openings')} failed {row.get('failed')}")
            if cal:
                pass_ms, gb = cal["pass_ns_median"] / 1e6, out["bytes"] / 1e9
                log(f"rows {rows} ctas {ctas}: plain read {cal['read_ms']:.3f} ms {gb / cal['read_ms'] * 1e3:.1f} GB/s, "
                    f"pass {pass_ms:.3f} ms {gb / pass_ms * 1e3:.1f} GB/s, pass over read {cal['tau']:.3f} "
                    f"(by the device's events {cal['tau_gpu']:.3f})")
            out["ok"] = ok
            (a.out / "rows.json").write_text(json.dumps(out, indent=1, default=str))
            if not ok:
                break
        if not ok:
            break
    print(json.dumps({"ok": ok, "rows": [(r["rows"], r["sketch_ctas"], r.get("calibration", {}).get("tau"))
                                         for r in out["rows"]]}), flush=True)
    # As prover.py ends: tearing down a whole-node store kept the process past its last output.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())
