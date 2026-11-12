"""The session's pass at each sketch row count in turn over one store on the visible GPUs, with no verifier: what a
pass at 64 rows costs beside one at 16, each over the plain read (`rows.sh` builds the libraries and runs it under a
lease). The registration of each row count is drawn here, as `audit.register` draws the verifier's.

For each of --rows, in the order given, on the same store (`device.GpuBackend`; its bytes are encoded once, at the
first), and at a row count for each of its receipt CTA counts in turn (--ctas-at; `--sketch-ctas` alone unless given;
a count may name an experiment's arms, `ARMS`, as in 264+p0), the gates once, at its first count, and the rest at each:

    gates      the device's gates at that registration (`GpuBackend.gate`), over a gate store on the same GPUs: its
               payload root, encode and decode, a pass's receipts and root against the reference's
               (`receipts.receipter` at the rows, `receipts.tree`), a session accepted, its openings checked by
               `audit.AuditRun`, and one that skips a sixteenth refused at a receipt
    openings   over the store itself, a pass's openings at --openings positions, the first and last block of every
               segment among them: each opened block's receipt in its leaf equal to the reference's, and the leaf
               under the pass's root (`receipts.verify_leaf`)
    whole      with --whole-store, at its first count: one pass over the store itself, into poisoned buffers, against
               a reference that runs none of the kernels' code (`whole_store`): every block's receipt, and the
               pass's root
    calibrate  the plain read, --calibration-passes chained passes queued as a session queues them, the chained pass
               against the plain pass at the same seed over the whole store, the device's signs against the
               reference's and a window's openings (`GpuBackend.calibrate`): tau is the pass over the read. A row
               keeps its slow passes (`slow`): which they were, and of the first 64 each one's time and each
               GPU's own part of it; and every pass's interval, time and each GPU's part (`every`)

Writes OUT/rows.json after each: the gates, the openings and the calibration of each row count and CTA count, in
order. Prints for each the plain read's and the pass's milliseconds and GB/s.

    python rows.py --lib session.so --kv kv.so --out OUT --rows 16 64 16 64 [--gib 496 --gpus 8, `device.py`'s options]
    python rows.py ... --rows 64 --ctas-at 64=96,128,264      which CTA count the 64-row pass is fastest at
    python rows.py ... --rows 64 16 --ctas-at 64=264,264+p0,264+after,264 16=128,128+p0   arms beside the pass as it is
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import device  # noqa: E402
import sketching  # noqa: E402
from device import audit, rb, skr  # noqa: E402


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


def reference(torch, seed: bytes, rows: int):
    """`receipts(blocks, j0)`: the receipts under pass `seed` at `rows` of the store's blocks j0 … held in `blocks` (a
    uint8 tensor of whole blocks, on any device), 64 bytes each, by the reference's definition and none of the kernels'
    code. σ is torch's float32 product of the blocks' bytes as int8 with the signs of `receipts.sign_bits` (exact:
    `sketching.sketch_torch`), and y is `SketchReceipter.receipt`, SHA-512 on the CPU: none of the kernels' tiles,
    packed signs, int8 tensor-core products or device hashing."""
    bits = np.unpackbits(skr.sign_bits(seed, rows), axis=1, bitorder="little")
    receipt, width = rb.SketchReceipter(seed, rows).receipt, 4 * rows

    def receipts(blocks, j0: int) -> bytes:
        e = blocks.view(torch.int8).view(-1, device.BLOCK)
        sigma = sketching.sketch_torch(torch, e, bits).cpu().numpy().astype("<i4").tobytes()
        return b"".join(receipt(j0 + k, sigma[k * width:(k + 1) * width]) for k in range(e.shape[0]))

    return receipts


def whole_store(backend: device.GpuBackend, rows: int, rng: np.random.Generator, step: int = 1 << 15) -> dict:
    """One pass over the backend's store, every buffer it writes poisoned first, against `reference` at `rows`: every
    block's receipt as the devices hold it after the pass, and the pass's root against `receipts.tree` over the
    reference's receipts. The reference reads the store's bytes where they are and shares no code with the pass."""
    s = backend.store
    seed = rb.seed(rng.bytes(rb.CHALLENGE_BYTES), 0, bytes(rb.DIGEST_BYTES))
    s.poison = True
    try:
        root, _ = s.run_pass(seed)
    finally:
        s.poison = False
    t0 = time.perf_counter()
    receipts, want, wrong, differ = reference(s.torch, seed, rows), [], [], 0
    for q, x in enumerate(s.seg_ring):
        gpu = s.gpus[s.seg_gpu[q]]
        with s.on(gpu):
            dev = s.torch.device("cuda", gpu.index)
            for a in range(0, s.n[q], step):
                b = min(a + step, s.n[q])
                ref = receipts(s.data_g[q][a * device.BLOCK:b * device.BLOCK].to(dev), s.base[q] + a)
                want.append(ref)
                if b"".join(device.digests(x["ys"][a:b])) != ref:
                    got = device.digests(x["ys"][a:b])
                    bad = [s.base[q] + a + k for k in range(b - a) if got[k] != ref[64 * k:64 * (k + 1)]]
                    differ += len(bad)
                    wrong += bad[:8 - len(wrong)]
    want = b"".join(want)
    salts = [device.store_salt(backend.salt_key, seed, c) for c in range(s.blocks // device.PER_LEAF)]
    tree = rb.tree(seed, s.plan, [want[64 * j:64 * (j + 1)] for j in range(s.blocks)], salts)
    same = root == tree.commitment.root
    return {"blocks": s.blocks, "receipts": s.blocks - differ, "first_wrong": wrong, "root": same,
            "reference_s": time.perf_counter() - t0, "ok": not differ and same}


ARMS = {"p0": "the side stream at priority 0, the main stream's, in place of −1",
        "after": "where no consumer fits beside the receipt grid, the side stream starts when the grid ends"}
"""An experiment's arms, which a --ctas-at count names after a +: the pass with one thing changed, beside the pass as
it is at the same count on the same store."""


def ctas_at(words: list[str], rows: list[int], default: int) -> dict[int, list[tuple[int, tuple[str, ...]]]]:
    """Each row count's receipt CTA counts, each with its arms (`ARMS`), from --ctas-at's ROWS=N[+ARM...][,N...] words;
    `default` alone, with none, where no word names the row count."""
    at = {r: [(default, ())] for r in rows}
    for word in words:
        r, _, counts = word.partition("=")
        runs = [(n, tuple(arms)) for n, *arms in (c.split("+") for c in counts.split(","))]
        if not r.isdigit() or int(r) not in at or not all(n.isdigit() and set(arms) <= set(ARMS) for n, arms in runs):
            raise SystemExit(f"--ctas-at {word}: ROWS=N[+ARM...][,N...] with ROWS one of --rows ({rows}), N a CTA "
                             f"count and ARM one of {sorted(ARMS)}")
        at[int(r)] = [(int(n), arms) for n, arms in runs]
    return at


def slow(trace: dict) -> dict:
    """A calibration's slow passes (`device.stall_report`: over `device.STALL` times the median): their indices, and
    for those the report details each one's interval, its time by the device's events and each GPU's own part of
    it, in ms. A pass ends when its last GPU does, so the parts say whether the GPUs were slow together."""
    return {"over_median": device.STALL, "passes": trace["slow_passes"],
            "each": [{k: x[k] for k in ("pass", "interval_ms", "gpu_ms", "part_ms")} for x in trace["stalls"]]}


def every(trace: dict, parts: dict) -> dict:
    """Every calibration pass, in order: its interval from the root before it (none for the first), its time by the
    device's events and each GPU's own part of it, in ms. A GPU's median over the run and the passes it was slow in
    are read from these, where `slow` keeps only the passes the node was slow in. With them, where the device kept
    it, the leader's wait for the last GPU and its top (`join_ms`, `top_ms`), the host's side of each pass
    (`device.host_side`, and `free_ms`, its launch's wait for a slot) and the offloader's time on it (`offload_ms`, of
    which `offload_copy_ms` is its own copy, with that copy's CPU time, minor faults and involuntary context
    switches): whether a pass waited on the cards or on the host, and whether the host worked or waited."""
    return {"interval_ms": [None] + [x / 1e6 for x in trace["intervals_ns"]],
            "gpu_ms": [round(x, 3) for x in parts["gpu_ms"]],
            "part_ms": [[round(x, 3) for x in p] for p in parts["part_ms"]],
            **{k: parts[k] for k in ("join_ms", "top_ms", "free_ms", "launch_at_ms", "launch_ms", "launch_cpu_ms",
                                     "root_at_ms", "offload_ms", "offload_copy_ms", "offload_copy_cpu_ms",
                                     "offload_copy_faults", "offload_copy_switched") if k in parts}}


def set_arms(backend: device.GpuBackend, arms: tuple[str, ...]) -> dict:
    """The backend's passes from here on with `arms` and no other: what a row records as its arms."""
    backend.set_side_priority(0 if "p0" in arms else -1)
    backend.set_side_after("after" in arms)
    return {"side_priority": backend.shape["side_priority"], "side_after": backend.shape["side_after"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, nargs="+", default=[rb.ROWS[1], rb.ROWS[0]], choices=rb.ROWS,
                    help="the row counts, in the order they run")
    ap.add_argument("--ctas-at", nargs="+", default=[], metavar="ROWS=N[+ARM...][,N...]",
                    help="the receipt grid's CTA counts at a row count, each in turn (default: --sketch-ctas alone); "
                         "an ARM is " + "; ".join(f"{k}: {v}" for k, v in ARMS.items()))
    ap.add_argument("--openings", type=int, default=256, help="the positions opened on the store each time")
    ap.add_argument("--whole-store", action="store_true",
                    help="at each row count's first CTA count, a pass over the store against the reference at every "
                         "block (about a minute of one core for 64 GiB)")
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
        for i, (ctas, arms) in enumerate(at[rows]):
            row = {"rows": rows, "sketch_ctas": ctas, "arms": list(arms)}
            out["rows"].append(row)
            try:
                backend.set_sketch_ctas(ctas)
                row["settings"] = set_arms(backend, arms)
                if i == 0 and k == 0:
                    row["encode"] = backend.encode(reg)
                elif i == 0:
                    backend.store.set_rows(rows)
                    backend.gates = backend.gate(reg)
                    if not backend.gates["ok"]:
                        raise RuntimeError(f"a gate failed, nothing is timed: {backend.gates}")
                if i == 0 and a.whole_store:
                    row["whole_store"] = whole_store(backend, rows, rng)
                    log(f"rows {rows}: the whole store against the reference {row['whole_store']}")
                    if not row["whole_store"]["ok"]:
                        raise RuntimeError(f"the store's pass is not the reference's: {row['whole_store']}")
                row["openings"] = openings(backend, rows, a.openings, rng)
                if not row["openings"]["ok"]:
                    raise RuntimeError(f"the store's openings are not the reference's: {row['openings']}")
                row["calibration"] = backend.calibrate()
                row["slow"] = slow(backend.calibration_trace)
                try:
                    row["every"] = every(backend.calibration_trace, backend.calibration_parts)
                except (KeyError, IndexError, TypeError, ValueError) as e:  # beside the row's figures, never in their way
                    row["every_failed"] = repr(e)
            except RuntimeError as e:
                row["failed"], ok = str(e), False
            row["gates"] = dict(backend.gates)
            cal = row.get("calibration")
            if i == 0:
                log(f"rows {rows}: gates {json.dumps(backend.gates, default=str)}")
            name = "+".join([str(ctas), *arms])
            log(f"rows {rows} ctas {name}: openings {row.get('openings')} failed {row.get('failed')}")
            if cal:
                pass_ms, gb = cal["pass_ns_median"] / 1e6, out["bytes"] / 1e9
                log(f"rows {rows} ctas {name}: plain read {cal['read_ms']:.3f} ms {gb / cal['read_ms'] * 1e3:.1f} GB/s, "
                    f"pass {pass_ms:.3f} ms {gb / pass_ms * 1e3:.1f} GB/s, pass over read {cal['tau']:.3f} "
                    f"(by the device's events {cal['tau_gpu']:.3f})")
            out["ok"] = ok
            (a.out / "rows.json").write_text(json.dumps(out, indent=1, default=str))
            if not ok:
                break
        if not ok:
            break
    print(json.dumps({"ok": ok, "rows": [(r["rows"], "+".join([str(r["sketch_ctas"]), *r["arms"]]),
                                          r.get("calibration", {}).get("tau")) for r in out["rows"]]}), flush=True)
    # As prover.py ends: tearing down a whole-node store kept the process past its last output.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())
