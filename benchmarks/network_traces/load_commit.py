"""load-commit: what the network certifier's commit of the weights' load costs on the auditor host, per core, for the demo's
frozen H200 shape (MQA, width 16,384, 52 layers, MLP 65,536, byte vocabulary, context 32; 130.10 GiB of INT8
W8A8 weights, P2-encoded into about 0.93·U of HBM).

Two commitments of the load, each measured exactly as its verifier recomputes it (each checked, on a sample, against the
reference it must equal):

* ``hm96``: the certifier's own (`certifier.transfer_rows`, `one_stage.registered`): every weight row a registered
  value's row under `hm96-sha512/row/v2`, its own 192-byte salt from the operating system, a frame-v3-sha512 tree over
  the rows. A row is what the circuit's registered reads take: an `S8AccumulatorPart_v1{K=256}` unit's weight row, 128
  u16 words (256 int8 weights). Per row: ``x`` (3 compressions after the row prefix's midstate), the mask ``M·y`` (192
  table rows of 64 bytes), ``c`` (2), the hm96 leaf (2), the frame leaf (2) and one tree node (3).
* ``store``: the P2 store's own root (PoUS's verification root, `pous.verifier`): each 2,056-byte block's
  `sha512/row/v2` digest (17 compressions after the prefix's midstate), its leaf (1) and one RFC 6962 node (2).

``raw`` is SHA-512 over 64 MiB buffers (the compression rate OpenSSL reaches on one core), and ``urandom`` the
operating system's generator, which samples the hm96 salts.

    python benchmarks/network_traces/load_commit.py --out DIR [--rows 65536] [--blocks 8192] [--procs 1,4,16]

Every rate is one process on one core; ``--procs`` runs that many processes at once, each the same per-core work, and
reports each count's aggregate, which is what the host's cores reach together. The extrapolation (``shape``) is the
frozen shape's load in core-seconds and, at each process count, in seconds of wall time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from verity.ml.commitments import hm96
from verity.ml.commitments.hm96 import batch as HB
from verity.ml.commitments.merkle import _FRAME, _LEAF, _NODE, _uint
from verity.ml.commitments.rowleaf import sha512_row_v2_prefix
from verity.core.protocols.sampled_proofs.one_stage import registered as G

INST = hm96.SHA512
KEY = INST.default_key
SCHEMA = G.SCHEMA_V2
#: the load's row: 128 u16 words, one S8AccumulatorPart_v1{K=256} unit's weight row
ROW_WORDS = 128
ROW_BYTES = 2 * ROW_WORDS
#: P2 at its registered point (`pous.schemes.p2`): 2,056-byte blocks of 2,054 payload bytes, 2^16 blocks a segment
P2_BLOCK, P2_PAYLOAD, P2_SEGMENT = 2056, 2054, 1 << 16
PROGRAM = "00" * 64

# the frozen shape: per layer, (rows, cols) of each INT8 weight matrix, K = cols
D, LAYERS, MLP, HEAD, VOCAB, CTX = 16384, 52, 65536, 64, 256, 32
MATRICES = {"wq": (D, D), "wk": (HEAD, D), "wv": (HEAD, D), "wo": (D, D), "up": (MLP, D), "down": (D, MLP)}


def shape() -> dict:
    """The load at the frozen shape: every INT8 matrix in rows of 256 weights (one part's K), each matrix its own value
    (none padded: each is a power of two rows) or all of them one value (padded to 2^30), the small tensors beside them,
    and the P2 store's blocks."""
    per_layer = {k: r * c // ROW_BYTES for k, (r, c) in MATRICES.items()}
    rows = LAYERS * sum(per_layer.values())
    weights = rows * ROW_BYTES
    # channel scales (f32), norms (bf16, two a layer and the final one), token and position embeddings (bf16), LM head
    small = 4 * LAYERS * sum(r for r, _c in MATRICES.values()) + 2 * D * (2 * LAYERS + 1) + 2 * D * (VOCAB + CTX) \
        + VOCAB * D + 4 * VOCAB
    total = weights + small
    blocks = -(-total // P2_PAYLOAD)
    pow2 = all(n & (n - 1) == 0 for n in per_layer.values())
    return {"rows_per_layer": per_layer, "weight_rows": rows, "weight_bytes": weights, "weight_gib": weights / 2**30,
            "small_bytes": small, "load_bytes": total, "p2_blocks": blocks, "store_bytes": blocks * P2_BLOCK,
            "store_gib": blocks * P2_BLOCK / 2**30, "values_per_matrix_unpadded": pow2,
            "one_value_padded_rows": 1 << (rows - 1).bit_length()}


# --- the certifier's hm96 commitment, streamed ------------------------------------------------------------------------

def hm96_leaves(dom, rows: np.ndarray, salts: bytes, start: int = 0) -> list[bytes]:
    """The frame-v3 leaves of `rows` (n × 256 bytes) at ranks ``start ..``, each under its salt: `registered.leaf` of
    `Registered.commit`'s ``b ‖ c``, batched (the mask in numpy, every hash from a midstate)."""
    n = rows.shape[0]
    xh = hashlib.sha512(sha512_row_v2_prefix(8 * ROW_BYTES))
    flat = rows.tobytes()
    xs = bytearray()
    for i in range(n):
        h = xh.copy()
        h.update(flat[ROW_BYTES * i:ROW_BYTES * (i + 1)])
        xs += h.digest()
    hs = HB.tree_leaves(INST, KEY, bytes(xs), salts)
    did, schema = dom.domain_id, SCHEMA.encode()
    head = _FRAME + len(_LEAF).to_bytes(4, "big") + _LEAF + len(did).to_bytes(8, "big") + did
    tail = len(schema).to_bytes(8, "big") + schema + (64).to_bytes(8, "big")
    out = []
    for i, h in enumerate(hs):
        r = _uint(start + i)
        rr = len(r).to_bytes(8, "big") + r
        out.append(hashlib.sha512(head + rr + rr + tail + h).digest())
    return out


def frame_nodes(dom, leaves: list[bytes]) -> bytes:
    """The frame-v3 root over a power of two of leaves, level by level."""
    did = dom.domain_id
    head = _FRAME + len(_NODE).to_bytes(4, "big") + _NODE + len(did).to_bytes(8, "big") + did
    level, depth = leaves, 0
    while len(level) > 1:
        lv = _uint(depth)
        pre = head + len(lv).to_bytes(8, "big") + lv
        nxt = []
        for j in range(len(level) // 2):
            ix = _uint(j)
            nxt.append(hashlib.sha512(pre + len(ix).to_bytes(8, "big") + ix + b"\0" * 7 + b"\x40" + level[2 * j]
                                      + b"\0" * 7 + b"\x40" + level[2 * j + 1]).digest())
        level, depth = nxt, depth + 1
    return level[0]


# --- the P2 store's root ----------------------------------------------------------------------------------------------

def store_leaves(blocks: bytes, salt_tag: bytes, start: int = 0) -> list[bytes]:
    """PoUS's leaves (`pous.verifier.leaf_hash`) of whole 2,056-byte blocks from ``start``, in segment 0's tag."""
    rh = hashlib.sha512(sha512_row_v2_prefix(8 * P2_BLOCK))
    out = []
    for i in range(len(blocks) // P2_BLOCK):
        h = rh.copy()
        h.update(blocks[P2_BLOCK * i:P2_BLOCK * (i + 1)])
        v = (start + i) % P2_SEGMENT
        out.append(hashlib.sha512(b"\x00" + salt_tag + v.to_bytes(2, "little") + h.digest()).digest())
    return out


def rfc6962(leaves: list[bytes]) -> bytes:
    level = leaves
    while len(level) > 1:
        level = [hashlib.sha512(b"\x01" + level[j] + level[j + 1]).digest() if j + 1 < len(level) else level[j]
                 for j in range(0, len(level), 2)]
    return level[0]


# --- exactness ----------------------------------------------------------------------------------------------------------

def exact() -> dict:
    """The streamed commitments against their references on a sample: the certifier's root against
    `Registered.commit`'s, the store's leaves and root against `pous.verifier`'s."""
    rng = np.random.default_rng(1)
    n = 64
    rows = rng.integers(0, 256, (n, ROW_BYTES), dtype=np.uint8)
    salts = os.urandom(n * INST.salt_bytes)
    words = [list(np.frombuffer(r.tobytes(), "<u2")) for r in rows]
    ref = G.Registered.commit(PROGRAM, "w", words, [salts[192 * i:192 * (i + 1)] for i in range(n)], 16, SCHEMA)
    dom = G.domain(PROGRAM, "w", ROW_WORDS, n)
    ours = frame_nodes(dom, hm96_leaves(dom, rows, salts))
    from verity.core.protocols.pous import verifier as PV
    from verity.core.protocols.pous.labels import salt_value, tag
    params = SimpleNamespace(blocks_per_segment=P2_SEGMENT, block_bytes=P2_BLOCK)
    salt = os.urandom(24)
    blocks = os.urandom(5 * P2_BLOCK)
    st = tag(salt_value(salt), 0).to_bytes(32, "little")
    mine = store_leaves(blocks, st)
    theirs = [PV.leaf_hash(params, salt, i, blocks[P2_BLOCK * i:P2_BLOCK * (i + 1)]) for i in range(5)]
    return {"hm96_root_equal": ours == ref.root, "store_leaves_equal": mine == theirs,
            "store_root_equal": rfc6962(mine) == PV.MerkleTree(theirs).root}


# --- the timings, one core each -----------------------------------------------------------------------------------------

def _time(fn, *a) -> float:
    t = time.perf_counter()
    fn(*a)
    return time.perf_counter() - t


def p2_decode(blocks: bytes, salt: bytes) -> None:
    """P2 v4's decode of whole blocks of segment 0 (`pous.schemes.p2.decode_segment`'s arithmetic and keys, without its
    codeword check, which random blocks fail): what a certifier that commits the decoded weights pays first."""
    from verity.core.protocols.pous.schemes import p2 as P2
    params = P2.P2Params(*P2.POINT)
    f, keys = P2._field(params), P2.segment_keys(params, salt, 0)
    for j in range(len(blocks) // P2_BLOCK):
        k, t = keys(j)
        c = int.from_bytes(blocks[P2_BLOCK * j:P2_BLOCK * (j + 1)], "big") % f.p
        (f.sq(f.sigma(f.sq(c), k)) - t) % f.p


def per_core(rows: int, blocks: int) -> dict:
    """One process's rates: µs a 256-byte row of the certifier's commitment, its parts, µs a 2,056-byte P2 block of the
    store's root and of its decode, raw SHA-512 and the generator."""
    rng = np.random.default_rng(os.getpid())
    data = rng.integers(0, 256, (rows, ROW_BYTES), dtype=np.uint8)
    dom = G.domain(PROGRAM, "w", ROW_WORDS, rows)
    t_salt = _time(os.urandom, rows * INST.salt_bytes)
    salts = os.urandom(rows * INST.salt_bytes)
    t_mask = _time(HB.masks, INST, KEY, salts)
    t0 = time.perf_counter()
    leaves = hm96_leaves(dom, data, salts)
    t_leaves = time.perf_counter() - t0
    t_nodes = _time(frame_nodes, dom, leaves)
    store = os.urandom(blocks * P2_BLOCK)
    st = os.urandom(32)
    t0 = time.perf_counter()
    sl = store_leaves(store, st)
    t_sleaves = time.perf_counter() - t0
    t_snodes = _time(rfc6962, sl)
    dec = max(1, blocks // 4)
    t_dec = _time(p2_decode, store[:dec * P2_BLOCK], os.urandom(24))
    buf = os.urandom(64 << 20)
    t_raw = min(_time(lambda: hashlib.sha512(buf).digest()) for _ in range(3))
    us = 1e6
    return {"rows": rows, "blocks": blocks,
            "hm96_row_us": (t_salt + t_leaves + t_nodes) / rows * us,
            "hm96_parts_us": {"salt": t_salt / rows * us, "mask": t_mask / rows * us,
                              "leaves (x, mask, c, hm96 leaf, frame leaf)": t_leaves / rows * us,
                              "nodes": t_nodes / rows * us},
            "store_block_us": (t_sleaves + t_snodes) / blocks * us, "p2_decode_block_us": t_dec / dec * us,
            "raw_sha512_mb_s": len(buf) / t_raw / 1e6, "raw_compression_ns": t_raw / (len(buf) / 128) * 1e9,
            "urandom_mb_s": rows * INST.salt_bytes / t_salt / 1e6}


def _worker(args) -> dict:
    return per_core(*args)


def scaled(procs: int, rows: int, blocks: int) -> dict:
    """`procs` processes at once, each `per_core`'s work: each one's rates and the aggregate."""
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(procs) as pool:
        got = pool.map(_worker, [(rows, blocks)] * procs)
    wall = time.perf_counter() - t0
    mean = lambda k: sum(g[k] for g in got) / procs  # noqa: E731
    return {"procs": procs, "wall_s": wall, "hm96_row_us": mean("hm96_row_us"), "store_block_us": mean("store_block_us"),
            "raw_sha512_mb_s": mean("raw_sha512_mb_s"),
            "hm96_mb_s_total": procs * ROW_BYTES / mean("hm96_row_us"),
            "store_mb_s_total": procs * P2_BLOCK / mean("store_block_us")}


def extrapolate(sh: dict, one: dict, runs: list[dict]) -> dict:
    """The frozen shape's load in core-seconds, measured (this Python on hashlib and numpy) and native (12 compressions a
    row at the measured raw rate, the mask as 384 nibble-table lookups of 64 bytes at 1 ns each, the salts at the
    measured generator rate), and each process count's wall seconds."""
    rows, blocks = sh["weight_rows"], sh["p2_blocks"]
    c = one["raw_compression_ns"] / 1e3
    native_row = 12 * c + 0.384 + INST.salt_bytes / one["urandom_mb_s"]
    native_block = 20 * c
    out = {"hm96_core_s": rows * one["hm96_row_us"] / 1e6, "store_core_s": blocks * one["store_block_us"] / 1e6,
           "p2_decode_core_s": blocks * one["p2_decode_block_us"] / 1e6,
           "hm96_native_core_s_est": rows * native_row / 1e6, "store_native_core_s_est": blocks * native_block / 1e6,
           "hm96_one_value_padded_core_s": (rows * one["hm96_row_us"]
                                            + (sh["one_value_padded_rows"] - rows) * one["hm96_parts_us"]["nodes"] * 2)
           / 1e6,
           "salt_witness_gb": rows * INST.salt_bytes / 1e9, "by_procs": {}}
    for r in runs:
        out["by_procs"][r["procs"]] = {"hm96_s": rows * r["hm96_row_us"] / 1e6 / r["procs"],
                                       "store_s": blocks * r["store_block_us"] / 1e6 / r["procs"],
                                       "hm96_gb_s": r["hm96_mb_s_total"] / 1e3, "store_gb_s": r["store_mb_s_total"] / 1e3}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="load_commit.py", description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=1 << 16, help="rows a process commits (a power of two)")
    ap.add_argument("--blocks", type=int, default=1 << 13, help="P2 blocks a process roots")
    ap.add_argument("--procs", default="1", help="process counts, comma-separated")
    a = ap.parse_args(argv)
    if a.rows & (a.rows - 1):
        raise SystemExit("--rows is a power of two")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ex = exact()
    if not all(ex.values()):
        raise SystemExit(f"a streamed commitment is not its reference's: {ex}")
    sh = shape()
    one = per_core(a.rows, a.blocks)
    runs = [scaled(int(p), a.rows, a.blocks) for p in a.procs.split(",")]
    res = {"host": {"node": platform.node(), "cpus": os.cpu_count(), "affinity": len(os.sched_getaffinity(0)),
                    "cpu": next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                                 if ln.startswith("model name")), None), "python": sys.version.split()[0],
                    "openssl": __import__("ssl").OPENSSL_VERSION},
           "exact": ex, "shape": sh, "per_core": one, "scaled": runs, "extrapolated": extrapolate(sh, one, runs)}
    (out / "load_commit.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    e = res["extrapolated"]
    print(f"LOAD {sh['weight_gib']:.2f} GiB weights, {sh['weight_rows']:,} rows, store {sh['store_gib']:.2f} GiB "
          f"({sh['p2_blocks']:,} blocks); per core: hm96 {one['hm96_row_us']:.2f} µs/row "
          f"({ROW_BYTES / one['hm96_row_us']:.1f} MB/s), store {one['store_block_us']:.2f} µs/block "
          f"({P2_BLOCK / one['store_block_us']:.0f} MB/s), raw SHA-512 {one['raw_sha512_mb_s']:.0f} MB/s; "
          f"core-s: hm96 {e['hm96_core_s']:,.0f} (native est. {e['hm96_native_core_s_est']:,.0f}), "
          f"store {e['store_core_s']:,.0f} (native est. {e['store_native_core_s_est']:,.0f}), P2 decode "
          f"{one['p2_decode_block_us']:.1f} µs/block, {e['p2_decode_core_s']:,.0f}", flush=True)
    for p, r in e["by_procs"].items():
        print(f"LOAD procs {p}: hm96 {r['hm96_s']:,.0f} s ({r['hm96_gb_s']:.2f} GB/s), store {r['store_s']:,.0f} s "
              f"({r['store_gb_s']:.2f} GB/s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
