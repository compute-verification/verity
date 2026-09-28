"""-h1's hashing on sm_120 (node 2): device gates, cost per byte, and decode's dependent chain at m = 32, n = k = 8,192.

  python3 h1_bench.py --cubin hash_sm120a.cubin --standin h1_standin.cubin --vectors h1_vectors.json --out h1_bench.json

Standard library only; the driver API through `run.py`'s `Dev`. The GPU is the one the lease named (`GPU_LEASE_UUID`, checked
against `cuDeviceGetUuid`); NVML reads that UUID's clocks.

**Gates, all before any timing** (a difference exits 4 with nothing timed). Each kernel form against `h1_vectors.json`, which
`h1_vectors.py` writes from `verity.ml.commitments`' frame-b3 and Pearl-C's h1 digests:
- the row leaves of `hash_rows` (as pushed) and `hash_rows_w` (tuned), at every chunk count 1 … 258: rows of 1,024c − 64
  bytes (exactly c chunks with the one-block header), 1,024c − 27 (37 bytes past a chunk boundary) and 1,024c, rows 0 and 1;
- A's tree at m = 32 (FP32 rows at k = 8,192, as `-h1` commits them, and BF16), `hash_nodes` level by level and
  `hash_tree_small` in one launch;
- the tile hashing at m = 32, n = 8,192 (decode's partial 32 x 64 tiles), as pushed (`hash_msg`, `hash_leaf`) and tuned
  (`hash_msg_v`, `hash_leaf_p`), with the tile tree both ways (level keys, then `hash_nodes` or `hash_tree_small`).

**Transcripts** (`--transcripts DIR`): after timing, each decode arm on -h1's format is replayed once, and its last call's A
rows, A's tree, C~, U, digests, tile leaves and tile tree go into DIR/<arm>.bin for `verify_h1.py`, the reference verifier.

**Cost per byte**: each form at prefill size (m = 8,192), a read-only L2 flush before each rep, bytes hashed over the median
time, per SM per SM clock too.

**Latency**: dependent compressions on one thread (`b3c`) and on four lanes (`b3c4`), SM cycles each.

**Decode, dependent chain** (`ARMS`): per call, A = f(y of the call before) (`derive_rows`, or `derive_rows_hash`, which hashes
A as it writes it), A's commitment in one of its forms (row leaves and tree), the
stand-in GEMM (`h1_standin.cu`: the weights' DRAM traffic, not a GEMM; four weight sets cycled, past twice the 128 MB L2), and
the post-GEMM hashing (message digests, tile leaves, tile tree). CALLS calls are captured into one CUDA graph and replayed;
per call is a replay over CALLS. The arms: `plain` (derive and the stand-in alone), `h1` (the kernels as pushed), `h1-tuned`
(serial), `h1-tuned-side` (the post-GEMM hashing on a second stream, two buffer slots, joined before a slot is reused).
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as R  # noqa: E402

u, i32, vp, sz, u64, P = C.c_uint, C.c_int, C.c_void_p, C.c_size_t, C.c_uint64, C.POINTER
EXTRA = {
    "cuStreamCreate": [P(vp), u], "cuStreamWaitEvent": [vp, vp, u], "cuStreamSynchronize": [vp],
    "cuStreamBeginCapture_v2": [vp, i32], "cuStreamEndCapture": [vp, P(vp)], "cuGraphInstantiateWithFlags": [P(vp), vp, C.c_ulonglong],
    "cuGraphLaunch": [vp, vp], "cuGraphExecDestroy": [vp], "cuGraphDestroy": [vp], "cuDeviceGetUuid_v2": [C.c_char_p, i32],
    "cuCtxGetStreamPriorityRange": [P(i32), P(i32)], "cuStreamCreateWithPriority": [P(vp), u, i32],
    "cuGraphGetNodes": [vp, vp, P(sz)], "cuGraphNodeGetType": [vp, P(i32)], "cuGraphKernelNodeGetAttribute": [vp, i32, vp],
}
USE_NODE_PRIORITY, ATTR_PRIORITY = 8, 8                     # CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY, CU_LAUNCH_ATTRIBUTE_PRIORITY
M, N, K = 32, 8192, 8192
CALLS, WSETS = 32, 4
MAX_DYN, SMEM_OPTIN = 8, 97
ROW_F32, ROW_BF16 = 4 * K, 2 * K
ROW, TILE = "verity/pouw/row/v1", "verity/pouw/tile/v1"


def shake(tag: str, n: int) -> bytes:
    return hashlib.shake_128(tag.encode()).digest(n)


class Sensors(R.Nvml):
    """run.py's NVML reader on the leased GPU's handle (by UUID; NVML's index 0 is another GPU on a shared node)."""

    def __init__(self, uuid: str | None):
        super().__init__()
        if self.lib and uuid:
            self.lib.nvmlDeviceGetHandleByUUID.argtypes = [C.c_char_p, P(vp)]
            h = vp()
            if self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), C.byref(h)) == 0:
                self.h = h
            else:
                self.lib = None


class Bench(R.Dev):
    def __init__(self, cubin: bytes, standin: bytes):
        super().__init__(cubin)
        R.bind(self.cu, EXTRA)
        self.mod2 = vp()
        self.ck("cuModuleLoadData", C.byref(self.mod2), standin)
        self.fns2: dict[str, vp] = {}
        self.s1, self.s2 = vp(), vp()
        self.ck("cuStreamCreate", C.byref(self.s1), 1)
        self.ck("cuStreamCreate", C.byref(self.s2), 1)
        least, greatest = i32(), i32()                       # the "priorities" arms: the chain's stream first, the side stream last
        self.ck("cuCtxGetStreamPriorityRange", C.byref(least), C.byref(greatest))
        self.h1, self.h2 = vp(), vp()
        self.ck("cuStreamCreateWithPriority", C.byref(self.h1), 1, greatest.value)
        self.ck("cuStreamCreateWithPriority", C.byref(self.h2), 1, least.value)
        self.h3 = vp()                                       # P2's A tree: beside the GEMM, never behind the tile hashing
        self.ck("cuStreamCreateWithPriority", C.byref(self.h3), 1, greatest.value)
        self.priority_range = (least.value, greatest.value)
        self.args: list = []
        self.dyn: dict[str, int] = {}

    def uuid(self) -> str:
        b = C.create_string_buffer(16)
        self.ck("cuDeviceGetUuid_v2", b, self.dev)
        h = b.raw.hex()
        return f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"

    def fn_any(self, name: str) -> vp:
        try:
            return self.fn(name)
        except RuntimeError:
            if name not in self.fns2:
                f = vp()
                self.ck("cuModuleGetFunction", C.byref(f), self.mod2, name.encode())
                self.fns2[name] = f
            return self.fns2[name]

    def go(self, stream, name: str, grid: int, block: int, p=(), i=(), smem: int = 0) -> None:
        a = R.Args()
        for j, v in enumerate(p):
            a.p[j] = v
        for j, v in enumerate(i):
            a.i[j] = v
        self.args.append(a)                                  # alive until the graph that captured it is gone
        params = (vp * 1)(C.addressof(a))
        self.args.append(params)
        f = self.fn_any(name)
        if smem > 48 * 1024 and self.dyn.get(name, 0) < smem:  # set once, before any capture reaches it
            self.ck("cuFuncSetAttribute", f, MAX_DYN, smem)
            self.dyn[name] = smem
        self.ck("cuLaunchKernel", f, max(1, grid), 1, 1, block, 1, 1, smem, stream, params, None)

    def event(self, stream) -> vp:
        e = vp()
        self.ck("cuEventCreate", C.byref(e), 2)              # CU_EVENT_DISABLE_TIMING
        self.ck("cuEventRecord", e, stream)
        return e

    def wait(self, stream, e) -> None:
        self.ck("cuStreamWaitEvent", stream, e, 0)

    def graph(self, body, origin=None, prio: bool = False) -> tuple[vp, list[int]]:
        """The captured graph, and its kernel nodes' priorities (instantiated to honour them when prio)."""
        origin = origin or self.s1
        self.ck("cuStreamBeginCapture_v2", origin, 0)        # CU_STREAM_CAPTURE_MODE_GLOBAL
        body()
        g, ge = vp(), vp()
        self.ck("cuStreamEndCapture", origin, C.byref(g))
        n = sz()
        self.ck("cuGraphGetNodes", g, None, C.byref(n))
        nodes = (vp * n.value)()
        self.ck("cuGraphGetNodes", g, nodes, C.byref(n))
        seen = set()
        for node in nodes:
            t, val = i32(), C.create_string_buffer(64)
            self.ck("cuGraphNodeGetType", node, C.byref(t))
            if t.value == 0:                                 # CU_GRAPH_NODE_TYPE_KERNEL
                self.ck("cuGraphKernelNodeGetAttribute", node, ATTR_PRIORITY, val)
                seen.add(int.from_bytes(val.raw[:4], "little", signed=True))
        self.ck("cuGraphInstantiateWithFlags", C.byref(ge), g, USE_NODE_PRIORITY if prio else 0)
        return ge, sorted(seen)

    def replay_ms(self, ge, reps: int) -> list[float]:
        out = []
        for _ in range(reps):
            a, b, ms = vp(), vp(), C.c_float()
            self.ck("cuEventCreate", C.byref(a), 0)
            self.ck("cuEventCreate", C.byref(b), 0)
            self.ck("cuEventRecord", a, self.s1)
            self.ck("cuGraphLaunch", ge, self.s1)
            self.ck("cuEventRecord", b, self.s1)
            self.ck("cuEventSynchronize", b)
            self.ck("cuEventElapsedTime", C.byref(ms), a, b)
            out.append(ms.value)
        return out


# ---- the kernels' launches, both forms --------------------------------------------------------------------------------------


def rows_w_words(rb: int) -> int:
    return 8 * ((64 + rb + 1023) // 1024)


def launch_rows(d: Bench, s, form: str, rows_p: int, out_p: int, dom_p: int, m: int, rb: int) -> None:
    if form == "pushed":
        d.go(s, "hash_rows", m, 256, (rows_p, out_p, dom_p), (m, rb))
    else:
        warps = 4
        d.go(s, "hash_rows_w", -(-m // warps), 32 * warps, (rows_p, out_p, dom_p), (m, rb), smem=4 * warps * rows_w_words(rb))


def launch_tree(d: Bench, s, form: str, tree_p: int, keys_p: int, dom_p: int, width: int) -> None:
    """A tree's levels over its level 0: the level keys (hash_node_keys, from the domain id), then hash_nodes a level at a
    time (as pushed) or hash_tree_small in one launch (tuned)."""
    depth = max(0, (width - 1).bit_length())
    d.go(s, "hash_node_keys", 1, 32, (keys_p, dom_p), (depth,))
    if form == "pushed" or width > 2048:
        off, w, lv = 0, width, 0
        while w > 1:
            d.go(s, "hash_nodes", -(-(w // 2) // 128), 128, (tree_p + off, tree_p + off + 32 * w, keys_p + 32 * lv), (lv, w // 2))
            off, w, lv = off + 32 * w, w // 2, lv + 1
    else:
        d.go(s, "hash_tree_small", 1, max(1, min(1024, width // 2)), (tree_p, keys_p), (width,))


def launch_a(d: Bench, s, form: str, rows_p: int, tree_p: int, keys_p: int, dom_p: int, m: int, rb: int) -> None:
    """A's commitment: row leaves and tree.  pushed: hash_rows and a hash_nodes launch per level; tuned: hash_rows_w and
    hash_tree_small.  (A fused one-CTA form, r20260930-074624-555a, put every chunk on one SM: 187 us against 37.)"""
    launch_rows(d, s, form, rows_p, tree_p, dom_p, m, rb)
    launch_tree(d, s, form, tree_p, keys_p, dom_p, m)


def b3s_template(schema: str) -> tuple[bytes, int]:
    """Format -h2's segment-key message with the rank, position, L and j left zero (hash_h2.cuh fills them), and its length."""
    name = schema.encode()
    head = b"\x05" + bytes(28) + len(name).to_bytes(2, "little") + name
    return head + bytes(64 - len(head)), len(head)


def launch_rows_h2(d: Bench, s, rows_p: int, out_p: int, dom_p: int, skeys_p: int, tmpl: tuple[int, int], m: int, rb: int,
                   rank0: int = 0) -> None:
    """-h2's row leaves (hash_rows_b3s): a CTA a row, a thread a 256-byte segment; skeys_p the segment tree's level keys."""
    nseg = max(1, -(-rb // 256))
    d.go(s, "hash_rows_b3s", m, 32 * -(-nseg // 32), (rows_p, out_p, dom_p, skeys_p, tmpl[0]), (m, rb, rank0, tmpl[1]))


def launch_tiles(d: Bench, s, form: str, c_p: int, u_p: int, dg_p: int, lv_p: int, tree_p: int, tkeys_p: int, keys: dict,
                 dom_t: int, rows: int, n: int, h2: tuple[int, int] | None = None) -> int:
    """Message digests, tile leaves, the tile tree's leaves and its levels; returns the tile count.  h2: the tile tree is
    -h2's (hash_leaves_b3s under this key template, then the levels under tkeys_p, computed beforehand)."""
    per, tn = n // 32, n // 64
    tiles = -(-rows // 64) * tn
    if form == "pushed":
        d.go(s, "hash_msg", -(-rows * per // 128), 128, (c_p, u_p, dg_p, keys["k_msg"]), (rows, n))
        d.go(s, "hash_leaf", -(-tiles // 64), 64, (dg_p, lv_p, keys["k_leaf"]), (rows, n))
    else:
        d.go(s, "hash_msg_v", -(-rows * per // 128), 128, (c_p, u_p, dg_p, keys["k_msg"]), (rows, n))
        d.go(s, "hash_leaf_p", -(-tiles // 32), 128, (dg_p, lv_p, keys["k_leaf"]), (rows, n))
    if h2:
        d.go(s, "hash_leaves_b3s", -(-tiles // 128), 128, (lv_p, tree_p, dom_t, h2[0]), (tiles, 32, 0, h2[1]))
        d.go(s, "hash_tree_small", 1, max(1, min(1024, tiles // 2)), (tree_p, tkeys_p), (tiles,))
        return tiles
    d.go(s, "hash_tree_leaves", -(-tiles // 128), 128, (lv_p, tree_p, dom_t), (tiles,))
    launch_tree(d, s, form, tree_p, tkeys_p, dom_t, tiles)
    return tiles


# ---- gates -------------------------------------------------------------------------------------------------------------------


def gates(d: Bench, V: dict) -> dict:
    out: dict = {"rows": {}, "trees": {}, "tiles": {}}
    bad: list[str] = []
    dom = d.put(bytes.fromhex(V["rows"]["domain"]))
    for form in ("pushed", "tuned"):
        wrong = 0
        for case in V["rows"]["cases"]:
            rb = case["rb"]
            rows = shake(f"gate rows {rb}", 2 * rb)
            rp, op = d.put(rows), d.alloc(64)
            launch_rows(d, None, form, rp, op, dom, 2, rb)
            d.sync()
            got = d.read(op, 64).hex()
            if got != case["leaves"][0] + case["leaves"][1]:
                wrong += 1
                bad.append(f"{form} rows rb={rb}")
            d.free(rp)
            d.free(op)
        out["rows"][form] = {"cases": len(V["rows"]["cases"]), "wrong": wrong}
    for label, t in V["trees"].items():
        rows = shake(t["tag"], t["m"] * t["rb"])
        dp = d.put(bytes.fromhex(t["domain"]))
        for form in ("pushed", "tuned"):
            rp, tp, kp = d.put(rows), d.alloc(32 * 2 * t["m"]), d.alloc(32 * 16)
            launch_a(d, None, form, rp, tp, kp, dp, t["m"], t["rb"])
            d.sync()
            ok = d.read(tp + 32 * (2 * t["m"] - 2), 32).hex() == t["root"]
            out["trees"][f"{label} {form}"] = ok
            if not ok:
                bad.append(f"{label} {form} root")
            d.free(rp)
            d.free(tp)
            d.free(kp)
        rp, tp, kp, cnt = d.put(rows), d.alloc(32 * 2 * t["m"]), d.alloc(32 * 16), d.alloc(16)
        d.go(None, "hash_node_keys", 1, 32, (kp, dp), (max(0, (t["m"] - 1).bit_length()),))
        d.go(None, "hash_rows_wt", -(-t["m"] // 4), 128, (rp, tp, dp, kp, cnt), (t["m"], t["rb"], t["m"]), smem=4 * 4 * rows_w_words(t["rb"]))
        d.sync()
        ok = d.read(tp + 32 * (2 * t["m"] - 2), 32).hex() == t["root"] and d.read(cnt, 4) == bytes(4)
        out["trees"][f"{label} one launch"] = ok
        if not ok:
            bad.append(f"{label} one launch root")
        for p_ in (rp, tp, kp, cnt):
            d.free(p_)
    # the producer: derive_rows_hash's A and tree against derive_rows, then hash_rows_wt, on the device
    for rb in (ROW_F32, ROW_BF16):
        y = d.put(shake("gate y", 2 * M * N))
        a1, a2, t1, t2 = d.alloc(M * rb), d.alloc(M * rb), d.alloc(32 * 2 * M), d.alloc(32 * 2 * M)
        dom, kp, c1, c2 = d.put(shake("gate producer dom", 32)), d.alloc(32 * 16), d.alloc(16), d.alloc(16)
        d.go(None, "hash_node_keys", 1, 32, (kp, dom), (5,))
        d.go(None, "derive_rows_hash", M // 4, 128, (y, a1, t1, dom, kp, c1), (M, rb, M, N), smem=4 * 4 * rows_w_words(rb))
        d.go(None, "derive_rows_wide", -(-M * rb // 2 // 256), 256, (y, a2), (M, rb, N))
        d.go(None, "hash_rows_wt", M // 4, 128, (a2, t2, dom, kp, c2), (M, rb, M), smem=4 * 4 * rows_w_words(rb))
        d.sync()
        ok = d.read(a1, M * rb) == d.read(a2, M * rb) and d.read(t1, 32 * 2 * M) == d.read(t2, 32 * 2 * M)
        out["trees"][f"producer {rb} B rows"] = ok
        if not ok:
            bad.append(f"producer {rb}")
        for p_ in (y, a1, a2, t1, t2, dom, kp, c1, c2):
            d.free(p_)
    b = V["blk"]
    rp, op = d.put(shake("gate blk rows", 2 * b["rb"])), d.alloc(64)
    kk, lk = d.put(bytes.fromhex(b["key"])), d.put(b"".join(bytes.fromhex(x) for x in b["levels"]))
    d.go(None, "hash_rows_blk", 2, 256, (rp, op, kk, lk), (2, b["rb"]))
    d.sync()
    ok = d.read(op, 64).hex() == b["roots"][0] + b["roots"][1]
    out["trees"]["block layout (candidate)"] = ok
    if not ok:
        bad.append("block layout")
    for sz, roots in V["seg"].items():
        d.go(None, "hash_rows_seg", 2, max(32, b["rb"] // int(sz)), (rp, op, kk, lk), (2, b["rb"], int(sz)))
        d.sync()
        ok = d.read(op, 64).hex() == roots[0] + roots[1]
        out["trees"][f"segment layout s={sz} (P1 prototype)"] = ok
        if not ok:
            bad.append(f"segment layout {sz}")
    for p_ in (rp, op, kk, lk):
        d.free(p_)
    keys = {k: d.put(bytes.fromhex(R.H1_KEYS[k])) for k in ("k_msg", "k_leaf")}
    for form, case in (("pushed", V["tiles"]["m32"]), ("tuned", V["tiles"]["m32"])):
        rows, n = case["rows"], case["n"]
        c = shake("gate c", 4 * M * n) + bytes(4 * (rows - M) * n)
        uu = shake("gate u", 4 * M * n) + bytes(4 * (rows - M) * n)
        cp, up = d.put(c), d.put(uu)
        dg, lv = d.alloc(32 * rows * (n // 32)), d.alloc(32 * 256)
        tp, dt, tk = d.alloc(32 * 2 * 256), d.put(bytes.fromhex(case["domain"])), d.alloc(32 * 16)
        tiles = launch_tiles(d, None, form, cp, up, dg, lv, tp, tk, keys, dt, rows, n)
        d.sync()
        checks = {"digests": hashlib.sha256(d.read(dg, 32 * rows * (n // 32))).hexdigest() == case["digests_sha256"],
                  "leaves": hashlib.sha256(d.read(lv, 32 * tiles)).hexdigest() == case["leaves_sha256"],
                  "root": d.read(tp + 32 * (2 * tiles - 2), 32).hex() == case["root"]}
        out["tiles"][form] = {"rows": rows, "tiles": tiles, **checks}
        bad += [f"tiles {form} {k}" for k, v in checks.items() if not v]
        for p in (cp, up, dg, lv, tp, dt, tk):
            d.free(p)
    h2 = V["h2"]
    tmpl = {sch: (d.put(t[0]), t[1]) for sch, t in ((ROW, b3s_template(ROW)), (TILE, b3s_template(TILE)))}
    dom, sk = d.put(bytes.fromhex(h2["rows"]["domain"])), d.alloc(32 * 16)
    d.go(None, "b3s_level_keys", 1, 32, (sk, dom), (10,))
    d.sync()
    ok = d.read(sk, 32 * 10).hex() == "".join(h2["rows"]["level_keys"])
    out["trees"]["h2 segment level keys, derived on the card"] = ok
    if not ok:
        bad.append("h2 segment level keys")
    wrong = 0
    for case in h2["rows"]["cases"]:
        rb = case["rb"]
        rp, op = d.put(shake(f"gate h2 rows {rb}", 2 * rb)), d.alloc(64)
        launch_rows_h2(d, None, rp, op, dom, sk, tmpl[ROW], 2, rb, rank0=3)
        d.sync()
        if d.read(op, 64).hex() != case["leaves"][0] + case["leaves"][1]:
            wrong += 1
            bad.append(f"h2 rows rb={rb}")
        d.free(rp)
        d.free(op)
    out["rows"]["h2"] = {"cases": len(h2["rows"]["cases"]), "wrong": wrong}
    da, ka, ska = d.put(bytes.fromhex(h2["tree"]["domain"])), d.alloc(32 * 16), d.alloc(32 * 16)
    rp, tp = d.put(shake("gate tree fp32", M * ROW_F32)), d.alloc(32 * 2 * M)
    d.go(None, "hash_node_keys", 1, 32, (ka, da), (5,))
    d.go(None, "b3s_level_keys", 1, 32, (ska, da), (10,))
    launch_rows_h2(d, None, rp, tp, da, ska, tmpl[ROW], M, ROW_F32)
    d.go(None, "hash_tree_small", 1, M // 2, (tp, ka), (M,))
    d.sync()
    ok = d.read(tp + 32 * (2 * M - 2), 32).hex() == h2["tree"]["root"]
    out["trees"]["A m=32 fp32 rows h2"] = ok
    if not ok:
        bad.append("h2 A tree root")
    case = V["tiles"]["m32"]
    rows, n = case["rows"], case["n"]
    cp, up = d.put(shake("gate c", 4 * M * n)), d.put(shake("gate u", 4 * M * n))
    dg, lv, ttp = d.alloc(32 * rows * (n // 32)), d.alloc(32 * 256), d.alloc(32 * 2 * 256)
    dt2, tk2 = d.put(bytes.fromhex(h2["tiles"]["domain"])), d.alloc(32 * 16)
    d.go(None, "hash_node_keys", 1, 32, (tk2, dt2), (7,))
    tiles = launch_tiles(d, None, "tuned", cp, up, dg, lv, ttp, tk2, keys, dt2, rows, n, h2=tmpl[TILE])
    d.sync()
    ok = d.read(ttp + 32 * (2 * tiles - 2), 32).hex() == h2["tiles"]["root"]
    out["tiles"]["h2"] = {"rows": rows, "tiles": tiles, "root": ok}
    if not ok:
        bad.append("h2 tile tree root")
    for p_ in (dom, sk, da, ka, ska, rp, tp, cp, up, dg, lv, ttp, dt2, tk2, *[t[0] for t in tmpl.values()]):
        d.free(p_)
    lc, p = V["lat_chain"], d.alloc(64)
    for mode, threads, name in ((0, 1, "b3c"), (1, 4, "b3c4")):
        d.go(None, "lat_chain", 1, threads, (p,), (lc["n"], mode))
        d.sync()
        ok = d.read(p + 16, 32).hex() == lc["cv"]
        out["trees"][f"lat_chain {name}"] = ok
        if not ok:
            bad.append(f"lat_chain {name}")
    d.free(p)
    out["failures"] = bad
    return out


# ---- timing ------------------------------------------------------------------------------------------------------------------


def flush(d: Bench, fl: int, sink: int) -> None:
    """A read of 512 MB (4x the L2) that writes nothing: no dirty line left behind."""
    d.go(None, "l2_flush", 4 * d.sms, 256, (fl, sink), ((512 << 20) // 4,))


def seg_rates(d: Bench, nv: R.Nvml, reps: int) -> dict:
    """Prefill's A-row leaves at 8,192 rows of 32 KB: -h1's two row kernels against P1's prototype at s = 64 and 256, the
    forms interleaved in each rep (order rotated), an L2 flush before each timing, the SM clock after it."""
    fl, sink = d.alloc(512 << 20), d.alloc(16)
    m, rb = 8192, ROW_F32
    rp, op, dom = d.alloc(m * rb), d.alloc(32 * m), d.put(bytes(32))
    kk, lk = d.put(shake("rate seg key", 32)), d.put(shake("rate seg levels", 32 * 9))
    forms = {"rows 32768 B pushed": lambda: launch_rows(d, None, "pushed", rp, op, dom, m, rb),
             "rows 32768 B tuned": lambda: launch_rows(d, None, "tuned", rp, op, dom, m, rb)}
    for sz in (64, 256):
        forms[f"rows 32768 B segment s={sz} (P1)"] = lambda sz=sz: d.go(None, "hash_rows_seg", m, rb // sz, (rp, op, kk, lk), (m, rb, sz))
    for body in forms.values():
        body()
    d.sync()
    ms: dict = {k: [] for k in forms}
    clk: dict = {k: [] for k in forms}
    order = list(forms)
    for rep in range(reps):
        for name in order[rep % len(order):] + order[:rep % len(order)]:
            flush(d, fl, sink)
            ms[name] += d.timed(forms[name], 1)
            clk[name].append(nv.read()["sm_mhz"])
    out = {}
    for name in forms:
        med = statistics.median(ms[name])
        out[name] = {"bytes": m * rb, "ms": med, "raw_ms": ms[name], "gb_s": m * rb / med / 1e6, "sm_mhz": clk[name],
                     "bytes_per_sm_clk": m * rb / (med * 1e-3) / (d.sms * max(statistics.median(clk[name]), 1) * 1e6)}
    for p in (fl, sink, rp, op, dom, kk, lk):
        d.free(p)
    return out


def h2_rates(d: Bench, nv: R.Nvml, reps: int, V: dict, tdir: str | None) -> dict:
    """Prefill at 8,192^3: A's row leaves under -h1's two kernels and -h2's, -h2's A tree levels, and the message digests and
    tile leaves (the same kernels under both formats), interleaved in each rep (order rotated) with an L2 flush before each
    timing and the SM clock after it.  Then -h2's whole A commitment once under the prefill domain, written as a transcript:
    the tree only, since A's row r is SHAKE128("h2 prefill A {r}") and verify_h1.py regenerates it."""
    fl, sink = d.alloc(512 << 20), d.alloc(16)
    m, n, rb = 8192, N, ROW_F32
    rp, op = d.put(b"".join(shake(f"h2 prefill A {r}", rb) for r in range(m))), d.alloc(32 * m)
    pa = V["decode"]["prefill_a2"]
    dom1, dom2, sk2, k2 = d.put(bytes(32)), d.put(bytes.fromhex(pa["domain"])), d.alloc(32 * 16), d.alloc(32 * 16)
    d.go(None, "b3s_level_keys", 1, 32, (sk2, dom2), (9,))
    d.go(None, "hash_node_keys", 1, 32, (k2, dom2), (max(0, (m - 1).bit_length()),))
    t = b3s_template(ROW)
    tr = (d.put(t[0]), t[1])
    keys = {kk: d.put(bytes.fromhex(R.H1_KEYS[kk])) for kk in ("k_msg", "k_leaf")}
    cp, up, dg = d.alloc(4 * m * n), d.alloc(4 * m * n), d.alloc(32 * m * (n // 32))
    tiles = (m // 64) * (n // 64)
    lv, tree = d.alloc(32 * tiles), d.alloc(32 * 2 * m)

    def a_tree_levels() -> None:
        off, w, lvl = 0, m, 0
        while w > 1:
            d.go(None, "hash_nodes", -(-(w // 2) // 128), 128, (tree + off, tree + off + 32 * w, k2 + 32 * lvl), (lvl, w // 2))
            off, w, lvl = off + 32 * w, w // 2, lvl + 1

    forms = {"A rows h1 pushed": lambda: launch_rows(d, None, "pushed", rp, op, dom1, m, rb),
             "A rows h1 tuned": lambda: launch_rows(d, None, "tuned", rp, op, dom1, m, rb),
             "A rows h2": lambda: launch_rows_h2(d, None, rp, op, dom2, sk2, tr, m, rb),
             "A tree levels (both formats)": a_tree_levels,
             "messages tuned (both formats)": lambda: d.go(None, "hash_msg_v", m * (n // 32) // 128, 128, (cp, up, dg, keys["k_msg"]), (m, n)),
             "tile leaves tuned (both formats)": lambda: d.go(None, "hash_leaf_p", tiles // 32, 128, (dg, lv, keys["k_leaf"]), (m, n))}
    for body in forms.values():
        body()
    d.sync()
    ms: dict = {k: [] for k in forms}
    clk: dict = {k: [] for k in forms}
    order = list(forms)
    for rep in range(reps):
        for name in order[rep % len(order):] + order[:rep % len(order)]:
            flush(d, fl, sink)
            ms[name] += d.timed(forms[name], 1)
            clk[name].append(nv.read()["sm_mhz"])
    out: dict = {}
    for name in forms:
        med = statistics.median(ms[name])
        out[name] = {"ms": med, "raw_ms": ms[name], "sm_mhz": clk[name]}
        if name.startswith("A rows"):
            out[name]["gb_s"] = m * rb / med / 1e6
    launch_rows_h2(d, None, rp, tree, dom2, sk2, tr, m, rb)
    a_tree_levels()
    d.sync()
    if tdir:
        os.makedirs(tdir, exist_ok=True)
        run = os.path.basename(os.environ.get("RESEARCH_RUN_DIR", "").rstrip("/")) or "local"
        body = d.read(tree, 32 * (2 * m - 1))
        head = {"run": run, "arm": "prefill A h2", "m": m, "rb": rb, "n": n, "hashing": "h2", "a": pa,
                "a_rows_tag": "h2 prefill A {r}", "sections": [["a_tree", len(body)]]}
        blob = json.dumps(head).encode() + b"\n" + body
        path = os.path.join(tdir, "prefill-A-h2.bin")
        open(path, "wb").write(blob)
        out["transcript"] = {"path": path, "sha256": hashlib.sha256(blob).hexdigest()}
    for p in (fl, sink, rp, op, dom1, dom2, sk2, k2, tr[0], *keys.values(), cp, up, dg, lv, tree):
        d.free(p)
    return out


def rates(d: Bench, nv: R.Nvml, reps: int) -> dict:
    """Bytes hashed per second at prefill size, each form, after an L2 flush (a 256 MB device copy) before each rep."""
    fl, sink = d.alloc(512 << 20), d.alloc(16)
    keys = {k: d.put(bytes.fromhex(R.H1_KEYS[k])) for k in ("k_msg", "k_leaf")}
    dom = d.put(bytes(32))
    m, n = 8192, 8192
    out = {}

    def timed(name: str, nbytes: int, body) -> None:
        body()
        d.sync()
        ms = []
        for _ in range(reps):
            flush(d, fl, sink)
            ms += d.timed(body, 1)
        med = statistics.median(ms)
        clk = nv.read()
        out[name] = {"bytes": nbytes, "ms": med, "raw_ms": ms, "gb_s": nbytes / med / 1e6,
                     "bytes_per_sm_clk": nbytes / (med * 1e-3) / (d.sms * max(clk["sm_mhz"], 1) * 1e6), "clocks": clk}

    for rb in (ROW_F32, ROW_BF16):
        rp, op = d.alloc(m * rb), d.alloc(32 * m)
        for form in ("pushed", "tuned"):
            timed(f"rows {rb} B {form}", m * rb, lambda form=form: launch_rows(d, None, form, rp, op, dom, m, rb))
        d.free(rp)
        d.free(op)
    cp, up, dg, lv = d.alloc(4 * m * n), d.alloc(4 * m * n), d.alloc(32 * m * (n // 32)), d.alloc(32 * (m // 64) * (n // 64))
    for form, kmsg, kleaf, bl in (("pushed", "hash_msg", "hash_leaf", 64), ("tuned", "hash_msg_v", "hash_leaf_p", 128)):
        timed(f"messages {form}", 8 * m * n,
              lambda kmsg=kmsg: d.go(None, kmsg, m * (n // 32) // 128, 128, (cp, up, dg, keys["k_msg"]), (m, n)))
        tiles = (m // 64) * (n // 64)
        grid = tiles // 64 if form == "pushed" else tiles // 32
        timed(f"tile leaves {form}", 32 * m * (n // 32),
              lambda kleaf=kleaf, bl=bl, grid=grid: d.go(None, kleaf, grid, bl, (dg, lv, keys["k_leaf"]), (m, n)))
    for p in (cp, up, dg, lv, fl):
        d.free(p)
    return out


#: decode's arms: (name, A's rows in bytes, A's commitment, the post-GEMM hashing).  A's commitment: "none"; "pushed" (hash_rows,
#: hash_node_keys, hash_nodes a level a launch); "tuned" (hash_rows_w, hash_node_keys, hash_tree_small); "keys" (tuned, the
#: level keys on the side stream from the call's start: they depend on the domain id only); "one" (the keys so, and
#: hash_rows_wt: rows and tree in one launch); "producer" (the keys so, and derive_rows_hash: A hashed by the kernel that
#: writes it). "pre", "onepre": "tuned" and "one" with the level keys computed once, before the chain (they depend on the
#: domain id alone, known from the salt and the matmul's index: a table at the epoch's start), so no call launches them.
#: Candidates that change the protocol, measured only: "overlap" (the one-launch commitment on the side stream beside the
#: GEMM, as if the noise didn't wait for A's root) and "blkpre" (the per-block Merkle row layout, hash_rows_blk, then
#: hash_tree_small, keys so). The post-GEMM hashing: "none", "pushed", "serial" (tuned, on the GEMM's stream), "side"
#: (tuned, on the side stream), "deferred" (the side stream's tiles of call i start once call i + 1's A commitment is done,
#: so they overlap only the GEMM). An arm named "…, priorities" runs the chain on the highest-priority stream and the side
#: stream at the lowest, in a graph instantiated to honour its nodes' priorities.
ARMS = [
    ("plain fp32 rows", ROW_F32, "none", "none"), ("plain bf16 rows", ROW_BF16, "none", "none"),
    ("h1 as pushed", ROW_F32, "pushed", "pushed"), ("h1 tuned", ROW_F32, "tuned", "serial"),
    ("h1 tuned, side stream", ROW_F32, "tuned", "side"),
    ("A as pushed", ROW_F32, "pushed", "none"), ("A tuned", ROW_F32, "tuned", "none"),
    ("A tuned, keys early", ROW_F32, "keys", "none"), ("A tuned, keys precomputed", ROW_F32, "pre", "none"),
    ("A in one launch, keys precomputed", ROW_F32, "onepre", "none"), ("A in its producer", ROW_F32, "producer", "none"),
    ("h1 tuned, keys precomputed, side stream", ROW_F32, "pre", "side"),
    ("h1 tuned, keys precomputed, side stream, priorities", ROW_F32, "pre", "side"),
    ("h1 A in one launch, keys precomputed, side stream", ROW_F32, "onepre", "side"),
    ("candidate: A overlapped with the GEMM", ROW_F32, "overlap", "none"),
    ("candidate: A block layout", ROW_F32, "blkpre", "none"),
    ("candidate: h1 A block layout, side stream", ROW_F32, "blkpre", "side"),
    ("candidate: h1 A block layout, side stream, priorities", ROW_F32, "blkpre", "side"),
    ("candidate: A segment layout s=64", ROW_F32, "seg64", "none"), ("candidate: A segment layout s=256", ROW_F32, "seg256", "none"),
    ("candidate: h1 A segment layout s=64, side stream, priorities", ROW_F32, "seg64", "side"),
    ("candidate: h1 A segment layout s=256, side stream, priorities", ROW_F32, "seg256", "side"),
    ("h1 tuned, keys precomputed, side stream deferred, priorities", ROW_F32, "pre", "deferred"),
    ("candidate: h1 A segment layout s=64, side stream deferred, priorities", ROW_F32, "seg64", "deferred"),
    ("candidate: h1 A segment layout s=256, side stream deferred, priorities", ROW_F32, "seg256", "deferred"),
    ("candidate: A bf16 rows, keys precomputed", ROW_BF16, "pre", "none"),
    ("candidate: h1 bf16 rows, keys precomputed, side stream", ROW_BF16, "pre", "side"),
    ("candidate: h1 bf16 rows, keys precomputed, side stream, priorities", ROW_BF16, "pre", "side"),
    ("h2 A alone", ROW_F32, "h2", "none"), ("h2, side stream, priorities", ROW_F32, "h2", "side"),
    ("h2, deferred side stream, priorities", ROW_F32, "h2", "deferred"),
    ("P2 prototype: h2 A alone, tree on its own stream, priorities", ROW_F32, "h2p2", "none"),
    ("P2 prototype: h2, tree on its own stream, deferred side stream, priorities", ROW_F32, "h2p2", "deferred"),
]


#: --preset p1: P1's prototype (hash_rows_seg, timing only) and its baselines, interleaved rep by rep
PRESETS = {"p1": ["plain fp32 rows", "A tuned, keys precomputed", "h1 tuned, keys precomputed, side stream, priorities",
                  "candidate: A block layout", "candidate: A segment layout s=64", "candidate: A segment layout s=256",
                  "candidate: h1 A block layout, side stream, priorities",
                  "candidate: h1 A segment layout s=64, side stream, priorities",
                  "candidate: h1 A segment layout s=256, side stream, priorities",
                  "h1 tuned, keys precomputed, side stream deferred, priorities",
                  "candidate: h1 A segment layout s=64, side stream deferred, priorities",
                  "candidate: h1 A segment layout s=256, side stream deferred, priorities"],
           "h2": ["plain fp32 rows", "A tuned, keys precomputed", "h1 tuned, keys precomputed, side stream, priorities",
                  "h2 A alone", "h2, side stream, priorities", "h2, deferred side stream, priorities",
                  "P2 prototype: h2 A alone, tree on its own stream, priorities",
                  "P2 prototype: h2, tree on its own stream, deferred side stream, priorities"]}

#: the A modes whose level keys go on the side stream at each call's start
EARLY = ("keys", "one", "producer", "overlap")


def latency(d: Bench) -> dict:
    """Dependent compressions on one thread and on four lanes: SM cycles per compression."""
    out, n = {}, 512
    p = d.alloc(64)
    for mode, threads, name in ((0, 1, "one thread (b3c)"), (1, 4, "four lanes (b3c4)")):
        d.go(None, "lat_chain", 1, threads, (p,), (n, mode))
        d.sync()
        cyc = []
        for _ in range(5):
            d.go(None, "lat_chain", 1, threads, (p,), (n, mode))
            d.sync()
            cyc.append(int.from_bytes(d.read(p, 8), "little") / n)
        out[name] = {"cycles_per_compression": statistics.median(cyc), "raw": cyc}
    return out


#: the decode arms whose last call's output verify_h1.py checks (every arm on -h1's format; the block layout is not)
VERIFY = [name for name, _rb, amode, _post in ARMS if not amode.startswith(("blk", "seg")) and not name.startswith("plain")]


def slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in name).strip("-").replace("--", "-")


def decode(d: Bench, nv: R.Nvml, reps: int, V: dict, tdir: str | None, only: list[str] | None = None) -> dict:
    """Per call of the dependent chain at m = 32, n = k = 8,192, each arm replayed as one graph of CALLS calls."""
    n, k, m = N, K, M
    depth = max(0, (m - 1).bit_length())
    keys = {kk: d.put(bytes.fromhex(R.H1_KEYS[kk])) for kk in ("k_msg", "k_leaf")}
    dom_a, dom_t = d.put(bytes.fromhex(V["decode"]["a"]["domain"])), d.put(bytes.fromhex(V["decode"]["t"]["domain"]))
    blk_key, blk_levels = d.put(shake("blk key", 32)), d.put(shake("blk levels", 32 * 9))
    h2d = {kk: d.put(bytes.fromhex(V["decode"][kk]["domain"])) for kk in ("a2", "t2")}
    a2_keys, a2_skeys, t2_keys = d.alloc(32 * 16), d.alloc(32 * 16), d.alloc(32 * 16)
    tmpl = {sch: (d.put(t[0]), t[1]) for sch, t in ((ROW, b3s_template(ROW)), (TILE, b3s_template(TILE)))}
    w = [d.put(shake(f"weights {s}", n * k)) for s in range(WSETS)]
    y = d.alloc(2 * m * n)
    rows = {rb: d.alloc(m * rb) for rb in (ROW_F32, ROW_BF16)}
    a_tree, a_keys, counter = d.alloc(32 * 2 * m), d.alloc(32 * 16), d.alloc(16)
    slots = [{"c": d.alloc(4 * 64 * n), "u": d.alloc(4 * 64 * n), "dg": d.alloc(32 * 64 * (n // 32)), "lv": d.alloc(32 * 256),
              "tree": d.alloc(32 * 2 * 256), "keys": d.alloc(32 * 16)} for _ in range(2)]
    out: dict = {"calls_per_graph": CALLS, "weight_sets": WSETS, "shape": f"m{m}-n{n}-k{k}", "priority_range": d.priority_range}

    def commit_a(s, amode: str, rb: int) -> None:
        if amode in ("pushed", "tuned"):
            launch_a(d, s, amode, rows[rb], a_tree, a_keys, dom_a, m, rb)
        elif amode in ("keys", "pre"):
            launch_rows(d, s, "tuned", rows[rb], a_tree, dom_a, m, rb)
            d.go(s, "hash_tree_small", 1, max(1, m // 2), (a_tree, a_keys), (m,))
        elif amode in ("one", "onepre", "overlap"):
            d.go(s, "hash_rows_wt", -(-m // 4), 128, (rows[rb], a_tree, dom_a, a_keys, counter), (m, rb, m), smem=4 * 4 * rows_w_words(rb))
        elif amode in ("seg64", "seg256"):
            sz = int(amode[3:])
            d.go(s, "hash_rows_seg", m, max(32, rb // sz), (rows[rb], a_tree, blk_key, blk_levels), (m, rb, sz))
            d.go(s, "hash_tree_small", 1, max(1, m // 2), (a_tree, a_keys), (m,))
        elif amode == "blkpre":
            d.go(s, "hash_rows_blk", m, 256, (rows[rb], a_tree, blk_key, blk_levels), (m, rb))
            d.go(s, "hash_tree_small", 1, max(1, m // 2), (a_tree, a_keys), (m,))
        elif amode == "h2":
            launch_rows_h2(d, s, rows[rb], a_tree, h2d["a2"], a2_skeys, tmpl[ROW], m, rb)
            d.go(s, "hash_tree_small", 1, max(1, m // 2), (a_tree, a2_keys), (m,))

    def tiles_on(s, sl: dict, h2: bool) -> None:
        if h2:
            launch_tiles(d, s, "tuned", sl["c"], sl["u"], sl["dg"], sl["lv"], sl["tree"], t2_keys, keys, h2d["t2"], m, n,
                         h2=tmpl[TILE])
        else:
            launch_tiles(d, s, "tuned", sl["c"], sl["u"], sl["dg"], sl["lv"], sl["tree"], sl["keys"], keys, dom_t, m, n)

    def call(i: int, rb: int, amode: str, post: str, st: dict) -> None:
        s1, s2 = st["streams"]
        sl = slots[i % 2]
        early = amode in EARLY
        if early:                                            # the level keys, from the domain id alone, beside the producer
            d.wait(s2, d.event(s1))
            d.go(s2, "hash_node_keys", 1, 32, (a_keys, dom_a), (depth,))
            ek = d.event(s2)
        if st.get("a_done") is not None:                     # the previous call's overlapped commitment has read its A
            d.wait(s1, st.pop("a_done"))
        if amode == "producer":
            d.wait(s1, ek)
            d.go(s1, "derive_rows_hash", m // 4, 128, (y, rows[rb], a_tree, dom_a, a_keys, counter), (m, rb, m, n),
                 smem=4 * 4 * rows_w_words(rb))
        else:
            d.go(s1, "derive_rows", -(-m * k // 256), 256, (y, rows[ROW_BF16]), (m * k,))
            if rb == ROW_F32:
                d.go(s1, "derive_rows", -(-m * k // 256), 256, (y, rows[ROW_F32]), (m * k,))
            if amode == "overlap":
                d.wait(s2, d.event(s1))
                commit_a(s2, amode, rb)
                st["a_done"] = d.event(s2)
            elif amode == "h2p2":                            # P2: the GEMM waits for the row leaves only, the tree beside it
                if st.get("tree_done") is not None:
                    d.wait(s1, st.pop("tree_done"))
                launch_rows_h2(d, s1, rows[rb], a_tree, h2d["a2"], a2_skeys, tmpl[ROW], m, rb)
                d.wait(d.h3, d.event(s1))
                d.go(d.h3, "hash_tree_small", 1, max(1, m // 2), (a_tree, a2_keys), (m,))
                st["tree_done"] = d.event(d.h3)
            elif amode != "none":
                if early:
                    d.wait(s1, ek)
                commit_a(s1, amode, rb)
        if st.get("pending") is not None:                    # "deferred": the previous call's tiles, now A's commitment is done
            d.wait(s2, d.event(s1))
            tiles_on(s2, *st.pop("pending"))
            st["joins"].append(d.event(s2))
        if post in ("side", "deferred") and len(st["joins"]) >= 2:   # the side stream's slot from two calls back is free again
            d.wait(s1, st["joins"][-2])
        a_in = rows[rb] if amode == "producer" else rows[ROW_BF16]   # the A this call's producer wrote
        d.go(s1, "standin_gemm", n * 32 // 256, 256, (w[i % WSETS], a_in, sl["c"], sl["u"], y), (m, n, k))
        if post == "none":
            return
        h2 = amode.startswith("h2")
        if post == "deferred":
            st["pending"] = (sl, h2)
            return
        if post == "side":
            d.wait(s2, d.event(s1))
            tiles_on(s2, sl, h2)
            st["joins"].append(d.event(s2))
        elif post == "pushed":
            launch_tiles(d, s1, "pushed", sl["c"], sl["u"], sl["dg"], sl["lv"], sl["tree"], sl["keys"], keys, dom_t, m, n)
        else:
            tiles_on(s1, sl, h2)

    arms = [a for a in ARMS if only is None or a[0] in only]
    if only is not None and len(arms) != len(only):
        raise SystemExit(f"preset names no arm: {sorted(set(only) - {x[0] for x in arms})}")
    graphs, prios = {}, {}
    for name, rb, amode, post in arms:
        streams = (d.h1, d.h2) if name.endswith(", priorities") else (d.s1, d.s2)

        def body(rb=rb, amode=amode, post=post, streams=streams):
            st: dict = {"joins": [], "streams": streams}
            for i in range(CALLS):
                call(i, rb, amode, post, st)
            if st.get("pending") is not None:
                d.wait(streams[1], d.event(streams[0]))
                tiles_on(streams[1], *st.pop("pending"))
                st["joins"].append(d.event(streams[1]))
            for e in [st.get("a_done"), st.get("tree_done")] + st["joins"][-1:]:
                if e is not None:
                    d.wait(streams[0], e)
            if amode in EARLY:
                d.wait(streams[0], d.event(streams[1]))
        graphs[name], prios[name] = d.graph(body, streams[0], name.endswith(", priorities"))
    d.go(None, "hash_node_keys", 1, 32, (a_keys, dom_a), (depth,))     # the "pre" arms' table (every arm writes the same keys)
    d.go(None, "hash_node_keys", 1, 32, (a2_keys, h2d["a2"]), (depth,))  # -h2's tables: A's tree, its segment trees, the tiles'
    d.go(None, "b3s_level_keys", 1, 32, (a2_skeys, h2d["a2"]), (9,))
    d.go(None, "hash_node_keys", 1, 32, (t2_keys, h2d["t2"]), (7,))
    d.sync()
    for ge in graphs.values():                               # warm every graph once
        d.replay_ms(ge, 1)
    raw = {name: [] for name in graphs}
    clk = {name: [] for name in graphs}
    for rep in range(reps):                                  # the arms interleaved, rotated each rep
        order = list(graphs)
        order = order[rep % len(order):] + order[:rep % len(order)]
        for name in order:
            raw[name] += d.replay_ms(graphs[name], 1)
            clk[name].append(nv.read()["sm_mhz"])
    for name, rb, amode, post in arms:
        ms = raw[name]
        out[name] = {"rows_bytes": rb, "a": amode, "post": post, "per_call_us": statistics.median(ms) * 1e3 / CALLS, "raw_ms": ms,
                     "sm_mhz": clk[name], "node_priorities": prios[name]}
    if tdir:                                                 # each verified arm once more; its last call's output, for the verifier
        os.makedirs(tdir, exist_ok=True)
        run = os.path.basename(os.environ.get("RESEARCH_RUN_DIR", "").rstrip("/")) or "local"
        sl = slots[(CALLS - 1) % 2]
        per, tiles = n // 32, (-(-m // 64)) * (n // 64)
        for name, rb, amode, post in arms:
            if name not in VERIFY:
                continue
            d.replay_ms(graphs[name], 1)
            d.sync()
            parts = [("a_rows", d.read(rows[rb], m * rb)), ("a_tree", d.read(a_tree, 32 * (2 * m - 1)))]
            if post != "none":
                parts += [("c", d.read(sl["c"], 4 * m * n)), ("u", d.read(sl["u"], 4 * m * n)), ("digests", d.read(sl["dg"], 32 * m * per)),
                          ("leaves", d.read(sl["lv"], 32 * tiles)), ("tile_tree", d.read(sl["tree"], 32 * (2 * tiles - 1)))]
            h2 = amode.startswith("h2")
            head = {"run": run, "arm": name, "m": m, "rb": rb, "n": n, "hashing": "h2" if h2 else "h1",
                    "a": V["decode"]["a2" if h2 else "a"], "t": V["decode"]["t2" if h2 else "t"],
                    "sections": [[k, len(v)] for k, v in parts]}
            blob = json.dumps(head).encode() + b"\n" + b"".join(v for _k, v in parts)
            path = os.path.join(tdir, slug(name) + ".bin")
            open(path, "wb").write(blob)
            out[name]["transcript"] = {"path": path, "sha256": hashlib.sha256(blob).hexdigest()}
    for name, r in out.items():
        if isinstance(r, dict) and "per_call_us" in r:
            plain = out[f"plain {'fp32' if r['rows_bytes'] == ROW_F32 else 'bf16'} rows"]["per_call_us"]  # every preset has its plain arm
            r["hashing_us"] = r["per_call_us"] - plain
            r["x_plain_standin"] = r["per_call_us"] / plain
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cubin", required=True)
    ap.add_argument("--standin", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--transcripts", help="a directory for the verified decode arms' transcripts (verify_h1.py)")
    ap.add_argument("--preset", choices=["all", *PRESETS], default="all",
                    help="p1: only P1's prototype and its baselines (decode arms and prefill row rates), no latency probe; "
                         "h2: format -h2, P2's prototype and their baselines, and prefill's -h2 A commitment as a transcript")
    args = ap.parse_args()
    cubin, standin = open(args.cubin, "rb").read(), open(args.standin, "rb").read()
    d = Bench(cubin, standin)
    want = os.environ.get("GPU_LEASE_UUID")
    nv = Sensors(want or d.uuid())
    res = {"device": d.name, "uuid": d.uuid(), "lease_uuid": want, "driver": d.driver, "sms": d.sms, "nvml_driver": nv.driver,
           "cubin_sha256": hashlib.sha256(cubin).hexdigest(), "standin_sha256": hashlib.sha256(standin).hexdigest(),
           "clock_label": "locked-2100", "clocks_idle": nv.read(), "time": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())}
    save = lambda: open(args.out, "w").write(json.dumps(res, indent=1))  # noqa: E731
    if want and want != res["uuid"]:
        res["error"] = "the visible device is not the leased one"
        save()
        sys.exit(3)
    V = json.load(open(args.vectors))
    res["gates"] = gates(d, V)
    save()
    print(json.dumps({"gates": res["gates"]["failures"], "rows": res["gates"]["rows"]}), flush=True)
    if res["gates"]["failures"]:
        sys.exit(4)
    res["preset"] = args.preset
    res["rates"] = (seg_rates(d, nv, args.reps) if args.preset == "p1" else
                    h2_rates(d, nv, args.reps, V, args.transcripts) if args.preset == "h2" else rates(d, nv, args.reps))
    save()
    res["latency"] = latency(d) if args.preset == "all" else {}
    save()
    res["decode"] = decode(d, nv, args.reps, V, args.transcripts, PRESETS.get(args.preset))
    res["clocks_after"] = nv.read()
    save()
    for name, r in res["decode"].items():
        if isinstance(r, dict):
            print(f"decode {name}: {r['per_call_us']:.1f} us per call, hashing {r['hashing_us']:.1f} us", flush=True)
    for name, r in res["latency"].items():
        print(f"latency {name}: {r['cycles_per_compression']:.0f} cycles a compression", flush=True)
    for name, r in res["rates"].items():
        if "ms" in r:
            print(f"rate {name}: {r['ms']:.4f} ms" + (f", {r['gb_s']:.0f} GB/s" if "gb_s" in r else ""), flush=True)


if __name__ == "__main__":
    main()
