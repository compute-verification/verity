"""-h1's hashing on sm_120 (node 2): device gates, cost per byte, and decode's dependent chain at m = 32, n = k = 8,192.

  python3 h1_bench.py --cubin hash_sm120a.cubin --standin h1_standin.cubin --vectors h1_vectors.json --out h1_bench.json

Standard library only; the driver API through `run.py`'s `Dev`. The GPU is the one the lease named (`GPU_LEASE_UUID`, checked
against `cuDeviceGetUuid`); NVML reads that UUID's clocks.

**Gates, all before any timing** (a difference exits 4 with nothing timed). Each kernel form against `h1_vectors.json`, which
`h1_vectors.py` writes from `verity.commitments`' frame-b3 and Pearl-C's h1 digests:
- the row leaves of `hash_rows` (as pushed) and `hash_rows_w` (tuned), at every chunk count 1 … 258: rows of 1,024c − 64
  bytes (exactly c chunks with the one-block header), 1,024c − 27 (37 bytes past a chunk boundary) and 1,024c, rows 0 and 1;
- A's tree at m = 32 (FP32 rows at k = 8,192, as `-h1` commits them, and BF16), `hash_nodes` level by level and
  `hash_tree_small` in one launch;
- the tile hashing at m = 32, n = 8,192 (decode's partial 32 x 64 tiles), as pushed (`hash_msg`, `hash_leaf`) and tuned
  (`hash_msg_v`, `hash_leaf_p`), with the tile tree both ways (level keys, then `hash_nodes` or `hash_tree_small`).

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
}
M, N, K = 32, 8192, 8192
CALLS, WSETS = 32, 4
MAX_DYN, SMEM_OPTIN = 8, 97
ROW_F32, ROW_BF16 = 4 * K, 2 * K


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

    def graph(self, body) -> vp:
        self.ck("cuStreamBeginCapture_v2", self.s1, 0)       # CU_STREAM_CAPTURE_MODE_GLOBAL
        body()
        g, ge = vp(), vp()
        self.ck("cuStreamEndCapture", self.s1, C.byref(g))
        self.ck("cuGraphInstantiateWithFlags", C.byref(ge), g, 0)
        return ge

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


def launch_tiles(d: Bench, s, form: str, c_p: int, u_p: int, dg_p: int, lv_p: int, tree_p: int, tkeys_p: int, keys: dict,
                 dom_t: int, rows: int, n: int) -> int:
    """Message digests, tile leaves, the tile tree's leaves and its levels; returns the tile count."""
    per, tn = n // 32, n // 64
    tiles = -(-rows // 64) * tn
    if form == "pushed":
        d.go(s, "hash_msg", -(-rows * per // 128), 128, (c_p, u_p, dg_p, keys["k_msg"]), (rows, n))
        d.go(s, "hash_leaf", -(-tiles // 64), 64, (dg_p, lv_p, keys["k_leaf"]), (rows, n))
    else:
        d.go(s, "hash_msg_v", -(-rows * per // 128), 128, (c_p, u_p, dg_p, keys["k_msg"]), (rows, n))
        d.go(s, "hash_leaf_p", -(-tiles // 32), 128, (dg_p, lv_p, keys["k_leaf"]), (rows, n))
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
    out["failures"] = bad
    return out


# ---- timing ------------------------------------------------------------------------------------------------------------------


def flush(d: Bench, fl: int, sink: int) -> None:
    """A read of 512 MB (4x the L2) that writes nothing: no dirty line left behind."""
    d.go(None, "l2_flush", 4 * d.sms, 256, (fl, sink), ((512 << 20) // 4,))


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
#: writes it). Candidates that change the protocol, measured only: "overlap" (the one-launch commitment on the side stream
#: beside the GEMM, as if the noise didn't wait for A's root) and "blk" (the per-block Merkle row layout, hash_rows_blk, then
#: hash_tree_small). The post-GEMM hashing: "none", "pushed", "serial" (tuned, on the GEMM's stream), "side" (tuned, on the
#: side stream).
ARMS = [
    ("plain fp32 rows", ROW_F32, "none", "none"), ("plain bf16 rows", ROW_BF16, "none", "none"),
    ("h1 as pushed", ROW_F32, "pushed", "pushed"), ("h1 tuned", ROW_F32, "tuned", "serial"),
    ("h1 tuned, side stream", ROW_F32, "tuned", "side"),
    ("A as pushed", ROW_F32, "pushed", "none"), ("A tuned", ROW_F32, "tuned", "none"),
    ("A tuned, keys early", ROW_F32, "keys", "none"), ("A in one launch", ROW_F32, "one", "none"),
    ("A in its producer", ROW_F32, "producer", "none"),
    ("h1 A in one launch, side stream", ROW_F32, "one", "side"), ("h1 A in its producer, side stream", ROW_F32, "producer", "side"),
    ("candidate: A overlapped with the GEMM", ROW_F32, "overlap", "none"),
    ("candidate: h1 A overlapped, side stream", ROW_F32, "overlap", "side"),
    ("candidate: A block layout", ROW_F32, "blk", "none"), ("candidate: h1 A block layout, side stream", ROW_F32, "blk", "side"),
    ("candidate: A bf16 rows in its producer", ROW_BF16, "producer", "none"),
    ("candidate: h1 bf16 rows in its producer, side stream", ROW_BF16, "producer", "side"),
]


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


def decode(d: Bench, reps: int) -> dict:
    """Per call of the dependent chain at m = 32, n = k = 8,192, each arm replayed as one graph of CALLS calls."""
    n, k, m = N, K, M
    depth = max(0, (m - 1).bit_length())
    keys = {kk: d.put(bytes.fromhex(R.H1_KEYS[kk])) for kk in ("k_msg", "k_leaf")}
    dom_a, dom_t = d.put(shake("dom a", 32)), d.put(shake("dom t", 32))
    blk_key, blk_levels = d.put(shake("blk key", 32)), d.put(shake("blk levels", 32 * 9))
    w = [d.put(shake(f"weights {s}", n * k)) for s in range(WSETS)]
    y = d.alloc(2 * m * n)
    rows = {rb: d.alloc(m * rb) for rb in (ROW_F32, ROW_BF16)}
    a_tree, a_keys, counter = d.alloc(32 * 2 * m), d.alloc(32 * 16), d.alloc(16)
    slots = [{"c": d.alloc(4 * 64 * n), "u": d.alloc(4 * 64 * n), "dg": d.alloc(32 * 64 * (n // 32)), "lv": d.alloc(32 * 256),
              "tree": d.alloc(32 * 2 * 256), "keys": d.alloc(32 * 16)} for _ in range(2)]
    out: dict = {"calls_per_graph": CALLS, "weight_sets": WSETS, "shape": f"m{m}-n{n}-k{k}"}

    def commit_a(s, amode: str, rb: int) -> None:
        if amode in ("pushed", "tuned"):
            launch_a(d, s, amode, rows[rb], a_tree, a_keys, dom_a, m, rb)
        elif amode == "keys":
            launch_rows(d, s, "tuned", rows[rb], a_tree, dom_a, m, rb)
            d.go(s, "hash_tree_small", 1, max(1, m // 2), (a_tree, a_keys), (m,))
        elif amode in ("one", "overlap"):
            d.go(s, "hash_rows_wt", -(-m // 4), 128, (rows[rb], a_tree, dom_a, a_keys, counter), (m, rb, m), smem=4 * 4 * rows_w_words(rb))
        elif amode == "blk":
            d.go(s, "hash_rows_blk", m, 256, (rows[rb], a_tree, blk_key, blk_levels), (m, rb))
            d.go(s, "hash_tree_small", 1, max(1, m // 2), (a_tree, a_keys), (m,))

    def call(i: int, rb: int, amode: str, post: str, st: dict) -> None:
        s1, s2 = d.s1, d.s2
        sl = slots[i % 2]
        early = amode in ("keys", "one", "producer", "overlap", "blk")
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
            elif amode != "none":
                if early:
                    d.wait(s1, ek)
                commit_a(s1, amode, rb)
        if post == "side" and len(st["joins"]) >= 2:         # the side stream's slot from two calls back is free again
            d.wait(s1, st["joins"][-2])
        a_in = rows[rb] if amode == "producer" else rows[ROW_BF16]   # the A this call's producer wrote
        d.go(s1, "standin_gemm", n * 32 // 256, 256, (w[i % WSETS], a_in, sl["c"], sl["u"], y), (m, n, k))
        if post == "none":
            return
        form = "pushed" if post == "pushed" else "tuned"
        if post == "side":
            d.wait(s2, d.event(s1))
            launch_tiles(d, s2, form, sl["c"], sl["u"], sl["dg"], sl["lv"], sl["tree"], sl["keys"], keys, dom_t, m, n)
            st["joins"].append(d.event(s2))
        else:
            launch_tiles(d, s1, form, sl["c"], sl["u"], sl["dg"], sl["lv"], sl["tree"], sl["keys"], keys, dom_t, m, n)

    graphs = {}
    for name, rb, amode, post in ARMS:
        def body(rb=rb, amode=amode, post=post):
            st: dict = {"joins": []}
            for i in range(CALLS):
                call(i, rb, amode, post, st)
            for e in [st.get("a_done")] + st["joins"][-1:]:
                if e is not None:
                    d.wait(d.s1, e)
            if amode in ("keys", "one", "producer", "overlap", "blk"):
                d.wait(d.s1, d.event(d.s2))
        graphs[name] = d.graph(body)
    for ge in graphs.values():                               # warm every graph once
        d.replay_ms(ge, 1)
    raw = {name: [] for name in graphs}
    for rep in range(reps):                                  # the arms interleaved, rotated each rep
        order = list(graphs)
        order = order[rep % len(order):] + order[:rep % len(order)]
        for name in order:
            raw[name] += d.replay_ms(graphs[name], 1)
    for name, rb, amode, post in ARMS:
        ms = raw[name]
        out[name] = {"rows_bytes": rb, "a": amode, "post": post, "per_call_us": statistics.median(ms) * 1e3 / CALLS, "raw_ms": ms}
    for name, r in out.items():
        if isinstance(r, dict) and "per_call_us" in r:
            plain = out[f"plain {'fp32' if r['rows_bytes'] == ROW_F32 else 'bf16'} rows"]["per_call_us"]
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
    res["rates"] = rates(d, nv, args.reps)
    save()
    res["latency"] = latency(d)
    save()
    res["decode"] = decode(d, args.reps)
    res["clocks_after"] = nv.read()
    save()
    for name, r in res["decode"].items():
        if isinstance(r, dict):
            print(f"decode {name}: {r['per_call_us']:.1f} us per call, hashing {r['hashing_us']:.1f} us", flush=True)
    for name, r in res["latency"].items():
        print(f"latency {name}: {r['cycles_per_compression']:.0f} cycles a compression", flush=True)
    for name, r in res["rates"].items():
        print(f"rate {name}: {r['gb_s']:.0f} GB/s, {r['bytes_per_sm_clk']:.2f} B/SM/clk", flush=True)


if __name__ == "__main__":
    main()
